"""Strict P/D handoff and independently selectable role-specific placement.

Unlike the toy proxy, localhost requests ignore shell proxy variables and
missing transfer metadata is an error, not a hidden full-prefill fallback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .policies import PlacementPolicy


def create_app(args):
    p_policy = PlacementPolicy(
        len(args.prefillers),
        args.prefill_policy,
        args.capacity_tokens,
        prefill_ms_per_token=args.prefill_ms,
        decode_ms_per_token=0,
    )
    d_policy = PlacementPolicy(
        len(args.decoders),
        args.decode_policy,
        args.capacity_tokens,
        prefill_ms_per_token=0,
        decode_ms_per_token=args.decode_ms,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a", buffering=1)

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
            app.state.client = client
            yield
        log.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "prefillers": args.prefillers,
            "decoders": args.decoders,
            "prefill_inflight": len(p_policy.reservations),
            "decode_inflight": len(d_policy.reservations),
        }

    @app.post("/v1/completions")
    async def completion(request: Request):
        payload = await request.json()
        tokens = payload.get("prompt")
        if (
            not isinstance(tokens, list)
            or not tokens
            or not all(isinstance(t, int) and 0 <= t < 2**32 for t in tokens)
        ):
            raise HTTPException(400, "PD proxy requires one token-ID prompt")
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        params = payload.get("kv_transfer_params") or {}
        if not isinstance(params, dict):
            raise HTTPException(400, "kv_transfer_params must be an object")
        optional = {
            key: value
            for key, value in params.items()
            if key.startswith("experiment.") or key == "lmcache.skip_save"
        }
        p_index, p_scores = p_policy.choose(rid, tokens, 0)
        p_start = time.monotonic()
        try:
            prefill = {
                **payload,
                "stream": False,
                "max_tokens": 1,
                "return_token_ids": False,
                "kv_transfer_params": {
                    **optional,
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                },
            }
            for key in ["stream_options", "min_tokens", "min_completion_tokens"]:
                prefill.pop(key, None)
            response = await app.state.client.post(
                args.prefillers[p_index] + "/v1/completions",
                json=prefill,
                headers={"X-Request-ID": rid},
            )
            response.raise_for_status()
            transfer = response.json().get("kv_transfer_params")
            if not transfer or not transfer.get("remote_block_ids"):
                raise RuntimeError("P did not return usable remote KV blocks")
            p_policy.finish(rid, tokens, True)
            log.write(
                json.dumps(
                    {
                        "event": "prefill",
                        "request_id": rid,
                        "worker": p_index,
                        "scores": [asdict(s) for s in p_scores],
                        "elapsed_s": time.monotonic() - p_start,
                        "remote_num_tokens": transfer.get("remote_num_tokens"),
                        "remote_engine_id": transfer.get("remote_engine_id"),
                    }
                )
                + "\n"
            )
            # D's score is output backlog, not P's remaining-prefill signal.
            d_index, d_scores = d_policy.choose(
                rid, tokens, int(payload.get("max_tokens", 16))
            )
            decode = {**payload, "kv_transfer_params": {**optional, **transfer}}
            upstream = await app.state.client.send(
                app.state.client.build_request(
                    "POST",
                    args.decoders[d_index] + "/v1/completions",
                    json=decode,
                    headers={"X-Request-ID": rid},
                ),
                stream=True,
            )
            if upstream.status_code >= 400:
                body = await upstream.aread()
                await upstream.aclose()
                raise RuntimeError(f"D HTTP {upstream.status_code}: {body[:1000]!r}")
        except BaseException as exc:
            p_policy.finish(rid, tokens, False)
            d_policy.finish(rid, tokens, False)
            log.write(
                json.dumps(
                    {
                        "event": "error",
                        "request_id": rid,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "error_repr": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise HTTPException(502, str(exc)) from exc

        def finished(success):
            d_policy.finish(rid, tokens, success)
            log.write(
                json.dumps(
                    {
                        "event": "decode",
                        "request_id": rid,
                        "worker": d_index,
                        "prefill_worker": p_index,
                        "success": success,
                        "elapsed_s": time.monotonic() - p_start,
                        "scores": [asdict(s) for s in d_scores],
                    }
                )
                + "\n"
            )

        if not payload.get("stream"):
            success = False
            try:
                body = json.loads(await upstream.aread())
                success = True
                return JSONResponse(body)
            finally:
                finished(success)
                await upstream.aclose()

        async def stream():
            done = False
            try:
                async for line in upstream.aiter_lines():
                    if line == "data: [DONE]":
                        done = True
                    yield (line + "\n").encode()
            finally:
                finished(done)
                await upstream.aclose()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Worker-Index": str(d_index), "X-Prefill-Index": str(p_index)},
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefillers", nargs="+", required=True)
    parser.add_argument("--decoders", nargs="+", required=True)
    parser.add_argument(
        "--prefill-policy",
        choices=["rr", "load", "threshold", "work", "cost"],
        default="rr",
    )
    parser.add_argument("--decode-policy", choices=["rr", "load", "work"], default="rr")
    parser.add_argument("--capacity-tokens", type=int, default=8192)
    parser.add_argument("--prefill-ms", type=float, default=0.05)
    parser.add_argument("--decode-ms", type=float, default=8)
    parser.add_argument("--port", type=int, default=8191)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    uvicorn.run(create_app(args), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
