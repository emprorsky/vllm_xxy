"""Streaming experimental ingress with a shared prefix-estimate baseline.

Run from the repository root with ``python -m ...router --help``.
The directory is explicitly inferred; it does not bypass engine KV validation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .policies import PlacementPolicy


def create_app(args: argparse.Namespace, policy=None) -> FastAPI:
    directory_source = getattr(args, "directory_source", "inferred_history")
    policy = policy or PlacementPolicy(
        len(args.workers),
        args.policy,
        args.capacity_tokens,
        prefill_ms_per_token=args.prefill_ms,
        decode_ms_per_token=args.decode_ms,
        balance_abs=args.balance_abs,
        balance_rel=args.balance_rel,
        eviction_weight=args.eviction_weight,
        ttl_s=args.ttl,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a", encoding="utf-8", buffering=1)

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600, connect=10),
            limits=httpx.Limits(max_connections=256),
            trust_env=False,
        ) as client:
            app.state.client = client
            yield
        log.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "policy": args.policy,
            "inflight": len(policy.reservations),
            "directory_source": directory_source,
            "workers": args.workers,
        }

    @app.get("/v1/models")
    async def models():
        response = await app.state.client.get(args.workers[0] + "/v1/models")
        return JSONResponse(response.json(), status_code=response.status_code)

    @app.post("/v1/completions")
    async def completion(request: Request):
        payload = await request.json()
        tokens = payload.get("prompt")
        if (
            not isinstance(tokens, list)
            or not tokens
            or not all(isinstance(t, int) and 0 <= t < 2**32 for t in tokens)
        ):
            raise HTTPException(400, "Router requires one token-ID prompt")
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        track_remaining = args.policy.endswith("_remaining")
        if track_remaining and payload.get("n", 1) != 1:
            raise HTTPException(400, "Remaining-work router currently requires n=1")
        try:
            worker, scores = policy.choose(
                request_id, tokens, int(payload.get("max_tokens", 16))
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        start = time.monotonic()
        event = {
            "request_id": request_id,
            "worker": worker,
            "policy": args.policy,
            "directory_source": directory_source,
            "timestamp": time.time(),
            "scores": [asdict(s) for s in scores],
        }
        log.write(json.dumps({**event, "event": "dispatch"}) + "\n")
        try:
            upstream_request = app.state.client.build_request(
                "POST",
                args.workers[worker] + "/v1/completions",
                json=payload,
                headers={"X-Request-ID": request_id},
            )
            upstream = await app.state.client.send(upstream_request, stream=True)
        except BaseException:
            policy.finish(request_id, tokens, False)
            raise
        if upstream.status_code >= 400 or not payload.get("stream", False):
            success = False
            try:
                body = await upstream.aread()
                success = upstream.status_code < 400
                log.write(
                    json.dumps(
                        {
                            **event,
                            "event": "finish",
                            "success": success,
                            "elapsed_s": time.monotonic() - start,
                        }
                    )
                    + "\n"
                )
                return Response(
                    body,
                    media_type=upstream.headers.get("content-type", "application/json"),
                    status_code=upstream.status_code,
                    headers={"X-Worker-Index": str(worker)},
                )
            finally:
                policy.finish(request_id, tokens, success)
                await upstream.aclose()

        async def stream():
            completed = False
            emitted = 0
            try:
                async for line in upstream.aiter_lines():
                    if line == "data: [DONE]":
                        completed = True
                    elif track_remaining and line.startswith("data: "):
                        try:
                            message = json.loads(line[6:])
                            emitted += sum(
                                len(choice.get("token_ids") or [])
                                for choice in message.get("choices", [])
                            )
                            emitted = max(
                                emitted,
                                (message.get("usage") or {}).get(
                                    "completion_tokens", 0
                                ),
                            )
                            policy.observe_output(request_id, emitted)
                        except (ValueError, TypeError, AttributeError):
                            # Preserve the upstream stream; missing token
                            # counts leave the conservative reservation intact.
                            pass
                    yield (line + "\n").encode()
            finally:
                policy.finish(request_id, tokens, completed)
                await upstream.aclose()
                log.write(
                    json.dumps(
                        {
                            **event,
                            "event": "finish",
                            "success": completed,
                            "elapsed_s": time.monotonic() - start,
                            "observed_output_tokens": emitted,
                        }
                    )
                    + "\n"
                )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Worker-Index": str(worker)},
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", nargs="+", required=True)
    parser.add_argument(
        "--policy",
        choices=[
            "rr",
            "load",
            "prefix",
            "threshold",
            "work",
            "cost",
            "work_remaining",
            "cost_remaining",
        ],
        default="cost",
    )
    parser.add_argument("--capacity-tokens", type=int, default=8192)
    parser.add_argument("--prefill-ms", type=float, default=0.05)
    parser.add_argument("--decode-ms", type=float, default=8.0)
    parser.add_argument("--balance-abs", type=int, default=4)
    parser.add_argument("--balance-rel", type=float, default=1.5)
    parser.add_argument("--eviction-weight", type=float, default=1)
    parser.add_argument("--ttl", type=float, default=60)
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("KVEXP_EVENT_ENDPOINTS"):
        from .events_router import create_event_app

        app = create_event_app(args, create_app)
    else:
        app = create_app(args)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
