"""Exercise failure cleanup and real adapter flag propagation without GPUs."""

import argparse
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from . import pd_proxy, router
from .policies import TierTelemetry, WriteAdmission


class BrokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        raise httpx.ReadError("injected response failure")
        yield b""  # pragma: no cover


class Upstream:
    def __init__(self, response):
        self.response = response

    def build_request(self, *args, **kwargs):
        return httpx.Request(*args, **kwargs)

    async def send(self, *args, **kwargs):
        return self.response

    async def post(self, *args, **kwargs):
        return httpx.Response(
            200,
            json={"kv_transfer_params": {"remote_block_ids": [1]}},
            request=httpx.Request("POST", "http://prefill"),
        )


def router_args(tmp_path):
    return argparse.Namespace(
        workers=["http://worker"],
        policy="cost",
        capacity_tokens=8192,
        prefill_ms=0.1,
        decode_ms=1,
        balance_abs=4,
        balance_rel=1.5,
        eviction_weight=1,
        ttl=60,
        log=tmp_path / "router.jsonl",
    )


@pytest.mark.parametrize("stream", [False, True])
def test_router_read_failure_releases_reservation(tmp_path, stream):
    app = router.create_app(router_args(tmp_path))
    response = httpx.Response(200, stream=BrokenStream())
    with TestClient(app) as client:
        app.state.client = Upstream(response)
        with pytest.raises(httpx.ReadError):
            client.post("/v1/completions", json={"prompt": [1] * 32, "stream": stream})
        assert client.get("/health").json()["inflight"] == 0


def test_router_incomplete_stream_does_not_populate_directory(tmp_path):
    args = router_args(tmp_path)
    app = router.create_app(args)
    with TestClient(app) as client:
        app.state.client = Upstream(
            httpx.Response(200, content=b'data: {"choices": []}\n\n')
        )
        for _ in range(2):
            assert (
                client.post(
                    "/v1/completions", json={"prompt": [1] * 32, "stream": True}
                ).status_code
                == 200
            )
        assert client.get("/health").json()["inflight"] == 0
    events = [json.loads(x) for x in args.log.read_text().splitlines()]
    assert all(
        e["scores"][0]["cached_tokens_est"] == 0
        for e in events
        if e["event"] == "dispatch"
    )
    assert all(not e["success"] for e in events if e["event"] == "finish")


def test_pd_malformed_decode_releases_both_ledgers(tmp_path):
    args = argparse.Namespace(
        prefillers=["http://p"],
        decoders=["http://d"],
        prefill_policy="rr",
        decode_policy="work",
        capacity_tokens=8192,
        prefill_ms=0.1,
        decode_ms=1,
        log=tmp_path / "pd.jsonl",
    )
    app = pd_proxy.create_app(args)
    with TestClient(app) as client:
        app.state.client = Upstream(httpx.Response(200, content=b"invalid JSON"))
        with pytest.raises(ValueError):
            client.post("/v1/completions", json={"prompt": [1] * 32})
        health = client.get("/health").json()
        assert health["prefill_inflight"] == health["decode_inflight"] == 0
    assert not json.loads(args.log.read_text().splitlines()[-1])["success"]


def test_adapter_sets_skip_before_base_tracker_and_freezes(monkeypatch):
    from .lmcache_adapter import LMCacheConnectorV1, WorkingSetLMCacheConnector

    monkeypatch.setattr(
        LMCacheConnectorV1, "get_num_new_matched_tokens", lambda *a: (0, False)
    )
    adapter = object.__new__(WorkingSetLMCacheConnector)
    adapter._admission = WriteAdmission(1024, 1, 256, mode="fixed", write_threshold=2)
    adapter._lmcache_engine = SimpleNamespace(
        load_specs={"id": SimpleNamespace(lmcache_cached_tokens=0)}
    )
    adapter._log = io.StringIO()
    adapter._telemetry = lambda: TierTelemetry(0, 4, 0, 0)
    request = SimpleNamespace(
        request_id="id",
        prompt_token_ids=[1] * 1024,
        kv_transfer_params={"experiment.session_id": "session"},
        sampling_params=SimpleNamespace(extra_args={"existing": 42}),
    )
    assert adapter.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.sampling_params.extra_args == {
        "existing": 42,
        "kv_transfer_params": {"lmcache.skip_save": True},
    }
    adapter.get_num_new_matched_tokens(request, 0)
    assert len(adapter._log.getvalue().splitlines()) == 1
