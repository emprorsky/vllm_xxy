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


def test_packed_layout_rejects_v2_before_storage_registration(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from .lmcache_adapter import LMCacheConnectorV1, WorkingSetLMCacheConnector

    calls = []
    monkeypatch.setattr(
        LMCacheConnectorV1, "register_kv_caches", lambda *a: calls.append("storage")
    )
    adapter = object.__new__(WorkingSetLMCacheConnector)
    adapter._exp_role = KVConnectorRole.WORKER
    adapter._lmcache_engine = SimpleNamespace(
        lmcache_engine=SimpleNamespace(gpu_connector=object())
    )
    with pytest.raises(ValueError, match="Packed K/V requires"):
        adapter.register_kv_caches({"layer": SimpleNamespace(ndim=4)})
    assert not calls


def test_packed_layout_is_discovered_before_storage_allocates(monkeypatch):
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV3

    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from .lmcache_adapter import LMCacheConnectorV1, WorkingSetLMCacheConnector

    calls = []
    connector = object.__new__(VLLMPagedMemGPUConnectorV3)
    connector.initialize_kvcaches_ptr = lambda **kw: calls.append("views")
    connector._initialize_kv_cache_pointers = lambda: calls.append("shapes")
    connector.engine_kv_format = "packed"
    adapter = object.__new__(WorkingSetLMCacheConnector)
    adapter._exp_role = KVConnectorRole.WORKER
    adapter._chunk = 256
    adapter._log = io.StringIO()
    adapter._start_observer = lambda: calls.append("observer")
    adapter._lmcache_engine = SimpleNamespace(
        lmcache_engine=SimpleNamespace(
            gpu_connector=connector,
            metadata=SimpleNamespace(get_shapes=lambda n: [(1, 36, n, 512)]),
        )
    )
    monkeypatch.setattr(
        LMCacheConnectorV1, "register_kv_caches", lambda *a: calls.append("storage")
    )
    adapter.register_kv_caches({"layer": SimpleNamespace(ndim=4)})
    assert calls == ["views", "shapes", "storage", "observer"]
    assert json.loads(adapter._log.getvalue())["cpu_shapes"] == [[1, 36, 256, 512]]


def test_stream_progress_counts_token_ids_instead_of_sse_messages(tmp_path):
    from .policies import PlacementPolicy

    observed = []

    class Observed(PlacementPolicy):
        def observe_output(self, request_id, emitted_tokens):
            observed.append(emitted_tokens)
            super().observe_output(request_id, emitted_tokens)

    args = router_args(tmp_path)
    args.policy = "work_remaining"
    policy = Observed(1, args.policy)
    app = router.create_app(args, policy=policy)
    data = (
        'data: {"choices": [{"token_ids": [1, 2]}]}\n\n'
        'data: {"choices": [{"token_ids": [3, 4, 5]}]}\n\n'
        'data: {"choices": [], "usage": {"completion_tokens": 5}}\n\n'
        "data: [DONE]\n\n"
    )
    with TestClient(app) as client:
        app.state.client = Upstream(httpx.Response(200, content=data.encode()))
        response = client.post(
            "/v1/completions", json={"prompt": [1] * 32, "stream": True}
        )
        assert response.status_code == 200
        assert not policy.reservations and not policy.output_progress
    assert observed == [2, 5, 5]
    assert (
        json.loads(args.log.read_text().splitlines()[-1])["observed_output_tokens"] == 5
    )


def test_code_probe_checks_execution_and_rejects_imports():
    from .quality_coding import execute_code

    tests = [([[1, 2, 3, 4]], 6), ([[]], 0)]
    assert execute_code(
        "def sum_even(nums):\n return sum(x for x in nums if x % 2 == 0)",
        "sum_even",
        tests,
    )["passed"]
    assert not execute_code("def sum_even(nums): return 6", "sum_even", tests)["passed"]
    assert not execute_code(
        "import os\ndef sum_even(nums): return 0", "sum_even", tests
    )["passed"]


@pytest.mark.parametrize("kind", ["router", "pd"])
def test_cancellation_during_upstream_wait_releases_ledger(tmp_path, kind):
    import asyncio

    async def exercise():
        ready = asyncio.Event()

        class Waiting(Upstream):
            async def send(self, *args, **kwargs):
                ready.set()
                await asyncio.Event().wait()

            async def post(self, *args, **kwargs):
                ready.set()
                await asyncio.Event().wait()

        if kind == "router":
            app = router.create_app(router_args(tmp_path))
        else:
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
        async with app.router.lifespan_context(app):
            app.state.client = Waiting(None)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                task = asyncio.create_task(
                    client.post("/v1/completions", json={"prompt": [1] * 32})
                )
                await asyncio.wait_for(ready.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                health = (await client.get("/health")).json()
                if kind == "router":
                    assert health["inflight"] == 0
                else:
                    assert health["prefill_inflight"] == health["decode_inflight"] == 0

    asyncio.run(exercise())
