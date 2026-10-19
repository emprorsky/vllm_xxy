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


def test_runner_rejects_live_port_before_launch():
    import socket

    from .run_experiment import check_ports_available

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="already occupied"):
            check_ports_available({"workers": [{"port": port}]})


@pytest.mark.parametrize("gpu_prefix", [0, 32, 288, 512])
def test_disk_capped_load_preserves_absolute_endpoint(monkeypatch, gpu_prefix):
    from .disk_adapter import DiskWorkingSetLMCacheConnector
    from .lmcache_adapter import WorkingSetLMCacheConnector

    endpoint = (gpu_prefix + 2048) // 256 * 256
    connector = DiskWorkingSetLMCacheConnector.__new__(DiskWorkingSetLMCacheConnector)
    connector._chunk = 256
    connector._log = io.StringIO()
    connector._lmcache_engine = SimpleNamespace(
        _max_tokens_per_load=2048,
        load_specs={
            "r": SimpleNamespace(
                vllm_cached_tokens=gpu_prefix, lmcache_cached_tokens=endpoint
            )
        },
    )
    monkeypatch.setattr(
        WorkingSetLMCacheConnector,
        "get_num_new_matched_tokens",
        lambda *args: (2048, False),
    )
    request = SimpleNamespace(request_id="r", num_tokens=4000)
    matched, asynchronous = connector.get_num_new_matched_tokens(request, gpu_prefix)
    assert matched + gpu_prefix == endpoint
    assert 0 <= matched <= 2048
    assert not asynchronous


def test_readiness_rejects_another_process_health_endpoint():
    import asyncio
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from .run_experiment import wait_ready

    class Healthy(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Healthy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    process = subprocess.Popen(["sleep", "10"])
    try:
        with pytest.raises(RuntimeError, match="not owned"):
            asyncio.run(wait_ready(server.server_port, process, timeout=2))
    finally:
        process.terminate()
        process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("contents", [b"abcd", b"ab", b"abce", None])
def test_disk_probe_only_accepts_complete_payload(tmp_path, contents):
    """A missing/short file cannot turn old CPU staging bytes into valid KV."""
    import hashlib
    import threading

    from .disk_adapter import DiskWorkingSetLMCacheConnector

    class LocalDiskBackend:
        use_odirect = False
        max_cache_size = 1024
        remove = staticmethod(lambda key, force=True: False)
        write_file = staticmethod(lambda buffer, path: None)

    disk = LocalDiskBackend()
    connector = DiskWorkingSetLMCacheConnector.__new__(DiskWorkingSetLMCacheConnector)
    connector._observer_started = False
    connector._lmcache_engine = SimpleNamespace(
        lmcache_engine=SimpleNamespace(
            storage_manager=SimpleNamespace(storage_backends={"disk": disk})
        )
    )
    connector._capacity = 1024
    connector._telemetry_path = tmp_path / "cpu-test.json"
    connector._log = io.StringIO()
    connector._stop_event = threading.Event()
    connector._stop_event.set()
    connector._start_observer()
    path = tmp_path / "payload"
    connector._verify_file_integrity = True
    connector._file_digests = {str(path): hashlib.sha256(b"abcd").hexdigest()}
    if contents is not None:
        path.write_bytes(contents)
    buffer = bytearray(b"xxxx")
    if contents == b"abcd":
        disk.read_file("key", buffer, path)
        assert bytes(buffer) == contents
        assert json.loads(connector._log.getvalue())["bytes"] == 4
    else:
        with pytest.raises((EOFError, FileNotFoundError, ValueError)):
            disk.read_file("key", buffer, path)
        assert not connector._log.getvalue()


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


def test_pd_preserves_session_but_remote_metadata_remains_authoritative(tmp_path):
    recorded = []

    class Capture(Upstream):
        async def post(self, *args, **kwargs):
            recorded.append(kwargs["json"]["kv_transfer_params"])
            return await super().post(*args, **kwargs)

        async def send(self, request, **kwargs):
            recorded.append(json.loads(request.content)["kv_transfer_params"])
            return self.response

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
        app.state.client = Capture(httpx.Response(200, json={"choices": []}))
        response = client.post(
            "/v1/completions",
            json={
                "prompt": [1] * 32,
                "kv_transfer_params": {
                    "experiment.session_id": "session",
                    "lmcache.skip_save": True,
                    "remote_block_ids": [999],
                },
            },
        )
        assert response.status_code == 200
    assert recorded[0]["do_remote_decode"] is True
    assert "remote_block_ids" not in recorded[0]
    assert recorded[1]["remote_block_ids"] == [1]
    assert all(row["experiment.session_id"] == "session" for row in recorded)
    assert all(row["lmcache.skip_save"] for row in recorded)


@pytest.mark.parametrize("optional_pd", [False, True])
def test_adapter_sets_skip_before_base_tracker_and_freezes(monkeypatch, optional_pd):
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
    adapter._optional_cpu_in_multi = optional_pd
    request = SimpleNamespace(
        request_id="id",
        prompt_token_ids=[1] * 1024,
        kv_transfer_params={"experiment.session_id": "session"},
        sampling_params=SimpleNamespace(extra_args={"existing": 42}),
    )
    expected_transfer = {"lmcache.skip_save": True}
    if optional_pd:
        request.kv_transfer_params["do_remote_decode"] = True
        request.sampling_params.extra_args["kv_transfer_params"] = {
            "do_remote_decode": True
        }
        expected_transfer["do_remote_decode"] = True
    assert adapter.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.sampling_params.extra_args == {
        "existing": 42,
        "kv_transfer_params": expected_transfer,
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
    import torch
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
    adapter._kv_bytes = 36864
    adapter._log = io.StringIO()
    adapter._start_observer = lambda: calls.append("observer")
    adapter._lmcache_engine = SimpleNamespace(
        lmcache_engine=SimpleNamespace(
            gpu_connector=connector,
            metadata=SimpleNamespace(
                get_shapes=lambda n: [torch.Size((1, 36, n, 512))],
                get_dtypes=lambda: [torch.bfloat16],
            ),
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


def test_cpu_payload_accounting_includes_inline_int8_scales():
    import torch

    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    from .lmcache_adapter import dense_payload_bytes_per_token

    for dtype, content_bytes, expected in [
        (torch.bfloat16, None, 36864),
        (torch.uint8, 264, 19008),
    ]:
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=dtype,
            state_content_bytes=content_bytes,
        )
        config = SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[f"layer{i}" for i in range(36)],
                    kv_cache_spec=spec,
                )
            ]
        )
        assert dense_payload_bytes_per_token(config) == expected


def test_failed_correctness_dependency_prevents_performance_server_start(tmp_path):
    import asyncio

    from .run_experiment import run

    check = tmp_path / "check.json"
    check.write_text(json.dumps({"status": "failed", "result": {"passed": False}}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"workers": [], "required_checks": [str(check)]}))
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="required correctness check"):
        asyncio.run(run(argparse.Namespace(config=config, output=output)))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["commands"] == []


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
