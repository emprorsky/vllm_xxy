"""Launch owned server processes, replay a trace, and retain full manifests."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

from .replay_client import replay
from .validate_outputs import validate


async def wait_ready(port, process, timeout=300):
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
        while time.monotonic() - start < timeout:
            if process.poll() is not None:
                raise RuntimeError(f"server port {port} exited {process.returncode}")
            try:
                response = await client.get(f"http://127.0.0.1:{port}/health")
                if response.status_code == 200:
                    parent = psutil.Process(process.pid)
                    owners = [parent, *parent.children(recursive=True)]
                    owned = False
                    for owner in owners:
                        with contextlib.suppress(psutil.NoSuchProcess):
                            owned |= any(
                                connection.laddr.port == port
                                and connection.status == psutil.CONN_LISTEN
                                for connection in owner.net_connections(kind="inet")
                            )
                    if not owned:
                        raise RuntimeError(
                            f"Health port {port} is not owned by launched process"
                        )
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"server port {port} not ready after {timeout}s")


def check_ports_available(config):
    ports = [w["port"] for w in config["workers"]]
    ports += [config[key]["port"] for key in ["router", "pd"] if key in config]
    if len(ports) != len(set(ports)):
        raise ValueError("Experiment contains duplicate service ports")
    for port in ports:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Experiment port already occupied: {port}") from exc


async def run(args):
    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(f"run directory already exists: {args.output}")
    args.output.mkdir(parents=True)
    if config.get("env", {}).get("LMCACHE_LOCAL_DISK") == "__PER_RUN_DISK_CACHE__":
        config["env"]["LMCACHE_LOCAL_DISK"] = str(args.output.resolve() / "disk-cache")
    process_list = []
    handles = []
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_OFFLINE"] = "1"
    env["KVEXP_STATS_DIR"] = str(args.output.resolve() / "admission")
    env.update(config.get("env", {}))
    manifest = {
        "started_at": time.time(),
        "config": config,
        "python": sys.executable,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "commands": [],
        "status": "starting",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "vllm", "transformers", "lmcache", "nixl"]
            if importlib.util.find_spec(name) is not None
        },
    }

    def save():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    save()
    try:
        for check_path in config.get("required_checks", []):
            check = json.loads(Path(check_path).read_text())
            if check.get("status") != "completed" or not check.get("result", {}).get(
                "passed"
            ):
                raise RuntimeError(
                    f"required correctness check did not pass: {check_path}"
                )
        check_ports_available(config)
        workers = config["workers"]
        for index, worker in enumerate(workers):
            argv = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                config["model"],
                "--served-model-name",
                "experiment",
                "--host",
                "127.0.0.1",
                "--port",
                str(worker["port"]),
            ]
            argv += config.get("server_args", []) + worker.get("args", [])
            worker_env = {
                **env,
                "CUDA_VISIBLE_DEVICES": str(worker["gpu"]),
                **worker.get("env", {}),
            }
            handle = (args.output / f"server_{index}.log").open("w")
            handles.append(handle)
            process = subprocess.Popen(
                argv,
                env=worker_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process_list.append(process)
            manifest["commands"].append(
                {"argv": argv, "gpu": worker["gpu"], "pid": process.pid}
            )
            save()
        await asyncio.gather(
            *(
                wait_ready(w["port"], p, config.get("startup_timeout", 300))
                for w, p in zip(workers, process_list)
            )
        )
        endpoints = [f"http://127.0.0.1:{w['port']}" for w in workers]
        endpoint = endpoints[0]
        if config.get("router") or config.get("pd"):
            router = config.get("router", config.get("pd"))
            port = router.get("port", 8190)
            if config.get("pd"):
                prefills = [e for e, w in zip(endpoints, workers) if w["role"] == "P"]
                decodes = [e for e, w in zip(endpoints, workers) if w["role"] == "D"]
                argv = [
                    sys.executable,
                    "-m",
                    "Documents_xxy.experiments_20261019.pd_proxy",
                    "--prefillers",
                    *prefills,
                    "--decoders",
                    *decodes,
                    "--prefill-policy",
                    router.get("prefill_policy", "rr"),
                    "--decode-policy",
                    router.get("decode_policy", "rr"),
                    "--port",
                    str(port),
                    "--log",
                    str(args.output / "router_decisions.jsonl"),
                ]
            else:
                argv = [
                    sys.executable,
                    "-m",
                    "Documents_xxy.experiments_20261019.router",
                    "--workers",
                    *endpoints,
                    "--policy",
                    router.get("policy", "cost"),
                    "--port",
                    str(port),
                    "--log",
                    str(args.output / "router_decisions.jsonl"),
                ]
            for name in [
                "capacity_tokens",
                "prefill_ms",
                "decode_ms",
                "balance_abs",
                "balance_rel",
                "eviction_weight",
                "ttl",
                "cost_profile",
                "max_batch",
            ]:
                if name in router and not (
                    config.get("pd")
                    and name
                    in {
                        "balance_abs",
                        "balance_rel",
                        "eviction_weight",
                        "ttl",
                        "cost_profile",
                        "max_batch",
                    }
                ):
                    argv.extend(["--" + name.replace("_", "-"), str(router[name])])
            handle = (args.output / "router.log").open("w")
            handles.append(handle)
            process = subprocess.Popen(
                argv,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process_list.append(process)
            manifest["commands"].append({"argv": argv, "pid": process.pid})
            await wait_ready(port, process, 30)
            endpoint = f"http://127.0.0.1:{port}"
        if config.get("validate_outputs", True):
            validation_endpoints = [endpoint] if config.get("pd") else endpoints
            await validate(
                config["model"],
                validation_endpoints,
                args.output / "quality_probes.json",
            )
            manifest["basic_output_gate"] = "passed"
        if config.get("coding_quality_probes", False):
            from .quality_coding import evaluate

            manifest["coding_quality"] = await evaluate(
                config["model"],
                [endpoint] if config.get("pd") else endpoints,
                args.output / "coding_quality.json",
            )
        if config.get("pd_fidelity_probes", False):
            from .pd_fidelity import validate_pd

            manifest["pd_fidelity"] = await validate_pd(
                config["model"],
                endpoint,
                prefills[0],
                decodes,
                args.output / "pd_fidelity.json",
            )
            if not manifest["pd_fidelity"]["passed"]:
                raise RuntimeError("P/D fidelity probes did not pass")
        if env.get("KVEXP_EVENT_ENDPOINTS"):
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                response = await client.get(endpoint + "/kv-event-state")
                response.raise_for_status()
                (args.output / "event_state_before.json").write_text(response.text)
        manifest["status"] = "replaying"
        save()
        replay_args = argparse.Namespace(
            trace=Path(config["trace"]),
            output=args.output / "replay",
            endpoint=endpoint,
            metrics_endpoints=endpoints,
            model="experiment",
            concurrency=config.get("concurrency", 8),
            gap_scale=config.get("gap_scale", 1),
            timeout=config.get("request_timeout", 300),
            slo_ttft=config.get("slo_ttft", 5),
            slo_tpot=config.get("slo_tpot", 0.1),
        )
        result = await replay(replay_args)
        if env.get("KVEXP_EVENT_ENDPOINTS"):
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                response = await client.get(endpoint + "/kv-event-state")
                response.raise_for_status()
                (args.output / "event_state_after.json").write_text(response.text)
        if result["successful"] != result["requests_expected"]:
            raise RuntimeError("replay contains failed or unattempted requests")
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for process in reversed(process_list):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        for process in process_list:
            try:
                await asyncio.to_thread(process.wait, timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        for handle in handles:
            handle.close()
        manifest["ended_at"] = time.time()
        save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
