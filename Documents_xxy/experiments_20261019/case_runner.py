"""Run a frozen selection of optional or confirmatory cases, interleaved."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from .extra_cases import configurations
from .run_experiment import run
from .suite import ROOT, cases
from .summarize import summarize

CORE = {
    "single",
    "offload",
    "capacity",
    "routing",
    "combined",
    "chunking",
    "pd1",
    "pd2",
    "replicas",
}


async def execute(args):
    selected = []
    for stage in args.stages:
        selected.extend(cases(stage) if stage in CORE else configurations(stage))
    if args.names:
        selected = [(n, c) for n, c in selected if n in args.names]
    if not selected:
        raise ValueError("no selected cases")
    remap = dict(pair.split(":") for pair in args.gpu_remap)
    for _, config in selected:
        for worker in config["workers"]:
            worker["gpu"] = int(remap.get(str(worker["gpu"]), worker["gpu"]))
            worker["port"] += args.worker_port_offset
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": time.time(),
        "repeats": args.repeats,
        "prefix": args.prefix,
        "cases": [{"name": n, "config": c} for n, c in selected],
    }
    (args.output / f"{args.prefix}selection_{int(time.time())}.json").write_text(
        json.dumps(plan, indent=2)
    )
    dependencies = (
        [args.wait_manifest] if args.wait_manifest else []
    ) + args.wait_manifests
    for dependency in dependencies:
        deadline = time.monotonic() + 10800
        while time.monotonic() < deadline:
            if dependency.exists():
                previous = json.loads(dependency.read_text())
                if (
                    previous["status"] in {"completed", "failed"}
                    and "ended_at" in previous
                ):
                    break
            await asyncio.sleep(2)
        else:
            raise TimeoutError("preceding campaign did not finish within 3h")
    for repeat in range(args.repeats):
        ordered = selected if repeat % 2 == 0 else list(reversed(selected))
        for name, config in ordered:
            run_id = f"{args.prefix}{name}_rep{repeat + 1:02d}"
            destination = args.output / run_id
            if (destination / "manifest.json").exists():
                print(json.dumps({"event": "skip_existing", "run": run_id}), flush=True)
                continue
            config_path = args.output / f"{run_id}.config.json"
            config_path.write_text(json.dumps(config, indent=2))
            progress = {"run": run_id, "status": "running", "timestamp": time.time()}
            progress_path = args.output.parent / "case_runner_progress.json"
            progress_path.write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
            try:
                await asyncio.wait_for(
                    run(argparse.Namespace(config=config_path, output=destination)),
                    1200,
                )
                progress["status"] = "completed"
            except Exception as exc:
                progress.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            progress_path.write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
            summarize(args.output, args.output.parent / "summary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--wait-manifest", type=Path)
    parser.add_argument("--wait-manifests", nargs="*", type=Path, default=[])
    parser.add_argument("--gpu-remap", nargs="*", default=[], metavar="SOURCE:TARGET")
    parser.add_argument("--worker-port-offset", type=int, default=0)
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
