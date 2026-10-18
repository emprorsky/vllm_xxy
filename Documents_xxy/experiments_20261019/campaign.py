"""Run independent stages sequentially and refresh reviewable evidence."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from .suite import ROOT, execute
from .summarize import summarize


async def campaign(args):
    if args.wait_manifest:
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if args.wait_manifest.exists():
                value = json.loads(args.wait_manifest.read_text())
                if value["status"] in {"completed", "failed"} and "ended_at" in value:
                    break
            await asyncio.sleep(2)
        else:
            raise TimeoutError("previous owned run did not finish")
    for stage in args.stages:
        progress = {"stage": stage, "status": "running", "updated_at": time.time()}
        (args.output.parent / "campaign_progress.json").write_text(
            json.dumps(progress, indent=2)
        )
        print(json.dumps(progress), flush=True)
        await execute(
            argparse.Namespace(
                stage=stage,
                output=args.output,
                repeats=args.repeats,
                only=None,
                timeout=1200,
            )
        )
        summarize(args.output, args.output.parent / "summary")
    (args.output.parent / "campaign_progress.json").write_text(
        json.dumps({"status": "finished", "updated_at": time.time()})
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        default=[
            "offload",
            "routing",
            "combined",
            "pd1",
            "pd2",
            "replicas",
            "capacity",
            "chunking",
        ],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--wait-manifest", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Campaign interpreter: {sys.executable}", flush=True)
    asyncio.run(campaign(args))


if __name__ == "__main__":
    main()
