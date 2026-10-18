"""Low-frequency host/GPU evidence; summed RSS includes shared-page duplication."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def number(path):
    try:
        return int(Path(path).read_text())
    except (OSError, ValueError):
        return None


def monitor(args):
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", buffering=1) as handle:
        while time.monotonic() - start < args.seconds:
            active = []
            pids = set()
            for path in args.runs.glob("*/manifest.json"):
                try:
                    manifest = json.loads(path.read_text())
                    if manifest["status"] not in {"starting", "replaying"}:
                        continue
                    active.append(path.parent.name)
                    for command in manifest["commands"]:
                        process = psutil.Process(command["pid"])
                        pids.add(process.pid)
                        pids.update(
                            child.pid for child in process.children(recursive=True)
                        )
                except (OSError, ValueError, psutil.NoSuchProcess):
                    continue
            processes = []
            for pid in pids:
                try:
                    process = psutil.Process(pid)
                    info = process.memory_info()
                    cpu = process.cpu_times()
                    processes.append(
                        {
                            "pid": pid,
                            "rss_bytes": info.rss,
                            "cpu_s": cpu.user + cpu.system,
                            "threads": process.num_threads(),
                        }
                    )
                except psutil.NoSuchProcess:
                    continue
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=10,
            )
            record = {
                "timestamp": time.time(),
                "active_runs": active,
                "processes": processes,
                "gpu_csv_columns": [
                    "index",
                    "memory_MiB",
                    "utilization_pct",
                    "power_W",
                    "temp_C",
                    "sm_MHz",
                    "memory_MHz",
                ],
                "gpu_csv": result.stdout.strip().splitlines(),
                "cgroup_memory_bytes": number("/sys/fs/cgroup/memory.current"),
                "shm_used_bytes": psutil.disk_usage("/dev/shm").used,
                "host_cpu_percent": psutil.cpu_percent(),
                "rss_sum_scope": "owned_process_RSS_shared_pages_may_duplicate",
            }
            handle.write(json.dumps(record) + "\n")
            time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=14400)
    parser.add_argument("--interval", type=float, default=5)
    monitor(parser.parse_args())


if __name__ == "__main__":
    main()
