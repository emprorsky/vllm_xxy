"""Byte-exact round trips through installed LMCache connectors, without a model."""

import argparse
import json
import time
from pathlib import Path

import torch
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemGPUConnectorV3,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata


def check(version, packed, staging, tokens, skip, dtype=torch.bfloat16, content=None):
    device = torch.device("cuda:0")
    layers, blocks, block, heads, width = 4, 64, 16, 2, 128
    generator = torch.Generator(device=device).manual_seed(17)
    shape = (
        (blocks, block, heads, content or 2 * width)
        if packed
        else (2, blocks, block, heads, width)
    )
    caches = [
        torch.randint(-127, 128, shape, generator=generator, device=device).to(dtype)
        for _ in range(layers)
    ]
    original = [t.clone() for t in caches]
    metadata = LMCacheMetadata(
        model_name="synthetic-byte-proof",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=(layers, 2, 256, heads, width),
    )
    cls = VLLMPagedMemGPUConnectorV3 if version == 3 else VLLMPagedMemGPUConnectorV2
    connector = cls.from_metadata(
        metadata, use_gpu=staging, device=device, layout_hints={"kv_layout": "NHD"}
    )
    connector.initialize_kvcaches_ptr(kvcaches=caches)
    if version == 3:
        connector._initialize_kv_cache_pointers()
    shape = metadata.get_shapes(tokens)[0]
    raw = torch.full(
        (shape.numel() * dtype.itemsize,), 0xA5, dtype=torch.uint8, pin_memory=True
    )
    obj = TensorMemoryObj(
        raw,
        MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=0,
            phy_size=raw.numel(),
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
            shapes=[shape],
            dtypes=[dtype],
        ),
        None,
    )
    slots = torch.randperm(blocks * block, generator=generator, device=device)[:tokens]
    connector.batched_from_gpu([obj], [0], [tokens], slot_mapping=slots)
    torch.cuda.synchronize()
    for cache in caches:
        cache.zero_()
    connector.batched_to_gpu(
        [obj], [0], [tokens], slot_mapping=slots, vllm_cached_tokens=skip
    )
    torch.cuda.synchronize()
    selected = slots[skip:]
    mismatched = 0
    checked = 0
    for restored, source in zip(caches, original):
        expected = torch.zeros_like(source)
        if packed:
            expected.flatten(0, 1)[selected] = source.flatten(0, 1)[selected]
        else:
            expected.flatten(1, 2)[:, selected] = source.flatten(1, 2)[:, selected]
        mismatched += int(
            (expected.view(torch.uint8) != restored.view(torch.uint8)).sum()
        )
        checked += expected.numel() * dtype.itemsize
    return {
        "connector_version": version,
        "packed": packed,
        "gpu_staging": staging,
        "tokens": tokens,
        "skip_prefix_tokens": skip,
        "memory_shape": list(shape),
        "engine_format": str(connector.engine_kv_format),
        "dtype": str(dtype),
        "content_width": content or 2 * width,
        "mismatched_bytes": mismatched,
        "checked_bytes": checked,
        "passed": mismatched == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-manifest", type=Path)
    parser.add_argument("--include-quantized", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.wait_manifest:
        while True:
            if args.wait_manifest.exists():
                previous = json.loads(args.wait_manifest.read_text())
                if (
                    previous["status"] in {"completed", "failed"}
                    and "ended_at" in previous
                ):
                    break
            time.sleep(2)
    results = []
    for version in [2, 3]:
        for packed in [False, True]:
            for staging in [False, True]:
                for tokens, skip in [(256, 0), (73, 16)]:
                    try:
                        row = check(version, packed, staging, tokens, skip)
                    except Exception as exc:
                        row = dict(
                            connector_version=version,
                            packed=packed,
                            gpu_staging=staging,
                            tokens=tokens,
                            skip_prefix_tokens=skip,
                            passed=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    results.append(row)
                    print(json.dumps(row), flush=True)
    if args.include_quantized:
        for staging in [False, True]:
            for tokens, skip in [(256, 0), (73, 16)]:
                try:
                    row = check(3, True, staging, tokens, skip, torch.uint8, 264)
                except Exception as exc:
                    row = {"connector_version": 3, "passed": False, "error": str(exc)}
                results.append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    passed = all(r["passed"] for r in results if r["connector_version"] == 3)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "status": "completed" if passed else "failed",
                "ended_at": time.time(),
                "result": {
                    "passed": passed,
                    "cases": len(results),
                    "expected_v2_packed_failures": 4,
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
