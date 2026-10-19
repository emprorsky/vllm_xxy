"""Byte-exact round trips through installed LMCache connectors, without a model."""

import argparse
import json
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


def check(version, packed, staging, tokens, skip):
    device = torch.device("cuda:0")
    layers, blocks, block, heads, width = 4, 64, 16, 2, 128
    generator = torch.Generator(device=device).manual_seed(17)
    shape = (
        (blocks, block, heads, 2 * width)
        if packed
        else (2, blocks, block, heads, width)
    )
    caches = [
        torch.randint(-127, 128, shape, generator=generator, device=device).to(
            torch.bfloat16
        )
        for _ in range(layers)
    ]
    original = [t.clone() for t in caches]
    metadata = LMCacheMetadata(
        model_name="synthetic-byte-proof",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
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
    raw = torch.full((shape.numel() * 2,), 0xA5, dtype=torch.uint8, pin_memory=True)
    obj = TensorMemoryObj(
        raw,
        MemoryObjMetadata(
            shape=shape,
            dtype=torch.bfloat16,
            address=0,
            phy_size=raw.numel(),
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
            shapes=[shape],
            dtypes=[torch.bfloat16],
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
            (expected.view(torch.int16) != restored.view(torch.int16)).sum()
        )
        checked += expected.numel()
    return {
        "connector_version": version,
        "packed": packed,
        "gpu_staging": staging,
        "tokens": tokens,
        "skip_prefix_tokens": skip,
        "memory_shape": list(shape),
        "engine_format": str(connector.engine_kv_format),
        "mismatched_elements": mismatched,
        "checked_elements": checked,
        "passed": mismatched == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
