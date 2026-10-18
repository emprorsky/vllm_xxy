from .event_directory import EventDirectory
from .policies import prefix_keys


def stored(blocks, tokens, parent=None, **kwargs):
    return {
        "type": "BlockStored",
        "block_hashes": blocks,
        "token_ids": tokens,
        "parent_block_hash": parent,
        "block_size": 16,
        "medium": "GPU",
        **kwargs,
    }


def test_native_hash_translation_remove_and_continuity():
    directory = EventDirectory(128)
    tokens = list(range(48))
    directory.ingest(0, [stored([11, 12], tokens[:32])], 0, b"a")
    directory.ingest(1, [stored([13], tokens[32:], parent=12)], 1, b"b")
    assert directory.match(prefix_keys(tokens), 2) == 0
    directory.observe(prefix_keys(tokens), 2)
    assert directory.match(prefix_keys(tokens), 2) == 48
    directory.ingest(2, [{"type": "BlockRemoved", "block_hashes": [11]}], 2, b"c")
    assert directory.match(prefix_keys(tokens), 3) == 0
    assert len(directory.blocks) == 2
    directory.ingest(3, [stored([11], tokens[:16])], 3, b"d")
    directory.observe(prefix_keys(tokens[:16]), 4)
    assert directory.match(prefix_keys(tokens), 4) == 48


def test_gap_and_new_epoch_invalidate_stale_prefixes():
    directory = EventDirectory(128)
    first = [1] * 16
    second = [2] * 16
    directory.ingest(0, [stored([1], first)], 0, b"a")
    directory.observe(prefix_keys(first), 0)
    assert directory.ingest(0, [stored([1], first)], 1, b"a") == "duplicate"
    assert directory.match(prefix_keys(first), 1) == 16
    directory.ingest(2, [stored([3], second, parent=1)], 2, b"c")
    assert directory.gaps == 1
    assert directory.match(prefix_keys(first), 2) == 0
    assert directory.unknown_parents == 1
    directory.ingest(0, [stored([4], second)], 3, b"new_epoch")
    assert directory.epochs == 1
    directory.observe(prefix_keys(second), 3)
    assert directory.match(prefix_keys(second), 3) == 16


def test_history_completion_cannot_create_confirmed_residency():
    directory = EventDirectory(128)
    keys = prefix_keys([1] * 32)
    directory.observe(keys, 0)
    assert directory.match(keys, 0) == 0
    directory.ingest(0, [stored([1], [1] * 16)], 0, b"a")
    directory.observe(keys, 1)
    assert directory.match(keys, 1) == 16
    assert directory.blocks[keys[0]].references == 2


def test_salt_and_partial_entries_are_excluded():
    directory = EventDirectory(128)
    directory.ingest(
        0,
        [
            stored([1], [1] * 16, extra_keys=[["salt"]]),
            stored([2], [2] * 8, block_size_override=8),
        ],
        0,
        b"a",
    )
    assert not directory.blocks
    assert directory.rejected == 2


def test_real_vllm_publisher_replay_and_live_removal(tmp_path):
    import asyncio
    import time

    from vllm.distributed.kv_events import (
        BlockRemoved,
        BlockStored,
        KVEventBatch,
        ZmqEventPublisher,
    )

    from .events_router import EventSubscriber

    async def exercise():
        endpoint = f"ipc://{tmp_path}/events"
        replay = f"ipc://{tmp_path}/replay"
        publisher = ZmqEventPublisher(0, endpoint=endpoint, replay_endpoint=replay)
        directory = EventDirectory(128)
        tokens = [42] * 32
        stored_event = BlockStored([11, 12], None, tokens, 16, None, "GPU", None)
        publisher.publish(KVEventBatch(time.time(), [stored_event]))
        subscriber = EventSubscriber(
            directory, endpoint, replay, (tmp_path / "events.jsonl").open("w")
        )
        try:
            await asyncio.sleep(0.15)
            await subscriber.start()
            assert directory.match(prefix_keys(tokens), time.monotonic()) == 0
            directory.observe(prefix_keys(tokens), time.monotonic())
            assert directory.match(prefix_keys(tokens), time.monotonic()) == 32
            await asyncio.sleep(0.15)
            publisher.publish(KVEventBatch(time.time(), [BlockRemoved([11], "GPU")]))
            deadline = time.monotonic() + 2
            while (
                directory.match(prefix_keys(tokens), time.monotonic())
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.01)
            assert directory.match(prefix_keys(tokens), time.monotonic()) == 0
            assert subscriber.errors == 0
        finally:
            await subscriber.close()
            publisher.shutdown()

    asyncio.run(exercise())


def test_replay_intermediate_stores_cannot_become_visible_to_routing():
    import asyncio
    import io
    import time
    from types import SimpleNamespace

    import msgspec

    from .events_router import EventSubscriber

    async def exercise():
        reached = asyncio.Event()
        release = asyncio.Event()
        tokens = [42] * 16
        messages = [
            [
                b"",
                b"",
                (0).to_bytes(8, "big"),
                msgspec.msgpack.encode([time.time(), [stored([1], tokens)]]),
            ],
            [
                b"",
                b"",
                (1).to_bytes(8, "big"),
                msgspec.msgpack.encode(
                    [time.time(), [{"type": "BlockRemoved", "block_hashes": [1]}]]
                ),
            ],
            [b"", b"", (-1).to_bytes(8, "big", signed=True), b""],
        ]

        class ReplaySocket:
            def connect(self, endpoint):
                pass

            async def send_multipart(self, frames):
                pass

            async def recv_multipart(self):
                if len(messages) == 2:
                    reached.set()
                    await release.wait()
                return messages.pop(0)

            def close(self, linger):
                pass

        subscriber = object.__new__(EventSubscriber)
        subscriber.directory = EventDirectory(128)
        subscriber.ctx = SimpleNamespace(socket=lambda kind: ReplaySocket())
        subscriber.replay_endpoint = "unused"
        subscriber.log = io.StringIO()
        task = asyncio.create_task(subscriber.replay())
        await asyncio.wait_for(reached.wait(), 1)
        subscriber.directory.observe(prefix_keys(tokens), time.monotonic())
        assert subscriber.directory.match(prefix_keys(tokens), time.monotonic()) == 0
        release.set()
        await task
        assert not subscriber.directory.blocks

    asyncio.run(exercise())


def test_completion_before_delayed_event_is_usable_but_removal_revokes_proof():
    directory = EventDirectory(128)
    tokens = [42] * 16
    keys = prefix_keys(tokens)
    directory.observe(keys, 0)
    assert directory.match(keys, 0) == 0
    directory.ingest(0, [stored([1], tokens)], 1, b"a")
    assert directory.match(keys, 1) == 16
    directory.ingest(1, [{"type": "BlockRemoved", "block_hashes": [1]}], 2, b"b")
    directory.ingest(2, [stored([1], tokens)], 3, b"new_pending_copy")
    assert directory.match(keys, 3) == 0
    directory.observe(keys, 4)
    assert directory.match(keys, 4) == 16
