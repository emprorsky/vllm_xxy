"""Optional KV-events subscription; history and confirmed modes share overhead."""

import asyncio
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager, suppress

import msgspec
import zmq
import zmq.asyncio

from .event_directory import EventDirectory
from .policies import PlacementPolicy


class EventSubscriber:
    def __init__(self, directory, endpoint, replay_endpoint, log):
        self.directory = directory
        self.endpoint = endpoint
        self.replay_endpoint = replay_endpoint
        self.log = log
        self.ctx = zmq.asyncio.Context.instance()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVHWM, 100000)
        self.socket.connect(endpoint)
        self.task = None
        self.errors = 0

    def apply(self, frames, target=None):
        _, sequence, payload = frames[-3:]
        sequence = int.from_bytes(sequence, "big", signed=True)
        if sequence < 0:
            return None
        batch = msgspec.msgpack.decode(payload)
        events = batch[1] if isinstance(batch, list) else batch["events"]
        directory = target if target is not None else self.directory
        status = directory.ingest(
            sequence,
            events,
            time.monotonic(),
            hashlib.blake2b(payload, digest_size=16).digest(),
        )
        self.log.write(
            json.dumps(
                {
                    "sequence": sequence,
                    "status": status,
                    "arrival_monotonic": time.monotonic(),
                    "payload_msgpack_hex": payload.hex(),
                }
            )
            + "\n"
        )
        return status

    async def replay(self):
        socket = self.ctx.socket(zmq.DEALER)
        socket.connect(self.replay_endpoint)
        try:
            await socket.send_multipart([b"", (0).to_bytes(8, "big")])
            self.directory.clear()
            snapshot = EventDirectory(
                self.directory.capacity_blocks * self.directory.block_size,
                self.directory.block_size,
                self.directory.ttl_s,
            )
            while True:
                frames = await asyncio.wait_for(socket.recv_multipart(), 3)
                if int.from_bytes(frames[-2], "big", signed=True) < 0:
                    break
                self.apply(frames, target=snapshot)
            # Publish a complete replay atomically; historical intermediate
            # stores must never be visible before their later removals.
            for name in ["blocks", "ready", "identities", "sequence", "seen"]:
                setattr(self.directory, name, getattr(snapshot, name))
            for name in ["gaps", "epochs", "unknown_parents", "rejected"]:
                setattr(
                    self.directory,
                    name,
                    getattr(self.directory, name) + getattr(snapshot, name),
                )
        finally:
            socket.close(linger=0)

    async def start(self):
        try:
            await self.replay()
        except (TimeoutError, zmq.ZMQError):
            self.errors += 1
            self.directory.clear()
        self.task = asyncio.create_task(self.listen())

    async def listen(self):
        while True:
            try:
                frames = await self.socket.recv_multipart()
                status = self.apply(frames)
                if status == "gap":
                    await self.replay()
            except asyncio.CancelledError:
                raise
            except (TimeoutError, ValueError, TypeError, KeyError, zmq.ZMQError):
                self.errors += 1
                self.directory.clear()

    async def close(self):
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        self.socket.close(linger=0)
        self.log.close()


def create_event_app(args, create_base_app):
    endpoints = json.loads(os.environ["KVEXP_EVENT_ENDPOINTS"])
    replays = json.loads(os.environ["KVEXP_EVENT_REPLAY_ENDPOINTS"])
    mode = os.environ.get("KVEXP_DIRECTORY_MODE", "events")
    if (
        mode not in {"history", "events"}
        or len(endpoints) != len(args.workers)
        or len(replays) != len(args.workers)
    ):
        raise ValueError("event endpoints/mode must match the immutable worker pool")
    policy = PlacementPolicy(
        len(args.workers),
        args.policy,
        args.capacity_tokens,
        prefill_ms_per_token=args.prefill_ms,
        decode_ms_per_token=args.decode_ms,
        balance_abs=args.balance_abs,
        balance_rel=args.balance_rel,
        eviction_weight=args.eviction_weight,
        ttl_s=args.ttl,
    )
    directories = [
        EventDirectory(args.capacity_tokens, ttl_s=args.ttl) for _ in endpoints
    ]
    if mode == "events":
        policy.directories = directories
    args.directory_source = (
        "events_and_completed_requests"
        if mode == "events"
        else "history_with_event_shadow"
    )
    app = create_base_app(args, policy=policy)
    base_lifespan = app.router.lifespan_context
    subscribers = [
        EventSubscriber(
            d,
            endpoint,
            replay,
            args.log.with_name(f"kv_events_worker_{index}.jsonl").open(
                "a", buffering=1
            ),
        )
        for index, (d, endpoint, replay) in enumerate(
            zip(directories, endpoints, replays)
        )
    ]

    @asynccontextmanager
    async def lifespan(app):
        async with base_lifespan(app):
            await asyncio.gather(*(subscriber.start() for subscriber in subscribers))
            try:
                yield
            finally:
                await asyncio.gather(
                    *(subscriber.close() for subscriber in subscribers)
                )

    app.router.lifespan_context = lifespan

    @app.get("/kv-event-state")
    async def state():
        return {
            "mode": mode,
            "workers": [
                {
                    "last_sequence": d.sequence,
                    "known_index_prefixes": len(d.blocks),
                    "completion_verified_prefixes": len(d.ready),
                    "gaps": d.gaps,
                    "epochs": d.epochs,
                    "unknown_parents": d.unknown_parents,
                    "rejected_entries": d.rejected,
                    "subscriber_errors": subscriber.errors,
                }
                for d, subscriber in zip(directories, subscribers)
            ],
        }

    return app
