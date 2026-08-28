# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import itertools
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def peek_order_key(self) -> tuple[Any, ...]:
        """Return the captured order key at the front of the queue."""
        pass

    @abstractmethod
    def peek_n(self, n: int) -> list[Request]:
        """Peek at the first n requests in queue order without removal."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def peek_n(self, n: int) -> list[Request]:
        """Peek at the first n requests in FCFS order without removal."""
        if n <= 0:
            return []
        return list(itertools.islice(self, n))

    def peek_order_key(self) -> tuple[Any, ...]:
        raise NotImplementedError("FCFS queues do not expose a comparable order key")


@dataclass(order=True, frozen=True)
class _PriorityQueueItem:
    key: tuple[Any, ...]
    request: Request = field(compare=False)


class PriorityRequestQueue(RequestQueue):
    """A heap ordered by an immutable key captured when a request is added."""

    def __init__(
        self, order_key: Callable[[Request], tuple[Any, ...]] | None = None
    ) -> None:
        self._order_key = order_key or self._default_order_key
        self._heap: list[_PriorityQueueItem] = []

    @staticmethod
    def _default_order_key(request: Request) -> tuple[Any, ...]:
        return (
            request.priority,
            request.arrival_time,
            request.request_id,
            id(request),
        )

    def peek_order_key(self) -> tuple[Any, ...]:
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0].key

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(
            self._heap,
            _PriorityQueueItem(self._order_key(request), request),
        )

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap).request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0].request

    def peek_n(self, n: int) -> list[Request]:
        """Peek at the first n requests in heap order without removal.

        Uses a frontier traversal from the heap root: only the children of
        popped nodes are pushed, so the cost is O(n log n) time and O(n)
        temporary space instead of copying the whole heap.
        """
        if n <= 0 or not self._heap:
            return []
        result: list[Request] = []
        # Frontier of (key, heap_index); the index breaks key ties the same
        # way the heap array itself does.
        frontier: list[tuple[tuple[Any, ...], int]] = [(self._heap[0].key, 0)]
        while frontier and len(result) < n:
            _, i = heapq.heappop(frontier)
            result.append(self._heap[i].request)
            for child in (2 * i + 1, 2 * i + 2):
                if child < len(self._heap):
                    heapq.heappush(frontier, (self._heap[child].key, child))
        return result

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by the configured key."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by the configured key."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        index = next(
            (i for i, item in enumerate(self._heap) if item.request is request),
            None,
        )
        if index is None:
            raise ValueError("request is not in queue")
        del self._heap[index]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [
            item for item in self._heap if item.request not in requests_to_remove
        ]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy).request


def create_request_queue(
    policy: SchedulingPolicy,
    order_key: Callable[[Request], tuple[Any, ...]] | None = None,
) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue(order_key)
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
