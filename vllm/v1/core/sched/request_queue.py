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

    @abstractmethod
    def peek_n_with_keys(self, n: int) -> list[tuple[tuple[Any, ...], Request]]:
        """Peek at requests and their captured queue-order keys."""
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

    def peek_n_with_keys(self, n: int) -> list[tuple[tuple[Any, ...], Request]]:
        raise NotImplementedError("FCFS queues do not expose comparable order keys")


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
        self._request_indices: dict[int, int] = {}

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
        item = _PriorityQueueItem(self._order_key(request), request)
        self._heap.append(item)
        index = len(self._heap) - 1
        self._request_indices[id(request)] = index
        self._sift_up(index)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return self._pop_at(0)

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
        return [request for _, request in self.peek_n_with_keys(n)]

    def peek_n_with_keys(self, n: int) -> list[tuple[tuple[Any, ...], Request]]:
        """Peek at requests with the immutable keys captured on enqueue."""
        if n <= 0 or not self._heap:
            return []
        result: list[tuple[tuple[Any, ...], Request]] = []
        # Frontier of (key, heap_index); the index breaks key ties the same
        # way the heap array itself does.
        frontier: list[tuple[tuple[Any, ...], int]] = [(self._heap[0].key, 0)]
        while frontier and len(result) < n:
            _, i = heapq.heappop(frontier)
            item = self._heap[i]
            result.append((item.key, item.request))
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
        index = self._request_indices.get(id(request))
        if index is None:
            raise ValueError("request is not in queue")
        self._pop_at(index)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        for request in requests:
            index = self._request_indices.get(id(request))
            if index is not None:
                self._pop_at(index)

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

    def _swap(self, first: int, second: int) -> None:
        self._heap[first], self._heap[second] = (
            self._heap[second],
            self._heap[first],
        )
        self._request_indices[id(self._heap[first].request)] = first
        self._request_indices[id(self._heap[second].request)] = second

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if not self._heap[index] < self._heap[parent]:
                break
            self._swap(index, parent)
            index = parent

    def _sift_down(self, index: int) -> None:
        size = len(self._heap)
        while True:
            left = 2 * index + 1
            if left >= size:
                return
            right = left + 1
            smallest = (
                right
                if right < size and self._heap[right] < self._heap[left]
                else left
            )
            if not self._heap[smallest] < self._heap[index]:
                return
            self._swap(index, smallest)
            index = smallest

    def _pop_at(self, index: int) -> Request:
        removed = self._heap[index]
        last = self._heap.pop()
        del self._request_indices[id(removed.request)]
        if index < len(self._heap):
            self._heap[index] = last
            self._request_indices[id(last.request)] = index
            parent = (index - 1) // 2
            if index > 0 and self._heap[index] < self._heap[parent]:
                self._sift_up(index)
            else:
                self._sift_down(index)
        return removed.request


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
