"""Async/await support for Talos computed methods."""

import asyncio
import functools
import hashlib
from collections.abc import Callable
from typing import Any, TypeVar, cast

from calyxos.graph.graph import ComputationGraph
from calyxos.graph.node import NodeType
from calyxos.tracking.context import get_current_frame, record_node_access

F = TypeVar("F", bound=Callable[..., Any])


def _compute_args_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Compute a stable hash for function arguments."""
    try:
        items = []
        for arg in args[1:]:  # Skip self
            items.append(repr(arg).encode())
        for k, v in sorted(kwargs.items()):
            items.append(f"{k}={v!r}".encode())
        content = b"|".join(items)
        return int(hashlib.md5(content).hexdigest(), 16)
    except Exception:
        # Fallback: use object ids for unhashable objects
        parts = []
        for arg in args[1:]:
            try:
                hash(arg)
                parts.append(str(hash(arg)))
            except TypeError:
                parts.append(str(id(arg)))
        for k, v in sorted(kwargs.items()):
            try:
                hash(v)
                parts.append(f"{k}={hash(v)}")
            except TypeError:
                parts.append(f"{k}={id(v)}")
        content = "|".join(parts).encode()
        return int(hashlib.md5(content).hexdigest(), 16)


def async_fn(func: F) -> F:
    """
    Decorator for async methods that should be memoized and dependency-tracked.

    Usage:
        class Model:
            @async_fn
            async def fetch_data(self, url: str) -> dict:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        return await resp.json()

            @async_fn
            async def process_data(self) -> list:
                data = await self.fetch_data("http://api.example.com/data")
                return [x * 2 for x in data['values']]

        obj = Model()
        result = await obj.process_data()
    """
    from calyxos.core.decorator import get_graph

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Get the computation graph for this object
        graph = get_graph(self)

        # Use the same ID as the graph (with override support)
        if hasattr(self, "_calyxos_override_id"):
            obj_id = self._calyxos_override_id
        else:
            obj_id = id(self)

        # Compute hash of arguments
        args_hash = _compute_args_hash((self,) + args, kwargs)

        # Get or create the node
        node = graph.get_or_create_node(
            method_name=func.__name__,
            args_hash=args_hash,
            node_type=NodeType.DERIVED,
            compute_fn=lambda: asyncio.create_task(func(self, *args, **kwargs)),
        )

        # Return cached value if valid
        if node.is_valid:
            return node.value

        # Record this node access in the current frame
        current_frame = get_current_frame()
        if current_frame is not None:
            record_node_access(obj_id, func.__name__, args_hash)

        # Evaluate the async function
        result = await func(self, *args, **kwargs)

        # Cache the result
        with graph._lock:
            node.value = result
            node.is_valid = True
            node.compute_count += 1

        return result

    return cast(F, wrapper)
