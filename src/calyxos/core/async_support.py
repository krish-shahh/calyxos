"""Async/await support for calyxos computed methods."""

import asyncio
import functools
import hashlib
from collections.abc import Callable
from typing import Any, TypeVar, cast

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
    """Decorator for async methods with memoization and dependency tracking.

    Uses the graph's ``async_evaluate_node`` so independent branches can
    be evaluated concurrently with ``asyncio.gather``.

    Usage::

        class Model:
            @async_fn
            async def fetch_data(self, url: str) -> dict:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        return await resp.json()
    """
    from calyxos.core.decorator import get_graph

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        graph = get_graph(self)

        if hasattr(self, "_calyxos_override_id"):
            obj_id = self._calyxos_override_id
        else:
            obj_id = id(self)

        args_hash = _compute_args_hash((self,) + args, kwargs)

        node = graph.get_or_create_node(
            method_name=func.__name__,
            args_hash=args_hash,
            node_type=NodeType.DERIVED,
            compute_fn=lambda: func(self, *args, **kwargs),
        )

        current_frame = get_current_frame()
        if current_frame is not None:
            record_node_access(obj_id, func.__name__, args_hash)

        return await graph.async_evaluate_node(node)

    wrapper._calyxos_node = True  # type: ignore[attr-defined]
    wrapper._calyxos_flags = None  # type: ignore[attr-defined]
    return cast(F, wrapper)


def async_map_node(source: str) -> Callable[[F], F]:
    """Async version of ``@map_node`` — evaluates per-element nodes
    concurrently with ``asyncio.gather``.

    Usage::

        class Pipeline:
            @node(NodeFlag.STORED)
            def urls(self) -> list[str]:
                return ["http://a.com", "http://b.com"]

            @async_map_node("urls")
            async def fetched(self, url: str) -> dict:
                ...  # async HTTP call
    """
    from calyxos.core.decorator import _compute_args_hash as sync_hash
    from calyxos.core.decorator import get_graph
    from calyxos.core.flags import NodeFlag

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            graph = get_graph(self)

            if hasattr(self, "_calyxos_override_id"):
                obj_id = self._calyxos_override_id
            else:
                obj_id = id(self)

            orch_hash = sync_hash((self,) + args, kwargs)
            map_prefix = f"_map_{func.__name__}"

            async def compute_orchestrated() -> list[Any]:
                collection = getattr(self, source)()

                # Create per-element nodes
                elem_nodes = []
                for element in collection:
                    elem_hash = sync_hash((self, element), {})

                    elem_nd = graph.get_or_create_node(
                        method_name=map_prefix,
                        args_hash=elem_hash,
                        node_type=NodeType.DERIVED,
                        compute_fn=lambda e=element: func(self, e),
                        flags=NodeFlag.NONE,
                    )

                    record_node_access(obj_id, map_prefix, elem_hash)
                    elem_nodes.append(elem_nd)

                # Evaluate all element nodes concurrently
                results = await asyncio.gather(
                    *[graph.async_evaluate_node(nd) for nd in elem_nodes]
                )
                return list(results)

            orch_nd = graph.get_or_create_node(
                method_name=func.__name__,
                args_hash=orch_hash,
                node_type=NodeType.DERIVED,
                compute_fn=compute_orchestrated,
                flags=NodeFlag.NONE,
            )

            current_frame = get_current_frame()
            if current_frame is not None:
                record_node_access(obj_id, func.__name__, orch_hash)

            return await graph.async_evaluate_node(orch_nd)

        wrapper._calyxos_node = True  # type: ignore[attr-defined]
        wrapper._calyxos_flags = NodeFlag.NONE  # type: ignore[attr-defined]
        wrapper._calyxos_map = True  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorator
