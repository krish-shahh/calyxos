"""Runtime execution context for dependency tracking."""

import contextvars
from dataclasses import dataclass, field

# Context variable to track the current evaluation stack
_evaluation_stack: contextvars.ContextVar[list["EvaluationFrame"]] = contextvars.ContextVar(
    "calyxos_evaluation_stack", default=[]
)


@dataclass
class EvaluationFrame:
    """Represents a single node evaluation in the call stack."""

    object_id: int
    method_name: str
    args_hash: int
    accessed_nodes: set[tuple[int, str, int]] = field(default_factory=set)


def push_frame(object_id: int, method_name: str, args_hash: int) -> EvaluationFrame:
    """Push a new evaluation frame onto the stack."""
    frame = EvaluationFrame(object_id=object_id, method_name=method_name, args_hash=args_hash)
    stack = _evaluation_stack.get().copy()
    stack.append(frame)
    _evaluation_stack.set(stack)
    return frame


def pop_frame() -> EvaluationFrame | None:
    """Pop the top evaluation frame from the stack."""
    stack = _evaluation_stack.get().copy()
    if stack:
        frame = stack.pop()
        _evaluation_stack.set(stack)
        return frame
    return None


def get_current_frame() -> EvaluationFrame | None:
    """Get the current (top) evaluation frame."""
    stack = _evaluation_stack.get()
    return stack[-1] if stack else None


def record_node_access(object_id: int, method_name: str, args_hash: int) -> None:
    """Record that the current frame accessed another node."""
    frame = get_current_frame()
    if frame is not None:
        frame.accessed_nodes.add((object_id, method_name, args_hash))


def get_evaluation_stack() -> list[EvaluationFrame]:
    """Get a copy of the current evaluation stack."""
    return _evaluation_stack.get().copy()


def reset_context() -> None:
    """Reset the evaluation context (for testing)."""
    _evaluation_stack.set([])
