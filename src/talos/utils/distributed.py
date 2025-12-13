"""Distributed node evaluation for parallel and remote execution."""

import json
from typing import Any, Callable

from talos.core.decorator import get_graph
from talos.graph.node import NodeType


class NodeExecutionPlan:
    """Plan for executing a node locally or remotely."""

    def __init__(self, method_name: str, args_hash: int, dependencies: list[str]) -> None:
        """Initialize execution plan."""
        self.method_name = method_name
        self.args_hash = args_hash
        self.dependencies = dependencies
        self.can_parallelize = len(dependencies) == 0
        self.can_remote = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize execution plan for remote transmission."""
        return {
            "method_name": self.method_name,
            "args_hash": self.args_hash,
            "dependencies": self.dependencies,
            "can_parallelize": self.can_parallelize,
        }


class DistributedExecutor:
    """
    Coordinate distributed execution of Talos nodes.

    Enables:
    - Parallel evaluation of independent nodes
    - Remote execution via worker processes or services
    - Custom scheduling strategies

    Usage:
        executor = DistributedExecutor(obj, workers=4)
        executor.schedule_parallel()
        result = executor.execute()
    """

    def __init__(self, obj: Any, workers: int = 1) -> None:
        """Initialize distributed executor."""
        self.obj = obj
        self.graph = get_graph(obj)
        self.workers = workers
        self.execution_plan: dict[str, NodeExecutionPlan] = {}
        self._build_plan()

    def _build_plan(self) -> None:
        """Build execution plan from graph."""
        for node in self.graph.get_all_nodes():
            # Find dependencies
            deps = [
                self.graph.nodes.get(key).method_name
                for key in node.children
                if key in self.graph.nodes
            ]

            plan = NodeExecutionPlan(node.method_name, node.args_hash, deps)
            self.execution_plan[node.method_name] = plan

    def get_parallelizable_nodes(self) -> list[str]:
        """Get nodes that can execute in parallel (no dependencies)."""
        return [
            name for name, plan in self.execution_plan.items() if plan.can_parallelize
        ]

    def get_critical_path(self) -> list[str]:
        """Get the longest dependency chain (critical path)."""
        # Simple topological sort to find longest path
        visited = set()
        longest_path = []

        def dfs(node_name: str, path: list[str]) -> list[str]:
            if node_name in visited:
                return path
            visited.add(node_name)

            plan = self.execution_plan[node_name]
            if not plan.dependencies:
                return path + [node_name]

            longest = path + [node_name]
            for dep in plan.dependencies:
                candidate = dfs(dep, longest)
                if len(candidate) > len(longest):
                    longest = candidate

            return longest

        # Start from each leaf node
        for name in self.execution_plan:
            path = dfs(name, [])
            if len(path) > len(longest_path):
                longest_path = path

        return longest_path

    def schedule_parallel(self) -> dict[int, list[str]]:
        """
        Schedule nodes for parallel execution.

        Returns dict mapping stage_number -> list of node names that can run in parallel.
        """
        stages: dict[int, list[str]] = {}
        computed = set()
        stage = 0

        while len(computed) < len(self.execution_plan):
            stage_nodes = []

            for name, plan in self.execution_plan.items():
                if name in computed:
                    continue
                # Can execute if all dependencies are computed
                if all(dep in computed for dep in plan.dependencies):
                    stage_nodes.append(name)

            if not stage_nodes:
                break  # Cycle detected or error

            stages[stage] = stage_nodes
            computed.update(stage_nodes)
            stage += 1

        return stages

    def estimate_speedup(self) -> float:
        """
        Estimate theoretical speedup from parallelization.

        Returns: speedup factor (e.g., 4.0 = 4x faster)
        """
        critical_path = self.get_critical_path()
        total_nodes = len(self.execution_plan)

        if len(critical_path) == 0:
            return 1.0

        # Amdahl's law: speedup ≈ total_work / (critical_path + (total_work - critical_path) / workers)
        parallelizable = total_nodes - len(critical_path)
        if parallelizable == 0:
            return 1.0

        speedup = total_nodes / (len(critical_path) + parallelizable / self.workers)
        return min(speedup, self.workers)  # Cap at worker count

    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of execution plan."""
        stages = self.schedule_parallel()
        critical_path = self.get_critical_path()
        speedup = self.estimate_speedup()

        return {
            "total_nodes": len(self.execution_plan),
            "parallelizable_nodes": len(self.get_parallelizable_nodes()),
            "critical_path_length": len(critical_path),
            "execution_stages": len(stages),
            "workers": self.workers,
            "estimated_speedup": speedup,
            "stages": stages,
        }

    def to_json(self) -> str:
        """Serialize execution plan for transmission to workers."""
        plan_dicts = {
            name: plan.to_dict() for name, plan in self.execution_plan.items()
        }
        return json.dumps(plan_dicts, indent=2)
