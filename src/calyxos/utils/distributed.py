"""Distributed node evaluation for parallel and remote execution."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from calyxos.core.decorator import get_graph
from calyxos.graph.node import NodeType


class NodeExecutionPlan:
    """Plan for executing a node locally or remotely."""

    def __init__(self, method_name: str, args_hash: int, dependencies: list[str]) -> None:
        self.method_name = method_name
        self.args_hash = args_hash
        self.dependencies = dependencies
        self.can_parallelize = len(dependencies) == 0
        self.can_remote = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "args_hash": self.args_hash,
            "dependencies": self.dependencies,
            "can_parallelize": self.can_parallelize,
        }


class DistributedExecutor:
    """Coordinate parallel execution of calyxos graph nodes.

    Stages of independent nodes are executed concurrently using a thread
    pool (``concurrent.futures.ThreadPoolExecutor``).

    Usage::

        executor = DistributedExecutor(obj, workers=4)
        results = executor.execute()   # evaluates all invalid nodes in parallel
    """

    def __init__(self, obj: Any, workers: int = 4) -> None:
        self.obj = obj
        self.graph = get_graph(obj)
        self.workers = workers
        self.execution_plan: dict[str, NodeExecutionPlan] = {}
        self._build_plan()

    def _build_plan(self) -> None:
        for node in self.graph.get_all_nodes():
            deps = [
                self.graph.nodes.get(key).method_name
                for key in node.children
                if key in self.graph.nodes
            ]
            plan = NodeExecutionPlan(node.method_name, node.args_hash, deps)
            self.execution_plan[node.method_name] = plan

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self) -> dict[str, Any]:
        """Evaluate all nodes in topological order, parallelising within
        each stage using a thread pool.

        Returns a dict mapping ``method_name -> computed value``.
        """
        stages = self.schedule_parallel()
        results: dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for _stage_num in sorted(stages):
                stage_nodes = stages[_stage_num]

                # Submit all nodes in this stage concurrently
                futures = {}
                for name in stage_nodes:
                    plan = self.execution_plan[name]
                    node = self.graph.nodes.get(
                        (self.graph.object_id, name, plan.args_hash)
                    )
                    if node is None:
                        continue
                    futures[pool.submit(self.graph.evaluate_node, node)] = name

                # Collect results
                for future in as_completed(futures):
                    name = futures[future]
                    results[name] = future.result()

        return results

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------

    def get_parallelizable_nodes(self) -> list[str]:
        return [
            name for name, plan in self.execution_plan.items() if plan.can_parallelize
        ]

    def get_critical_path(self) -> list[str]:
        visited: set[str] = set()
        longest_path: list[str] = []

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

        for name in self.execution_plan:
            path = dfs(name, [])
            if len(path) > len(longest_path):
                longest_path = path

        return longest_path

    def schedule_parallel(self) -> dict[int, list[str]]:
        """Schedule nodes for parallel execution.

        Returns dict mapping ``stage_number -> [node_names]`` where each
        stage can run in parallel.
        """
        stages: dict[int, list[str]] = {}
        computed: set[str] = set()
        stage = 0

        while len(computed) < len(self.execution_plan):
            stage_nodes = []
            for name, plan in self.execution_plan.items():
                if name in computed:
                    continue
                if all(dep in computed for dep in plan.dependencies):
                    stage_nodes.append(name)
            if not stage_nodes:
                break
            stages[stage] = stage_nodes
            computed.update(stage_nodes)
            stage += 1

        return stages

    def estimate_speedup(self) -> float:
        critical_path = self.get_critical_path()
        total_nodes = len(self.execution_plan)

        if len(critical_path) == 0:
            return 1.0

        parallelizable = total_nodes - len(critical_path)
        if parallelizable == 0:
            return 1.0

        speedup = total_nodes / (len(critical_path) + parallelizable / self.workers)
        return min(speedup, self.workers)

    def get_execution_summary(self) -> dict[str, Any]:
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
        plan_dicts = {
            name: plan.to_dict() for name, plan in self.execution_plan.items()
        }
        return json.dumps(plan_dicts, indent=2)
