"""Tests for dependency tracking and graph construction."""

from calyxos import fn, stored
from calyxos.core.decorator import get_graph, set_stored


class TestDependencyTracking:
    """Test runtime dependency capture."""

    def test_simple_dependency(self) -> None:
        """Test that simple method dependencies are captured."""

        class Calculator:
            @fn
            def x(self) -> int:
                return 10

            @fn
            def double_x(self) -> int:
                return self.x() * 2

        calc = Calculator()
        graph = get_graph(calc)

        # Trigger evaluation
        assert calc.double_x() == 20

        # Check dependencies
        nodes = graph.get_all_nodes()
        double_x_node = next(n for n in nodes if n.method_name == "double_x")
        x_node = next(n for n in nodes if n.method_name == "x")

        # double_x should depend on x
        assert x_node.key() in double_x_node.children
        assert double_x_node.key() in x_node.parents

    def test_complex_dependency_chain(self) -> None:
        """Test multi-level dependency chains."""

        class Model:
            @fn
            def a(self) -> int:
                return 1

            @fn
            def b(self) -> int:
                return self.a() + 1

            @fn
            def c(self) -> int:
                return self.b() + 1

        model = Model()
        graph = get_graph(model)

        # Trigger full evaluation
        assert model.c() == 3

        # Check chain: c depends on b depends on a
        nodes = {n.method_name: n for n in graph.get_all_nodes()}
        c_node = nodes["c"]
        b_node = nodes["b"]
        a_node = nodes["a"]

        assert b_node.key() in c_node.children
        assert a_node.key() in b_node.children

    def test_conditional_dependency(self) -> None:
        """Test that conditional dependencies are captured correctly."""

        class Conditional:
            @fn
            def use_x(self) -> bool:
                return True

            @fn
            def x(self) -> int:
                return 10

            @fn
            def y(self) -> int:
                return 20

            @fn
            def result(self) -> int:
                # Dependency on use_x is always recorded
                if self.use_x():
                    return self.x()
                else:
                    return self.y()

        cond = Conditional()
        graph = get_graph(cond)

        # First evaluation: uses x
        assert cond.result() == 10

        nodes = {n.method_name: n for n in graph.get_all_nodes()}
        result_node = nodes["result"]

        # result depends on use_x (always evaluated)
        assert nodes["use_x"].key() in result_node.children

        # result also depends on x (evaluated this time)
        assert nodes["x"].key() in result_node.children

    def test_diamond_dependency(self) -> None:
        """Test diamond dependency pattern: D depends on B and C, both depend on A."""

        class Diamond:
            @fn
            def a(self) -> int:
                return 1

            @fn
            def b(self) -> int:
                return self.a() + 1

            @fn
            def c(self) -> int:
                return self.a() + 2

            @fn
            def d(self) -> int:
                return self.b() + self.c()

        diamond = Diamond()
        graph = get_graph(diamond)

        assert diamond.d() == 1 + 1 + 1 + 2  # 5

        nodes = {n.method_name: n for n in graph.get_all_nodes()}
        a_node = nodes["a"]

        # a should have two parents: b and c
        assert len(a_node.parents) == 2

    def test_no_spurious_dependencies(self) -> None:
        """Test that unrequested values are not recorded as dependencies."""

        class NoSpurious:
            @fn
            def x(self) -> int:
                return 10

            @fn
            def y(self) -> int:
                return 20

            @fn
            def result(self) -> int:
                return self.x()  # Only depends on x, not y

        obj = NoSpurious()
        graph = get_graph(obj)

        assert obj.result() == 10

        nodes = {n.method_name: n for n in graph.get_all_nodes()}
        result_node = nodes["result"]
        x_node = nodes["x"]

        # result should only depend on x, not y
        assert x_node.key() in result_node.children
        # y may not even have a node since it was never accessed
        if "y" in nodes:
            assert nodes["y"].key() not in result_node.children


class TestInvalidation:
    """Test invalidation propagation."""

    def test_invalidation_from_stored_change(self) -> None:
        """Test that changing a stored value invalidates dependents."""

        class WithStored:
            @stored
            def input_value(self) -> int:
                return 10

            @fn
            def double(self) -> int:
                return self.input_value() * 2

        obj = WithStored()
        graph = get_graph(obj)

        assert obj.double() == 20

        double_node = next(n for n in graph.get_all_nodes() if n.method_name == "double")
        assert double_node.is_valid

        # Change stored value
        set_stored(obj, "input_value", 15)

        # double should now be invalid
        assert not double_node.is_valid

        # Accessing it should recompute
        assert obj.double() == 30

    def test_invalidation_propagates_downstream(self) -> None:
        """Test that invalidation propagates to all downstream dependents."""

        class PropagatingInvalidation:
            @stored
            def base(self) -> int:
                return 1

            @fn
            def level1(self) -> int:
                return self.base() + 1

            @fn
            def level2(self) -> int:
                return self.level1() + 1

            @fn
            def level3(self) -> int:
                return self.level2() + 1

        obj = PropagatingInvalidation()
        graph = get_graph(obj)

        # Evaluate all levels
        assert obj.level3() == 4

        # All should be valid
        assert all(n.is_valid for n in graph.get_all_nodes() if n.method_name != "base")

        # Invalidate base
        set_stored(obj, "base", 10)

        # All derived nodes should be invalid
        level1 = next(n for n in graph.get_all_nodes() if n.method_name == "level1")
        level2 = next(n for n in graph.get_all_nodes() if n.method_name == "level2")
        level3 = next(n for n in graph.get_all_nodes() if n.method_name == "level3")

        assert not level1.is_valid
        assert not level2.is_valid
        assert not level3.is_valid

    def test_only_affected_nodes_invalidated(self) -> None:
        """Test that invalidation only affects downstream nodes."""

        class SelectiveInvalidation:
            @stored
            def x(self) -> int:
                return 10

            @fn
            def x_doubled(self) -> int:
                return self.x() * 2

            @fn
            def unrelated(self) -> int:
                return 42

        obj = SelectiveInvalidation()
        graph = get_graph(obj)

        # Evaluate both
        assert obj.x_doubled() == 20
        assert obj.unrelated() == 42

        # Invalidate x
        set_stored(obj, "x", 15)

        x_doubled = next(n for n in graph.get_all_nodes() if n.method_name == "x_doubled")
        unrelated = next(n for n in graph.get_all_nodes() if n.method_name == "unrelated")

        # x_doubled should be invalid
        assert not x_doubled.is_valid

        # unrelated should still be valid
        assert unrelated.is_valid
