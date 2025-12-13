"""Tests for introspection utilities."""

from calyxos import enable_dir, fn, get_calyxos_methods, list_computed_methods, list_stored_methods, stored


class TestDirSupport:
    """Test __dir__ patching for Talos objects."""

    def test_enable_dir_includes_calyxos_methods(self) -> None:
        """Test that enable_dir() adds Talos methods to dir()."""

        class Model:
            @stored
            def stored_value(self) -> int:
                return 10

            @fn
            def computed_value(self) -> int:
                return self.stored_value() * 2

        obj = Model()
        _ = obj.stored_value()
        _ = obj.computed_value()

        # Before enable_dir, Talos methods might not be in dir()
        dir_before = set(dir(obj))

        # After enable_dir, they should be
        enable_dir(obj)
        dir_after = set(dir(obj))

        assert "stored_value" in dir_after
        assert "computed_value" in dir_after

    def test_dir_shows_sorted_list(self) -> None:
        """Test that dir() returns a sorted list."""

        class Model:
            @fn
            def zebra(self) -> int:
                return 1

            @fn
            def apple(self) -> int:
                return 2

        obj = Model()
        _ = obj.zebra()
        _ = obj.apple()

        enable_dir(obj)
        result = dir(obj)

        assert isinstance(result, list)
        assert result == sorted(result)


class TestGetTalosMethods:
    """Test get_calyxos_methods() introspection."""

    def test_get_calyxos_methods_returns_dict(self) -> None:
        """Test that get_calyxos_methods returns method info."""

        class Model:
            @stored
            def stored_val(self) -> int:
                return 42

            @fn
            def computed_val(self) -> int:
                return self.stored_val() * 2

        obj = Model()
        _ = obj.stored_val()
        _ = obj.computed_val()

        methods = get_calyxos_methods(obj)

        assert "stored_val" in methods
        assert "computed_val" in methods

        assert methods["stored_val"]["type"] == "stored"
        assert methods["computed_val"]["type"] == "derived"

        assert methods["stored_val"]["is_valid"] is True
        assert methods["computed_val"]["is_valid"] is True

        assert methods["stored_val"]["value"] == 42
        assert methods["computed_val"]["value"] == 84

        # stored_val has compute_count 0 because it's created on first access
        # computed_val has compute_count 1 because it was evaluated
        assert methods["stored_val"]["compute_count"] == 0
        assert methods["computed_val"]["compute_count"] == 1


class TestListMethods:
    """Test list_stored_methods() and list_computed_methods()."""

    def test_list_stored_methods(self) -> None:
        """Test listing only stored methods."""

        class Model:
            @stored
            def stored_a(self) -> int:
                return 1

            @stored
            def stored_b(self) -> int:
                return 2

            @fn
            def computed(self) -> int:
                return self.stored_a() + self.stored_b()

        obj = Model()
        _ = obj.stored_a()
        _ = obj.stored_b()
        _ = obj.computed()

        stored_list = list_stored_methods(obj)

        assert "stored_a" in stored_list
        assert "stored_b" in stored_list
        assert "computed" not in stored_list
        assert stored_list == sorted(stored_list)

    def test_list_computed_methods(self) -> None:
        """Test listing only computed methods."""

        class Model:
            @stored
            def stored_val(self) -> int:
                return 1

            @fn
            def computed_a(self) -> int:
                return self.stored_val() + 1

            @fn
            def computed_b(self) -> int:
                return self.stored_val() + 2

        obj = Model()
        _ = obj.stored_val()
        _ = obj.computed_a()
        _ = obj.computed_b()

        computed = list_computed_methods(obj)

        assert "computed_a" in computed
        assert "computed_b" in computed
        assert "stored_val" not in computed
        assert computed == sorted(computed)

    def test_empty_lists_for_unaccessed_methods(self) -> None:
        """Test that unaccessed methods don't appear in lists."""

        class Model:
            @fn
            def never_called(self) -> int:
                return 1

            @fn
            def called(self) -> int:
                return 2

        obj = Model()
        _ = obj.called()  # Only call one

        computed = list_computed_methods(obj)

        # Only the called method should have a node
        assert "called" in computed
        assert "never_called" not in computed
