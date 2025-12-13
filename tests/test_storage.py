"""Tests for storage backends and persistence."""

import tempfile
from pathlib import Path

from calyxos import JSONStorage, SQLiteStorage, fn, stored
from calyxos.core.decorator import get_graph, set_stored
from calyxos.core.persistence import load_object, save_object


class TestSQLiteStorage:
    """Test SQLite storage backend."""

    def test_save_and_load(self) -> None:
        """Test saving and loading stored values."""

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteStorage(db_path)

            class Account:
                @stored
                def balance(self) -> float:
                    return 100.0

            account = Account()
            account_id = id(account)

            # First access creates the node
            _ = account.balance()

            # Modify stored value
            set_stored(account, "balance", 250.0)

            # Save
            save_object(account, backend)

            # Load into new instance with the SAME object id key
            # (In real usage, you'd use a UUID or identifier, not Python id())
            account2 = Account()
            account2._calyxos_override_id = account_id  # Use same ID as original
            load_object(account2, backend)

            assert account2.balance() == 250.0

    def test_multiple_stored_values(self) -> None:
        """Test saving multiple stored values."""

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteStorage(db_path)

            class Portfolio:
                @stored
                def cash(self) -> float:
                    return 1000.0

                @stored
                def shares(self) -> int:
                    return 100

            portfolio = Portfolio()
            portfolio_id = id(portfolio)

            # First access creates the nodes
            _ = portfolio.cash()
            _ = portfolio.shares()

            # Modify both
            set_stored(portfolio, "cash", 2000.0)
            set_stored(portfolio, "shares", 200)

            # Save
            save_object(portfolio, backend)

            # Load into new instance with same ID
            portfolio2 = Portfolio()
            portfolio2._calyxos_override_id = portfolio_id
            load_object(portfolio2, backend)

            assert portfolio2.cash() == 2000.0
            assert portfolio2.shares() == 200

    def test_exists_check(self) -> None:
        """Test the exists() method."""

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteStorage(db_path)

            class Data:
                @stored
                def value(self) -> int:
                    return 42

            data = Data()
            obj_id = id(data)

            assert not backend.exists(obj_id)

            # Create the node first
            _ = data.value()

            save_object(data, backend)

            assert backend.exists(obj_id)

    def test_delete(self) -> None:
        """Test the delete() method."""

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteStorage(db_path)

            class Data:
                @stored
                def value(self) -> int:
                    return 42

            data = Data()
            obj_id = id(data)

            _ = data.value()  # Create node
            save_object(data, backend)
            assert backend.exists(obj_id)

            backend.delete(obj_id)
            assert not backend.exists(obj_id)


class TestJSONStorage:
    """Test JSON storage backend."""

    def test_save_and_load(self) -> None:
        """Test saving and loading with JSON storage."""

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(tmpdir)

            class Account:
                @stored
                def balance(self) -> float:
                    return 100.0

            account = Account()
            account_id = id(account)
            _ = account.balance()  # Create node
            set_stored(account, "balance", 500.0)

            save_object(account, backend)

            account2 = Account()
            account2._calyxos_override_id = account_id
            load_object(account2, backend)

            assert account2.balance() == 500.0

    def test_file_location(self) -> None:
        """Test that files are created in the correct location."""

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(tmpdir)

            class Data:
                @stored
                def value(self) -> int:
                    return 42

            data = Data()
            obj_id = id(data)

            _ = data.value()  # Create node
            save_object(data, backend)

            expected_file = Path(tmpdir) / f"object_{obj_id}.json"
            assert expected_file.exists()

    def test_multiple_objects(self) -> None:
        """Test storing multiple objects."""

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(tmpdir)

            class Counter:
                @stored
                def value(self) -> int:
                    return 0

            c1 = Counter()
            c2 = Counter()
            c1_id = id(c1)
            c2_id = id(c2)

            _ = c1.value()  # Create node
            _ = c2.value()  # Create node

            set_stored(c1, "value", 10)
            set_stored(c2, "value", 20)

            save_object(c1, backend)
            save_object(c2, backend)

            c1_loaded = Counter()
            c2_loaded = Counter()
            c1_loaded._calyxos_override_id = c1_id
            c2_loaded._calyxos_override_id = c2_id

            load_object(c1_loaded, backend)
            load_object(c2_loaded, backend)

            assert c1_loaded.value() == 10
            assert c2_loaded.value() == 20


class TestPersistenceRoundtrip:
    """Test full persistence roundtrip with derived values."""

    def test_rehydration_rebuilds_derived(self) -> None:
        """Test that loading an object rebuilds derived values lazily."""

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteStorage(db_path)

            class Model:
                @stored
                def input_value(self) -> int:
                    return 10

                @fn
                def doubled(self) -> int:
                    return self.input_value() * 2

                @fn
                def tripled(self) -> int:
                    return self.input_value() * 3

            # Create, modify, and save
            model = Model()
            model_id = id(model)
            _ = model.input_value()  # Create node
            set_stored(model, "input_value", 20)
            save_object(model, backend)

            # Load into fresh instance with same ID
            model2 = Model()
            model2._calyxos_override_id = model_id
            load_object(model2, backend)

            # Stored value should be restored
            assert model2.input_value() == 20

            # Derived values should recompute correctly
            assert model2.doubled() == 40
            assert model2.tripled() == 60

            # Check that nodes exist and have correct values
            graph = get_graph(model2)
            doubled_node = next(n for n in graph.get_all_nodes() if n.method_name == "doubled")
            assert doubled_node.value == 40

    def test_partial_evaluation_after_load(self) -> None:
        """Test that not all derived nodes are evaluated after load."""

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteStorage(db_path)

            class Model:
                @stored
                def base(self) -> int:
                    return 1

                @fn
                def computed_a(self) -> int:
                    return self.base() + 1

                @fn
                def computed_b(self) -> int:
                    return self.base() + 2

            model = Model()
            model_id = id(model)
            _ = model.base()  # Create node
            save_object(model, backend)

            model2 = Model()
            model2._calyxos_override_id = model_id
            load_object(model2, backend)

            graph = get_graph(model2)

            # Access stored node to create it
            _ = model2.base()

            # Stored node should be valid
            base_node = next(n for n in graph.get_all_nodes()
                            if n.method_name == "base")
            assert base_node.is_valid

            # Access one derived value
            assert model2.computed_a() == 2

            # Only computed_a should be computed, not computed_b
            computed_a = next(n for n in graph.get_all_nodes()
                             if n.method_name == "computed_a")
            assert computed_a.compute_count == 1

            # computed_b was never accessed, so it might not have a node yet
            computed_b_nodes = [n for n in graph.get_all_nodes()
                               if n.method_name == "computed_b"]
            if computed_b_nodes:
                computed_b = computed_b_nodes[0]
                assert computed_b.compute_count == 0
