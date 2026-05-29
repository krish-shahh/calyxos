"""Tests for storage backends and persistence."""

import tempfile
from pathlib import Path

from calyxos import JSONStorage, SQLiteStorage, fn, stored
from calyxos.core.decorator import get_graph, set_stored
from calyxos.core.persistence import load_object, save_object


class TestSQLiteStorage:
    """Test SQLite storage backend with string keys."""

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteStorage(db_path)

            class Account:
                @stored
                def balance(self) -> float:
                    return 100.0

            account = Account()
            _ = account.balance()
            set_stored(account, "balance", 250.0)

            save_object(account, backend, key="acct-1")

            # Load into a brand-new object — no id() hack needed
            account2 = Account()
            load_object(account2, backend, key="acct-1")
            assert account2.balance() == 250.0

    def test_cross_process_persistence(self) -> None:
        """Simulates cross-process: save, discard object, load into new one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteStorage(db_path)

            class Model:
                @stored
                def x(self) -> int:
                    return 0

            m1 = Model()
            _ = m1.x()
            set_stored(m1, "x", 42)
            save_object(m1, backend, key="my-model")
            del m1

            # "New process" — different backend instance, new object
            backend2 = SQLiteStorage(db_path)
            m2 = Model()
            load_object(m2, backend2, key="my-model")
            assert m2.x() == 42

    def test_multiple_stored_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteStorage(Path(tmpdir) / "test.db")

            class Portfolio:
                @stored
                def cash(self) -> float:
                    return 1000.0

                @stored
                def shares(self) -> int:
                    return 100

            p = Portfolio()
            _ = p.cash()
            _ = p.shares()
            set_stored(p, "cash", 2000.0)
            set_stored(p, "shares", 200)

            save_object(p, backend, key="portfolio-main")

            p2 = Portfolio()
            load_object(p2, backend, key="portfolio-main")
            assert p2.cash() == 2000.0
            assert p2.shares() == 200

    def test_exists_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteStorage(Path(tmpdir) / "test.db")

            class Data:
                @stored
                def value(self) -> int:
                    return 42

            d = Data()
            _ = d.value()

            assert not backend.exists("data-1")
            save_object(d, backend, key="data-1")
            assert backend.exists("data-1")

    def test_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteStorage(Path(tmpdir) / "test.db")

            class Data:
                @stored
                def value(self) -> int:
                    return 42

            d = Data()
            _ = d.value()
            save_object(d, backend, key="data-1")
            assert backend.exists("data-1")

            backend.delete("data-1")
            assert not backend.exists("data-1")


class TestJSONStorage:
    """Test JSON storage backend with string keys."""

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(tmpdir)

            class Account:
                @stored
                def balance(self) -> float:
                    return 100.0

            account = Account()
            _ = account.balance()
            set_stored(account, "balance", 500.0)

            save_object(account, backend, key="acct-1")

            account2 = Account()
            load_object(account2, backend, key="acct-1")
            assert account2.balance() == 500.0

    def test_file_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(tmpdir)

            class Data:
                @stored
                def value(self) -> int:
                    return 42

            d = Data()
            _ = d.value()
            save_object(d, backend, key="my-data")

            expected_file = Path(tmpdir) / "my-data.json"
            assert expected_file.exists()

    def test_multiple_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(tmpdir)

            class Counter:
                @stored
                def value(self) -> int:
                    return 0

            c1 = Counter()
            c2 = Counter()
            _ = c1.value()
            _ = c2.value()
            set_stored(c1, "value", 10)
            set_stored(c2, "value", 20)

            save_object(c1, backend, key="counter-1")
            save_object(c2, backend, key="counter-2")

            c1_loaded = Counter()
            c2_loaded = Counter()
            load_object(c1_loaded, backend, key="counter-1")
            load_object(c2_loaded, backend, key="counter-2")

            assert c1_loaded.value() == 10
            assert c2_loaded.value() == 20


class TestPersistenceRoundtrip:
    """Test full persistence roundtrip with derived values."""

    def test_rehydration_rebuilds_derived(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteStorage(Path(tmpdir) / "test.db")

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

            model = Model()
            _ = model.input_value()
            set_stored(model, "input_value", 20)
            save_object(model, backend, key="model-1")

            model2 = Model()
            load_object(model2, backend, key="model-1")

            assert model2.input_value() == 20
            assert model2.doubled() == 40
            assert model2.tripled() == 60

            graph = get_graph(model2)
            doubled_node = next(n for n in graph.get_all_nodes() if n.method_name == "doubled")
            assert doubled_node.value == 40

    def test_partial_evaluation_after_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteStorage(Path(tmpdir) / "test.db")

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
            _ = model.base()
            save_object(model, backend, key="model-2")

            model2 = Model()
            load_object(model2, backend, key="model-2")
            _ = model2.base()

            graph = get_graph(model2)
            base_node = next(n for n in graph.get_all_nodes() if n.method_name == "base")
            assert base_node.is_valid

            assert model2.computed_a() == 2

            computed_a = next(n for n in graph.get_all_nodes() if n.method_name == "computed_a")
            assert computed_a.compute_count == 1

            computed_b_nodes = [n for n in graph.get_all_nodes() if n.method_name == "computed_b"]
            if computed_b_nodes:
                assert computed_b_nodes[0].compute_count == 0
