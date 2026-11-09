import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from catalogue.domain import Product, ProductQuery
from catalogue.errors import CatalogueExistsError, RepositoryError
from catalogue.persistence.sqlite import SQLiteProductRepository, initialize_catalogue


def test_imported_fixture_has_the_required_schema_and_distribution(database):
    with closing(sqlite3.connect(database)) as connection:
        assert [row[1] for row in connection.execute("PRAGMA table_info(products)")] == [
            "id",
            "name",
            "category",
            "price",
        ]
        assert dict(
            connection.execute("SELECT category, count(*) FROM products GROUP BY category")
        ) == {"laptop": 5, "smartphone": 5, "tablet": 5}


@pytest.mark.parametrize(
    "criteria, expected_ids",
    [
        ({"category": "laptop"}, [1, 2, 3, 4, 5]),
        ({"category": "laptop", "max_price": 69999}, [1, 2]),
        ({"category": "laptop", "max_price": 70000}, [1, 2, 3]),
        ({"category": "laptop", "max_price": 50000}, [1, 2]),
        ({"category": "tablet", "max_price": 0}, []),
        ({"category": "other"}, []),
        ({"category": "smartphone", "sort_by": "price_desc"}, [10, 8, 9, 7, 6]),
        ({"category": "smartphone", "max_price": 50000}, [6, 7, 8, 9]),
        (
            {"category": "smartphone", "max_price": 50000, "sort_by": "price_desc"},
            [8, 9, 7, 6],
        ),
    ],
)
def test_search_filters_and_sorts_exactly(database, criteria, expected_ids):
    results = SQLiteProductRepository(database).search(ProductQuery(**criteria))
    assert [product.id for product in results] == expected_ids


def test_shared_adapter_can_be_used_by_batch_threads(database):
    repository = SQLiteProductRepository(database)
    query = ProductQuery(category="laptop", max_price=50000)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(repository.search, [query] * 12))
    assert all([product.id for product in products] == [1, 2] for products in results)


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(RepositoryError):
        SQLiteProductRepository(path).search(ProductQuery(category="laptop"))
    assert not path.exists()


def test_initialization_preserves_existing_data(database):
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE products SET price = 123 WHERE id = 1")
    original_bytes = database.read_bytes()
    with pytest.raises(CatalogueExistsError):
        initialize_catalogue(database)
    assert database.read_bytes() == original_bytes


def test_initialization_preserves_an_unrelated_file(tmp_path):
    path = tmp_path / "important.txt"
    path.write_text("existing user content")
    with pytest.raises(CatalogueExistsError):
        initialize_catalogue(path)
    assert path.read_text() == "existing user content"


def test_failed_initialization_leaves_no_partial_database(tmp_path):
    path = tmp_path / "failed.sqlite3"
    product = Product(id=1, name="Laptop", category="laptop", price=50000)
    with pytest.raises(RepositoryError):
        initialize_catalogue(path, [product, product])
    assert not path.exists()


def test_initialization_creates_an_empty_catalogue_by_default(tmp_path):
    path = tmp_path / "empty.sqlite3"
    initialize_catalogue(path)
    assert SQLiteProductRepository(path).search(ProductQuery(category="laptop")) == []


@pytest.mark.parametrize(
    "row",
    [
        (16, "Product", "other", 100),
        (16, "Product", "laptop", -1),
        (16, "Product", "laptop", 1.5),
        (16, " ", "laptop", 100),
        (0, "Product", "laptop", 100),
    ],
)
def test_database_constraints_reject_invalid_records(database, row):
    with closing(sqlite3.connect(database)) as connection, connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO products VALUES (?, ?, ?, ?)", row)


def test_search_is_read_only(database):
    original_bytes = database.read_bytes()
    SQLiteProductRepository(database).search(ProductQuery(category="tablet"))
    assert database.read_bytes() == original_bytes


def test_concurrent_initializers_publish_one_complete_database(tmp_path, products):
    path = tmp_path / "shared.sqlite3"

    def initialize():
        try:
            initialize_catalogue(path, products)
            return True
        except CatalogueExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: initialize(), range(2))) == [False, True]
    assert SQLiteProductRepository(path).integrity_check() == "ok"
    assert len(SQLiteProductRepository(path).search(ProductQuery(category="laptop"))) == 5


def test_storage_errors_preserve_the_original_cause(tmp_path):
    with pytest.raises(RepositoryError) as captured:
        SQLiteProductRepository(tmp_path / "absent.sqlite3").search(ProductQuery(category="laptop"))
    assert isinstance(captured.value.__cause__, sqlite3.Error)
