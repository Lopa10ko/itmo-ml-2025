from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from catalogue.domain import ProductQuery
from catalogue.entrypoints import populate
from catalogue.errors import InputDataError, RepositoryError
from catalogue.persistence.sqlite import SQLiteProductRepository, populate_catalogue
from catalogue.population import generate_products, load_population
from catalogue.settings import load_settings


def test_generation_is_repeatable_and_preserves_examples(products):
    generated = generate_products(products, 30, 42)
    assert generated == generate_products(list(reversed(products)), 30, 42)
    assert Counter(product.category for product in generated) == {
        "laptop": 30,
        "smartphone": 30,
        "tablet": 30,
    }
    assert len({product.id for product in generated}) == len(generated)
    assert len({product.name for product in generated}) == len(generated)
    assert all(product in generated for product in products)


def test_increasing_size_preserves_existing_generated_identifiers(products):
    smaller = generate_products(products, 30, 42)
    larger = {product.id: product for product in generate_products(products, 50, 42)}
    assert all(larger[product.id] == product for product in smaller)


def test_seed_changes_generated_prices_without_changing_examples(products):
    first = generate_products(products, 30, 42)
    second = generate_products(products, 30, 17)
    assert first != second
    assert first[: len(products)] == second[: len(products)] == products


def test_examples_supply_price_boundaries_ties_and_ranges(products):
    generated = generate_products(products, 30, 42)
    for category in {product.category for product in products}:
        prices = [product.price for product in generated if product.category == category]
        anchors = [product.price for product in products if product.category == category]
        assert min(prices) == min(anchors)
        assert max(prices) == max(anchors)
        assert len(set(prices)) < len(prices)
    laptop_prices = {product.price for product in generated if product.category == "laptop"}
    assert {49999, 50000, 50001, 69999, 70000, 70001, 79999, 80000, 80001} <= laptop_prices


@pytest.mark.parametrize(
    "examples,count",
    [(lambda p: [], 30), (lambda p: p[:5], 30), (lambda p: p + [p[0]], 30), (lambda p: p, 4)],
)
def test_invalid_population_inputs_are_rejected(products, examples, count):
    with pytest.raises(InputDataError):
        generate_products(examples(products), count, 42)


def test_population_creates_parent_directories_and_is_idempotent(tmp_path, products):
    path = tmp_path / "nested" / "data" / "catalogue.sqlite3"
    generated = generate_products(products, 30, 42)
    populate_catalogue(path, generated)
    first = path.read_bytes()
    populate_catalogue(path, generated)
    assert path.read_bytes() == first
    repository = SQLiteProductRepository(path)
    assert repository.integrity_check() == "ok"
    assert len(repository.search(ProductQuery(category="laptop"))) == 30


def test_population_extends_an_existing_catalogue_and_supports_larger_sizes(database, products):
    generated = generate_products(products, 30, 42)
    populate_catalogue(database, generated)
    populate_catalogue(database, generate_products(products, 50, 42))
    repository = SQLiteProductRepository(database)
    assert len(repository.search(ProductQuery(category="laptop"))) == 50
    actual = {product.id: product for product in repository.search(ProductQuery(category="laptop"))}
    assert all(
        actual[product.id] == product for product in products if product.category == "laptop"
    )


def test_population_preserves_modified_data_and_does_not_partially_insert(database, products):
    import sqlite3

    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE products SET price=123 WHERE id=1")
    before = database.read_bytes()
    with pytest.raises(RepositoryError, match="No data was changed"):
        populate_catalogue(database, generate_products(products, 30, 42))
    assert database.read_bytes() == before


def test_population_preserves_unrelated_rows(database, products):
    import sqlite3

    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("INSERT INTO products VALUES (999, 'Custom laptop', 'laptop', 123)")
    before = database.read_bytes()
    with pytest.raises(RepositoryError):
        populate_catalogue(database, generate_products(products, 30, 42))
    assert database.read_bytes() == before


def test_simultaneous_population_does_not_duplicate_products(tmp_path, products):
    path = tmp_path / "catalogue.sqlite3"
    generated = generate_products(products, 30, 42)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: populate_catalogue(path, generated), range(2)))
    assert len(SQLiteProductRepository(path).search(ProductQuery(category="laptop"))) == 30


def test_generated_catalogue_demonstrates_price_limits_and_stable_sorting(tmp_path, products):
    path = tmp_path / "catalogue.sqlite3"
    generated = generate_products(products, 30, 42)
    populate_catalogue(path, generated)
    repository = SQLiteProductRepository(path)
    for limit in (49999, 50000, 50001, 69999, 70000, 70001):
        actual = repository.search(
            ProductQuery(category="laptop", max_price=limit, sort_by="price_desc")
        )
        expected = sorted(
            (
                product
                for product in generated
                if product.category == "laptop" and product.price <= limit
            ),
            key=lambda product: (-product.price, product.id),
        )
        assert actual == expected
        assert 0 < len(actual) < 30
    assert repository.search(ProductQuery(category="tablet", max_price=0)) == []


def test_population_command_does_not_require_a_database_token_or_review_fixtures(
    configure, product_file, tmp_path
):
    configure(
        PRODUCTS_PATH=product_file,
        ITMO_LLM_TOKEN="",
        QUERIES_PATH="missing.json",
        DIALOGUES_PATH="missing.json",
    )
    assert populate() == 0
    assert populate() == 0
    assert (
        len(
            SQLiteProductRepository(tmp_path / "products.sqlite3").search(
                ProductQuery(category="tablet")
            )
        )
        == 30
    )


def test_population_size_and_seed_come_from_dotenv(configure, product_file):
    configure(PRODUCTS_PATH=product_file, PRODUCTS_PER_CATEGORY=40, POPULATION_SEED=7)
    result = load_population(load_settings())
    assert len(result) == 120


def test_invalid_product_file_does_not_create_a_database(configure, tmp_path):
    source = tmp_path / "products.json"
    source.write_text('[{"id":1,"name":"Bad","category":"laptop","price":-1}]')
    configure(PRODUCTS_PATH=source)
    assert populate() == 2
    assert not (tmp_path / "products.sqlite3").exists()
