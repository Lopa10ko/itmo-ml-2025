import json
from pathlib import Path

from catalogue.entrypoints import initialize, search


def test_initialization_and_search_use_dotenv(tmp_path, product_file, capsys, configure):
    database = tmp_path / "catalogue.sqlite3"
    configure(
        DATABASE_PATH=database,
        PRODUCT_QUERY=json.dumps({"category": "laptop", "max_price": 50000}),
        PRODUCTS_PATH=product_file,
    )
    assert initialize() == 0
    capsys.readouterr()
    assert search() == 0
    output = json.loads(capsys.readouterr().out)
    assert [product["id"] for product in output["products"]] == [1, 2]


def test_empty_results_have_a_message(database, capsys, configure):
    configure(
        DATABASE_PATH=database, PRODUCT_QUERY=json.dumps({"category": "tablet", "max_price": 0})
    )
    assert search() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["products"] == []
    assert output["message"] == "No products found."


def test_invalid_query_does_not_touch_storage(tmp_path, capsys, configure):
    database = tmp_path / "missing.sqlite3"
    configure(
        DATABASE_PATH=database, PRODUCT_QUERY=json.dumps({"category": "laptop", "max_price": True})
    )
    assert search() == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err
    assert not database.exists()


def test_unsupported_category_does_not_touch_storage(tmp_path, capsys, configure):
    database = tmp_path / "missing.sqlite3"
    configure(DATABASE_PATH=database, PRODUCT_QUERY=json.dumps({"category": "other"}))
    assert search() == 2
    assert "not supported" in capsys.readouterr().err
    assert not database.exists()


def test_missing_database_is_reported(tmp_path, capsys, configure):
    database = tmp_path / "missing.sqlite3"
    configure(DATABASE_PATH=database, PRODUCT_QUERY=json.dumps({"category": "laptop"}))
    assert search() == 2
    assert "Error:" in capsys.readouterr().err
    assert not database.exists()


def test_existing_database_is_preserved(tmp_path, capsys, configure):
    database = tmp_path / "catalogue.sqlite3"
    configure(DATABASE_PATH=database)
    assert initialize() == 0
    capsys.readouterr()
    original_bytes = database.read_bytes()
    assert initialize() == 2
    assert "No data was changed" in capsys.readouterr().err
    assert database.read_bytes() == original_bytes


def test_custom_products_are_loaded_from_configured_file(tmp_path, capsys, configure):
    database = tmp_path / "custom.sqlite3"
    products = tmp_path / "custom.json"
    products.write_text(
        json.dumps([{"id": 101, "name": "Custom tablet", "category": "tablet", "price": 123}])
    )
    configure(
        DATABASE_PATH=database,
        PRODUCT_QUERY=json.dumps({"category": "tablet"}),
        PRODUCTS_PATH=products,
    )
    assert initialize() == 0
    capsys.readouterr()
    assert search() == 0
    output = json.loads(capsys.readouterr().out)
    assert [product["id"] for product in output["products"]] == [101]


def test_invalid_products_are_rejected_before_database_creation(tmp_path, capsys, configure):
    database = tmp_path / "invalid.sqlite3"
    products = tmp_path / "invalid.json"
    products.write_text('[{"id":1,"name":"Laptop","category":"laptop","price":-1}]')
    configure(DATABASE_PATH=database, PRODUCTS_PATH=products)
    assert initialize() == 2
    assert "Error:" in capsys.readouterr().err
    assert not database.exists()


def test_search_requires_a_configured_query(tmp_path, capsys, configure):
    database = tmp_path / "missing.sqlite3"
    configure(DATABASE_PATH=database)
    assert search() == 2
    assert "PRODUCT_QUERY must be set" in capsys.readouterr().err
    assert not database.exists()


def test_configuration_errors_do_not_expose_values(capsys, configure):
    Path(".env").write_text("DATABASE_PATH=example.sqlite3\nUNEXPECTED_KEY=private-value")
    assert initialize() == 2
    output = capsys.readouterr()
    assert "private-value" not in output.err
    assert "unexpected_key" in output.err.lower()
