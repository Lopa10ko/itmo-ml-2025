from pathlib import Path

import pytest
from pydantic import ValidationError

from catalogue.domain import ProductQuery
from catalogue.errors import ConfigurationError
from catalogue.settings import Settings, load_settings


def test_settings_load_from_dotenv(configure):
    configure(
        DATABASE_PATH="products.sqlite3",
        PRODUCTS_PATH="fixtures/products.json",
        PRODUCT_QUERY='{"category":"laptop","max_price":50000}',
    )
    settings = Settings()
    assert settings.database_path == Path("products.sqlite3")
    assert settings.products_path == Path("fixtures/products.json")
    assert settings.product_query == ProductQuery(category="laptop", max_price=50000)


def test_blank_optional_settings_are_ignored(configure):
    configure()
    assert Settings().products_path is None
    assert Settings().product_query is None


def test_database_path_is_required(configure):
    path = configure()
    path.write_text(
        "\n".join(
            line for line in path.read_text().splitlines() if not line.startswith("DATABASE_PATH=")
        )
    )
    with pytest.raises(ValidationError):
        Settings()


def test_process_environment_cannot_override_dotenv(configure, monkeypatch):
    configure(DATABASE_PATH="from-file.sqlite3")
    monkeypatch.setenv("DATABASE_PATH", "from-process.sqlite3")
    assert Settings().database_path == Path("from-file.sqlite3")


def test_process_environment_does_not_replace_missing_dotenv(configure, monkeypatch):
    configure().unlink()
    monkeypatch.setenv("DATABASE_PATH", "from-process.sqlite3")
    with pytest.raises(ConfigurationError):
        load_settings()


@pytest.mark.parametrize("query", ['{"category":"laptop","max_price":"50000"}', "not-json"])
def test_invalid_queries_are_rejected(configure, query):
    configure(PRODUCT_QUERY=query)
    with pytest.raises(ConfigurationError):
        load_settings()


def test_unknown_settings_do_not_expose_values(configure):
    configure(UNEXPECTED_KEY="private-value")
    with pytest.raises(ConfigurationError) as captured:
        load_settings()
    assert "private-value" not in str(captured.value)
