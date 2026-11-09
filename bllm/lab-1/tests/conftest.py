from pathlib import Path

import pytest
from pydantic import TypeAdapter

from catalogue.domain import Product
from catalogue.persistence.sqlite import initialize_catalogue


@pytest.fixture
def product_file() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "products.json"


@pytest.fixture
def products(product_file: Path) -> list[Product]:
    return TypeAdapter(list[Product]).validate_json(product_file.read_bytes())


@pytest.fixture
def database(tmp_path: Path, products: list[Product]) -> Path:
    path = tmp_path / "catalogue ü ?# products.sqlite3"
    initialize_catalogue(path, products)
    return path


@pytest.fixture
def configure(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    defaults = (root / ".env.example").read_text()
    monkeypatch.chdir(tmp_path)

    def write(**overrides):
        values = dict(line.split("=", 1) for line in defaults.splitlines() if line)
        for key, value in values.items():
            if key.endswith("_PATH"):
                values[key] = str(root / value)
        values.update(
            {
                "DATABASE_PATH": str(tmp_path / "products.sqlite3"),
                "ARTIFACTS_PATH": str(tmp_path / "artifacts"),
                "QUERY_TEXT": "",
                "PRODUCT_QUERY": "",
                "PRODUCTS_PATH": "",
            }
        )
        values.update({key: str(value) for key, value in overrides.items()})
        Path(".env").write_text("\n".join(f"{key}={value}" for key, value in values.items()))
        return Path(".env")

    return write


@pytest.fixture
def prompts():
    from catalogue.llm.prompts import PromptSet

    root = Path(__file__).parents[1] / "prompts"
    return PromptSet(
        *(
            root.joinpath(name).read_text()
            for name in (
                "classification.txt",
                "extraction.txt",
                "json.txt",
                "tools.txt",
                "search_tool.txt",
                "request.txt",
            )
        )
    )
