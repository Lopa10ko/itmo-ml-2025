import json

import pytest
from pydantic import ValidationError

from catalogue.domain import Product, ProductQuery


def test_query_defaults_schema_and_json_round_trip():
    query = ProductQuery(category="laptop")
    assert json.loads(query.model_dump_json()) == {
        "category": "laptop",
        "max_price": None,
        "sort_by": "price_asc",
    }
    assert ProductQuery.model_validate_json(query.model_dump_json()) == query
    schema = ProductQuery.model_json_schema()
    assert schema["required"] == ["category"]
    assert schema["properties"]["category"]["enum"] == ["laptop", "smartphone", "tablet", "other"]
    assert all(field["description"] for field in schema["properties"].values())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"category": "refrigerator"},
        {"category": "laptop' OR 1=1 --"},
        {"category": "laptop", "max_price": -1},
        {"category": "laptop", "max_price": "50000"},
        {"category": "laptop", "max_price": True},
        {"category": "laptop", "max_price": 50000.0},
        {"category": "laptop", "sort_by": "price; DROP TABLE products"},
        {"category": "laptop", "sql": "SELECT * FROM products"},
    ],
)
def test_query_rejects_invalid_input(payload):
    with pytest.raises(ValidationError):
        ProductQuery.model_validate(payload)


def test_query_is_an_immutable_value_object():
    query = ProductQuery(category="tablet", max_price=0)
    assert query == ProductQuery(category="tablet", max_price=0)
    with pytest.raises(ValidationError):
        query.max_price = 100


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": 0},
        {"name": "   "},
        {"category": "other"},
        {"price": -1},
        {"price": True},
    ],
)
def test_product_rejects_invalid_records(overrides):
    record = {"id": 1, "name": "Laptop", "category": "laptop", "price": 50000}
    with pytest.raises(ValidationError):
        Product.model_validate(record | overrides)
