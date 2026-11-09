from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from catalogue.bootstrap import Services
from catalogue.llm.models import QueryResult
from catalogue.review.inputs import QueryCase, ReviewInputs


class ReportTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    rows: list[list[JsonValue]]

    @model_validator(mode="after")
    def validate_width(self) -> "ReportTable":
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("Every report row must match the table's column count.")
        return self


class SectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    title: str
    source: str
    records: list[dict[str, JsonValue]]
    table: ReportTable
    notes: list[str] = Field(default_factory=list)
    expected_cases: list[str]


class ReviewResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    started_at: datetime
    completed_at: datetime | None = None
    mode: Literal["live", "offline"]
    metadata: dict[str, JsonValue]
    sections: list[SectionResult] = Field(default_factory=list)


@dataclass(frozen=True)
class ReviewContext:
    services: Services
    inputs: ReviewInputs

    def score_parameters(self, case: QueryCase, result: QueryResult) -> dict[str, JsonValue]:
        record = result.record()
        record.update(
            {
                "case_id": case.id,
                "expected": case.expected.model_dump(mode="json"),
                "field_checks": {
                    key: result.parameters is not None and getattr(result.parameters, key) == value
                    for key, value in case.expected.model_dump().items()
                },
                "correct": result.parameters == case.expected,
            }
        )
        return record

    def score_search(self, case: QueryCase, result: QueryResult) -> dict[str, JsonValue]:
        expected = case.expected
        matching = [
            product
            for product in self.inputs.products
            if product.category == expected.category
            and (expected.max_price is None or product.price <= expected.max_price)
        ]
        expected_ids = [
            product.id
            for product in sorted(
                matching,
                key=lambda product: (
                    -product.price if expected.sort_by == "price_desc" else product.price,
                    product.id,
                ),
            )
        ]
        record = self.score_parameters(case, result)
        record.update(
            {
                "expected_product_ids": [identifier for identifier in expected_ids],
                "products_correct": [product.id for product in result.products] == expected_ids,
                "correct": result.parameters == expected
                and [product.id for product in result.products] == expected_ids,
            }
        )
        return record


@dataclass(frozen=True)
class Experiment:
    name: str
    run: Callable[[ReviewContext], SectionResult]
