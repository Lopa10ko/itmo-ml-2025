from collections import Counter
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from catalogue.domain import Product, ProductCategory, ProductQuery
from catalogue.errors import InputDataError
from catalogue.population import load_population
from catalogue.settings import Settings


def read_data[Value](path: Path, adapter: TypeAdapter[Value]) -> Value:
    try:
        return adapter.validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise InputDataError(f"Could not load valid input data from {path}.") from error


class QueryCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str | None = None
    text: str = Field(min_length=1)
    expected: ProductQuery
    interpretation: str | None = None


class DialogueCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    turns: list[QueryCase] = Field(min_length=3)


class ReviewPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_products: int = Field(ge=15)
    minimum_per_category: int = Field(ge=4)
    studies: list[str]
    offline_studies: list[str]
    basic_cases: list[str] = Field(min_length=3)
    languages: list[str] = Field(min_length=2)
    json_cases: list[str] = Field(min_length=7)
    search_cases: list[str] = Field(min_length=10)
    tool_cases: list[str] = Field(min_length=7)
    thinking_cases: list[str] = Field(min_length=2)
    ungrounded_cases: list[str] = Field(min_length=3)
    thinking_modes: list[bool]
    context_modes: list[str]
    json_method: str
    tool_method: str
    ping_request: str
    ungrounded_prompt: str

    @model_validator(mode="after")
    def validate_comparisons(self) -> "ReviewPlan":
        if set(self.thinking_modes) != {False, True}:
            raise ValueError("Thinking comparison requires disabled and enabled modes.")
        if set(self.context_modes) != {"state", "history"}:
            raise ValueError("Dialogue comparison requires state and history contexts.")
        if not set(self.thinking_cases).issubset(self.json_cases):
            raise ValueError("Thinking cases must reuse structured-output cases.")
        if not set(self.json_cases).issubset(self.tool_cases):
            raise ValueError("Tool calling must include all structured-output cases.")
        if not set(self.ungrounded_cases).issubset(self.search_cases):
            raise ValueError("Ungrounded comparisons must reuse search questions.")
        return self


class RetryCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    responses: list[str] = Field(min_length=1)
    max_attempts: int = Field(ge=1, le=3)
    expected_attempts: int
    expected_searches: int


class InvalidResponse(BaseModel):
    name: str
    response: str
    expected_stage: str


class FailureCases(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: str
    errors: list[InvalidResponse] = Field(min_length=3)
    retry: list[RetryCase]
    confirmation_response: str
    confirmation_choices: list[bool]
    absent_tool_response: str


class ReviewInputs(BaseModel):
    products: list[Product]
    queries: list[QueryCase]
    dialogues: list[DialogueCase] = Field(min_length=2)
    plan: ReviewPlan
    failures: FailureCases

    @classmethod
    def load(cls, settings: Settings) -> "ReviewInputs":
        if settings.products_path is None:
            raise InputDataError("Set PRODUCTS_PATH in .env before running review scenarios.")
        try:
            result = cls(
                products=load_population(settings),
                queries=read_data(settings.queries_path, TypeAdapter(list[QueryCase])),
                dialogues=read_data(settings.dialogues_path, TypeAdapter(list[DialogueCase])),
                plan=read_data(settings.review_plan_path, TypeAdapter(ReviewPlan)),
                failures=read_data(settings.error_cases_path, TypeAdapter(FailureCases)),
            )
        except ValidationError as error:
            raise InputDataError(
                "Review inputs do not satisfy the experiment requirements."
            ) from error
        counts: Counter[str] = Counter(product.category for product in result.products)
        if len(result.products) < result.plan.minimum_products or any(
            counts[category] < result.plan.minimum_per_category
            for category in get_args(ProductCategory)
        ):
            raise InputDataError(
                "Review input contains insufficient products or category examples."
            )
        if len({product.id for product in result.products}) != len(result.products):
            raise InputDataError("Product identifiers must be unique.")
        identifiers = [case.id for case in result.queries]
        if None in identifiers or len(set(identifiers)) != len(identifiers):
            raise InputDataError("Review query identifiers must be present and unique.")
        requested = [
            *result.plan.basic_cases,
            *result.plan.json_cases,
            *result.plan.search_cases,
            *result.plan.tool_cases,
            *result.plan.thinking_cases,
            *result.plan.ungrounded_cases,
        ]
        if not set(requested).issubset(identifiers):
            raise InputDataError("The review plan references unknown query identifiers.")
        return result

    def cases(self, identifiers: list[str]) -> list[QueryCase]:
        indexed = {case.id: case for case in self.queries}
        return [indexed[identifier] for identifier in identifiers]
