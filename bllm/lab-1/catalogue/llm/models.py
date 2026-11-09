from dataclasses import dataclass, field
from enum import StrEnum

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from catalogue.domain import Product, ProductQuery
from catalogue.errors import InputDataError


class Outcome(StrEnum):
    VALIDATED = "validated"
    FOUND = "found"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported_category"
    CONFIRMATION_REQUIRED = "confirmation_required"
    PARSE_ERROR = "parse_error"
    INVALID_TOOL = "invalid_tool"
    FAILED = "failed"


class ExecutionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(ge=1, le=3)
    require_confirmation: bool = False
    confirmed: bool = False


@dataclass(frozen=True)
class QueryRequest:
    text: str
    current: ProductQuery | None = None
    history: tuple[BaseMessage, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise InputDataError("A non-empty product request is required.")
        if self.current is not None and self.history:
            raise InputDataError("Supply current parameters or dialogue history, not both.")

    def variables(self) -> dict[str, object]:
        return {
            "query": self.text,
            "context": f"Current parameters: {self.current.model_dump_json()}\n"
            if self.current
            else "",
            "history": list(self.history),
        }


class ModelCall(BaseModel):
    attempt: int
    elapsed_seconds: float
    content: JsonValue
    usage: dict[str, JsonValue] | None
    tool_calls: list[dict[str, JsonValue]]
    invalid_tool_calls: list[dict[str, JsonValue]]
    reasoning: dict[str, JsonValue]


@dataclass
class QueryResult:
    request_id: str
    input: str
    parameters: ProductQuery | None = None
    products: list[Product] = field(default_factory=list)
    status: Outcome = Outcome.FAILED
    attempts: int = 0
    model_calls: list[ModelCall] = field(default_factory=list)
    elapsed_seconds: float = 0
    error: str | None = None

    def record(self) -> dict[str, JsonValue]:
        return {
            "request_id": self.request_id,
            "input": self.input,
            "parameters": self.parameters.model_dump(mode="json") if self.parameters else None,
            "products": [product.model_dump(mode="json") for product in self.products],
            "status": self.status.value,
            "attempts": self.attempts,
            "model_calls": [call.model_dump(mode="json") for call in self.model_calls],
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }
