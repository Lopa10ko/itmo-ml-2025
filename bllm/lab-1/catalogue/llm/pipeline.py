from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import JsonValue, ValidationError

from catalogue.domain import Product, ProductQuery
from catalogue.errors import (
    ConfigurationError,
    ModelRequestError,
    RepositoryError,
    ToolCallError,
    UnsupportedCategory,
)
from catalogue.evidence import Journal, json_value
from catalogue.llm.client import invoke_model
from catalogue.llm.extraction import PipelineParts
from catalogue.llm.models import ExecutionPolicy, ModelCall, Outcome, QueryRequest, QueryResult


@dataclass
class RequestExecution:
    parts: PipelineParts
    policy: ExecutionPolicy
    journal: Journal
    request: QueryRequest
    reasoning_fields: tuple[str, ...]
    result: QueryResult = field(init=False)

    def __post_init__(self) -> None:
        self.result = QueryResult(uuid4().hex, self.request.text)
        self.record("input_received", text=self.request.text)

    def record(self, event: str, **details: JsonValue) -> None:
        self.journal.record(self.result.request_id, event, **details)

    def begin(self, variables: dict[str, object]) -> dict[str, object]:
        self.result.attempts += 1
        return variables

    def format_prompt(self, variables: dict[str, object]) -> PromptValue:
        try:
            return self.parts.extractor.prompt.invoke(variables)
        except KeyError as error:
            raise ConfigurationError(
                "Prompt variables do not match the configured templates."
            ) from error

    def invoke(self, prompt: PromptValue) -> AIMessage:
        attempt = self.result.attempts
        self.record(
            "prompt_formed",
            attempt=attempt,
            messages=[
                {"role": message.type, "content": json_value(message.content)}
                for message in prompt.to_messages()
            ],
        )
        self.record("llm_started", attempt=attempt)
        started = perf_counter()
        try:
            message = invoke_model(self.parts.extractor.model, prompt)
        except ModelRequestError:
            self.record("llm_failed", attempt=attempt)
            raise
        call = ModelCall.model_validate(
            {
                "attempt": attempt,
                "elapsed_seconds": perf_counter() - started,
                "content": json_value(message.content),
                "usage": message.usage_metadata,
                "tool_calls": message.tool_calls,
                "invalid_tool_calls": message.invalid_tool_calls,
                "reasoning": {
                    key: value
                    for key, value in message.additional_kwargs.items()
                    if key in self.reasoning_fields
                },
            }
        )
        self.result.model_calls.append(call)
        self.record(
            "llm_completed", attempt=attempt, elapsed_seconds=call.elapsed_seconds, usage=call.usage
        )
        return message

    def parse(self, message: AIMessage) -> ProductQuery:
        try:
            query = self.parts.extractor.parse(message)
        except OutputParserException as error:
            self.record(
                "parse_failed",
                attempt=self.result.attempts,
                error_type=type(error).__name__,
                cause_type=type(error.__cause__).__name__,
                stage="validation"
                if isinstance(error.__cause__, ValidationError)
                else "json_parsing",
            )
            if self.result.attempts < self.policy.max_attempts:
                self.record("retry_scheduled", next_attempt=self.result.attempts + 1)
            raise
        self.result.parameters = query
        self.result.status = Outcome.VALIDATED
        self.record("parsed", attempt=self.result.attempts)
        self.record("validated", parameters=query.model_dump(mode="json"))
        return query

    def find(self, query: ProductQuery) -> list[Product]:
        if (
            self.policy.require_confirmation
            and query.category != "other"
            and query.max_price is None
            and not self.policy.confirmed
        ):
            self.result.status = Outcome.CONFIRMATION_REQUIRED
            self.result.error = "Confirm the unrestricted search in .env before proceeding."
            self.record("search_skipped", reason="confirmation_required")
            return []
        self.result.status = Outcome.FAILED
        self.record("search_started", parameters=query.model_dump(mode="json"))
        try:
            products = self.parts.search(query)
        except UnsupportedCategory:
            self.result.status = Outcome.UNSUPPORTED
            self.result.error = "The requested product category is not supported."
            self.record("search_skipped", reason="unsupported_category")
            return []
        except RepositoryError:
            self.record("search_failed")
            raise
        self.result.products = products
        self.result.status = Outcome.FOUND if products else Outcome.EMPTY
        self.record("search_completed", count=len(products))
        return products

    def extraction_chain(self) -> Runnable[dict[str, object], ProductQuery]:
        return (
            RunnableLambda(self.begin)
            | RunnableLambda(self.format_prompt)
            | RunnableLambda(self.invoke)
            | RunnableLambda(self.parse)
        ).with_retry(
            stop_after_attempt=self.policy.max_attempts,
            retry_if_exception_type=(OutputParserException,),
            wait_exponential_jitter=False,
        )

    def complete(self, chain: Runnable[dict[str, object], object]) -> QueryResult:
        started = perf_counter()
        try:
            chain.invoke(self.request.variables())
        except OutputParserException:
            self.result.status = Outcome.PARSE_ERROR
            self.result.error = (
                f"Model output failed validation after {self.result.attempts} attempt(s)."
            )
            self.record("retry_exhausted", attempts=self.result.attempts)
        except ToolCallError as error:
            self.result.status = Outcome.INVALID_TOOL
            self.result.error = str(error)
            self.record("tool_rejected", reason=str(error))
        finally:
            self.result.elapsed_seconds = perf_counter() - started
            self.record(
                "output_delivered", status=self.result.status.value, count=len(self.result.products)
            )
        return self.result


@dataclass(frozen=True)
class CataloguePipeline:
    parts: PipelineParts
    policy: ExecutionPolicy
    journal: Journal
    reasoning_fields: tuple[str, ...] = ()

    def execution(self, request: QueryRequest) -> RequestExecution:
        return RequestExecution(
            self.parts, self.policy, self.journal, request, self.reasoning_fields
        )

    def extract(self, request: QueryRequest) -> QueryResult:
        execution = self.execution(request)
        return execution.complete(execution.extraction_chain())

    def run(self, request: QueryRequest) -> QueryResult:
        execution = self.execution(request)
        return execution.complete(execution.extraction_chain() | RunnableLambda(execution.find))
