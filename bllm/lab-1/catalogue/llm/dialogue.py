from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from catalogue.domain import ProductQuery
from catalogue.llm.models import QueryRequest, QueryResult
from catalogue.llm.pipeline import CataloguePipeline


class DialogueContext(Protocol):
    def request(self, text: str) -> QueryRequest: ...

    def update(self, text: str, result: QueryResult) -> None: ...


@dataclass
class StateContext:
    current: ProductQuery | None = None

    def request(self, text: str) -> QueryRequest:
        return QueryRequest(text, current=self.current)

    def update(self, text: str, result: QueryResult) -> None:
        if result.parameters is not None:
            self.current = result.parameters


@dataclass
class HistoryContext:
    messages: list[BaseMessage] = field(default_factory=list)

    def request(self, text: str) -> QueryRequest:
        return QueryRequest(text, history=tuple(self.messages))

    def update(self, text: str, result: QueryResult) -> None:
        self.messages.append(HumanMessage(content=text))
        self.messages.append(
            AIMessage(
                content=result.parameters.model_dump_json()
                if result.parameters
                else str(result.error)
            )
        )


@dataclass
class DialogueSession:
    pipeline: CataloguePipeline
    context: DialogueContext

    def ask(self, text: str) -> QueryResult:
        result = self.pipeline.run(self.context.request(text))
        self.context.update(text, result)
        return result
