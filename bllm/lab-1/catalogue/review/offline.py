from dataclasses import dataclass
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import JsonValue, ValidationError

from catalogue.application import SearchProducts
from catalogue.domain import Product, ProductQuery, ProductRepository
from catalogue.llm.extraction import json_parts, tool_parts
from catalogue.llm.models import ExecutionPolicy, QueryRequest
from catalogue.llm.pipeline import CataloguePipeline
from catalogue.review.models import ReportTable, ReviewContext, SectionResult


@dataclass
class CountingRepository:
    repository: ProductRepository
    calls: int = 0

    def search(self, query: ProductQuery) -> list[Product]:
        self.calls += 1
        return self.repository.search(query)


class ScriptedToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolModel":
        return self


def failure_handling(context: ReviewContext) -> SectionResult:
    services, cases = context.services, context.inputs.failures
    parser: PydanticOutputParser[ProductQuery] = PydanticOutputParser(pydantic_object=ProductQuery)
    records: list[dict[str, JsonValue]] = []
    rows: list[list[JsonValue]] = []
    for case in cases.errors:
        try:
            parser.invoke(case.response)
        except OutputParserException as error:
            stage = "validation" if isinstance(error.__cause__, ValidationError) else "json_parsing"
            record: dict[str, JsonValue] = {
                "case_id": case.name,
                "response": case.response,
                "exception": type(error).__name__,
                "cause": type(error.__cause__).__name__,
                "stage": stage,
                "correct": stage == case.expected_stage,
            }
        else:
            record = {"case_id": case.name, "response": case.response, "correct": False}
        records.append(record)
        rows.append(
            [
                case.name,
                case.response,
                f"Исключение: {record.get('exception')}; причина: {record.get('cause')}; этап: {record.get('stage')}",
                record["correct"],
            ]
        )
    for retry_case in cases.retry:
        repository = CountingRepository(services.repository)
        model = FakeMessagesListChatModel(
            responses=[AIMessage(content=text) for text in retry_case.responses]
        )
        pipeline = CataloguePipeline(
            json_parts(
                model,
                services.prompts,
                SearchProducts(repository),
                services.settings.search_tool_name,
            ),
            ExecutionPolicy(max_attempts=retry_case.max_attempts),
            services.journal,
        )
        result = pipeline.run(QueryRequest(cases.request))
        record = result.record()
        record.update(
            {
                "case_id": retry_case.name,
                "searches": repository.calls,
                "correct": result.attempts == retry_case.expected_attempts
                and repository.calls == retry_case.expected_searches,
            }
        )
        records.append(record)
        rows.append(
            [
                retry_case.name,
                "Число попыток модели и вызовов поиска",
                f"Модель: {result.attempts}; поиск: {repository.calls}; статус: {result.status.value}",
                record["correct"],
            ]
        )
    for confirmed in cases.confirmation_choices:
        repository = CountingRepository(services.repository)
        model = FakeMessagesListChatModel(
            responses=[AIMessage(content=cases.confirmation_response)]
        )
        pipeline = CataloguePipeline(
            json_parts(
                model,
                services.prompts,
                SearchProducts(repository),
                services.settings.search_tool_name,
            ),
            ExecutionPolicy(max_attempts=1, require_confirmation=True, confirmed=confirmed),
            services.journal,
        )
        result = pipeline.run(QueryRequest(cases.request))
        record = result.record()
        record.update(
            {
                "case_id": f"confirmation/{confirmed}",
                "searches": repository.calls,
                "correct": repository.calls == int(confirmed),
            }
        )
        records.append(record)
        rows.append(
            [
                record["case_id"],
                "Число попыток модели и вызовов поиска",
                f"Модель: {result.attempts}; поиск: {repository.calls}; статус: {result.status.value}",
                record["correct"],
            ]
        )
    repository = CountingRepository(services.repository)
    pipeline = CataloguePipeline(
        tool_parts(
            ScriptedToolModel(responses=[AIMessage(content=cases.absent_tool_response)]),
            services.prompts,
            SearchProducts(repository),
            services.settings.search_tool_name,
        ),
        ExecutionPolicy(max_attempts=1),
        services.journal,
    )
    result = pipeline.run(QueryRequest(cases.request))
    record = result.record()
    record.update(
        {
            "case_id": "absent_tool",
            "searches": repository.calls,
            "correct": result.status == "invalid_tool" and repository.calls == 0,
        }
    )
    records.append(record)
    rows.append(
        [
            "absent_tool",
            "Число попыток модели и вызовов поиска",
            f"Модель: {result.attempts}; поиск: {repository.calls}; статус: {result.status.value}",
            record["correct"],
        ]
    )
    return SectionResult(
        name="failure_handling",
        title="Ошибки, повторные попытки и подтверждение",
        source="scripted_model_real_sqlite",
        records=records,
        expected_cases=[str(record["case_id"]) for record in records],
        table=ReportTable(
            columns=["Сценарий", "Проверяемые данные", "Фактический результат", "Верно"],
            rows=rows,
        ),
        notes=[
            "Ответы модели в этом разделе заданы заранее. Используются те же парсер, механизм повторных попыток LCEL и репозиторий SQLite, что и в основном пайплайне. События запроса можно найти в `events.jsonl` по его `request_id`."
        ],
    )
