from collections.abc import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import JsonValue

from catalogue.errors import ConfigurationError
from catalogue.evidence import json_value
from catalogue.llm.client import invoke_model
from catalogue.llm.dialogue import DialogueContext, DialogueSession, HistoryContext, StateContext
from catalogue.llm.models import QueryRequest, QueryResult
from catalogue.review.models import ReportTable, ReviewContext, SectionResult


def tokens(result: QueryResult, name: str) -> int | None:
    values = [(call.usage or {}).get(name) for call in result.model_calls]
    return (
        sum(value for value in values if isinstance(value, int))
        if values and all(isinstance(value, int) for value in values)
        else None
    )


def basic_chain(context: ReviewContext) -> SectionResult:
    services, plan = context.services, context.inputs.plan
    cases = context.inputs.cases(plan.basic_cases)
    model = services.models.get()
    prompt = services.prompts.classification_template()
    chain = prompt | RunnableLambda(lambda value: invoke_model(model, value)) | StrOutputParser()
    inputs = [{"query": case.text, "language": plan.languages[0]} for case in cases]
    answers = chain.batch(inputs, config={"max_concurrency": services.settings.batch_concurrency})
    direct = invoke_model(model, prompt.invoke(inputs[0]))
    records: list[dict[str, JsonValue]] = [
        {
            "case_id": case.id,
            "input": case.text,
            "answer": str(answer),
            "correct": isinstance(answer, str),
        }
        for case, answer in zip(cases, answers, strict=True)
    ]
    records.append(
        {
            "kind": "return_types",
            "direct_type": type(direct).__name__,
            "chain_type": type(answers[0]).__name__,
            "is_string": isinstance(answers[0], str),
            "messages": [
                {"role": message.type, "content": json_value(message.content)}
                for message in prompt.invoke(inputs[0]).to_messages()
            ],
        }
    )
    language_notes = []
    for language in plan.languages[1:]:
        answer = str(chain.invoke({"query": cases[0].text, "language": language}))
        language_notes.append(f"Дополнительный запуск с `language={language}`: {answer}")
        records.append(
            {
                "kind": "language_example",
                "language": language,
                "answer": answer,
            }
        )
    return SectionResult(
        name="basic_chain",
        title="Базовая LCEL-цепочка",
        source="university_live",
        records=records,
        expected_cases=plan.basic_cases,
        table=ReportTable(
            columns=["Запрос", "Ответ"],
            rows=[[case.text, str(answer)] for case, answer in zip(cases, answers, strict=True)],
        ),
        notes=[
            f"Прямой вызов модели вернул `{type(direct).__name__}`. После парсера получен `{type(answers[0]).__name__}`; проверка `isinstance(result, str)` подтверждает, что это строка.",
            *language_notes,
        ],
    )


def structured(context: ReviewContext) -> SectionResult:
    plan = context.inputs.plan
    pipeline = context.services.pipelines.create(plan.json_method)
    records: list[dict[str, JsonValue]] = []
    rows: list[list[JsonValue]] = []
    for case in context.inputs.cases(plan.json_cases):
        result = pipeline.extract(QueryRequest(case.text))
        record = context.score_parameters(case, result)
        records.append(record)
        rows.append([case.id, case.text, record["parameters"], record["field_checks"]])
    return SectionResult(
        name="structured",
        title="Структурированный ответ: проверка полей",
        source="university_live",
        records=records,
        expected_cases=plan.json_cases,
        table=ReportTable(
            columns=["ID", "Запрос", "Полученные параметры", "Проверка полей"], rows=rows
        ),
        notes=[
            case.interpretation
            for case in context.inputs.cases(plan.json_cases)
            if case.interpretation
        ],
    )


def search(context: ReviewContext) -> SectionResult:
    plan = context.inputs.plan
    pipeline = context.services.pipelines.create(plan.json_method)
    records: list[dict[str, JsonValue]] = []
    rows: list[list[JsonValue]] = []
    for case in context.inputs.cases(plan.search_cases):
        result = pipeline.run(QueryRequest(case.text))
        record = context.score_search(case, result)
        records.append(record)
        rows.append(
            [
                case.text,
                record["parameters"],
                [f"{product.id}: {product.name} ({product.price})" for product in result.products],
                record["correct"],
            ]
        )
    return SectionResult(
        name="search",
        title="Поиск по SQLite",
        source="university_live",
        records=records,
        expected_cases=plan.search_cases,
        table=ReportTable(
            columns=[
                "Запрос пользователя",
                "Структурированные параметры",
                "Найденные товары",
                "Корректность",
            ],
            rows=rows,
        ),
    )


def tools(context: ReviewContext) -> SectionResult:
    plan = context.inputs.plan
    pipeline = context.services.pipelines.create(plan.tool_method)
    records: list[dict[str, JsonValue]] = []
    rows: list[list[JsonValue]] = []
    for case in context.inputs.cases(plan.tool_cases):
        result = pipeline.run(QueryRequest(case.text))
        record = context.score_search(case, result)
        records.append(record)
        rows.append(
            [
                case.id,
                result.model_calls[-1].content,
                json_value(result.model_calls[-1].tool_calls),
                record["parameters"],
                result.status.value,
                record["correct"],
            ]
        )
    return SectionResult(
        name="tools",
        title="Поиск как инструмент",
        source="university_live",
        records=records,
        expected_cases=plan.tool_cases,
        table=ReportTable(
            columns=["ID", "Текст ответа", "Вызовы `tool_calls`", "Параметры", "Статус", "Верно"],
            rows=rows,
        ),
        notes=[
            "Найденные товары возвращены приложению. После поиска модель повторно не вызывается."
        ],
    )


def thinking(context: ReviewContext) -> SectionResult:
    plan = context.inputs.plan
    records: list[dict[str, JsonValue]] = []
    rows: list[list[JsonValue]] = []
    identifiers = []
    for enabled in plan.thinking_modes:
        pipeline = context.services.pipelines.create(plan.json_method, thinking=enabled)
        for case in context.inputs.cases(plan.thinking_cases):
            result = pipeline.extract(QueryRequest(case.text))
            identifier = f"{case.id}/thinking={enabled}"
            record = context.score_parameters(case, result)
            record.update({"case_id": identifier, "thinking": enabled})
            identifiers.append(identifier)
            records.append(record)
            rows.append(
                [
                    case.id,
                    enabled,
                    record["parameters"],
                    result.elapsed_seconds,
                    tokens(result, "output_tokens"),
                    result.attempts,
                    result.status.value,
                ]
            )
    probe_case = context.inputs.cases(plan.thinking_cases)[0]
    probe_pipeline = context.services.pipelines.create(plan.json_method, thinking=True)
    prompt = probe_pipeline.parts.extractor.prompt.invoke(QueryRequest(probe_case.text).variables())
    diagnostic = context.services.models.raw_reasoning(prompt)
    records.append({"kind": "raw_reasoning_probe", **diagnostic})
    raw_message = diagnostic.get("message")
    observed = [
        field
        for field in context.services.settings.reasoning_fields
        if isinstance(raw_message, dict) and raw_message.get(field)
    ]
    reasoning_note = (
        "Поля рассуждения с непустым значением в исходном ответе: "
        + (", ".join(observed) or "не обнаружены")
        + ". Парсеру передаётся поле `content`. Полное диагностическое сообщение сохранено в `results.json`."
    )
    return SectionResult(
        name="thinking",
        title="Сравнение режимов размышления",
        source="university_live",
        records=records,
        expected_cases=identifiers,
        table=ReportTable(
            columns=[
                "ID",
                "Режим размышления",
                "Параметры",
                "Время, с",
                "Выходные токены",
                "Попытки",
                "Результат разбора",
            ],
            rows=rows,
        ),
        notes=[
            "Исходное сообщение сервера сохранено в записи `raw_reasoning_probe` этого раздела в `results.json`. Время диагностического запроса не включено в сравнение.",
            reasoning_note,
        ],
    )


def dialogues(context: ReviewContext) -> SectionResult:
    plan = context.inputs.plan
    builders: dict[str, Callable[[], DialogueContext]] = {
        "state": StateContext,
        "history": HistoryContext,
    }
    records: list[dict[str, JsonValue]] = []
    rows: list[list[JsonValue]] = []
    identifiers = []
    for mode in plan.context_modes:
        try:
            create_context = builders[mode]
        except KeyError as error:
            raise ConfigurationError(f"Unknown dialogue context: {mode}.") from error
        for dialogue in context.inputs.dialogues:
            session = DialogueSession(
                context.services.pipelines.create(plan.json_method), create_context()
            )
            for number, case in enumerate(dialogue.turns, 1):
                result = session.ask(case.text)
                identifier = f"{dialogue.id}/{mode}/{number}"
                record = context.score_search(case, result)
                record.update(
                    {
                        "case_id": identifier,
                        "dialogue": dialogue.id,
                        "context": mode,
                        "turn": number,
                    }
                )
                identifiers.append(identifier)
                records.append(record)
                rows.append(
                    [
                        dialogue.id,
                        mode,
                        number,
                        case.text,
                        record["expected"],
                        record["parameters"],
                        record["correct"],
                        tokens(result, "input_tokens"),
                    ]
                )
    return SectionResult(
        name="dialogues",
        title="Уточняющие запросы: параметры и история диалога",
        source="university_live",
        records=records,
        expected_cases=identifiers,
        table=ReportTable(
            columns=[
                "Диалог",
                "Контекст",
                "Ход",
                "Реплика",
                "Ожидаемые параметры",
                "Полученные параметры",
                "Верно",
                "Входные токены",
            ],
            rows=rows,
        ),
    )


def without_database(context: ReviewContext) -> SectionResult:
    plan = context.inputs.plan
    prompt = ChatPromptTemplate.from_messages(
        [("system", plan.ungrounded_prompt), ("human", "{query}")]
    )
    model = context.services.models.get()
    records: list[dict[str, JsonValue]] = []
    rows: list[list[JsonValue]] = []
    for case in context.inputs.cases(plan.ungrounded_cases):
        message = invoke_model(model, prompt.invoke({"query": case.text}))
        actual = context.services.search.execute(case.expected)
        records.append(
            {
                "case_id": case.id,
                "input": case.text,
                "answer": json_value(message.content),
                "actual_products": [product.model_dump(mode="json") for product in actual],
                "usage": json_value(message.usage_metadata),
            }
        )
        rows.append([case.text, json_value(message.content), [product.id for product in actual]])
    return SectionResult(
        name="without_database",
        title="Ответы без доступа к каталогу",
        source="university_live",
        records=records,
        expected_cases=plan.ungrounded_cases,
        table=ReportTable(
            columns=["Запрос", "Ответ модели без доступа к базе", "ID найденных товаров"], rows=rows
        ),
    )
