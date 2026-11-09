import json
from collections import defaultdict

from pydantic import JsonValue, ValidationError

from catalogue.bootstrap import Services
from catalogue.errors import EvidenceError
from catalogue.evidence import json_value
from catalogue.review.inputs import ReviewInputs
from catalogue.review.models import ReviewResults


def load_results(services: Services) -> ReviewResults:
    try:
        return ReviewResults.model_validate(
            services.store.read_json(services.settings.results_filename)
        )
    except ValidationError as error:
        raise EvidenceError("The saved review results have an invalid structure.") from error


def validate_results(results: ReviewResults, inputs: ReviewInputs) -> None:
    if results.completed_at is None:
        raise EvidenceError("The review is incomplete; rerun the scenarios before reporting.")
    expected = inputs.plan.studies if results.mode == "live" else inputs.plan.offline_studies
    actual = [section.name for section in results.sections]
    if actual != expected:
        raise EvidenceError("The saved review sections do not match the configured plan.")
    for section in results.sections:
        identifiers = [record["case_id"] for record in section.records if "case_id" in record]
        if identifiers != section.expected_cases or len(set(section.expected_cases)) != len(
            section.expected_cases
        ):
            raise EvidenceError(f"The {section.name} evidence has missing or duplicate cases.")
        if any(record.get("correct") is False for record in section.records):
            raise EvidenceError(
                f"The {section.name} experiment contains failed expectations; inspect results.json."
            )

    if results.metadata.get("catalogue") != [
        product.model_dump(mode="json") for product in inputs.products
    ]:
        raise EvidenceError(
            "The saved catalogue differs from the configured population; rerun the experiments."
        )


def cell(value: JsonValue) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Да" if value else "Нет"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text.replace("|", "\\|").replace("\n", "<br>") or "—"


def build_report(services: Services, inputs: ReviewInputs) -> None:
    results = load_results(services)
    validate_results(results, inputs)
    try:
        template = services.settings.report_template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EvidenceError("Could not read the configured report template.") from error
    source_labels = {
        "university_live": "университетская модель",
        "scripted_model_real_sqlite": "заранее заданные ответы модели",
    }
    sections = []
    for section in results.sections:
        lines = [
            f"## {section.title}",
            "",
            f"Источник ответов: {source_labels.get(section.source, section.source)}.",
            "",
            "| " + " | ".join(section.table.columns) + " |",
            "| " + " | ".join("---" for _ in section.table.columns) + " |",
        ]
        lines.extend(
            "| " + " | ".join(cell(value) for value in row) + " |" for row in section.table.rows
        )
        lines.extend("\n" + note for note in section.notes)
        sections.append("\n".join(lines))
    grouped: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    for event in read_journal(services):
        grouped[str(event["request_id"])].append(event)
    recovered = next(
        (
            sequence
            for sequence in grouped.values()
            if any(event.get("event") == "retry_scheduled" for event in sequence)
            and any(event.get("event") == "search_completed" for event in sequence)
        ),
        None,
    )
    if recovered is not None:
        excerpt = [
            {key: value for key, value in event.items() if key != "messages"} for event in recovered
        ]
        sections.append(
            "## Фрагмент журнала: успешный повтор после ошибки\n\n```json\n"
            + json.dumps(excerpt, ensure_ascii=False, indent=2)
            + "\n```"
        )
    metadata_lines = [
        f"Режим выполнения: `{results.mode}`. Начало: {results.started_at.isoformat()}. Завершение: {results.completed_at.isoformat() if results.completed_at else ''}. Время указано в UTC.",
        "",
        "| Параметр | Значение |",
        "| --- | --- |",
    ]
    metadata_labels = {
        "database_integrity": "Проверка целостности SQLite",
        "versions": "Версии библиотек",
        "model": "Модель",
        "base_url": "Адрес API",
        "temperature": "Температура генерации",
        "max_tokens": "Лимит выходных токенов",
        "retry_attempts": "Максимальное число попыток",
        "population": "Параметры генерации каталога",
        "connection": "Проверка подключения",
    }
    metadata_lines.extend(
        f"| {metadata_labels.get(key, key)} | {cell(value)} |"
        for key, value in results.metadata.items()
        if key not in {"catalogue", "schema", "selected_studies"}
    )
    catalogue = results.metadata.get("catalogue")
    if isinstance(catalogue, list):
        metadata_lines.extend(
            [
                "",
                "### Каталог эксперимента",
                "",
                "| ID | Название | Категория | Цена, руб. |",
                "| --- | --- | --- | --- |",
            ]
        )
        for product in catalogue:
            if isinstance(product, dict):
                metadata_lines.append(
                    "| "
                    + " | ".join(
                        cell(product.get(key)) for key in ("id", "name", "category", "price")
                    )
                    + " |"
                )
    metadata = "\n".join(metadata_lines)
    services.store.write_text(
        services.settings.report_filename,
        template.replace("{{metadata}}", metadata).replace("{{sections}}", "\n\n".join(sections)),
    )


def read_journal(services: Services) -> list[dict[str, JsonValue]]:
    try:
        lines = (
            (services.store.directory / services.settings.events_filename)
            .read_text(encoding="utf-8")
            .splitlines()
        )
        events: list[dict[str, JsonValue]] = []
        for line in lines:
            event = json_value(json.loads(line))
            if not isinstance(event, dict) or not isinstance(event.get("request_id"), str):
                raise EvidenceError("An execution journal event has an invalid structure.")
            events.append(event)
        return events
    except (OSError, ValueError) as error:
        raise EvidenceError("Could not read a valid execution journal.") from error


def verify(services: Services, inputs: ReviewInputs) -> None:
    results = load_results(services)
    validate_results(results, inputs)
    try:
        paths = [
            services.store.directory / name
            for name in (
                services.settings.results_filename,
                services.settings.events_filename,
                services.settings.report_filename,
            )
        ]
        contents = [path.read_text(encoding="utf-8") for path in paths]
    except (OSError, ValueError) as error:
        raise EvidenceError(
            "The report, results and execution journal must all be readable."
        ) from error
    by_request: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    for event in read_journal(services):
        by_request[str(event["request_id"])].append(event)
    for sequence in by_request.values():
        names = [event.get("event") for event in sequence]
        attempts = [
            event.get("attempt") for event in sequence if event.get("event") == "llm_started"
        ]
        if attempts != list(range(1, len(attempts) + 1)) or len(attempts) > 3:
            raise EvidenceError("The execution journal contains invalid attempt numbering.")
        if names.count("search_started") > 1 or names.count("search_completed") > 1:
            raise EvidenceError("A request performed multiple catalogue searches.")
        if "search_started" in names and (
            "validated" not in names or names.index("validated") > names.index("search_started")
        ):
            raise EvidenceError("The execution journal contains a search before validation.")
        if "retry_exhausted" in names and "search_started" in names:
            raise EvidenceError("A failed extraction reached the database.")
    for secret in services.store.secrets:
        if secret and any(secret in content for content in contents):
            raise EvidenceError("A credential was found in the saved evidence.")
    for section in results.sections:
        for record in section.records:
            request_id = record.get("request_id")
            if isinstance(request_id, str) and request_id not in by_request:
                raise EvidenceError("A saved request has no execution journal events.")
    if not by_request:
        raise EvidenceError("The execution journal contains no requests.")
    print(
        f"Verified {len(results.sections)} sections and {len(by_request)} journaled requests ({results.mode})."
    )
