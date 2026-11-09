import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from catalogue.application import SearchProducts
from catalogue.domain import ProductQuery
from catalogue.errors import InputDataError, RepositoryError
from catalogue.evidence import Journal
from catalogue.llm.dialogue import DialogueSession, StateContext
from catalogue.llm.extraction import PipelineParts, json_parts, tool_parts
from catalogue.llm.models import ExecutionPolicy, QueryRequest
from catalogue.llm.pipeline import CataloguePipeline
from catalogue.persistence.sqlite import SQLiteProductRepository
from catalogue.review.offline import CountingRepository, ScriptedToolModel

VALID = '{"category":"laptop","max_price":50000,"sort_by":"price_asc"}'


def make_pipeline(database, prompts, responses, attempts=3, confirmed=False, confirmation=False):
    repository = CountingRepository(SQLiteProductRepository(database))
    model = FakeMessagesListChatModel(responses=[AIMessage(content=text) for text in responses])
    pipeline = CataloguePipeline(
        json_parts(model, prompts, SearchProducts(repository), "search_products"),
        ExecutionPolicy(
            max_attempts=attempts, require_confirmation=confirmation, confirmed=confirmed
        ),
        Journal(),
    )
    return pipeline, repository


@pytest.mark.parametrize(
    "responses,attempts,expected_attempts,searches,status",
    [
        (["not-json", VALID], 3, 2, 1, "found"),
        (["not-json", VALID], 1, 1, 0, "parse_error"),
        (["not-json"], 3, 3, 0, "parse_error"),
        ([VALID], 3, 1, 1, "found"),
        (['{"category":"laptop","max_price":-1}'], 3, 3, 0, "parse_error"),
    ],
)
def test_retry_and_validation_boundary(
    database, prompts, responses, attempts, expected_attempts, searches, status
):
    pipeline, repository = make_pipeline(database, prompts, responses, attempts)
    result = pipeline.run(QueryRequest("Find laptops"))
    assert result.status == status
    assert result.attempts == expected_attempts
    assert repository.calls == searches
    events = [event["event"] for event in pipeline.journal.events]
    if searches:
        assert (
            events.index("validated")
            < events.index("search_started")
            < events.index("search_completed")
        )
        assert [product.id for product in result.products] == [1, 2]
    else:
        assert "search_started" not in events


def test_database_errors_do_not_restart_generation(database, prompts):
    pipeline, _ = make_pipeline(database, prompts, [VALID])

    def fail(query):
        raise RepositoryError("database unavailable")

    pipeline = CataloguePipeline(
        PipelineParts(pipeline.parts.extractor, fail), pipeline.policy, pipeline.journal
    )
    with pytest.raises(RepositoryError):
        pipeline.run(QueryRequest("Find laptops"))
    assert sum(event["event"] == "llm_started" for event in pipeline.journal.events) == 1
    assert any(event["event"] == "search_failed" for event in pipeline.journal.events)


@pytest.mark.parametrize(
    "confirmed,searches,status", [(False, 0, "confirmation_required"), (True, 1, "found")]
)
def test_confirmation_precedes_search(database, prompts, confirmed, searches, status):
    pipeline, repository = make_pipeline(
        database, prompts, ['{"category":"laptop"}'], confirmed=confirmed, confirmation=True
    )
    assert pipeline.run(QueryRequest("Find laptops")).status == status
    assert repository.calls == searches


def test_unsupported_category_does_not_query_database(database, prompts):
    pipeline, repository = make_pipeline(database, prompts, ['{"category":"other"}'])
    assert pipeline.run(QueryRequest("Refrigerators")).status == "unsupported_category"
    assert repository.calls == 0


@pytest.mark.parametrize(
    "message,status,searches",
    [
        (AIMessage(content="No call"), "invalid_tool", 0),
        (
            AIMessage(content="", tool_calls=[{"name": "unknown", "args": {}, "id": "a"}]),
            "invalid_tool",
            0,
        ),
        (
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_products", "args": {"category": "laptop"}, "id": "a"},
                    {"name": "search_products", "args": {"category": "tablet"}, "id": "b"},
                ],
            ),
            "invalid_tool",
            0,
        ),
        (
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {"name": "search_products", "args": "bad", "id": "a", "error": "invalid"}
                ],
            ),
            "invalid_tool",
            0,
        ),
        (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_products",
                        "args": {"category": "laptop", "max_price": True},
                        "id": "a",
                    }
                ],
            ),
            "parse_error",
            0,
        ),
        (
            AIMessage(
                content="",
                tool_calls=[{"name": "search_products", "args": json.loads(VALID), "id": "a"}],
            ),
            "found",
            1,
        ),
    ],
)
def test_tool_calls_validate_before_search(database, prompts, message, status, searches):
    repository = CountingRepository(SQLiteProductRepository(database))
    pipeline = CataloguePipeline(
        tool_parts(
            ScriptedToolModel(responses=[message]),
            prompts,
            SearchProducts(repository),
            "search_products",
        ),
        ExecutionPolicy(max_attempts=3),
        Journal(),
    )
    result = pipeline.run(QueryRequest("Laptops"))
    assert result.status == status
    assert repository.calls == searches
    if searches:
        assert len(result.model_calls) == 1
        assert [product.id for product in result.products] == [1, 2]


def test_failed_turn_retains_last_valid_state(database, prompts):
    pipeline, repository = make_pipeline(database, prompts, [VALID, "not-json"], attempts=1)
    context = StateContext()
    session = DialogueSession(pipeline, context)
    session.ask("Laptops")
    session.ask("Refine")
    assert context.current == ProductQuery.model_validate_json(VALID)
    assert repository.calls == 1


def test_batch_requests_have_isolated_execution_state(database, prompts):
    pipeline, repository = make_pipeline(database, prompts, [VALID])
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                lambda text: pipeline.run(QueryRequest(text)), ["First", "Second", "Third"]
            )
        )
    assert repository.calls == 3
    assert len({result.request_id for result in results}) == 3
    assert all(result.attempts == 1 for result in results)
    assert all(
        sum(
            event["event"] == "llm_started" and event["request_id"] == result.request_id
            for event in pipeline.journal.events
        )
        == 1
        for result in results
    )


def test_state_and_history_cannot_be_mixed():
    with pytest.raises(InputDataError, match="not both"):
        QueryRequest("Laptops", ProductQuery(category="laptop"), (AIMessage(content=VALID),))


def test_history_prompt_has_no_state_reset(database, prompts):
    captured = []

    def respond(prompt):
        captured.extend(prompt.to_messages())
        return AIMessage(content=VALID)

    pipeline, _ = make_pipeline(database, prompts, [VALID])
    extractor = pipeline.parts.extractor
    from catalogue.llm.extraction import JsonExtractor

    replacement = JsonExtractor(RunnableLambda(respond), extractor.prompt, extractor.parser)
    pipeline = CataloguePipeline(
        PipelineParts(replacement, pipeline.parts.search), pipeline.policy, pipeline.journal
    )
    pipeline.run(
        QueryRequest(
            "Switch category", history=(HumanMessage(content="Previous"), AIMessage(content=VALID))
        )
    )
    assert captured[1].content == "Previous"
    assert captured[2].content == VALID
    assert "Current parameters" not in captured[-1].content


def test_new_extractor_requires_no_pipeline_changes(database, prompts):
    pipeline, _ = make_pipeline(database, prompts, [VALID])

    class FixedExtractor:
        model = pipeline.parts.extractor.model
        prompt = pipeline.parts.extractor.prompt

        def parse(self, message):
            return ProductQuery(category="tablet", max_price=0)

    extended = CataloguePipeline(
        PipelineParts(FixedExtractor(), pipeline.parts.search), pipeline.policy, pipeline.journal
    )
    assert extended.run(QueryRequest("Tablets")).status == "empty"


def test_programming_errors_are_not_hidden(database, prompts):
    pipeline, _ = make_pipeline(database, prompts, [VALID])

    def broken(query):
        raise TypeError("implementation bug")

    pipeline = CataloguePipeline(
        PipelineParts(pipeline.parts.extractor, broken), pipeline.policy, pipeline.journal
    )
    with pytest.raises(TypeError, match="implementation bug"):
        pipeline.run(QueryRequest("Laptops"))


def test_json_format_instructions_are_system_only(database, prompts):
    pipeline, _ = make_pipeline(database, prompts, [VALID])
    messages = pipeline.parts.extractor.prompt.invoke(
        QueryRequest("User request").variables()
    ).to_messages()
    assert messages[0].type == "system"
    assert '"category"' in messages[0].content
    assert messages[-1].type == "human"
    assert messages[-1].content.rstrip().endswith("User request")


def test_basic_chain_parses_and_batches_parameterized_prompts(prompts):
    from langchain_core.output_parsers import StrOutputParser

    model = FakeMessagesListChatModel(responses=[AIMessage(content="laptop")])
    chain = prompts.classification_template() | model | StrOutputParser()
    assert (
        chain.batch(
            [
                {"query": "Laptop", "language": language}
                for language in ["English", "Russian", "French"]
            ]
        )
        == ["laptop"] * 3
    )


def test_transport_error_is_not_a_parser_retry(database, prompts):
    from openai import OpenAIError

    from catalogue.errors import ModelRequestError
    from catalogue.llm.extraction import JsonExtractor

    calls = []

    def unavailable(prompt):
        calls.append(prompt)
        raise OpenAIError("sensitive provider response")

    pipeline, _ = make_pipeline(database, prompts, [VALID])
    original = pipeline.parts.extractor
    extractor = JsonExtractor(RunnableLambda(unavailable), original.prompt, original.parser)
    pipeline = CataloguePipeline(
        PipelineParts(extractor, pipeline.parts.search), pipeline.policy, pipeline.journal
    )
    with pytest.raises(ModelRequestError) as captured:
        pipeline.run(QueryRequest("Laptop"))
    assert "sensitive" not in str(captured.value)
    assert isinstance(captured.value.__cause__, OpenAIError)
    assert len(calls) == 1
    assert not any(event["event"] == "search_started" for event in pipeline.journal.events)


def test_unsupported_category_never_requires_confirmation(database, prompts):
    pipeline, repository = make_pipeline(
        database, prompts, ['{"category":"other"}'], confirmation=True
    )
    assert pipeline.run(QueryRequest("Refrigerators")).status == "unsupported_category"
    assert repository.calls == 0
