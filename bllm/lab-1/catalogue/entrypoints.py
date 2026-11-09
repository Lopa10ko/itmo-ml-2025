import json
import sys
from collections.abc import Callable
from functools import wraps

from pydantic import JsonValue, TypeAdapter

from catalogue.bootstrap import application
from catalogue.domain import Product
from catalogue.errors import CatalogueError, InputDataError
from catalogue.evidence import json_value, redact
from catalogue.llm.client import invoke_model
from catalogue.llm.models import Outcome, QueryRequest
from catalogue.persistence.sqlite import initialize_catalogue, populate_catalogue
from catalogue.population import load_population
from catalogue.review import scenarios
from catalogue.review.inputs import ReviewInputs, read_data
from catalogue.review.models import Experiment, ReviewContext
from catalogue.review.offline import failure_handling
from catalogue.review.report import build_report
from catalogue.review.report import verify as verify_results
from catalogue.review.runner import ReviewRunner


def command(action: Callable[[], int]) -> Callable[[], int]:
    @wraps(action)
    def execute() -> int:
        try:
            return action()
        except CatalogueError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2

    return execute


def print_json(value: JsonValue, secrets: tuple[str, ...]) -> None:
    print(json.dumps(redact(value, secrets), ensure_ascii=False, indent=2))


@command
def initialize() -> int:
    with application() as services:
        path = services.settings.products_path
        products = read_data(path, TypeAdapter(list[Product])) if path else []
        initialize_catalogue(services.settings.database_path, products)
        print(f"Catalogue created: {services.settings.database_path}")
    return 0


@command
def search() -> int:
    with application() as services:
        settings = services.settings
        if settings.query_text:
            result = services.pipelines.create(settings.extraction_method).run(
                QueryRequest(settings.query_text)
            )
            print_json(result.record(), services.store.secrets)
            return (
                2
                if result.status in {Outcome.FAILED, Outcome.PARSE_ERROR, Outcome.INVALID_TOOL}
                else 0
            )
        if settings.product_query is None:
            raise InputDataError(
                "QUERY_TEXT or PRODUCT_QUERY must be set in .env before searching."
            )
        products = services.search.execute(settings.product_query)
        print_json(
            {
                "query": settings.product_query.model_dump(mode="json"),
                "products": [product.model_dump(mode="json") for product in products],
                "message": "Products found." if products else "No products found.",
            },
            services.store.secrets,
        )
    return 0


@command
def populate() -> int:
    with application() as services:
        products = load_population(services.settings)
        populate_catalogue(services.settings.database_path, products)
        print(f"Catalogue ready: {len(products)} products.")
    return 0


@command
def ping() -> int:
    with application() as services:
        inputs = ReviewInputs.load(services.settings)
        message = invoke_model(services.models.get(), inputs.plan.ping_request)
        print_json(
            {
                "model": services.settings.llm_model,
                "answer": json_value(message.content),
                "thinking": False,
            },
            services.store.secrets,
        )
    return 0


def run_review(live: bool) -> int:
    runner = ReviewRunner(
        tuple(
            Experiment(name, action)
            for name, action in (
                ("basic_chain", scenarios.basic_chain),
                ("structured", scenarios.structured),
                ("search", scenarios.search),
                ("tools", scenarios.tools),
                ("thinking", scenarios.thinking),
                ("dialogues", scenarios.dialogues),
                ("without_database", scenarios.without_database),
                ("failure_handling", failure_handling),
            )
        )
    )
    with application() as services:
        inputs = ReviewInputs.load(services.settings)
        populate_catalogue(services.settings.database_path, inputs.products)
        runner.run(ReviewContext(services, inputs), live)
    return 0


@command
def experiments() -> int:
    return run_review(live=True)


@command
def offline() -> int:
    return run_review(live=False)


@command
def report() -> int:
    with application() as services:
        build_report(services, ReviewInputs.load(services.settings))
        print(f"Report: {services.store.directory / services.settings.report_filename}")
    return 0


@command
def verify() -> int:
    with application() as services:
        verify_results(services, ReviewInputs.load(services.settings))
    return 0
