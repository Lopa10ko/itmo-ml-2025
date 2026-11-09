from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version

from pydantic import JsonValue

from catalogue.domain import ProductQuery
from catalogue.errors import ConfigurationError
from catalogue.evidence import json_value
from catalogue.llm.client import invoke_model
from catalogue.review.models import Experiment, ReviewContext, ReviewResults


@dataclass(frozen=True)
class ReviewRunner:
    experiments: tuple[Experiment, ...]

    def run(self, context: ReviewContext, live: bool) -> ReviewResults:
        services, inputs = context.services, context.inputs
        indexed = {experiment.name: experiment for experiment in self.experiments}
        selected = inputs.plan.studies if live else inputs.plan.offline_studies
        if (
            len(indexed) != len(self.experiments)
            or len(set(selected)) != len(selected)
            or not set(selected).issubset(indexed)
        ):
            raise ConfigurationError(
                "Review experiment names must be registered, known and unique."
            )
        services.store.remove(services.settings.report_filename)
        services.store.write_text(services.settings.events_filename, "")
        services.journal.events.clear()
        metadata: dict[str, JsonValue] = {
            "catalogue": [product.model_dump(mode="json") for product in inputs.products],
            "schema": ProductQuery.model_json_schema(),
            "database_integrity": services.repository.integrity_check(),
            "versions": {
                name: version(name)
                for name in ("langchain", "langchain-core", "langchain-openai", "pydantic")
            },
            "model": services.settings.llm_model,
            "base_url": str(services.settings.llm_base_url),
            "temperature": services.settings.llm_temperature,
            "max_tokens": services.settings.llm_max_tokens,
            "retry_attempts": services.settings.retry_attempts,
            "population": {
                "products_per_category": services.settings.products_per_category,
                "seed": services.settings.population_seed,
            },
            "selected_studies": [name for name in selected],
        }
        results = ReviewResults(
            started_at=datetime.now(UTC), mode="live" if live else "offline", metadata=metadata
        )
        self.save(context, results)
        if live:
            started = datetime.now(UTC)
            message = invoke_model(services.models.get(), inputs.plan.ping_request)
            metadata["connection"] = {
                "answer": json_value(message.content),
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
                "thinking": False,
            }
            self.save(context, results)
        for name in selected:
            print(f"Running {name}...", flush=True)
            results.sections.append(indexed[name].run(context))
            self.save(context, results)
        results.completed_at = datetime.now(UTC)
        self.save(context, results)
        return results

    def save(self, context: ReviewContext, results: ReviewResults) -> None:
        context.services.store.write_json(
            context.services.settings.results_filename, results.model_dump(mode="json")
        )
