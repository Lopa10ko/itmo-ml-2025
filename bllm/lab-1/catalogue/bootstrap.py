from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from catalogue.application import SearchProducts
from catalogue.errors import ConfigurationError
from catalogue.evidence import EvidenceStore, Journal
from catalogue.llm.client import ModelProvider
from catalogue.llm.extraction import PipelineParts, json_parts, tool_parts
from catalogue.llm.models import ExecutionPolicy
from catalogue.llm.pipeline import CataloguePipeline
from catalogue.llm.prompts import PromptSet
from catalogue.persistence.sqlite import SQLiteProductRepository
from catalogue.settings import Settings, load_settings

PartsBuilder = Callable[[BaseChatModel, PromptSet, SearchProducts, str], PipelineParts]


@dataclass
class PipelineFactory:
    settings: Settings
    prompts: PromptSet
    search: SearchProducts
    journal: Journal
    models: ModelProvider
    builders: Mapping[str, PartsBuilder]

    def create(
        self,
        method: str,
        thinking: bool = False,
        policy: ExecutionPolicy | None = None,
        model: BaseChatModel | None = None,
    ) -> CataloguePipeline:
        try:
            builder = self.builders[method]
        except KeyError as error:
            raise ConfigurationError(f"Unknown extraction method: {method}.") from error
        parts = builder(
            model if model is not None else self.models.get(thinking),
            self.prompts,
            self.search,
            self.settings.search_tool_name,
        )
        return CataloguePipeline(
            parts,
            policy if policy is not None else self.default_policy(),
            self.journal,
            self.settings.reasoning_fields,
        )

    def default_policy(self) -> ExecutionPolicy:
        return ExecutionPolicy(
            max_attempts=self.settings.retry_attempts,
            require_confirmation=self.settings.require_price_confirmation,
            confirmed=self.settings.unrestricted_search_confirmed,
        )


@dataclass(frozen=True)
class Services:
    settings: Settings
    store: EvidenceStore
    journal: Journal
    prompts: PromptSet
    repository: SQLiteProductRepository
    search: SearchProducts
    models: ModelProvider
    pipelines: PipelineFactory


@contextmanager
def application() -> Iterator[Services]:
    settings = load_settings()
    secrets = (settings.itmo_llm_token.get_secret_value(),) if settings.itmo_llm_token else ()
    store = EvidenceStore(settings.artifacts_path, secrets)
    journal = Journal(
        secrets=secrets,
        sink=lambda event: store.append_event(settings.events_filename, event),
    )
    prompts = PromptSet.load(settings)
    repository = SQLiteProductRepository(settings.database_path)
    search = SearchProducts(repository)
    models = ModelProvider(settings)
    pipelines = PipelineFactory(
        settings,
        prompts,
        search,
        journal,
        models,
        builders={"json": json_parts, "tools": tool_parts},
    )
    try:
        yield Services(settings, store, journal, prompts, repository, search, models, pipelines)
    finally:
        models.close()
