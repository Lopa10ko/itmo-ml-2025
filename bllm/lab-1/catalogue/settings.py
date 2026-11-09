from pathlib import Path

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
)

from catalogue.domain import ProductQuery
from catalogue.errors import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    database_path: Path
    products_path: Path | None = None
    products_per_category: int = Field(ge=4)
    population_seed: int
    product_query: ProductQuery | None = None
    query_text: str | None = None
    itmo_llm_token: SecretStr | None = None
    llm_base_url: AnyHttpUrl
    llm_model: str = Field(min_length=1)
    llm_timeout_seconds: float = Field(gt=0)
    llm_max_tokens: int = Field(gt=0)
    llm_temperature: float = Field(ge=0, le=2)
    retry_attempts: int = Field(ge=1, le=3)
    batch_concurrency: int = Field(ge=1)
    extraction_method: str = Field(min_length=1)
    reasoning_fields: tuple[str, ...]
    search_tool_name: str = Field(min_length=1)
    classification_prompt_path: Path
    extraction_prompt_path: Path
    json_prompt_path: Path
    tool_prompt_path: Path
    tool_description_path: Path
    request_prompt_path: Path
    artifacts_path: Path
    results_filename: str = Field(pattern=r"^[^/\\]+\.json$")
    events_filename: str = Field(pattern=r"^[^/\\]+\.jsonl$")
    report_filename: str = Field(pattern=r"^[^/\\]+\.md$")
    queries_path: Path
    dialogues_path: Path
    review_plan_path: Path
    error_cases_path: Path
    report_template_path: Path
    require_price_confirmation: bool
    unrestricted_search_confirmed: bool

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (dotenv_settings,)


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as error:
        fields = "; ".join(
            f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
            for item in error.errors(include_input=False, include_context=False)
        )
        raise ConfigurationError(f"Invalid .env configuration: {fields}") from error
    except (SettingsError, OSError) as error:
        raise ConfigurationError("Could not read .env configuration.") from error
