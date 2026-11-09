from dataclasses import dataclass
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from catalogue.errors import ConfigurationError
from catalogue.settings import Settings


def read_prompt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise ConfigurationError(f"Could not read prompt template {path}.") from error
    if not text:
        raise ConfigurationError(f"Prompt template is empty: {path}.")
    return text


@dataclass(frozen=True)
class PromptSet:
    classification: str
    extraction: str
    json: str
    tools: str
    tool_description: str
    request: str

    @classmethod
    def load(cls, settings: Settings) -> "PromptSet":
        return cls(
            classification=read_prompt(settings.classification_prompt_path),
            extraction=read_prompt(settings.extraction_prompt_path),
            json=read_prompt(settings.json_prompt_path),
            tools=read_prompt(settings.tool_prompt_path),
            tool_description=read_prompt(settings.tool_description_path),
            request=read_prompt(settings.request_prompt_path),
        )

    def classification_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.classification),
                ("human", "{query}"),
            ]
        )

    def extraction_template(self, instructions: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.extraction + "\n" + instructions),
                MessagesPlaceholder("history", optional=True),
                ("human", self.request),
            ]
        )
