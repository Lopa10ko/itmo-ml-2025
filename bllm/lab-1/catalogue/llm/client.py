from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from pydantic import JsonValue, ValidationError

from catalogue.errors import ConfigurationError, ModelRequestError
from catalogue.settings import Settings

ChatModel = Runnable[LanguageModelInput, AIMessage]


def invoke_model(model: ChatModel, prompt: PromptValue | str) -> AIMessage:
    try:
        message = model.invoke(prompt)
    except OpenAIError as error:
        status = getattr(error, "status_code", None)
        detail = f", HTTP {status}" if status is not None else ""
        raise ModelRequestError(
            f"Model request failed ({type(error).__name__}{detail})."
        ) from error
    if not isinstance(message, AIMessage):
        raise ModelRequestError("The chat model returned an unexpected message type.")
    return message


class ModelProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._models: dict[bool, ChatOpenAI] = {}

    def get(self, thinking: bool = False) -> ChatOpenAI:
        if not self.settings.itmo_llm_token:
            raise ConfigurationError("Set ITMO_LLM_TOKEN in .env before a live model request.")
        if thinking not in self._models:
            try:
                self._models[thinking] = ChatOpenAI(
                    api_key=self.settings.itmo_llm_token,
                    base_url=str(self.settings.llm_base_url),
                    model=self.settings.llm_model,
                    timeout=self.settings.llm_timeout_seconds,
                    max_completion_tokens=self.settings.llm_max_tokens,
                    temperature=self.settings.llm_temperature,
                    max_retries=0,
                    cache=False,
                    use_responses_api=False,
                    extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
                )
            except ValidationError as error:
                raise ConfigurationError("The model client configuration is invalid.") from error
        return self._models[thinking]

    def raw_reasoning(self, prompt: PromptValue) -> dict[str, JsonValue]:
        model = self.get(thinking=True)
        try:
            response = model.client.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system" if message.type == "system" else "user",
                        "content": message.content,
                    }
                    for message in prompt.to_messages()
                ],
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
        except OpenAIError as error:
            raise ModelRequestError("The raw reasoning diagnostic request failed.") from error
        return {
            "source": "raw_provider_diagnostic",
            "message": response.choices[0].message.model_dump(mode="json"),
            "usage": response.usage.model_dump(mode="json") if response.usage else None,
        }

    def close(self) -> None:
        for model in self._models.values():
            model.root_client.close()
