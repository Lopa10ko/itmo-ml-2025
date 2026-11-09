from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import ValidationError

from catalogue.application import SearchProducts
from catalogue.domain import Product, ProductQuery
from catalogue.errors import ToolCallError
from catalogue.llm.client import ChatModel
from catalogue.llm.prompts import PromptSet

Search = Callable[[ProductQuery], list[Product]]


class QueryExtractor(Protocol):
    @property
    def model(self) -> ChatModel: ...

    @property
    def prompt(self) -> ChatPromptTemplate: ...

    def parse(self, message: AIMessage) -> ProductQuery: ...


@dataclass(frozen=True)
class JsonExtractor:
    model: ChatModel
    prompt: ChatPromptTemplate
    parser: PydanticOutputParser[ProductQuery]

    def parse(self, message: AIMessage) -> ProductQuery:
        return cast(ProductQuery, self.parser.invoke(message))


@dataclass(frozen=True)
class ToolExtractor:
    model: ChatModel
    prompt: ChatPromptTemplate
    tool_name: str

    def parse(self, message: AIMessage) -> ProductQuery:
        if message.invalid_tool_calls:
            raise ToolCallError("Malformed tool arguments; no search was executed.")
        if not message.tool_calls:
            raise ToolCallError("The model did not call the search tool.")
        if len(message.tool_calls) != 1 or message.tool_calls[0]["name"] != self.tool_name:
            raise ToolCallError(f"Expected exactly one {self.tool_name} call.")
        try:
            return ProductQuery.model_validate(message.tool_calls[0]["args"])
        except ValidationError as error:
            raise OutputParserException("Search tool arguments failed validation.") from error


@dataclass(frozen=True)
class PipelineParts:
    extractor: QueryExtractor
    search: Search


def json_parts(
    model: BaseChatModel, prompts: PromptSet, search: SearchProducts, tool_name: str
) -> PipelineParts:
    parser: PydanticOutputParser[ProductQuery] = PydanticOutputParser(pydantic_object=ProductQuery)
    prompt = prompts.extraction_template(prompts.json).partial(
        format_instructions=parser.get_format_instructions()
    )
    return PipelineParts(JsonExtractor(model, prompt, parser), search.execute)


def tool_parts(
    model: BaseChatModel, prompts: PromptSet, search: SearchProducts, tool_name: str
) -> PipelineParts:
    @tool(tool_name, args_schema=ProductQuery, description=prompts.tool_description)
    def search_products(
        category: str, max_price: int | None = None, sort_by: str = "price_asc"
    ) -> list[Product]:
        query = ProductQuery.model_validate(
            {
                "category": category,
                "max_price": max_price,
                "sort_by": sort_by,
            }
        )
        return search.execute(query)

    def execute(query: ProductQuery) -> list[Product]:
        return cast(list[Product], search_products.invoke(query.model_dump()))

    prompt = prompts.extraction_template(prompts.tools).partial(tool_name=tool_name)
    return PipelineParts(
        ToolExtractor(model.bind_tools([search_products]), prompt, tool_name), execute
    )
