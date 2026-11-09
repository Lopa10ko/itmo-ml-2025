from dataclasses import dataclass

from pydantic import ValidationError

from catalogue.domain import Product, ProductQuery, ProductRepository
from catalogue.errors import InputDataError, UnsupportedCategory


@dataclass(frozen=True)
class SearchProducts:
    repository: ProductRepository

    def execute(self, query: ProductQuery) -> list[Product]:
        try:
            query = ProductQuery.model_validate(query)
        except ValidationError as error:
            raise InputDataError("The search parameters are invalid.") from error
        if query.category == "other":
            raise UnsupportedCategory("The requested product category is not supported.")
        return self.repository.search(query)
