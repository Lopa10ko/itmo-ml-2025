from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

ProductCategory = Literal["laptop", "smartphone", "tablet"]
QueryCategory = Literal[ProductCategory, "other"]
SortOrder = Literal["price_asc", "price_desc"]


class ProductQuery(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, revalidate_instances="always"
    )

    category: QueryCategory = Field(
        description="Product category: laptop, smartphone, tablet, or other if unsupported."
    )
    max_price: int | None = Field(
        default=None,
        ge=0,
        description="Inclusive maximum price in whole rubles; null means no price limit.",
    )
    sort_by: SortOrder = Field(
        default="price_asc",
        description="Price order: price_asc (default) or price_desc.",
    )


class Product(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    id: int = Field(gt=0)
    name: str = Field(min_length=1, pattern=r"\S")
    category: ProductCategory
    price: int = Field(ge=0)


class ProductRepository(Protocol):
    def search(self, query: ProductQuery) -> list[Product]: ...
