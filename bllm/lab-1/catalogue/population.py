from collections import defaultdict
from collections.abc import Sequence
from random import Random
from typing import get_args

from pydantic import TypeAdapter, ValidationError

from catalogue.domain import Product, ProductCategory
from catalogue.errors import InputDataError
from catalogue.settings import Settings


def generate_products(
    examples: Sequence[Product], products_per_category: int, seed: int
) -> list[Product]:
    if len({product.id for product in examples}) != len(examples):
        raise InputDataError("Example product identifiers must be unique.")
    grouped: dict[str, list[Product]] = defaultdict(list)
    for product in sorted(examples, key=lambda product: product.id):
        grouped[product.category].append(product)
    categories = get_args(ProductCategory)
    if any(not grouped[category] for category in categories):
        raise InputDataError("Provide at least one example product for every supported category.")
    if any(len(grouped[category]) > products_per_category for category in categories):
        raise InputDataError("PRODUCTS_PER_CATEGORY must accommodate all example products.")
    products = list(examples)
    first_identifier = max(product.id for product in examples) + 1
    for position, category in enumerate(categories):
        anchors = grouped[category]
        minimum = min(product.price for product in anchors)
        maximum = max(product.price for product in anchors)
        boundaries = sorted(
            {
                product.price + offset
                for product in anchors
                for offset in (-1, 0, 1)
                if minimum <= product.price + offset <= maximum
            }
        )
        random = Random(f"{seed}:{category}")
        for index in range(products_per_category - len(anchors)):
            price = (
                boundaries[index] if index < len(boundaries) else random.randint(minimum, maximum)
            )
            products.append(
                Product.model_validate(
                    {
                        "id": first_identifier + index * len(categories) + position,
                        "name": f"{anchors[index % len(anchors)].name} variant {index + 1:02}",
                        "category": category,
                        "price": price,
                    }
                )
            )
    return sorted(products, key=lambda product: product.id)


def load_population(settings: Settings) -> list[Product]:
    if settings.products_path is None:
        raise InputDataError("Set PRODUCTS_PATH in .env before populating the catalogue.")
    try:
        adapter = TypeAdapter(list[Product])
        examples = adapter.validate_json(settings.products_path.read_bytes())
    except (OSError, ValidationError) as error:
        raise InputDataError(
            f"Could not load valid product examples from {settings.products_path}."
        ) from error
    return generate_products(examples, settings.products_per_category, settings.population_seed)
