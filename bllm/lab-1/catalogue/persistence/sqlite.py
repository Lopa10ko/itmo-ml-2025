import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from catalogue.domain import Product, ProductQuery, SortOrder
from catalogue.errors import CatalogueExistsError, RepositoryError

SORT_DIRECTIONS: dict[SortOrder, str] = {
    "price_asc": "ASC",
    "price_desc": "DESC",
}


def initialize_catalogue(path: Path, products: Iterable[Product] = ()) -> None:
    destination = path.resolve()
    if destination.exists():
        raise CatalogueExistsError(f"Catalogue already exists: {path}. No data was changed.")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        schema = files("catalogue.persistence").joinpath("schema.sql").read_text(encoding="utf-8")
        with TemporaryDirectory(dir=destination.parent) as temporary:
            database = Path(temporary) / destination.name
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("BEGIN")
                connection.execute(schema)
                connection.executemany(
                    "INSERT INTO products (id, name, category, price) VALUES (?, ?, ?, ?)",
                    (
                        (product.id, product.name, product.category, product.price)
                        for product in products
                    ),
                )
            os.link(database, destination)
    except FileExistsError as error:
        raise CatalogueExistsError(
            f"Catalogue already exists: {path}. No data was changed."
        ) from error
    except (sqlite3.Error, OSError) as error:
        raise RepositoryError(f"Could not initialize the catalogue at {path}.") from error


class SQLiteProductRepository:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def search(self, query: ProductQuery) -> list[Product]:
        statement = "SELECT id, name, category, price FROM products WHERE category = ?"
        parameters: list[str | int] = [query.category]
        if query.max_price is not None:
            statement += " AND price <= ?"
            parameters.append(query.max_price)
        statement += f" ORDER BY price {SORT_DIRECTIONS[query.sort_by]}, id ASC"
        try:
            with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                return [
                    Product.model_validate(dict(row))
                    for row in connection.execute(statement, parameters)
                ]
        except (sqlite3.Error, OSError, ValidationError) as error:
            raise RepositoryError(f"Could not read valid products from {self.path}.") from error

    def integrity_check(self) -> str:
        try:
            with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)) as connection:
                row = connection.execute("PRAGMA integrity_check").fetchone()
                return str(row[0])
        except sqlite3.Error as error:
            raise RepositoryError(f"Could not check catalogue integrity at {self.path}.") from error


def populate_catalogue(path: Path, products: list[Product]) -> None:
    expected = {product.id: product for product in products}
    if len(expected) != len(products):
        raise RepositoryError("Population requires unique product identifiers.")
    if not path.exists():
        try:
            initialize_catalogue(path, products)
            return
        except CatalogueExistsError:
            pass
    try:
        with (
            closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True)) as connection,
            connection,
        ):
            connection.execute("BEGIN IMMEDIATE")
            connection.row_factory = sqlite3.Row
            actual = [
                Product.model_validate(dict(row))
                for row in connection.execute("SELECT id, name, category, price FROM products")
            ]
            if any(expected.get(product.id) != product for product in actual):
                raise RepositoryError(
                    "Existing products differ from the generated catalogue; configure a new DATABASE_PATH. No data was changed."
                )
            existing = {product.id for product in actual}
            connection.executemany(
                "INSERT INTO products (id, name, category, price) VALUES (?, ?, ?, ?)",
                (
                    (product.id, product.name, product.category, product.price)
                    for product in products
                    if product.id not in existing
                ),
            )
    except (sqlite3.Error, OSError, ValidationError) as error:
        raise RepositoryError(f"Could not populate the catalogue at {path}.") from error
