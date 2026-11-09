import pytest

from catalogue.application import SearchProducts
from catalogue.domain import Product, ProductQuery
from catalogue.errors import InputDataError, UnsupportedCategory


class RecordingRepository:
    def __init__(self, products):
        self.products = products
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.products


def test_search_uses_the_repository_port():
    products = [Product(id=1, name="Laptop", category="laptop", price=50000)]
    repository = RecordingRepository(products)
    query = ProductQuery(category="laptop", max_price=50000)
    assert SearchProducts(repository).execute(query) == products
    assert repository.queries == [query]


def test_empty_search_returns_a_list():
    assert (
        SearchProducts(RecordingRepository([])).execute(
            ProductQuery(category="tablet", max_price=0)
        )
        == []
    )


def test_unsupported_category_never_touches_the_repository():
    repository = RecordingRepository([])
    with pytest.raises(UnsupportedCategory, match="not supported"):
        SearchProducts(repository).execute(ProductQuery(category="other"))
    assert repository.queries == []


def test_bypassed_validation_cannot_reach_the_repository():
    repository = RecordingRepository([])
    invalid = ProductQuery.model_construct(category="laptop", max_price=-1)
    with pytest.raises(InputDataError):
        SearchProducts(repository).execute(invalid)
    assert repository.queries == []


def test_storage_failure_is_not_retried():
    class FailingRepository:
        calls = 0

        def search(self, query):
            self.calls += 1
            raise OSError("storage unavailable")

    repository = FailingRepository()
    with pytest.raises(OSError, match="storage unavailable"):
        SearchProducts(repository).execute(ProductQuery(category="laptop"))
    assert repository.calls == 1
