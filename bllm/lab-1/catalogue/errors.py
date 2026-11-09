class CatalogueError(Exception):
    pass


class ConfigurationError(CatalogueError):
    pass


class InputDataError(CatalogueError):
    pass


class RepositoryError(CatalogueError):
    pass


class CatalogueExistsError(RepositoryError):
    pass


class ModelRequestError(CatalogueError):
    pass


class ToolCallError(CatalogueError):
    pass


class EvidenceError(CatalogueError):
    pass


class UnsupportedCategory(CatalogueError):
    pass
