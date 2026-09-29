class AppError(Exception):
    """Base application error."""


class RetrievalError(AppError):
    pass


class EmbeddingError(AppError):
    pass


class VectorStoreError(AppError):
    pass


class MemoryError(AppError):
    pass


class DocumentError(AppError):
    pass


class ValidationError(AppError):
    pass


class ConfigurationError(AppError):
    pass


class RuntimeError(AppError):
    pass


class ConversationStoreError(AppError):
    pass
