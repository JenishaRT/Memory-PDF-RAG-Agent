class ApplicationError(Exception):
    """
    Base exception for application level errors.
    """


class ConfigurationError(ApplicationError):
    """
    Raised when application configuration is invalid.
    """


class ContractError(ApplicationError):
    """
    Raised when an internal contract is violated.
    """


class ConversationStoreError(ApplicationError):
    """
    Raised when conversation persistence fails.
    """


class RetrievalError(ApplicationError):
    """
    Raised when retrieval fails.
    """


class MemoryError(ApplicationError):
    """
    Raised when memory processing fails.
    """


class DocumentError(ApplicationError):
    """
    Raised when document processing fails.
    """


class LLMError(ApplicationError):
    """
    Raised when an LLM provider fails.
    """


class EmbeddingError(ApplicationError):
    """
    Raised when an embedding provider fails.
    """


class VectorStoreError(ApplicationError):
    """
    Raised when a vector store operation fails.
    """