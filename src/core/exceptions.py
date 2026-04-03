"""Domain-oriented errors (map infra errors to API-friendly messages in handlers)."""


class DomainError(Exception):
    """Base class for application-specific errors."""


class KnowledgeBaseError(DomainError):
    """Vector store empty or retrieval failed in a user-visible way."""


class IngestionError(DomainError):
    """Document processing / chunking failed."""


class QueryGuardrailError(DomainError):
    """Query blocked by optional content policy (e.g. suspected PII)."""
