from enum import Enum
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.config import Settings


class InferenceProvider(str, Enum):
    """Supported inference providers for LLMs and embeddings."""
    VERTEX = "vertex"
    OPENAI_COMPAT = "openai_compat"
    ANTHROPIC = "anthropic"


def get_llm(settings: Settings, **kwargs: Any) -> BaseChatModel:
    """
    Returns a BaseChatModel instance based on the configured inference provider.

    Args:
        settings: The central application Settings.
        **kwargs: Additional keyword arguments passed directly to the model constructor
                 (e.g., temperature, max_tokens).

    Returns:
        A configured BaseChatModel instance.

    Raises:
        ValueError: If an unknown inference provider is specified in settings.
    """
    provider = settings.inference_provider
    model_name = settings.inference_llm_model

    if provider == InferenceProvider.VERTEX:
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(model_name=model_name, **kwargs)

    elif provider == InferenceProvider.OPENAI_COMPAT:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            base_url=settings.inference_base_url,
            api_key=settings.inference_api_key,
            **kwargs
        )

    elif provider == InferenceProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            api_key=settings.inference_api_key,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown inference provider: {provider}")


def get_embeddings(settings: Settings, **kwargs: Any) -> Embeddings:
    """
    Returns an Embeddings instance based on the configured inference provider.

    Args:
        settings: The central application Settings.
        **kwargs: Additional keyword arguments passed directly to the embeddings constructor.

    Returns:
        A configured Embeddings instance.

    Raises:
        NotImplementedError: If the chosen provider does not support native embeddings (e.g., Anthropic).
        ValueError: If an unknown inference provider is specified in settings.
    """
    provider = settings.inference_provider
    model_name = settings.inference_embedding_model

    if provider == InferenceProvider.VERTEX:
        from langchain_google_vertexai import VertexAIEmbeddings
        return VertexAIEmbeddings(model_name=model_name, **kwargs)

    elif provider == InferenceProvider.OPENAI_COMPAT:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name,
            base_url=settings.inference_base_url,
            api_key=settings.inference_api_key,
            **kwargs
        )

    elif provider == InferenceProvider.ANTHROPIC:
        raise NotImplementedError("Anthropic does not provide native text embeddings. Use openai_compat or vertex for embeddings.")

    else:
        raise ValueError(f"Unknown inference provider: {provider}")
