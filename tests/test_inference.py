from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Settings
from src.core.inference import get_embeddings, get_llm


@pytest.fixture
def mock_langchain_modules():
    """Mock the provider SDK imports to test the factory functions without needing credentials."""
    mock_vertex = MagicMock()
    mock_openai = MagicMock()
    mock_anthropic = MagicMock()

    with patch.dict('sys.modules', {
        'langchain_google_vertexai': mock_vertex,
        'langchain_openai': mock_openai,
        'langchain_anthropic': mock_anthropic
    }):
        yield {
            'vertex': mock_vertex,
            'openai': mock_openai,
            'anthropic': mock_anthropic
        }


def test_get_llm_vertex(mock_langchain_modules):
    settings = Settings(
        inference_provider='vertex',
        inference_llm_model='gemini-2.5-flash'
    )
    # The temperature kwarg tests passing extra params
    get_llm(settings, temperature=0.5)
    mock_langchain_modules['vertex'].ChatVertexAI.assert_called_once_with(
        model_name='gemini-2.5-flash', temperature=0.5
    )


def test_get_llm_openai_compat(mock_langchain_modules):
    settings = Settings(
        inference_provider='openai_compat',
        inference_llm_model='gpt-4o',
        inference_base_url='http://localhost:8000',
        inference_api_key='sk-123'
    )
    get_llm(settings, temperature=0)
    mock_langchain_modules['openai'].ChatOpenAI.assert_called_once_with(
        model='gpt-4o',
        base_url='http://localhost:8000',
        api_key='sk-123',
        temperature=0
    )


def test_get_llm_anthropic(mock_langchain_modules):
    settings = Settings(
        inference_provider='anthropic',
        inference_llm_model='claude-3-opus-20240229',
        inference_api_key='sk-ant-123'
    )
    get_llm(settings, max_tokens=1000)
    mock_langchain_modules['anthropic'].ChatAnthropic.assert_called_once_with(
        model='claude-3-opus-20240229',
        api_key='sk-ant-123',
        max_tokens=1000
    )


def test_get_llm_invalid(mock_langchain_modules):
    settings = Settings()
    # Bypass Pydantic validation to test the error path
    settings.inference_provider = 'invalid'  # type: ignore
    with pytest.raises(ValueError, match="Unknown inference provider: invalid"):
        get_llm(settings)


def test_get_embeddings_vertex(mock_langchain_modules):
    settings = Settings(
        inference_provider='vertex',
        inference_embedding_model='text-embedding-004'
    )
    get_embeddings(settings)
    mock_langchain_modules['vertex'].VertexAIEmbeddings.assert_called_once_with(
        model_name='text-embedding-004'
    )


def test_get_embeddings_openai_compat(mock_langchain_modules):
    settings = Settings(
        inference_provider='openai_compat',
        inference_embedding_model='text-embedding-ada-002',
        inference_base_url='http://localhost:8000',
        inference_api_key='sk-123'
    )
    get_embeddings(settings)
    mock_langchain_modules['openai'].OpenAIEmbeddings.assert_called_once_with(
        model='text-embedding-ada-002',
        base_url='http://localhost:8000',
        api_key='sk-123'
    )


def test_get_embeddings_anthropic(mock_langchain_modules):
    settings = Settings(
        inference_provider='anthropic',
        inference_embedding_model='any'
    )
    with pytest.raises(NotImplementedError, match="Anthropic does not provide native text embeddings"):
        get_embeddings(settings)


def test_get_embeddings_invalid(mock_langchain_modules):
    settings = Settings()
    # Bypass Pydantic validation to test the error path
    settings.inference_provider = 'invalid'  # type: ignore
    with pytest.raises(ValueError, match="Unknown inference provider: invalid"):
        get_embeddings(settings)
