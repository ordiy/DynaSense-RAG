import pytest
from unittest.mock import MagicMock, patch
from src.pdf_extract import extract_image_captions_from_pdf, extract_pdf_content
import base64

@pytest.fixture
def mock_settings():
    with patch("src.core.config.get_settings") as mock_core_settings:
        settings = MagicMock()
        settings.image_caption_enabled = True
        mock_core_settings.return_value = settings
        yield settings

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value.content = "A test image description."
    return llm

@patch("pypdf.PdfReader")
def test_captions_disabled(mock_reader, mock_settings):
    mock_settings.image_caption_enabled = False
    captions = extract_image_captions_from_pdf(b"fake_pdf_data")
    assert captions == []
    mock_reader.assert_not_called()

@patch("pypdf.PdfReader")
def test_captions_skips_small_images(mock_reader, mock_settings, mock_llm):
    mock_page = MagicMock()
    mock_image = MagicMock()
    mock_image.data = b"0" * 1024  # 1KB image, < 10KB
    mock_page.images = [mock_image]
    
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_reader.return_value = mock_pdf

    captions = extract_image_captions_from_pdf(b"fake_pdf_data", llm=mock_llm)
    assert captions == []
    mock_llm.invoke.assert_not_called()

@patch("pypdf.PdfReader")
def test_captions_calls_vision_llm(mock_reader, mock_settings, mock_llm):
    mock_page = MagicMock()
    mock_image = MagicMock()
    mock_image.data = b"0" * (10 * 1024 + 1)  # > 10KB
    mock_page.images = [mock_image]
    
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_reader.return_value = mock_pdf

    captions = extract_image_captions_from_pdf(b"fake_pdf_data", llm=mock_llm)
    
    assert len(captions) == 1
    assert captions[0] == "[图片描述] A test image description."
    mock_llm.invoke.assert_called_once()
    
    # Check that human message is formatted correctly
    args, kwargs = mock_llm.invoke.call_args
    messages = args[0]
    assert len(messages) == 1
    content = messages[0].content
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]['type'] == 'text'
    assert content[1]['type'] == 'image_url'
    
    expected_b64 = base64.b64encode(mock_image.data).decode('utf-8')
    assert content[1]['image_url']['url'] == f"data:image/png;base64,{expected_b64}"

@patch("pypdf.PdfReader")
def test_captions_caps_at_10(mock_reader, mock_settings, mock_llm):
    mock_page = MagicMock()
    # Create 15 images
    images = []
    for _ in range(15):
        img = MagicMock()
        img.data = b"0" * (10 * 1024 + 1)
        images.append(img)
    mock_page.images = images
    
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_reader.return_value = mock_pdf

    captions = extract_image_captions_from_pdf(b"fake_pdf_data", llm=mock_llm)
    
    assert len(captions) == 10
    assert mock_llm.invoke.call_count == 10

@patch("pypdf.PdfReader")
def test_captions_handles_llm_failure(mock_reader, mock_settings, mock_llm):
    mock_page = MagicMock()
    mock_image = MagicMock()
    mock_image.data = b"0" * (10 * 1024 + 1)
    mock_page.images = [mock_image]
    
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_reader.return_value = mock_pdf

    # Make LLM raise an exception
    mock_llm.invoke.side_effect = RuntimeError("API down")

    captions = extract_image_captions_from_pdf(b"fake_pdf_data", llm=mock_llm)
    
    assert captions == []

@patch("src.pdf_extract.extract_text_from_pdf_bytes")
@patch("src.pdf_extract.extract_tables_as_markdown")
@patch("src.pdf_extract.extract_image_captions_from_pdf")
def test_extract_pdf_content_includes_captions(mock_captions, mock_tables, mock_text):
    mock_text.return_value = "Main text."
    mock_tables.return_value = ["| Table |\n| --- |"]
    mock_captions.return_value = ["[图片描述] Image 1.", "[图片描述] Image 2."]

    result = extract_pdf_content(b"fake_data")
    
    assert "Main text." in result
    assert "| Table |" in result
    assert "[图片描述] Image 1.\n[图片描述] Image 2." in result
    assert result == "Main text.\n\n| Table |\n| --- |\n\n[图片描述] Image 1.\n[图片描述] Image 2."

