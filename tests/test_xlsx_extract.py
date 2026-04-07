"""Tests for XLSX text extraction (no Vertex / DB)."""
from __future__ import annotations

from io import BytesIO

import pytest

from src.xlsx_extract import XlsxExtractError, extract_text_from_xlsx_bytes


def _make_xlsx_bytes(sheets: dict[str, list[list]]) -> bytes:
    """Build a minimal real XLSX in-memory using openpyxl."""
    import openpyxl

    wb = openpyxl.Workbook()
    first = True
    for sheet_name, rows in sheets.items():
        if first:
            ws = wb.active
            ws.title = sheet_name
            first = False
        else:
            ws = wb.create_sheet(sheet_name)
        for row in rows:
            ws.append(row)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_extract_empty_bytes_raises():
    with pytest.raises(XlsxExtractError, match="Empty"):
        extract_text_from_xlsx_bytes(b"")


def test_extract_invalid_bytes_raises():
    with pytest.raises(XlsxExtractError, match="Invalid|corrupted"):
        extract_text_from_xlsx_bytes(b"not an xlsx file")


def test_extract_single_sheet():
    data = _make_xlsx_bytes({"Data": [["Name", "Score"], ["Alice", 95], ["Bob", 87]]})
    result = extract_text_from_xlsx_bytes(data)
    assert "Sheet: Data" in result
    assert "Name" in result
    assert "Alice" in result
    assert "95" in result


def test_extract_multiple_sheets():
    data = _make_xlsx_bytes({
        "Sheet1": [["A", "B"], [1, 2]],
        "Sheet2": [["X", "Y"], [3, 4]],
    })
    result = extract_text_from_xlsx_bytes(data)
    assert "Sheet: Sheet1" in result
    assert "Sheet: Sheet2" in result
    assert "A" in result and "X" in result


def test_extract_empty_sheet_skipped():
    data = _make_xlsx_bytes({"EmptySheet": [], "RealSheet": [["Hello", "World"]]})
    result = extract_text_from_xlsx_bytes(data)
    assert "EmptySheet" not in result
    assert "RealSheet" in result


def test_extract_empty_rows_skipped():
    data = _make_xlsx_bytes({"Sheet1": [["A", "B"], [], ["C", "D"]]})
    result = extract_text_from_xlsx_bytes(data)
    assert "A" in result
    assert "C" in result
