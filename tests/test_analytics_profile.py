"""Analytics profiling (pandas) — no full app / Vertex import."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.core.analytics_profile import is_allowed_tabular_filename, profile_tabular_bytes


def test_is_allowed_tabular_filename():
    assert is_allowed_tabular_filename("a.csv") is True
    assert is_allowed_tabular_filename("x.XLSX") is True
    assert is_allowed_tabular_filename("bad.pdf") is False


def test_profile_csv_basic():
    raw = b"name,amount\nfoo,10\nbar,20\n"
    out = profile_tabular_bytes(raw, "t.csv", max_rows=1000, max_cols=64)
    assert out["row_count"] == 2
    assert out["column_count"] == 2
    names = {c["name"] for c in out["columns"]}
    assert names == {"name", "amount"}
    amt = next(c for c in out["columns"] if c["name"] == "amount")
    assert amt["numeric_stats"] is not None
    assert amt["numeric_stats"]["mean"] == 15.0


def test_profile_rejects_too_many_rows():
    buf = io.StringIO()
    pd.DataFrame({"x": range(5)}).to_csv(buf, index=False)
    raw = buf.getvalue().encode()
    with pytest.raises(ValueError, match="Too many rows"):
        profile_tabular_bytes(raw, "huge.csv", max_rows=3, max_cols=64)


def test_profile_xlsx_roundtrip():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    raw = bio.getvalue()
    out = profile_tabular_bytes(raw, "t.xlsx", max_rows=1000, max_cols=64)
    assert out["row_count"] == 2
    assert out["column_count"] == 2
