"""
Controlled tabular profiling (pandas). No user code execution — fixed summaries only.

Used for CSV/TSV/XLSX uploads to support reporting-style analytics alongside RAG.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

def is_allowed_tabular_filename(filename: str | None) -> bool:
    fn = (filename or "").lower()
    return any(fn.endswith(s) for s in (".csv", ".tsv", ".txt", ".xlsx"))


def _to_json_scalar(v: Any) -> Any:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return float(v) if isinstance(v, np.floating) else int(v)
    if isinstance(v, float):
        return float(v)
    if isinstance(v, (str, int, bool)):
        return v
    return str(v)


def _read_dataframe(content: bytes, filename: str, max_rows: int) -> pd.DataFrame:
    ext = Path(filename or "data.csv").suffix.lower()
    buf = io.BytesIO(content)

    if ext == ".xlsx":
        try:
            import openpyxl  # noqa: F401  # imported for read_excel engine
        except ImportError as e:
            raise ValueError("XLSX support requires the 'openpyxl' package.") from e
        df = pd.read_excel(buf, engine="openpyxl", nrows=max_rows + 1, header=0)
    elif ext == ".tsv":
        df = pd.read_csv(buf, sep="\t", nrows=max_rows + 1, encoding_errors="replace")
    else:
        # .csv, .txt — comma-separated
        df = pd.read_csv(buf, nrows=max_rows + 1, encoding_errors="replace")

    return df


def profile_tabular_bytes(
    content: bytes,
    filename: str,
    *,
    max_rows: int = 100_000,
    max_cols: int = 256,
) -> dict[str, Any]:
    """
    Build a JSON-serializable profile: shape, per-column nulls, numeric stats, top categories.

    Raises ValueError for unsupported format or limits exceeded.
    """
    if not content.strip():
        raise ValueError("Empty file.")

    if not is_allowed_tabular_filename(filename):
        raise ValueError(
            "Unsupported file type. Use .csv, .tsv, .txt (tab-separated), or .xlsx."
        )

    df = _read_dataframe(content, filename, max_rows)

    if len(df) > max_rows:
        raise ValueError(f"Too many rows (>{max_rows}). Split the file or raise the server limit.")

    if df.shape[1] > max_cols:
        raise ValueError(f"Too many columns (>{max_cols}).")

    # Compact column names for JSON
    df.columns = [str(c) for c in df.columns]

    columns_out: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        n = len(s)
        null_pct = round(100.0 * null_count / n, 4) if n else 0.0

        entry: dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": null_pct,
        }

        non_null = s.dropna()
        if non_null.empty:
            entry["n_unique"] = 0
            entry["top_values"] = []
            entry["numeric_stats"] = None
            columns_out.append(entry)
            continue

        nu = non_null.nunique(dropna=True)
        entry["n_unique"] = int(nu)

        if pd.api.types.is_numeric_dtype(s):
            ns = non_null.astype(float)
            entry["numeric_stats"] = {
                "min": _to_json_scalar(ns.min()),
                "max": _to_json_scalar(ns.max()),
                "mean": _to_json_scalar(ns.mean()),
                "std": _to_json_scalar(ns.std()) if len(ns) > 1 else 0.0,
            }
            vc = non_null.value_counts().head(5)
            entry["top_values"] = [
                {"value": _to_json_scalar(k), "count": int(v)} for k, v in vc.items()
            ]
        else:
            entry["numeric_stats"] = None
            vc = non_null.astype(str).value_counts().head(5)
            entry["top_values"] = [
                {"value": _to_json_scalar(k), "count": int(v)} for k, v in vc.items()
            ]

        columns_out.append(entry)

    return {
        "filename": Path(filename or "upload").name,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": columns_out,
    }
