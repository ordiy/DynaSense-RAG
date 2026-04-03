"""
Backward-compatible entrypoint: ``uvicorn src.app:app``.

Application assembly lives in ``src.api.main`` (routers, static files, LangSmith init).
"""
from __future__ import annotations

import uvicorn

from src.api.main import app

__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
