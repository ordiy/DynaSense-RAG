"""HTML pages (engineer console + customer portal)."""

from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["pages"])

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")


@router.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@router.get("/demo", response_class=HTMLResponse)
async def read_customer_portal():
    portal_path = os.path.join(STATIC_DIR, "portal.html")
    with open(portal_path, "r", encoding="utf-8") as f:
        return f.read()


@router.get("/portal")
async def portal_alias():
    return RedirectResponse(url="/demo", status_code=302)
