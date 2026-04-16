"""
Devtools console routes — lightweight debug UI for agents and memory exploration.

Mounted at /console by cli.py via dynamic import. These pages are developer tools
and do not affect the core API or production behavior.
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/console", tags=["devtools"])

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


@router.get("/agents", response_class=HTMLResponse)
async def agent_console(request: Request):
    """Agent debug console — inspect agent config, chat batches, and produced contexts."""
    return templates.TemplateResponse("agent_console.html", {"request": request})


@router.get("/memory", response_class=HTMLResponse)
async def memory_explorer(request: Request):
    """Memory explorer — browse memory snapshots, semantic search, and context list."""
    return templates.TemplateResponse("memory_explorer.html", {"request": request})
