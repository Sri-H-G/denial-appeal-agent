import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import run_agent

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Denial Appeal Agent")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _sse_format(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/api/run-appeal")
def run_appeal():
    def event_stream():
        for event in run_agent():
            yield _sse_format(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")