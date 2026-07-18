import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import run_agent
from app.tools import DATA_DIR

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Denial Appeal Agent")

# Metadata for the frontend's "source document" badge -- not used by the
# agent itself, just descriptive info so the UI can show what's really
# being processed (real vs. synthetic, filename, size).
CASE_META = {
    "synthetic": {
        "label": "Synthetic demo case",
        "patient": "Jane Doe / Meridian Health Plan",
        "source_type": "synthetic",
        "policy_file": "policy_rx114.txt",
    },
    "uhc_real": {
        "label": "Real-world case",
        "patient": "Marcus Bell / UnitedHealthcare",
        "source_type": "real",
        "policy_file": "real_world/uhc_adalimumab_policy.txt",
    },
}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _sse_format(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/api/case-info")
def case_info(case: str = "synthetic"):
    """
    Returns descriptive metadata about a case's source policy document,
    including its real character count on disk -- used by the frontend
    to render the "source document" badge before/while the agent runs.
    """
    meta = CASE_META.get(case)
    if meta is None:
        return {"error": f"Unknown case '{case}'"}

    policy_path = DATA_DIR / meta["policy_file"]
    char_count = len(policy_path.read_text()) if policy_path.exists() else None

    return {**meta, "case": case, "policy_char_count": char_count}


@app.get("/api/run-appeal")
def run_appeal(case: str = "synthetic"):
    """
    Streams agent events as Server-Sent Events. Each event is one step
    of the tool-use loop: an assistant thought, a tool call, a tool
    result, or the final drafted letter.
    """

    def event_stream():
        for event in run_agent(case=case):
            yield _sse_format(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")