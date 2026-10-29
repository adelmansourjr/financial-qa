"""Local, validated HTTP interface with an evidence explorer."""

from __future__ import annotations

from pathlib import Path
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .data import Document, load_examples
from .http_guard import ResearchHTTPGuard
from .planner import LocalLLM, Planner
from .selective import SemanticBaseline
from .service import answer


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=2000)
    table: list[list[str]] = Field(min_length=1, max_length=100)
    paragraphs: list[str] = Field(default_factory=list, max_length=30)


def create_app(
    planner: Planner | None = None, data_path: Path = Path("data/tatqa_dev.json")
) -> FastAPI:
    app = FastAPI(title="Grounded Financial QA", version="0.1.0")
    app.add_middleware(ResearchHTTPGuard)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    engine = planner or SemanticBaseline()
    # The local model has one slot, and risk planners can revoke shared state.
    # Reject overlapping inference instead of queueing private documents through
    # timeouts or racing a fail-closed transition. Health/static routes stay free.
    inference_slot = BoundedSemaphore(1)
    public_examples = []
    if data_path.exists():
        # The UI uses already explored examples, never the frozen holdout.
        public_examples = [
            example for example in load_examples(data_path) if example.answer_type == "arithmetic"
        ][:40]

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "backend": type(engine).__name__,
            "reasoning": getattr(
                engine,
                "state",
                {"version": 2, "certified": False, "state": "calibration_not_applied"},
            ),
        }

    @app.get("/examples")
    def examples() -> list[dict]:
        return [
            {"id": example.id, "question": example.question, "document_id": example.document.id}
            for example in public_examples
        ]

    @app.get("/examples/{identifier}")
    def example(identifier: str) -> dict:
        selected = next((item for item in public_examples if item.id == identifier), None)
        if selected is None:
            raise HTTPException(404, "Example is not available in the public demo catalog")
        return {
            "question": selected.question,
            "table": selected.document.table,
            "paragraphs": selected.document.paragraphs,
            "source": {
                "dataset": "TAT-QA development; CC BY 4.0",
                "document_id": selected.document.id,
                "url": "https://nextplusplus.github.io/TAT-QA/",
            },
        }

    @app.post("/answer")
    def predict(request: AnswerRequest) -> dict:
        if sum(len(cell) for row in request.table for cell in row) > 30_000:
            raise HTTPException(422, "Table exceeds the context budget")
        if (
            any(len(row) > 30 for row in request.table)
            or sum(map(len, request.paragraphs)) > 30_000
        ):
            raise HTTPException(422, "Document exceeds the context budget")
        if not request.table[0] or any(len(row) != len(request.table[0]) for row in request.table):
            raise HTTPException(422, "Table must be rectangular and have a header row")
        document = Document(
            "user-document", tuple(map(tuple, request.table)), tuple(request.paragraphs)
        )
        if not inference_slot.acquire(blocking=False):
            raise HTTPException(
                503, "Inference is busy; retry shortly", headers={"Retry-After": "1"}
            )
        try:
            response = answer(document, request.question, engine)
            response["reasoning"] = getattr(
                engine,
                "state",
                {"version": 2, "certified": False, "state": "calibration_not_applied"},
            )
            return response
        finally:
            inference_slot.release()

    return app


def local_llm_app() -> FastAPI:
    return create_app(LocalLLM())
