"""
FastAPI server exposing the Clinical Triage OpenEnv API.
Endpoints: POST /reset, POST /step, GET /state, GET /health
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.env import ClinicalTriageEnv
from server.models import (
    ClinicalAction,
    ResetResult,
    StateResult,
    StepResult,
    TaskID,
)


# ──────────────────────────────────────────────
# Request / Response bodies
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = TaskID.EASY  # defaults to task_easy if not provided (handles empty {} body)


class StepRequest(BaseModel):
    content: str


# ──────────────────────────────────────────────
# App lifecycle
# ──────────────────────────────────────────────

_env: ClinicalTriageEnv = ClinicalTriageEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = ClinicalTriageEnv()
    yield


app = FastAPI(
    title        = "Clinical Triage Navigator — OpenEnv",
    description  = "Real-world emergency department triage environment for RL agent evaluation.",
    version      = "1.0.0",
    lifespan     = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness probe — returns 200 when server is ready."""
    return {"status": "ok", "env": "clinical-triage-env", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResult)
async def reset(request: ResetRequest) -> ResetResult:
    """
    Start a new episode.
    task_id must be one of: task_easy | task_medium | task_hard
    """
    valid_tasks = {TaskID.EASY, TaskID.MEDIUM, TaskID.HARD}
    if request.task_id not in valid_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task_id '{request.task_id}'. Choose from: {sorted(valid_tasks)}",
        )
    return _env.reset(task_id=request.task_id)


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest) -> StepResult:
    """
    Submit one agent action.
    - task_easy:  one of IMMEDIATE | URGENT | SEMI_URGENT | NON_URGENT
    - task_medium: comma-separated test codes e.g. CBC,CMP,ECG
    - task_hard:  structured discharge plan text
    """
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=422, detail="Action content must not be empty.")
    try:
        action = ClinicalAction(content=request.content)
        return _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=StateResult)
async def state() -> StateResult:
    """Return current episode state without modifying it."""
    return _env.state()


# ──────────────────────────────────────────────
# Dev entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.main:app", host="0.0.0.0", port=port, reload=False)
