"""FastAPI application exposing the campus market environment."""

from __future__ import annotations

from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from campus_market_env.models import (
    CampusMarketAction,
    CampusMarketState,
    CampusMarketStepResult,
)
from campus_market_env.server.environment import CampusMarketEnv

InfoValue = str | int | float | bool


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = None


class StateResponse(BaseModel):
    state: CampusMarketState


class HealthResponse(BaseModel):
    status: str


def create_app(environment: CampusMarketEnv | None = None) -> FastAPI:
    app = FastAPI(
        title="Campus Market Environment",
        version="0.1.0",
        description="OpenEnv-compatible campus market reinforcement learning environment.",
    )
    env = environment or CampusMarketEnv()
    lock = Lock()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/reset", response_model=CampusMarketStepResult)
    def reset_environment(payload: ResetRequest) -> CampusMarketStepResult:
        with lock:
            observation = env.reset(seed=payload.seed)
            current_state = env.state
        info: dict[str, InfoValue] = {
            "episode_id": current_state.episode_id,
            "executed_day": current_state.current_day,
            "executed_phase": current_state.current_phase,
            "next_day": current_state.current_day,
            "next_phase": current_state.current_phase,
            "total_steps": current_state.total_steps,
            "done": current_state.done,
        }
        return CampusMarketStepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info=info,
        )

    @app.post("/step", response_model=CampusMarketStepResult)
    def step_environment(action: CampusMarketAction) -> CampusMarketStepResult:
        try:
            with lock:
                observation, reward, done, info = env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CampusMarketStepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    @app.get("/state", response_model=StateResponse)
    def get_state() -> StateResponse:
        try:
            with lock:
                current_state = env.state
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StateResponse(state=current_state)

    return app


app = create_app()
