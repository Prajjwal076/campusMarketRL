"""OpenEnv-compatible environment implementation."""

from __future__ import annotations

import random

from campus_market_env._compat import Environment
from campus_market_env.models import (
    CampusMarketAction,
    CampusMarketObservation,
    CampusMarketState,
)
from campus_market_env.server.engine import build_initial_observation, compute_step
from campus_market_env.server.state_manager import create_initial_state, transition_after_step

InfoValue = str | int | float | bool
InfoDict = dict[str, InfoValue]


class CampusMarketEnv(Environment):
    """Deterministic campus market environment with seeded randomness."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = 0 if seed is None else seed
        self._rng = random.Random(self._seed)
        self._episode_counter = 0
        self._state: CampusMarketState | None = None
        self._last_observation: CampusMarketObservation | None = None

    def reset(self, seed: int | None = None) -> CampusMarketObservation:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)

        self._episode_counter += 1
        self._state = create_initial_state(
            episode_id=f"episode-{self._episode_counter:04d}-seed-{self._seed}",
        )
        self._last_observation = build_initial_observation(
            state=self._state,
            base_seed=self._next_step_seed(),
        )
        return self._last_observation

    def step(
        self,
        action: CampusMarketAction,
    ) -> tuple[CampusMarketObservation, float, bool, InfoDict]:
        current_state = self.state
        if current_state.done:
            raise RuntimeError("Environment episode is complete. Call reset() before step().")
        if self._last_observation is None:
            raise RuntimeError("Environment observation state is unavailable. Call reset().")

        validated_action = CampusMarketAction.model_validate(action)
        step_result = compute_step(
            state=current_state,
            action=validated_action,
            previous_observation=self._last_observation,
            base_seed=self._next_step_seed(),
        )
        next_state = transition_after_step(
            state=current_state,
            revenue=step_result.observation.revenue,
            satisfaction=step_result.observation.customer_satisfaction,
        )
        self._state = next_state
        self._last_observation = step_result.observation

        info: InfoDict = {
            "episode_id": current_state.episode_id,
            "executed_day": current_state.current_day,
            "executed_phase": current_state.current_phase,
            "next_day": next_state.current_day,
            "next_phase": next_state.current_phase,
            "total_steps": next_state.total_steps,
            "done": next_state.done,
            **step_result.debug,
        }
        return step_result.observation, step_result.reward, next_state.done, info

    @property
    def state(self) -> CampusMarketState:
        if self._state is None:
            raise RuntimeError("Environment has not been reset.")
        return self._state

    def _next_step_seed(self) -> int:
        return self._rng.randint(1, 1_000_000_000)
