"""Training wrapper that replays verified placement phases before PPO control."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from _stair_rl_contract import (
    compose_bounded_residual_action,
    placement_phase_ready,
)


class DeterministicPolicy(Protocol):
    """Minimal Stable-Baselines-style inference contract used by the wrapper."""

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, object]: ...


class PlacementPhaseTrainingEnv(gym.Wrapper):
    """Expose only one placement phase while deterministic policies replay its prefix.

    The underlying episode remains physically continuous. ``reset`` advances the
    simulator through every earlier leg and inter-leg transfer, then returns the
    first observation controlled by the trainable target-leg PPO. Prefix actions
    and rewards are intentionally invisible to PPO.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        target_leg: str,
        precursor_policies: Mapping[str, DeterministicPolicy],
        target_base_policy: DeterministicPolicy | None = None,
        target_residual_scale: float = 0.25,
        target_residual_mask: np.ndarray | None = None,
        maximum_reset_attempts: int = 8,
        maximum_precursor_steps: int = 1800,
        cache_phase_snapshot: bool = True,
    ) -> None:
        super().__init__(env)
        if maximum_reset_attempts < 1:
            raise ValueError("maximum_reset_attempts must be positive")
        if maximum_precursor_steps < 1:
            raise ValueError("maximum_precursor_steps must be positive")

        self.raw_env = env.unwrapped
        self.target_leg = str(target_leg)
        sequence = tuple(self.raw_env.placement_sequence_legs)
        if self.target_leg not in sequence:
            raise ValueError(
                f"Target leg {self.target_leg!r} is outside {sequence}"
            )
        self.target_position = sequence.index(self.target_leg)
        self.required_precursor_legs = sequence[: self.target_position]
        self.precursor_policies = dict(precursor_policies)
        missing = sorted(
            set(self.required_precursor_legs) - set(self.precursor_policies)
        )
        if missing:
            raise ValueError(f"Missing precursor policies for: {missing}")
        self.target_base_policy = target_base_policy
        self.target_residual_scale = float(target_residual_scale)
        if self.target_base_policy is not None and not (
            0.0 < self.target_residual_scale <= 1.0
        ):
            raise ValueError("target_residual_scale must be within (0, 1]")
        self.target_residual_mask = (
            None
            if target_residual_mask is None
            else np.asarray(target_residual_mask, dtype=np.float32).copy()
        )
        if self.target_residual_mask is not None and (
            self.target_residual_mask.shape != self.action_space.shape
            or np.any(self.target_residual_mask < 0.0)
            or np.any(self.target_residual_mask > 1.0)
        ):
            raise ValueError("target_residual_mask must match the action space")

        self.maximum_reset_attempts = int(maximum_reset_attempts)
        self.maximum_precursor_steps = int(maximum_precursor_steps)
        self.cache_phase_snapshot = bool(cache_phase_snapshot)
        self.phase_snapshot: object | None = None
        self.reset_calls = 0
        self.reset_attempts = 0
        self.failed_precursor_attempts = 0
        self.successful_precursor_attempts = 0
        self.total_precursor_steps = 0
        self.cached_phase_restores = 0
        self.failed_cached_phase_restores = 0
        self.last_precursor_steps = 0
        self.last_precursor_failure_reasons: list[str] = []
        self.latest_observation: np.ndarray | None = None

    def _capture_phase_snapshot(self) -> None:
        if not self.cache_phase_snapshot:
            return
        capture = getattr(
            self.raw_env,
            "capture_placement_phase_snapshot",
            None,
        )
        if callable(capture):
            self.phase_snapshot = capture()

    def _restore_phase_snapshot(
        self,
        *,
        seed: int | None,
        options: dict[str, Any] | None,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        if self.phase_snapshot is None:
            return None
        restore = getattr(
            self.raw_env,
            "restore_placement_phase_snapshot",
            None,
        )
        if not callable(restore):
            self.phase_snapshot = None
            return None
        try:
            observation, info = restore(
                self.phase_snapshot,
                seed=seed,
                options=options,
            )
        except (RuntimeError, ValueError):
            self.failed_cached_phase_restores += 1
            self.phase_snapshot = None
            return None
        if not self._target_phase_ready():
            self.failed_cached_phase_restores += 1
            self.phase_snapshot = None
            return None
        self.cached_phase_restores += 1
        self.last_precursor_steps = 0
        self.last_precursor_failure_reasons = []
        self.latest_observation = np.asarray(
            observation,
            dtype=np.float32,
        ).copy()
        return observation, self._phase_info(
            info,
            attempt=0,
            precursor_steps=0,
            snapshot_restored=True,
        )

    def _target_phase_ready(self) -> bool:
        return placement_phase_ready(
            sequence_legs=self.raw_env.placement_sequence_legs,
            completed_legs=self.raw_env.completed_placement_legs,
            active_leg=self.raw_env.placement_swing_leg,
            transfer_active=self.raw_env.placement_transfer_active,
            target_leg=self.target_leg,
        )

    def _predict_precursor_action(self, observation: np.ndarray) -> np.ndarray:
        if self.raw_env.placement_transfer_active:
            return np.zeros(self.action_space.shape, dtype=np.float32)
        active_leg = str(self.raw_env.placement_swing_leg)
        if active_leg == self.target_leg:
            raise RuntimeError(
                "Target phase began before its precursor state was accepted"
            )
        try:
            policy = self.precursor_policies[active_leg]
        except KeyError as exc:
            raise RuntimeError(
                f"No deterministic precursor policy for {active_leg}"
            ) from exc
        action, _ = policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def _phase_info(
        self,
        info: Mapping[str, Any],
        *,
        attempt: int,
        precursor_steps: int,
        snapshot_restored: bool = False,
    ) -> dict[str, Any]:
        return {
            **dict(info),
            "phase_training_ready": True,
            "phase_training_target_leg": self.target_leg,
            "phase_training_reset_attempt": int(attempt),
            "phase_training_precursor_steps": int(precursor_steps),
            "phase_training_snapshot_restored": bool(snapshot_restored),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.reset_calls += 1
        restored = self._restore_phase_snapshot(seed=seed, options=options)
        if restored is not None:
            return restored
        last_info: dict[str, Any] = {}
        for attempt in range(1, self.maximum_reset_attempts + 1):
            self.reset_attempts += 1
            attempt_seed = None if seed is None else int(seed) + attempt - 1
            observation, info = self.env.reset(
                seed=attempt_seed,
                options=options,
            )
            if self._target_phase_ready():
                self.successful_precursor_attempts += 1
                self.last_precursor_steps = 0
                self._capture_phase_snapshot()
                self.latest_observation = np.asarray(
                    observation,
                    dtype=np.float32,
                ).copy()
                return observation, self._phase_info(
                    info,
                    attempt=attempt,
                    precursor_steps=0,
                )

            for precursor_steps in range(1, self.maximum_precursor_steps + 1):
                action = self._predict_precursor_action(observation)
                observation, _, terminated, truncated, info = self.env.step(action)
                self.total_precursor_steps += 1
                if self._target_phase_ready():
                    self.successful_precursor_attempts += 1
                    self.last_precursor_steps = precursor_steps
                    self.last_precursor_failure_reasons = []
                    self._capture_phase_snapshot()
                    self.latest_observation = np.asarray(
                        observation,
                        dtype=np.float32,
                    ).copy()
                    return observation, self._phase_info(
                        info,
                        attempt=attempt,
                        precursor_steps=precursor_steps,
                    )
                if terminated or truncated:
                    last_info = dict(info)
                    break
            else:
                last_info = {
                    "failure_reasons": ["phase_training_precursor_timeout"]
                }

            self.failed_precursor_attempts += 1
            self.last_precursor_failure_reasons = list(
                last_info.get("failure_reasons", ())
            )

        raise RuntimeError(
            "Could not reach placement training phase "
            f"{self.target_leg!r} after {self.maximum_reset_attempts} attempts; "
            f"last failures={self.last_precursor_failure_reasons}"
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._target_phase_ready():
            raise RuntimeError(
                f"PPO action requested outside target phase {self.target_leg!r}"
            )
        residual_action = np.asarray(action, dtype=np.float32)
        applied_action = residual_action
        base_action: np.ndarray | None = None
        if self.target_base_policy is not None:
            if self.latest_observation is None:
                raise RuntimeError("Target phase has no current observation")
            predicted, _ = self.target_base_policy.predict(
                self.latest_observation,
                deterministic=True,
            )
            base_action = np.asarray(predicted, dtype=np.float32)
            applied_action = compose_bounded_residual_action(
                base_action,
                residual_action,
                residual_scale=self.target_residual_scale,
                residual_mask=self.target_residual_mask,
            )
        observation, reward, terminated, truncated, info = self.env.step(
            applied_action
        )
        self.latest_observation = np.asarray(observation, dtype=np.float32).copy()
        result_info = dict(info)
        result_info["phase_training_residual_scale"] = (
            self.target_residual_scale
            if self.target_base_policy is not None
            else None
        )
        result_info["phase_training_base_action_max_abs"] = (
            float(np.max(np.abs(base_action)))
            if base_action is not None
            else None
        )
        result_info["phase_training_residual_action_max_abs"] = float(
            np.max(np.abs(residual_action))
        )
        return observation, float(reward), terminated, truncated, result_info

    def training_stats(self) -> dict[str, object]:
        """Return JSON-safe prefix replay evidence for the training report."""

        return {
            "target_leg": self.target_leg,
            "required_precursor_legs": list(self.required_precursor_legs),
            "target_action_mode": (
                "frozen_base_plus_bounded_ppo_residual"
                if self.target_base_policy is not None
                else "direct_ppo_action"
            ),
            "target_residual_scale": (
                self.target_residual_scale
                if self.target_base_policy is not None
                else None
            ),
            "target_residual_active_action_indices": (
                np.flatnonzero(self.target_residual_mask).tolist()
                if self.target_residual_mask is not None
                else None
            ),
            "maximum_reset_attempts": self.maximum_reset_attempts,
            "maximum_precursor_steps": self.maximum_precursor_steps,
            "phase_snapshot_cache_enabled": self.cache_phase_snapshot,
            "phase_snapshot_cached": self.phase_snapshot is not None,
            "cached_phase_restores": self.cached_phase_restores,
            "failed_cached_phase_restores": (
                self.failed_cached_phase_restores
            ),
            "reset_calls": self.reset_calls,
            "reset_attempts": self.reset_attempts,
            "failed_precursor_attempts": self.failed_precursor_attempts,
            "successful_precursor_attempts": self.successful_precursor_attempts,
            "total_precursor_steps": self.total_precursor_steps,
            "last_precursor_steps": self.last_precursor_steps,
            "last_precursor_failure_reasons": list(
                self.last_precursor_failure_reasons
            ),
        }
