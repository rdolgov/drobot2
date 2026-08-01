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
        target_base_mask: np.ndarray | None = None,
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
        self.target_base_mask = (
            None
            if target_base_mask is None
            else np.asarray(target_base_mask, dtype=np.float32).copy()
        )
        if self.target_base_mask is not None and (
            self.target_base_policy is None
            or self.target_base_mask.shape != self.action_space.shape
            or np.any(self.target_base_mask < 0.0)
            or np.any(self.target_base_mask > 1.0)
        ):
            raise ValueError(
                "target_base_mask requires a base policy and must match the "
                "action space"
            )
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
        self.target_steps = 0
        self.maximum_target_swing_lift_m = 0.0
        self.minimum_target_base_clearance_m = float("inf")
        self.minimum_target_support_margin_m = float("inf")
        self.minimum_target_support_contact_fraction = 1.0
        self.maximum_target_support_slip_m = 0.0
        self.minimum_target_upright_cosine = 1.0
        self.maximum_target_goal_hold_steps = 0
        self.maximum_target_desired_lift_m = 0.0
        self.initial_target_swing_reference: np.ndarray | None = None
        self.maximum_target_swing_reference_change_rad = 0.0
        self.minimum_target_swing_reference_delta_rad: np.ndarray | None = None
        self.maximum_target_swing_reference_delta_rad: np.ndarray | None = None
        self.initial_target_swing_actual: np.ndarray | None = None
        self.maximum_target_swing_actual_change_rad = 0.0
        self.minimum_target_swing_actual_delta_rad: np.ndarray | None = None
        self.maximum_target_swing_actual_delta_rad: np.ndarray | None = None
        self.maximum_target_residual_action_abs = 0.0
        self.maximum_target_applied_action_abs = 0.0

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
            if self.target_base_mask is not None:
                base_action = base_action * self.target_base_mask
            applied_action = compose_bounded_residual_action(
                base_action,
                residual_action,
                residual_scale=self.target_residual_scale,
                residual_mask=self.target_residual_mask,
            )
        elif self.target_residual_mask is not None:
            applied_action = np.clip(
                residual_action * self.target_residual_mask,
                self.action_space.low,
                self.action_space.high,
            ).astype(np.float32)
        observation, reward, terminated, truncated, info = self.env.step(
            applied_action
        )
        self.latest_observation = np.asarray(observation, dtype=np.float32).copy()
        result_info = dict(info)
        self.target_steps += 1
        lift_by_leg = dict(result_info.get("maximum_foot_lift_m_by_leg", {}))
        self.maximum_target_swing_lift_m = max(
            self.maximum_target_swing_lift_m,
            float(
                result_info.get(
                    "placement_swing_lift_m",
                    lift_by_leg.get(self.target_leg, 0.0),
                )
            ),
        )
        self.minimum_target_base_clearance_m = min(
            self.minimum_target_base_clearance_m,
            float(result_info.get("base_clearance_m", float("inf"))),
        )
        self.minimum_target_support_margin_m = min(
            self.minimum_target_support_margin_m,
            float(result_info.get("placement_support_margin_m", float("inf"))),
        )
        self.minimum_target_support_contact_fraction = min(
            self.minimum_target_support_contact_fraction,
            float(result_info.get("placement_support_contact_fraction", 1.0)),
        )
        self.maximum_target_support_slip_m = max(
            self.maximum_target_support_slip_m,
            float(result_info.get("maximum_support_slip_m", 0.0)),
        )
        self.minimum_target_upright_cosine = min(
            self.minimum_target_upright_cosine,
            float(result_info.get("placement_upright_cosine", 1.0)),
        )
        self.maximum_target_goal_hold_steps = max(
            self.maximum_target_goal_hold_steps,
            int(result_info.get("placement_goal_hold_step_count", 0)),
        )
        self.maximum_target_desired_lift_m = max(
            self.maximum_target_desired_lift_m,
            float(result_info.get("placement_desired_lift_m", 0.0)),
        )
        swing_reference_value = result_info.get(
            "placement_swing_reference_joint_positions_rad"
        )
        if swing_reference_value is not None:
            swing_reference = np.asarray(
                swing_reference_value,
                dtype=np.float32,
            )
            if self.initial_target_swing_reference is None:
                self.initial_target_swing_reference = swing_reference.copy()
                self.minimum_target_swing_reference_delta_rad = np.zeros_like(
                    swing_reference
                )
                self.maximum_target_swing_reference_delta_rad = np.zeros_like(
                    swing_reference
                )
            reference_delta = (
                swing_reference - self.initial_target_swing_reference
            )
            self.minimum_target_swing_reference_delta_rad = np.minimum(
                self.minimum_target_swing_reference_delta_rad,
                reference_delta,
            )
            self.maximum_target_swing_reference_delta_rad = np.maximum(
                self.maximum_target_swing_reference_delta_rad,
                reference_delta,
            )
            self.maximum_target_swing_reference_change_rad = max(
                self.maximum_target_swing_reference_change_rad,
                float(np.max(np.abs(reference_delta))),
            )
        swing_actual_value = result_info.get(
            "placement_swing_actual_joint_positions_rad"
        )
        if swing_actual_value is not None:
            swing_actual = np.asarray(swing_actual_value, dtype=np.float32)
            if self.initial_target_swing_actual is None:
                self.initial_target_swing_actual = swing_actual.copy()
                self.minimum_target_swing_actual_delta_rad = np.zeros_like(
                    swing_actual
                )
                self.maximum_target_swing_actual_delta_rad = np.zeros_like(
                    swing_actual
                )
            actual_delta = swing_actual - self.initial_target_swing_actual
            self.minimum_target_swing_actual_delta_rad = np.minimum(
                self.minimum_target_swing_actual_delta_rad,
                actual_delta,
            )
            self.maximum_target_swing_actual_delta_rad = np.maximum(
                self.maximum_target_swing_actual_delta_rad,
                actual_delta,
            )
            self.maximum_target_swing_actual_change_rad = max(
                self.maximum_target_swing_actual_change_rad,
                float(np.max(np.abs(actual_delta))),
            )
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
        result_info["phase_training_applied_action_max_abs"] = float(
            np.max(np.abs(applied_action))
        )
        self.maximum_target_residual_action_abs = max(
            self.maximum_target_residual_action_abs,
            result_info["phase_training_residual_action_max_abs"],
        )
        self.maximum_target_applied_action_abs = max(
            self.maximum_target_applied_action_abs,
            result_info["phase_training_applied_action_max_abs"],
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
                else (
                    "masked_direct_ppo_action"
                    if self.target_residual_mask is not None
                    else "direct_ppo_action"
                )
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
            "target_steps": self.target_steps,
            "maximum_target_swing_lift_m": self.maximum_target_swing_lift_m,
            "minimum_target_base_clearance_m": (
                None
                if not np.isfinite(self.minimum_target_base_clearance_m)
                else self.minimum_target_base_clearance_m
            ),
            "minimum_target_support_margin_m": (
                None
                if not np.isfinite(self.minimum_target_support_margin_m)
                else self.minimum_target_support_margin_m
            ),
            "minimum_target_support_contact_fraction": (
                self.minimum_target_support_contact_fraction
            ),
            "maximum_target_support_slip_m": self.maximum_target_support_slip_m,
            "minimum_target_upright_cosine": (
                self.minimum_target_upright_cosine
            ),
            "maximum_target_goal_hold_steps": (
                self.maximum_target_goal_hold_steps
            ),
            "maximum_target_desired_lift_m": (
                self.maximum_target_desired_lift_m
            ),
            "maximum_target_swing_reference_change_rad": (
                self.maximum_target_swing_reference_change_rad
            ),
            "minimum_target_swing_reference_delta_rad": (
                None
                if self.minimum_target_swing_reference_delta_rad is None
                else self.minimum_target_swing_reference_delta_rad.tolist()
            ),
            "maximum_target_swing_reference_delta_rad": (
                None
                if self.maximum_target_swing_reference_delta_rad is None
                else self.maximum_target_swing_reference_delta_rad.tolist()
            ),
            "maximum_target_swing_actual_change_rad": (
                self.maximum_target_swing_actual_change_rad
            ),
            "minimum_target_swing_actual_delta_rad": (
                None
                if self.minimum_target_swing_actual_delta_rad is None
                else self.minimum_target_swing_actual_delta_rad.tolist()
            ),
            "maximum_target_swing_actual_delta_rad": (
                None
                if self.maximum_target_swing_actual_delta_rad is None
                else self.maximum_target_swing_actual_delta_rad.tolist()
            ),
            "maximum_target_residual_action_abs": (
                self.maximum_target_residual_action_abs
            ),
            "maximum_target_applied_action_abs": (
                self.maximum_target_applied_action_abs
            ),
        }
