"""Training wrapper that replays verified placement phases before PPO control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from _policy_transfer import predict_with_observation_prefix
from _stair_rl_contract import (
    compose_bounded_residual_action,
    expand_compact_masked_action,
    placement_phase_ready,
    placement_transfer_ready,
)


class DeterministicPolicy(Protocol):
    """Minimal Stable-Baselines-style inference contract used by the wrapper."""

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, object]: ...


class FrozenBaseResidualPolicy:
    """Expose a frozen full-action base plus a bounded residual as one policy.

    The stair sequence already uses nested skills: V17 supplies the full
    rear-right swing baseline and compact V35 corrects its three swing joints.
    Phase PPO must see that exact composition as its frozen base before it can
    learn disjoint support-joint corrections without replacing the verified
    clearance motion.
    """

    def __init__(
        self,
        *,
        base_policy: DeterministicPolicy,
        residual_policy: DeterministicPolicy,
        action_space: gym.spaces.Box,
        residual_scale: float,
        base_mask: np.ndarray | None = None,
        residual_mask: np.ndarray | None = None,
        compact_residual_action: bool = False,
    ) -> None:
        self.base_policy = base_policy
        self.residual_policy = residual_policy
        self.action_space = action_space
        self.action_shape = tuple(action_space.shape)
        if len(self.action_shape) != 1:
            raise ValueError("composed policy action space must be flat")
        self.residual_scale = float(residual_scale)
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must be within (0, 1]")
        self.base_mask = self._validated_mask(base_mask, "base_mask")
        self.residual_mask = self._validated_mask(
            residual_mask,
            "residual_mask",
        )
        self.compact_residual_action = bool(compact_residual_action)
        if self.compact_residual_action and self.residual_mask is None:
            raise ValueError(
                "compact_residual_action requires residual_mask"
            )
        observation_spaces = (
            getattr(base_policy, "observation_space", None),
            getattr(residual_policy, "observation_space", None),
        )
        if any(space is None for space in observation_spaces):
            raise ValueError("composed policies require observation spaces")
        self.observation_space = max(
            observation_spaces,
            key=lambda space: int(space.shape[0]),
        )

    def _validated_mask(
        self,
        value: np.ndarray | None,
        label: str,
    ) -> np.ndarray | None:
        if value is None:
            return None
        mask = np.asarray(value, dtype=np.float32).copy()
        if mask.shape != self.action_shape:
            raise ValueError(f"{label} must match the action space")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError(f"{label} must be binary")
        return mask

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, object]:
        """Predict the same nested frozen action used by composed evaluation."""

        base_action, _ = predict_with_observation_prefix(
            self.base_policy,
            observation,
            deterministic=deterministic,
        )
        base_action = np.asarray(base_action, dtype=np.float32)
        if base_action.shape != self.action_shape:
            raise ValueError("base policy action must match the action space")
        if self.base_mask is not None:
            base_action = base_action * self.base_mask

        residual_action, state = predict_with_observation_prefix(
            self.residual_policy,
            observation,
            deterministic=deterministic,
        )
        residual_action = np.asarray(residual_action, dtype=np.float32)
        if self.compact_residual_action:
            residual_action = expand_compact_masked_action(
                residual_action,
                self.residual_mask,
            )
        elif residual_action.shape != self.action_shape:
            raise ValueError(
                "residual policy action must match the action space"
            )
        return (
            compose_bounded_residual_action(
                base_action,
                residual_action,
                residual_scale=self.residual_scale,
                residual_mask=self.residual_mask,
            ),
            state,
        )


class PlacementPhaseTrainingEnv(gym.Wrapper):
    """Expose one placement or transfer phase after deterministic prefix replay.

    The underlying episode remains physically continuous. ``reset`` advances the
    simulator through every earlier leg and inter-leg transfer, then returns the
    first observation controlled by the trainable target-leg PPO. With
    ``train_transfer`` enabled, PPO starts at the inter-leg transfer into the
    target leg and the wrapper ends its episode when that transfer is accepted.
    Prefix actions and rewards are intentionally invisible to PPO.
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
        compact_residual_action: bool = False,
        train_transfer: bool = False,
        transfer_post_hold_seconds: float = 0.0,
        train_post_transfer_hold_only: bool = False,
        maximum_reset_attempts: int = 8,
        maximum_precursor_steps: int = 1800,
        cache_phase_snapshot: bool = True,
        initial_phase_snapshot: Mapping[str, object] | None = None,
        initial_phase_snapshot_mode: str | None = None,
        transfer_unload_thresholds_n: Sequence[float] = (),
        transfer_unload_successes_per_level: int = 2,
        transfer_upright_cosines: Sequence[float] = (),
    ) -> None:
        super().__init__(env)
        if maximum_reset_attempts < 1:
            raise ValueError("maximum_reset_attempts must be positive")
        if maximum_precursor_steps < 1:
            raise ValueError("maximum_precursor_steps must be positive")

        self.raw_env = env.unwrapped
        self.raw_action_space = env.action_space
        self.target_leg = str(target_leg)
        self.train_transfer = bool(train_transfer)
        self.transfer_unload_thresholds_n = tuple(
            float(value) for value in transfer_unload_thresholds_n
        )
        self.transfer_unload_successes_per_level = int(
            transfer_unload_successes_per_level
        )
        self.transfer_upright_cosines = tuple(
            float(value) for value in transfer_upright_cosines
        )
        if self.transfer_unload_successes_per_level < 1:
            raise ValueError(
                "transfer_unload_successes_per_level must be positive"
            )
        if self.transfer_unload_thresholds_n and not self.train_transfer:
            raise ValueError("transfer unload curriculum requires train_transfer")
        if self.transfer_unload_thresholds_n:
            if not all(
                np.isfinite(value) and value > 0.0
                for value in self.transfer_unload_thresholds_n
            ):
                raise ValueError(
                    "transfer unload thresholds must be finite and positive"
                )
            if any(
                current <= following
                for current, following in zip(
                    self.transfer_unload_thresholds_n,
                    self.transfer_unload_thresholds_n[1:],
                    strict=False,
                )
            ):
                raise ValueError(
                    "transfer unload thresholds must be strictly descending"
                )
        if self.transfer_upright_cosines:
            if len(self.transfer_upright_cosines) != len(
                self.transfer_unload_thresholds_n
            ):
                raise ValueError(
                    "transfer upright gates must match unload thresholds"
                )
            if not all(
                np.isfinite(value) and 0.0 < value <= 1.0
                for value in self.transfer_upright_cosines
            ):
                raise ValueError(
                    "transfer upright gates must be finite and within (0, 1]"
                )
            if any(
                current > following
                for current, following in zip(
                    self.transfer_upright_cosines,
                    self.transfer_upright_cosines[1:],
                    strict=False,
                )
            ):
                raise ValueError(
                    "transfer upright gates must be nondecreasing"
                )
        self.transfer_unload_curriculum_level = 0
        self.transfer_unload_level_successes = 0
        self.transfer_unload_successes_by_level = [
            0 for _ in self.transfer_unload_thresholds_n
        ]
        self.transfer_unload_curriculum_transitions: list[dict[str, float | int]] = []
        if self.transfer_unload_thresholds_n:
            self._validate_and_apply_transfer_unload_threshold(
                self.transfer_unload_thresholds_n[0],
                validate_final=True,
            )
        self.train_post_transfer_hold_only = bool(
            train_post_transfer_hold_only
        )
        if self.train_post_transfer_hold_only and not self.train_transfer:
            raise ValueError(
                "train_post_transfer_hold_only requires train_transfer"
            )
        self.transfer_post_hold_seconds = float(transfer_post_hold_seconds)
        if self.transfer_post_hold_seconds < 0.0:
            raise ValueError("transfer_post_hold_seconds cannot be negative")
        if self.transfer_post_hold_seconds > 0.0 and not self.train_transfer:
            raise ValueError(
                "transfer_post_hold_seconds requires train_transfer"
            )
        if (
            self.train_post_transfer_hold_only
            and self.transfer_post_hold_seconds <= 0.0
        ):
            raise ValueError(
                "train_post_transfer_hold_only requires a positive hold"
            )
        self.transfer_post_hold_steps = int(
            round(
                self.transfer_post_hold_seconds
                * float(getattr(self.raw_env, "control_hz", 0.0))
            )
        )
        if self.transfer_post_hold_seconds > 0.0 and (
            self.transfer_post_hold_steps < 1
        ):
            raise ValueError(
                "transfer_post_hold_seconds is below one control step"
            )
        self.transfer_post_hold_steps_remaining = 0
        self.target_mode = (
            "post_transfer_hold"
            if self.train_post_transfer_hold_only
            else ("inter_leg_transfer" if self.train_transfer else "placement")
        )
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
        if missing and initial_phase_snapshot is None:
            raise ValueError(f"Missing precursor policies for: {missing}")
        self.target_base_policy = target_base_policy
        self.target_base_mask = (
            None
            if target_base_mask is None
            else np.asarray(target_base_mask, dtype=np.float32).copy()
        )
        if self.target_base_mask is not None and (
            self.target_base_policy is None
            or self.target_base_mask.shape != self.raw_action_space.shape
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
            self.target_residual_mask.shape != self.raw_action_space.shape
            or np.any(self.target_residual_mask < 0.0)
            or np.any(self.target_residual_mask > 1.0)
        ):
            raise ValueError("target_residual_mask must match the action space")
        self.compact_residual_action = bool(compact_residual_action)
        if self.compact_residual_action and self.target_residual_mask is None:
            raise ValueError(
                "compact_residual_action requires target_residual_mask"
            )
        self.compact_action_indices = (
            np.flatnonzero(self.target_residual_mask).astype(np.int64)
            if self.compact_residual_action
            else np.arange(self.raw_action_space.shape[0], dtype=np.int64)
        )
        if self.compact_residual_action and not self.compact_action_indices.size:
            raise ValueError("compact_residual_action requires an active joint")
        if self.compact_residual_action:
            self.action_space = gym.spaces.Box(
                low=np.asarray(self.raw_action_space.low)[self.compact_action_indices],
                high=np.asarray(self.raw_action_space.high)[self.compact_action_indices],
                dtype=np.float32,
            )

        self.maximum_reset_attempts = int(maximum_reset_attempts)
        self.maximum_precursor_steps = int(maximum_precursor_steps)
        self.cache_phase_snapshot = bool(cache_phase_snapshot)
        self.initial_phase_snapshot_supplied = initial_phase_snapshot is not None
        self.phase_snapshot: object | None = (
            deepcopy(initial_phase_snapshot)
            if initial_phase_snapshot is not None
            else None
        )
        self.phase_snapshot_mode = (
            str(initial_phase_snapshot_mode)
            if initial_phase_snapshot_mode is not None
            else (
                "inter_leg_transfer" if self.train_transfer else self.target_mode
            )
        )
        expected_snapshot_mode = (
            "inter_leg_transfer" if self.train_transfer else self.target_mode
        )
        if self.phase_snapshot_mode != expected_snapshot_mode:
            raise ValueError(
                "initial_phase_snapshot_mode does not match the target mode: "
                f"{self.phase_snapshot_mode!r} != {expected_snapshot_mode!r}"
            )
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
        self.completed_target_transfers = 0
        self.completed_target_transfer_holds = 0
        self.maximum_target_transfer_balance_error_m = 0.0
        self.maximum_target_transfer_body_rate_rad_s = 0.0
        self.minimum_target_transfer_swing_load_n = float("inf")
        transfer_training_config = dict(
            getattr(self.raw_env, "inter_leg_transfer_config", {}).get(
                "training_reward",
                {},
            )
        )
        self.transfer_balance_progress_reward_per_m = float(
            transfer_training_config.get(
                "balance_target_error_progress_per_m",
                0.0,
            )
        )
        self.transfer_support_margin_progress_reward_per_m = float(
            transfer_training_config.get(
                "support_margin_progress_per_m",
                0.0,
            )
        )
        self.transfer_swing_load_reduction_reward_per_n = float(
            transfer_training_config.get(
                "swing_load_reduction_per_n",
                0.0,
            )
        )
        self.transfer_progress_reward_clip_m = float(
            transfer_training_config.get(
                "maximum_progress_m_per_step",
                0.010,
            )
        )
        self.transfer_load_progress_reward_clip_n = float(
            transfer_training_config.get(
                "maximum_load_progress_n_per_step",
                1.0,
            )
        )
        transfer_reward_values = (
            self.transfer_balance_progress_reward_per_m,
            self.transfer_support_margin_progress_reward_per_m,
            self.transfer_swing_load_reduction_reward_per_n,
            self.transfer_progress_reward_clip_m,
            self.transfer_load_progress_reward_clip_n,
        )
        if not all(np.isfinite(value) for value in transfer_reward_values):
            raise ValueError("transfer training reward values must be finite")
        if any(value < 0.0 for value in transfer_reward_values):
            raise ValueError(
                "transfer training reward values cannot be negative"
            )
        self.previous_transfer_balance_error_m: float | None = None
        self.previous_transfer_support_margin_m: float | None = None
        self.previous_transfer_swing_load_n: float | None = None
        self.cumulative_transfer_progress_reward = 0.0
        self.cumulative_transfer_balance_error_progress_m = 0.0
        self.cumulative_transfer_support_margin_progress_m = 0.0
        self.cumulative_transfer_swing_load_reduction_n = 0.0

    def _validate_and_apply_transfer_unload_threshold(
        self,
        threshold_n: float,
        *,
        validate_final: bool = False,
    ) -> None:
        """Set the target leg's live gate while preserving the deployment limit."""

        transfer_config = getattr(self.raw_env, "inter_leg_transfer_config", None)
        if not isinstance(transfer_config, dict):
            raise ValueError("raw environment has no mutable inter-leg transfer config")
        overrides = transfer_config.setdefault("override_by_next_swing_leg", {})
        target_config = overrides.setdefault(self.target_leg, {})
        deployment_threshold = float(
            target_config.get(
                "maximum_swing_unloaded_load_n",
                transfer_config.get("maximum_swing_unloaded_load_n", 0.0),
            )
        )
        if validate_final and not np.isclose(
            self.transfer_unload_thresholds_n[-1],
            deployment_threshold,
            atol=1e-9,
        ):
            raise ValueError(
                "final transfer unload threshold must equal the configured "
                f"deployment gate ({deployment_threshold:g} N)"
            )
        target_config["maximum_swing_unloaded_load_n"] = float(threshold_n)
        if self.transfer_upright_cosines:
            deployment_upright = float(
                target_config.get(
                    "minimum_upright_cosine",
                    transfer_config.get(
                        "minimum_upright_cosine",
                        self.raw_env.placement_reference_config[
                            "minimum_success_upright_cosine"
                        ],
                    ),
                )
            )
            if validate_final and not np.isclose(
                self.transfer_upright_cosines[-1],
                deployment_upright,
                atol=1e-9,
            ):
                raise ValueError(
                    "final transfer upright gate must equal the configured "
                    f"deployment gate ({deployment_upright:g})"
                )
            target_config["minimum_upright_cosine"] = (
                self.transfer_upright_cosines[
                    self.transfer_unload_curriculum_level
                ]
            )

    def _record_transfer_unload_curriculum_success(self) -> None:
        if not self.transfer_unload_thresholds_n:
            return
        level = self.transfer_unload_curriculum_level
        self.transfer_unload_successes_by_level[level] += 1
        self.transfer_unload_level_successes += 1
        if (
            self.transfer_unload_level_successes
            < self.transfer_unload_successes_per_level
            or level >= len(self.transfer_unload_thresholds_n) - 1
        ):
            return
        previous_threshold = self.transfer_unload_thresholds_n[level]
        self.transfer_unload_curriculum_level += 1
        self.transfer_unload_level_successes = 0
        next_threshold = self.transfer_unload_thresholds_n[
            self.transfer_unload_curriculum_level
        ]
        self._validate_and_apply_transfer_unload_threshold(next_threshold)
        self.transfer_unload_curriculum_transitions.append(
            {
                "completed_target_transfers": self.completed_target_transfers,
                "from_threshold_n": previous_threshold,
                "to_threshold_n": next_threshold,
            }
        )

    @staticmethod
    def _transfer_balance_error_m(info: Mapping[str, Any]) -> float | None:
        error_xy = info.get("placement_balance_target_error_xy_m")
        if error_xy is not None:
            vector = np.asarray(error_xy, dtype=np.float64).reshape(-1)
            if vector.shape == (2,) and np.all(np.isfinite(vector)):
                return float(np.linalg.norm(vector))
        scalar = info.get("placement_transfer_base_target_error_m")
        if scalar is None or not np.isfinite(float(scalar)):
            return None
        return max(0.0, float(scalar))

    def _reset_transfer_progress_baseline(
        self,
        info: Mapping[str, Any],
    ) -> None:
        self.previous_transfer_balance_error_m = (
            self._transfer_balance_error_m(info)
        )
        margin = info.get("placement_support_margin_m")
        self.previous_transfer_support_margin_m = (
            float(margin)
            if margin is not None and np.isfinite(float(margin))
            else None
        )
        swing_load = info.get("placement_transfer_swing_total_load_n")
        self.previous_transfer_swing_load_n = (
            max(0.0, float(swing_load))
            if swing_load is not None and np.isfinite(float(swing_load))
            else None
        )

    def _transfer_progress_reward(
        self,
        info: Mapping[str, Any],
    ) -> tuple[float, float, float, float]:
        if not self.train_transfer:
            return 0.0, 0.0, 0.0, 0.0
        balance_error = self._transfer_balance_error_m(info)
        margin_value = info.get("placement_support_margin_m")
        support_margin = (
            float(margin_value)
            if margin_value is not None
            and np.isfinite(float(margin_value))
            else None
        )
        swing_load_value = info.get("placement_transfer_swing_total_load_n")
        swing_load = (
            max(0.0, float(swing_load_value))
            if swing_load_value is not None
            and np.isfinite(float(swing_load_value))
            else None
        )
        balance_progress = 0.0
        if (
            balance_error is not None
            and self.previous_transfer_balance_error_m is not None
        ):
            balance_progress = float(
                self.previous_transfer_balance_error_m - balance_error
            )
        margin_progress = 0.0
        if (
            support_margin is not None
            and self.previous_transfer_support_margin_m is not None
        ):
            margin_progress = float(
                support_margin - self.previous_transfer_support_margin_m
            )
        swing_load_reduction = 0.0
        if (
            swing_load is not None
            and self.previous_transfer_swing_load_n is not None
        ):
            swing_load_reduction = float(
                self.previous_transfer_swing_load_n - swing_load
            )
        self.previous_transfer_balance_error_m = balance_error
        self.previous_transfer_support_margin_m = support_margin
        self.previous_transfer_swing_load_n = swing_load
        clip_m = self.transfer_progress_reward_clip_m
        clipped_balance_progress = float(
            np.clip(balance_progress, -clip_m, clip_m)
        )
        clipped_margin_progress = float(
            np.clip(margin_progress, -clip_m, clip_m)
        )
        clipped_swing_load_reduction = float(
            np.clip(
                swing_load_reduction,
                -self.transfer_load_progress_reward_clip_n,
                self.transfer_load_progress_reward_clip_n,
            )
        )
        progress_reward = (
            self.transfer_balance_progress_reward_per_m
            * clipped_balance_progress
            + self.transfer_support_margin_progress_reward_per_m
            * clipped_margin_progress
            + self.transfer_swing_load_reduction_reward_per_n
            * clipped_swing_load_reduction
        )
        self.cumulative_transfer_progress_reward += progress_reward
        self.cumulative_transfer_balance_error_progress_m += (
            clipped_balance_progress
        )
        self.cumulative_transfer_support_margin_progress_m += (
            clipped_margin_progress
        )
        self.cumulative_transfer_swing_load_reduction_n += (
            clipped_swing_load_reduction
        )
        return (
            progress_reward,
            clipped_balance_progress,
            clipped_margin_progress,
            clipped_swing_load_reduction,
        )

    def _capture_phase_snapshot(
        self,
        *,
        post_transfer_hold: bool = False,
    ) -> None:
        if not self.cache_phase_snapshot:
            return
        capture = getattr(
            self.raw_env,
            "capture_placement_phase_snapshot",
            None,
        )
        if callable(capture):
            self.phase_snapshot = capture()
            if post_transfer_hold:
                self.phase_snapshot_mode = "post_transfer_hold"
            elif self.train_transfer:
                self.phase_snapshot_mode = "inter_leg_transfer"
            else:
                self.phase_snapshot_mode = self.target_mode

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
        if self.phase_snapshot_mode == "post_transfer_hold":
            self.transfer_post_hold_steps_remaining = (
                self.transfer_post_hold_steps
            )
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
        readiness = placement_phase_ready
        if self.train_transfer and self.transfer_post_hold_steps_remaining <= 0:
            readiness = placement_transfer_ready
        return readiness(
            sequence_legs=self.raw_env.placement_sequence_legs,
            completed_legs=self.raw_env.completed_placement_legs,
            active_leg=self.raw_env.placement_swing_leg,
            transfer_active=self.raw_env.placement_transfer_active,
            target_leg=self.target_leg,
        )

    def _predict_precursor_action(self, observation: np.ndarray) -> np.ndarray:
        if self.raw_env.placement_transfer_active:
            return np.zeros(self.raw_action_space.shape, dtype=np.float32)
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
        action, _ = predict_with_observation_prefix(
            policy,
            observation,
            deterministic=True,
        )
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
            "phase_training_target_mode": self.target_mode,
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
        self.transfer_post_hold_steps_remaining = 0
        restored = self._restore_phase_snapshot(seed=seed, options=options)
        if restored is not None:
            self._reset_transfer_progress_baseline(restored[1])
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
                phase_info = self._phase_info(
                    info,
                    attempt=attempt,
                    precursor_steps=0,
                )
                self._reset_transfer_progress_baseline(phase_info)
                return observation, phase_info

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
                    phase_info = self._phase_info(
                        info,
                        attempt=attempt,
                        precursor_steps=precursor_steps,
                    )
                    self._reset_transfer_progress_baseline(phase_info)
                    return observation, phase_info
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
            f"Could not reach {self.target_mode} training phase "
            f"{self.target_leg!r} after {self.maximum_reset_attempts} attempts; "
            f"last failures={self.last_precursor_failure_reasons}"
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._target_phase_ready():
            raise RuntimeError(
                "PPO action requested outside target "
                f"{self.target_mode} {self.target_leg!r}"
            )
        policy_action = np.asarray(action, dtype=np.float32)
        if policy_action.shape != self.action_space.shape:
            raise ValueError(
                "phase PPO action shape differs from the exposed action space: "
                f"{policy_action.shape} != {self.action_space.shape}"
            )
        residual_action = policy_action
        if self.compact_residual_action:
            residual_action = np.zeros(
                self.raw_action_space.shape,
                dtype=np.float32,
            )
            residual_action[self.compact_action_indices] = policy_action
        applied_action = residual_action
        base_action: np.ndarray | None = None
        if self.target_base_policy is not None:
            if self.latest_observation is None:
                raise RuntimeError("Target phase has no current observation")
            predicted, _ = predict_with_observation_prefix(
                self.target_base_policy,
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
                self.raw_action_space.low,
                self.raw_action_space.high,
            ).astype(np.float32)
        if (
            self.train_post_transfer_hold_only
            and self.raw_env.placement_transfer_active
        ):
            applied_action = np.zeros(
                self.raw_action_space.shape,
                dtype=np.float32,
            )
        observation, reward, terminated, truncated, info = self.env.step(
            applied_action
        )
        self.latest_observation = np.asarray(observation, dtype=np.float32).copy()
        result_info = dict(info)
        (
            transfer_progress_reward,
            transfer_balance_progress_m,
            transfer_margin_progress_m,
            transfer_swing_load_reduction_n,
        ) = self._transfer_progress_reward(result_info)
        reward = float(reward) + transfer_progress_reward
        result_info["phase_training_transfer_progress_reward"] = (
            transfer_progress_reward
        )
        result_info["phase_training_transfer_balance_error_progress_m"] = (
            transfer_balance_progress_m
        )
        result_info["phase_training_transfer_support_margin_progress_m"] = (
            transfer_margin_progress_m
        )
        result_info["phase_training_transfer_swing_load_reduction_n"] = (
            transfer_swing_load_reduction_n
        )
        transfer_completed = bool(
            self.train_transfer
            and result_info.get("placement_transfer_completed_event")
            and not self.raw_env.placement_transfer_active
            and self.raw_env.placement_swing_leg == self.target_leg
        )
        transfer_hold_completed = False
        if transfer_completed:
            self.completed_target_transfers += 1
            self._record_transfer_unload_curriculum_success()
            self.transfer_post_hold_steps_remaining = (
                self.transfer_post_hold_steps
            )
            if self.transfer_post_hold_steps > 0:
                self._capture_phase_snapshot(post_transfer_hold=True)
            terminated = bool(self.transfer_post_hold_steps == 0)
        elif self.train_transfer and self.transfer_post_hold_steps_remaining > 0:
            self.transfer_post_hold_steps_remaining -= 1
            transfer_hold_completed = bool(
                self.transfer_post_hold_steps_remaining == 0
                and not terminated
                and not truncated
            )
            if transfer_hold_completed:
                self.completed_target_transfer_holds += 1
                terminated = True
        if transfer_completed and self.transfer_post_hold_steps == 0:
            transfer_hold_completed = True
            self.completed_target_transfer_holds += 1
        if transfer_hold_completed:
            reward = float(reward) + float(
                getattr(self.raw_env, "reward_config", {}).get("success", 0.0)
            )
        result_info["phase_training_transfer_completed"] = transfer_completed
        result_info["phase_training_transfer_post_hold_completed"] = (
            transfer_hold_completed
        )
        result_info["phase_training_transfer_post_hold_remaining_steps"] = (
            self.transfer_post_hold_steps_remaining
        )
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
        self.maximum_target_transfer_balance_error_m = max(
            self.maximum_target_transfer_balance_error_m,
            float(
                result_info.get(
                    "placement_transfer_base_target_error_m",
                    0.0,
                )
            ),
        )
        self.maximum_target_transfer_body_rate_rad_s = max(
            self.maximum_target_transfer_body_rate_rad_s,
            float(result_info.get("placement_transfer_body_rate_rad_s", 0.0)),
        )
        transfer_swing_load = result_info.get(
            "placement_transfer_swing_total_load_n"
        )
        if transfer_swing_load is not None:
            self.minimum_target_transfer_swing_load_n = min(
                self.minimum_target_transfer_swing_load_n,
                float(transfer_swing_load),
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
            np.max(np.abs(policy_action))
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
            "target_mode": self.target_mode,
            "train_post_transfer_hold_only": (
                self.train_post_transfer_hold_only
            ),
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
            "compact_residual_action": self.compact_residual_action,
            "policy_action_size": int(self.action_space.shape[0]),
            "raw_action_size": int(self.raw_action_space.shape[0]),
            "maximum_reset_attempts": self.maximum_reset_attempts,
            "maximum_precursor_steps": self.maximum_precursor_steps,
            "phase_snapshot_cache_enabled": self.cache_phase_snapshot,
            "initial_phase_snapshot_supplied": (
                self.initial_phase_snapshot_supplied
            ),
            "phase_snapshot_cached": self.phase_snapshot is not None,
            "phase_snapshot_mode": self.phase_snapshot_mode,
            "transfer_unload_curriculum": {
                "thresholds_n": list(self.transfer_unload_thresholds_n),
                "successes_per_level": self.transfer_unload_successes_per_level,
                "active_level": self.transfer_unload_curriculum_level,
                "active_threshold_n": (
                    self.transfer_unload_thresholds_n[
                        self.transfer_unload_curriculum_level
                    ]
                    if self.transfer_unload_thresholds_n
                    else None
                ),
                "upright_cosines": list(self.transfer_upright_cosines),
                "active_upright_cosine": (
                    self.transfer_upright_cosines[
                        self.transfer_unload_curriculum_level
                    ]
                    if self.transfer_upright_cosines
                    else None
                ),
                "successes_at_active_level": self.transfer_unload_level_successes,
                "successes_by_level": list(
                    self.transfer_unload_successes_by_level
                ),
                "transitions": list(
                    self.transfer_unload_curriculum_transitions
                ),
            },
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
            "completed_target_transfers": self.completed_target_transfers,
            "transfer_post_hold_seconds": self.transfer_post_hold_seconds,
            "transfer_post_hold_steps": self.transfer_post_hold_steps,
            "completed_target_transfer_holds": (
                self.completed_target_transfer_holds
            ),
            "transfer_progress_reward": {
                "balance_target_error_progress_per_m": (
                    self.transfer_balance_progress_reward_per_m
                ),
                "support_margin_progress_per_m": (
                    self.transfer_support_margin_progress_reward_per_m
                ),
                "swing_load_reduction_per_n": (
                    self.transfer_swing_load_reduction_reward_per_n
                ),
                "maximum_progress_m_per_step": (
                    self.transfer_progress_reward_clip_m
                ),
                "maximum_load_progress_n_per_step": (
                    self.transfer_load_progress_reward_clip_n
                ),
                "cumulative_reward": (
                    self.cumulative_transfer_progress_reward
                ),
                "cumulative_balance_error_progress_m": (
                    self.cumulative_transfer_balance_error_progress_m
                ),
                "cumulative_support_margin_progress_m": (
                    self.cumulative_transfer_support_margin_progress_m
                ),
                "cumulative_swing_load_reduction_n": (
                    self.cumulative_transfer_swing_load_reduction_n
                ),
            },
            "maximum_target_transfer_balance_error_m": (
                self.maximum_target_transfer_balance_error_m
            ),
            "maximum_target_transfer_body_rate_rad_s": (
                self.maximum_target_transfer_body_rate_rad_s
            ),
            "minimum_target_transfer_swing_load_n": (
                None
                if not np.isfinite(self.minimum_target_transfer_swing_load_n)
                else self.minimum_target_transfer_swing_load_n
            ),
        }
