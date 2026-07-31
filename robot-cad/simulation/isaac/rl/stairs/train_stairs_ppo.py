"""Train a separate PPO policy for climbing the Drobot stair world."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import torch
import torch._dynamo  # noqa: F401
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _policy_transfer import (  # noqa: E402
    EXPANDABLE_INPUT_WEIGHTS,
    physical_action_output_ratios,
    transfer_policy_state,
)
from _run_support import (  # noqa: E402
    build_model_manifest,
    expected_ppo_algorithm_contract,
    file_hash_records,
    model_manifest_path,
    read_model_manifest,
    sha256_file,
    validate_ppo_algorithm_contract,
    write_model_manifest,
)
from _stair_rl_contract import (  # noqa: E402
    config_for_height_stage,
    progress_gate_failures,
    stair_observation_fields,
)

parser = argparse.ArgumentParser(
    description="Train the separate Drobot stair-climbing PPO policy."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_stairs_v1.yaml"),
)
parser.add_argument("--world", default=None)
parser.add_argument(
    "--height-stage",
    default=None,
    help="Apply one stair_height_stages entry declared by the config.",
)
parser.add_argument(
    "--fixed-active-steps",
    type=int,
    default=None,
    help="Pin training to one declared stair count instead of advancing mastery.",
)
parser.add_argument(
    "--output-dir",
    default="simulation/isaac/output/rl/ppo-stairs-v1",
)
parser.add_argument("--total-timesteps", type=int, default=None)
parser.add_argument("--seed", type=int, default=142)
parser.add_argument("--device", default="cpu")
parser.add_argument(
    "--resume",
    default=None,
    help="Resume a stairs PPO model with a matching .contract.json manifest.",
)
parser.add_argument(
    "--initialize-from-flat",
    default=None,
    help=(
        "Initialize a new 57-input stairs policy from a 48-input flat-walk "
        "policy. Policy weights transfer; optimizer state does not."
    ),
)
parser.add_argument(
    "--initialize-from-stairs",
    default=None,
    help=(
        "Initialize a new same-shape stair policy from another stair model. "
        "Policy parameters transfer; optimizer state does not."
    ),
)
parser.add_argument(
    "--allow-unverified-resume",
    action="store_true",
    help="Allow a deliberate resume from a stairs model without its manifest.",
)
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Run a 512-step end-to-end pipeline check; not convergence evidence.",
)
parser.add_argument(
    "--initialize-only",
    action="store_true",
    help=(
        "Save the initialized/transferred policy and a current manifest "
        "without collecting a PPO rollout."
    ),
)
parser.add_argument("--gui", action="store_true")
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


initialization_options = (
    args.resume,
    args.initialize_from_flat,
    args.initialize_from_stairs,
)
if sum(value is not None for value in initialization_options) > 1:
    parser.error(
        "--resume, --initialize-from-flat, and --initialize-from-stairs "
        "are mutually exclusive"
    )
if args.total_timesteps is not None and args.total_timesteps <= 0:
    parser.error("--total-timesteps must be positive")
if args.smoke_test and args.initialize_only:
    parser.error("--smoke-test and --initialize-only are mutually exclusive")
config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if int(config.get("schema_version", 0)) != 1:
    parser.error(f"Unsupported stairs config schema: {config.get('schema_version')}")
try:
    config = config_for_height_stage(config, args.height_stage)
except ValueError as exc:
    parser.error(str(exc))
task_config = dict(config["task"])
if args.fixed_active_steps is not None:
    maximum_steps = int(task_config["staircase"]["step_count"])
    if args.fixed_active_steps < 1 or args.fixed_active_steps > maximum_steps:
        parser.error(
            f"--fixed-active-steps must be within 1..{maximum_steps}"
        )
    curriculum = dict(task_config["curriculum"])
    curriculum["levels"] = [
        {"start_fraction": 0.0, "active_steps": args.fixed_active_steps}
    ]
    task_config["curriculum"] = curriculum
ppo_config = dict(config["ppo"])
residual_policy_config = dict(task_config.get("residual_policy", {}))
if bool(residual_policy_config.get("enabled", False)) and args.initialize_from_flat:
    parser.error(
        "Residual stair policies use their configured frozen base model and "
        "cannot also use --initialize-from-flat"
    )
policy_observation_size = len(
    stair_observation_fields(
        task_config["staircase"]["terrain_sample_offsets_m"],
        include_navigation_observation=bool(
            task_config.get("include_navigation_observation", False)
        ),
        include_foot_progress_observation=bool(
            task_config.get("include_foot_progress_observation", False)
        ),
    )
)
world_path = _resolve_project_path(args.world or task_config["world"])
world_dependency_paths = tuple(
    _resolve_project_path(value)
    for value in task_config.get("world_dependencies", ())
)
output_dir = _resolve_project_path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / "training_report.json"

if args.smoke_test:
    total_timesteps = args.total_timesteps or 512
    rollout_steps = min(128, total_timesteps)
    batch_size = min(64, rollout_steps)
    epochs = min(2, int(ppo_config["epochs"]))
    checkpoint_frequency = max(128, rollout_steps)
    curriculum_total_timesteps = total_timesteps
else:
    total_timesteps = args.total_timesteps or int(ppo_config["total_timesteps"])
    rollout_steps = int(ppo_config["rollout_steps"])
    batch_size = int(ppo_config["batch_size"])
    epochs = int(ppo_config["epochs"])
    checkpoint_frequency = int(ppo_config["checkpoint_frequency_steps"])
    curriculum_total_timesteps = int(ppo_config["total_timesteps"])
if rollout_steps < 2 or batch_size < 2 or rollout_steps % batch_size:
    parser.error("PPO rollout_steps must be divisible by batch_size and both >= 2")
training_mode = "smoke" if args.smoke_test else "full"
algorithm_contract = expected_ppo_algorithm_contract(
    ppo_config,
    training_mode=training_mode,
    rollout_steps=rollout_steps,
    batch_size=batch_size,
    epochs=epochs,
    observation_size=policy_observation_size,
    action_size=12,
)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "width": 1280,
        "height": 720,
    }
)

import stable_baselines3  # noqa: E402
from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from _run_support import validate_model_manifest  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402


class TrainingControlCallback(BaseCallback):
    def __init__(
        self,
        *,
        task_config: dict[str, object],
        total_for_curriculum: int,
        report_path: Path,
    ) -> None:
        super().__init__(verbose=0)
        self.task_config = task_config
        self.total_for_curriculum = max(1, int(total_for_curriculum))
        self.curriculum_config = dict(task_config["curriculum"])
        self.curriculum_mode = str(
            self.curriculum_config.get("mode", "timesteps")
        )
        if self.curriculum_mode not in {"timesteps", "mastery"}:
            raise ValueError(
                f"Unsupported curriculum mode: {self.curriculum_mode}"
            )
        self.mastery_window = int(
            self.curriculum_config.get("mastery_window_episodes", 40)
        )
        self.mastery_minimum = int(
            self.curriculum_config.get("mastery_minimum_episodes", 20)
        )
        self.mastery_success_rate = float(
            self.curriculum_config.get("mastery_success_rate", 0.70)
        )
        self.level_outcomes: deque[bool] = deque(maxlen=self.mastery_window)
        self.watchdog = dict(task_config.get("progress_watchdog", {}))
        self.watchdog_enabled = bool(self.watchdog.get("enabled", False))
        self.report_path = report_path
        self.report_interval_steps = int(
            self.watchdog.get("report_interval_steps", 10000)
        )
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.first_step_climb_episodes = 0
        self.maximum_step_reached = 0
        self.maximum_base_elevation_gain_m = 0.0
        self.last_progress_step = 0
        self.last_report_step = -self.report_interval_steps
        self.initial_gate_evaluated = False
        self.initial_gate_passed = False
        self.aborted = False
        self.abort_reason: str | None = None

    def _raw_env(self) -> QuadrupedStairsEnv:
        return self.training_env.envs[0].unwrapped

    def snapshot(self) -> dict[str, object]:
        climb_rate = (
            self.first_step_climb_episodes / self.completed_episodes
            if self.completed_episodes
            else 0.0
        )
        raw = self._raw_env()
        return {
            "status": "ABORTED" if self.aborted else "RUNNING",
            "timesteps": int(self.num_timesteps),
            "curriculum_mode": self.curriculum_mode,
            "active_step_count": int(raw.active_step_count),
            "pending_active_step_count": int(raw.pending_active_step_count),
            "completed_episodes": self.completed_episodes,
            "successful_episodes": self.successful_episodes,
            "first_step_climb_episodes": self.first_step_climb_episodes,
            "first_step_climb_rate": climb_rate,
            "maximum_step_reached": self.maximum_step_reached,
            "maximum_base_elevation_gain_m": (
                self.maximum_base_elevation_gain_m
            ),
            "last_progress_step": self.last_progress_step,
            "initial_gate_evaluated": self.initial_gate_evaluated,
            "initial_gate_passed": self.initial_gate_passed,
            "abort_reason": self.abort_reason,
            "watchdog_config": self.watchdog,
            "curriculum_transitions": list(raw.curriculum_transitions),
        }

    def _write_report(self) -> None:
        climb_rate = (
            self.first_step_climb_episodes / self.completed_episodes
            if self.completed_episodes
            else 0.0
        )
        success_rate = (
            self.successful_episodes / self.completed_episodes
            if self.completed_episodes
            else 0.0
        )
        self.logger.record("stair/completed_episodes", self.completed_episodes)
        self.logger.record("stair/success_rate", success_rate)
        self.logger.record("stair/first_step_climb_rate", climb_rate)
        self.logger.record("stair/maximum_step_reached", self.maximum_step_reached)
        self.logger.record(
            "stair/maximum_base_elevation_gain_m",
            self.maximum_base_elevation_gain_m,
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as stream:
            json.dump(self.snapshot(), stream, indent=2)
            stream.write("\n")

    def _record_episode(self, metrics: dict[str, object]) -> None:
        self.completed_episodes += 1
        active_steps = int(metrics["active_step_count"])
        succeeded = bool(metrics["stairs_completed"])
        highest_step = int(metrics["highest_step_reached"])
        elevation_gain = float(metrics["maximum_base_elevation_gain_m"])
        minimum_elevation = min(
            float(self.watchdog.get("minimum_base_elevation_gain_m", 0.02)),
            (
                active_steps
                * float(self.task_config["staircase"]["rise_m"])
                * 0.5
            ),
        )
        physically_reached_first_step = (
            highest_step >= 1 and elevation_gain >= minimum_elevation
        )
        if succeeded:
            self.successful_episodes += 1
        if physically_reached_first_step:
            self.first_step_climb_episodes += 1
        if highest_step > self.maximum_step_reached or succeeded:
            self.last_progress_step = int(self.num_timesteps)
        self.maximum_step_reached = max(self.maximum_step_reached, highest_step)
        self.maximum_base_elevation_gain_m = max(
            self.maximum_base_elevation_gain_m,
            elevation_gain,
        )

        raw = self._raw_env()
        if active_steps == raw.active_step_count:
            self.level_outcomes.append(succeeded)
        if self.curriculum_mode != "mastery":
            return
        maximum_steps = int(self.task_config["staircase"]["step_count"])
        if (
            raw.active_step_count < maximum_steps
            and len(self.level_outcomes) >= self.mastery_minimum
        ):
            success_rate = sum(self.level_outcomes) / len(self.level_outcomes)
            if success_rate >= self.mastery_success_rate:
                raw.set_training_level(
                    raw.active_step_count + 1,
                    reason="mastery",
                    evidence={
                        "episodes": len(self.level_outcomes),
                        "success_rate": success_rate,
                        "timesteps": int(self.num_timesteps),
                    },
                )
                self.level_outcomes.clear()
                self.last_progress_step = int(self.num_timesteps)

    def _check_watchdog(self) -> None:
        if not self.watchdog_enabled or self.aborted:
            return
        gate_steps = int(self.watchdog["initial_gate_steps"])
        if not self.initial_gate_evaluated and self.num_timesteps >= gate_steps:
            self.initial_gate_evaluated = True
            minimum_episodes = int(self.watchdog["minimum_completed_episodes"])
            minimum_climbs = int(
                self.watchdog["minimum_first_step_climb_episodes"]
            )
            minimum_rate = float(
                self.watchdog["minimum_first_step_climb_rate"]
            )
            failures = progress_gate_failures(
                completed_episodes=self.completed_episodes,
                first_step_climb_episodes=self.first_step_climb_episodes,
                minimum_completed_episodes=minimum_episodes,
                minimum_first_step_climb_episodes=minimum_climbs,
                minimum_first_step_climb_rate=minimum_rate,
            )
            if failures:
                self.aborted = True
                self.abort_reason = (
                    "No measurable stair/elevation progress at the "
                    f"{gate_steps}-step gate: " + "; ".join(failures)
                )
            else:
                self.initial_gate_passed = True

        if self.initial_gate_passed:
            stagnation_steps = int(self.watchdog["stagnation_abort_steps"])
            if self.num_timesteps - self.last_progress_step >= stagnation_steps:
                self.aborted = True
                self.abort_reason = (
                    "No new stair level or successful episode for "
                    f"{stagnation_steps} training steps"
                )

    def _on_step(self) -> bool:
        raw = self._raw_env()
        if self.curriculum_mode == "timesteps":
            progress = min(
                1.0,
                self.num_timesteps / self.total_for_curriculum,
            )
            raw.set_training_progress(progress)
        for info in self.locals.get("infos", ()):
            metrics = info.get("episode_metrics")
            if isinstance(metrics, dict):
                self._record_episode(metrics)
        self._check_watchdog()
        if (
            self.aborted
            or self.num_timesteps - self.last_report_step
            >= self.report_interval_steps
        ):
            self._write_report()
            self.last_report_step = int(self.num_timesteps)
        return not self.aborted

    def _on_training_end(self) -> None:
        self._write_report()


class CheckpointManifestCallback(BaseCallback):
    def __init__(
        self,
        *,
        save_frequency: int,
        save_path: Path,
        config_path: Path,
        world_path: Path,
        world_dependencies: tuple[Path, ...],
        raw_env: QuadrupedStairsEnv,
        algorithm_contract: dict[str, object],
        seed: int,
        transferred_from: Path | None,
        resumed_from: Path | None,
        inherited_transfer: dict[str, object] | None,
    ) -> None:
        super().__init__(verbose=0)
        self.save_frequency = int(save_frequency)
        self.save_path = save_path
        self.config_path = config_path
        self.world_path = world_path
        self.world_dependencies = world_dependencies
        self.raw_env = raw_env
        self.algorithm_contract = algorithm_contract
        self.seed = seed
        self.transferred_from = transferred_from
        self.resumed_from = resumed_from
        self.inherited_transfer = inherited_transfer

    def _on_step(self) -> bool:
        if self.n_calls % self.save_frequency:
            return True
        model_path = (
            self.save_path
            / f"drobot_stairs_ppo_{self.num_timesteps}_steps.zip"
        )
        if model_path.is_file():
            manifest = build_model_manifest(
                model_path=model_path,
                config_path=self.config_path,
                world_path=self.world_path,
                world_dependencies=self.world_dependencies,
                environment_contract=self.raw_env.contract,
                algorithm_contract=self.algorithm_contract,
                training_seed=self.seed,
                transferred_from=self.transferred_from,
                resumed_from=self.resumed_from,
                inherited_transfer=self.inherited_transfer,
            )
            write_model_manifest(model_manifest_path(model_path), manifest)
        return True


report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "config": str(config_path),
    "config_sha256": sha256_file(config_path),
    "world": str(world_path),
    "world_sha256": sha256_file(world_path) if world_path.is_file() else None,
    "world_dependencies": [
        {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for path in world_dependency_paths
    ],
    "output_dir": str(output_dir),
    "smoke_test": args.smoke_test,
    "initialize_only": args.initialize_only,
    "training_mode": training_mode,
    "algorithm_contract_requested": algorithm_contract,
    "requested_total_timesteps": total_timesteps,
    "requested_seed": args.seed,
    "height_stage": args.height_stage,
    "fixed_active_steps": args.fixed_active_steps,
    "device_request": args.device,
    "isaac_sim_version": "6.0.1",
    "torch_version": torch.__version__,
    "stable_baselines3_version": stable_baselines3.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    ),
}
exit_code = 1
raw_env: QuadrupedStairsEnv | None = None
monitored_env = None
start_time = time.perf_counter()

try:
    if not world_path.is_file():
        raise FileNotFoundError(
            f"{world_path}. Generate it with create_stairs_world.py first."
        )
    file_hash_records(world_dependency_paths)
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human" if args.gui else None,
    )
    monitored_env = Monitor(raw_env, filename=str(output_dir / "monitor.csv"))
    policy_kwargs = {
        "activation_fn": torch.nn.ELU,
        "net_arch": list(ppo_config["policy_hidden_layers"]),
    }
    if "initial_log_std" in ppo_config:
        policy_kwargs["log_std_init"] = float(ppo_config["initial_log_std"])
    transferred_from: Path | None = None
    resume_path: Path | None = None
    inherited_transfer: dict[str, object] | None = None
    reset_num_timesteps = True
    if args.resume:
        resume_path = _resolve_project_path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_verification = validate_model_manifest(
            model_path=resume_path,
            config_path=config_path,
            world_path=world_path,
            world_dependencies=world_dependency_paths,
            environment_contract=raw_env.contract,
            allow_unverified=args.allow_unverified_resume,
            expected_algorithm_contract=algorithm_contract,
        )
        if resume_verification["status"] == "PASS":
            resume_manifest = read_model_manifest(resume_path)
            saved_transfer = resume_manifest.get("transferred_from")
            if isinstance(saved_transfer, dict):
                inherited_transfer = dict(saved_transfer)
        model = PPO.load(
            str(resume_path),
            env=monitored_env,
            device=args.device,
        )
        algorithm_verification = validate_ppo_algorithm_contract(
            model,
            algorithm_contract,
        )
        model.set_random_seed(args.seed)
        reset_num_timesteps = False
        report["resume_checkpoint"] = str(resume_path)
        report["resume_contract_verification"] = resume_verification
    else:
        model = PPO(
            "MlpPolicy",
            monitored_env,
            learning_rate=float(ppo_config["learning_rate"]),
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=epochs,
            gamma=float(ppo_config["gamma"]),
            gae_lambda=float(ppo_config["gae_lambda"]),
            clip_range=float(ppo_config["clip_range"]),
            ent_coef=float(ppo_config["entropy_coefficient"]),
            vf_coef=float(ppo_config["value_coefficient"]),
            max_grad_norm=float(ppo_config["max_gradient_norm"]),
            target_kl=(
                float(ppo_config["target_kl"])
                if ppo_config.get("target_kl") is not None
                else None
            ),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(output_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        if bool(ppo_config.get("zero_action_mean_init", False)):
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
            report["zero_action_mean_initialization"] = {
                "weight": "all_zero",
                "bias": "all_zero",
            }
        if args.initialize_from_stairs:
            transferred_from = _resolve_project_path(
                args.initialize_from_stairs
            )
            if not transferred_from.is_file():
                raise FileNotFoundError(transferred_from)
            source_model = PPO.load(str(transferred_from), device=args.device)
            source_observation_size = int(source_model.observation_space.shape[0])
            target_observation_size = int(model.observation_space.shape[0])
            if tuple(source_model.action_space.shape) != tuple(
                model.action_space.shape
            ):
                raise RuntimeError("Stair source and target action shapes differ")
            source_state = source_model.policy.state_dict()
            target_state = model.policy.state_dict()
            if source_observation_size == target_observation_size:
                mismatched = [
                    name
                    for name, tensor in target_state.items()
                    if name not in source_state
                    or tuple(source_state[name].shape) != tuple(tensor.shape)
                ]
                if set(source_state) != set(target_state) or mismatched:
                    raise RuntimeError(
                        "Stair policy parameter contracts differ: "
                        f"mismatched={mismatched}"
                    )
                transferred_state = source_state
                stair_transfer_report = {
                    "mode": "exact_same_shape",
                    "expanded_inputs": [],
                }
            elif source_observation_size < target_observation_size:
                transferred_state, stair_transfer_report = transfer_policy_state(
                    source_state,
                    target_state,
                    source_observation_size=source_observation_size,
                )
                stair_transfer_report["mode"] = "expanded_observation"
                if stair_transfer_report["skipped"]:
                    raise RuntimeError(
                        "Expanded stair policy transfer skipped parameters: "
                        f"{stair_transfer_report['skipped']}"
                    )
            else:
                raise RuntimeError(
                    "Stair source observation is larger than the target: "
                    f"{source_observation_size}>{target_observation_size}"
                )
            model.policy.load_state_dict(transferred_state, strict=True)
            report["stair_policy_transfer"] = {
                "source_model": str(transferred_from),
                "source_model_sha256": sha256_file(transferred_from),
                "parameter_count": len(source_state),
                "optimizer_transferred": False,
                "source_observation_size": source_observation_size,
                "target_observation_size": target_observation_size,
                **stair_transfer_report,
            }
            del source_model
        if args.initialize_from_flat:
            transferred_from = _resolve_project_path(args.initialize_from_flat)
            if not transferred_from.is_file():
                raise FileNotFoundError(transferred_from)
            source_model = PPO.load(str(transferred_from), device=args.device)
            source_observation_shape = tuple(source_model.observation_space.shape)
            source_action_shape = tuple(source_model.action_space.shape)
            source_activation = getattr(
                source_model.policy.activation_fn,
                "__name__",
                str(source_model.policy.activation_fn),
            )
            if source_observation_shape != (48,) or source_action_shape != (12,):
                raise RuntimeError(
                    "Flat source must have observation/action shapes (48,)/(12,), "
                    f"got {source_observation_shape}/{source_action_shape}"
                )
            if source_activation != "ELU":
                raise RuntimeError(
                    f"Flat source activation must be ELU, got {source_activation}"
                )
            transfer_config = dict(
                task_config.get("flat_policy_transfer", {})
            )
            output_ratios = None
            if bool(
                transfer_config.get("preserve_physical_action_mean", False)
            ):
                output_ratios = physical_action_output_ratios(
                    raw_env.dof_names,
                    dict(transfer_config["source_action_scale_rad"]),
                    dict(task_config["action_scale_rad"]),
                )
            transferred_state, transfer_report = transfer_policy_state(
                source_model.policy.state_dict(),
                model.policy.state_dict(),
                source_observation_size=48,
                action_output_ratios=output_ratios,
            )
            expected_exact_count = len(model.policy.state_dict()) - len(
                EXPANDABLE_INPUT_WEIGHTS
            )
            if (
                transfer_report["expanded_inputs"]
                != list(EXPANDABLE_INPUT_WEIGHTS)
                or transfer_report["copied_exact_count"] != expected_exact_count
                or transfer_report["skipped"]
            ):
                raise RuntimeError(
                    f"Flat policy transfer was not exact: {transfer_report}"
                )
            model.policy.load_state_dict(transferred_state, strict=True)
            if "initial_log_std" in ppo_config:
                model.policy.log_std.data.fill_(
                    float(ppo_config["initial_log_std"])
                )
                transfer_report["log_std_overridden_after_transfer"] = float(
                    ppo_config["initial_log_std"]
                )
            report["flat_policy_transfer"] = {
                "source_model": str(transferred_from),
                "source_model_sha256": sha256_file(transferred_from),
                **transfer_report,
            }
            del source_model
        algorithm_verification = validate_ppo_algorithm_contract(
            model,
            algorithm_contract,
        )

    actual_training_seed = int(model.seed) if model.seed is not None else args.seed
    report["training_seed"] = actual_training_seed
    report["ppo_algorithm_verification"] = algorithm_verification
    initial_curriculum_progress = min(
        1.0,
        float(model.num_timesteps) / max(1, curriculum_total_timesteps),
    )
    if str(task_config["curriculum"].get("mode", "timesteps")) == "timesteps":
        raw_env.set_training_progress(initial_curriculum_progress)
    training_control_callback: TrainingControlCallback | None = None
    if not args.initialize_only:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_frequency,
            save_path=str(checkpoint_dir),
            name_prefix="drobot_stairs_ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        training_control_callback = TrainingControlCallback(
            task_config=task_config,
            total_for_curriculum=curriculum_total_timesteps,
            report_path=output_dir / "progress_watchdog.json",
        )
        callbacks = CallbackList(
            [
                checkpoint_callback,
                CheckpointManifestCallback(
                    save_frequency=checkpoint_frequency,
                    save_path=checkpoint_dir,
                    config_path=config_path,
                    world_path=world_path,
                    world_dependencies=world_dependency_paths,
                    raw_env=raw_env,
                    algorithm_contract=algorithm_contract,
                    seed=actual_training_seed,
                    transferred_from=transferred_from,
                    resumed_from=resume_path,
                    inherited_transfer=inherited_transfer,
                ),
                training_control_callback,
            ]
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
    aborted_no_progress = bool(
        training_control_callback is not None
        and training_control_callback.aborted
    )
    final_model_base = output_dir / (
        "drobot_stairs_ppo_aborted"
        if aborted_no_progress
        else (
            "drobot_stairs_ppo_initialized"
            if args.initialize_only
            else "drobot_stairs_ppo_final"
        )
    )
    model.save(str(final_model_base))
    final_model_path = final_model_base.with_suffix(".zip")
    if not final_model_path.is_file():
        raise RuntimeError(f"Stable-Baselines3 did not save {final_model_path}")
    final_manifest = build_model_manifest(
        model_path=final_model_path,
        config_path=config_path,
        world_path=world_path,
        world_dependencies=world_dependency_paths,
        environment_contract=raw_env.contract,
        algorithm_contract=algorithm_contract,
        training_seed=actual_training_seed,
        transferred_from=transferred_from,
        resumed_from=resume_path,
        inherited_transfer=inherited_transfer,
    )
    final_manifest_path = model_manifest_path(final_model_path)
    write_model_manifest(final_manifest_path, final_manifest)
    report.update(
        {
            "status": (
                "ABORTED_NO_PROGRESS" if aborted_no_progress else "PASS"
            ),
            "actual_total_timesteps": int(model.num_timesteps),
            "model": str(final_model_path),
            "model_bytes": final_model_path.stat().st_size,
            "model_manifest": str(final_manifest_path),
            "model_parameter_count": int(
                sum(parameter.numel() for parameter in model.policy.parameters())
            ),
            "ppo": {
                **algorithm_contract,
            },
            "initial_curriculum_progress": initial_curriculum_progress,
            "environment_contract": raw_env.contract,
            "curriculum_transitions": raw_env.curriculum_transitions,
            "progress_watchdog": (
                training_control_callback.snapshot()
                if training_control_callback is not None
                else {
                    "status": "NOT_RUN",
                    "reason": "initialize_only",
                }
            ),
            "recent_completed_episodes": raw_env.completed_episode_metrics,
            "elapsed_seconds": time.perf_counter() - start_time,
            "scope": (
                "Policy initialization without PPO updates"
                if args.initialize_only
                else (
                    "Pipeline validation only"
                    if args.smoke_test
                    else (
                        "Automatically aborted stair PPO training"
                        if aborted_no_progress
                        else "Single-environment stair PPO training"
                    )
                )
            ),
            "terrain_input": (
                "Analytic height profile used for simulation learning; not "
                "yet a hardware sensor pipeline."
            ),
        }
    )
    exit_code = 3 if aborted_no_progress else 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
    report["elapsed_seconds"] = time.perf_counter() - start_time
finally:
    if monitored_env is not None:
        monitored_env.close()
    elif raw_env is not None:
        raw_env.close()
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_STAIRS_TRAIN_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
