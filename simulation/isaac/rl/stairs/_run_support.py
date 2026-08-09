"""Shared provenance helpers for stairs train/evaluate/record entry points."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

MANIFEST_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest_path(model_path: Path) -> Path:
    return Path(str(model_path) + ".contract.json")


def file_hash_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"World dependency is missing: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    return records


def expected_ppo_algorithm_contract(
    ppo_config: Mapping[str, object],
    *,
    training_mode: str,
    rollout_steps: int,
    batch_size: int,
    epochs: int,
    observation_size: int,
    action_size: int,
) -> dict[str, object]:
    if training_mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported training mode: {training_mode}")
    contract = {
        "algorithm": "stable_baselines3.PPO",
        "policy_class": "ActorCriticPolicy",
        "training_mode": training_mode,
        "observation_shape": [int(observation_size)],
        "action_shape": [int(action_size)],
        "policy_hidden_layers": [
            int(value) for value in ppo_config["policy_hidden_layers"]
        ],
        "activation": "ELU",
        "rollout_steps": int(rollout_steps),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": round(float(ppo_config["learning_rate"]), 12),
        "gamma": round(float(ppo_config["gamma"]), 12),
        "gae_lambda": round(float(ppo_config["gae_lambda"]), 12),
        "clip_range": round(float(ppo_config["clip_range"]), 12),
        "entropy_coefficient": round(
            float(ppo_config["entropy_coefficient"]),
            12,
        ),
        "value_coefficient": round(float(ppo_config["value_coefficient"]), 12),
        "maximum_gradient_norm": round(
            float(ppo_config["max_gradient_norm"]),
            12,
        ),
        "normalize_advantage": True,
    }
    if ppo_config.get("target_kl") is not None:
        contract["target_kl"] = round(float(ppo_config["target_kl"]), 12)
    return contract


def describe_ppo_model(model, *, training_mode: str) -> dict[str, object]:
    policy_network = model.policy.mlp_extractor.policy_net
    hidden_layers = [
        int(layer.out_features)
        for layer in policy_network
        if layer.__class__.__name__ == "Linear"
    ]
    activation_fn = getattr(model.policy, "activation_fn", None)
    activation_name = getattr(activation_fn, "__name__", str(activation_fn))
    contract = {
        "algorithm": f"{model.__class__.__module__.split('.')[0]}.{model.__class__.__name__}",
        "policy_class": model.policy.__class__.__name__,
        "training_mode": training_mode,
        "observation_shape": [int(value) for value in model.observation_space.shape],
        "action_shape": [int(value) for value in model.action_space.shape],
        "policy_hidden_layers": hidden_layers,
        "activation": activation_name,
        "rollout_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "epochs": int(model.n_epochs),
        "learning_rate": round(float(model.lr_schedule(1.0)), 12),
        "gamma": round(float(model.gamma), 12),
        "gae_lambda": round(float(model.gae_lambda), 12),
        "clip_range": round(float(model.clip_range(1.0)), 12),
        "entropy_coefficient": round(float(model.ent_coef), 12),
        "value_coefficient": round(float(model.vf_coef), 12),
        "maximum_gradient_norm": round(float(model.max_grad_norm), 12),
        "normalize_advantage": bool(model.normalize_advantage),
    }
    if model.target_kl is not None:
        contract["target_kl"] = round(float(model.target_kl), 12)
    return contract


def validate_ppo_algorithm_contract(
    model,
    expected_contract: Mapping[str, object],
) -> dict[str, object]:
    expected = dict(expected_contract)
    training_mode = str(expected.get("training_mode", ""))
    actual = describe_ppo_model(model, training_mode=training_mode)
    if actual != expected:
        mismatches = {
            key: {
                "saved_or_expected": expected.get(key),
                "loaded_model": actual.get(key),
            }
            for key in sorted(set(expected) | set(actual))
            if expected.get(key) != actual.get(key)
        }
        raise RuntimeError(f"Loaded PPO algorithm contract differs: {mismatches}")
    return {
        "status": "PASS",
        "verified_fields": list(expected),
        "contract": actual,
    }


def read_model_manifest(model_path: Path) -> dict[str, object]:
    path = model_manifest_path(model_path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_model_manifest(
    *,
    model_path: Path,
    config_path: Path,
    world_path: Path,
    world_dependencies: Iterable[Path],
    environment_contract: dict[str, object],
    algorithm_contract: dict[str, object],
    training_seed: int,
    transferred_from: Path | None,
    resumed_from: Path | None = None,
    inherited_transfer: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if inherited_transfer is not None:
        transfer_record: dict[str, object] | None = dict(inherited_transfer)
    elif transferred_from is not None:
        transfer_record = {
            "model": str(transferred_from),
            "model_sha256": sha256_file(transferred_from),
            "policy_parameters_only": True,
            "optimizer_state": False,
        }
    else:
        transfer_record = None
    resume_record = None
    if resumed_from is not None:
        resume_manifest_path = model_manifest_path(resumed_from)
        resume_record = {
            "model": str(resumed_from),
            "model_sha256": sha256_file(resumed_from),
            "manifest": str(resume_manifest_path),
            "manifest_sha256": (
                sha256_file(resume_manifest_path)
                if resume_manifest_path.is_file()
                else None
            ),
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task_id": environment_contract["task_id"],
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "world": str(world_path),
        "world_sha256": sha256_file(world_path),
        "world_dependencies": file_hash_records(world_dependencies),
        "training_seed": int(training_seed),
        "environment_contract": environment_contract,
        "algorithm_contract": algorithm_contract,
        "transferred_from": transfer_record,
        "resumed_from": resume_record,
    }


def write_model_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def validate_model_manifest(
    *,
    model_path: Path,
    config_path: Path,
    world_path: Path,
    world_dependencies: Iterable[Path],
    environment_contract: dict[str, object],
    allow_unverified: bool,
    expected_algorithm_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    path = model_manifest_path(model_path)
    if not path.is_file():
        if allow_unverified:
            return {
                "status": "SKIPPED",
                "reason": "manifest_missing_and_override_enabled",
                "manifest": str(path),
            }
        raise FileNotFoundError(
            f"Model contract manifest is missing: {path}. "
            "Use --allow-unverified-model only for deliberate recovery."
        )
    manifest = read_model_manifest(model_path)
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task_id": environment_contract["task_id"],
        "model_sha256": sha256_file(model_path),
        "config_sha256": sha256_file(config_path),
        "world_sha256": sha256_file(world_path),
        "world_dependencies": file_hash_records(world_dependencies),
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        if allow_unverified:
            return {
                "status": "SKIPPED",
                "reason": "manifest_mismatch_and_override_enabled",
                "manifest": str(path),
                "actual": actual,
                "expected": expected,
                "algorithm_contract": manifest.get("algorithm_contract"),
            }
        raise RuntimeError(
            f"Model contract mismatch: actual={actual}, expected={expected}"
        )
    saved_contract = dict(manifest.get("environment_contract", {}))
    semantic_fields = (
        "dof_names",
        "observation_fields",
        "observation_size",
        "action_size",
        "physics_steps_per_control",
        "staircase",
    )
    mismatches = {
        field: {
            "saved": saved_contract.get(field),
            "runtime": environment_contract.get(field),
        }
        for field in semantic_fields
        if saved_contract.get(field) != environment_contract.get(field)
    }
    if mismatches:
        if allow_unverified:
            return {
                "status": "SKIPPED",
                "reason": "environment_mismatch_and_override_enabled",
                "manifest": str(path),
                "mismatches": mismatches,
                "algorithm_contract": manifest.get("algorithm_contract"),
            }
        raise RuntimeError(f"Saved/runtime environment contract differs: {mismatches}")
    saved_algorithm_contract = dict(manifest.get("algorithm_contract", {}))
    if not saved_algorithm_contract:
        raise RuntimeError("Model manifest has no PPO algorithm contract")
    if (
        expected_algorithm_contract is not None
        and saved_algorithm_contract != dict(expected_algorithm_contract)
    ):
        raise RuntimeError(
            "Saved/requested PPO algorithm contract differs: "
            f"saved={saved_algorithm_contract}, "
            f"requested={dict(expected_algorithm_contract)}"
        )
    return {
        "status": "PASS",
        "manifest": str(path),
        "verified_fields": list(expected) + list(semantic_fields),
        "algorithm_contract": saved_algorithm_contract,
        "training_seed": manifest.get("training_seed"),
        "transferred_from": manifest.get("transferred_from"),
        "resumed_from": manifest.get("resumed_from"),
    }
