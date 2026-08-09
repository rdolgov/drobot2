from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAAC_DIR = PROJECT_ROOT / "simulation" / "isaac"
if str(ISAAC_DIR) not in sys.path:
    sys.path.insert(0, str(ISAAC_DIR))

from experiments.stair_feasibility._contract import (  # noqa: E402
    current_policy_front_lift_m,
    signed_support_margin_m,
    step_targets,
    target_limit_failures,
    trial_gate_failures,
    validate_config,
)


@pytest.fixture
def config() -> dict:
    path = (
        ISAAC_DIR
        / "experiments"
        / "stair_feasibility"
        / "real_stair_feasibility.yaml"
    )
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_real_stair_config_and_targets_fit_hard_limits(config: dict) -> None:
    experiment = validate_config(config)
    shifted_base_x = (
        float(experiment["reset_base_x_m"])
        - float(experiment["weight_shift"]["backward_m"])
    )
    for height in experiment["riser_heights_m"]:
        targets = step_targets(
            experiment,
            riser_height_m=float(height),
            shifted_base_x_m=shifted_base_x,
        )
        assert target_limit_failures(
            targets,
            margin_rad=float(
                experiment["acceptance"]["joint_limit_margin_rad"]
            ),
        ) == ()


def test_current_ppo_action_box_cannot_reach_real_riser(config: dict) -> None:
    experiment = validate_config(config)
    lift = current_policy_front_lift_m(experiment)

    assert lift == pytest.approx(0.0889461548)
    assert lift < min(experiment["riser_heights_m"])


def test_support_margin_is_positive_inside_and_negative_outside() -> None:
    triangle = ((0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1))

    assert signed_support_margin_m((-0.05, 0.0), triangle) > 0.0
    assert signed_support_margin_m((0.1, -0.1), triangle) < 0.0


def test_trial_gate_reports_physical_failures(config: dict) -> None:
    acceptance = config["experiment"]["acceptance"]
    metrics = {
        "edge_clearance_m": 0.0,
        "tread_contact_hold_s": 0.0,
        "support_contact_fraction": 0.5,
        "minimum_support_polygon_margin_m": -0.1,
        "landing_height_error_m": 0.1,
        "maximum_support_tip_slip_m": 0.1,
        "maximum_body_tilt_deg": 30.0,
        "maximum_base_drop_m": 0.1,
        "maximum_abs_joint_error_rad": 0.5,
        "pd_saturation_fraction": 1.0,
        "riser_strike": True,
        "nonfoot_step_collision": True,
        "tread_contact_achieved": False,
    }

    failures = trial_gate_failures(metrics, acceptance)

    assert "riser_strike=true" in failures
    assert "nonfoot_step_collision=true" in failures
    assert "tread_contact_achieved=false" in failures
    assert len(failures) == 13
