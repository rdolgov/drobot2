"""Pure reference arithmetic tests, authored but not run for the V44 change."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "isaac/rl/parallel_walking/generate_flat_contact_reference.py"
SPEC = importlib.util.spec_from_file_location("v44_reference_generator", PATH)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@pytest.mark.parametrize("speed,frequency", [(.03, .35), (.05, .45)])
def test_reference_has_flat_support_and_rate_feasible_corrective_band(speed, frequency):
    payload = generator.make_reference(speed, frequency)
    assert payload["sample_count"] == 512
    assert max(payload["generation"]["maximum_joint_target_velocity_deg_s"]) <= 90.0
    for i, row in enumerate(payload["joint_position_rad"]):
        for abd, hip, knee, offset, _sign in generator.LEGS:
            assert row[abd] == 0.0
            if (i / 512 + offset) % 1 < generator.DUTY:
                assert row[hip] + row[knee] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_swing_lifts_lowest_edge_not_only_center(sign):
    forward = sign * .08
    clearance = .010
    hip, knee = generator.foot_pose(forward, clearance, sign)
    down = generator.PROXIMAL_M * math.cos(hip) + generator.DISTAL_CONTACT_M * math.cos(hip + knee)
    edge_down = down + generator.SOLE_HALF_LENGTH_M * abs(math.sin(hip + knee))
    flat_down = math.sqrt(generator.PROXIMAL_M**2 - forward**2) + generator.DISTAL_CONTACT_M
    assert flat_down - edge_down == pytest.approx(clearance, abs=1e-8)


def test_quintic_has_zero_endpoint_acceleration():
    # Analytic derivatives of10u^3−15u^4+6u^5, not finite-difference simulation.
    for u in (0.0, 1.0):
        assert 30 * u**2 - 60 * u**3 + 30 * u**4 == 0.0
        assert 60 * u - 180 * u**2 + 120 * u**3 == 0.0
    assert generator.smoothstep(0.0) == 0.0
    assert generator.smoothstep(1.0) == 1.0


def test_existing_reference_is_not_overwritten(tmp_path, monkeypatch):
    old = tmp_path / "v44_flat_contact_low.json"
    old.write_text("existing baseline", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [str(PATH), "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        generator.main()
    assert old.read_text(encoding="utf-8") == "existing baseline"
    assert not (tmp_path / "v44_flat_contact_high.json").exists()
