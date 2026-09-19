"""Same 90-degree objective, with a less narrow heading reward."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Keep the stdlib launcher independent of Isaac/gym package imports.
METRIC_COLUMNS = (
    "full_climb", "fall", "any_placement", "front_pair", "rear_pair", "all_first",
    "unique_feet", "supported_advance_m", "speed_world_x_m_s", "slip_m_s",
    "max_yaw_deg", "sideways_motion_fraction", "any_supported_lift",
    "valid_depth_fraction", "rate_limited_fraction", "episode_seconds",
    "peak_clearance_m", "selected_support_margin_m",
)
STRATEGIES = {"sideways_dense": {
    "yaw_deg": 90.0, "yaw_weight": 2.0, "placement_weight": 0.0,
    "transfer_weight": 0.0, "turn_rate_weight": 0.5,
    "heading_shape": "cosine", "turn_rate_width_rad_s": 0.25,
}}
