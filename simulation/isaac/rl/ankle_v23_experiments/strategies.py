"""Only these heading objectives and shaping weights vary between arms."""
STRATEGIES = {
    "free": dict(yaw_deg=0., yaw_weight=0., placement_weight=0., transfer_weight=0.),
    "oblique": dict(yaw_deg=45., yaw_weight=2., placement_weight=0., transfer_weight=0.),
    "sideways": dict(yaw_deg=90., yaw_weight=2., placement_weight=0., transfer_weight=0.),
    "place_first": dict(yaw_deg=0., yaw_weight=0., placement_weight=3., transfer_weight=0.),
    "place_transfer": dict(yaw_deg=0., yaw_weight=0., placement_weight=3., transfer_weight=2.),
    "oblique_place": dict(yaw_deg=45., yaw_weight=2., placement_weight=3., transfer_weight=2.),
}

METRIC_COLUMNS = (
    "full_climb", "fall", "any_placement", "front_pair", "rear_pair", "all_first",
    "unique_feet", "supported_advance_m", "speed_world_x_m_s", "slip_m_s",
    "max_yaw_deg", "sideways_motion_fraction", "any_supported_lift",
    "valid_depth_fraction", "rate_limited_fraction", "episode_seconds",
    "peak_clearance_m", "selected_support_margin_m",
)
