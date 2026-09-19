"""Matched continuations: progress shaping and pre-contact terrain penalties."""
METRIC_COLUMNS = (
    "full_climb", "fall", "any_placement", "front_pair", "rear_pair", "all_first",
    "unique_feet", "supported_advance_m", "speed_world_x_m_s", "slip_m_s",
    "max_yaw_deg", "sideways_motion_fraction", "any_supported_lift",
    "valid_depth_fraction", "rate_limited_fraction", "episode_seconds",
    "peak_clearance_m", "selected_support_margin_m",
)
COMMON = dict(yaw_deg=90., yaw_weight=2., placement_weight=0., transfer_weight=0.,
              turn_rate_weight=.5, heading_shape="cosine", turn_rate_width_rad_s=.25)
STRATEGIES = {
    "continue_turn": dict(COMMON, potential=False, contact_penalties=False),
    "potential": dict(COMMON, potential=True, contact_penalties=False),
    "potential_edges": dict(COMMON, potential=True, contact_penalties=True),
}
