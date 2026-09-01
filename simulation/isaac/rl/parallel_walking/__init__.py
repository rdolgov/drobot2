"""Command-conditioned, vectorized Drobot walking task registration."""

import gymnasium as gym

from . import agents


def _register(task_id: str, env_cfg: str, runner_cfg: str) -> None:
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.commanded_walking_env:DrobotCommandedWalkingEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.commanded_walking_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
        },
    )


_register(
    "Drobot-Commanded-Walk-Forward-Direct",
    "DrobotCommandedWalkingForwardEnvCfg",
    "DrobotCommandedWalkingForwardPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Directional-Direct",
    "DrobotCommandedWalkingDirectionalEnvCfg",
    "DrobotCommandedWalkingDirectionalPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Smooth-Payload-Direct",
    "DrobotCommandedWalkingSmoothPayloadEnvCfg",
    "DrobotCommandedWalkingSmoothPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Low-Speed-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingLowSpeedExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingLowSpeedExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Low-Speed-Crawl-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingLowSpeedCrawlExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingLowSpeedCrawlExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Higher-Speed-Straight-Crawl-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingHigherSpeedStraightCrawlExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingHigherSpeedStraightCrawlExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Padded-Feet-Forward-Bias-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingPaddedFeetForwardBiasExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingPaddedFeetForwardBiasExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Robust-Straight-Low-Stance-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingRobustStraightLowStanceExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingRobustStraightLowStanceExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Balanced-Four-Leg-Straight-Crawl-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingBalancedFourLegStraightCrawlExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingBalancedFourLegStraightCrawlExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Adaptive-Asymmetric-Four-Leg-Straight-Crawl-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingAdaptiveAsymmetricFourLegStraightCrawlExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingAdaptiveAsymmetricFourLegStraightCrawlExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Forward-Biased-Cycle-Gated-Four-Leg-Straight-Crawl-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingForwardBiasedCycleGatedFourLegStraightCrawlExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingForwardBiasedCycleGatedFourLegStraightCrawlExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Schedule-Matched-Support-Straight-Crawl-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingScheduleMatchedSupportStraightCrawlExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingScheduleMatchedSupportStraightCrawlExternalRearPayloadPPORunnerCfg",
)
_register(
    "Drobot-Commanded-Walk-Symmetry-Gated-Robust-Straight-Crawl-External-Rear-Payload-Direct",
    "DrobotCommandedWalkingSymmetryGatedRobustStraightCrawlExternalRearPayloadEnvCfg",
    "DrobotCommandedWalkingSymmetryGatedRobustStraightCrawlExternalRearPayloadPPORunnerCfg",
)
