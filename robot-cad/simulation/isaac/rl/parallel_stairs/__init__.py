"""Pure-RL, vectorized Drobot stair task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Drobot-Pure-Stairs-Direct",
    entry_point=f"{__name__}.pure_stairs_env:DrobotPureStairsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pure_stairs_env_cfg:DrobotPureStairsEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DrobotPureStairsPPORunnerCfg"
        ),
    },
)


def _register_variant(task_id: str, env_cfg: str, runner_cfg: str) -> None:
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.pure_stairs_env:DrobotPureStairsEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pure_stairs_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}"
            ),
        },
    )


_register_variant(
    "Drobot-Pure-Stairs-Hip-Direct",
    "DrobotPureStairsHipEnvCfg",
    "DrobotPureStairsHipPPORunnerCfg",
)
_register_variant(
    "Drobot-Pure-Stairs-Low-Hip-Direct",
    "DrobotPureStairsLowHipEnvCfg",
    "DrobotPureStairsLowHipPPORunnerCfg",
)
_register_variant(
    "Drobot-Pure-Stairs-Sideways-Hip-Direct",
    "DrobotPureStairsSidewaysHipEnvCfg",
    "DrobotPureStairsSidewaysHipPPORunnerCfg",
)
