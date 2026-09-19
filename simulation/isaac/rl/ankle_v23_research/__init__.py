"""Research-inspired reward continuations with unchanged stair mechanics."""
import gymnasium as gym
from ankle_v23_experiments.strategies import STRATEGIES as BASE_STRATEGIES
from .strategies import STRATEGIES

# This registry extension exists only in this follow-up worker's process.
BASE_STRATEGIES.update(STRATEGIES)
gym.register(id="Drobot-Ankle-V23-Research", entry_point=__name__ + ".env:ResearchEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:ResearchCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:ResearchPPOCfg"})
