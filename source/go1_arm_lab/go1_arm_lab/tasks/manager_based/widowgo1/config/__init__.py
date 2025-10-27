
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import flat_env_cfg
from .agents import rsl_rl_ppo_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Flat-widowgo1",
    entry_point="lab_play.tasks.manager_based.widowgo1.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.widowgo1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.Widowgo1FlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Flat-widowgo1-Play",
    entry_point="lab_play.tasks.manager_based.widowgo1.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.widowgo1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.Widowgo1FlatPPORunnerCfg,
    },
)
