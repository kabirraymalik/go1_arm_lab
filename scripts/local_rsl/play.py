
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import pandas as pd
# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from local_rsl.runners import OnPolicyRunner
from isaaclab.utils.dict import print_dict

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from go1_arm_lab.tasks.manager_based.widowgo1.config.rsl_rl_ppo_cfg import Widowgo1RslRlOnPolicyRunnerCfg
from local_rsl.wrappers.VecEnvWrapper import RslRlVecEnvWrapper

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, export_policy_as_jit, export_policy_as_onnx
import numpy as np


def prepare_obs(env, agent_cfg:Widowgo1RslRlOnPolicyRunnerCfg):
    """
        Modify the order of observations containing historical information for easier network reading. Try to avoid modifications if possible!

        Args:
            env: The environment
            agent_cfg: Agent configuration

        Returns:
            total(num_hist, num_prop): Indices of the modified observation order
            obs_new(num_envs, num_prop * num_hist): Modified observation order indices corresponding to the observations
        Notes:
            Try to avoid modifications if possible!

        Example:
            In env.step, the original observation follows right-to-left order, after modification it becomes left-to-right

            For example, the original observation is:
                obs = ang_vel(3) * 10(num_hist) + joint_pos(18) * 10(num_hist) + joint_vel(18) * 10(num_hist)
            After env.step, the observation (obs) becomes structured as follows:
                obs_old = ang_vel_timestep_10 -> ang_vel_timestep_1, joint_pos_timestep_10 -> joint_pos_timestep_1, joint_vel_timestep_10 -> joint_vel_timestep_1
            We need to modify the observation order to:
                obs_new = ang_vel_timestep_1, joint_pos_timestep_1, joint_vel_timestep_1 -> ang_vel_timestep_10, joint_pos_timestep_10, joint_vel_timestep_10
    """
    total = np.zeros((agent_cfg.policy.num_hist, agent_cfg.policy.num_prop)) 
    obs_new = torch.zeros(env.num_envs, agent_cfg.policy.num_prop * agent_cfg.policy.num_hist).to(env.device)

    lst, length = env.get_obs_list_length()
    lst = [item for item in lst if not item.startswith("policy-priv_")]

    result_dict = {}
    for i in range(len(lst)):
        c = np.array(list(range( sum(length[: i + 1 ]) - int(length[i] / agent_cfg.policy.num_hist), sum(length[: i + 1]) )))
        result_dict[lst[i]] = c

    key_list = list(result_dict.keys())
    a1_list = []
    for i in range(agent_cfg.policy.num_hist):
        for j in range(len(lst)):
            a1 = np.concatenate([
                result_dict[key_list[j]] - (i) * result_dict[key_list[j]].shape[0]])
            a1_list.append(a1)
            if j == len(lst) - 1:
                a1_list = np.concatenate(a1_list)
                total[i, :] = a1_list
                a1_list = []
    return total, obs_new

def change_obs_order(obs, obs_new, total, env, agent_cfg):

    """
        Modify the order of observations containing historical information for easier network reading. Try to avoid modifications if possible!

        Args:
            obs(num_envs, num_prop * num_hist): The input to the actor and critic network
            obs_new(num_envs, num_prop * num_hist): Modified observation order indices corresponding to the observations
            total(num_hist, num_prop): Indices of the modified observation order
            env: The environment
            agent_cfg: Agent configuration

        Returns:
            obs(num_envs, num_prop * num_hist): The input to the actor and critic network
            obs_new(num_envs, num_prop * num_hist): Modified observation order indices corresponding to the observations(need to reset)

        Notes:
            Try to avoid modifications if possible!

        Example:
            In env.step, the original observation follows right-to-left order, after modification it becomes left-to-right

            For example, the original observation is:
                obs = ang_vel(3) * 10(num_hist) + joint_pos(18) * 10(num_hist) + joint_vel(18) * 10(num_hist)
            After env.step, the observation (obs) becomes structured as follows:
                obs_old = ang_vel_timestep_10 -> ang_vel_timestep_1, joint_pos_timestep_10 -> joint_pos_timestep_1, joint_vel_timestep_10 -> joint_vel_timestep_1
            We need to modify the observation order to:
                obs_new = ang_vel_timestep_1, joint_pos_timestep_1, joint_vel_timestep_1 -> ang_vel_timestep_10, joint_pos_timestep_10, joint_vel_timestep_10
    """
    for i in range(10):
        obs_1 = obs[:, total[i, :]].to(env.device)
        obs_new = torch.cat([obs_new, obs_1], dim = -1)

    # only prop obs:    
    obs = obs_new[:, agent_cfg.policy.num_prop  * agent_cfg.policy.num_hist:] 
    
    # prop and priv obs:
    # obs = torch.cat([obs_new[:, agent_cfg.policy.num_prop  * agent_cfg.policy.num_hist :], 
    #                  obs[:, agent_cfg.policy.num_prop  * agent_cfg.policy.num_hist :]], dim=-1)  
    
    obs_new = torch.zeros(env.num_envs, agent_cfg.policy.num_prop * agent_cfg.policy.num_hist).to(env.device)
    return obs, obs_new


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg: Widowgo1RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording
    if args_cli.video:
        if not hasattr(env, 'render_mode') or env.render_mode is None:
            env.render_mode = "rgb_array"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )

    # reset environment
    obs, _ = env.get_observations()
    total, obs_new = prepare_obs(env, agent_cfg)
    timestep = 0
  
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            timestep += 1
            obs, obs_new = change_obs_order(obs, obs_new, total, env, agent_cfg)
            actions = policy(obs, hist_encoding=True) # no priv obs

            obs, _, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

