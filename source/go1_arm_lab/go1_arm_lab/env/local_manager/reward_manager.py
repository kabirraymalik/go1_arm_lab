from __future__ import annotations
import torch
from isaaclab.managers import RewardManager as RewardManagerBase

class RewardManager(RewardManagerBase):

    def __init__(self,cfg, env):
        super().__init__(cfg, env)
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.arm_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) 


    def compute(self, dt: float) -> tuple[torch.Tensor,torch.Tensor]:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        self.arm_reward_buf[:] = 0.0 
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # check if the term is a special term for arm
            if name.startswith("end_effector"):  ## TODO: 
                self.arm_reward_buf += value
            else:
                self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value

            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt

        return self._reward_buf, self.arm_reward_buf
