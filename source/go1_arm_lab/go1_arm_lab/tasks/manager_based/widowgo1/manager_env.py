
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs import ManagerBasedRLEnv
from . import local_manager

class ManagerRLEnv(ManagerBasedRLEnv):
    """Configuration for the loco-manipulation environment."""
    def __init__(self, cfg, render_mode, **kwargs):
        super().__init__(cfg=cfg)
        self._sim_step_counter = 0

    def load_managers(self):
        super().load_managers()
        self.reward_manager = local_manager.RewardManager(self.cfg.rewards, self)
        self.observation_manager = local_manager.ObservationManager(self.cfg.observations, self)

    def step(self, action) :
        self.action_manager.process_action(action.to(self.device))
        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf, arm_reward_buf = self.reward_manager.compute(dt=self.step_dt)
        
        # Store arm reward in extras for downstream processing
        self.extras["arm_reward"] = arm_reward_buf

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # -- update command
        self.command_manager.compute(dt=self.step_dt)

        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras (standard interface)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras