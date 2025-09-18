from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import ObservationManager as ObservationManagerBase


class ObservationManager(ObservationManagerBase):

    def compute_obs(self):
        num_prop: int = 0  # The number of proprioceptive observations.

        num_priv: int = 0   # The number of privileged observations.

        num_history: int = 0  # The length of history.

        for group_name in self._group_obs_term_names:

            # check ig group name is valid
            if group_name not in self._group_obs_term_names:
                raise ValueError(
                    f"Unable to find the group '{group_name}' in the observation manager."
                    f" Available groups are: {list(self._group_obs_term_names.keys())}"
                )
            # iterate over all the terms in each group
            group_term_names = self._group_obs_term_names[group_name]

            # read attributes for each term
            obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])  
            for term_name, term_cfg in obs_terms:
                if term_name.startswith("priv_"):
                    num_priv += (term_cfg.func(self._env, **term_cfg.params).clone()).shape[1]
                else:
                    num_prop += (term_cfg.func(self._env, **term_cfg.params).clone()).shape[1]
                    num_history = term_cfg.history_length 

        return num_history, num_prop, num_priv
