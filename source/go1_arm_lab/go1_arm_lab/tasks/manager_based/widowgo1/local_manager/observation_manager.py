# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom Observation Manager with time-first (interleaved) history organization.

This observation manager reorganizes observation history to group observations by timestep
rather than by observation term.

Format comparison:
    Official IsaacLab format (term-first):
        [a(t-2), a(t-1), a(t), b(t-2), b(t-1), b(t), c(t-2), c(t-1), c(t)]
        Each term's complete history is grouped together (aaa|bbb|ccc)

    This implementation (time-first):
        [a(t-2), b(t-2), c(t-2), a(t-1), b(t-1), c(t-1), a(t), b(t), c(t)]
        All observations from the same timestep are grouped together (abc|abc|abc)

Usage:
    from custom_obs_manager import ObservationManager
    
    # The manager will automatically apply time-first interleaving when:
    # 1. Multiple observation terms have history enabled
    # 2. All history terms have the same history_length
    # 3. Terms are concatenated (concatenate_terms=True)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ObservationManager as ObservationManagerBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ObservationManager(ObservationManagerBase):
    """Custom Observation Manager with time-first history organization.
    
    This manager inherits from IsaacLab's official ObservationManager and modifies
    the observation concatenation order to group observations by timestep rather
    than by observation term.
    
    Benefits:
        - More intuitive for temporal processing (e.g., temporal CNNs, RNNs)
        - Observations from the same moment are adjacent in memory
        - Easier to interpret observation history
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the custom observation manager.
        
        Args:
            cfg: The configuration object or dictionary (dict[str, ObservationGroupCfg]).
            env: The environment instance.
        """
        # Call parent constructor
        super().__init__(cfg, env)
        
        # Compute metadata for interleaving
        self._interleave_metadata: dict[str, dict] = {}
        self._compute_interleave_metadata()
        print("[INFO] Local Observation Manager:")
        # Print detailed information about observation manager
        print(self)
        
    def _compute_interleave_metadata(self):
        """Analyze observation groups and compute metadata for interleaving.
        
        For each observation group, this method determines:
        1. Whether interleaving should be applied
        2. Specifications for each observation term (dimensions, history length)
        3. How to reorganize the observations
        """
        for group_name in self._group_obs_term_names:
            metadata = {
                "should_interleave": False,
                "history_length": None,
                "term_specs": [],
            }
            
            # Analyze each term in the group
            term_names = self._group_obs_term_names[group_name]
            term_cfgs = self._group_obs_term_cfgs[group_name]
            
            for term_name, term_cfg in zip(term_names, term_cfgs):
                # Check if term has flattened history
                has_history = (
                    term_cfg.history_length > 0 and 
                    term_cfg.flatten_history_dim
                )
                
                if has_history:
                    # Get single-step observation dimension
                    single_obs = term_cfg.func(self._env, **term_cfg.params)
                    dims_per_step = single_obs.shape[1]
                    
                    spec = {
                        "name": term_name,
                        "has_history": True,
                        "history_length": term_cfg.history_length,
                        "dims_per_step": dims_per_step,
                        "total_dims": dims_per_step * term_cfg.history_length,
                    }
                    
                    # Check for consistent history length
                    if metadata["history_length"] is None:
                        metadata["history_length"] = term_cfg.history_length
                    elif metadata["history_length"] != term_cfg.history_length:
                        # Cannot interleave with mismatched history lengths
                        print(
                            f"[ObservationManager] Warning: Group '{group_name}' has "
                            f"terms with different history lengths "
                            f"({metadata['history_length']} vs {term_cfg.history_length}). "
                            f"Time-first interleaving will not be applied."
                        )
                        metadata["should_interleave"] = False
                        metadata["term_specs"].append(spec)
                        break
                else:
                    # Term without history
                    obs = term_cfg.func(self._env, **term_cfg.params)
                    spec = {
                        "name": term_name,
                        "has_history": False,
                        "history_length": 0,
                        "dims_per_step": obs.shape[1],
                        "total_dims": obs.shape[1],
                    }
                
                metadata["term_specs"].append(spec)
            
            # Determine if interleaving should be applied
            history_terms = [s for s in metadata["term_specs"] if s["has_history"]]
            
            # Interleave if:
            # 1. Multiple terms have history
            # 2. All history terms have same history_length
            # 3. Terms are concatenated
            if (
                len(history_terms) > 1 and
                metadata["history_length"] is not None and
                self._group_obs_concatenate.get(group_name, False)
            ):
                metadata["should_interleave"] = True
            
            self._interleave_metadata[group_name] = metadata

    def compute_group(
        self, 
        group_name: str, 
        update_history: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute observations for a group with optional time-first interleaving.
        
        This method:
        1. Calls the parent implementation to get standard observations
        2. Applies time-first interleaving if conditions are met
        
        Args:
            group_name: The name of the group for which to compute observations.
            update_history: Whether to update the history buffer.
            
        Returns:
            Observations as a concatenated tensor (with interleaving if applicable)
            or as a dictionary (if concatenate_terms=False).
        """
        # Get observations using parent implementation
        obs = super().compute_group(group_name, update_history=update_history)
        
        # Check if interleaving is needed
        metadata = self._interleave_metadata.get(group_name, {})
        
        if metadata.get("should_interleave", False) and isinstance(obs, torch.Tensor):
            # Apply time-first interleaving
            obs = self._apply_interleaving(obs, metadata)
        
        return obs

    def _apply_interleaving(self, obs: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Reorganize observations from term-first to time-first format.
        
        Args:
            obs: Observations in term-first format [num_envs, total_dims]
                 Format: [term1_all_history, term2_all_history, term3_all_history, ...]
            metadata: Metadata about observation terms and their specifications
            
        Returns:
            Observations in time-first format [num_envs, total_dims]
            Format: [all_terms_at_t-N, all_terms_at_t-N+1, ..., all_terms_at_t]
        """
        num_envs = obs.shape[0]
        history_length = metadata["history_length"]
        term_specs = metadata["term_specs"]
        
        # Calculate the starting position of each term in the concatenated obs
        term_positions = []
        current_pos = 0
        
        for spec in term_specs:
            term_positions.append({
                "spec": spec,
                "start": current_pos,
                "end": current_pos + spec["total_dims"],
            })
            current_pos += spec["total_dims"]
        
        # Build interleaved observation by iterating through timesteps
        interleaved_chunks = []
        
        # Process each timestep from oldest to newest
        for t in range(history_length):
            # For each term at timestep t
            for pos_info in term_positions:
                spec = pos_info["spec"]
                
                if spec["has_history"]:
                    # Extract the slice corresponding to timestep t
                    dims = spec["dims_per_step"]
                    start_idx = pos_info["start"] + t * dims
                    end_idx = start_idx + dims
                    
                    interleaved_chunks.append(obs[:, start_idx:end_idx])
        
        # Append non-history terms at the end
        for pos_info in term_positions:
            spec = pos_info["spec"]
            
            if not spec["has_history"]:
                interleaved_chunks.append(obs[:, pos_info["start"]:pos_info["end"]])
        
        # Concatenate all chunks along the feature dimension
        interleaved_obs = torch.cat(interleaved_chunks, dim=1)
        
        return interleaved_obs

    def __str__(self) -> str:
        """Generate string representation with interleaving information."""
        # Get base string from parent
        msg = super().__str__()
        
        # Add custom interleaving information
        msg += "\n" + "=" * 80 + "\n"
        msg += "Time-First Observation Manager Configuration\n"
        msg += "=" * 80 + "\n"
        
        for group_name, metadata in self._interleave_metadata.items():
            msg += f"\nGroup: '{group_name}'\n"
            
            if metadata.get("should_interleave", False):
                msg += "  Status: TIME-FIRST INTERLEAVING ENABLED\n"
                msg += f"  History Length: {metadata['history_length']}\n"
                msg += "  Format: abc|abc|abc (observations grouped by timestep)\n"
                msg += "  Terms with history:\n"
                
                for spec in metadata["term_specs"]:
                    if spec["has_history"]:
                        msg += (
                            f"    - {spec['name']}: "
                            f"{spec['dims_per_step']} dims × "
                            f"{spec['history_length']} steps = "
                            f"{spec['total_dims']} dims total\n"
                        )
            else:
                history_count = sum(1 for s in metadata["term_specs"] if s["has_history"])
                
                if history_count > 1:
                    msg += "  Status: ○ Standard format (interleaving disabled)\n"
                    msg += "  Reason: Terms have different history lengths\n"
                elif history_count == 1:
                    msg += "  Status: ○ Standard format (interleaving not needed)\n"
                    msg += "  Reason: Only one term has history\n"
                else:
                    msg += "  Status: ○ No history terms in this group\n"
        
        msg += "\n" + "=" * 80 + "\n"
        
        return msg