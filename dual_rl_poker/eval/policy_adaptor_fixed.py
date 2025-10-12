"""
Fixed policy adaptors for exact OpenSpiel evaluation.

This corrects the legal action handling to only include legal actions
in the policy dictionaries, fixing the action space mismatch issue.
"""

import torch
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod

from eval.policy_adaptor import PolicyAdaptor, TabularCFRPolicyAdaptor


class FixedDeepCFRPolicyAdaptor(PolicyAdaptor):
    """Fixed policy adaptor for Deep CFR algorithms with proper legal action handling."""
    
    def extract_policy(self, algorithm, game_wrapper) -> Dict[str, Dict[int, float]]:
        """
        Extract neural network policy from Deep CFR solver.
        
        Args:
            algorithm: Deep CFR algorithm instance
            game_wrapper: Game wrapper instance
            
        Returns:
            Dictionary mapping info states to action probability dictionaries with only legal actions
        """
        policy_dict = {}
        
        # Get all information states from the game
        info_states = self._get_all_info_states(game_wrapper)
        
        for info_state_str, state_encoding in info_states.items():
            with torch.no_grad():
                # Get network output
                state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
                
                if hasattr(algorithm, 'strategy_network'):
                    # Deep CFR has strategy network - call with strategy type
                    network_output = algorithm.strategy_network(state_tensor, network_type='strategy')
                    if isinstance(network_output, dict) and 'policy' in network_output:
                        action_probs = network_output['policy']
                    else:
                        # Fallback if dict doesn't have policy key
                        action_probs = torch.softmax(network_output, dim=-1)
                elif hasattr(algorithm, 'regret_network'):
                    # Alternative: use regret network and apply softmax
                    regret_output = algorithm.regret_network(state_tensor)
                    if isinstance(regret_output, dict) and 'advantages' in regret_output:
                        action_probs = torch.softmax(regret_output['advantages'], dim=-1)
                    else:
                        action_probs = torch.softmax(regret_output, dim=-1)
                    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                else:
                    raise ValueError("Algorithm must have strategy_network or regret_network")
                
                # Apply legal actions mask and extract only legal actions
                legal_actions_mask = self._get_legal_actions_mask(
                    info_state_str, game_wrapper
                )
                if legal_actions_mask is not None:
                    masked_probs = action_probs * torch.FloatTensor(legal_actions_mask)
                    normalized_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
                    
                    # Extract only legal actions for OpenSpiel compatibility
                    legal_actions = [i for i, mask_val in enumerate(legal_actions_mask) if mask_val > 0]
                    probs_numpy = normalized_probs.squeeze(0).numpy()
                    
                    # Create dictionary with ONLY legal action indices and their probabilities
                    legal_policy = {}
                    for action_idx in legal_actions:
                        legal_policy[action_idx] = float(probs_numpy[action_idx])
                    
                    policy_dict[info_state_str] = legal_policy
                else:
                    # Fallback to all actions if no mask available
                    action_probs_numpy = action_probs.squeeze(0).numpy()
                    policy_dict[info_state_str] = {i: float(prob) for i, prob in enumerate(action_probs_numpy)}
        
        return policy_dict
    
    def _get_all_info_states(self, game_wrapper) -> Dict[str, np.ndarray]:
        """Get all information states and their encodings from the game wrapper."""
        info_states = {}
        
        # Use game wrapper's encoding method
        if hasattr(game_wrapper, 'get_all_info_states'):
            return game_wrapper.get_all_info_states()
        
        # Fallback: traverse game states
        game = game_wrapper.game if hasattr(game_wrapper, 'game') else game_wrapper._game
        
        def traverse(state):
            if state.is_terminal():
                return
                
            if state.is_chance_node():
                # Handle chance nodes - traverse all outcomes
                for action, prob in state.chance_outcomes():
                    child = state.clone()
                    child.apply_action(action)
                    traverse(child)
                return
            
            # Decision node - record the info state
            player = state.current_player()
            info_state_str = state.information_state_string(player)
            
            if info_state_str not in info_states:
                # Encode the state
                if hasattr(game_wrapper, 'encode_state'):
                    encoding = game_wrapper.encode_state(state)
                else:
                    # Use information state tensor if available
                    encoding = state.information_state_tensor(player)
                
                info_states[info_state_str] = np.array(encoding)
            
            # Traverse children
            for action in state.legal_actions():
                child = state.clone()
                child.apply_action(action)
                traverse(child)
        
        traverse(game.new_initial_state())
        return info_states

    def _get_legal_actions_mask(self, info_state_str: str, game_wrapper) -> Optional[np.ndarray]:
        """Get legal actions mask for an information state."""
        if hasattr(game_wrapper, 'get_legal_actions_mask'):
            return game_wrapper.get_legal_actions_mask(info_state_str)
        
        # Try to extract from the game directly
        try:
            game = game_wrapper.game if hasattr(game_wrapper, 'game') else game_wrapper._game
            
            # Find the state corresponding to this info state
            def find_state(state, target_info_state):
                if state.is_terminal():
                    return None
                
                if state.is_chance_node():
                    for action, _ in state.chance_outcomes():
                        child = state.clone()
                        child.apply_action(action)
                        result = find_state(child, target_info_state)
                        if result is not None:
                            return result
                    return None
                
                player = state.current_player()
                if state.information_state_string(player) == target_info_state:
                    return state
                
                for action in state.legal_actions():
                    child = state.clone()
                    child.apply_action(action)
                    result = find_state(child, target_info_state)
                    if result is not None:
                        return result
                
                return None
            
            state = find_state(game.new_initial_state(), info_state_str)
            if state is not None:
                num_actions = game.num_distinct_actions()
                mask = np.zeros(num_actions)
                for action in state.legal_actions():
                    mask[action] = 1.0
                return mask
        except Exception as e:
            print(f"Error getting legal actions mask for {info_state_str}: {e}")
        
        return None


class FixedSDCFRPolicyAdaptor(FixedDeepCFRPolicyAdaptor):
    """Fixed policy adaptor for SD-CFR algorithms (inherits from fixed DeepCFR)."""
    pass


class FixedARMACPolicyAdaptor(PolicyAdaptor):
    """Fixed policy adaptor for ARMAC algorithms with proper legal action handling."""
    
    def extract_policy(self, algorithm, game_wrapper) -> Dict[str, Dict[int, float]]:
        """
        Extract actor network policy for ARMAC.
        
        Args:
            algorithm: ARMAC algorithm instance
            game_wrapper: Game wrapper instance
            
        Returns:
            Dictionary mapping info states to action probability dictionaries with only legal actions
        """
        policy_dict = {}
        
        # Get all information states from the game
        info_states = self._get_all_info_states(game_wrapper)
        
        for info_state_str, state_encoding in info_states.items():
            with torch.no_grad():
                # Get actor network output
                state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
                
                if hasattr(algorithm, 'actor'):
                    # ARMAC has actor network
                    actor_output = algorithm.actor(state_tensor)
                    if isinstance(actor_output, dict) and 'action_probs' in actor_output:
                        action_probs = actor_output['action_probs']
                    else:
                        action_probs = torch.softmax(actor_output, dim=-1)
                else:
                    raise ValueError("ARMAC algorithm must have actor network")
                
                # Apply legal actions mask and extract only legal actions
                legal_actions_mask = self._get_legal_actions_mask(
                    info_state_str, game_wrapper
                )
                if legal_actions_mask is not None:
                    masked_probs = action_probs * torch.FloatTensor(legal_actions_mask)
                    normalized_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
                    
                    # Extract only legal actions for OpenSpiel compatibility
                    legal_actions = [i for i, mask_val in enumerate(legal_actions_mask) if mask_val > 0]
                    probs_numpy = normalized_probs.squeeze(0).numpy()
                    
                    # Create dictionary with ONLY legal action indices and their probabilities
                    legal_policy = {}
                    for action_idx in legal_actions:
                        legal_policy[action_idx] = float(probs_numpy[action_idx])
                    
                    policy_dict[info_state_str] = legal_policy
                else:
                    # Fallback to all actions if no mask available
                    action_probs_numpy = action_probs.squeeze(0).numpy()
                    policy_dict[info_state_str] = {i: float(prob) for i, prob in enumerate(action_probs_numpy)}
        
        return policy_dict
    
    def _get_all_info_states(self, game_wrapper) -> Dict[str, np.ndarray]:
        """Get all information states and their encodings from the game wrapper."""
        # Reuse implementation from FixedDeepCFRPolicyAdaptor
        adaptor = FixedDeepCFRPolicyAdaptor()
        return adaptor._get_all_info_states(game_wrapper)

    def _get_legal_actions_mask(self, info_state_str: str, game_wrapper) -> Optional[np.ndarray]:
        """Get legal actions mask for an information state."""
        # Reuse implementation from FixedDeepCFRPolicyAdaptor
        adaptor = FixedDeepCFRPolicyAdaptor()
        return adaptor._get_legal_actions_mask(info_state_str, game_wrapper)


def create_fixed_policy_adaptor(algorithm_name: str) -> PolicyAdaptor:
    """
    Factory function to create appropriate fixed policy adaptor.
    
    Args:
        algorithm_name: Name of the algorithm ('deep_cfr', 'sd_cfr', 'armac', 'tabular_cfr')
        
    Returns:
        Appropriate fixed policy adaptor instance
    """
    if algorithm_name.lower() in ['deep_cfr', 'deepcfr']:
        return FixedDeepCFRPolicyAdaptor()
    elif algorithm_name.lower() in ['sd_cfr', 'sdcfr', 'sd-cfr']:
        return FixedSDCFRPolicyAdaptor()
    elif algorithm_name.lower() in ['armac']:
        return FixedARMACPolicyAdaptor()
    elif algorithm_name.lower() in ['tabular_cfr', 'tabular']:
        return TabularCFRPolicyAdaptor()  # Use original for tabular
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def evaluate_algorithm_policy_fixed(algorithm, game_wrapper, evaluator, algorithm_name: str):
    """
    Convenience function to evaluate an algorithm's policy with fixed adaptors.
    
    Args:
        algorithm: Trained algorithm instance
        game_wrapper: Game wrapper instance  
        evaluator: Exact evaluator instance
        algorithm_name: Name of the algorithm
        
    Returns:
        Evaluation result
    """
    adaptor = create_fixed_policy_adaptor(algorithm_name)
    policy_dict = adaptor.extract_policy(algorithm, game_wrapper)
    result = evaluator.evaluate_policy(policy_dict)
    return result