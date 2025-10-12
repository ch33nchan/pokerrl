"""
Policy adaptors for converting between algorithm outputs and OpenSpiel evaluators.

This module provides adaptors that convert neural network policies and algorithm
outputs into the format expected by OpenSpiel's exact evaluators.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PolicyAdaptor(ABC):
    """Base class for policy adaptors."""
    
    @abstractmethod
    def extract_policy(self, algorithm, game_wrapper) -> Dict[str, np.ndarray]:
        """Extract policy dictionary from algorithm for OpenSpiel evaluation."""
        pass


class DeepCFRPolicyAdaptor(PolicyAdaptor):
    """Policy adaptor for Deep CFR algorithms."""
    
    def extract_policy(self, algorithm, game_wrapper) -> Dict[str, np.ndarray]:
        """
        Extract strategy network policy for Deep CFR.
        
        Args:
            algorithm: Deep CFR algorithm instance
            game_wrapper: Game wrapper instance
            
        Returns:
            Dictionary mapping info states to action probability vectors
        """
        policy_dict = {}
        
        # Get all information states from the game
        info_states = self._get_all_info_states(game_wrapper)
        
        for info_state_str, state_encoding in info_states.items():
            with torch.no_grad():
                # Get strategy network output
                state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0)
                
                if hasattr(algorithm, 'strategy_network'):
                    # Deep CFR has separate strategy network
                    strategy_output = algorithm.strategy_network(state_tensor, network_type='strategy')
                    if isinstance(strategy_output, dict) and 'policy' in strategy_output:
                        action_probs = torch.softmax(strategy_output['policy'], dim=-1)
                    else:
                        action_probs = torch.softmax(strategy_output, dim=-1)
                elif hasattr(algorithm, 'regret_network'):
                    # Use regret network and convert to policy
                    regret_output = algorithm.regret_network(state_tensor, network_type='regret')
                    if isinstance(regret_output, dict) and 'advantages' in regret_output:
                        regrets = regret_output['advantages']
                    elif isinstance(regret_output, dict) and 'regrets' in regret_output:
                        regrets = regret_output['regrets']
                    else:
                        regrets = regret_output
                    
                    # Convert regrets to policy via regret matching
                    positive_regrets = torch.clamp(regrets, min=0.0)
                    regret_sum = positive_regrets.sum(dim=-1, keepdim=True)
                    
                    # Handle case where all regrets are negative (uniform policy)
                    epsilon = 1e-6
                    if regret_sum < epsilon:
                        action_probs = torch.ones_like(positive_regrets) / positive_regrets.shape[-1]
                    else:
                        action_probs = positive_regrets / regret_sum
                    
                    # Add epsilon smoothing for numerical stability
                    action_probs = action_probs + epsilon
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
                    legal_probs = normalized_probs.squeeze(0).numpy()[legal_actions]
                    
                    # Create dictionary with legal action indices
                    legal_policy = {}
                    for i, action_idx in enumerate(legal_actions):
                        legal_policy[action_idx] = legal_probs[i]
                    
                    policy_dict[info_state_str] = legal_policy
                else:
                    # Fallback to all actions if no mask available
                    action_probs_numpy = action_probs.squeeze(0).numpy()
                    policy_dict[info_state_str] = {i: prob for i, prob in enumerate(action_probs_numpy)}
        
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
            
            # Traverse all legal actions
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
                if state.is_terminal() or state.is_chance_node():
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
        except:
            pass
        
        return None


class SDCFRPolicyAdaptor(DeepCFRPolicyAdaptor):
    """Policy adaptor for SD-CFR algorithms (inherits from DeepCFR)."""
    pass


class ARMACPolicyAdaptor(PolicyAdaptor):
    """Policy adaptor for ARMAC algorithms."""
    
    def extract_policy(self, algorithm, game_wrapper) -> Dict[str, np.ndarray]:
        """
        Extract actor network policy for ARMAC.
        
        Args:
            algorithm: ARMAC algorithm instance
            game_wrapper: Game wrapper instance
            
        Returns:
            Dictionary mapping info states to action probability vectors
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
                    legal_probs = normalized_probs.squeeze(0).numpy()[legal_actions]
                    
                    # Create dictionary with legal action indices
                    legal_policy = {}
                    for i, action_idx in enumerate(legal_actions):
                        legal_policy[action_idx] = legal_probs[i]
                    
                    policy_dict[info_state_str] = legal_policy
                else:
                    # Fallback to all actions if no mask available
                    action_probs_numpy = action_probs.squeeze(0).numpy()
                    policy_dict[info_state_str] = {i: prob for i, prob in enumerate(action_probs_numpy)}
        
        return policy_dict
    
    def _get_all_info_states(self, game_wrapper) -> Dict[str, np.ndarray]:
        """Get all information states and their encodings from the game wrapper."""
        # Reuse implementation from DeepCFRPolicyAdaptor
        adaptor = DeepCFRPolicyAdaptor()
        return adaptor._get_all_info_states(game_wrapper)
    
    def _get_legal_actions_mask(self, info_state_str: str, game_wrapper) -> Optional[np.ndarray]:
        """Get legal actions mask for an information state."""
        # Reuse implementation from DeepCFRPolicyAdaptor
        adaptor = DeepCFRPolicyAdaptor()
        return adaptor._get_legal_actions_mask(info_state_str, game_wrapper)


class TabularCFRPolicyAdaptor(PolicyAdaptor):
    """Policy adaptor for tabular CFR algorithms."""
    
    def extract_policy(self, algorithm, game_wrapper) -> Dict[str, np.ndarray]:
        """
        Extract tabular policy from tabular CFR solver.
        
        Args:
            algorithm: Tabular CFR algorithm instance
            game_wrapper: Game wrapper instance
            
        Returns:
            Dictionary mapping info states to action probability vectors
        """
        policy_dict = {}
        
        # Get the tabular policy from the CFR solver
        if hasattr(algorithm, 'solver'):
            # This is a TabularCFRAgent
            tabular_policy = algorithm.solver.average_policy()
            
            # Get the game and traverse to find all information states
            game = game_wrapper.game if hasattr(game_wrapper, 'game') else algorithm.solver.get_game()
            
            def traverse_states(state):
                if state.is_terminal() or state.is_chance_node():
                    if state.is_chance_node():
                        # Traverse chance node outcomes
                        for action, _ in state.chance_outcomes():
                            child_state = state.clone()
                            child_state.apply_action(action)
                            traverse_states(child_state)
                    return
                
                # This is a player decision node
                player = state.current_player()
                info_state_str = state.information_state_string(player)
                
                if info_state_str not in policy_dict:
                    # Get action probabilities from the tabular policy
                    action_probs = tabular_policy.action_probabilities(state)
                    
                    # Convert to numpy array
                    if action_probs:
                        # action_probs is a list of (action, probability) tuples
                        max_action = max(action for action, _ in action_probs.items())
                        prob_array = np.zeros(max_action + 1)
                        for action, prob in action_probs.items():
                            prob_array[action] = prob
                        policy_dict[info_state_str] = prob_array
                
                # Traverse all legal actions
                for action in state.legal_actions():
                    child_state = state.clone()
                    child_state.apply_action(action)
                    traverse_states(child_state)
            
            # Start traversal from initial state
            traverse_states(game.new_initial_state())
        
        return policy_dict


def create_policy_adaptor(algorithm_name: str) -> PolicyAdaptor:
    """
    Factory function to create appropriate policy adaptor.
    
    Args:
        algorithm_name: Name of the algorithm ('deep_cfr', 'sd_cfr', 'armac', 'tabular_cfr')
        
    Returns:
        Appropriate policy adaptor instance
    """
    if algorithm_name.lower() in ['deep_cfr', 'deepcfr']:
        return DeepCFRPolicyAdaptor()
    elif algorithm_name.lower() in ['sd_cfr', 'sdcfr', 'sd-cfr']:
        return SDCFRPolicyAdaptor()
    elif algorithm_name.lower() in ['armac']:
        return ARMACPolicyAdaptor()
    elif algorithm_name.lower() in ['tabular_cfr', 'tabular']:
        return TabularCFRPolicyAdaptor()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def evaluate_algorithm_policy(algorithm, game_wrapper, evaluator, algorithm_name: str):
    """
    Convenience function to evaluate an algorithm's policy.
    
    Args:
        algorithm: Algorithm instance
        game_wrapper: Game wrapper instance
        evaluator: OpenSpiel exact evaluator
        algorithm_name: Name of the algorithm
        
    Returns:
        EvaluationResult from the evaluator
    """
    # Create appropriate policy adaptor
    adaptor = create_policy_adaptor(algorithm_name)
    
    # Extract policy dictionary
    policy_dict = adaptor.extract_policy(algorithm, game_wrapper)
    
    logger.info(f"Extracted policy for {len(policy_dict)} information states")
    
    # Evaluate using exact evaluator
    result = evaluator.evaluate_policy(policy_dict)
    
    logger.info(
        f"Evaluation complete: NashConv={result.nash_conv:.6f}, "
        f"Exploitability={result.exploitability:.6f}"
    )
    
    return result