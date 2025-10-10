import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
import random
import json
from collections import defaultdict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CFRAgent(BasePokerPlayer):
    def __init__(self, name: str = "CFR_Agent", iterations: int = 1000):
        super().__init__()
        self.name = name
        self.iterations = iterations
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.strategy = defaultdict(lambda: defaultdict(float))
        self.action_mapping = {'fold': 0, 'call': 1, 'raise': 2, 'check': 3}
        
        logger.info(f"CFRAgent {name} initialized with {iterations} iterations")
    
    def declare_action(self, valid_actions, hole_card, round_state):
        information_set = self._get_information_set(hole_card, round_state)
        
        strategy = self._get_strategy(information_set, valid_actions)
        
        action_probs = [strategy.get(action['action'], 0.0) for action in valid_actions]
        
        if sum(action_probs) == 0:
            action_probs = [1.0 / len(valid_actions)] * len(valid_actions)
        else:
            total_prob = sum(action_probs)
            action_probs = [p / total_prob for p in action_probs]
        
        chosen_action_idx = np.random.choice(len(valid_actions), p=action_probs)
        chosen_action = valid_actions[chosen_action_idx]
        
        return chosen_action['action'], chosen_action['amount']
    
    def _get_information_set(self, hole_card, round_state) -> str:
        street = round_state['street']
        pot_size = round_state['pot']['main']['amount']
        
        hand_strength = self._calculate_hand_strength(hole_card, round_state.get('community_card', []))
        
        action_history = self._get_action_history(round_state)
        
        info_set = f"{street}_{hand_strength:.2f}_{pot_size}_{action_history}"
        return info_set
    
    def _calculate_hand_strength(self, hole_card, community_card) -> float:
        try:
            if len(community_card) >= 3:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card),
                    community_card=gen_cards(community_card)
                )
            else:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card)
                )
            return win_rate
        except:
            return 0.5
    
    def _get_action_history(self, round_state) -> str:
        history = []
        for action in round_state.get('action_histories', {}).get(round_state['street'], []):
            history.append(action['action'][0])
        return ''.join(history[-5:])
    
    def _get_strategy(self, information_set: str, valid_actions: List[Dict]) -> Dict[str, float]:
        strategy = {}
        
        for action in valid_actions:
            action_name = action['action']
            regret = self.regret_sum[information_set][action_name]
            strategy[action_name] = max(regret, 0)
        
        total_strategy = sum(strategy.values())
        
        if total_strategy > 0:
            for action_name in strategy:
                strategy[action_name] /= total_strategy
        else:
            uniform_prob = 1.0 / len(valid_actions)
            for action in valid_actions:
                strategy[action['action']] = uniform_prob
        
        return strategy
    
    def update_regrets(self, information_set: str, action_utilities: Dict[str, float], counterfactual_value: float):
        for action, utility in action_utilities.items():
            regret = utility - counterfactual_value
            self.regret_sum[information_set][action] += regret
    
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class DeepCFRAgent(BasePokerPlayer):
    def __init__(self, name: str = "DeepCFR_Agent", hidden_dim: int = 256):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim
        
        self.value_network = self._build_value_network()
        self.strategy_network = self._build_strategy_network()
        
        self.optimizer_value = torch.optim.Adam(self.value_network.parameters(), lr=0.001)
        self.optimizer_strategy = torch.optim.Adam(self.strategy_network.parameters(), lr=0.001)
        
        self.memory = []
        self.training_step = 0
        
        logger.info(f"DeepCFRAgent {name} initialized with hidden_dim={hidden_dim}")
    
    def _build_value_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def _build_strategy_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4),
            nn.Softmax(dim=-1)
        )
    
    def declare_action(self, valid_actions, hole_card, round_state):
        state_features = self._encode_state(hole_card, round_state)
        
        with torch.no_grad():
            strategy = self.strategy_network(state_features)
        
        action_probs = self._map_strategy_to_actions(strategy, valid_actions)
        
        chosen_action_idx = np.random.choice(len(valid_actions), p=action_probs)
        chosen_action = valid_actions[chosen_action_idx]
        
        return chosen_action['action'], chosen_action['amount']
    
    def _encode_state(self, hole_card, round_state) -> torch.Tensor:
        features = torch.zeros(64)
        
        hand_strength = self._calculate_hand_strength(hole_card, round_state.get('community_card', []))
        features[0] = hand_strength
        
        pot_size = round_state['pot']['main']['amount']
        features[1] = min(pot_size / 1000.0, 1.0)
        
        street_encoding = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street_idx = street_encoding.get(round_state['street'], 0)
        features[2 + street_idx] = 1.0
        
        action_history = self._encode_action_history(round_state)
        features[6:6+len(action_history)] = torch.tensor(action_history[:58])
        
        return features.unsqueeze(0)
    
    def _calculate_hand_strength(self, hole_card, community_card) -> float:
        try:
            if len(community_card) >= 3:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card),
                    community_card=gen_cards(community_card)
                )
            else:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card)
                )
            return win_rate
        except:
            return 0.5
    
    def _encode_action_history(self, round_state) -> List[float]:
        action_encoding = {'fold': 0.0, 'call': 0.33, 'raise': 0.66, 'check': 1.0}
        
        history = []
        for street_actions in round_state.get('action_histories', {}).values():
            for action in street_actions:
                action_name = action['action']
                history.append(action_encoding.get(action_name, 0.5))
        
        return history[-58:] if len(history) > 58 else history + [0.0] * (58 - len(history))
    
    def _map_strategy_to_actions(self, strategy: torch.Tensor, valid_actions: List[Dict]) -> List[float]:
        strategy_array = strategy.squeeze().numpy()
        
        action_mapping = {'fold': 0, 'call': 1, 'raise': 2, 'check': 3}
        
        action_probs = []
        for action in valid_actions:
            action_idx = action_mapping.get(action['action'], 3)
            action_probs.append(strategy_array[action_idx])
        
        total_prob = sum(action_probs)
        if total_prob > 0:
            action_probs = [p / total_prob for p in action_probs]
        else:
            action_probs = [1.0 / len(valid_actions)] * len(valid_actions)
        
        return action_probs
    
    def train_networks(self, batch_size: int = 32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        states = torch.stack([item['state'] for item in batch])
        values = torch.tensor([item['value'] for item in batch], dtype=torch.float32)
        strategies = torch.stack([item['strategy'] for item in batch])
        
        predicted_values = self.value_network(states).squeeze()
        value_loss = F.mse_loss(predicted_values, values)
        
        predicted_strategies = self.strategy_network(states)
        strategy_loss = F.kl_div(F.log_softmax(predicted_strategies, dim=-1), strategies, reduction='batchmean')
        
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()
        
        self.optimizer_strategy.zero_grad()
        strategy_loss.backward()
        self.optimizer_strategy.step()
        
        self.training_step += 1
    
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class NFSPAgent(BasePokerPlayer):
    def __init__(self, name: str = "NFSP_Agent", hidden_dim: int = 256, epsilon: float = 0.1):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        
        self.q_network = self._build_q_network()
        self.target_q_network = self._build_q_network()
        self.policy_network = self._build_policy_network()
        
        self.optimizer_q = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.optimizer_policy = torch.optim.Adam(self.policy_network.parameters(), lr=0.001)
        
        self.replay_buffer = []
        self.policy_buffer = []
        self.update_target_frequency = 1000
        self.training_step = 0
        
        logger.info(f"NFSPAgent {name} initialized with epsilon={epsilon}")
    
    def _build_q_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4)
        )
    
    def _build_policy_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 4),
            nn.Softmax(dim=-1)
        )
    
    def declare_action(self, valid_actions, hole_card, round_state):
        state_features = self._encode_state(hole_card, round_state)
        
        if random.random() < self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state_features)
            action_probs = self._q_values_to_action_probs(q_values, valid_actions)
        else:
            with torch.no_grad():
                policy = self.policy_network(state_features)
            action_probs = self._map_policy_to_actions(policy, valid_actions)
        
        chosen_action_idx = np.random.choice(len(valid_actions), p=action_probs)
        chosen_action = valid_actions[chosen_action_idx]
        
        return chosen_action['action'], chosen_action['amount']
    
    def _encode_state(self, hole_card, round_state) -> torch.Tensor:
        features = torch.zeros(64)
        
        hand_strength = self._calculate_hand_strength(hole_card, round_state.get('community_card', []))
        features[0] = hand_strength
        
        pot_size = round_state['pot']['main']['amount']
        features[1] = min(pot_size / 1000.0, 1.0)
        
        street_encoding = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street_idx = street_encoding.get(round_state['street'], 0)
        features[2 + street_idx] = 1.0
        
        return features.unsqueeze(0)
    
    def _calculate_hand_strength(self, hole_card, community_card) -> float:
        try:
            if len(community_card) >= 3:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card),
                    community_card=gen_cards(community_card)
                )
            else:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card)
                )
            return win_rate
        except:
            return 0.5
    
    def _q_values_to_action_probs(self, q_values: torch.Tensor, valid_actions: List[Dict]) -> List[float]:
        q_array = q_values.squeeze().numpy()
        
        action_mapping = {'fold': 0, 'call': 1, 'raise': 2, 'check': 3}
        
        valid_q_values = []
        for action in valid_actions:
            action_idx = action_mapping.get(action['action'], 3)
            valid_q_values.append(q_array[action_idx])
        
        exp_q = np.exp(np.array(valid_q_values) - np.max(valid_q_values))
        action_probs = exp_q / np.sum(exp_q)
        
        return action_probs.tolist()
    
    def _map_policy_to_actions(self, policy: torch.Tensor, valid_actions: List[Dict]) -> List[float]:
        policy_array = policy.squeeze().numpy()
        
        action_mapping = {'fold': 0, 'call': 1, 'raise': 2, 'check': 3}
        
        action_probs = []
        for action in valid_actions:
            action_idx = action_mapping.get(action['action'], 3)
            action_probs.append(policy_array[action_idx])
        
        total_prob = sum(action_probs)
        if total_prob > 0:
            action_probs = [p / total_prob for p in action_probs]
        else:
            action_probs = [1.0 / len(valid_actions)] * len(valid_actions)
        
        return action_probs
    
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class PluriBusStyleAgent(BasePokerPlayer):
    def __init__(self, name: str = "Pluribus_Agent", abstraction_levels: int = 3):
        super().__init__()
        self.name = name
        self.abstraction_levels = abstraction_levels
        
        self.strategy_abstraction = self._build_strategy_abstraction()
        self.blueprint_strategy = defaultdict(lambda: defaultdict(float))
        
        self.search_depth = 3
        self.monte_carlo_samples = 100
        
        logger.info(f"PluriBusStyleAgent {name} initialized with {abstraction_levels} abstraction levels")
    
    def _build_strategy_abstraction(self) -> Dict[str, List[str]]:
        return {
            'preflop': ['tight', 'loose', 'aggressive'],
            'flop': ['conservative', 'balanced', 'aggressive'],
            'turn': ['value', 'bluff', 'check'],
            'river': ['value', 'bluff', 'fold']
        }
    
    def declare_action(self, valid_actions, hole_card, round_state):
        abstract_state = self._abstract_game_state(hole_card, round_state)
        
        if self._should_use_blueprint(round_state):
            action_probs = self._get_blueprint_strategy(abstract_state, valid_actions)
        else:
            action_probs = self._monte_carlo_search(hole_card, round_state, valid_actions)
        
        chosen_action_idx = np.random.choice(len(valid_actions), p=action_probs)
        chosen_action = valid_actions[chosen_action_idx]
        
        return chosen_action['action'], chosen_action['amount']
    
    def _abstract_game_state(self, hole_card, round_state) -> str:
        hand_strength = self._calculate_hand_strength(hole_card, round_state.get('community_card', []))
        
        if hand_strength > 0.7:
            strength_bucket = 'strong'
        elif hand_strength > 0.4:
            strength_bucket = 'medium'
        else:
            strength_bucket = 'weak'
        
        pot_size = round_state['pot']['main']['amount']
        pot_bucket = 'small' if pot_size < 20 else 'medium' if pot_size < 100 else 'large'
        
        street = round_state['street']
        
        return f"{street}_{strength_bucket}_{pot_bucket}"
    
    def _calculate_hand_strength(self, hole_card, community_card) -> float:
        try:
            if len(community_card) >= 3:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card),
                    community_card=gen_cards(community_card)
                )
            else:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card)
                )
            return win_rate
        except:
            return 0.5
    
    def _should_use_blueprint(self, round_state) -> bool:
        pot_size = round_state['pot']['main']['amount']
        return pot_size < 50
    
    def _get_blueprint_strategy(self, abstract_state: str, valid_actions: List[Dict]) -> List[float]:
        if abstract_state not in self.blueprint_strategy:
            self._initialize_blueprint_strategy(abstract_state, valid_actions)
        
        strategy = self.blueprint_strategy[abstract_state]
        
        action_probs = []
        for action in valid_actions:
            action_name = action['action']
            action_probs.append(strategy.get(action_name, 0.0))
        
        total_prob = sum(action_probs)
        if total_prob > 0:
            action_probs = [p / total_prob for p in action_probs]
        else:
            action_probs = [1.0 / len(valid_actions)] * len(valid_actions)
        
        return action_probs
    
    def _initialize_blueprint_strategy(self, abstract_state: str, valid_actions: List[Dict]):
        strategy = {}
        
        if 'strong' in abstract_state:
            strategy = {'fold': 0.05, 'call': 0.25, 'raise': 0.65, 'check': 0.05}
        elif 'medium' in abstract_state:
            strategy = {'fold': 0.15, 'call': 0.55, 'raise': 0.25, 'check': 0.05}
        else:
            strategy = {'fold': 0.60, 'call': 0.25, 'raise': 0.10, 'check': 0.05}
        
        valid_action_names = [action['action'] for action in valid_actions]
        filtered_strategy = {k: v for k, v in strategy.items() if k in valid_action_names}
        
        total_prob = sum(filtered_strategy.values())
        if total_prob > 0:
            for action_name in filtered_strategy:
                filtered_strategy[action_name] /= total_prob
        
        self.blueprint_strategy[abstract_state] = filtered_strategy
    
    def _monte_carlo_search(self, hole_card, round_state, valid_actions: List[Dict]) -> List[float]:
        action_values = {}
        
        for action in valid_actions:
            action_name = action['action']
            total_value = 0.0
            
            for _ in range(self.monte_carlo_samples):
                value = self._simulate_action_outcome(hole_card, round_state, action)
                total_value += value
            
            action_values[action_name] = total_value / self.monte_carlo_samples
        
        values = list(action_values.values())
        if max(values) == min(values):
            return [1.0 / len(valid_actions)] * len(valid_actions)
        
        exp_values = np.exp(np.array(values) - np.max(values))
        action_probs = exp_values / np.sum(exp_values)
        
        return action_probs.tolist()
    
    def _simulate_action_outcome(self, hole_card, round_state, action: Dict) -> float:
        hand_strength = self._calculate_hand_strength(hole_card, round_state.get('community_card', []))
        
        pot_size = round_state['pot']['main']['amount']
        
        if action['action'] == 'fold':
            return -pot_size * 0.1
        elif action['action'] == 'call':
            return hand_strength * pot_size - action.get('amount', 0)
        elif action['action'] == 'raise':
            return hand_strength * pot_size * 1.5 - action.get('amount', 0)
        else:
            return hand_strength * pot_size
    
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class CommercialBotInterface(BasePokerPlayer):
    def __init__(self, name: str = "Commercial_Bot", bot_type: str = "aggressive"):
        super().__init__()
        self.name = name
        self.bot_type = bot_type
        
        self.strategy_profiles = {
            'aggressive': {'fold': 0.15, 'call': 0.25, 'raise': 0.55, 'check': 0.05},
            'conservative': {'fold': 0.35, 'call': 0.45, 'raise': 0.15, 'check': 0.05},
            'balanced': {'fold': 0.25, 'call': 0.40, 'raise': 0.30, 'check': 0.05}
        }
        
        self.adaptation_factor = 0.1
        self.opponent_model = defaultdict(lambda: defaultdict(int))
        
        logger.info(f"CommercialBotInterface {name} initialized as {bot_type}")
    
    def declare_action(self, valid_actions, hole_card, round_state):
        hand_strength = self._calculate_hand_strength(hole_card, round_state.get('community_card', []))
        
        base_strategy = self.strategy_profiles[self.bot_type].copy()
        
        adapted_strategy = self._adapt_strategy_to_opponent(base_strategy, round_state)
        
        hand_adjusted_strategy = self._adjust_for_hand_strength(adapted_strategy, hand_strength)
        
        action_probs = self._map_strategy_to_actions(hand_adjusted_strategy, valid_actions)
        
        chosen_action_idx = np.random.choice(len(valid_actions), p=action_probs)
        chosen_action = valid_actions[chosen_action_idx]
        
        return chosen_action['action'], chosen_action['amount']
    
    def _calculate_hand_strength(self, hole_card, community_card) -> float:
        try:
            if len(community_card) >= 3:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card),
                    community_card=gen_cards(community_card)
                )
            else:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=gen_cards(hole_card)
                )
            return win_rate
        except:
            return 0.5
    
    def _adapt_strategy_to_opponent(self, base_strategy: Dict[str, float], round_state) -> Dict[str, float]:
        opponent_aggression = self._estimate_opponent_aggression(round_state)
        
        adapted_strategy = base_strategy.copy()
        
        if opponent_aggression > 0.6:
            adapted_strategy['fold'] *= 1.2
            adapted_strategy['call'] *= 0.8
            adapted_strategy['raise'] *= 0.9
        elif opponent_aggression < 0.3:
            adapted_strategy['fold'] *= 0.8
            adapted_strategy['call'] *= 0.9
            adapted_strategy['raise'] *= 1.3
        
        total_prob = sum(adapted_strategy.values())
        for action in adapted_strategy:
            adapted_strategy[action] /= total_prob
        
        return adapted_strategy
    
    def _estimate_opponent_aggression(self, round_state) -> float:
        total_actions = 0
        aggressive_actions = 0
        
        for street_actions in round_state.get('action_histories', {}).values():
            for action in street_actions:
                total_actions += 1
                if action['action'] in ['raise', 'bet']:
                    aggressive_actions += 1
        
        return aggressive_actions / total_actions if total_actions > 0 else 0.3
    
    def _adjust_for_hand_strength(self, strategy: Dict[str, float], hand_strength: float) -> Dict[str, float]:
        adjusted_strategy = strategy.copy()
        
        if hand_strength > 0.8:
            adjusted_strategy['raise'] *= 1.5
            adjusted_strategy['fold'] *= 0.3
        elif hand_strength > 0.6:
            adjusted_strategy['call'] *= 1.2
            adjusted_strategy['raise'] *= 1.1
        elif hand_strength < 0.3:
            adjusted_strategy['fold'] *= 1.4
            adjusted_strategy['raise'] *= 0.5
        
        total_prob = sum(adjusted_strategy.values())
        for action in adjusted_strategy:
            adjusted_strategy[action] /= total_prob
        
        return adjusted_strategy
    
    def _map_strategy_to_actions(self, strategy: Dict[str, float], valid_actions: List[Dict]) -> List[float]:
        action_probs = []
        
        for action in valid_actions:
            action_name = action['action']
            action_probs.append(strategy.get(action_name, 0.0))
        
        total_prob = sum(action_probs)
        if total_prob > 0:
            action_probs = [p / total_prob for p in action_probs]
        else:
            action_probs = [1.0 / len(valid_actions)] * len(valid_actions)
        
        return action_probs
    
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class BaselineAgentFactory:
    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> BasePokerPlayer:
        if agent_type.lower() == 'cfr':
            return CFRAgent(**kwargs)
        elif agent_type.lower() == 'deep_cfr':
            return DeepCFRAgent(**kwargs)
        elif agent_type.lower() == 'nfsp':
            return NFSPAgent(**kwargs)
        elif agent_type.lower() == 'pluribus':
            return PluriBusStyleAgent(**kwargs)
        elif agent_type.lower() == 'commercial':
            return CommercialBotInterface(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    @staticmethod
    def get_available_agents() -> List[str]:
        return ['cfr', 'deep_cfr', 'nfsp', 'pluribus', 'commercial']
    
    @staticmethod
    def create_benchmark_suite() -> List[BasePokerPlayer]:
        return [
            CFRAgent(name="CFR_Baseline"),
            DeepCFRAgent(name="DeepCFR_Baseline"),
            NFSPAgent(name="NFSP_Baseline"),
            PluriBusStyleAgent(name="Pluribus_Baseline"),
            CommercialBotInterface(name="Commercial_Baseline", bot_type="balanced")
        ]