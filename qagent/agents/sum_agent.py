import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
import json
import time
from collections import deque

from qagent.core.neural_architecture import SUMNeuralArchitecture
from qagent.core.loss_functions import MultiObjectiveLossManager
from qagent.environments.poker_environment import PokerEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SUMAgent(BasePokerPlayer):
    def __init__(self, 
                 name: str = "SUM_Agent",
                 num_strategies: int = 8,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 0.001,
                 lambda_commitment: float = 0.3,
                 lambda_deception: float = 0.1,
                 is_copy: bool = False):
        
        super().__init__()
        self.name = name
        self.device = torch.device(device)
        self.num_strategies = num_strategies
        
        self.neural_architecture = SUMNeuralArchitecture(
            num_strategies=num_strategies
        ).to(self.device)
        
        self.poker_env = PokerEnvironment()
        self.training_mode = True
        self.experience_buffer = deque(maxlen=10000)

        if not is_copy:
            self.loss_manager = MultiObjectiveLossManager(
                lambda_commitment=lambda_commitment,
                lambda_deception=lambda_deception,
                use_adaptive_weights=True
            )
            
            self.optimizer = torch.optim.Adam(
                self.neural_architecture.parameters(),
                lr=learning_rate,
                weight_decay=1e-5
            )
            self.performance_history = deque(maxlen=100)
        
        self.game_statistics = {
            'total_hands': 0,
            'wins': 0,
            'losses': 0,
            'total_winnings': 0,
            'commitment_decisions': 0,
            'successful_bluffs': 0,
            'strategy_switches': 0
        }
        
        self.current_round_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'commitments': [],
            'hand_strength': []
        }
        
        logger.info(f"SUMAgent {name} initialized with {num_strategies} strategies on {device}")
    
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            game_state = self.poker_env.encode_game_state(round_state, self._get_player_id(round_state))
            action_history = self.poker_env.encode_action_history(self._extract_action_history(round_state))
            
            for key in game_state:
                if torch.is_tensor(game_state[key]):
                    game_state[key] = game_state[key].to(self.device)
            
            for key in action_history:
                if torch.is_tensor(action_history[key]):
                    action_history[key] = action_history[key].to(self.device)
            
            if self.training_mode:
                action_tensor, model_outputs = self.neural_architecture.get_action(game_state, action_history)
                
                self._store_experience(game_state, action_history, model_outputs, valid_actions)
            else:
                with torch.no_grad():
                    action_tensor, model_outputs = self.neural_architecture.get_action(game_state, action_history)
            
            chosen_action = self.poker_env.decode_action(action_tensor, valid_actions)
            
            self._update_statistics(model_outputs, chosen_action)
            
            # Return action string and amount integer as expected by PyPokerEngine
            action_name = chosen_action['action']
            action_amount = chosen_action.get('amount', 0)
            
            # Ensure amount is an integer (should already be normalized by decode_action)
            action_amount = int(action_amount) if action_amount is not None else 0
            
            logger.debug(f"Agent {self.name} declaring action: {action_name}, amount: {action_amount} (type: {type(action_amount)})")
            
            return action_name, action_amount
            
        except Exception as e:
            logger.error(f"Error in declare_action: {e}")
            return valid_actions[0]['action'], valid_actions[0]['amount']
    
    def _get_player_id(self, round_state) -> int:
        try:
            # Find this agent's position in the seats
            seats = round_state.get('seats', [])
            for i, seat in enumerate(seats):
                if seat.get('name') == self.name or seat.get('uuid') == getattr(self, 'uuid', None):
                    return i
            # Fallback to next_player or 0
            return round_state.get('next_player', 0)
        except:
            return 0
    
    def _extract_action_history(self, round_state) -> List[Dict]:
        history = []
        
        try:
            action_histories = round_state.get('action_histories', {})
            
            for street, actions in action_histories.items():
                for action in actions:
                    history.append({
                        'action': action['action'],
                        'amount': action.get('amount', 0),
                        'street': street
                    })
            
            return history[-20:]
            
        except Exception as e:
            logger.warning(f"Error extracting action history: {e}")
            return []
    
    def _store_experience(self, game_state: Dict, action_history: Dict, model_outputs: Dict, valid_actions: List):
        experience = {
            'game_state': {k: v.cpu() if torch.is_tensor(v) else v for k, v in game_state.items()},
            'action_history': {k: v.cpu() if torch.is_tensor(v) else v for k, v in action_history.items()},
            'model_outputs': {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_outputs.items()},
            'valid_actions': valid_actions,
            'timestamp': time.time()
        }
        
        self.experience_buffer.append(experience)
    
    def _update_statistics(self, model_outputs: Dict, chosen_action: Dict):
        self.game_statistics['total_hands'] += 1
        
        if torch.is_tensor(model_outputs.get('commitment_prob')):
            commitment_prob = model_outputs['commitment_prob'].item()
            if commitment_prob > 0.5:
                self.game_statistics['commitment_decisions'] += 1
        
        if chosen_action['action'] in ['raise', 'bet']:
            hand_strength = model_outputs.get('uncertainty', torch.tensor(0.5))
            if torch.is_tensor(hand_strength):
                hand_strength = hand_strength.item()
            
            if hand_strength < 0.3:
                self.game_statistics['successful_bluffs'] += 1
    
    def train_step(self, batch: List[Dict], batch_size: int, **kwargs) -> Dict[str, float]:
        if not batch:
            return {'error': 'Insufficient experience data'}
        
        try:
            predictions = self._forward_batch(batch)
            targets = self._create_targets(batch)
            game_context = self._create_game_context(batch)
            
            # Debug tensor shapes
            logger.debug(f"Predictions shapes: strategies={predictions['strategies'].shape}, weights={predictions['weights'].shape}")
            logger.debug(f"Targets shapes: actions={targets['actions'].shape}, commitment={targets['commitment'].shape}")
            logger.debug(f"Game context shapes: hand_strength={game_context['hand_strength'].shape}")
            
            performance_metrics = self._calculate_performance_metrics()
            
            loss_outputs = self.loss_manager.compute_loss(
                predictions, targets, game_context, performance_metrics
            )
            
            self.optimizer.zero_grad()
            loss_outputs['total_loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.neural_architecture.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            return {
                'total_loss': loss_outputs['total_loss'].item(),
                'strategy_loss': loss_outputs['strategy_loss'].item(),
                'commitment_loss': loss_outputs['commitment_loss'].item(),
                'deception_loss': loss_outputs['deception_loss'].item(),
                'regularization_loss': loss_outputs.get('regularization_loss', torch.tensor(0.0)).item()
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Error in train_step: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _sample_batch(self, batch_size: int) -> List[Dict]:
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        return [self.experience_buffer[i] for i in indices]
    
    def _forward_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_game_states = {}
        batch_action_histories = {}
        
        # Process game states with consistent tensor shapes
        for key in batch[0]['game_state'].keys():
            tensors = []
            for exp in batch:
                tensor = exp['game_state'][key].to(self.device)
                # Ensure consistent shape - flatten if needed
                if tensor.dim() > 1:
                    tensor = tensor.flatten()
                elif tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                tensors.append(tensor)
            
            # Stack tensors with same shape
            if len(set(t.shape for t in tensors)) == 1:
                batch_game_states[key] = torch.stack(tensors)
            else:
                # Pad to same size if needed
                max_size = max(t.shape[0] for t in tensors)
                padded_tensors = []
                for t in tensors:
                    if t.shape[0] < max_size:
                        padding = torch.zeros(max_size - t.shape[0], device=self.device)
                        t = torch.cat([t, padding])
                    padded_tensors.append(t)
                batch_game_states[key] = torch.stack(padded_tensors)
        
        # Process action histories with consistent tensor shapes
        for key in batch[0]['action_history'].keys():
            tensors = []
            for exp in batch:
                tensor = exp['action_history'][key].to(self.device)
                # Ensure consistent shape - flatten if needed
                if tensor.dim() > 1:
                    tensor = tensor.flatten()
                elif tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                tensors.append(tensor)
            
            # Stack tensors with same shape
            if len(set(t.shape for t in tensors)) == 1:
                batch_action_histories[key] = torch.stack(tensors)
            else:
                # Pad to same size if needed
                max_size = max(t.shape[0] for t in tensors)
                padded_tensors = []
                for t in tensors:
                    if t.shape[0] < max_size:
                        padding = torch.zeros(max_size - t.shape[0], device=self.device)
                        t = torch.cat([t, padding])
                    padded_tensors.append(t)
                batch_action_histories[key] = torch.stack(padded_tensors)
        
        return self.neural_architecture(batch_game_states, batch_action_histories)
    
    def _create_targets(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        
        action_targets = torch.zeros(batch_size, 4, device=self.device)
        commitment_targets = torch.zeros(batch_size, device=self.device)
        
        for i, exp in enumerate(batch):
            valid_actions = exp['valid_actions']
            
            uniform_prob = 1.0 / len(valid_actions)
            for j, action in enumerate(valid_actions):
                action_mapping = {'fold': 0, 'call': 1, 'raise': 2, 'check': 3}
                action_idx = action_mapping.get(action['action'], 3)
                action_targets[i, action_idx] = uniform_prob
            
            hand_strength = exp['game_state'].get('hand_strength', torch.tensor(0.5))
            if torch.is_tensor(hand_strength):
                # Handle different tensor dimensions
                if hand_strength.dim() > 0:
                    hand_strength = hand_strength.flatten()[0].item()
                else:
                    hand_strength = hand_strength.item()
            elif not isinstance(hand_strength, (int, float)):
                hand_strength = 0.5
            
            commitment_targets[i] = 1.0 if hand_strength > 0.6 else 0.0
        
        return {
            'actions': action_targets,
            'commitment': commitment_targets
        }
    
    def _create_game_context(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        
        hand_strengths = torch.zeros(batch_size, device=self.device)
        pot_sizes = torch.zeros(batch_size, device=self.device)
        stack_sizes = torch.zeros(batch_size, device=self.device)
        
        for i, exp in enumerate(batch):
            # Extract hand_strength with proper tensor handling
            hand_strength = exp['game_state'].get('hand_strength', torch.tensor(0.5))
            if torch.is_tensor(hand_strength):
                if hand_strength.dim() > 0:
                    hand_strengths[i] = hand_strength.flatten()[0]
                else:
                    hand_strengths[i] = hand_strength
            else:
                hand_strengths[i] = float(hand_strength)
            
            # Extract pot_size with proper tensor handling
            pot_size = exp['game_state'].get('pot_size', torch.tensor(10.0))
            if torch.is_tensor(pot_size):
                if pot_size.dim() > 0:
                    pot_sizes[i] = pot_size.flatten()[0]
                else:
                    pot_sizes[i] = pot_size
            else:
                pot_sizes[i] = float(pot_size)
            
            # Extract stack_sizes with proper tensor handling
            stack_size = exp['game_state'].get('stack_sizes', torch.tensor(200.0))
            if torch.is_tensor(stack_size):
                if stack_size.dim() > 0:
                    stack_sizes[i] = stack_size.flatten()[0]
                else:
                    stack_sizes[i] = stack_size
            else:
                stack_sizes[i] = float(stack_size)
        
        return {
            'hand_strength': hand_strengths,
            'pot_size': pot_sizes,
            'stack_sizes': stack_sizes
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        if self.game_statistics['total_hands'] == 0:
            return {'win_rate': 0.5, 'commitment_rate': 0.0, 'bluff_success_rate': 0.0}
        
        win_rate = self.game_statistics['wins'] / self.game_statistics['total_hands']
        commitment_rate = self.game_statistics['commitment_decisions'] / self.game_statistics['total_hands']
        
        bluff_attempts = max(1, self.game_statistics['total_hands'] // 10)
        bluff_success_rate = self.game_statistics['successful_bluffs'] / bluff_attempts
        
        return {
            'win_rate': win_rate,
            'commitment_rate': commitment_rate,
            'bluff_success_rate': bluff_success_rate
        }
    
    def self_play_training(self, num_episodes: int = 1000, save_frequency: int = 100):
        logger.info(f"Starting self-play training for {num_episodes} episodes")
        
        training_losses = []
        
        for episode in range(num_episodes):
            try:
                opponent = SUMAgent(
                    name=f"SUM_Opponent_{episode}",
                    num_strategies=self.num_strategies,
                    device=str(self.device)
                )
                
                opponent.neural_architecture.load_state_dict(
                    self.neural_architecture.state_dict()
                )
                opponent.training_mode = False
                
                tournament_result = self.poker_env.run_tournament(
                    self, opponent, num_games=10
                )
                
                self._process_tournament_result(tournament_result)
                
                if len(self.experience_buffer) >= 32:
                    loss_info = self.train_step(batch_size=32)
                    training_losses.append(loss_info)
                
                if episode % save_frequency == 0 and episode > 0:
                    self._save_checkpoint(episode, training_losses)
                    logger.info(f"Episode {episode}: {self._get_training_summary()}")
                
            except Exception as e:
                logger.error(f"Error in episode {episode}: {e}")
                continue
        
        logger.info("Self-play training completed")
        return training_losses
    
    def _process_tournament_result(self, result: Dict):
        if not result.get('success', False):
            return
        
        player_results = result.get('player_results', {})
        
        for player_name, player_result in player_results.items():
            if player_name == self.name:
                winnings = player_result.get('winnings', 0)
                
                if winnings > 0:
                    self.game_statistics['wins'] += 1
                else:
                    self.game_statistics['losses'] += 1
                
                self.game_statistics['total_winnings'] += winnings
        
        self.performance_history.append({
            'timestamp': time.time(),
            'winnings': self.game_statistics['total_winnings'],
            'win_rate': self.game_statistics['wins'] / max(1, self.game_statistics['total_hands']),
            'hands_played': self.game_statistics['total_hands']
        })
    
    def _save_checkpoint(self, episode: int, training_losses: List[Dict]):
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.neural_architecture.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'game_statistics': self.game_statistics,
            'training_losses': training_losses[-100:],
            'performance_history': list(self.performance_history)
        }
        
        checkpoint_path = f"checkpoints/sum_agent_episode_{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.neural_architecture.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.game_statistics = checkpoint.get('game_statistics', self.game_statistics)
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def _get_training_summary(self) -> str:
        total_hands = self.game_statistics['total_hands']
        wins = self.game_statistics['wins']
        win_rate = wins / max(1, total_hands)
        
        commitment_rate = self.game_statistics['commitment_decisions'] / max(1, total_hands)
        
        return (f"Hands: {total_hands}, Win Rate: {win_rate:.3f}, "
                f"Commitment Rate: {commitment_rate:.3f}, "
                f"Total Winnings: {self.game_statistics['total_winnings']}")
    
    def evaluate_against_baseline(self, baseline_agent, num_games: int = 1000) -> Dict[str, Any]:
        logger.info(f"Evaluating against {baseline_agent.name} for {num_games} games")
        
        self.training_mode = False
        
        start_time = time.time()
        
        tournament_result = self.poker_env.run_tournament(
            self, baseline_agent, num_games=num_games
        )
        
        end_time = time.time()
        
        evaluation_result = {
            'opponent': baseline_agent.name,
            'num_games': num_games,
            'duration': end_time - start_time,
            'tournament_result': tournament_result
        }
        
        if tournament_result.get('success', False):
            player_results = tournament_result.get('player_results', {})
            sum_result = player_results.get(self.name, {})
            
            evaluation_result.update({
                'sum_winnings': sum_result.get('winnings', 0),
                'sum_win_rate': sum_result.get('win_rate', 0.0),
                'mbb_per_100': self._calculate_mbb_per_100(sum_result.get('winnings', 0), num_games),
                'hands_per_second': tournament_result.get('hands_per_second', 0)
            })
        
        self.training_mode = True
        
        return evaluation_result
    
    def _calculate_mbb_per_100(self, winnings: int, num_games: int) -> float:
        if num_games == 0:
            return 0.0
        
        big_blind = 2
        mbb_per_100 = (winnings / (num_games / 100)) / big_blind * 1000
        
        return mbb_per_100
    
    def get_strategy_analysis(self) -> Dict[str, Any]:
        commitment_stats = self.neural_architecture.get_commitment_stats()
        
        return {
            'game_statistics': self.game_statistics,
            'commitment_stats': commitment_stats,
            'performance_history': list(self.performance_history)[-10:],
            'model_parameters': {
                'num_strategies': self.num_strategies,
                'device': str(self.device),
                'total_parameters': sum(p.numel() for p in self.neural_architecture.parameters())
            }
        }
    
    def set_training_mode(self, training: bool):
        self.training_mode = training
        if training:
            self.neural_architecture.train()
        else:
            self.neural_architecture.eval()
    
    def receive_game_start_message(self, game_info):
        self.neural_architecture.reset_commitment_history()
        self.current_round_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'commitments': [],
            'hand_strength': []
        }
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        try:
            my_uuid = None
            for seat in round_state['seats']:
                if seat['name'] == self.name:
                    my_uuid = seat['uuid']
                    break
            
            if my_uuid:
                for winner in winners:
                    if winner['uuid'] == my_uuid:
                        self.game_statistics['wins'] += 1
                        break
                else:
                    self.game_statistics['losses'] += 1
            
        except Exception as e:
            logger.warning(f"Error processing round result: {e}")