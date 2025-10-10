import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from dataclasses import dataclass
from tqdm import tqdm

from qagent.agents.sum_agent import SUMAgent
from qagent.environments.poker_environment import PokerEnvironment
from qagent.core.neural_architecture import SUMNeuralArchitecture
from qagent.core.loss_functions import MultiObjectiveLossManager
from pypokerengine.players import BasePokerPlayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    total_hands: int = 10_000_000
    max_epochs: int = 1000
    parallel_games: int = 64
    batch_size: int = 256
    learning_rate: float = 0.001
    save_frequency: int = 100_000
    evaluation_frequency: int = 500_000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "training_logs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_strategies: int = 8
    lambda_commitment: float = 0.3
    lambda_deception: float = 0.1
    gradient_clip_norm: float = 1.0
    target_update_frequency: int = 1000
    experience_buffer_size: int = 100_000

class ParallelGameManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.game_environments = []
        for i in range(config.parallel_games):
            env = PokerEnvironment(
                initial_stack=200,
                small_blind=1,
                big_blind=2
            )
            self.game_environments.append(env)
        
        logger.info(f"ParallelGameManager initialized with {config.parallel_games} parallel games")

    def run_parallel_games(
        self, 
        main_agent_copies: List[SUMAgent], 
        opponent_agent_copies: List[SUMAgent], 
        hands_per_game: int
    ) -> List[Dict]:
        num_games = self.config.parallel_games
        game_results = []

        logger.info(f"Running {num_games} games sequentially (hands_per_game={hands_per_game})")

        for i in range(num_games):
            env = self.game_environments[i]
            main_agent_copy = main_agent_copies[i]
            opponent_agent_copy = opponent_agent_copies[i]

            try:
                result = self._run_single_game(
                    env, main_agent_copy, opponent_agent_copy, hands_per_game, i
                )
                game_results.append(result)
            except Exception as e:
                logger.error(f"Game execution failed in game {i}: {e}", exc_info=True)
                game_results.append(self._create_error_result(i))
            
        return game_results

    def _run_single_game(self, env: PokerEnvironment, agent1: SUMAgent, agent2: SUMAgent, hands_per_game: int, game_id: int) -> Dict:
        try:
            start_time = time.time()
            actual_hands = min(hands_per_game, 5)
            
            logger.debug(f"Starting game {game_id} with {actual_hands} hands")
            result = env.run_tournament(agent1, agent2, num_games=actual_hands)
            end_time = time.time()
            execution_time = end_time - start_time
            
            if not result.get('success', False):
                logger.warning(f"Game {game_id} reported failure: {result.get('error', 'Unknown error')}")
                return self._create_error_result(game_id)
            
            result.update({
                'game_id': game_id,
                'execution_time': execution_time,
                'hands_per_game': actual_hands,
                'original_hands_requested': hands_per_game
            })
            
            logger.debug(f"Game {game_id} completed successfully in {execution_time:.2f}s with {actual_hands} hands")
            return result
            
        except Exception as e:
            logger.error(f"Error in game {game_id}: {e}")
            import traceback
            logger.error(f"Game {game_id} traceback: {traceback.format_exc()}")
            return self._create_error_result(game_id)
    
    def _create_error_result(self, game_id: int = -1) -> Dict:
        return {
            'game_id': game_id,
            'success': False,
            'error': 'Game execution failed',
            'execution_time': 0,
            'hands_per_game': 0,
            'player_results': {}
        }

class ExperienceReplayBuffer:
    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.device = torch.device(device)
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
        logger.info(f"ExperienceReplayBuffer initialized with capacity {capacity}")
    
    def add_experience(self, experience: Dict, priority: float = 1.0):
        self.buffer.append(experience)
        self.priorities.append(priority)

    def add_batch(self, experiences: List[Dict]):
        for experience in experiences:
            self.add_experience(experience)
    
    def sample_batch(self, batch_size: int, prioritized: bool = True) -> List[Dict]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if prioritized and len(self.priorities) == len(self.buffer):
            priorities = np.array(self.priorities)
            probabilities = priorities / np.sum(priorities)
            
            indices = np.random.choice(
                len(self.buffer),
                size=batch_size,
                replace=False,
                p=probabilities
            )
        else:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class SelfPlayTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        self.main_agent = SUMAgent(
            name="SUM_Main",
            num_strategies=config.num_strategies,
            device=config.device,
            learning_rate=config.learning_rate,
            lambda_commitment=config.lambda_commitment,
            lambda_deception=config.lambda_deception
        )
        
        self.target_agent = SUMAgent(
            name="SUM_Target",
            num_strategies=config.num_strategies,
            device=config.device,
            learning_rate=config.learning_rate,
            lambda_commitment=config.lambda_commitment,
            lambda_deception=config.lambda_deception
        )
        
        self.target_agent.neural_architecture.load_state_dict(
            self.main_agent.neural_architecture.state_dict()
        )
        self.target_agent.set_training_mode(False)
        
        self.game_manager = ParallelGameManager(config)
        self.experience_buffer = ExperienceReplayBuffer(
            capacity=config.experience_buffer_size,
            device=config.device
        )
        
        self.training_statistics = {
            'current_epoch': 0,
            'total_hands_played': 0,
            'total_training_steps': 0,
            'total_games': 0,
            'successful_games': 0,
            'failed_games': 0,
            'average_game_duration': 0.0,
            'win_rate_history': [],
            'loss_history': [],
            'commitment_rate_history': [],
            'strategy_diversity_history': []
        }
        
        self.performance_tracker = {
            'average_win_rate': 0.0,
            'total_loss': 0.0,
            'recent_win_rates': deque(maxlen=100),
            'recent_losses': deque(maxlen=1000),
            'hands_per_second': deque(maxlen=50)
        }
        
        logger.info(f"SelfPlayTrainer initialized on {config.device}")
    
    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting self-play training for {self.config.total_hands:,} hands")
        
        start_time = time.time()
        hands_played = 0
        training_step = 0
        
        with tqdm(total=self.config.total_hands, desc="Self-Play Training", unit="hands") as pbar:
            while self.training_statistics['total_hands_played'] < self.config.total_hands:
                pbar.set_description(f"Epoch {self.training_statistics['current_epoch']+1}/{self.config.max_epochs}")

                # Core training activities
                self._run_training_epoch(pbar)

                # Update and log performance metrics
                self._update_performance_metrics()
                self._log_training_progress()

                # Check for model improvement and save if necessary
                if self._is_model_improved():
                    self.save_checkpoint(is_best=True)

                # Prepare for the next epoch
                self.training_statistics['current_epoch'] += 1
                if self.training_statistics['current_epoch'] >= self.config.max_epochs:
                    logger.info("Maximum number of epochs reached. Ending training.")
                    break
        
            pbar.close()
        
        total_duration = time.time() - start_time
        
        final_checkpoint = self._save_final_checkpoint(hands_played, training_step, total_duration)
        
        logger.info(f"Training completed: {hands_played:,} hands in {total_duration:.2f}s")
        
        return {
            'total_hands': hands_played,
            'total_training_steps': training_step,
            'total_duration': total_duration,
            'final_checkpoint': final_checkpoint,
            'training_statistics': self.training_statistics
        }
    
    def _run_training_epoch(self, pbar: tqdm):
        epoch_start_time = time.time()
        
        # Run parallel games
        game_results, agent_copies = self._run_parallel_games()
        
        # Process results and extract experiences
        hands_this_epoch = self._process_game_results(game_results, agent_copies)
        
        # Update the main progress bar
        pbar.update(hands_this_epoch)
        
        # Perform training steps if enough experience is available
        if len(self.experience_buffer) >= self.config.batch_size:
            num_training_steps = (len(self.experience_buffer) // self.config.batch_size)
            total_loss = 0
            for _ in range(num_training_steps):
                loss_info = self._training_step()
                if 'total_loss' in loss_info:
                    total_loss += loss_info['total_loss']
            
            avg_loss = total_loss / num_training_steps if num_training_steps > 0 else 0
            self.performance_tracker['recent_losses'].append(avg_loss)
        
        # Update target network periodically
        if self.training_statistics['total_training_steps'] % self.config.target_update_frequency == 0:
            self._update_target_network()
            
        epoch_duration = time.time() - epoch_start_time
        hands_per_second = hands_this_epoch / epoch_duration if epoch_duration > 0 else 0
        self.performance_tracker['hands_per_second'].append(hands_per_second)

        # Update progress bar postfix
        pbar.set_postfix({
            "Win Rate": f"{self.performance_tracker.get('average_win_rate', 0.0):.3f}",
            "Loss": f"{np.mean(list(self.performance_tracker['recent_losses'])) if self.performance_tracker['recent_losses'] else 0.0:.4f}",
            "HPS": f"{hands_per_second:.1f}",
            "Buffer": f"{len(self.experience_buffer)}",
        })

    def _run_parallel_games(self) -> Tuple[List[Dict], List[SUMAgent]]:
        hands_per_game = min(20, max(5, self.config.total_hands // (self.config.parallel_games * 50)))
        
        main_agent_copies = [
            self._create_lightweight_agent_copy(self.main_agent, f"{self.main_agent.name}_game_{i}")
            for i in range(self.config.parallel_games)
        ]
        
        opponent_agent_copies = [
            self._create_lightweight_agent_copy(self._create_opponent_agent(), f"SUM_Opponent_game_{i}")
            for i in range(self.config.parallel_games)
        ]
        
        game_results = self.game_manager.run_parallel_games(
            main_agent_copies,
            opponent_agent_copies,
            hands_per_game=hands_per_game
        )
        
        return game_results, main_agent_copies

    def _create_lightweight_agent_copy(self, original_agent: SUMAgent, new_name: str) -> SUMAgent:
        copy_agent = SUMAgent(
            name=new_name,
            num_strategies=original_agent.num_strategies,
            device=original_agent.device,
            is_copy=True
        )
        copy_agent.neural_architecture.load_state_dict(original_agent.neural_architecture.state_dict())
        copy_agent.set_training_mode(True)  # Set to training mode to collect experience
        return copy_agent
    
    def _create_opponent_agent(self) -> SUMAgent:
        opponent = SUMAgent(
            name="SUM_Opponent",
            num_strategies=self.config.num_strategies,
            device=self.config.device
        )
        
        if np.random.random() < 0.8:
            opponent.neural_architecture.load_state_dict(
                self.target_agent.neural_architecture.state_dict()
            )
        else:
            opponent.neural_architecture.load_state_dict(
                self.main_agent.neural_architecture.state_dict()
            )
        
        opponent.set_training_mode(False)
        
        return opponent
    
    def _process_game_results(self, game_results: List[Dict], agent_copies: List[SUMAgent]) -> int:
        logger.info(f"Processing {len(game_results)} game results...")
        hands_this_epoch = 0
        successful_games = 0
        win_rates_this_epoch = []
        
        for result in game_results:
            if result.get('success', False):
                hands_in_game = result.get('total_hands', 0)
                hands_this_epoch += hands_in_game
                successful_games += 1

                player_results = result.get('player_results', {})
                # Assuming player_0 is always the main agent
                if 'player_0' in player_results:
                    win_rate = player_results['player_0'].get('win_rate', 0.5)
                    win_rates_this_epoch.append(win_rate)
            else:
                logger.warning(f"Skipping failed game: {result.get('game_id', 'N/A')}")

        if win_rates_this_epoch:
            self.performance_tracker['recent_win_rates'].extend(win_rates_this_epoch)

        self.training_statistics['total_hands_played'] += hands_this_epoch
        self.training_statistics['successful_games'] += successful_games
        self.training_statistics['failed_games'] += len(game_results) - successful_games
        self.training_statistics['total_games'] += len(game_results)
        
        # After all games in the epoch are processed, extract experiences from the agent copies
        self._extract_experiences_from_agents(agent_copies)
        
        return hands_this_epoch

    def _extract_experiences_from_agents(self, agent_copies: List[SUMAgent]):
        """
        Extracts all experiences from the agent copies' buffers and moves them to the trainer's buffer.
        """
        total_experiences = 0
        for agent in agent_copies:
            if not hasattr(agent, 'experience_buffer'):
                logger.warning(f"Agent {agent.name} does not have an experience_buffer attribute.")
                continue

            agent_buffer = agent.experience_buffer
            if not agent_buffer:
                logger.warning(f"Agent {agent.name}'s experience buffer is empty. No experiences to extract.")
                continue

            experiences = list(agent_buffer)
            self.experience_buffer.add_batch(experiences)
            total_experiences += len(experiences)
            
            agent_buffer.clear()
        
        if total_experiences > 0:
            logger.info(f"Extracted {total_experiences} experiences from {len(agent_copies)} agents.")
        else:
            logger.warning("No experiences were extracted from any agent.")

    def _extract_experiences_from_agent(self):
        """
        Extracts all experiences from the main agent's buffer and moves them to the trainer's buffer.
        This should be called once per training epoch, after all parallel games have finished.
        """
        if not hasattr(self.main_agent, 'experience_buffer'):
            logger.warning("Main agent does not have an experience_buffer attribute.")
            return

        agent_buffer = self.main_agent.experience_buffer
        if not agent_buffer:
            logger.warning("Agent's experience buffer is empty. No experiences to extract.")
            return

        new_experiences = list(agent_buffer)
        agent_buffer.clear()

        self.experience_buffer.extend(new_experiences)
        
        logger.info(f"Moved {len(new_experiences)} experiences from agent to trainer's buffer.")
        logger.info(f"Trainer experience buffer size: {len(self.experience_buffer)}")


    def _calculate_experience_priority(self, experience: Dict) -> float:
        # Placeholder for priority calculation
        base_priority = 1.0
        
        player_results = game_result.get('player_results', {})
        main_result = player_results.get(self.main_agent.name, {})
        winnings = main_result.get('winnings', 0)
        
        if winnings > 0:
            base_priority *= 1.5
        elif winnings < 0:
            base_priority *= 1.2
        
        model_outputs = experience.get('model_outputs', {})
        uncertainty = model_outputs.get('uncertainty', torch.tensor(0.5))
        if torch.is_tensor(uncertainty):
            uncertainty_value = uncertainty.item()
            if uncertainty_value > 0.7:
                base_priority *= 1.3
        
        return base_priority
    
    def _training_step(self) -> Dict[str, float]:
        batch = self.experience_buffer.sample_batch(
            self.config.batch_size,
            prioritized=True
        )
        
        if len(batch) < self.config.batch_size // 2:
            return {'error': 'Insufficient batch size'}
        
        try:
            loss_info = self.main_agent.train_step(batch, len(batch))
            
            self.training_statistics['total_training_steps'] += 1
            self.training_statistics['loss_history'].append(loss_info)
            
            return loss_info
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return {'error': str(e)}
    
    def _update_target_network(self):
        self.target_agent.neural_architecture.load_state_dict(
            self.main_agent.neural_architecture.state_dict()
        )
        logger.info("Target network updated")
    
    def _evaluate_progress(self, hands_played: int):
        logger.info(f"Evaluating progress at {hands_played:,} hands")
        
        commitment_stats = self.main_agent.neural_architecture.get_commitment_stats()
        commitment_rate = commitment_stats.get('commitment_rate', 0.0)
        
        self.training_statistics['commitment_rate_history'].append(commitment_rate)
        
        strategy_analysis = self.main_agent.get_strategy_analysis()
        
        evaluation_summary = {
            'hands_played': hands_played,
            'commitment_rate': commitment_rate,
            'recent_win_rate': np.mean(list(self.performance_tracker['recent_win_rates'])) if self.performance_tracker['recent_win_rates'] else 0.5,
            'average_hands_per_second': np.mean(list(self.performance_tracker['hands_per_second'])) if self.performance_tracker['hands_per_second'] else 0,
            'strategy_analysis': strategy_analysis
        }
        
        evaluation_file = os.path.join(
            self.config.log_dir,
            f"evaluation_{hands_played}.json"
        )
        
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2, default=str)
        
        logger.info(f"Evaluation saved: {evaluation_file}")
    
    def _save_checkpoint(self, hands_played: int, training_step: int):
        checkpoint_data = {
            'hands_played': hands_played,
            'training_step': training_step,
            'model_state_dict': self.main_agent.neural_architecture.state_dict(),
            'optimizer_state_dict': self.main_agent.optimizer.state_dict(),
            'target_model_state_dict': self.target_agent.neural_architecture.state_dict(),
            'training_statistics': self.training_statistics,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_hands_{hands_played}.pt"
        )
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Public method to save a checkpoint, typically called when model improves"""
        hands_played = self.training_statistics.get('total_hands_played', 0)
        training_step = self.training_statistics.get('total_training_steps', 0)
        
        # Use the existing _save_checkpoint method
        self._save_checkpoint(hands_played, training_step)
        
        # If this is the best model so far, also save it as the best model
        if is_best:
            best_model_path = os.path.join(
                self.config.checkpoint_dir,
                "best_model.pt"
            )
            checkpoint_data = {
                'hands_played': hands_played,
                'training_step': training_step,
                'model_state_dict': self.main_agent.neural_architecture.state_dict(),
                'optimizer_state_dict': self.main_agent.optimizer.state_dict(),
                'target_model_state_dict': self.target_agent.neural_architecture.state_dict(),
                'training_statistics': self.training_statistics,
                'config': self.config.__dict__
            }
            torch.save(checkpoint_data, best_model_path)
            logger.info(f"Best model saved: {best_model_path}")
    
    def _save_final_checkpoint(self, hands_played: int, training_step: int, total_duration: float) -> str:
        final_checkpoint = {
            'hands_played': hands_played,
            'training_step': training_step,
            'total_duration': total_duration,
            'model_state_dict': self.main_agent.neural_architecture.state_dict(),
            'optimizer_state_dict': self.main_agent.optimizer.state_dict(),
            'training_statistics': self.training_statistics,
            'config': self.config.__dict__,
            'final_performance': {
                'average_win_rate': np.mean(list(self.performance_tracker['recent_win_rates'])) if self.performance_tracker['recent_win_rates'] else 0.5,
                'average_hands_per_second': np.mean(list(self.performance_tracker['hands_per_second'])) if self.performance_tracker['hands_per_second'] else 0,
                'commitment_rate': self.main_agent.neural_architecture.get_commitment_stats().get('commitment_rate', 0.0)
            }
        }
        
        final_path = os.path.join(
            self.config.checkpoint_dir,
            f"final_model_{int(time.time())}.pt"
        )
        
        torch.save(final_checkpoint, final_path)
        logger.info(f"Final checkpoint saved: {final_path}")
        
        return final_path

    def _update_performance_metrics(self):
        # Calculate average win rate if available
        if self.performance_tracker['recent_win_rates']:
            avg_win_rate = np.mean(self.performance_tracker['recent_win_rates'])
            self.performance_tracker['average_win_rate'] = avg_win_rate
            # Reset for the next evaluation period
            self.performance_tracker['recent_win_rates'] = []
        else:
            avg_win_rate = self.performance_tracker.get('average_win_rate', 0.0)

        # Update other metrics as needed
        # For example, you could calculate average loss, explore/exploit ratio, etc.
        # This is where you would add more detailed metric calculations.
        pass

    def _log_training_progress(self):
        logger.info(f"Epoch {self.training_statistics['current_epoch']} Summary:")
        logger.info(f"  Total Hands: {self.training_statistics['total_hands_played']}")
        logger.info(f"  Average Win Rate: {self.performance_tracker.get('average_win_rate', 'N/A'):.4f}")
        logger.info(f"  Total Loss: {self.performance_tracker.get('total_loss', 'N/A')}")
        logger.info(f"  Successful Games: {self.training_statistics['successful_games']}")
        logger.info(f"  Failed Games: {self.training_statistics['failed_games']}")

    def _is_model_improved(self) -> bool:
        # Placeholder for improvement logic
        return True
    
    def _log_progress(self, hands_played: int, training_step: int, hands_per_second: float):
        if hands_played % 50000 == 0 or hands_played < 1000:
            recent_win_rate = np.mean(list(self.performance_tracker['recent_win_rates'])) if self.performance_tracker['recent_win_rates'] else 0.5
            
            recent_losses = list(self.performance_tracker['recent_losses'])[-10:] if self.performance_tracker['recent_losses'] else []
            avg_loss = np.mean([loss.get('total_loss', 0) for loss in recent_losses if isinstance(loss, dict)]) if recent_losses else 0
            
            progress_pct = (hands_played / self.config.total_hands) * 100
            
            logger.info(
                f"Progress: {progress_pct:.1f}% | "
                f"Hands: {hands_played:,}/{self.config.total_hands:,} | "
                f"Steps: {training_step:,} | "
                f"Win Rate: {recent_win_rate:.3f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Speed: {hands_per_second:.1f} hands/s"
            )
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.main_agent.neural_architecture.load_state_dict(checkpoint['model_state_dict'])
            self.main_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'target_model_state_dict' in checkpoint:
                self.target_agent.neural_architecture.load_state_dict(checkpoint['target_model_state_dict'])
            
            self.training_statistics = checkpoint.get('training_statistics', self.training_statistics)
            
            logger.info(f"Checkpoint loaded successfully: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        return {
            'training_statistics': self.training_statistics,
            'current_performance': {
                'recent_win_rate': np.mean(list(self.performance_tracker['recent_win_rates'])) if self.performance_tracker['recent_win_rates'] else 0.5,
                'recent_hands_per_second': np.mean(list(self.performance_tracker['hands_per_second'])) if self.performance_tracker['hands_per_second'] else 0,
                'experience_buffer_size': len(self.experience_buffer)
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.main_agent.neural_architecture.parameters()),
                'device': str(self.device),
                'num_strategies': self.config.num_strategies
            }
        }

def run_self_play_training(config: TrainingConfig = None) -> Dict[str, Any]:
    if config is None:
        config = TrainingConfig()
    
    trainer = SelfPlayTrainer(config)
    
    try:
        results = trainer.train()
        return results
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return trainer.get_training_summary()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {'error': str(e), 'summary': trainer.get_training_summary()}

if __name__ == "__main__":
    config = TrainingConfig(
        total_hands=1_000_000,
        parallel_games=32,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    results = run_self_play_training(config)
    print(json.dumps(results, indent=2, default=str))