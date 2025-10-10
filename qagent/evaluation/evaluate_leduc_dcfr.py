"""
Evaluates a trained Deep CFR agent for Leduc Hold'em against baseline bots.
"""
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qagent.environments.leduc_holdem import LeducHoldem
from qagent.agents.dcfr_leduc import DCFRAgent, RegretNet
from qagent.agents.leduc_bots import RandomBot, TightAggressiveBot, LoosePassiveBot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "leduc_dcfr_strategy_net.pt"
NUM_GAMES = 10000

def load_dcfr_agent(model_path: str, game: LeducHoldem) -> DCFRAgent:
    """Loads the DCFR agent's strategy network."""
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        print("Please run the training script for the DCFR agent first.")
        sys.exit(1)
        
    agent = DCFRAgent(game)
    agent.strategy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.strategy_net.to(DEVICE)
    agent.strategy_net.eval()
    print(f"DCFR agent loaded from {model_path}")
    return agent

def play_game(game: LeducHoldem, agent1, agent2, log_game: bool = False) -> float:
    """
    Plays a single game of Leduc Hold'em between two agents.
    Returns the payoff for agent1.
    """
    state = game.get_initial_state()
    agents = [agent1, agent2]
    game_history = []

    while not game.is_terminal(state):
        if game.is_chance_node(state):
            # Sample a single outcome for evaluation
            outcomes = game.sample_chance_outcome(state)
            _, next_s = outcomes[np.random.randint(len(outcomes))]
            if log_game:
                if state['round'] == 0 and len(next_s['private_cards']) == 2:
                    game_history.append(f"--- PRE-FLOP ---")
                    game_history.append(f"Player 0 (DCFR) dealt: {game.card_to_str(next_s['private_cards'][0])}")
                    game_history.append(f"Player 1 (Bot) dealt: {game.card_to_str(next_s['private_cards'][1])}")
                elif state['round'] == 1 and next_s.get('board_card') is not None:
                    game_history.append(f"--- FLOP ---")
                    game_history.append(f"Board card dealt: {game.card_to_str(next_s['board_card'])}")
            state = next_s
            continue

        current_player_idx = game.get_current_player(state)
        current_agent = agents[current_player_idx]
        
        action = current_agent.get_action(state)
        
        if log_game:
            agent_name = "DCFR" if agents[current_player_idx] is agent1 else "Bot"
            action_str = game.action_to_str(action)
            game_history.append(f"Player {current_player_idx} ({agent_name}) action: {action_str}")

        state = game.get_next_state(state, action)

    payoff = game.get_payoff(state, 0)

    # Log if the DCFR agent (agent1) lost money
    if log_game and payoff < 0:
        print("\n--- Losing Game Log ---")
        for entry in game_history:
            print(entry)
        final_hands = f"Final Hands -> P0 (DCFR): {game.card_to_str(state['private_cards'][0])}, P1 (Bot): {game.card_to_str(state['private_cards'][1])}"
        if state['board_card'] is not None:
            final_hands += f", Board: {game.card_to_str(state['board_card'])}"
        print(final_hands)
        print(f"Payoff for DCFR Agent: {payoff}")
        print("-----------------------\n")

    return payoff


def evaluate_agent(dcfr_agent, bot_agent, game, num_games, log_games=False):
    """Evaluates the DCFR agent against a given bot."""
    total_reward = 0
    
    # DCFR agent plays as player 0
    for i in range(num_games // 2):
        # Only log a few games to avoid spamming
        should_log = log_games and i < 5 
        total_reward += play_game(game, dcfr_agent, bot_agent, log_game=should_log)
        
    # DCFR agent plays as player 1
    for i in range(num_games // 2):
        # Only log a few games to avoid spamming
        should_log = log_games and i < 5
        # When bot is player 0, payoff is inverted for DCFR agent
        payoff = play_game(game, bot_agent, dcfr_agent, log_game=should_log)
        total_reward -= payoff
        
    return total_reward / num_games

class DCFRAgentWrapper:
    """A wrapper to make the DCFR trainer conform to the bot's get_action interface."""
    def __init__(self, agent, agent_type='baseline'):
        self.agent = agent
        self.game = agent.game
        self.agent_type = agent_type

    def get_action(self, state):
        action_probs = self.get_action_probabilities(state)
        legal_actions = self.game.get_legal_actions(state)
        
        if np.sum(action_probs) > 0:
            return np.random.choice(np.arange(self.game.num_actions), p=action_probs)
        else:
            # Fallback if strategy has zero probability for all legal actions
            return np.random.choice(legal_actions)

    def get_action_probabilities(self, state):
        """Returns the probability distribution over actions for the current state."""
        legal_actions = self.game.get_legal_actions(state)
        
        if self.agent_type == 'baseline':
            info_set_tensor = self.agent._encode_info_set(state)
            with torch.no_grad():
                strategy_logits = self.agent.strategy_net(info_set_tensor.to(DEVICE))
            strategy = torch.softmax(strategy_logits, dim=0).cpu().numpy()

        elif self.agent_type == 'lstm':
            private_card, board_card, history = self.agent._encode_info_set(state)
            with torch.no_grad():
                strategy_logits = self.agent.strategy_net(
                    private_card.to(DEVICE), 
                    board_card.to(DEVICE), 
                    history.to(DEVICE)
                )
            strategy = torch.softmax(strategy_logits.squeeze(0), dim=0).cpu().numpy()

        elif self.agent_type == 'transformer':
            private_card, board_card, history = self.agent._encode_info_set(state)
            with torch.no_grad():
                strategy_logits = self.agent.strategy_net(
                    private_card.to(DEVICE), 
                    board_card.to(DEVICE), 
                    history.to(DEVICE)
                )
            strategy = torch.softmax(strategy_logits.squeeze(0), dim=0).cpu().numpy()
            
        else:
            raise ValueError(f"Unknown agent type in wrapper: {self.agent_type}")

        legal_mask = np.zeros(self.game.num_actions)
        legal_mask[legal_actions] = 1
        masked_strategy = strategy * legal_mask
        
        prob_sum = np.sum(masked_strategy)
        if prob_sum > 0:
            return masked_strategy / prob_sum
        else:
            # Fallback to uniform
            final_strategy = np.zeros(self.game.num_actions)
            if legal_actions:
                final_strategy[legal_actions] = 1.0 / len(legal_actions)
            return final_strategy


def main():
    game = LeducHoldem()
    
    # Load the trained DCFR agent
    dcfr_trainer = load_dcfr_agent(MODEL_PATH, game)
    dcfr_agent = DCFRAgentWrapper(dcfr_trainer)
    
    # Define the bots to evaluate against
    bots = {
        "RandomBot": RandomBot(game),
        "TightAggressiveBot": TightAggressiveBot(game),
        "LoosePassiveBot": LoosePassiveBot(game)
    }
    
    print(f"\n--- Starting Evaluation ({NUM_GAMES} games per bot) ---")
    
    results = {}
    for bot_name, bot_agent in bots.items():
        print(f"Evaluating against {bot_name}...")
        # Enable logging only for the TightAggressiveBot
        log_this_match = (bot_name == "TightAggressiveBot")
        avg_reward = evaluate_agent(dcfr_agent, bot_agent, game, NUM_GAMES, log_games=log_this_match)
        results[bot_name] = avg_reward
        print(f"Average reward vs {bot_name}: {avg_reward:.4f}")
        
    print("\n--- Evaluation Complete ---")
    for bot_name, avg_reward in results.items():
        print(f"DCFR Agent vs {bot_name}: {avg_reward:.4f} avg reward per game.")

if __name__ == "__main__":
    main()
