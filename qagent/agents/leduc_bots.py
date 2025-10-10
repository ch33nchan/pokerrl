
import numpy as np
import random

from qagent.environments.leduc_holdem import LeducHoldem

class BaseBot:
    def __init__(self, game=None):
        self.game = game if game is not None else LeducHoldem()

    def get_action(self, state):
        raise NotImplementedError

class RandomBot(BaseBot):
    """A bot that chooses randomly from legal actions."""
    def get_action(self, state):
        legal_actions = self.game.get_legal_actions(state)
        if not legal_actions:
            return None
        return random.choice(legal_actions)

class TightAggressiveBot(BaseBot):
    """
    A bot that plays a tight-aggressive style.
    - Folds with low-ranking cards.
    - Bets/Raises with high-ranking cards.
    - Calls with medium-ranking cards.
    """
    def get_action(self, state):
        legal_actions = self.game.get_legal_actions(state)
        if not legal_actions:
            return None

        hand_card = state['private_cards'][self.game.get_current_player(state)]
        board_card = state['board_card']

        # Simple hand strength evaluation
        hand_strength = self.evaluate_strength(hand_card, board_card)

        # Action mapping: 0=Fold, 1=Call/Check, 2=Bet/Raise
        fold_action, call_action, raise_action = 0, 1, 2

        if hand_strength == 'weak':
            # If folding is an option, fold. Otherwise, check.
            return fold_action if fold_action in legal_actions else call_action
        elif hand_strength == 'strong':
            # If raising is an option, raise. Otherwise, bet/call.
            return raise_action if raise_action in legal_actions else call_action
        else: # medium
            # Call/Check, avoid folding if possible
            return call_action if call_action in legal_actions else random.choice(legal_actions)

    def evaluate_strength(self, hand_card, board_card):
        """Evaluates hand strength based on card ranks."""
        # Card ranks: J=0, Q=1, K=2
        if board_card is not None and hand_card == board_card:
            return 'strong' # Pair
        
        if hand_card == 2: # King
            return 'strong'
        elif hand_card == 0: # Jack
            return 'weak'
        else: # Queen
            return 'medium'

class LoosePassiveBot(BaseBot):
    """
    A bot that plays a loose-passive style.
    - Rarely folds.
    - Prefers to check/call.
    - Rarely bets/raises.
    """
    def get_action(self, state):
        legal_actions = self.game.get_legal_actions(state)
        if not legal_actions:
            return None

        # Action mapping: 0=Fold, 1=Call/Check, 2=Bet/Raise
        fold_action, call_action, raise_action = 0, 1, 2

        # Avoid folding if possible
        if len(legal_actions) > 1 and fold_action in legal_actions:
            legal_actions.remove(fold_action)

        # Prefer calling/checking
        if call_action in legal_actions:
            # Occasionally bet/raise, but mostly call
            if random.random() < 0.1 and raise_action in legal_actions:
                return raise_action
            return call_action
        
        # If only fold and raise are options, choose randomly
        return random.choice(legal_actions)

class CFRBot:
    def __init__(self, cfr_agent):
        self.cfr_agent = cfr_agent
        self.game = LeducHoldem()

    def get_action(self, state):
        legal_actions = self.game.get_legal_actions(state)
        if not legal_actions:
            return None
        
        # The CFR agent's policy is often a numpy array over all actions
        policy = self.cfr_agent.get_average_strategy(state)
        
        # Ensure policy is a valid probability distribution over legal actions
        legal_policy = [policy[a] for a in legal_actions]
        policy_sum = sum(legal_policy)
        
        if policy_sum > 0:
            normalized_policy = [p / policy_sum for p in legal_policy]
            return np.random.choice(legal_actions, p=normalized_policy)
        else:
            # Fallback to uniform random if policy has no weight on legal actions
            return random.choice(legal_actions)
