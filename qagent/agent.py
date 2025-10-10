import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

class Actor(nn.Module):
    """Actor network for policy approximation."""
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    """Critic network for value function approximation."""
    def __init__(self, state_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.network(state)

class ActorCriticAgent:
    """A2C agent that contains Actor and Critic networks."""
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, entropy_beta=0.01, hidden_size=256, device='cpu'):
        self.device = device
        self.actor = Actor(state_dim, action_dim, hidden_size).to(self.device)
        self.critic = Critic(state_dim, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

    def select_action(self, state, action_mask):
        """Selects an action based on the current policy and action mask."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(self.device)
        
        action_probs = self.actor(state_tensor)
        
        # Apply the action mask
        masked_probs = action_probs * action_mask_tensor
        # Re-normalize probabilities
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        else:
            # Fallback if all valid actions have zero probability
            masked_probs = action_mask_tensor / action_mask_tensor.sum()
        
        dist = Categorical(masked_probs)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state_tensor)
        
        self.log_probs.append(log_prob.unsqueeze(0))
        self.values.append(value)
        
        return action.item(), entropy

    def update(self):
        """Updates the actor and critic networks."""
        if not self.rewards:
            return 0.0, 0.0, 0.0

        # Convert lists to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        log_probs = torch.cat(self.log_probs).squeeze() # Squeeze to make it [seq_len]
        values = torch.cat(self.values).squeeze()
        masks = torch.tensor(self.masks, dtype=torch.float32).to(self.device)
        
        # Calculate returns (discounted rewards)
        returns = []
        R = 0
        # Ensure last state value is used for bootstrapping if episode not done
        next_val = self.critic(torch.from_numpy(np.zeros(self.actor.network[0].in_features)).float().unsqueeze(0).to(self.device)) if len(self.masks) > 0 and self.masks[-1] > 0 else 0

        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + self.gamma * R * m
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantage
        advantage = returns - values
        
        # Calculate losses
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        # Total loss
        total_loss = actor_loss + critic_loss
        
        # Optimize
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # Clear buffers
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.masks = []
        
        return actor_loss.item(), critic_loss.item()

    def save_model(self, path=".", episode=None):
        """Saves the model weights."""
        if not os.path.exists(path):
            os.makedirs(path)
        
        actor_filename = f"actor_e{episode}.pt" if episode is not None else "actor.pt"
        critic_filename = f"critic_e{episode}.pt" if episode is not None else "critic.pt"

        torch.save(self.actor.state_dict(), os.path.join(path, actor_filename))
        torch.save(self.critic.state_dict(), os.path.join(path, critic_filename))

    def load_model(self, path="."):
        """Loads the model weights."""
        actor_path = os.path.join(path, "actor.pt")
        critic_path = os.path.join(path, "critic.pt")

        if not os.path.exists(actor_path):
            # If the simple name doesn't exist, find the latest checkpoint
            files = [f for f in os.listdir(path) if f.startswith('actor_e') and f.endswith('.pt')]
            if not files:
                raise FileNotFoundError(f"No actor model file found in {path}")
            latest_actor = max(files, key=lambda f: int(f.split('_e')[1].split('.pt')[0]))
            actor_path = os.path.join(path, latest_actor)
            
        if not os.path.exists(critic_path):
            files = [f for f in os.listdir(path) if f.startswith('critic_e') and f.endswith('.pt')]
            if not files:
                raise FileNotFoundError(f"No critic model file found in {path}")
            latest_critic = max(files, key=lambda f: int(f.split('_e')[1].split('.pt')[0]))
            critic_path = os.path.join(path, latest_critic)

        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor.eval()
        self.critic.eval()