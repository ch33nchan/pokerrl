import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def plot_learning_curve(reward_data: pd.DataFrame, save_path: str, window: int = 100):
    """
    Plots the learning curve (e.g., rewards per episode) over time.

    Args:
        reward_data (pd.DataFrame): DataFrame with 'episode' and 'reward' columns.
        save_path (str): Path to save the plot image.
        window (int): Window size for the rolling average.
    """
    plt.figure(figsize=(12, 7))
    sns.lineplot(x='episode', y='reward', data=reward_data, alpha=0.4, label='Per-Episode Reward')
    
    reward_data['rolling_avg'] = reward_data['reward'].rolling(window=window).mean()
    sns.lineplot(x='episode', y='rolling_avg', data=reward_data, label=f'Rolling Average (window={window})')
    
    plt.title('Agent Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_reward_distribution(reward_data: pd.DataFrame, save_path: str):
    """
    Plots the distribution of rewards using a histogram and KDE.

    Args:
        reward_data (pd.DataFrame): DataFrame with a 'reward' column.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(reward_data['reward'], kde=True, bins=30)
    plt.title('Distribution of Episode Rewards')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_agent_comparison(performance_data: pd.DataFrame, save_path: str, metric: str = 'mean_reward'):
    """
    Compares the performance of different agents or training stages using a bar plot.

    Args:
        performance_data (pd.DataFrame): DataFrame with 'agent_name' and a metric column (e.g., 'mean_reward').
        save_path (str): Path to save the plot image.
        metric (str): The performance metric to plot.
    """
    plt.figure(figsize=(12, 7))
    sns.barplot(x='agent_name', y=metric, data=performance_data)
    plt.title(f'Agent Performance Comparison ({metric})')
    plt.xlabel('Agent / Training Stage')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45, ha='right')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_win_rate_pie(win_loss_draw: dict, save_path: str):
    """
    Creates a pie chart showing the win, loss, and draw percentages.

    Args:
        win_loss_draw (dict): A dictionary with 'wins', 'losses', 'draws' keys.
        save_path (str): Path to save the plot image.
    """
    labels = ['Wins', 'Losses', 'Draws']
    sizes = [win_loss_draw.get('wins', 0), win_loss_draw.get('losses', 0), win_loss_draw.get('draws', 0)]
    colors = ['#4CAF50', '#F44336', '#FFC107']
    
    if sum(sizes) == 0:
        print("Warning: No win/loss data to plot.")
        return

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Win/Loss/Draw Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
