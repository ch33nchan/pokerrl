//! Rust environment implementation for ARMAC dual RL poker
//!
//! This module provides high-performance environment implementations
//! for poker games with deterministic replay support and efficient
//! batch processing capabilities.

use ndarray::Array1;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod kuhn_poker;
pub mod leduc_poker;
pub use kuhn_poker::KuhnPokerEnv;
pub use leduc_poker::LeducPokerEnv;

#[derive(Error, Debug)]
pub enum EnvError {
    #[error("Invalid action: {action} for state")]
    InvalidAction { action: i32 },
    #[error("Game is already terminal")]
    AlreadyTerminal,
    #[error("Invalid player: {player}")]
    InvalidPlayer { player: i32 },
    #[error("Encoding error: {message}")]
    EncodingError { message: String },
    #[error("RNG error: {message}")]
    RngError { message: String },
}

pub type EnvResult<T> = Result<T, EnvError>;

/// Base trait for all game environments
pub trait Environment: Send + Sync {
    /// Reset the environment to initial state
    fn reset(&mut self, seed: Option<u64>) -> EnvResult<()>;

    /// Get current observation
    fn observation(&self) -> Array1<f32>;

    /// Get legal actions mask
    fn legal_actions(&self) -> Array1<i32>;

    /// Get current player (0 or 1), or -1 for terminal
    fn current_player(&self) -> i32;

    /// Check if state is terminal
    fn is_terminal(&self) -> bool;

    /// Get rewards for all players
    fn rewards(&self) -> Array1<f32>;

    /// Step the environment with an action
    fn step(&mut self, action: i32) -> EnvResult<(Array1<f32>, f32, bool)>;

    /// Get information state encoding
    fn info_state(&self, player: i32) -> Array1<f32>;

    /// Get game-specific information
    fn game_info(&self) -> GameInfo;

}

/// Game information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameInfo {
    pub name: String,
    pub num_players: usize,
    pub num_actions: usize,
    pub max_game_length: usize,
    pub observation_shape: Vec<usize>,
    pub action_shape: Vec<usize>,
}

/// Environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    pub seed: Option<u64>,
    pub batch_size: usize,
    pub max_steps: Option<usize>,
    pub reward_scale: f32,
    pub gamma: f32,
    pub deterministic: bool,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            seed: None,
            batch_size: 1,
            max_steps: None,
            reward_scale: 1.0,
            gamma: 0.99,
            deterministic: false,
        }
    }
}

/// Python bindings for the Rust environment
#[pymodule]
fn dual_rl_poker_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KuhnPokerEnv>()?;
    m.add_class::<LeducPokerEnv>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kuhn_poker_basic() {
        let config = EnvConfig {
            seed: Some(42),
            ..Default::default()
        };

        let mut env = KuhnPokerEnv::new(config);
        env.reset(Some(42)).unwrap();

        assert_eq!(env.current_player(), 0);
        assert!(!env.is_terminal());
        assert_eq!(env.legal_actions().len(), 2);

        let (obs, reward, done) = env.step(0).unwrap();
        assert_eq!(
            obs.len(),
            env.game_info().observation_shape.iter().product()
        );
        assert_eq!(done, false);
    }

}
