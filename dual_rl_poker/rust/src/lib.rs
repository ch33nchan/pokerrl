//! Rust environment implementation for ARMAC dual RL poker
//!
//! This module provides high-performance environment implementations
//! for poker games with deterministic replay support and efficient
//! batch processing capabilities.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use parking_lot::RwLock;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use thiserror::Error;

pub mod env_batch;
pub mod kuhn_poker;
pub mod leduc_poker;
pub mod replay_buffer;
pub mod utils;

pub use env_batch::EnvBatch;
pub use kuhn_poker::KuhnPokerEnv;
pub use leduc_poker::LeducPokerEnv;
pub use replay_buffer::ReplayBuffer;
pub use utils::{DeterministicRng, RngState};

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

    /// Clone the environment state
    fn clone_state(&self) -> Box<dyn Environment>;

    /// Restore environment from cloned state
    fn restore_state(&mut self, state: &Box<dyn Environment>) -> EnvResult<()>;
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

/// Step result for environment interaction
#[derive(Debug, Clone)]
pub struct StepResult {
    pub observations: Array2<f32>,
    pub rewards: Array1<f32>,
    pub dones: Array1<bool>,
    pub infos: Vec<EnvInfo>,
}

/// Environment information for each step
#[derive(Debug, Clone)]
pub struct EnvInfo {
    pub legal_actions: Array1<i32>,
    pub current_player: i32,
    pub is_terminal: bool,
    pub rewards: Array1<f32>,
    pub info_state: Array1<f32>,
}

/// Python bindings for the Rust environment
#[pymodule]
fn dual_rl_poker_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EnvBatch>()?;
    m.add_class::<ReplayBuffer>()?;
    m.add_class::<KuhnPokerEnv>()?;
    m.add_class::<LeducPokerEnv>()?;
    m.add_function(wrap_pyfunction!(create_environment, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_environment, m)?)?;
    Ok(())
}

/// Create environment by name
#[pyfunction]
fn create_environment(
    game_name: &str,
    config: Option<EnvConfig>,
) -> PyResult<Box<dyn Environment>> {
    let config = config.unwrap_or_default();

    match game_name {
        "kuhn_poker" => {
            let mut env = KuhnPokerEnv::new(config);
            env.reset(config.seed)?;
            Ok(Box::new(env))
        }
        "leduc_poker" => {
            let mut env = LeducPokerEnv::new(config);
            env.reset(config.seed)?;
            Ok(Box::new(env))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown game: {}",
            game_name
        ))),
    }
}

/// Benchmark environment performance
#[pyfunction]
fn benchmark_environment(
    game_name: &str,
    batch_size: usize,
    num_steps: usize,
) -> PyResult<BenchmarkResult> {
    let config = EnvConfig {
        batch_size,
        ..Default::default()
    };

    let start = std::time::Instant::now();

    match game_name {
        "kuhn_poker" => {
            let mut env = EnvBatch::<KuhnPokerEnv>::new(batch_size, config);
            env.reset(Some(42))?;

            let mut total_steps = 0;
            let mut total_episodes = 0;

            for _ in 0..num_steps {
                let actions = env.sample_legal_actions()?;
                let _results = env.step(&actions)?;
                total_steps += batch_size;

                // Count completed episodes
                let dones = env.dones();
                total_episodes += dones.iter().filter(|&&done| done).count();
            }

            let elapsed = start.elapsed();

            Ok(BenchmarkResult {
                game_name: game_name.to_string(),
                batch_size,
                num_steps: total_steps,
                num_episodes: total_episodes,
                elapsed_seconds: elapsed.as_secs_f64(),
                steps_per_second: total_steps as f64 / elapsed.as_secs_f64(),
                episodes_per_second: total_episodes as f64 / elapsed.as_secs_f64(),
            })
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown game for benchmark: {}",
            game_name
        ))),
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub game_name: String,
    pub batch_size: usize,
    pub num_steps: usize,
    pub num_episodes: usize,
    pub elapsed_seconds: f64,
    pub steps_per_second: f64,
    pub episodes_per_second: f64,
}

impl BenchmarkResult {
    pub fn to_dict(&self) -> pyo3::PyObject {
        Python::with_gil(|py| {
            py.eval(
                &format!(
                    r#"{{
                    "game_name": "{}",
                    "batch_size": {},
                    "num_steps": {},
                    "num_episodes": {},
                    "elapsed_seconds": {:.6},
                    "steps_per_second": {:.2},
                    "episodes_per_second": {:.2}
                }}"#,
                    self.game_name,
                    self.batch_size,
                    self.num_steps,
                    self.num_episodes,
                    self.elapsed_seconds,
                    self.steps_per_second,
                    self.episodes_per_second
                ),
                None,
                None,
            )
            .unwrap()
            .to_object(py)
        })
    }
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

    #[test]
    fn test_env_batch() {
        let config = EnvConfig {
            batch_size: 4,
            seed: Some(42),
            ..Default::default()
        };

        let mut batch = EnvBatch::<KuhnPokerEnv>::new(4, config);
        batch.reset(Some(42)).unwrap();

        assert_eq!(batch.batch_size(), 4);
        assert_eq!(batch.observations().nrows(), 4);

        let actions = batch.sample_legal_actions().unwrap();
        assert_eq!(actions.len(), 4);

        let results = batch.step(&actions).unwrap();
        assert_eq!(results.observations.nrows(), 4);
        assert_eq!(results.rewards.len(), 4);
        assert_eq!(results.dones.len(), 4);
    }
}
