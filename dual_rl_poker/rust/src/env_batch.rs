//! Batch environment implementation for efficient parallel processing
//!
//! This module provides a high-performance batch environment that can
//! handle multiple game instances in parallel with deterministic replay
//! support and efficient memory management.

use crate::{EnvConfig, EnvError, EnvInfo, EnvResult, Environment, StepResult};
use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray2};
use parking_lot::RwLock;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Batch environment for parallel processing
#[pyclass]
pub struct EnvBatch<T: Environment + Clone> {
    environments: Vec<T>,
    config: EnvConfig,
    rng: Pcg64Mcg,
    batch_size: usize,
    observations: Array2<f32>,
    legal_actions: Vec<Array1<i32>>,
    current_players: Array1<i32>,
    dones: Array1<bool>,
    rewards: Array1<f32>,
    info_states: Vec<Array1<f32>>,
}

impl<T: Environment + Clone> EnvBatch<T> {
    /// Create a new batch environment
    pub fn new(batch_size: usize, config: EnvConfig) -> Self {
        let mut environments = Vec::with_capacity(batch_size);

        // Create individual environments
        for i in 0..batch_size {
            let mut env_config = config.clone();
            if let Some(seed) = config.seed {
                env_config.seed = Some(seed.wrapping_add(i as u64));
            }

            // Note: This requires T to implement a constructor from EnvConfig
            // In practice, each game type will have its own factory method
            panic!("Environment must be created through specific factory methods");
        }

        let obs_dim = if batch_size > 0 {
            environments[0]
                .game_info()
                .observation_shape
                .iter()
                .product()
        } else {
            0
        };

        Self {
            environments,
            config,
            rng: Pcg64Mcg::seed_from_u64(config.seed.unwrap_or(42)),
            batch_size,
            observations: Array2::zeros((batch_size, obs_dim)),
            legal_actions: Vec::with_capacity(batch_size),
            current_players: Array1::zeros(batch_size),
            dones: Array1::from_elem(batch_size, false),
            rewards: Array1::zeros(batch_size),
            info_states: Vec::with_capacity(batch_size),
        }
    }

    /// Create batch environment from existing environments
    pub fn from_environments(environments: Vec<T>, config: EnvConfig) -> EnvResult<Self> {
        let batch_size = environments.len();
        let obs_dim = if batch_size > 0 {
            environments[0]
                .game_info()
                .observation_shape
                .iter()
                .product()
        } else {
            0
        };

        let mut batch = Self {
            environments,
            config,
            rng: Pcg64Mcg::seed_from_u64(config.seed.unwrap_or(42)),
            batch_size,
            observations: Array2::zeros((batch_size, obs_dim)),
            legal_actions: Vec::with_capacity(batch_size),
            current_players: Array1::zeros(batch_size),
            dones: Array1::from_elem(batch_size, false),
            rewards: Array1::zeros(batch_size),
            info_states: Vec::with_capacity(batch_size),
        };

        batch.update_observations()?;
        Ok(batch)
    }

    /// Update internal observations from all environments
    fn update_observations(&mut self) -> EnvResult<()> {
        for (i, env) in self.environments.iter().enumerate() {
            self.observations.row_mut(i).assign(&env.observation());
            self.legal_actions[i] = env.legal_actions();
            self.current_players[i] = env.current_player();
            self.dones[i] = env.is_terminal();
            self.rewards[i] = if env.is_terminal() {
                env.rewards()[env.current_player().max(0) as usize]
            } else {
                0.0
            };
            self.info_states[i] = env.info_state(env.current_player().max(0));
        }
        Ok(())
    }

    /// Reset all environments
    pub fn reset(&mut self, seed: Option<u64>) -> EnvResult<()> {
        if let Some(seed) = seed {
            self.rng = Pcg64Mcg::seed_from_u64(seed);
        }

        // Reset environments in parallel
        self.environments
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, env)| {
                let env_seed = self.rng.gen::<u64>().wrapping_add(i as u64);
                let _ = env.reset(Some(env_seed));
            });

        self.update_observations()?;
        Ok(())
    }

    /// Step all environments with given actions
    pub fn step(&mut self, actions: &[i32]) -> EnvResult<StepResult> {
        if actions.len() != self.batch_size {
            return Err(EnvError::InvalidAction {
                action: actions.len() as i32,
            });
        }

        // Step environments in parallel
        let results: Vec<EnvResult<(Array1<f32>, f32, bool)>> = self
            .environments
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(env, &action)| {
                if env.is_terminal() {
                    return Ok((env.observation(), 0.0, true));
                }
                env.step(action)
            })
            .collect();

        // Process results
        let mut infos = Vec::with_capacity(self.batch_size);
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok((_, reward, done)) => {
                    self.rewards[i] = reward * self.config.reward_scale;
                    self.dones[i] = done;
                }
                Err(e) => return Err(e),
            }
        }

        // Reset terminal environments
        for (i, done) in self.dones.iter().enumerate() {
            if *done {
                let env_seed = self.rng.gen::<u64>();
                let _ = self.environments[i].reset(Some(env_seed));
            }
        }

        self.update_observations()?;

        // Create info objects
        for i in 0..self.batch_size {
            infos.push(EnvInfo {
                legal_actions: self.legal_actions[i].clone(),
                current_player: self.current_players[i],
                is_terminal: self.dones[i],
                rewards: Array1::from_elem(
                    self.environments[i].game_info().num_players,
                    self.rewards[i],
                ),
                info_state: self.info_states[i].clone(),
            });
        }

        Ok(StepResult {
            observations: self.observations.clone(),
            rewards: self.rewards.clone(),
            dones: self.dones.clone(),
            infos,
        })
    }

    /// Sample legal actions for all environments
    pub fn sample_legal_actions(&self) -> EnvResult<Vec<i32>> {
        let mut actions = Vec::with_capacity(self.batch_size);

        for i in 0..self.batch_size {
            if self.dones[i] {
                actions.push(0); // Default action for terminal states
            } else {
                let legal_actions = &self.legal_actions[i];
                if legal_actions.len() > 0 {
                    let idx = self.rng.gen_range(0..legal_actions.len());
                    actions.push(legal_actions[idx]);
                } else {
                    actions.push(0);
                }
            }
        }

        Ok(actions)
    }

    /// Get current observations
    pub fn observations(&self) -> &Array2<f32> {
        &self.observations
    }

    /// Get current dones
    pub fn dones(&self) -> &Array1<bool> {
        &self.dones
    }

    /// Get current rewards
    pub fn rewards(&self) -> &Array1<f32> {
        &self.rewards
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get environment configuration
    pub fn config(&self) -> &EnvConfig {
        &self.config
    }

    /// Get RNG state for deterministic replay
    pub fn get_rng_state(&self) -> RngState {
        RngState::from_pcg64(&self.rng)
    }

    /// Set RNG state for deterministic replay
    pub fn set_rng_state(&mut self, state: &RngState) -> EnvResult<()> {
        self.rng = state.to_pcg64().map_err(|_| EnvError::RngError {
            message: "Invalid RNG state".to_string(),
        })?;
        Ok(())
    }

    /// Get environment states for debugging
    pub fn get_env_states(&self) -> Vec<Vec<u8>> {
        self.environments
            .iter()
            .map(|env| {
                // Serialize environment state
                // This is a simplified version - in practice you'd want proper serialization
                vec![]
            })
            .collect()
    }
}

#[pymethods]
impl<T: Environment + Clone + Send + Sync + 'static> EnvBatch<T> {
    /// Create new batch environment (Python interface)
    #[new]
    fn py_new(batch_size: usize, config: Option<EnvConfig>) -> PyResult<Self> {
        let config = config.unwrap_or_default();

        // This is a placeholder - actual implementation would need
        // specific factory methods for each environment type
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Use specific factory methods like create_kuhn_batch()",
        ))
    }

    /// Reset environments (Python interface)
    fn reset(&mut self, seed: Option<u64>) -> PyResult<()> {
        self.reset(seed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Reset failed: {}", e)))
    }

    /// Step environments (Python interface)
    fn step(&mut self, actions: Vec<i32>) -> PyResult<StepResult> {
        self.step(&actions)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Step failed: {}", e)))
    }

    /// Get observations as NumPy array
    fn get_observations<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        self.observations.view().into_pyarray(py)
    }

    /// Get rewards as NumPy array
    fn get_rewards<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.rewards.view().into_pyarray(py)
    }

    /// Get dones as NumPy array
    fn get_dones<'py>(&self, py: Python<'py>) -> &'py PyArray1<bool> {
        self.dones.view().into_pyarray(py)
    }

    /// Sample legal actions
    fn sample_legal_actions(&self) -> PyResult<Vec<i32>> {
        self.sample_legal_actions().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Sampling failed: {}", e))
        })
    }

    /// Get RNG state
    fn get_rng_state(&self) -> PyResult<Vec<u8>> {
        let state = self.get_rng_state();
        serde_json::to_vec(&state).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e))
        })
    }

    /// Set RNG state
    fn set_rng_state(&mut self, state_bytes: Vec<u8>) -> PyResult<()> {
        let state: RngState = serde_json::from_slice(&state_bytes).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Deserialization failed: {}", e))
        })?;

        self.set_rng_state(&state).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Setting RNG state failed: {}", e))
        })
    }

    /// Get batch size
    fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get statistics
    fn get_stats(&self) -> PyResult<serde_json::Value> {
        let stats = serde_json::json!({
            "batch_size": self.batch_size,
            "num_terminal": self.dones.iter().filter(|&&done| done).count(),
            "avg_reward": self.rewards.iter().sum::<f32>() / self.batch_size as f32,
            "config": self.config
        });
        Ok(stats)
    }
}

/// Factory function for Kuhn Poker batch
pub fn create_kuhn_batch(
    batch_size: usize,
    config: EnvConfig,
) -> EnvResult<EnvBatch<crate::KuhnPokerEnv>> {
    let mut environments = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let mut env_config = config.clone();
        if let Some(seed) = config.seed {
            env_config.seed = Some(seed.wrapping_add(i as u64));
        }

        let mut env = crate::KuhnPokerEnv::new(env_config);
        env.reset(env_config.seed)?;
        environments.push(env);
    }

    EnvBatch::from_environments(environments, config)
}

/// Factory function for Leduc Poker batch
pub fn create_leduc_batch(
    batch_size: usize,
    config: EnvConfig,
) -> EnvResult<EnvBatch<crate::LeducPokerEnv>> {
    let mut environments = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let mut env_config = config.clone();
        if let Some(seed) = config.seed {
            env_config.seed = Some(seed.wrapping_add(i as u64));
        }

        let mut env = crate::LeducPokerEnv::new(env_config);
        env.reset(env_config.seed)?;
        environments.push(env);
    }

    EnvBatch::from_environments(environments, config)
}
