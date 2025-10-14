//! Replay buffer implementation for deterministic replay and training
//!
//! This module provides a comprehensive replay buffer system that supports
//! deterministic episode storage, retrieval, and verification for the
//! ARMAC training system with Rust environment integration.

use crate::utils::{Episode, EpisodeBuilder, RngState, Step};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use parking_lot::RwLock;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Replay buffer with deterministic replay support
#[pyclass]
pub struct ReplayBuffer {
    /// Internal buffer storage
    buffer: Arc<RwLock<InternalBuffer>>,
    /// Configuration
    config: ReplayConfig,
}

/// Internal buffer implementation
#[derive(Debug)]
struct InternalBuffer {
    /// Episodes stored in buffer
    episodes: Vec<Episode>,
    /// Episode indices by ID for fast lookup
    episode_index: HashMap<String, usize>,
    /// Circular buffer indices for efficient storage
    write_idx: usize,
    /// Number of episodes currently stored
    size: usize,
    /// Statistics tracking
    stats: BufferStats,
}

/// Replay buffer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Maximum number of episodes to store
    pub max_episodes: usize,
    /// Maximum number of steps per episode
    pub max_steps_per_episode: usize,
    /// Whether to compress episodes
    pub compress: bool,
    /// Priority mode (uniform, reward, recent)
    pub priority_mode: PriorityMode,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

/// Priority modes for episode sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityMode {
    /// Uniform random sampling
    Uniform,
    /// Prioritize by total reward
    Reward,
    /// Prioritize recent episodes
    Recent,
    /// Prioritize by episode length
    Length,
}

/// Buffer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferStats {
    /// Total number of episodes
    pub total_episodes: usize,
    /// Total number of steps
    pub total_steps: usize,
    /// Average reward per episode
    pub avg_reward: f32,
    /// Average episode length
    pub avg_length: f32,
    /// Minimum reward
    pub min_reward: f32,
    /// Maximum reward
    pub max_reward: f32,
    /// Reward standard deviation
    pub reward_std: f32,
    /// Buffer utilization
    pub utilization: f32,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            max_episodes: 10000,
            max_steps_per_episode: 1000,
            compress: false,
            priority_mode: PriorityMode::Uniform,
            seed: None,
        }
    }
}

impl Default for BufferStats {
    fn default() -> Self {
        Self {
            total_episodes: 0,
            total_steps: 0,
            avg_reward: 0.0,
            avg_length: 0.0,
            min_reward: f32::INFINITY,
            max_reward: f32::NEG_INFINITY,
            reward_std: 0.0,
            utilization: 0.0,
        }
    }
}

impl ReplayBuffer {
    /// Create new replay buffer
    pub fn new(config: ReplayConfig) -> Self {
        let buffer = InternalBuffer {
            episodes: Vec::with_capacity(config.max_episodes),
            episode_index: HashMap::new(),
            write_idx: 0,
            size: 0,
            stats: BufferStats::default(),
        };

        Self {
            buffer: Arc::new(RwLock::new(buffer)),
            config,
        }
    }

    /// Add episode to buffer
    pub fn add_episode(&self, episode: Episode) -> Result<(), String> {
        let mut buffer = self.buffer.write();

        // Validate episode
        if episode.steps.is_empty() {
            return Err("Episode has no steps".to_string());
        }

        if episode.steps.len() > self.config.max_steps_per_episode {
            return Err(format!(
                "Episode too long: {} > {}",
                episode.steps.len(),
                self.config.max_steps_per_episode
            ));
        }

        // Check if episode already exists
        if buffer.episode_index.contains_key(&episode.id) {
            return Err(format!("Episode {} already exists", episode.id));
        }

        // Add episode to buffer
        if buffer.size < self.config.max_episodes {
            // Buffer not full, append
            buffer.episodes.push(episode.clone());
            buffer.episode_index.insert(episode.id.clone(), buffer.size);
            buffer.size += 1;
        } else {
            // Buffer full, overwrite oldest
            let old_id = buffer.episodes[buffer.write_idx].id.clone();
            buffer.episode_index.remove(&old_id);

            buffer.episodes[buffer.write_idx] = episode.clone();
            buffer
                .episode_index
                .insert(episode.id.clone(), buffer.write_idx);
            buffer.write_idx = (buffer.write_idx + 1) % self.config.max_episodes;
        }

        // Update statistics
        self.update_stats(&mut buffer);

        Ok(())
    }

    /// Get episode by ID
    pub fn get_episode(&self, id: &str) -> Option<Episode> {
        let buffer = self.buffer.read();
        buffer
            .episode_index
            .get(id)
            .map(|&idx| buffer.episodes[idx].clone())
    }

    /// Sample episodes based on priority mode
    pub fn sample_episodes(&self, n: usize) -> Vec<Episode> {
        let buffer = self.buffer.read();

        if buffer.size == 0 {
            return Vec::new();
        }

        let sample_size = n.min(buffer.size);
        let mut episodes = Vec::with_capacity(sample_size);

        match self.config.priority_mode {
            PriorityMode::Uniform => {
                use rand::seq::SliceRandom;
                use rand::SeedableRng;
                use rand_pcg::Pcg64Mcg;

                let mut rng = Pcg64Mcg::seed_from_u64(self.config.seed.unwrap_or(42));
                let indices: Vec<usize> = (0..buffer.size).collect();
                let sampled_indices: Vec<usize> = indices
                    .choose_multiple(&mut rng, sample_size)
                    .cloned()
                    .collect();

                for &idx in &sampled_indices {
                    episodes.push(buffer.episodes[idx].clone());
                }
            }
            PriorityMode::Reward => {
                // Sample episodes weighted by absolute reward
                let mut weights: Vec<f32> = buffer.episodes[..buffer.size]
                    .iter()
                    .map(|ep| ep.total_reward.abs())
                    .collect();

                let total_weight: f32 = weights.iter().sum();
                if total_weight > 0.0 {
                    for weight in &mut weights {
                        *weight /= total_weight;
                    }
                }

                use rand::seq::IteratorRandom;
                use rand::SeedableRng;
                use rand_pcg::Pcg64Mcg;

                let mut rng = Pcg64Mcg::seed_from_u64(self.config.seed.unwrap_or(42));

                for _ in 0..sample_size {
                    let idx = (0..buffer.size)
                        .map(|i| (i, weights[i]))
                        .filter(|(_, w)| *w > 0.0)
                        .choose_weighted(&mut rng, |(_, w)| *w)
                        .map(|(i, _)| i)
                        .unwrap_or(rng.gen_range(0..buffer.size));

                    episodes.push(buffer.episodes[idx].clone());
                }
            }
            PriorityMode::Recent => {
                // Sample most recent episodes
                let start_idx = if buffer.size >= sample_size {
                    buffer.size - sample_size
                } else {
                    0
                };

                for i in start_idx..buffer.size {
                    episodes.push(buffer.episodes[i].clone());
                }
            }
            PriorityMode::Length => {
                // Sample episodes weighted by length
                let mut weights: Vec<f32> = buffer.episodes[..buffer.size]
                    .iter()
                    .map(|ep| ep.length as f32)
                    .collect();

                let total_weight: f32 = weights.iter().sum();
                if total_weight > 0.0 {
                    for weight in &mut weights {
                        *weight /= total_weight;
                    }
                }

                use rand::seq::IteratorRandom;
                use rand::SeedableRng;
                use rand_pcg::Pcg64Mcg;

                let mut rng = Pcg64Mcg::seed_from_u64(self.config.seed.unwrap_or(42));

                for _ in 0..sample_size {
                    let idx = (0..buffer.size)
                        .map(|i| (i, weights[i]))
                        .choose_weighted(&mut rng, |(_, w)| *w)
                        .map(|(i, _)| i)
                        .unwrap_or(rng.gen_range(0..buffer.size));

                    episodes.push(buffer.episodes[idx].clone());
                }
            }
        }

        episodes
    }

    /// Get all episodes
    pub fn get_all_episodes(&self) -> Vec<Episode> {
        let buffer = self.buffer.read();
        buffer.episodes[..buffer.size].to_vec()
    }

    /// Get buffer statistics
    pub fn get_stats(&self) -> BufferStats {
        let buffer = self.buffer.read();
        buffer.stats.clone()
    }

    /// Clear buffer
    pub fn clear(&self) {
        let mut buffer = self.buffer.write();
        buffer.episodes.clear();
        buffer.episode_index.clear();
        buffer.write_idx = 0;
        buffer.size = 0;
        buffer.stats = BufferStats::default();
    }

    /// Update buffer statistics
    fn update_stats(&self, buffer: &mut InternalBuffer) {
        if buffer.size == 0 {
            buffer.stats = BufferStats::default();
            return;
        }

        let total_reward: f32 = buffer.episodes[..buffer.size]
            .iter()
            .map(|ep| ep.total_reward)
            .sum();

        let total_steps: usize = buffer.episodes[..buffer.size]
            .iter()
            .map(|ep| ep.length)
            .sum();

        let min_reward = buffer.episodes[..buffer.size]
            .iter()
            .map(|ep| ep.total_reward)
            .fold(f32::INFINITY, f32::min);

        let max_reward = buffer.episodes[..buffer.size]
            .iter()
            .map(|ep| ep.total_reward)
            .fold(f32::NEG_INFINITY, f32::max);

        // Calculate reward standard deviation
        let avg_reward = total_reward / buffer.size as f32;
        let reward_variance: f32 = buffer.episodes[..buffer.size]
            .iter()
            .map(|ep| {
                let diff = ep.total_reward - avg_reward;
                diff * diff
            })
            .sum();

        let reward_std = if buffer.size > 1 {
            (reward_variance / (buffer.size - 1) as f32).sqrt()
        } else {
            0.0
        };

        buffer.stats = BufferStats {
            total_episodes: buffer.size,
            total_steps,
            avg_reward,
            avg_length: total_steps as f32 / buffer.size as f32,
            min_reward,
            max_reward,
            reward_std,
            utilization: buffer.size as f32 / self.config.max_episodes as f32,
        };
    }

    /// Save buffer to file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = self.buffer.read();
        let save_data = SerializableBuffer {
            episodes: buffer.episodes[..buffer.size].to_vec(),
            config: self.config.clone(),
            stats: buffer.stats.clone(),
        };

        let json = serde_json::to_string_pretty(&save_data)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load buffer from file
    pub fn load_from_file(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let save_data: SerializableBuffer = serde_json::from_str(&json)?;

        let mut buffer = self.buffer.write();
        buffer.episodes = save_data.episodes;
        buffer.episode_index.clear();

        for (idx, episode) in buffer.episodes.iter().enumerate() {
            buffer.episode_index.insert(episode.id.clone(), idx);
        }

        buffer.size = buffer.episodes.len();
        buffer.write_idx = buffer.size % self.config.max_episodes;
        buffer.stats = save_data.stats;

        Ok(())
    }

    /// Verify episode integrity
    pub fn verify_episode(&self, episode: &Episode) -> Result<(), String> {
        // Check basic structure
        if episode.id.is_empty() {
            return Err("Episode ID is empty".to_string());
        }

        if episode.steps.is_empty() {
            return Err("Episode has no steps".to_string());
        }

        // Check step sequence
        for (i, step) in episode.steps.iter().enumerate() {
            if step.t != i {
                return Err(format!(
                    "Step index mismatch: expected {}, got {}",
                    i, step.t
                ));
            }

            if step.observation.is_empty() {
                return Err(format!("Step {} has empty observation", i));
            }

            if !step.legal_actions.contains(&step.action) {
                return Err(format!(
                    "Step {} action {} not in legal actions {:?}",
                    i, step.action, step.legal_actions
                ));
            }
        }

        // Check total reward consistency
        let calculated_total: f32 = episode.steps.iter().map(|s| s.reward).sum();
        if (calculated_total - episode.total_reward).abs() > 1e-6 {
            return Err(format!(
                "Total reward mismatch: calculated {}, expected {}",
                calculated_total, episode.total_reward
            ));
        }

        // Check terminal state consistency
        if let Some(last_step) = episode.steps.last() {
            if last_step.done && episode.steps.len() != episode.length {
                return Err(format!(
                    "Episode length mismatch: steps {}, length {}",
                    episode.steps.len(),
                    episode.length
                ));
            }
        }

        Ok(())
    }

    /// Get episodes by environment type
    pub fn get_episodes_by_env(&self, env_type: &str) -> Vec<Episode> {
        let buffer = self.buffer.read();
        buffer.episodes[..buffer.size]
            .iter()
            .filter(|ep| ep.env_type == env_type)
            .cloned()
            .collect()
    }

    /// Get episodes in reward range
    pub fn get_episodes_by_reward_range(&self, min_reward: f32, max_reward: f32) -> Vec<Episode> {
        let buffer = self.buffer.read();
        buffer.episodes[..buffer.size]
            .iter()
            .filter(|ep| ep.total_reward >= min_reward && ep.total_reward <= max_reward)
            .cloned()
            .collect()
    }
}

/// Serializable buffer format for file I/O
#[derive(Debug, Serialize, Deserialize)]
struct SerializableBuffer {
    episodes: Vec<Episode>,
    config: ReplayConfig,
    stats: BufferStats,
}

#[pymethods]
impl ReplayBuffer {
    #[new]
    fn py_new(max_episodes: Option<usize>, seed: Option<u64>) -> PyResult<Self> {
        let config = ReplayConfig {
            max_episodes: max_episodes.unwrap_or(10000),
            seed,
            ..Default::default()
        };
        Ok(Self::new(config))
    }

    fn add_episode(&self, episode_data: serde_json::Value) -> PyResult<()> {
        let episode: Episode = serde_json::from_value(episode_data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid episode data: {}", e))
        })?;

        self.add_episode(episode).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to add episode: {}", e))
        })
    }

    fn get_episode(&self, id: &str) -> PyResult<Option<serde_json::Value>> {
        if let Some(episode) = self.get_episode(id) {
            let json = serde_json::to_value(episode).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e))
            })?;
            Ok(Some(json))
        } else {
            Ok(None)
        }
    }

    fn sample_episodes(&self, n: usize) -> PyResult<Vec<serde_json::Value>> {
        let episodes = self.sample_episodes(n);
        let mut json_episodes = Vec::with_capacity(episodes.len());

        for episode in episodes {
            let json = serde_json::to_value(episode).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e))
            })?;
            json_episodes.push(json);
        }

        Ok(json_episodes)
    }

    fn get_stats(&self) -> PyResult<serde_json::Value> {
        let stats = self.get_stats();
        let json = serde_json::to_value(stats).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e))
        })?;
        Ok(json)
    }

    fn clear(&self) -> PyResult<()> {
        self.clear();
        Ok(())
    }

    fn save_to_file(&self, path: &str) -> PyResult<()> {
        self.save_to_file(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Save failed: {}", e)))
    }

    fn load_from_file(&mut self, path: &str) -> PyResult<()> {
        self.load_from_file(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Load failed: {}", e)))
    }

    fn verify_episode(&self, episode_data: serde_json::Value) -> PyResult<()> {
        let episode: Episode = serde_json::from_value(episode_data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid episode data: {}", e))
        })?;

        self.verify_episode(&episode).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Verification failed: {}", e))
        })
    }

    fn get_size(&self) -> PyResult<usize> {
        let buffer = self.buffer.read();
        Ok(buffer.size)
    }

    fn get_capacity(&self) -> PyResult<usize> {
        Ok(self.config.max_episodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{EpisodeBuilder, RngState};

    #[test]
    fn test_replay_buffer_basic() {
        let config = ReplayConfig::default();
        let buffer = ReplayBuffer::new(config);

        // Create test episode
        let episode = EpisodeBuilder::new("test_ep".to_string(), 42, "kuhn_poker".to_string())
            .add_step(
                vec![0.5, 0.3],
                0,
                1.0,
                false,
                vec![0, 1],
                0,
                RngState::from_pcg64(&rand_pcg::Pcg64Mcg::seed_from_u64(42)),
            )
            .add_step(
                vec![0.6, 0.4],
                1,
                -1.0,
                true,
                vec![0, 1],
                1,
                RngState::from_pcg64(&rand_pcg::Pcg64Mcg::seed_from_u64(43)),
            )
            .build();

        buffer.add_episode(episode).unwrap();

        assert_eq!(buffer.get_size().unwrap(), 1);

        let retrieved = buffer.get_episode("test_ep").unwrap();
        assert!(retrieved.is_some());

        let stats = buffer.get_stats().unwrap();
        assert_eq!(stats.total_episodes, 1);
        assert_eq!(stats.total_steps, 2);
    }

    #[test]
    fn test_buffer_capacity() {
        let config = ReplayConfig {
            max_episodes: 2,
            ..Default::default()
        };
        let buffer = ReplayBuffer::new(config);

        // Add 3 episodes to test eviction
        for i in 0..3 {
            let episode =
                EpisodeBuilder::new(format!("ep_{}", i), 42 + i as u64, "kuhn_poker".to_string())
                    .add_step(
                        vec![0.5],
                        0,
                        i as f32,
                        true,
                        vec![0, 1],
                        0,
                        RngState::from_pcg64(&rand_pcg::Pcg64Mcg::seed_from_u64(42 + i as u64)),
                    )
                    .build();

            buffer.add_episode(episode).unwrap();
        }

        assert_eq!(buffer.get_size().unwrap(), 2); // Should evict oldest

        // First episode should be evicted
        assert!(buffer.get_episode("ep_0").unwrap().is_none());
        // Last two episodes should remain
        assert!(buffer.get_episode("ep_1").unwrap().is_some());
        assert!(buffer.get_episode("ep_2").unwrap().is_some());
    }

    #[test]
    fn test_episode_verification() {
        let config = ReplayConfig::default();
        let buffer = ReplayBuffer::new(config);

        // Valid episode
        let valid_episode = EpisodeBuilder::new("valid".to_string(), 42, "kuhn_poker".to_string())
            .add_step(
                vec![0.5],
                0,
                1.0,
                false,
                vec![0, 1],
                0,
                RngState::from_pcg64(&rand_pcg::Pcg64Mcg::seed_from_u64(42)),
            )
            .build();

        assert!(buffer.verify_episode(&valid_episode).is_ok());

        // Invalid episode (wrong step index)
        let mut invalid_episode = valid_episode.clone();
        invalid_episode.steps[0].t = 1;
        assert!(buffer.verify_episode(&invalid_episode).is_err());
    }
}
