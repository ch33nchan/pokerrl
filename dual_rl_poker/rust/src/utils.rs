//! Utility modules for RNG management and deterministic replay
//!
//! This module provides deterministic random number generation,
//! state serialization, and replay verification utilities
//! for the Rust environment implementation.

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Deterministic RNG state for replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RngState {
    /// PCG64 state
    pub state: u128,
    /// PCG64 increment
    pub inc: u128,
}

impl RngState {
    /// Create from PCG64 RNG
    pub fn from_pcg64(rng: &Pcg64Mcg) -> Self {
        Self {
            state: rng.get_state()[0],
            inc: rng.get_state()[1],
        }
    }

    /// Convert to PCG64 RNG
    pub fn to_pcg64(&self) -> Result<Pcg64Mcg, &'static str> {
        Pcg64Mcg::from_state(self.state, self.inc)
    }
}

/// Deterministic RNG wrapper
pub struct DeterministicRng {
    rng: Pcg64Mcg,
    initial_state: RngState,
}

impl DeterministicRng {
    /// Create new deterministic RNG with seed
    pub fn new(seed: u64) -> Self {
        let rng = Pcg64Mcg::seed_from_u64(seed);
        let initial_state = RngState::from_pcg64(&rng);

        Self { rng, initial_state }
    }

    /// Get current state
    pub fn get_state(&self) -> RngState {
        RngState::from_pcg64(&self.rng)
    }

    /// Set state
    pub fn set_state(&mut self, state: &RngState) -> Result<(), &'static str> {
        self.rng = state.to_pcg64()?;
        Ok(())
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.rng = self.initial_state.to_pcg64().unwrap();
    }

    /// Generate random boolean
    pub fn gen_bool(&mut self) -> bool {
        self.rng.gen_bool(0.5)
    }

    /// Generate random range
    pub fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        self.rng.gen_range(range)
    }

    /// Shuffle a slice
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        self.rng.shuffle(slice);
    }

    /// Get underlying RNG reference
    pub fn rng(&mut self) -> &mut Pcg64Mcg {
        &mut self.rng
    }
}

/// Replay buffer for deterministic training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    /// Stored episodes
    episodes: Vec<Episode>,
    /// Maximum number of episodes
    max_episodes: usize,
    /// Current capacity
    capacity: usize,
}

/// Single episode in replay buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Episode ID
    pub id: String,
    /// Seed for reproducibility
    pub seed: u64,
    /// Environment type
    pub env_type: String,
    /// Steps in episode
    pub steps: Vec<Step>,
    /// Total reward
    pub total_reward: f32,
    /// Episode length
    pub length: usize,
}

/// Single step in episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// Step index
    pub t: usize,
    /// Observation
    pub observation: Vec<f32>,
    /// Action taken
    pub action: i32,
    /// Reward received
    pub reward: f32,
    /// Done flag
    pub done: bool,
    /// Legal actions
    pub legal_actions: Vec<i32>,
    /// Current player
    pub current_player: i32,
    /// RNG state after step
    pub rng_state: RngState,
    /// Additional info
    pub info: HashMap<String, serde_json::Value>,
}

impl ReplayBuffer {
    /// Create new replay buffer
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: Vec::with_capacity(max_episodes),
            max_episodes,
            capacity: max_episodes,
        }
    }

    /// Add episode to buffer
    pub fn add_episode(&mut self, episode: Episode) -> Result<(), &'static str> {
        if self.episodes.len() >= self.max_episodes {
            // Remove oldest episode
            self.episodes.remove(0);
        }

        self.episodes.push(episode);
        Ok(())
    }

    /// Get episode by ID
    pub fn get_episode(&self, id: &str) -> Option<&Episode> {
        self.episodes.iter().find(|ep| ep.id == id)
    }

    /// Get all episodes
    pub fn get_episodes(&self) -> &[Episode] {
        &self.episodes
    }

    /// Get sample of episodes
    pub fn sample_episodes(&self, n: usize) -> Vec<&Episode> {
        use rand::seq::SliceRandom;

        let mut indices: Vec<usize> = (0..self.episodes.len()).collect();
        indices.shuffle(&mut rand::thread_rng());

        indices
            .into_iter()
            .take(n.min(self.episodes.len()))
            .map(|i| &self.episodes[i])
            .collect()
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.episodes.clear();
    }

    /// Get statistics
    pub fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        if self.episodes.is_empty() {
            stats.insert(
                "total_episodes".to_string(),
                serde_json::Value::Number(0.into()),
            );
            return stats;
        }

        let total_reward: f32 = self.episodes.iter().map(|ep| ep.total_reward).sum();
        let avg_reward = total_reward / self.episodes.len() as f32;
        let total_length: usize = self.episodes.iter().map(|ep| ep.length).sum();
        let avg_length = total_length as f32 / self.episodes.len() as f32;

        stats.insert(
            "total_episodes".to_string(),
            serde_json::Value::Number(self.episodes.len().into()),
        );
        stats.insert(
            "avg_reward".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(avg_reward as f64).unwrap()),
        );
        stats.insert(
            "avg_length".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(avg_length as f64).unwrap()),
        );
        stats.insert(
            "total_steps".to_string(),
            serde_json::Value::Number(total_length.into()),
        );

        stats
    }

    /// Save to file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from file
    pub fn load_from_file(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let buffer: ReplayBuffer = serde_json::from_str(&json)?;
        self.episodes = buffer.episodes;
        self.max_episodes = buffer.max_episodes;
        self.capacity = buffer.capacity;
        Ok(())
    }
}

/// Episode builder for constructing episodes step by step
pub struct EpisodeBuilder {
    episode: Episode,
}

impl EpisodeBuilder {
    /// Create new episode builder
    pub fn new(id: String, seed: u64, env_type: String) -> Self {
        Self {
            episode: Episode {
                id,
                seed,
                env_type,
                steps: Vec::new(),
                total_reward: 0.0,
                length: 0,
            },
        }
    }

    /// Add step to episode
    pub fn add_step(
        &mut self,
        observation: Vec<f32>,
        action: i32,
        reward: f32,
        done: bool,
        legal_actions: Vec<i32>,
        current_player: i32,
        rng_state: RngState,
    ) -> &mut Self {
        let step = Step {
            t: self.episode.length,
            observation,
            action,
            reward,
            done,
            legal_actions,
            current_player,
            rng_state,
            info: HashMap::new(),
        };

        self.episode.total_reward += reward;
        self.episode.length += 1;
        self.episode.steps.push(step);
        self
    }

    /// Add info to last step
    pub fn add_info(&mut self, key: String, value: serde_json::Value) -> &mut Self {
        if let Some(last_step) = self.episode.steps.last_mut() {
            last_step.info.insert(key, value);
        }
        self
    }

    /// Build episode
    pub fn build(self) -> Episode {
        self.episode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_state_serialization() {
        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let state = RngState::from_pcg64(&rng);

        let json = serde_json::to_string(&state).unwrap();
        let restored: RngState = serde_json::from_str(&json).unwrap();

        let mut restored_rng = restored.to_pcg64().unwrap();
        assert_eq!(rng.gen::<u64>(), restored_rng.gen::<u64>());
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(2);

        // Add episodes
        let episode1 = EpisodeBuilder::new("ep1".to_string(), 42, "kuhn_poker".to_string())
            .add_step(
                vec![0.5],
                0,
                1.0,
                true,
                vec![0, 1],
                0,
                RngState::from_pcg64(&Pcg64Mcg::seed_from_u64(42)),
            )
            .build();

        let episode2 = EpisodeBuilder::new("ep2".to_string(), 43, "kuhn_poker".to_string())
            .add_step(
                vec![0.3],
                1,
                -1.0,
                true,
                vec![0, 1],
                1,
                RngState::from_pcg64(&Pcg64Mcg::seed_from_u64(43)),
            )
            .build();

        buffer.add_episode(episode1).unwrap();
        buffer.add_episode(episode2).unwrap();

        assert_eq!(buffer.get_episodes().len(), 2);

        // Test eviction when exceeding capacity
        let episode3 = EpisodeBuilder::new("ep3".to_string(), 44, "kuhn_poker".to_string())
            .add_step(
                vec![0.7],
                0,
                0.5,
                true,
                vec![0, 1],
                0,
                RngState::from_pcg64(&Pcg64Mcg::seed_from_u64(44)),
            )
            .build();

        buffer.add_episode(episode3).unwrap();
        assert_eq!(buffer.get_episodes().len(), 2); // Should evict oldest

        // Test statistics
        let stats = buffer.get_stats();
        assert_eq!(stats["total_episodes"], 2);
    }
}
