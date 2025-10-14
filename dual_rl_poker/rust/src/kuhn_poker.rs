//! Kuhn Poker environment implementation in Rust
//!
//! This module provides a high-performance implementation of Kuhn Poker
//! with deterministic replay support, efficient state management, and
//! Python bindings for integration with the ARMAC training system.

use crate::{EnvConfig, EnvError, EnvResult, Environment, GameInfo};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Kuhn Poker game state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KuhnPokerState {
    /// Player cards [0, 1, 2] representing J, Q, K
    pub cards: [i32; 2],
    /// Current betting round (0 for first, 1 for second)
    pub round: i32,
    /// Current player (0 or 1)
    pub current_player: i32,
    /// Pot size
    pub pot: i32,
    /// Player contributions to pot
    pub contributions: [i32; 2],
    /// Betting history
    pub history: Vec<i32>,
    /// Is terminal state
    pub terminal: bool,
    /// Payoffs for each player
    pub payoffs: [f32; 2],
}

impl Default for KuhnPokerState {
    fn default() -> Self {
        Self {
            cards: [0, 0],
            round: 0,
            current_player: 0,
            pot: 2, // Ante from both players
            contributions: [1, 1],
            history: Vec::new(),
            terminal: false,
            payoffs: [0.0, 0.0],
        }
    }
}

/// Kuhn Poker environment
#[pyclass]
pub struct KuhnPokerEnv {
    /// Current game state
    state: KuhnPokerState,
    /// Random number generator
    rng: Pcg64Mcg,
    /// Environment configuration
    config: EnvConfig,
    /// Game information
    game_info: GameInfo,
}

impl KuhnPokerEnv {
    /// Create new Kuhn Poker environment
    pub fn new(config: EnvConfig) -> Self {
        let game_info = GameInfo {
            name: "kuhn_poker".to_string(),
            num_players: 2,
            num_actions: 2, // Check/Call (0) or Bet/Fold (1)
            max_game_length: 6,
            observation_shape: vec![6], // [card, pot, round, player, legal_actions_mask_2]
            action_shape: vec![1],
        };

        Self {
            state: KuhnPokerState::default(),
            rng: Pcg64Mcg::seed_from_u64(config.seed.unwrap_or(42)),
            config,
            game_info,
        }
    }

    /// Deal cards to players
    fn deal_cards(&mut self) -> EnvResult<()> {
        let mut cards = [0, 1, 2];
        self.rng.shuffle(&mut cards);
        self.state.cards = [cards[0], cards[1]];
        Ok(())
    }

    /// Check if action is legal
    fn is_legal_action(&self, action: i32) -> bool {
        if self.state.terminal {
            return false;
        }

        match self.state.round {
            0 => {
                // First round: can check (0) or bet (1)
                action == 0 || action == 1
            }
            1 => {
                // Second round: depends on previous action
                if self.state.history.is_empty() {
                    return false;
                }
                let last_action = self.state.history[self.state.history.len() - 1];
                match last_action {
                    0 => {
                        // Previous was check, can check or bet
                        action == 0 || action == 1
                    }
                    1 => {
                        // Previous was bet, can call or fold
                        action == 0 || action == 1 // 0=call, 1=fold in this context
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Apply action to current state
    fn apply_action(&mut self, action: i32) -> EnvResult<()> {
        if !self.is_legal_action(action) {
            return Err(EnvError::InvalidAction { action });
        }

        self.state.history.push(action);

        match (self.state.round, action) {
            (0, 0) => {
                // First round, check
                self.state.current_player = 1 - self.state.current_player;

                // Check if both players checked
                if self.state.history.len() == 2
                    && self.state.history[0] == 0
                    && self.state.history[1] == 0
                {
                    self.resolve_showdown()?;
                } else if self.state.history.len() == 2 {
                    // Move to second round
                    self.state.round = 1;
                    self.state.current_player = 0; // Player 0 acts first in second round
                }
            }
            (0, 1) => {
                // First round, bet
                self.state.contributions[self.state.current_player as usize] += 1;
                self.state.pot += 1;
                self.state.current_player = 1 - self.state.current_player;
            }
            (1, 0) => {
                // Second round, check or call
                let last_action = self.state.history[self.state.history.len() - 2];
                if last_action == 0 {
                    // Check after check
                    self.state.current_player = 1 - self.state.current_player;
                    self.resolve_showdown()?;
                } else {
                    // Call bet
                    self.state.contributions[self.state.current_player as usize] += 1;
                    self.state.pot += 1;
                    self.resolve_showdown()?;
                }
            }
            (1, 1) => {
                // Second round, bet or fold
                let last_action = self.state.history[self.state.history.len() - 2];
                if last_action == 0 {
                    // Bet after check
                    self.state.contributions[self.state.current_player as usize] += 1;
                    self.state.pot += 1;
                    self.state.current_player = 1 - self.state.current_player;
                } else {
                    // Fold
                    let winner = 1 - self.state.current_player;
                    self.state.payoffs[winner as usize] = self.state.pot as f32;
                    self.state.payoffs[self.state.current_player as usize] =
                        -(self.state.pot as f32);
                    self.state.terminal = true;
                }
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Resolve showdown and determine payoffs
    fn resolve_showdown(&mut self) -> EnvResult<()> {
        let winner = if self.state.cards[0] > self.state.cards[1] {
            0
        } else {
            1
        };
        self.state.payoffs[winner as usize] = self.state.pot as f32;
        self.state.payoffs[1 - winner as usize] = -(self.state.pot as f32);
        self.state.terminal = true;
        Ok(())
    }

    /// Get legal actions mask
    fn get_legal_actions_mask(&self) -> Array1<i32> {
        let mut mask = Array1::zeros(2);
        if !self.state.terminal {
            for action in 0..2 {
                if self.is_legal_action(action) {
                    mask[action as usize] = 1;
                }
            }
        }
        mask
    }
}

impl Environment for KuhnPokerEnv {
    fn reset(&mut self, seed: Option<u64>) -> EnvResult<()> {
        if let Some(seed) = seed {
            self.rng = Pcg64Mcg::seed_from_u64(seed);
        }

        self.state = KuhnPokerState::default();
        self.deal_cards()?;
        Ok(())
    }

    fn observation(&self) -> Array1<f32> {
        let mut obs = Array1::zeros(6);

        // Player's card (normalized)
        obs[0] = self.state.cards[self.state.current_player as usize] as f32 / 2.0;

        // Pot size (normalized, assume max pot is 4)
        obs[1] = self.state.pot as f32 / 4.0;

        // Round (normalized)
        obs[2] = self.state.round as f32;

        // Current player
        obs[3] = self.state.current_player as f32;

        // Legal actions mask
        let legal_mask = self.get_legal_actions_mask();
        obs[4] = legal_mask[0] as f32;
        obs[5] = legal_mask[1] as f32;

        obs
    }

    fn legal_actions(&self) -> Array1<i32> {
        let mut actions = Vec::new();
        for action in 0..2 {
            if self.is_legal_action(action) {
                actions.push(action);
            }
        }
        Array1::from(actions)
    }

    fn current_player(&self) -> i32 {
        if self.state.terminal {
            -1
        } else {
            self.state.current_player
        }
    }

    fn is_terminal(&self) -> bool {
        self.state.terminal
    }

    fn rewards(&self) -> Array1<f32> {
        Array1::from(self.state.payoffs.to_vec())
    }

    fn step(&mut self, action: i32) -> EnvResult<(Array1<f32>, f32, bool)> {
        if self.state.terminal {
            return Err(EnvError::AlreadyTerminal);
        }

        let player = self.state.current_player;
        self.apply_action(action)?;

        let reward = if self.state.terminal {
            self.state.payoffs[player as usize]
        } else {
            0.0
        };

        Ok((
            self.observation(),
            reward * self.config.reward_scale,
            self.state.terminal,
        ))
    }

    fn info_state(&self, player: i32) -> Array1<f32> {
        let mut info = Array1::zeros(8);

        // Player's card
        info[0] = self.state.cards[player as usize] as f32 / 2.0;

        // Opponent's card (hidden during game)
        info[1] = -1.0; // Unknown

        // Pot size
        info[2] = self.state.pot as f32 / 4.0;

        // Round
        info[3] = self.state.round as f32;

        // Current player
        info[4] = self.state.current_player as f32;

        // Player's contribution
        info[5] = self.state.contributions[player as usize] as f32 / 2.0;

        // Opponent's contribution
        info[6] = self.state.contributions[1 - player as usize] as f32 / 2.0;

        // Betting history (simplified)
        info[7] = if self.state.history.is_empty() {
            0.0
        } else {
            self.state.history[self.state.history.len() - 1] as f32
        };

        info
    }

    fn game_info(&self) -> GameInfo {
        self.game_info.clone()
    }

    fn clone_state(&self) -> Box<dyn Environment> {
        Box::new(KuhnPokerEnv {
            state: self.state.clone(),
            rng: Pcg64Mcg::seed_from_u64(42), // Rng state not critical for cloned state
            config: self.config.clone(),
            game_info: self.game_info.clone(),
        })
    }

    fn restore_state(&mut self, state: &Box<dyn Environment>) -> EnvResult<()> {
        let other = state
            .as_any()
            .downcast_ref::<KuhnPokerEnv>()
            .ok_or_else(|| EnvError::EncodingError {
                message: "Cannot restore from different environment type".to_string(),
            })?;

        self.state = other.state.clone();
        Ok(())
    }
}

#[pymethods]
impl KuhnPokerEnv {
    #[new]
    fn py_new(config: Option<EnvConfig>) -> PyResult<Self> {
        Ok(Self::new(config.unwrap_or_default()))
    }

    fn reset(&mut self, seed: Option<u64>) -> PyResult<()> {
        self.reset(seed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Reset failed: {}", e)))
    }

    fn observation(&self) -> PyArray1<f32> {
        self.observation()
            .into_pyarray(Python::acquire_gil().asgil())
    }

    fn legal_actions(&self) -> PyArray1<i32> {
        self.legal_actions()
            .into_pyarray(Python::acquire_gil().asgil())
    }

    fn current_player(&self) -> i32 {
        self.current_player()
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal()
    }

    fn rewards(&self) -> PyArray1<f32> {
        self.rewards().into_pyarray(Python::acquire_gil().asgil())
    }

    fn step(&mut self, action: i32) -> PyResult<(PyArray1<f32>, f32, bool)> {
        let (obs, reward, done) = self.step(action).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Step failed: {}", e))
        })?;

        let py = Python::acquire_gil().asgil();
        let obs_array = obs.into_pyarray(py);
        Ok((obs_array, reward, done))
    }

    fn info_state(&self, player: i32) -> PyArray1<f32> {
        self.info_state(player)
            .into_pyarray(Python::acquire_gil().asgil())
    }

    fn get_game_info(&self) -> PyResult<serde_json::Value> {
        let info = serde_json::json!({
            "name": self.game_info.name,
            "num_players": self.game_info.num_players,
            "num_actions": self.game_info.num_actions,
            "max_game_length": self.game_info.max_game_length,
            "observation_shape": self.game_info.observation_shape,
            "action_shape": self.game_info.action_shape
        });
        Ok(info)
    }

    fn get_state(&self) -> PyResult<serde_json::Value> {
        let state_json = serde_json::json!({
            "cards": self.state.cards,
            "round": self.state.round,
            "current_player": self.state.current_player,
            "pot": self.state.pot,
            "contributions": self.state.contributions,
            "history": self.state.history,
            "terminal": self.state.terminal,
            "payoffs": self.state.payoffs
        });
        Ok(state_json)
    }

    fn get_rng_state(&self) -> PyResult<Vec<u8>> {
        let state = crate::utils::RngState::from_pcg64(&self.rng);
        serde_json::to_vec(&state).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e))
        })
    }

    fn set_rng_state(&mut self, state_bytes: Vec<u8>) -> PyResult<()> {
        let state: crate::utils::RngState = serde_json::from_slice(&state_bytes).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Deserialization failed: {}", e))
        })?;

        self.rng = state
            .to_pcg64()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Invalid RNG state"))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "KuhnPokerEnv(terminal={}, player={}, pot={})",
            self.state.terminal, self.state.current_player, self.state.pot
        )
    }
}

impl fmt::Display for KuhnPokerEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KuhnPoker: cards={:?}, player={}, pot={}, terminal={}",
            self.state.cards, self.state.current_player, self.state.pot, self.state.terminal
        )
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

        // Test check action
        let (obs, reward, done) = env.step(0).unwrap();
        assert!(!done);
        assert_eq!(reward, 0.0);

        // Second player checks
        let (obs, reward, done) = env.step(0).unwrap();
        assert!(done); // Showdown should complete
    }

    #[test]
    fn test_kuhn_poker_betting() {
        let config = EnvConfig {
            seed: Some(42),
            ..Default::default()
        };

        let mut env = KuhnPokerEnv::new(config);
        env.reset(Some(42)).unwrap();

        // First player bets
        let (obs, reward, done) = env.step(1).unwrap();
        assert!(!done);
        assert_eq!(env.state.pot, 3); // 2 ante + 1 bet

        // Second player calls
        let (obs, reward, done) = env.step(0).unwrap();
        assert!(done);
        assert_eq!(env.state.pot, 4); // 2 ante + 1 bet + 1 call
    }

    #[test]
    fn test_invalid_actions() {
        let config = EnvConfig {
            seed: Some(42),
            ..Default::default()
        };

        let mut env = KuhnPokerEnv::new(config);
        env.reset(Some(42)).unwrap();

        // Invalid action
        let result = env.step(2);
        assert!(result.is_err());
    }
}
