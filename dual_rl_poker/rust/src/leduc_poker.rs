//! Leduc Poker environment implementation in Rust
//!
//! This module provides a high-performance implementation of Leduc Poker
//! with deterministic replay support, efficient state management, and
//! Python bindings for integration with the ARMAC training system.

use crate::{EnvConfig, EnvError, EnvResult, Environment, GameInfo};
use ndarray::Array1;
use numpy::PyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Card values in Leduc Poker
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Card {
    Jack = 0,
    Queen = 1,
    King = 2,
}

impl Card {
    /// Get numeric value
    pub fn value(&self) -> i32 {
        *self as i32
    }

    /// Get card from value
    pub fn from_value(value: i32) -> Option<Self> {
        match value {
            0 => Some(Card::Jack),
            1 => Some(Card::Queen),
            2 => Some(Card::King),
            _ => None,
        }
    }

    /// Check if cards match (pair)
    pub fn matches(&self, other: &Card) -> bool {
        self == other
    }
}

/// Betting rounds in Leduc Poker
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Round {
    PreFlop = 0,
    Flop = 1,
    Showdown = 2,
}

/// Leduc Poker game state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeducPokerState {
    /// Player cards
    pub player_cards: [Option<Card>; 2],
    /// Community card
    pub community_card: Option<Card>,
    /// Current betting round
    pub round: Round,
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
    /// Number of actions in current round
    pub round_actions: i32,
    /// Last bet size
    pub last_bet: i32,
}

impl Default for LeducPokerState {
    fn default() -> Self {
        Self {
            player_cards: [None, None],
            community_card: None,
            round: Round::PreFlop,
            current_player: 0,
            pot: 4, // 2 ante from each player
            contributions: [2, 2],
            history: Vec::new(),
            terminal: false,
            payoffs: [0.0, 0.0],
            round_actions: 0,
            last_bet: 0,
        }
    }
}

/// Leduc Poker environment
#[pyclass]
pub struct LeducPokerEnv {
    /// Current game state
    state: LeducPokerState,
    /// Random number generator
    rng: Pcg64Mcg,
    /// Environment configuration
    config: EnvConfig,
    /// Game information
    game_info: GameInfo,
    /// Deck for dealing
    deck: Vec<Card>,
}

impl LeducPokerEnv {
    /// Create new Leduc Poker environment
    pub fn new(config: EnvConfig) -> Self {
        let game_info = GameInfo {
            name: "leduc_poker".to_string(),
            num_players: 2,
            num_actions: 3, // Fold, Call, Raise
            max_game_length: 12,
            observation_shape: vec![10], // Extended observation space
            action_shape: vec![1],
        };

        Self {
            state: LeducPokerState::default(),
            rng: Pcg64Mcg::seed_from_u64(config.seed.unwrap_or(42)),
            config,
            game_info,
            deck: Vec::new(),
        }
    }

    /// Initialize deck for Leduc Poker (6 cards: 2 of each rank)
    fn initialize_deck(&mut self) {
        self.deck = vec![
            Card::Jack,
            Card::Jack,
            Card::Queen,
            Card::Queen,
            Card::King,
            Card::King,
        ];
    }

    /// Shuffle deck
    fn shuffle_deck(&mut self) {
        self.deck.shuffle(&mut self.rng);
    }

    /// Deal cards to players
    fn deal_cards(&mut self) -> EnvResult<()> {
        self.initialize_deck();
        self.shuffle_deck();

        // Deal private cards
        self.state.player_cards[0] = Some(self.deck[0]);
        self.state.player_cards[1] = Some(self.deck[1]);

        // Community card will be dealt after pre-flop betting
        self.state.community_card = None;

        Ok(())
    }

    /// Deal community card for flop
    fn deal_community_card(&mut self) -> EnvResult<()> {
        if self.deck.len() >= 2 {
            self.state.community_card = Some(self.deck[2]);
            Ok(())
        } else {
            Err(EnvError::EncodingError {
                message: "No cards left for community card".to_string(),
            })
        }
    }

    /// Check if action is legal
    fn is_legal_action(&self, action: i32) -> bool {
        if self.state.terminal {
            return false;
        }

        // Actions: 0=Fold, 1=Call, 2=Raise
        match action {
            0 => true, // Can always fold
            1 => {
                // Can call if there's a bet to match
                self.state.last_bet > 0 || self.state.round_actions == 0
            }
            2 => {
                // Can raise (limit to 2 raises per round)
                self.state.round_actions < 4
            }
            _ => false,
        }
    }

    /// Get call amount
    fn get_call_amount(&self) -> i32 {
        if self.state.last_bet > 0 {
            self.state.last_bet - self.state.contributions[self.state.current_player as usize]
        } else {
            0
        }
    }

    /// Get raise amount
    fn get_raise_amount(&self) -> i32 {
        match self.state.round {
            Round::PreFlop => 2,
            Round::Flop => 4,
            _ => 0,
        }
    }

    /// Apply action to current state
    fn apply_action(&mut self, action: i32) -> EnvResult<()> {
        if !self.is_legal_action(action) {
            return Err(EnvError::InvalidAction { action });
        }

        self.state.history.push(action);
        self.state.round_actions += 1;

        match action {
            0 => {
                // Fold
                let winner = 1 - self.state.current_player;
                self.state.payoffs[winner as usize] = self.state.pot as f32;
                self.state.payoffs[self.state.current_player as usize] = -(self.state.pot as f32);
                self.state.terminal = true;
            }
            1 => {
                // Call
                let call_amount = self.get_call_amount();
                self.state.contributions[self.state.current_player as usize] += call_amount;
                self.state.pot += call_amount;

                // Check if betting round is complete
                if self.state.round_actions >= 2 {
                    self.advance_round()?;
                } else {
                    self.state.current_player = 1 - self.state.current_player;
                }
            }
            2 => {
                // Raise
                let raise_amount = self.get_raise_amount();
                let total_bet = self.get_call_amount() + raise_amount;
                self.state.contributions[self.state.current_player as usize] += total_bet;
                self.state.pot += total_bet;
                self.state.last_bet = self.state.contributions[self.state.current_player as usize];

                self.state.current_player = 1 - self.state.current_player;
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Advance to next round
    fn advance_round(&mut self) -> EnvResult<()> {
        match self.state.round {
            Round::PreFlop => {
                // Move to flop
                self.state.round = Round::Flop;
                self.deal_community_card()?;
                self.state.round_actions = 0;
                self.state.last_bet = 0;
                self.state.current_player = 0; // Player 0 acts first in flop
            }
            Round::Flop => {
                // Move to showdown
                self.state.round = Round::Showdown;
                self.resolve_showdown()?;
            }
            Round::Showdown => {
                self.state.terminal = true;
            }
        }

        Ok(())
    }

    /// Resolve showdown and determine payoffs
    fn resolve_showdown(&mut self) -> EnvResult<()> {
        let player_card = self.state.player_cards[0].ok_or_else(|| EnvError::EncodingError {
            message: "Player 0 has no card".to_string(),
        })?;
        let opponent_card = self.state.player_cards[1].ok_or_else(|| EnvError::EncodingError {
            message: "Player 1 has no card".to_string(),
        })?;

        // Determine winner
        let winner = if let Some(community_card) = self.state.community_card {
            // With community card, check for pairs
            let player_pair = player_card.matches(&community_card);
            let opponent_pair = opponent_card.matches(&community_card);

            if player_pair && !opponent_pair {
                0
            } else if opponent_pair && !player_pair {
                1
            } else if player_pair && opponent_pair {
                // Both have pairs, compare card values
                if player_card.value() > opponent_card.value() {
                    0
                } else {
                    1
                }
            } else {
                // No pairs, compare private cards
                if player_card.value() > opponent_card.value() {
                    0
                } else {
                    1
                }
            }
        } else {
            // No community card, compare private cards
            if player_card.value() > opponent_card.value() {
                0
            } else {
                1
            }
        };

        self.state.payoffs[winner as usize] = self.state.pot as f32;
        self.state.payoffs[1 - winner as usize] = -(self.state.pot as f32);
        self.state.terminal = true;

        Ok(())
    }

    /// Get legal actions mask
    fn get_legal_actions_mask(&self) -> Array1<i32> {
        let mut mask = Array1::zeros(3);
        if !self.state.terminal {
            for action in 0..3 {
                if self.is_legal_action(action) {
                    mask[action as usize] = 1;
                }
            }
        }
        mask
    }

    /// Normalize card value for observation
    fn normalize_card(&self, card: Option<Card>) -> f32 {
        match card {
            Some(Card::Jack) => 0.0,
            Some(Card::Queen) => 0.5,
            Some(Card::King) => 1.0,
            None => -1.0,
        }
    }
}

impl Environment for LeducPokerEnv {
    fn reset(&mut self, seed: Option<u64>) -> EnvResult<()> {
        if let Some(seed) = seed {
            self.rng = Pcg64Mcg::seed_from_u64(seed);
        }

        self.state = LeducPokerState::default();
        self.deal_cards()?;
        Ok(())
    }

    fn observation(&self) -> Array1<f32> {
        let mut obs = Array1::zeros(10);

        // Player's card (normalized)
        obs[0] = self.normalize_card(self.state.player_cards[self.state.current_player as usize]);

        // Community card (normalized)
        obs[1] = self.normalize_card(self.state.community_card);

        // Pot size (normalized, assume max pot is 20)
        obs[2] = self.state.pot as f32 / 20.0;

        // Round (normalized)
        obs[3] = self.state.round as i32 as f32 / 2.0;

        // Current player
        obs[4] = self.state.current_player as f32;

        // Player's contribution (normalized)
        obs[5] = self.state.contributions[self.state.current_player as usize] as f32 / 10.0;

        // Opponent's contribution (normalized)
        obs[6] = self.state.contributions[1 - self.state.current_player as usize] as f32 / 10.0;

        // Last bet (normalized)
        obs[7] = self.state.last_bet as f32 / 10.0;

        // Legal actions mask
        let legal_mask = self.get_legal_actions_mask();
        obs[8] = legal_mask[0] as f32; // Fold
        obs[9] = legal_mask[1] as f32; // Call

        obs
    }

    fn legal_actions(&self) -> Array1<i32> {
        let mut actions = Vec::new();
        for action in 0..3 {
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
        let mut info = Array1::zeros(12);

        // Player's card
        info[0] = self.normalize_card(self.state.player_cards[player as usize]);

        // Opponent's card (hidden during game)
        info[1] = -1.0;

        // Community card
        info[2] = self.normalize_card(self.state.community_card);

        // Pot size
        info[3] = self.state.pot as f32 / 20.0;

        // Round
        info[4] = self.state.round as i32 as f32 / 2.0;

        // Current player
        info[5] = self.state.current_player as f32;

        // Player's contribution
        info[6] = self.state.contributions[player as usize] as f32 / 10.0;

        // Opponent's contribution
        info[7] = self.state.contributions[1 - player as usize] as f32 / 10.0;

        // Last bet
        info[8] = self.state.last_bet as f32 / 10.0;

        // Round actions
        info[9] = self.state.round_actions as f32 / 4.0;

        // Betting history (last 2 actions)
        if self.state.history.len() > 0 {
            info[10] = self.state.history[self.state.history.len() - 1] as f32 / 2.0;
        }
        if self.state.history.len() > 1 {
            info[11] = self.state.history[self.state.history.len() - 2] as f32 / 2.0;
        }

        info
    }

    fn game_info(&self) -> GameInfo {
        self.game_info.clone()
    }

}

#[pymethods]
impl LeducPokerEnv {
    #[new]
    fn py_new(seed: Option<u64>) -> PyResult<Self> {
        let mut config = EnvConfig::default();
        config.seed = seed;
        Ok(Self::new(config))
    }

    fn reset(&mut self, seed: Option<u64>) -> PyResult<()> {
        <Self as Environment>::reset(self, seed)
            .map_err(|e| PyRuntimeError::new_err(format!("Reset failed: {}", e)))
    }

    #[pyo3(name = "observation")]
    fn py_observation<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f32>> {
        let obs = <Self as Environment>::observation(self);
        Ok(PyArray1::from_vec(py, obs.to_vec()))
    }

    #[pyo3(name = "legal_actions")]
    fn py_legal_actions<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<i32>> {
        let acts = <Self as Environment>::legal_actions(self);
        Ok(PyArray1::from_vec(py, acts.to_vec()))
    }

    fn current_player(&self) -> i32 {
        <Self as Environment>::current_player(self)
    }

    fn is_terminal(&self) -> bool {
        <Self as Environment>::is_terminal(self)
    }

    #[pyo3(name = "rewards")]
    fn py_rewards<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f32>> {
        let rewards = <Self as Environment>::rewards(self);
        Ok(PyArray1::from_vec(py, rewards.to_vec()))
    }

    #[pyo3(name = "step")]
    fn py_step<'py>(&mut self, py: Python<'py>, action: i32) -> PyResult<(&'py PyArray1<f32>, f32, bool)> {
        let (obs, reward, done) = <Self as Environment>::step(self, action)
            .map_err(|e| PyRuntimeError::new_err(format!("Step failed: {}", e)))?;
        Ok((PyArray1::from_vec(py, obs.to_vec()), reward, done))
    }

    #[pyo3(name = "info_state")]
    fn py_info_state<'py>(&self, py: Python<'py>, player: i32) -> PyResult<&'py PyArray1<f32>> {
        let info = <Self as Environment>::info_state(self, player);
        Ok(PyArray1::from_vec(py, info.to_vec()))
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
        let player_cards_json = serde_json::json!([
            self.state.player_cards[0].map(|c| c.value()),
            self.state.player_cards[1].map(|c| c.value())
        ]);

        let state_json = serde_json::json!({
            "player_cards": player_cards_json,
            "community_card": self.state.community_card.map(|c| c.value()),
            "round": self.state.round as i32,
            "current_player": self.state.current_player,
            "pot": self.state.pot,
            "contributions": self.state.contributions,
            "history": self.state.history,
            "terminal": self.state.terminal,
            "payoffs": self.state.payoffs,
            "round_actions": self.state.round_actions,
            "last_bet": self.state.last_bet
        });
        Ok(state_json)
    }

    fn __repr__(&self) -> String {
        format!(
            "LeducPokerEnv(terminal={}, player={}, round={}, pot={})",
            self.state.terminal, self.state.current_player, self.state.round as i32, self.state.pot
        )
    }
}

impl fmt::Display for LeducPokerEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LeducPoker: player_cards={:?}, community={:?}, player={}, round={}, pot={}, terminal={}",
            self.state.player_cards,
            self.state.community_card,
            self.state.current_player,
            self.state.round as i32,
            self.state.pot,
            self.state.terminal
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leduc_poker_basic() {
        let config = EnvConfig {
            seed: Some(42),
            ..Default::default()
        };

        let mut env = LeducPokerEnv::new(config);
        env.reset(Some(42)).unwrap();

        assert_eq!(env.current_player(), 0);
        assert!(!env.is_terminal());
        assert_eq!(env.legal_actions().len(), 3); // Fold, Call, Raise

        // Test fold action
        let (obs, reward, done) = env.step(0).unwrap();
        assert!(done);
        assert_ne!(reward, 0.0);
    }

    #[test]
    fn test_leduc_poker_betting_rounds() {
        let config = EnvConfig {
            seed: Some(42),
            ..Default::default()
        };

        let mut env = LeducPokerEnv::new(config);
        env.reset(Some(42)).unwrap();

        // Both players call
        let (obs, reward, done) = env.step(1).unwrap();
        assert!(!done);

        let (obs, reward, done) = env.step(1).unwrap();
        assert!(!done); // Should advance to flop

        // Should be in flop round now
        assert_eq!(env.state.round, Round::Flop);
        assert!(env.state.community_card.is_some());
    }

    #[test]
    fn test_leduc_poker_raise() {
        let config = EnvConfig {
            seed: Some(42),
            ..Default::default()
        };

        let mut env = LeducPokerEnv::new(config);
        env.reset(Some(42)).unwrap();

        // First player raises
        let (obs, reward, done) = env.step(2).unwrap();
        assert!(!done);
        assert_eq!(env.state.last_bet, 4); // 2 ante + 2 raise

        // Second player calls
        let (obs, reward, done) = env.step(1).unwrap();
        assert!(!done); // Should advance to flop
    }

    #[test]
    fn test_leduc_poker_showdown() {
        let config = EnvConfig {
            seed: Some(42),
            ..Default::default()
        };

        let mut env = LeducPokerEnv::new(config);
        env.reset(Some(42)).unwrap();

        // Play through to showdown
        env.step(1).unwrap(); // Call
        env.step(1).unwrap(); // Call - advance to flop
        env.step(1).unwrap(); // Call
        env.step(1).unwrap(); // Call - should reach showdown

        assert!(env.is_terminal());
        assert!(env.state.payoffs[0] != 0.0 || env.state.payoffs[1] != 0.0);
    }
}
