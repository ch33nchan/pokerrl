//! Leduc Poker environment implementation in Rust
//!
//! This module provides a high-performance implementation of Leduc Poker
//! with deterministic replay support, efficient state management, and
//! Python bindings for integration with the ARMAC training system.

use crate::{EnvConfig, EnvError, EnvResult, Environment, GameInfo};
use ndarray::Array1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
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
    /// Player card identifiers (0..5 to disambiguate duplicate ranks)
    pub player_card_ids: [Option<i32>; 2],
    /// Community card
    pub community_card: Option<Card>,
    /// Community card identifier
    pub community_card_id: Option<i32>,
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
    /// Last bet size
    pub last_bet: i32,
    /// Highest contribution to match in the current round
    pub current_bet: i32,
    /// Number of raises in the current round
    pub raises_in_round: i32,
    /// Track if previous action was a check (used to detect check-check)
    pub last_action_was_check: bool,
}

impl Default for LeducPokerState {
    fn default() -> Self {
        Self {
            player_cards: [None, None],
            player_card_ids: [None, None],
            community_card: None,
            community_card_id: None,
            round: Round::PreFlop,
            current_player: 0,
            pot: 2, // 1 ante from each player
            contributions: [1, 1],
            history: Vec::new(),
            terminal: false,
            payoffs: [0.0, 0.0],
            last_bet: 0,
            current_bet: 1,
            raises_in_round: 0,
            last_action_was_check: false,
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
    deck: Vec<(Card, i32)>,
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
            (Card::Jack, 0),
            (Card::Jack, 1),
            (Card::Queen, 2),
            (Card::Queen, 3),
            (Card::King, 4),
            (Card::King, 5),
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
        let (card0, id0) = self.deck[0];
        let (card1, id1) = self.deck[1];
        self.state.player_cards[0] = Some(card0);
        self.state.player_card_ids[0] = Some(id0);
        self.state.player_cards[1] = Some(card1);
        self.state.player_card_ids[1] = Some(id1);

        // Community card will be dealt after pre-flop betting
        self.state.community_card = None;
        self.state.community_card_id = None;
        self.state.current_bet = self.state.contributions[0];
        self.state.raises_in_round = 0;
        self.state.last_action_was_check = false;

        Ok(())
    }

    /// Deal community card for flop
    fn deal_community_card(&mut self) -> EnvResult<()> {
        if self.deck.len() < 3 {
            return Err(EnvError::EncodingError {
                message: "No cards left for community card".to_string(),
            });
        }

        // Find next unused card
        for (card, id) in &self.deck {
            let already_used = self.state.player_card_ids.iter().any(|&c| c == Some(*id));
            if !already_used {
                self.state.community_card = Some(*card);
                self.state.community_card_id = Some(*id);
                return Ok(());
            }
        }

        Err(EnvError::EncodingError {
            message: "Unable to select community card".to_string(),
        })
    }

    /// Check if action is legal
    fn is_legal_action(&self, action: i32) -> bool {
        if self.state.terminal {
            return false;
        }

        let to_call = self.get_call_amount();
        let max_raises_per_round = 2;

        // Actions: 0=Fold, 1=Call, 2=Raise
        match action {
            0 => to_call > 0,
            1 => true, // Check (when to_call == 0) or call (when to_call > 0)
            2 => self.state.raises_in_round < max_raises_per_round,
            _ => false,
        }
    }

    /// Get call amount
    fn get_call_amount(&self) -> i32 {
        let player = self.state.current_player as usize;
        let to_call = self.state.current_bet - self.state.contributions[player];
        to_call.max(0)
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
        let player = self.state.current_player as usize;
        let opponent = 1 - player;
        let to_call = self.get_call_amount();

        match action {
            0 => {
                // Fold (only legal when facing a bet)
                let winner = opponent as i32;
                let payoff = self.state.contributions[player] as f32;
                self.state.payoffs[winner as usize] = payoff;
                self.state.payoffs[player] = -payoff;
                self.state.terminal = true;
            }
            1 => {
                if to_call > 0 {
                    // Call: match the outstanding bet and advance the round
                    self.state.contributions[player] += to_call;
                    self.state.pot += to_call;
                    self.state.current_bet = self.state.contributions[player];
                    self.state.last_bet = to_call;
                    self.state.last_action_was_check = false;
                    self.advance_round()?;
                } else {
                    // Check: if both players checked consecutively, advance
                    if self.state.last_action_was_check {
                        self.state.last_action_was_check = false;
                        self.advance_round()?;
                    } else {
                        self.state.last_action_was_check = true;
                        self.state.current_player = opponent as i32;
                    }
                }
            }
            2 => {
                // Raise or bet
                let raise_amount = self.get_raise_amount();
                let total_bet = to_call + raise_amount;
                self.state.contributions[player] += total_bet;
                self.state.pot += total_bet;
                self.state.last_bet = total_bet;
                self.state.current_bet = self.state.contributions[player];
                self.state.raises_in_round += 1;
                self.state.last_action_was_check = false;
                self.state.current_player = opponent as i32;
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
                self.state.raises_in_round = 0;
                self.state.last_bet = 0;
                self.state.current_bet = self.state.contributions[0].max(self.state.contributions[1]);
                self.state.current_player = 0; // Player 0 acts first in flop
                self.state.last_action_was_check = false;
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

        if !self.state.terminal {
            self.state.last_action_was_check = false;
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

        // Determine winner (None indicates a tie)
        let outcome = if let Some(community_card) = self.state.community_card {
            // With community card, check for pairs
            let player_pair = player_card.matches(&community_card);
            let opponent_pair = opponent_card.matches(&community_card);

            if player_pair && !opponent_pair {
                Some(0)
            } else if opponent_pair && !player_pair {
                Some(1)
            } else if player_pair && opponent_pair {
                match player_card.value().cmp(&opponent_card.value()) {
                    Ordering::Greater => Some(0),
                    Ordering::Less => Some(1),
                    Ordering::Equal => None,
                }
            } else {
                // No pairs, compare private cards
                match player_card.value().cmp(&opponent_card.value()) {
                    Ordering::Greater => Some(0),
                    Ordering::Less => Some(1),
                    Ordering::Equal => None,
                }
            }
        } else {
            // No community card, compare private cards
            match player_card.value().cmp(&opponent_card.value()) {
                Ordering::Greater => Some(0),
                Ordering::Less => Some(1),
                Ordering::Equal => None,
            }
        };

        if let Some(winner) = outcome {
            let payoff = self.state.contributions[1 - winner as usize] as f32;
            self.state.payoffs[winner as usize] = payoff;
            self.state.payoffs[1 - winner as usize] = -payoff;
        } else {
            self.state.payoffs = [0.0, 0.0];
        }
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
        obs[9] = legal_mask[1] as f32; // Call / Check

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
        info[9] = self.state.raises_in_round as f32 / 2.0;

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
    fn py_observation(&self) -> PyResult<Vec<f32>> {
        let obs = <Self as Environment>::observation(self);
        Ok(obs.to_vec())
    }

    #[pyo3(name = "legal_actions")]
    fn py_legal_actions(&self) -> PyResult<Vec<i32>> {
        let acts = <Self as Environment>::legal_actions(self);
        Ok(acts.to_vec())
    }

    fn current_player(&self) -> i32 {
        <Self as Environment>::current_player(self)
    }

    fn is_terminal(&self) -> bool {
        <Self as Environment>::is_terminal(self)
    }

    #[pyo3(name = "rewards")]
    fn py_rewards(&self) -> PyResult<Vec<f32>> {
        let rewards = <Self as Environment>::rewards(self);
        Ok(rewards.to_vec())
    }

    #[pyo3(name = "step")]
    fn py_step(&mut self, action: i32) -> PyResult<(Vec<f32>, f32, bool)> {
        let (obs, reward, done) = <Self as Environment>::step(self, action)
            .map_err(|e| PyRuntimeError::new_err(format!("Step failed: {}", e)))?;
        Ok((obs.to_vec(), reward, done))
    }

    #[pyo3(name = "info_state")]
    fn py_info_state(&self, player: i32) -> PyResult<Vec<f32>> {
        let info = <Self as Environment>::info_state(self, player);
        Ok(info.to_vec())
    }

    fn get_game_info<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let info = PyDict::new(py);
        info.set_item("name", &self.game_info.name)?;
        info.set_item("num_players", self.game_info.num_players)?;
        info.set_item("num_actions", self.game_info.num_actions)?;
        info.set_item("max_game_length", self.game_info.max_game_length)?;
        info.set_item("observation_shape", self.game_info.observation_shape.clone())?;
        info.set_item("action_shape", self.game_info.action_shape.clone())?;
        Ok(info)
    }

    fn get_state<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let state = PyDict::new(py);
        let player_card_ranks: Vec<Option<i32>> = self
            .state
            .player_cards
            .iter()
            .map(|card| card.map(|c| c.value()))
            .collect();
        let player_card_ids: Vec<Option<i32>> = self
            .state
            .player_card_ids
            .iter()
            .map(|id| *id)
            .collect();
        state.set_item("player_cards", player_card_ranks)?;
        state.set_item("player_card_ids", player_card_ids)?;
        state.set_item(
            "community_card",
            self.state.community_card.map(|card| card.value()),
        )?;
        state.set_item("community_card_id", self.state.community_card_id)?;
        state.set_item("round", self.state.round as i32)?;
        state.set_item("current_player", self.state.current_player)?;
        state.set_item("pot", self.state.pot)?;
        state.set_item("contributions", self.state.contributions.to_vec())?;
        state.set_item("history", self.state.history.clone())?;
        state.set_item("terminal", self.state.terminal)?;
        state.set_item("payoffs", self.state.payoffs.to_vec())?;
        state.set_item("last_bet", self.state.last_bet)?;
        state.set_item("current_bet", self.state.current_bet)?;
        state.set_item("raises_in_round", self.state.raises_in_round)?;
        state.set_item("last_action_was_check", self.state.last_action_was_check)?;
        Ok(state)
    }

    fn __repr__(&self) -> String {
        format!(
            "LeducPokerEnv(terminal={}, player={}, round={}, pot={}, raises={})",
            self.state.terminal,
            self.state.current_player,
            self.state.round as i32,
            self.state.pot,
            self.state.raises_in_round
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
        assert_eq!(env.legal_actions().to_vec(), vec![1, 2]); // Check/Call or Raise

        // Test opening check
        let (_obs, reward, done) = env.step(1).unwrap();
        assert!(!done);
        assert_eq!(reward, 0.0);
        assert_eq!(env.current_player(), 1);
    }

    #[test]
    fn test_leduc_poker_betting_rounds() {
        let config = EnvConfig {
            seed: Some(42),
            ..Default::default()
        };

        let mut env = LeducPokerEnv::new(config);
        env.reset(Some(42)).unwrap();

        // Both players check
        let (_obs, _reward, done) = env.step(1).unwrap();
        assert!(!done);

        let (_obs, _reward, done) = env.step(1).unwrap();
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
        let (_obs, _reward, done) = env.step(2).unwrap();
        assert!(!done);
        assert_eq!(env.state.last_bet, 2); // Initial raise amount

        // Second player calls
        let (_obs, _reward, done) = env.step(1).unwrap();
        assert!(!done); // Should advance to flop
        assert_eq!(env.state.round, Round::Flop);
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
        env.step(2).unwrap(); // Player 0 raises
        env.step(1).unwrap(); // Player 1 calls - advance to flop
        env.step(2).unwrap(); // Player 0 raises on flop
        env.step(1).unwrap(); // Player 1 calls - should reach showdown

        assert!(env.is_terminal());
        assert!(env.state.payoffs[0] != 0.0 || env.state.payoffs[1] != 0.0);
    }
}
