from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearMDP:
    """Synthetic finite-horizon MDP with explicit state-action features."""

    name: str
    horizon: int
    num_states: int
    num_actions: int
    feature_dim: int
    features: np.ndarray
    rewards: np.ndarray
    transition_probs: np.ndarray
    initial_state_dist: np.ndarray


@dataclass(frozen=True)
class LinearOfflineTask:
    """Linear task plus a behavior policy used for offline data collection."""

    mdp: LinearMDP
    behavior_policy: np.ndarray
    support_threshold: int
    description: str


@dataclass(frozen=True)
class LinearTransition:
    step: int
    state: int
    action: int
    feature: np.ndarray
    reward: float
    next_state: int


@dataclass(frozen=True)
class LinearOfflineDataset:
    transitions: list[LinearTransition]
    num_episodes: int


@dataclass(frozen=True)
class StepRegressionData:
    features: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class LinearPlannerResult:
    policy: np.ndarray
    q_values: np.ndarray
    q_hat: np.ndarray
    bonuses: np.ndarray
    step_thetas: list[np.ndarray]
