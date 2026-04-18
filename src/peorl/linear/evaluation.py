from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from peorl.linear.types import LinearMDP


@dataclass(frozen=True)
class LinearSupportStats:
    expected_chosen_count: float
    low_support_mass: float
    root_action: int
    root_action_count: float
    chosen_feature_novelty: float


def evaluate_linear_policy(mdp: LinearMDP, policy: np.ndarray) -> float:
    values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=float)
    for step in range(mdp.horizon - 1, -1, -1):
        q_values = mdp.rewards[step] + np.einsum(
            "san,n->sa",
            mdp.transition_probs[step],
            values[step + 1],
        )
        values[step] = np.sum(policy[step] * q_values, axis=1)
    return float(np.dot(mdp.initial_state_dist, values[0]))


def solve_linear_task_optimal_policy(mdp: LinearMDP) -> tuple[np.ndarray, np.ndarray]:
    q_values = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=float)
    policy = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)

    for step in range(mdp.horizon - 1, -1, -1):
        q_values[step] = mdp.rewards[step] + np.einsum(
            "san,n->sa",
            mdp.transition_probs[step],
            values[step + 1],
        )
        greedy_actions = np.argmax(q_values[step], axis=1)
        policy[step, np.arange(mdp.num_states), greedy_actions] = 1.0
        values[step] = q_values[step, np.arange(mdp.num_states), greedy_actions]

    return policy, q_values


def feature_covariance_trace(features: np.ndarray) -> float:
    if features.size == 0:
        return 0.0
    covariance = features.T @ features
    return float(np.trace(covariance))


def compute_linear_policy_q_values(mdp: LinearMDP, policy: np.ndarray) -> np.ndarray:
    q_values = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=float)
    for step in range(mdp.horizon - 1, -1, -1):
        q_values[step] = mdp.rewards[step] + np.einsum(
            "san,n->sa",
            mdp.transition_probs[step],
            values[step + 1],
        )
        values[step] = np.sum(policy[step] * q_values[step], axis=1)
    return q_values


def compute_linear_state_occupancy(mdp: LinearMDP, policy: np.ndarray) -> np.ndarray:
    occupancy = np.zeros((mdp.horizon, mdp.num_states), dtype=float)
    current = mdp.initial_state_dist.copy()
    for step in range(mdp.horizon):
        occupancy[step] = current
        state_action_mass = current[:, None] * policy[step]
        current = np.einsum("sa,san->n", state_action_mass, mdp.transition_probs[step])
    return occupancy


def compute_linear_action_agreement_mass(
    mdp: LinearMDP,
    lhs_policy: np.ndarray,
    rhs_policy: np.ndarray,
) -> float:
    occupancy = compute_linear_state_occupancy(mdp, lhs_policy)
    lhs_actions = np.argmax(lhs_policy, axis=2)
    rhs_actions = np.argmax(rhs_policy, axis=2)
    agreement = 0.0
    for step in range(mdp.horizon):
        agreement += float(np.dot(occupancy[step], (lhs_actions[step] == rhs_actions[step]).astype(float)))
    return agreement / float(mdp.horizon)


def compute_linear_support_stats(
    mdp: LinearMDP,
    policy: np.ndarray,
    counts: np.ndarray,
    support_threshold: int,
    bonuses: np.ndarray | None = None,
) -> LinearSupportStats:
    occupancy = compute_linear_state_occupancy(mdp, policy)
    greedy_actions = np.argmax(policy, axis=2)
    expected_chosen_count = 0.0
    low_support_mass = 0.0
    chosen_feature_novelty = 0.0

    for step in range(mdp.horizon):
        chosen_counts = counts[step, np.arange(mdp.num_states), greedy_actions[step]]
        expected_chosen_count += float(np.dot(occupancy[step], chosen_counts))
        low_support_mass += float(np.dot(occupancy[step], (chosen_counts < support_threshold).astype(float)))
        if bonuses is not None:
            chosen_feature_novelty += float(np.dot(occupancy[step], bonuses[step, np.arange(mdp.num_states), greedy_actions[step]]))

    root_state = int(np.argmax(mdp.initial_state_dist))
    root_action = int(greedy_actions[0, root_state])
    root_action_count = float(counts[0, root_state, root_action])

    return LinearSupportStats(
        expected_chosen_count=expected_chosen_count,
        low_support_mass=low_support_mass,
        root_action=root_action,
        root_action_count=root_action_count,
        chosen_feature_novelty=chosen_feature_novelty / float(mdp.horizon),
    )
