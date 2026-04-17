from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from peorl.envs.tabular import TabularMDP


@dataclass(frozen=True)
class SupportStats:
    expected_chosen_count: float
    low_support_mass: float
    root_action: int
    root_action_count: float


def compute_policy_q_values(mdp: TabularMDP, policy: np.ndarray) -> np.ndarray:
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


def evaluate_policy(mdp: TabularMDP, policy: np.ndarray) -> float:
    values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=float)
    for step in range(mdp.horizon - 1, -1, -1):
        q_values = mdp.rewards[step] + np.einsum(
            "san,n->sa",
            mdp.transition_probs[step],
            values[step + 1],
        )
        values[step] = np.sum(policy[step] * q_values, axis=1)
    return float(np.dot(mdp.initial_state_dist, values[0]))


def solve_optimal_policy(mdp: TabularMDP) -> tuple[np.ndarray, np.ndarray]:
    q_values = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=float)
    policy = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)

    for step in range(mdp.horizon - 1, -1, -1):
        q_values[step] = mdp.rewards[step] + np.einsum(
            "san,n->sa",
            mdp.transition_probs[step],
            values[step + 1],
        )
        best_actions = np.argmax(q_values[step], axis=1)
        policy[step, np.arange(mdp.num_states), best_actions] = 1.0
        values[step] = q_values[step, np.arange(mdp.num_states), best_actions]

    return policy, q_values


def compute_state_occupancy(mdp: TabularMDP, policy: np.ndarray) -> np.ndarray:
    occupancy = np.zeros((mdp.horizon, mdp.num_states), dtype=float)
    current = mdp.initial_state_dist.copy()

    for step in range(mdp.horizon):
        occupancy[step] = current
        state_action_mass = current[:, None] * policy[step]
        current = np.einsum("sa,san->n", state_action_mass, mdp.transition_probs[step])

    return occupancy


def compute_support_stats(
    mdp: TabularMDP,
    policy: np.ndarray,
    counts: np.ndarray,
    support_threshold: int,
) -> SupportStats:
    occupancy = compute_state_occupancy(mdp, policy)
    greedy_actions = np.argmax(policy, axis=2)

    expected_chosen_count = 0.0
    low_support_mass = 0.0
    for step in range(mdp.horizon):
        chosen_counts = counts[step, np.arange(mdp.num_states), greedy_actions[step]]
        expected_chosen_count += float(np.dot(occupancy[step], chosen_counts))
        low_support_mass += float(np.dot(occupancy[step], (chosen_counts < support_threshold).astype(float)))

    root_state = int(np.argmax(mdp.initial_state_dist))
    root_action = int(greedy_actions[0, root_state])
    root_action_count = float(counts[0, root_state, root_action])

    return SupportStats(
        expected_chosen_count=expected_chosen_count,
        low_support_mass=low_support_mass,
        root_action=root_action,
        root_action_count=root_action_count,
    )


def compute_action_agreement_mass(
    mdp: TabularMDP,
    lhs_policy: np.ndarray,
    rhs_policy: np.ndarray,
) -> float:
    occupancy = compute_state_occupancy(mdp, lhs_policy)
    lhs_actions = np.argmax(lhs_policy, axis=2)
    rhs_actions = np.argmax(rhs_policy, axis=2)

    agreement = 0.0
    for step in range(mdp.horizon):
        agreement += float(np.dot(occupancy[step], (lhs_actions[step] == rhs_actions[step]).astype(float)))
    return agreement / float(mdp.horizon)
