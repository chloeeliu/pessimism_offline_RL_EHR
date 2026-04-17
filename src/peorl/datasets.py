from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from peorl.envs.tabular import TabularMDP


@dataclass(frozen=True)
class Transition:
    step: int
    state: int
    action: int
    reward: float
    next_state: int


@dataclass(frozen=True)
class OfflineDataset:
    transitions: list[Transition]
    num_episodes: int


@dataclass(frozen=True)
class EmpiricalModel:
    counts: np.ndarray
    reward_sums: np.ndarray
    transition_counts: np.ndarray
    reward_hat: np.ndarray
    transition_hat: np.ndarray


def collect_dataset(
    mdp: TabularMDP,
    behavior_policy: np.ndarray,
    num_episodes: int,
    seed: int,
) -> OfflineDataset:
    rng = np.random.default_rng(seed)
    transitions: list[Transition] = []

    for _ in range(num_episodes):
        state = mdp.sample_initial_state(rng)
        for step in range(mdp.horizon):
            action = int(rng.choice(mdp.num_actions, p=behavior_policy[step, state]))
            reward, next_state = mdp.sample_step(step, state, action, rng)
            transitions.append(
                Transition(
                    step=step,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                )
            )
            state = next_state

    return OfflineDataset(transitions=transitions, num_episodes=num_episodes)


def build_empirical_model(mdp: TabularMDP, dataset: OfflineDataset) -> EmpiricalModel:
    counts = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    reward_sums = np.zeros_like(counts)
    transition_counts = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions, mdp.num_states), dtype=float)

    for transition in dataset.transitions:
        counts[transition.step, transition.state, transition.action] += 1.0
        reward_sums[transition.step, transition.state, transition.action] += transition.reward
        transition_counts[transition.step, transition.state, transition.action, transition.next_state] += 1.0

    denom = np.maximum(counts, 1.0)
    reward_hat = reward_sums / denom
    transition_hat = transition_counts / denom[..., None]

    return EmpiricalModel(
        counts=counts,
        reward_sums=reward_sums,
        transition_counts=transition_counts,
        reward_hat=reward_hat,
        transition_hat=transition_hat,
    )
