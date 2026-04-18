from __future__ import annotations

import numpy as np

from peorl.linear.types import LinearMDP, LinearOfflineDataset, LinearTransition, StepRegressionData


def collect_linear_dataset(
    mdp: LinearMDP,
    behavior_policy: np.ndarray,
    num_episodes: int,
    seed: int,
) -> LinearOfflineDataset:
    rng = np.random.default_rng(seed)
    transitions: list[LinearTransition] = []

    for _ in range(num_episodes):
        state = int(rng.choice(mdp.num_states, p=mdp.initial_state_dist))
        for step in range(mdp.horizon):
            action = int(rng.choice(mdp.num_actions, p=behavior_policy[step, state]))
            reward = float(rng.binomial(1, mdp.rewards[step, state, action]))
            next_state = int(rng.choice(mdp.num_states, p=mdp.transition_probs[step, state, action]))
            transitions.append(
                LinearTransition(
                    step=step,
                    state=state,
                    action=action,
                    feature=mdp.features[state, action].copy(),
                    reward=reward,
                    next_state=next_state,
                )
            )
            state = next_state

    return LinearOfflineDataset(transitions=transitions, num_episodes=num_episodes)


def build_step_regression_data(
    dataset: LinearOfflineDataset,
    step: int,
    targets: np.ndarray,
) -> StepRegressionData:
    rows = [transition for transition in dataset.transitions if transition.step == step]
    if not rows:
        return StepRegressionData(features=np.zeros((0, 0)), targets=np.zeros(0))

    features = np.stack([transition.feature for transition in rows], axis=0)
    target_values = np.asarray([targets[index] for index in range(len(rows))], dtype=float)
    return StepRegressionData(features=features, targets=target_values)


def step_transitions(
    dataset: LinearOfflineDataset,
    step: int,
) -> list[LinearTransition]:
    return [transition for transition in dataset.transitions if transition.step == step]


def count_state_actions(
    mdp: LinearMDP,
    dataset: LinearOfflineDataset,
) -> np.ndarray:
    counts = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    for transition in dataset.transitions:
        counts[transition.step, transition.state, transition.action] += 1.0
    return counts
