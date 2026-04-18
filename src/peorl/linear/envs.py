from __future__ import annotations

import numpy as np

from peorl.linear.types import LinearMDP, LinearOfflineTask


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=float)
    vec[index] = 1.0
    return vec


def _make_feature_tensor(num_states: int, num_actions: int, feature_dim: int) -> np.ndarray:
    features = np.zeros((num_states, num_actions, feature_dim), dtype=float)
    for state in range(num_states):
        for action in range(num_actions):
            features[state, action, 0] = 1.0
            features[state, action, 1] = state / max(num_states - 1, 1)
            features[state, action, 2] = action / max(num_actions - 1, 1)
            if feature_dim > 3:
                features[state, action, 3] = 1.0 if action == 1 else 0.0
            if feature_dim > 4:
                features[state, action, 4] = 1.0 if state in {1, 2} else 0.0
            if feature_dim > 5:
                features[state, action, 5] = 1.0 if state == 2 and action == 1 else 0.0
    return features


def _make_linear_branching_family_task(
    *,
    name: str,
    description: str,
    feature_dim: int,
    support_threshold: int,
    root_risky_prob: float,
    hidden_optimal_behavior_prob: float,
) -> LinearOfflineTask:
    num_states = 6
    num_actions = 2
    horizon = 3
    features = _make_feature_tensor(num_states, num_actions, feature_dim)

    rewards = np.zeros((horizon, num_states, num_actions), dtype=float)
    transitions = np.zeros((horizon, num_states, num_actions, num_states), dtype=float)

    root, safe_mid, risky_mid, safe_leaf, hidden_leaf, bad_leaf = range(num_states)

    if feature_dim > 6:
        features[risky_mid, 0, 6] = 1.0
        features[risky_mid, 1, 6] = 0.35
    if feature_dim > 7:
        features[hidden_leaf, 0, 7] = 1.0

    rewards[0, root, 0] = 0.10
    transitions[0, root, 0, safe_mid] = 1.0
    transitions[0, root, 1, risky_mid] = 1.0
    transitions[0, safe_mid:, :, bad_leaf] = 1.0

    rewards[1, safe_mid, 0] = 0.20
    rewards[1, safe_mid, 1] = 0.05
    transitions[1, safe_mid, 0, safe_leaf] = 1.0
    transitions[1, safe_mid, 1, bad_leaf] = 1.0

    rewards[1, risky_mid, 0] = 0.10
    rewards[1, risky_mid, 1] = 0.30
    transitions[1, risky_mid, 0, hidden_leaf] = 1.0
    transitions[1, risky_mid, 1, bad_leaf] = 1.0
    transitions[1, safe_leaf:, :, bad_leaf] = 1.0

    rewards[2, safe_leaf, 0] = 0.50
    rewards[2, safe_leaf, 1] = 0.05
    rewards[2, hidden_leaf, 0] = 0.85
    rewards[2, hidden_leaf, 1] = 0.05
    rewards[2, bad_leaf, 0] = 0.40
    rewards[2, bad_leaf, 1] = 0.00
    transitions[2, :, :, bad_leaf] = 1.0

    initial_state_dist = _one_hot(root, num_states)
    behavior_policy = np.zeros((horizon, num_states, num_actions), dtype=float)
    behavior_policy[:, :, 0] = 1.0
    behavior_policy[0, root] = np.array([1.0 - root_risky_prob, root_risky_prob], dtype=float)
    behavior_policy[1, safe_mid] = np.array([0.90, 0.10], dtype=float)
    behavior_policy[1, risky_mid] = np.array(
        [hidden_optimal_behavior_prob, 1.0 - hidden_optimal_behavior_prob],
        dtype=float,
    )
    behavior_policy[2, safe_leaf] = np.array([0.95, 0.05], dtype=float)
    behavior_policy[2, hidden_leaf] = np.array([1.0, 0.0], dtype=float)
    behavior_policy[2, bad_leaf] = np.array([0.90, 0.10], dtype=float)

    mdp = LinearMDP(
        name=name,
        horizon=horizon,
        num_states=num_states,
        num_actions=num_actions,
        feature_dim=feature_dim,
        features=features,
        rewards=rewards,
        transition_probs=transitions,
        initial_state_dist=initial_state_dist,
    )
    return LinearOfflineTask(
        mdp=mdp,
        behavior_policy=behavior_policy,
        support_threshold=support_threshold,
        description=description,
    )


def make_linear_bandit_task(feature_dim: int = 6, support_threshold: int = 3) -> LinearOfflineTask:
    num_states = 1
    num_actions = 8
    horizon = 1

    features = _make_feature_tensor(num_states, num_actions, feature_dim)
    rewards = np.zeros((horizon, num_states, num_actions), dtype=float)
    rewards[0, 0] = np.array([0.78, 0.48, 0.45, 0.42, 0.40, 0.38, 0.36, 0.62], dtype=float)

    transitions = np.zeros((horizon, num_states, num_actions, num_states), dtype=float)
    transitions[:, :, :, 0] = 1.0
    initial_state_dist = _one_hot(0, num_states)

    behavior_policy = np.zeros((horizon, num_states, num_actions), dtype=float)
    behavior_policy[0, 0] = np.array([0.38, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09, 0.05], dtype=float)

    mdp = LinearMDP(
        name="linear_spurious_bandit",
        horizon=horizon,
        num_states=num_states,
        num_actions=num_actions,
        feature_dim=feature_dim,
        features=features,
        rewards=rewards,
        transition_probs=transitions,
        initial_state_dist=initial_state_dist,
    )
    return LinearOfflineTask(
        mdp=mdp,
        behavior_policy=behavior_policy,
        support_threshold=support_threshold,
        description=(
            "One-step linear bandit scaffold with a weakly covered distractor action. "
            "This is the first linear sanity-check task."
        ),
    )


def make_linear_branching_task(feature_dim: int = 6, support_threshold: int = 2) -> LinearOfflineTask:
    return _make_linear_branching_family_task(
        name="linear_branching_mdp",
        description=(
            "Finite-horizon linear branching scaffold with a rarely supported hidden-optimal path. "
            "This is the main Phase 2 planning environment."
        ),
        feature_dim=feature_dim,
        support_threshold=support_threshold,
        root_risky_prob=0.10,
        hidden_optimal_behavior_prob=0.10,
    )


def make_linear_intrinsic_task(
    feature_dim: int = 6,
    root_risky_prob: float = 0.06,
    hidden_optimal_behavior_prob: float = 0.0,
    support_threshold: int = 2,
) -> LinearOfflineTask:
    return _make_linear_branching_family_task(
        name="linear_intrinsic_branching_mdp",
        description=(
            "Three-step linear branching task where the logged data never shows the optimal continuation "
            "of a weakly visited branch. The safe branch is learnable, but the hidden optimal path is "
            "effectively off-support."
        ),
        feature_dim=feature_dim,
        support_threshold=support_threshold,
        root_risky_prob=root_risky_prob,
        hidden_optimal_behavior_prob=hidden_optimal_behavior_prob,
    )


def make_linear_near_intrinsic_task(
    feature_dim: int = 6,
    root_risky_prob: float = 0.10,
    hidden_optimal_behavior_prob: float = 0.10,
    support_threshold: int = 2,
) -> LinearOfflineTask:
    return _make_linear_branching_family_task(
        name="linear_near_intrinsic_branching_mdp",
        description=(
            "Three-step linear branching task with a rarely observed optimal continuation. "
            "The hidden branch is no longer fully off-support, so the learner may recover it "
            "once enough rare evidence accumulates."
        ),
        feature_dim=feature_dim,
        support_threshold=support_threshold,
        root_risky_prob=root_risky_prob,
        hidden_optimal_behavior_prob=hidden_optimal_behavior_prob,
    )


def make_linear_task(name: str, **kwargs: float | int) -> LinearOfflineTask:
    normalized = name.strip().lower()
    if normalized == "linear_bandit":
        return make_linear_bandit_task(**kwargs)
    if normalized == "linear_branching":
        return make_linear_branching_task(**kwargs)
    if normalized == "linear_intrinsic":
        return make_linear_intrinsic_task(**kwargs)
    if normalized == "linear_near_intrinsic":
        return make_linear_near_intrinsic_task(**kwargs)
    raise ValueError(
        f"Unknown linear task {name!r}. Expected 'linear_bandit', 'linear_branching', "
        "'linear_intrinsic', or 'linear_near_intrinsic'."
    )
