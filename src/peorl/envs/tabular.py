from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TabularMDP:
    """Finite-horizon tabular MDP with bounded rewards in [0, 1]."""

    name: str
    horizon: int
    num_states: int
    num_actions: int
    rewards: np.ndarray
    transition_probs: np.ndarray
    initial_state_dist: np.ndarray

    def sample_initial_state(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.num_states, p=self.initial_state_dist))

    def sample_step(
        self,
        step: int,
        state: int,
        action: int,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        reward_mean = float(self.rewards[step, state, action])
        reward = float(rng.binomial(1, reward_mean))
        next_state = int(rng.choice(self.num_states, p=self.transition_probs[step, state, action]))
        return reward, next_state


@dataclass(frozen=True)
class OfflineTask:
    """A tabular environment paired with a dataset collection policy."""

    mdp: TabularMDP
    behavior_policy: np.ndarray
    support_threshold: int
    description: str


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=float)
    vec[index] = 1.0
    return vec


def make_bandit_task(
    num_actions: int = 8,
    best_reward: float = 0.78,
    distractor_reward: float = 0.62,
    distractor_behavior_prob: float = 0.05,
    support_threshold: int = 3,
) -> OfflineTask:
    horizon = 1
    num_states = 1

    rewards = np.zeros((horizon, num_states, num_actions), dtype=float)
    rewards[0, 0, 0] = best_reward
    rewards[0, 0, 1:] = np.array([0.48, 0.45, 0.42, 0.40, 0.38, 0.36, distractor_reward][: num_actions - 1])

    transitions = np.zeros((horizon, num_states, num_actions, num_states), dtype=float)
    transitions[:, :, :, 0] = 1.0

    initial_state_dist = _one_hot(0, num_states)

    behavior_policy = np.zeros((horizon, num_states, num_actions), dtype=float)
    behavior_policy[0, 0, 0] = 0.38
    if num_actions > 2:
        remaining_prob = 1.0 - behavior_policy[0, 0, 0] - distractor_behavior_prob
        behavior_policy[0, 0, 1:-1] = remaining_prob / (num_actions - 2)
    behavior_policy[0, 0, -1] = distractor_behavior_prob

    mdp = TabularMDP(
        name="spurious_bandit",
        horizon=horizon,
        num_states=num_states,
        num_actions=num_actions,
        rewards=rewards,
        transition_probs=transitions,
        initial_state_dist=initial_state_dist,
    )
    return OfflineTask(
        mdp=mdp,
        behavior_policy=behavior_policy,
        support_threshold=support_threshold,
        description=(
            "One-step bandit with a weakly covered distractor arm. "
            "Greedy plug-in estimation can over-pick a lucky rare arm."
        ),
    )


def make_branching_task(
    root_risky_prob: float = 0.08,
    risky_success_reward: float = 0.45,
    support_threshold: int = 2,
) -> OfflineTask:
    horizon = 3
    num_states = 5
    num_actions = 2

    rewards = np.zeros((horizon, num_states, num_actions), dtype=float)
    transitions = np.zeros((horizon, num_states, num_actions, num_states), dtype=float)

    root, safe_mid, risky_mid, good_leaf, bad_leaf = range(num_states)

    rewards[0, root, 0] = 0.10
    rewards[0, root, 1] = 0.00
    transitions[0, root, 0, safe_mid] = 1.0
    transitions[0, root, 1, risky_mid] = 1.0

    rewards[0, safe_mid:, :] = 0.0
    rewards[0, safe_mid, :] = 0.0
    rewards[0, risky_mid, :] = 0.0
    rewards[0, good_leaf, :] = 0.0
    rewards[0, bad_leaf, :] = 0.0
    transitions[0, safe_mid:, :, bad_leaf] = 1.0

    rewards[1, safe_mid, 0] = 0.25
    rewards[1, safe_mid, 1] = 0.05
    transitions[1, safe_mid, 0, good_leaf] = 1.0
    transitions[1, safe_mid, 1, bad_leaf] = 1.0

    rewards[1, risky_mid, 0] = risky_success_reward
    rewards[1, risky_mid, 1] = 0.05
    transitions[1, risky_mid, 0, bad_leaf] = 1.0
    transitions[1, risky_mid, 1, bad_leaf] = 1.0

    rewards[1, good_leaf, :] = 0.0
    rewards[1, bad_leaf, :] = 0.0
    transitions[1, good_leaf, :, bad_leaf] = 1.0
    transitions[1, bad_leaf, :, bad_leaf] = 1.0

    rewards[2, good_leaf, 0] = 0.55
    rewards[2, good_leaf, 1] = 0.10
    rewards[2, bad_leaf, 0] = 0.10
    rewards[2, bad_leaf, 1] = 0.00
    transitions[2, :, :, bad_leaf] = 1.0

    initial_state_dist = _one_hot(root, num_states)

    behavior_policy = np.zeros((horizon, num_states, num_actions), dtype=float)
    behavior_policy[:, :, 0] = 1.0
    behavior_policy[0, root] = np.array([1.0 - root_risky_prob, root_risky_prob], dtype=float)
    behavior_policy[1, safe_mid] = np.array([0.90, 0.10], dtype=float)
    behavior_policy[1, risky_mid] = np.array([0.80, 0.20], dtype=float)
    behavior_policy[2, good_leaf] = np.array([0.95, 0.05], dtype=float)
    behavior_policy[2, bad_leaf] = np.array([0.80, 0.20], dtype=float)

    mdp = TabularMDP(
        name="branching_mdp",
        horizon=horizon,
        num_states=num_states,
        num_actions=num_actions,
        rewards=rewards,
        transition_probs=transitions,
        initial_state_dist=initial_state_dist,
    )
    return OfflineTask(
        mdp=mdp,
        behavior_policy=behavior_policy,
        support_threshold=support_threshold,
        description=(
            "Three-step branching MDP with a well-covered safe branch and an "
            "under-covered risky branch that can be overestimated by the greedy plug-in baseline."
        ),
    )


def make_intrinsic_uncertainty_task(
    root_risky_prob: float = 0.06,
    hidden_optimal_behavior_prob: float = 0.0,
    support_threshold: int = 2,
) -> OfflineTask:
    horizon = 3
    num_states = 6
    num_actions = 2

    rewards = np.zeros((horizon, num_states, num_actions), dtype=float)
    transitions = np.zeros((horizon, num_states, num_actions, num_states), dtype=float)

    root, safe_mid, risky_mid, safe_leaf, hidden_leaf, bad_leaf = range(num_states)

    rewards[0, root, 0] = 0.10
    rewards[0, root, 1] = 0.00
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

    mdp = TabularMDP(
        name="intrinsic_branching_mdp",
        horizon=horizon,
        num_states=num_states,
        num_actions=num_actions,
        rewards=rewards,
        transition_probs=transitions,
        initial_state_dist=initial_state_dist,
    )
    return OfflineTask(
        mdp=mdp,
        behavior_policy=behavior_policy,
        support_threshold=support_threshold,
        description=(
            "Three-step branching MDP where the logged data never shows the optimal continuation "
            "of a weakly visited branch. The safe branch is learnable, but the hidden optimal path "
            "is off-support, so pessimism should fail gracefully rather than recover it."
        ),
    )


def make_near_intrinsic_task(
    root_risky_prob: float = 0.10,
    hidden_optimal_behavior_prob: float = 0.10,
    support_threshold: int = 2,
) -> OfflineTask:
    task = make_intrinsic_uncertainty_task(
        root_risky_prob=root_risky_prob,
        hidden_optimal_behavior_prob=hidden_optimal_behavior_prob,
        support_threshold=support_threshold,
    )
    mdp = TabularMDP(
        name="near_intrinsic_branching_mdp",
        horizon=task.mdp.horizon,
        num_states=task.mdp.num_states,
        num_actions=task.mdp.num_actions,
        rewards=task.mdp.rewards,
        transition_probs=task.mdp.transition_probs,
        initial_state_dist=task.mdp.initial_state_dist,
    )
    return OfflineTask(
        mdp=mdp,
        behavior_policy=task.behavior_policy,
        support_threshold=task.support_threshold,
        description=(
            "Three-step branching MDP with a rarely observed optimal continuation. "
            "The hidden branch is no longer fully off-support, so the learner may recover it "
            "once enough rare evidence accumulates."
        ),
    )


def make_task(name: str, **kwargs: float | int) -> OfflineTask:
    normalized = name.strip().lower()
    if normalized == "bandit":
        return make_bandit_task(**kwargs)
    if normalized == "branching":
        return make_branching_task(**kwargs)
    if normalized == "intrinsic":
        return make_intrinsic_uncertainty_task(**kwargs)
    if normalized == "near_intrinsic":
        return make_near_intrinsic_task(**kwargs)
    raise ValueError(f"Unknown task {name!r}. Expected 'bandit', 'branching', 'intrinsic', or 'near_intrinsic'.")
