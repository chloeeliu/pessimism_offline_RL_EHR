from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from peorl.datasets import EmpiricalModel
from peorl.envs.tabular import TabularMDP


@dataclass(frozen=True)
class PlannerResult:
    policy: np.ndarray
    q_values: np.ndarray
    q_hat: np.ndarray
    v_values: np.ndarray
    bonuses: np.ndarray


def _run_value_iteration(
    mdp: TabularMDP,
    model: EmpiricalModel,
    beta: float,
) -> PlannerResult:
    q_hat = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    q_values = np.zeros_like(q_hat)
    bonuses = np.zeros_like(q_hat)
    v_values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=float)
    policy = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)

    for step in range(mdp.horizon - 1, -1, -1):
        q_hat[step] = model.reward_hat[step] + np.einsum(
            "san,n->sa",
            model.transition_hat[step],
            v_values[step + 1],
        )
        remaining_steps = mdp.horizon - step
        bonuses[step] = beta * remaining_steps / np.sqrt(np.maximum(model.counts[step], 1.0))
        q_values[step] = np.clip(q_hat[step] - bonuses[step], 0.0, float(remaining_steps))

        greedy_actions = np.argmax(q_values[step], axis=1)
        policy[step, np.arange(mdp.num_states), greedy_actions] = 1.0
        v_values[step] = q_values[step, np.arange(mdp.num_states), greedy_actions]

    return PlannerResult(
        policy=policy,
        q_values=q_values,
        q_hat=q_hat,
        v_values=v_values,
        bonuses=bonuses,
    )


def fit_greedy_plugin(mdp: TabularMDP, model: EmpiricalModel) -> PlannerResult:
    return _run_value_iteration(mdp=mdp, model=model, beta=0.0)


def fit_pessimistic_vi(mdp: TabularMDP, model: EmpiricalModel, beta: float) -> PlannerResult:
    return _run_value_iteration(mdp=mdp, model=model, beta=beta)


def fit_support_masked_vi(
    mdp: TabularMDP,
    model: EmpiricalModel,
    min_count: int,
) -> PlannerResult:
    q_hat = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    q_values = np.zeros_like(q_hat)
    bonuses = np.zeros_like(q_hat)
    v_values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=float)
    policy = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)

    for step in range(mdp.horizon - 1, -1, -1):
        q_hat[step] = model.reward_hat[step] + np.einsum(
            "san,n->sa",
            model.transition_hat[step],
            v_values[step + 1],
        )
        supported = model.counts[step] >= min_count
        fallback_actions = np.argmax(model.counts[step], axis=1)
        q_values[step] = np.where(supported, q_hat[step], -1e9)
        no_supported = ~np.any(supported, axis=1)
        q_values[step, no_supported, fallback_actions[no_supported]] = q_hat[step, no_supported, fallback_actions[no_supported]]

        greedy_actions = np.argmax(q_values[step], axis=1)
        policy[step, np.arange(mdp.num_states), greedy_actions] = 1.0
        q_values[step] = np.where(q_values[step] < -1e8, 0.0, q_values[step])
        v_values[step] = q_hat[step, np.arange(mdp.num_states), greedy_actions]

    return PlannerResult(
        policy=policy,
        q_values=q_values,
        q_hat=q_hat,
        v_values=v_values,
        bonuses=bonuses,
    )
