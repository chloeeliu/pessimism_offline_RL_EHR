from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from peorl.linear.data import count_state_actions, step_transitions
from peorl.linear.regression import confidence_width, fit_ridge_regression, predict_linear_values
from peorl.linear.types import LinearMDP, LinearOfflineDataset, LinearPlannerResult, StepRegressionData


@dataclass(frozen=True)
class LinearFitConfig:
    ridge: float = 1.0
    beta: float = 0.0


def fit_single_step_linear_bandit(
    mdp: LinearMDP,
    regression_data: StepRegressionData,
    config: LinearFitConfig,
) -> LinearPlannerResult:
    """First working linear planner: one-step greedy/pessimistic bandit fit."""

    if mdp.horizon != 1:
        raise ValueError("fit_single_step_linear_bandit currently only supports horizon-1 tasks.")
    if regression_data.features.shape[1] != mdp.feature_dim:
        raise ValueError("Feature dimension mismatch between regression data and LinearMDP.")

    theta, covariance = fit_ridge_regression(
        features=regression_data.features,
        targets=regression_data.targets,
        ridge=config.ridge,
    )
    flat_features = mdp.features.reshape(-1, mdp.feature_dim)
    flat_q_hat = predict_linear_values(flat_features, theta)
    flat_bonus = confidence_width(flat_features, covariance, beta=config.beta)

    q_hat = flat_q_hat.reshape(mdp.num_states, mdp.num_actions)
    bonuses = flat_bonus.reshape(mdp.num_states, mdp.num_actions)
    q_values = np.clip(q_hat - bonuses, 0.0, float(mdp.horizon))

    policy = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    greedy_actions = np.argmax(q_values, axis=1)
    policy[0, np.arange(mdp.num_states), greedy_actions] = 1.0

    return LinearPlannerResult(
        policy=policy,
        q_values=q_values[None, ...],
        q_hat=q_hat[None, ...],
        bonuses=bonuses[None, ...],
        step_thetas=[theta],
    )


def prepare_bandit_regression_targets(
    dataset: LinearOfflineDataset,
) -> np.ndarray:
    return np.asarray([transition.reward for transition in dataset.transitions], dtype=float)


def _fit_step_model(
    mdp: LinearMDP,
    dataset: LinearOfflineDataset,
    step: int,
    next_values: np.ndarray,
    config: LinearFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rows = step_transitions(dataset, step)
    if not rows:
        covariance = config.ridge * np.eye(mdp.feature_dim, dtype=float)
        theta = np.zeros(mdp.feature_dim, dtype=float)
        return theta, covariance

    features = np.stack([transition.feature for transition in rows], axis=0)
    targets = np.asarray(
        [transition.reward + next_values[transition.next_state] for transition in rows],
        dtype=float,
    )
    return fit_ridge_regression(features=features, targets=targets, ridge=config.ridge)


def fit_linear_pevi(
    mdp: LinearMDP,
    dataset: LinearOfflineDataset,
    config: LinearFitConfig,
) -> LinearPlannerResult:
    q_hat = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    q_values = np.zeros_like(q_hat)
    bonuses = np.zeros_like(q_hat)
    v_values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=float)
    policy = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    step_thetas: list[np.ndarray] = []

    flat_features = mdp.features.reshape(-1, mdp.feature_dim)
    for step in range(mdp.horizon - 1, -1, -1):
        theta, covariance = _fit_step_model(mdp, dataset, step, v_values[step + 1], config)
        step_thetas.insert(0, theta)
        flat_q_hat = predict_linear_values(flat_features, theta)
        scale = mdp.horizon - step
        flat_bonus = confidence_width(flat_features, covariance, beta=config.beta * scale)
        q_hat[step] = flat_q_hat.reshape(mdp.num_states, mdp.num_actions)
        bonuses[step] = flat_bonus.reshape(mdp.num_states, mdp.num_actions)
        q_values[step] = np.clip(q_hat[step] - bonuses[step], 0.0, float(scale))
        greedy_actions = np.argmax(q_values[step], axis=1)
        policy[step, np.arange(mdp.num_states), greedy_actions] = 1.0
        v_values[step] = q_values[step, np.arange(mdp.num_states), greedy_actions]

    return LinearPlannerResult(
        policy=policy,
        q_values=q_values,
        q_hat=q_hat,
        bonuses=bonuses,
        step_thetas=step_thetas,
    )


def fit_linear_support_masked_vi(
    mdp: LinearMDP,
    dataset: LinearOfflineDataset,
    ridge: float,
    min_count: int,
) -> LinearPlannerResult:
    counts = count_state_actions(mdp, dataset)
    q_hat = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    q_values = np.zeros_like(q_hat)
    bonuses = np.zeros_like(q_hat)
    v_values = np.zeros((mdp.horizon + 1, mdp.num_states), dtype=float)
    policy = np.zeros((mdp.horizon, mdp.num_states, mdp.num_actions), dtype=float)
    step_thetas: list[np.ndarray] = []

    flat_features = mdp.features.reshape(-1, mdp.feature_dim)
    for step in range(mdp.horizon - 1, -1, -1):
        theta, _ = _fit_step_model(mdp, dataset, step, v_values[step + 1], LinearFitConfig(ridge=ridge, beta=0.0))
        step_thetas.insert(0, theta)
        q_hat[step] = predict_linear_values(flat_features, theta).reshape(mdp.num_states, mdp.num_actions)
        supported = counts[step] >= min_count
        fallback_actions = np.argmax(counts[step], axis=1)
        q_values[step] = np.where(supported, q_hat[step], -1e9)
        no_supported = ~np.any(supported, axis=1)
        q_values[step, no_supported, fallback_actions[no_supported]] = q_hat[step, no_supported, fallback_actions[no_supported]]
        greedy_actions = np.argmax(q_values[step], axis=1)
        policy[step, np.arange(mdp.num_states), greedy_actions] = 1.0
        q_values[step] = np.where(q_values[step] < -1e8, 0.0, q_values[step])
        v_values[step] = q_hat[step, np.arange(mdp.num_states), greedy_actions]

    return LinearPlannerResult(
        policy=policy,
        q_values=q_values,
        q_hat=q_hat,
        bonuses=bonuses,
        step_thetas=step_thetas,
    )
