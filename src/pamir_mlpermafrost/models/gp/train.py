import copy
import math

import gpytorch
import mlflow
import numpy as np
import torch
from icecream import ic as print
from loguru import logger


def train_mll(
    model,
    X_train_scaled_tensor: torch.Tensor,
    y_train_scaled_tensor: torch.Tensor,
    optimizer,
    n_iters=500,
    patience=10,
    tolerance=1e-3,
):
    # 3. Improved training with convergence monitoring
    likelihood = model.likelihood

    assert not contains_nans(X_train_scaled_tensor), (
        "X_train_scaled_tensor contains NaNs"
    )
    assert not contains_nans(y_train_scaled_tensor), (
        "y_train_scaled_tensor contains NaNs"
    )

    model.train()
    likelihood.train()

    stopper = EarlyStopper(
        patience=patience, rel_min_delta=tolerance, warmup=50, ema_beta=0.9
    )

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(n_iters):
        optimizer.zero_grad()
        output = model(X_train_scaled_tensor)
        loss = -mll(output, y_train_scaled_tensor)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        log_message = f"Iter {i + 1}/{n_iters} - Loss: {loss.item():.6f}"
        if (i + 1) % 10 == 0:
            logger.info(log_message)
        else:
            logger.debug(log_message)
        mlflow.log_metrics({"loss": loss.item()}, step=i)

        # Check for convergence
        if stopper(loss.item(), i):
            logger.info(f"Early stopping at iteration {i + 1}")
            break

    return model, losses


def contains_nans(arr: torch.Tensor):
    """
    Check if a tensor contains NaN values.

    Parameters
    ----------
    arr : torch.Tensor
        The tensor to check for NaN values.

    Returns
    -------
    bool
        True if NaN values are present, False otherwise.
    """
    return torch.isnan(arr).any().item()  # Convert to Python boolean


class EarlyStopper:
    def __init__(self, patience=15, rel_min_delta=1e-3, warmup=50, ema_beta=None):
        self.patience = patience
        self.rel_min_delta = rel_min_delta
        self.warmup = warmup
        self.ema_beta = ema_beta
        self.best = math.inf
        self.wait = 0
        self.ema = None
        self.best_state = None
        self.best_iter = -1

    def __call__(self, value, step, model=None, likelihood=None):
        """
        Call the stopper to check if training should be stopped.

        :param value: The current value to evaluate.
        :param step: The current iteration step.
        :param model: The model to save if early stopping occurs.
        :param likelihood: The likelihood to save if early stopping occurs.
        :return: True if training should stop, False otherwise.
        """
        return self.update(value, step, model, likelihood)

    def _smooth(self, x):
        if self.ema_beta is None:
            return x
        self.ema = (
            x
            if self.ema is None
            else self.ema_beta * self.ema + (1 - self.ema_beta) * x
        )
        return self.ema

    def update(self, value, step, model=None, likelihood=None):
        v = self._smooth(value)
        if self.best is math.inf:
            self.best = v
            if model and likelihood:
                self.best_state = (
                    copy.deepcopy(model.state_dict()),
                    copy.deepcopy(likelihood.state_dict()),
                )
                self.best_iter = step
            return False

        thresh = self.rel_min_delta * abs(self.best)
        if (self.best - v) > thresh:  # improved enough
            self.best = v
            self.wait = 0
            if model and likelihood:
                self.best_state = (
                    copy.deepcopy(model.state_dict()),
                    copy.deepcopy(likelihood.state_dict()),
                )
                self.best_iter = step
        else:
            self.wait += 1

        return (step >= self.warmup) and (self.wait >= self.patience)
