import gpytorch
import numpy as np
import torch
from loguru import logger

from ...preprocessing.scalers import StandardScaler_toTensor


def train_mll(
    model,
    X_train_scaled_tensor: torch.Tensor,
    y_train_scaled_tensor: torch.Tensor,
    n_iters=500,
    learning_rate=0.02,
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

    # Better optimizer and learning rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )  # Lower learning rate
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    prev_loss = float("inf")
    no_improve_count = 0
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

        # Check for convergence
        diff = prev_loss - loss.item()  # diff < 0 if loop has higher loss
        if diff < tolerance:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info(f"Converged at iteration {i + 1}")
                break
        else:
            no_improve_count = max(no_improve_count - 1, 0)
        logger.trace(f"{no_improve_count=}")
        prev_loss = loss.item()

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
