import gpytorch
import numpy as np
import torch
from loguru import logger

from ..data import StandardScaler_toTensor


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

        if (i + 1) % 10 == 0:
            print(f"Iter {i + 1}/{n_iters} - Loss: {loss.item():.6f}")

        # Check for convergence
        diff = prev_loss - loss.item()  # diff < 0 if loop has higher loss
        if diff < tolerance:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Converged at iteration {i + 1}")
                break
        else:
            no_improve_count = max(no_improve_count - 1, 0)
        logger.debug(f"{no_improve_count=}")
        prev_loss = loss.item()

    return model, losses


def eval(
    model,
    X_test_scaled_tensor: torch.Tensor,
    y_test_scaled_tensor: torch.Tensor,
    scaler_y: StandardScaler_toTensor,
):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    likelihood = model.likelihood

    # 4. Proper evaluation on test set
    model.eval()
    likelihood.eval()

    y_test = scaler_y.inverse_transform(y_test_scaled_tensor.reshape(-1, 1))

    # Make predictions on scaled test data
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_test_scaled_tensor))

        # Get predictions in scaled space
        y_pred_scaled_tensor = observed_pred.mean
        y_pred_std_scaled_tensor = observed_pred.stddev

        # Transform back to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled_tensor.reshape(-1, 1))
        # Note: std needs special handling for scaling transformation
        y_pred_std = y_pred_std_scaled_tensor * scaler_y.scale_[0]

    # Calculate metrics in original scale
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test Set Performance (Original Scale):")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Predicted Std Dev: {y_pred_std.mean():.4f}")

    return {"mse": mse, "rmse": np.sqrt(mse), "mae": mae, "r2": r2}, y_pred, y_pred_std


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
