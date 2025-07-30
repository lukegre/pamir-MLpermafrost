import gpytorch
import numpy as np
import torch

from ..datasets import StandardScaler_toTensor


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Better kernel for mixed continuous/discrete data
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                # ARD for different length scales per dimension
                ard_num_dims=train_x.shape[1]
            ),
        )

        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def make_model(
    X_train_scaled_tensor: torch.Tensor, y_train_scaled_tensor: torch.Tensor
) -> GPModel:
    # Create improved model with scaled data
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = GPModel(X_train_scaled_tensor, y_train_scaled_tensor, likelihood)

    return model


def train(
    model,
    X_train_scaled_tensor: torch.Tensor,
    y_train_scaled_tensor: torch.Tensor,
    n_iters=500,
    learning_rate=0.02,
    patience=10,
    tolerance=1e-6,
):
    """
    Trains the GP model using the provided training data and parameters.

    Args:
        model: The GPModel instance to train.
        X_train_scaled_tensor (torch.Tensor): Scaled training features.
        y_train_scaled_tensor (torch.Tensor): Scaled training targets.
        n_iters (int, optional): Maximum number of training iterations. Default is 500.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.02.
        patience (int, optional): Number of iterations with minimal improvement before stopping. Default is 10.
        tolerance (float, optional): Minimum change in loss to be considered as improvement. Default is 1e-6.

    Returns:
        tuple:
            model (GPModel): The trained GP model.
            losses (list of float): List of negative log marginal likelihood losses per iteration.
    """
    # 3. Improved training with convergence monitoring
    likelihood = model.likelihood

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
        diff = prev_loss - loss.item()
        # if difference is negative, it means the loss has increased
        if diff < tolerance:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Converged at iteration {i + 1}")
                break
        else:
            no_improve_count = 0

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
