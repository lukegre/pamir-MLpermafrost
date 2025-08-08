import gpytorch
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ...preprocessing.scalers import StandardScaler_toTensor


def eval(
    model,
    X_test_scaled_tensor: torch.Tensor,
    y_test_scaled_tensor: torch.Tensor,
    scaler_y: StandardScaler_toTensor,
):
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

    return {"mse": mse, "rmse": np.sqrt(mse), "mae": mae, "r2": r2}
