import gpytorch
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from ..datasets import StandardScaler_toTensor


def predict(
    model,
    inference_dataset: Dataset,
    scaler_y: StandardScaler_toTensor,
    num_workers=5,
):
    device = next(model.parameters()).device

    # Make predictions on scaled inference data
    likelihood = model.likelihood

    model.eval()
    likelihood.eval()

    logger.info("Preparing datasets")
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=1,
        num_workers=3,
        shuffle=False,
        pin_memory=True,  # Set to True for potentially faster data transfer to GPU
    )

    predictions_mean_scaled = []
    predictions_variance_scaled = []

    logger.info("starting inference")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i, (batch_x,) in enumerate(inference_loader):
            batch_x = batch_x.to(device)
            observed_pred_inference = likelihood(model(batch_x))
            predictions_mean_scaled.append(observed_pred_inference.mean.cpu())
            predictions_variance_scaled.append(observed_pred_inference.variance.cpu())

            logging_message = f"Processed batch {i + 1}/{len(inference_loader)}"
            if not ((i + 1) % 1):
                logger.info(logging_message)
            else:
                logger.debug(logging_message)

    # Concatenate predictions from all batches
    # f_mean_inference_scaled = torch.cat(predictions_mean_scaled)
    # f_variance_inference_scaled = torch.cat(predictions_variance_scaled)

    # # Transform predictions back to original scale
    # f_mean_inference_original = scaler_y.inverse_transform(
    #     f_mean_inference_scaled
    # )

    return predictions_mean_scaled, predictions_variance_scaled


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
