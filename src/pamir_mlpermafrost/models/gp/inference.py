import gpytorch
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from ...preprocessing.scalers import StandardScaler_toTensor


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
