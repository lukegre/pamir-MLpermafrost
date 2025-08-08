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
    num_workers=4,
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
        num_workers=num_workers,
        shuffle=False,
        prefetch_factor=2,
    )

    predictions_mean_scaled = []
    predictions_variance_scaled = []

    logger.info("starting inference")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i, (batch_x,) in enumerate(inference_loader):
            batch_x = batch_x.to(device)
            observed_pred_inference = likelihood(model(batch_x))

            pred_mean = (
                observed_pred_inference.mean.cpu() * scaler_y.scale_[0]
                + scaler_y.mean_[0]
            )
            pred_var = observed_pred_inference.variance.cpu() * scaler_y.scale_[0]

            predictions_mean_scaled.append(pred_mean)
            predictions_variance_scaled.append(pred_var)

            logging_message = f"Processed batch {i + 1}/{len(inference_loader)}"
            if not ((i + 1) % 1):
                logger.info(logging_message)
            else:
                logger.debug(logging_message)

    return predictions_mean_scaled, predictions_variance_scaled
