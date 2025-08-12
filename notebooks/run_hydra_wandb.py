import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import xarray as xr
from rich import print

import hydra
import pamir_mlpermafrost as pamir

PLOT_VARS = {
    "yhat_avg": dict(robust=True, cmap="RdBu_r", center=0),
    "yhat_std": dict(robust=True),
    "temperature": dict(robust=True, cmap="RdBu_r", center=273.15),
    "altitude": dict(robust=True, cmap="terrain"),
    "aspect": dict(cmap="twilight_r"),
    "surface_index": dict(robust=True, cmap="turbo", vmin=0, vmax=5),
}


@hydra.main(
    version_base=None,
    config_path="../src/pamir_mlpermafrost/conf",
    config_name="laptop-jupyter",
)
def main(cfg):
    cfg = pamir.utils.process_hydra_config(cfg)

    run = wandb.init(
        project=cfg.wandb.project, config=cfg.toDict(), name=cfg.wandb.run_name
    )
    run_dir = pathlib.Path(cfg.run_dir)
    run.log_artifact(pathlib.Path(cfg.run_dir) / ".hydra", "hydra")

    train_X, train_y, test_X, test_y, scaler_X, scaler_y = load_training_data(cfg)
    scaler_X.save_params(run_dir / "scaler_X.json")
    scaler_y.save_params(run_dir / "scaler_y.json")

    model = cfg.model(
        train_X,
        train_y,
        likelihood=cfg.likelihood(),
        covar_module=cfg.covar_module(),
    ).to(cfg.device)

    model, losses = model.fit(
        train_X,
        train_y,
        cfg.optimizer(model.parameters()),
        n_iters=500,
        patience=40,
        tolerance=1e-3,
    )
    torch.save(model.state_dict(), run_dir / "model_state.pt")

    scores = evaluate_model(model, train_X, train_y, test_X, test_y, scaler_y)
    scores.to_json(run_dir / "scores.json", indent=2)

    output = inference(
        model,
        scaler_X,
        scaler_y,
        cfg,
        isel_subset={"y": slice(2800, 3300), "x": slice(1400, 2500)},
    )

    fig, axs = plot_results(output)
    fig.savefig(run_dir / "results.png", bbox_inches="tight", dpi=120, transparent=True)

    run.log_artifact(run_dir / "results.png", "results")
    run.log_artifact(run_dir / "scores.json", "scores")


def load_training_data(cfg):
    """
    Load the training data and preprocess it according to the configuration.
    """
    # Load the training data
    data = cfg.data.training.load().loc[["S180_exp4", "N180_exp4"]]

    # Preprocess the data
    data_X, data_y = cfg.preprocessing.training(data)

    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = cfg.preprocessing.train_test_split(
        data_X, data_y
    )

    # Scale the features and target variable
    scaler_X = cfg.scalers.features.fit(train_X)
    scaler_y = cfg.scalers.target.fit(train_y.to_frame())

    train_X_scaled = scaler_X.transform(train_X)
    train_y_scaled = scaler_y.transform(train_y.to_frame()).squeeze()

    test_X_scaled = scaler_X.transform(test_X)
    test_y_scaled = scaler_y.transform(test_y.to_frame()).squeeze()

    return (
        train_X_scaled,
        train_y_scaled,
        test_X_scaled,
        test_y_scaled,
        scaler_X,
        scaler_y,
    )


def evaluate_model(model, train_X, train_y, test_X, test_y, scaler_y):
    scores_train = model.score(train_X, train_y, scaler_y)
    scores_test = model.score(test_X, test_y, scaler_y)

    for key in scores_train:
        wandb.log({f"train_{key}": scores_train[key], f"test_{key}": scores_test[key]})

    return (
        pd.DataFrame({"train": scores_train, "test": scores_test})
        .rename(index=lambda x: x.upper())
        .round(2)
    )


def inference(
    model,
    scaler_X,
    scaler_y,
    cfg,
    isel_subset: dict = {},
    chunksizes: dict = {"x": 250, "y": 250},
):
    from functools import partial

    ds_spatial = cfg.data.inference.load().isel(**isel_subset).chunk(chunksizes)

    chunk_processor = partial(pamir.data.processors.process_X, features=cfg.features)
    dataset = pamir.preprocessing.datasets.DatasetXarraySpatial(
        ds_spatial.persist(),
        chunk_processor=chunk_processor,
        scaler=scaler_X,
    )

    yhat_avg, yhat_std = pamir.models.gp.inference.predict(model, dataset, scaler_y)

    name = cfg.target
    yhat_avg = (
        dataset.reconstruct_output(yhat_avg)
        .rename(f"yhat_avg")
        .assign_attrs(long_name=f"{name} average")
    )
    yhat_std = (
        dataset.reconstruct_output(yhat_std)
        .rename(f"yhat_std")
        .assign_attrs(long_name=f"{name} standard deviation")
    )

    out = xr.merge([yhat_avg, yhat_std, ds_spatial])

    return out


def plot_results(ds):
    n_vars = len(PLOT_VARS)
    n_col = 3
    n_row = (n_vars + n_col - 1) // n_col  # Calculate number of rows needed
    figsize = (n_col * 4 + 2, n_row * 2 + 0.5)

    fig, axs = plt.subplots(
        n_row, n_col, figsize=figsize, sharex=True, sharey=True, dpi=200
    )

    for key, ax in zip(PLOT_VARS.keys(), axs.flat):
        da = ds[key]
        props = PLOT_VARS[key]
        da.plot.imshow(ax=ax, **props)

    fig.tight_layout()
    return fig, axs


if __name__ == "__main__":
    main()
