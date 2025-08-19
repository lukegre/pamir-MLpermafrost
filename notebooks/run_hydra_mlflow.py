import pathlib

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import random
import torch
import xarray as xr
import warnings
import pamir_mlpermafrost as pamir
from rich import print


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="linear_operator.utils.interpolation|numcodecs"
)

PLOT_VARS = {
    "yhat_avg": dict(cmap="RdBu_r", vmin=-10, vmax=10),
    "yhat_std": dict(vmin=0, vmax=3.5),
    # "temperature": dict(robust=True, cmap="RdBu_r", center=273.15),
    # "altitude": dict(robust=True, cmap="terrain"),
    # "aspect": dict(cmap="twilight_r"),
    # "surface_index": dict(robust=True, cmap="turbo", vmin=0, vmax=5),
}

@hydra.main(
    version_base=None,
    config_path="../src/pamir_mlpermafrost/conf",
    config_name="gp_split_rbf")
def main(cfg):
    # --- MLflow setup ---
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)  # e.g. "file:./mlruns" or http uri
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    cfg = pamir.utils.process_hydra_config(cfg)
    
    g = set_seed(cfg.seed)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        output_dir = pathlib.Path(run.info.artifact_uri.replace("file://", ""))

        mlflow.log_params(cfg.toDict())
        mlflow.log_artifacts(pathlib.Path(cfg.run_dir) / ".hydra", "hydra")

        train_X, train_y, test_X, test_y, scaler_X, scaler_y = load_training_data(cfg)
        mlflow.log_dict(scaler_X.to_dict(), "scaler_X.json")
        mlflow.log_dict(scaler_y.to_dict(), "scaler_y.json")

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
            n_iters=1000,
            patience=10,
            tolerance=1e-3,
        )
        scores = evaluate_model(model, train_X, train_y, test_X, test_y, scaler_y)

        torch.save(model.state_dict(), output_dir / "model_state.pt")
        mlflow.pytorch.log_model(model)

        # rq_lengthscales = get_rbf_lengthscales(model, cfg.features, kernel_trace='covar_module.kernels.0.kernels.1.base_kernel')
        # mlflow.log_dict(rq_lengthscales, "rq_lengthscales_trained.json")

        output = inference(
            model,
            scaler_X,
            scaler_y,
            cfg,
            # isel_subset={"y": slice(2800, 3300), "x": slice(1400, 2500)},
        )

        plot_results(output)
        log_netcdf(output, cfg)


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
        train_X_scaled.to(cfg.device),
        train_y_scaled.to(cfg.device),
        test_X_scaled.to(cfg.device),
        test_y_scaled.to(cfg.device),
        scaler_X,
        scaler_y,
    )


def evaluate_model(model, train_X, train_y, test_X, test_y, scaler_y):
    scores_train = model.score(train_X, train_y, scaler_y)
    scores_test = model.score(test_X, test_y, scaler_y)

    for key in scores_train:
        mlflow.log_metric(f"train_{key}", scores_train[key])
        mlflow.log_metric(f"test_{key}", scores_test[key])

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
    chunksizes: dict = {"x": 300, "y": 300},
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
    
    mlflow.tracking.multimedia.COMPRESSED_IMAGE_SIZE = 512

    mask = ds.surface_index > 0

    pixel_aspect = 1 / np.cos(np.deg2rad(ds.y.mean().item()))
    fig_aspect = pixel_aspect * 1.1
    fig_w = 10
    fig_h = fig_w / fig_aspect

    for i, key in enumerate(PLOT_VARS.keys()):
        da = ds[key].where(mask)
        props = PLOT_VARS[key]
        
        fig, axs = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        da.plot.imshow(ax=axs, **props)
        fig.tight_layout()
        image = fig_to_pilimage(fig)
        mlflow.log_image(image, key='images', step=i)


def fig_to_pilimage(fig):
    """
    Convert a matplotlib figure to a PIL image.
    """
    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img


def set_seed(seed: int = 42) -> torch.Generator:
    """
    Set seeds for reproducibility across Python, NumPy, PyTorch (CPU + GPU).

    Returns:
        torch.Generator: a seeded torch generator for use in DataLoader or elsewhere.
    """
    import os 

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def log_netcdf(ds:xr.Dataset, cfg):
    from tempfile import TemporaryDirectory
    from pathlib import Path
    import mlflow

    ds = ds[['yhat_avg', 'yhat_std']]
    ds = ds.rename(
        yhat_avg=f"yhat_avg_{cfg.target}",
        yhat_std=f"yhat_std_{cfg.target}")
    ds = ds.astype('float32')

    for key in ds:
        da = ds[[key]]
        enc = {v: {"zlib": True, "complevel": 4} for v in da.data_vars}  # optional compression
        with TemporaryDirectory() as td:
            path = Path(td) / f"output_{key}.nc"
            da.to_netcdf(path, engine="h5netcdf", encoding=enc)  
            mlflow.log_artifact(str(path), artifact_path="results")


def get_rbf_lengthscales(model, features:list[str], kernel_trace='covar_module.kernels.0.kernels.1.base_kernel')->dict:
    features = np.array(features)
    state_dict = model.state_dict()
    rbf_lengthscales = state_dict[kernel_trace + '.raw_lengthscale'].cpu().numpy().squeeze()
    rbf_columns = features[state_dict[kernel_trace + '.active_dims'].cpu().numpy().squeeze()]
    return dict(zip(rbf_columns, rbf_lengthscales))


if __name__ == "__main__":
    main()
