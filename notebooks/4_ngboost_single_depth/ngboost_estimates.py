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
    "yhat_std": dict(vmin=0, vmax=4.5),
    # "temperature": dict(robust=True, cmap="RdBu_r", center=273.15),
    # "altitude": dict(robust=True, cmap="terrain"),
    # "aspect": dict(cmap="twilight_r"),
    # "surface_index": dict(robust=True, cmap="turbo", vmin=0, vmax=5),
}

@hydra.main(
    version_base=None,
    config_path="../src/pamir_mlpermafrost/conf",
    config_name="ngboost")
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

        train_X, train_y, test_X, test_y = load_training_data(cfg)

        # create model
        # train model
        #   train with CV
        # log model

        if cfg.get('plot_map', True) or cfg.get('write_netcdf'):
            # inference
            if cfg.plot_map:
                plot_results(output)
            if cfg.write_netcdf:
                log_netcdf(output, cfg)


def load_training_data(cfg):
    """
    Load the training data and preprocess it according to the configuration.
    """
    # Load the training data
    data = cfg.data.training.load()
    data = data.reset_index().set_index(["y", "x"])

    # Preprocess the data
    data_X, data_y = cfg.preprocessing.training(data)

    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = cfg.preprocessing.train_test_split(
        data_X, data_y
    )

    plot_train_test_split(train_X, test_X)

    return (
        train_X,
        train_y,
        test_X,
        test_y,
    )


def make_models():
    import ngboost
    from itertools import product

    param_ranges = {
        'max_depth': [3, 6, 9, 15, 25], 
        'min_samples_leaf': [5, 10, 20, 30, 40, 50], 
        'learning_rate': [0.01]}
    
    param_list = product(*param_ranges.values())
    param_list = [dict(zip(param_ranges.keys(), values)) for values in param_list]
    
    
    models = []
    for params in param_list:
        models += ngboost.NGBRegressor(
            Base=ngboost.learners.DecisionTreeRegressor(
                splitter='random',
                min_impurity_decrease=0.005,
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf']), 
            n_estimators=5_000, 
            verbose_eval=False, 
            early_stopping_rounds=5,
            learning_rate=params['learning_rate']),
    return models


def _get_figsize(x: pd.Series|xr.DataArray, y: pd.Series|xr.DataArray, fig_w: float = 10):
    """
    Calculate figure size based on the aspect ratio of the data.
    """
    pixel_aspect = 1 / np.cos(np.deg2rad(float(y.mean())))
    fig_aspect = pixel_aspect * 1.1
    fig_h = fig_w / fig_aspect
    return fig_w, fig_h
    

def plot_train_test_split(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,):

    len_train = len(train_X)
    len_test = len(test_X)

    coords_train = train_X.index.to_frame()
    coords_test = test_X.index.to_frame()
    coords = pd.concat([coords_train, coords_test])

    figsize = _get_figsize(coords['x'], coords['y'])

    fig, axs = plt.subplots(figsize=figsize, dpi=150)
    coords_train.plot(x='x', y='y', kind='scatter', c='b', ax=axs, label=f'Train (n={len_train})')
    coords_test.plot(x='x', y='y', kind='scatter', c='r', ax=axs, label=f'Test (n={len_test})')

    frac = len_test / (len_train + len_test)
    axs.set_title(f"Train/Test Split (Test Size: {frac:.2f})")
    axs.legend()

    fig.tight_layout()
    image = fig_to_pilimage(fig)
    mlflow.log_image(image, key='train_test_split', step=0)



def plot_results(ds):

    mask = ds.surface_index > 0
    figsize = _get_figsize(ds.x, ds.y, fig_w=10)

    for i, key in enumerate(PLOT_VARS.keys()):
        da = ds[key].where(mask)
        props = PLOT_VARS[key]

        fig, axs = plt.subplots(figsize=figsize, dpi=150)
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
    fig.savefig(buf, format='png', transparent=True, bbox_inches='tight')
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

