import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ..data.dem_utils import calc_aspect_cos_sin

FEATURES_DEFAULT = (
    "altitude",
    "slope_angle",
    "aspect_cos",
    "aspect_sin",
    "surface_index",
    "albedo",
    "emissivity",
    "snow_melt_doy",
    "temperature",
    "precipitation",
    "temp2m_DJF_q05",
    "temp2m_JJA_q05",
    "temp2m_MAM_q05",
    "temp2m_SON_q05",
    "temp2m_DJF_q50",
    "temp2m_JJA_q50",
    "temp2m_MAM_q50",
    "temp2m_SON_q50",
    "temp2m_DJF_q95",
    "temp2m_JJA_q95",
    "temp2m_MAM_q95",
    "temp2m_SON_q95",
    "precip_DJF_q05",
    "precip_JJA_q05",
    "precip_MAM_q05",
    "precip_SON_q05",
    "precip_DJF_q50",
    "precip_JJA_q50",
    "precip_MAM_q50",
    "precip_SON_q50",
    "precip_DJF_q95",
    "precip_JJA_q95",
    "precip_MAM_q95",
    "precip_SON_q95",
)


class StandardScaler_toTensor(StandardScaler):
    """
    transform (pd.DataFrame | pd.Series | np.ndarray --> torch.Tensor)
    inverse_transform (torch.Tensor --> pd.DataFrame | pd.Series | np.ndarray)
    """

    def __init__(self, device="cpu", skip_columns: list[int] = [], **kwargs):
        """
        Initialize the StandardScaler_toTensor.
        transform (pd.DataFrame | pd.Series | np.ndarray --> torch.Tensor)
        inverse_transform (torch.Tensor --> pd.DataFrame | pd.Series | np.ndarray)

        Parameters
        ----------
        device : str, optional
            The device to which the tensor will be moved (default is 'cpu').
        skip_columns : list, optional
            List of columns to skip during scaling (default is an empty list).
            Keeps these columns unchanged during transformation.
        **kwargs : additional keyword arguments
            Additional parameters to pass to the StandardScaler constructor.
            This allows for customization of the scaler, such as `with_mean` and `with_std` parameters.
        """
        super(StandardScaler_toTensor, self).__init__(**kwargs)
        self.device = device
        self.skip_columns = skip_columns

    def fit(self, X, y=None):
        super(StandardScaler_toTensor, self).fit(X, y)

        if isinstance(X, pd.DataFrame):
            # Store feature names for inverse transformation
            self.feature_names_in_ = X.columns.tolist()
        elif isinstance(X, pd.Series):
            self.feature_names_in_ = X.name
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = None

        # return the mean and scale for the skipped columns
        if self.skip_columns:
            self.mean_[self.skip_columns] = 0
            self.scale_[self.skip_columns] = 1

        return self

    def transform(self, X):
        X_scaled = super(StandardScaler_toTensor, self).transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

    def inverse_transform(self, X_scaled_tensor):
        X_scaled = X_scaled_tensor.cpu().numpy()
        X_original = super(StandardScaler_toTensor, self).inverse_transform(X_scaled)

        if isinstance(self.feature_names_in_, list):
            X_original = pd.DataFrame(data=X_original, columns=self.feature_names_in_)
        elif isinstance(self.feature_names_in_, str):
            X_original = pd.Series(data=X_original, name=self.feature_names_in_)
        return X_original


def process_X_data(
    df: pd.DataFrame, features: tuple[str, ...] = FEATURES_DEFAULT
) -> pd.DataFrame:
    """
    Load data from a DataFrame and return features and target series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A tuple containing the features DataFrame and the target Series.
    """

    features = list(features)

    default_renames = {"albedo_modis": "albedo", "emissivity_aster": "emissivity"}
    renames = {k: v for k, v in default_renames.items() if k in df.columns}
    df = df.rename(columns=renames)

    if "stratigraphy_index" in df.columns and "surface_index" in features:
        df["surface_index"] = convert_strat_index_to_surface_index(
            df["stratigraphy_index"]
        )

    if "aspect" in df.columns and (
        "aspect_cos" in features or "aspect_sin" in features
    ):
        df = df.pipe(calc_aspect_cos_sin)

    data_X = df[features]

    return data_X


def load_training_data(fname_data="../data/training/training_data-k1500-pamir_ns180-expX.parquet", sel=None, target="ground_temp_2m"):

    if isinstance(target, str):
        target = [target]

    df = pd.read_parquet(fname_data)

    if sel is not None:
        df = df.loc[sel]

    df = df.dropna(subset=target)

    train_X = process_X_data(df)
    train_Y = df[target]

    train_X = train_X.dropna()
    train_Y = train_Y.loc[train_X.index]

    return train_X, train_Y


def load_inference_data_from_zarr(ds_spatial, features):
    ds_spatial = ds_spatial.drop_vars("spatial_ref", errors="ignore")
    df_inference = ds_spatial.to_dataframe()

    inference_X = process_X_data(df_inference, features=features)

    return inference_X


def train_test_split(
    data_X: pd.DataFrame, data_y: pd.Series, stratified_columns="surface_index"
) -> tuple:
    from sklearn import model_selection

    # train test split with stratification
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data_X,
        data_y,
        test_size=0.2,
        random_state=42,
        stratify=data_X[stratified_columns],
    )

    return X_train, X_test, y_train, y_test


def convert_strat_index_to_surface_index(data: pd.Series) -> pd.Series:
    """
    assume that the stratigraphy index is [1, 5], [2, 6], [3, 7], [4, 8],  but 0 values also exist
    So we need to convert to surface index which is 0, 1, 2, 3, 4
    """
    out = (data - 1).astype(int) % 4 + 1
    return out
