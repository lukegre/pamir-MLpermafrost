import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

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

    def __init__(self, skip_columns: list[int] = [], **kwargs):
        """
        Initialize the StandardScaler_toTensor.
        transform (pd.DataFrame | pd.Series | np.ndarray --> torch.Tensor)
        inverse_transform (torch.Tensor --> pd.DataFrame | pd.Series | np.ndarray)

        Parameters
        ----------
        skip_columns : list, optional
            List of columns to skip during scaling (default is an empty list).
            Keeps these columns unchanged during transformation.
        **kwargs : additional keyword arguments
            Additional parameters to pass to the StandardScaler constructor.
            This allows for customization of the scaler, such as `with_mean` and `with_std` parameters.
        """
        super(StandardScaler_toTensor, self).__init__(**kwargs)
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
        return torch.tensor(X_scaled, dtype=torch.float32, device="cpu")

    def inverse_transform(self, X_scaled_tensor):
        X_scaled = X_scaled_tensor.cpu().numpy()
        X_original = super(StandardScaler_toTensor, self).inverse_transform(X_scaled)

        if isinstance(self.feature_names_in_, list):
            X_original = pd.DataFrame(data=X_original, columns=self.feature_names_in_)
        elif isinstance(self.feature_names_in_, str):
            X_original = pd.Series(data=X_original, name=self.feature_names_in_)
        return X_original

    def save(self, fname: str):
        """
        Save the scaler to a file.

        Parameters
        ----------
        fname : str
            The filename to save the scaler to.
        """
        import joblib

        joblib.dump(self, fname)

    @classmethod
    def from_joblib(cls, fname: str):
        """
        Load the scaler from a file.

        Parameters
        ----------
        fname : str
            The filename to load the scaler from.

        Returns
        -------
        StandardScaler_toTensor
            The loaded scaler.
        """
        import joblib

        return joblib.load(fname)


class DatasetXarraySpatial(Dataset):
    def __init__(
        self, xarray_dataset, chunk_processor: callable, scaler: StandardScaler_toTensor
    ):
        assert len(xarray_dataset.dims) == 2, "Dataset must be 2D (time, features)"
        assert "x" in xarray_dataset.dims and "y" in xarray_dataset.dims, (
            "Dataset must have 'x' and 'y' dimensions"
        )

        self.xarray_dataset = xarray_dataset
        self.chunk_processor = chunk_processor
        self.scaler = scaler
        if scaler is not None:
            assert isinstance(scaler, StandardScaler_toTensor), (
                "Scaler must be an instance of StandardScaler_toTensor"
            )
            self.scaler = scaler
        self.chunk_mapper = None
        self._make_chunk_index()

    def __len__(self):
        return len(self.chunk_mapper)

    def __getitem__(self, idx):
        ds = self._get_chunk(idx)
        df = ds.to_dataframe()
        X = self.chunk_processor(df)
        X_tensor_scaled = self.scaler.transform(X)
        return X_tensor_scaled

    def _get_chunk(self, idx):
        """
        If the dataset is chunked, this method returns the
        chunk at the specified index for the entire dataset.
        """
        if self.chunk_mapper is None:
            self._make_chunk_index()

        if idx not in self.chunk_mapper:
            raise IndexError(
                f"Chunk index {idx} out of range. Available chunks: {list(self.chunk_mapper.keys())}"
            )

        chunk_selector = self.chunk_mapper[idx]
        ds_chunk = self.xarray_dataset.isel(**chunk_selector)
        return ds_chunk

    def _make_chunk_index(self):
        from itertools import product

        ds = self.xarray_dataset
        chunk_mapper = {}

        xchunks = list(ds.chunks["x"])
        ychunks = list(ds.chunks["y"])

        xchunks_cum0 = np.cumsum([0] + xchunks[:-1]).tolist()
        xchunks_cum1 = np.cumsum(xchunks).tolist()
        ychunks_cum0 = np.cumsum([0] + ychunks[:-1]).tolist()
        ychunks_cum1 = np.cumsum(ychunks).tolist()

        xslices = [slice(i0, i1) for i0, i1 in zip(xchunks_cum0, xchunks_cum1)]
        yslices = [slice(i0, i1) for i0, i1 in zip(ychunks_cum0, ychunks_cum1)]

        xy_slices = list(product(xslices, yslices))

        for i, (xs, ys) in enumerate(xy_slices):
            chunk_mapper[i] = dict(x=xs, y=ys)

        self.chunk_mapper = chunk_mapper

    def reconstruct_output(self, output: list) -> xr.DataArray:
        """
        Reconstruct the output from the dataset.
        This method is a placeholder and should be implemented
        based on the specific requirements of the output format.
        """
        # For now, just return the output as is
        assert len(output) == len(self.chunk_mapper), (
            "Output length must match the number of chunks"
        )

        key0 = list(self.xarray_dataset.data_vars)[0]
        dummy = self.xarray_dataset[key0].compute().rename("reconstructed") * np.nan

        for i, chunk in enumerate(output):
            chunk_iselector = self.chunk_mapper[i]
            # assign chunk to the corresponding slice in the dummy DataArray
            y_size = chunk_iselector["y"].stop - chunk_iselector["y"].start
            x_size = chunk_iselector["x"].stop - chunk_iselector["x"].start

            x_sel = dummy.x.isel(x=chunk_iselector["x"])
            y_sel = dummy.y.isel(y=chunk_iselector["y"])
            dummy.loc[y_sel, x_sel] = chunk.reshape(y_size, x_size)

        return dummy


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


def load_training_data(fname_data, sel=None, target="ground_temp_2m"):
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

    m = (train_Y != 0).values.ravel()  # remove zero values from target
    train_X = train_X[m]
    train_Y = train_Y[m]

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
