import pandas as pd
import xarray as xr


def process_X(df: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
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
    from .dem_utils import calc_aspect_cos_sin

    features = list(features)

    renames = {"albedo_modis": "albedo", "emissivity_aster": "emissivity"}

    df = df.rename(columns=renames, errors="ignore")

    # checks to see if we need to compute some features
    has_strat_index = "stratigraphy_index" in df.columns
    has_surface_index = "surface_index" in features
    has_aspect = "aspect" in features
    has_cos_sin = "aspect_cos" in features or "aspect_sin" in features

    if has_strat_index and has_surface_index:
        df["surface_index"] = _convert_strat_to_surface_idx(df["stratigraphy_index"])
    if has_aspect or has_cos_sin:
        df = df.pipe(calc_aspect_cos_sin)

    data_X = df[features]

    return data_X


def process_Xy(
    df: pd.DataFrame, features: tuple[str, ...], target: str
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.dropna(subset=target)

    train_X = process_X(df, features)
    train_y = df[target]

    train_X = train_X.dropna()
    train_y = train_y.loc[train_X.index]

    m = (train_y != 0).values.ravel()  # remove zero values from target
    train_X = train_X[m]
    train_y = train_y[m]

    return train_X, train_y


def _convert_strat_to_surface_idx(data: pd.Series) -> pd.Series:
    """
    assume that the stratigraphy index is [1, 5], [2, 6], [3, 7], [4, 8],  but 0 values also exist
    So we need to convert to surface index which is 0, 1, 2, 3, 4
    """
    out = (data - 1).astype(int) % 4 + 1
    return out
