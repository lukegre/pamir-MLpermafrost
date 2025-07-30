import numpy as np
import pandas as pd
import xarray as xr


def calc_flip_aspect_north_south(aspect):
    """
    Flips aspect so that North is 180 and South is 0,
    but keeps East 90 and West 270.
    """
    aspect = -aspect % 360
    aspect = (aspect - 180) % 360
    return aspect


def _calc_aspect_cos_sin(
    aspect: np.ndarray | pd.Series,
) -> tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]:
    """
    Converts aspect to cosine and sine components.
    """
    aspect_rad = np.deg2rad(aspect)
    cos = np.cos(aspect_rad)
    sin = np.sin(aspect_rad)

    if isinstance(aspect, pd.Series):
        cos = cos.rename("aspect_cos")
        sin = sin.rename("aspect_sin")

    return cos, sin


def calc_aspect_cos_sin(
    data: pd.DataFrame | xr.DataArray, aspect_name: str = "aspect"
) -> pd.DataFrame | xr.DataArray:
    """
    Adds cosine and sine components of aspect to the DataFrame.
    """
    cos, sin = _calc_aspect_cos_sin(data[aspect_name])
    data["aspect_cos"] = cos
    data["aspect_sin"] = sin

    if isinstance(data, pd.DataFrame):
        data = data.drop(columns=[aspect_name, "spatial_ref"], errors="ignore")
    elif isinstance(data, xr.DataArray):
        data = data.drop_vars([aspect_name, "spatial_ref"], errors="ignore")

    return data
