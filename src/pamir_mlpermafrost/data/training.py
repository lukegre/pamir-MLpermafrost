import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

_valid_dataset = """
<xarray.Dataset> Size: ...
Dimensions:                    (tag depth, time_1YS, time, y, y,
                                variable, metrics, thermal_class)
Coordinates:
  * depth                      (depth) float64
  * metrics                    (metrics) <U
  * tag                        (tag) int64
  * thermal_class              (thermal_class) <U
  * time                       (time) datetime64[ns]
  * time_1YS                   (time_1YS) datetime64[ns]
  * variable                   (variable) <U
  * x                          (x) float64
  * y                          (y) float64
Data variables:
    active_layer_temp_metrics  (tag, metrics, time_1YS) float32
    cluster_labels             (y, x) int64
    cluster_params             (tag, variable) float64
    elevation                  (tag) float32
    ground_class_mask          (tag, depth, time_1YS) float32
    ground_class_thickness     (tag, thermal_class, time_1YS) float32
    permafrost_temp_metrics    (tag, metrics, time_1YS) float32
    temperature                (tag, depth, time) float32
"""


def get_training_data_table(
    ds: xr.Dataset,
    experiment_name=None,
    ground_temp_depths=[-2, -5, -10],
    progressbar=False,
) -> pd.DataFrame:
    validate_output_dataset(ds)

    df = get_cluster_params_table(ds)

    df["active_layer_thickness"] = get_active_layer_thickness(ds)
    df["permafrost_thickness"] = get_permafrost_thickness(ds)
    df["active_layer_temperature"] = get_active_layer_temperature(ds)
    df["permafrost_temperature"] = get_permafrost_temperature(ds)

    df = df.assign(
        **get_ground_temperature(ds, progressbar=progressbar, depths=ground_temp_depths)
    )

    df["profile"] = get_profile_from_tag(df.index)

    if experiment_name is not None:
        current_index = df.index.name
        df = df.reset_index()
        df["experiment"] = experiment_name
        df = df.set_index(["experiment", current_index])

    return df


def validate_output_dataset(ds: xr.Dataset) -> bool:
    required_vars = ["cluster_params", "ground_class_thickness", "temperature"]
    present_vars = [var in ds for var in required_vars]
    missing_variables = ", ".join(
        [var for var, present in zip(required_vars, present_vars) if not present]
    )
    message = "Dataset is not valid - missing variables: " + missing_variables
    message += "\nValid dataset structure is:\n" + _valid_dataset
    assert all(present_vars), message
    return True


def get_profile_from_tag(tag: np.ndarray, n_profiles: int | None = None) -> np.ndarray:
    if n_profiles is None:
        n_profiles = len(np.unique(tag))
    if n_profiles <= 0:
        raise ValueError("Number of profiles must be a positive integer.")

    n_chars = len(str(n_profiles))

    tag_str = pd.Series(tag, dtype=np.str_)
    tag_str = tag_str.str[-n_chars:]

    tag = tag_str.astype(np.int64).values

    return tag


def get_cluster_params_table(ds: xr.Dataset) -> pd.DataFrame:
    return ds.cluster_params.to_series().unstack()


def get_permafrost_thickness(
    ds: xr.Dataset, agg_func="max", dim="time_1YS"
) -> pd.Series:
    """
    Get the permafrost depth from the dataset.
    """
    da = ds.ground_class_thickness.sel(thermal_class="permafrost")
    func = getattr(da, agg_func, None)
    if func is None:
        raise ValueError(
            f"Aggregation function '{agg_func}' is not valid for xarray DataArray."
        )
    da = func(dim=dim)
    da.name = "permafrost_depth"
    da.attrs["units"] = "m"
    da.attrs["description"] = (
        "Permafrost depth averaged over the specified dimension. "
        f"Aggregation function used: `{agg_func}` over dimension `{dim}`"
    )
    ser = da.to_series()
    ser.attrs = da.attrs.copy()
    return ser


def _get_layer_temperature(
    ds: xr.Dataset, layer_name: str, metric="mean", dim="time_1YS"
) -> pd.Series:
    """
    Get the temperature for a specific layer from the dataset.
    """
    assert isinstance(layer_name, str), "Layer name must be a string."
    assert isinstance(metric, str), (
        "Metric must be a string representing the metric name."
    )

    da = ds[layer_name].sel(metrics=metric).mean(dim=dim)
    da.name = f"{layer_name}_{metric}"

    ser = da.to_pandas()
    return ser


def get_active_layer_temperature(
    ds: xr.Dataset, metric="mean", dim="time_1YS"
) -> pd.Series:
    ser = _get_layer_temperature(ds, "active_layer_temp_metrics", metric, dim)
    ser.name = "active_layer_temperature"
    ser.attrs["units"] = "C"
    return ser


def get_permafrost_temperature(
    ds: xr.Dataset, metric="mean", dim="time_1YS"
) -> pd.Series:
    ser = _get_layer_temperature(ds, "permafrost_temp_metrics", metric, dim)
    ser.name = "permafrost_temperature"
    ser.attrs["units"] = "C"
    return ser


def get_active_layer_thickness(
    ds: xr.Dataset, agg_func="mean", dim="time_1YS"
) -> pd.Series:
    """
    Get the active layer thickness from the dataset.
    """
    da = ds.ground_class_thickness.sel(thermal_class="active_layer").squeeze()

    func = getattr(da, agg_func, None)
    if func is None:
        raise ValueError(
            f"Aggregation function '{agg_func}' is not valid for xarray DataArray."
        )

    dim_min = da[dim].min()
    dim_max = da[dim].max()
    if dim_min.dtype == "datetime64[ns]" and dim_max.dtype == "datetime64[ns]":
        dim_min = dim_min.dt.strftime("%Y-%m-%d").item()
        dim_max = dim_max.dt.strftime("%Y-%m-%d").item()

    da = func(dim=dim, skipna=True)
    da.name = "active_layer_thickness"
    da.attrs["units"] = "m"
    da.attrs["description"] = (
        "Active layer thickness averaged over the specified dimension. "
        f"Aggregation function used: `{agg_func}` over dimension `{dim}` with "
        f"original span `{dim_min} - {dim_max}`"
    )

    ser = da.to_series()
    ser.attrs = da.attrs.copy()
    return ser


def get_ground_temperature(
    ds: xr.Dataset,
    depths: float | list[float] = [-2, -5, -10],
    agg_func="mean",
    dim="time",
    progressbar=False,
) -> pd.DataFrame:
    if progressbar:
        from tqdm.dask import TqdmCallback as ProgressBar
    else:
        from ..utils import DummyContextManager as ProgressBar

    key = "temperature"

    da = ds[key].sel(depth=depths, method="nearest")

    func = getattr(da, agg_func, None)
    if func is None:
        raise ValueError(
            f"Aggregation function '{agg_func}' is not valid for xarray DataArray."
        )

    da = func(dim=dim)
    da["depth"] = [f"ground_temp_{abs(d):.2g}m" for d in da.depth.values]

    with ProgressBar(desc=f"{agg_func}( {key}@{depths} )"):
        df = da.to_series().unstack()

    return df


def get_collocated_spatial_data(
    spatial: xr.Dataset, table: pd.DataFrame, lat_name="latitude", lon_name="longitude"
) -> pd.DataFrame:
    """
    Collocate spatial data with the provided table based on latitude and longitude.

    Parameters:
    - spatial: xr.Dataset containing spatial variables.
    - table: pd.DataFrame with 'latitude' and 'longitude' columns.
    - lat_name: Name of the latitude column in the table.
    - lon_name: Name of the longitude column in the table.

    Returns:
    - xr.Dataset with collocated spatial data.
    """

    selector = (
        table[[lat_name, lon_name]]
        .rename(columns={lat_name: "y", lon_name: "x"})
        .to_xarray()
    )

    ds_sel = spatial.sel(**selector, method="nearest")

    df_sel = ds_sel.to_dataframe()
    df_sel = df_sel.loc[table.index]

    df_sel[f"table_y"] = table[lat_name]
    df_sel[f"table_x"] = table[lon_name]

    df_sel["x_diff"] = df_sel["x"] - df_sel[f"table_x"]
    df_sel["y_diff"] = df_sel["y"] - df_sel[f"table_y"]

    tolerance = 100 / 111_100  # 100m in degrees
    df_sel["within_tolerance"] = (df_sel["x_diff"].abs() < tolerance) & (
        df_sel["y_diff"].abs() < tolerance
    )

    if not df_sel["within_tolerance"].all():
        logger.warning(
            "Some points in the table are not within the specified tolerance of the spatial data. "
            "Check the 'within_tolerance' column for details."
        )

    return df_sel
