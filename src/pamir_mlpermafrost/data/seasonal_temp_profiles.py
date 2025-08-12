import numpy as np
import xarray as xr
from tqdm.dask import TqdmCallback as ProgressBarDask


def get_season_names(bins):
    import calendar

    names = []
    for i0 in range(len(bins) - 1):
        i1 = i0 + 1
        months = np.ceil(np.arange(bins[i0], bins[i1])).astype(int)
        letters = "".join([calendar.month_abbr[m][0] for m in months])
        names.append(letters)

    return names


def get_seasonal_quantiles(
    fname, bins=[0.5, 3.5, 6.5, 9.5, 12.5], quantiles=[0.25, 0.5, 0.75]
):
    ds = xr.open_zarr(fname, consolidated=True)

    bins = [0.5, 3.5, 6.5, 9.5, 12.5]
    labels = get_season_names(bins)

    with ProgressBarDask():
        temp_profiles = (
            ds["temperature"]
            .sel(depth=slice(-0.05, -20))
            .groupby_bins("time.month", bins=bins, labels=labels)
            .quantile(quantiles)
            .rename(month_bins="season")
            .compute()
            .chunk(dict(quantile=1, season=1))
        )

    temp_profiles["season"] = temp_profiles["season"].astype(str)
    return temp_profiles


def get_gev_quantile(arr, q=0.99):
    """
    Fit a Generalized Extreme Value (GEV) distribution to the data and return the quantile at q.
    """
    from scipy.stats import genextreme as gev

    arr = arr[(arr != 0) & ~np.isnan(arr)]  # remove zeros and NaNs

    # Fit GEV (SciPy uses shape 'c' = -xi)
    c, loc, scale = gev.fit(arr)  # MLE
    x_q = gev.ppf(q, c, loc, scale)  # quantile function

    return x_q


def adjust_temp_quantiles_to_stationary(
    temperature,
    deep=slice(-17, -20),
    dims=["quantile", "season"],
    gev_quantile_threshold=0.99,
):
    """
    Adjust temperature profiles to account for non-stationarity by removing the offset at deep depths.
    Adjusts to the median temperature at deep depths, assuming that these depths are stable over time.

    Problem with this approach is that processes occurring at the surface (e.g., freeze-thaw) temperatures
    are shifted, thus not respecting the physical processes.
    """
    sigma_temp_deep = temperature.sel(depth=deep).std(dims)
    thresh = get_gev_quantile(sigma_temp_deep.values, q=gev_quantile_threshold)

    deep_stable_temp = sigma_temp_deep.mean("depth") < thresh
    good_profiles = temperature.sel(tag=deep_stable_temp.values)

    # this part a bit ugly since it's hardcoded to the dims and quantile, very input-specific
    deep_temp_mean = good_profiles.sel(depth=deep).median(["depth", "season"])
    ref = deep_temp_mean.sel(quantile=0.5)
    deep_temp_offset = deep_temp_mean - ref
    profiles_corrected = good_profiles - deep_temp_offset

    profiles_low_std = (
        profiles_corrected.sel(depth=-20).std(["season", "quantile"]) < 0.1
    )
    assert profiles_low_std.all(), (
        "Not all profiles have low standard deviation at -20m depth."
    )

    return profiles_corrected
