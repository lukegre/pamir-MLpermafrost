import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pamir_mlpermafrost.utils.chained_upath import ChainedUPath


def plot_depths(da, path_to_zarr: str, title="", info="", **kwargs):
    props = (
        dict(
            figsize=(12, 7.5),
            col="depth",
            col_wrap=4,
            robust=True,
            cbar_kwargs={"pad": 0.02},
        )
        | kwargs
    )

    fg = da.plot.imshow(**props)

    fg.fig.suptitle(title, fontsize=14, weight="bold", ha="left", x=0.05, y=1.02)
    fg.set_titles(
        template="Depth: {value} m",
        loc="left",
        weight="bold",
        x=0.019,
        y=0.97,
        va="bottom",
        zorder=0,
    )
    fg.set_titles(template="")
    fg.set_xlabels("")

    add_plot_meta(fg.axs, info, path_to_zarr)

    return fg.fig, fg.axs, fg.cbar


def add_plot_meta(axs, info, path, **kwargs):
    if isinstance(axs, plt.Axes):
        axs = np.array([[axs]])

    fig = axs.flat[0].figure

    path = ChainedUPath(path)

    if info:
        info = f"$\\mathbf{{Info}}$:    {info}\n"

    text = (
        f"{info}"
        f"$\\mathbf{{File}}$:    {path.name}\n"
        f"$\\mathbf{{S3}}$:     {str(path.parent).replace('simplecache::', '')}\n"
        f"$\\mathbf{{Date}}$:   {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    x = axs[-1, 0].get_position().x0
    y0 = axs[-1, 0].get_position().y0
    dh = axs[-1, 0].get_position().height
    y = y0 - dh * fig.subplotpars.hspace

    props = dict(x=x, y=y, s=text, va="top", size=8, family="monospace") | kwargs

    fig.text(**props)
