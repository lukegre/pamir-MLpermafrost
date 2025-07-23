import matplotlib.pyplot as plt


def plot_hist_from_series(series, **kwargs):
    stats_text = series.describe().to_string(float_format="%.4g")

    props = {"histtype": "stepfilled", "bins": 51, "color": "0.7", "zorder": 1} | kwargs

    axs = series.plot.hist(**props)

    axs.grid(which="major", axis="y", lw=0.5, zorder=2, color="k", alpha=0.2)
    axs.axvline(0, color="k", linestyle=":", lw=0.5, label="No change")

    axs.set_ylabel("Count")

    axs.text(
        0.02,
        0.98,
        stats_text,
        transform=axs.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()
    return axs.figure, axs
