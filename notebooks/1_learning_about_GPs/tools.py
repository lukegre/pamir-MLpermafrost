import gpytorch
import numpy as np
from matplotlib import pyplot as plt


class plotarray(np.ndarray):
    def plot(self, *args, **kwargs):
        ax = kwargs.pop("ax", plt.gca())
        ax.plot(self, *args, **kwargs)
        return ax


def plot(
    X_train,
    y_train,
    X_test,
    mu_s,
    std_s,
    mu_prior=None,
    std_prior=None,
    samples_prior=None,
):
    # Plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(X_train, y_train, "ko", label="Training points", zorder=10)
    ax.plot(X_test, mu_s, "b", label="Mean prediction", zorder=9)

    if samples_prior is not None:
        ax.plot(X_test, samples_prior.T, "k", alpha=0.05)
    if mu_prior is not None and std_prior is not None:
        ax.fill_between(
            X_test.ravel(),
            mu_prior - 2 * std_prior,
            mu_prior + 2 * std_prior,
            color="grey",
            alpha=0.15,
            label="±2σ prior band",
        )

    ax.fill_between(
        X_test.ravel(),
        mu_s - 2 * std_s,
        mu_s + 2 * std_s,
        color="blue",
        alpha=0.15,
        label="±2 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(ncol=2)
    ax.grid(False)
    ax.axhline(0, color="black", lw=0.5, ls="--")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig, ax


def pretty_kernel(kernel, **replacements):
    def recurse(k):
        if hasattr(k, "kernels"):  # Composite kernel (Additive/Product)
            op = "+" if isinstance(k, gpytorch.kernels.AdditiveKernel) else "*"
            return f"({f' {op} '.join(recurse(subk) for subk in k.kernels)})"
        elif isinstance(k, gpytorch.kernels.ScaleKernel):
            return f"{k.__class__.__name__}({recurse(k.base_kernel)})"
        else:
            return k.__class__.__name__

    clipped = recurse(kernel)
    out = clipped
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out
