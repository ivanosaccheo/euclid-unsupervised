import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


def plot_latent_space(
    mu,
    labels,
    dims=(0, 1, 2, 3, 4),
    unlabeled_value=-1,
    colors=("goldenrod", "crimson", "blue"),
    names=("Star", "Galaxy", "QSO"),
    gridsize=100,
    mincnt=1,
    scatter_size=2,
    scatter_alpha=0.25,
    contour_levels=4,
):
    mu = np.asarray(mu)
    labels = np.asarray(labels)

    z = mu[:, dims]
    n_dim = len(dims)

    # compact grid: (n_dim-1) x (n_dim-1)
    fig, axes = plt.subplots(n_dim-1, n_dim-1, figsize=(10, 10))

    classes = [c for c in np.unique(labels) if c != unlabeled_value]
    unlabeled_mask = labels == unlabeled_value

    for i in range(1, n_dim):
        for j in range(i):

            row = i - 1
            col = j

            ax = axes[row, col]

            # background (hexbin)
            bg_mask = unlabeled_mask if np.any(unlabeled_mask) else np.ones(len(z), dtype=bool)

            ax.hexbin(
                z[bg_mask, j],
                z[bg_mask, i],
                gridsize=gridsize,
                mincnt=mincnt,
                norm=LogNorm(),
                cmap="Greys",
                rasterized=True,
            )

            # labeled overlay
            for c, color, name in zip(classes, colors, names):
                mask = labels == c

                ax.scatter(
                    z[mask, j],
                    z[mask, i],
                    s=scatter_size,
                    alpha=scatter_alpha,
                    color=color,
                    rasterized=True,
                )

                if np.sum(mask) > 50:
                    sns.kdeplot(
                        x=z[mask, j],
                        y=z[mask, i],
                        ax=ax,
                        levels=contour_levels,
                        color=color,
                        linewidths=1.2,
                    )

            # labels only on edges
            if row == n_dim - 2:
                ax.set_xlabel(f"$z_{dims[j]}$")
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(f"$z_{dims[i]}$")
            else:
                ax.set_yticklabels([])

    # hide unused upper-right panels
    for i in range(n_dim-1):
        for j in range(n_dim-1):
            if j > i:
                axes[i, j].axis("off")

    # legend
    handles = [
        plt.Line2D([], [], color=color, label=name)
        for color, name in zip(colors, names)
    ]
    fig.legend(handles=handles, loc="upper right")

    fig.subplots_adjust(
        left=0.07, right=0.97,
        bottom=0.07, top=0.97,
        wspace=0.05, hspace=0.05
    )

    return fig
    return fig



