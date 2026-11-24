import os

import matplotlib.pyplot as plt
import seaborn as sns


def heatmap(piv, xlabel, ylabel, title, fname):
    # Clamp the color scale at 0.85: a diverged run (acc ~ 0.10) would flatten everything else
    os.makedirs("figures", exist_ok=True)
    vmin = max(piv.min().min(), 0.85)
    sns.heatmap(piv, annot=True, fmt=".4f", cmap="viridis", vmin=vmin,
                cbar_kws={"label": "validation accuracy"})
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.savefig(f"figures/{fname}.pdf", bbox_inches="tight")
    plt.show()
