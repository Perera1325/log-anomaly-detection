import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrices(summary, path):
    """One confusion-matrix panel per detector (needs TP/FP/FN/TN columns)."""
    fig, axes = plt.subplots(1, len(summary), figsize=(3.4 * len(summary), 3.4))
    for ax, (_, r) in zip(np.atleast_1d(axes), summary.iterrows()):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], ["Normal", "Anomaly"])
        ax.set_yticks([0, 1], ["Normal", "Anomaly"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{r['Model']}\nF1={r['F1']:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_agreement_heatmap(kappa, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(kappa.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(kappa)), kappa.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(kappa)), kappa.index)
    for i in range(len(kappa)):
        for j in range(len(kappa)):
            ax.text(j, i, f"{kappa.iloc[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_title("Model agreement (Cohen's kappa)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
