import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR
from analysis.statistical import load_all_results, compute_avg_rankings, compute_rank_matrix


os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_boxplot(df, metric="f1", save=True):
    metric_df = df[df["metric"] == metric].copy()
    plt.figure(figsize=(12, 6))
    order = metric_df.groupby("smote")["mean"].median().sort_values().index.tolist()
    sns.boxplot(data=metric_df, x="smote", y="mean", order=order, palette="Set2")
    plt.title(f"Distribucija {metric.upper()} po SMOTE varijantama")
    plt.xlabel("SMOTE varijanta")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"boxplot_{metric}.pdf")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved boxplot to {path}")


def plot_cd_diagram(df, metric="f1", save=True):
    """Kritični dijagram — statističke grupe."""
    try:
        rank_matrix = compute_rank_matrix(df, metric)
        avg_ranks = rank_matrix.mean().sort_values()
        n_datasets = rank_matrix.shape[0]
        n_algos = rank_matrix.shape[1]

        from scipy.stats import studentized_range
        q_alpha = studentized_range.ppf(0.95, n_algos, (n_datasets - 1) * (n_algos - 1))
        cd = q_alpha * np.sqrt(n_algos * (n_algos + 1) / (6 * n_datasets))

        fig, ax = plt.subplots(figsize=(10, 4))
        y_positions = np.arange(len(avg_ranks))
        ax.scatter(avg_ranks.values, y_positions, s=60, zorder=5)

        for i, (algo, rank) in enumerate(avg_ranks.items()):
            ax.text(rank + 0.02, i, algo, va="center", fontsize=10)

        for i in range(len(avg_ranks)):
            group = [i]
            for j in range(i + 1, len(avg_ranks)):
                if avg_ranks.iloc[j] - avg_ranks.iloc[i] <= cd:
                    group.append(j)
                else:
                    break
            if len(group) > 1:
                y_center = np.mean(group)
                ax.plot(
                    [avg_ranks.iloc[group[0]], avg_ranks.iloc[group[-1]]],
                    [y_center, y_center],
                    "k-", linewidth=2,
                )

        ax.set_title(f"CD dijagram — {metric.upper()}")
        ax.set_xlabel("Prosječan rang")
        ax.set_ylim(-1, len(avg_ranks))
        ax.invert_yaxis()
        plt.tight_layout()
        if save:
            path = os.path.join(FIGURES_DIR, f"cd_diagram_{metric}.pdf")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved CD diagram to {path}")
    except Exception as e:
        print(f"  CD diagram failed: {e}")


def plot_heatmap(df, metric="f1", save=True):
    pivot = df[df["metric"] == metric].pivot_table(
        index="dataset", columns="smote", values="mean", aggfunc="median",
    )
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={"label": metric.upper()})
    plt.title(f"Heatmap — {metric.upper()} (Data × SMOTE)")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"heatmap_{metric}.pdf")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved heatmap to {path}")


def generate_all_plots(metric="f1"):
    df = load_all_results()
    print(f"\n  Generating plots for {metric.upper()}...")
    plot_boxplot(df, metric)
    plot_cd_diagram(df, metric)
    plot_heatmap(df, metric)
