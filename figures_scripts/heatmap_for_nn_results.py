import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
def plot_metrics_heatmap(
    df: pd.DataFrame,
    row_label_col: str,
    metric_cols=("precision", "recall", "f1", "accuracy"),
    sort_by = None,
    ascending: bool = False,
    figsize=(10, 6),
    cmap="viridis",
    vmin= 0.0,
    vmax= 1.0,
    fmt=".3f",
    title: str = "Metrics Heatmap",
):
    """
    df: DataFrame with one row per configuration and metric columns in [0,1].
    row_label_col: column name used as the row label (e.g., 'config' / 'setting').
    """
    missing = [c for c in (row_label_col, *metric_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    plot_df = df[[row_label_col, *metric_cols]].copy()
    plot_df = plot_df.set_index(row_label_col)

    if sort_by is not None:
        if sort_by not in metric_cols:
            raise ValueError(f"sort_by must be one of {metric_cols}, got '{sort_by}'")
        # plot_df = plot_df.sort_values(by=sort_by, ascending=ascending)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_df,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Add this line

    ax.set_xlabel("Metric")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig, ax

def compute_metrics(df):
    df = df.copy()

    df["precision"] = df["TP"] / (df["TP"] + df["FP"])
    df["recall"]    = df["TP"] / (df["TP"] + df["FN"])
    df["f1"]        = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"])
    df["accuracy"]  = (df["TP"] + df["TN"]) / (
        df["TP"] + df["FP"] + df["FN"] + df["TN"]
    )

    return df


# Example usage:
if __name__ == "__main__":
    # Sample data
    figures_path = '/Users/rinatsaban/Library/CloudStorage/OneDrive-mail.tau.ac.il/TAU/master/docs for writing/figures'

    data = [

        {"config": "Both Evaluation Set", "TP": 10919 , "FP": 4357 , "FN": 1976 , "TN": 4140},
        {"config": "Both Test Set", "TP": 4868 , "FP": 2792 , "FN": 7396 , "TN": 2706},

        {"config": "'OR' Evaluation Set", "TP": 9963 , "FP": 4049 , "FN": 2933 , "TN": 4447},
        {"config": "'OR' Test Set", "TP": 3017  , "FP": 1495  , "FN": 9247  , "TN": 4003},
    ]

    df = pd.DataFrame(data)
    df_metrics = compute_metrics(df)

    fig, ax = plot_metrics_heatmap(
        df_metrics,
        row_label_col="config",
        metric_cols=("precision", "recall", "f1", "accuracy"),
        sort_by="f1",
        ascending=False,
        title="Model Performance Comparison",
        cmap="viridis"
    )

    fig.savefig(os.path.join(figures_path, "mmd_model_comparison.png"))