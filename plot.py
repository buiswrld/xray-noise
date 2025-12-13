import os
import re
import glob

import pandas as pd
import matplotlib.pyplot as plt

EVAL_DIR = os.path.join("checkpoints", "eval")
PLOTS_DIR = os.path.join("checkpoints", "plots")


def load_results(eval_dir=EVAL_DIR):
    pattern = re.compile(r"(?P<model>.+)-gauss(?P<gauss>-?\d+)-poiss(?P<poiss>-?\d+)\.csv")

    records = []

    for path in glob.glob(os.path.join(eval_dir, "*.csv")):
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if not m:
            print(f"Skipping {fname}: name does not match pattern model-gaussX-poissY.csv")
            continue

        model = m.group("model")
        gauss = int(m.group("gauss"))
        poiss = int(m.group("poiss"))

        df = pd.read_csv(path)
        if df.shape[0] != 1:
            print(f"Warning: {fname} has {df.shape[0]} rows; expected 1. Using first row.")
        row = df.iloc[0].to_dict()

        row.update({
            "model": model,
            "gauss": gauss,
            "poiss": poiss,
        })
        records.append(row)

    if not records:
        raise RuntimeError(f"No matching CSV files found in {eval_dir}")

    return pd.DataFrame(records)


def plot_metric_vs_noise(df, noise_type, metric, metric_label, out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    if noise_type == "poiss":
        # Poisson-only curves
        sub = df[df["gauss"] == 0].copy()
        x_col = "poiss"
        title_noise = "Poisson"
    elif noise_type == "gauss":
        # Gaussian-only curves
        sub = df[df["poiss"] == 0].copy()
        x_col = "gauss"
        title_noise = "Gaussian"
    else:
        raise ValueError("noise_type must be 'poiss' or 'gauss'")

    if sub.empty:
        print(f"No rows for noise_type={noise_type} (gauss==0 or poiss==0). Skipping {metric}.")
        return

    plt.figure()
    for model, g in sub.groupby("model"):
        g_sorted = g.sort_values(by=x_col)
        xs = g_sorted[x_col].values
        ys = g_sorted[metric].values
        plt.plot(xs, ys, marker="o", label=model)

    plt.xlabel(f"{title_noise} noise intensity")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs {title_noise} noise")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fname = f"{metric}_vs_{noise_type}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_metric_vs_both(df, metric, metric_label, out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    sub = df[(df["gauss"] == df["poiss"]) & (df["gauss"] != 0)].copy()

    if sub.empty:
        print(f"No rows where gauss == poiss != 0. Skipping combined plot for {metric}.")
        return

    x_col = "gauss"  # same as poiss in this filtered subset

    plt.figure()
    for model, g in sub.groupby("model"):
        g_sorted = g.sort_values(by=x_col)
        xs = g_sorted[x_col].values
        ys = g_sorted[metric].values
        plt.plot(xs, ys, marker="o", label=model)

    plt.xlabel("Shared noise intensity (Poisson + Gaussian)")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs combined Poisson + Gaussian noise")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fname = f"{metric}_vs_both.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    df = load_results()

    metrics = {
        "test_auroc": "AUROC",
        "test_auprc": "AUPRC",
        "test_f1": "F1-score",
    }

    for metric, label in metrics.items():
        # vs Poisson-only (gauss = 0)
        plot_metric_vs_noise(df, noise_type="poiss", metric=metric, metric_label=label)

        # vs Gaussian-only (poiss = 0)
        plot_metric_vs_noise(df, noise_type="gauss", metric=metric, metric_label=label)

        # vs BOTH (gauss == poiss != 0)
        plot_metric_vs_both(df, metric=metric, metric_label=label)


if __name__ == "__main__":
    main()