import os
import re
import glob
import pandas as pd

EVAL_DIR = os.path.join("checkpoints", "eval")
SUMMARY_DIR = os.path.join("checkpoints", "summary")

def load_all_results(eval_dir=EVAL_DIR):
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


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    df = load_all_results()
    df = df.sort_values(by=["model", "gauss", "poiss"])

    all_path = os.path.join(SUMMARY_DIR, "all_models_noise_summary.csv")
    df.to_csv(all_path, index=False)
    print(f"Saved global summary to {all_path}")

    # one CSV per model
    for model, g in df.groupby("model"):
        g_sorted = g.sort_values(by=["gauss", "poiss"])
        out_path = os.path.join(SUMMARY_DIR, f"{model}_noise_summary.csv")
        g_sorted.to_csv(out_path, index=False)
        print(f"Saved per-model summary for {model} to {out_path}")


if __name__ == "__main__":
    main()
