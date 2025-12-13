import os, csv, glob, random
import argparse
from pathlib import Path

def list_imgs(d):
    exts = ("*.png","*.jpg","*.jpeg","*.JPG","*.JPEG","*.PNG")
    out = []
    for e in exts:
        out += glob.glob(os.path.join(d, e))
    return out

def main(data_root, val_frac=0.12, seed=42, out_csv="data/labels.csv"):
    random.seed(seed)
    data_root = Path(data_root)

    train_norm = list_imgs(data_root/"train"/"NORMAL")
    train_pneu = list_imgs(data_root/"train"/"PNEUMONIA")
    tiny_val_norm = list_imgs(data_root/"val"/"NORMAL")
    tiny_val_pneu = list_imgs(data_root/"val"/"PNEUMONIA")
    train_norm += tiny_val_norm
    train_pneu += tiny_val_pneu

    test_norm  = list_imgs(data_root/"test"/"NORMAL")
    test_pneu  = list_imgs(data_root/"test"/"PNEUMONIA")

    def strat_split(paths, frac):
        paths = paths[:]
        random.shuffle(paths)
        k = int(len(paths)*frac)
        return paths[k:], paths[:k]

    trN, vaN = strat_split(train_norm, val_frac)
    trP, vaP = strat_split(train_pneu, val_frac)

    rows = []
    for p in trN: rows.append((p, 0, "train"))
    for p in trP: rows.append((p, 1, "train"))
    for p in vaN: rows.append((p, 0, "val"))
    for p in vaP: rows.append((p, 1, "val"))
    for p in test_norm: rows.append((p, 0, "test"))
    for p in test_pneu: rows.append((p, 1, "test"))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path","label","split"])
        w.writerows(rows)

    print(f"Wrote {out_csv}")
    print(f"train={len(trN)+len(trP)}, val={len(vaN)+len(vaP)}, test={len(test_norm)+len(test_pneu)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to Kaggle chest_xray folder")
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="data/labels.csv")
    args = ap.parse_args()
    main(**vars(args))
