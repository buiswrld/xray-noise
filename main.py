import argparse, os, random, csv
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryPrecision, BinaryRecall, BinaryF1Score

from classifier import ClassificationTask
from dataset import ChestXrayDataset


#utils

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

def seed_worker(worker_id): 
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def default_transforms(img_size=224, is_train=False):
    tf = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    if is_train:
        tf.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    return transforms.Compose(tf)

def make_loader(
    root,
    split,
    batch,
    workers,
    poisson_intensity=0.0,
    gaussian_intensity=0.0,
    img_size=224,
    shuffle=False,
    seed=12345,
):
    """
    Builds a DataLoader using ChestXrayDataset.
    We seed each worker for reproducible NumPy draws (Poisson noise).
    """
    ds = ChestXrayDataset(
        root_dir=root,
        split=split,
        transform=default_transforms(img_size, is_train=(split == "train")),
        poisson_intensity=poisson_intensity,
        gaussian_intensity=gaussian_intensity,
    )

    g = torch.Generator()
    g.manual_seed(seed)  # controls shuffling order
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )


# subcommands

def cmd_train(args):
    seed_everything(args.seed)
    os.makedirs(args.ckptdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    # clean train/val loaders (no noise)
    train_loader = make_loader(
        args.data_root, "train", args.batch_size, args.num_workers,
        poisson_intensity=0.0, gaussian_intensity=0.0,
        img_size=args.img_size, shuffle=True, seed=args.seed
    )
    val_loader = make_loader(
        args.data_root, "val", args.batch_size, args.num_workers,
        poisson_intensity=0.0, gaussian_intensity=0.0,
        img_size=args.img_size, shuffle=False, seed=args.seed
    )

    # model (select the backbone via --model)
    model = ClassificationTask(backbone=args.model, lr=args.lr)

    ckpt_cb = ModelCheckpoint(
        dirpath=args.ckptdir,
        filename=f"{args.model}-" + "{epoch:02d}-{val_auroc:.3f}",
        monitor="val_auroc", mode="max", save_top_k=1
    )
    es_cb = EarlyStopping(monitor="val_auroc", mode="max", patience=args.patience)
    logger = TensorBoardLogger(save_dir=args.logdir, name=args.model)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[ckpt_cb, es_cb],
        logger=logger,
        precision="32",
        deterministic=True,
        devices=args.devices,
        accelerator=args.accelerator,
        limit_train_batches=args.limit_train_batches
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Best checkpoint:", ckpt_cb.best_model_path)


def _pretty_print_one_row(row: dict, keys: list):
    colw = {k: max(len(k), len(f"{row.get(k,'')}")) for k in keys}
    def line(sep="+", fill="-"):
        return sep + sep.join(fill*(colw[k]+2) for k in keys) + sep
    def rowfmt(r):
        cells = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            cells.append(f" {v:>{colw[k]}} ")
        return "|" + "|".join(cells) + "|"
    print("\nResult:")
    print(line())
    print(rowfmt({k:k for k in keys}))
    print(line())
    print(rowfmt(row))
    print(line())

def cmd_test(args):
    seed_everything(args.seed)


    if args.model is None:
        model = ClassificationTask.load_from_checkpoint(args.model_ckpt)
    else:
        model = ClassificationTask(backbone=args.model, lr=1e-3)
    model.eval()

    if args.accelerator == "mps":
        device = torch.device("mps")
    elif args.accelerator == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    sev_p = float(args.poiss)   # Poisson
    sev_g = float(args.gauss)   # Gaussian

    loader = make_loader(
        args.data_root, "test", args.batch_size, args.num_workers,
        poisson_intensity=sev_p, gaussian_intensity=sev_g,
        img_size=args.img_size, shuffle=False, seed=args.seed
    )
    tag = "clean" if sev_p == 0.0 and sev_g == 0.0 else f"poisson_{sev_p:g}_gaussian_{sev_g:g}"
    print(f"\n==> Testing panel: {tag}")

    auroc     = BinaryAUROC()
    auprc     = BinaryAveragePrecision()
    prec_m    = BinaryPrecision()
    rec_m     = BinaryRecall()
    f1_m      = BinaryF1Score()

    tp = fp = tn = fn = 0
    rows = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(torch.int64)

            logits = model(x)
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits)

            probs_cpu = probs.cpu()
            y_cpu = y.cpu()

            auroc.update(probs_cpu, y_cpu)
            auprc.update(probs_cpu, y_cpu)
            prec_m.update(probs_cpu, y_cpu)
            rec_m.update(probs_cpu, y_cpu)
            f1_m.update(probs_cpu, y_cpu)

            preds = (probs_cpu >= 0.5).to(torch.int64)
            tp += int(((preds == 1) & (y_cpu == 1)).sum())
            fp += int(((preds == 1) & (y_cpu == 0)).sum())
            tn += int(((preds == 0) & (y_cpu == 0)).sum())
            fn += int(((preds == 0) & (y_cpu == 1)).sum())

            for p, label in zip(probs_cpu.tolist(), y_cpu.tolist()):
                rows.append([p, label])

    prevalence = (tp + fn) / (tp + fp + tn + fn)
    res = {
        "panel": tag,
        "prevalence_y1": prevalence,
        "test_auroc": float(auroc.compute().item()),
        "test_auprc": float(auprc.compute().item()),
        "test_precision": float(prec_m.compute().item()),
        "test_recall": float(rec_m.compute().item()),
        "test_f1": float(f1_m.compute().item()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    print("\nResult:")
    print(res)

    # save csv of results
    csv_dir = os.path.join(os.path.dirname(args.model_ckpt), "eval")
    os.makedirs(csv_dir, exist_ok=True)
    ck_base = os.path.basename(args.model_ckpt).replace(".ckpt", "")
    model_name = ck_base.split("-")[0]

    gauss_str = str(int(args.gauss))
    poiss_str = str(int(args.poiss))

    csv_path = os.path.join(csv_dir, f"{model_name}-gauss{gauss_str}-poiss{poiss_str}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(res.keys())
        writer.writerow(res.values())

    print(f"Saved metrics to {csv_path}")




# CLI args

def build_parser():
    p = argparse.ArgumentParser(description="Chest X-ray robustness experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train args
    pt = sub.add_parser("train", help="Train a model on clean data")
    pt.add_argument("--data_root", required=True, help="Folder with train/, val/, test/ subdirs")
    pt.add_argument("--model", default="resnet18", choices=["resnet18","densenet121","custom"])
    pt.add_argument("--batch_size", type=int, default=32)
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--img_size", type=int, default=224)
    pt.add_argument("--num_workers", type=int, default=8)
    pt.add_argument("--ckptdir", default="checkpoints")
    pt.add_argument("--logdir", default="runs")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--devices", default=1, type=int)
    pt.add_argument("--accelerator", default="gpu")  # or "cpu"
    pt.add_argument("--patience", type=int, default=5)
    pt.add_argument("--limit_train_batches", type=float, default=1.0)
    pt.add_argument("--labels_csv", default="labels.csv", help="Path to CSV file with image paths and labels")

    # test args
    pe = sub.add_parser("test", help="Evaluate a checkpoint on ONE noise severity")
    pe.add_argument("--data_root", required=True, help="Folder with train/, val/, test/ subdirs")
    pe.add_argument("--model_ckpt", required=True, help="Path to .ckpt")
    pe.add_argument("--gauss", type=int, default=0, help="gaussian noise")
    pe.add_argument("--poiss", type=int, default=0, help="poisson noise")
    pe.add_argument("--model", default=None, choices=[None,"resnet18","densenet121","custom"])
    pe.add_argument("--batch_size", type=int, default=32)
    pe.add_argument("--img_size", type=int, default=224)
    pe.add_argument("--num_workers", type=int, default=8)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--devices", default=1, type=int)
    pe.add_argument("--accelerator", default="gpu")
    pe.add_argument("--labels_csv", default="labels.csv", help="Path to CSV file with image paths and labels")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "test":
        cmd_test(args)
    else:
        raise ValueError(f"Unknown command {args.cmd}")

if __name__ == "__main__":
    main()
