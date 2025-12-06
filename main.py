import argparse, os, random, csv
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from classifier import ClassificationTask
from dataset import ChestXrayDataset


#utils

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

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

    def seed_worker(worker_id):
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

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
        precision="16-mixed",
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

    # load model from checkpoint
    if args.model is None:
        model = ClassificationTask.load_from_checkpoint(args.model_ckpt)
    else:
        model = ClassificationTask(backbone=args.model, lr=1e-3)

    trainer = pl.Trainer(
        logger=False, enable_checkpointing=False,
        devices=args.devices, accelerator=args.accelerator,
        precision="16-mixed", deterministic=True
    )

    # build test loader with specified noise severity
    sev = float(args.noise)
    loader = make_loader(
        args.data_root, "test", args.batch_size, args.num_workers,
        poisson_intensity=sev, gaussian_intensity=0.0,
        img_size=args.img_size, shuffle=False, seed=args.seed
    )
    tag = "clean" if sev == 0.0 else f"poisson_{sev:g}"
    print(f"\n==> Testing panel: {tag}")

    out = trainer.test(model, dataloaders=loader, ckpt_path=args.model_ckpt, verbose=False)
    metrics = out[0] if isinstance(out, list) else out
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["panel"] = tag

    # Choose the columns that likely exist from ClassificationTask logs
    keys = ["panel"]
    for k in ["test_auroc", "test_auprc", "test_loss", "test_precision", "test_recall", "test_f1"]:
        if k in metrics:
            keys.append(k)

    _pretty_print_one_row(metrics, keys)

    # save to CSV next to the model checkpoint
    csv_dir = os.path.join(os.path.dirname(args.model_ckpt), "eval")
    os.makedirs(csv_dir, exist_ok=True)
    ck = os.path.basename(args.model_ckpt).replace(".ckpt","")
    csv_path = os.path.join(csv_dir, f"metrics_{ck}_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        r_out = {k: (round(metrics[k], 6) if isinstance(metrics.get(k), float) else metrics.get(k)) for k in keys}
        w.writerow(r_out)
    print(f"Saved: {csv_path}")


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
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--devices", default=1, type=int)
    pt.add_argument("--accelerator", default="gpu")  # or "cpu"
    pt.add_argument("--patience", type=int, default=5)
    pt.add_argument("--limit_train_batches", type=float, default=1.0)

    # test args
    pe = sub.add_parser("test", help="Evaluate a checkpoint on ONE noise severity")
    pe.add_argument("--data_root", required=True, help="Folder with train/, val/, test/ subdirs")
    pe.add_argument("--model_ckpt", required=True, help="Path to .ckpt")
    pe.add_argument("--noise", type=float, required=True, help="Poisson severity (0.0 = clean)")
    pe.add_argument("--model", default=None, choices=[None,"resnet18","densenet121","custom"])
    pe.add_argument("--batch_size", type=int, default=32)
    pe.add_argument("--img_size", type=int, default=224)
    pe.add_argument("--num_workers", type=int, default=8)
    pe.add_argument("--seed", type=int, default=42)
    pe.add_argument("--devices", default=1, type=int)
    pe.add_argument("--accelerator", default="gpu")

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
