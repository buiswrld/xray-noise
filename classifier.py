import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from models import build_model

class ClassificationTask(pl.LightningModule):
    def __init__(self, backbone="resnet18", lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(backbone, in_channels=1)
        self.lr = lr
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()

    def forward(self, x):
        return self.model(x).squeeze(1)  # logits [B]

    def _step(self, batch, stage):
        x, y = batch
        y = y.float()
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        probs = torch.sigmoid(logits)
        self.log(f"{stage}_loss", loss, prog_bar=(stage!="train"), on_step=False, on_epoch=True)
        self.log(f"{stage}_auroc", self.auroc(probs, y), prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_auprc", self.auprc(probs, y), prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}_precision", self.prec(probs, y), on_epoch=True)
        self.log(f"{stage}_recall",    self.rec(probs, y),  on_epoch=True)
        self.log(f"{stage}_f1",        self.f1(probs, y),   on_epoch=True)
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _): return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
