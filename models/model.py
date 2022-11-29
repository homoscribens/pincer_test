import torch

from transformers import AutoModelForSequenceClassification

import pytorch_lightning as pl

class SentimentClassifier(pl.LightningModule):
    def __init__(self, model_name, lr=1e-5):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.lr = lr

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
