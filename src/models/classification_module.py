from typing import Dict, Union

import pytorch_lightning as pl
import torch
from torch import nn, optim
from sklearn.metrics import (
        recall_score,
        precision_score,
        accuracy_score,
        f1_score,
        roc_auc_score
    )


class ClassificationLightningModule(pl.LightningModule):

    """
    Pytorch Lightning Module for ResNet18.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Union[nn.Module],
        optimizer: Union[optim.SGD, optim.Adam],
        optimizer_params: Dict,
        scheduler: Union[
            optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.ReduceLROnPlateau
        ],
        scheduler_params: Dict,
        scheduler_interval: str,
    ):
        """
        Initialize TableNet Module.
        Args:
            model
            loss
            optimizer
            optimizer_params
            scheduler
            scheduler_params
            scheduler_interval
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval

    def forward(self, batch):
        """
        Perform forward-pass.
        Args:
            batch (tensor): Batch of images to perform forward-pass.
        Returns (Tuple[tensor, tensor]): Table, Column prediction.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        images, labels, __ = batch

        images = images.to(device)
        labels = labels.to(device)

        output = self.forward(images)

        output = output.to(device)

        target = labels.long()
        target = target.to(device)

        targets_one_hot = torch.zeros(target.shape[0], 2)
        targets_one_hot = targets_one_hot.to(device)
        targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = self.loss(output, targets_one_hot)

        target = target.to("cpu")
        output = output.to("cpu")

        threshold = 0.5
        predicted_labels = (output >= threshold).long()  # Convert probabilities to binary predictions
        predictions = torch.argmax(predicted_labels, dim=1)

        auc = roc_auc_score(target, predictions)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_auc", auc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        images, labels, __ = batch

        images = images.to(device)
        labels = labels.to(device)

        output = self.forward(images)

        output = output.to(device)
        target = labels.long()

        target = target.to(device)

        targets_one_hot = torch.zeros(target.shape[0], 2)
        targets_one_hot = targets_one_hot.to(device)
        targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = self.loss(output, targets_one_hot)

        target = target.to("cpu")
        output = output.to("cpu")

        threshold = 0.5
        predicted_labels = (output >= threshold).long()  # Convert probabilities to binary predictions
        predictions = torch.argmax(predicted_labels, dim=1)

        auc = roc_auc_score(target, predictions)

        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_auc", auc, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        images, labels, __ = batch

        images = images.to(device)
        labels = labels.to(device)

        output = self.forward(images)

        output = output.to(device)

        target = labels.long()

        target = target.to(device)

        targets_one_hot = torch.zeros(target.shape[0], 2)
        targets_one_hot = targets_one_hot.to(device)
        targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = self.loss(output, targets_one_hot)

        target = target.to("cpu")
        output = output.to("cpu")

        threshold = 0.5
        predicted_labels = (output >= threshold).long()  # Convert probabilities to binary predictions
        predictions = torch.argmax(predicted_labels, dim=1)

        precision = precision_score(target, predictions, zero_division=0)
        recall = recall_score(target, predictions)
        accuracy = accuracy_score(target, predictions)
        f1 = f1_score(target, predictions)
        auc = roc_auc_score(target, predictions)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_precision", precision, on_epoch=True)
        self.log("test_recall", recall, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)
        self.log("test_f1_score", f1, on_epoch=True)
        self.log("test_accuracy", auc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configure optimizer for pytorch lighting.
        Returns: optimizer and scheduler for pytorch lighting.
        """
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.scheduler(optimizer, **self.scheduler_params)
        scheduler = {
            "scheduler": scheduler,
            "monitor": "validation_loss",
            "interval": self.scheduler_interval,
        }

        return [optimizer], [scheduler]
