import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import ConfusionMatrixDisplay
import wandb
import numpy as np

import io
import PIL.Image
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import inspect

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)

parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import configuration as config

class ImageClassificationLightningModule(LightningModule):
    """
    LightningModule implementation for image classification tasks.

    Args:
        model: Model instance representing the Convolution Network Model.
        loss_fn: Loss function used for training.
        metrics: Metrics used for evaluation.
        vectorized_metrics: Metrics that return a vector as a result of the computing.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
        weight_decay (float): Weight decay for the optimizer.

    Attributes:
        model: Model instance representing the Convolution Network Model.
        loss_fn: Loss function used for training.
        train_metrics: Metrics used for training evaluation.
        val_metrics: Metrics used for validation evaluation.
        test_metrics: Metrics used for testing evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
    """

    def __init__(
        self,
        model,
        loss_fn,
        metrics,
        vectorized_metrics,
        lr,
        scheduler_max_it,
        weight_decay=0,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

        # General stats
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Per class stats
        self.train_vect_metrics = vectorized_metrics.clone(prefix="train/")
        self.val_vect_metrics = vectorized_metrics.clone(prefix="val/")
        self.test_vect_metrics = vectorized_metrics.clone()

        # Parameters for the module
        self.scheduler_max_it = scheduler_max_it
        self.weight_decay = weight_decay
        self.lr = lr

        # Aux parameters for the vectorial stats (per class stats)
        self.all_preds = []
        self.all_targets = []
        self.test_vect_metrics_result = {}

    def forward(self, X):
        """
        Forward pass of the Convolutional model.

        Args:
            X: Input tensor.

        Returns:
            Output tensor.
        """
        outputs = self.model(X)
        return outputs

    # One db online visualizer to see training?
    def _plot_cm(self, y_true, pred):
        """
        Plot a confusion matrix on the log.

        Args:
            y_true: numpy array with the true labels.
            pred: numpy array with the predicted labels.
        """
        wandb.log(
            {
                "train/conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=pred,
                    class_names= config.CLASS_NAMES
                ),
            }
        )

    def _common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Tuple containing the ground truth labels, predicted outputs, and loss value.
        """
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        return y, y_hat, loss

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        y, y_hat, loss = self._common_step(batch, batch_idx)

        # Refine value for prediction for metrics
        y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)

        self.train_metrics.update(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "train/labels": y, "train/predictions": y_hat}

    def on_train_epoch_end(self):
        """
        Callback function called at the end of each training epoch.
        Computes and logs the training metrics.
        """
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        y, y_hat, loss = self._common_step(batch, batch_idx)

        y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        self.val_metrics.update(y_hat, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "val/labels": y, "val/predictions": y_hat}

    def on_validation_epoch_end(self):
        """
        Callback function called at the end of each validation epoch.
        Computes and logs the validation metrics.
        """
        self.log_dict(
            self.val_metrics.compute(), on_step=False, on_epoch=True, prog_bar=True
        )

        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        y, y_hat, loss = self._common_step(batch, batch_idx)

        y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        self.all_targets.append(y)
        self.all_preds.append(y_hat)

        self.test_metrics.update(y_hat, y)
        self.test_vect_metrics.update(y_hat, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "test/labels": y, "test/predictions": y_hat}

    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        all_preds = torch.cat(self.all_preds).cpu().detach().numpy()
        all_targets = torch.cat(self.all_targets).cpu().detach().numpy()

        # Send the plot to one db
        self._plot_cm(all_preds, all_targets)

        self.test_vect_metrics_result = self.test_vect_metrics.compute()
        self.test_vect_metrics.reset()
        self.all_preds = []
        self.all_targets = []

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple containing the optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_max_it)
        return [optimizer], [scheduler]
