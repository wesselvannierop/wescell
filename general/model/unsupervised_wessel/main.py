from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
import torch
from torchvision.utils import save_image

from general.data.data_pair_creation import DataTransformPair
from general.metrics.metrics import CellConfluencyMetrics
from general.model.iic_loss import IICloss
from general.model.unet.unet_decoder import UNetDecoderWithSoftmax

# Local imports
from general.model.unet.unet_encoder import UNetEncoder
from general.utils import get_confusion_matrix_overlaid_mask, remap


class UnsupervisedWesCell(pl.LightningModule):
    def __init__(
        self,
        nr_of_clusters: int = 2,
        consider_neighbouring_pixels: int = 1,
        entropy_coeff: float = 1.0,
        optimizer: str = "adamw",
        lr: float = 5e-4,
        save_folder: str = None,
    ) -> None:
        """The unsupervised model that was proposed by Wessel van Nierop during his MSc Thesis
        project implemented using PyTorch Lightning. Read the thesis for details.

        Args:
            nr_of_clusters (int, optional): Number of classes the model should try to identify. Defaults to 2.
            consider_neighbouring_pixels (int, optional): How many neighbouring pixels the model should optimize for. Defaults to 1.
            entropy_coeff (float, optional): How much weight to add to maximizing marginal entropy (equal class predictions). Defaults to 1.0.
            optimizer (str, optional): String with the name of the optimizer. Defaults to "adamw".
            lr (float, optional): Learning rate of the optimizer. Defaults to 5e-4.
            save_folder (str, optional): If set to a folder path, during validation this model will save its predictions there. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters()
        assert (
            nr_of_clusters == 2
        ), "Validation for more clusters not implemented, needs Hungarian matching"
        self.nr_of_clusters = nr_of_clusters
        self.consider_neighbouring_pixels = consider_neighbouring_pixels
        self.entropy_coeff = entropy_coeff
        self.optimizer = optimizer
        self.lr = lr

        self.backbone = UNetEncoder()
        self.head = UNetDecoderWithSoftmax(self.backbone.embed_dims, n_classes=self.nr_of_clusters)
        self.metrics = CellConfluencyMetrics()

        self.inv = DataTransformPair.inv
        self.iic_loss = IICloss(
            consider_neighbouring_pixels=consider_neighbouring_pixels,
            entropy_coeff=entropy_coeff,
            nr_of_clusters=self.nr_of_clusters,
        )

        if save_folder is not None:
            self.save_folder = Path(save_folder)
            self.save_folder.mkdir(exist_ok=True)

    def training_step(self, batch, _):
        # Extract data
        (transformed0, transformed1), (transform0, transform1), name = batch

        # Forward through network, and invert geometric transformations
        classes0 = self.inv(self(transformed0), transform0)
        classes1 = self.inv(self(transformed1), transform1)

        # Mask black, padded areas and compute loss (max mutual information)
        mask = (classes0 != 0).logical_and(classes1 != 0)
        loss, marginal_entropy, conditional_entropy = self.iic_loss(
            classes0 * mask, classes1 * mask
        )

        # Log to tensorboard
        self.log("train_loss", loss)
        self.log("Marginal Entropy", marginal_entropy)
        self.log("Conditional Entropy", conditional_entropy)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target, names = batch
        pred = self(image)

        binary_pred = torch.argmax(pred, dim=1, keepdim=True)
        self.metrics.update(binary_pred, target)

        # Log pred to tensorboard
        if batch_idx == 0:
            self.validation_step_outputs = (binary_pred[0, 0], target[0, 0])

        # Save preds if save_folder is set
        if self.save_folder is not None:
            for binary_pred_, name in zip(binary_pred, names):
                save_image(binary_pred_.float(), str(self.save_folder / f"{name}_predicted.png"))

    def predict_step(self, batch, *_):
        assert self.save_folder is not None, "Predict step only works when save_folder is set."

        image, names = batch
        pred = self(image)
        pred = torch.argmax(pred, dim=1, keepdim=True)
        for pred_, name in zip(pred, names):
            save_image(pred_.float(), str(self.save_folder / f"{name}_predicted.png"))

    def on_validation_epoch_end(self):
        # Compute metrics, log to tensorboard and reset the metrics for reuse
        report, match = self.metrics.compute(return_match=True)
        logs = self.metrics.logger_format(report, log_group="ValMetrics")
        for log in logs:
            self.log(**log)
        self.metrics.reset()

        # Also print the metrics
        print("\n\n")
        pprint(report, sort_dicts=False)

        # Invert pred when the model learns inverted classes compared to the target
        for map_from, map_to in match.items():
            self.invert_pred = map_from != map_to

        # Log pred and confusion visualization to tensorboard
        if self.logger is not None:
            pred, target = self.validation_step_outputs
            self.logger.experiment.add_image(
                "clusters",
                pred / (self.nr_of_clusters - 1),
                dataformats="HW",
                global_step=self.trainer.global_step,
            )

            if match is not None:
                pred = remap(pred, match)
            confusion = get_confusion_matrix_overlaid_mask(target.cpu().numpy(), pred.cpu().numpy())
            self.logger.experiment.add_image(
                "confusion",
                confusion / 255,
                dataformats="HWC",
                global_step=self.trainer.global_step,
            )
            self.logger.experiment.flush()

    test_step = validation_step
    on_test_epoch_end = on_validation_epoch_end

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer {self.optim} not supported")

    def forward(self, image):
        embeddings = self.backbone(image)
        return self.head(*embeddings)

    def output_to_pred(self, pred):
        return torch.argmax(pred, dim=1, keepdim=True)

    def predict(self, image):
        return self.output_to_pred(self(image))
