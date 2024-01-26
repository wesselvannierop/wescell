import pytorch_lightning as pl


class TrainValDataModule(pl.LightningDataModule):
    def __init__(
        self, train_datamodule: pl.LightningDataModule, val_datamodule: pl.LightningDataModule
    ):
        """Wrapper class to allow for training on a different dataset.

        Args:
            train_datamodule (pl.LightningDataModule): Initialized train datamodule (pytorch_lightning)
            val_datamodule (pl.LightningDataModule): Initialized val datamodule (pytorch_lightning)
        """
        super().__init__()
        self.train_datamodule = train_datamodule
        self.val_datamodule = val_datamodule

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_datamodule.setup(stage)
        if stage == "validate":
            self.val_datamodule.setup(stage)

    def class_id_to_name(self, i: int):
        return self.val_datamodule.class_id_to_name(i)

    def train_dataloader(self):
        return self.train_datamodule.train_dataloader()

    def val_dataloader(self):
        self.val_datamodule.setup("validate")
        return self.val_datamodule.val_dataloader()
