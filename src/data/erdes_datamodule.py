from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
#from torchvision.transforms import transforms
#from src.data.components.utils import resize
from src.data.components.erdes_dataset import VideoDataset


class ERDESDataModule(LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        size: Tuple[int, int, int],
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.size = size
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        #self.transforms = resize(size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        pass  # No downloading necessary

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train:
            self.data_train = VideoDataset(csv_path=self.train_csv, size=self.size)

        if not self.data_val:
            self.data_val = VideoDataset(csv_path=self.val_csv, size=self.size)

        if not self.data_test:
            self.data_test = VideoDataset(csv_path=self.test_csv, size=self.size)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    datamodule = ERDESDataModule(
        train_csv="train.csv",
        val_csv="val.csv",
        test_csv="test.csv",
        size=(128, 128, 128),
        batch_size=4,
        num_workers=2,
        pin_memory=True,
)