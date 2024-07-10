from typing import Any, Callable, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from torchvision import transforms as transform_lib
from torchvision.datasets import CocoDetection
from pathlib import Path


class COCODataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
            ]
        )
        self.target_transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
            ]
        )


        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        data_dir = Path(self.data_dir)
        train_annotations = data_dir / "annotations" / "instances_train2017.json"
        val_annotations = data_dir / "annotations" / "instances_val2017.json"
        test_annotations = data_dir / "annotations" / "instances_test2017.json" 

        if not self.data_train and train_annotations.exists():
            self.data_train = CocoDetection(
                root=data_dir,
                annFile=train_annotations,
                transform=self.transforms,
            )
        if not self.data_val and val_annotations.exists():
            self.data_val = CocoDetection(
                root=data_dir,
                annFile=val_annotations,
                transform=self.transforms,
            )
        if not self.data_test and test_annotations.exists():
            self.data_test = CocoDetection(
                root=data_dir,
                annFile=test_annotations,
                transform=self.transforms,
            )

    
        

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        assert self.data_train is not None, "Data not loaded yet."
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        assert self.data_val is not None, "Data not loaded yet."
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        assert self.data_test is not None, "Data not loaded yet."
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = MNISTDataModule()
