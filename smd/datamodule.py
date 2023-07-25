import os
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import numpy.typing as npt
from numalogic.tools.data import TimeseriesDataModule, StreamingDataset
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader


class SMDDataModule(TimeseriesDataModule):
    def __init__(
            self,
            data_dir: str,
            group_id: int = 1,
            sub_id: int = 1,
            seq_len: int = 12,
            batch_size: int = 64,
            preproc_transforms: Optional[list] = None,
            split_ratios: Sequence[float] = (0.5, 0.2, 0.3),
            *args,
            **kwargs,
    ):
        super().__init__(data=None, seq_len=seq_len, batch_size=batch_size, *args, **kwargs)

        self.group_id = group_id
        self.sub_id = sub_id

        if len(split_ratios) != 3 or sum(split_ratios) != 1.0:
            raise ValueError("Sum of all the 3 ratios should be 1.0")

        self.split_ratios = split_ratios
        self.data_dir = data_dir
        if preproc_transforms:
            if len(preproc_transforms) > 1:
                self.transforms = make_pipeline(preproc_transforms)
            else:
                self.transforms = preproc_transforms[0]
        else:
            self.transforms = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._train_labels = None
        self._val_labels = None
        self._test_labels = None

    @property
    def val_data(self) -> npt.NDArray[float]:
        return self.val_dataset.data

    @property
    def train_data(self) -> npt.NDArray[float]:
        return self.train_dataset.data

    @property
    def test_data(self) -> npt.NDArray[float]:
        return self.test_dataset.data

    @property
    def val_labels(self) -> npt.NDArray[int]:
        return self._val_labels

    @property
    def train_labels(self) -> npt.NDArray[int]:
        return self._train_labels

    @property
    def test_labels(self) -> npt.NDArray[int]:
        return self._test_labels

    def _preprocess(self, data: npt.NDArray[float], fit=True):
        if self.transforms:
            if fit:
                self.transforms.fit(data)
            return self.transforms.transform(data)
        return data

    def setup(self, stage: str) -> None:
        data, labels = self.read_data()
        val_size = np.floor(self.split_ratios[1] * len(data)).astype(int)
        test_size = np.floor(self.split_ratios[2] * len(data)).astype(int)

        if stage == "fit":
            train_data = data[: -(val_size + test_size)]
            train_data = self._preprocess(train_data, fit=True)
            self._train_labels = labels[: -(val_size + test_size)]

            val_data = data[val_size:test_size]
            val_data = self._preprocess(val_data, fit=False)
            self._val_labels = labels[val_size:test_size]

            self.train_dataset = StreamingDataset(train_data, self.seq_len)
            self.val_dataset = StreamingDataset(val_data, self.seq_len)

            print(f"Train size: {train_data.shape}\nVal size: {val_data.shape}")

        if stage in ("test", "predict"):
            test_data = data[-test_size:]
            test_data = self._preprocess(test_data, fit=False)
            self._test_labels = labels[-test_size:]
            self.test_dataset = StreamingDataset(test_data, self.seq_len)
            print(f"Test size: {test_data.shape}")

    def read_data(self) -> tuple[npt.NDArray[float], npt.NDArray[int]]:
        df = pd.read_csv(
            os.path.join(self.data_dir, "test", f"machine-{self.group_id}-{self.sub_id}.txt")
        )
        label_df = pd.read_csv(
            os.path.join(self.data_dir, "test_label", f"machine-{self.group_id}-{self.sub_id}.txt")
        )
        return df.to_numpy(), label_df.to_numpy(dtype=int)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    dm = SMDDataModule(data_dir="data/", group_id=1, sub_id=1, seq_len=12)
    dm.setup("predict")
