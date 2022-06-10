from typing import List, Union, Optional, Tuple

import albumentations as album
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import VOCDetection

from haia.constants.types import TYPE_PAIR_SET
from haia.utils.data import normalize_transformation, to_prediction


class VOCDataset(Dataset):
    """VOC Dataset

    Parameters
    ----------
    folder : str
        Path to the folder containing the VOC dataset.
    data_type : str = 'yolo'
        Type of data to be returned. If yolo the data will be returned for yolo
        task, if not as normal pair of image and bounding boxes.
    images_transformations: torch.nn.Sequential, optional = None
        The transforms to be applied to the images.
    """

    def __init__(
        self,
        folder: str,
        data_type: str = "yolo",
        *,
        image_set: str = "train",
        images_transformations: Optional[album.Compose] = None,
    ):
        if data_type not in ["yolo", "bbox"]:
            raise ValueError(
                f"{data_type} is not a valid data type, please use yolo or bbox"
            )
        self.data_type = data_type
        self.data = VOCDetection(root=folder, image_set=image_set)
        self._len = len(self.data)
        self.transforms = images_transformations

    def __len__(self):
        return self._len

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[List[TYPE_PAIR_SET], TYPE_PAIR_SET]:
        if isinstance(idx, int):
            return to_prediction(*self.data[idx], transforms=self.transforms)

        result_samples = []
        for index in range(idx.start or 0, idx.stop or self._len, idx.step or 1):
            image, data = to_prediction(*self.data[index], transforms=self.transforms)
            result_samples.append((image, data))
        return result_samples


def load_datasets(
    data_folder: str, dimensions: int, batch_size: int, training_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """Load the datasets.

    Parameters
    ----------
    data_folder: str
        Folder where the datasets are stored.
    dimensions: int
        Width or height of the image.
    batch_size: int
        Batch size.
    training_ratio: float
        Training ratio of the training dataset.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Datasets
        for the training and validation."""
    normalization = normalize_transformation(dimensions)

    data = VOCDataset(
        data_folder, images_transformations=normalization, image_set="trainval"
    )

    training_length = int(len(data) * training_ratio)
    validation_length = len(data) - training_length
    training, validation = random_split(data, [training_length, validation_length])

    training = DataLoader(training, shuffle=True, batch_size=batch_size)
    validation = DataLoader(validation, shuffle=True, batch_size=batch_size)

    return training, validation
