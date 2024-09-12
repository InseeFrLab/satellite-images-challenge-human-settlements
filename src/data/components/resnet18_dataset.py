from typing import Optional
import torch
from albumentations import Compose
from torch.utils.data import Dataset


class ResNet18_Dataset(Dataset):
    """
    Custom Dataset class.
    """

    def __init__(
        self,
        X,
        y,
        ids: Optional[dict] = None,
        transforms: Optional[Compose] = None,
    ):
        """
        Constructor.

        Args:
            X: images
            y: labels
            transforms (Compose): List of transforms to apply to the images.
        """
        self.X = X
        self.y = y
        self.ids = ids
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]: Tuple containing the
            image, label, and metadata.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.X[idx]
        img = img.astype(float)

        label = int(self.y[idx])
        label = torch.tensor(label)

        if self.transforms:
            sample = self.transforms(image=img)
            img = sample["image"]
        else:
            img = torch.tensor(img.astype(float))

        img = img.type(torch.float)
        label = label.type(torch.float)
        if self.ids and idx in self.ids.keys():
            id_image = self.ids[idx]
        else:
            id_image = idx

        metadata = {"ID": idx, "id": id_image}

        return img, label, metadata

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.X)
