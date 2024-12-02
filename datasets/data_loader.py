import os
import csv
from enum import Enum
from PIL import Image

import numpy as np

from datasets.data_structure import ImageData
from datasets.data_parser import DataParser


class Partition(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class DataLoader:
    def __init__(self, data_path, partition=Partition.TRAIN, limit: int = None, sequential_classes: bool = False):
        self.data_path = data_path
        
        if not os.path.exists(f"{self.data_path}/annotations.csv"):
            DataParser.parse(self.data_path)
            
        with open(f"{self.data_path}/annotations.csv", "r") as csvfile:
            # Load from CSV
            reader = csv.DictReader(csvfile)
            self.data = [ImageData(**row) for row in reader]
            
            # Filter by partition
            self.data = [image for image in self.data if image.partition == partition.value]

        if limit and len(self.data) > limit:
            self.data = self.data[:limit]

        if sequential_classes:
            self.reorder_classes()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Open image and return the whole object
        self.data[index].image = Image.open(f"{self.data_path}/Img/img_align_celeba/{self.data[index].filename}")
        return self.data[index]

    def unique_classes(self):
        return set(item.id for item in self.data)
    
    def reorder_classes(self):
        """Reorder classes to be sequential starting from 0 to len(unique_classes)."""
        unique_classes = sorted(self.unique_classes())
        class_mapping = {class_id: i for i, class_id in enumerate(unique_classes)}
        for item in self.data:
            item.id = class_mapping[item.id]

class DataLoaderTorchWrapper(DataLoader):
    """Wrapper for DataLoader to allow automatic pytorch batching."""
    def __getitem__(self, index):
        item = super().__getitem__(index)

        image = np.array(item.image)
        assert image.ndim == 3, f"Image should have 3 dimensions, got {image.ndim}"

        if image.shape[2] == 3:  # Engines need images in RGB format [3, H, W]
            image = image.transpose(2, 0, 1)

        return {
            "image": image,
            "class": item.id,
            # "attributes": item.attributes()
        }
