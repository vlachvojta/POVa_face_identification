import os
import csv
from enum import Enum

from PIL import Image
import cv2
import numpy as np

from datasets.data_structure import ImageData
from datasets.data_parser import DataParser
from datasets.image_preprocessor import ImagePreProcessor

class Partition(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class DataLoader:
    def __init__(self, data_path, partition=Partition.TRAIN, filter_class = 0, filter_attributes = [],
                 sequential_classes: bool = False, limit: int = None, balance_subset: bool = False,
                 image_preprocessor: ImagePreProcessor = None):
        self.data_path = data_path
        self.image_preprocessor = image_preprocessor

        if not os.path.exists(f"{self.data_path}/annotations.csv"):
            DataParser.parse(self.data_path)

        with open(f"{self.data_path}/annotations.csv", "r") as csvfile:
            # Load from CSV
            reader = csv.DictReader(csvfile)
            self.data = [ImageData(**row) for row in reader]
            
            # Filter by partition
            self.data = [image for image in self.data if image.partition == partition.value]

        if filter_class != 0:
            # Classes begin from 1
            self.data = [image for image in self.data if image.id == filter_class]

        if filter_attributes:
            # Filter by attributes (check if filter_attributes is a subset of image attributes)
            self.data = [image for image in self.data if set([attr.name for attr in filter_attributes]) <= set(image.attributes())]

        if sequential_classes:
            self.reorder_classes()

        if balance_subset:
            self.data = self.balance_subset(limit)
        else:
            # create unbalanced subset by selecting first `limit` elements
            if limit and len(self.data) > limit:
                self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Open image and return the whole object
        self.data[index].image = Image.open(f"{self.data_path}/Img/img_align_celeba/{self.data[index].filename}")

        if self.image_preprocessor:
            save_image = self.data[index].id == 42  # save debug image
            image = np.array(self.data[index].image)
            image = self.image_preprocessor(image, save_image=save_image, image_src=self.data[index].filename)
            self.data[index].image = image

        return self.data[index]

    def unique_classes(self):
        return set(item.id for item in self.data)

    def reorder_classes(self):
        """Reorder classes to be sequential starting from 0 to len(unique_classes)."""
        unique_classes = sorted(self.unique_classes())
        class_mapping = {class_id: i for i, class_id in enumerate(unique_classes)}
        for item in self.data:
            item.id = class_mapping[item.id]

    def balance_subset(self, limit: int = None) -> list:
        """Assure classes have more than one sample in the dataset by creating a subset of the data.
        1) Use the first quarter of the data with whatever is available.
        2) Add the rest of the data, but only if the class is not already present in the subset.
        """
        if limit is not None:
            max_subset_len = limit
        else:
            max_subset_len = len(self.data)

        # get unique classes from the first quarter of the data
        data = self.data[:max_subset_len // 4]
        unique_classes = set(item.id for item in data)
        data_len = len(data)

        # add the rest of the data
        for item in self.data[max_subset_len // 4:]:
            if item.id in unique_classes:
                data.append(item)
                data_len += 1
                unique_classes.add(item.id)

                if data_len >= max_subset_len:
                    break

        return data


class DataLoaderTorchWrapper(DataLoader):
    """Wrapper for DataLoader to allow automatic pytorch batching."""
    def __getitem__(self, index):
        item = super().__getitem__(index)

        image = np.array(item.image)
        # orig_image = image.copy()
        assert image.ndim == 3, f"Image should have 3 dimensions (H, W, C) or (C, H, W), got {image.ndim} dimensions with shape {image.shape}"

        if image.shape[2] == 3:  # Engines need images in RGB format [3, H, W]
            image = image.transpose(2, 0, 1)

        return {
            "image": image,
            "class": item.id,
        }
