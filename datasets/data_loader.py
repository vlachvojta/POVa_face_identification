import os
import csv
from enum import Enum

from PIL import Image
import cv2
import numpy as np

from datasets.data_structure import ImageData
from datasets.data_parser import DataParser


class Partition(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class DataLoader:
    def __init__(self, data_path, partition=Partition.TRAIN, filter_class = 0, filter_attributes = [],
                 sequential_classes: bool = False, limit: int = None, balance_subset: bool = False):
        self.data_path = data_path

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

class Normalization(Enum):
    MEAN_STD = "mean_std"
    MIN_MAX = "min_max"
    _0_1 = "0_1"
    _1_1 = "1_1"

class Squarify(Enum):
    CROP = "crop"
    PAD = "pad"
    AROUND_FACE = "around_face"

class DataLoaderTorchWrapper(DataLoader):
    """Wrapper for DataLoader to allow automatic pytorch batching."""
    def __init__(self, data_path, face_detection_engine=None, squarify: Squarify = None,
                 normalize: Normalization = None, resize: int = None, *args, **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.face_detection_engine = face_detection_engine
        self.squarify = squarify
        self.normalize = normalize
        self.resize = resize

    def __getitem__(self, index):
        item = super().__getitem__(index)

        image = np.array(item.image)
        orig_image = image.copy()
        assert image.ndim == 3, f"Image should have 3 dimensions (H, W, C) or (C, H, W), got {image.ndim} dimensions with shape {image.shape}"

        # # Crop face from image
        # if self.face_detection_engine:
        #     faces, probs = self.face_detection_engine.crop_faces(image)
        #     if len(faces) > 1:
        #         # pick the face with the highest probability
        #         image = faces[np.argmax(probs)]
        #     elif len(faces) == 1:
        #         image = faces[0]

        #     if image.shape[0] == 0 or image.shape[1] == 0:
        #         print(f"Skipping face detection on image {item.filename} as it has zero width or height after face detection. shape: {image.shape}")
        #         image = orig_image

        if self.squarify:
            if self.squarify == Squarify.PAD:
                # pad the image to make it square
                h, w = image.shape[:2]
                if h > w:
                    pad = (h - w) // 2
                    image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
                elif w > h:
                    pad = (w - h) // 2
                    image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
            elif self.squarify == Squarify.AROUND_FACE:
                # detect face box and crop the image around the face
                assert self.face_detection_engine, "Face detection engine must be provided to squarify around face."
                result = self.face_detection_engine(image)[0]
                if result:
                    face = result['box']
                    l, t, r, b = face
                    l, t, r, b = int(l), int(t), int(r), int(b)
                    # crop face and pad to make it square
                    if b - t > r - l:
                        pad = (b - t - (r - l)) // 2
                        l = max(0, l - pad)
                        r = min(image.shape[1], r + pad)
                    elif r - l > b - t:
                        pad = (r - l - (b - t)) // 2
                        t = max(0, t - pad)
                        b = min(image.shape[0], b + pad)

                    t = max(0, t)
                    b = min(image.shape[0], b)
                    l = max(0, l)
                    r = min(image.shape[1], r)
                    # complete squarify if needed
                    h, w = b - t, r - l
                    if h > w:
                        pad = (h - w) // 2
                        l = max(0, l - pad)
                        r = min(image.shape[1], r + pad)
                    elif w > h:
                        pad = (w - h) // 2
                        t = max(0, t - pad)
                        b = min(image.shape[0], b + pad)
                    image = image[t:b, l:r]

                else:
                    print(f"Skipping squarify around face on image {item.filename} as no face was detected.")
                    image = orig_image

        if self.resize:
            image = cv2.resize(image, (self.resize, self.resize))

        if self.normalize:
            if self.normalize == Normalization.MEAN_STD:
                image = (image - 127.5) / 128.0
            elif self.normalize == Normalization.MIN_MAX:
                image = (image - image.min()) / (image.max() - image.min())
            elif self.normalize == Normalization._0_1:
                image = image / 255.0
            elif self.normalize == Normalization._1_1:
                image = image / 127.5 - 1.0
            else:
                raise ValueError(f"Unknown normalization method: {self.normalize}")

        if image.shape[2] == 3:  # Engines need images in RGB format [3, H, W]
            image = image.transpose(2, 0, 1)

        return {
            "image": image,
            "class": item.id,
            # "attributes": item.attributes()
        }
