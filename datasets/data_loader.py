import os
import csv
from enum import Enum
from PIL import Image

from datasets.data_structure import ImageData
from datasets.data_parser import DataParser


class Partition(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class DataLoader:
    def __init__(self, data_path, partition=Partition.TRAIN):
        self.data_path = data_path
        
        if not os.path.exists(f"{self.data_path}/annotations.csv"):
            DataParser.parse(self.data_path)
            
        with open(f"{self.data_path}/annotations.csv", "r") as csvfile:
            # Load from CSV
            reader = csv.DictReader(csvfile)
            self.data = [ImageData(**row) for row in reader]
            
            # Filter by partition
            self.data = [image for image in self.data if image.partition == partition.value]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Open image and return the whole object
        self.data[index].image = Image.open(f"{self.data_path}/Img/img_align_celeba/{self.data[index].filename}")
        return self.data[index]
    