import os
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

# add current working directory + parent to path
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from datasets.ORL_dataset import ORLDataset


class FaceEmbeddingEngine:
    DIM = 42  # Change this to the dimension of the face embeddings in every subclass
    def __call__(self, input_image):
        if isinstance(input_image, list):
            return [self.__call__(i) for i in input_image]

        if isinstance(input_image, str):
            input_image = self.open_image(input_image)

        return self.extract_face_embeddings(input_image)

    def open_image(self, image_path: str) -> np.ndarray:
        return np.array(Image.open(image_path))

    # All face embedding engines should implement this method:
    # def extract_face_embeddings(self, image: np.ndarray):
    #     ...


class ResnetEmbeddingEngine(FaceEmbeddingEngine):
    INPUT_PIXELS = 160
    DIM = 512

    def __init__(self, model_name: str = 'vggface2', device: str = 'cuda'):
        # so far tested models: 'vggface2', 'casia-webface'
        self.model = InceptionResnetV1(pretrained=model_name).eval().to(device)
        self.device = device

    def extract_face_embeddings(self, image: np.ndarray):
        image = torch.tensor(image).float().to(self.device)

        if len(image.shape) == 3:
            assert image.shape[0] == 3, f"Image should have 3 channels in a shape (channels, height, width), got {image.shape[0]} channels in shape {image.shape}"
            image = image.unsqueeze(0)
            return_first = True
        elif len(image.shape) == 4:
            assert image.shape[1] == 3, f"Image should have 3 channels in a shape (batch, channels, height, width), got {image.shape[1]} channels in shape {image.shape}"
            return_first = False
        else:
            assert False, f"Image shape should have three dimensions (channels, height, width) or four (batch, channels, height, width), got {image.shape}"

        # resize image to input_pixels
        image = F.interpolate(image, size=(self.INPUT_PIXELS, self.INPUT_PIXELS), mode='bilinear', align_corners=False)

        embedding = self.model(image).detach().cpu().numpy()
        print(f'embedding image {image.shape} -> {embedding.shape}')
        if return_first:
            return embedding[0]
        return embedding
