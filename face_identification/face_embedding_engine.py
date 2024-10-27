import os
import sys
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

# add current working directory + parent to path
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from datasets.ORL_dataset import ORLDataset


class FaceEmbeddingEngine:
    def __call__(self, input_image):
        if isinstance(input_image, list):
            return [self.__call__(i) for i in input_image]

        if isinstance(input_image, str):
            input_image = np.array(Image.open(input_image))

        return self.extract_face_embeddings(input_image)

    # All face embedding engines should implement this method:
    # def extract_face_embeddings(self, image: np.ndarray):
    #     ...


class ResnetEmbeddingEngine(FaceEmbeddingEngine):
    def __init__(self, min_face_size: int = 160, model_name: str = 'vggface2'):
        # so far tested models: 'vggface2', 'casia-webface'
        self.model = InceptionResnetV1(pretrained=model_name).eval()

        self.min_face_size = min_face_size

    def extract_face_embeddings(self, image: np.ndarray):
        if len(image.shape) == 2:
            # convert grayscale image to 3 channel torch tensor with batch dimension
            image = np.stack([image, image, image], axis=2)

        image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0)

        # resize image to min_face_size if needed
        if image.shape[2] < self.min_face_size or image.shape[3] < self.min_face_size:
            image = torch.nn.functional.interpolate(image, size=(160, 160), mode='bilinear', align_corners=False)

        return self.model(image).detach().cpu().numpy()[0]
