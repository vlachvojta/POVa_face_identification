import os
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

# add parent of this file to path to enable importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ORL_dataset import ORLDataset
from face_identification.face_embedding_models import BasicResnet, FacenetPytorchWrapper
from common import utils
from face_identification.train import load_model


class FaceEmbeddingEngine:
    """All face embedding engines should inherit from this class and then:
      - implement extract_face_embeddings method
      - have a __init__ method that initializes the model
      - have self.model attribute (should be a torch model)
    """
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def __call__(self, input_image):
        if isinstance(input_image, list):
            return [self.__call__(i) for i in input_image]

        if isinstance(input_image, str):
            input_image = self.open_image(input_image)

        embedding = self.extract_face_embeddings(input_image)  # all face embedding engines should implement this method

        if self.verbose:
            print(f'{self.__class__.__name__}.{self.model.__class__.__name__}: {input_image.shape} -> {embedding.shape}')

        return embedding

    def open_image(self, image_path: str) -> np.ndarray:
        return np.array(Image.open(image_path))


class FacenetEmbeddingEngine(FaceEmbeddingEngine):
    INPUT_PIXELS = 160
    DIM = 512

    def __init__(self, model_name: str = 'vggface2', device: str = 'cuda', **kwargs):
        super().__init__(**kwargs)
        self.model = InceptionResnetV1(pretrained=model_name).eval().to(device)
        # so far tested models: 'vggface2', 'casia-webface'
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

        if return_first:
            return embedding[0]
        return embedding

class BasicResnetEmbeddingEngine(FaceEmbeddingEngine):
    INPUT_PIXELS = 160
    DIM = 512

    def __init__(self, model_name: str = 'resnet50', device: str = 'cuda', **kwargs):
        super().__init__(**kwargs)
        self.model = BasicResnet(embedding_size=self.DIM).eval().to(device)
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

        if return_first:
            return embedding[0]
        return embedding

class TrainedEmbeddingEngine(FaceEmbeddingEngine):
    INPUT_PIXELS = 160

    def __init__(self, trained_path: str, device: str = 'cuda', **kwargs):
        super().__init__(**kwargs)
        self.model, self.trained_steps = load_model(trained_path, device=device)
        self.model.eval()
        print(f'Loaded model {self.model.__class__.__name__} from {trained_path} trained for {self.trained_steps} steps.')
        self.device = device

        self.DIM = self.model.embedding_size

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

        embedding = self.model(image).detach().cpu().numpy()

        if return_first:
            return embedding[0]
        return embedding


def example_usage_one_image():
    import matplotlib.pyplot as plt
    engine = FacenetEmbeddingEngine()
    dataset = ORLDataset()

    # test the engine with one image
    image = dataset[0][0]
    print(f'image shape: {image.shape}')
    embedding = engine(image)
    print(f'embedding shape: {embedding.shape}')
    
    # show image
    plt.imshow(image.transpose(1, 2, 0))
    plt.show()


def example_usage_batch_images():
    import matplotlib.pyplot as plt
    engine = FacenetEmbeddingEngine()
    dataset = ORLDataset()

    # test the engine with multiple images (batch)
    first_five_images = dataset.images[:5]
    print(f'image batch shape: {first_five_images.shape}')
    embedding = engine(first_five_images)
    print(f'embedding batch shape: {embedding.shape}')

    # show images
    fig, axs = plt.subplots(1, 5, figsize=(15, 6))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(first_five_images[i].transpose(1, 2, 0))
        ax.axis('off')
    plt.show()


if __name__ == '__main__':
    example_usage_one_image()
    example_usage_batch_images()
