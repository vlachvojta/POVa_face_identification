import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from enum import Enum

# add current working directory + parent to path
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from datasets.ORL_dataset import ORLDataset


class DistanceFunction(Enum):
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine'


class FaceIdentificationEngine:
    def __init__(self, min_face_size: int = 160, known_embeddings: list = None):
        self.mtcnn = MTCNN(keep_all=True)
        # self.model = InceptionResnetV1(pretrained='casia-webface').eval()
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

        self.min_face_size = min_face_size
        self.known_embeddings = known_embeddings

    def extract_face_embeddings(self, image: np.ndarray):
        if len(image.shape) == 2:
            # convert grayscale image to 3 channel torch tensor with batch dimension
            image = np.stack([image, image, image], axis=2)

        image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0)

        if image.shape[2] < self.min_face_size or image.shape[3] < self.min_face_size:
            image = torch.nn.functional.interpolate(image, size=(160, 160), mode='bilinear', align_corners=False)

        return self.model(image).detach().cpu().numpy()[0]

    def identify_face(self, image, known_embeddings, distance_function: DistanceFunction = DistanceFunction.COSINE):
        # Load the image
        print(f'{image.shape = }')
        image_embedding = self.extract_face_embeddings(image)
        print(f'{image_embedding.shape = }')
        # query_embedding = self.extract_face_embeddings(image)

        if image_embedding is None:
            return None, None

        if distance_function == DistanceFunction.EUCLIDEAN:
            # Compute distances using Euclidean distance
            distances = np.linalg.norm(known_embeddings - image_embedding, axis=1)
            closest_match_index = np.argmin(distances)
        elif distance_function == DistanceFunction.COSINE:
            # Compute distances using cosine similarity
            distances = F.cosine_similarity(torch.tensor(known_embeddings), torch.tensor(image_embedding), dim=1)
            closest_match_index = torch.argmax(distances).item()

        return closest_match_index, distances[closest_match_index]

    def process_image_with_load(self, image_path):
        # Load the image
        img = Image.open(image_path)

        # Detect faces
        boxes, _ = self.mtcnn.detect(img)

        if boxes is not None:
            # Align faces
            aligned = self.mtcnn(img)

            # Compute embeddings
            embedding = self.extract_face_embeddings(aligned)
            return embedding

        else:
            print("No faces detected in the image.")
            return None


if __name__ == '__main__':
    dataset = ORLDataset()
    engine = FaceIdentificationEngine()

    # load images directly from the dataset and not from the file system

    # create known embeddings fro every image class in the dataset
    known_embeddings = []
    for i in range(40):
        embedding = engine.extract_face_embeddings(dataset.images[i*10 + 1])
        known_embeddings.append(embedding)

    # Identify a query face
    query_image = dataset.images[32]
    match_index, distance = engine.identify_face(query_image, np.array(known_embeddings))

    if match_index is not None:
        print(f"Closest match: Face {match_index}, Distance: {distance}")

