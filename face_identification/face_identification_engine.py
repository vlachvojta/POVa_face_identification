import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from enum import Enum

# add current working directory + parent to path
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from datasets.ORL_dataset import ORLDataset
from face_identification.face_embedding_engine import FaceEmbeddingEngine


class DistanceFunction(Enum):
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine'


class FaceIdentificationEngine:
    def __init__(self, embedding_model: callable, min_face_size: int = 160, 
                 known_embeddings: list = None, distance_function: DistanceFunction = DistanceFunction.COSINE):
        # TODO handle face detection and alignment

        self.embedding_model = embedding_model
        self.min_face_size = min_face_size
        self.known_embeddings = known_embeddings
        self.distance_function = distance_function

    def identify_face(self, image):
        # Load the image
        image_embedding = self.embedding_model(image)
        # query_embedding = self.extract_face_embeddings(image)

        if image_embedding is None:
            return None, None

        if self.distance_function == DistanceFunction.EUCLIDEAN:
            # Compute distances using Euclidean distance
            distances = np.linalg.norm(self.known_embeddings - image_embedding, axis=1)
            closest_match_index = np.argmin(distances)
        elif self.distance_function == DistanceFunction.COSINE:
            # Compute distances using cosine similarity
            distances = F.cosine_similarity(torch.tensor(self.known_embeddings), torch.tensor(image_embedding), dim=1)
            closest_match_index = torch.argmax(distances).item()

        return closest_match_index, distances

    # def process_image_with_load(self, image_path):
    #     # old code to load image, detect faces, align faces and extract embeddings
    #     # Load the image
    #     img = Image.open(image_path)
    #     # Detect faces
    #     boxes, _ = self.mtcnn.detect(img)
    #     if boxes is not None:
    #         # Align faces
    #         aligned = self.mtcnn(img)
    #         # Compute embeddings
    #         embedding = self.extract_face_embeddings(aligned)
    #         return embedding
    #     else:
    #         print("No faces detected in the image.")
    #         return None


def test_engine_with_ORL_dataset():
    dataset = ORLDataset()

    embedding_engine = FaceEmbeddingEngine()

    # create known embeddings fro every image class in the dataset
    known_embeddings = []
    for i in range(40):
        embedding = embedding_engine(dataset.images[i*10])
        known_embeddings.append(embedding)

    identification_engine = FaceIdentificationEngine(embedding_engine, known_embeddings=known_embeddings)

    # Identify face
    query_image = dataset.images[5]
    match_index, distances = identification_engine.identify_face(query_image)
    distance = distances[match_index]

    if match_index is not None:
        print(f"Closest match: Face {match_index}, Distance: {distance}")

    # display distances in a histogram
    import matplotlib.pyplot as plt
    plt.hist(distances, bins=30)
    plt.title('Distances to known embeddings')
    plt.show()

if __name__ == '__main__':
    test_engine_with_ORL_dataset()
