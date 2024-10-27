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
from face_identification.face_embedding_engine import FaceEmbeddingEngine, ResnetEmbeddingEngine


class DistanceFunction(Enum):
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine'


class FaceIdentificationEngine:
    def __init__(self, embedding_model: FaceEmbeddingEngine, known_embeddings: np.ndarray,
                 distance_function: DistanceFunction = DistanceFunction.COSINE,
                 class_ids: list[str] = None):
        # TODO handle face detection (and alignment)

        self.embedding_model = embedding_model

        assert len(known_embeddings) > 0, "Known embeddings must not be empty."
        assert len(known_embeddings.shape) == 2, "Known embeddings must be a 2D array."
        self.known_embeddings = known_embeddings

        self.distance_function = distance_function

        if class_ids is None:
            class_ids = [str(i) for i in range(len(known_embeddings))]
        assert len(class_ids) == len(known_embeddings), "Number of class IDs must match number of known embeddings."
        self.class_ids = class_ids

    def __call__(self, input_image):
        if isinstance(input_image, list):
            return [self.__call__(i) for i in input_image]

        return self.identify_face(input_image)

    def identify_face(self, image) -> (str | None, dict[int, float] | None):
        image_embedding = self.embedding_model(image)

        if image_embedding is None:
            return None, None

        # Compute distances using the specified distance function
        if self.distance_function == DistanceFunction.EUCLIDEAN:
            distances = np.linalg.norm(self.known_embeddings - image_embedding, axis=1)
            closest_match_index = np.argmin(distances)
        elif self.distance_function == DistanceFunction.COSINE:
            distances = F.cosine_similarity(torch.tensor(self.known_embeddings), torch.tensor(image_embedding), dim=1)
            closest_match_index = torch.argmax(distances).item()

        # return closest match class id and distances as a dict(key: class_id, value: distance)
        distances = {self.class_ids[i]: distances[i].item() for i in range(len(distances))}

        return self.class_ids[closest_match_index], distances

def test_engine_with_ORL_dataset():
    dataset = ORLDataset()
    embedding_engine = ResnetEmbeddingEngine()

    # create known embeddings fro every image class in the dataset by taking the first image of each class
    known_embeddings = []
    for i in range(40):
        embedding = embedding_engine(dataset.images[i*10])
        known_embeddings.append(embedding)
    known_embeddings = np.array(known_embeddings)

    identification_engine = FaceIdentificationEngine(embedding_engine, known_embeddings=known_embeddings)

    # Identify face
    query_image = dataset.images[5]
    match_index, distances = identification_engine.identify_face(query_image)
    distance = distances[match_index]

    if match_index is not None:
        print(f"Closest match: Face {match_index}, Distance: {distance}")

    print(f'distances: {distances}')

    # # display distances in a histogram
    # import matplotlib.pyplot as plt
    # plt.hist(distances, bins=30)
    # plt.title('Distances to known embeddings')
    # plt.savefig('distances_histogram.png')
    # plt.clf() # clear the plot
    # # plt.show()

if __name__ == '__main__':
    test_engine_with_ORL_dataset()
