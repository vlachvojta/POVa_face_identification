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


def distance_criterium_is_max(distance_function: DistanceFunction) -> bool:
    if distance_function == DistanceFunction.EUCLIDEAN:
        return False
    elif distance_function == DistanceFunction.COSINE:
        return True
    else:
        raise ValueError(f"Unknown distance function: {distance_function}")


class ClassEmbeddingStyle(Enum):
    FIRST = 'first'
    RANDOM = 'random'
    MEAN = 'mean'
    MEDIAN = 'median'
    MIN = 'min'
    MAX = 'max'


class FaceIdentificationEngine:
    def __init__(self, embedding_model: FaceEmbeddingEngine,
                 images: list[np.ndarray], target_classes: list[str],
                 class_embedding_file: str, force_new_class_embeddings: bool = False,
                 distance_function: DistanceFunction = DistanceFunction.COSINE,
                 class_embedding_style: ClassEmbeddingStyle = ClassEmbeddingStyle.FIRST):
        # TODO handle face detection (and alignment)

        self.embedding_model = embedding_model
        self.distance_function = distance_function
        self.class_embedding_style = class_embedding_style
        self.class_embedding_file = class_embedding_file if class_embedding_file.endswith('.npy') else class_embedding_file + '.npy'
        self.class_ids_file = class_embedding_file.replace('.npy', '_class_ids.npy')
        self.force_new_class_embeddings = force_new_class_embeddings

        self.distance_criterium = 'max' if distance_function == DistanceFunction.COSINE else 'min'

        assert len(images) > 0, "Images must not be empty."
        assert len(images) == len(target_classes), "Number of images must match number of class IDs."

        self.class_embeddings, self.class_ids = self.load_or_create_class_embeddings(images, target_classes)

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
            distances = np.linalg.norm(self.class_embeddings - image_embedding, axis=1)
            closest_match_index = np.argmin(distances)
        elif self.distance_function == DistanceFunction.COSINE:
            distances = F.cosine_similarity(torch.tensor(self.class_embeddings), torch.tensor(image_embedding), dim=1)
            closest_match_index = torch.argmax(distances).item()

        # return closest match class id and distances as a dict(key: class_id, value: distance)
        distances = {self.class_ids[i]: distances[i].item() for i in range(len(distances))}

        return self.class_ids[closest_match_index], distances

    def load_or_create_class_embeddings(self, images: list[np.ndarray], target_classes: list[str]) -> (np.ndarray, list[str]):
        class_embeddings = None
        class_ids = None

        if (not self.force_new_class_embeddings and 
            (os.path.exists(self.class_embedding_file) and os.path.exists(self.class_ids_file))):
                class_embeddings = np.load(self.class_embedding_file)
                class_ids = np.load(self.class_ids_file)

                # check if class_ids are the same as the ones provided
                unique_classes_ref = list(set(target_classes))
                unique_classes = list(set(class_ids))
                if sorted(unique_classes) != sorted(unique_classes_ref):
                    print(f"Class IDs in the file {self.class_ids_file} do not match the provided target classes.")
                    print("Creating new class embeddings.")
                    print(f'')
                    class_embeddings = None
                    class_ids = None
                else:
                    print(f'Loaded class embeddings from {self.class_embedding_file}')
                    print('')

        if class_embeddings is None or class_ids is None:
            class_embeddings, class_ids = create_class_embeddings(self.embedding_model, images, target_classes, self.class_embedding_style)

        # check correctness of class embeddings and class ids
        assert len(class_embeddings.shape) == 2, "Class embeddings must be a 2D array."
        assert len(class_ids) == len(class_embeddings), "Number of class IDs must match number of known embeddings."

        # save class_embeddings and class_ids to file
        os.makedirs(os.path.dirname(self.class_embedding_file), exist_ok=True)
        np.save(self.class_embedding_file, class_embeddings)
        np.save(self.class_ids_file, class_ids)
        
        return class_embeddings, class_ids


def create_class_embeddings(embedding_engine: FaceEmbeddingEngine, images: list[np.ndarray], target_classes: list[str],
                            class_embedding_style: ClassEmbeddingStyle = ClassEmbeddingStyle.FIRST):
    assert len(images) > 0, "Images must not be empty."
    assert len(images) == len(target_classes), f"Number of images must match number of target classes. Got {len(images)} images and {len(target_classes)} target classes."

    # create a unique class id for every class
    # assume the classes can be mixed e.g. classes: ['A', 'B', 'A', 'C', 'B', 'A']
    unique_classes = {}
    class_ids = []
    for class_name in target_classes:
        if class_name not in unique_classes:
            unique_classes[class_name] = len(unique_classes)
        class_ids.append(unique_classes[class_name])

    # create class embeddings for every unique class
    class_embeddings = np.zeros((len(unique_classes), embedding_engine.DIM))
    class_order = []
    for class_name, class_id in unique_classes.items():
        # class images = images where class_id == class_ids
        class_images = images[np.where(np.array(class_ids) == class_id)]
        print(f'creating class embedding for {class_name} with {len(class_images)} images')

        class_embedding = create_class_embedding(embedding_engine, class_images, class_embedding_style)

        # class_embeddings.append(class_embedding)
        class_embeddings[class_id] = class_embedding
        class_order.append(class_name)

    return class_embeddings, class_order


def create_class_embedding(embedding_engine: FaceEmbeddingEngine, images: list[np.ndarray], class_embedding_style: ClassEmbeddingStyle = ClassEmbeddingStyle.FIRST):
    assert len(images) > 0, "Images must not be empty."

    if class_embedding_style == ClassEmbeddingStyle.FIRST:
        return embedding_engine(images[0])
    elif class_embedding_style == ClassEmbeddingStyle.RANDOM:
        random_idx = np.random.randint(len(images))
        return embedding_engine(images[random_idx])

    embeddings = embedding_engine(images)

    if class_embedding_style == ClassEmbeddingStyle.MEAN:
        return np.mean(embeddings, axis=0)
    elif class_embedding_style == ClassEmbeddingStyle.MEDIAN:
        return np.median(embeddings, axis=0)
    elif class_embedding_style == ClassEmbeddingStyle.MIN:
        return np.min(embeddings, axis=0)
    elif class_embedding_style == ClassEmbeddingStyle.MAX:
        return np.max(embeddings, axis=0)
    else:
        raise ValueError(f"Unknown class embedding style: {class_embedding_style}")


def test_engine_with_ORL_dataset():
    dataset = ORLDataset()
    embedding_engine = ResnetEmbeddingEngine(device='cpu')

    # initialize face identification engine
    force_new_class_embeddings = False
    identification_engine = FaceIdentificationEngine(embedding_engine, dataset.images, dataset.targets,
                                                     class_embedding_style=ClassEmbeddingStyle.MEAN,
                                                     class_embedding_file='face_identification/tmp/class_embeddings.npy',
                                                     force_new_class_embeddings=force_new_class_embeddings)

    # Identify face
    test_image_index = 12
    query_image = dataset.images[test_image_index]
    print(f'Testing image {test_image_index} from class {dataset.targets[test_image_index]}')
    match_index, distances = identification_engine(query_image)
    distance = distances[match_index]

    print(f"Closest match: Face {match_index}, Distance: {distance}")
    print(f'distances: {distances}')


if __name__ == '__main__':
    test_engine_with_ORL_dataset()
