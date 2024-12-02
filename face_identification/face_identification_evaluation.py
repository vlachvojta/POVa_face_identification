import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from enum import Enum
from dataclasses import dataclass

# add parent of this file to path to enable importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ORL_dataset import ORLDataset
from face_identification.face_embedding_engine import FaceEmbeddingEngine, ResnetEmbeddingEngine
from face_identification.face_identification_engine import FaceIdentificationEngine, DistanceFunction, ClassEmbeddingStyle, distance_criterium_is_max


class TopNHits:
    def __init__(self, n: int):
        self.n = n
        self.hits = 0
        self.misses = 0

    def update(self, is_hit: bool):
        if is_hit:
            self.hits += 1
        else:
            self.misses += 1

    def __str__(self):
        return f'Top {self.n} hits: {self.hits}, misses: {self.misses}'

@dataclass
class ClassStat:
    hits: int = 0
    total: int = 0

    def add_hit(self):
        self.hits += 1
        self.total += 1

    def add_miss(self):
        self.total += 1

    def __str__(self):
        if self.total == 0:
            return 'No hits or misses.'
        return f'{self.hits / self.total * 100:.2f}% ({self.hits}/{self.total})'


class FaceIdentificationEvaluation:
    def __init__(self, identification_engine: FaceIdentificationEngine):
        self.identification_engine = identification_engine

        # default int dict for class hits
        self.class_hits = {}
        self.top_n_hits = []

        self.skipped = 0
        self.processed = 0
        self.target_hit = 0
        self.top_3_hit = 0
        self.top_5_hit = 0
        self.top_10_hit = 0

        self.reverse_ranking = distance_criterium_is_max(identification_engine.distance_function)
    
    def __call__(self, images, target_classes: list[str]):
        # TODO: add support for batch processing
        assert len(images) == len(target_classes), "Number of images must match number of class IDs."

        self.class_hits = {class_id: ClassStat() for class_id in set(target_classes)}
        self.top_n_hits = np.zeros(len(self.class_hits), dtype=int)

        for i, (image, target_class) in enumerate(zip(images, target_classes)):
            if i and i % 10 == 0:
                print(f'Processed {i}/{len(images)} images.')
                self.print_stats()
                print('')

            if not isinstance(image, np.ndarray):
                print(f'Skipping image {i} as it is not a numpy array.')
                self.skipped += 1
                continue

            match_class, distances = self.identification_engine(image)
            match_class_distance = distances[match_class]
            # print(f'Match class: {match_class} with distance: {match_class_distance}')
            # print(f'all distances: {distances}')

            if match_class == target_class:
                self.target_hit += 1
                self.class_hits[target_class].add_hit()
            else:
                self.class_hits[target_class].add_miss()

            self.update_top_n_hits(distances, target_class)

            self.processed += 1

        self.print_stats()

    def print_stats(self):
        print(f'Target hits: {self.target_hit/self.processed*100:.2f}% ({self.target_hit}/{self.processed})')
        print(f'Skipped: {self.skipped}')

        print(f'Top N hits:')
        for top_n in [1, 2, 3, 5]:
            if len(self.top_n_hits) >= top_n:
                hits = self.top_n_hits[top_n-1]
                print(f'\ttop_{top_n}: {hits/self.processed*100:.2f}% ({hits}/{self.processed})')

        print(f'Class hits:')
        for class_id, class_stat in self.class_hits.items():
            if class_stat.total > 0:
                print(f'\t{class_id}: {class_stat}')

        print('')

    def update_top_n_hits(self, distances: dict[str, float], target_class: str):
        # order distances and get target class rank
        sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=self.reverse_ranking)
        target_class_rank = [i for i, (class_id, _) in enumerate(sorted_distances) if class_id == target_class][0]

        # increment top_n_hits >= target_class_rank
        increment = np.zeros(len(self.top_n_hits), dtype=int)
        increment[target_class_rank:] = 1
        self.top_n_hits += increment


def test_evaluation_with_ORL_dataset():
    dataset = ORLDataset()
    embedding_engine = ResnetEmbeddingEngine(device='cpu')

    # initialize face identification engine
    force_new_class_embeddings = False
    identification_engine = FaceIdentificationEngine(embedding_engine, dataset.images, dataset.targets,
                                                     class_embedding_style=ClassEmbeddingStyle.MEAN,
                                                     class_embedding_file='face_identification/tmp/class_embeddings.npy',
                                                     force_new_class_embeddings=force_new_class_embeddings)

    evaluation = FaceIdentificationEvaluation(identification_engine)

    evaluation(dataset.images, dataset.targets)


if __name__ == '__main__':
    test_evaluation_with_ORL_dataset()
