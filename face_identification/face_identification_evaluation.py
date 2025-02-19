import os
import sys
import numpy as np
from dataclasses import dataclass
import shutil

import torch

# add parent of this file to path to enable importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ORL_dataset import ORLDataset
from datasets.data_loader import DataLoaderTorchWrapper as CelebADataLoader
from datasets.data_loader import Partition
from datasets.image_preprocessor import ImagePreProcessor, Squarify, Normalization, ImagePreProcessorMTCNN, ImagePreProcessorResnet
from face_detection.face_detection_engine import FaceDetectionEngine
from face_identification.face_embedding_engine import FaceEmbeddingEngine, FacenetEmbeddingEngine, BasicResnetEmbeddingEngine, TrainedEmbeddingEngine
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

        percentage = f'{self.hits / self.total * 100:.2f}%'
        return f'{percentage:<7} ({self.hits}/{self.total})'


class FaceIdentificationEvaluation:
    def __init__(self, identification_engine: FaceIdentificationEngine, print_interval: int = 10):
        self.identification_engine = identification_engine
        self.print_interval = print_interval

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
            if i and self.print_interval and i % self.print_interval == 0:
                print(f'Processed {i}/{len(images)} images.')
                self.print_stats()
                print('')

            if not isinstance(image, np.ndarray):
                print(f'Skipping image {i} as it is not a numpy array.')
                self.skipped += 1
                continue

            match_class, distances = self.identification_engine(image)
            # match_class_distance = distances[match_class]
            # print(f'Match class: {match_class} with distance: {match_class_distance}')
            # print(f'all distances: {distances}')

            if match_class == target_class:
                self.target_hit += 1
                self.class_hits[target_class].add_hit()
            else:
                self.class_hits[target_class].add_miss()

            self.update_top_n_hits(distances, target_class)

            self.processed += 1

        if self.print_interval:
            self.print_stats()
        self.print_stats(limited=True)

        return self.target_hit / self.processed

    def print_stats(self, limited: bool = False):
        print(f'Target hits: {self.target_hit/self.processed*100:.2f}% ({self.target_hit}/{self.processed})')
        print(f'Skipped: {self.skipped}')

        print(f'Top N hits: (target class is in top N classes according to distance)')
        for top_n in [1, 2, 3, 5]:
            if len(self.top_n_hits) >= top_n:
                hits = self.top_n_hits[top_n-1]
                percentage = f'{hits/self.processed*100:.2f}%'
                print(f'\ttop_{top_n}: {percentage:<7} ({hits}/{self.processed})')
        if limited:
            return

        print(f'Class hits:')
        for class_id in sorted(self.class_hits, key=lambda x: int(x)):
            class_stat = self.class_hits[class_id]
            if class_stat.total > 0:
                class_id_text = f'{class_id}:'
                print(f'\t{class_id_text:>8} {class_stat}')

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
    embedding_engine = FacenetEmbeddingEngine(device='cpu', verbose=False)

    # initialize face identification engine
    identification_engine = FaceIdentificationEngine(embedding_engine, dataset.images, dataset.targets,
                                                     class_embedding_style=ClassEmbeddingStyle.MEAN,
                                                     class_embedding_file='tmp/class_embeddings.npy',
                                                     force_new_class_embeddings=False)

    evaluation = FaceIdentificationEvaluation(identification_engine, print_interval=10)

    evaluation(dataset.images, dataset.targets)


def test_resnet_sanity_check():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_detection_engine = FaceDetectionEngine(device=device, keep_all=False)
    embedding_engine = BasicResnetEmbeddingEngine(device=device, verbose=False)

    output_path = 'tmp-renders/'
    shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    results=[]

    # resnet_preprocessor = ImagePreProcessorResnet(squarify=Squarify.AROUND_FACE_SQUISH)
    # test_preprocessing_config(face_detection_engine, embedding_engine, None, None, None, results, preprocessor=resnet_preprocessor)
    # resnet_preprocessor = ImagePreProcessorResnet(squarify=Squarify.AROUND_FACE)
    # test_preprocessing_config(face_detection_engine, embedding_engine, None, None, None, results, preprocessor=resnet_preprocessor)
    # test_preprocessing_config(
    #     face_detection_engine, embedding_engine, 
    #     Normalization.IMAGE_NET, Squarify.AROUND_FACE, 160, results)

    # test every option in data_loader.py
    # normalize_options = [Normalization.IMAGE_NET, Normalization._0_1, Normalization._1_1, Normalization.MEAN_STD, Normalization.MIN_MAX]
    padding_options = [None, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    # normalize_options = [Normalization._1_1, Normalization.MEAN_STD, Normalization.MIN_MAX]
    squarify_options = [Squarify.AROUND_FACE, Squarify.AROUND_FACE_SQUISH] #, Squarify.AROUND_FACE_STRICT, None, Squarify.CROP]
    # resize_options = [160, 224]

    for padding in padding_options:
        for squarify in squarify_options:
            print(f'\nTesting: Padding: {padding}, Squarify: {squarify}')
            preprocessor = ImagePreProcessorResnet(padding_around_face=padding, squarify=squarify)
            test_preprocessing_config(face_detection_engine, embedding_engine, None, None, None, results, preprocessor=preprocessor)

    # for normalize in normalize_options:
    #     for squarify in squarify_options:
    #         for resize in resize_options:
    #             print(f'\nTesting: Normalize: {normalize}, Squarify: {squarify}, Resize: {resize}')
    #             test_preprocessing_config(face_detection_engine, embedding_engine, normalize, squarify, resize, results)

    print('\n--------------------')
    print('Final results:')
    for normalize, squarify, resize, accuracy, padding in results:
        print(f'Accuracy: {accuracy:.3f}, Normalize: {normalize}, Squarify: {squarify}, Resize: {resize}, Padding: {padding}')

def test_preprocessing_of_CelebA_images_val_set():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_detection_engine = FaceDetectionEngine(device=device, keep_all=False)
    embedding_engine = FacenetEmbeddingEngine(device=device, verbose=False)

    output_path = 'tmp-renders/'
    shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # test every option in data_loader.py
    normalize_options = [Normalization._0_1, Normalization._1_1, Normalization.MEAN_STD, Normalization.MIN_MAX]
    squarify_options = [Squarify.AROUND_FACE, None, Squarify.CROP]
    resize_options = [160]

    results=[]

    mtcnn_strict_preprocessor = ImagePreProcessorMTCNN()
    test_preprocessing_config(face_detection_engine, embedding_engine, None, None, None, results, preprocessor=mtcnn_strict_preprocessor)
    test_preprocessing_config(face_detection_engine, embedding_engine, None, Squarify.AROUND_FACE_STRICT, None, results)

    for normalize in normalize_options:
        for squarify in squarify_options:
            for resize in resize_options:
                print(f'\nTesting: Normalize: {normalize}, Squarify: {squarify}, Resize: {resize}')
                test_preprocessing_config(face_detection_engine, embedding_engine, normalize, squarify, resize, results)

    print('\n--------------------')
    print('Final results:')
    for normalize, squarify, resize, accuracy in results:
        print(f'Accuracy: {accuracy:.3f}, Normalize: {normalize}, Squarify: {squarify}, Resize: {resize}')

def test_preprocessing_config(face_detection_engine, embedding_engine, normalize, squarify, resize, results, preprocessor=None):
    if preprocessor is None:
        preprocessor = ImagePreProcessor(face_detection_engine=face_detection_engine,
                                        squarify=squarify, normalize=normalize, resize=resize)

    val_dataset = CelebADataLoader(data_path='../../datasets/CelebA/', partition=Partition.VAL,
                                   sequential_classes=True, balance_classes=True, limit=1000,
                                   image_preprocessor=preprocessor)
    print('')
    accuracy = evaluate_dataset(val_dataset, embedding_engine)

    normalize = preprocessor.normalize.name if preprocessor.normalize else None
    squarify = preprocessor.squarify.name if preprocessor.squarify else None
    resize = preprocessor.resize if preprocessor.resize else None
    padding = preprocessor.padding_around_face if preprocessor.padding_around_face else None

    results.append((normalize, squarify, resize, accuracy, padding))
    # results.append((normalize, squarify, resize, accuracy, preprocessor.padding_around_face))
    print(f'results so far:')
    for normalize, squarify, resize, accuracy, padding in results:
        print(f'Accuracy: {accuracy:.3f}, Normalize: {normalize}, Squarify: {squarify}, Resize: {resize}, Padding: {padding}')

def evaluate_dataset(val_dataset, embedding_engine):
    all_images = []
    all_classes = []

    # for i in range(50):
    for i in range(len(val_dataset)):
        item = val_dataset[i]
        all_images.append(item['image'])
        all_classes.append(item['class'])

    all_images = np.array(all_images)
    all_classes = np.array(all_classes)

    # initialize face identification engine
    identification_engine = FaceIdentificationEngine(embedding_engine, all_images, all_classes,
                                                     class_embedding_style=ClassEmbeddingStyle.FIRST,
                                                     class_embedding_file='tmp/class_embeddings.npy',
                                                     force_new_class_embeddings=True)

    evaluation = FaceIdentificationEvaluation(identification_engine, print_interval=None) #100)

    # all_images = all_images[:50]
    # all_classes = all_classes[:50]

    all_images, all_classes = delete_first_of_each_class(all_images, all_classes)

    return evaluation(all_images, all_classes)

def delete_first_of_each_class(images, classes):
    unique_classes = np.unique(classes)
    for class_id in unique_classes:
        idx = np.where(classes == class_id)[0][0]
        images = np.delete(images, idx, axis=0)
        classes = np.delete(classes, idx)

    return images, classes

def evaluate_trained_model(embedding_engine, all_images, all_classes, images_without_first, classes_without_first):
    identification_engine = FaceIdentificationEngine(embedding_engine, all_images, all_classes,
                                                     class_embedding_style=ClassEmbeddingStyle.MEAN,
                                                     class_embedding_file='tmp/class_embeddings.npy',
                                                     force_new_class_embeddings=True)

    evaluation = FaceIdentificationEvaluation(identification_engine, print_interval=None) #100)

    return evaluation(images_without_first, classes_without_first)

def get_all_data(dataset: CelebADataLoader):
    all_images = []
    all_classes = []

    for i in range(len(dataset)):
        item = dataset[i]
        all_images.append(item['image'])
        all_classes.append(item['class'])

    all_images = np.array(all_images)
    all_classes = np.array(all_classes)

    return all_images, all_classes

def test_trained_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocessorMTCNN = ImagePreProcessorMTCNN()
    preprocessorResnet = ImagePreProcessorResnet()
    print(f'Getting all data from CelebA validation set with MTCNN and Resnet preprocessing...')

    limit=1000
    datasetMTCNN = CelebADataLoader(data_path='../../datasets/CelebA/', partition=Partition.VAL,
                                    sequential_classes=True, balance_classes=True, limit=limit,
                                    image_preprocessor=preprocessorMTCNN)
    datasetResnet = CelebADataLoader(data_path='../../datasets/CelebA/', partition=Partition.VAL,
                                    sequential_classes=True, balance_classes=True, limit=limit,
                                    image_preprocessor=preprocessorResnet)
    
    imagesMTCNN, classesMTCNN = get_all_data(datasetMTCNN)
    images_MTCNN_without_first, classes_MTCNN_without_first = imagesMTCNN, classesMTCNN #delete_first_of_each_class(imagesMTCNN, classesMTCNN)
    imagesResnet, classesResnet = get_all_data(datasetResnet)
    images_Resnet_without_first, classes_Resnet_without_first = imagesResnet, classesResnet #delete_first_of_each_class(imagesResnet, classesResnet)

    print(f'Data loaded')

    train_root = '../../training/'
    trained_model_paths=[
        'resnet_05',
        'facenet_05',
        'facenet_05_eyeglasses',
        'resnet_05_eyeglasses',
        'facenet_05_blurry',
        'resnet_05_blurry',
    ]

    embedding_engines = [
        [BasicResnetEmbeddingEngine(device=device, verbose=False), 'BasicResnetEmbeddingEngine'],
        [FacenetEmbeddingEngine(device=device, verbose=False), 'FacenetEmbeddingEngine'],
    ]

    # Load trained models as embedding engines
    for trained_model_path in trained_model_paths:
        trained_model_path = os.path.join(train_root, trained_model_path)
        embedding_engine = TrainedEmbeddingEngine(trained_model_path, device=device, verbose=False)
        embedding_engines.append([embedding_engine, trained_model_path])
    
    print(f'resting {len(embedding_engines)} trained models...')

    results = []

    for embedding_engine, trained_model_path in embedding_engines:
        model_class = embedding_engine.model.__class__.__name__
        print(f'Trained model: {model_class}')

        if 'Facenet' in model_class or 'InceptionResnetV1' in model_class:
            result = evaluate_trained_model(embedding_engine, imagesMTCNN, classesMTCNN, images_MTCNN_without_first, classes_MTCNN_without_first)
            results.append((trained_model_path, result))
        elif 'BasicResnet' in model_class:
            result = evaluate_trained_model(embedding_engine, imagesResnet, classesResnet, images_Resnet_without_first, classes_Resnet_without_first)
            results.append((trained_model_path, result))
        else:
            raise ValueError(f'Unknown model class: {model_class}, do not know how to preprocess images.')

    print('\n--------------------')
    for trained_model_path, result in results:
        print(f'Trained model: {trained_model_path}, Accuracy: {result:.3f}')


if __name__ == '__main__':
    test_evaluation_with_ORL_dataset()
    # test_preprocessing_of_CelebA_images_val_set()
    # test_resnet_sanity_check()
    # test_trained_models()
