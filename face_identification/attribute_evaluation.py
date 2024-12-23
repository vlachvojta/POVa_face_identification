import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

# add parent of this file to path to enable importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_loader import DataLoader as CelebADataLoader
from datasets.data_loader import Partition
from datasets.data_structure import Attribute
from datasets.image_preprocessor import ImagePreProcessorMTCNN
from face_identification.face_embedding_engine import FacenetEmbeddingEngine
from prettytable import PrettyTable


def get_accuracy(data, distance_threshold):
    accuracy = {}
    for attr_name, class_dict in data.items():
        miss = 0
        hit = 0
        count = 0
        for class_id, embeddings in class_dict.items():
            pairs = [(x, y) for x in embeddings[0] for y in embeddings[1]]
            count += len(pairs)
            for pair in pairs:
                distance = F.cosine_similarity(torch.from_numpy(pair[0]).unsqueeze(0), torch.from_numpy(pair[1]).unsqueeze(0))
                if distance < distance_threshold:
                    miss += 1
                else:
                    hit += 1
                    
        value = 0 if count == 0 else hit / count
        accuracy[attr_name] = [value, count]
    return accuracy

def evaluate_attribute_accuracy():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_engine = FacenetEmbeddingEngine(device=device, verbose=False)
    
    preprocessor = ImagePreProcessorMTCNN()
    val_dataset = CelebADataLoader(data_path='../../datasets/CelebA/', partition=Partition.VAL,
                                   sequential_classes=True, balance_classes=True,# limit=100,
                                   image_preprocessor=preprocessor)

    # Init data structure
    data = {}
    for attr in Attribute:
        data[attr.name] = {}
        for i in val_dataset.unique_classes():
            data[attr.name][i] = [[], []]
            
    # Load all images, create embeddings
    for i in range(len(val_dataset)):
        item = val_dataset[i]
        image = np.array(item.image).transpose(2, 0, 1)
        embedding = embedding_engine(image)

        for attr in Attribute:
            if attr.name in item.attributes():
                data[attr.name][item.id][0].append(embedding)
            else:
                data[attr.name][item.id][1].append(embedding)
    
    accuracy_0 = get_accuracy(data, 0)
    accuracy_0_4 = get_accuracy(data, 0.4)
    accuracy_0_6 = get_accuracy(data, 0.6)
    accuracy_0_7 = get_accuracy(data, 0.7)
    accuracy_0_8 = get_accuracy(data, 0.8)

    combined_accuracy = {}
    for attr in Attribute:
        combined_accuracy[attr.name] = [
            accuracy_0[attr.name][0],
            accuracy_0_4[attr.name][0],
            accuracy_0_6[attr.name][0],
            accuracy_0_7[attr.name][0],
            accuracy_0_8[attr.name][0],
            accuracy_0[attr.name][1]
        ]
    
    table = PrettyTable()
    table.field_names = ["Attribute", "Accuracy (0)", "Accuracy (0.4)", "Accuracy (0.6)", "Accuracy (0.7)", "Accuracy (0.8)", "Number of pairs"]

    sorted_accuracy = sorted(combined_accuracy.items(), key=lambda item: item[1][0], reverse=True)
    for attr, values in sorted_accuracy:
        table.add_row([attr, f"{values[0]:.2f}", f"{values[1]:.2f}", f"{values[2]:.2f}", f"{values[3]:.2f}", f"{values[4]:.2f}", values[5]])

    print(table)

if __name__ == "__main__":
    evaluate_attribute_accuracy()