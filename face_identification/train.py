import os
import sys
import logging
import argparse
import time
import json
import re
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
import torchvision.utils as vutils

# add parent of this file to path to enable importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_loader import DataLoaderTorchWrapper as CelebADataLoader
from datasets.data_loader import Partition
from datasets.image_preprocessor import ImagePreProcessorMTCNN
from face_detection.face_detection_engine import FaceDetectionEngine
from face_identification.face_embedding_models import FacenetPytorchWrapper, NetUtils
from face_identification.face_embedding_models import *  # TODO: try to eliminate this import
from face_identification.training_monitor import TrainingMonitor
from common import utils

logging.basicConfig(level=logging.DEBUG)


def parse_arguments():
    parser = argparse.ArgumentParser(usage="Train a model for face identification.")

    # ClearML
    parser.add_argument("--name", type=str, required=False,
                        help="Task name for ClearML.")
    parser.add_argument("--project-name", type=str, default="xvlach_POVa_face_identification",
                        help="Project name for ClearML.")

    # Data paths
    parser.add_argument("-d", "--dataset-path", required=True, type=str, help="Path to the dataset folder.")
    parser.add_argument("-o", "--output-path", required=True, type=str, help="Path to the output folder.")
    parser.add_argument("--config", type=str, default=None, help="Path to model config")
    parser.add_argument("--render", action="store_true", help="Render validation samples.")
    parser.add_argument("--detect-faces", action="store_true", help="Detect faces in images using MTCNN from facenet_pytorch.")

    # Trainer settings
    parser.add_argument("--max-iter", default=100_000, type=int)
    parser.add_argument("--view-step", default=50, type=int, help="Number of training iterations between validation.")
    parser.add_argument("--save-step", default=100, type=int, help="Number of training iterations between model saving.")
    parser.add_argument("--val-size", default=None, type=int, help="Number of samples to use for validation, leave None to use all samples.")

    # Optimization settings
    parser.add_argument("--batch-size", "-b", default=16, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--weight-decay", "-w", default=0.01, type=float)

    # Image backbone settings
    # parser.add_argument("--image-backbone", default="resnet50", type=str, help="Image encoder name from the TIMM library.")
    # parser.add_argument("--image-backbone-depth", default=0, type=int, help="Depth from which to take the output feature map -- 0 means the last feature map")
    # parser.add_argument("--freeze-image-backbone", action="store_true", help="Disables training of the image backbone.")

    # parser.add_argument("--image-width", default=1024, type=int, help="Images are resized to this width.")
    # parser.add_argument("--image-height", default=1024, type=int, help="Images are resized to this height.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.info(args)
    # logging.getLogger("cv2").setLevel(level=logging.ERROR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on: {device}")

    logging.info("Loading datasets ...")    
    preprocessor = ImagePreProcessorMTCNN(device=device) if args.detect_faces else None

    trn_dataset = CelebADataLoader(
        args.dataset_path, partition=Partition.TRAIN, sequential_classes=True, image_preprocessor=preprocessor)
    val_dataset = CelebADataLoader(
        args.dataset_path, partition=Partition.VAL, sequential_classes=True, image_preprocessor=preprocessor, limit=args.val_size, balance_classes=True)

    logging.info(f"Train dataset:      {len(trn_dataset)} samples with {len(trn_dataset.unique_classes())} unique classes")
    logging.info(f"Validation dataset: {len(val_dataset)} samples with {len(val_dataset.unique_classes())} unique classes")

    trn_dataloader = torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, persistent_workers=True, num_workers=4)

    val_dataloaders = {
        # 'trn': torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=False, persistent_workers=True, num_workers=2),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, persistent_workers=True, num_workers=2)
    }

    model, trained_steps = load_model(args.output_path)
    model = model.to(device)
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.debug(f'learnable parameters: {learnable_params}')

    logging.info("Starting training ...")
    monitor = TrainingMonitor(name=args.name, project_name=args.project_name, output_path=args.output_path)
    try:
        train(
            model=model,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            trn_dataloader=trn_dataloader,
            val_dataloaders=val_dataloaders,
            max_iter=args.max_iter,
            view_step=args.view_step,
            save_step=args.save_step,
            monitor=monitor,
            output_path=args.output_path,
            render=args.render,
            trained_steps=trained_steps,
        )
        logging.info("Training finished.")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")

    logging.info("DONE")


def train(
    model: torch.nn.Module,
    device: torch.device,
    trn_dataloader: torch.utils.data.DataLoader,
    val_dataloaders: dict[str, torch.utils.data.DataLoader],
    lr: float,
    weight_decay: float,
    max_iter: int,
    view_step: int,
    save_step: int,
    monitor: TrainingMonitor,
    output_path: str,
    render: bool=False,
    trained_steps: int=0,
):
    logging.info(f"Training on {len(trn_dataloader.dataset)} samples. with unique classes {len(trn_dataloader.dataset.unique_classes())}")

    # ArcFace Loss with Cosine Similarity
    criterion = losses.ArcFaceLoss(
        num_classes=len(trn_dataloader.dataset.unique_classes()),
        embedding_size=512,
        margin=0.5,
        scale=64,
        distance=CosineSimilarity()
    ).to(device)

    miner = miners.TripletMarginMiner(
        margin=0.2,
        type_of_triplets='hard'
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    iteration = trained_steps
    train_losses = []
    t1 = time.time()

    for batch_data in trn_dataloader:
        iteration += 1
        if iteration > max_iter:
            break

        model.train()

        images = batch_data["image"].to(device).float()
        classes = batch_data["class"].to(device)

        # print(f'images.shape: {images.shape}')
        embeddings = model(images)
        # print('embeddings.shape:', embeddings.shape)

        hard_pairs = miner(embeddings, classes)

        loss = criterion(embeddings, classes, hard_pairs)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        if render:
            image_name = f"{iteration}_loss_{loss.item():.4f}.png"
            render_batch_images(images = images, path = f"{output_path}/train", filename = image_name)

        if iteration % view_step == 0:
            t2 = time.time()

            monitor.iterations.append(iteration)
            monitor.add_value(f"train_loss", sum(train_losses) / len(train_losses))

            train_losses = []

            for val_name, val_data_loader in val_dataloaders.items():
                validate(
                    model=model,
                    data_loader=val_data_loader,
                    data_loader_name=val_name,
                    criterion=criterion,
                    device=device,
                    monitor=monitor,
                    render=render,
                    output_path=output_path,
                    training_iter = iteration,
                )

            monitor.add_value("view_time", t2 - t1)
            log_string = monitor.get_last_string()
            print(f"\n{log_string}")

            monitor.report_results()
            monitor.save_csv()
            t1 = time.time()

        if iteration % save_step == 0:
            save_model(model, output_path, iteration)

def validate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    data_loader_name: str,
    output_path: str,
    training_iter: int,
    criterion: torch.nn.Module,
    device: torch.device,
    monitor: TrainingMonitor,
    render: bool=False,
):
    logging.info("Validating ...")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_losses = []

    model.eval()
    with torch.no_grad():
        embeddings_all = torch.empty((len(data_loader.dataset), model.embedding_size), device=device)
        classes_all = torch.empty((len(data_loader.dataset)), device=device, dtype=torch.int64)
        images_all = torch.empty((len(data_loader.dataset), *next(iter(data_loader))["image"].shape[1:]), device="cpu")

        for i, batch_data in enumerate(data_loader):
            images = batch_data["image"].to(device).float()
            classes = batch_data["class"].to(device)

            embeddings = model(images)
            
            loss = criterion(embeddings, classes)
            total_loss += loss.item() * len(images)
            batch_losses.append((i, loss.item(), images))

            total_samples += len(images)

            # normalize embeddings + store them and classes for accuracy calculation
            # embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            start = i*len(embeddings)
            end = start + len(embeddings)
            embeddings_all[start:end] = embeddings
            classes_all[start:end] = classes
            images_all[start:end] = images
    
    embeddings_all = torch.nn.functional.normalize(embeddings_all, dim=1)
    
    worst_batches = sorted(batch_losses, key=lambda x: x[1], reverse=True)[:3]
    
    if render:
        for batch_index, loss, images in worst_batches:
            image_name = f"{training_iter}_{batch_index}_loss_{loss:.4f}.png"
            render_batch_images(images=images, path=f"{output_path}/val", filename=image_name)


    # calculate average loss
    avg_loss = total_loss / total_samples
    monitor.add_value("val_loss", avg_loss)
    logging.info(f"Validation Loss: {avg_loss:.4f}")

    # calculate similarity matrix + bool matrix for same or different classes
    similarities_all = torch.mm(embeddings_all, embeddings_all.T)
    same_or_diff_all = (classes_all.unsqueeze(0) == classes_all.unsqueeze(1)).to(device)

    # setting diagonal to False to avoid self pairing
    mask = torch.eye(similarities_all.size(0), device=device, dtype=torch.bool)
    similarities_all = similarities_all.masked_select(~mask).view(similarities_all.size(0), -1)
    same_or_diff_all = same_or_diff_all.masked_select(~mask).view(same_or_diff_all.size(0), -1)

    thresholds = [0.0, 0.4, 0.8]
    for threshold in thresholds:
        calculate_accuracy_with_threshold(
            threshold=threshold,
            similarities_all=similarities_all,
            same_or_diff_all=same_or_diff_all,
            monitor=monitor
        )
        
    if render:
        least_similar_pairs = []
        similarities = []
        class_pairs = []
        image_indices = []
        all_pairs = []

        for i in range(similarities_all.size(0)):
            dissimilarities = similarities_all[i]
            least_similar_index = torch.argmin(dissimilarities).item()

            if classes_all[i] == classes_all[least_similar_index]:
                least_similar_pairs.append((i, least_similar_index))
                similarities.append(dissimilarities[least_similar_index].item())
                class_pairs.append((classes_all[i].item(), classes_all[least_similar_index].item()))
                image_indices.append((i, least_similar_index))

        for img_idx_1, img_idx_2 in least_similar_pairs:
            image1 = images_all[img_idx_1]
            image2 = images_all[img_idx_2]
            all_pairs.append((image1, image2))

        if len(all_pairs) > 0:
            render_pairs(
                pairs=all_pairs,
                similarities=similarities,
                classes=class_pairs,
                indices=image_indices,
                path=f"{output_path}/val/false_negatives",
                filename=f"{training_iter}_least_similar.png"
            )

def calculate_accuracy_with_threshold(
    threshold: float,
    similarities_all: torch.Tensor,
    same_or_diff_all: torch.Tensor,
    monitor: TrainingMonitor
):
    correct_all = ((similarities_all >= threshold) == same_or_diff_all).sum().item()
    correct_positive_all = ((similarities_all >= threshold) & same_or_diff_all).sum().item()
    correct_negative_all = ((similarities_all < threshold) & ~same_or_diff_all).sum().item()
    total_positive_all = same_or_diff_all.sum().item()
    total_all = (same_or_diff_all.size(0) * (same_or_diff_all.size(0) - 1))

    # calculate precision, recall, f1 score
    precision = correct_positive_all / (correct_positive_all + correct_negative_all)
    recall = correct_positive_all / total_positive_all
    f1_score = 2 * (precision * recall) / (precision + recall)

    monitor.add_value(f"val{threshold:.2f}_precision", precision)
    monitor.add_value(f"val{threshold:.2f}_recall", recall)
    monitor.add_value(f"val{threshold:.2f}_f1_score", f1_score)

    val_accuracy = correct_all / total_all
    monitor.add_value(f"val{threshold:.2f}_accuracy", val_accuracy)


def load_model(path: str, device = 'cpu') -> tuple[NetUtils, int]:
    """Load model from the last checkpoint in the given path.
    Return the model and the number of trained steps."""
    assert path is not None, "Output path must not be None."
    assert os.path.exists(path), f"Path {path} does not exist. Create it with the config.json file including model_class key."

    config_path = os.path.join(path, 'config.json')
    assert os.path.exists(config_path), f"Config file {config_path} does not exist. Create it with the model_class key."
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    model_name = utils.find_last_model(path)
    if not model_name:  # no previous checkpoint found
        model = init_model_class(model_config)
        logging.info(f'Created new model of class {model.__class__.__name__}.')
        return model, 0

    trained_steps = 0
    match_obj = re.match(rf'\S+_(\d+).pth', model_name)
    if match_obj:
        trained_steps = int(match_obj.groups(1)[0])

    model = init_model_class(model_config)
    model.load_state_dict(torch.load(os.path.join(path, model_name), map_location=device))
    model.to(device)
    logging.info(f'Loaded model from {os.path.join(path, model_name)} with {trained_steps} steps.')

    return model, trained_steps

def init_model_class(model_config):
    # if not utils.class_exists(model_config['model_class']):
    #     raise ValueError(f'Class {model_config["model_class"]} does not exist.')

    try: 
        model = eval(model_config['model_class']).from_config(model_config)
    except Exception as e:
        raise ValueError(f'Model class {model_config["model_class"]} does not exist or does not have from_config method.')
    return model

def save_model(model: NetUtils, output_path: str, trained_steps: int):
    model_path = os.path.join(output_path, f'{model.__class__.__name__}_{trained_steps}.pth')
    torch.save(model.state_dict(), model_path)

    # save config
    config_path = os.path.join(output_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(model.config, f, indent=4)

    return model_path

def render_batch_images(images: torch.Tensor, path: str, filename: str):
   
    os.makedirs(path, exist_ok=True)
    
    images_np = images.cpu().numpy()
    images_np = images_np.transpose(0, 2, 3, 1)
    
    images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())
    images_np = (images_np * 255).astype(np.uint8)
    
    batch_size = images_np.shape[0]
    grid_side = math.ceil(math.sqrt(batch_size))
    grid_size = (grid_side, grid_side)
    
    rows, cols = grid_size
    img_height, img_width, img_channels = images_np.shape[1:]
    canvas_height = rows * img_height
    canvas_width = cols * img_width
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    
    for idx, image in enumerate(images_np):
        row, col = divmod(idx, cols)
        if row >= rows:
            break
        img = Image.fromarray(image)
        canvas.paste(img, (col * img_width, row * img_height))
    
    output_file = os.path.join(path, filename)
    canvas.save(output_file)

def render_pairs(pairs, similarities, classes, indices, path, filename):

    os.makedirs(path, exist_ok=True)
    
    num_pairs = len(pairs)

    image_height, image_width = pairs[0][0].shape[1], pairs[0][0].shape[2]
    text_height = 30
    total_height = num_pairs * (image_height + text_height)
    total_width = image_width * 2

    canvas = Image.new("RGB", (total_width, total_height), color=(0, 0, 0))
    font = ImageFont.load_default()

    for idx, ((image1, image2), similarity, (class1, class2), (index1, index2)) in enumerate(zip(pairs, similarities, classes, indices)):
        image1 = image1.cpu().numpy().transpose(1, 2, 0)
        image2 = image2.cpu().numpy().transpose(1, 2, 0)

        image1 = (image1 - image1.min()) / (image1.max() - image1.min()) * 255
        image2 = (image2 - image2.min()) / (image2.max() - image2.min()) * 255
        image1 = image1.astype(np.uint8)
        image2 = image2.astype(np.uint8)

        image1_pil = Image.fromarray(image1)
        image2_pil = Image.fromarray(image2)

        concatenated_image = Image.new("RGB", (image_width * 2, image_height))
        concatenated_image.paste(image1_pil, (0, 0))
        concatenated_image.paste(image2_pil, (image_width, 0))

        y_offset = idx * (image_height + text_height)
        canvas.paste(concatenated_image, (0, y_offset))

        draw = ImageDraw.Draw(canvas)
        text = f"similarity: {similarity:.2f} | id1: {class1} | id2: {class2} | index1: {index1} | index2: {index2}"
        text_position = (10, y_offset + image_height + 5)
        draw.text(text_position, text, fill=(255, 255, 255), font=font)

    canvas.save(f"{path}/{filename}")

if __name__ == "__main__":
    main()