import os
import sys
import logging
import argparse
import time
# import random
from typing import Optional
from collections import defaultdict
import json
import re

import numpy as np
import torch
import cv2
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
# from clearml import Task

# add parent of this file to path to enable importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_loader import DataLoader as CelebADataLoader
from datasets.data_loader import Partition
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
    parser.add_argument("--project-name", required=False, type=str, help="xvlach_POVa_face_identification")

    # Data paths
    parser.add_argument("-d", "--dataset-path", required=True, type=str, help="Path to the dataset folder.")
    parser.add_argument("-o", "--output-path", required=True, type=str, help="Path to the output folder.")
    parser.add_argument("--config", type=str, default=None, help="Path to model config")
    parser.add_argument("--render", action="store_true", help="Render validation samples.")

    # Trainer settings
    parser.add_argument("--max-iter", default=100_000, type=int)
    parser.add_argument("--view-step", default=50, type=int, help="Number of training iterations between validation.")

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
    # this is a script taken from the layout organizer project but now I am modifying it to work with the face identification project
    args = parse_arguments()
    logging.info(args)
    # logging.getLogger("cv2").setLevel(level=logging.ERROR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on: {device}")

    logging.info("Loading datasets ...")  # TODO put this back on after testing
    trn_dataset = CelebADataLoader(args.dataset_path, partition=Partition.TRAIN)
    val_dataset = CelebADataLoader(args.dataset_path, partition=Partition.VAL, limit=100)
    logging.info(f"Train dataset:      {len(trn_dataset)} samples")
    logging.info(f"Validation dataset: {len(val_dataset)} samples")

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
    monitor = TrainingMonitor(name=args.name, project_name=args.project_name)
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
            render=args.render,
            monitor=monitor,
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
    monitor: TrainingMonitor,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    iteration = trained_steps

    train_losses = []

    t1 = time.time()

    print(f'exiting'); exit(0)

    
    for batch_data in trn_dataloader:
        if iteration > max_iter:
            break

        model.train()
        pages_seen += batch_data["bbox"].size()[0]

        iteration += 1

        bboxes = batch_data["bbox"].to(device)
        query_types = batch_data["query_type"].to(device)
        gt_bbox = batch_data["gt_bbox"].to(device)
        images = None
        mask = gt_bbox != -1
        
        if model.config.use_img_input:
            images = batch_data["image"].to(device)

        optimizer.zero_grad()

        pred, _ = model(
            x=bboxes,
            images=images,
            query_types=query_types,
        )
        loss = criterion(pred, gt_bbox)
        # aggregated_loss = aggregate_loss(loss, mask, loss_aggregation)
        # loss = aggregated_loss.mean()
        loss.backward()
        optimizer.step()

        train_losses.extend(loss.tolist())

        # Compute prediction accuracy, but ignore the -1 labels
        pred_labels = torch.argmax(pred, dim=1)
        bboxes_seen += torch.sum(mask).item()
        bboxes_predicted_right += torch.sum(pred_labels[mask] == gt_bbox[mask]).item()

        # Page accuracy
        page_predictions = (pred_labels == gt_bbox) & mask
        correct_per_page = torch.sum(page_predictions, dim=1)
        valid_bboxes_per_page = torch.sum(mask, dim=1)
        page_acc = correct_per_page / valid_bboxes_per_page
        page_accuracies.extend(page_acc.tolist())

        if iteration % view_step == 0:
            t2 = time.time()
            assert len(page_accuracies) == pages_seen

            monitor.iterations.append(iteration)
            monitor.add_value(f"train_loss", sum(train_losses) / len(train_losses))
            monitor.add_value("train_acc_region", bboxes_predicted_right / bboxes_seen)
            monitor.add_value("train_acc_page", sum(page_accuracies) / pages_seen)

            train_losses = []
            page_accuracies = []
            bboxes_predicted_right = 0
            bboxes_seen = 0
            pages_seen = 0

            for val_name, val_data_loader in val_dataloaders.items():
                validate(
                    model=model,
                    data_loader=val_data_loader,
                    data_loader_name=val_name,
                    criterion=criterion,
                    device=device,
                    monitor=monitor,
                    # loss_aggregation=loss_aggregation,
                    render=render,
                    # max_test_samples=max_test_samples,
                )

            monitor.add_value("view_time", t2 - t1)
            log_string = monitor.get_last_string()
            print(f"{log_string}")

            torch.save(model.state_dict(), "./last.pth")
            
            monitor.report_results()
            monitor.save_csv(".")
            t1 = time.time()


def validate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    data_loader_name: str,
    criterion: torch.nn.Module,
    device: torch.device,
    monitor: TrainingMonitor,
    # loss_aggregation: str,
    render: bool=False,
    # max_test_samples: int=None,
):
    model.eval()

    pages_seen = 0
    bboxes_seen = 0
    bboxes_predicted_right = 0

    page_accuracies = []
    val_loss = []

    for batch_id, batch_data in enumerate(data_loader):
        pages_seen += batch_data["bbox"].shape[0]

        bboxes = batch_data["bbox"].to(device)
        query_types = batch_data["query_type"].to(device)
        images = None
        gt_bbox = batch_data["gt_bbox"].to(device)
        mask = gt_bbox != -1

        if model.config.use_img_input or render:
            images = batch_data["image"].to(device)

        with torch.no_grad():
            pred, _ = model(
                x=bboxes,
                images=images,
                query_types=query_types,
            )
            loss = criterion(pred, gt_bbox)
            # aggregated_loss = aggregate_loss(loss, mask, loss_aggregation)

        val_loss.extend(loss.tolist())

        pred_labels = torch.argmax(pred, dim=1)
        valid_bbox_count = torch.sum(mask).item()
        bboxes_seen += valid_bbox_count

        # Bbox accuracy
        bboxes_predicted_right += torch.sum(pred_labels[mask] == gt_bbox[mask]).item()

        # Page accuracy
        page_predictions = (pred_labels == gt_bbox) & mask
        correct_per_page = torch.sum(page_predictions, dim=1)
        valid_bboxes_per_page = torch.sum(mask, dim=1)
        page_acc = correct_per_page / valid_bboxes_per_page
        page_accuracies.extend(page_acc.tolist())

        if render:
            worst_sample_idx = int(torch.argmin(page_acc).item())
            worst_sample_acc = float(page_acc[worst_sample_idx])
            valid_count = torch.sum(mask[worst_sample_idx]).item()
            worst_sample_bboxes = bboxes[worst_sample_idx][:valid_count]
            worst_sample_img = images[worst_sample_idx]

            # # Hard predictions
            # worst_sample_preds = pred_labels[worst_sample_idx][:valid_count]
            # img = render_reading_order(
            #     image=worst_sample_img,
            #     bboxes=worst_sample_bboxes,
            #     preds=worst_sample_preds,
            # )

            # # Soft predictions
            # prob = torch.nn.functional.softmax(pred[worst_sample_idx][:valid_count, :valid_count], dim=0)
            # order = find_shortest_path(1-prob.cpu().numpy().T)
            # order = transform_order_to_successor(order)
            # img_soft = render_reading_order(
            #     image=worst_sample_img,
            #     bboxes=worst_sample_bboxes,
            #     preds=order,
            # )

            iteration = monitor.iterations[-1]
            folder = os.path.join(".", f"step-{iteration}_{data_loader_name}")
            os.makedirs(folder, exist_ok=True)
            # cv2.imwrite(os.path.join(folder, f"{iteration:06d}_{batch_id:03d}_{worst_sample_acc:.3f}.jpg"), img)
            # cv2.imwrite(os.path.join(folder, f"{iteration:06d}_{batch_id:03d}_{worst_sample_acc:.3f}_soft.jpg"), img_soft)

        # if max_test_samples is not None and pages_seen >= max_test_samples:
        #     break

    bbox_accuracy = bboxes_predicted_right / bboxes_seen
    page_accuracy = sum(page_accuracies) / pages_seen

    monitor.add_value(f"{data_loader_name}_loss", sum(val_loss) / len(val_loss))
    monitor.add_value(f"{data_loader_name}_acc_region", bbox_accuracy)
    monitor.add_value(f"{data_loader_name}_acc_page", page_accuracy)

    if data_loader_name == "tst" and len(monitor.values["tst_acc_page"]) > 1 and page_accuracy > max(monitor.values["tst_acc_page"][:-1]):
        logging.info(f"Found new best model with page accuracy {page_accuracy:.4f}, saving.")
        torch.save(model.state_dict(), "./best.pth")

    model.train()


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


if __name__ == "__main__":
    main()