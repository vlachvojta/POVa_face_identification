import os
import sys
import logging
import argparse
import time
import random
from typing import Optional

import torch
import cv2
import numpy as np
from collections import defaultdict

# add parent of this file to path to enable importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.DEBUG)

# from organizer.image_backbone import ImageBackbone
# from organizer.dataset import LayoutDataset
# from organizer.model import MultimodalLayoutTransformerConfig, MultimodalLayoutTransformer
# from organizer.augmentation import OnlineBBoxAugmentor
# from organizer.utils import find_shortest_path, render_reading_order, transform_order_to_successor

from clearml import Task


def parse_arguments():
    parser = argparse.ArgumentParser(usage="Train a model for face identification.")

    # ClearML
    parser.add_argument("--name", type=str, default="vlach_face_identification",
                        help="Task name for ClearML.")
    parser.add_argument("--project-name", required=False, type=str, help="Project name for ClearML.")

    # Data paths
    parser.add_argument("-d", "--dataset-path", required=True, type=str, help="Path to the dataset folder.")
    parser.add_argument("--config", type=str, default=None, help="Path to model config")
    parser.add_argument("--render", action="store_true", help="Render validation samples.")

    # Trainer settings
    parser.add_argument("--max-iter", default=100_000, type=int)
    parser.add_argument("--view-step", default=50, type=int, help="Number of training iterations between validation.")

    # Optimization settings
    parser.add_argument("--batch-size", "-b", default=64, type=int)
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

    # if args.seed is not None:
    #     logging.debug(f"Settings random seed to {args.seed}")
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on: {device}")

    # bbox_augmentor = None
    # if args.augment_bboxes:
    #     logging.debug(f"Creating bbox augmentor with sigma={args.sigma} and ratio={args.ratio} ...")
    #     bbox_augmentor = OnlineBBoxAugmentor(sigma=args.sigma, ratio=args.ratio)
    #     logging.debug("Bbox augmentor created.")

    if args.config is not None:
        logging.debug("Config specified, creating from file. Ommiting all manual settings.")
        config = MultimodalLayoutTransformerConfig.from_file(args.config)
    else:
        logging.error("No config specified, creating from manual settings.")
        # logging.debug("Creating config from manual settings ...")
        # config = MultimodalLayoutTransformerConfig(
        #     input_dropout=args.input_dropout,
        #     layers=args.layers,
        #     dim=args.dim,
        #     heads=args.heads,
        #     dropout=args.dropout,
        #     residual_input=args.residual_input,
        #     positional_embedding=args.pos_embedding,
        #     pos_table_size=args.table_size,
        #     pos_strategy=args.pos_strategy,
        #     image_backbone=args.image_backbone,
        #     image_backbone_depth=args.image_backbone_depth,
        #     freeze_image_backbone=args.freeze_image_backbone,
        #     image_width=args.image_width,
        #     image_height=args.image_height,
        #     max_bbox_count=args.max_bbox_count,
        #     use_query_type=args.query_type,
        #     class_queries=args.class_queries,
        #     use_img_input=args.use_img,
        #     image_encoder=args.image_encoder,
        #     layers_encoder=args.layers_encoder,
        #     heads_encoder=args.heads_encoder,
        #     dropout_encoder=args.dropout_encoder,
        # )
        config.save()
    logging.debug("Config created.")
    print(config)

    logging.debug("Creating train dataset ...")
    trn_dataset = LayoutDataset(
        xml_path=args.xml_train,
        img_path=args.img,
        max_bbox_count=config.max_bbox_count,
        bbox_augmentor=bbox_augmentor,
        image_size=(config.image_height, config.image_width),
        line_based=args.line_based,
    )
    logging.debug(f"Train dataset created. trn_samples: {len(trn_dataset)}")

    logging.debug("Creating validation dataset ...")
    val_dataset = LayoutDataset(
        xml_path=args.xml_val,
        img_path=args.img,
        max_bbox_count=config.max_bbox_count,
        image_size=(config.image_height, config.image_width),
        line_based=args.line_based,
    )
    logging.debug(f"Validation dataset created. val_samples: {len(val_dataset)}")

    trn_data_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, persistent_workers=True, num_workers=4)

    val_data_loaders = {
        # 'trn': torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=False, persistent_workers=True, num_workers=2),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, persistent_workers=True, num_workers=2)
    }

    # image_backbone = None
    # if config.use_img_input:
    #     logging.debug(f"Creating image backbone with name: {config.image_backbone} ...")
    #     image_backbone = ImageBackbone(
    #         name=config.image_backbone,
    #         depth=config.image_backbone_depth,
    #         device=device,
    #         image_size=(config.image_height, config.image_width),
    #     )
    #     logging.debug("Image backbone created.")

    logging.debug("Creating model ...")
    # TODO create model
    # model = MultimodalLayoutTransformer(
    #     config=config,
    #     device=device,
    #     image_backbone=image_backbone,
    # )
    model = model.to(device)
    model.train()
    logging.debug("Model created.")
    print(model)

    logging.info("Starting training ...")
    monitor = TrainingMonitor(name=args.name, project_name=args.project_name)
    try:
        train(
            model=model,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_dataloader=trn_data_loader,
            val_data_loaders=val_data_loaders,
            max_iter=args.max_iter,
            view_step=args.view_step,
            # loss_aggregation=args.loss_aggregation,
            render=args.render,
            monitor=monitor,
            # max_test_samples=args.max_test_samples,
        )
        logging.info("Training finished.")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")

    logging.info("DONE")


class TrainingMonitor:
    def __init__(
        self,
        name: Optional[str]=None,
        project_name: Optional[str]=None,
    ):
        self.iterations: list[int] = []
        self.values: dict[str, list[float]] = defaultdict(list)

        self.name = name if name is not None else "training"
        self.project_name = project_name
        self.log_name = f"{self.project_name}_{self.name}" if self.project_name is not None else self.name

        self.task = Task.init(project_name=project_name, task_name=name, continue_last_task=True) if project_name is not None else None

    def add_value(
        self,
        key: str,
        value: float,
    ) -> None:
        self.values[key].append(value)
        if self.task is not None:
            series, title = key.split("_", 1)
            self.task.logger.report_scalar(title=title, series=series, value=value, iteration=self.iterations[-1])

    def report_results(self, digits: int=4) -> None:
        if self.task is None:
            return

        for k, v in self.values.items():
            if "loss" in k:
                self.task.logger.report_single_value(k, round(min(v), digits))
            elif "acc" in k:
                self.task.logger.report_single_value(k, round(max(v), digits))
            else:
                continue

    def save_csv(self, path: str, ) -> None:
        p = os.path.join(path, f"{self.log_name}.csv")
        with open(p, "w") as f_csv:
            keys = sorted(list(self.values.keys()))
            f_csv.write("iteration," + ",".join(keys) + "\n")
            for i, it in enumerate(self.iterations):
                f_csv.write(f"{it}," + ",".join([str(self.values[k][i]) for k in keys]) + "\n")

    # def load_from_csv(self, path: str) -> None:
    #     p = os.path.join(path, f"{self.log_name}.csv")
    #     with open(p, "r") as f_csv:
    #         lines = f_csv.readlines()
    #         keys = lines[0].strip().split(",")[1:]
    #         for line in lines[1:]:
    #             parts = line.strip().split(",")
    #             self.iterations.append(int(parts[0]))
    #             for i, k in enumerate(keys):
    #                 self.values[k].append(float(parts[i+1]))

    def get_last_string(self):
        return f"ITER {self.iterations[-1]} " + " ".join([f"{k} {v[-1]:.4f}" for k, v in self.values.items()])


def train(
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    val_data_loaders: dict[str, torch.utils.data.DataLoader],
    lr: float,
    weight_decay: float,
    max_iter: int,
    view_step: int,
    monitor: TrainingMonitor,
    # loss_aggregation: str,
    render: bool=False,
    # max_test_samples: int=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none").to(device)

    iteration = 0
    bboxes_seen = 0
    pages_seen = 0
    bboxes_predicted_right = 0

    page_accuracies = []
    train_loss = []

    t1 = time.time()

    for epoch in range(10000):
        if iteration > max_iter:
            break
        for batch_data in train_dataloader:
            if iteration > max_iter:
                break
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

            train_loss.extend(loss.tolist())

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
                monitor.add_value(f"train_loss", sum(train_loss) / len(train_loss))
                monitor.add_value("train_acc_region", bboxes_predicted_right / bboxes_seen)
                monitor.add_value("train_acc_page", sum(page_accuracies) / pages_seen)

                train_loss = []
                page_accuracies = []
                bboxes_predicted_right = 0
                bboxes_seen = 0
                pages_seen = 0

                for val_name, val_data_loader in val_data_loaders.items():
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
            cv2.imwrite(os.path.join(folder, f"{iteration:06d}_{batch_id:03d}_{worst_sample_acc:.3f}.jpg"), img)
            cv2.imwrite(os.path.join(folder, f"{iteration:06d}_{batch_id:03d}_{worst_sample_acc:.3f}_soft.jpg"), img_soft)

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





if __name__ == "__main__":
    main()