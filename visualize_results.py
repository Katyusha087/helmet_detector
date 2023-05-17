import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
import numpy as np
import random


def plot_metrics(train_losses, val_losses, train_metric_data, val_metric_data, metric_names, model_name):
    epochs = len(train_losses)

    output_dir = f"output/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    for i, (train_metrics, val_metrics, metric_name) in enumerate(zip(train_metric_data, val_metric_data, metric_names),
                                                                  start=1):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color='blue')
        plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss vs. Epoch")

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_metrics, label=f"Train {metric_name}", color='blue')
        plt.plot(range(1, epochs + 1), val_metrics, label=f"Val {metric_name}", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.title(f"{metric_name} vs. Epoch")

        plt.savefig(f"{output_dir}/{metric_name}_vs_epoch.png")
        plt.close()



def visualize_predictions(images, targets, outputs, model_name, num_samples=5, score_threshold=0.5, class_names=None,
                          start_index=0):
    images = [(img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for img in images]

    output_dir = f"output/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    if class_names is None:
        class_names = {1: "helmet", 2: "head"}
    for i in range(min(num_samples, len(images))):
        image = images[i]
        gt_boxes = targets[i]['boxes'].cpu().numpy()
        gt_labels = targets[i]['labels'].cpu().numpy()
        pred_boxes = outputs[i]['boxes'].detach().cpu().numpy()
        pred_labels = outputs[i]['labels'].detach().cpu().numpy()
        pred_scores = outputs[i]['scores'].detach().cpu().numpy()
        # Отсечь предсказанные ограничивающие рамки с низкими оценками
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        # Преобразовать метки классов в имена классов
        gt_labels = [class_names[label] for label in gt_labels]
        pred_labels = [f"{class_names[label]} ({score:.2f})" for label, score in zip(pred_labels, pred_scores)]

        # Визуализировать ограничивающие рамки для истинных значений и предсказаний
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # Преобразование изображения в тензор с форматом CxHxW
        gt_colors = ["green"] * len(gt_boxes)
        pred_colors = ["red"] * len(pred_boxes)
        image_with_gt = draw_bounding_boxes(image_tensor, torch.tensor(gt_boxes, dtype=torch.int64), labels=gt_labels,
                                            colors=gt_colors)
        image_with_pred = draw_bounding_boxes(image_tensor, torch.tensor(pred_boxes, dtype=torch.int64),
                                              labels=pred_labels, colors=pred_colors)

        # Отображение изображений
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image_with_gt.permute(1, 2, 0).numpy())
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(image_with_pred.permute(1, 2, 0).numpy())
        plt.title(f"Predictions ({model_name})")
        plt.axis("off")

        # Сохранить изображение
        plt.savefig(f"{output_dir}/prediction_{start_index + i + 1}.png")
        plt.close()
