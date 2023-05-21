import torch
from torch import nn
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint

def train(model, device, train_loader, optimizer, epoch, print_interval=10):
    model.train()
    total_loss = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if batch_idx % print_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss: {avg_loss:.6f}")


def evaluate(model, device, test_loader, iou_threshold=0.5, score_threshold=0.5):
    model.eval()

    num_true_positives = 0
    num_false_positives = 0
    num_false_negatives = 0

    bbox_loss = nn.SmoothL1Loss()
    cls_loss = nn.BCEWithLogitsLoss()
    total_loss = 0

    num_images_to_plot = 5
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for idx, output in enumerate(outputs):
                gt_boxes = targets[idx]['boxes']
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                pred_labels = output['labels']

                keep = pred_scores > score_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

                if pred_boxes.shape[0] == 0:
                    continue

                ious = box_iou(gt_boxes, pred_boxes)
                num_predictions = pred_boxes.shape[0]
                num_targets = gt_boxes.shape[0]
                matched_gt_boxes = torch.zeros_like(pred_boxes)
                matched_gt_labels = torch.zeros(num_predictions, dtype=torch.float32).to(pred_scores.device)

                for i in range(num_targets):
                    max_iou, max_j = ious[i].max(0)
                    if max_iou >= iou_threshold:
                        matched_gt_boxes[max_j] = gt_boxes[i]
                        matched_gt_labels[max_j] = 1.0
                        num_true_positives += 1
                    else:
                        num_false_negatives += 1

                num_false_positives += num_predictions - torch.sum(matched_gt_labels).item()

                current_loss = bbox_loss(pred_boxes, matched_gt_boxes) + cls_loss(pred_scores.unsqueeze(-1),
                                                                                  matched_gt_labels.unsqueeze(-1))
                total_loss += current_loss.item()
                metric = MeanAveragePrecision()

                preds = [
                    dict(
                        boxes=pred_boxes,
                        scores=pred_scores,
                        labels=pred_labels,
                    )
                ]

                target = [
                    dict(
                        boxes=matched_gt_boxes,
                        labels=matched_gt_labels,
                    )
                ]

                metric.update(preds, target)
                # print(bbox_loss(pred_boxes, matched_gt_boxes))
                # print(cls_loss(pred_scores.unsqueeze(-1), matched_gt_labels.unsqueeze(-1)))

    eps = 1e-6
    precision = num_true_positives / (num_true_positives + num_false_positives + eps)
    recall = num_true_positives / (num_true_positives + num_false_negatives + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    average_loss_per_epoch = total_loss / len(test_loader)
    map_score = metric.compute()
    print("Все метрики:")
    print(map_score)
    print("Mean Average Precision:", map_score.map)
    print(
        f"Average Loss: {average_loss_per_epoch:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
    return average_loss_per_epoch, precision, recall, f1_score, map_score.map
