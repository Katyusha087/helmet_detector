import os

import torch
from tensorboard import summary
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

from model import get_model
from train_utils import train, evaluate
from dataset import ObjectDetectionDataset, collate_fn
from visualize_results import plot_metrics, visualize_predictions
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration
model_name = 'ssd'  # Options: 'faster_rcnn', 'ssd'
num_epochs = 5
batch_size = 12
learning_rate = 1e-4
iou_threshold = 0.5
score_threshold = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
train_dataset = ObjectDetectionDataset("train", model_name, transform=ToTensor())
val_dataset = ObjectDetectionDataset("val", model_name, transform=ToTensor())


os.makedirs(f"output/{model_name}", exist_ok=True)

for i in range(5):
    image, target = train_dataset[i]
    image = ToPILImage()(image)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    boxes = target['boxes']
    labels = target['labels']

    for k, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin, str(labels[k].item()), color='white',
                 bbox=dict(facecolor='red', edgecolor='none', boxstyle='round,pad=0.2'))
    fig.savefig(os.path.join(f"output/{model_name}", f"image_{i+1}.png"))
    plt.close(fig)


print(f"Train dataset: {len(train_dataset)}")
print(f"Val dataset: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = get_model(model_name)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
train_f1_scores = []
val_f1_scores = []

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    train_loss, train_precision, train_recall, train_f1_score = evaluate(model, device,
                                                                         train_loader,
                                                                         iou_threshold=iou_threshold)
    val_loss, val_precision, val_recall, val_f1_score = evaluate(model, device, val_loader,
                                                                 iou_threshold=iou_threshold)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_f1_scores.append(train_f1_score)
    val_f1_scores.append(val_f1_score)

metrics_df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Train Loss': train_losses,
    'Val Loss': val_losses,
    'Train F1-score': train_f1_scores,
    'Val F1-score': val_f1_scores
})

metrics_df.to_csv(f"output/{model_name}/metrics.csv", index=False)

torch.save(model.state_dict(), f"output/{model_name}/model.pth")
