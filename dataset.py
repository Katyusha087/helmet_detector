import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize
from sklearn.model_selection import train_test_split


def collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets


class ObjectDetectionDataset(Dataset):
    def __init__(self, split, model_name, transform=None, root_dir="SafetyHelmetDetection", image_dir="images",
                 annot_dir="annotations", test_size=0.2):
        self.root = root_dir
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.split = split
        self.model_name = model_name
        self.transform = transform if transform else ToTensor()
        self.resize = Resize((416, 416))  # Resize to 416x416
        self.image_files = self.get_image_files()
        if self.split == 'train':
            self.image_files, _ = train_test_split(self.image_files, test_size=test_size, random_state=42)
        elif self.split == 'val':
            _, self.image_files = train_test_split(self.image_files, test_size=test_size, random_state=42)

    def get_image_files(self):
        image_files = sorted(os.listdir(os.path.join(self.root, self.image_dir)))
        return self.filter_images(image_files)

    def filter_images(self, image_files):
        return [image_file for image_file in image_files if self.has_valid_class(image_file)]

    def has_valid_class(self, image_file):
        annot_path = os.path.join(self.root, self.annot_dir, image_file.replace('.png', '.xml'))
        tree = ET.parse(annot_path)
        root = tree.getroot()
        return any(obj.find('name').text in ['helmet', 'head'] for obj in root.findall('object'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.resize(image)
        annot_path = os.path.join(self.root, self.annot_dir, self.image_files[idx].replace('.png', '.xml'))

        boxes, labels = self.get_boxes_and_labels(annot_path, image.size)
        assert boxes, f"Annotation file {annot_path} does not contain any labels"

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        image = self.transform(image)
        return image, target

    def get_boxes_and_labels(self, annot_path, img_size):
        tree = ET.parse(annot_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name == 'person':
                continue
            label = 1 if name == 'helmet' else 2 if name == 'head' else 0
            labels.append(label)

            bndbox = obj.find('bndbox')
            box = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ]
            boxes.append(box)
        return boxes, labels
