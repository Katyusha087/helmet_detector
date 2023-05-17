import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights


def get_model(model_name):
    if model_name == 'faster_rcnn':
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 3
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                   num_classes)
    elif model_name == 'ssd':
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
        num_classes = 3
        for idx, block in enumerate(model.head.classification_head.children()):
            if isinstance(block, nn.Sequential):
                old_layer = block[-1]
                new_layer = nn.Conv2d(old_layer.in_channels, num_classes, kernel_size=old_layer.kernel_size,
                                      stride=old_layer.stride, padding=old_layer.padding)
                new_layer.weight.data.normal_(0, 0.01)
                new_layer.bias.data.zero_()
                block[-1] = new_layer
    else:
        raise ValueError('Invalid model name')

    return model
