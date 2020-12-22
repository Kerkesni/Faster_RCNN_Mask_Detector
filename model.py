import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def getModel():
    #load Faster R-CNN pre-trained on COCO dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #Change the classifier head with a new one:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 4 #(3 + background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def loadModelWithWeights(weights_path):
    model = getModel()
    model.load_state_dict(torch.load(weights_path))
    return model