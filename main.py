from dataset import getDatasets, getLoaders
from model import getModel, loadModelWithWeights
from train import train_for_epochs

from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

# Temporary
import cv2
import torchvision.transforms as T

# Set class  boxcolors
classes_color = {'with_mask':[0, 255, 0], 'without_mask':[0 ,0, 255], 'mask_weared_incorrect':[255, 165, 0]}
classes_index = {1:'with_mask', 2:'without_mask', 3:'mask_weared_incorrect'}

# Show image with bounding boxes that have a score > 0.8
def draw_results(image, lables):
    # Set class  boxcolors
    classes_color = {'with_mask':'g', 'without_mask':'r', 'mask_weared_incorrect':'tab:orange'}
    classes_index = {1:'with_mask', 2:'without_mask', 3:'mask_weared_incorrect'}
    # Transpose and plot image
    image = np.transpose(image, (1,2,0))
    # Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    # Adding Bonding Boxes to image
    bounding_boxes = lables[0]['boxes'].to('cpu').numpy()
    box_lables = lables[0]['labels'].to('cpu').numpy()
    box_scores = lables[0]['scores'].to('cpu').numpy()

    for index, box_coord in enumerate(bounding_boxes):
        if box_scores[index] < 0.8:
            continue
        box_class = box_lables[index]
        xmin, ymin, xmax, ymax = box_coord
        rect = patches.Rectangle((xmin,ymax), (xmax-xmin), -(ymax-ymin), linewidth=1, edgecolor=classes_color[classes_index[box_class]], facecolor='none')
        ax.add_patch(rect)
    plt.show()

data_folder = './Data'
root_folder = './'
weights_path = './weights/weights.pt'

# Setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Loading Datasets
train_dataset, val_dataset, test_dataset = getDatasets(data_folder)

# Loading model
model = loadModelWithWeights(weights_path)
model = model.to(device)

model.eval()
with torch.no_grad():
    image = train_dataset.__getitem__(0)[0].to(device)
    bboxes = model([image])
    # TODO : Show IoU
    # TODO calculate mAP
    draw_results(image.to('cpu'), bboxes)
