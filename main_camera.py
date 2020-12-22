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

weights_path = './weights/weights.pt'

# Setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# Loading model
model = loadModelWithWeights(weights_path)
model = model.to(device)

model.eval()
with torch.no_grad():

    convert = T.ToTensor()
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        image = convert(Image.fromarray(frame)).to(device)
        bboxes = model([image])
       
        bounding_boxes = bboxes[0]['boxes'].to('cpu').numpy().astype(int)
        box_lables = bboxes[0]['labels'].to('cpu').numpy()
        box_scores = bboxes[0]['scores'].to('cpu').numpy()

        args_c1 = np.intersect1d(np.argwhere(box_scores > 0.8), np.argwhere(box_lables == 1))
        args_c2 = np.intersect1d(np.argwhere(box_scores > 0.8), np.argwhere(box_lables == 2))
        args_c3 = np.intersect1d(np.argwhere(box_scores > 0.45), np.argwhere(box_lables == 3))

        args = np.concatenate((args_c1, args_c2, args_c3))

        for index in args:
            box_class = box_lables[index]
            xmin, ymin, xmax, ymax = bounding_boxes[index].flatten()
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), classes_color[classes_index[box_class]], 1) 
            frame = cv2.putText(frame, classes_index[box_class], (xmax-xmin, ymin), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, classes_color[classes_index[box_class]], 1) 
            frame = cv2.putText(frame, f'{box_scores[index]*100:.2f}%', (xmax-xmin, ymax), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, classes_color[classes_index[box_class]], 1) 

        cv2.imshow('frame', frame)
        cv2.waitKey(100)





