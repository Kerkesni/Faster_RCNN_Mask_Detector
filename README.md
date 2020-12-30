# Faster_RCNN_Mask_Detector
## Description
Faster RCNN model trained to detect whether people are wearing masks correctly, incorrectly or if they are not wearing it at all.

## Model Used
A Faster R-CNN model, pre-trained on the COCO dataset

## Dataset Used for training
- [Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection)

## Results after 20 epochs
![Detection Results](./results/output.gif)

| Classes | mAP |
| ----------- | ----------- |
| With_Mask | 0.87 |
| Without_Mask | 0.73 |
| mask_weared_incorrect | 0.46 |