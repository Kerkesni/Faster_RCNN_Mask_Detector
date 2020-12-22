import numpy as np

# Calculates the IoU between two bounding boxes returns 1 or 0
def InersectionOverUnion(boxA, boxB, threshold = 0.5):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return 1 if iou > threshold else 0

# Calculates Average Precision for a class
def getAPForClass(predictions_boxes, score, truth_boxes):
    GTP = len(predictions_boxes)
    AP = 0
    if len(predictions_boxes) > 0:
        predictions_boxes = predictions_boxes[0].numpy()
    if len(truth_boxes) > 0:
        truth_boxes = truth_boxes[0].numpy()
    if len(score) > 0:
        score = score[0].numpy()

    for truth_box in truth_boxes:
        for index, pred_box in enumerate(predictions_boxes):
            AP += score[index]*InersectionOverUnion(pred_box, truth_box)
    return AP/GTP
    
# Calculates Average Precision for all classes
def getAPForAll(predictions, truth, classes = 3):
    AP = []
    for cl in range(1, classes+1):

        pred_labels = predictions['labels'].to('cpu')
        pred_boxes = predictions['boxes'].to('cpu')
        pred_score = predictions['scores'].to('cpu')

        class_indexes_truth = np.argwhere(truth['labels'] == cl)
        class_indexes_pred = np.argwhere(pred_labels == cl)

        class_boxes_truth = [truth['boxes'][i] for i in class_indexes_truth]
        class_boxes_pred = [pred_boxes[i] for i in class_indexes_pred]

        class_score_pred = [pred_score[i] for i in class_indexes_pred]

        AP.append(getAPForClass(class_boxes_pred, class_score_pred, class_boxes_truth))
    
    return AP



