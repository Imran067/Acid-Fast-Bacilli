import torch
from torchvision.ops import nms

def merge_boxes(preds, scores, iou=0.3):
    keep = nms(preds, scores, iou)
    return preds[keep], scores[keep]
