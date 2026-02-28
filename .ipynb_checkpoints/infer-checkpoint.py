import cv2
import torch
from model import get_model
from utils import merge_boxes
from tile import TILE, OVERLAP

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model().to(device)
model.load_state_dict(torch.load("afb_fcos.pth"))
model.eval()

img = cv2.imread("data/images/afb_351ua.bmp")
H,W,_ = img.shape

all_boxes, all_scores = [], []

for y in range(0, H-TILE, TILE-OVERLAP):
    for x in range(0, W-TILE, TILE-OVERLAP):
        tile = img[y:y+TILE, x:x+TILE]
        t = torch.tensor(tile/255.).permute(2,0,1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(t)[0]

        for b,s in zip(out["boxes"], out["scores"]):
            if s > 0.4:
                b = b.cpu().numpy()
                all_boxes.append([b[0]+x,b[1]+y,b[2]+x,b[3]+y])
                all_scores.append(s.cpu())

if len(all_boxes):
    boxes = torch.tensor(all_boxes)
    scores = torch.tensor(all_scores)
    boxes, scores = merge_boxes(boxes, scores)

    print("AFB detected:", len(boxes))
else:
    print("No AFB detected")
