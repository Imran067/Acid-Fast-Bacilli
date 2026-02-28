import os, json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class AFBDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.imgs = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.tf = T.ToTensor()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(f"{self.img_dir}/{img_name}").convert("RGB")
        img = self.tf(img)

        lbl_path = f"{self.lbl_dir}/{img_name.replace('.png','.json')}"
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                boxes = json.load(f)

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0,4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.ones(len(boxes), dtype=torch.int64)
            }

        return img, target
