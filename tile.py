import cv2
import os
import json

def create_tiles(
    image_dir,
    annotation_dir,
    out_img_dir,
    out_lbl_dir,
    tile_size=256,
    overlap=64
):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for fname in os.listdir(image_dir):
        if not fname.endswith("unannotated.bmp"):
            continue

        img_path = os.path.join(image_dir, fname)
        ann_path = os.path.join(
            annotation_dir, fname.replace("unannotated.bmp", "annotated.json")
        )

        img = cv2.imread(img_path)
        H, W, _ = img.shape

        boxes = []
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                boxes = json.load(f)["boxes"]

        for y in range(0, H - tile_size + 1, tile_size - overlap):
            for x in range(0, W - tile_size + 1, tile_size - overlap):

                tile = img[y:y+tile_size, x:x+tile_size]
                tile_boxes = []

                for b in boxes:
                    x1,y1,x2,y2 = b
                    if x1 < x+tile_size and x2 > x and y1 < y+tile_size and y2 > y:
                        tile_boxes.append([
                            max(0, x1-x), max(0, y1-y),
                            min(tile_size, x2-x), min(tile_size, y2-y)
                        ])

                tile_id = f"{fname[:-4]}_{x}_{y}"
                cv2.imwrite(
                    os.path.join(out_img_dir, tile_id + ".png"),
                    tile
                )

                with open(
                    os.path.join(out_lbl_dir, tile_id + ".json"), "w"
                ) as f:
                    json.dump(tile_boxes, f)
