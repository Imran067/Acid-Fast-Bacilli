import torch
from torch.utils.data import DataLoader
from dataset import AFBDataset
from model import get_model
from tqdm import tqdm

def train_model(
    tile_img_dir,
    tile_lbl_dir,
    epochs=20,
    batch_size=4,
    lr=1e-4,
    model_out="afb_fcos.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = AFBDataset(tile_img_dir, tile_lbl_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, targets in tqdm(loader):
            imgs = [i.to(device) for i in imgs]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

            losses = model(imgs, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss = {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), f"afb_fcos{epoch}.pth")

    torch.save(model.state_dict(), model_out)
    print(f"Model saved to {model_out}")
