"""
Trains the Afterimage autoencoder on STL-10 (13,000 general-purpose photos,
96x96 RGB). GPU strongly recommended -- see ../RUNBOOK.md for timing.

After training, run export_decoder.py to produce the browser-loadable
frozen decoder -- this script only produces the full checkpoint (both
encoder and decoder), which is not itself shipped anywhere.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from model import Autoencoder, RES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, "data")


def load_data():
    tfm = transforms.ToTensor()
    train = torchvision.datasets.STL10(root=DATA_DIR, split="train", download=True, transform=tfm)
    test = torchvision.datasets.STL10(root=DATA_DIR, split="test", download=True, transform=tfm)
    return torch.utils.data.ConcatDataset([train, test])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-size", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "cnn_weights.pt"))
    ap.add_argument("--preview-out", default=os.path.join(OUT_DIR, "cnn_recon_preview.png"))
    args = ap.parse_args()

    print(f"device: {DEVICE}", flush=True)
    data = load_data()
    n = len(data)
    train_set, val_set = torch.utils.data.random_split(
        data, [n - args.val_size, args.val_size], generator=torch.Generator().manual_seed(0))
    print(f"total={n} train={len(train_set)} val={len(val_set)}", flush=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_x = torch.stack([val_set[i][0] for i in range(len(val_set))]).to(DEVICE)

    model = Autoencoder().to(DEVICE)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_t0 = time.time()
        total_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            loss = F.mse_loss(out, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        train_mse = total_loss / len(train_set)

        model.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(model(val_x), val_x).item()
        print(f"epoch {epoch:4d}  train_mse={train_mse:.5f}  val_mse={val_mse:.5f}  "
              f"epoch_time={time.time()-ep_t0:.2f}s  total={time.time()-t0:.1f}s", flush=True)

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}", flush=True)

    model.eval()
    with torch.no_grad():
        out_val = model(val_x[:8]).cpu()
    from PIL import Image
    grid = Image.new("RGB", (RES * 8, RES * 2))
    for i in range(8):
        orig = (val_x[i].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        recon = (out_val[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        grid.paste(Image.fromarray(orig), (i * RES, 0))
        grid.paste(Image.fromarray(recon), (i * RES, RES))
    grid.save(args.preview_out)
    print(f"saved {args.preview_out} (top row = original, bottom row = reconstruction)", flush=True)


if __name__ == "__main__":
    main()
