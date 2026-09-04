"""
The mechanism test: freeze the trained decoder entirely (as it will be
shipped) and, for photos the network never saw during training, optimize
only the latent vector via gradient descent to best reproduce each photo.

This mirrors exactly what runs client-side in the browser (see
../decoder-manifest.json and the TensorFlow.js port in ../index.html) --
running it here in PyTorch is how the mechanism gets validated before
trusting the JavaScript port, and it's also a useful quality regression
check after retraining: if mean_search_mse creeps up, something in the
architecture or training run got worse.
"""
import argparse
import os

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from model import Autoencoder, RES, CH, LATENT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, "data")


def load_val_images(n):
    tfm = transforms.ToTensor()
    train = torchvision.datasets.STL10(root=DATA_DIR, split="train", download=True, transform=tfm)
    test = torchvision.datasets.STL10(root=DATA_DIR, split="test", download=True, transform=tfm)
    data = torch.utils.data.ConcatDataset([train, test])
    total = len(data)
    _, val_set = torch.utils.data.random_split(
        data, [total - 40, 40], generator=torch.Generator().manual_seed(0))
    return torch.stack([val_set[i][0] for i in range(n)]).to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(OUT_DIR, "cnn_weights.pt"))
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "latent_search_result.png"))
    args = ap.parse_args()

    model = Autoencoder().to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    targets = load_val_images(args.n_images)
    n = targets.shape[0]

    with torch.no_grad():
        z_enc = model.encode(targets)
        out_enc = model.decode(z_enc)
        mse_enc = F.mse_loss(out_enc, targets, reduction="none").mean(dim=[1, 2, 3])

    latent = torch.zeros(n, LATENT, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([latent], lr=args.lr)
    for t in range(1, args.iters + 1):
        opt.zero_grad()
        out = model.decode(latent)
        loss = F.mse_loss(out, targets)
        loss.backward()
        opt.step()
        if t % 200 == 0 or t == args.iters:
            print(f"iter {t:4d}  loss={loss.item():.5f}", flush=True)

    with torch.no_grad():
        out_search = model.decode(latent)
        mse_search = F.mse_loss(out_search, targets, reduction="none").mean(dim=[1, 2, 3])

    print("per-image mse (latent-search-only vs via-encoder):", flush=True)
    for i in range(n):
        print(f"  img {i}: search={mse_search[i].item():.5f}  encoder={mse_enc[i].item():.5f}", flush=True)
    print(f"mean mse: latent-search-only={mse_search.mean().item():.5f}  "
          f"via-encoder={mse_enc.mean().item():.5f}", flush=True)

    from PIL import Image
    grid = Image.new("RGB", (RES * n, RES * 3))
    for i in range(n):
        orig = (targets[i].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        search = (out_search[i].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        enc = (out_enc[i].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        grid.paste(Image.fromarray(orig), (i * RES, 0))
        grid.paste(Image.fromarray(search), (i * RES, RES))
        grid.paste(Image.fromarray(enc), (i * RES, RES * 2))
    grid.save(args.out)
    print(f"saved {args.out} (rows: original | latent-search-only | via-encoder)", flush=True)


if __name__ == "__main__":
    main()
