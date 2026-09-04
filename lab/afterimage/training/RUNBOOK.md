# Retraining runbook

Everything here runs locally. No cloud credentials, no server-side storage of
any kind is involved at any step.

## 1. Set up an isolated environment

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```

Install pinned, known-working versions:
```bash
pip install -r requirements.lock.txt
```

`requirements.lock.txt` pins an NVIDIA CUDA 12.4 build of PyTorch (the
`+cu124` suffix). If you don't have an NVIDIA GPU, or have a different CUDA
version, install from `requirements.txt` instead and let pip pick the right
build for your machine (training will fall back to CPU and take
proportionally longer -- see timing below):

```bash
pip install -r requirements.txt
```

## 2. Train

```bash
python train.py --epochs 200
```

STL-10 (13,000 images, ~2.6GB) downloads automatically into `data/` on first
run and is cached there for subsequent runs. `data/` is gitignored -- never
commit it.

Timing on an RTX-class GPU: ~5-6s/epoch, so 200 epochs is roughly 20 minutes.
On CPU this will be dramatically slower (likely multiple hours) -- start with
a short smoke test before committing to a long run:

```bash
python train.py --epochs 5
```

Check `cnn_recon_preview.png` (top row = original photos, bottom row =
reconstructions) before trusting a longer run. `val_mse` should be
decreasing and comfortably below 0.02 by epoch 5; if it's diverging or stuck,
something is wrong before you spend the time on 200 epochs.

Output: `cnn_weights.pt` -- the full model (encoder + decoder). This file is
gitignored and never committed; it's an intermediate artifact, not the
shipped asset.

## 3. Validate the actual deployment mechanism

The site never uses the encoder -- it freezes the decoder and optimizes a
latent vector via gradient descent for each uploaded photo. Confirm that
still works well on the freshly trained decoder before shipping it:

```bash
python verify_latent_search.py
```

This prints, per held-out image, the reconstruction error from pure latent
search vs. from the paired encoder, and saves `latent_search_result.png`
(rows: original | latent-search-only | via-encoder). Latent search should be
at or below the encoder's error on most images -- if it's now consistently
*worse*, treat that as a regression and investigate before exporting.

## 4. Export the frozen decoder for the browser

```bash
python export_decoder.py
```

This fuses every ConvTranspose2d + BatchNorm2d pair into a single affine
transform, permutes every array from PyTorch's layout into TensorFlow.js's,
and writes:

- `../decoder-manifest.json` -- layer order, shapes, activations
- `../weights/*.bin` -- raw float32 arrays referenced by the manifest
- `reference_latent.json` + `reference_output.png` -- a fixed latent vector
  and its known-correct decoder output, for sanity-checking the JS port

The script re-runs the exported (fused, permuted) weights through a plain
NumPy forward pass and asserts the output matches the original model's
`decode()` almost exactly (max abs diff < 1e-4) before it lets you finish.
If that assertion fails, do not proceed to update the site -- something in
the fusion math or the layout permutes is wrong, and the exported weights
would silently produce different reconstructions in the browser than what
`verify_latent_search.py` just validated.

## 5. Update the site

Copy `decoder-manifest.json` and `weights/` into `lab/afterimage/` (already
their default location if you ran the scripts from `training/` as shown
above). Open `lab/afterimage/index.html` locally and confirm the reference
case still matches `reference_output.png` before considering the retrain
done -- see the "developer check" note in `../CLAUDE.md`.
