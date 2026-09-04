"""
Exports the frozen decoder half of a trained Afterimage autoencoder into a
browser-loadable format (raw float32 binaries + a JSON manifest describing
shapes and layer order). The encoder is never exported -- it's only needed
during training, not at inference time.

Each ConvTranspose2d + BatchNorm2d pair is fused into a single affine
transform folded directly into the conv weight and bias, so the shipped
decoder is pure conv/dense + activation with no separate batchnorm step:

    conv_out = convT(x)                      # no bias yet
    y = gamma/sqrt(var+eps) * (conv_out + conv_bias - mean) + beta
      = scale * conv_out + [beta + scale*(conv_bias - mean)]
      = convT(x, weight=weight*scale) + fused_bias

PyTorch and TensorFlow.js disagree on tensor layout, so every array is
permuted on export:
  - Linear.weight: torch (out, in)              -> tfjs dense kernel (in, out)
  - ConvTranspose2d.weight: torch (in, out, kh, kw) -> tfjs conv2dTranspose
    filter (kh, kw, out, in)

After exporting, this script reloads the exported arrays and re-runs the
fused forward pass in pure PyTorch against several random latent vectors,
asserting it matches the original model's decode() output almost exactly.
This is the safety net for the fusion math and layout permutes -- if this
assertion passes, any mismatch found later is in the JavaScript port, not
in the exported weights.
"""
import argparse
import json
import os

import numpy as np
import torch

from model import Autoencoder, LATENT, RES, CH

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fuse_convT_bn(convT, bn):
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    scale = gamma / torch.sqrt(var + bn.eps)
    conv_bias = convT.bias.detach() if convT.bias is not None else torch.zeros_like(mean)
    fused_bias = beta + scale * (conv_bias - mean)
    fused_weight = convT.weight.detach() * scale.view(1, -1, 1, 1)  # scale along out_channels (dim 1)
    return fused_weight, fused_bias


def to_tfjs_convT(weight_torch):
    # torch (in, out, kh, kw) -> tfjs conv2dTranspose filter (kh, kw, out, in)
    return weight_torch.permute(2, 3, 1, 0).contiguous().numpy().astype(np.float32)


def to_tfjs_dense(weight_torch):
    # torch Linear.weight (out, in) -> tfjs dense kernel (in, out)
    return weight_torch.detach().t().contiguous().numpy().astype(np.float32)


def save_array(name, arr, manifest):
    path = os.path.join(OUT_DIR, "weights", f"{name}.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.astype(np.float32).tofile(path)
    manifest[name] = {"shape": list(arr.shape), "file": f"weights/{name}.bin"}


def export(weights_path, out_manifest_path):
    model = Autoencoder()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    manifest = {"latent_dim": LATENT, "res": RES, "channels": CH, "layers": []}
    arrays = {}

    dec_fc_w = to_tfjs_dense(model.dec_fc.weight)
    dec_fc_b = model.dec_fc.bias.detach().numpy().astype(np.float32)
    arrays["dec_fc_weight"] = dec_fc_w
    arrays["dec_fc_bias"] = dec_fc_b
    manifest["layers"].append({
        "type": "dense", "weight": "dec_fc_weight", "bias": "dec_fc_bias", "activation": "none",
        "reshape_chw": [256, 6, 6],
        "reshape_note": "flat output is channel-major (matches torch's view(256,6,6)) -- "
                         "reshape to [256,6,6] THEN transpose to HWC [6,6,256], never reshape "
                         "straight to [6,6,256] or channels get scrambled",
    })

    convT_bn_pairs = [
        (model.dec[0], model.dec[1]),  # convT1, bn1
        (model.dec[3], model.dec[4]),  # convT2, bn2
        (model.dec[6], model.dec[7]),  # convT3, bn3
    ]
    for i, (convT, bn) in enumerate(convT_bn_pairs, start=1):
        fw, fb = fuse_convT_bn(convT, bn)
        arrays[f"convT{i}_weight"] = to_tfjs_convT(fw)
        arrays[f"convT{i}_bias"] = fb.numpy().astype(np.float32)
        manifest["layers"].append({
            "type": "conv2dTranspose", "weight": f"convT{i}_weight", "bias": f"convT{i}_bias",
            "stride": 2, "padding": 1, "activation": "relu",
        })

    # final convT4: no batchnorm follows it, so no fusion -- export as-is
    convT4 = model.dec[9]
    arrays["convT4_weight"] = to_tfjs_convT(convT4.weight.detach())
    arrays["convT4_bias"] = convT4.bias.detach().numpy().astype(np.float32)
    manifest["layers"].append({
        "type": "conv2dTranspose", "weight": "convT4_weight", "bias": "convT4_bias",
        "stride": 2, "padding": 1, "activation": "sigmoid",
    })

    for name, arr in arrays.items():
        save_array(name, arr, manifest)

    with open(out_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"exported {len(arrays)} arrays + manifest to {out_manifest_path}", flush=True)

    return model, manifest, arrays


def verify(model, manifest, arrays, n_trials=5, atol=1e-4):
    """Re-run the fused, permuted weights through a plain PyTorch forward pass
    and check it matches model.decode() almost exactly. Only fusion/layout
    bugs would show up here -- this never touches JS."""
    rng = torch.Generator().manual_seed(0)
    max_diff = 0.0
    for _ in range(n_trials):
        z = torch.randn(1, LATENT, generator=rng)
        with torch.no_grad():
            expected = model.decode(z).numpy()

        x = z.numpy() @ arrays["dec_fc_weight"] + arrays["dec_fc_bias"]
        # torch's h.view(1,256,6,6) is channel-major (NCHW) -- reshape into that
        # order first, THEN transpose to NHWC. Reshaping straight to (1,6,6,256)
        # would silently scramble which values land in which channel.
        x = x.reshape(1, 256, 6, 6).transpose(0, 2, 3, 1)
        for i in range(1, 4):
            w = arrays[f"convT{i}_weight"]  # (kh,kw,out,in)
            b = arrays[f"convT{i}_bias"]
            x = conv_transpose_nhwc(x, w, stride=2, padding=1)
            x = x + b
            x = np.maximum(x, 0)  # relu
        w4, b4 = arrays["convT4_weight"], arrays["convT4_bias"]
        x = conv_transpose_nhwc(x, w4, stride=2, padding=1) + b4
        x = 1 / (1 + np.exp(-x))  # sigmoid

        got = x.transpose(0, 3, 1, 2)  # NHWC -> NCHW to compare against torch
        diff = np.max(np.abs(got - expected))
        max_diff = max(max_diff, diff)

    print(f"verify: max abs diff over {n_trials} random latents = {max_diff:.2e}", flush=True)
    assert max_diff < atol, f"fused/exported weights diverge from original model by {max_diff}"
    print("verify: PASSED -- exported weights are numerically equivalent to the trained model", flush=True)


def conv_transpose_nhwc(x, w, stride, padding):
    """Reference conv_transpose2d in numpy, NHWC input / (kh,kw,out,in) filter,
    used only to verify the export -- not performance-critical."""
    n, h, wid, cin = x.shape
    kh, kw, cout, cin_w = w.shape
    assert cin == cin_w
    out_h = (h - 1) * stride - 2 * padding + kh
    out_w = (wid - 1) * stride - 2 * padding + kw
    out = np.zeros((n, out_h + 2 * padding, out_w + 2 * padding, cout), dtype=np.float32)
    for i in range(h):
        for j in range(wid):
            patch = x[:, i, j, :]  # (n, cin)
            contrib = np.einsum('nc,pqoc->npqo', patch, w)  # (n, kh, kw, cout)
            oi, oj = i * stride, j * stride
            out[:, oi:oi + kh, oj:oj + kw, :] += contrib
    return out[:, padding:padding + out_h, padding:padding + out_w, :]


def save_reference_case(model, out_dir):
    """A fixed latent vector + its decoder output, saved so the JS port can
    later be checked against a known-correct answer instead of just 'looks right'."""
    torch.manual_seed(42)
    z = torch.randn(1, LATENT)
    with torch.no_grad():
        out = model.decode(z).numpy()[0]  # (C,H,W)
    img = (np.clip(out, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(img).save(os.path.join(out_dir, "reference_output.png"))
    with open(os.path.join(out_dir, "reference_latent.json"), "w") as f:
        json.dump({"latent": z.numpy()[0].tolist()}, f)
    print(f"saved reference_latent.json + reference_output.png to {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(OUT_DIR, "cnn_weights.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "decoder-manifest.json"))
    args = ap.parse_args()

    model, manifest, arrays = export(args.weights, args.out)
    verify(model, manifest, arrays)
    save_reference_case(model, OUT_DIR)
