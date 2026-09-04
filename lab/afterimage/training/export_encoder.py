"""
Exports the trained encoder for use as a fast warm-start in the browser: one
forward pass gives a decent starting latent vector, and the existing
per-photo gradient search refines it from there instead of from zero. The
encoder is never used on its own to produce the final result -- see
../CLAUDE.md for why (it measurably reconstructs worse than search alone).

Mirrors export_decoder.py's approach: fuse every Conv2d + BatchNorm2d pair
into a single affine transform folded into the conv weight/bias, permute
every array from PyTorch's layout into TensorFlow.js's, and self-verify the
export against the original PyTorch model before writing anything the site
will load.

Layout notes (mirror image of export_decoder.py's, read that file's
docstring first if this doesn't make sense on its own):
  - Linear.weight: torch (out, in)                 -> tfjs dense kernel (in, out)
  - Conv2d.weight: torch (out, in, kh, kw)          -> tfjs conv2d filter (kh, kw, in, out)
  - encoder's flatten (h.view(N,-1) on an NCHW tensor) is channel-major --
    the JS port must transpose NHWC->NCHW-equivalent BEFORE flattening, or
    enc_fc's weights get matched against scrambled input, same class of bug
    the decoder's reshape note warns about.
"""
import json
import os

import numpy as np
import torch

from model import Autoencoder, LATENT

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fuse_conv_bn(conv, bn):
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    scale = gamma / torch.sqrt(var + bn.eps)
    conv_bias = conv.bias.detach() if conv.bias is not None else torch.zeros_like(mean)
    fused_bias = beta + scale * (conv_bias - mean)
    fused_weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)  # scale along out_channels (dim 0)
    return fused_weight, fused_bias


def to_tfjs_conv(weight_torch):
    # torch (out, in, kh, kw) -> tfjs conv2d filter (kh, kw, in, out)
    return weight_torch.permute(2, 3, 1, 0).contiguous().numpy().astype(np.float32)


def to_tfjs_dense(weight_torch):
    return weight_torch.detach().t().contiguous().numpy().astype(np.float32)


def save_array(name, arr, manifest):
    path = os.path.join(OUT_DIR, "encoder-weights", f"{name}.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.astype(np.float32).tofile(path)
    manifest[name] = {"shape": list(arr.shape), "file": f"encoder-weights/{name}.bin"}


def conv2d_nhwc(x, w, stride, padding):
    """Reference conv2d in numpy, NHWC input / (kh,kw,in,out) filter -- used
    only to verify the export, not performance-critical."""
    n, h, wid, cin = x.shape
    kh, kw, cin_w, cout = w.shape
    assert cin == cin_w
    xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (wid + 2 * padding - kw) // stride + 1
    out = np.zeros((n, out_h, out_w, cout), dtype=np.float32)
    for oi in range(out_h):
        for oj in range(out_w):
            i0, j0 = oi * stride, oj * stride
            patch = xp[:, i0:i0 + kh, j0:j0 + kw, :]  # (n,kh,kw,cin)
            out[:, oi, oj, :] = np.einsum('nhwc,hwco->no', patch, w)
    return out


def export(weights_path, out_manifest_path):
    model = Autoencoder()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    manifest = {"latent_dim": LATENT, "layers": []}
    arrays = {}

    conv_bn_pairs = [
        (model.enc[0], model.enc[1]),
        (model.enc[3], model.enc[4]),
        (model.enc[6], model.enc[7]),
        (model.enc[9], model.enc[10]),
    ]
    for i, (conv, bn) in enumerate(conv_bn_pairs, start=1):
        fw, fb = fuse_conv_bn(conv, bn)
        arrays[f"conv{i}_weight"] = to_tfjs_conv(fw)
        arrays[f"conv{i}_bias"] = fb.numpy().astype(np.float32)
        manifest["layers"].append({
            "type": "conv2d", "weight": f"conv{i}_weight", "bias": f"conv{i}_bias",
            "stride": 2, "padding": 1, "activation": "relu",
        })

    enc_fc_w = to_tfjs_dense(model.enc_fc.weight)
    enc_fc_b = model.enc_fc.bias.detach().numpy().astype(np.float32)
    arrays["enc_fc_weight"] = enc_fc_w
    arrays["enc_fc_bias"] = enc_fc_b
    manifest["layers"].append({
        "type": "dense", "weight": "enc_fc_weight", "bias": "enc_fc_bias", "activation": "none",
        "flatten_note": "input must be transposed NHWC->NCHW-equivalent [N,C,H,W] BEFORE "
                         "flattening -- torch's h.view(N,-1) on the conv output is channel-major",
    })

    for name, arr in arrays.items():
        save_array(name, arr, manifest)
    with open(out_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"exported {len(arrays)} arrays + manifest to {out_manifest_path}", flush=True)
    return model, manifest, arrays


def verify(model, arrays, n_trials=5, atol=1e-3):
    rng = torch.Generator().manual_seed(1)
    max_diff = 0.0
    for _ in range(n_trials):
        x = torch.rand(1, 3, 96, 96, generator=rng)  # matches ToTensor()'s [0,1] range
        with torch.no_grad():
            expected = model.encode(x).numpy()

        h = x.permute(0, 2, 3, 1).numpy()  # NCHW -> NHWC
        for i in range(1, 5):
            w, b = arrays[f"conv{i}_weight"], arrays[f"conv{i}_bias"]
            h = conv2d_nhwc(h, w, stride=2, padding=1) + b
            h = np.maximum(h, 0)
        h_nchw = h.transpose(0, 3, 1, 2)  # NHWC -> NCHW, matches torch's channel-major flatten
        h_flat = h_nchw.reshape(1, -1)
        got = h_flat @ arrays["enc_fc_weight"] + arrays["enc_fc_bias"]

        diff = np.max(np.abs(got - expected))
        max_diff = max(max_diff, diff)

    print(f"verify: max abs diff over {n_trials} random inputs = {max_diff:.2e}", flush=True)
    assert max_diff < atol, f"fused/exported encoder weights diverge from original model by {max_diff}"
    print("verify: PASSED", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(OUT_DIR, "cnn_weights.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "encoder-manifest.json"))
    args = ap.parse_args()
    model, manifest, arrays = export(args.weights, args.out)
    verify(model, arrays)
