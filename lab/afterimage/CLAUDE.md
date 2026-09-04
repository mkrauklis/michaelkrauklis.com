# Afterimage

Reconstructs an uploaded photo from a small latent vector, using a decoder
that was trained once and never changes. The name is the idea: the decoder
holds a lasting impression of everything it saw during training, and a new
photo is rebuilt from a trace of it rather than the photo itself.

## The core idea (read this before touching the model)

There are two entirely different phases, and conflating them is the most
likely source of confusing bugs:

**Training** (`training/`, Python, run once, offline, on a GPU): a
convolutional encoder and decoder are trained together on 13,000 STL-10
photos to reconstruct their input through a 256-number bottleneck.

**Rendering** (`index.html`, JavaScript, runs per-visitor, in the browser):
only the trained *decoder* ships, frozen, as a static asset
(`decoder-manifest.json` + `weights/*.bin`). The encoder is discarded after
training and never leaves the training machine. For a new photo, there is no
encoder call at all -- instead, a latent vector starts at zero and is
optimized via ~1500 steps of Adam gradient descent, through the frozen
decoder, to minimize the pixel difference against the uploaded photo. This is
the same "GAN inversion" idea used by DeepDream/style-transfer demos: freeze
the network, optimize the input.

Why this, instead of just running the encoder on the uploaded photo? Because
it measurably reconstructs better -- see `training/verify_latent_search.py`
and the results referenced in `training/RUNBOOK.md`. The encoder is a
compression shortcut learned during training; per-photo optimization has no
such shortcut and consistently finds a better-fitting point in latent space.

**No photo is ever uploaded to a server.** Both the crop and the entire
latent search run client-side. This must stay true — don't add a fetch/XHR
that sends image data anywhere.

## Architecture

```
Encoder (training only): 4 conv layers (stride 2 each, 96px -> 6px,
  3 -> 32 -> 64 -> 128 -> 256 channels) + 1 dense head -> 256-d latent
Decoder (trained + shipped): 1 dense head + 4 transposed-conv layers
  (stride 2 each, 6px -> 96px, 256 -> 128 -> 64 -> 32 -> 3 channels)
```

Every ConvTranspose2d is followed by BatchNorm2d + ReLU except the last,
which goes straight to sigmoid. See `training/model.py` for the exact
PyTorch definition — treat that file as the single source of truth for the
architecture; the JS decoder in `index.html` must stay numerically
equivalent to it (see "Modifying the decoder" below).

## Directory layout

```
lab/afterimage/
  index.html              the actual page (upload, crop, search, learning section)
  CLAUDE.md                this file
  decoder-manifest.json    layer shapes/order for the shipped decoder (committed)
  weights/*.bin             raw float32 arrays for the shipped decoder (committed)
  training/                 Python, never shipped to the site
    model.py                 architecture definition (source of truth)
    train.py                 trains encoder+decoder together, GPU
    verify_latent_search.py  validates the frozen-decoder+search mechanism
    export_decoder.py        fuses BatchNorm into conv weights, exports for JS,
                              self-verifies the export against the PyTorch model
    requirements.txt         loose deps
    requirements.lock.txt    exact pinned versions (generated from a clean venv)
    RUNBOOK.md                step-by-step retraining instructions
    data/, cnn_weights.pt, *.png   gitignored, regenerated locally
```

## Why TensorFlow.js and not ONNX Runtime Web / Transformers.js

The rendering phase needs backpropagation through the decoder with respect
to an *input* (the latent vector), run ~1500 times per photo. That's a
training-loop primitive, not inference. ONNX Runtime Web and Transformers.js
are inference-only in their mainstream form. TensorFlow.js is the one
mainstream browser framework built for custom training loops
(`tf.variable()` + `tf.train.adam().minimize()`), with `tf.conv2dTranspose`
as a first-class differentiable op and a WebGL-accelerated backend — without
GPU acceleration, ~1500 iterations of a 4-layer conv-transpose backward pass
in plain JS would take tens of minutes, not seconds.

This is a deliberate departure from Ridgeline's zero-dependency approach —
Ridgeline's DFT is ~50 lines to hand-roll, a correct conv-transpose backward
pass is not something worth hand-rolling and debugging in vanilla JS.

**The page forces the WASM backend, not WebGL — this was not optional.**
TFJS's default WebGL backend silently runs at 16-bit float precision on some
GPUs/drivers (`tf.backend().floatPrecision()` returns 16 despite
`WEBGL_RENDER_FLOAT32_ENABLED` being true — a real hardware/driver ceiling,
not a config flag you can override). In testing, that precision loss
underflowed the small gradient values this optimization depends on late in
the 1500-step search: overall brightness/color still roughly converged, but
fine spatial detail never resolved, producing a nearly uniform blurry blob
instead of a recognizable photo. Switching to the WASM backend (true float32,
loaded via `@tensorflow/tfjs-backend-wasm`) fixed it completely and actually
converged faster per-iteration than the broken WebGL path did, at some
wall-clock cost (WASM is single-threaded CPU execution, not GPU-parallel —
expect ~30s-2min for the full search depending on device). If you're ever
tempted to switch back to WebGL for speed, first confirm
`tf.backend().floatPrecision() === 32` on your target devices, and re-run the
`?debugLatent=reference` check after a full 1500-iteration search (not just
on the single forward pass, which will match regardless of precision) to
confirm quality didn't silently regress.

## Modifying the decoder (retraining, changing the architecture)

Follow `training/RUNBOOK.md` exactly, in order. The critical step is that
`export_decoder.py` re-runs its own exported weights through a plain NumPy
forward pass and asserts the result matches the original PyTorch model
almost exactly (`max abs diff < 1e-4`) before letting you proceed — if
that assertion fails, the bug is in the fusion math or the PyTorch→TFJS
layout permutes, and you must not update the site's `weights/` from that
export.

After exporting, there's a second check the export script can't do for you:
it also writes `training/reference_latent.json` and
`training/reference_output.png` — a fixed latent vector and its known-good
decoder output. Before considering a retrain done, open
`index.html?debugLatent=reference` locally (the debug hook lives at the
bottom of the `<script>` block) and confirm the two images it renders
side-by-side match. This is the only thing that actually catches a bug in
the *JavaScript* port itself, since the Python-side assertion only proves
the export is faithful to the PyTorch model, not that the JS decoder is
faithful to the export. Concretely: max per-channel diff should be ~1/255
(uint8 rounding), not a visibly different image.

If you change the architecture (layer count, channel widths, kernel size,
latent dimension), you must update the JS decoder in `index.html` to match —
it is a hand-written mirror of `training/model.py`, not something that reads
the architecture generically from the manifest. `decoder-manifest.json`
mainly exists to hand shapes and array names between the two, not to make
the JS side architecture-agnostic.

## Testing changes

No test suite for `index.html` (static page, same as Ridgeline). Golden path
to verify manually: upload a photo → crop to square → run reconstruction →
confirm it visibly resembles the source (soft/impressionistic is expected
and fine; unrecognizable is not) → check the learning section renders and
its diagrams match the current architecture if you've changed it.
