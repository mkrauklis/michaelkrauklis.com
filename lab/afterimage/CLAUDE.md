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
both the encoder and decoder ship as frozen static assets, but they are used
completely differently:

1. The **encoder** runs once, forward-only, on the uploaded photo, to get a
   fast starting-point latent vector. This is a speed optimization, nothing
   more — see "Why the encoder is shipped after all" below.
2. Starting from that point, the **decoder** is optimized against via ~500
   steps of Adam gradient descent on the latent vector itself (not the
   decoder's weights) to minimize pixel error against the uploaded photo.
   This is the same "GAN inversion" idea used by DeepDream/style-transfer
   demos: freeze the network, optimize the input.
3. A further ~400 steps then *also* optimize a small per-photo adapter
   layered onto the decoder's last layer — see "The personalization adapter"
   below.

Why per-photo optimization instead of just trusting the encoder's output
directly? Because it measurably reconstructs better — see
`training/verify_latent_search.py` and the results in `training/RUNBOOK.md`.
The encoder is a compression shortcut learned to work reasonably for *any*
photo in one shot; per-photo optimization has no such shortcut and
consistently finds a better-fitting point in latent space for *this* photo.

**No photo is ever uploaded to a server.** The crop, the encoder pass, and
the entire optimization run client-side. This must stay true — don't add a
fetch/XHR that sends image data anywhere.

## Why the encoder is shipped after all

Earlier versions of this project shipped the decoder only, starting the
latent search from zero every time — simpler, but slow (30s-2min with no
visible progress for the first stretch, since zero is a bad starting point
in a 256-dimensional space). Running the encoder once first (a few
milliseconds) gets the search started from a point already close to a
plausible photo, which cuts the *first* search phase's iteration count and
wall-clock time substantially without touching final quality — phase 1 only
refines, it doesn't replace, the encoder's guess. The encoder is never used
as the final answer; nothing shortcuts straight from encoder output to
displayed result.

Cost: this doubles the shipped weight size (~12MB decoder + ~12MB encoder =
~24MB total). That's the deliberate trade being made — bandwidth for speed.

## The personalization adapter ("head")

After the latent-only search plateaus, a second phase optimizes four small
per-channel vectors layered onto the decoder's *last* transposed-conv layer
only:

```
xIn  = x * scaleIn  + shiftIn    (scaleIn, shiftIn: 32 numbers each -- applied
                                   to the last layer's input channels)
y    = convTranspose4(xIn)
y    = y * scaleOut + shiftOut   (scaleOut, shiftOut: 3 numbers each -- applied
                                   to the final RGB output channels)
```

70 numbers total (32+32+3+3), initialized to the identity (`scale=1,
shift=0`) so phase 2 starts exactly where phase 1 left off and only
diverges as far as it actually helps. This is a small per-photo artifact,
genuinely distinct from the latent vector — it gets its own QR code (see
"Sharing" below).

**This is a FiLM-style affine reparameterization, not a LoRA-style delta to
the conv weight matrix — that distinction is load-bearing, not stylistic.**
A rank-r low-rank delta applied directly to the conv filter (`W + A@B`,
passed as the filter argument to `conv2dTranspose`) needs gradients with
respect to that filter tensor when you backprop through it — TFJS's WASM
backend has no `Conv2DBackpropFilter` kernel, so an optimizer trying to
train A and B that way throws mid-search, silently, inside a promise if you
aren't watching for it (found this the hard way — it happened deep into
phase 2, past the point where a cursory smoke test would catch it). The
scale/shift adapter never uses a tensor as a *filter* argument — the actual
conv weight (`arrays.convT4_weight`) stays a constant array the whole time —
so every gradient it needs is with respect to *inputs*, the same kind of
gradient the latent search already relies on successfully on every backend.
If you're tempted to make the adapter more expressive (e.g. a real weight
delta), confirm first that whatever backend you're targeting actually
implements the backward kernels involved, on real target hardware, not just
in a quick single-iteration test — the failure mode is late and silent.

## Architecture

```
Encoder (shipped, forward-only warm start): 4 conv layers (stride 2 each,
  96px -> 6px, 3 -> 32 -> 64 -> 128 -> 256 channels) + 1 dense head -> 256-d latent
Decoder (shipped, optimized against per-photo): 1 dense head + 4 transposed-conv
  layers (stride 2 each, 6px -> 96px, 256 -> 128 -> 64 -> 32 -> 3 channels)
Adapter (computed per-photo, phase 2 only, never trained offline): scale/shift
  on the last conv layer's input (32+32) and output (3+3) -- see above
```

Every conv/conv-transpose layer is followed by BatchNorm2d + ReLU except the
decoder's last, which goes straight to sigmoid. See `training/model.py` for
the exact PyTorch definition of the encoder/decoder — treat that file as the
single source of truth for their architecture; the JS port in `index.html`
must stay numerically equivalent to it (see "Modifying the network" below).
The adapter has no PyTorch counterpart at all — it's purely a browser-side,
per-photo construct with a fixed, hand-picked shape (see above); there's
nothing to export or retrain for it.

## Directory layout

```
lab/afterimage/
  index.html                the actual page (upload, crop, search, share, explore, learning section)
  CLAUDE.md                  this file
  decoder-manifest.json      layer shapes/order for the shipped decoder (committed)
  weights/*.bin               raw float32 arrays for the shipped decoder (committed)
  encoder-manifest.json      layer shapes/order for the shipped encoder (committed)
  encoder-weights/*.bin       raw float32 arrays for the shipped encoder (committed)
  training/                   Python, never shipped to the site
    model.py                   architecture definition (source of truth)
    train.py                   trains encoder+decoder together, GPU
    verify_latent_search.py    validates the frozen-decoder+search mechanism
    export_decoder.py          fuses BatchNorm into conv weights, exports for JS,
                                self-verifies the export against the PyTorch model
    export_encoder.py          same, for the encoder
    requirements.txt           loose deps
    requirements.lock.txt      exact pinned versions (generated from a clean venv)
    RUNBOOK.md                  step-by-step retraining instructions
    data/, cnn_weights.pt, *.png, encoder-weights/, decoder-manifest.json(*)
      gitignored, regenerated locally -- (*) training/'s own copies of the
      manifests are export_decoder.py/export_encoder.py's raw output before
      being copied up to the shipped location above; don't confuse the two
```

## Why TensorFlow.js and not ONNX Runtime Web / Transformers.js

The rendering phase needs backpropagation through the decoder with respect
to an *input* (the latent vector, and later the adapter), run hundreds of
times per photo. That's a training-loop primitive, not inference. ONNX
Runtime Web and Transformers.js are inference-only in their mainstream form.
TensorFlow.js is the one mainstream browser framework built for custom
training loops (`tf.variable()` + `tf.train.adam().minimize()`), with
`tf.conv2dTranspose` as a first-class differentiable op and a
WebGL-accelerated backend — without GPU acceleration, hundreds of iterations
of a 4-layer conv-transpose backward pass in plain JS would take tens of
minutes, not seconds.

This is a deliberate departure from Ridgeline's zero-dependency approach —
Ridgeline's DFT is ~50 lines to hand-roll, a correct conv-transpose backward
pass is not something worth hand-rolling and debugging in vanilla JS.

**The page picks its backend at runtime — WebGL if it's genuinely float32,
WASM otherwise — this was not optional.** TFJS's WebGL backend can silently
run at 16-bit float precision on some GPUs/drivers
(`tf.backend().floatPrecision()` returns 16 despite
`WEBGL_RENDER_FLOAT32_ENABLED` being true — a real hardware/driver ceiling,
not a config flag you can override). In testing, that precision loss
underflowed the small gradient values this optimization depends on late in
the search: overall brightness/color still roughly converged, but fine
spatial detail never resolved, producing a nearly uniform blurry blob
instead of a recognizable photo. `chooseBackend()` in `index.html` tries
WebGL first, checks `floatPrecision()`, and only falls back to WASM (true
float32, loaded via `@tensorflow/tfjs-backend-wasm`) if WebGL didn't deliver
real float32 — WASM is single-threaded CPU execution, not GPU-parallel, so
expect a real wall-clock cost when that fallback triggers (worth it for
correctness; a fast wrong answer is worse than a slow right one). If you're
ever debugging a quality regression, check `tf.getBackend()` and
`tf.backend().floatPrecision()` first, and re-run the
`?debugLatent=reference` check after a *full* search (not just the single
forward pass it does today, which matches regardless of precision) if you
extend that debug hook.

## Sharing: QR codes and URLs

A reconstruction produces two independent artifacts, each small enough to
round-trip through a QR code and a URL:

- **latent vector** (256 numbers) → quantized to 256 bytes (int8, clipped to
  ±4 — chosen empirically as generous headroom for optimized latents, not a
  hard mathematical bound) → Base45-encoded → `?z=...`
- **adapter** (70 numbers, stored as deltas from identity) → quantized to 70
  bytes (int8, clipped to ±1.5) → Base45-encoded → `?h=...`

Both live in the same URL when both exist (`?z=...&h=...`); the latent alone
is also independently valid and reconstructs through the shared decoder at
the phase-1-only quality level. Two QR codes are generated (`buildShareArtifacts()`)
so both fidelity levels are shareable separately.

**Base45 was chosen because its 45-character alphabet is exactly QR's
alphanumeric-mode charset** — the same reasoning behind its use in EU COVID
certificates. That efficiency only actually applies when the *entire* QR
payload stays within that alphabet, though — wrapping the payload in a URL
(`https://.../?z=...`) reintroduces lowercase letters and other characters
outside it, which forces the whole QR into byte mode regardless. This is a
real, deliberate trade-off (URLs make the codes directly scannable/openable
by any camera app, which the alphanumeric-mode size advantage doesn't
outweigh), not an oversight — the resulting QR codes are still comfortably
within capacity (a few hundred bytes against a multi-kilobyte ceiling), just
not maximally compact. Don't "fix" this by dropping the URL wrapper without
reconsidering the "scan and open" UX it enables.

**`encodeURIComponent()` around each payload before embedding it in a URL is
required, not optional.** Base45's alphabet includes `%`, `+`, `/`, and `:`.
A raw `%` immediately followed by two characters that happen to look like
hex digits (both plausible from Base45's alphanumeric alphabet) gets
silently percent-decoded by `URLSearchParams` on the reading end, corrupting
the payload — probabilistically, not deterministically, which is exactly
the kind of bug that passes a quick test and fails intermittently in the
field. Found via an actual jsQR round-trip test, not by inspection — if you
change the encoding scheme, re-verify with a real encode → QR → jsQR decode →
`URLSearchParams.get()` round-trip, not just a visual QR scan.

Reading a QR code back (`readQrImage()`, wired to the "load a shared
impression" upload control) uses `jsQR` against an uploaded *image* of a QR
code — there's no live camera capture, by design, to keep this consistent
with the rest of the site never requesting camera/microphone access.
`parseShareText()` accepts either a bare `z=...&h=...` fragment or a full
URL, so a decoded QR payload and a pasted URL both work through the same
path. Loading from a URL param (`urlParamHook`) and loading from a scanned
QR image both funnel through `loadFromParams()`, which also regenerates that
loaded impression's own QR codes via `buildShareArtifacts()` — so a shared
impression can always be re-shared, not just viewed once.

## Modifying the network (retraining, changing the architecture)

Follow `training/RUNBOOK.md` exactly, in order. The critical step is that
`export_decoder.py`/`export_encoder.py` each re-run their own exported
weights through a plain NumPy forward pass and assert the result matches
the original PyTorch model almost exactly before letting you proceed — if
either assertion fails, the bug is in the fusion math or the PyTorch→TFJS
layout permutes, and you must not update the site's `weights/`/
`encoder-weights/` from that export.

After exporting, there's a second check the decoder export script can't do
for you: it also writes `training/reference_latent.json` and
`training/reference_output.png` — a fixed latent vector and its known-good
decoder output. Before considering a retrain done, open
`index.html?debugLatent=reference` locally (the debug hook lives at the
bottom of the `<script>` block) and confirm the two images it renders
side-by-side match (max per-channel diff should be ~1/255, uint8 rounding —
not a visibly different image). This is the only thing that actually catches
a bug in the *JavaScript decoder's forward pass*, since the Python-side
assertion only proves the export is faithful to the PyTorch model. It does
**not** exercise the encoder, the optimization loop, or the adapter — for
those, do one real upload-through-reconstruction pass by hand and confirm
the result resembles the source photo.

If you change either network's architecture (layer count, channel widths,
kernel size, latent dimension), you must update the matching JS function in
`index.html` (`encode()` or `decodeFull()`) to match — each is a
hand-written mirror of `training/model.py`, not something that reads the
architecture generically from its manifest. The manifests mainly exist to
hand shapes and array names between Python and JS, not to make the JS side
architecture-agnostic. Changing the adapter's shape additionally requires
updating `ADAPTER_IN`/`ADAPTER_OUT`/`ADAPTER_CLIP`/`ADAPTER_LEN` and the
quantize/dequantize functions together — they all assume the current 70-number
layout.

## Testing changes

No test suite for `index.html` (static page, same as Ridgeline). Golden path
to verify manually: upload a photo → crop to square → run reconstruction
(watch it through both phases; "personalizing" should start automatically
after "refining" finishes) → confirm it visibly resembles the source
(soft/impressionistic is expected and fine; unrecognizable is not) → check
both QR codes render and actually decode (a real jsQR round-trip, not just
"a QR-looking pattern appeared") → reload the page with the resulting `?z=`
URL and confirm it reproduces the same image deterministically → try
"nudge it" in the explore section and confirm the canvas updates live. Watch
the browser console for errors during the *personalizing* phase specifically
— that's where a backend-kernel-support regression (see "The personalization
adapter" above) would surface.
