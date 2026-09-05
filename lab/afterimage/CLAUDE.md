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

After the latent-only search plateaus, a second phase optimizes per-channel
scale/shift vectors layered onto the input of the decoder's **last three**
transposed-conv layers (channels 128, 64, 32 -- see `training/model.py`),
plus the very last layer's output:

```
for each of the last 3 conv-transpose layers:
  xIn = x * scaleIn_k + shiftIn_k   (one scaleIn/shiftIn pair per layer,
                                      sized to that layer's input channels)
  x   = convTranspose_k(xIn)
y = x * scaleOut + shiftOut         (scaleOut, shiftOut: 3 numbers each --
                                      applied only after the final layer)
```

454 numbers total (`(128+64+32)*2 + 3*2`), initialized to the identity
(`scale=1, shift=0`) so phase 2 starts exactly where phase 1 left off and
only diverges as far as it actually helps. This is a small per-photo
artifact, genuinely distinct from the latent vector — it gets its own QR
code (see "Sharing" below).

**One layer wasn't enough.** An earlier version of this adapter touched only
the last layer (70 numbers) and the effect was real but barely visible —
measured error improvement was consistently under 30%, and by eye the two
images looked almost identical. Spreading the same kind of adjustment across
the last three layers instead of one (measured improvement jumped to
50-70%+ in testing) gave it enough surface area to actually matter, without
changing the underlying trick at all — still pure input-side FiLM, just
applied more times. If you're tuning this further, `ADAPTER_LAYER_CHANNELS`
in `index.html` is the single place that controls how many layers and which
ones; extending it further back toward the latent vector should keep
working the same way, since nothing about the technique depends on being
near the output.

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
conv weights (`arrays.convT2_weight` through `convT4_weight`, whichever
layers `ADAPTER_LAYER_CHANNELS` targets) stay constant arrays the whole
time — so every gradient it needs is with respect to *inputs*, the same
kind of gradient the latent search already relies on successfully on every
backend, no matter how many layers it's spread across. If you're tempted to
make the adapter more expressive (e.g. a real weight delta), confirm first
that whatever backend you're targeting actually
implements the backward kernels involved, on real target hardware, not just
in a quick single-iteration test — the failure mode is late and silent.

## Architecture

```
Encoder (shipped, forward-only warm start): 4 conv layers (stride 2 each,
  96px -> 6px, 3 -> 32 -> 64 -> 128 -> 256 channels) + 1 dense head -> 256-d latent
Decoder (shipped, optimized against per-photo): 1 dense head + 4 transposed-conv
  layers (stride 2 each, 6px -> 96px, 256 -> 128 -> 64 -> 32 -> 3 channels)
Adapter (computed per-photo, phase 2 only, never trained offline): scale/shift
  on the last 3 conv layers' inputs (128+128, 64+64, 32+32) and the final
  output (3+3) -- 454 numbers total, see above
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

`index.html` links two root-level shared files rather than duplicating their contents:
`/nav.js` (breadcrumb header) and `/theme.css` (palette, base typography, and the panel/
dropzone/button/hero styles shared with Ridgeline — see the root `CLAUDE.md`). Afterimage's own
`<style>` block overrides `--accent`/`--accent-dim` to violet (theme.css's default is amber) and
otherwise holds only what's genuinely specific to this page: `.wrap`'s max-width, the
progress bar, the result-grid/gauge styles, `#debugPanel`. Because those links are root-relative,
opening `index.html` directly over `file://` won't pick them up — test through a local static
server serving the repo root instead (see "Testing changes" below).

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
- **adapter** (454 numbers, stored as deltas from identity) → quantized to 454
  bytes (int8, clipped to ±1.5) → Base45-encoded → `?h=...`

Both live in the same URL when both exist (`?z=...&h=...`); the latent alone
is also independently valid and reconstructs through the shared decoder at
the phase-1-only quality level. Three QR codes are generated
(`buildShareArtifacts()`): latent-only ("the essence" — reconstructs alone),
adapter-only ("the key" — `?h=...` with no `z`, reconstructs nothing by
itself), and both together. See "Split-key sharing" below for why the
adapter-only code exists at all.

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

Reading a QR code back (`readQrImages()`, wired to the "load a shared
impression" upload control) uses `jsQR` against uploaded *images* of QR
codes — there's no live camera capture, by design, to keep this consistent
with the rest of the site never requesting camera/microphone access.
`parseShareText()` accepts either a bare `z=...&h=...` fragment or a full
URL, so a decoded QR payload and a pasted URL both work through the same
path. Loading from a URL param (`urlParamHook`) and loading from scanned QR
images both funnel through `loadFromParams()`, which also regenerates that
loaded impression's own QR codes via `buildShareArtifacts()` — so a shared
impression can always be re-shared, not just viewed once.

## Color: black-to-white only, except where color is literal

`valueToColor()` (used by both the live latent heatmap and the explore grid) and the head
fingerprint's spokes/dots all use a single grayscale scale — dark = smaller/negative, light =
larger/positive. Earlier versions used amber-vs-teal to encode sign, which turned out to just add
a second thing to decode for no real benefit: sign is already recoverable from magnitude/geometry
(spoke length grows outward for positive, inward for negative), so a second arbitrary hue only
added visual noise on top of an already-dense grid. The one deliberate exception is the head
fingerprint's 3 outer accent triangles, which stay real red/green/blue — that's not an arbitrary
color choice, it's a literal, self-explanatory mapping to the decoder's actual R/G/B output
channels. If you add a new visualization, default to the grayscale scale unless a color genuinely
*means* something specific the way RGB does here.

## The head fingerprint (`renderHeadFingerprint`)

The adapter's numbers are plotted in polar coordinates, not a flat grid — **3 concentric rings**,
one per personalized decoder layer (innermost = the layer closest to the latent vector, outermost
= the layer closest to the final image, in `ADAPTER_LAYER_CHANNELS` order), each ring made of
spokes (angle = which channel within that layer, length = that channel's gain delta, a dot at the
tip = its shift delta), plus 3 colored accent marks past the outer ring for the RGB output-channel
adjustments. This isn't decorative: every visual property maps to one specific number,
deterministically, so two different photos' adapters never look alike and the same photo always
looks the same. It reads as a mandala/sunburst, which was the point — it's meant to be worth
printing and asking about, not just informative. If you change `ADAPTER_LAYER_CHANNELS`, the ring
count/radii and per-ring spoke count follow automatically (`rings` is derived from it); only
`ADAPTER_OUT`'s accent placement is still hardcoded to 3.

## Proving the head matters (`showHeadProof`)

Making a personalization step is easy; making its effect legible is a separate problem. After
training finishes (and after loading any shared impression that includes an adapter),
`showHeadProof()` decodes the *same* latent vector twice — once with `adapter: null`, once with
the real adapter — so the only difference between the two rendered images is the adapter itself,
nothing else. When a target photo is available (a live training run, not a loaded-from-URL
impression), it also reports the actual MSE for both, and the percentage the adapter improved on
it — a real number, not just "does this look sharper to you." This exists because a purely visual
before/after is easy to eyeball as "basically the same" even when it isn't; the number settles it.

## The loss curve (`renderLossCurve`)

`checkpoint()` records `{iteration, mse}` at every progress update into `lossHistory` and redraws
the chart each time — this is the literal gradient descent, not a decorative progress bar. A
dashed vertical line marks `phase1End` (where phase 2, the personalization pass, begins), which is
usually visible as a small upward jump in the curve: phase 2 adds 454 new free parameters, so the
loss briefly gets *worse* right at the transition before dropping again as they're fit. That jump
is expected and worth leaving visible, not smoothing away — it's honest evidence that two separate
optimizations are happening, not one continuous one.

## Result-canvas spinners and the status ticker

Each of the four canvases that only get real content partway through training (impression,
latent heatmap, head fingerprint, loss curve) has a `.spinner` sibling inside a `.canvas-wrap`,
shown by `resetSpinners()` at the start of `reconstruct()` and hidden individually the moment
that specific canvas gets its first real paint — `hideSpinner('headSpinner')` only fires inside
`checkpoint()`'s `if (adapter)` branch, for instance, so it correctly stays spinning through all
of phase 1, since there's genuinely nothing to show there until phase 2 starts. `loadFromParams()`
(no training, so no loss curve at all) hides all four immediately rather than leaving any of them
spinning forever.

`startStatusTicker()`/`stopStatusTicker()` cross-fade through `STATUS_MESSAGES`, a mix of
plain-English explanation and lighter asides, under the progress bar — purely there so the
several-second gaps between checkpoints have *something* happening, not a technical read-out.
Keep additions to that list at the same tl;dr level (no jargon a first-time visitor wouldn't
already have from the page copy above) and roughly the same tone mix (mostly plain explanation,
a few genuinely light ones) rather than letting it drift toward either all-technical or all-jokes.

## Split-key sharing: essence, key, and both-together

Beyond the two independent artifacts described below (latent, adapter), `buildShareArtifacts()`
also draws a **third** QR containing *only* the adapter (`?h=...`, no `z`) — deliberately
undecodable by itself, since `loadFromParams` has nothing to decode without a latent. Paired with
the latent-only QR (which *does* decode alone, just at lower fidelity), this gives a genuine
two-piece gift mechanic: give the "essence" to one person and the "key" to another, and neither
piece alone gives the sharp result. The "load a shared impression" flow (`readQrImages()`) accepts
multiple files in one drop (or sequential drops across separate visits to the page — `pendingZ`/
`pendingH` persist in module scope for the session) and auto-classifies each by content rather
than requiring the user to say which slot a scan belongs to; it re-renders through the normal
`loadFromParams()` path as soon as a latent is available, upgrading in place if a matching adapter
arrives later. If you change the QR payload format, all three QR variants (latent-only,
adapter-only, combined) need to stay parseable by `parseShareText()`. Each of the three QR cards
in the "save & share" section sits directly next to a live re-render of exactly what it carries
(`shareEssenceLatentCanvas`, `shareKeyHeadCanvas`, `shareBothLatentCanvas`+`shareBothHeadCanvas`)
so the claim "this code carries these numbers" is something you can see, not just prose asserting
it — `buildShareArtifacts()` renders all of them from the same `latentArr`/`adapterArrs` it QR-
encodes. The key/both rows (plus their `<hr>` dividers, `shareKeyDivider`/`shareBothDivider`) are
hidden together as a unit when there's no adapter — don't toggle the QR canvas's own `.col` without
also toggling its paired visualization `.col`, or you'll end up with an orphaned viz next to
nothing.

The "prove the head matters" comparison (above, in the training section) also gets a third image
when it can: `proofOriginalCol`/`proofOriginalCanvas` show the actual downsampled source photo
next to the with/without-head renders, but only for a live training run — a loaded shared
impression has no source photo to show (by design, nothing was ever uploaded anywhere for it), so
that column stays hidden in that path.

## The latent explorer (`setupExplore`)

The slider is an **absolute position** on the same -4..+4 scale the grids are colored on (styled
with a literal black-to-white CSS gradient background so it visually doubles as that axis) — not a
delta from wherever a dimension happened to land. Every selected cell gets set to exactly that one
value, together (`currentLatent()`: `arr[i] = currentPosition()` for `i in selected`). This
replaced an earlier delta-based design where the slider *added* to each cell's base value — which
was genuinely confusing, because dragging to the slider's minimum didn't reliably produce black:
a cell already sitting near the top of its range would land somewhere in the middle after a
uniform "-4" offset, not at the scale's true minimum. Absolute positioning fixes that by
construction: drag to -4, everything selected *is* -4, full stop.

Selection is click-and-drag box-select on the grid (`onGridDown`/`onWindowMove`/`onWindowUp`,
mirroring the crop-box dragger's window-level-listener pattern elsewhere in this file) — a single
click is just a zero-movement 1x1 drag. Plain drag replaces the selection with the dragged box;
shift-drag unions a second box onto the existing selection instead of replacing it; a plain click
(no drag, no shift) on a cell that was already the *entire* prior selection toggles it off, so
quick single-cell selection still feels like clicking a toggle. Because `setupExplore()` can be
called repeatedly across one page session (train → share → load a different impression → explore
again), the window-level `mousemove`/`mouseup` listeners are cleaned up explicitly via
`exploreDragCleanup` at the top of each call — the clone-and-replace trick used for the other
controls doesn't reach listeners attached to `window` itself, only to elements that get replaced.

Changing the selected region calls `syncSliderToSelection()`, which snaps the slider to the
*average* of the newly-selected cells' original base values — so selecting a different region
never makes the image jump; it just shows you where that region already sits before you've dragged
anything. `currentDistance()` is no longer a closed-form `delta * sqrt(n)` (that only worked when
every selected cell moved by the same *offset*) — now that different cells can sit different
distances from an absolute target position, it's a real per-dimension sum:
`sqrt(sum((position - base[i])^2))` over the selected set.

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
updating `ADAPTER_LAYER_CHANNELS` (which `ADAPTER_IN` and `ADAPTER_LEN`
derive from automatically), `ADAPTER_OUT`/`ADAPTER_CLIP`, and `decodeFull()`'s
`startLi` slicing logic together with the quantize/dequantize functions and
`renderHeadFingerprint()`'s ring layout — they all assume the current
454-number, 3-layer structure.

## Testing changes

No test suite for `index.html` (static page, same as Ridgeline). Golden path
to verify manually: upload a photo → crop to square → run training (watch it
through both phases; "personalizing" should start automatically after
"training" finishes, and the loss curve should show a small jump right at
that transition) → confirm the result visibly resembles the source
(soft/impressionistic is expected and fine; unrecognizable is not) → confirm
the "latent only" vs. "latent + head" comparison shows two visibly different
images with a plausible MSE improvement, not an identical pair or a NaN →
check all three QR codes render and actually decode (a real jsQR round-trip,
not just "a QR-looking pattern appeared") — the latent-only and combined
codes should each reconstruct something by themselves, the adapter-only code
should reconstruct nothing until paired with a latent → reload the page with
the resulting `?z=` URL and confirm it reproduces the same image
deterministically → in the "load a shared impression" box, drop the
latent-only and adapter-only QR images in separately (either order) and
confirm it renders the blurrier version after the first and upgrades after
the second → in the explore section, drag-select a region on the grid,
confirm shift-drag adds a second region instead of replacing the first, and
confirm a plain click on an already-fully-selected single cell toggles it
off → drag the position slider to its minimum with something selected and
confirm the selected cells render as actual black (`rgb(8,8,8)`), not some
partial value — this is the exact bug the absolute-positioning redesign
fixed, so it's the one regression test that matters most in this section →
train a second photo in the same page session afterward and confirm the
explore section still works (catches window-listener leaks from
`exploreDragCleanup` not firing). Watch the browser console for errors
during the *personalizing* phase specifically — that's where a
backend-kernel-support regression (see "The personalization adapter" above)
would surface, and it's more likely to show up now that personalization
touches three layers instead of one.
