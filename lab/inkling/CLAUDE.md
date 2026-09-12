# Inkling

A real 100→18→26 neural network — plain feedforward, trained by real backpropagation, no library,
no pretrained weights — that you teach by hand: draw a letter, say what it is, click Train, and
watch the actual forward pass and gradient step happen on screen. Then spell a word with it and
watch the decoded guess sharpen as you teach it more letters. Originated as a prototype reviewed
in a separate `claude.ai` conversation, then adapted into this repo's conventions and shipped
directly (per direct instruction — "I trust you to publish").

## Architecture

`index.html` plus a vendored `qrcode.min.js` (davidshimjs/qrcodejs v1.0.0, MIT — the exact same
file already vendored for Echo State; see that tool's CLAUDE.md for why it's local instead of a
CDN import). Links the two site-wide shared files (`/nav.js`, `/theme.css`) like every other tool.
No build step, no bundler, no ML library — the entire network (forward pass, softmax,
cross-entropy gradient, backprop, SGD update) is ~60 lines of hand-written array math, matching
this site's standing rule: real computation, nothing simulated or faked for effect.

Accent is a rose/crimson (`--accent:#c81d4f`, `--accent-dim:#7a1030`) — distinct from every other
tool's accent (amber, violet, rust, cyan, teal) and thematically apt for a *drawing* tool ("ink").
A second, page-local `--cool:#2f6fe0` (blue) exists specifically for the excitatory/inhibitory
diverging color scale described below — it isn't a shared `theme.css` token, the same way Vectis
and Neural Viaduct each carry their own extra non-standard hex constants beyond the shared
palette. `--paper:#efe7d6` is the scratchpad's literal paper color, used for legibility against
the site's otherwise dark theme (drawing in dark ink on a light paper reads far better than white
ink on black). One extra Google Fonts import (`JetBrains Mono`) is needed beyond what `theme.css`
already provides (Fraunces + Inter) — used for every numeric/label/mono-styled element (chip
counts, ledger, letter labels in the diagrams, stage captions), matching the prototype's own
type system rather than flattening it into the site's two-typeface default.

## The network (`forward`, `forwardAndGrad`, `applyGrad`, `runEpochs`)

Deliberately the plainest possible architecture, stated directly in the "How this actually works"
copy: `100 pixels → 18 hidden (tanh) → 26 output (softmax, A–Z)`. Forward pass:
`a1 = tanh(W1x + b1)`, `ŷ = softmax(W2·a1 + b2)`. Loss is cross-entropy against whatever letter
you picked. Backward pass is the textbook chain rule: `δ2 = ŷ − y` (softmax+cross-entropy's
famously simple combined gradient), `δ1 = (W2ᵀδ2) ⊙ (1 − a1²)` (undoing tanh), then every `W`/`b`
nudged by `−η·∂L`. `forwardAndGrad()` returns both the activations *and* the gradients from one
pass — the stage animation (below) visualizes exactly this returned object, not a separately
recomputed one, so what's drawn on screen is guaranteed to match what `applyGrad()` actually
applies a moment later.

Pressing Train does two distinct things, not one: `forwardAndGrad()` + the stage animation runs
once, immediately, on your new example specifically (so the animation always shows *this*
drawing's real forward/backward pass) — then, after the animation finishes, `runEpochs(16, 0.45)`
re-shuffles and re-trains on **the entire accumulated dataset**, not just the new example. This is
why earlier letters don't get forgotten as new ones arrive: every Train click is a fresh, full
mini-training run over everything taught so far, not an incremental single-example update layered
on top of the last. `dataset` and `counts` grow monotonically until Reset.

**Seeded, not random-random**: `rngState`/`rnd()` is a hand-rolled xorshift32 PRNG (deterministic
given its fixed seed `88675123`), the same "seeded, not `Math.random()`" discipline Echo State
uses (there via mulberry32 — a different algorithm, which is fine; there's no requirement to share
one PRNG across tools, just to avoid true nondeterminism in initialization/shuffling). `initNet()`
draws `W1`/`W2` from this PRNG at fixed scales (0.35 / 0.45) and small biases (0.05).

## The seed dataset (`SEED`, `seedTrain`) — why the page never opens blank

Six letters (H, I, A, B, C, T) are hand-authored as literal 10×10 ASCII-art patterns, each trained
in two horizontal-shift variants (`patternToGrid(rows, dx)`, `dx` 0 or 1) — 12 examples total, run
through 60 epochs at a higher learning rate (0.5) than live training uses (0.45), *before* the
page shows anything to a visitor. This means: the letter chips already show non-zero counts, the
"look inside" tiles already show real (not random-noise) receptive fields, and the seeded test
word "HAT" (drawn programmatically from the same `SEED` patterns, not typed) already decodes to
something close to correct on first paint. All of this is a genuine consequence of the same
training code a visitor's own Train clicks run — not a separately hardcoded "looks trained" state.
`Reset network` re-runs `initNet()` + `seedTrain()` from scratch, returning to this exact same
starting point, not to a blank/untrained network.

## Drawing: one real canvas, two visible sizes (`makeScratchpad`, `openExpander`/`closeExpander`)

Every pad on the page (the main teaching pad, each of the 8 word-spelling mini-pads) is a small,
static *thumbnail* canvas — actual drawing only ever happens in one shared, much larger
`#expandCanvas` overlay, opened by clicking any thumbnail (`openExpander(padApi, label)`) and
committed back on "Done," an outside click, or Esc (`closeExpander(true)`). This exists because a
10×10-downsampled letter drawn accurately at 60×60 CSS pixels (a word pad's on-page size) is
genuinely hard to do with a mouse or a finger — the shared expander is always a generous
`min(78vmin, 520px)` square regardless of which tiny pad opened it. Reopening a pad that already
has ink on it scales the existing thumbnail *up* into the expander first (`isDirty()` +
`drawImage`), so continuing to refine a drawing never starts over from blank. `pointerdown`/
`pointermove`/`pointerup` (not separate mouse/touch handlers) is deliberate — one code path covers
mouse, touch, and pen input uniformly.

**`getGrid()`'s downsample is a real resample, not a fake one**: it draws the full-resolution
canvas into a 10×10 offscreen canvas via `drawImage` (letting the browser's own image scaling do
the area-averaging), then reads back `getImageData` and converts luminance to ink density
(`1 − luminance`, clamped). This is genuine pixel data feeding the network, not a synthetic
stand-in — the same "verify against the real thing" discipline as every other tool here.

## The stage animation (`animateSequence`) — visualizing the exact pass that just ran

Four overlapping phases over 1.7 seconds, driven by one `requestAnimationFrame` loop and simple
time-windowed opacity ramps (`p2`/`p3`/`p4`, each a `clamp` of elapsed time): input raster fades
in, then forward lines light up input→hidden (colored by each hidden unit's actual activation
sign — rose for positive, blue for negative, via `divergingColor`), then hidden→output lines light
up **only for the true label and the network's current prediction** (not all 26 outputs — showing
every output's edge would be visual noise; the two that matter are the true letter and whatever
the net guessed, which are the same dot when it already gets it right), then dashed
right-to-left lines animate the actual backward gradients (`dz2`, `dz1`) — including a moving
dash offset (`lineDashOffset = -el/12`) so the backward signal visibly *flows* rather than sitting
static. The caption text changes at the same fixed time boundaries the opacity ramps use (350ms,
750ms, 1150ms) — if you retune the ramp timings, update the caption thresholds together, or the
text will visibly desync from what's on screen.

## Look inside: tiles and heatmap (`refreshLookInside`)

Each of the 18 hidden-unit tiles reshapes that unit's 100 real `W1` weights back into a 10×10 grid
and colors every cell with `divergingColor` (rose = excitatory/positive, blue =
inhibitory/negative, magnitude = saturation toward that hue) — this is a literal receptive field,
not an approximation or a stylized stand-in, the same "draw the real weights" principle Echo
State's node-link diagram and Neural Viaduct's block thumbnails both already establish elsewhere
on this site. Tile border color is that same unit's bias, using the same diverging scale. The
26×18 heatmap to the right is `W2` in full: one column per output letter, one row per hidden
unit, so a visitor can see which hidden-layer "features" each letter's output actually leans on.
Both redraw on every `refreshLookInside()` call — after every Train, every Reset, and every
successful weight Load — so they're always the live state of the actual network, never a cached
snapshot.

## Export: quantization, the QR payload, and the two other export paths

**Quantization** (`quantizePayload`, 8-bit or 4-bit toggle): finds the single largest-magnitude
parameter across the whole flattened network, derives one shared linear scale from it
(`maxAbs/levels`, 127 levels at 8-bit, 7 at 4-bit — deliberately asymmetric-looking but correct:
signed range is `-levels-1..levels`), and quantizes every weight/bias to that scale. The payload
is a plain byte array: a float32 scale header (4 bytes), then `[bits, GRID, HID, OUT]` (4 more
bytes) so a decoder knows the exact shape without guessing, then the packed weights — one byte
each at 8-bit, two nibbles per byte at 4-bit. `refreshLedger()` recomputes and redraws the byte
counts *and* the QR code together on every toggle — there's no separate "regenerate QR" step, the
ledger and the QR always describe the same payload.

**QR download** (`#downloadQrBtn`): qrcodejs renders a hidden `<canvas>` plus a visible `<img>`
whose `src` it already set to that canvas's own `toDataURL('image/png')` — same confirmed behavior
already documented in Echo State's CLAUDE.md ("Two Zazzle links" section doesn't apply here, but
the img-first QR-download trick is identical) — so the button just grabs that `<img>`'s `src`
directly, falling back to encoding the canvas itself only if qrcodejs's DOM ever changes.

**Full-precision copy/paste** (`fullPrecisionJSON`, `weightsOut`/`weightsIn`) is a completely
separate export path from the QR — raw, unquantized floats as JSON, meant for exact
restoration (paste back in, `loadWeightsBtn` validates the shape matches `GRID`/`HID`/`OUT`
before accepting it) rather than for physically carrying the network anywhere. Loading weights
restores the network's numbers only, not the `dataset` array of drawn examples that produced
them — teaching is one-way within a session unless the tab stays open, stated directly in the
"How this actually works" copy so this isn't a silent surprise.

**Portrait export** (`portraitBtn`/`#downloadPortraitBtn`): composites the 18 tiles, a scaled copy
of the heatmap, and the current decoded test word into one 560×520 canvas, rendered to
`#portraitImg` and only then exposed for download — `#downloadPortraitRow` starts hidden and is
only revealed after a real portrait has been rendered, the same "no download button before there's
something real to download" convention every other tool's spinner-gated downloads already follow.
The prototype's own version only offered "right-click to save"; a proper one-click download button
was added here to match this site's own established bar (every other tool provides a real
Download button, not a right-click instruction) — this is the one deliberate deviation from a
straight port of the prototype, not a case of "the prototype forgot something."

## QR capacity: 8-bit doesn't actually fit, and defaulting to it was a real bug

The prototype assumed the 8-bit quantized payload would "fit in a QR code" without ever checking
against the real vendored library's actual capacity. It doesn't: this network has 2,312
parameters, so the 8-bit payload is 2,320 raw bytes → ~3,096 base64 characters, and `new
QRCode(...)` (davidshimjs/qrcodejs, `CorrectLevel.L`) throws past a real ceiling confirmed by
binary search directly against this exact vendored file to be ~2,950 base64 characters — a
`TypeError` inside the library's own internal array indexing, not a caught/graceful failure. Left
unhandled, that throw aborted every top-level `<script>` statement after it, which is why the
first shipped version crashed silently the moment `refreshLedger()` ran on page load.

Fixed three ways, all in `refreshLedger()`: (1) `QR_MAX_B64_CHARS` (`2900`, a rounded-down safety
margin under the empirical ~2,950 ceiling) is compared against the payload's
*actual* base64 length before ever calling `new QRCode(...)`, replacing the old and simply wrong
`current <= 1500` raw-byte check that had no real basis; (2) the `new QRCode(...)` call is also
wrapped in try/catch as a second line of defense, so a future change to quantization or network
size can never again take down the whole page's script execution the way it did here — it degrades
to a fallback message in `#qrHolder` instead, and disables `#downloadQrBtn`; (3) the page's default
quantization was flipped from 8-bit to 4-bit (`quantBits = 4`, `#q4` starts with the `on` class)
specifically so the page opens with a *working* QR by default rather than the fallback message —
8-bit is still selectable and correctly shows the "too dense" message rather than crashing.

If `HID`/`GRID`/`OUT` ever change (making the network smaller), 8-bit may start fitting again on
its own — that's fine, the length check is computed live from the actual payload every time, not
hardcoded to this network's current 2,312-parameter size.

## Zazzle link is a placeholder, not a verified template

Same situation as every other tool's *first* ship: `#shareSection`'s "make it a gift" link reuses
the same known-working generic mug template ID (`256206116898885602`) and the site's standard
`?rf=238054754631086278` ambassador param, since no Inkling-specific curated product exists yet.
Unlike Echo State and Ridgeline, there's no hero CTA here at all yet either — those only got added
once a *real, specific* product existed, and this tool was shipped without one being provided.
Add a real hero CTA the same way (right after the intro paragraph, `?rf=` included) once a real
product exists — don't fabricate one or guess a product URL in the meantime.

## Testing changes

No test suite — static page. Verify via a local static server (root-relative `/nav.js` and
`/theme.css` mean `file://` won't pick them up). Golden path: confirm the page opens with the
seed letters already taught (chip counts non-zero for H/I/A/B/C/T) and the seeded "HAT" word
pre-filled and decoding to something close to correct → click the main pad, confirm the expander
opens at a large size, draw a letter, click Done, confirm the thumbnail shows what you drew →
pick a letter chip, click Train, confirm the stage animation actually plays through all four
phases (forward lighting up, prediction, dashed backward flow) and the caption text changes in
sync → confirm chip counts, the look-inside tiles, and the heatmap all visibly update after
training completes → draw a full word across several mini-pads (via the expander on each),
confirm it decodes and appends to the attempts log → toggle 8-bit/4-bit, confirm the ledger byte
counts and the QR code both update together → click "Download QR as PNG" and confirm a real,
non-empty PNG saves → copy the full-precision weights, clear/retrain a little, then paste the
copied JSON back into "Load weights" and confirm the look-inside tiles snap back to the earlier
state (a real round-trip, not just "no error thrown") → confirm the page opens on 4-bit with a
real QR already rendered, then switch to 8-bit and confirm it shows the "too dense" fallback
message (not a crash) with the download button disabled, then switch back to 4-bit and confirm
the QR renders again → render a portrait, confirm the download
button only appears after rendering (not before), and confirm the downloaded PNG actually contains
the tiles/heatmap/word, not a blank canvas → click Reset network twice (confirm the first click
only arms a confirmation toast) and confirm it returns to the exact seed state, not an empty one.
