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

## Why weight decay: a real training-instability bug, diagnosed against the real math, not guessed

**Reported directly**: training letters in sequence could cause the decoded test word to
"collapse" — training H, then A, then H again could turn every position of a three-letter word to
whatever letter was just trained (e.g. "HHT" → train A → "AAT" → train H → "HHH"), rather than each
click just correcting *its own* letter's classification. The user's own hypothesis was "the
training step size is too high."

**That hypothesis turned out to be backwards, verified empirically before changing anything.** A
temporary debug harness (`window.__debugInkling`, removed before shipping — never leave this kind
of hook in) drew realistic, position/size-jittered letter strokes directly onto the real teach pad
and ran the exact real `forwardAndGrad`+`runEpochs` path (bypassing only the decorative
`requestAnimationFrame` stage animation, which never resolves under headless/background automation
— a documented gotcha elsewhere in this codebase — but has zero effect on the actual weights).
Sweeping learning rate and epoch count independently against the same jittered training sequence
showed **lower** LR/epochs producing *more* "every position becomes the same letter" collapses
(e.g. LR 0.15/6 epochs: 4 of 7 steps degenerate), while the *existing* LR 0.45/16 epochs baseline
already produced zero fully-degenerate collapses in that same test — reducing step size the way the
report suggested would have made this measurably worse, not better.

**The actual mechanism**: with a shared-weight softmax classifier, only a handful of examples, and
*no regularization at all*, `applyGrad()`'s unconstrained SGD updates let the output layer's
weights/biases grow large and overconfident. Once a class's weights are large enough, a single new
training example's gradient can swing the softmax's argmax for *other*, already-correctly-
classified inputs too — not just correct its own — because the hidden layer hasn't yet developed
well-separated per-letter features this early, so different letters can still activate overlapping
hidden units, and a large output weight there ends up "winning" for more than just the letter it
was pushed toward. This is a real, well-known small-sample SGD failure mode (catastrophic
interference from unregularized weight growth), not something specific to this one report.

**The fix**: a small L2 weight-decay term, `WEIGHT_DECAY = 0.02`, applied inside `applyGrad()` to
every `W1`/`W2` weight (not biases — see the code comment for why) before the gradient step:
`row[k] = row[k]*(1 - lr*decay) - dv*g.x[k]`. This directly counters unbounded weight growth
without changing `LR`/`EPOCHS` at all. Re-run against the same jittered test sequence that produced
0-4 degenerate collapses across various LR/epoch combos, adding decay alone (keeping the original
0.45/16) dropped it to **zero** degenerate collapses and reliable convergence to the fully correct
decode, including from a genuinely blank post-Reset network (which reproduced the reported "HHH"-
style collapse on its very first training step, then recovered and stayed stable from the fourth
step onward) and in a longer, more varied 10-step, 6-letter sequence. If this ever needs revisiting,
re-run the same kind of jittered-sequence sweep before changing `LR`/`EPOCHS`/`WEIGHT_DECAY` by
feel — the naive "lower LR must be more stable" intuition is specifically the one this bug disproved.

## The seed dataset (`SEED`, `seedTrain`) — why the page never opens blank

Six letters (H, I, A, B, C, T) are hand-authored as literal 10×10 ASCII-art patterns, each trained
in two horizontal-shift variants (`patternToGrid(rows, dx)`, `dx` 0 or 1) — 12 examples total, run
through 60 epochs at a higher learning rate (0.5) than live training uses (0.45), *before* the
page shows anything to a visitor. This means: the letter chips already show non-zero counts and
the "look inside" tiles already show real (not random-noise) receptive fields on first paint. All
of this is a genuine consequence of the same training code a visitor's own Train clicks run — not
a separately hardcoded "looks trained" state.

**`Reset network` does *not* call `seedTrain()`** — it only calls `initNet()`, genuinely returning
to fresh random weights, zeroed `dataset`/`counts`, and a re-run `decodeWordPads()` so the word
display reflects the new (garbage) guesses immediately. This was a real, reported bug: the first
shipped version called `initNet(); seedTrain();` on Reset, so a visitor who clicked Reset expecting
a blank slate still saw the six seed letters' counts and their already-shaped "look inside" tiles
— Reset silently did nothing observable to the trained state, only to whatever the visitor had
personally added on top of the seed. Only the very first page load seeds; Reset is a genuine wipe.

## Drawing: one real canvas, opened as a two-step teach wizard (`makeScratchpad`, `openExpander`/`closeExpander`)

Every pad on the page (the teach pad, each of the 8 word-spelling mini-pads) is a small canvas —
actual drawing only ever happens in one shared, much larger `#expandCanvas` overlay. The teach
pad's own canvas (`pad`, created via `makeScratchpad(document.createElement('canvas'), 220)`) is
never appended to the visible page at all anymore — it exists purely as an off-screen data holder
that `doTrain()` reads via `pad.getGrid()`, exactly the same as before, just without a redundant
inline thumbnail taking up page space (removed on direct feedback: "we don't need the full pad and
all the space it takes up").

**`openExpander(padApi, label, opts)` now branches on `opts.wizard`.** Word-pad mode (`opts`
omitted) is unchanged from the original design: draw, then "Done" (or an outside click, or Esc)
commits the ink straight back to that pad's thumbnail via `commitExpanderInk()`. Teach mode
(`{wizard:true}`) instead swaps "Done" for "Next →", and clicking Next moves the *same modal card*
to a second, distinct body (`showLetterStep()`/`showDrawStep()` toggle two sibling `hidden`
attributes) showing the wizard's own 26-letter picker (`#modalLetterChips` — a second, independent
chip grid from the page's `#letterChips`, since that one is now just a read-only-except-for-opening
overview, see below) and a "Train" button that starts disabled until a letter is picked
(`selectWizardLetter()`). Only "Train" ever calls `doTrain()` in wizard mode — outside-click/Esc/a
mid-wizard cancel all just discard via `closeExpander(false-ish)` (checked as `!expandWizard` before
committing), since committing ink or starting training on an accidental dismiss would be a much
worse surprise than losing an unsubmitted drawing.

**A real, easy-to-reintroduce CSS bug found here**: `.expand-actions[hidden]` doesn't work if
`.expand-actions{ display:flex }` is declared without it — a class selector's `display` beats the
browser's UA-stylesheet `[hidden]{display:none}` rule on specificity, so setting the `hidden`
*property* in JS silently does nothing and both action rows (draw-step's Clear/Next and
letter-step's Back/Train) render on top of each other. Fixed with an explicit
`.expand-actions[hidden]{ display:none; }` override. If you add another element that's toggled via
`.hidden` inside `.expand-card` and it has its own `display` rule, it needs the same override.

**Clicking a chip on the main page also opens this wizard, pre-aimed at that letter**
(`openExpander(pad, 'Draw the letter "'+letter+'"', {wizard:true, presetLetter:letter})`) — the
modal's own header names the letter, and `wizardLetter`/the modal chip grid's active state are
pre-set from `opts.presetLetter` in `openExpander()`, so Train is already enabled the moment
drawing finishes; the visitor can still pick a different letter in step 2 if they clicked the wrong
chip. This was a direct, explicit request ("if you click on a letter it should open the modal but
should tell you that you should be drawing the letter on which you clicked") — before this, the
main-page chip grid was purely a read-only overview with no click behavior at all.

**`getGrid()`'s downsample is a real resample, not a fake one**: it draws the full-resolution
canvas into a 10×10 offscreen canvas via `drawImage` (letting the browser's own image scaling do
the area-averaging), then reads back `getImageData` and converts luminance to ink density
(`1 − luminance`, clamped). This is genuine pixel data feeding the network, not a synthetic
stand-in — the same "verify against the real thing" discipline as every other tool here.

## The stage animation (`animateSequence`, `idleStage`) — real per-pixel connections, not a bundled line

**This went through a full redesign, not a tweak.** The first shipped version treated the entire
100-pixel input as a single bundled line per hidden neuron, originating from one fixed point at the
raster's edge — direct feedback was that this was "super weak" and didn't actually show "how the
letter becomes pixels which feed to neurons then to the chosen letter and back." The current
version draws a real line for every (active pixel, hidden neuron) pair: `activePixels(x, ...)`
finds every grid cell above the same 0.05 ink threshold `drawInputRaster` uses to render it, and
for each one, `fwdC`/`bwdC` precompute the *actual* per-connection numbers — `x[k]*W1[h][k]` for
the forward pass, `dz1[h]*W1[h][k]` (the real per-connection contribution to that pixel's gradient)
for backprop — once per `animateSequence()` call, not per frame. A blank pixel contributes exactly
zero to the weighted sum, so it correctly gets no line at all; this isn't a simplification, it's
what the math already says. Typical hand-drawn letters have 15-40 active cells, so this is at most
a few hundred lines per hidden neuron pass, not the 100×18 worst case a fully-inked square would
produce — verified to stay smooth in testing, not just assumed cheap.

**Three explicit named phases** replace the old four overlapping opacity ramps: `p1` (250-800ms,
pixels → hidden), `p2` (800-1300ms, hidden → output, unchanged from the original design — still
only draws edges to the true label and the current prediction, not all 26 outputs), `p3`
(1300-2000ms total, backward — output→hidden via the real `dz2` gradient, *and* hidden→the same
active pixels via the real `dz1[h]*W1[h][k]` gradient, so the correction visibly retraces the exact
path the forward pass just lit up, all the way back to the pixels). Captions change at the same
boundaries the phases use — if you retune the timing, update both together or the text will desync
from what's on screen, the same trap the original version already had to watch for.

**Persistent column labels** (`drawStageLabels()`, called from both `idleStage()` and every
animation frame) directly state the architecture on the diagram itself: "100 pixels", "18 hidden
neurons (tanh) — one hidden layer —", "26 outputs (softmax, A–Z) / argmax picks the letter" — added
after direct feedback that a visitor had no way to tell from the diagram alone that there's only
one hidden layer, or how a letter gets chosen from the output layer.

**Idle-state hidden/output dots are colored by real bias** (`b1[h]`/`b2[o]` via `divergingColor`),
not a flat gray — direct feedback ("the network diagram should be colored based on the
weights/biases"). There's no activation to color by before any input arrives, but there is always a
real, current bias, the same number the "look inside" tiles' borders already use — so the training
diagram reflects the network's actual current state even at rest, rather than looking like an inert
placeholder until the first drawing. Once an example is drawn, the animated hidden dots switch to
coloring by that example's real activation (`pre.a1[h]`, unchanged from the original design) —
that's an even more specific real number than bias alone, so it isn't touched by this change.

## Keeping the teach panel on one screen (`STAGE_H`, chip sizing)

Direct request: "the training should all be able to fit on one screen along with the validation
section below." "Validation" here is the `#stageWordReadout` line already sitting directly under
the diagram (see below) — the ask was to shrink the whole teach panel (trigger, chip overview,
diagram, legend, readout) enough that it reads as one continuous view rather than something you
scroll through mid-interaction. Two changes did almost all of the work:

- **`STAGE_H` dropped from 300 to 200** (`STAGE_W` unchanged at 760), with `INBOX`/`hiddenPos`/
  `outputPos`/`drawStageLabels()`'s label y-coordinates all rescaled to fit the shorter canvas.
  Because the canvas is responsive (`width:100%; max-width:760px; height:auto`), this aspect-ratio
  change directly shrinks the *rendered* height at every screen size, not just on wide viewports —
  confirmed on both desktop and a 375px mobile width. `OUTPUT_LABEL_SPLIT_Y` (the y-threshold
  deciding whether an output letter's label draws above or below its dot) is a named constant now,
  not a hardcoded `155`, specifically so it stays in sync if this geometry changes again.
- **The `.chips` grid shrank from 34px to 27px chips** (with proportionally smaller gap and count
  badge) — this 26-chip, multi-row grid was the single largest contributor to the panel's height
  after the diagram itself. `.draw-trigger`'s padding and icon size were trimmed slightly too.
  27px is smaller than the ~44px touch target guidelines usually recommend; accepted deliberately
  since this on-page grid is a secondary shortcut to a letter (mistapping just opens the wizard
  pre-aimed at the wrong letter, correctable in step 2), not the primary teaching path — the
  wizard's own in-modal letter picker is unaffected and stays at full chip size.

Verified directly, not just eyeballed: the teach panel (trigger through the readout line, excluding
the section's own heading/intro prose) went from roughly 694px to 585px tall on desktop, and its
overall section height dropped from ~869px to ~760px — under a typical viewport height. On a 375px
mobile width the panel alone is ~715px (still under an 812px mobile viewport), though the full
section including the heading and wrapped intro paragraph runs a bit taller there — accepted, since
shortening the explanatory copy wasn't part of the request and the interactive panel is the part
that actually needed to read as one view.

## Page order: the word comes first, teaching comes second

Sections were reordered from teach→word to word→teach (now "01 — the goal / Give it a word" then
"02 — teach / Teach it a letter"), on direct request: "the example word that's pre-loaded, that
should be the first thing we do before we train. Give it the word, then we train it to try and
understand the word." The reordering is purely a DOM/copy change — both sections' element IDs are
untouched, so nothing downstream needed to change to support it. Section 01's intro paragraph now
frames the word as the goal ("the network hasn't learned much yet, so it may well misread it —
teach it letters in the next step and watch this decode improve"), and a compact, read-only echo of
the same decode (`#stageWordReadout`) sits directly under section 02's training diagram — "reads
today's word (step 01, above) as: X" — added on direct request ("we should see the word right below
the training diagram") so the connection between "you just trained" and "here's how the word
decode changed" doesn't require scrolling back up to section 01. `renderDecoded(str)` is the single
function both `#wordOutput` (the full display) and `#stageWordReadout` (the compact one) read from
— it updates both together, so they can never desync.

**"Decode now" was removed entirely** — direct request: "just do that any time the letters change
or we train." It was already redundant: `sp.onCommit` (fires when a word-pad's ink is committed)
and the per-pad "clear" button already called `decodeWordPads()`/`renderDecoded()` on their own;
the button only mattered for the (already rare) case of wanting a fresh attempts-log entry without
changing anything. Both handlers now also call `pushAttempt()`, so every real decode moment — a
letter changing, a clear, or a Train — logs an entry the same way, with no separate manual path. A
new **"Clear all"** button (`#clearAllWordBtn`) was added in the old button's place, clearing every
word pad and re-decoding in one click — the per-pad "clear" links under each mini-pad still exist
for clearing just one letter.

## The seeded word looks hand-drawn, not like pixel-art bricks

The word pads' pre-loaded "HAT" used to render by filling a solid square for every `patternToGrid`
cell above 0.5 — literally the training data's own blocky rasterization, drawn straight onto the
60×60 mini-pad canvas. At that scale it read as pixel-art bricks, not handwriting — direct
feedback: "the pre-loaded word should look like handwriting, not the brick letters we have now."
Fixed with `HANDWRITTEN_STROKES` (hand-authored pen paths, in 0-9 grid-cell coordinates, currently
covering H/A/T — the three letters actually shown) and `drawHandwrittenLetter()`, which strokes
each path with a thick, round-capped/joined line — the same visual language as the live freehand
drawing input uses (`expandCtx`'s own `lineCap='round'`), rather than `patternToGrid`'s discrete
grid squares. `seedWord()` falls back to the old block-fill rendering for any letter without a
hand-authored stroke path, so this degrades gracefully rather than silently drawing nothing if the
seeded word ever changes to include a letter outside H/A/T.

**This is a display-only change — `SEED`/`patternToGrid`/`seedTrain()` are untouched**, and that's
a deliberate, accepted tradeoff, not an oversight: `decodeWordPads()` re-samples whatever is
*currently drawn* on a word pad's canvas, independent of whatever data trained the network, so
there's no requirement that the two match pixel-for-pixel. The practical effect is that the
initial "HAT" decode is no longer reliably close to correct on first paint the way it used to be
(the hand-stroke shapes differ enough from the ASCII-block shapes the seed-trained network actually
learned that the very first decode can come out wrong, e.g. "QII" instead of "HAT" in one observed
run) — but this now reads as *consistent* with section 01's own copy ("the network hasn't learned
much yet, so it may well misread it"), rather than as a regression. If a pixel-perfect initial
decode ever matters again, the real fix is regenerating `seedTrain()`'s training examples from the
same `HANDWRITTEN_STROKES` paths (rasterized the same way `getGrid()` already resamples real ink)
for all six seed letters, not reverting this rendering change.

## Look inside: tiles and heatmap (`refreshLookInside`)

**The section's intro copy states the architecture up front, in plain language, before describing
what the tiles/heatmap show** — added on direct feedback ("without understanding the network
architecture I don't even truly know what the 'look inside' is showing me"). It now opens with "the
whole network is only three layers: your drawing becomes 100 pixel values... 18 hidden neurons...
26 output neurons" before the receptive-field explanation, and the tile/heatmap caption below was
rewritten to say plainly that a tile *is* one hidden neuron's real 100 weights reshaped to the
drawing grid, and the heatmap is the *next* layer's weights, one row per hidden neuron in the same
order as the tiles — spelling out the connection between the two visuals explicitly rather than
assuming a reader already holds the architecture in their head.

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

## Export: QR code, the network-shape diagram, and two other export paths

**Quantization is hardcoded to 4-bit (`QUANT_BITS = 4`), not a toggle.** An 8-bit/4-bit choice used
to sit in the main "04 — save & share" panel alongside a full byte-count ledger table; removed on
direct feedback ("doesn't need all the quantization options, just go with 4-bit... doesn't need all
the metrics"). There was never a real choice to offer anyway — see "QR capacity" below, 8-bit
doesn't reliably fit a QR code at this network's size while 4-bit always does. `quantizePayload()`
itself is unchanged (still takes a `bits` argument, just always called with `4` now): finds the
single largest-magnitude parameter, derives one shared linear scale (`maxAbs/levels`, 7 levels at
4-bit, signed range `-8..7`), quantizes every weight/bias to it, and packs
`[float32 scale][bits, GRID, HID, OUT][packed weights, two nibbles per byte]`. The byte-count detail
that used to live in the on-page ledger now lives as prose in "how this actually works" instead.

**The network-weights diagram (`renderArchDiagram`, `#archCanvas`, `#downloadArchBtn`) sits right
next to the QR code**, added on direct request: "I want the thing that someone might print on a
mug: the QR code and the actual network architecture... isn't that something we should be able to
download?" Its first version was a generic three-box-and-two-arrows schematic (a canvas port of the
same static SVG already in "how this actually works") — a real, follow-up request pushed past that:
*"I want this to show the actual weights... seems like we could draw all that here, couldn't we? At
least a cool visual."* The current version draws every real connection this network has, not an
illustration of the shape: for each of the 100 input pixels, a line to each of the 18 hidden dots
colored by the sign of that exact `W1[h][k]` weight (rose positive, blue negative) and opacity by
its magnitude relative to the network's own largest weight; the same again from each hidden dot to
each of the 26 lettered output dots via `W2`. Hidden and output dots are filled by that neuron's
real bias via `divergingColor()`, the identical convention `refreshLookInside()`'s tiles already
use — this is a second, denser view of the *same* live `W1`/`b1`/`W2`/`b2` arrays, not a separate
approximation of them. The input grid itself has no single "the weight" for a pixel (it has 18, one
per hidden neuron), so it's filled by `impact[k] = mean_h(|W1[h][k]|)`, the one honest single-number
summary of that pixel's overall influence — this is what makes a trained network's diagram visibly
different from a fresh/random one (a trained "H" produces a visible receptive-field-shaped blob in
that grid; a freshly-reset network shows uniform noise). Connections below `|w|/maxAbs < 0.12` are
skipped entirely rather than drawn at near-zero opacity, both for legibility (≤100×18 + 18×26 ≈
2,268 possible lines is already dense) and because a genuinely negligible weight isn't meaningfully
"a real connection" worth ink. Every position is still derived from the real `GRID`/`HID`/`OUT`
constants, not hand-tuned to "18 dots," so the layout stays correct if the network's shape ever
changes. Unlike the original schematic (rendered once at load, since the *shape* never changes),
this version is re-rendered every time the weights actually do — wired into the same call sites as
`refreshLookInside()` (after Train, after Reset, after a successful weight Load) — so it's always a
live picture of the current network, exactly like the tiles and heatmap are. Downloadable via
`archCanvas.toDataURL('image/png')`, same as every other canvas export on this page.

**QR download** (`#downloadQrBtn`): qrcodejs renders a hidden `<canvas>` plus a visible `<img>`
whose `src` it already set to that canvas's own `toDataURL('image/png')` — same confirmed behavior
already documented in Echo State's CLAUDE.md ("Two Zazzle links" section doesn't apply here, but
the img-first QR-download trick is identical) — so the button just grabs that `<img>`'s `src`
directly, falling back to encoding the canvas itself only if qrcodejs's DOM ever changes.

**Full-precision copy/paste** (`fullPrecisionJSON`, `weightsOut`/`weightsIn`) now lives inside "how
this actually works," not the main share panel — moved there on direct request, alongside the
quantization/byte-count prose it's already adjacent to, since it's the more technical of the two
export paths and the main panel is meant to stay down to "the two things worth printing." Mechanics
unchanged: raw, unquantized floats as JSON, `loadWeightsBtn` validates the shape matches
`GRID`/`HID`/`OUT` before accepting it, restores the network's numbers only (not the `dataset` array
of drawn examples) — teaching is one-way within a session unless the tab stays open.

**Portrait export** (`portraitBtn`/`#downloadPortraitBtn`): composites the 18 tiles, a scaled copy
of the heatmap, and the current decoded test word into one 560×520 canvas, rendered to
`#portraitImg` and only then exposed for download — `#downloadPortraitRow` starts hidden and is
only revealed after a real portrait has been rendered, the same "no download button before there's
something real to download" convention every other tool's spinner-gated downloads already follow.
Its on-page description was rewritten after direct feedback that "portrait" as a name didn't
communicate anything ("I still have no idea what that network portrait is") — the paragraph now
spells out literally what gets composited (the step-03 tiles, the step-03 heatmap, the step-01
word) rather than assuming the "portrait" metaphor was self-explanatory.

## QR capacity: 8-bit doesn't actually fit, and defaulting to it was a real bug

The prototype assumed the 8-bit quantized payload would "fit in a QR code" without ever checking
against the real vendored library's actual capacity. It doesn't: this network has 2,312
parameters, so the 8-bit payload is 2,320 raw bytes → ~3,096 base64 characters, and `new
QRCode(...)` (davidshimjs/qrcodejs, `CorrectLevel.L`) throws past a real ceiling confirmed by
binary search directly against this exact vendored file to be ~2,950 base64 characters — a
`TypeError` inside the library's own internal array indexing, not a caught/graceful failure. Left
unhandled, that throw aborted every top-level `<script>` statement after it, which is why the
first shipped version crashed silently the moment `refreshLedger()` ran on page load.

Fixed two ways, both in `refreshLedger()`: (1) `QR_MAX_B64_CHARS` (`2900`, a rounded-down safety
margin under the empirical ~2,950 ceiling) is compared against the payload's *actual* base64
length before ever calling `new QRCode(...)`, replacing the old and simply wrong `current <= 1500`
raw-byte check that had no real basis; (2) the `new QRCode(...)` call is also wrapped in try/catch
as a second line of defense, so a future change to network size can never again take down the whole
page's script execution the way it did here — it degrades to a fallback message in `#qrHolder`
instead, and disables `#downloadQrBtn`. Quantization was also flipped from an 8-bit default to a
permanent, hardcoded 4-bit (see "Export," above) — at this network's size 4-bit's ~1,552 base64
characters always clears the ~2,900 ceiling with room to spare, so the fallback path is a real
safety net for if the network ever grows, not something a visitor can currently trigger.

If `HID`/`GRID`/`OUT` ever change (making the network smaller), 8-bit may start fitting again on
its own — that's fine, the length check is computed live from the actual payload every time, not
hardcoded to this network's current 2,312-parameter size.

## Zazzle link is a placeholder, not a verified template

Same situation as every other tool's *first* ship: `#shareSection`'s "make it a gift" link reuses
the same known-working generic mug template ID (`256206116898885602`) and the site's standard
`?rf=238054754631086278` ambassador param, since no Inkling-specific curated product exists yet.
Its copy now suggests the specific two-sided use case the QR-plus-diagram pairing above is built
for (QR on one side, network diagram on the other), matching Echo State's established two-sided-
mug precedent, while still pointing at the same generic template link — no real Inkling-specific
product has been verified to exist yet. Unlike Echo State and Ridgeline, there's no hero CTA here
at all yet either — those only got added once a *real, specific* product existed, and this tool was
shipped without one being provided. Add a real hero CTA the same way (right after the intro
paragraph, `?rf=` included) once a real product exists — don't fabricate one or guess a product URL
in the meantime.

## Testing changes

No test suite — static page. Verify via a local static server (root-relative `/nav.js` and
`/theme.css` mean `file://` won't pick them up). Golden path: confirm the page opens with section
01 ("Give it a word") showing the hand-drawn-looking "HAT" first, section 02 ("Teach it a letter")
below it, and the seed letters already taught (chip counts non-zero for H/I/A/B/C/T) → click "Tap
to draw a letter and teach it", confirm the wizard opens to the draw step, draw something, click
"Next →", confirm it switches to the letter-picker step with Train disabled until a letter is
picked, pick one, click Train, confirm the modal closes and the stage animation plays real
per-active-pixel fan-out lines from the input raster into the hidden column (not a single bundled
line), then hidden→output, then dashed backward lines all the way back to the same input pixels →
confirm the compact "reads today's word... as: X" line under the diagram updates in sync with
section 01's own decoded word → back on the page, click a specific letter chip (not the generic
trigger), confirm the wizard opens with that letter named in the header and already pre-selected/
Train-enabled in the letter-picker step → in section 01, change one word-pad letter and confirm it
re-decodes and logs a new attempt with no separate button click needed, then click "Clear all" and
confirm every pad clears and it re-decodes to blanks in one action → click "Reset network" twice
(first click only arms the confirmation toast) and confirm chip counts and `trainStatus` both go to
exactly 0, the look-inside tiles snap to random noise (not the seed letters' shapes), the word
decode updates to reflect the now-random network (not a stale pre-reset value), and the network-
weights diagram's input grid visibly loses whatever letter-shaped pattern it had and shows uniform
noise instead → confirm the QR code and the network-weights diagram both render in step 04 with no
quantization toggle or ledger table visible, and both "Download QR as PNG" and "Download diagram as
PNG" save real, non-empty PNGs → train one more letter and confirm the diagram's connections/dot
colors visibly change afterward (it re-renders on every Train, not just at page load) →
open "how this actually works" and confirm the full-precision copy/load weights UI is there (not in
step 04) → copy the full-precision weights, clear/retrain a little, then paste the
copied JSON back into "Load weights" and confirm the look-inside tiles snap back to the earlier
state (a real round-trip, not just "no error thrown") → render a portrait, confirm the download
button only appears after rendering (not before), and confirm the downloaded PNG actually contains
the tiles/heatmap/word, not a blank canvas.
