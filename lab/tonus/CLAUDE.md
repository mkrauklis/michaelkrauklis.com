# Tonus

Generate a 2D dataset — six built-in shapes, or your own drawn/uploaded points — and watch seven
genuinely different machine learning models each compute their own real decision boundary over
it: a closed-form linear fit, a gradient-descent-trained logistic curve, a hinge-loss linear
classifier, k-nearest-neighbors, Gaussian Naive Bayes, a hand-rolled random forest, and a real
backprop-trained neural network (via TensorFlow.js). Started life as a spec drafted in a separate
conversation, sanity-checked against this site's actual conventions before anything was built
(see "Where this started" below), then built directly in this repo.

## Where this started, and what changed before building

The original spec (drafted elsewhere, transcribed and sanity-checked in this repo's working
notes before being discarded — see the "Where this started" convention other tools' CLAUDE.md
files describe, e.g. Vectis's `vectis-spec.md`) called for TailwindCSS, TypeScript, WebGL, Web
Workers, and XGBoost. None of those shipped:

- **TailwindCSS and TypeScript** would both require a build step, which this entire site
  deliberately doesn't have (see root `CLAUDE.md`). Tonus is plain inline `<script>` JS and
  `theme.css`'s existing components, same as every other tool here.
- **WebGL and Web Workers** were dropped as solving a performance problem that doesn't exist —
  the "100×100 spatial grid" is 10,000 cells, trivial for a plain 2D canvas fill, and every
  classical model here trains in milliseconds on a few hundred points. The one genuinely
  expensive computation (the neural network) already gets real GPU acceleration through
  TensorFlow.js's own WebGL backend without needing a Worker on top of it.
- **XGBoost was dropped entirely, not swapped for another tree-boosting library.** There's no
  realistic "vanilla JS XGBoost" — real XGBoost builds are inference engines for a model already
  trained offline in Python, not something you train live from a browser click. More importantly,
  once asked what XGBoost was actually *for* in this spec (direct answer: "alternatives to neural
  nets... compare to some more traditional machine learning methods"), a second tree-ensemble
  method next to Random Forest wasn't the best use of a model slot anyway. It was replaced with
  **k-Nearest Neighbors** and **Gaussian Naive Bayes** — two paradigms (instance-based, and
  properly probabilistic) that weren't represented at all otherwise. The final seven-model roster
  spans linear, probabilistic, instance-based, tree-ensemble, and deep-learning approaches, which
  tells a much richer "here's what's actually different about these techniques" story than three
  tree/ensemble variants would have.

**The name** was chosen from a short Latin-word brainstorm, once "alternatives to neural nets /
traditional ML methods" and "something Latin evoking hyperparameter tuning" were both settled.
*Tonus* is Latin for "tension, pitch" — the literal etymological root of "tune." Two other
finalists are worth recording in case a future tool wants them: **Temperies** (Latin for "a
proper mixture/tempering," the root of "temperature" and "temper" as in tempering steel — richer
metaphor, iterative refinement toward an optimal state) and **Gradus** (Latin for "a step,"
root of "gradient" — the most mathematically literal of the three, since most of these models
converge via actual gradient steps).

## Architecture

`index.html` plus a vendored `qrcode.min.js` (davidshimjs/qrcodejs v1.0.0, MIT — the identical
file already vendored for Echo State and Inkling; see Echo State's CLAUDE.md for why it's local
instead of a CDN import) and one CDN script, `@tensorflow/tfjs@4.22.0`, loaded the same way
Afterimage already loads it. No build step, no bundler — every dataset generator and every model
except the neural network is hand-written array math, matching this site's standing rule of real,
inspectable computation. Links the two site-wide shared files (`/nav.js`, `/theme.css`) like
every other tool.

Accent is a periwinkle-indigo blue (`--accent:#6f8fe0`), distinct from every other tool's accent
(amber, teal, violet, rust, cyan, rose). The two *data class* colors (`--c0`/`--c1`) are
deliberately separate constants from the page accent — they encode which class a point belongs
to, not which tool this is, the same separation Inkling keeps between its rose "accent" and its
blue "cool" excite/inhibit color. They're the Okabe-Ito colorblind-safe orange/blue pair
(`#e69f00`/`#0072b2`) — the same verified pair Inkling's colorblind toggle already uses (see that
tool's CLAUDE.md) — not a fresh guess at "looks distinct enough," and not a toggle here at all:
since data-encoding color is the *entire point* of this tool (every visualization is built around
"which of these two colors is this"), there's no reason to ship a less-safe default and gate the
better one behind an opt-in switch the way Inkling and Echo State do for their own diagrams.

Every data point (on the main plot, the export grid's panels, and the hero canvas) is drawn with
a light `#e9e7de` stroke around its fill — a first version used a dark, semi-transparent stroke,
which all but disappeared against the heatmap's own dark-ish mid-tones right where points most
need to stay legible (near the decision boundary itself, where the background color is closest to
50/50 between the two classes and therefore darkest/most desaturated in the lerp).

## Dataset generators (`GENERATORS`, seeded via `mulberry32`)

Six built-in shapes (`linear`, `xor`, `circles`, `spirals`, `moons`, `hidden`), plus a seventh
`points` type that covers both hand-drawn and CSV-uploaded data (no seed to replay for either, so
they're serialized as raw points instead — see "Sharing," below). Every generator takes the same
seeded PRNG (`mulberry32`, the same "deterministic, not `Math.random()`" discipline Echo State and
Inkling already use) so a given seed always produces the exact same dataset — the seed alone,
without ever transmitting the points themselves, is what makes a shareable link small.

`genSpirals`/`genMoons` run their output through `centerAndFit()` (recenter on the point cloud's
own mean, then uniformly scale so the furthest point lands at 0.92) rather than hand-tuning
magic-number offsets to land inside `[-1,1]` — their underlying parametric formulas don't
naturally center or bound themselves, and a scale-and-recenter pass is more robust to future
tweaks than re-deriving fixed constants by hand every time the shape changes slightly.

**"Hidden message"** renders the text "AI" onto an offscreen 240×240 canvas, then labels each
randomly-sampled point by whether it lands on drawn ink or not — invisible on a plain scatter
plot (the point density looks uniform either way), only resolved once a model with enough
capacity actually finds the boundary. Same idea as the spec's original "concealed data" pitch,
implemented directly rather than as a separate curated preset — there was never a meaningful
difference between "the hidden message dataset" and "a specific instance of it" once it's just
one generator with one hardcoded message.

## Models

Each model exposes the same two-function shape: a `train*(points, params)` that returns whatever
state it needs, and a `predict*(model, x1, x2)` returning a 0–1 confidence used both for the
0.5-threshold class call and for the heatmap's color. The neural network is the one exception —
see below.

- **Linear Regression** solves the normal equations directly (`solve3x3`, Gaussian elimination
  with partial pivoting) rather than iterating — genuinely the one model here computed exactly
  in closed form, not approximated by gradient descent. It's also the one model classifying by
  accident: it was never trained to output a probability, just repurposed via a 0.5 threshold,
  which is exactly why it's usually the worst performer on anything non-linear.
- **Logistic Regression** is real batch gradient descent on cross-entropy loss (300 epochs,
  fixed learning rate 0.6) — the same `(prediction − label)` gradient form Inkling's own output
  layer uses, just for a 3-parameter model instead of a 64→18→26 network.
- **Linear (hinge loss)** is the Pegasos algorithm (Shalev-Shwartz, Singer & Srebro) — stochastic
  subgradient descent on hinge loss with L2 regularization, learning rate `1/(λt)`. Deliberately
  labeled "linear classifier, hinge loss," never "SVM" — a true SVM is defined by its dual
  formulation and support vectors, and this is a primal, linearly-parameterized model trained to
  a similar-looking objective, not the same algorithm.
- **k-Nearest Neighbors** has no training step at all — `trainKNN` just stores the points and
  `k`. It's the one model that can't generalize past its own data by construction, which is the
  whole pedagogical point of including it: turn `k` down to 1 and the boundary traces every
  single point's own tiny neighborhood.
- **Gaussian Naive Bayes** is the one properly probabilistic model — everything else either fits
  a boundary directly or votes among neighbors; this one models each class's own per-axis
  Gaussian distribution (independent axes, the "naive" assumption) and applies Bayes' rule. Two
  Gaussian classes make the posterior collapse into the same sigmoid shape logistic regression
  produces, but the two boundaries diverge whenever the classes' spreads differ, since NB is
  actually modeling variance and logistic regression isn't.
- **Random Forest** (`buildTree`/`trainRF`) is a real bootstrap-aggregated ensemble: each tree
  is grown on an independent resample-with-replacement of the actual data, splitting greedily on
  whichever of the two axes and threshold most reduces Gini impurity, recursing to a configurable
  max depth. A cell's confidence is the plain average of every tree's leaf value. The
  characteristic stepped, orthogonal boundary isn't a stylistic choice — it's the direct
  consequence of every individual tree only ever being able to cut the space with straight
  horizontal/vertical lines.
- **Multi-Layer Perceptron** (`trainMLP`/`predictMLPGrid`) is the one model trained with real
  backpropagation through more than one layer, via TensorFlow.js (`tf.sequential`, 1-4 configurable
  dense layers, 2-64 neurons each, ReLU/sigmoid/tanh, Adam optimizer, binary cross-entropy loss).
  Trains to real convergence rather than a fixed, user-chosen epoch count (see "The actual fix"
  section below for why, and for the real bugs that section's whole design went through), and gets
  its own live architecture diagram right under its sliders (see "Architecture diagram" below).
  **Known simplification, not yet hit as a real bug**: unlike Afterimage's decoder, this doesn't
  check for or fall back from low-precision WebGL (`floatPrecision()` returning 16 instead of 32 —
  see Afterimage's CLAUDE.md for why that's a real GPU/driver ceiling, not a config flag). At this
  network's tiny size the precision loss that broke Afterimage's much deeper gradient search
  hasn't shown up in testing; if an MLP boundary ever looks visibly wrong for no other reason,
  that's the first thing worth checking.

## `model.fit()` and `requestAnimationFrame`: three attempts before the real fix

**Superseded by the section below — kept as the record of what was actually tried and why each
attempt failed, since the failures themselves are the useful part.** The final, shipped fix is a
global `requestAnimationFrame` monkey-patch plus tf.js's own default yielding, not any of the
approaches described here.

**Attempt 1 — do nothing (tf.js defaults).** The very first working version of `trainMLP()` called
`model.fit(...)` with no special options, and every training run simply hung — the returned
promise never resolved, `trainStatus` stayed on "training…" forever, no error. Confirmed the
mechanism directly: `tf.nextFrame()` — which `model.fit()` calls internally between epochs to
yield control back to the browser — is backed by `requestAnimationFrame`, and
`requestAnimationFrame` **never fires while `document.visibilityState !== 'visible'`** (confirmed
directly: `tf.nextFrame().then(...)` left unresolved for 2+ seconds while the tab reported
`hidden`). Same root cause several other tools in this repo have already hit for their own canvas
animation loops (see Inkling's and Vectis's CLAUDE.md) — the first time it turned out to affect a
*library's* internal implementation rather than this codebase's own code.

**Attempt 2 — `yieldEvery: 'never'` for the whole training run.** This does fix the hang (skips
the `tf.nextFrame()` yield entirely) and was shipped briefly. It was wrong: **direct user report,
in a real browser tab, not this automation environment — "when I change anything it all freezes
up."** Removing tf.js's yielding removes the exact mechanism that's supposed to keep the page
responsive during a multi-second computation; at this network's largest allowed size (4 layers,
64 neurons, up to 250 epochs) that's a real, multi-second unbroken block of the main thread, not a
theoretical concern.

**Attempt 3 — one `model.fit()` call per single epoch, `yieldEvery:'never'` on each, yielding via
a plain `setTimeout` in between.** This does keep the page responsive (verified: the live epoch
counter visibly advanced across separate polls during training, which is only possible if the
main thread is actually yielding control back to the event loop between epochs) — but introduced
a *different*, also-real problem: benchmarked directly, isolated from the rest of the app, calling
`fit()` repeatedly with `epochs:1` each time cost roughly a **flat ~1000ms per call**, regardless
of how many epochs that call actually covered (`epochs:1` and `epochs:10` both took ~1000ms; the
real per-epoch compute, ~5-40ms, was negligible next to that fixed cost). Training 250 epochs this
way would take minutes, not seconds — fixed the freeze, broke usability a different way.

## The actual fix: patch `requestAnimationFrame` once, before tf.js loads, and let it yield normally

The real problem was never "yielding is expensive" — attempt 3's benchmark makes that clear (the
per-epoch compute is cheap; something else was slow). It's that `requestAnimationFrame` doesn't
fire in a hidden tab, so every approach that depended on it (attempt 1) hung, and every attempt to
route around it by hand (attempts 2 and 3) fought tf.js's own scheduling instead of fixing the
actual broken primitive. The fix addresses that directly: a `<script>` tag right before the
`tfjs` CDN `<script>` tag replaces `window.requestAnimationFrame` with a plain
`setTimeout(cb, 16)`-based implementation, unconditionally, for the whole page:

```js
window.requestAnimationFrame = function(cb){ return setTimeout(function(){ cb(performance.now()); }, 16); };
```

This has to happen **before** tf.js's own script tag, not from inside this page's main script —
tf.js reads `requestAnimationFrame` once, when its module initializes, and keeps that reference;
patching it later (confirmed directly, the hard way) has no effect on what tf.js already captured.
This page has no other real use for `requestAnimationFrame` — the hero canvas is a one-shot static
render, not a continuous animation — so replacing it globally here has no visible side effect of
its own, and it makes tf.js's real default yielding (no `yieldEvery` override at all, letting
`fit()` yield via its own internal `'auto'` policy) work correctly instead of being disabled or
reimplemented by hand.

`trainMLP()` now makes exactly **one** `model.fit()` call, for up to `MAX_EPOCHS` (250), using
tf.js's own `callbacks.onEpochEnd` hook — the idiomatic way to get per-epoch progress and to stop
early (`model.stopTraining = true`) once loss stops meaningfully improving over the last 10 epochs,
replacing the "train until it converges, capped at 250" behavior a user-facing epoch-count slider
used to leave to guesswork. This also directly enables the live "training… epoch N/250" status
text (`onProgress`) and the "known this many epochs" cap is now a real training decision, not a
number someone has to pick.

**A second bug this restructuring surfaced and fixed**: since `model.fit()` can't be cancelled
once started, a superseded training run (e.g. the user drags "neurons per layer" again before the
previous drag's retrain finishes) used to keep running all the way to its own convergence or
`MAX_EPOCHS`, blocking every newer request behind it — JS is single-threaded, so a new `fit()`
call can't even *start* until the old one's promise settles. `onEpochEnd` now also checks a passed-
in `isStale()` function (`mySeq !== trainSeq`, the same staleness token `recompute()` already used
elsewhere) and calls `model.stopTraining = true` immediately if a newer request has since
superseded this one — so a stale run gives up its hold on the main thread within one epoch instead
of running to completion first. Verified directly: started training the default network, then
immediately switched to Logistic Regression before the neural network had even reached its first
epoch — the classical model's own (fast, synchronous) result appeared within ~2 seconds rather
than waiting behind the abandoned neural-network run.

**A genuine testing-environment limit, not glossed over**: this automation environment's
`document.visibilityState` stays `'hidden'` for the whole session (same documented limitation
elsewhere in this repo), and tf.js's *own* internal handling of a hidden tab appears to fall back
to a much slower polling cadence even with the `requestAnimationFrame` patch in place (`tf.nextFrame()`
measured at 400-1000ms per call here, vs. the patch's own intended ~16ms) — this could not be
fully resolved or timed end-to-end from inside this environment, and the browser-automation tool
used to drive it here also appears to reset in-page progress whenever one of its own calls times
out, rather than letting training continue in the background. What *was* verified directly: no
permanent hang (real epoch-by-epoch progress, confirmed across many separate checks), no console
errors under repeated stress, correct final results once a run does complete, and the stale-abort
mechanism working within a couple of seconds. Full wall-clock training speed in a normal, visible,
focused browser tab — the actual target environment — should be spot-checked directly there before
assuming this is fully tuned; there is no technical reason to expect the ~1s/epoch figures measured
here to reflect real usage, since none of the slow paths this investigation found are gated on
anything except `document.visibilityState`, which is only ever `hidden` in this sandbox.

## Architecture diagram (`renderMLPArch`, `mlpArchCanvas`)

Direct request: "we need a visual of the neural net architecture." Lives directly under the MLP's
hyperparameter sliders (not in "how this actually works" — it's meant to be looked at *while*
tuning the sliders, not read about afterward). `mlpLayerSizes(params)` derives the node-count-per-
layer array (`[2, width, width, ..., 1]`) straight from the same `params` object driving the
sliders, so it's structurally impossible for the diagram to drift out of sync with what's actually
configured.

**Two states, not one.** Moving a layers/width/activation slider calls `renderMLPArchIfCurrent()`
immediately — before the debounced retrain even fires — redrawing the diagram's *shape* right
away (neutral gray edges, since there's no trained model matching this shape yet). Once training
actually completes, the same function redraws it colored by the network's real weights (blue
positive, rust/`--danger` negative, opacity by magnitude relative to that layer's own largest
weight — the same diverging-by-magnitude convention every other real-weight visualization on this
site uses). `mlpModelMatchesParams()` is the guard that decides which state to show: it checks the
live model's actual tensor count and first-layer width against the *current* `params`, so a model
that finished training against an now-outdated slider position is never mistakenly drawn as if it
were current — this exact "stale model, current sliders" state is common now that hyperparameter
drags update the diagram structurally before the real retrain lands. A caption under the canvas
(`#mlpArchNote`) states in plain language which of the two states is showing, since "gray lines"
vs. "blue/rust lines" isn't self-explanatory without it.

At the largest allowed size (4 layers × 64 neurons) this draws up to roughly 16,500 individual
edges; confirmed this stays a fast, single synchronous draw (not a per-frame animation, so it only
needs to be fast once per change, not 60 times a second) — no separate performance mitigation
(e.g. edge culling) was needed in testing.

## Export comparison grid: labels used to sit on top of the heatmap they were labeling

**Direct report: "I can't read the words on the export PNG."** The bug was a real geometry
mistake, not a font/contrast issue on its own: `panelH` (each panel's height) left only 10px of
margin below where a panel's own heatmap ended, and the label text was drawn at `canvas.height -
14` — a coordinate *inside* that occupied region, not below it. Canvas drawing leaves permanent
pixels; a `ctx.save()`/`clip()`/`ctx.restore()` scope only affects what's drawn *during* that
scope, not what a later, unscoped `fillText()` call paints on top of pixels already there — so the
label text was landing directly on the heatmap's own (often light-colored, sometimes near-white)
background, not on the dark base fill it looked like it should be sitting on. Fixed by genuinely
widening the margin (`panelH = canvas.height - 100`, up from `- 60`) so every label sits entirely
within a real, untouched band of the base `#10131a` fill — verified directly by sampling pixel
colors in that exact band after a real export (dominant color: `rgb(16,19,26)`, i.e. the base fill,
not heatmap orange or blue) rather than trusting the geometry by eye. Font size and weight were
also bumped (title 22px→30px, panel labels 14px→20px, both bold) while this was being fixed, since
the original sizing was designed around the *intended* clean background, not tested against the
actual overlap bug.

## Throbbers (`.spinner-inline`, `#computeSpinner`, `#exportSpinner`)

Direct request: "we need some sort of throbber while it's thinking for pretty much all the
steps." A single `#computeSpinner` (the same rotating-ring `.spinner-inline` pattern already
established in Inkling, copied verbatim rather than reinvented) sits next to the accuracy readout
and covers every path through `recompute()` — which is every dataset change, every model switch,
and every hyperparameter tweak, not just the neural network, since all of them funnel through that
one function. `recompute()` now shows the spinner and does one real `setTimeout(0)` yield *before*
running any computation, classical or not — most classical models finish in a handful of
milliseconds, too fast for a spinner to ever actually paint without this, but Random Forest at a
high tree count and depth on a large dataset is a real, measurable exception, and there was
previously no visual feedback at all for the gap between "you changed something" and "the boundary
updated," on any step. The export button gets its own separate `#exportSpinner`, shown for the
entire multi-model export (including, now, live per-epoch text during that export's own neural-
network panel).

## A real bug: switching to an empty custom dataset while the Neural Network is selected

**Also found directly during testing**, right after fixing the hang above. Selecting "Custom
points" (which starts with zero points) while the Neural Network model was already active threw
`tensor2d() requires shape to be provided when values are a flat/TypedArray` — `tf.tensor2d([])`
has nothing to infer a shape from — and then every *subsequent* recompute threw `Container
'sequential_1' is already disposed`, repeating on every future retrain attempt. The second error
is a direct consequence of the first: `trainMLP()` disposes the previous model immediately, before
building the new one, but the first error happened *after* that dispose and before
`state.mlpModel` was ever reassigned — so `state.mlpModel` was left pointing at an already-disposed
model, and the next retrain's dispose call hit that same stale, already-disposed reference. Fixed
at the source rather than patched at the symptom: `recompute()` now checks `state.points.length`
first and returns immediately (clearing `trainedModel`/`mlpModel`, showing "No points yet") before
attempting to train *anything*, of any kind — there was never a case where training on zero points
was meaningful to begin with, for any model, not just the neural network. `computeGridProbs()`
also short-circuits to `null` on zero points as a second line of defense, so a future model type
added here can't reintroduce this exact class of bug by skipping the check in `recompute()` alone.
Verified the fix by reproducing the exact original sequence (select Neural Network, then switch to
an empty custom dataset, then draw one point) and confirming no console errors and a real 100%
accuracy result once a point exists.

## Custom points: drawing and CSV upload share one internal representation

Both "draw your own" and "upload a CSV" set `state.datasetType = 'points'` and populate
`state.points` directly — there's no meaningful difference between them once the points exist, so
they share one code path (and one share-payload encoding) rather than two. Drawing
(`plotCanvas`'s `mousedown`/window-level `mousemove`/`mouseup` — window-level so a drag that
leaves the canvas mid-gesture still completes, the same pattern Vectis's and Ridgeline's own
draggers use) adds a point on click and continues adding points as the cursor moves at least 0.05
units in data space from the last one, using whichever class the two-button toggle currently has
selected. `parseCSV()` expects `x1,x2,label` columns (header row optional — detected by checking
whether the first cell parses as a number), and always min-max rescales the parsed values into
`[-1,1]` regardless of their original scale, per the original spec's normalization requirement.

## Sharing: the same quantized-payload-in-a-URL scheme as Echo State and Inkling

A 12-byte header — dataset index, model index, a 4-byte seed, a 2-byte point count, and 4 bytes
of model-specific hyperparameters (meaning depends on which model is selected: regularization
strength for the hinge-loss classifier on a log scale, `k` for k-NN, tree count and max depth for
Random Forest, or layer count/width/activation for the neural network — training itself always
runs to real convergence rather than a chosen epoch count, so there's nothing to encode for it)
— covers every
built-in dataset in just those 12 bytes, since the seed alone is enough to regenerate the exact
same points. A `points`-type dataset has no seed to replay, so its actual points are appended
after the header instead, 3 bytes each (`x1`/`x2` quantized to int8, `y` as a plain 0/1 byte) —
still small enough that even a few hundred hand-drawn points comfortably clears the ~2,900
base64-character QR ceiling documented in Inkling's CLAUDE.md. Verified end-to-end, not just
assumed symmetric: generated a link from a live 6-point custom dataset with Random Forest
selected, loaded that URL completely fresh, and confirmed the dataset, model, hyperparameters, and
every point (up to int8 quantization's expected small rounding) all matched the original session's
displayed accuracy.

## Export: the comparison grid, not the mug print layout literally

The original spec called for a hardcoded 4-panel PNG (Logistic Regression, Random Forest, a 3-layer
network, and gradient-boosted trees) at a fixed 3:1 mug-wrap aspect ratio with an embedded QR code.
Built close to that literally, with two adjustments: the fourth panel is k-Nearest Neighbors (the
gradient-boosted-trees slot doesn't exist anymore — see "Where this started," above), and the QR
in the bottom-right corner encodes the *current* share payload (same `buildPayload()`/
`toBase64Url()` the main share section uses) so scanning the exported print reopens this exact
dataset, not a generic link. Each panel retrains its own model fresh from the current dataset via
the same `renderPanel()` function the main plot canvas uses — this guarantees the exported image
shows the same real computation the interactive view would, not a separately-maintained preview
that could silently drift out of sync with it.

## Zazzle link is a placeholder, not a verified template

Same situation every other tool's first ship starts from: `#shareSection`'s mug-template link
reuses the same generic "upload your own image" mug template ID and the site's standard
`?rf=238054754631086278` ambassador param, since no Tonus-specific curated product exists yet.
Copy is written to point specifically at the exported comparison grid (the thing this tool
actually produces worth printing), not a generic "make it a gift" line. Replace with a real,
verified Tonus-specific product link once one exists — don't fabricate one in the meantime.

## Testing changes

No test suite — static page. Verify via a local static server (root-relative `/nav.js` and
`/theme.css` mean `file://` won't pick them up). Golden path: confirm the page opens with the
Moons dataset and Logistic Regression selected, a real heatmap and scattered points visible, and a
training-accuracy readout well under 100% (moons genuinely isn't linearly separable) → switch to
XOR and confirm Logistic Regression and the hinge-loss linear classifier both land near 50%
accuracy while k-Nearest Neighbors, Random Forest, and the Neural Network all reach 100% — this is
the core pedagogical claim of the whole tool, so it's the single most important thing to
re-verify if any model's math is ever touched → confirm Gaussian Naive Bayes lands well below 100%
on XOR too (its independence assumption is a textbook failure case for exactly this dataset) →
drag every hyperparameter slider for k-NN/Random Forest/the Neural Network and confirm the
boundary and accuracy actually change, not just the displayed number → for the Neural Network
specifically, confirm there's no epoch-count slider at all, confirm the architecture diagram
redraws its *shape* the instant a layers/width slider moves (before the retrain finishes — check
the "structure only" caption) and switches to real-weight coloring with the "colored by this
network's real trained weights" caption once a training run completes → switch to "Custom points"
with **zero points already present** while the Neural Network is selected and confirm no console
errors and a clean "No points yet" message (the exact regression documented above) → draw a small
shape, confirm points are added continuously while dragging and the boundary retrains after
release → upload a small CSV with `x1,x2,label` columns at an arbitrary scale (not already in
`[-1,1]`) and confirm the points land inside the plot, correctly rescaled → copy the share link,
open it in a fresh tab/session, and confirm the dataset, model, every hyperparameter, and (for a
custom dataset) every point round-trip correctly → click "Export comparison grid," confirm all
four panels render with visibly different boundaries for the same underlying points, every panel
label and the title are cleanly legible against the plain dark background (not overlapping any
panel's own heatmap colors — the exact bug documented above), a QR code appears in the bottom-right
corner, and "Download comparison as PNG" saves a real, non-empty file → confirm a spinner
(`#computeSpinner`) is visible next to the accuracy readout during *every* kind of change (dataset,
model, and hyperparameter, not just Neural Network ones) and a separate spinner covers the whole
"Export comparison grid" action → start training the Neural Network with a large layer/width
setting, then immediately switch to a fast classical model before it's finished, and confirm the
classical model's real result appears within a couple of seconds rather than waiting behind the
abandoned neural-network run (the stale-training-abort fix, `isStale` in `trainMLP`).

One environment-specific trap worth knowing before touching `trainMLP()` again: this automation
environment's `document.visibilityState` stays `'hidden'` throughout (a documented limitation
elsewhere in this repo, e.g. Vectis's and Inkling's CLAUDE.md), and testing here found that tf.js's
own internal handling of a hidden tab is *slower*, not just blocked, even past the
`requestAnimationFrame` monkey-patch described above — full wall-clock training-speed
verification could not be completed from inside this environment (see "A genuine testing-
environment limit" above) and should be spot-checked in a real, focused browser tab after any
future change to `trainMLP()`. Don't assume every timing oddity seen while testing here is *just*
the environment, either, the way the empty-dataset bug above genuinely wasn't — check for real
console errors and incorrect final results first, and only attribute pure slowness (not
incorrectness) to this limitation.
