# Tonus

Generate a 2D dataset — six built-in shapes, or your own drawn/uploaded points — and watch eight
genuinely different machine learning models each compute their own real decision boundary over
it: a closed-form linear fit, a gradient-descent-trained logistic curve, a hinge-loss linear
classifier, a real kernelized SVM, k-nearest-neighbors, Gaussian Naive Bayes, a hand-rolled random
forest, and a real backprop-trained neural network (via TensorFlow.js). Started life as a spec
drafted in a separate conversation, sanity-checked against this site's actual conventions before anything was built
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
  properly probabilistic) that weren't represented at all otherwise. The roster (later joined by a
  real kernelized SVM — see "SVM (RBF kernel)" below) spans linear, kernel, probabilistic,
  instance-based, tree-ensemble, and deep-learning approaches, which tells a much richer "here's
  what's actually different about these techniques" story than three tree/ensemble variants would
  have.

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

**Point count is no longer a visitor-facing control.** A "Points" slider (30-300) used to sit
right above the "New random seed" button — direct feedback was that this isn't a decision worth
asking a first-time visitor to make ("the number of points isn't something that we need to expose
to the user"). `state.pointCount` now just defaults to a fixed `200` and there's no UI to change
it for a generated dataset; "if they want more, they can switch to custom" — Custom points (draw
or CSV-upload) was already completely unbounded by this slider, so that escape hatch already
existed and needed no changes. The share payload still carries a 2-byte point count (unused by any
current UI, but harmless to keep — a previously-shared link that happened to encode some other
count still decodes and regenerates correctly, `decoded.pointCount || 200` covers the case where
an old/malformed payload has none).

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
  subgradient descent on hinge loss with L2 regularization (`lambda`, fixed at `0.05`, no longer a
  slider — see "Hyperparameters removed from the UI" below), learning rate `1/(λt)`. Deliberately
  labeled "linear classifier, hinge loss," never "SVM" — a true SVM is defined by its dual
  formulation and support vectors, and this is a primal, linearly-parameterized model trained to
  a similar-looking objective, not the same algorithm.
- **SVM (RBF kernel)** (`trainSVM`/`predictSVM`) is what actually earns the name the model above
  deliberately avoids — added on direct request ("add a svc/svm option"). Trains in the *dual*,
  via kernelized Pegasos (the same paper as the linear one, generalized to kernels the way the
  paper itself describes) — one running coefficient (`alpha[i]`) per training point rather than
  one weight per axis, updated by checking whether that point currently violates the margin
  (`y_i · f(x_i) < 1`) using an RBF kernel (`exp(-γ‖a-b‖²)`) instead of a plain dot product for
  every similarity comparison. A point's coefficient stays at exactly zero unless it ever
  triggers an update — only the points that do are "support vectors" in the literal sense, and
  only they ever influence a prediction. Two hyperparameters, `gamma` (kernel width) and `lambda`
  (regularization) — both fixed at `10` and `0.001` respectively, no longer sliders (see
  "Hyperparameters removed from the UI" below). Verified directly: on XOR, concentric rings, and
  moons — none linearly separable — this model reached 95-100% accuracy where the linear/hinge-loss
  models and Gaussian Naive Bayes all stayed well below 100%, confirming the kernel is doing real,
  non-linear work and not just quietly falling back to a linear fit. Grid/point prediction is
  O(grid × training points) per render (every cell sums a kernel evaluation against every nonzero-
  alpha training point) — the same complexity class k-NN's own grid prediction already has, and
  measured fast enough at this tool's point-count ceiling (300) not to need any special handling.
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
  dense layers, 2-64 neurons each, Adam optimizer, binary cross-entropy loss). The activation
  function was originally a third slider (ReLU/sigmoid/tanh) — removed on direct feedback ("even
  the activation function doesn't show us much. Just use ReLU and call it a day"): unlike layer
  count and width, which visibly change the boundary's shape and complexity, swapping activation
  functions on a network this shallow didn't produce a difference worth a whole control for it.
  `params.activation` still exists internally (permanently `'relu'`) rather than being ripped out
  of `trainMLP`/`renderMLPArch`'s signatures — the two functions that use it don't need to know it
  can no longer vary, and hardcoding the value at the one place it's set is simpler than threading
  a literal through every call site. Trains to real convergence rather than a fixed, user-chosen
  epoch count (see "The actual fix"
  section below for why, and for the real bugs that section's whole design went through), and gets
  its own live architecture diagram right under its sliders (see "Architecture diagram" below).
  **Known simplification, not yet hit as a real bug**: unlike Afterimage's decoder, this doesn't
  check for or fall back from low-precision WebGL (`floatPrecision()` returning 16 instead of 32 —
  see Afterimage's CLAUDE.md for why that's a real GPU/driver ceiling, not a config flag). At this
  network's tiny size the precision loss that broke Afterimage's much deeper gradient search
  hasn't shown up in testing; if an MLP boundary ever looks visibly wrong for no other reason,
  that's the first thing worth checking.

## Hyperparameters removed from the UI: fixed constants instead

Direct feedback, delivered as a batch of mobile-testing notes: "remove the linear regression
hyperparameters [meaning the linear hinge-loss classifier's regularization slider — Linear
Regression itself, the closed-form model, never had any]. remove the SVM hyperparameters. Just go
with a kernel width of 10 and margin strength a 0.001." Both models' `paramRow()` branches were
deleted from `syncModelUI()` outright — they now fall through to the same generic "No
hyperparameters for this one." message every hyperparameter-free model (Linear Regression,
Logistic Regression, Gaussian Naive Bayes) already shows, rather than being special-cased with an
empty box of their own.

The values themselves live as top-level constants next to the models they configure —
`LINSVM_LAMBDA = 0.05` (unchanged from the old slider's default; no new value was requested for
it) and `SVM_GAMMA = 10, SVM_LAMBDA = 0.001` (the exact values requested) — rather than folded back
into `state.params.lambda`/`state.params.gamma` the way the sliders used to write them. That
distinction matters: the linear hinge-loss model and the SVM used to *share* `params.lambda` on
purpose (see "SVM (RBF kernel)" above, before this change — they were "the same kind of quantity
for two different algorithms"), which was fine when a user could only ever be adjusting one of the
two sliders at a time. Now that both are fixed, they need to be fixed at *different* numbers
(`0.05` vs. `0.001`), so they can no longer share one field — `trainLinSVM` still reads
`params.lambda`, while the SVM's own two values live in `state.params.svmGamma`/`svmLambda`
instead and get remapped into a plain `{ gamma, lambda }` object at each of the two call sites
(`recompute()`'s dispatch, and `trainForExport()`) that actually invoke `trainSVM` — `trainSVM`
and `predictSVM` themselves are untouched, since they only ever cared about a `params.gamma`/
`params.lambda` shape, not where those numbers originally came from.

**The share-link payload got simpler, not more complicated, from this.** The 12-byte header used
to spend bytes 8-9 encoding whichever of `lambda`/`gamma` the current model's now-removed sliders
had been set to (`lambdaToByte`/`gammaToByte`, log-scale quantization to fit a hyperparameter
spanning orders of magnitude into one byte). Since neither model has anything left to encode,
those branches — and the four now-fully-unused `lambdaToByte`/`byteToLambda`/`gammaToByte`/
`byteToGamma` helper functions — were deleted rather than left dead. A link generated with the
linear hinge-loss classifier or the SVM selected now just leaves header bytes 8-9 at their default
0 (identical to what Linear Regression, Logistic Regression, and Naive Bayes already did, since
none of them ever had per-model bytes to write either) — decoding always resolves those two
models' hyperparameters to the fixed constants above regardless of what (if anything) is sitting
in those unused bytes, so an old link generated before this change still opens correctly, just
with the new fixed values instead of whatever custom gamma/lambda it used to carry.

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

**Two states, not one.** Moving a layers/width slider calls `renderMLPArchIfCurrent()`
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

## Throbbers (`.spinner-inline`, `.spinner-big`, `#computeSpinner`, `#plotOverlay`, `#exportSpinner`)

Direct request: "we need some sort of throbber while it's thinking for pretty much all the
steps." First version: a single `#computeSpinner` (the rotating-ring `.spinner-inline` pattern
already established in Inkling) next to the accuracy readout, shown around every path through
`recompute()`, with a `setTimeout(0)` yield before any computation so the spinner had at least one
chance to paint before a fast classical model finished.

**That version shipped, and the very next report was "I'm not getting any spinners at all."**
Verified directly, not dismissed: `spinner.hidden` genuinely did flip to `false` immediately after
a model/dataset click and the element's computed `display` genuinely became `inline-block` at that
instant — the show/hide *logic* was correct. The actual problem is a real, well-documented browser
behavior: a DOM mutation that gets reverted before the browser's next scheduled paint may never be
painted at all — there is no guarantee of "one paint per macrotask," only "paint when there's
something new to show and a frame is due." Most classical models here finish in low single-digit
milliseconds even after the yield, comfortably within one 16ms frame budget, so the spinner's
entire visible window could complete without ever actually reaching the screen. A 13px ring
sitting next to small status text was also just easy to miss even on the rare frame where it did
render — not the most likely place a user is looking while watching the boundary redraw.

**Two changes fixed this, not one.** `MIN_SPINNER_MS = 300` enforces a real minimum visible
duration — `recompute()`'s `hideSpinner()` checks how long the spinner has actually been shown and
waits out the remainder before hiding it, so even an instant computation now holds the spinner
on-screen for a genuinely perceivable stretch. And a second, much larger spinner (`.spinner-big`,
44px) now lives in `#plotOverlay`, a semi-transparent dark layer positioned directly over
`#plotCanvas` itself (`.canvas-wrap` + `position:absolute; inset:0`) — shown and hidden in lockstep
with `#computeSpinner` from the same `showSpinner()`/`hideSpinner()` helpers, so there's no way for
one to show without the other. This puts unmissable, centered feedback exactly where a user's eyes
already are (watching the boundary), rather than relying on a small icon in a status line they may
not be looking at. The export button keeps its own separate `#exportSpinner`, shown for the entire
multi-model export.

**A third overlay, `#mlpArchOverlay`, was added later directly over the Neural Network's own
architecture diagram** — direct feedback ("we need to show a spinner on the neural network as well
when it's training") after mobile testing. The plot's own overlay already covered every model,
MLP included, from a strict "does `recompute()` show and hide it" standpoint — the actual gap was
distance, not logic: while dragging a layers/width slider, the thing a visitor is looking at is the
architecture diagram right under that slider, not the plot canvas down in step 3, which on a phone
is very likely scrolled well out of view. `syncModelUI()`'s `mlp` branch now wraps `mlpArchCanvas`
in its own `.canvas-wrap`/`.canvas-overlay`/`.spinner-big` (the identical trio `#plotOverlay`
already uses, just a second instance), and `recompute()`'s `showSpinner()`/`hideSpinner()` toggle
it too, guarded by `if (archOverlay)` rather than a `state.model === 'mlp'` check — the element
only exists in the DOM at all while the Neural Network's params are showing (`syncModelUI()` tears
down and rebuilds `#modelParams` on every model switch), so its mere presence already implies the
right model is selected. The zero-points early-return branch (see "A real bug" below) also needed
an explicit `archOverlay.hidden = true` of its own, since that branch returns before ever calling
`showSpinner()`/`hideSpinner()` in the same pass — without it, an MLP training run that gets
superseded by someone clearing all custom points mid-training could leave the architecture
spinner stuck spinning forever, a real latent bug caught while wiring this up rather than one
that had already been reported.

## Progress bars for Neural Network training (`.progress-track`, `.progress-fill`)

Direct request: "when the neural network is training can you show a progress bar." A spinner (the
`#plotOverlay`/`#mlpArchOverlay` pair above) only ever says "something is happening" — the Neural
Network is the one model here where real, known progress actually exists (`epoch`/`MAX_EPOCHS`,
already flowing through `trainMLP()`'s `onProgress` callback for the "training… epoch N/250" text),
so it's the one case that earns an actual fill bar instead of just motion. `.progress-track` is a
plain hand-rolled div pair (an 8px rounded track plus an inner `.progress-fill` whose `width` is
set directly in JS as a `%` string) rather than a native `<progress>` element — consistent with
this site's general preference for CSS it fully controls over relying on a browser's own default
widget styling, and it reuses `var(--accent)` the same way every other Tonus indicator does.

**Three separate bars, not one, because there are three separate places training is visible**,
each already established by an earlier fix in this same file:
- `#mlpProgressTrack`, under the main accuracy/status row in step 3 — the primary, always-visible
  location.
- `#mlpArchProgressTrack`, under the architecture diagram in step 2 — the same "this is what a
  mobile visitor is actually looking at while dragging a layers/width slider" reasoning that
  justified `#mlpArchOverlay` above applies equally here; a lone bar down in step 3 would be just
  as easy to miss as the original single spinner was.
- `#exportProgressTrack`, under the "Export comparison grid" button — the export loop trains a
  fresh Neural Network from scratch for its own panel (see "Export: the comparison grid" below),
  and that one retrain is by far the slowest part of an otherwise-fast multi-model export.

All three read from the same underlying numbers, via a single `setMLPProgress(pct)` helper
(defined once, inside `recompute()`, closed over that call's own DOM lookups) that updates the two
main-page bars together — the export bar is updated separately, inline in the export loop, since
that code path doesn't go through `recompute()` at all and trains its own standalone `trainMLP()`
call. **Percent is computed as `epoch / MAX_EPOCHS`, not epoch count against however many epochs
this particular run happens to take** — since training can (and often does) stop early once loss
converges, a bar normalized against "epochs actually run" would jump to 100% at an unpredictable,
inconsistent point every time; normalizing against the fixed 250-epoch ceiling means the bar's
speed is at least consistent run to run, even though it will frequently stop short of full on a
run that converges early. That's an accepted, expected trade — a progress bar for a process with
no fixed known length is inherently approximate, and "consistent but sometimes incomplete" reads
better than "reaches 100% every time by a different, arbitrary definition of 100%."

**Visibility is wired into the exact same `showSpinner()`/`hideSpinner()` bracket the spinners
already use**, guarded by `state.model === 'mlp'` at show time (unlike the spinner overlays, which
guard by DOM presence — the progress tracks are static HTML, always in the document, so presence
alone can't distinguish "Neural Network selected" the way it does for the arch overlay) — a
classical model's near-instant `recompute()` never shows these bars at all, matching the "spinner
without meaningful progress data" reasoning that keeps them Neural-Network-specific. The zero-
points early-return branch (see "A real bug" below) hides both main-page tracks explicitly for the
same reason it already explicitly hides `#mlpArchOverlay` — that branch returns before the shared
`hideSpinner()` ever runs. The export bar is shown/hidden locally within its own `if (mkey ===
'mlp')` block in the export loop, immediately before that panel's `trainMLP()` call and immediately
after it resolves — it never stays visible while the other three (fast, synchronous) panels
render, since there's nothing left for it to represent once the Neural Network panel is done.

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
strength for the hinge-loss classifier on a log scale, regularization *and* kernel width for the
RBF SVM (both log-scale, via the same `lambdaToByte`/`gammaToByte` pair), `k` for k-NN, tree count
and max depth for Random Forest, or layer count/width for the neural network — training
itself always runs to real convergence rather than a chosen epoch count, so there's nothing to
encode for it) — covers every
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

Same situation every other tool's first ship starts from: the mug-template link reuses the same
generic "upload your own image" mug template ID and the site's standard `?rf=238054754631086278`
ambassador param, since no Tonus-specific curated product exists yet. Replace with a real,
verified Tonus-specific product link once one exists — don't fabricate one in the meantime.

**The user-facing copy doesn't say any of that, on purpose.** An earlier version's visible text
read "...no verified Tonus-specific design exists yet, so this is the same generic upload-your-
own-image mug template used as a placeholder elsewhere on this site" — accurate, and exactly the
kind of thing that talks a visitor out of clicking before they've even seen what they'd be
buying. UX review caught this directly: that sentence belongs in this file, for whoever picks up
this code next, not in front of someone deciding whether to make a purchase. The current copy
("Whatever's on the canvas above is worth putting on something...") says only what's true and
relevant to the visitor — that the link goes to a real, working mug template — without either
lying about a curated product existing or talking the CTA down before it gets a chance.

## The mug CTA had no way to see the product before clicking

Direct follow-up: exporting, downloading a PNG, and then manually uploading it to a blank Zazzle
template with zero preview of the result along the way was too many disconnected steps before a
visitor had any idea what they'd end up with. `.mug-mockup` is a small live preview — a CSS-drawn
mug body and handle, no image assets — whose body's `background-image` is set directly from
`plotCanvas.toDataURL()` every time `renderPlot()` runs, so it always shows *this exact* boundary,
live, with no separate export step required to see it. This is deliberately a lightweight
illustration (a rounded rect with an inset shadow suggesting a ceramic curve, not a photorealistic
product render) — the goal was "you can see roughly what this would look like" before deciding to
export and upload, not a pixel-perfect mockup. `a.primary` is a new page-local rule (theme.css
scopes `.primary`/`.secondary` to `<button>` only) since the actual "Make it a mug" CTA has to be
a real outbound `<a>`, not a button faking a link.

## UX pass: less math on the page, more math in the implementation

A UX review plus direct follow-up ("too mathy in many regards") both landed on the same finding:
"How this actually works" opened by default and led with four literal formula blocks (`(XᵀX) w =
Xᵀy`, `K(a,b) = exp(−γ‖a−b‖²)`, etc.) — accurate, but exactly the kind of thing that makes a
casual visitor bounce before ever touching the actual tool. Two changes, not one: the disclosure
now starts **collapsed** (matching every other tool on this site whose deep-dive isn't itself the
main draw), and every `.math-block` formula was removed in favor of plain-language explanation —
"fits the single straight line that best fits the points, solved directly in one step" instead of
the normal-equations formula it's describing. This is a **content** decision, not a rigor one:
the actual implementation didn't change at all, and the real math is still exactly as correct and
inspectable in the source as it ever was (see the model-by-model breakdown earlier in this file)
— this section just stopped trying to also be that file's replacement for a general audience.
`Multi-Layer Perceptron` was also renamed to `Neural Network` in this pass, matching
`MODEL_LABELS.mlp` — the deep-dive heading and the pill label had quietly drifted to two different
names for the same model.

**Same UX review, a smaller companion fix**: every hyperparameter slider (`paramRow`'s new
optional `hint` argument) now shows one short, italic, plain-language sentence underneath it —
"lower = hugs the training points tightly; higher = a simpler, more cautious line," not another
formula. Direct feedback was that a slider's technical name and current number don't say anything
about *why* you'd move it, leaving a first-time visitor to guess-and-check with no intuition to
build on. Deliberately the opposite direction from a "how it works" deep-dive: one sentence, right
where the decision is actually being made, not a paragraph to go read elsewhere.

**Reversed later, on direct instruction: "How it works is supposed to be expanded by default."**
`<details class="disclosure advanced">` now carries `open` again. The reasoning above for
collapsing it in the first place hasn't been invalidated — the content still opened with formula
blocks back then, which really was the problem — but by the time this reversal landed, the math
had already been stripped out in favor of plain-language explanation (that part of this section
still stands), so "collapsed to avoid scaring off a casual visitor with formulas" was solving a
problem that no longer existed in the same form. Don't collapse this again without checking
whether the actual objection (dense math, not the disclosure's open/closed state) has resurfaced.

## The drawing instructions were disconnected from the drawing surface

UX review finding: "Custom points" mode's instructions and class toggle live in step 1 (where
every other dataset choice lives), but the canvas you actually draw on is down in step 3 — a
first-time visitor following "draw on the plot below" literally would click empty space right
where they're standing, a scroll away from where anything happens. Rather than duplicating the
drawing surface into two places (step 1 would need its own canvas, kept in sync with step 3's —
a real source of drift risk for no good reason), a `#jumpToDrawBtn` ("↓ Start drawing") smooth-
scrolls to `.canvas-wrap` and adds a brief accent-colored pulse (`jump-highlight`, 1.5s) so the
canvas is unmistakable the moment it comes into view. Deliberately a manual button, not an
automatic scroll-on-select — jumping the page out from under someone the instant they click a
pill would be a worse surprise than leaving them to click one more button.

## "Clear points" needed a confirm — drawing is a slower, more invested action

UX review finding: unlike picking a different preset dataset (an instant, freely-reversible
choice), freehand drawing can represent real invested time, and `#clearDrawBtn` erased it in one
click with no way back. Fixed with the exact same arm-then-confirm pattern Inkling's own "Reset
network" button already uses — `#clearDrawToast` ("click again to confirm") fades in on the first
click via `.toast`/`.toast.show`'s opacity transition, and only a second click within 2.6 seconds
actually clears `state.points`. Also guarded against clearing an already-empty pad (`if
(!state.points.length) return;`) — no reason to make someone confirm erasing nothing.

## Showing which points the model got wrong, not just the percentage

UX review finding: "Training accuracy: 87%" is honest, but abstract — it doesn't say *which*
points are still tripping the model up, and that's a much more concrete, inspectable thing to
look at than a number. `predictCurrentPoints()` was factored out of `computeAccuracy()` (it used
to compute this inline, once) specifically so `renderPlot()` could call the exact same function
for the exact same purpose — the accuracy percentage and the red rings marking wrong points are
now guaranteed to agree with each other, since they're reading the same predictions, not two
separately-computed ones that could silently drift apart. `renderPanel()`'s new `predictions`
parameter draws a red ring (`#e5484d`, distinct from both class colors) around any point whose
predicted class doesn't match its real label — verified directly: a genuinely 100%-accurate
render (k-NN on the default dataset) produces zero red pixels, and an 87%-accurate one produces a
real, non-zero ring count matching roughly the expected number of missed points. Deliberately
**not** carried into the export/comparison-grid panels — those are meant to be a clean image worth
printing, not a live debugging view, and a print covered in red error rings isn't what someone
wants on a mug.

## "Surprise me" — a first-time visitor's default landing state has no obvious payoff

UX review finding: a fresh page load is Moons + Logistic Regression, a real but unremarkable
~85% — nothing in that default state hints at why this tool is worth playing with. `#surpriseMeBtn`
(in the hero, right under the intro paragraph) jumps to one of three curated `SURPRISES` pairings
— a straight-line model against a dataset no straight line can separate (XOR/Linear Regression,
concentric rings/Logistic Regression, spirals/the linear hinge-loss model) — each a genuinely
dramatic, honest failure, not a staged one, verified directly (concentric rings + Logistic
Regression landed at 65% in testing, XOR-style failures land near 50%). The callout underneath the
plot names real alternatives that actually fix it for that specific pairing, and clears itself the
moment the visitor picks anything themselves (a normal dataset or model pill click) so it can never
sit there as stale advice for a boundary it no longer describes. Reuses the same scroll-and-
highlight mechanism `#jumpToDrawBtn` already established, rather than inventing a second one.

## The export CTA never fired at the moment it was actually earned

UX review finding: the mug/export call to action sat as a static line at the very bottom of the
page regardless of what the visitor had just accomplished — genuinely great moments (an SVM just
solving a spiral, k-NN nailing a hand-drawn shape) never got connected to "you should save this."
`#exportNudge` appears directly under the accuracy readout the moment `computeAccuracy()` returns
0.95 or higher, and disappears again the instant it doesn't (recomputed fresh every `recompute()`
call, not a one-time trigger that could go stale) — a harder dataset or a worse hyperparameter
correctly makes it vanish again. Its link smooth-scrolls to `#shareSection` using the same
`scrollIntoView` pattern `#jumpToDrawBtn` already established, rather than a plain `<a href="#...">`
anchor jump, which would have been an abrupt instant cut with no `scroll-behavior:smooth` set
anywhere on this page. Deliberately threshold-based on the number alone, not filtered by which
model earned it — k-NN hitting 100% by memorizing its own points is a less impressive "win" than
an SVM genuinely solving a hard boundary, but singling out specific models to exclude felt more
arbitrary than just trusting the same honest number the accuracy readout itself already shows.

## The export grid ignored whatever model you'd actually just gotten excited about

UX review finding: `EXPORT_MODELS` was a hardcoded `['logreg', 'rf', 'knn', 'mlp']` regardless of
what was selected — if a visitor had just picked the SVM to solve their own hand-drawn spiral, the
export retrained four completely different models from scratch and never showed the result they
were actually proud of. `computeExportModelList()` now always puts `state.model` in
the first slot, filling the rest from the same `DEFAULT_EXPORT_MODELS` list minus whichever entry
is now redundant with it — so the current model is always represented, the panel count stays at
a fixed 4, and if the current model already *is* one of the defaults (e.g. Random Forest), nothing
changes from before other than which slot it's ordered into. This required generalizing the
export loop's training dispatch (`trainForExport()`) from the three hardcoded models it used to
know about (`logreg`/`rf`/`knn`, plus a separate `mlp` special case) to the same full
trainer/predictor maps `recompute()` and `computeGridProbs()` already use — every non-MLP model in
this file's roster can now legitimately appear in an export panel, not just three of them.
Verified directly: exporting with SVM selected produced a real, non-erroring 4-panel render with
SVM included; exporting with Random Forest selected (already one of the defaults) still produced
exactly 4 distinct panels with no duplicate.

## The exported comparison grid had no way to say it was out of date

Direct feedback: "when we change the dataset or hyperparameters for any of the models we need to
at least indicate that comparison grid is stale." Once an export exists, nothing about the page
distinguished "this PNG reflects exactly what's on screen right now" from "this PNG was rendered
three datasets ago and someone's about to download it thinking it's current" — the export panel
just sits there, unchanged, until someone clicks the button again. `markExportStale()` is called
as the very first line of `recompute()` — the single choke point every dataset pick, model pick,
hyperparameter slider, draw stroke, and CSV upload already funnels through — so it fires on
anything that could actually change what an export would show, without needing to be wired into
each of those input handlers separately. It sets `state.exportStale = true` and reveals
`#exportStaleNote`, a `var(--danger)`-colored line directly under the export button; the export
button's own click handler clears it (`state.exportStale = false`) once a fresh render actually
completes. `hasExported` gates the whole thing on both sides — the note can't appear for a
first-time visitor who's never exported anything (there's nothing to be stale relative to), and
`markExportStale()` itself is a no-op until `hasExported` flips true, so idle page-load recomputes
before any export don't do pointless work setting a flag nothing is checking yet. Deliberately
"at least indicate," per the request's own wording — this doesn't auto-regenerate the export or
block the download button, it just tells you to look again before you trust it.

## Testing changes

No test suite — static page. Verify via a local static server (root-relative `/nav.js` and
`/theme.css` mean `file://` won't pick them up). Golden path: confirm the page opens with the
Moons dataset and Logistic Regression selected, no "Points" slider anywhere in step 1 (just "New
random seed"), a real heatmap and scattered points visible, and a training-accuracy readout well
under 100% (moons genuinely isn't linearly separable) → switch to XOR and confirm Logistic
Regression and the hinge-loss linear classifier both land near 50% accuracy while k-Nearest
Neighbors, Random Forest, the SVM (RBF kernel), and the Neural Network all reach 90%+ — this is the
core pedagogical claim of the whole tool, so it's the single most important thing to re-verify if
any model's math is ever touched → confirm Gaussian Naive Bayes lands well below 100% on XOR too
(its independence assumption is a textbook failure case for exactly this dataset) → select the
linear hinge-loss classifier and the SVM in turn and confirm each shows "No hyperparameters for
this one." rather than any slider — this is the one thing to re-check most carefully if either
`LINSVM_LAMBDA`/`SVM_GAMMA`/`SVM_LAMBDA` is ever touched, since these are the two models where a
regression would be silent (no error, just a quietly wrong fixed value) → drag every hyperparameter
slider for k-NN/Random Forest/the Neural Network and confirm the boundary and accuracy actually
change, not just the displayed number → for the Neural Network specifically, confirm there's no
epoch-count slider at all, confirm the architecture diagram redraws its *shape* the instant a
layers/width slider moves (before the retrain finishes — check the "structure only" caption) and
switches to real-weight coloring with the "colored by this network's real trained weights" caption
once a training run completes, and confirm the diagram's own overlay spinner (`#mlpArchOverlay`)
is visible while that training run is in progress and gone once it completes — separately from,
not instead of, the plot's own overlay — since this is the one model whose relevant spinner sits
somewhere other than the plot canvas → for that same run, confirm both `#mlpProgressTrack` (under
the accuracy readout) and `#mlpArchProgressTrack` (under the architecture diagram) become visible,
their fill widths climb together in lockstep with the "epoch N/250" text (not just once at the
end), and both disappear the moment training finishes — pick a network large enough (4 layers, 64
neurons) to actually stay mid-training long enough to sample more than one frame of progress, since
a small/fast network can finish before a poll even catches it once → confirm neither progress bar
ever appears for a classical model (there's no meaningful "progress" for a synchronous compute) →
switch to "Custom points"
with **zero points already present** while the Neural Network is selected and confirm no console
errors and a clean "No points yet" message (the exact regression documented above) → draw a small
shape, confirm points are added continuously while dragging and the boundary retrains after
release → upload a small CSV with `x1,x2,label` columns at an arbitrary scale (not already in
`[-1,1]`) and confirm the points land inside the plot, correctly rescaled → copy the share link,
open it in a fresh tab/session, and confirm the dataset, model, every hyperparameter, and (for a
custom dataset) every point round-trip correctly → with the Neural Network selected (so it claims
the export's first panel — see "The export grid ignored whatever model" below), click "Export
comparison grid" and confirm `#exportProgressTrack` appears and its fill climbs while that one
panel trains, then disappears again before the remaining three (fast, synchronous) panels
render — it shouldn't still be sitting there through the rest of the export → confirm all
four panels render with visibly different boundaries for the same underlying points, every panel
label and the title are cleanly legible against the plain dark background (not overlapping any
panel's own heatmap colors — the exact bug documented above), a QR code appears in the bottom-right
corner, "Download comparison as PNG" saves a real, non-empty file, and `#exportStaleNote` is
hidden right after that export finishes → change the dataset (or the model, or any hyperparameter)
and confirm `#exportStaleNote` appears without needing to click anything else, then click "Export
comparison grid" again and confirm it disappears and the newly-rendered panels actually reflect
the changed state — not the previous export's → confirm **both** the
overlay spinner centered on the plot canvas (`#plotOverlay`/`.spinner-big`) and the small inline
one next to the accuracy readout (`#computeSpinner`) are actually visible — not just toggled in
the DOM — for *every* kind of change (dataset, model, and hyperparameter, not just Neural Network
ones), and that each stays up for a real, perceivable moment even for an instant classical-model
change (the exact "I'm not getting any spinners at all" bug documented above — verifying
`hidden` flips correctly in script is not sufficient, since that was already true when the bug was
real; look at the page, not just the attribute) → confirm a separate spinner covers the whole
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
