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
(amber, teal, violet, rust, cyan, rose). The two *data class* colors — orange-red for class 0,
cyan-blue for class 1 (`--c0`/`--c1`) — are deliberately separate constants from the page accent,
following the original spec's own "Orange/Red vs Blue/Cyan" convention; they encode which class a
point belongs to, not which tool this is, the same separation Inkling keeps between its rose
"accent" and its blue "cool" excite/inhibit color.

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
  **Known simplification, not yet hit as a real bug**: unlike Afterimage's decoder, this doesn't
  check for or fall back from low-precision WebGL (`floatPrecision()` returning 16 instead of 32 —
  see Afterimage's CLAUDE.md for why that's a real GPU/driver ceiling, not a config flag). At this
  network's tiny size the precision loss that broke Afterimage's much deeper gradient search
  hasn't shown up in testing; if an MLP boundary ever looks visibly wrong for no other reason,
  that's the first thing worth checking.

## A real bug: `model.fit()` silently hangs forever without `yieldEvery: 'never'`

**Found directly during testing, not theorized.** The very first working version of `trainMLP()`
called `model.fit(...)` with tf.js's defaults, and every single training run simply hung — the
returned promise never resolved, `trainStatus` stayed on "training…" forever, no error, no
timeout. Confirmed the actual mechanism before touching anything: `tf.nextFrame()` — which
`model.fit()` calls internally between epochs to yield control back to the browser, so a long
training run doesn't freeze the page — is backed by `requestAnimationFrame`, and
`requestAnimationFrame` **never fires while `document.visibilityState !== 'visible'`** (confirmed
directly: `tf.nextFrame().then(...)` left unresolved for 2+ seconds while the tab reported
`hidden`). This is the exact same root cause several other tools in this repo have already hit for
their own canvas `requestAnimationFrame` loops (see Inkling's and Vectis's CLAUDE.md) — the first
time it's turned out to affect a *library's* internal implementation rather than this codebase's
own animation code.

Unlike those other cases, this one has a real fix rather than just a testing workaround: tf.js's
`fit()` accepts a `yieldEvery` option (`'auto' | 'batch' | 'epoch' | 'never' | <ms>`), and passing
`yieldEvery: 'never'` skips the `tf.nextFrame()` yield entirely. Verified directly, isolated from
the rest of the app: the identical `fit()` call with `yieldEvery:'never'` completed in ~1.4
seconds and produced a correctly-trained tiny XOR network, in the same hidden-tab environment
where the default hung indefinitely. This isn't just a workaround for a broken test harness,
either — every model this tool trains is small enough (2D input, at most 4×64 dense layers, at
most 300 epochs, at most a few hundred points) to finish in well under a couple of seconds
regardless, so there's no real responsiveness reason to yield mid-training in the first place, and
skipping the yield makes the tool robust against a real user backgrounding the tab mid-train too
— not just an automation quirk. If a future model added here is large enough that yielding
actually matters for UI responsiveness, revisit this specific option rather than assuming it's
always safe to skip.

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
Random Forest, or layer count/width/activation/epochs for the neural network) — covers every
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
boundary and accuracy actually change, not just the displayed number → switch to "Custom points"
with **zero points already present** while the Neural Network is selected and confirm no console
errors and a clean "No points yet" message (the exact regression documented above) → draw a small
shape, confirm points are added continuously while dragging and the boundary retrains after
release → upload a small CSV with `x1,x2,label` columns at an arbitrary scale (not already in
`[-1,1]`) and confirm the points land inside the plot, correctly rescaled → copy the share link,
open it in a fresh tab/session, and confirm the dataset, model, every hyperparameter, and (for a
custom dataset) every point round-trip correctly → click "Export comparison grid," confirm all
four panels render with visibly different boundaries for the same underlying points, a QR code
appears in the bottom-right corner, and "Download comparison as PNG" saves a real, non-empty file.

One environment-specific trap worth knowing before touching `trainMLP()` again: this automation
environment's `document.visibilityState` stays `'hidden'` throughout (a documented limitation
elsewhere in this repo, e.g. Vectis's and Inkling's CLAUDE.md), which is exactly the condition
that exposed the `yieldEvery` bug above. If a training-related change ever seems to hang during
testing here, check `document.visibilityState` and `tf.nextFrame()`'s resolution directly before
assuming the code itself is broken — but don't assume every hang here is *just* the environment,
either, the way the empty-dataset bug above genuinely wasn't.
