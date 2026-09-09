# Neural Viaduct

Runs two real, off-the-shelf, ImageNet-pretrained networks — MobileNetV1 (plain) and
MobileNetV2 (residual) — on the visitor's own photo, entirely client-side, and visualizes each
network's actual block-by-block intermediate activations side by side, plus a live-computed
"how much of the input survives" stat, each network's real top-5 prediction, and a real
params/accuracy comparison. The pitch: skip connections are most of the reason vision networks
can go deep without falling apart, and this page shows the shortcut literally happening — and
measures its effect — rather than just describing it.

## Architecture

`index.html` only, linking `/nav.js` and `/theme.css` like every other lab tool (see root
`CLAUDE.md`'s "Site-wide shared files"). Own `<style>` block only holds the block-diagram canvas,
prediction bars, and stat table — everything else comes from `theme.css`'s defaults. Accent color
is a rust/terracotta (`#c9553d`) — a deliberate nod to International Orange, the actual paint color
of real viaducts and bridges, distinct from Ridgeline's amber, Afterimage's violet, and Vectis's
cyan.

No build step, no bundler. External JS: `@tensorflow/tfjs`, `@tensorflow/tfjs-backend-wasm`, and
`@tensorflow-models/mobilenet` — all from jsDelivr, same CDN-script pattern as every other
external dependency on this site (Afterimage's tfjs, Vectis's transformers.js).

## Why MobileNetV1 vs. MobileNetV2, not VGG19 vs. ResNet50

The original spec (`pons-spec.md`, the tool's working-name predecessor) called for VGG19 vs.
ResNet50 — real, famous, and dramatic (VGG19: 143.7M params / 72.4% top-1; ResNet50: 25.6M params
/ 80.9% top-1), but VGG19 has **no published quantized weights anywhere** — only fp32, ~548MB.
Even after ruling out VGG16-int8 (132MB, but a different named architecture than the spec called
for) and self-quantizing VGG19 ourselves (~135MB, but requiring a brand-new dedicated GitHub repo
just to host the binary, plus a Python conversion pipeline), the total footprint and hosting
mechanics kept growing more elaborate than a personal lab-tool site should need. Direct user
feedback mid-build: *"If it's too big for github it's too big to run in the browser."* — both the
byte count and the idea of a repo dedicated to hosting weights were the objection, not just one or
the other.

**MobileNetV1 (2017, plain) vs. MobileNetV2 (2018, residual)** solves both at once:

- Already hosted by TensorFlow.js's own official `@tensorflow-models/mobilenet` npm package,
  served from jsDelivr — **zero new repo, zero self-hosting, zero quantization pipeline.**
- Combined download ≈ 30MB (V1: 4.2M params, 70.6% top-1; V2: 3.4M params, 72.0% top-1) — smaller
  than a single Vectis CLIP tower (Xenova's int8 CLIP text or vision tower alone is ~65-90MB).
- Same authors' lineage, same design philosophy, differing specifically in the presence of the
  skip connection — MobileNetV2's own paper explicitly credits the improvement over V1 to adding
  inverted-residual connections. This isolates the one variable (skip vs. no skip) more honestly
  than VGG-vs-ResNet50 ever did, which also differs in filter sizes, FC-head design, and overall
  philosophy — several confounding variables at once, not one.

**The honest tradeoff, stated directly in the deep-dive section:** the published payoff numbers
are far subtler than VGG/ResNet50's "twice as deep, a sixth of the params, way more accurate"
story — V2 is only modestly smaller and modestly more accurate than V1. Don't paper over this;
the page says so explicitly rather than implying a bigger effect than what's real. (The live
retention stat, added later — see below — turned out to carry a more dramatic real difference
than the published params/accuracy numbers do.)

## How intermediate activations are extracted (no graph surgery needed)

The naive assumption going in was that pulling a real intermediate feature map out of an
already-exported inference graph would need offline graph surgery (adding named outputs to the
ONNX/TFJS graph before shipping it) or switching to a Keras `LayersModel` format specifically
for its named-layer access. **Neither is true.** `@tensorflow-models/mobilenet`'s `.load()`
returns a wrapper whose `.model` property is a plain `tf.GraphModel`, and `GraphModel.execute()`
accepts an array of **internal node names**, not just the graph's declared outputs — confirmed
directly against the real loaded model before writing any page code:

```js
const net = await mobilenet.load({version:1, alpha:1.0});
const nodeNames = Object.keys(net.model.executor.graph.nodes); // every op in the graph, by name
const out = net.model.execute({images: inputTensor}, ['module_apply_default/.../Relu6']);
```

This works for any named tensor in the graph, including ones nobody declared as a model output —
which is also how the 10 real residual-add nodes in MobileNetV2 were found (`nodeNames.filter(n
=> /add/i.test(n))` against the *actual loaded graph*, not inferred from reading about the
architecture): `expanded_conv_{2,4,5,7,8,9,11,12,14,15}/add`, exactly 10 of MobileNetV2's 17
bottleneck blocks — matching the known architecture (blocks that change stride or channel count
can't have an identity shortcut) but verified against the real artifact, not assumed. This is the
same "verify against the real model, not against what you expect it to do" discipline as
[Vectis's CLIP anisotropy fix](../vectis/CLAUDE.md) — a technique worth remembering for any future
tool that needs to look inside an off-the-shelf `GraphModel` without re-exporting it.

`index.html` generates the exact node-name lists at runtime (`V1_NODES`, `V2_PROJECT_NODES`,
`V2_ADD_NODES` in the main script) rather than hardcoding 32+ verbose strings by hand, since the
naming pattern is regular and generation is far less error-prone than transcription.
`V2_RESIDUAL_BLOCKS` is the hardcoded `Set([2,4,5,7,8,9,11,12,14,15])` found via the probe above —
if a future TFJS/mobilenet version changes this internal graph, re-run the same
`Object.keys(...).filter(/add/i)` probe against the new model before trusting this set again.

## `project/BatchNorm` vs. `add`: a real bug, not just a naming choice

The first shipped version of this page fetched only each block's `project/BatchNorm` node as "the
block's output," for every block, residual or not. **That's wrong for every residual block.**
`project/BatchNorm` is `F(x)` — the block's own computed transform, sitting *before* the shortcut
adds the input back in. The actual output of a residual block, the thing the shortcut is
responsible for, is the separate `add` node: `F(x) + x`. Using `project` alone meant the diagram
and every downstream computation had never once reflected the shortcut's effect, on any block, the
entire time — direct user feedback caught this at the "what story is it telling? the skips aren't
even that obvious" stage, and it turned out the reason was structural, not just a styling problem.

Confirmed how much this matters before fixing anything: for one real block, the mean absolute
difference between `project` and `add` was *larger* than `project`'s own mean absolute value — the
identity path can contribute more signal than the block's learned transform does. `index.html` now
fetches both node sets in one pass and `buildTrueOutputs()` picks the right one per stage: `add`
wherever `V2_RESIDUAL_BLOCKS` says a shortcut exists, `project` everywhere else (there's nothing
else there — no shortcut, no add node). Every thumbnail and every retention-stat calculation reads
from this corrected array, never from raw `project` values directly.

## A real WebGL bug: reading back an `add` node hangs forever

While fixing the above, reading back the `add` node's tensor via `.data()` **hung indefinitely**
under the WebGL backend — confirmed directly, isolated down to a single node, single tensor,
single `.data()` call: `execute()` itself returns instantly and reports the correct shape, but the
returned tensor's `.data()` promise never resolves, no error, no timeout, nothing. The same exact
node reads back correctly in well under a second on the WASM backend. Root cause almost certainly
WebGL's operation-fusion optimizer: `add` is normally immediately consumed by whatever the graph
does next, and asking for it directly, as a standalone output nobody declared, seems to leave no
real texture behind to read from. This is a strictly worse failure mode than a wrong answer — a
wrong-but-fast result is at least visible and debuggable; a promise that never resolves just looks
like a frozen page. **This page now forces WASM and does not attempt WebGL at all** — see
`chooseBackend()`, which no longer tries WebGL first the way Afterimage's does. If a future change
here ever needs GPU speed back, re-verify this exact repro (fetch one `add` node alone, time out
the `.data()` call) before trusting WebGL with any non-declared internal node again; it may be
fine for ordinary declared outputs; it was not fine for this one.

## The preprocessing pitfall (found by testing, not assumed)

`net.classify(imgEl, 5)` handles its own preprocessing internally and is trustworthy as-is. But
feeding a *raw* `tf.browser.fromPixels()` tensor (uint8, [0,255]) directly into
`model.execute({images: ...}, [...])` for the custom intermediate-activation path gives **wrong,
inconsistent-with-`classify()`** predictions — confirmed directly: a test image that `classify()`
correctly scored as "ping-pong ball" at 21.2% came back as a completely different top class from
the naive manual pipeline. The fix, found by inspecting the loaded model's own
`net.inputMin`/`inputMax`/`normalizationConstant` properties (`0`, `1`, `1/255`) and confirmed by
re-running the manual pipeline: divide by 255 to land in `[0,1]` **before** `expandDims` — the
graph's own `hub_input/Mul`/`hub_input/Sub` nodes handle any further internal normalization
themselves.

```js
function preprocess(imgEl){
  return tf.tidy(() => {
    let x = tf.browser.fromPixels(imgEl);
    x = tf.image.resizeBilinear(x, [224,224]);
    x = x.toFloat().div(255);          // <-- the step that's easy to skip and silently wrong
    return x.expandDims(0);
  });
}
```

After this fix, a manual `execute()` call with both the intermediate node and the model's own
logits node in one pass reproduced `classify()`'s probabilities almost exactly (softmax by hand,
matching to 5+ decimal places) — confirming the same preprocessed tensor is safe to reuse for
both the diagram's activations and (if ever needed) hand-rolled predictions. The page still calls
`net.classify()` separately for the predictions panel rather than hand-rolling softmax + a
1001-class label lookup — `classify()` is already correct and tested, and a second forward pass
on a network this small costs nothing worth avoiding the well-tested path for.

## The block-flow diagram (`renderArch`, `drawRow`, `renderThumbToCanvas`)

One `<canvas>`, two rows (`ROW1_Y`/`ROW2_Y`), each block rendered as a small square. Every
thumbnail is that block's real *corrected* output tensor (see `buildTrueOutputs()` above) reduced
to a single `[H,W]` map by averaging across channels (all values are post-ReLU6 or post-add of two
post-ReLU6-derived tensors — a plain mean is a reasonable single-number summary either way),
normalized to *its own* min/max (raw activation magnitude varies a lot block to block, so a shared
scale across the whole network would make early blocks look uniformly dim and late blocks
uniformly bright regardless of their actual internal contrast), then tinted from a dark base color
to the page's accent color — a single meaningful scalar-to-color mapping (intensity of
activation), consistent with the site's established "grayscale/single-hue unless a color means
something specific" convention (see Afterimage's CLAUDE.md, "Color: black-to-white only, except
where color is literal").

**Arc visibility went through a real revision, not just a tweak.** The first version drew arcs
*underneath* the thumbnails, thin (1.6px), grayish (`rgba(154,160,174,0.55)`), with a shallow
30px peak — direct feedback: "the skips aren't even that obvious," and looking at an actual
screenshot, they were nearly invisible against the dark background. The current version draws
arcs **on top of** the thumbnails (drawn after, in a second pass over the row), thick (3px), bright
(`#e9e7de`, the page's near-white ink color, not a muted gray), with a taller 40px peak and a
small filled dot marking exactly where the shortcut lands — plus the landing tile's own border
switches to the accent color so the connection between "this arc" and "this tile" is unambiguous.
Row layout uses named offsets *from the tile's own top edge* (`ARC_TOP_GAP`, `ARC_PEAK_HEIGHT`,
`LABEL_GAP_ABOVE_PEAK`) rather than tuning each row's label/arc position by hand against the
other — changing one no longer risks the arc peak colliding with the label text above it, which is
exactly the kind of thing that went unnoticed with the old hand-tuned single `LABEL_OFFSET`
constant. Arcs are drawn **only** for blocks where `V2_RESIDUAL[i]` is true — verified
pixel-by-pixel after building it, not just assumed correct from the code: sampling the canvas at a
residual block's arc position finds the arc's bright stroke color; sampling the equivalent
position over a non-residual gap finds only the background fill. If you change `THUMB`/`GAP`/
`START_X`/the offset constants, re-verify this the same way — a geometry mistake here would
silently draw arcs in the wrong place while still "looking like a diagram."

Both rows animate their reveal together over one shared `requestAnimationFrame` timeline
(`animateReveal`), each row computing its own `revealCount` as a fraction of its own block count
so a 14-stage and an 18-stage row still finish at the same moment despite the different lengths.
`window.__forceRenderArch(frac)` is a **testing-only hook** (same pattern as Vectis's
`window.__forceRender`, for the same reason: `requestAnimationFrame` pauses when
`document.visibilityState !== 'visible'`, which some automated browser tools report as "active"
even when it isn't) — don't remove it, but don't wire it into any real page control either.

## The retention stat (`computeRetention`, `renderRetention`) — the actual proof, as a number

Direct feedback on the first shipped version: "what story is it telling? ... the ending pictures
are basically the same. The predictions aren't much different either." All true, and not really
fixable by making the existing visuals prettier — thumbnails reduced to a single mean-brightness
map inherently wash out most of the structure that would make two blocks look meaningfully
different, and two moderate-depth, already-well-trained networks (neither one is deep enough to
hit the classic degradation problem — see the deep-dive section) simply don't produce a dramatic
visual difference in their raw forward-pass activations. That's not a bug to fix, it's just not
where this particular pair's real difference shows up.

The real difference shows up in **how much of a block's input survives into its output** — which
is exactly what a shortcut is *for*, and it's directly measurable from the same forward pass
already being computed, no extra model runs needed. For every MobileNetV2 block with a shortcut,
`computeRetention()` takes cosine similarity between that block's input (the previous stage's true
output) and its true output (the `add` result) — a residual block's output is *literally*
`input + F(x)`, so a high similarity to the input is the expected, measurable signature of the
shortcut doing its job. MobileNetV1 has no shortcuts to measure the same way, so it needs a fair
baseline: `V1_MATCHING_PAIRS` (`[3,5,7,8,9,10,11,13]`, verified against the real loaded model's
block shapes) are the specific consecutive V1 blocks whose input and output happen to share the
same shape anyway — the only pairs where the same cosine-similarity comparison is even
dimensionally meaningful, computed the identical way, network structure aside.

This was validated against multiple different synthetic test photos before being trusted, not
assumed to generalize from one lucky run: MobileNetV2's average was consistently around 55-77%,
MobileNetV1's consistently around 35-36%, across photos with entirely different colors and
shapes. The gap is real and directionally stable, not an artifact of one specific test image — but
the exact percentages **will** vary photo to photo (that's the point: it's computed live, not
canned), so don't hardcode an expected number anywhere or treat a specific run's value as a
regression baseline. `retainV1`/`retainV2` (module-scoped) are also read by the PNG export so the
downloaded image carries this photo's actual numbers, not a placeholder.

## Scope trims from the original spec

- **No curated fallback example photos** (spec §8 asked for a small set). Upload-only for this
  first version — a real, deliberate scope cut to keep the build shippable, not an oversight.
  Revisit if it turns out people bounce at the upload step.
- **No PCA-to-RGB feature-map projection** (spec §8's other listed candidate for the thumbnail
  reduction). Channel-averaging into a single accent-tinted heatmap was chosen instead — simpler,
  consistent with the site's existing color conventions, and applied identically to both networks,
  which is the actual fairness requirement the spec cared about, not the specific reduction
  method.
- **"Depth" isn't presented as a single number.** Unlike VGG/ResNet's clean "16 vs. 50 layers,"
  MobileNet's block structure doesn't map onto a simple depth count both networks share cleanly.
  The stat card instead reports what the tool actually demonstrates: parameters, top-1 accuracy,
  and blocks-with-a-skip-connection (0 of 13 vs. 10 of 17) — real, specific, and directly tied to
  what's on screen, rather than forcing an apples-to-oranges depth comparison.
- **Magnet only, no t-shirt**, per spec §7 — every output here is built from the visitor's own
  photo, so it gets the same personal/private-gift framing as Vectis's photo-mode outputs, not
  Ridgeline's general-purpose merch framing.

## Page flow: purpose → the lab → save & share → deep dive

Reworked once already, deliberately tighter than the first shipped version: direct feedback was
"It's... way too complicated. We need it to be much simpler. Purpose. Lab. Call to action
(zazzle). Then all the details in the learn more section." The first version had five separately
step-tagged sections (upload, watch it flow, predictions, payoff, save & share) — now it's three:
the hero + one short paragraph (purpose), a single `#labSection` holding upload, the diagram, the
retention stat, predictions, and the stat table all as sub-panels of one continuous experience
(not five numbered steps), `#shareSection` (the Zazzle CTA), and `#learnSection` (collapsed,
everything else: the plain-vs-residual mechanics prose that used to sit inline above the diagram,
the degradation-problem explanation, the historical He et al. result, why MobileNet instead of
VGG/ResNet50, and further reading). If you're adding a new capability to the lab experience,
default to a new sub-panel inside `#labSection` rather than a new top-level numbered section —
that's the whole point of this rework, and drifting back to many small sections is the specific
thing that got flagged as "too complicated."

## Testing changes

No test suite — static page. Verify via a local static server (root-relative `/nav.js` and
`/theme.css` mean `file://` won't pick them up). Golden path: upload a photo → click "send it
through both networks" → confirm both rows of the diagram fill in with real (non-blank,
non-identical-looking) thumbnails, not a blank canvas → confirm the bright arcs are actually
visible over MobileNetV2's row, roughly matching the "10 of 17" count in the label → confirm the
retention bars show two different, plausible percentages (not 0%, not NaN, not identical) with
MobileNetV2 reading meaningfully higher than MobileNetV1 → confirm both prediction panels show 5
rows each with plausible-looking class names and bars → confirm the stat table and share section
appear → click "Download PNG" and confirm the downloaded file contains the diagram plus all four
stat rows (including this run's actual retention percentages, not a placeholder), not a blank
image.

Watch the console for the backend — it should always read `wasm`, never `webgl` (see "A real
WebGL bug" above; this page deliberately never attempts WebGL). If you ever reintroduce a WebGL
attempt, the specific regression to check for isn't a wrong answer, it's a page that looks frozen
partway through "running both networks…" with no console error at all — that hang is silent by
default, so don't rely on the console alone to catch it; time the actual pipeline run.
