# Vectis

Lets someone pick two words as axes — anything, "formal" and "playful" — then plots words,
phrases, or their own photos on that 2D map based on nothing but CLIP cosine similarity to each
axis word. Also does word-vector arithmetic and a node-link "network" view, both built from the
exact same two numbers per item that are already on screen. Built from a locked spec
(`vectis-spec.md`, not checked into the repo) — see that document for the product rationale;
this file is about how it's actually implemented and why.

## Architecture

`index.html` + `worker.js`, plus the two site-wide shared files (`/nav.js`, `/theme.css` — see
root `CLAUDE.md`). Accent color is a cyan (`#57c2e0`) not otherwise used on the site — amber is
Ridgeline's, violet is Afterimage's.

**Why a CDN import, breaking the site's usual "no external JS" rule**: Ridgeline and Afterimage
both do all their math by hand, in inline `<script>`. Vectis can't — it needs an actual trained
vision-and-language model (CLIP) to turn a word or photo into a vector, and hand-rolling that is
obviously out of scope. `worker.js` is a `type: "module"` Web Worker that does
`import ... from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'` — a genuine,
deliberate exception to the site's "no bundler, no external JS dependencies" convention. It's
scoped as tightly as possible: only `worker.js` imports anything, `index.html` itself stays
dependency-free, and the import is pinned to an exact version so a jsdelivr/npm update can't
silently change behavior.

**Why a Worker at all**: loading two ONNX models and running inference on them can take a
noticeable moment. Doing it on the main thread would freeze the axis-swap animation and the
angle/cosine demo (both plain `requestAnimationFrame` canvas loops) every time someone adds an
item. `index.html` talks to the worker with a small request/response wrapper
(`callWorker(type, payload)` returns a Promise, keyed by an incrementing request id in a
`pending` Map) plus a side-channel `progress` message type the worker sends independent of any
particular request, since progress belongs to "the model is loading" rather than to whichever
call happened to trigger that load.

## The model

`Xenova/clip-vit-base-patch32`, loaded as two independent towers so a text-only user never pays
for the (larger) vision tower:
- **Text tower** (`AutoTokenizer` + `CLIPTextModelWithProjection`) loads immediately on page
  load (`warmText()`), since axis words always need it — there's no useful state before it's
  ready, so `setControlsEnabled(false)` disables axis/add-text inputs until it resolves.
- **Vision tower** (`AutoProcessor` + `CLIPVisionModelWithProjection`) loads lazily, on the
  first image drop (`warmVisionIfNeeded()`), so someone who only ever plots words never
  downloads it.

Both emit `text_embeds`/`image_embeds` from the same joint space; the worker L2-normalizes every
row before sending it back (`normalizeRows()`), so `cosine()` on the main thread is just a dot
product. `env.allowLocalModels = false` keeps transformers.js from trying anything except the
public hub.

## The rescaling formula, and why it's not just raw cosine

Spec requirement: axis words act as "the two reference poles" for a min/max rescale, so the plot
always uses its full -1..1 range regardless of how tightly raw similarities cluster (they do
cluster — see the CLIP quirks below). `recomputeAxes()` builds the pool for the X axis as
`[cosine(axisXWord, axisXWord), cosine(axisYWord, axisXWord), ...items.rawX]` — i.e. the axis
word's similarity to itself (always 1, so this is effectively always the pool's max) and the
*other* axis word's similarity to this one (a real, sometimes-surprising lower bound), alongside
every item's raw value. Y is the mirror image. This guarantees the axis word itself anchors its
own positive extreme, and everything else — including the other axis word — gets stretched to
fit underneath it. `rangeX/rangeY` fall back to `1` if the pool is degenerate (e.g. both axis
words embed identically) to avoid dividing by zero.

Every recompute increments `axisSeq`; a recompute whose embedding lookup resolves after a newer
one has already started is discarded (`if(mySeq !== axisSeq) return`). This matters because
axis-word changes and item-adds both call `recomputeAxes()`, and a slow one (an axis word not
yet in `textCache`) shouldn't clobber a faster one that started later.

## Two real CLIP quirks this surfaces, on purpose

- **Raw text-text cosine similarity runs hot.** Two arbitrary short phrases in CLIP's text space
  routinely score 0.85-0.95 raw cosine similarity, whether or not they're related. This is
  exactly the problem the spec's rescaling step exists to paper over — see `updateInfoPanel()`'s
  raw-numbers readout, which shows this plainly rather than hiding it (`raw similarity to "calm":
  0.914 • raw similarity to "chaotic": 0.913`, before rescaling stretches that thin gap to fill
  the whole axis). This is expected, not a bug — resist the urge to "fix" it by, say, subtracting
  a baseline; that would be exactly the kind of hidden per-item correction the spec's §6
  constraint rules out.
- **Text-image cosine similarity lives in a completely different, much lower numeric range**
  than text-text (CLIP's well-documented "modality gap") — verified directly against a real
  photo during development: raw similarity to a word landed around 0.22-0.23, versus ~0.9 for
  that same word against other words. Combined with the min/max rescale, this means dropping one
  photo onto a text-heavy map tends to shove it straight into a corner, regardless of what the
  photo actually shows — the image isn't reading as "extreme," it's reading as "a different
  modality." `updateMerchGating()` also computes this and toggles `#modalityNote`, a plain
  on-page warning, whenever the current item set mixes both kinds. Don't remove this note without
  also fixing the underlying gap (a nontrivial, out-of-scope change) — without it, a mixed plot
  looks broken instead of explained.

## Network view

Deliberately reuses each item's already-rescaled `(x, y)` position as its node position (scaled
into canvas space the same way the compass view does) rather than running a separate
force-directed layout. This was a scope call, not just a shortcut: the spec's §6 constraint is
that *every* view must be traceable to the same two on-screen similarity numbers, and a
from-scratch physics layout would be a second, independent way of deciding where things go —
exactly the "second hidden computation" the spec calls out and rejects. Edges are drawn between
any pair whose 2D origin-vector cosine similarity (`cosine2D()`, the same function the ranked
list and arithmetic "closest to" result use) exceeds `0.55`, with opacity/glow scaled by how far
above that floor they are.

## Vector arithmetic

`A op1 B [op2 C]` is computed directly on the already-rescaled positions (`A.pos.x + opAB *
B.pos.x`, etc.), clamped to `[-1, 1]`, then run through the exact same `cosine2D()`-based ranking
as a normal selection to report what it's closest to. No embedding call happens here — that's
the point (spec §4). The result renders as a dashed line + diamond marker (`state.ghost`) so it
reads visually as distinct from a real, embedded item.

## Animation

`liveXY(item, now)` interpolates between `item.animFrom` and `item.animTo` with a 650ms
`easeInOutCubic`, recomputed fresh every time `recomputeAxes()` runs (whether from an axis-word
change or a new item arriving) by capturing the *current* on-screen position as the new
`animFrom`. A persistent `requestAnimationFrame` loop (`loop()` → `render()`) always redraws
both the compass and network canvases regardless of whether anything is actually animating —
simpler than dirty-checking, and cheap enough for a canvas this size.

## Export

`exportMode('compass'|'network')` draws onto a fresh, off-screen `LOGICAL`×`LOGICAL` canvas by
calling `drawCompass`/`drawNetwork` with that canvas's own context — both draw functions take
`ctx` as an explicit first argument rather than closing over the on-screen canvas's context, so
export never depends on which view is currently showing on screen, and doesn't need to
temporarily swap anything.

## Monetization

- AdSense: same script tag + `google-adsense-account` meta, same publisher id as the rest of the
  site.
- Zazzle: two separate "create your own" templates (magnet, t-shirt), each a plain link with the
  same `?rf=` ambassador param already used by Ridgeline — no API integration. Per spec §9, the
  t-shirt link is gated behind `updateMerchGating()`'s `allText` check (every current item must
  be `kind: 'text'`) since a shirt printed with someone's own uploaded photo is a private/personal
  object, not something to wear in public; the magnet is offered unconditionally.

## Shared theme.css change

Added a generic `input[type=text], input[type=search], select` rule to the root `theme.css` —
Vectis is the first lab tool whose primary input is typed text rather than a photo or a slider,
and neither Ridgeline nor Afterimage needed this chrome. Kept in the shared file rather than
duplicated locally so a future tool gets it for free, per the root `CLAUDE.md`'s stated
preference for relying on `theme.css` defaults before writing new page-local CSS.

## Testing changes

No test suite — static page + a Web Worker. Serve the repo root (`python -m http.server` or
equivalent) and browse to `/lab/vectis/`; opening `index.html` directly over `file://` won't
pick up `/nav.js` or `/theme.css`, and module workers generally won't load over `file://` at all.

Golden path: wait for "Language model ready," confirm the four example items
(thunderstorm/library/rollercoaster/meditation) land at distinct positions, change an axis word
and confirm both the label and every point's position update, click a point and confirm the raw
numbers + ranked list appear, run a vector-arithmetic combination and confirm a dashed ghost
point appears with a sensible "closest to" result, toggle to Network and confirm edges render,
and download both PNGs. Separately, drop in an actual photo and confirm the vision-tower
progress bar appears, the item eventually gets a real position (not stuck at pending), and the
modality-gap note appears once the plot has both kinds. This was all verified end-to-end against
the real hosted model during development (not mocked) — CDN/model-hub access is required for any
of it to work, so if you're testing somewhere network-restricted, expect the text-model row to
sit at "Loading language model…" forever and everything downstream of it to stay inert.
