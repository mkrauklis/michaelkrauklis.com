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

**Why a CDN import**: Ridgeline does all its math by hand in inline `<script>` with zero external
JS, but Afterimage already broke from that — it loads TensorFlow.js, `tfjs-backend-wasm`,
`qrcode-generator`, and `jsQR` from jsdelivr, because training a live autoencoder head in the
browser isn't something you hand-roll. Vectis follows that same established precedent, not a new
one: it needs an actual pretrained vision-and-language model (CLIP) to turn a word or photo into
a vector, so `worker.js` — a `type: "module"` Web Worker — does
`import ... from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'` (transformers.js,
running the model via ONNX Runtime Web under the hood, rather than tfjs — CLIP's public ONNX
builds live on the Hugging Face hub, and transformers.js is what knows how to fetch and run
them). It's still scoped as tightly as this pattern allows: only `worker.js` imports anything,
`index.html` itself stays dependency-free, and the import is pinned to an exact version so a
jsdelivr/npm update can't silently change behavior.

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

## The rescaling formula, and why it's corpus-only, not pole-anchored

The spec's original text called for axis words to act as "the two reference poles" for a min/max
rescale — pooling each item's raw score together with the axis word's similarity to itself
(always exactly 1) and to the other axis word, then normalizing against that pool. **This shipped
first and was wrong in practice**, caught from a real screenshot: every item came out clustered
in one corner, heavily biased negative, the plot never centered. The cause was the phantom pole —
cosine similarity of 1 is only reachable by a word compared against itself, so no real item could
ever get near that end of the range, and every actual item got compressed into whatever fraction
of the range was left underneath it.

The fix, direct from that feedback: **normalize purely against the corpus of scores the current
items have actually achieved** — no axis self-similarity, no phantom poles. For each axis,
`recomputeAxes()` takes the min and max of `item.rawX` (or `rawY`) across every embedded item and
maps that range to exactly `[-1, 1]`. This guarantees whichever item scored highest lands at the
positive extreme and whichever scored lowest lands at the negative extreme, no matter what the
actual raw numbers are — "even if everything is negative the lowest and highest become the
normalizing min/max," per the instruction that fixed it. A single item (no meaningful range yet)
is centered at the origin instead of forced to an edge; two-plus items get the full min/max
treatment, guarded against a zero-width range (`rangeX > 1e-9`) for the case where every item
scores identically.

**The tradeoff this creates, and why there's a permanent note about it, not a conditional one**:
stretching *whatever* range exists to fill the whole axis means a razor-thin, noise-sized raw
margin gets displayed with exactly the same visual confidence as a wide, meaningful one. Verified
directly: eight animal words spanned only 0.046 of raw similarity to "legs" (0.859–0.905), and
"worm" — which has no legs — outscored "dog". This isn't a bug in the normalization (it's doing
exactly what was asked) or an isolated fluke (the page's own default example spans just
0.025–0.033 on its two axes) — thin raw margins are the normal case for CLIP text-text
similarity, not an exceptional one. That ruled out a threshold-triggered "unusually thin!"
warning, which would misrepresent the common case as rare. Instead, `#spreadNote` always shows
the live, current range on both axes whenever there are 2+ items, framed as a standing fact about
how to read the plot rather than an alarm. Don't turn this back into a conditional warning
without re-checking that assumption.

Every recompute increments `axisSeq`; a recompute whose embedding lookup resolves after a newer
one has already started is discarded (`if(mySeq !== axisSeq) return`). This matters because
axis-word changes and item-adds both call `recomputeAxes()`, and a slow one (an axis word not
yet in `textCache`) shouldn't clobber a faster one that started later.

## Axis labels: no arrows, ever

The left/right end-labels originally used `writing-mode: vertical-rl` (to fit in the plot's
narrow side margins) plus a literal `←` character to mean "this end is the *less* pole" — e.g.
`← less scary` down the left edge. This actively lied about direction: Unicode gives directional
arrows a "rotate" orientation for vertical text layout, so browsers auto-rotate `←` 90° clockwise
inside `vertical-rl` text, and a left-pointing arrow rotated 90° clockwise points *up*. Caught
directly from a user screenshot asking why the arrow for "less scary" pointed up. The bottom
label had the same character used for a vertical axis's "less" direction, which was never
correct either, rotated glyph or not — down isn't left.

The fix drops arrows entirely rather than trying to pick a "correct" one per edge: labels are now
plain horizontal text at all four positions (`.axis-label.left`/`.right` no longer set
`writing-mode`, just a `width: 60px` so "LESS SCARY" wraps to two short lines instead of running
into the plot), and direction is instead communicated the way the top/bottom labels always did it
— position (the labeled word sits at its own positive edge) plus `.axis-label.pos`'s accent
color and bold weight marking the positive pole, with the negative pole left in plain dim ink.
That combination doesn't depend on a reader parsing a glyph at all. Don't reintroduce a directional
arrow character inside rotated/vertical text without checking Unicode's vertical orientation
property for it first — this exact failure mode is easy to reintroduce by accident.

## Two real CLIP quirks this surfaces, on purpose

- **Raw text-text cosine similarity runs hot and clusters tight** — see the rescaling section
  above for the full account (the phantom-pole bug it caused, the corpus-only fix, and why
  `#spreadNote` is permanent rather than conditional). This is expected model behavior, not a
  bug — resist the urge to "fix" it by, say, subtracting a baseline; that would be exactly the
  kind of hidden per-item correction the spec's §6 constraint rules out. `updateInfoPanel()`'s
  raw-numbers readout is what makes it checkable at all — don't remove it.
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

`drawCompass` explicitly fills `#0b0e14` before drawing, matching `drawNetwork` — the on-screen
`<canvas>` gets its dark background from CSS, but a freshly created export canvas has none, so
without this fill the exported compass PNG came out with a *transparent* background that any
viewer without its own dark theme (a phone's photo viewer, for instance) renders on white,
leaving the light-gray grid and labels almost invisible. Caught from an actual exported PNG the
user sent back.

## Monetization

- AdSense: same script tag + `google-adsense-account` meta, same publisher id as the rest of the
  site.
- Zazzle: two separate "create your own" templates (magnet, t-shirt), each a plain link with the
  same `?rf=` ambassador param already used by Ridgeline — no API integration. Per spec §9, the
  t-shirt link is gated behind `updateMerchGating()`'s `allText` check (every current item must
  be `kind: 'text'`) since a shirt printed with someone's own uploaded photo is a private/personal
  object, not something to wear in public; the magnet is offered unconditionally.

## Shared theme.css changes

- Added a generic `input[type=text], input[type=search], select` rule — Vectis is the first lab
  tool whose primary input is typed text rather than a photo or a slider, and neither Ridgeline
  nor Afterimage needed this chrome. Kept in the shared file rather than duplicated locally so a
  future tool gets it for free, per the root `CLAUDE.md`'s stated preference for relying on
  `theme.css` defaults before writing new page-local CSS.
- Added `display: block` to `.dropzone`. A `<label>` is inline by default, and an inline
  element's border/background doesn't grow to enclose a second wrapped line of text — it just
  lets that line spill out past the box. Ridgeline and Afterimage never hit this because their
  dropzone label text is short enough to never wrap; Vectis's ("Click or drop images here — each
  one gets its own point") does wrap on a narrow phone screen, and the overflowing second line
  visually collided with the chip list sitting right below it. Fixing it in the shared rule
  rather than locally, since it's a defect in the shared component itself, not something specific
  to Vectis's copy.

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
