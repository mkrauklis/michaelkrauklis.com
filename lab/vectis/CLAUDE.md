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

**Layout**: sections "2. your two axes" and "3. what's on the map" sit in a left `.col`, the
compass/network plot in a right `.col`, both inside one `.row` (theme.css's generic flex
row/column pair, reused rather than page-specific CSS). This replaced an earlier fully-stacked
layout after direct feedback that adding a word and then having to scroll down to see it land
made the core "type something, watch it plot" loop feel disconnected. `.row`'s `flex-wrap: wrap`
means this collapses to the original stacked order automatically on anything too narrow for both
columns (roughly under ~865px, the two `flex-basis` values plus gap) — including every mobile
width — so no separate mobile-specific markup was needed.

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
exactly what was asked) or an isolated fluke — the page's original default example (calm/chaotic)
spanned just 0.025–0.033 on its two axes, and even the current, much better-behaved default
(ocean/mountain) still only spans 0.058–0.102. Thin raw margins are the normal case for CLIP
text-text similarity, not an exceptional one. That ruled out a threshold-triggered "unusually
thin!" warning, which would misrepresent the common case as rare. Instead, `#spreadNote` always shows
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

## Choosing axis words: topics beat traits, and it's measurable

Three separate rounds of live testing hit the same wall: "calm"/"chaotic" put a thunderstorm as
*less* chaotic than a library; "legs" ranked a worm above a dog; "safe"/"hairy" put a zebra as the
least hairy thing on the map and ranked snake as safer than bunny. The instinct was that the
specific word pair was just badly chosen — but a direct test disproved the easy fix: querying the
quick-similarity-check widget's own machinery for `cosine(axisXWord, axisYWord)` across several
antonym-style trait pairs (calm/chaotic 0.97, formal/playful 0.95, warm/cold 0.98,
peaceful/violent 0.96, simple/complex 0.97, natural/artificial 0.96) showed **every** one sitting
in essentially the same 0.95-0.98 band — this isn't a property of any one pair, it's what
antonym-adjectives do in CLIP's text space generally. Topically-different *nouns*
(ocean/mountain 0.91, food/technology 0.94, nature/city 0.93, forest/desert 0.90) score
measurably — if not dramatically — lower. The likely reason: CLIP's text tower is trained to
align with photo captions, which describe *what's depicted* far more reliably than they apply
mood/trait adjectives, so two nouns naming different subjects separate better than two adjectives
naming opposite ends of one trait.

Two things follow from this, both shipped:
- **The default example changed** from calm/chaotic + thunderstorm/library/rollercoaster/
  meditation to ocean/mountain + surfing/summit/coral reef/ski lodge/sailboat/hiking trail —
  verified live to place every item in a way that actually matches its real-world domain (ocean
  items score high on "ocean," mountain items high on "mountain"), unlike every trait-pair
  example tried. This is curation of what a first-time visitor sees, not a claim that this pair
  is uniquely "correct" — a fresh visitor typing their own trait-adjective pair will still hit the
  same wall, which is exactly why the second fix exists.
- **`#axisSimNote` now shows `cosine(axisXEmbed, axisYEmbed)` for whatever pair someone actually
  typed**, every time, with a plain-language bucket (`>=0.95` "expect a noisy, hard-to-trust
  plot," `0.90-0.95` "some noise is normal," `<0.90` "relatively distinct, as these things go").
  This is the durable fix — it makes the problem self-diagnosable for any axis pair, not just the
  shipped one, and it's what actually explains a bad result on the spot instead of leaving someone
  to independently rediscover this the way today's testing did. The static guidance paragraph
  right above it (topics beat traits, with worked examples) exists so people can avoid the trap
  before they hit it, not just after.

Don't try to "fix" this by picking yet another axis-word pair and calling it solved — there is no
pair that scores meaningfully better than ~0.90, and the real fix is the always-visible
diagnostic, not example curation.

**A further wrinkle found the same way: broad, generic category words underperform even other
topic nouns** — reported directly ("I'll use a dimension like 'person' or 'crowd' and it does
really poorly"). Verified with "person"/"landscape" against a set of items deliberately varying
in how crowded they are: "packed stadium" scored *lower* raw similarity to "person" (0.915) than
"empty room" (0.938) or "solo hiker" (0.942) did, and landed near the bottom of the plot's
"person" axis instead of the top. Reworded to "crowded"/"empty" and it still misranked the same
item ("packed stadium" landed on the *un*-crowded side). Two follow-up checks ruled out easy
fixes rather than just accepting the first bad result: (1) it's not a phrasing/prompt-template
issue — `cosine("person", "a photo of a person")` is 0.98, so wrapping a word in a fuller caption-
style template barely moves its own embedding, meaning it can't meaningfully change how anything
else ranks against it either; (2) it's not fixable by picking different but equally-generic
words — "crowded"/"empty" hit the identical failure mode as "person"/"landscape" on the same test
item. The likely cause: words this generic (person, crowd, thing, crowded, empty) show up across
such a huge fraction of caption-training data that they don't anchor a specific direction in the
embedding space the way a vivid, concrete noun like "ocean" does — closer to the "hubness"
problem documented in high-dimensional embedding spaces generally than to anything specific to
CLIP. No code fix exists for this the way the corpus-normalization fix existed for the centering
bug — it's a real limit of the underlying representation. Handled the only honest way available:
extended the axis-word tips paragraph to call out broad category words by name as an additional,
worse-than-average case, on top of "topics beat traits." Don't spend more effort chasing a
phrasing or template trick for this specific complaint without new evidence — both obvious ones
were tried and ruled out above.

## Two real CLIP quirks this surfaces, on purpose

- **Raw text-text cosine similarity runs hot and clusters tight** — see the rescaling section
  above for the full account (the phantom-pole bug it caused, the corpus-only fix, and why
  `#spreadNote` is permanent rather than conditional). This is expected model behavior, not a
  bug — resist the urge to "fix" it by, say, subtracting a baseline; that would be exactly the
  kind of hidden per-item correction the spec's §6 constraint rules out. The info panel used to
  show each selected item's raw per-axis similarity numbers directly (`#infoRaw`) as the way to
  check this — removed on direct feedback ("just make the title '{label} Similarity' and get rid
  of the raw similarity stuff"), since the ranked list plus the always-visible `#axisSimNote` /
  `#spreadNote` diagnostics already make the same point without a per-item number dump. If this
  quirk ever needs to be checkable per-item again, that's what to extend — don't resurrect
  `#infoRaw` itself, it was removed on purpose.
- **Text-image cosine similarity lives in a completely different, much lower numeric range**
  than text-text (CLIP's well-documented "modality gap") — verified directly against a real
  photo during development: raw similarity to a word landed around 0.22-0.23, versus ~0.9 for
  that same word against other words. See "Modality mode" below for how this is handled.

## Modality mode: words or photos, never both

Two earlier attempts tried to make mixed word-and-photo plots work: normalizing each kind against
only its own kind (fixed the visual squashing, but made a photo's position permanently
incomparable to a word's — pointless to plot together at all), then a gap-correction step
estimating and subtracting the offset between CLIP's image- and text-embedding clouds, per Liang
et al., ["Mind the Gap"](https://arxiv.org/abs/2203.02053) (NeurIPS 2022). The correction was
real and directionally worked, but with only one or two photos to estimate the offset from — the
realistic common case — it was too unreliable to trust, and was called out as such directly:
"you didn't solve the modality issue."

The actual fix: stop trying to make two things comparable that CLIP itself doesn't represent
comparably, and don't let the plot mix them at all. `state.mode` is `'text'` or `'image'` — never
both — controlled by the `#modeToggle` buttons (styled like the Compass/Network toggle).
Switching modes clears `state.items` outright rather than trying to hide or preserve the other
kind's items; there's no partial state where both kinds coexist even transiently. Switching to
`'image'` also proactively calls `warmVisionIfNeeded()` so the vision-tower progress bar starts
before the user even opens the file picker, rather than only after their first drop.

This is a deliberate departure from spec §2 step 2, which described items as "typed text/phrases,
or uploaded images" without restricting mixing. That assumption didn't survive contact with the
actual model's behavior — CLIP's cross-modal raw similarity isn't reliable enough, at the sample
sizes this tool actually sees, to plot both kinds on one honest scale. Don't resurrect mixing (or
a gap-correction attempt) without first re-establishing that the underlying reliability problem
has actually changed — it hasn't just gone undiscovered, it was tried twice and rejected.

With this in place, `recomputeAxes()` normalizes `embeddedItems` as one combined pool
unconditionally (see "Rescaling" above) — safe because that array can now only ever contain one
kind at a time, not because cross-modal comparability was ever actually solved.

## Network view

Deliberately reuses each item's already-rescaled `(x, y)` position as its node position (scaled
into canvas space the same way the compass view does) rather than running a separate
force-directed layout. This was a scope call, not just a shortcut: the spec's §6 constraint is
that *every* view must be traceable to the same two on-screen similarity numbers, and a
from-scratch physics layout would be a second, independent way of deciding where things go —
exactly the "second hidden computation" the spec calls out and rejects.

**Edges use Euclidean distance between the plotted points, not `cosine2D()`.** They started out
using the same origin-vector cosine similarity the ranked list uses, on the theory that reusing
one function everywhere was simpler and more consistent with the "no second computation"
principle. In practice that shipped a graph that visibly failed on two counts, both reported
directly from a real screenshot: the network only ever connected points that shared an origin
angle (i.e., pointed the same general direction from center), so anything on the opposite side of
the plot — however close it actually sat to something else — could never get an edge, and every
edge inside one visual cluster came out looking close to the same weight, because points in one
cluster naturally share a similar angle from origin *too*. Cosine-from-origin measures "same
direction," not "close together," and only the second one is what the compass view already
teaches the eye to look for. Switched to plain Euclidean distance between the two items'
positions (`Math.hypot(dx, dy)`), connecting anything within `DIST_THRESHOLD = 1.15` and scaling
opacity/width/glow by `1 - dist/DIST_THRESHOLD`. This is still spec-compliant — §6 explicitly
allows "cosine similarity (or distance)" for this view, unlike §3's ranked list, which is locked
to origin-vector cosine and must stay that way. Don't swap the ranked list to distance to "match"
this — they're allowed to differ on purpose.

## Vector arithmetic was removed

Spec §4 called for word-vector arithmetic (`A − B + C`, reporting the closest existing item to
the result) via a dedicated panel with item/operator selects. It shipped, worked correctly, and
was removed anyway on direct feedback ("remove the vector arithmetic. That's just confusing.").
There's no bug history here — don't re-add it without being asked; if a future spec revision
wants it back, `git log` has the original implementation (selects, `state.ghost` diamond marker,
the arithmetic result readout) to resurrect rather than rebuild from scratch.

## Most-/least-similar arcs

When a point is selected in the compass view, `drawCompass()` sweeps a green arc to its
single most-similar other item and a red arc to its single least-similar one — both ranked via
`rankOthers(item)`, the exact same origin-vector-cosine function `updateInfoPanel()` uses for its
ranked list (refactored out of `updateInfoPanel()` specifically so the two can never silently
disagree).

**The arcs are centered on the origin, not drawn point-to-point** — a first version connected the
selected point directly to each target with a bowed bezier curve, which read as just a curvier
version of the plain lines already radiating from center, with no obvious link to what "similar"
actually meant (direct feedback: make it "obvious that arc length is how similarity is being
calculated"). The current version (`drawSimilarityArc()`) instead draws along a circle centered
on the origin, sweeping from the selected item's angular position to the target's. That sweep is
computed as the literal shortest angular difference between the two origin-vectors (`Math.atan2`
on each, normalized to the shorter way around a full circle) — which is exactly `acos(similarity)
`, the same angle-vs-cosine relationship the "try it — angle and cosine" demo teaches by hand
further down the page. A short arc means "very similar," a long one (up to half the circle, at
similarity −1) means "very different" — length reads directly as dissimilarity, no separate
number required.

**Both radii are derived from one shared cap, not computed independently per arc.** This took
three iterations to actually fix, each caught from a real screenshot:
1. *Shared literal radius* (the selected item's own distance, used for both arcs): put green and
   red on the exact same circle whenever that distance happened to be the limiting one for both,
   so the arc drawn second (red) painted directly over the first (green) wherever their sweeps
   overlapped. Sometimes only one arc appeared to render at all.
2. *Independent fraction per arc*, each of its own `Math.min(selectedDist, targetDist)`: fixed
   the "arc swings past a closer point" problem, but not the overlap — reported directly as
   "we're still sometimes drawing the red over the green." Which of the two distances (selected's
   or that specific target's) was the limiting one varied independently per arc with the actual
   data, so the two computed radii could still land close together or coincide by coincidence.
3. **Current**: compute `cap = Math.min(selDist, bestDist, worstDist)` — the smallest of *all
   three* distances involved — once, then derive both radii from that single shared value with
   fixed, different fractions (`cap * 0.45` for green, `cap * 0.75` for red). Since both radii
   scale off the same number, the ratio between them is fixed and guaranteed distinct regardless
   of what's plotted, not just usually distinct. Red is also drawn *before* green (green is
   z-ordered last) as a second line of defense per direct instruction — green's sweep is always
   the shorter of the two, so if anything ever still overlaps, the smaller arc stays visibly on
   top instead of disappearing under the larger one. Don't revert to independent per-arc radius
   math without re-solving the coincidence problem this specifically fixes.

Below a small pixel threshold the arc is skipped entirely rather than drawing an invisible/
degenerate sliver. Compass-view only; the network view has its own distance-based edges already
serving a similar purpose there.

## Layout

Sections "2. your two axes" and "3. what's on the map" sit in a left `.col`, the compass/network
plot in a right `.col`, both inside one `.row` (theme.css's generic flex row/column pair, reused
rather than page-specific CSS) — added after direct feedback that adding a word and then having
to scroll down to see it land made the core "type something, watch it plot" loop feel
disconnected. `.row`'s `flex-wrap: wrap` collapses this to the original stacked order
automatically on anything too narrow for both columns (roughly under ~865px, the two
`flex-basis` values plus gap) — including every mobile width — so no separate mobile markup was
needed.

Within the left column, the static "pick topics, not traits" guidance paragraph and the
`#spreadNote` diagnostic are tucked into a collapsed-by-default `<details class="disclosure">`
("Tips for picking axis words"). They were originally always-visible paragraphs, but stacked
together with the axis inputs and the live `#axisSimNote` line, they made that column
dramatically taller than the plot beside it — reported directly as "all the explanation text on
#2 is killing the layout." `#axisSimNote` (the one-line, always-current "these two words are
X similar" readout) stays outside the accordion since it's short and the single most actionable
diagnostic; the longer static tip and the multi-line spread breakdown are what got tucked away.

The "3. what's on the map" panel had a similar problem in miniature: the mode toggle originally
sat under a paragraph explaining *why* words and photos are kept separate (the modality-gap
history above). Removed outright on direct feedback ("the user doesn't care. Just have the
split.") — the toggle buttons themselves are self-explanatory, and the reasoning already lives
here and in "how this actually works" for anyone who does want it.

## The embedding table (in "how this actually works")

`#embeddingTable`, populated by `updateEmbeddingTable()` (called from `recomputeAxes()`, so it
tracks every add/remove/axis-change automatically), lists every currently-plotted item's literal
`(x, y)` position — added on direct request specifically to tie the abstract explanation above it
to the visitor's own concrete session, and to open with a plain "you just generated N real CLIP
embeddings" line rather than assuming the reader already believes something happened. It's
deliberately just a restatement of `item.pos`, the exact same numbers the compass plot and the
info panel's raw-similarity line already use — not a new computation, so it can't drift from what
's actually plotted.

**Built via DOM methods (`createElement`/`textContent`), not template-string `innerHTML`** — same
reasoning as the ranked-list fix below. `item.text` is unescaped user input (typed text or a
photo's filename); interpolating it into an HTML string and assigning `innerHTML` would let a
name like `<img src=x onerror=...>` execute as markup. Verified directly: adding an item with
that exact text renders the literal string everywhere (chip, ranked list, embedding table) with
no script execution. If you touch this function, keep using element methods — don't collapse it
back into a template-string `innerHTML` assignment for brevity.

**`updateInfoPanel()`'s ranked-list rows had the same latent bug**, present since the ranked list
first shipped and unrelated to this feature — building each row's markup as a template string
assigned to `innerHTML`, with `item.text` interpolated directly into it — was fixed alongside the
new table, for the same reason and the same way (explicit `createElement`/`textContent` calls,
appended via `.append(...)`).

## Intro copy

The top-of-page intro used to spend two paragraphs on CLIP mechanics and the model-download
privacy note before a first-time visitor ever saw the tool itself. Trimmed to two sentences —
what an embedding is, and that picking two axes turns that into a map — with the CLIP mechanics,
the privacy/download details, and the "same space of numbers" explanation all moved into the
first paragraph of the "how this actually works" disclosure instead, ahead of the pre-existing
cosine-similarity explanation there. Don't grow the top intro back into a mechanics lecture;
if something's worth explaining in depth, it almost always belongs below the tool, not above it.

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
  t-shirt link is gated behind `updateMerchGating()`'s `state.mode === 'text'` check (a shirt
  printed with someone's own uploaded photo is a private/personal object, not something to wear
  in public); the magnet is offered unconditionally. Now that word/photo mode is exclusive (see
  "Modality mode"), this check simplified from "every current item is text" to just the mode
  flag — equivalent in practice, since items can no longer be mixed.

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

Golden path: wait for "Language model ready," confirm the six example items
(surfing/summit/coral reef/ski lodge/sailboat/hiking trail) land at distinct positions roughly
matching their real-world domain, change an axis word and confirm the label, `#axisSimNote`, and
every point's position update and `#embeddingTable` (in "how this actually works") updates its
rows to match, click a point and confirm the "{label} Similarity" title and ranked list appear
alongside a green arc and a red arc, both centered on the origin (not point-to-point) and at
visibly different radii (never overlapping/overpainting each other), with the red one visibly
longer whenever its similarity is more negative than the green one's is positive, expand
the "Tips for picking axis words" accordion and confirm `#spreadNote` has live numbers in it,
toggle to Network and confirm edges vary visibly in weight and some cross the plot's center
(not just within one visual cluster), and download both PNGs. Separately, click the Photos mode
button, confirm the word items disappear and the vision-tower progress bar starts immediately
(before dropping anything), drop in two different real photos and confirm they land at distinct
positions rather than both collapsing to the same point, then switch back to Words mode and
confirm the map is empty again (not a leftover mix of both kinds) and "Reset to example" still
repopulates the six word items.

One easy-to-miss testing trap: this page's whole render loop runs on `requestAnimationFrame`,
which browsers fully pause whenever `document.visibilityState !== 'visible'` — every visual check
(arcs, animations, exported PNGs) will then silently fail with no console error, even though
everything else (model loading, item embedding, DOM updates) works fine, since those don't depend
on rAF. Don't trust a browser-automation tool's own "this tab is active/selected" bookkeeping as
proof the page is actually visible — confirmed directly, more than once, that a tab reported as
active still had `document.visibilityState === 'hidden'` (and `document.hasFocus()` false),
silently stalling `drawCompass()`/`drawNetwork()` from ever running. Check
`document.visibilityState` from inside the page itself before trusting a "nothing changed"
result. If it's stuck hidden and switching tabs doesn't fix it, don't wait on it — call the
render function directly for a one-off synchronous draw (a temporary
`window.__x = () => render(performance.now())`, invoked from the test script, works and doesn't
depend on rAF at all); reading canvas pixel data back afterward works regardless of page
visibility, only the automatic scheduling is affected. Remove any such debug hook before
shipping.

This was all verified end-to-end against the real hosted model during development (not mocked) —
CDN/model-hub access is required for any of it to work, so if you're testing somewhere
network-restricted, expect the text-model row to sit at "Loading language model…" forever and
everything downstream of it to stay inert.
