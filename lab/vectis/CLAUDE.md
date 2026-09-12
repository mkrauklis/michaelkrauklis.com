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

## The models — two of them, not one

This used to be one model (CLIP) doing everything. It's now **two**, chosen per `state.mode`, not
by user choice — see "Two embedding models, one per mode" below for why and how this was decided.
`env.allowLocalModels = false` keeps transformers.js from trying anything except the public hub,
for both.

- **General text model** (`Xenova/all-MiniLM-L6-v2`, loaded via `pipeline('feature-extraction',
  ...)`) — used for every word-to-word comparison in Words mode: axis words, typed items, and the
  quick-similarity-check widget. Loads immediately on page load (`warmGeneralText()`), since
  Words mode is the default and always needs it.
- **CLIP** (`Xenova/clip-vit-base-patch32`), loaded as two independent towers, used **only** in
  Photos mode:
  - **Text tower** (`AutoTokenizer` + `CLIPTextModelWithProjection`) embeds the axis words —
    the only way they end up in the same space as an uploaded photo at all.
  - **Vision tower** (`AutoProcessor` + `CLIPVisionModelWithProjection`) embeds the photos.
  - Both load together, lazily, the first time Photos mode is actually selected
    (`warmVisionIfNeeded()`, which now also calls `warmClipTextIfNeeded()` in parallel) — someone
    who only ever plots words never downloads either CLIP tower.

CLIP's two towers emit `text_embeds`/`image_embeds` from the same joint space; the worker
L2-normalizes every row before sending it back (`normalizeRows()` for CLIP, `pipeline`'s own
`normalize: true` for the general model), so `cosine()` on the main thread is just a dot product
either way. The general model's embeddings and CLIP's embeddings are **not comparable to each
other** — different space, different dimensionality (384 vs. 512) — which is exactly why they're
kept in separate caches (`generalTextCache`/`clipTextCache`) and never mixed; see "Two embedding
models" below.

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

## Text embedding centering — the actual fix for "everything's noise"

Everything in the next section ("Choosing axis words") and several fixes before it (the
corpus-only rescale, the `#axisSimNote`/`#spreadNote` diagnostics, the "topics beat traits"
guidance) were built on top of one unexamined assumption: that CLIP's raw text-text cosine
similarity clustering at 0.85-0.98 regardless of actual relatedness was just an inherent,
unfixable property of the model, best documented and worked around rather than fixed. That
assumption held for a long time and drove real, working fixes (the corpus normalization is still
correct and still needed) — but it turned out to be wrong. There *is* a fix, and it changes the
numbers throughout this file dramatically enough that older sections below describe the
*pre-fix* behavior; they're kept as history (the reasoning and the process of finding this were
real), but don't trust their specific numbers as current.

**What's actually happening**: this is the well-documented "anisotropy" (or representation
degeneration) problem in transformer sentence embeddings — most of a short text's embedding
points along one shared, content-independent "generic text" direction, so any two embeddings
look artificially similar to each other regardless of what they actually say. It's a different
phenomenon from the "modality gap" (image vs. text) that was investigated and ultimately
sidestepped earlier — this one only involves the text tower, and unlike the modality gap, it
turned out to have a fix simple enough to actually ship.

**The fix**: subtract a fixed reference mean from every text embedding, then renormalize to unit
length, before computing any cosine similarity. This is the standard treatment for anisotropic
sentence embeddings (mean-centering, sometimes paired with whitening in the literature) — not a
bespoke rescaling invented for this tool. The reference mean was derived once, offline, from
~225 words deliberately spanning many unrelated semantic categories (animals, nature, colors,
emotions, materials, food, technology, tools, places, professions, abstract concepts, body parts,
time, personality traits — plus specific words from this session's bad-result reports, like
"leech," "hairy," "crowded"), embedded live through the real shipped model, averaged, and
hardcoded as `TEXT_EMBED_MEAN` in `text-embed-mean.js`. `centerAndNormalize()` in `index.html`
applies it, called from inside `embedTexts()`'s caching layer — so every consumer (axis words,
item text, the quick-similarity-check widget) gets the corrected embedding transparently, and
nothing needs to special-case it.

**Verified directly, before shipping, not assumed**: this was the whole point — earlier sections
in this file repeatedly say "verified directly" about things that later turned out to still be
broken in a way the correction fixes. Before writing any of this:
- `cosine(leech, hairy)`: 0.95 raw → **-0.001** centered. (A leech has no hair; the tool used to
  rank it as the single hairiest thing on the map.)
- `cosine(fish, hairy)`: 0.92 raw → **-0.040** centered. (Correctly negative now.)
- `cosine(dog, cat)`: 0.94 raw → **0.364** centered, the *highest* of every pair tested — pets
  correctly identified as the most related pair, instead of being indistinguishable from
  `cosine(dog, computer)` (0.93 raw → -0.035 centered).
- The shipped default example (ocean/mountain + the six-item set) went from every item sitting
  near the diagonal to genuine anti-correlation: surfing (0.365 ocean, -1.000 mountain), summit
  (-1.000 ocean, 0.282 mountain), hiking trail (-0.151 ocean, 1.000 mountain) — ocean items score
  low on "mountain" and vice versa, which never reliably happened pre-correction.
- The user's exact reported failure case (axes "fire"/"hairy"; items including monkey,
  charizard, fish) went from collapsed-diagonal to: monkey (0.39 fire, **1.00 hairy** — correct,
  monkeys have fur), charizard (**1.00 fire**, -0.35 hairy — correct, a fire-breathing reptile
  isn't hairy), fish (0.02 fire, **-1.00 hairy** — correct, the exact complaint that triggered
  this investigation).

**A second, related finding from the same testing pass**: after centering, a word's *residual*
magnitude before renormalization (`Math.sqrt(sum of (embedding - mean)^2)`) is itself
informative. Broad, generic words sit close to the mean and have a small residual — "thing"
(0.11), "something" (0.12), "stuff" (0.13), "person" (0.14) — versus concrete, specific words,
which sit further away — "ocean" (0.33), "dog" (0.28), "charizard" (0.40), "coral reef" (0.51).
A small residual means more of that word's final (renormalized) direction is determined by
whatever's left after removing the generic direction — i.e. less signal, more noise, a shakier
axis. This is the actual, measured mechanism behind "broad category words underperform," not the
disproven claim that "topic nouns are just inherently better than trait words" (post-centering,
"hot"/"furry" at 0.063 similarity to each other is *more* independent than "ocean"/"mountain" at
0.202 — the opposite of what the tool claimed before this fix, when that specific comparison was
never actually tested against corrected numbers). Don't reintroduce the old "concrete nouns beat
adjectives" framing; the generic-vs-specific / residual-magnitude framing is the one with
evidence behind it.

**Everything downstream got recalibrated for the new scale, since post-centering similarity
values live in a completely different range** (mostly -0.15 to 0.4, versus 0.85-0.98 raw):
`#axisSimNote`'s thresholds moved from `>=0.95`/`>=0.90` to `>=0.35`/`>=0.15`; `#spreadNote`
dropped its blanket "margins this thin are always normal for CLIP" claim (no longer true — a
0.2-0.5 range is now typical, not thin) in favor of a conditional note that only flags a
genuinely thin (`<0.1`) spread; the axis-word tips paragraph's "ocean/mountain beats hot/furry"
claim was corrected per the residual-magnitude finding above. If you change `TEXT_EMBED_MEAN` or
the underlying model, re-run the verification pairs above and re-check all of these thresholds —
they're calibrated to this specific correction, not universal constants.

**If the CLIP model ID in `worker.js` ever changes**, `TEXT_EMBED_MEAN` must be regenerated —
it's specific to `Xenova/clip-vit-base-patch32`'s text tower, not a general-purpose constant.
Procedure: temporarily expose `embedTexts` on `window` (as was done during this investigation,
then removed before shipping), embed a similarly large and diverse word batch through the new
model, average the results, and replace the array in `text-embed-mean.js`. Re-verify against the
pairs above before trusting the new mean.

## Choosing axis words: topics beat traits, and it's measurable

*(Historical record of the investigation that led to the fix above — the specific similarity
numbers quoted below are all pre-centering/raw values and no longer reflect what the tool
computes. The reasoning that led here was real and worth keeping; don't recalibrate anything
against these numbers.)*

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
  typed**, every time, with a plain-language bucket (originally `>=0.95` "expect a noisy,
  hard-to-trust plot," `0.90-0.95` "some noise is normal," `<0.90` "relatively distinct, as these
  things go" — see below for why the top threshold moved to `0.93` and the wording got more
  specific). This is the durable fix — it makes the problem self-diagnosable for any axis pair,
  not just the shipped one, and it's what actually explains a bad result on the spot instead of
  leaving someone
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
CLIP. At the time, no code fix seemed to exist for this — handled by extending the axis-word tips
paragraph to call out broad category words by name.

**Update, once text embedding centering shipped (see that section above)**: this partially
resolved after all, just not via a phrasing trick — centering gives a precise, measured
explanation for *why* generic words underperform (their residual after mean-subtraction is
small, so their post-renormalization direction carries more noise relative to signal — see the
numbers in "Text embedding centering") and does measurably improve their behavior along with
everything else's. It doesn't make them as reliable as concrete words — a small residual is still
a small residual — so the tips paragraph still calls them out, now with the correct mechanism
instead of the disproven "generic-across-everything" framing. Don't spend more effort chasing a
phrasing or template trick for this specific complaint without new evidence — both obvious ones
were tried and ruled out above, before centering existed.

**The "topics beat traits" framing itself got corrected** after direct pushback: the on-page
examples (good: ocean/mountain; bad: calm/chaotic, hairy/safe) read as "pick a topic, not a
trait," but that's not actually the tool's intended semantics, and "hairy"/"safe" was never a
pair of opposite traits in the first place — it's two *unrelated* trait words that happened to
also perform badly (see the person/crowd finding above). What Vectis's axes are actually meant to
be is two **independent** dimensions, the way you'd pick two features to classify something by
("hot" and "furry," "kind" and "fast") — not two ends of one spectrum ("hot" and "cold"), which
are redundant with each other by construction, not independent. Checked directly before rewriting
anything: `cosine(hot, cold)` = 0.97, `cosine(kind, unkind)` = 0.99, `cosine(loud, quiet)` = 0.97
(antonym pairs) versus `cosine(hot, furry)` = 0.95, `cosine(loud, furry)` = 0.95, `cosine(kind,
fast)` = 0.98 (unrelated pairs) — genuinely independent adjective pairs score only marginally
better than antonym pairs, nowhere near as well as concrete topic nouns (ocean/mountain 0.91).
So both lessons are true and stay in the tips text together: prefer independent dimensions over
spectrum endpoints (the *correct mental model* for how this tool's axes work, regardless of raw
numbers), and prefer concrete/specific words over broad/generic ones (the stronger *empirical*
lever, per the finding above) — don't collapse back to a single "topics vs. traits" axis of
advice, the two corrections address different failure modes and both are worth keeping.

**The `#axisSimNote` severity threshold was recalibrated after a real report made the old one
look too lenient.** "dangerous"/"hairy" (0.94-0.95, landing just under the original `>=0.95`
cutoff) put every one of eight test items — anaconda, alligator, frog, gorilla, leech, explosion,
fire, nuclear warfare — almost exactly on the plot's diagonal, and "leech" came out as the single
*most hairy* item on the map despite having no hair. Verified directly, per-item, via the
quick-similarity-check machinery rather than assumed: raw similarity to "dangerous" and to
"hairy" moved in near-lockstep for every single item (leech: 0.97 / 0.95; frog: 0.87 / 0.85;
gorilla: 0.93 / 0.93; full numbers in the commit that made this change) — exactly what you'd
expect when two axis words are themselves 0.94-0.95 similar: an item's similarity to one is
almost fully determined by its similarity to the other, so independently min/max-normalizing
each axis still produces two nearly-identical rescaled coordinates per item, hence the diagonal.
"Leech reading as maximally hairy" isn't a separate bug on top of that — once the two axes
collapse into one, whichever item scores highest on that one shared dimension (here, apparently
something like "unpleasant/dangerous," which leech scores highest on) automatically reads as the
extreme on *both* axes, hair having nothing to do with it. This is the exact failure mode
`#axisSimNote` exists to warn about — it just had the wrong threshold and undersold the severity
of what actually happens. Moved the top bucket from `>=0.95` to `>=0.93`, and reworded it from
vague ("expect a noisy, hard-to-trust plot") to specifically describe the diagonal-collapse
symptom, since that's what someone will actually see and should be able to recognize by name.
If another real example lands just under `0.93` and still shows this same collapse, lower the
threshold again rather than assuming this one data point pinned the exact cutoff.

**Update: this whole "dangerous"/"hairy" case is what directly motivated the text embedding
centering fix** (see that section, well above this one). The `0.93`/`0.90` thresholds here
describe the *raw*, pre-centering similarity scale and are no longer what the code uses — after
centering, similarity values live in roughly a -0.15 to 0.4 range instead of 0.85-0.98, so the
live thresholds moved to `0.35`/`0.15`. Kept this section as-is since it's the accurate account
of how the problem was first caught and diagnosed; just don't use its numbers to sanity-check
current behavior.

## A deeper limitation: properties that aren't visible in a photo (or a sentence)

Reported directly, with a concrete example: on a "speed" axis, **worm ranked higher than
cheetah**. Investigated properly rather than patched on the spot — tested directly against the
real shipped pipeline (centering included) before theorizing anything:

- Axis "speed": greyhound (0.10, reasonable) then **snail** (0.07), then snake, then **worm**
  (0.05) — with **cheetah near the very bottom** (-0.01), behind sloth's only slightly lower
  -0.05.
- Axis "fast": **sloth outranks cheetah** (0.03 vs. 0.01).
- Axis "quickness": **sloth ranks #1 of 11 animals** (0.14), ahead of cheetah (0.09).

Not an isolated fluke of the word "speed": the identical shape of failure showed up on
**"loud"** — goldfish (a mute animal) ranked loudest; lion ranked *last*, behind mouse. Meanwhile,
properties that are directly, statically visible in a photo worked cleanly on the same items:
"spotted" correctly ranked cheetah/dalmatian/leopard above golden retriever/black bear; "colorful"
correctly ranked parrot above a cardboard box.

**The pattern**: CLIP's text tower (and, it turns out, general text embeddings too — see below)
reliably separates properties that are *directly, statically visible in a photograph* — spotted,
colorful, hairy, dangerous-looking all work, because the axis word and the item co-occur in the
same photos often enough to teach the model a real direction for it. Properties that are
*behavioral, temporal, auditory, or otherwise not depicted in a typical static photo* — speed,
loudness, and almost certainly others like intelligence or lifespan — have no such shared visual
grounding to learn from, and the resulting "similarity" is close to noise, sometimes inverted
from reality. This is not the anisotropy problem centering already fixes, and centering does not
touch it — it's a ceiling on what a fixed word/sentence embedding can represent *at all*, not a
computational error to correct with more/different vector math.

**Tested and ruled out: switching to a different, non-CLIP model doesn't fix it either.** Before
concluding this was fundamental rather than CLIP-specific, `Xenova/all-MiniLM-L6-v2` — a general
sentence-embedding model, trained on regular web sentences and paraphrase pairs, not photo
captions — was tested against the same animals, both as bare words and as full comparative
sentences ("This animal is very fast." vs. "This animal is a sloth."). Sentence-framing is the
input shape this model actually trained on, so it was the most favorable test that could
reasonably be tried. Result: **sloth still ranked #1 fastest** (0.586), ahead of cheetah (0.531)
— worse in relative terms than the bare-word version. Two architecturally unrelated models, four
phrasing variants, the same failure. The likely reason: a comparative fact like "cheetahs are
faster than sloths" is *relational* — true of a pair, not a property either animal "has" the way
"spotted" is directly true of a cheetah on its own — and no model trained this way (predicting
what co-occurs with what) has a mechanism to retrieve a relational fact from a single word's fixed
embedding, regardless of what text it was trained on. ("Loud" moved somewhat in the right
direction with sentence-framing under the general model — elephant/lion/owl all correctly beat
goldfish — so sentence-framing isn't *worthless*, just nowhere near reliable enough to trust as a
fix; this is why the general model is used for its own real benefits, below, not as a claimed
solution to this specific problem.)

**What actually addresses this**: not a smarter embedding computation (tested, doesn't exist for
this at the word/sentence level), but giving the visitor direct control — see "Manual override"
below. An LLM asked to reason explicitly ("cheetahs are faster than sloths") could likely get this
right, since that's genuine comparative reasoning rather than a fixed-vector lookup — but that
would mean either a server call (breaking this site's "nothing leaves your browser" guarantee,
upheld by every tool here) or an impractically large in-browser model, and was explicitly not
pursued for that reason. If this ever gets revisited, that trade-off needs to be made
consciously and by name, not slipped in as an implementation detail.

**This limitation was only documented here, not to visitors, until a real user hit it live** —
reproduced directly with the exact "fast"/"hairy" axis pair (sloth landed furthest out on "fast,"
ahead of cheetah, matching the numbers above almost exactly) before concluding it was the same
known issue rather than a new regression. The `.select-hint` paragraph right under the compass
(always visible, not tucked in the collapsed "Tips" disclosure) now names this directly —
properties you can see or read work reliably, behavioral ones like speed/loudness often don't,
and that's a real limit of the technique rather than a bug — immediately next to the reminder that
dragging a point is the actual fix. Don't move this explanation into the collapsed disclosure or
delete it as "redundant with CLAUDE.md" — CLAUDE.md is never read by an actual visitor, and this
exact gap (a real limitation, fully understood and even fixed via manual override, but invisible
to the person hitting it) is what produced the "this isn't working at all" report in the first
place.

## Two embedding models, one per mode

Direct follow-up once the above was diagnosed: *"are there actually solutions? A different
embedding model? A manual mode..."* — both were pursued, not just the manual one, since testing
showed they solve genuinely different problems and stack rather than compete.

**Why a general text model helps anyway, despite not fixing "speed"**: CLIP's text tower is
fundamentally a *caption* embedder — trained to match photos, not to compare sentences to each
other — which is a real mismatch for Words mode, where nothing being compared is ever a photo.
`Xenova/all-MiniLM-L6-v2` (a proper sentence-transformers model, contrastively fine-tuned
specifically so cosine similarity between two sentences means something) gives a measurably wider,
more usable raw similarity range without CLIP's severe anisotropy to begin with, and correctly
handles the exact cases CLIP needed centering to fix (dog/cat vs. dog/computer; leech/hairy
correctly negative) at least as well. It was made the **default and only** text-mode model, not an
opt-in toggle, on direct instruction — a toggle was scoped first, but "don't even make it an
option, just use that for text and CLIP for images" is simpler for a visitor and was already the
right call anyway: Photos mode *cannot* use the general model even as an option, since axis words
compared against photos must stay in CLIP's joint space, so the two modes were always going to
need different models regardless of whether Words mode's choice was configurable.

**Centering is still needed, but with its own separately-derived mean.** `GENERAL_TEXT_EMBED_MEAN`
(`text-embed-mean-general.js`) is a 384-dimension mean over ~180 diverse words, computed and
verified the same way `TEXT_EMBED_MEAN` was (see "Text embedding centering," above) — **not**
interchangeable with CLIP's mean; different model, different dimensionality, would silently
produce garbage (or a dimension-mismatch crash) if swapped. Verified before shipping: dog/cat 0.42
vs. dog/computer 0.03 (clean separation), leech/hairy -0.01 and fish/hairy -0.06 (correctly
negative) — centering helps this model too, for the same reason it helped CLIP.

**Diagnostic thresholds are calibrated per model, not shared**, because the two models' raw
similarity distributions aren't identical even after centering: antonym pairs (warm/cold,
loud/quiet) landed around 0.60 on the general model — higher than the equivalent CLIP-era
antonym clustering — while topic-noun and independent-trait pairs (ocean/mountain 0.19,
hot/furry 0.12) stayed low, same shape as CLIP's own calibration just shifted. `#axisSimNote`'s
top threshold (`collapseThreshold` in `recomputeAxes()`) is `0.4` in Words mode, `0.35` in Photos
mode (CLIP, unchanged from before); the middle "some noise is normal" cutoff (`0.15`) is shared,
since both models' "clearly distinct" pairs land in a similar low range. If either model or its
mean ever changes, re-run the calibration pairs in this section (and the ones under "Text
embedding centering") before trusting these numbers again.

**Implementation**: `embedTexts()` routes by `state.mode` — `'image'` uses CLIP (`model: 'clip'`
in the worker message, `clipTextCache`, `TEXT_EMBED_MEAN`), anything else uses the general model
(`model: 'general'` or omitted, `generalTextCache`, `GENERAL_TEXT_EMBED_MEAN`). Every caller
(axis words in `recomputeAxes()`, typed items in `addTextItem()`, the quick-similarity-check
widget) goes through this one function and gets the mode-appropriate model automatically — nothing
downstream needs to know or care which model actually ran. `worker.js`'s `ensureGeneralText()`
loads `Xenova/all-MiniLM-L6-v2` via `pipeline('feature-extraction', ..., { pooling: 'mean',
normalize: true })` rather than hand-rolled mean-pooling over raw token outputs — `pipeline`'s
built-in pooling is attention-mask-aware (correctly ignores padding tokens), which hand-rolling
would need to replicate exactly to avoid silently corrupting every embedding with padding noise.

## Manual override: dragging a point directly

The actual, reliable fix for the "speed"/"loud" class of failure above — not a smarter
computation (there isn't one), but letting the visitor's own judgment override the model's when
it's wrong. Direct instruction, considered alongside a proposed alternative (a 7-point Likert
selector per item) and resolved in favor of continuous free-drag: strictly more expressive (any
position in `[-1, 1]`, not one of 7 discrete rungs) while still trivially capable of landing on
any of the 7 gradients someone might have wanted, and it reuses the compass view's existing
click-to-select interaction rather than adding new per-item UI chrome.

**Mechanism**: `item.manualPos` (`{x, y}` or `null`). `normalizeGroup()` always prefers it over
the computed `nx`/`ny` for that item's final `pos` — but still computes that item's real
`rawX`/`rawY` and includes it in the min/max range used to place *everything else*, so a pinned
item keeps contributing honestly to the corpus range rather than distorting it or opting out
silently. Dragging is implemented as mousedown-hit-test / window-level mousemove / window-level
mouseup on `#plotCanvas` (window-level, not canvas-level, so a drag that leaves the canvas
mid-gesture still completes — same pattern as Afterimage's crop-box dragger and Ridgeline's
manual skyline override). A plain click with no movement (`dragMoved` stays `false`) still just
selects, exactly as before it existed — the existing click-to-select behavior is a special case
of a zero-distance drag, not a separate code path competing with this one.

**Compass view only**, gated by `if(state.view !== 'compass') return;` in the mousedown handler —
Network view's node positions are the same underlying `item.pos` (so a compass-view pin is
still visible there), but dragging a node in a distance-based graph layout doesn't have the same
obvious "this is where I think it belongs" semantics free-positioning has on the compass's
labeled axes, and wasn't asked for.

**Visual indicator**: a dashed amber ring around any pinned point (`drawCompass()`) — amber
specifically because it's the one color Vectis's cyan-dominated palette never otherwise uses, so
"this one's different" reads without a legend. Verified pixel-by-pixel after building it (same
discipline as the arc-position fixes elsewhere in this file), not just assumed correct from the
code — sampling the canvas at the ring's expected radius around a freshly-dragged point finds the
amber stroke; the same check with no drag finds nothing there.

**Clearing pins**: `clearManualPins()` resets every item's `manualPos` to `null`, called whenever
either axis word changes (`axisXInput`/`axisYInput`'s `change` handlers) — a hand-placed position
only means something relative to the axis it was placed on; silently carrying it over to a
newly-labeled axis would display a stale, no-longer-meaningful position as if it were still
current. Per-item, `resetItemPosition(id)` (wired to a "Reset to computed position" button in the
info panel, shown only when the selected item has a `manualPos`) clears just that one item and
re-runs the full `recomputeAxes()` — deliberately not a lighter-weight "just recompute this one
item's position," so there's exactly one code path that ever assigns a position, with no second,
slightly-different path for the unpinned case to drift out of sync with.

**Testing note**: verifying the amber ring actually renders hit the same `requestAnimationFrame`-
pauses-when-hidden issue documented elsewhere in this file (`document.visibilityState ===
'hidden'` even when a browser-automation tool reports the tab as active) — confirmed directly,
worked around with a temporary `window.__forceRender = () => render(performance.now())`, removed
before shipping. If you're testing this again, don't trust a "the ring isn't there" result without
checking `document.visibilityState` first.

## Two real CLIP quirks this surfaces, on purpose

- **Raw text-text cosine similarity used to run hot and cluster tight** — see the rescaling
  section above for the corpus-only fix and why `#spreadNote` is permanent rather than
  conditional, both still accurate. This bullet originally argued against "fixing" the clustering
  itself by subtracting a baseline, on the theory that doing so would be the kind of hidden
  per-item correction the spec's §6 constraint rules out. **That reasoning didn't hold up, and
  text embedding centering (see that section, above) does exactly this — subtracts a baseline —
  and is the actual fix.** The distinction that makes it spec-compliant rather than a violation:
  §6 rules out a *second*, hidden similarity computation that could disagree with what's plotted
  (e.g. comparing in the full embedding space while showing something else on screen). Centering
  isn't a second computation running alongside the first — it's a preprocessing step applied
  uniformly, upstream, to every text embedding before *the* similarity is computed once, the same
  category of thing as the worker already unit-normalizing embeddings or the corpus rescale
  transforming raw cosine into a plotted position. There's still exactly one number per item per
  axis, and it's still what's shown and plotted everywhere, consistently — nothing to disagree
  with. Don't re-litigate this as a §6 violation without re-reading this distinction.

  The info panel used to show each selected item's raw per-axis similarity numbers directly
  (`#infoRaw`) — removed on direct feedback ("just make the title '{label} Similarity' and get
  rid of the raw similarity stuff"), since the ranked list plus the always-visible
  `#axisSimNote` / `#spreadNote` diagnostics already make the same point without a per-item
  number dump. If this needs to be checkable per-item again, that's what to extend — don't
  resurrect `#infoRaw` itself, it was removed on purpose.
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
deliberately just a restatement of `item.pos`, the exact same numbers the compass plot uses — not
a new computation, so it can't drift from what's actually plotted. (The info panel used to show
this same kind of raw-number breakdown per selected item too, via `#infoRaw` — removed since,
see "Two real CLIP quirks" above; this table is now the only place the literal numbers live.)

Immediately after the table, a short paragraph makes clear this 2-axis view is a teaching
simplification, not how embeddings work in the wild: production systems use the full, uncurated
vector directly (hundreds of numbers) rather than two hand-picked, human-readable axes — added on
direct request ("a little blurb about how real embeddings aren't typically curated and often
have hundreds of dimensions"). Keep this near the table, since the table is what makes "only two
numbers" concrete enough for the caveat to land.

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

**The hero paragraph (in `<header class="hero">`, above everything else) is a separate piece of
copy from that trimmed intro**, and got rewritten for a different reason: it explained what the
tool *does* (pick two words, plot things) without ever saying what it's *for* — direct feedback
("we need a blurb at the top that explains what this is for, not just what it does"). It now
opens with the real-world hook (search/recommendation/classification systems all place things in
a similarity space; this is that trick shrunk to two dimensions you can see) before getting to
the mechanics. It also carried the site's original "chaotic"/"calm" example pairing — an antonym
pair, the exact thing "Choosing axis words" above documents as bad practice — so it picked up the
same "hot"/"furry", "kind"/"fast" independent-dimension examples used in the axis-word tips,
for consistency across the page. Keep both pieces of copy's examples in sync if either changes
again; having the hero pitch model the opposite of what the tips paragraph recommends undercuts
the tips.

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
button, confirm the word items disappear, **both** progress rows start (vision *and* language —
Photos mode now warms CLIP's text tower alongside the vision tower, not just vision) before
dropping anything, and both rows actually reach "ready" (the language-model row previously got
stuck on "Loading…" forever after switching modes — a real bug, fixed by making
`warmClipTextIfNeeded()` update that row's state instead of only `warmGeneralText()` doing so;
re-check this specifically if you touch either warm function again), drop in two different real
photos and confirm they land at distinct positions rather than both collapsing to the same point,
then switch back to Words mode and confirm the map is empty again (not a leftover mix of both
kinds) and "Reset to example" still repopulates the six word items.

Also: drag a plotted point somewhere else entirely and confirm it snaps to the cursor live, gets a
dashed amber ring, and the info panel shows a "Reset to computed position" button; click that
button and confirm the point returns to its original computed position exactly; change either
axis word afterward (with a fresh drag in place) and confirm the pin is gone and the point is back
to auto-placement under the new axis, not still sitting at the old manual position.

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
