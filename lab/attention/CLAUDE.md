# Attention

Type a sentence and watch real scaled dot-product attention — the mechanism at the core of
every modern language model — run on your own words, live: tokens, embeddings, Query/Key/Value,
softmax attention weights, a KV cache, and next-token prediction, then a downloadable "sentence
portrait" and a Zazzle hand-off. Built from a full spec plus a working first-pass prototype
handed off together (see "Where this started" below), then implemented end to end in this repo.

## Where this started

Two artifacts were handed off together: a Claude Docs spec ("Attention — Concept Lab Spec," a
full page-structure and stage-by-stage design document) and a separate working prototype
covering only Stage 1 (tokenize/embed/position) and Stage 3 (attention) — explicitly scoped in
the spec's own "Build plan" section as "a working first pass... since those two together already
deliver the hook and the core 'aha.' Stages 2, 4, 5, the portrait export, and Go Deeper are
scoped above but not yet built." Direct instruction: implement the *full* spec, "dynamic,
interactive," with icons, examples, demos, explanation, and a Zazzle link — not just the
prototype's narrower slice. Everything in this file describes decisions made closing that gap.

The prototype itself was a reasonable proof of concept but didn't follow this site's own
conventions — a fully bespoke color system and typography (`--violet`, `--coral`, Georgia
serif headings, a hand-rolled iOS-style toggle switch) with no link to `/theme.css` or `/nav.js`
at all. None of that shipped as-is: the underlying math (mulberry32, hash-seeded embeddings,
sinusoidal position encoding, scaled dot-product attention, softmax, the chord-diagram
visualization) was sound and carried over; every visual convention was rebuilt against
`theme.css`'s shared components (`.panel`, `.pill`, `.toggle-group`, `details.disclosure`,
`canvas.stage` patterns) the same way every other tool in this lab already does — see the root
`CLAUDE.md`'s standing instruction to rely on `theme.css` before writing new CSS.

**The spec left three open questions; all three were resolved here, not deferred:**
- *Real pretrained embeddings vs. fully synthetic ones* — stayed synthetic. See "Why the
  embeddings and projections are fixed, not trained" below for the actual reasoning, which
  turned out to matter more than the open question's own framing suggested.
- *Tokenizer* — a simplified suffix-splitting stand-in, not real BPE, not plain whitespace
  either. See "Tokenization" below.
- *How many heads in the main pipeline* — one. Multi-head comparison is Go Deeper's own reveal;
  showing more than one head in the main pipeline would spoil that reveal and contradict Go
  Deeper's own design constraint ("a light re-skin of a mechanic already built," which requires
  the main pipeline to have already established the single-head mechanic Go Deeper then extends).

## Architecture

`index.html` plus a vendored `qrcode.min.js` (the identical file already vendored in Tonus's,
Echo State's, and Inkling's own folders — see Echo State's CLAUDE.md for why it's local instead
of a CDN import). No build step, no bundler, no server, no external API calls, no training step
— every stage's math is plain inline `<script>` array arithmetic, matching this site's standing
rule of real, inspectable computation running entirely client-side. Links the two site-wide
shared files (`/nav.js`, `/theme.css`) like every other tool.

Accent is an emerald green (`--accent:#4fd17a`, `--accent-dim:#235c3d`) — distinct from every
other tool's accent (amber, teal, violet, cyan, rose, rust, periwinkle-blue; see each tool's own
CLAUDE.md for its own color). Query/Key/Value each additionally get their own fixed hue
(`--q-color` reuses the page accent since Query is the "active" lens Stage 4 traces attention
*from*; `--k-color` violet, `--v-color` amber) so the three offshoots in Stage 2 — and every KV
table cell downstream — read as three different *kinds* of thing, not three shades of one color.

## The math pipeline (`computePipeline`)

One function, `computePipeline(tokens, opts)`, is the single source of truth every stage reads
from: tokens → embeddings (+ optional sinusoidal position) → Query/Key/Value for one head →
scaled dot-product scores (optionally causally masked) → softmax → weighted blend of Values into
each token's contextualized output. Every stage below is a different *view* onto one of this
function's return fields (`baseEmb`, `finalEmb`, `Q`/`K`/`V`, `weights`, `output`) — nothing is
computed twice by two different code paths that could silently drift apart from each other.

`opts.headIdx` selects which of `HEADS[0..3]`'s fixed projection matrices to use (main pipeline
always uses head 0; Go Deeper's multi-head comparison is the only consumer of heads 1-3),
`opts.posOn` toggles whether position gets added before projection, and `opts.causal` masks out
every `j > i` score to `-Infinity` before softmax (Go Deeper's causal-masking tool is the only
place this is ever `true`).

## Why the embeddings and projections are fixed, not trained

This is the load-bearing decision in the whole tool, and it resolves the spec's own open
question about real vs. synthetic embeddings in a way the spec's framing didn't quite anticipate.

Every token's embedding is `tokenEmbedding(tok)` — a deterministic hash-seeded fingerprint
(`mulberry32(hashStr('emb:'+tok.toLowerCase()))`), not a lookup into any real pretrained table.
"bank" always gets the same 16 numbers; those numbers carry no relationship to "river," "loan,"
or anything else. The Query/Key/Value projection matrices (`HEADS[h].Wq/Wk/Wv`) are likewise
small, fixed, seeded-random matrices, generated once at page load and never updated.

**The reasoning isn't "training is out of scope, so approximate it" — it's that a real
embedding table wouldn't actually fix the thing that matters.** The spec's Stage 3 payoff line
("this is why 'bank' next to 'river' looks different from 'bank' next to 'loan'") implies real
semantic disambiguation. But the attention *pattern* a visitor actually watches is driven by the
Query/Key **projections**, not by the embeddings feeding into them — and those projections would
still be untrained, fixed, random numbers even with a real pretrained embedding table underneath
them. Shipping a real embedding asset would have added real weight (and a real "which words are
in the vocabulary" boundary to explain) for a benefit that doesn't survive to the part of the
pipeline a visitor is actually looking at in Stage 4. So the honest, load-bearing claim is
narrower than the spec's original payoff line, and the copy says so directly, in two places:

- **Inline, in Stage 4 itself** (`#attnHonesty`, always visible, not gated behind a disclosure):
  "the dot products, the softmax, and the blending are exactly the real operation... these
  Query/Key lenses are fixed, untrained numbers, not weights learned from billions of real
  sentences... the mechanism is authentic; the judgment behind it isn't."
- **In "How this actually works,"** at more length, including the specific reasoning above for
  *why* a real embedding table wouldn't have actually fixed the underlying issue.

This is the same discipline Vectis's CLAUDE.md documents for its own anisotropy/centering
investigation and Tonus's for Linear Regression's honestly-bad performance — don't claim a demo
proves something it structurally can't, and say so where the visitor can actually see it, not
only in a file they'll never read.

## Tokenization

Real BPE tokenizers learn their merge vocabulary from millions of documents. `splitSubword()` is
a hand-picked stand-in: split on whitespace, peel trailing punctuation into its own token
(unchanged from the prototype), then check a fixed suffix list and split the matched suffix into
its own token with a leading `##` — the same marker convention real WordPiece/BPE tokenizers use
for a continuation piece, so a split reads as recognizable rather than arbitrary.

**This went through one real, caught-before-shipping bug.** The first suffix list treated `ing`,
`ed`, `er`, `est`, `ly`, `s` all the same way, gated only by "word longer than 4 letters, stem at
least 3 letters" — which meant ordinary short words that merely *end* in a common letter sequence
split nonsensically: "river" → "riv" + "##er", "water" → "wat" + "##er". Caught directly, not
theorized, by reading the actual rendered Stage 1 tokens for the page's own default sentence.
Fixed by splitting the suffix list in two: longer, more distinctive suffixes (`tion`, `sion`,
`ness`, `ment`, `able`, `ible`, `ing`, `est`) keep a shorter minimum word length (6) and stem
length (4), while short, generic suffixes (`ed`, `er`, `ly`, `s`) — real suffixes, but far more
prone to false positives on ordinary short words — need a longer word (8+) and a longer remaining
stem (5+) before they're allowed to trigger at all. Verified directly afterward: "river" and
"water" no longer split, while "swimming" → "swimm"/"##ing" and "celebration" → "celebra"/"##tion"
still do. The trade this makes is deliberate — sometimes missing a real split is fine; a silly
one (like "riv"/"##er") actively undermines the "look, subword tokenization" teaching moment
instead of demonstrating it.

## Positional encoding

The one piece of math in this entire tool that's the *actual* published formula, not a
simplification of it — alternating sine/cosine at position-and-dimension-dependent frequencies,
exactly as in Vaswani et al. Copy says this explicitly ("this one *is* the real formula... no
shortcuts here") specifically because everything else on the page is qualified one way or
another, and a visitor reading carefully deserves to know which parts aren't simplified at all.

## Next-token prediction: reusing attention's own similarity math, not a fixed-vocabulary matrix

A real model's final layer projects its last hidden state through an unembedding matrix sized to
its entire vocabulary. That doesn't work cleanly here, because the candidate word list is
deliberately *dynamic* — it's `VOCAB_BASE` (roughly 150 hand-picked common English words) plus
whatever unique words are actually in the visitor's own sentence, deduplicated
(`candidateVocab()`), specifically so the model can plausibly "choose" to reuse a word the
visitor already typed. A fixed-size matrix keyed to array position can't accommodate a candidate
list whose membership changes with every sentence.

The fix: `Wu` (a fixed matrix, `QKV_DIM → EMB_DIM`) projects an attention output back into
embedding space, and each candidate word's score is `dot(that projected query, tokenEmbedding(word))
/ sqrt(EMB_DIM)` — **literally the same dot-product-similarity operation attention itself uses**,
just comparing against a candidate word list instead of other tokens in the sentence. Since
`tokenEmbedding()` already works for any string, this handles an arbitrary, sentence-dependent
candidate list with no size mismatch, and it's a more honest design than it might look: "predicting
the next word" and "attending to other tokens" are being presented as literally the same kind of
operation here, because — modulo the vocabulary being a candidate list instead of the rest of the
sentence — they structurally are.

**Why generated continuations read as word soup, and why that's not hidden.** Real fluency in a
trained model comes from `Wu` (and every other matrix) having learned real structure from
language. Here every matrix is random and fixed, so the ranking a visitor sees is a real softmax
over a real similarity computation, but the *specific* words that rank highly are close to
arbitrary. "How this actually works" says this directly rather than letting a visitor wonder why
the "AI" seems to pick bad words — the mechanism (project, compare, softmax, sample) is real; the
fluency a trained model has learned on top of it is not, and was never in scope for a page with
no server and no training step.

**Sampling vs. greedy decoding, without using either term.** `#autoPickCheck` genuinely changes
behavior, not just cosmetically: checked, "Generate next token" always takes the highest-ranked
candidate (`ranked[0]`); unchecked, it draws a real weighted random sample from the *full*
distribution (`weightedSample()`, not just the top-8 slice rendered in the bar chart — sampling
from a truncated distribution would silently exclude a real, if small, chance of a good sample).
Clicking a specific candidate row always picks that exact word regardless of the checkbox — that
path is "pick one yourself," a separate, more direct action than "generate."

## The KV cache and generation stepper (Stages 5-6, wired together)

The spec lists these as two stages (4: KV cache: 5: prediction) but describes them as one real
control flow — generate, watch the table grow, see the next prediction, repeat — so they share
one piece of state (`gen`) and one action (`genStep()`), rather than two disconnected buttons a
visitor has to click in a specific order to get a coherent result.

`gen.tokens` starts as a copy of the *main* pipeline's current tokens (`genReset()`, called
whenever the sentence input actually changes — `genSyncIfNeeded()` compares against
`gen.baseSentence` every render so Stage 5-6 always continues from what's currently typed, never
a stale earlier sentence) and grows by one real token per `genStep()` call. Each step recomputes
the *entire* sequence's Key/Value via `computePipeline` (head 0, position on) — genuinely
re-deriving every row, not just appending a precomputed one — because the whole pedagogical point
of the cache toggle is to visually distinguish "recompute everything" from "reuse what's already
there," and that distinction has to be real to demonstrate honestly.

**`cacheOn` only controls which rows visually flash, and a running counter, never the actual
output.** `renderKVTable(flashAll)` adds the `.flash` CSS class (a brief background-pulse
keyframe) to every row when `!cacheOn`, or only the newest row when `cacheOn` — verified directly
by scripting both paths and counting flashed rows (all N when off, exactly 1 when on). The
"redundant K/V computations avoided" counter increments by `gen.rows.length` (the row count
*before* the new token was added) on every cache-on step — a real count of how many rows a
naive from-scratch reimplementation would have redone at that step, not a timing measurement
(everything here is small enough to be instant either way, and the copy says so explicitly,
matching the spec's own caution not to let "cache off" imply a different, worse *output* rather
than purely a redundancy difference).

After `MAX_GEN_STEPS` (5) real generation steps, `#genDoneBanner` appears with a link to the
portrait section — the spec's own "loop-back" instruction, so this doesn't read as an
open-ended toy that undercuts the "30 minutes, steps 1-4" promise the spec makes for the rest of
the page.

## The sentence portrait

`renderPortrait()` redraws, from scratch, straight off the *current* pipeline state, every time
the sentence changes: the sentence as a wrapped title, the attention chord diagram (via
`drawAttentionCanvas`, a canvas port of the same geometry `renderAttentionSVG` uses for the
on-page SVG — two renderers, one shared geometry, so the exported portrait can never show a
different pattern than what's on screen), a strip of Stage 1's own embedding heatmap cells along
the bottom third, and a dense small-multiple render of the current KV cache rows (or, if no
generation has happened yet, the main sentence's own K/V) — "reads almost like circuitry,"
matching the spec's explicit aesthetic note and the same "real computation as texture" idea
Neural Viaduct's own thumbnail already established for this site. No separate export pipeline
that could drift out of sync with what a visitor actually saw — it's the same data, redrawn at
print resolution.

The small shirt-mockup preview (`#shirtPrint`, in the Zazzle section) uses the exact same
"render to an offscreen canvas, set a plain positioned `<div>`'s `background-image` from its
`toDataURL()`" technique Tonus's mug mockup already established — an earlier draft tried
embedding a live `<canvas>` inside an SVG via `<foreignObject>`, which is a real technique but an
untested, more fragile one on this site (nothing else here relies on `foreignObject`); switched
to the proven pattern before shipping rather than being the first tool to find out whether it
holds up.

**Sharing is just the sentence, URL-encoded.** Unlike Tonus's or Echo State's share links (which
pack a seed, model choice, and several hyperparameters into a byte-quantized payload), everything
in Attention is fully deterministic from the sentence text alone — every embedding, projection,
and attention weight the page will ever compute for a given string is already fixed before the
visitor types a single character of it. So the share link is just `?s=<encodeURIComponent(sentence)>`,
no packing scheme needed at all.

## Zazzle CTA — a real, individually verified template, not a guess

The spec's own product tier for this tool is a t-shirt (the "wearable" framing running through
the whole "printable output" section). A first draft used a fabricated-looking product ID that
turned out, on verification, to be a genuine 404 — caught before shipping by actually loading it.
The *next* candidate — reusing Vectis's own "magnet" product ID on the theory that a numeric ID
already live elsewhere on this site must be safe to reuse — turned out to resolve to a real,
live page, but for a **coffee mug**, not a magnet or a t-shirt (Zazzle's URL slug text is
apparently cosmetic; only the numeric ID actually selects the product). That means Vectis's own
"magnet" link may itself be mislabeled — worth a look independently of this tool, but out of
scope to fix here without being asked. Neither near-miss shipped. The actual link
(`personalized_one_of_a_kind_photo_collage_t_shirt-235429040115913948`) was found via a live
Zazzle search for "create your own t shirt," opened directly, and confirmed to be a real,
purchasable, genuinely customizable "upload your own image" t-shirt template (title, price, and
size/style options all present) before it was wired in — the same discipline every other
external link on this site follows, and the reason it took three attempts here instead of one.
Carries the standard `?rf=238054754631086278` ambassador param like every other Zazzle link on
this site.

## Go Deeper — four isolated replays, not four new pipelines

Collapsed by default (`<details class="disclosure advanced">`, no `open` — matching the majority
convention across this site's other tools' own advanced disclosures; Attention's most important
honesty caveat already lives inline in Stage 4, not gated behind this section, so there's no
strong case for reopening it the way Tonus's was on direct request). All four reuse
`computePipeline`/`renderAttentionSVG`/`renderTokenCards`/`nextTokenDistribution` with different
options rather than separate bespoke code paths, per the spec's own instruction that each one
should be "a light re-skin of a mechanic already built... no new heavy engineering."

- **Multi-head comparison**: a toggle-group (2/3/4 heads) renders that many small
  `renderAttentionSVG` panels side by side. Head 0's panel is deliberately labeled "same as
  Stage 4" — it's the literal same computation, not a coincidentally similar one, which is the
  whole point of introducing "heads" as "more of what you already saw," not a new concept.
- **Positional encoding, isolated**: two `renderTokenCards` calls side by side, `posOn: false`
  vs. `posOn: true`, on the current sentence — a focused replay of Stage 1's own toggle with
  nothing else changing, exactly as specced.
- **Causal masking**: one more `renderAttentionSVG`, fed `computePipeline(..., {causal: true/false})`.
  Verified directly that toggling it actually removes edges (61 rendered paths at a representative
  sentence with no mask, 26 with one) rather than just relabeling the same diagram.
  Deliberately **not** wired into the Stage 5-6 generation flow — generation is trivially causal
  already, since future tokens don't exist yet at each step; this toggle exists to show what an
  explicit mask does to a *fixed, fully-typed* sentence, which is a different (and equally real)
  question.
- **Temperature / sampling**: a slider re-runs `nextTokenDistribution` at the chosen temperature
  against the *main* pipeline's current last-token output (not the generation stepper's state —
  this is meant as a standalone "what does this knob do" exploration, not tied to an in-progress
  generation run). Verified the direction is correct: at temperature 0.3 the top candidate's
  share was consistently higher (sharper) than at 2.0 (flatter, closer to uniform across ~150
  candidates) for the same sentence.

## Testing changes

No test suite — static page. Verify via a local static server (root-relative `/nav.js` and
`/theme.css` mean `file://` won't pick them up). Golden path: confirm the page opens with the
default sentence ("the river bank was steep near the old bridge") and the hero canvas, Stage 1
token strips, Stage 2 Q/K/V strips, and Stage 4's attention diagram all render with no console
errors → type a new sentence and confirm every stage updates live within the debounce window, no
button required → click an example pill and confirm the sentence, every stage, and the pill's own
`.selected` state all update together → flip Stage 1's position toggle and confirm the strip
colors visibly shift (position vectors are additive, so this should never be a no-op for a
non-trivial sentence) → click a word in Stage 4's diagram and confirm the spotlight dims every
other connection and the "attending most to" note names real, correctly-ranked neighbors →
confirm the inline honesty note (`#attnHonesty`) is visible without expanding anything → click
"Generate next token" repeatedly and confirm the KV table grows by exactly one row per click,
Stage 6's bar chart re-ranks against the new last token each time, and the done banner appears
at exactly 5 steps, not before → toggle cache off and confirm *every* row gets the `.flash` class
on the next generate; toggle back on and confirm only the newest row does; confirm the "redundant
computations avoided" counter only increments while cache is on → click "Reset generation" and
confirm the table collapses back to the current sentence's own tokens → open "Go deeper" and
confirm the multi-head grid changes panel count when the toggle changes, the positional-encoding
comparison shows two genuinely different strip patterns for the same tokens, the causal-mask
toggle visibly removes edges rather than just relabeling, and the temperature slider's top-candidate
percentage moves in the correct direction at both ends → download the portrait PNG and confirm
it's a real, non-trivial file whose attention pattern matches what Stage 4 currently shows, not a
stale render from an earlier sentence → copy the share link, open it in a fresh tab, and confirm
the exact sentence (and therefore every downstream stage) reproduces exactly → click "Put it on a
shirt" and confirm it's still resolving to a real, live, purchasable t-shirt template (Zazzle
product pages do occasionally get retired — re-verify by loading the link directly, the same way
it was originally checked, don't just trust that the URL still looks right) → check mobile width
specifically for the example pills (they wrap their own long sentence text onto multiple lines
within the pill shape rather than overflowing the viewport — confirmed directly, not just assumed
from `flex-wrap`) and for the head-comparison grid (should stack to one column on a narrow screen
via `.head-panel`'s `flex: 1 1 220px`).
