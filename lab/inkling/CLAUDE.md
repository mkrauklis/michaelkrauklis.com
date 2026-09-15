# Inkling

A real 64→18→26 neural network — plain feedforward, trained by real backpropagation, no library,
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

## The hero canvas (`drawHeroInkling`) — a decorative illustration, not a live computation

Direct request, matching the pattern other tools already use: *"create a visual in the top right
when you actually open Inkling that shows a hand-drawn letter going into our neural architecture
and creating the identified letter."* Converted the hero from a single `<div class="wrap">` to
the shared `<div class="wrap hero-grid">` layout (`#heroCanvas`, sized and styled entirely by
`theme.css`'s shared rule — nothing page-specific needed there), matching Echo State's and Neural
Viaduct's own hero-grid headers.

**This is a fixed illustration, not a rendering of any real forward pass** — the same choice
Neural Viaduct made for its own hero (a decorative arc over a line of dots, see that tool's
CLAUDE.md), for the same reason: the page deliberately opens with no letter drawn and a freshly
randomized, untrained network (see "The page opens genuinely blank" below), so there is nothing
*real* yet worth animating. `HERO_LETTER` is a hand-authored 8×8 block "A" (matching `GRID`, so it
reads as this tool's actual input shape, not an arbitrary size), fed through hand-drawn fan-out
lines to 7 illustrative "hidden" dots and on to 5 illustrative "output" dots, one of which glows —
`HN`/`ON` are deliberately *not* `HID`/`OUT`; showing all 18/26 at this canvas's small size would
be unreadable, and the point is to illustrate the mechanism, not to be a literal miniature of the
real diagram. Everything about *how* it's colored, though, is real and shared, not reinvented:
`divergingColor`/`ACCENT_RGB`/`COOL_RGB` for the hidden dots and forward connections (the same
rose/blue-or-colorblind-orange/blue convention every other real diagram on this page uses) and
`C_GOOD` for the one glowing "this is the answer" output dot (the same green used everywhere else
on the page for a correct prediction). A visitor who scrolls down and sees the real stage
animation or network diagram is looking at the same visual language they just saw in the hero,
not a different one.

Reads its own `canvas.clientWidth`/`clientHeight` at draw time (the Neural Viaduct `drawHero()`
pattern) rather than a fixed size, since `#heroCanvas` is `width:100%` and collapses to a single
column under 800px (`theme.css`'s `hero-grid` media query) — redrawn on `window.resize` and once
more from the colorblind toggle's handler (alongside `refreshLookInside`/`renderArchDiagram`/
`idleStage`) so it recolors in step with every other real diagram on the page rather than being
the one thing left showing the old palette.

## The network (`forward`, `forwardAndGrad`, `applyGrad`, `runEpochs`)

Deliberately the plainest possible architecture, stated directly in the "How this actually works"
copy: `64 pixels → 18 hidden (tanh) → 26 output (softmax, A–Z)`. Forward pass:
`a1 = tanh(W1x + b1)`, `ŷ = softmax(W2·a1 + b2)`. Loss is cross-entropy against whatever letter
you picked. Backward pass is the textbook chain rule: `δ2 = ŷ − y` (softmax+cross-entropy's
famously simple combined gradient), `δ1 = (W2ᵀδ2) ⊙ (1 − a1²)` (undoing tanh), then every `W`/`b`
nudged by `−η·∂L`. `forwardAndGrad()` returns both the activations *and* the gradients from one
pass — the stage animation (below) visualizes exactly this returned object, not a separately
recomputed one, so what's drawn on screen is guaranteed to match what `applyGrad()` actually
applies a moment later.

Pressing Train does two distinct things, not one: `forwardAndGrad()` + the stage animation runs
once, immediately, on your new example specifically (so the animation always shows *this*
drawing's real forward/backward pass) — then, after the animation finishes, `runEpochs(24, 0.45)`
re-shuffles and re-trains on **the entire accumulated dataset**, not just the new example. This is
why earlier letters don't get forgotten as new ones arrive: every Train click is a fresh, full
mini-training run over everything taught so far, not an incremental single-example update layered
on top of the last. `dataset` and `counts` grow monotonically until Reset. Each real example also
adds `AUG_COUNT` (4) elastically-jittered synthetic copies of itself to `dataset` at the same time
(see "Real hand tremor is non-rigid" below) — `counts` only reflects real taught examples, but
`dataset.length` (and therefore every epoch's real workload) is 5x that.

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

## The real fix for real handwriting: input centering, not more hyperparameter tuning

**Reported directly, later, and much more seriously**: real human handwriting — "a dozen examples
of each letter," trying to teach the word MIKE — simply wouldn't train reliably. Worse: *"if
anything it seems to get worse with more examples."* That symptom is the opposite of ordinary
underfitting (more data should never make a healthy classifier worse) and pointed at something
structurally wrong, not just an undertuned hyperparameter.

**Root cause, found by reading `getGrid()` again, not by tuning**: it squashed the *entire* source
canvas down to the 10×10 grid — `drawImage(canvas, 0, 0, dW, dH)` with no source rectangle — with
no cropping, no centering, no scale normalization. Two drawings of the same letter at a different
size or position on the pad produced two completely different 10×10 grids, because a plain fully-
connected network (no convolution, nothing translation- or scale-invariant) has no way to recognize
"the same shape, somewhere else." Every additional real, inconsistently-placed example of a letter
therefore added *contradictory* raw-pixel signal for the network to reconcile, not confirming
signal — which is exactly why more examples made things worse, not better. The clean, consistently-
centered strokes used to validate the weight-decay fix above never exposed this, because they were
always drawn at the same size and position on purpose.

**The fix**: `getGrid()` now finds the ink's actual bounding box on the *full-resolution* source
canvas first (a real per-pixel luminance scan, threshold `0.12` at the time — raised to `0.28` later;
see "Thin strokes were being erased by low-quality downsampling" further down for why), then crops a
padded square (40% margin) centered on that bounding box — not the whole canvas — before downsampling to 10×10,
falling back to the whole canvas only if nothing was drawn at all. This is the same idea MNIST-style
datasets bake in by construction (digits pre-centered and normalized before a classifier ever sees
them); a plain FC network here needs the same treatment done live, since nothing normalizes a
visitor's raw drawing for it otherwise. Because `getGrid()` lives inside the one shared
`makeScratchpad()` used by both the teach pad and every word pad, this applies uniformly to training
data and decode-time input alike — no separate code path to keep in sync.

**Weight decay (see the section above) then had to be reverted — `WEIGHT_DECAY = 0` now, not
`0.02`.** Verified directly, not assumed: a temporary debug harness (same pattern as before, removed
before shipping) trained synthetic M/I/K/E strokes with realistic position/scale jitter (a stand-in
for real handwriting's natural inconsistency) and tested genuine held-out generalization — fresh
jittered examples never seen during training, not just re-scoring the training set. With centering
alone, `0.02` decay generalized fine for two letters but confidently misclassified others (K read as
E with ~0% true confidence in one run); dropping decay to `0` and re-running the identical sequence
fixed it completely — every letter, high confidence, repeatably. Decay had been tuned against tiny
(7-10 example) synthetic sequences and genuinely fixed the narrower bug it targeted, but it was
never validated against a larger, more realistic dataset, and it turned out to be actively capping
the weight magnitudes a real, harder multi-letter classification task needs to converge well. The
original catastrophic-collapse scenario was re-tested with decay at `0` and centering in place, and
did **not** reoccur beyond the one, unavoidable state any classifier is in after its very first
training example (everything reads as that one letter, which recovers within the next couple of
Train clicks) — meaning centering was the real fix for *both* bugs, and decay was papering over a
symptom of the same underlying problem rather than addressing it.

**Confirmed end-to-end in the real UI, not just synthetically**: drew genuine, imperfectly
positioned/sized letters (M, I, K, E) by hand via real mouse drags, at deliberately different sizes
and screen positions each time — a fresh, differently-drawn "M" was misclassified after 1-2 trained
examples, then correctly recognized after a 3rd real example was added, the expected direction of
improvement finally working as it should. If you touch `getGrid()`'s cropping math or `WEIGHT_DECAY`
again, re-validate with genuinely varied (not just clean, consistently-placed) synthetic strokes and
a real held-out generalization check — training-set self-confidence alone hid this exact bug for
as long as it went undetected.

## Thin strokes were being erased by low-quality downsampling — the actual dominant bug (`imageSmoothingQuality`, ink threshold 0.12→0.28)

**Reported again after the augmentation fix above, with real numbers**: taught M/I/K/E with 5-6 real
examples *each* (well past anything the augmentation testing above required), and a fresh "MIKE"
still decoded wrong (M correct, I/K/E confused with each other). That result flatly contradicted
the augmentation fix's own validation, which predicted good results by 3 real examples per letter.
The user's own guess at the cause — *"would anti-aliasing possibly help?"* — pointed straight at it
and turned out to be the real, dominant bug this whole investigation had been missing.

**Root cause, found by extracting the exact grid `getGrid()` produces for a real drawn "I" and
looking at the actual numbers, not by reasoning about the code**: two compounding bugs in the same
function, both invisible unless you inspect raw pixel output.

1. **The bounding-box scan's ink threshold (`1-lum > 0.12`) was too close to the pad's own
   decorative background.** `paintPaper()` draws faint gridlines (`rgba(30,20,10,.08)` over
   `C_PAPER`) on every scratchpad, including every word pad — measured directly, that blend comes
   out to `1-lum ≈ 0.131`, just *above* the old 0.12 threshold. For a letter with substantial ink
   coverage this noise is negligible next to the real bounding box. For a sparse letter like "I" (a
   single thin stroke), the gridlines were the *only* thing touching all four edges of the canvas,
   so they won the bounding-box computation outright: `minX`/`maxX`/`minY`/`maxY` landed at the
   canvas's own edges (confirmed directly: bbox `[0,0,74,74]` on a 75×75 canvas for a letter whose
   real ink occupied a ~2px-wide strip) instead of tightly around the actual ink.
2. **Even with a correct bounding box, `drawImage`'s default resize quality drops thin content
   entirely.** Chromium's default `imageSmoothingQuality` is `"low"`; minifying a ~100px-wide crop
   down to 8px in one `drawImage` call at that quality can lose a stroke only 1-2 physical pixels
   wide completely. Confirmed directly, not assumed: extracting the grid twice from the identical
   canvas, once with default smoothing and once with `imageSmoothingQuality:'high'` forced —
   default came back as *exactly* the flat paper value in all 64 cells (zero trace of the ink
   whatsoever); high-quality came back with a clearly readable vertical-stroke gradient in the
   correct two columns.

Both bugs hit thin/sparse letters (I, K's diagonals, E's strokes) far harder than dense ones (M,
whose four overlapping strokes gave it enough raw ink mass to survive both problems reasonably
intact) — which is exactly the failure pattern reported (M fine, I/K/E confused with each other)
and exactly why throwing more real examples or more augmentation at it never fully fixed it: those
levers can't recover information the pixel-extraction step had already destroyed before training
ever saw it. Every letter's *training* examples were degrading through this same broken pipeline
too, not just decode-time input, so the network was matching one degraded, noise-like signature
against another rather than learning real shape differences for these letters.

**The fix**: raised the ink-detection threshold from `0.12` to `0.28` (clears the ~0.131 gridline
noise with a wide margin, still far below any real ink's 0.6+ values) and explicitly set
`octx.imageSmoothingEnabled = true; octx.imageSmoothingQuality = 'high';` before the final
`drawImage` downsample. Both lines live in the one shared `getGrid()`, so the fix applies uniformly
to the teach pad and all 8 word pads, training and decode-time input alike, same as every other fix
to this function.

**Verified with real numbers, not just "looks fixed"**: re-ran the exact same test that exposed the
bug — M/I/K/E, 3 real examples each (the same count the augmentation fix's own validation used) —
and got a fresh "MIKE" fully correct, twice in a row. Re-ran at the harder 1-real-example-per-letter
case too (this tool's hardest realistic scenario) and got 3/4 correct (only E missed), a real jump
over the 2/4 this same case scored before this fix. This is very likely the fix that should have
been found first — the augmentation and resolution work above are still real, valid improvements
(non-rigid tremor is still a real, separate source of variance this doesn't address), but they were
fighting a much smaller problem than this one while this pixel-destroying bug sat underneath both
of them the entire time.

**If `getGrid()`'s crop/threshold/downsample logic is touched again**: extract and print the actual
grid values for a deliberately thin/sparse test letter (not just a thick one like M or H) before
trusting a change — this bug was invisible to every prior round of testing in this file specifically
because every prior synthetic test happened to use letters with enough ink mass to survive it. A
diagnostic worth keeping in your back pocket: render the grid as ASCII art (`#`/`+`/`.`/` ` by value
threshold) directly from the real `getGrid()` output (not a reimplementation — this investigation's
first attempt at a "clean" reimplementation quietly diverged from the real function and gave a false
signal) to eyeball whether a shape is actually surviving the pipeline.

## Real hand tremor is non-rigid — centering wasn't the whole fix (`elasticJitter`, `AUG_COUNT`, `GRID=8`)

**Reported again after the centering fix above shipped**: still not working, on a fresh attempt to
teach a whole word from real handwriting. The user's own diagnosis was correct and is worth quoting
because it's the actual mechanism: *"when I'm actually writing I'm less consistent than you're
expecting... that's a big part of the problem the kernels in CNNs solve."*

**Why centering wasn't enough**: bounding-box centering (previous section) corrects *rigid*
variance — a letter drawn bigger, smaller, or off to one side. It does nothing for *non-rigid*
variance — the same hand drawing the same letter twice with the stroke's exact path wobbling a
little differently each time, pixel by pixel. A plain fully-connected network has no built-in
tolerance for a shape landing on slightly different pixels (that's exactly the local-shift
tolerance a CNN's convolutional kernels provide and this network doesn't have). My own prior
validation of the centering fix missed this: the synthetic test strokes used *rigid* jitter only
(translate/scale/rotate the same fixed path), which centering directly corrects — so that test
could never have caught a non-rigid problem in the first place. Caught here by building a second,
more realistic synthetic test that jitters each stroke *segment* independently (simulating real
tremor along the path, not just moving the whole letter), which reproduced real, meaningful
accuracy loss even with centering already in place.

**What was tried and measured** (self-contained parallel test network, same math as the real app,
genuine held-out generalization checks — never training on the exact examples being tested for
accuracy):
- Lower resolution alone (8×8, 6×6): 8×8 gave a small real improvement; 6×6 did not help (too
  coarse, loses shape detail).
- Blurring the input before downsampling: a small real improvement, roughly comparable to 8×8 alone.
- **Data augmentation** — training each real drawn example alongside several synthetically-perturbed
  copies of itself — was the clearly largest lever. The first pass at this measured augmentation by
  having the test's synthetic letter *generator* redraw fresh independent examples, which isn't a
  technique available in the shipped app (there's no "true" generator for a real visitor's drawing,
  only the one grid actually captured) — an important distinction, since that version of the result
  would have been fake progress if shipped as-is.
- Re-tested with the only mechanism actually available in production: **elastic distortion** —
  warping the one real captured grid through a smooth random per-pixel displacement field (the same
  augmentation technique Simard et al. used for MNIST in 2003; not invented for this tool). This is
  what's shipped (`elasticJitter`/`smoothField`, defined just above `doTrain`).

**Honest numbers, not a "solved" claim**: under the realistic non-rigid noise simulation, baseline
(no augmentation) held-out accuracy averaged ~78% across repeated runs; with elastic-distortion
augmentation (`AUG_COUNT=4` copies per real example, `ELASTIC_MAG=0.8`, `ELASTIC_COARSE=3`) it
averaged ~88% across repeated runs at the same noise level. A single lucky run hit a perfect score
first — repeating it 4 more times immediately (35, 31, 36, 37 out of 40, not 40/40 again) showed
real run-to-run variance and that the perfect run was not the reliable expected outcome. That
discipline — never trust one run, especially a suspiciously good one — should carry forward to any
future tuning here. ~88% is a real, meaningful improvement over ~78%, not a complete fix; a
visitor's actual handwriting may still occasionally need a couple of retrained examples of a
letter that keeps getting misread, same as it did before.

**`GRID` dropped from 10 to 8 (`GRIDN` 100 → 64) in the same change**, directly answering the
user's resolution question: tested and found to not hurt (a coarser grid tested as good or
slightly *better* than 10×10 once combined with augmentation, likely because there's less exact-
pixel detail for tremor to disagree about) while also shrinking the QR payload (~1552 → ~1120
base64 characters at this dataset size) as a real secondary benefit, not just a hoped-for one.
`EPOCHS` raised 16 → 24 alongside augmentation, since the combination tested better than either
change alone (each real example now trains alongside 4 synthetic copies, so more epochs over that
larger effective dataset kept helping instead of plateauing early).

**If `GRID` ever changes again**: check for stale hardcoded references to the old size — this pass
found and fixed three (`drawStageLabels`'s and the network-diagram canvas's `'100 pixels'` labels,
now `GRIDN+' pixels'`; the QR ledger note, which had hardcoded the literal string `'10×'` instead of
deriving both dimensions from `GRID`) plus static prose in "how this actually works" and the
look-inside tiles paragraph that had to be hand-updated since they're not JS-driven. None of these
threw errors or looked broken at a glance — they just quietly said "100 pixels" and "10×10" next to
a network that was actually 64 pixels and 8×8, which is exactly the kind of drift that's easy to
ship unnoticed. Grep the file for `GRID`, `GRIDN`, and any literal `10` or `100` near them before
trusting a future resolution change is fully applied.

**Verified end-to-end in the real UI**, not just in the synthetic test harness: drew real K, H, and
A strokes by hand via mouse drags (deliberately imperfect, not identical repeats), trained each
through the actual wizard, and confirmed `dataset.length` grew by 5 per Train click (1 real + 4
augmented, matching `AUG_COUNT`), the stage animation and look-inside views rendered correctly at
the new 8×8 resolution with no console errors, and the word decode picked up each newly-taught
letter correctly while leaving untaught letters to fall back to whatever letter's shape they're
nearest to — the same behavior as before, just now backed by a genuinely more tremor-tolerant
network.

## The page opens genuinely blank — no seed training, at load or on Reset

`SEED` (six letters, H/I/A/B/C/T, hand-authored as literal 10×10 ASCII-art patterns) and
`patternToGrid()` still exist, but **nothing calls `seedTrain()` anymore, including at page load.**
An earlier version pre-trained those six letters (12 examples, 60 epochs) before a visitor ever saw
the page, specifically so the letter chips and "look inside" tiles never looked empty on first
paint. That was reversed on direct feedback: *"let's start with a random network, no training. It's
not intuitive that someone needs to go back and reset to get their own network."* A first-time
visitor's very first "look inside" or word decode is now genuinely *their* network, from a random
`initNet()`, not a demo they'd have to clear first. `SEED`/`patternToGrid` are only still used as
the fallback path in `seedWord()`, for rendering the pre-loaded "HAT" word pads on letters without a
hand-authored `HANDWRITTEN_STROKES` entry. Never actually exercised today — H/A/T all have
`HANDWRITTEN_STROKES` entries — but if a future word ever needs this fallback for a `SEED` letter,
note `patternToGrid()` loops `r<GRID, c<GRID` (now 8) against `SEED`'s still-10-wide/10-tall
strings, so it'd silently render only the pattern's top-left 8×8 corner. Fine for display-only use,
but worth knowing before assuming the fallback draws the whole authored shape.

**`Reset network` and page load now do the exact same thing** (`initNet()` alone, nothing else) —
this also fixed a real, separately-reported bug where Reset used to call `initNet(); seedTrain();`,
silently restoring the same six-letter demo instead of genuinely blanking the network. There's now
only one "blank" state in the whole tool, not two different ones depending on how you got there.

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

**Clicking a chip on the main page also opens this wizard, pre-aimed at that letter, and now skips
the letter-picker step entirely** (`openExpander(pad, 'Draw the letter "'+letter+'"',
{wizard:true, presetLetter:letter})`). The modal's header names the letter, and — direct follow-up
feedback after the preset-chip flow first shipped still showing a redundant confirmation screen:
*"they shouldn't have to confirm it... it should go right to train instead of confirming the
letter"* — `expandPreset` (`true` whenever `opts.presetLetter` was given) makes the draw step's own
action button read "Train ↳" instead of "Next →", and clicking it commits the ink and calls
`doTrain(wizardLetter)` directly, closing the modal without ever showing `#letterStepBody` at all.
The generic "tap to draw a letter and teach it" trigger (no letter known ahead of time) still goes
through the original two-step Next → pick-a-letter → Train flow, since there's a real choice to
make there that a chip click has already made. `$('expandFoot')`'s instructional text also branches
on `expandPreset` ("draw the letter above, then Train" vs. the generic "draw, then Next") so the
modal's own copy matches which flow is actually active.

**`getGrid()`'s downsample is a real resample, not a fake one**: it draws the full-resolution
canvas into an 8×8 offscreen canvas via `drawImage` (letting the browser's own image scaling do
the area-averaging), then reads back `getImageData` and converts luminance to ink density
(`1 − luminance`, clamped). This is genuine pixel data feeding the network, not a synthetic
stand-in — the same "verify against the real thing" discipline as every other tool here.

**The teach wizard used to silently carry over old ink on open — a real, reported bug**: *"when we
open the drawing modal while training, the old letter remains. I want a fresh board every time I
click the button or letter that opens the modal."* The carry-over line right after `paintPaper` in
`openExpander` (`if (padApi.isDirty()) expandCtx.drawImage(padApi.canvasEl, ...)`) exists so
*re-opening a word pad* to touch up a letter doesn't discard what's already drawn there — genuinely
useful in that mode. But `openExpander(pad, ...)` is the same function for the teach wizard, and the
teach `pad`'s dirty flag can still be set from whatever was last drawn into it, so the exact same
line was quietly reappearing the previous letter every time the wizard reopened, regardless of
whether it came from the generic trigger or a preset-letter chip. Fixed with a single
`if (expandWizard) padApi.clear();` right before that block — the teach wizard now always starts
from a genuinely blank board, while word-pad mode (`expandWizard` false) is completely unaffected
and still carries ink over as before. Verified both open paths (`#openTeachBtn` and a letter chip)
land on a blank canvas after a full train cycle, and confirmed a word pad's existing ink still
carries over correctly on reopen (checked real pixel data both times, not just visually).

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
a few hundred lines per hidden neuron pass, not the 64×18 worst case a fully-inked square would
produce — verified to stay smooth in testing, not just assumed cheap.

**Three explicit named phases** replace the old four overlapping opacity ramps: `p1` (250-800ms,
pixels → hidden), `p2` (800-1300ms, hidden → output, unchanged from the original design — still
only draws edges to the true label and the current prediction, not all 26 outputs), `p3`
(1300-2000ms total, backward — output→hidden via the real `dz2` gradient, *and* hidden→the same
active pixels via the real `dz1[h]*W1[h][k]` gradient, so the correction visibly retraces the exact
path the forward pass just lit up, all the way back to the pixels). Captions change at the same
boundaries the phases use — if you retune the timing, update both together or the text will desync
from what's on screen, the same trap the original version already had to watch for.

**Second redesign: lines → packets** (`drawPacket`). Direct request: *"rather than lines, we have
packets that go from the letters to the hidden layer neurons to the letters that were predicted...
back props... via packets as well"* — the "network packets of information traveling through the
network" framing. Every one of the four connection blocks above (pixel→hidden, hidden→output,
output→hidden, hidden→pixel) draws the exact same real per-connection number it always did
(`fwdC`/`contribs`/`dz2`/`bwdC`, unchanged), just through `drawPacket(ctx, from, to, t, size, fill)`
— a small filled square at `lerp(from, to, t)` — instead of a `stroke()`'d line whose *opacity*
carried the phase progress. `t` is the same phase-progress value (`p1`, `p2`, `p3a`, or `p3b` — see
below) every packet in that phase shares, so every in-flight packet moves in lockstep: a visible
wave arriving together, not independent timers. A packet is only drawn while `0 < p < 1` (in
flight); at `p===1` it's arrived and been
absorbed into the neuron it was headed to, so it simply stops being drawn rather than sitting there
at full brightness the way the old lines did once their phase finished — packets are inherently
transient, unlike a line. Forward packets keep the same excite/inhibit accent/cool coloring by
sign; backward packets keep the same neutral `C_INK` coloring the old dashed backprop lines used
(this is the correction signal itself, not a signed weight — the legend already draws this
distinction and didn't need to change). The marching-dashes (`setLineDash`/`lineDashOffset`) that
used to convey backprop's direction are gone entirely — a packet moving from output to hidden to
pixel already shows direction on its own, nothing dashed needs to fake it.

Verifying this one is awkward in this specific automation environment and worth knowing before
trying again: `document.visibilityState` reports `"hidden"` here regardless of `tabs_select`, and
`requestAnimationFrame` never fires at all while hidden — so `frame()` only ever runs once, forced
by a `computer` screenshot call, at whatever `el` has accumulated by the time that screenshot's
round-trip completes (usually already past every phase, landing on the final resting frame). That
made it impossible to visually catch a mid-flight packet through this tool; correctness here rests
on the transformation being mechanical (same numbers, same gating, `stroke()` swapped for
`drawPacket()`) plus a clean, error-free idle end-state after several full train cycles. A real
browser tab (this bug doesn't exist outside automation) will show the actual traveling packets
smoothly — check there if `drawPacket`'s math is ever touched again.

**Third redesign: backward is two sequential sub-phases, not one simultaneous one** (`p3a`, `p3b`).
Direct follow-up after the packets shipped: *"I want the backprop from the letters to the neurons
to happen before the packets going from the neurons to the pixels"* — the single `p3` window
(1300-1950ms) used to gate both the output→hidden packets and the hidden→pixel packets at once, so
both legs of backprop arrived on screen simultaneously. That's not just a visual preference to
accommodate — it's actually *more* correct: the chain rule computes `dz2` (the output layer's
error) and only *then* derives `dz1` (`dz1[h] = (W2ᵀdz2) ⊙ (1-a1²)`, a real data dependency on
`dz2` already existing) before there's anything meaningful to send on toward the pixels. `p3` split
into `p3a = clamp((el-1300)/325,0,1)` (output→hidden) and `p3b = clamp((el-1625)/325,0,1)`
(hidden→pixel, starting only once `p3a` reaches 1) — same total backward window, same 650ms, just
sequenced instead of overlapped. The two backprop captions (previously one shared "backprop → the
correction flows back through the same connections, all the way to the pixels" for the whole
window) split the same way, at the same `el>1300 && el<=1625` / `el>1625` boundary as the packet
timing, so the caption never claims a leg is happening before its packets actually appear.

**Persistent column labels** (`drawStageLabels()`, called from both `idleStage()` and every
animation frame) directly state the architecture on the diagram itself: "`GRIDN` pixels" (64 today,
derived live rather than hardcoded — see "Real hand tremor is non-rigid" above for why that matters),
"18 hidden neurons (tanh) — one hidden layer —", "26 outputs (softmax, A–Z) / argmax picks the
letter" — added
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

**A second round went further, on direct follow-up feedback**: *"there's still tons of empty
space... the tap to draw is huge, reset network could be on the same line."* The trigger button, the
`trainStatus` count, and Reset used to sit in a two-column `.row` (chips-and-label on the left,
status-and-Reset stacked on the right) — since the right column was much shorter than the chip grid
on the left, the row's height was set by the taller column, leaving the shorter one surrounded by
dead space. Replaced with `.teach-header`, one flex row holding the trigger (`flex:1 1 240px`),
`trainStatus`, and the Reset button/toast together, wrapping naturally on narrow screens instead of
reserving a whole second column. `.draw-trigger` itself also shrank again (icon 36px→28px, padding
`.7rem 1.1rem`→`.5rem .9rem`, font `1.05rem`→`.95rem`) so it reads as one control among several in
that row rather than a dominant full-width block above them. The chip grid's own label moved below
`.teach-header`, directly above the chips, since it no longer needs to share a row with anything.

**A third round, on further direct feedback, removed content rather than just tightening spacing.**
The chip-grid label ("examples taught so far — tap a letter to teach it directly") was deleted
outright — `trainStatus` in `.teach-header` already states the count, and the chips' own click
behavior speaks for itself. The idle-state stage caption ("Draw a letter to watch it learn.") was
also removed — `#stageCaption` now starts empty in the HTML and `resetBtn`'s handler explicitly
clears it back to `''` too (it previously only ever got a real value from `animateSequence()`'s own
captions, so without that explicit reset it would otherwise keep showing whatever the *last*
animation's caption said, e.g. "backprop → ..." well after a Reset — a real, if minor, staleness
bug this cleanup also happened to catch). `.chips`' own chip width shrank again (27px→24px) and its
`margin-top` tightened (`.7rem`→`.5rem`) now that there's no label above it to space away from —
at the page's own `.wrap` max-width (900px), all 26 chips now fit on a single row instead of
wrapping to two. The word-decode readout and the colorblind-friendly toggle were merged onto one
line (`display:flex; justify-content:space-between`) instead of two separate stacked lines, on
direct request to place the toggle "to the right of 'reads today's word'".

**The stage canvas's own bottom margin was also trimmed** (`STAGE_H` 200→160, with
`INBOX`/`hiddenPos`/`outputPos`/`OUTPUT_LABEL_SPLIT_Y` all rescaled together) — the diagram's own
drawn content bottomed out around y≈132 inside a 200px-tall canvas, leaving a real ~50-68px band of
unused dark space below it that read as "padding at the bottom of the neural network card." The new
geometry keeps the lowest content (the second output row's letter labels) within about 30px of
`STAGE_H`, rather than leaving a visibly empty region between the diagram and the card's own border.

## The compact readout shows the actual handwriting, not just the decoded letters

Direct follow-up: *"update this so it shows the handwritten... values (repeated)... 'Your network
translates [][][] to XXX'... it should show the handwriting and what it gets translated to with
the current network."* The compact "reads today's word" line under the training diagram used to be
text-only (`stageWordReadout` alone). `syncStageWordThumbs()` now renders each currently-drawn word
pad's *real* ink into a small, uniformly-sized `.word-thumb` canvas (22px CSS size, backed by a
44×44 bitmap for a crisp downscale) via `drawImage(sp.canvasEl, 0, 0, 44, 44)` — a literal copy of
that exact pad's current pixels, not a re-rendering or a placeholder glyph — and inserts one per
non-blank pad into `#stageWordThumbs`, read left-to-right in the same order as the word pads
themselves. The line now reads "Your network translates [H][A][T] to CDD" with real handwriting
thumbnails standing in for `[H][A][T]`.

**Wired into `renderDecoded()` itself, not a separate call site.** `renderDecoded(str)` was already
the single function both `#wordOutput` (the full step-01 display) and `#stageWordReadout` (the
compact text) read from, specifically so they can never desync — `syncStageWordThumbs()` was added
as its first line for the same reason: every trigger that already calls `renderDecoded()` (a word
pad being drawn on or cleared, "Clear all", every Train, Reset, and weight Load) automatically keeps
the thumbnails current too, with no new call sites to remember to wire up. If a future change adds
another way to alter the word pads, make sure it still routes through `renderDecoded()` rather than
updating `#wordOutput`/`#stageWordReadout` directly, or the thumbnails will silently stop tracking.

## Page order: the word comes first, teaching comes second

**Superseded by "The wizard redesign" further down** — the "01"/"02" numbering, the pre-loaded
seed word, and `#stageWordReadout`'s "reads today's word (step 01, above)" wording described here
are all from before the wizard restructure. Kept as the historical record of *why* word-before-
teach was the right call in the first place (that reasoning still holds, it just now spans more
steps) — don't treat the specific step numbers or the seeded-word details below as current.

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

**Superseded by "The wizard redesign" further down — there is no seeded word anymore.** Every
word pad starts genuinely blank now, and `SEED`/`patternToGrid`/`HANDWRITTEN_STROKES`/
`drawHandwrittenLetter`/`seedWord()` were all deleted outright, not just disabled — a visitor
writes their own first word from the very first thing they see, which is the whole point of the
new step 2 ("here's what a random network reads *your* word as"). Kept below as the historical
record of a real bug this once fixed, in case a future seeded-example feature is ever added back
and needs the same "look like handwriting, not pixel-art bricks" lesson.

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

**This is a display-only change — `SEED`/`patternToGrid` are untouched**, and that's a deliberate,
accepted tradeoff, not an oversight: `decodeWordPads()` re-samples whatever is
*currently drawn* on a word pad's canvas, independent of whatever data trained the network, so
there's no requirement that the two match pixel-for-pixel. The practical effect is that the
initial "HAT" decode used to be reliably close to correct on first paint back when the page seed-
trained on load — moot now that it doesn't (see "The page opens genuinely blank," above): a fresh
random network has no learned associations to match or mismatch against these strokes either way,
so the word simply decodes to whatever a random network decodes anything to until you actually
teach it something.

## The training tip (`computeTrainingTip`) — a real self-consistency check, not a guess at intent

**Hidden by default now (`SHOW_TRAINING_TIP = false`, see "The wizard redesign" further down) —
not deleted.** Direct feedback, after living with it for a while: even the honestly-computed
self-consistency version documented below still *reads* as an accuracy claim ("Looking solid," a
specific confidence percentage), and this tool has no ground truth to justify that framing any
more here than the rejected "suggest which letter is wrong" framing did. The computation itself
is untouched and still runs on every train/reset/load — only whether `#trainingTip` is ever
un-hidden changed. Revisit this if a more honestly-framed version (one that doesn't read as
measuring accuracy) is ever designed.

Direct request: analyze the output and suggest which letter would benefit most from another
example — "if someone's test input is HTT and they have the output HHH ... they should probably
train more T's." **That exact framing isn't something the tool can honestly compute**: a word pad
is just a drawing, with no separate "this position is supposed to be a T" label ever attached to
it, so there's no ground truth to compare a decode against and no way to know a decode is "wrong"
the way a person reading it can — the tool would have to guess at what the visitor intended to
spell, and guessing wrong would be worse than saying nothing.

What it *can* honestly compute, using only real, already-available data: for every letter that's
been taught at least once, replay every one of *that letter's own* training examples back through
the current network (`forward(x).probs[y]`, the real softmax probability the network assigns to
the correct class) and average them. A letter sitting at low self-confidence is either under-taught
(often because it only has one example) or genuinely being confused with something else already
taught — either way, teaching it one more example is the best next move, and this is computable
without ever needing to know what a test word was "supposed" to say. `#trainingTip` shows the
single worst-scoring taught letter by name with its real average confidence and example count
(`computeTrainingTip()`), or a "looking solid" message when every taught letter already averages
above 80% self-confidence — hidden entirely (`hidden` attribute) whenever `dataset` is empty, since
there's nothing yet to compute a real suggestion from. Recomputed at the same points
`refreshLookInside()`/`renderArchDiagram()` already are (after Train, Reset, and a successful
weight Load), so it's never stale relative to the current weights.

## The "Looking solid" tip was measuring the wrong thing (`MIN_REAL_FOR_CONFIDENCE`, `real` flag on `dataset` entries)

**Reported directly, with a screenshot**: taught M/I/K/E (one real example each) to spell "MIKE,"
the tip said *"Looking solid... recognizing its own examples with high confidence,"* and the word
still decoded wrong. Not a vague complaint — a specific, reproducible claim that the tool's own
"things look good" signal was lying.

**First, a real methodology mistake of my own, caught before blaming the app**: verifying this
requires driving the actual teach wizard and reading a real decode back, not just asserting JS
state. Scripting that hit two real snags, both worth recording so a future verification pass
doesn't repeat them:
1. `document.visibilityState` reports `"hidden"` for this tab throughout this automation
   environment, always, regardless of `tabs_select`/fronting — and `animateSequence()`'s
   `requestAnimationFrame` loop *never fires at all* while hidden (not throttled — zero callbacks).
   Since `doTrain()`'s real work (`runEpochs`, `refreshLookInside`, `refreshLedger`, re-enabling
   `openTeachBtn`/`resetBtn`) all lives inside `animateSequence(...).then(...)`, a script that clicks
   Train and moves on leaves training genuinely stuck mid-flight — `dataset.push` happened, the real
   weight update did not, and the UI is left with `openTeachBtn` permanently disabled. The
   workaround: a `computer` tool screenshot call between steps forces one real compositor frame,
   which is enough for one rAF tick to fire — and since `animateSequence`'s `frame()` checks real
   elapsed `Date.now()` time rather than counting frames, by the time that one tick fires (after the
   round-trip latency of the intervening tool calls) enough wall-clock time has already passed that
   it resolves the whole animation immediately. Any future scripted test of the teach flow needs a
   forced-paint step between clicking Train and checking the result, or it will silently test a
   stuck page.
2. **The first "clean" retest after the fix above shipped was itself contaminated**, and produced
   exactly the kind of scary, wrong-looking result that could have led to reverting a real fix for
   a bug that didn't exist in it. Word pads 0/1/2 start pre-loaded with the seeded "HAT" ink; a word
   pad's own click handler opens the *shared* `#expandCanvas` pre-loaded with whatever's already on
   that pad (so you can keep editing a drawing, not just replace it), and drawing new ink into it
   without clicking Clear first draws on top of, not over, the old letter. A fresh "K" drawn into
   pad 2 without clearing decoded as a blend of the drawn K and the original hand-authored T
   underneath it — confirmed directly by re-implementing `getGrid()`'s exact crop/downsample math
   against that pad's raw canvas pixels and rendering it as ASCII art, which showed T's literal top
   crossbar plus K's diagonals in the same grid. This alone was producing decodes like "MEEK" and
   "MKII" that looked like the augmentation fix had failed outright. Clicking `#expandClear` before
   drawing into any pre-seeded pad fixed it immediately. Any test that reuses word pads 0-2 without
   clearing first is not testing what it thinks it's testing.

**Once actually clean, the real signal was much smaller and much more informative than either the
scary contaminated result or a simple "still broken":**
- **1 real example per letter** (M, I, K, E, exactly matching the reported scenario): a fresh
  drawing of "MIKE" decoded 2/4 correct.
- **3 real examples per letter** (same four letters, same test): a fresh "MIKE" decoded 3/4 correct.

This is the expected, honest shape of the augmentation fix, not evidence it doesn't work:
`elasticJitter` (see "Real hand tremor is non-rigid" above) warps the *one* real drawing you gave
it — it regularizes around that example, it cannot manufacture the genuine shape-diversity that
only comes from a person actually drawing the letter more than once. One real example of a
26-way class is a genuinely hard one-shot problem for any method, augmented or not; the fix's
tested ~88% held-out accuracy (vs. ~78% baseline) was always measured at a realistic multi-example
regime, never at n=1, and n=1 was never going to be "solved" by better regularization alone.

**The actual, fixable bug this exposed**: `computeTrainingTip()`'s confidence check ran against
`dataset`, which now contains each real example *plus* `AUG_COUNT` synthetic near-duplicates of it
— trivially easy for the network to fit even off one real drawing, since all five are minor warps
of the exact same ink. High self-confidence at n=1 was therefore not a lie about the math (the
network genuinely was confident on those five near-identical inputs) but was a real, misleading
signal about generalization, stated to the visitor as unqualified reassurance right when the tool
should have been the most honest. Fixed by tagging each `dataset` entry `real:true`/`real:false`
(the pushed-in-`doTrain` original vs. its `elasticJitter` copies) and adding
`MIN_REAL_FOR_CONFIDENCE = 3` (the number the test above actually supports): "Looking solid" now
requires *both* every letter averaging >80% self-confidence *and* every letter having at least 3
real drawings behind it. Below that real-count threshold but still confident, a new message says so
plainly — *"that's partly the network matching its own practice copies of what you drew, not proof
it'll read a genuinely fresh attempt"* — and names whichever letter has the fewest real examples,
same pattern as the existing low-confidence branch. Verified in the real UI: taught the same letter
once (message correctly hedges, names it, says "1 real drawing"), twice more (still hedges, "2 real
drawings"), a third time (switches to "Looking solid" only once real count and confidence both
clear the bar).

**If `AUG_COUNT` or the augmentation mechanism ever changes**, re-check this tip's honesty
specifically — it's the one place in the tool that makes a confidence claim to the visitor, and it's
exactly the kind of thing that silently drifts back to "measuring the training set" if a future
change adds more synthetic data without re-deriving what "real" means for that data.

## Colorblind-friendly palette (`ACCENT_RGB`/`COOL_RGB`, `#colorblindToggle`)

Direct request: offer better colors for colorblindness, or at least an option. The rose/blue pair
(`C_ACCENT`/`C_COOL`) is this page's default excites/inhibits diverging scale, used everywhere a
real signed value gets drawn: `divergingColor()` (tiles, hidden/output bias dots), the heatmap, the
stage animation's forward/backward lines, and the network-weights diagram's connection lines. All
of those read from two mutable module-level variables, `ACCENT_RGB`/`COOL_RGB`, rather than the
frozen `hexToRgb(C_ACCENT)`/`hexToRgb(C_COOL)` constants they used to be — flipping those two
variables and re-rendering is enough to update every one of those visualizations consistently from
one place, with nothing left pointing at the old colors. The colorblind-safe alternative is
orange `#E69F00` / blue `#0072B2` (the Okabe-Ito palette, a standard, widely-recommended
colorblind-safe diverging pair — chosen over inventing a new pair by feel), toggled by
`#colorblindToggle` and persisted across visits via `localStorage['inklingColorblind']` (read once,
synchronously, at the top of the script — before `ACCENT_RGB`/`COOL_RGB` are even initialized —
wrapped in try/catch since `localStorage` can throw in a locked-down browsing context and this
should never be what breaks the page).

**The legend's two color swatches are kept in sync explicitly, not left pointing at CSS.** They
used to be plain inline `style="background:var(--accent)"`/`var(--cool)"`, tied to the page's CSS
custom properties rather than the JS color variables — toggling colorblind mode would have updated
every canvas but left the legend showing the old rose/blue, directly contradicting what the
diagrams actually then looked like. `syncLegendSwatches()` sets `#legendAccentSwatch`/
`#legendCoolSwatch`'s `background` directly from the live `ACCENT_RGB`/`COOL_RGB`, called once at
load and again inside the toggle's `change` handler, so the legend can never drift out of sync with
what's actually drawn. Deliberately scoped to just these data-encoding colors — the toggle does not
touch the site's own `--accent`/`--cool` CSS custom properties or any button/branding color, since
those don't encode signed data and aren't part of what colorblind accessibility here is about.

## Look inside: tiles and heatmap (`refreshLookInside`)

**The section's intro copy states the architecture up front, in plain language, before describing
what the tiles/heatmap show** — added on direct feedback ("without understanding the network
architecture I don't even truly know what the 'look inside' is showing me"). It now opens with "the
whole network is only three layers: your drawing becomes 64 pixel values... 18 hidden neurons...
26 output neurons" before the receptive-field explanation, and the tile/heatmap caption below was
rewritten to say plainly that a tile *is* one hidden neuron's real 64 weights reshaped to the
drawing grid, and the heatmap is the *next* layer's weights, one row per hidden neuron in the same
order as the tiles — spelling out the connection between the two visuals explicitly rather than
assuming a reader already holds the architecture in their head.

Each of the 18 hidden-unit tiles reshapes that unit's 64 real `W1` weights back into an 8×8 grid
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
illustration of the shape: for each of the 64 input pixels, a line to each of the 18 hidden dots
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
skipped entirely rather than drawn at near-zero opacity, both for legibility (≤64×18 + 18×26 ≈
1,620 possible lines is already dense) and because a genuinely negligible weight isn't meaningfully
"a real connection" worth ink. Every position is still derived from the real `GRID`/`HID`/`OUT`
constants, not hand-tuned to "18 dots," so the layout stays correct if the network's shape ever
changes. Unlike the original schematic (rendered once at load, since the *shape* never changes),
this version is re-rendered every time the weights actually do — wired into the same call sites as
`refreshLookInside()` (after Train, after Reset, after a successful weight Load) — so it's always a
live picture of the current network, exactly like the tiles and heatmap are. Downloadable via
`archCanvas.toDataURL('image/png')`, same as every other canvas export on this page.

**No text anywhere on this diagram, and every layer vertically centered** — direct request. Removed
the "Inkling" title, the "this network's real trained weights" subtitle, every output dot's `A`-`Z`
letter label, the `GRIDN+' pixels'`/`'18 hidden (tanh)'`/`'26 outputs'` axis captions, and the
bottom "N real weights and biases..." line — this diagram is meant to print clean as pure
weights-and-connections, with any explanation living in the surrounding page instead. Removing the
text also freed up the vertical margins it used to need, so `ARCH_IN`/`ARCH_HID_TOP`/
`ARCH_HID_BOTTOM`/`ARCH_OUT_TOP`/`ARCH_OUT_BOTTOM` are now all derived from `ARCH_H/2` (each layer's
own span centered on the canvas's mid-height) instead of the old fixed offsets that existed to leave
room for a title above and axis labels below. If text or asymmetric layers are ever added back here,
recheck the centering math rather than assuming the old offsets are still meaningful.

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

## `lab/inkling/thumbnail.jpg` is a cropped real export, not a separate asset

Inkling shipped for a while with `lab/index.html`'s card pointing at a `thumbnail.jpg` that didn't
exist yet (silently falling back to the text-only card via the `onerror` handler, per the site-wide
convention — nothing broke, it just showed no thumbnail). The image is a tight 600×600 crop of a
real `#archCanvas` "Download diagram as PNG" export — the input pixel grid, the fan of real
connections, and the hidden-neuron column, all real trained weights, not staged art. The diagram
itself is much wider than tall (920×640 in its exported, DPR-doubled form), so the crop deliberately
favors the *left* side (input grid through the hidden column) over including the output column too —
the input grid's solid color block is the most immediately recognizable part at 96×96 icon size, and
fitting the full width into a square would have meant either cropping through the grid itself or
adding real letterboxing, both worse options. If the diagram's layout changes (e.g., the layer
spacing from the vertical-centering work above), re-export and re-crop rather than assuming the old
600×600 window still frames the right part of the image.

## The wizard redesign — a step-by-step flow, not a page of independent sections

Direct, extensive feedback (dictated, so summarized here rather than quoted at length): the page
"doesn't grab the user," asked someone to read three or four sentences before doing anything, and
should instead walk them through the process like a wizard — write a word, see it fail, then fix
it. This touched nearly every section on the page; the pieces below are all one coordinated
change, not independent tweaks.

**New flow, five numbered sections instead of four:**
1. **`#step-word` "Write a word"** — copy cut to one sentence. Word pads start genuinely blank
   (see "The seeded word looks hand-drawn" above — that whole feature was deleted, not just
   disabled) so a visitor's first action is writing *their own* word, not looking at a canned
   "HAT." The always-visible attempts log became a collapsed-by-default `<details
   class="disclosure">` ("Past attempts") — direct feedback that it "takes up a lot of real
   estate" without earning it; still there, just not competing with the actual task for space.
2. **`#step-guess` "See its first guess" (new)** — shows the same live decode step 1 just
   produced, framed as a random network's guess, and states flatly that it's wrong. This doesn't
   (and can't) check per-letter correctness — this tool still has no ground truth for what a
   visitor meant to write, the same real limitation "The training tip" section above documents —
   but it doesn't need one here: a freshly-initialized network's weights are pure random noise, so
   *any* non-empty decode from it is guaranteed nonsense regardless of what was actually drawn.
   That's a true statement about the network's state, not a claim about being able to read a
   specific intended word, which is what keeps this honest where the training tip's old framing
   wasn't. `renderDecoded()` now updates a second readout (`#firstGuessOutput`/
   `#firstGuessStatus`) alongside the existing `#wordOutput`/`#stageWordReadout`, from the same
   single function, so all of them stay in sync automatically — no separate update path to drift.
3. **`#step-teach` "Teach it a letter"** — intro paragraph removed. The excite/inhibit/backprop
   legend (`#stageLegend`) now starts `hidden` and is only revealed during the very first Train
   click ever (`doTrain()`'s `isFirstTrain = dataset.length === 0`, checked *before* the push),
   then hidden again once that first animation finishes — direct feedback wanted the explanation
   to exist once, not as permanent chrome under the diagram forever after. Re-triggers correctly
   after Reset, since `dataset.length` genuinely returns to 0 there too.
4. **`#shareSection` "Take it with you"** (moved up, was after Look Inside) — the "render a
   network portrait" button, its whole explanatory paragraph, and `renderArchDiagram`-adjacent
   portrait-export code (`$('portraitBtn')`'s click handler, `#portraitImg`, `#downloadPortraitRow`/
   `#downloadPortraitBtn`) are deleted outright, not hidden — direct feedback: "I still don't
   understand what that even does... just get rid of it." `#qrNote` is now static, simple prose
   ("Scan this to come back to this exact page with your trained network already loaded") instead
   of `refreshLedger()`-generated byte-count text — the byte-count/quantization detail moved into
   the deep-dive instead (below).
5. **`#step-lookinside` "Look inside the network"** (moved down, was before Take it with you) —
   cut to one sentence ("A live picture of the network's actual weights and biases — no averages,
   no stand-ins"). The fuller explanation that used to live here moved into "How this actually
   works" as its own "Look inside the network, in detail" subsection, verbatim in substance.

**"How this actually works" now opens expanded (`<details ... open>`) instead of collapsed** —
direct feedback that there was "no reason why we should hide all of that information," alongside a
standing complaint that the section itself "is actually kind of light." Both are addressed
together: opening it by default, and genuinely adding content rather than just un-hiding what was
already there — the relocated Look Inside explanation (above) and a new "The QR code is quantized"
subsection are both net-new material, not copy moved around for its own sake. That second
subsection also fixed a real, separate staleness bug found while writing it: the old copy said
"Total parameters: 100×18+18 hidden... = 2,312... At 8 bits that's 2,312 bytes... At 4 bits it's
1,156 bytes" — numbers from when `GRID` was still 10 (100 inputs), never updated when it dropped
to 8 (64 inputs; see "Real hand tremor is non-rigid" for that change). Correct current numbers are
64×18+18 hidden, 18×26+26 output = **1,664** parameters, 1,664 bytes at 8-bit, 832 bytes at 4-bit —
matching the ~1,120-base64-character payload the live QR ledger has reported all along (the
*computed* ledger note was always correct; only this hardcoded prose paragraph had drifted). The
new subsection also explicitly states what the old copy left implicit — that scanning the QR code
back in reconstructs a network *close to* the live one, not byte-identical to it — and repeats the
actual QR code (`renderQrInto()`, factored out of `refreshLedger()` so both `#qrHolder` and the new
`#qrHolderDeep` render the identical payload from one function) right next to that explanation,
per direct request: "repeat the QR code down there."

**A real `[hidden]` CSS bug, caught by testing this rather than assumed correct — the exact same
class of bug `.expand-actions[hidden]` already has a documented fix for, elsewhere on this page.**
`.legend{ display:flex; ... }` is an *author* rule, which beats the *user-agent* stylesheet's
`[hidden]{ display:none }` regardless of selector specificity (author rules always win over UA
defaults in the cascade, a tier above specificity math) — so setting `$('stageLegend').hidden =
true` silently did nothing visually; `getComputedStyle(el).display` read `"flex"` with `.hidden ===
true` at the same time, confirmed directly, not assumed from reading the CSS. Fixed the same way
`.expand-actions[hidden]` was: an explicit `.legend[hidden]{ display:none; }` override. **If you
add `hidden` toggling to any *new* element on this page, check whether its own class sets a
`display` property before trusting the attribute alone — this is now the second time this exact
mistake has shipped on this file.**

**`SHOW_TRAINING_TIP = false`** — see "The training tip" section above for the reasoning (it reads
as an accuracy claim this tool has no ground truth to back up, the same problem step 2's wording
was written carefully to avoid). The `Reset` button's "no seedTrain()" test/behavior and the
`#resetBtn` double-click-to-confirm flow are unchanged by any of this — direct confirmation to
keep it exactly as it was.

## Step 1's prediction/attempts moved out entirely, not just hidden

Direct feedback: "get rid of the prediction and past attempts in step one. Those should be
exclusive to steps two and three." `#wordOutput` (the full decode display) and the "Past
attempts" `<details>` block were both **deleted from `#step-word`'s markup**, not hidden with
CSS — step 1 is now just the word pads, "Clear all," and the Next button, exactly the "prompt
them to write a word" framing the original wizard redesign called for, with nothing downstream
of that action visible yet. "Past attempts" moved down into `#step-teach` (step 3), still a
collapsed-by-default `<details>`, sitting after the stage diagram/readout — it's a training-time
log, and step 3 is the only place training happens. Step 2 (`#step-guess`) already had its own
`#firstGuessOutput`/`#firstGuessStatus` readout, untouched by this change.

Removing `#wordOutput` meant `renderDecoded(str)` — the one function every word-pad
change/train/reset/load already funnels through — could no longer unconditionally write to it.
Rather than null-guard a reference to an element that will now never exist again, the `el`/
`wordOutput` lookup was deleted from the function outright (the other three targets —
`#stageWordReadout`, `#firstGuessOutput`, `#firstGuessStatus` — are still individually
null-checked, since they're real, current elements). If step 1 ever gets its own decode display
back, that's a new element with a new lookup, not a revival of this one.

## The translate-readout spinner and highlight flash (`#translateSpinner`, `.just-updated`)

Direct feedback: during a Train click, `#stageWordReadout` ("Your network translates ... to
X") silently sits on its *old* value for the entire ~2s stage animation plus the `runEpochs`
pass after it — "it's not clear that that's being updated." Two additions, both scoped to
`doTrain()` only (a word pad being drawn on or cleared already gives its own immediate visual
feedback — the ink itself — so neither the spinner nor the flash fires for that path):

- **`#translateSpinner`** (a small CSS-only rotating ring, `.spinner-inline`) sits right next to
  the readout, `hidden` by default. `doTrain()` un-hides it the moment training starts (right
  alongside `setControlsEnabled(false)`) and hides it again inside the `.then()` callback once
  `renderDecoded()` has actually run with the new weights — so it's a literal "this value is
  stale, a real computation is in flight" indicator, not a fixed-duration decoration.
  `.spinner-inline[hidden]{ display:none; }` is declared proactively, before this element ever
  shipped with a bug — `.spinner-inline{ display:inline-block; ... }` is exactly the kind of
  author rule that the file's two prior `[hidden]` bugs (`.expand-actions`, `.legend`, both
  documented above) came from, so the override went in from the start rather than waiting for a
  third real regression to teach the lesson again.
- **`.just-updated`** (`readout-flash` keyframes: a brief flash to `var(--good)` with a text
  glow, fading back to inherit over 900ms) marks the *moment* the readout actually changes.
  `renderDecoded(str, flash)` takes a new second parameter — `true` only from `doTrain()`'s
  `.then()` callback, never from the word-pad-edit/clear/reset/load call sites, since those
  already have their own obvious cause-and-effect (you just watched yourself draw or clear
  something) and don't need a second highlight layered on. When `flash` is true, it removes then
  re-adds the class with a forced reflow (`void compact.offsetWidth`) in between — restarts the
  CSS animation from the top even if a fast second Train click retriggers it while the first
  flash is still fading, rather than the browser silently no-op'ing a redundant `classList.add()`
  on a class that's technically already present.

Verified end-to-end via a temporary debug harness (`window.__debugInkling`, exposing `doTrain`/
`pad`/`wordPads`/`decodeWordPads`/`renderDecoded`, removed before shipping — same pattern as
every other debug harness documented in this file): drew directly onto the teach pad and a word
pad's real canvases, then called `doTrain()` with `window.requestAnimationFrame` temporarily
patched to invoke its callback synchronously with `performance.now()+5000` (jumping straight past
the animation's `TOTAL=2000` window in one deterministic call, since this automation environment's
`document.visibilityState` stays `'hidden'` throughout — the same known limitation documented
elsewhere in this file — so a real rAF-driven completion never arrives here). Confirmed directly:
spinner hidden before the call, visible immediately after `doTrain()` starts, hidden again once
its promise resolves; the readout's text changed to the newly-trained letter's decode; and
`stageWordReadout.classList.contains('just-updated')` was true right after resolution.

## Heatmap axis labels (`refreshLookInside`)

Direct feedback: the `W2` heatmap's row numbers (0-17) had no label saying what they were
indexing. Canvas grew from 440×306 to 460×330 (`leftM` 22→36 for a widened left margin, plus a
new bottom margin) to make room for two axis titles, drawn after the existing grid/number
rendering: `'output letter (A–Z)'` centered below the grid (the columns already carry real A-Z
letter labels above them, but nothing said what the axis *was*), and `'hidden neuron (0–17, same
order as tiles)'` rotated -90° in the widened left margin, replacing what used to be bare, unlabeled
numbers. "Same order as tiles" is a deliberate callback to the tiles-vs-heatmap connection this
section's intro paragraph already makes in prose ("one row per hidden neuron in the same order as
the tiles") — the axis label repeats it right where a reader is actually looking at that row
index, not just once in the surrounding paragraph. Verified by sampling non-background pixel
counts in both label regions directly from the live canvas (`getImageData`), rather than by
screenshot — this session's browser pane wasn't compositing frames at all (a different, harder
failure than the usual `document.visibilityState` issue, which at least resolves with a forced
paint) — confirming both regions have real drawn content and aren't silently clipped or blank.

## Testing changes

No test suite — static page. Verify via a local static server (root-relative `/nav.js` and
`/theme.css` mean `file://` won't pick them up). Golden path: confirm the page opens with step 1
("Write a word") showing every word pad genuinely blank (no pre-seeded letters anywhere) and
**no decode display or "Past attempts" disclosure anywhere in step 1** — both now live
exclusively in later steps (step 1 is just the pads, "Clear all," and Next), step 2
("See its first guess") showing "draw a word in step 1 first" until something's drawn, step 3
("Teach it a letter") below with `trainStatus` reading exactly 0 examples across 0 letters, every
chip's count at 0, `#trainingTip` hidden, `#stageLegend` hidden (nothing trained yet, `.legend`
should be `getComputedStyle(...).display === "none"`, not just have `.hidden === true` — see "The
wizard redesign" above for why that distinction matters here specifically), `#translateSpinner`
hidden, and a collapsed "Past attempts" disclosure sitting under the stage readout → draw a letter
into a step-1 word pad, confirm step 2 updates to show that same decode with a red "Wrong!" message
and a "Next →" button → click it (or just scroll — these are anchor jumps, not gated navigation)
into step 3, click the generic "Tap to draw a letter and teach it" trigger, draw something, click
"Next →", pick a letter, click Train, confirm the modal closes, `#stageLegend` becomes visible
(`display:flex`) for this first training only, `#translateSpinner` becomes visible the moment
training starts and hides again once it resolves, the stage animation plays real per-active-pixel
packets traveling pixel→hidden→output then backward output→hidden→pixel (two sequential legs, see
the packets section above), and once the animation resolves `#stageLegend` goes back to
`display:none`, the readout under the diagram briefly flashes green (`.just-updated`, see above)
if it changed, and `#trainingTip` stays hidden throughout (it should never appear at all with
`SHOW_TRAINING_TIP = false`) → train a *second* letter and confirm the legend does **not**
reappear (only the very first training ever, or the first after a Reset, should show it) → back on
the page, click a *specific* letter chip (not the generic trigger), confirm the wizard opens with
that letter named directly in the header ('Draw the letter "X"'), the draw step's own button
already reads "Train ↳" (not "Next →"), and clicking it after drawing commits and trains
immediately — the letter-picker step should never appear at all for this path → in step 1, change
one word-pad letter and confirm it re-decodes and logs a new (collapsed-by-default) attempt with no
separate button click needed, then click "Clear all" and confirm every pad clears and it re-decodes
to blanks in one action, and confirm step 2 falls back to its "draw a word first" state again too →
click "Reset network" twice (first click only arms the confirmation toast) and confirm chip counts
and `trainStatus` both go to exactly 0, the look-inside tiles snap to random noise, the word decode
updates to reflect the now-random network, and the network-weights diagram's input grid visibly
loses whatever letter-shaped pattern it had and shows uniform noise instead → check the
"Colorblind-friendly colors" checkbox and confirm the legend swatches (next time they're shown),
the look-inside tiles, the heatmap, the stage diagram, and the network-weights diagram all switch
from rose/blue to orange/blue together, with nothing left showing the old colors; reload the page
and confirm the checkbox and the colors both persisted → confirm step 4 ("Take it with you") shows
the QR code with the simple one-line note (no byte-count text) and the network-weights diagram, in
that order, with no "render a network portrait" button or section anywhere on the page, and both
"Download QR as PNG" and "Download diagram as PNG" save real, non-empty PNGs → confirm step 5
("Look inside the network") comes *after* step 4, not before, with just the one-sentence intro,
and the heatmap shows both axis titles ("output letter (A–Z)" below the grid, "hidden neuron
(0–17, same order as tiles)" rotated in the left margin) rather than bare unlabeled row numbers →
train one more letter and confirm the diagram's connections/dot colors visibly change afterward
(it re-renders on every Train, not just at page load) → confirm "how this actually works" is
**expanded on page load**, not collapsed, and contains (in order) the architecture diagram/math, a
"Look inside the network, in detail" subsection, a "The QR code is quantized" subsection with a
*second*, independently-rendered QR code (`#qrHolderDeep`) showing the same payload as step 4's,
and the full-precision copy/load weights UI → copy the full-precision weights, clear/retrain a
little, then paste the copied JSON back into "Load weights" and confirm the look-inside tiles snap
back to the earlier state (a real round-trip, not just "no error thrown").

Also: at the page's own `.wrap` max-width (900px), confirm all 26 chips sit on a single row rather
than wrapping to two → confirm the stage diagram at idle shows no caption text at all (not even a
placeholder sentence), and that Reset clears any caption text left over from a previous Train's
animation rather than leaving it stuck showing something like "backprop → ..." → confirm the
word-decode readout and the colorblind toggle sit on the same line at full width (they may wrap to
two lines on narrow viewports, which is expected) → confirm there's no large empty band of unused
canvas below the diagram's lowest content (the bottom output row's letter labels) before the card's
own border.

**Real handwriting generalization** (the thing that actually matters most — a synthetic golden path
passing doesn't guarantee this): draw the *same* letter three or four times, deliberately at
different sizes and different positions on the pad each time (not the same careful stroke
repeated), training each as the same letter. Then draw a *fresh* example of that letter, differently
sized/positioned again, into a word pad and confirm it decodes correctly — and confirm accuracy on
that fresh example gets *better*, not worse, as you go from one training example to three or four.
If it gets worse with more (real, varied) examples, `getGrid()`'s centering/cropping is broken —
check it before suspecting anything else. Vary more than just size and position, too: draw the
letter with genuinely different stroke tremor each time (not the same careful path traced
repeatedly), not just at different sizes/positions — see "Real hand tremor is non-rigid" above for
why a test that only varies size/position can pass while the harder, more realistic case still
fails.
