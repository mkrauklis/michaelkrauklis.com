# Hunch

Two candy jars, a secret coin flip, and a real, exact application of Bayes' theorem — draw candies
one at a time, watch a "hunch" bar update after each clue, lock in a final guess, then get a
personalized "Certified Bayes Detective" badge to put on a shirt. Built directly from a one-line
request: "a walk through to help a 6th grader understand Bayes theorem in a fun way, ending with a
zazzle T-shirt link to show off what they learned" — no separate spec or prototype handoff this
time (contrast [[feedback-artifact-spec-handoff]], which describes the Attention workflow), so the
whole page structure, the game mechanic, and the pedagogical approach were designed directly in
this repo before anything was built.

## The name

Short-listed against the same "single evocative word, not the literal technical term" convention
every other tool here uses (Tonus, Vectis, Afterimage, Ridgeline, Echo State). *Hunch* was picked
because it's the one word that's simultaneously the everyday term a 6th grader already has for "a
guess based on a feeling" **and** literally what a Bayesian prior is — the page never has to
awkwardly introduce what a "prior" is before explaining why it matters, because "hunch" already
carries the right meaning on its own. The page's own tagline ("Every guess is a hunch waiting for
a clue") and the badge's engraved quote both lean on this same word doing double duty.

## Why a candy-jar guessing game, not a coin/dice abstraction

The classic textbook framing of Bayes' theorem (urns, balls, sometimes literally just "Event A"
and "Event B") is accurate but has nothing for a 6th grader to hold onto. Two jars with visibly
different candy mixes, a coin flip that's secret, and a "case" you're trying to crack is the same
underlying math with an actual story wrapped around it — closer to what Gerd Gigerenzer's natural-
frequency research (see "Learn more" on the page itself) found actually helps people reason about
conditional probability, and closer to "fun" than "here is a formula."

**Two candy colors, not three or more.** More colors would make the natural-frequency grid (see
below) more visually interesting but would also mean more sliders, more legend entries, and a
noticeably more crowded case log — for a first pass aimed at a 6th grader, two clearly-labeled
outcomes keeps every screen readable at a glance. The colors are **Orange** and **Blueberry**
(never just "the orange one"/"the blue one" without the flavor name in the copy) mapped to the
Okabe-Ito colorblind-safe orange/blue pair (`--c-orange:#e69f00`, `--c-blue:#0072b2`) — the exact
same verified pair Tonus's two-class visualization and Inkling's colorblind toggle already use
(see Tonus's CLAUDE.md for why that specific pair was chosen), not a fresh guess at "these look
different enough."

**Accent color** is a bubblegum fuchsia (`--accent:#e0559c`, `--accent-dim:#6b2d54`) — distinct
from every other tool's accent (amber, teal, violet, rose/crimson, periwinkle-blue, sky-blue,
rust, emerald green) and deliberately playful rather than clinical, matching a tool whose whole
framing is a fun guessing game rather than a serious ML/math instrument.

## The core loop: setup → active case → reveal

`state.phase` is one of `setup` / `active` / `revealed`. This is the one piece of real state
machine complexity on the page, and it exists for a specific pedagogical reason: **the jar
recipes and the coin's own fairness have to be visible and adjustable before a case starts, and
then frozen for the whole case.** Letting a visitor change Jar A's mix mid-case would silently
invalidate every math update already shown for that case — freezing the sliders (`setSlidersEnabled`)
the moment "Start the case" is clicked is what makes "the odds shown after each draw are the real,
unchanging odds for this exact case" actually true, not just asserted in copy.

- **Draws are with replacement**, and there's no upper bound on how many you can take from one
  case — an infinite candy jar is a simplification a 6th grader will accept instantly and a real
  statistician would recognize as the standard textbook assumption anyway (an urn "with
  replacement"). Modeling a finite, depleting jar would require tracking exact remaining counts
  per color per jar and switching the update rule to a hypergeometric one instead of a simple
  Bernoulli likelihood — real, but a second, harder concept this page was never trying to teach.
- **The guess toggle only unlocks after at least one draw** (`history.length >= 1`), on purpose —
  guessing off the bare coin odds alone, with zero clues, isn't wrong, but it also doesn't
  demonstrate anything this page is about. Requiring at least one clue first keeps every reveal
  actually showing Bayesian updating having done something.
- **The scoreboard (`casesSolved`/`casesTotal`) persists across "New Case" clicks, but not across
  a page reload** — same as every other piece of state on this site (no localStorage, no backend).
  It's there purely for the "detective" framing's own sense of a running record, and feeds directly
  into the exported badge.

## The natural-frequency grid, and why it isn't the primary formula

Stage 3 ("do the math") renders a 10×10 grid of 100 dots: the top `nA` dots represent "the 100
imaginary repeats of this case where the coin picked Jar A," the bottom `100-nA` represent Jar B,
and within each block dots are colored orange/blue proportional to that jar's own real recipe.
Only the dots whose color matches the clue you actually just saw get a ring around them — count
the rings in each half, and the ratio *is* the updated hunch, no algebra required. This is a
directly-implemented version of the "natural frequency" technique Gigerenzer's research popularized
specifically because turning a probability word problem into a counting problem measurably
improves comprehension versus presenting the same problem as `P(A|B)` notation cold.

**Deliberately not the first thing the page shows.** The actual textbook formula, `P(A|clue) =
P(clue|A) × P(A) ÷ P(clue)`, lives inside `details.disclosure.advanced` ("How this actually
works"), open by default (matching the now-established site convention — see Tonus's CLAUDE.md,
"Reversed later, on direct instruction: 'How it works is supposed to be expanded by default.'") but
visually and narratively *after* the grid, not before it. This mirrors the Attention feedback
pass's calibration lesson almost exactly (see `lab/attention/CLAUDE.md`'s six-item feedback
list) — introduce the concept in plain, concrete terms a reader can already picture, and only
*then* hand them the standard vocabulary as a translation layer for anything they might see
elsewhere, via a small dictionary table (Starting hunch → Prior → `P(A)`, etc.) rather than
assuming the jargon on first contact.

**The grid is intentionally approximate where the hunch bar is exact.** 100 dots can only
represent whole-number percentages, so `renderMathGrid()`'s own caption says outright that this
picture rounds to the nearest dot and that the real, exact number is whatever the hunch bar above
already shows — the grid's job is building intuition for *why* the number moved the way it did,
not being the page's source of truth for the number itself. `updateHunch()` itself does the real,
unrounded floating-point Bayes update every time; the grid is a separate, purely-illustrative
rendering computed from the same inputs.

## Real randomness, not seeded/deterministic

Every other tool on this site that involves randomness (Echo State, Inkling, Tonus) seeds a
`mulberry32` PRNG specifically so a result can be captured in a share link and reproduced exactly
on a different visit — that's the whole reason those tools bother with a seed at all. Hunch has no
share-link feature (there's nothing here that benefits from replaying an *exact* coin flip and
sequence of draws — the entire point is that a fresh case is a fresh, honest trial), so
`drawColorFrom()`/the coin flip in `startCaseBtn`'s handler both call `Math.random()` directly.
Given the page's own premise is literally "here's what real chance looks like," using true
browser randomness rather than a reproducible PRNG is the more honest choice here, not an
oversight.

## The badge (`renderBadge`, `#badgeCanvas`) and a real bug it surfaced

The badge is a 720×900 canvas: a fuchsia double border, "CERTIFIED / Bayes Detective," a hand-drawn
magnifying-glass icon (two canvas primitives — an arc and a rounded line, no image asset), the
visitor's own typed name (`#detectiveName`, defaults to "Anonymous Detective" if left blank), then
four lines of real session stats (cases cracked, the last case's actual jar recipes, how many clues
it took, and the confidence at reveal) via a small hand-rolled `wrapText()` helper, and a closing
engraved quote plus the page's own URL.

**Revised into an actual certificate, on direct request ("make the certificate cooler and more
official, include some jars visuals, and add the ability to download in white")** — three separate
asks, all addressed:
- **More official**: a small-caps "CERTIFICATE OF ACHIEVEMENT" header above the existing
  "CERTIFIED" line, a rotated-square flourish at each of the four border corners
  (`drawCornerFlourish`), a formal sentence structure ("This certifies that [name] has applied real
  Bayes' theorem to crack N of M cases" — falls back to "is ready to crack their first case" before
  any case exists, rather than showing a nonsensical "0 of 0"), a hand-drawn medal/ribbon seal
  (`drawSeal`/`drawStar` — concentric circles, a five-point star, two ribbon-tail triangles) in the
  bottom-right corner, and a "CASE FILE NO. ####" plus a real issue date in the bottom-left,
  mimicking an actual certificate's serial number and date fields. The case file number is a
  random 4-digit number generated once per page load (`caseFileNo`) — decorative, in keeping with
  the whole "certified detective" conceit already being fictional, not a claim of real record-
  keeping.
- **Jar visuals**: `drawMiniJar()` (a simplified version of `drawJar()`'s own rounded-rect-plus-
  candies logic, parameterized by size so it scales down cleanly) draws Jar A and Jar B — using the
  *real* recipe from the last solved case, or the current slider values before any case exists, so
  the badge's jars always match an actual jar you set up — flanking the magnifying glass icon.
- **Downloadable in white**: `BADGE_THEMES` holds a `dark` and a `light` palette (background, ink,
  dim text, and jar-line colors each swap; the fuchsia accent and the orange/blue candy colors stay
  the same in both, since they read fine against either background). A `Dark`/`White` toggle-group
  (`#badgeThemeToggle`) sets `badgeTheme` and re-renders; the download filename includes the theme
  (`bayes-detective-badge-dark.png` / `-light.png`) so a visitor who downloads both doesn't have one
  silently overwrite the other. Verified both themes render legibly by decoding real canvas output
  to PNG, not by inspection alone — the light theme in particular needed the jar outline color
  (`jarLine`) darkened from the dark theme's pale `#4a5064` to a mid-gray `#8a8f9b`, since the dark
  theme's own line color barely showed up against a near-white background.

**A real bug, caught during verification, not shipped:** `ctx.textAlign` was set to `'center'`
for the title block above and never reset before the four stat lines were drawn via `wrapText()`,
which calls `fillText(line, x, y)` with `x` meant to be a **left** edge. With `textAlign` still
`'center'`, every wrapped line rendered centered *on* that left-edge coordinate instead of
starting there — the visible symptom (confirmed by rendering an actual badge and reading the
pixels, not just inspecting the source) was every stat line clipped on the left, e.g. "ses cracked:
1 out of 1" instead of "Cases cracked: 1 out of 1." Fixed by explicitly setting
`ctx.textAlign = 'left'` immediately before that block and `= 'center'` again immediately after
(the closing quote and URL line below it rely on being centered). Caught by decoding the canvas's
own `toDataURL()` output to a real PNG and reading it as an image rather than trusting the
rendering code by inspection — the same "verify the pixels, not just the logic" discipline Tonus's
CLAUDE.md documents for its own spinner-visibility bug.

**The download is disabled until at least one case is resolved** (`downloadBadgeBtn.disabled` only
flips false inside `revealBtn`'s handler) — a badge certifying zero solved cases isn't a badge
worth printing, and disabling the button (with a status line explaining why) is clearer than
generating one full of zeroes.

## Zazzle link and the thumbnail

Reuses the exact same verified "Personalized One Of A Kind Photo Collage T-Shirt" template
(`…-235429040115913948`) already wired up for Attention's own portrait export, with the same
`?rf=238054754631086278` ambassador param. This isn't a case of pattern-matching a different
product ID and assuming it's the right one (the caution [[feedback-artifact-spec-handoff]] and
[[feedback-zazzle-referral]] both warn about) — it's the literal same generic "upload your own
square-ish image to a shirt" product, reused for the same use case (a downloaded PNG a visitor
uploads themselves), and it was re-verified live in this session (loaded the page, confirmed the
title still reads "Personalized One Of A Kind Photo Collage T-Shirt | Zazzle") rather than assumed
stable since Attention shipped.

`thumbnail.jpg` was originally a crop of the badge canvas's own top section — replaced on direct
feedback ("the icons for these are stinkers... an icon of one of the jars") once it shipped on
`lab/index.html`'s 96×96 card icon: the badge crop read as an unrecognizable wall of small text at
that size, not as "a jar" the way the request wanted. The thumbnail is now a tight, real crop of
`#jarACanvas`'s own rendering (the exact `drawJar()` output, captured via an offscreen canvas scaled
8× with `imageSmoothingEnabled = false` for a crisp, non-blurry crop, filled with the page's own
`#0c0e13` background first since a bare `drawImage` onto a fresh canvas leaves transparent pixels
that composite to white once flattened to JPEG) — real tool output, just a different, more
recognizable piece of it than the certificate.

## What a math-major-calibrated read of this page should confirm

Unlike Attention (whose audience was explicitly "a math major without a CS background"), this
page's stated audience is a 6th grader — a stricter calibration bar, not a looser one. Concretely:
no unexplained jargon anywhere outside the one clearly-marked "How this actually works" disclosure
(the words "prior," "likelihood," "posterior," and "Bayesian" never appear in the primary copy at
all — only inside that disclosure's own vocabulary table, explicitly translating from the plain
words already used), every number the page states out loud has just been shown concretely (the
grid before the formula, never the reverse), and the "why this matters" section's three real-world
examples (spam filters, weather forecasts, medical tests) are each two sentences of plain
description with no technical terms of their own.

## Second revamp: the certificate becomes a joke, and the page gets tighter

Direct feedback, delivered alongside Matrix Matrix and Attention all being demoted to the bottom
of `lab/index.html`: "nobody cares about a certificate. We want a joke or something that's
relevant and mathy... they're too confusing still. Please make them simpler and more direct."
Two separate fixes:

**The badge became "I updated my prior."** `renderBadge()` now draws a plain, loud joke instead of
a formal certificate: "REAL BAYES' THEOREM — NOT CLIP ART" (small, accent), "I UPDATED MY PRIOR"
(huge, bold), a row of candy dots colored by the last case's real Jar A recipe, then the actual
number swing from that case — `{startConfidence}% → {confidence}%` — at 96px, and "...and all I
got was this T-shirt." underneath. `startConfidence` is a new field on `lastCase`, computed at
reveal time as the coin's own starting odds for whichever jar turned out to be correct
(`state.secretJar === 'A' ? state.priorA : (1 - state.priorA)`) — a real, honest "before" number to
pair with the real "after" number already being tracked, not a made-up one. Everything from the
old certificate concept — the name input, the formal "this certifies that" sentence, the seal, the
case file number, the corner flourishes, the two full-size jar illustrations — was deleted, not
just hidden; `drawMiniJar`/`drawStar`/`drawSeal`/`drawCornerFlourish` and the now-orphaned
`wrapText` helper are gone. The Dark/White theme toggle survived, since it was never the problem —
a joke still needs to print cleanly on a light shirt.

**The main flow got shorter and less dense**, without cutting any real content: the "do the math"
paragraphs were tightened (e.g. "Before this clue, your hunch was 50% Jar A. Out of 100 imaginary
cases, about 50 would come from..." → "Your hunch before this clue: 50%... Picture 100 cases like
this one — about 50 come from..."), the reveal-panel copy dropped its more philosophical asides, and
the "How this actually works" disclosure — the formula and vocabulary table — now starts
**collapsed**, not open, so the page's default view is the natural-frequency grid and plain
language only, with the formal notation genuinely opt-in rather than front-loaded.

**A real bug fixed along the way**: `hunchCaption` and the natural-frequency walkthrough both wrote
"a orange candy" — a plain `'a ' + color` concatenation that never accounted for "orange" starting
with a vowel sound. Fixed with a direct ternary (`color === 'orange' ? 'an ' : 'a '`) rather than a
general a/an heuristic, since there are only ever two possible colors.

`lab/index.html`'s card description was also trimmed — it no longer mentions "your own detective
badge on a T-shirt," since the artifact backing that claim no longer exists in that form.

## Testing notes

No test suite — static page. Verify via the local static server (root-relative `/nav.js` and
`/theme.css` need `http://`, not `file://`). Golden path: load the page, confirm both jar canvases
render real, distinct dot patterns matching their slider values (8/10 and 3/10 orange by default) →
click a preset pill and confirm both sliders and both jar canvases update together → start a case,
confirm sliders lock and "Draw a candy" enables → draw several candies and confirm the hunch bar,
case log, and natural-frequency grid all update together after each draw, and that the grid's own
ring count ratio matches the hunch bar's percentage (up to the grid's whole-dot rounding) → confirm
the guess toggle stays disabled until at least one draw has happened → lock a guess, reveal, and
confirm the scoreboard increments correctly for both a correct and an incorrect guess (verified
both branches directly) → click "New Case" and confirm sliders re-enable, the hunch bar resets to
the coin's own odds, and the scoreboard total is preserved, not reset → type a detective name and
confirm the badge canvas and the shirt-mockup preview both update live → download the badge PNG and
open it directly to confirm every stat line is fully legible with no left-edge clipping (the exact
regression documented above — re-check this specifically if `renderBadge()`'s `ctx.textAlign` calls
are ever touched again) → confirm no console errors through the entire flow.

One environment-specific note: this automation environment's screenshot tool intermittently times
out ("the page did not finish rendering in time") on this page, the same documented rendering
artifact noted in Tonus's and other tools' CLAUDE.md files — it is not evidence of a real bug by
itself. Verification here was done by reading the accessibility tree (`get_page_text`/`find`),
driving interactions via both real clicks and direct `.click()`/event dispatch through
`javascript_tool`, and — for anything visual, like the badge — decoding the canvas's own
`toDataURL()` output to a real PNG and reading the actual pixels, rather than trusting a timed-out
screenshot tool either way.
