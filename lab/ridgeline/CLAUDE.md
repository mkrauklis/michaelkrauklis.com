# Ridgeline

Turns a photo of a skyline into a printable graphic (t-shirt/mug/poster style), and
optionally shows the Fourier-series math that builds the outline as an animated or
step-by-step reconstruction.

## Architecture

`index.html`, plus two site-wide shared files it links rather than duplicates: `/nav.js` (the
breadcrumb header) and `/theme.css` (palette, base typography, and the panel/dropzone/button/
hero styles shared with Afterimage — see the root `CLAUDE.md`'s "Site-wide shared files"
section). Ridgeline's own `<style>` block only holds what's actually specific to it: `.wrap`'s
max-width, the step/merch-picker visualization CSS (`.toggle-group`, `#gridOutput`, `.step-row`
and friends, `.style-cards`, `.swatch-row`). No build step, no bundler, no external JS
dependencies beyond that — everything else is inline `<script>`. Everything runs client-side in
the browser; no photo is ever uploaded to a server (this is stated in the page footer — keep it
true).

Ridgeline doesn't override `--accent` — amber is `theme.css`'s default, and Ridgeline is the
page that default was written for. If a future tool wants a different brand color (Afterimage
uses violet), override `--accent`/`--accent-dim` in that page's own `<style>` block, after the
`<link>` tag.

External resources loaded: Google Fonts (Fraunces, Inter, via `theme.css`) and the AdSense
script tag. That's the entire dependency surface.

## Pipeline (in order)

1. **Upload** — drag/drop or file picker, read via `FileReader` → `Image`.
2. **Silhouette extraction** — for each of `N` x-samples, scan down the column for the
   sky/land boundary. This went through four real designs, each one fixing a real failure the
   previous one had (see the note at the end of this section on why synthetic tests alone
   didn't catch most of these):
   - **A brightness threshold** (with a per-column adaptive shift) assumes sky and land fall
     on opposite sides of some cutoff — true for a dark ridge on a bright sky, false for a
     snow-bright peak on a darker sky, shaky for a hazy far ridge nearly the same brightness
     as the sky behind it, and outright broken for a column that goes sky→bright snow→dark
     forest (two opposite-polarity edges stacked in one column).
   - **A local step score** (compare each row's own above/below neighborhood, threshold via
     Otsu on the score distribution) fixes the polarity problem but not a textured sky — a
     dramatic, cloud-heavy sky has plenty of *local* steps of its own (a hard cloud edge scores
     just as high as real land), so the very first cloud edge from the top used to win,
     collapsing the whole line into the sky. This is the one that looked fine on synthetic
     flat-fill test images and then failed badly on an actual dramatic mountain-lake photo.
   - **The third design** (median/MAD against a single fixed reference band) sampled a
     reference from the very top strip of the photo (sky, almost always), and walked each
     column down looking for where the brightness *permanently* left that reference. The
     reference was summarized by **median and MAD** (median absolute deviation), not mean and
     standard deviation — a sky that's part blue, part bright cloud is two populations in one
     sample, and a mean/stdev gets dragged toward the cloud population enough to make a second,
     brighter cloud further down the frame look like it's still within range. This shipped, was
     validated against real cloudy test photos, and worked — until a real photo with a smooth
     vertical sky gradient (skies routinely get lighter toward the horizon) showed it was
     fundamentally blind to gradients: a fixed reference has no way to tell "the sky is
     gradually getting lighter" apart from "we've left the sky," so every column falsely
     triggered at nearly the same row regardless of sensitivity, long before the real ridge. A
     user report ("I can't get the line to even go down to the ridge line") is what surfaced
     this — the failure was invisible in this project's own prior test photos, which happened
     not to have a pronounced gradient.
   - **The current design** keeps the median/MAD idea but **detrends the reference**: instead
     of one fixed band, it samples **two** bands (`bandMedian()`) near the presumed-sky edge of
     the photo and fits a straight line through their two median brightnesses, extrapolated to
     every row (`expectedSky(y)`) — this cancels a smooth gradient out before it can accumulate
     into a false departure, while keeping the reference otherwise fixed (not re-adapted every
     row it scans past). That "fixed, not adaptive" property is load-bearing, not incidental —
     an intermediate attempt at this fix made the reference *adaptively track* whatever the scan
     had recently passed (to chase gradients), which did fix the gradient case but broke the
     cloudy-sky case: an adaptive reference re-centers on a cloud just as readily as it
     re-centers on land, so it lost the one thing that let a fixed reference tell a *permanent*
     land boundary apart from a *transient* cloud edge (see "Bias correction" below — the
     majority/lookahead vote depends on that permanence). Detrending fixes the gradient blindness
     without touching that property. `mad` is pooled across both reference bands together and
     floored at a small absolute minimum (`Math.max(3, ...)`) rather than the bare
     mathematical floor of 1 — a single very large, very uniform, very bright cloud sitting
     exactly where both reference bands sample from can otherwise produce a near-zero MAD, which
     turns any tiny real variation elsewhere into an enormous, meaningless z-score.
   - Per column, a sliding-window median (`histMedian()`, over an incrementally-updated 256-bin
     luminance histogram — an exact sort per window, at `N` columns × up to `h` rows per
     extraction, was too slow) gives a z-score against `expectedSky(y)` at every row; the
     boundary is the first row (scanning from the sky side down) where *most* of a lookahead
     window — not just that one row — has cleared the z-score cutoff. "Most, not all" absorbs a
     single sunlit rock face or gap that would otherwise read as a brief return to "sky."
     "First, not strongest" is what makes a weak-but-real ridge higher up beat a stronger edge
     lower in the same column, like a lake reflecting the same ridge a second time near the
     bottom of the frame.
   - The **"Sensitivity" slider** scales how many MADs from the expected (detrended) sky
     brightness counts as "no longer sky" (higher accepts a smaller departure). **"Flip it"**
     means "trace the boundary as seen from the far side" — it samples both reference bands from
     the *bottom* of the photo instead and scans upward — rather than assuming an inverted
     brightness polarity, since the detector no longer cares which side is darker. The elevation
     array still gets a 5-wide median filter (`medianSmooth`) afterward for single-column noise
     (a bird, a lens artifact) on top of all of this.
   - **A known remaining limit**: if a real photo has heavy, bright, uniform cloud covering
     *exactly* the top ~16% of the frame that both reference bands sample from (not light cirrus
     with real blue gaps, but dense, near-total coverage right at the very top edge), the
     detrended reference can still be built from cloud rather than sky and mislead the detector —
     confirmed with a deliberately adversarial synthetic test photo, not just theorized. This is
     a harder case than any of the three real photos this algorithm has been validated against
     so far. The **manual override** below exists specifically for cases like this, where no
     automatic threshold gets it right — reach for that rather than chasing a fully general
     automatic fix for every possible sky.
   - **Why synthetic tests weren't enough (repeatedly)**: the second design was validated
     against flat-fill synthetic images and looked correct until it met a real dramatic sky. The
     third design was validated against real cloudy photos and looked correct until it met a
     real gradient sky. Both times, the synthetic tests used to build confidence just didn't
     contain the specific thing that broke the design in the field. If you change this algorithm
     again: validate against a real photo with the *specific* property the previous fix already
     handles (a busy/cloudy sky) *and* a synthetic gradient-sky test *and* a synthetic
     heavy-uniform-cloud-at-the-top test — this history has now produced three separate,
     unrelated failure modes, and a change that silently regresses any one of them is exactly
     the kind of bug that hides until a real user hits it.
   - **Bias correction**: both the sliding-window median and the majority-vote lookahead only
     register a break once *most* of what they're looking at has crossed into land, so the raw
     row they land on is already partway into land — a predictable overshoot of about
     `winRows/2 + (1-MAJORITY)*lookahead` back toward the sky. `detectSkyline()` corrects for
     this directly (the `bias` constant) rather than leaving the line to visibly float above
     the real ridge. If `winRows`, `lookahead`, or `MAJORITY` change, this correction needs to
     move with them.
   - **Auto sensitivity, and why "smoothest wins" isn't quite right**: `autoTuneAndExtract()`
     runs `detectSkyline()` at a spread of sensitivities and uses roughness (mean column-to-
     column jump) to pick one, because a wrong sensitivity reacts to noise and visibly jumps
     around — a real skyline doesn't. But the single smoothest candidate isn't always the most
     correct one: a *lenient* sensitivity triggers on weak evidence, and on a sufficiently
     adversarial sky (see the known remaining limit above) that early, wrong trigger can be just
     as consistent column-to-column as the real ridge is — or more so, collapsing to a nearly
     flat line that reads as *extremely* smooth despite being flatly wrong. So it finds the best
     roughness achievable first, then walks candidates strict-to-lenient and stops at the first
     one already close to that best (see the `+1.5` tolerance) — roughness rules out
     sensitivities that are clearly too strict (visibly erratic), it isn't used to go hunting for
     the single smoothest result on offer. It also filters out any candidate whose elevation
     span is under 5% of the photo's height before roughness is even considered: a real ridge has
     genuine vertical variation, and a falsely-uniform line collapses to just a few pixels of
     span no matter how smooth it looks — confirmed directly against a test photo built to
     trigger this (correct candidates: 40-85% of height in span; the falsely-flat one: under 2%).
3. **Fourier decomposition** — the elevation profile is mirrored (`M = 2N`) to force
   periodicity, then run through a hand-rolled DFT (`dft()`). Components are sorted by
   amplitude, largest first, so reconstructions add the most structurally important
   waves first.
4. **Reconstruction / rendering** — `reconstructAt()` rebuilds the ridge from the DC
   term plus the first `count` components. Used for: the merch print canvas, the
   step-by-step panel grid, and the two video animation styles (sequential summation,
   rotating epicycle arms).
5. **Export** — high-res PNG for the merch design; PNG for the step panels; WebM
   (via `MediaRecorder` + `canvas.captureStream`) for the animation.

## Manual override: drawing the line by hand

Automatic detection can't win every photo — see the known remaining limit above, plus fog, a
ridge nearly the same tone as the sky, or a foreground object crossing the horizon. Rather than
chasing a fully general automatic fix for all of these, `sourceCanvas` accepts a direct
click-and-drag: `manualStart`/`manualMove`/`manualEnd` write straight into `state.elevations` at
whatever sample index the cursor's x maps to (`setElevationAtX`), interpolating between mousemove
events (`manualDrawSegment`) so a fast drag doesn't leave gaps. `redrawLineOverlay()` — the same
function `extractAndPreview()` uses — is called on every move for live feedback, and
`computeFourier()` re-runs once on mouseup so the rest of the pipeline (merch preview, step
panels, video) picks up the hand-drawn shape immediately. A light `medianSmooth(elevations, 3)`
runs on mouseup only (not during the drag itself, which would fight the cursor) to clean up
mouse-jitter without erasing the shape actually drawn — a much smaller window than the auto
path's 5, since the input here is already a human's intentional line, not noisy per-column
detection.

There's no special "manual mode" toggle — dragging on the canvas always overrides whatever's
currently in `state.elevations`, and touching a sensitivity/outline-detail slider always
re-runs `detectSkyline()` and replaces it right back, the same as it always did. One line, two
ways to set it, no mode state to get out of sync. `#resetLineBtn` just calls `extractAndPreview()`
again to discard a manual edit and go back to auto-detection. Both `mousemove`/`mouseup` (and
`touchend`) listen on `window`, not `sourceCanvas`, so a drag that leaves the canvas mid-gesture
still completes correctly — mirrors the window-level-listener pattern already used for
Afterimage's crop-box dragger.

## Conventions specific to this file

- **True-to-photo scale**: `fitScale = h/sourceCanvas.height` is used everywhere a
  reconstruction is drawn so that wave magnitude stays visually consistent across every
  panel/frame and the final overlay-on-photo step. Don't introduce a different scale
  factor for a new view without a good reason — it'll look inconsistent with the rest.
- **DFT is O(N·M)** and runs synchronously on the main thread. `N` (outline detail) is
  capped at 300 via the slider `max` for this reason — raising it materially slows down
  every slider interaction, not just the initial extraction.
- Step/epicycle-arm counts are always clamped to `state.comps.length` in
  `updateStepBounds()` — there's no benefit to letting a slider request more waves than
  exist.
- Color tokens are CSS custom properties in `:root` (`--amber`, `--teal`, `--danger`,
  etc.) — reuse these rather than hardcoding new hex values so the palette stays
  coherent if it's ever retuned.
- **Stack style's row count** (`merchState.stackRows`, default 8) is a slider
  (`#merchStackRowsRange`, min 3, max 20 — kept in sync with `MERCH_STACK_MIN_ROWS`/
  `MERCH_STACK_MAX_ROWS` in `drawMerchStack()`) threaded through `currentMerchOpts()`
  like every other merch option. This went through a real redesign, not just a wider
  range, after direct feedback caught three separate problems with the first version:

  1. **Off-by-one, by design intent rather than arithmetic**: the old code treated
     `rows` as the *total* line count, so the photo's own bold ridge line silently
     consumed one of the requested rows — asking for 3 only ever showed 2 faint lines
     above the photo. Fixed by making `rows` mean *only* the faint build-up lines
     (`counts = [1, 2, ..., rows]`); one additional bolder "result" line (the photo's
     ridge, or an extra standalone line with no photo) is always appended after that,
     never counted against the slider's number.
  2. **A taller stack used to mean a taller (or photo-squeezing) graphic**: with a
     photo, each wave row took a *fixed* `contentH*0.08` slice regardless of row count,
     so more rows directly grew the total space the wave stack consumed and shrank
     the photo panel — this was the original reason the max was capped at 10 at all.
     Fixed by reserving a **fixed total** height for the whole wave stack
     (`waveStackH`, a constant share of `contentH`) and dividing *that* by the row
     count — more rows now packs the same lines closer together (overlapping at high
     counts, which is fine — direct feedback: "I don't care if they overlap") instead
     of growing the graphic or eating into the photo panel. This is *why* the max
     could safely become 20 instead of 10 — the original constraint that capped it is
     gone.
  3. **The final line's own fidelity used to just be `rows` itself** — capped at
     3-20 Fourier components no matter how much real detail the photo actually has, so
     the "sharp result" line barely changed between slider positions. `finalCount` now
     ramps from ~15% up to 100% of `state.comps.length` as the slider moves from its
     min to its max (`t = (rows-MIN)/(MAX-MIN)`, `finalCount = totalComps*(0.15+0.85*t)`),
     so cranking the slider actually sharpens the one line that's supposed to look like
     the real result — verified by computing `finalCount` at both ends against a
     realistic `totalComps` (~220, this file's own outline-detail ceiling), not just
     eyeballing a low-detail test silhouette where the difference is barely visible.

  If either of `MERCH_STACK_MIN_ROWS`/`MAX_ROWS` or the slider's own `min`/`max`
  attributes change, update both together — the fidelity ramp's `t` calculation
  reads the constants, not the DOM element.

## Monetization

- AdSense: script tag + `google-adsense-account` meta tag in `<head>`, publisher id
  `ca-pub-5936916546458743` (matches the rest of the site — don't change independently).
- Two Zazzle links now, serving different jobs (same pattern as Echo State — see that tool's
  CLAUDE.md for the fuller writeup of the distinction):
  - **The hero CTA** ("Want to see what this looks like as a gift?", right under the intro
    paragraph) links to a real, specific, finished product — "Glacier National Park — Traced in
    Waves" (`glacier_national_park_traced_in_waves_mug-256459456782885365`), a mug built from a
    real Ridgeline export of an actual Glacier National Park photo. This sells the idea at a
    glance, before a visitor has touched the tool.
  - **`#shareSection`'s mug template link** (further down) is the generic "upload your own
    image" template — a plain link (not an API integration) to a pre-made blank template with
    the `?rf=<ambassador_id>` cross-promotion param, where a visitor downloads *their own*
    PNG and uploads it themselves. There's no server-side handoff of the design either way.
  - Both links carry `?rf=238054754631086278` — every Zazzle link on this site does; never add
    one without it.

## Page flow: explanation → tool → save & share → deep dive

The Zazzle link used to sit inline at the bottom of the merch-picker step ("03 — your print"),
mixed in with the style/color controls. Pulled out into its own `#shareSection` ("04 — save &
share"), revealed at the same time as the merch section, right after it — a small change made
specifically to match the flow Vectis landed on and that the other tools were asked to follow:
a brief explanation, the interactive tool, a distinct save/share step with the merch link, then
the "how it works" deep-dive last. Ridgeline's intro was already short (one hero paragraph) and
the math deep-dive (`composeSection`) was already a collapsed `<details>` near the bottom, so
this was the only piece out of place — don't fold the Zazzle link back into the merch step.

## Testing changes

No test suite — this is a static page. Verify changes via a local static server (root-relative
`/nav.js` and `/theme.css` links mean opening `index.html` directly over `file://` won't pick
them up — serve the repo root, e.g. `python -m http.server`, and browse to `/lab/ridgeline/`).
Run the golden path: upload a photo → check the auto-extracted outline looks right → render
steps/video → download. Check both the "Line" and "Stack" merch styles, and both video styles
(sequential summation, epicycle arms), since they share the reconstruction math but have
separate drawing code paths. For "Stack" specifically, drag the "Number of lines" slider to both
ends (3 and 20) and confirm: the count of faint lines actually shown matches the slider (not one
fewer), the photo panel's size stays constant across the whole range (only the line spacing
should change), and — using a photo with real jagged detail, not a smooth test silhouette — the
bold final line visibly sharpens as the slider goes up rather than looking the same at every
position (see "Conventions specific to this file" above for the exact formula and why a
low-detail test image won't show this last one).

If you touch `detectSkyline()` or `autoTuneAndExtract()`, also check: a synthetic photo with a
smooth vertical sky gradient and a jagged silhouette (a `<canvas>` gradient fill plus a filled
path is enough — no real photo needed) extracts a line that actually follows the silhouette, not
a flat line near the top, at every sensitivity from strict to lenient; a synthetic photo with
scattered soft-edged clouds and real gaps of blue between them still finds the real ridge, not a
cloud edge; and the manual override actually works end-to-end — drag across the photo, confirm
the line follows the cursor live, release, confirm the step-panel/video preview downstream picks
up the hand-drawn shape (not silently still using the old auto-detected one), then click "Reset
to auto-detected" and confirm it goes back to the varying, ridge-following line rather than the
flat manually-drawn one.
