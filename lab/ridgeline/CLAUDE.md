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
   sky/land boundary. This used to be a brightness threshold (with a per-column adaptive
   shift), but that assumes sky and land fall on opposite sides of some cutoff — true for a
   dark ridge on a bright sky, false for a snow-bright peak on a darker sky, and shaky for a
   hazy far ridge that's nearly the same brightness as the sky behind it (all three showed up
   as real failures: a distant hazy mountain rising above a much higher-contrast near
   treeline, snow peaks brighter than a cloudy sky with a lake reflecting a second copy of
   the same ridge lower in the frame, and a deep-blue-sky/white-peak photo where the same
   column goes sky→bright snow→dark forest, i.e. two edges of opposite polarity stacked in
   one column). What's actually universal is a *step*: the region just above a real boundary
   and the region just below it have reliably different average brightness, regardless of
   which one is darker. So each candidate row `y` in each column gets a step score —
   `|meanAbove - meanBelow|` over two `winRows`-tall windows straddling `y`, divided by
   `sqrt(varAbove + varBelow + 16)` (a two-sample t-statistic, via `rectStats`) so a big gap
   that's just photo grain scores lower than the same-sized gap between two genuinely uniform
   regions. `buildIntegral()` builds a summed-area table of luminance and luminance² once per
   extraction so every window's mean/variance is an O(1) lookup (`rectStats`) instead of a
   pixel rescan — there are `N × h` of these queries per extraction. Otsu's method
   (`otsuSplit`, generalized from "split 256 brightness levels" to "split any set of
   non-negative numbers") then finds where, across the *whole* photo's score distribution,
   "flat" ends and "edge" begins — the same trick the old brightness threshold used, just
   applied to "how step-like is this" instead of "how bright is this," which is what makes it
   agnostic to which side is the bright one. Per column, the first row (scanning from the sky
   side down) whose score clears that cutoff wins — "first" rather than "strongest overall"
   is what keeps a photo from latching onto a sharper-but-wrong edge further down the column,
   like the lake reflection case above; the mountain-above-treeline case is why it's "first"
   rather than "only": a column can have a weak qualifying edge higher up and a much
   stronger one lower down, and the weak one is usually the real ridge. The whole elevation
   array still gets a 5-wide median filter (`medianSmooth`) afterward for single-column noise
   (a bird, a lens artifact). The old brightness-threshold's `CONFIRM`-consecutive-rows check
   isn't needed any more — a real edge's score rises and falls smoothly across the averaging
   window rather than spiking on one row, unlike raw per-pixel brightness, so single-row
   crossing is already stable. The "Sensitivity" slider now scales the Otsu cutoff directly
   (higher accepts a weaker edge) instead of nudging a brightness level, and "Flip it" now
   picks the *last* qualifying row instead of the first (scanning from the ground side up)
   rather than assuming an inverted brightness polarity, since the detector no longer cares
   which side is darker. Sample count, sensitivity, and flip are all still user-adjustable via
   the "fine-tune" disclosure, but the goal of this rewrite was for the default, un-tuned
   result to be right far more often.
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

## Monetization

- AdSense: script tag + `google-adsense-account` meta tag in `<head>`, publisher id
  `ca-pub-5936916546458743` (matches the rest of the site — don't change independently).
- Zazzle affiliate: a plain link (not an API integration) to a pre-made mug template
  with `?rf=<ambassador_id>` cross-promotion param. User manually downloads the PNG and
  uploads it to Zazzle themselves — there's no server-side handoff of the design.

## Testing changes

No test suite — this is a static page. Verify changes via a local static server (root-relative
`/nav.js` and `/theme.css` links mean opening `index.html` directly over `file://` won't pick
them up — serve the repo root, e.g. `python -m http.server`, and browse to `/lab/ridgeline/`).
Run the golden path: upload a photo → check the auto-extracted outline looks right → render
steps/video → download. Check both the "Line" and "Stack" merch styles, and both video styles
(sequential summation, epicycle arms), since they share the reconstruction math but have
separate drawing code paths.
