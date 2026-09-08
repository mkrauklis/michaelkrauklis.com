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
   sky/land boundary. This went through three real designs before landing on one that held up
   against actual test photos (not just synthetic ones — see the note at the end of this
   section on why that distinction mattered here):
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
   - **The current design**: sample a reference band from the very top strip of the photo (sky,
     almost always), and walk each column down looking for where the brightness *permanently*
     leaves that reference, not just locally steps away from its immediate neighbors. The
     reference is summarized by **median and MAD** (median absolute deviation), not mean and
     standard deviation (`bandMedianMad()`) — a sky that's part blue, part bright cloud is two
     populations in one sample, and a mean/stdev gets dragged toward the cloud population
     enough to make a second, brighter cloud further down the frame look like it's still
     within range. Median/MAD aren't dragged the same way. Per column, a sliding-window
     median (`histMedian()`, over an incrementally-updated 256-bin luminance histogram — an
     exact sort per window, at `N` columns × up to `h` rows per extraction, was too slow) gives
     a z-score against that reference at every row; the boundary is the first row (scanning
     from the sky side down) where *most* of a lookahead window — not just that one row — has
     cleared the z-score cutoff. "Most, not all" absorbs a single sunlit rock face or gap that
     would otherwise read as a brief return to "sky." "First, not strongest" is what makes a
     weak-but-real ridge higher up beat a stronger edge lower in the same column, like a lake
     reflecting the same ridge a second time near the bottom of the frame.
   - The **"Sensitivity" slider** scales how many MADs from the reference counts as "no longer
     sky" (higher accepts a smaller departure). **"Flip it"** now means "trace the boundary as
     seen from the far side" — it samples the reference band from the *bottom* strip instead
     and scans upward — rather than assuming an inverted brightness polarity, since the
     detector no longer cares which side is darker. The elevation array still gets a 5-wide
     median filter (`medianSmooth`) afterward for single-column noise (a bird, a lens
     artifact) on top of all of this.
   - **Why synthetic tests weren't enough**: the second design above was validated against
     flat-fill synthetic images (a clean two-tone ridge, a low-contrast ridge above a stronger
     treeline) and looked correct — those tests just don't have the thing that broke it, a
     genuinely textured/dramatic sky. The fix came from running the actual code against real
     test photos (a phone shot with light cirrus clouds, a dramatic stock photo with heavy
     cumulus and a lake). If you change this algorithm again, validate against a real, busy-sky
     photo, not only clean synthetic fills — they will not catch this failure mode.
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
     correct one: a *lenient* sensitivity triggers on weak evidence, and a cloud-vs-cloud
     transition high in a busy sky can be just as consistent column-to-column as the real ridge
     is, so "lowest roughness wins" outright would happily pick that early, wrong answer over a
     later, right one at a stricter setting. So it finds the best roughness achievable first,
     then walks candidates strict-to-lenient and stops at the first one already close to that
     best (see the `+1.5` tolerance) — roughness rules out sensitivities that are clearly too
     strict (visibly erratic), it isn't used to go hunting for the single smoothest result on
     offer.
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
separate drawing code paths.
