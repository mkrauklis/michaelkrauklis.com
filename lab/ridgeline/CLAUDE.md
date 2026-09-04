# Ridgeline

Turns a photo of a skyline into a printable graphic (t-shirt/mug/poster style), and
optionally shows the Fourier-series math that builds the outline as an animated or
step-by-step reconstruction.

## Architecture

Single self-contained file: `index.html`. No build step, no bundler, no external JS
dependencies — inline `<style>` and `<script>` blocks only. Everything runs client-side
in the browser; no photo is ever uploaded to a server (this is stated in the page
footer — keep it true).

External resources loaded: Google Fonts (Fraunces, Inter) and the AdSense script tag.
That's the entire dependency surface.

## Pipeline (in order)

1. **Upload** — drag/drop or file picker, read via `FileReader` → `Image`.
2. **Silhouette extraction** — for each of `N` x-samples, scan down the column for the
   sky/land boundary. Threshold auto-picked via Otsu's method; sky-vs-land polarity
   auto-guessed by comparing top/bottom band luminance (`guessInvert`). Both are
   user-adjustable via the "fine-tune" disclosure if the auto-guess is wrong.
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

No test suite — this is a static page. Verify changes by opening `index.html` directly
(or via a local static server) and running the golden path: upload a photo → check the
auto-extracted outline looks right → render steps/video → download. Check both the
"Line" and "Stack" merch styles, and both video styles (sequential summation, epicycle
arms), since they share the reconstruction math but have separate drawing code paths.
