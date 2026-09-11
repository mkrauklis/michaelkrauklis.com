# Echo State

Reads whatever text a visitor types, one character at a time, through a small untrained
recurrent network (an *echo state network* — the tool is named directly for the architecture).
Traces the path the network's hidden state takes through space as a unique line for that exact
input, shows the same weights lighting up live in a node-link diagram, exports either as a
high-res print, and — the tool's centerpiece — packs the whole quantized model plus the typed
text into a QR code that, when scanned, reopens this same page and replays the exact reservoir
live. Originated as a prototype built and reviewed in a separate `claude.ai` conversation before
being adapted into this repo's conventions.

## Architecture

`index.html`, plus the two site-wide shared files (`/nav.js`, `/theme.css` — see root
`CLAUDE.md`'s "Site-wide shared files"), plus one page-local dependency: `qrcode.min.js`
(davidshimjs/qrcodejs v1.0.0, MIT license, vendored unmodified rather than loaded from a CDN —
see "Why vendor the QR library" below). No build step, no bundler; everything else is inline
`<script>`. Runs entirely client-side — no text is ever sent anywhere, which is also what makes
the QR-replay feature honest: the model has to travel entirely inside the code because there is
no server to fetch it from.

Accent color is `var(--teal)` — `theme.css`'s teal token, unclaimed by any other tool at the time
this shipped (Ridgeline: default amber, Afterimage: violet, Vectis: a custom cyan `#57c2e0`,
Neural Viaduct: a custom rust `#c9553d`). `theme.css` doesn't define `--teal-dim`, so this page
defines its own `--accent-dim: #355952` locally, the same way Vectis and Viaduct each define a
custom `--accent-dim` for their own non-token accent colors.

## Why vendor the QR library instead of a CDN import

The site's default is no external JS at runtime; Vectis's `transformers.js` CDN import is the
one documented exception, justified there by the library being far too large to vendor. qrcodejs
is the opposite case — about 20KB unminified from cdnjs, MIT-licensed, and simple enough that
vendoring a static local copy costs nothing and removes a runtime dependency on cdnjs.cloudflare.com
entirely. Given this tool's own pitch is "nothing about your text is uploaded anywhere," not
depending on a third-party host at runtime is the more consistent choice, not just a style
preference — do the same for any future small utility library before reaching for a CDN link.

## The reservoir (`buildReservoir`, `runForward`)

Weights are **not trained** — `buildReservoir(seed, H, targetRadius)` fills `Win` (H×6) and
`Wrec` (H×H) with a seeded PRNG (`mulberry32`, the same tiny deterministic generator Vectis and
Ridgeline don't need but this tool does, since "reroll" has to mean "a new but reproducible-if-
reseeded reservoir," not literal unrepeatable `Math.random()` noise for the *generation* step —
though the reroll button itself does pick its next seed via `Math.random()`), then rescales
`Wrec` so its spectral radius (estimated via power iteration in `specRadius()`, 80 iterations
against a fixed non-degenerate starting vector — this is a real numerical estimate, not a
closed-form shortcut) hits the slider's chosen target exactly. Each character updates the hidden
state by `h_t = tanh(Win·x_t + Wrec·h_{t-1} + b)` (`runForward()`), where `x_t` is a 6-bit code
(`code6()`) for that character's index into a 44-symbol alphabet (`ALPHABET` — a–z, 0–9, space,
and `. , ! ? & ' -`; anything else maps to space via `charIndex()`). Six bits rather than a
44-wide one-hot is deliberate: it keeps `Win` at `H×6` instead of `H×44`, which is most of why
parameter count (`H×(H+7)`, stated directly in the deep-dive) stays small enough to fit a QR code
at all, and — because there's no training objective either encoding needs to be friendly to —
there's no accuracy cost to picking the more compact one.

**Spectral radius is the one control that actually changes the network's *behavior*, not just its
starting point.** Below ~0.55 the state contracts to a fixed point fast, so the trace collapses
toward one spot and the network effectively forgets everything but the last couple of characters.
Above ~1.05 small differences compound instead of decaying, so the trace can swing unpredictably
and two similar-looking inputs can end up nowhere near each other. The useful range is in between
— this is stated directly in the deep-dive (`regimeFor()`'s three labels: "fixed point — short
memory," "rich dynamics," "saturating / chaotic") rather than left for a visitor to discover only
by accident.

## Projection (`pca2`, `project`) — hand-rolled, matching the rest of the site's "no black box" rule

`H` can be 2, 4, 8, or 16. At `H=2` the raw hidden state already *is* a 2D point (`project()`
returns it directly, and the caption says so explicitly — "plotted directly, no projection
needed"). For every other `H`, `pca2()` computes the top two principal components of the actual
sequence of hidden states from **this specific run** — covariance matrix by hand, top eigenvector
via power iteration (`powerIterVec`, 120 iterations against a fixed start vector, same pattern as
`specRadius`), second eigenvector via one deflation step against the first. This is a real,
per-input PCA, not a fixed or precomputed projection — the 2D shape a 16-dimensional reservoir
traces for one piece of text has no reason to align with the shape a different piece of text
traces in the same space, which is part of why two inputs' traces look genuinely different rather
than just offset copies of each other.

## The activation diagram vs. the structure diagram — same layout, different meaning

`computeNodeLayout(H)` is shared by both `renderDiagram()` (the deep-dive's static structure
view) and `renderActivationDiagram()` (the live view in step 02) specifically so the two read as
two views of *the same* network rather than unrelated drawings. The static one fills each hidden
node by its fixed **bias** (the network's permanent wiring, unaffected by input) and never
changes as playback moves; the live one fills each hidden node by **`states[state.t]`** — the
actual current activation — and lights up whichever of the six input nodes correspond to the
character just read. Edge color is always sign (teal positive, rust negative) and opacity is
always magnitude relative to that weight matrix's own max absolute value, in both diagrams —
picking a fresh color/opacity convention for one and not the other would break the "same network,
two views" framing this depends on.

## The QR payload: `buildPayload()` / `decodePayload()` — a real round trip, not a decorative export

The naive version of "put the model in a QR code" would just be a screenshot or a link to a
hosted JSON blob — neither actually satisfies "nothing lives on a server." `buildPayload(model,
H, text)` instead quantizes every real weight (`Win`, `Wrec`, `bias` — flattened in that fixed
order) to a single shared int8 scale derived from the run's own max absolute weight
(`scaleByte = round(maxAbs*40)`, clamped to a byte), packs `[H, scaleByte, ...quantized weights,
...UTF-8 text bytes]` into one `Uint8Array`, and base64url-encodes it into the page's own URL as
`?d=`. Because there's no "correct" trained version being approximated here — an untrained
reservoir's weights don't have a ground truth to preserve fidelity to — int8 quantization loses
nothing that matters: **the quantized weights are the model**, not an approximation of some other
canonical one.

`decodePayload()` is the exact inverse, and `initFromUrl()` (bottom of the main script, before
the first `recompute()`) runs it against `location.search` on every page load. If a `?d=` is
present and decodes cleanly, its `{H, model, text}` become the starting state — `state.replayModel`
holds the exact rebuilt weight matrices, and `recompute()` uses `state.replayModel ||
buildReservoir(...)` so a replay never regenerates a fresh reservoir by accident. This is what
makes the QR code's promise literally true: scanning it doesn't link to a recording or a static
image, it reopens this same page and reruns the identical forward pass, byte for byte, using
weights that traveled entirely inside the URL. `#replayNote` tells the visitor this is happening;
typing new text while replaying keeps the borrowed weights and runs the new text through them
(the same "personality," a new input) — but touching H, spectral radius, or "reroll" calls
`exitReplay()` first, since none of those make sense against a fixed foreign reservoir, and
silently ignoring them would leave the page looking broken instead of just switching back to
normal generative mode.

If you ever change `buildPayload()`'s byte layout, `decodePayload()` must change with it in the
same commit — they're the two halves of one format, and a mismatch would silently produce
garbage weights rather than an error (a decoded `H` out of a sane range is the only sanity check
`decodePayload()` performs, via the `H < 1 || H > 64` guard).

## Print export (`renderPrint`, `drawNetworkPrint`)

Two styles, matching the pattern established in step 02: "Trace" re-renders the same PCA path
used on-screen at high resolution (`computeLayout` + Catmull-Rom, identical math, just a bigger
canvas); "Network — final frame" (`drawNetworkPrint`) draws the same node-link structure as the
live diagram but frozen at the last character read, monochrome in whatever accent color the
visitor picked — a clean single-hue "lit circuit" meant for merch, not for reading exact values
(unlike the on-page diagrams, which use sign-colored edges specifically so a visitor can tell
positive from negative at a glance). Download is a direct `canvas.toDataURL('image/png')` +
synthetic anchor click, the same pattern Ridgeline's merch download uses — no capability check or
bespoke save flow, since this is a plain static page rather than a `claude.ai`-hosted artifact.

## Zazzle link is a placeholder, not a verified template

`#shareSection`'s mug template URL is the **same product ID already used by Ridgeline's own mug
CTA** (`256206116898885602`, with the same `?rf=` ambassador parameter), reused deliberately as a
known-working stand-in rather than a dedicated Echo-State template — the original prototype's own
"design your own t-shirt" link reused this same numeric ID under a *different* product slug, which
is almost certainly wrong (Zazzle product IDs are per-listing; a mug and a t-shirt would not
share one). Replace this with a real Echo-State-specific template link once one exists; don't
carry forward the prototype's mismatched slug/ID pairing if you ever look at its version again.

## Testing changes

No test suite — static page. Verify via a local static server (root-relative `/nav.js` and
`/theme.css` mean `file://` won't pick them up). Golden path: type text → confirm the trace (step
02) and both node diagrams update live → scrub and play back → change H and spectral radius,
confirm the regime label changes at the documented thresholds → reroll → switch print style/color/
background and download a PNG → confirm the QR code renders and its ledger byte counts look sane
→ **scan or manually open the QR's URL (or just append its own `?d=...` to the address bar) and
confirm the page reloads with `#replayNote` visible, the same H and text pre-filled, and the trace
identical to what was exported** — this last step is the one thing that would be easy to ship
silently broken (a bug in `decodePayload()` alone wouldn't show up anywhere except on an actual
round trip), so don't skip it after touching `buildPayload()`, `decodePayload()`, or `initFromUrl()`.
Also confirm touching H/radius/reroll while replaying calls `exitReplay()` (the note disappears)
rather than silently fighting the fixed replayed weights.
