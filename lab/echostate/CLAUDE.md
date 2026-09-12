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
— this is stated directly next to the slider (`regimeFor()`'s three labels: "fixed point — short
memory," "rich dynamics," "saturating / chaotic") rather than left for a visitor to discover only
by accident. **The slider itself lives in the "How this actually works" deep-dive**, not the main
personalize step — direct feedback was that it's a "how it's made" control, not a "give it
something to remember" one, so section 01 now holds only the text input and the reroll button;
the slider is still fully live (still calls `recompute()` on `input`), it's just presented right
alongside the explanation of what it does.

## Reservoir size is fixed at 8 (`FIXED_H`)

There used to be a 2/4/8/16 toggle in section 01; direct feedback simplified this to a single
fixed size, so every newly-generated reservoir here uses `FIXED_H = 8` and there's no H control
in the UI at all. `state.H` still exists as a real field (not hardcoded everywhere `H` appears)
specifically so a **replayed** link can carry a different `H` — `decodePayload()` stays fully
general (any `H` 1–64) for backward compatibility with links generated before this change, and
every H-dependent function (`computeNodeLayout`, `project`, `renderUnroll`, etc.) still takes `H`
as a parameter rather than assuming 8. `exitReplay()`-adjacent handlers (reroll, the spectral
radius slider) explicitly reset `state.H = FIXED_H` when breaking out of replay, so leaving a
replayed odd-`H` link never gets stuck applied to a freshly generated reservoir.

## Projection (`pca2`, `project`) — hand-rolled, matching the rest of the site's "no black box" rule

At `H=2` the raw hidden state already *is* a 2D point (`project()` returns it directly) — this
path only matters for an old replayed link, since new reservoirs are always `H=8` now. For every
other `H`, `pca2()` computes the top two principal components of the actual sequence of hidden
states from **this specific run** — covariance matrix by hand, top eigenvector via power
iteration (`powerIterVec`, 120 iterations against a fixed start vector, same pattern as
`specRadius`), second eigenvector via one deflation step against the first. This is a real,
per-input PCA, not a fixed or precomputed projection — the 2D shape doesn't have to align between
two different pieces of text, which is part of why two inputs' traces genuinely differ rather than
being offset copies of each other. The plain-language explanation of this used to sit as a caption
directly under the trace in step 02 (dynamically worded per-`H`); direct feedback moved it into
the deep-dive as fixed prose instead, alongside the formula it's explaining.

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

## The unrolled-recurrence diagram (`renderUnroll()`) — answering "do we unroll it, or update in place?"

Direct question worth answering precisely, because both readings are half-right: `runForward()`
is a **plain loop**, not a literal unrolled computation graph — there's no backpropagation-
through-time, no training, no graph object at all. Each step computes a brand-new `newH` from the
previous `h` (`h_t = tanh(Win·x_t + Wrec·h_{t-1} + b)`) and only ever looks one step back; it is
not "adding to" or mutating the prior state in place. But it's also not *only* keeping a running
current value the way a naive in-place update might — `states` (the array `runForward()` returns)
keeps **every** intermediate state, one per character, specifically because the trace/scrubber/
playback UI in step 02 needs the full history, not just wherever the loop currently is.

`renderUnroll()` makes this concrete rather than leaving it as prose: it samples up to 6 of
`cached.states` (all of them, if the text is short enough that there are 6 or fewer), evenly
spaced via `Math.round(i*(n-1)/(maxFrames-1))` so both endpoints are always included, and draws
each sampled state as its own column of `H` small circles (same sign/magnitude color convention
as `renderActivationDiagram()` — teal positive, rust negative), labeled above by the character
that produced it (`(start)` for the initial all-zero state), connected to the next column by a
plain arrow. The arrow is the entire point: it's the only new information flowing between
columns, which is what "reaches back one step" actually looks like. This diagram always reflects
the *current* full text regardless of the step-02 scrubber's position — it's illustrating the
general mechanism, not tracking playback.

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
(the same "personality," a new input) — but touching the spectral radius slider or "reroll" calls
`exitReplay()` first (and resets `state.H` back to `FIXED_H`), since neither makes sense against a
fixed foreign reservoir, and silently ignoring them would leave the page looking broken instead of
just switching back to normal generative mode.

The share section used to show a full byte-ledger table (`renderLedger()` — hidden units,
parameter count, weight/message bytes, estimated QR version, etc.) next to the QR code; direct
feedback ("get rid of all those metrics, just show the QR code") removed that table and the
`estimateVersion()`/`QR_CAP_M` helpers that only existed to feed it. `cached.url` is still computed
every `recompute()` (now inline, right after `buildPayload()`), since `renderQR()` needs it — it's
just no longer displayed as a table. A `#downloadQrBtn` was added in its place: qrcodejs actually
renders to a hidden `<canvas>` and a *visible* `<img>` whose `src` it already set to that canvas's
own `toDataURL('image/png')` (confirmed directly — `#qrcode img` already carries a `data:image/png`
URL after `renderQR()` runs, no extra encode step needed), so the button just grabs that `<img>`'s
`src` directly, falling back to encoding the canvas itself only if qrcodejs's DOM ever changes.
This exists specifically to support printing the signature and the QR code as two separate mug
sides (see "Zazzle link" below) — a visitor needs the QR as its own file for that, not just as a
pixel graphic sitting in the page.

If you ever change `buildPayload()`'s byte layout, `decodePayload()` must change with it in the
same commit — they're the two halves of one format, and a mismatch would silently produce
garbage weights rather than an error (a decoded `H` out of a sane range is the only sanity check
`decodePayload()` performs, via the `H < 1 || H > 64` guard).

## Print export (`renderPrint`, `drawNetworkPrint`) — "Trace" and "Fingerprint"

Two styles, matching the pattern established in step 02: "Trace" re-renders the same PCA path
used on-screen at high resolution (`computeLayout` + Catmull-Rom, identical math, just a bigger
canvas); "Fingerprint" (`data-style="network"` internally — not renamed, to avoid touching the
payload/state wiring — `drawNetworkPrint()`) draws the same node-link structure as the live
diagram but frozen at the last character read, monochrome in whatever accent color the visitor
picked — a clean single-hue "lit circuit" meant for merch, not for reading exact values (unlike
the on-page diagrams, which use sign-colored edges specifically so a visitor can tell positive
from negative at a glance). The button label reads "Fingerprint" (renamed from "Network — final
frame" so both options read as two kinds of *signature*, matching the "Pick a Trace or Fingerprint
— that's your unique signature" framing added to the section's own copy); the downloaded filename
follows the same rename (`echo-state-trace.png` / `echo-state-fingerprint.png` — keep these two in
sync if the label ever changes again, they drifted out of sync once already going into this
change). Download is a direct `canvas.toDataURL('image/png')` + synthetic anchor click, the same
pattern Ridgeline's merch download uses — no capability check or bespoke save flow, since this is
a plain static page rather than a `claude.ai`-hosted artifact. "Print the text underneath" now
**defaults off** (was on) — a clean, caption-free signature is what a visitor wants for the
two-sided-mug idea below.

## Two Zazzle links, two different jobs — don't conflate them

There are now two separate Zazzle links on this page, and they serve different purposes:

- **The hero CTA** ("Want to see what this looks like as a gift? Here's an example on a mug,"
  right under the intro paragraph) links to a **real, live, specific product** — the
  "Automate the C-Suite First" mug (`automate_the_c_suite_first_mug-256161326676791673`, with
  the site's standard `?rf=238054754631086278` ambassador param). This is a curated, pre-made
  example meant to sell the *idea* of "you can put this on a mug" at a glance, before a visitor
  has even touched the tool — it is not something a visitor's own input feeds into. This is the
  first tool on the site to get this pattern; if other tools get their own real product links,
  put them in the same spot (top of the hero, right after the intro paragraph) for consistency.
- **`#shareSection`'s mug template link** (below) is the generic "make it a gift" DIY flow — a
  visitor downloads *their own* PNG/QR output and uploads it to a blank "upload your own image"
  template themselves. This one is still a placeholder (see below), unrelated to the real product
  link above. Don't merge these two into one link or assume fixing one fixes the other — the hero
  link points at a finished product; the share-section link points at a blank template a visitor
  personalizes themselves.

## Zazzle link is a placeholder, not a verified template

`#shareSection`'s mug template URL is the **same product ID already used by Ridgeline's own mug
CTA** (`256206116898885602`, with the same `?rf=` ambassador parameter), reused deliberately as a
known-working stand-in rather than a dedicated Echo-State template — the original prototype's own
"design your own t-shirt" link reused this same numeric ID under a *different* product slug, which
is almost certainly wrong (Zazzle product IDs are per-listing; a mug and a t-shirt would not
share one). Replace this with a real Echo-State-specific template link once one exists; don't
carry forward the prototype's mismatched slug/ID pairing if you ever look at its version again.

The section's copy now explicitly suggests a two-sided mug: the signature (caption off) on one
side, the QR code — via `#downloadQrBtn`, see above — on the other, so the mug stays scannable
without the code interrupting the design. This was direct user feedback, not a guess at what's
popular for mugs generally; don't walk the suggestion back to a single-image mug without reason.

## Testing changes

No test suite — static page. Verify via a local static server (root-relative `/nav.js` and
`/theme.css` mean `file://` won't pick them up). Golden path: type text → confirm the trace (step
02) and both node diagrams update live → scrub and play back → open the deep-dive and move the
spectral radius slider, confirm the regime label changes at the documented thresholds and the
trace/diagrams above recompute → confirm the unrolled-recurrence diagram shows one column per
sampled state with a plausible arrow chain and its own text matches the actual sample count →
reroll → switch print style (Trace/Fingerprint)/color/background, confirm "print the text
underneath" starts unchecked, and download a PNG (check the filename matches the style: `-trace`
vs. `-fingerprint`) → confirm the QR code renders with **no ledger table** next to it → click
"Download QR as PNG" and confirm a real PNG saves (not an empty file) → **scan or manually open
the QR's URL (or just append its own `?d=...` to the address bar) and confirm the page reloads
with `#replayNote` visible, the same text pre-filled, and the trace identical to what was
exported** — this last step is the one thing that would be easy to ship silently broken (a bug in
`decodePayload()` alone wouldn't show up anywhere except on an actual round trip), so don't skip
it after touching `buildPayload()`, `decodePayload()`, or `initFromUrl()`. Also confirm touching
the spectral radius slider or reroll while replaying calls `exitReplay()` (the note disappears and
`state.H` resets to `FIXED_H`) rather than silently fighting the fixed replayed weights.
