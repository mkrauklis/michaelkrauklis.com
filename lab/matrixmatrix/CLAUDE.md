# Matrix Matrix

What a matrix is, a few things it can represent, and exactly what multiplying two of them means —
worked by hand from a 2×2, up to a 4×4, then across two matrices of genuinely different shapes
(1×2 · 2×4), plus an interactive shape checker to test the general rule. Built directly from
[[project-lab-future-topics]]'s own backlog: matrix multiplication was flagged as a prerequisite
concept Attention leans on without fully teaching, during that tool's own "math-inclined 8th
grader" calibration pass — this is that gap, closed as its own standalone piece rather than a
bigger inline aside on Attention's page.

## Direct request, structured exactly as given

The build order was specified directly, not inferred: "Start with what is a matrix. Then what a
matrix could represent. Then what multiplying matrices means with an example. 2x2 to start. Then
4x4. Then heterogenous, maybe 1x2 and then 2x4." Stages 1-3 map onto that literally, in that order,
including the specific example shapes named (1×2, 2×4). "Call the lab Matrix Matrix" set the name
directly — a double meaning (the math object, and "The Matrix" as pop-culture shorthand for the
same idea) rather than a name that had to be brainstormed.

## The Zazzle plan changed mid-build, twice

Original direct request: "link to a funny zazzle design for matrix multiplication grounded in
industry humor." Searched Zazzle live for an existing one first (`matrix multiplication funny`,
`matrix does not commute`, `linear algebra funny`) rather than guessing a URL — nothing close to
actual industry humor turned up, just generic elementary-school "do the math" designs. Reported
that back rather than forcing a weak match. Direct follow-up: "Let's make our own," then
immediately refined further: "We just need the image, I can create the product." So this page
generates and exports the design (Stage 6) but deliberately has **no Zazzle link or CTA at all** —
the user is wiring up the actual product separately. Don't add a generic "upload your own design"
template link here the way Attention/Hunch do unless asked — that would contradict the explicit
"we just need the image" scope.

## The design itself: a real, recognizable error message, not a generic joke

The generated PNG (`renderDesign`, Stage 6) mimics a genuine shape-mismatch error — `ValueError:
shapes (3,4) and (5,6) not aligned: 4 (dim 1) != 5 (dim 0)` — the exact style of error every
numpy/PyTorch/TensorFlow user has hit, styled like a terminal, under a deadpan "Make sure the inner
dimensions match" caption set in the site's own serif (Fraunces), mimicking a motivational-poster
format applied to a real practitioner's pain point. This is "industry humor" in the literal sense
requested — funny specifically to someone who has actually multiplied real matrices in code and
hit this exact wall — rather than a generic math pun. The `>>> attention_output = A @ B` prompt
line at the top is a deliberate, subtle nod back to why this tool exists at all (Attention's own
Query/Key/Value reshaping), without needing anyone to know that to get the joke.

## Accent color and the "A"/"B" hues

`--accent:#22d3ee` (electric cyan) — distinct from every existing tool's accent (amber, teal,
violet, rose/crimson, periwinkle-blue, sky-blue, rust, emerald green, fuchsia). Chosen for
saturation/vividness contrast against the existing muted teal (Echo State) specifically, the same
"distinguish by temperature/saturation, not just hue" reasoning Viaduct's rust and Inkling's
crimson already use against each other. `--a-color`/`--b-color` (cyan/amber) are separate,
non-brand constants used only to distinguish "this row belongs to matrix A" from "this column
belongs to matrix B" in the multiplication demos' highlight states — the same separation Tonus
keeps between its page accent and its two data-class colors.

## Three representations, chosen to also set up Stage 3

Stage 2 doesn't just list arbitrary examples — each one was picked because it pays off later:
- **A table of data** — the most literal reading, no interactivity needed, just establishes rows
  = things, columns = properties before anything gets more abstract.
- **A picture** — an 8×8 grid of the same numbers shown two ways at once (a plain number grid and
  actual colored squares from a hand-authored pixel-art heart), making concrete that "a matrix" and
  "a picture" can be the literal same object, not an analogy.
- **A transformation recipe** — a 2×2 matrix applied live to a square (rotate/scale/flip/shear
  presets), with the copy explicitly calling out that transforming *one point* is already a matrix
  multiplication (a 2×2 times a single point's 2 numbers) — this is the direct setup for Stage 3,
  which then shows exactly how that multiplication works and scales the same rule up to much
  bigger matrices.

## Hero canvas got real bracket notation, and the thumbnail follows it

The hero canvas originally drew each matrix as plain bordered boxes — no brackets. Direct feedback
on `lab/index.html`'s card icons ("the icons for these are stinkers... an icon of a matrix") landed
on the actual missing piece: bracket notation (`[ a b ; c d ]`) is the one glyph anyone who's seen
linear algebra actually recognizes as "a matrix" on sight, and a grid of bordered boxes reads more
like a spreadsheet or a game board. `drawBracket()` draws real bracket shapes (an open rectangle
missing its vertical middle — two horizontal ticks plus a vertical spine) on both sides of each of
the hero's three matrices, in that matrix's own color. This is a genuine improvement to the page
itself, not just a change made for the sake of a thumbnail.

`thumbnail.jpg` is a fresh, larger render of the same bracket-notation idea — a standalone 1280×1280
canvas showing just matrix A `[[2,0],[1,3]]` at export resolution, tightly cropped to its own real
content (via a numpy bounding-box check against the background color, not a hand-guessed crop
rectangle) and resized down to 640×640. Built as a one-off render sharing the hero's exact drawing
logic rather than reusing the hero canvas directly, since the hero draws all three matrices side by
side at page scale — a crop tight enough to read as an icon would have cut into B or C, and a crop
loose enough to include only A would have wasted most of the frame on empty background.

## A real closure bug, caught by testing every editable cell, not just the first

**The bug**: `renderGrid()`'s editable-cell branch declared `var input = document.createElement
('input')` inside the nested row/column loop, then wrapped only `(i, j)` in the loop's
closure-fixing IIFE — `input` itself was left as a plain `var`, function-scoped to the whole
`renderGrid` call. Every cell's `input` event handler closed over that *same* variable by
reference, not by value, so by the time any handler actually fired (after the entire grid had
finished rendering), `input` had already been reassigned three more times and pointed at the
*last* cell created. Editing any cell except the last one silently read and applied the last
cell's current value instead of the one actually being edited.

**How it was caught**: not by reading the code again, but by scripting the actual interaction —
set cell (0,0)'s value to 9 via a real `input` event, then read back all four cells' values.
Cell (0,0) showed `3`, matching neither the original value (`2`) nor the edit (`9`) — it was the
unedited value of cell (1,1), the last cell in render order, confirming the exact mechanism before
touching the fix. Fixed by adding `input` as a third parameter to the existing IIFE
(`(function(i, j, input){ ... })(i, j, input)`), giving every cell's closure its own correctly-
scoped reference — consistent with the file's existing all-`var` style rather than switching just
this one declaration to `let`. Re-verified after the fix: editing cell (0,0) and, separately, cell
(1,0), each updated only its own value and left the other three cells untouched, and the live
recomputed answer matched by hand.

**Why this matters as a lesson for this file specifically**: every one of the three multiplication
demos (2×2, 4×4, 1×2·2×4) and the shape explorer all route through this same `renderGrid()`
function — a bug here wasn't isolated to one demo, it would have silently broken editing on all of
them at once except the very last cell in each grid. Any future change to `renderGrid()`'s editable
branch should re-run this exact test (edit a non-last cell, confirm only that cell's value and the
recomputed answer change) before shipping.

## Revamp: a punchier design, and one fewer representation in Stage 2

Direct feedback, alongside Hunch and Attention both being demoted to the bottom of
`lab/index.html`: "especially matrix matrix and hunch, they need better artifacts that can end up
on t-shirts... they're too confusing still. Please make them simpler and more direct." The design
concept (a real shape-mismatch error under a deadpan caption) was already the right idea — the
execution wasn't landing it hard enough. Fixed by restructuring, not replacing: dropped the small
`>>> attention_output = A @ B` prompt line entirely (extra clutter competing with the actual joke),
made "ValueError" and the shape-mismatch text much larger and centered instead of small and
left-aligned in a corner, and kept "MAKE SURE THE INNER DIMENSIONS MATCH." as the punchline but in
bold sans instead of serif, matching the louder, more poster-like hierarchy the redesign was going
for — lead with the big thing, not the fine print.

**Stage 2 lost its third representation.** "A table of data" (a static 3×3 student-scores table)
was cut entirely — it was the least essential of the three: it didn't set up anything later on the
page the way "a picture" (real pixel-grid rendering) and "a set of instructions for reshaping
something" (the live transform demo, which directly previews Stage 3's own multiplication) both do.
Cutting it removed a full panel and its now-dead `table.data-table` CSS, and shortened the page
without touching anything Stage 3's three worked examples, Stage 4's shape checker, or Stage 5's
real-world tie-ins — all of which were explicit, specific requests in the original build and
weren't second-guessed here.

**The densest paragraphs were tightened** (Stage 3's lead, the "different shapes" callout, Stage
4's intro), and **"How this actually works" now starts collapsed** instead of open — it's the
"precise version" by its own subtitle, not something a first-time reader needs pushed in front of
them immediately.

## Testing notes

No test suite — static page. Verify via the local static server. Golden path: confirm the intro
3×2 grid, the data table, the pixel heart (both as numbers and as a real rendered picture), and the
transform demo's default identity square all render with no console errors → click each transform
preset and confirm the filled square visibly rotates/scales/flips/shears to match its own 2×2
matrix, shown live in the readout → in the 2×2 demo, click a non-default C cell and confirm the
correct row/column highlight and explanation appear, then edit a non-last A or B cell specifically
(the exact regression above) and confirm only that cell's value changes and the recomputed C and
explanation are correct → click "Randomize" on the 4×4 demo and confirm all 16 A and 16 B values
change together and C recomputes correctly → in the 1×2·2×4 demo, confirm the explanation text
correctly handles negative numbers in its product breakdown → in the shape explorer, confirm a
matching case shows a green "compatible" verdict and the correct output shape, and an intentional
mismatch (edit only one of the two "k" values) shows a red mismatch state with the actual two
differing numbers named in the verdict text → download the design PNG and open it directly to
confirm the error text, caption, and footer are all legible with no overlapping or clipped text →
confirm no console errors through the entire flow.

One environment-specific note, same as every other tool built in this environment: the automation
screenshot tool intermittently times out here ("the page did not finish rendering in time") — not
evidence of a real bug by itself. Verification was done via `get_page_text`/`find` for structure,
real clicks plus direct `.click()`/event dispatch via `javascript_tool` for interaction, and
decoding canvas `toDataURL()` output to real PNGs (the pixel heart, the hero canvas, the exported
design) to confirm actual rendered pixels rather than trusting a timed-out screenshot either way.
