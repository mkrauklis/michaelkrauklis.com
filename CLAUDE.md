# Working conventions

- Always commit finished work — don't leave changes uncommitted at the end of a session.
- Commit directly to `main`. No feature branches or PRs needed for this repo unless asked.
  If we need more sophisticated commit workflows later, we'll adjust this.

# Site-wide shared files

Two root-level files are included, unmodified, by every page on the site (root `index.html`,
`lab/index.html`, and every tool under `lab/*/`) via a plain `<script src="/nav.js">` /
`<link rel="stylesheet" href="/theme.css">` tag — same mechanism, no build step, no bundler:

- **`/nav.js`** — the shared breadcrumb header. Reads `location.pathname` and the page's
  `<title>` to build its trail, so a new page needs nothing beyond the one `<script>` tag to
  get a correct breadcrumb. The `/lab/` section is displayed to visitors as **"The Concept
  Lab"** (breadcrumb crumb reads "Concept Lab"; page titles, headings, and footer credit lines
  use the full "The Concept Lab") — this was renamed from the generic "The Lab" specifically to
  signal what the tools are *for*: taking a concept and making it something you build, break,
  and watch work, not just read about. Keep this name consistent across `lab/index.html`,
  every tool's footer ("Part of The Concept Lab by Michael Krauklis."), and the homepage link —
  don't let a new tool drift back to the old generic name. The breadcrumb also renders a tiny
  18×18 `.site-nav-icon` (the site favicon, `/echo-state-trace-mkconceptlab.png`, browser-scaled
  down from its real 512×512 — no separate small asset needed for this) immediately to the left
  of "Michael Krauklis," linking `/` — direct request, on every page, since this is the one
  shared component every page already includes. If you edit `nav.js` and test it locally by
  reloading a page you've already loaded earlier in the same browser session, you may see the
  *old* cached copy despite the file on disk being correct — Chrome caches this exact URL
  aggressively across navigations since it's identical on every page. Confirm a real change
  actually shipped with a cache-busted fetch (`fetch('/nav.js?x='+Date.now(), {cache:'no-store'})`)
  or by injecting a fresh `<script src="/nav.js?x=...">` and checking its output, not just by
  reloading the page and eyeballing it.
- **`/theme.css`** — the shared palette, base typography, and the component styles the tool
  pages share almost verbatim (`.panel`, `.dropzone`, buttons, `header.hero`, disclosure/
  advanced, etc.). A page opts into a specific accent color by overriding `--accent`/
  `--accent-dim` in its own `<style>` block, placed *after* the `<link>` tag so the override
  wins. Deliberately has no blanket `max-width` on `<p>` — an earlier per-page version capped
  every paragraph at 62ch for readability, which looked lopsided wherever a lead paragraph sat
  directly above a full-width panel or grid (the text stopping well short of the panel below
  it). Letting paragraphs fill their container fixed that everywhere at once; if a specific
  paragraph genuinely wants to be a narrow, short teaser, give it an explicit class (see
  `p.lede` on the two content pages) rather than reintroducing a blanket cap.

## The homepage's LinkedIn / GitHub / Concept Lab cards

Direct request: give both of root `index.html`'s `.links` cards a real icon, "whatever you have
permission to use" for LinkedIn (a third, GitHub, was added later the same way). Every card is a
single clickable `<a class="card">` (icon + heading + description all one link target, matching
the `.writing-item` pattern already used further down this same page) rather than just the `<h2>`
text being a link — `.card:hover`'s amber border and `.card h2`'s explicit `color:var(--ink)`
mirror `.writing-item`/`.writing-item h3` exactly, for the same reason: once the *whole* card is
the link, the heading shouldn't look like a separate, differently-styled link floating inside it.
`.links` is a plain wrapping flexbox (`flex:1; min-width:220px` per card) specifically so a third
card could be added later without needing any layout changes — it wasn't built assuming exactly
two cards.

- **LinkedIn**: a plain inline SVG — a rounded `#0A66C2` (LinkedIn's own brand blue) square with
  a bold white "in" wordmark — rather than an external icon font or a fetched brand-asset file.
  Simple enough to hand-draw exactly, keeps the site's zero-runtime-icon-dependency posture, and
  LinkedIn's own brand guidelines permit using their mark to link to a real profile, which this
  is.
- **GitHub**: same convention, a rounded `#181717` (GitHub's own brand black) square with the
  real, widely-reproduced Octocat mark path (the same path data used by, e.g., the simple-icons
  project) traced as a single inline `<path>`, not a screenshot or a fetched asset. Links to
  `https://github.com/mkrauklis` — verified live (page title read "mkrauklis (Michael Krauklis) ·
  GitHub") before linking it, same discipline as every other external link added to this site.
- **The Concept Lab**: `/lab-mosaic.jpg`, a 2×2 composite built from four of the *actual* tool
  thumbnails already living under `lab/*/thumbnail.jpg` (Inkling, Echo State, Afterimage,
  Ridgeline — picked for visual variety at small size, not the newest four; Neural Viaduct's
  thumbnail reads as a fairly uniform orange blur this small, and Vectis's is mostly small text
  labels, neither of which survives a 64px card icon legibly) — real tool output, the same
  "build it from what the tool actually produces" rule `lab/index.html`'s own thumbnails follow,
  just tiled instead of single. Built once, offline, with a small Pillow script
  (`ImageOps.fit` to crop each source thumbnail to a square tile, pasted onto a `#10131a`-filled
  canvas with a 12px gap between tiles so two adjacent near-black thumbnails don't visually merge
  into one) — not regenerated automatically, so if `lab/index.html`'s own thumbnail set changes
  meaningfully, regenerate this file by hand rather than assuming it stays representative forever.
  `.card-icon` (40×40, `object-fit:cover`) is the shared sizing/shape class for every card's
  icon; the mosaic `<img>` additionally carries an `onerror="this.remove()"` fallback (the
  inline LinkedIn SVG can't fail to load, so it doesn't need one), identical in spirit to
  `lab/index.html`'s own `.tool-thumb` handling — a missing file collapses back to a clean
  icon-less card rather than a broken-image glyph.

**Layout: icon sits inline with the heading, not beside the whole card.** The original layout put
each card's icon in a 64×64 side column next to a `.card-body` holding both heading and
description, row-flex, vertically centered. That worked with two cards but broke once a third
(GitHub) landed in the same row — the icon column ate enough width from the text column that
"The Concept Lab" wrapped into three cramped lines (direct report: "the text on the main links...
is way too big and is throwing off the layout"). Fixed by restructuring `.card` to a column-flex
layout with a `.card-head` row (icon + `<h2>` together, `align-items:center`, shrunk to 40×40)
followed by the `<p>` description as a full-width sibling below it — the description no longer
has to share horizontal space with the icon at all, and the icon only ever competes with one
short heading line instead of the whole card's text. `.card-body` (the old text-column wrapper)
was removed entirely once nothing needed it.

The Concept Lab card's own description used to name a specific example tool ("for example,
Ridgeline, a Fourier-transform mountain silhouette reconstructor") — caught in a UX review as a
staleness trap: it named the *oldest* tool, not the newest or most representative, and would need
a manual update every time a new tool shipped if left as a specific example at all. Reworded to
describe the collection generically ("real backpropagation, real Fourier transforms, real
embeddings, running entirely in your browser") so it stays accurate without needing to change
every time `lab/index.html`'s own list grows.

## The homepage's certification badges link to real, verified individual pages

The six `.cert-badge` icons used to be plain `<div>`s with no link at all — everything rolled up
to one generic "See all certifications on LinkedIn" line below the grid. UX review flagged this:
a recruiter or client wanting to verify one *specific* cert had no faster path than the LinkedIn
aggregate page. Each badge is now its own `<a class="cert-badge">` pointing at that exact
certification's real Credly verification page (`https://www.credly.com/badges/<id>`) — fetched
directly from the real badge wallet at `https://www.credly.com/users/michael-krauklis.05da83e9`
and spot-verified (loaded one badge page and confirmed it publicly shows "This badge was issued
to Michael Krauklis" with the matching issue/expiry dates, no login required) before wiring any
of them in, not guessed from a naming pattern. The six IDs, in the order they appear on the page:

| Badge | Credly badge ID |
|---|---|
| Machine Learning – Specialty | `10dc1565-9938-4a4e-a2b9-199023cec16c` |
| Machine Learning Engineer – Associate | `f537964d-1c37-41ba-930a-2f9fd47fceb4` |
| AI Practitioner | `224fe209-6e83-41a5-9a1e-1dd6607e76bf` |
| Data Engineer – Associate | `84e87661-b3d1-4a4d-b1e2-02cd156d8018` |
| Solutions Architect – Associate | `b9493f91-0740-4290-a410-44d0e83cba6c` |
| SysOps Administrator – Associate | `df3b27f7-9c81-44fe-aa1d-328955b041c6` |

The Credly wallet has more badges than these six (an early-adopter variant of two of these, an
expired older Solutions Architect and Machine Learning Specialty, plus DevOps Engineer
Professional, Developer Associate, Cloud Practitioner, and Data Analytics Specialty) — the
homepage only ever showed this curated set of six, and this change didn't expand that; it just
made the six already shown individually verifiable instead of only reachable in aggregate. If the
curated set ever changes, re-fetch the wallet rather than assuming an ID is still current — early-
adopter and expired badges sit right next to the current ones in the same list and are easy to
grab by mistake if you're not checking each one's own issue/expiry dates against the wallet page.

When adding a new lab tool, link both files and rely on `theme.css`'s defaults before writing
new CSS — copying a whole `<style>` block from an existing tool page (the old pattern) is how
the palette and the paragraph-width bug drifted out of sync across pages in the first place.

`lab/index.html`'s tool list is ordered newest-first, top to bottom (currently Inkling, Echo
State, Neural Viaduct, Vectis, Afterimage, Ridgeline) — add a new tool's card at the *top* of
`.tool-list`, not the bottom. Each card also gets a `/lab/<tool>/thumbnail.jpg` — build it from
the tool's own real output, not placeholder art, but crop it to a **tight square around the
actual content** before saving it (no padding/letterboxing baked into the file — if the tool's
own canvas export has empty margin around the interesting part, crop that out first). It's
displayed as a small fixed-size icon (`.tool-thumb`, 96×96, `object-fit:cover`, a slight
`border-radius`) sitting to the *right* of each card's text and vertically centered via `.tool`'s
`align-items:center` — deliberately a consistent fixed size across every card rather than scaled
to that card's own (variable) text height, which is what an earlier version tried and which
needed a whole measure-and-apply script to work around a genuine flexbox limitation (a flex
item's stretched cross size can't drive its own aspect-ratio-derived main size in one layout
pass). A fixed icon size sidesteps that entirely — don't reintroduce per-card height-matching
without a real reason. `onerror` on the `<img>` removes the whole `.tool-thumb-link` (not just
the image) if `thumbnail.jpg` doesn't exist yet, collapsing back to a clean text-only card.

A third shared file, **`/echo-state-trace-mkconceptlab.png`**, is the site's favicon — a real
Echo State trace of the text "MKConceptLab" (default reservoir: seed 42, H=8, spectral radius
0.9), not a hand-drawn icon. Every page's `<head>` links it directly (`<link rel="icon"
type="image/png" href="/echo-state-trace-mkconceptlab.png">`, placed right before the
`theme.css` link) rather than relying on the browser's automatic `/favicon.ico` fallback. If the
site ever gets a different mascot/favicon, update all 9 pages' `<link>` tags together in one
commit (root, `lab/index.html`, and 7 tools as of this writing — check `lab/index.html`'s own
newest-first list for the current count) — there's no shared head-include mechanism, so this one
tag is duplicated per page the same way the SEO meta block below is.

# SEO

The live site is served from **`michaelkrauklis.me`**, not `.com` — the git remote's name is
`.com` but the actual canonical domain is set by the root `CNAME` file. Always use `.me` in
canonical URLs, `og:url`, and the sitemap; don't assume the domain from the repo name.

Every page (root `index.html`, `lab/index.html`, every tool) carries the same block of tags right
after `<title>`, before the `theme.css` link: `<meta name="description">`, `<link
rel="canonical">`, `og:type`/`og:site_name`/`og:title`/`og:description`/`og:url`/`og:image`, and
`twitter:card`/`twitter:title`/`twitter:description`/`twitter:image`. `og:title`/`og:description`
duplicate the meta description and `<title>` rather than being written separately — there's no
reason for them to diverge on a site this size.

**`og:image`/`twitter:image` were added site-wide, in one pass, across all 9 pages** — this was
deliberately held off on until it could be done consistently everywhere at once, not one page at
a time (see git history for the earlier, longer-standing version of this note). Every image used
is a **real, already-existing asset already vetted for another purpose on this site** — no new
image was created or generated for this:
- Every tool page uses its own `/lab/<tool>/thumbnail.jpg` (the same real, tightly-cropped tool
  output already used as that tool's card icon on `lab/index.html`).
- `lab/index.html` uses `/lab-mosaic.jpg` (the real 2×2 composite of four tool thumbnails already
  used as the homepage's own Concept Lab card icon).
- Root `index.html` uses `/echo-state-trace-mkconceptlab.png` — the site's own favicon (a real
  Echo State trace of the text "MKConceptLab"), reused here as the one page's own image rather
  than duplicating `lab-mosaic.jpg` on both root and `lab/index.html`; it's already the visual
  mark visitors associate with this site from the browser tab, and it's a genuine artifact of the
  site's own real work, not stock art.
`twitter:card` stays `summary` (not `summary_large_image`) even now that every page has an image
— all of these source images are square (480×480, 512×512, or 600×600), and `summary_large_image`
expects/crops toward a roughly 2:1 landscape aspect, which would crop a square image awkwardly on
platforms that enforce it. `summary`'s smaller, uncropped thumbnail treatment suits a square image
better. If a real landscape hero image is ever made for a specific page, revisit `twitter:card`
for that page only — there's no need to change every page's card type just because one page's
image happens to be a better fit for the wider format.

`sitemap.xml` and `robots.txt` live at the repo root (served at `/sitemap.xml` and `/robots.txt`
automatically, no config needed). **A new lab tool needs a `<url>` entry added to `sitemap.xml`**
(with today's date as `<lastmod>`) in the same commit that adds its page — easy to forget since
nothing breaks visibly if you skip it, search engines just never find out the page exists.
`lab/index.html`'s own newest-first list is the reminder to cross-check against when adding a
tool anyway.
