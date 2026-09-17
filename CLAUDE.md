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
  `.card-icon` (64×64, `object-fit:cover`) is the shared sizing/shape class for both cards'
  icons; the mosaic `<img>` additionally carries an `onerror="this.remove()"` fallback (the
  inline LinkedIn SVG can't fail to load, so it doesn't need one), identical in spirit to
  `lab/index.html`'s own `.tool-thumb` handling — a missing file collapses back to a clean
  icon-less card rather than a broken-image glyph.

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
site ever gets a different mascot/favicon, update all 7 pages' `<link>` tags together in one
commit — there's no shared head-include mechanism, so this one tag is duplicated per page the
same way the SEO meta block below is.

# SEO

The live site is served from **`michaelkrauklis.me`**, not `.com` — the git remote's name is
`.com` but the actual canonical domain is set by the root `CNAME` file. Always use `.me` in
canonical URLs, `og:url`, and the sitemap; don't assume the domain from the repo name.

Every page (root `index.html`, `lab/index.html`, every tool) carries the same block of tags right
after `<title>`, before the `theme.css` link: `<meta name="description">`, `<link
rel="canonical">`, `og:type`/`og:site_name`/`og:title`/`og:description`/`og:url`, and
`twitter:card` (`summary`, not `summary_large_image` — there's no `og:image` anywhere on the site
yet, no logo or screenshot asset exists to use for one; add real `og:image` tags site-wide
together, in one pass, if that ever gets made, rather than letting pages drift inconsistent one at
a time). `og:title`/`og:description` duplicate the meta description and `<title>` rather than
being written separately — there's no reason for them to diverge on a site this size.

`sitemap.xml` and `robots.txt` live at the repo root (served at `/sitemap.xml` and `/robots.txt`
automatically, no config needed). **A new lab tool needs a `<url>` entry added to `sitemap.xml`**
(with today's date as `<lastmod>`) in the same commit that adds its page — easy to forget since
nothing breaks visibly if you skip it, search engines just never find out the page exists.
`lab/index.html`'s own newest-first list is the reminder to cross-check against when adding a
tool anyway.
