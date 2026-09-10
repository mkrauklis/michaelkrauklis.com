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
  don't let a new tool drift back to the old generic name.
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

When adding a new lab tool, link both files and rely on `theme.css`'s defaults before writing
new CSS — copying a whole `<style>` block from an existing tool page (the old pattern) is how
the palette and the paragraph-width bug drifted out of sync across pages in the first place.

`lab/index.html`'s tool list is ordered newest-first, top to bottom (currently Neural Viaduct,
Vectis, Afterimage, Ridgeline) — add a new tool's card at the *top* of `.tool-list`, not the bottom.

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
