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
  get a correct breadcrumb.
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
