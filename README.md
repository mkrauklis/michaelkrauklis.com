# michaelkrauklis.com

Personal site and portfolio — bio, certifications, and talks/writing, plus a
`/lab` of small interactive tools.

## Site structure

```
/                     portfolio home
/lab/                 index of tools
/lab/ridgeline/       Ridgeline — Fourier mountain-silhouette tool
CNAME                 custom domain config for GitHub Pages
```

## Deploying with GitHub Pages

1. Push this repo to GitHub.
2. In the repo settings, go to **Pages**.
3. Under **Source**, choose the `main` branch and `/ (root)` folder.
4. Under **Custom domain**, enter `michaelkrauklis.com` (this repo already
   includes a `CNAME` file with that domain).
5. In your DNS provider (e.g. GoDaddy), point the apex domain at GitHub
   Pages with four `A` records, host `@`:
   ```
   185.199.108.153
   185.199.109.153
   185.199.110.153
   185.199.111.153
   ```
   Optionally add a `CNAME` record for `www` pointing to
   `<your-github-username>.github.io`.
6. Once DNS resolves, check **Enforce HTTPS** in the Pages settings.

## Lab tools

### Ridgeline (`/lab/ridgeline/`)

A mountain range is a stack of sine waves. Ridgeline traces the silhouette of
an uploaded photo, decomposes it with a discrete Fourier transform, and shows
the shape assembling itself one frequency at a time — as a still image
(panels ordered by wave magnitude) or a video (sequential summation, or
classic rotating-arm/epicycle drawing).

Everything runs client-side in the browser. No photo is ever uploaded to a
server, and no build step is required.

**How it works:** the silhouette is sampled into N points, mirrored to make
it periodic, and run through a discrete Fourier transform (`O(N²)`, computed
in-browser). Frequency components are sorted by magnitude, largest first.
Reconstructions are built by summing the top-*k* components — the video and
panel modes just vary *k* (or, for the arms mode, animate the phase).
Epicycle arm count defaults to however many components carry ~95% of the
signal's energy.

## Ads

AdSense Auto ads is enabled via the loader script in `<head>` of
`lab/ridgeline/index.html` (client `ca-pub-5936916546458743`), plus the
site-root `ads.txt`. Google places ads automatically — no manual unit
markup needed. The `<div id="adSlot">` near the bottom of that page is
just a labeled note, not a functional ad slot.

## License

MIT — see `LICENSE`.
