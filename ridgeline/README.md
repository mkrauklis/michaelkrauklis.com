# Ridgeline

A mountain range is a stack of sine waves. Ridgeline traces the silhouette of an
uploaded photo, decomposes it with a discrete Fourier transform, and shows the
shape assembling itself one frequency at a time — as a still image (panels
ordered by wave magnitude) or a video (sequential summation, or classic
rotating-arm/epicycle drawing).

Everything runs client-side in the browser. No photo is ever uploaded to a
server, and no build step is required — it's a single `index.html`.

## Try it

Open `index.html` directly in a browser, or serve the repo with GitHub Pages
(see below).

1. Upload a photo with a clear sky/land silhouette.
2. Adjust the detection threshold until the traced line hugs the real ridge
   (use the invert checkbox if the sky is darker than the land, e.g. a
   backlit or dusk shot).
3. Choose **Image** for a downloadable panel-sequence PNG, or **Video** for
   an animated `.webm` — sequential summation or rotating arms.

## How it works

- The silhouette is sampled into N points, mirrored to make it periodic, and
  run through a discrete Fourier transform (`O(N²)`, computed in-browser).
- Frequency components are sorted by magnitude, largest first.
- Reconstructions are built by summing the top-*k* components — the video
  and panel modes just vary *k* (or, for the arms mode, animate the phase).
- Epicycle arm count defaults to however many components carry ~95% of the
  signal's energy.

## Site structure

```
/                     portfolio home (michaelkrauklis.com)
/lab/                 index of tools
/lab/ridgeline/       this tool
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

## Ads

There's a labeled placeholder `<div id="adSlot">` near the bottom of
`index.html`. Once you have an approved ad network account (e.g. AdSense),
drop the unit's script snippet in there.

## License

MIT — see `LICENSE`.
