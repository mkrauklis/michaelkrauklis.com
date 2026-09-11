// Vectis's embedding worker. Runs off the main thread so downloading/running the CLIP
// model never janks the plot's animations or the page while it's thinking.
//
// Loaded as a module worker (`new Worker(url, { type: 'module' })`) so it can `import`
// transformers.js straight from a CDN -- see ../CLAUDE.md's "Why a CDN import" note for
// why that's a deliberate exception to the site's usual "no external JS" rule.
import {
  AutoTokenizer, CLIPTextModelWithProjection,
  AutoProcessor, CLIPVisionModelWithProjection,
  pipeline,
  RawImage, env,
} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = false;

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
// Word-to-word comparisons use a general sentence-embedding model instead of CLIP's text tower
// -- see ../CLAUDE.md's "Two embedding models, one per mode" section for why. CLIP's text tower
// is still used, unchanged, for Photos mode, since that's the only space a photo can be compared
// against a word in at all. Loaded via `pipeline('feature-extraction', ...)` rather than the
// direct AutoTokenizer/AutoModel classes used for CLIP below -- attention-mask-aware mean
// pooling (ignoring padding tokens) is the standard, correct way to turn this model's per-token
// output into one sentence vector, and `pipeline`'s built-in `pooling`/`normalize` options do
// exactly that without hand-rolling the mask arithmetic ourselves.
const GENERAL_TEXT_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

let tokenizer, textModel, processor, visionModel, generalTextExtractor;
let textReady = null;
let visionReady = null;
let generalTextReady = null;

function post(msg) { self.postMessage(msg); }

function progressFor(tower) {
  return (p) => {
    // transformers.js reports one callback per file in the model repo; forward the
    // interesting ones so the page can show real percentages instead of a spinner.
    if (p && (p.status === 'progress' || p.status === 'done' || p.status === 'ready')) {
      post({ type: 'progress', tower, status: p.status, file: p.file, progress: p.progress || 0 });
    }
  };
}

function ensureText() {
  if (!textReady) {
    textReady = (async () => {
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: progressFor('text') });
      textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID, { progress_callback: progressFor('text'), quantized: true });
    })();
  }
  return textReady;
}

function ensureVision() {
  if (!visionReady) {
    visionReady = (async () => {
      processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: progressFor('vision') });
      visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { progress_callback: progressFor('vision'), quantized: true });
    })();
  }
  return visionReady;
}

function ensureGeneralText() {
  if (!generalTextReady) {
    generalTextReady = (async () => {
      generalTextExtractor = await pipeline('feature-extraction', GENERAL_TEXT_MODEL_ID, { progress_callback: progressFor('text') });
    })();
  }
  return generalTextReady;
}

function normalizeRows(data, rows, dim) {
  const out = [];
  for (let r = 0; r < rows; r++) {
    let sumSq = 0;
    const row = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      const v = data[r * dim + i];
      row[i] = v;
      sumSq += v * v;
    }
    const norm = Math.sqrt(sumSq) || 1;
    for (let i = 0; i < dim; i++) row[i] /= norm;
    out.push(Array.from(row));
  }
  return out;
}

self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    if (type === 'warmText') {
      await ensureText();
      post({ type: 'ready', id, tower: 'text' });
    } else if (type === 'warmVision') {
      await ensureVision();
      post({ type: 'ready', id, tower: 'vision' });
    } else if (type === 'warmGeneralText') {
      await ensureGeneralText();
      post({ type: 'ready', id, tower: 'text' });
    } else if (type === 'embedText') {
      // `model` selects which text encoder this batch goes through -- 'general' (the default,
      // used for Words mode) or 'clip' (used only for axis words in Photos mode, so they land in
      // the same space as the uploaded photos). See CLAUDE.md's "Two embedding models" section.
      const { texts, model } = e.data;
      if (model === 'clip') {
        await ensureText();
        const inputs = tokenizer(texts, { padding: true, truncation: true });
        const { text_embeds } = await textModel(inputs);
        const dim = text_embeds.dims[text_embeds.dims.length - 1];
        const embeddings = normalizeRows(text_embeds.data, texts.length, dim);
        post({ type: 'result', id, embeddings });
      } else {
        await ensureGeneralText();
        const out = await generalTextExtractor(texts, { pooling: 'mean', normalize: true });
        const dim = out.dims[out.dims.length - 1];
        const embeddings = [];
        for (let r = 0; r < texts.length; r++) embeddings.push(Array.from(out.data.slice(r * dim, (r + 1) * dim)));
        post({ type: 'result', id, embeddings });
      }
    } else if (type === 'embedImage') {
      await ensureVision();
      const { bitmap } = e.data;
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const raw = new RawImage(imgData.data, canvas.width, canvas.height, 4);
      const inputs = await processor(raw);
      const { image_embeds } = await visionModel(inputs);
      const dim = image_embeds.dims[image_embeds.dims.length - 1];
      const embeddings = normalizeRows(image_embeds.data, 1, dim);
      post({ type: 'result', id, embeddings });
    }
  } catch (err) {
    post({ type: 'error', id, message: (err && err.message) ? err.message : String(err) });
  }
};
