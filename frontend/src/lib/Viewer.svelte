<script lang="ts">
  import { TURBO, turboGradientCss } from "./colormap";

  interface Props {
    imageUrl: string;
    heatmapUrl: string | null;
    label: string;
  }

  let { imageUrl, heatmapUrl, label }: Props = $props();

  let opacity = $state(0.5);
  let visible = $state(true);
  let canvas: HTMLCanvasElement;
  let base = $state<HTMLImageElement | null>(null);
  let overlay = $state<HTMLCanvasElement | null>(null);
  let error = $state("");

  function loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`could not load ${url}`));
      img.src = url;
    });
  }

  // The heatmap PNG is 8-bit grayscale (0 = no evidence, 255 = the map's peak). Colour it
  // with turbo and make alpha follow the value, so regions without evidence stay clear.
  function colorize(img: HTMLImageElement): HTMLCanvasElement {
    const out = document.createElement("canvas");
    out.width = img.naturalWidth;
    out.height = img.naturalHeight;
    const ctx = out.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const pixels = ctx.getImageData(0, 0, out.width, out.height);
    const d = pixels.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = d[i];
      const [r, g, b] = TURBO[v];
      d[i] = r;
      d[i + 1] = g;
      d[i + 2] = b;
      d[i + 3] = v;
    }
    ctx.putImageData(pixels, 0, 0);
    return out;
  }

  $effect(() => {
    let stale = false;
    loadImage(imageUrl)
      .then((img) => !stale && (base = img))
      .catch((e: Error) => !stale && (error = e.message));
    return () => (stale = true);
  });

  $effect(() => {
    let stale = false;
    overlay = null;
    if (heatmapUrl) {
      loadImage(heatmapUrl)
        .then((img) => !stale && (overlay = colorize(img)))
        .catch((e: Error) => !stale && (error = e.message));
    }
    return () => (stale = true);
  });

  $effect(() => {
    if (!base) return;
    canvas.width = base.naturalWidth;
    canvas.height = base.naturalHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(base, 0, 0);
    if (overlay && visible) {
      ctx.globalAlpha = opacity;
      ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
    }
  });
</script>

<div class="viewer">
  <div class="stage">
    <canvas bind:this={canvas} aria-label={`Chest X-ray with ${label} heatmap`}></canvas>
    {#if !base && !error}
      <span class="loading">Loading image…</span>
    {/if}
  </div>

  <div class="controls">
    <label class="toggle">
      <input type="checkbox" bind:checked={visible} />
      Heatmap
    </label>
    <label class="slider">
      Opacity
      <input type="range" min="0" max="1" step="0.05" bind:value={opacity} disabled={!visible} />
      <output>{Math.round(opacity * 100)}%</output>
    </label>
    <div class="legend" aria-label="Heatmap scale">
      <span>low</span>
      <span class="ramp" style:background={turboGradientCss()}></span>
      <span>peak</span>
    </div>
  </div>
  <p class="caption">
    Showing <strong>{label}</strong>. The map is scaled to its own peak, so it shows
    <em>where</em> the evidence is, not how strong it is; compare classes by probability.
  </p>
  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}
</div>

<style>
  .viewer {
    background: var(--viewer-bg);
    border-radius: var(--radius);
    padding: 0.75rem;
    color: #d6dae0;
  }

  .stage {
    display: grid;
    place-items: center;
    min-height: 320px;
  }

  canvas {
    display: block;
    max-width: 100%;
    max-height: 72vh;
    width: auto;
    height: auto;
    image-rendering: auto;
  }

  .loading {
    color: #9aa3af;
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.75rem 1.5rem;
    margin-top: 0.75rem;
    font-size: 0.9rem;
  }

  .toggle,
  .slider {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .slider input {
    width: 140px;
    accent-color: #5b8def;
  }

  output {
    min-width: 3ch;
    font-variant-numeric: tabular-nums;
  }

  .legend {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-left: auto;
    color: #9aa3af;
  }

  .ramp {
    width: 120px;
    height: 10px;
    border-radius: 3px;
  }

  .caption {
    margin: 0.6rem 0 0;
    font-size: 0.85rem;
    color: #9aa3af;
  }
</style>
