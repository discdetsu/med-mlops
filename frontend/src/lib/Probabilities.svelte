<script lang="ts">
  import type { Prediction } from "./api";

  interface Props {
    prediction: Prediction;
    selected: string;
    onselect: (name: string) => void;
  }

  let { prediction, selected, onselect }: Props = $props();

  const pct = (p: number) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
</script>

<div class="probs" role="radiogroup" aria-label="Findings; choose one to show its heatmap">
  {#each prediction.classes as name (name)}
    {@const p = prediction.probabilities[name]}
    {@const t = prediction.thresholds[name]}
    {@const positive = prediction.positives.includes(name)}
    <button
      class="row"
      class:selected={name === selected}
      class:positive
      role="radio"
      aria-checked={name === selected}
      onclick={() => onselect(name)}
    >
      <span class="name">
        {name}
        {#if positive}<span class="flag">above threshold</span>{/if}
      </span>
      <span class="value">{pct(p)}</span>
      <span class="track" aria-hidden="true">
        <span class="fill" style:width={`${p * 100}%`}></span>
        <span class="marker" style:left={`${t * 100}%`} title={`threshold ${pct(t)}`}></span>
      </span>
      <span class="threshold muted">threshold {pct(t)}</span>
    </button>
  {/each}
</div>

<style>
  .probs {
    display: grid;
    gap: 0.35rem;
  }

  .row {
    display: grid;
    grid-template-columns: 1fr auto;
    grid-template-areas:
      "name value"
      "track track"
      "threshold threshold";
    gap: 0.3rem 0.75rem;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.55rem 0.65rem;
    cursor: pointer;
  }

  .row:hover {
    background: var(--surface-2);
  }

  .row.selected {
    border-color: var(--accent);
    background: var(--surface-2);
  }

  .name {
    grid-area: name;
    font-weight: 600;
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
  }

  .flag {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--positive);
    background: var(--positive-bg);
    border-radius: 999px;
    padding: 0.05rem 0.5rem;
  }

  .value {
    grid-area: value;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }

  .track {
    grid-area: track;
    position: relative;
    height: 8px;
    background: var(--surface-2);
    border-radius: 4px;
    overflow: visible;
  }

  .row.selected .track {
    background: var(--border);
  }

  .fill {
    position: absolute;
    inset: 0 auto 0 0;
    background: var(--bar);
    border-radius: 4px;
  }

  .positive .fill {
    background: var(--bar-positive);
  }

  .marker {
    position: absolute;
    top: -4px;
    bottom: -4px;
    width: 2px;
    margin-left: -1px;
    background: var(--text);
  }

  .threshold {
    grid-area: threshold;
    font-size: 0.78rem;
  }
</style>
