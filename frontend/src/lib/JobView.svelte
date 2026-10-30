<script lang="ts">
  import { ApiError, getJob, type Job } from "./api";
  import Feedback from "./Feedback.svelte";
  import Probabilities from "./Probabilities.svelte";
  import Viewer from "./Viewer.svelte";

  let { id }: { id: string } = $props();

  let job = $state<Job | null>(null);
  let error = $state("");
  let selected = $state("");

  // Poll until the worker reports; slow down after the first few seconds.
  $effect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let polls = 0;
    let cancelled = false;

    async function poll() {
      try {
        job = await getJob(id);
        error = "";
      } catch (e) {
        error = e instanceof Error ? e.message : String(e);
        if (e instanceof ApiError && e.status === 404) return;
      }
      if (cancelled || (job && job.status !== "queued")) return;
      polls += 1;
      timer = setTimeout(poll, polls < 10 ? 700 : 2000);
    }

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  });

  // Start on the most probable finding.
  $effect(() => {
    const p = job?.prediction;
    if (p && !selected) {
      selected = [...p.classes].sort((a, b) => p.probabilities[b] - p.probabilities[a])[0];
    }
  });

  const mlflowBase = `${location.protocol}//${location.hostname}:5000`;
</script>

{#if error && !job}
  <p class="error" role="alert">{error}</p>
{:else if !job || job.status === "queued"}
  <section class="card waiting" aria-live="polite">
    <span class="spinner" aria-hidden="true"></span>
    <div>
      <strong>Queued{job ? `: ${job.filename}` : ""}</strong>
      <p class="muted">Waiting for an inference worker. This usually takes a second or two.</p>
    </div>
  </section>
{:else if job.status === "failed"}
  <section class="card">
    <h1>Inference failed</h1>
    <p class="error" role="alert">{job.error || "The worker could not process this file."}</p>
    <a class="btn" href="#/">Upload another image</a>
  </section>
{:else if job.prediction}
  {@const p = job.prediction}
  <div class="title">
    <h1>{job.filename}</h1>
    <a class="btn" href="#/">New upload</a>
  </div>
  <div class="layout">
    <Viewer imageUrl={p.image_url} heatmapUrl={selected ? p.heatmap_urls[selected] : null} label={selected} />
    <aside>
      <section class="card">
        <h2>Findings</h2>
        <p class="muted small">
          Select a finding to show its heatmap. The marker is the model version's decision
          threshold, chosen on the validation set.
        </p>
        <Probabilities prediction={p} {selected} onselect={(name) => (selected = name)} />
      </section>
      <Feedback jobId={job.id} prediction={p} />
      <section class="card lineage">
        <h2>Provenance</h2>
        <dl>
          <dt>Model</dt>
          <dd>
            <a href={`${mlflowBase}/#/models/${p.model_name}/versions/${p.model_version}`}>
              {p.model_name} v{p.model_version}
            </a>
          </dd>
          <dt>Preprocess</dt>
          <dd>v{p.preprocess_version}</dd>
          <dt>Dataset rev</dt>
          <dd><code>{p.dataset_rev || "-"}</code></dd>
          <dt>MLflow run</dt>
          <dd><code>{p.mlflow_run_id ? p.mlflow_run_id.slice(0, 12) : "-"}</code></dd>
          <dt>Worker time</dt>
          <dd>{Math.round(p.latency_ms)} ms</dd>
          <dt>Job</dt>
          <dd><code>{job.id.slice(0, 8)}</code></dd>
        </dl>
      </section>
    </aside>
  </div>
{/if}

<style>
  .title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 0.75rem 0 1rem;
  }

  h1 {
    font-size: 1.25rem;
    overflow-wrap: anywhere;
  }

  h2 {
    font-size: 1.05rem;
    margin-bottom: 0.4rem;
  }

  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(300px, 1fr);
    gap: 1rem;
    align-items: start;
  }

  @media (max-width: 860px) {
    .layout {
      grid-template-columns: 1fr;
    }
  }

  aside {
    display: grid;
    gap: 1rem;
  }

  .small {
    font-size: 0.85rem;
    margin-top: 0;
  }

  .waiting {
    display: flex;
    gap: 1rem;
    align-items: center;
    max-width: 720px;
    margin: 1.5rem auto;
  }

  .waiting p {
    margin: 0.2rem 0 0;
  }

  .spinner {
    width: 28px;
    height: 28px;
    flex: none;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .spinner {
      animation-duration: 3s;
    }
  }

  .lineage dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.3rem 1rem;
    margin: 0;
    font-size: 0.9rem;
  }

  .lineage dt {
    color: var(--muted);
  }

  .lineage dd {
    margin: 0;
    overflow-wrap: anywhere;
  }
</style>
