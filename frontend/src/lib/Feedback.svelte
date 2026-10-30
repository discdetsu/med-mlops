<script lang="ts">
  import { sendFeedback, type Prediction } from "./api";

  interface Props {
    jobId: string;
    prediction: Prediction;
  }

  let { jobId, prediction }: Props = $props();

  let mode = $state<"idle" | "correcting" | "sent">("idle");
  let labels = $state<Record<string, boolean>>({});
  let comment = $state("");
  let busy = $state(false);
  let error = $state("");

  function startCorrecting() {
    labels = Object.fromEntries(
      prediction.classes.map((c) => [c, prediction.positives.includes(c)]),
    );
    mode = "correcting";
  }

  async function submit(verdict: "agree" | "correct") {
    busy = true;
    error = "";
    try {
      await sendFeedback(jobId, verdict === "agree" ? { verdict, comment } : { verdict, labels, comment });
      mode = "sent";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
</script>

<section class="card">
  <h2>Reviewer feedback</h2>
  {#if mode === "sent"}
    <p role="status">
      Thanks. Recorded against model version {prediction.model_version}; it feeds the delayed
      performance metrics and the labeling queue.
    </p>
  {:else}
    <p class="muted small">
      Do the findings above the threshold match the image? Corrections go into the labeling
      loop.
    </p>

    {#if mode === "correcting"}
      <fieldset>
        <legend>Findings present</legend>
        {#each prediction.classes as name (name)}
          <label>
            <input type="checkbox" bind:checked={labels[name]} />
            {name}
          </label>
        {/each}
      </fieldset>
    {/if}

    <label class="comment">
      <span>Comment <span class="muted">(optional)</span></span>
      <textarea bind:value={comment} rows="2" maxlength="2000"></textarea>
    </label>

    <div class="actions">
      {#if mode === "correcting"}
        <button class="btn primary" disabled={busy} onclick={() => submit("correct")}>
          Submit correction
        </button>
        <button class="btn" disabled={busy} onclick={() => (mode = "idle")}>Cancel</button>
      {:else}
        <button class="btn primary" disabled={busy} onclick={() => submit("agree")}>Agree</button>
        <button class="btn" disabled={busy} onclick={startCorrecting}>Correct labels</button>
      {/if}
    </div>
  {/if}
  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}
</section>

<style>
  h2 {
    font-size: 1.05rem;
    margin-bottom: 0.4rem;
  }

  .small {
    font-size: 0.85rem;
    margin-top: 0;
  }

  fieldset {
    border: 1px solid var(--border);
    border-radius: 8px;
    margin: 0.75rem 0;
    padding: 0.5rem 0.9rem 0.75rem;
    display: grid;
    gap: 0.3rem;
  }

  legend {
    padding: 0 0.3rem;
    font-weight: 600;
  }

  .comment {
    display: grid;
    gap: 0.3rem;
    margin: 0.75rem 0;
  }

  textarea {
    resize: vertical;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
  }

  .actions {
    display: flex;
    gap: 0.5rem;
  }
</style>
