<script lang="ts">
  import { createJob } from "./api";

  let dragging = $state(false);
  let busy = $state(false);
  let error = $state("");
  let input: HTMLInputElement;

  async function submit(file: File | undefined) {
    if (!file || busy) return;
    busy = true;
    error = "";
    try {
      const job = await createJob(file);
      location.hash = `#/jobs/${job.id}`;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    dragging = false;
    submit(event.dataTransfer?.files[0]);
  }
</script>

<section class="card">
  <h1>Upload a chest X-ray</h1>
  <p class="muted">
    A frontal (PA or AP) chest radiograph as PNG, JPEG, or DICOM, up to 32 MB. The model returns
    a probability for Atelectasis, Cardiomegaly, Effusion, Mass, and Pneumothorax, with a class
    activation map showing which regions drove each one.
  </p>

  <div
    class="drop"
    class:dragging
    role="button"
    tabindex="0"
    aria-busy={busy}
    ondragover={(e) => {
      e.preventDefault();
      dragging = true;
    }}
    ondragleave={() => (dragging = false)}
    ondrop={onDrop}
    onclick={() => input.click()}
    onkeydown={(e) => (e.key === "Enter" || e.key === " ") && input.click()}
  >
    {#if busy}
      <span>Uploading…</span>
    {:else}
      <strong>Drop a file here</strong>
      <span class="muted">or click to choose one</span>
    {/if}
  </div>
  <input
    bind:this={input}
    type="file"
    accept=".png,.jpg,.jpeg,.dcm,image/png,image/jpeg,application/dicom"
    hidden
    onchange={(e) => submit((e.currentTarget as HTMLInputElement).files?.[0])}
  />

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}
</section>

<style>
  section {
    max-width: 720px;
    margin: 1.5rem auto;
  }

  h1 {
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
  }

  .drop {
    margin-top: 1rem;
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 3rem 1rem;
    display: grid;
    place-items: center;
    gap: 0.25rem;
    text-align: center;
    cursor: pointer;
    background: var(--surface-2);
    transition:
      border-color 0.15s,
      background 0.15s;
  }

  .drop:hover,
  .drop.dragging {
    border-color: var(--accent);
  }
</style>
