<script lang="ts">
  import JobView from "./lib/JobView.svelte";
  import Upload from "./lib/Upload.svelte";

  // Hash routes keep nginx config trivial: "#/" uploads, "#/jobs/<id>" shows one job.
  function parse(hash: string): string | null {
    const match = /^#\/jobs\/([0-9a-f-]{36})$/i.exec(hash);
    return match ? match[1] : null;
  }

  let jobId = $state(parse(location.hash));

  $effect(() => {
    const onHash = () => (jobId = parse(location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  });
</script>

<div class="banner" role="note">
  <strong>Not for clinical use.</strong>
  Research demo, not a medical device. Predictions and heatmaps are not diagnoses.
</div>

<header>
  <a class="brand" href="#/">CXR findings</a>
  <span class="muted">5 findings · ONNX · class activation maps</span>
  <nav>
    <a href="/admin/">Admin</a>
  </nav>
</header>

<main>
  {#if jobId}
    {#key jobId}
      <JobView id={jobId} />
    {/key}
  {:else}
    <Upload />
  {/if}
</main>

<style>
  .banner {
    position: sticky;
    top: 0;
    z-index: 10;
    background: var(--banner-bg);
    color: var(--banner-text);
    text-align: center;
    padding: 0.45rem 1rem;
    font-size: 0.9rem;
  }

  header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    flex-wrap: wrap;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.1rem 1rem 0.5rem;
  }

  .brand {
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--text);
    text-decoration: none;
  }

  nav {
    margin-left: auto;
  }

  main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0.5rem 1rem 3rem;
  }
</style>
