import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// `npm run dev` proxies the API to the nginx of a running `make up`, so the UI can be
// worked on with hot reload against the real backend.
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
