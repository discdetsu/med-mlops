# frontend/

Svelte 5 + Vite UI, built into the `nginx` image (`Dockerfile`), which also serves as the gateway (`compose/nginx/default.conf`).

- **Upload** a PNG, JPEG, or DICOM; the page follows the job until a worker reports.
- **Viewer:** the image with the selected finding's class activation map on top (turbo colormap, alpha following the map), an opacity slider, and a heatmap toggle. A plain canvas, not Cornerstone3D: the worker sends an 8-bit display PNG and one grayscale CAM per class, so the browser never decodes DICOM.
- **Findings:** per-class probability bars with the model version's threshold as a marker. Selecting one switches the heatmap.
- **Feedback:** agree, or correct the labels per class, with an optional comment.
- **Provenance:** model version (linked to MLflow), preprocess version, dataset rev, MLflow run, worker time.
- A "Not for clinical use" banner on every view.

```sh
npm install
npm run dev     # http://localhost:5173, proxying /api to the nginx of `make up`
npm run check   # svelte-check (TypeScript)
npm run build   # dist/, what the image serves
```
