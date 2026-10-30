// Types and calls for the public API (backend/src/cxr_backend/jobs/schemas.py).

export type JobStatus = "queued" | "succeeded" | "failed";

export interface Prediction {
  model_name: string;
  model_version: string;
  preprocess_version: string;
  mlflow_run_id: string;
  dataset_rev: string;
  classes: string[];
  probabilities: Record<string, number>;
  thresholds: Record<string, number>;
  positives: string[];
  latency_ms: number;
  image_url: string;
  heatmap_urls: Record<string, string>;
}

export interface Job {
  id: string;
  status: JobStatus;
  filename: string;
  format: string;
  created_at: string;
  finished_at: string | null;
  error: string;
  prediction: Prediction | null;
}

export interface FeedbackIn {
  verdict: "agree" | "correct";
  labels?: Record<string, boolean>;
  comment?: string;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    throw new ApiError(response.status, await errorText(response));
  }
  return (await response.json()) as T;
}

async function errorText(response: Response): Promise<string> {
  if (response.status === 429) return "Too many requests; wait a moment and try again.";
  if (response.status === 413) return "The file is too large (32 MB at most).";
  try {
    const body = await response.json();
    // Ninja: {"detail": "..."} for our errors, {"detail": [{msg, loc}]} for validation.
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    // not JSON
  }
  return `Request failed (${response.status})`;
}

export function createJob(file: File): Promise<Job> {
  const form = new FormData();
  form.append("file", file);
  return call<Job>("/api/jobs", { method: "POST", body: form });
}

export function getJob(id: string): Promise<Job> {
  return call<Job>(`/api/jobs/${encodeURIComponent(id)}`);
}

export function sendFeedback(id: string, body: FeedbackIn): Promise<unknown> {
  return call(`/api/jobs/${encodeURIComponent(id)}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
