import type { BinaryMatrix, JobState, RampResult, RampStatus, SimulationPreset, SimulationRequest, TerrainGrid } from "./types";

function defaultApiBase(): string {
  if (typeof window === "undefined") return "http://127.0.0.1:8000";
  if (window.location.port === "5173") {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return window.location.origin;
}

export const API_BASE = import.meta.env.VITE_API_BASE ?? defaultApiBase();

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    } catch {
      // Keep the HTTP status when the response is not JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function getPresets(): Promise<Record<string, SimulationPreset>> {
  return json("/api/v1/presets");
}

export async function getPreset(key = "clean_high_altitude_demo_10ghz"): Promise<SimulationPreset> {
  const payload = await getPresets();
  return payload[key] ?? payload.clean_high_altitude_demo_10ghz;
}

export function getIdealTerrain(): Promise<TerrainGrid> {
  return json("/api/v1/terrain/ideal-flat");
}

export function createSimulation(request: SimulationRequest): Promise<{ job_id: string; status: string; derived_metrics: JobState["derived_metrics"]; warnings: string[] }> {
  return json("/api/v1/simulations", { method: "POST", body: JSON.stringify(request) });
}

export function getJob(jobId: string): Promise<JobState> {
  return json(`/api/v1/simulations/${jobId}`);
}

export function cancelJob(jobId: string): Promise<JobState> {
  return json(`/api/v1/simulations/${jobId}`, { method: "DELETE" });
}

export async function getBinaryMatrix(url: string): Promise<BinaryMatrix> {
  const response = await fetch(`${API_BASE}${url}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const rows = Number(response.headers.get("X-Rows"));
  const cols = Number(response.headers.get("X-Cols"));
  const data = new Float32Array(await response.arrayBuffer());
  if (!rows || !cols || data.length !== rows * cols) {
    throw new Error("二进制矩阵尺寸与响应头不一致");
  }
  return { data, rows, cols };
}

export function getSimulationFrame(jobId: string, frameIndex: number): Promise<BinaryMatrix> {
  return getBinaryMatrix(`/api/v1/simulations/${jobId}/frames/${frameIndex}`);
}

export function jobWebSocketUrl(jobId: string): string {
  const url = new URL(API_BASE);
  const protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${url.host}/api/v1/simulations/${jobId}/events`;
}

export function getRampStatus(): Promise<RampStatus> {
  return json("/api/v1/datasets/ramp/status");
}

export function getRampFrames(): Promise<{ frames: { id: string; name: string }[] }> {
  return json("/api/v1/datasets/ramp/frames?limit=50");
}

export function processRamp(frameId?: string): Promise<RampResult> {
  return json("/api/v1/datasets/ramp/process", {
    method: "POST",
    body: JSON.stringify({ frame_id: frameId || null }),
  });
}

export function getRampMatrix(resultId: string, view: "measured" | "ideal", channel = "combined"): Promise<BinaryMatrix> {
  return getBinaryMatrix(`/api/v1/datasets/ramp/results/${resultId}/rd?view=${view}&channel=${encodeURIComponent(channel)}`);
}

export function rampCameraUrl(resultId: string): string {
  return `${API_BASE}/api/v1/datasets/ramp/results/${resultId}/camera`;
}
