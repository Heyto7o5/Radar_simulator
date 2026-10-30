import { API_BASE } from "./api";
import type { Catalog, ComparisonRecord, Matrix, Preview, SceneJob, SceneRequest } from "./workbenchTypes";

export class HttpError extends Error { constructor(message: string, public status: number) { super(message); } }
export function readableError(message: string): string {
  if (message.includes("intersects antenna")) return "目标与天线位置相交，请调整目标或阵元位置。";
  if (message.includes("intersects radar")) return "目标与雷达位置相交，请调整场景。";
  if (message.includes("two scene jobs already active")) return "已有两个仿真任务在运行，请先停止任务或稍后重试。";
  if (message.includes("exceeds 2 GB")) return "结果超过 2 GB，请减少帧数、扫描范围或关闭原始 ADC 保存。";
  return message;
}
export async function requestJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API_BASE + path, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const value = (await response.json()).detail;
      detail = typeof value === "string" ? value : Array.isArray(value) ? value.map(x => `${x.loc?.slice(1).join(".") ?? ""}: ${x.msg}`).join("；") : JSON.stringify(value);
    } catch { /* Preserve the transport error. */ }
    throw new HttpError(readableError(detail), response.status);
  }
  return response.json() as Promise<T>;
}
export const getWorkbenchPresets = () => requestJSON<Record<string, SceneRequest>>("/api/v1/workbench/presets");
export const getPreview = (request: SceneRequest, auto: boolean, signal?: AbortSignal) => requestJSON<Preview>("/api/v1/workbench/preview", { method: "POST", body: JSON.stringify({ request, auto_acquisition: auto }), signal });
export const createScene = (request: SceneRequest) => requestJSON<SceneJob>("/api/v1/scene-simulations", { method: "POST", body: JSON.stringify(request) });
export const getSceneJob = (id: string) => requestJSON<SceneJob>(`/api/v1/scene-simulations/${id}`);
export const cancelScene = (id: string) => requestJSON<SceneJob>(`/api/v1/scene-simulations/${id}`, { method: "DELETE" });
export const getSavedRequest = (id: string) => requestJSON<SceneRequest>(`/api/v1/workbench/jobs/${id}/request`);
export function eventURL(id: string) {
  const url = new URL(API_BASE); url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `/api/v1/scene-simulations/${id}/events`; return url.toString();
}
export async function matrix(path: string, signal?: AbortSignal): Promise<Matrix> {
  const response = await fetch(API_BASE + path, { signal });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail ?? message; } catch { /* Transport error. */ }
    throw new HttpError(String(message), response.status);
  }
  const rows = Number(response.headers.get("X-Rows")), cols = Number(response.headers.get("X-Cols"));
  if (response.headers.get("X-Dtype") !== "float32" || response.headers.get("X-Order") !== "C") throw new Error("显示数据格式不匹配");
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== rows * cols * 4 || !rows || !cols) throw new Error("显示矩阵尺寸不匹配");
  const values = new Float32Array(bytes.byteLength / 4), view = new DataView(bytes);
  for (let i = 0; i < values.length; i++) { values[i] = view.getFloat32(i * 4, true); if (!Number.isFinite(values[i])) throw new Error("显示数据包含无效数值"); }
  return { data: values, rows, cols, reference: Number(response.headers.get("X-Reference-Power")) };
}
export function getSceneView(id: string, frame: number, options: Record<string, string | number>, signal?: AbortSignal) {
  const params = new URLSearchParams(Object.entries(options).map(([k, v]) => [k, String(v)]));
  return matrix(`/api/v1/workbench/jobs/${id}/frames/${frame}/view?${params}`, signal);
}
export const getCatalog = () => requestJSON<Catalog>("/api/v1/comparisons/ramp/catalog");
export const prepareReal = (frame_id: string, signal?: AbortSignal) => requestJSON<ComparisonRecord>("/api/v1/comparisons/ramp/prepare", { method: "POST", body: JSON.stringify({ frame_id }), signal });
export const simulateComparison = (id: string, request: SceneRequest) => requestJSON<SceneJob>(`/api/v1/comparisons/ramp/records/${id}/simulate`, { method: "POST", body: JSON.stringify({ request }) });
export const realMatrix = (id: string, channel: string, signal?: AbortSignal) => matrix(`/api/v1/comparisons/ramp/records/${id}/rd?channel=${encodeURIComponent(channel)}`, signal);
export const comparisonMatrix = (id: string, job: string, channel: string, signal?: AbortSignal) => matrix(`/api/v1/comparisons/ramp/records/${id}/simulations/${job}/rd?channel=${encodeURIComponent(channel)}`, signal);
export const cameraURL = (path: string) => API_BASE + path;
export const isAbort = (reason: unknown) => reason instanceof DOMException && reason.name === "AbortError";

export const checkScene = (request: SceneRequest, motion = true) => requestJSON<import("./workbenchTypes").CheckReport>(motion ? "/api/v1/scene-motion-check" : "/api/v1/scene-check", { method: "POST", body: JSON.stringify(request) });
export const listExperiments = () => requestJSON<import("./workbenchTypes").ExperimentSummary[]>("/api/v1/workbench/experiments");
export const getExperiment = (id: string) => requestJSON<import("./workbenchTypes").Experiment>(`/api/v1/workbench/experiments/${id}`);
export const saveExperiment = (id: string, title: string, display: Record<string, unknown>) => requestJSON(`/api/v1/workbench/experiments/${id}/save`, { method: "POST", body: JSON.stringify({ title, display }) });
export const getCases = () => requestJSON<import("./workbenchTypes").CaseSpec[]>("/api/v1/workbench/cases");
export const getCaseRequest = (id: string) => requestJSON<SceneRequest>(`/api/v1/workbench/cases/${id}/request`);
export const getGeometry = (id: string) => requestJSON<import("./workbenchTypes").SceneGeometry>(`/api/v1/workbench/jobs/${id}/geometry`);
export const getAuxiliary = (id: string, frame: number, signal?: AbortSignal) => requestJSON<import("./workbenchTypes").Auxiliary>(`/api/v1/workbench/jobs/${id}/frames/${frame}/auxiliary`, { signal });
export const getDisplayData = (id: string, frame: number, channel: string, signal?: AbortSignal) => requestJSON<import("./workbenchTypes").DisplayData>(`/api/v1/workbench/jobs/${id}/frames/${frame}/display-data?channel=${encodeURIComponent(channel)}`, { signal });
export const exportURL = (id: string) => API_BASE + `/api/v1/workbench/experiments/${id}/export`;
export function downloadJSON(value: unknown, name: string) { const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" })); const a = document.createElement("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }

export const normalizeScene = (request: unknown) => requestJSON<SceneRequest>("/api/v1/workbench/normalize", { method: "POST", body: JSON.stringify(request) });
