import { useCallback, useEffect, useRef, useState } from "react";
import { cancelScene, createScene, eventURL, getSceneJob, HttpError, readableError } from "../workbenchApi";
import type { SceneJob, SceneRequest } from "../workbenchTypes";

const terminal = (job: SceneJob) => ["complete", "cancelled", "failed"].includes(job.status);
const sleep = (ms: number) => new Promise(resolve => window.setTimeout(resolve, ms));

/** One request producer, one active job, latest draft wins. GET/WS never paint a stale generation. */
export function useSceneRunner(create: (request: SceneRequest) => Promise<SceneJob> = createScene) {
  const [job, setJob] = useState<SceneJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const generation = useRef(0), active = useRef<string | null>(null);
  const desired = useRef<{ request: SceneRequest; version: number } | null>(null);
  const pumping = useRef(false), mounted = useRef(true), dispose = useRef<() => void>(() => undefined);
  const stopping = useRef<Promise<SceneJob | undefined> | null>(null);
  const createRef = useRef(create); createRef.current = create;
  const jobRef = useRef<SceneJob | null>(null);

  const watch = useCallback((id: string, version: number) => {
    dispose.current(); let closed = false, fetching = false, failures = 0;
    const accept = (next: SceneJob) => {
      if (closed || version !== generation.current || !mounted.current) return;
      const prior = jobRef.current;
      if (prior?.job_id === next.job_id && (next.frames.length < prior.frames.length || (terminal(prior) && !terminal(next)))) return;
      jobRef.current = next; setJob(next); setSubmitting(false);
      if (next.status === "failed") setError(readableError(next.error ?? "仿真失败"));
      else setError(null);
      if (terminal(next)) { window.clearInterval(timer); socket?.close(); }
    };
    const refresh = async () => {
      if (closed || fetching) return;
      fetching = true;
      try { const next = await getSceneJob(id); failures = 0; accept(next); }
      catch (reason) {
        if (!closed && version === generation.current && mounted.current) {
          failures++;
          if (reason instanceof HttpError && reason.status === 404) {
            window.clearInterval(timer); socket?.close(); active.current = null; setJob(null); setSubmitting(false);
            setError("任务已失效，请重新运行。");
          } else if (failures > 2) setError("暂时无法获取进度，正在重连；不会重复创建任务。");
        }
      } finally { fetching = false; }
    };
    let socket: WebSocket | undefined;
    const timer = window.setInterval(() => { void refresh(); }, 1000);
    try {
      socket = new WebSocket(eventURL(id));
      socket.onmessage = message => {
        if (closed || version !== generation.current) return;
        try { const value = JSON.parse(message.data); if (value.type === "snapshot") accept(value.job); else if (value.type !== "heartbeat") void refresh(); }
        catch { void refresh(); }
      };
      // Polling remains active if the socket is unavailable or closes unexpectedly.
      socket.onerror = () => { void refresh(); };
    } catch { /* HTTP progress polling is a fully supported fallback. */ }
    dispose.current = () => { closed = true; window.clearInterval(timer); socket?.close(); };
    void refresh();
  }, []);

  const stopPrevious = useCallback(async () => {
    const preceding = stopping.current ? await stopping.current : undefined;
    const id = active.current; active.current = null;
    if (!id) return preceding;
    const operation = (async () => { try {
      let previous = await cancelScene(id);
      const deadline = Date.now() + 30000;
      while (!terminal(previous)) {
        if (Date.now() > deadline) throw new Error("上一任务仍在停止，请稍后重试");
        await sleep(150); previous = await getSceneJob(id);
      }
      return previous;
    } catch (reason) { if (!(reason instanceof HttpError && reason.status === 404)) throw reason; } })();
    stopping.current = operation;
    try { return await operation; } finally { if (stopping.current === operation) stopping.current = null; }
  }, []);

  const pump = useCallback(async () => {
    if (pumping.current) return;
    pumping.current = true;
    try {
      while (desired.current) {
        let next = desired.current; desired.current = null;
        await stopPrevious();
        // Coalesce edits that arrived while cancellation was in flight.
        if (desired.current) { next = desired.current; desired.current = null; }
        if (next.version !== generation.current) continue;
        let created: SceneJob;
        try { created = await createRef.current(next.request); }
        catch (reason) { if (next.version !== generation.current) continue; throw reason; }
        active.current = created.job_id;
        if (!mounted.current || next.version !== generation.current) { await stopPrevious(); continue; }
        jobRef.current = created; setJob(created); setSubmitting(false); watch(created.job_id, next.version);
      }
    } catch (reason) {
      if (mounted.current) { setError(reason instanceof Error ? reason.message : String(reason)); setSubmitting(false); }
    } finally {
      pumping.current = false;
      if (mounted.current && !desired.current) setSubmitting(false);
    }
  }, [stopPrevious, watch]);

  const run = useCallback((request: SceneRequest) => {
    const version = ++generation.current;
    desired.current = { request, version }; dispose.current(); setJob(null); jobRef.current = null; setError(null); setSubmitting(true);
    void pump();
  }, [pump]);

  const invalidate = useCallback(() => {
    const version = ++generation.current; desired.current = null; dispose.current(); setJob(null); jobRef.current = null; setError(null);
    if (!pumping.current) void stopPrevious().catch(reason => { if (mounted.current && version === generation.current) setError(String(reason)); });
    setSubmitting(false);
  }, [stopPrevious]);

  const stop = useCallback(async () => {
    ++generation.current; desired.current = null; dispose.current(); setSubmitting(true);
    try { const stopped = await stopPrevious(); if (mounted.current && stopped) { jobRef.current = stopped; setJob(stopped); } }
    catch (reason) { if (mounted.current) setError(String(reason)); }
    finally { if (mounted.current) setSubmitting(false); }
  }, [stopPrevious]);

  const resume = useCallback(async (id: string) => {
    const version = ++generation.current; dispose.current(); active.current = id;
    const current = await getSceneJob(id);
    if (version !== generation.current || !mounted.current) return;
    jobRef.current = current; setJob(current); watch(id, version);
  }, [watch]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; dispose.current(); };
  }, []);
  return { job, error, submitting, busy: submitting || Boolean(job && !terminal(job)), run, invalidate, stop, resume };
}
