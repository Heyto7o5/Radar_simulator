import { ChevronLeft, ChevronRight, Play, RotateCcw, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cameraURL, comparisonMatrix, getCatalog, HttpError, isAbort, prepareReal, realMatrix, simulateComparison } from "../workbenchApi";
import { clone, type Catalog, type ComparisonRecord, type Matrix, type SceneRequest } from "../workbenchTypes";
import { useSceneRunner } from "../hooks/useSceneRunner";
import { Heatmap, SceneSketch, format } from "../components/WorkbenchCharts";
import { ObjectsEditor, Switch } from "../components/ScenarioEditor";

const STORAGE = "radar.comparison.v1";
interface Saved { frameId: string; request?: SceneRequest; disabled?: string[]; motion?: boolean; live?: boolean; jobId?: string; recordId?: string; signature?: string }
function readSaved(): Saved | null { try { return JSON.parse(localStorage.getItem(STORAGE) ?? "null"); } catch { return null; } }
function emptyTemplate(record: ComparisonRecord): SceneRequest { return { ...clone(record.acquisition), schema_version: "1.0", name: "RAMP 手工场景", scene: { objects: [], terrain: null, seed: 42, amplitude_reference_range_m: 10, max_scatterers: 4096 }, sequence: { frame_count: 1, frame_interval_s: .1, start_time_s: 0 } }; }

export function ComparePage() {
  const [catalog, setCatalog] = useState<Catalog | null>(null), [selected, setSelected] = useState("");
  const [record, setRecord] = useState<ComparisonRecord | null>(null), [input, setInput] = useState<SceneRequest | null>(null);
  const [disabled, setDisabled] = useState<string[]>([]), [motion, setMotion] = useState(true), [live, setLive] = useState(true);
  const [channel, setChannel] = useState("combined"), [floor, setFloor] = useState(-80), [real, setReal] = useState<Matrix | null>(null), [sim, setSim] = useState<Matrix | null>(null);
  const [sourceTab, setSourceTab] = useState<"camera" | "labels">("camera"), [inputTab, setInputTab] = useState<"scene" | "edit">("scene");
  const [loading, setLoading] = useState(false), [error, setError] = useState<string | null>(null), [cameraFailed, setCameraFailed] = useState(false), [reload, setReload] = useState(0);
  const submitted = useRef(""), restoredJob = useRef<Saved | null>(null), saved = useRef(readSaved());
  const create = useCallback((request: SceneRequest) => { if (!record) return Promise.reject(new Error("请先载入实测帧")); return simulateComparison(record.record_id, request); }, [record]);
  const runner = useSceneRunner(create);
  const effective = useMemo(() => {
    if (!input) return null;
    const result = clone(input); result.scene.objects = result.scene.objects.filter(o => !disabled.includes(o.id));
    if (!motion) result.scene.objects.forEach(o => { o.velocity_ned_mps = [0, 0, 0]; });
    return result;
  }, [input, disabled, motion]);
  const signature = effective ? JSON.stringify({ record: record?.record_id, request: effective }) : "";
  const currentSequence = catalog?.sequences.find(sequence => sequence.frames.some(frame => frame.id === selected));
  const frameList = currentSequence?.frames ?? [], selectedIndex = frameList.findIndex(frame => frame.id === selected);

  async function refreshCatalog() {
    setError(null);
    try {
      const result = await getCatalog(); setCatalog(result);
      if (!result.available) { setSelected(""); return; }
      const all = result.sequences.flatMap(s => s.frames), previous = saved.current?.frameId;
      if (previous && all.some(f => f.id === previous)) setSelected(previous);
      else { const sequence = result.sequences.find(s => s.id === "2019_05_09_cm1s003") ?? result.sequences[0]; setSelected((sequence.frames.find(f => f.frame_index === 451) ?? sequence.frames[Math.floor(sequence.frames.length / 2)]).id); }
    } catch (reason) { setError(String(reason)); }
  }
  useEffect(() => { void refreshCatalog(); }, []);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController(); let active = true;
    setLoading(true); setError(null); setRecord(null); setInput(null); setReal(null); setSim(null); setCameraFailed(false); submitted.current = "";
    void prepareReal(selected, controller.signal).then(async value => {
      if (!active) return;
      const stored = saved.current?.frameId === selected ? saved.current : null;
      const request = stored?.request ?? value.template ?? emptyTemplate(value);
      const excluded = stored?.disabled ?? [], moving = stored?.motion ?? true;
      setRecord(value); setInput(request); setDisabled(excluded); setMotion(moving); setLive(stored?.live ?? true); setChannel("combined");
      setSourceTab(value.camera_available ? "camera" : "labels");
      const resolved = clone(request); resolved.scene.objects = resolved.scene.objects.filter(o => !excluded.includes(o.id));
      if (!moving) resolved.scene.objects.forEach(o => { o.velocity_ned_mps = [0, 0, 0]; });
      const key = JSON.stringify({ record: value.record_id, request: resolved });
      if (stored?.jobId && stored.recordId === value.record_id && stored.signature === key) {
        submitted.current = key;
        try { await runner.resume(stored.jobId); if (!active) return; restoredJob.current = stored; }
        catch (reason) { if (active) { submitted.current = ""; if (!(reason instanceof HttpError && reason.status === 404)) setError("旧任务无法恢复，可以重新运行。"); } }
      }
    }).catch(reason => { if (active && !isAbort(reason)) setError(String(reason)); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; controller.abort(); };
  }, [selected, reload]);

  useEffect(() => {
    if (!record || !effective || !live || loading || !effective.scene.objects.length || submitted.current === signature) return;
    const timer = window.setTimeout(() => { submitted.current = signature; runner.run(effective); }, 650);
    return () => window.clearTimeout(timer);
  }, [signature, live, loading, runner.run]);

  useEffect(() => {
    if (!record || !input || record.frame_id !== selected) return;
    const next: Saved = { frameId: selected, request: input, disabled, motion, live };
    if (runner.job && submitted.current === signature) Object.assign(next, { jobId: runner.job.job_id, recordId: record.record_id, signature });
    saved.current = next;
    try { localStorage.setItem(STORAGE, JSON.stringify(next)); } catch { /* Optional persistence. */ }
  }, [record, input, disabled, motion, live, runner.job?.job_id]);

  useEffect(() => {
    if (!record) { setReal(null); return; }
    const controller = new AbortController(); setReal(null);
    void realMatrix(record.record_id, channel, controller.signal).then(value => { if (!controller.signal.aborted) setReal(value); }).catch(reason => { if (!isAbort(reason)) setError(String(reason)); });
    return () => controller.abort();
  }, [record?.record_id, channel]);

  useEffect(() => {
    if (!record || !runner.job?.frames.length) { setSim(null); return; }
    const controller = new AbortController(); setSim(null);
    void comparisonMatrix(record.record_id, runner.job.job_id, channel, controller.signal).then(value => { if (!controller.signal.aborted) setSim(value); }).catch(reason => { if (!isAbort(reason)) setError(String(reason)); });
    return () => controller.abort();
  }, [record?.record_id, runner.job?.job_id, Boolean(runner.job?.frames.length), channel]);

  function invalidate() { runner.invalidate(); submitted.current = ""; restoredJob.current = null; setSim(null); setError(null); }
  function changeFrame(id: string) {
    invalidate(); setRecord(null); setInput(null); setReal(null); setChannel("combined"); saved.current = { frameId: id, live };
    try { localStorage.setItem(STORAGE, JSON.stringify(saved.current)); } catch { /* Optional. */ }
    setSelected(id);
  }
  function run() { if (!effective?.scene.objects.length || !record) return; submitted.current = signature; setSim(null); runner.run(effective); }
  function restore() { if (!record) return; invalidate(); setInput(clone(record.template ?? emptyTemplate(record))); setDisabled([]); setMotion(true); }
  const showError = error ?? runner.error;
  return <div className="wb-page wb-comparison">
    <header className="wb-page-header"><div><h1>实测对比</h1><p>同一帧的公开数据与可编辑仿真输入，使用相同处理网格。</p></div><div className="wb-toolbar"><label className="wb-live"><input type="checkbox" checked={live} onChange={event => setLive(event.target.checked)} aria-label="自动更新对比仿真" />自动更新</label>{runner.busy && <button className="wb-button secondary" onClick={() => { setLive(false); void runner.stop(); }}><Square size={14} />停止</button>}<button className="wb-button primary" data-testid="run-comparison" disabled={!record || !effective?.scene.objects.length || loading} onClick={run}><Play size={16} />{runner.busy ? "按输入重算" : "运行仿真对比"}</button></div></header>
    {showError && <div className="wb-alert error" role="alert">{showError}<button className="wb-text-button" onClick={() => { invalidate(); setReload(v => v + 1); }}>重新载入当前帧</button></div>}
    {!catalog?.available ? <div className="wb-empty-source"><h2>{catalog ? "尚无可用的原始雷达数据" : "正在读取数据目录…"}</h2><p>{catalog?.configured === false ? "数据位置尚未配置。可以继续使用场景仿真；这里不会用合成数据代替实测。" : "需要本地公开 ADC 和配套标签；没有数据时不生成虚假的实测图。"}</p><button className="wb-button secondary" onClick={() => { void refreshCatalog(); }}>重新检测数据</button></div> : <>
      <div className="wb-data-toolbar">
        <label className="wb-field"><span>数据序列 · RAMP 77 GHz</span><select aria-label="数据序列" value={currentSequence?.id ?? ""} onChange={event => { const sequence = catalog.sequences.find(s => s.id === event.target.value)!; changeFrame((sequence.frames.find(f => f.frame_index === 451) ?? sequence.frames[Math.floor(sequence.frames.length / 2)]).id); }}>{catalog.sequences.map(sequence => <option value={sequence.id} key={sequence.id}>{sequence.name}</option>)}</select></label>
        <label className="wb-field"><span>原始帧</span><select aria-label="原始帧" value={selected} onChange={event => changeFrame(event.target.value)}>{frameList.map(frame => <option value={frame.id} key={frame.id}>{frame.name}{frame.camera_available ? " · 有相机" : " · 无相机"}</option>)}</select></label>
        <div className="wb-frame-arrows"><button className="wb-icon-button" aria-label="上一原始帧" disabled={selectedIndex <= 0} onClick={() => changeFrame(frameList[selectedIndex - 1].id)}><ChevronLeft size={17} /></button><button className="wb-icon-button" aria-label="下一原始帧" disabled={selectedIndex < 0 || selectedIndex >= frameList.length - 1} onClick={() => changeFrame(frameList[selectedIndex + 1].id)}><ChevronRight size={17} /></button></div>
        <label className="wb-field"><span>两侧显示通道</span><select aria-label="对比显示通道" value={channel} onChange={event => setChannel(event.target.value)} disabled={!record}><option value="combined">8 通道功率平均</option>{record?.channels.map(name => <option value={name} key={name}>{name.replace(/tx(\d+)_rx(\d+)/, (_all, tx, rx) => `Tx${Number(tx) + 1} / Rx${Number(rx) + 1}`)}</option>)}</select></label>
        <label className="wb-field"><span>两侧动态范围</span><select aria-label="对比动态范围" value={floor} onChange={event => setFloor(Number(event.target.value))}><option value={-60}>60 dB</option><option value={-80}>80 dB</option><option value={-100}>100 dB</option></select></label>
      </div>
      <div className="wb-source-line"><span className="wb-source-badge" data-testid="real-source">{record?.is_real ? "真实公开 ADC" : "正在载入真实数据"}</span><span>77 GHz · 2Tx4Rx · 128 采样 × 255 循环</span><span>{loading ? "正在读取原始帧…" : record ? `帧 ${record.frame_index.toString().padStart(6, "0")}` : "—"}</span><a href={catalog.source_url} target="_blank" rel="noreferrer">数据来源 ↗</a></div>
      <div className="wb-compare-grid">
        <section className="wb-panel"><div className="wb-panel-head"><h2>实测场景 / 目标</h2><div className="wb-tabs compact"><button className={sourceTab === "camera" ? "active" : ""} onClick={() => { setCameraFailed(false); setSourceTab("camera"); }} disabled={!record?.camera_available}>{cameraFailed ? "重试相机" : "配套相机"}</button><button className={sourceTab === "labels" ? "active" : ""} onClick={() => setSourceTab("labels")}>公开标注</button></div></div>
          <div className="wb-source-visual">{loading ? <div className="wb-empty-chart">正在载入场景…</div> : sourceTab === "camera" && record?.camera_url && !cameraFailed ? <img key={record.record_id} data-testid="real-camera" src={cameraURL(record.camera_url)} alt="当前雷达帧在公开数据集中同名配套的真实相机图像" onError={() => { setCameraFailed(true); setSourceTab("labels"); }} /> : <SceneSketch realLabels={record?.labels ?? []} height={290} />}</div>
          <div className="wb-caption">{sourceTab === "camera" && record?.camera_url && !cameraFailed ? "数据集同名配套相机；未将雷达标签伪投影到图像。" : record?.camera_available && !cameraFailed ? "公开二维位置标注，不是物体三维表面模型。" : "此帧无可用相机；当前显示公开位置标注，不是实物照片。"}</div>
        </section>
        <section className="wb-panel"><div className="wb-panel-head"><h2>真实 RD</h2><span className="wb-muted">原始 ADC 处理</span></div><Heatmap matrix={real} x={record?.range_m} y={record?.velocity_mps} title="当前原始帧真实 RD" height={360} floorDb={floor} empty={loading ? "正在处理真实 ADC…" : "请选择原始帧"} /><div className="wb-caption">真实与仿真共用本帧实测功率参考；切换通道同时作用于两侧。</div></section>
        <section className="wb-panel"><div className="wb-panel-head"><h2>仿真目标输入</h2><div className="wb-tabs compact"><button className={inputTab === "scene" ? "active" : ""} onClick={() => setInputTab("scene")}>场景</button><button className={inputTab === "edit" ? "active" : ""} onClick={() => setInputTab("edit")}>编辑输入</button><button className="wb-icon-button" aria-label="恢复标签输入" title="恢复标签输入" onClick={restore} disabled={!record}><RotateCcw size={14} /></button></div></div>
          <div className="wb-input-visual">{inputTab === "scene" ? <SceneSketch request={effective} height={290} /> : input ? <div className="wb-input-fields"><Switch label="使用输入目标速度" checked={motion} onChange={v => { invalidate(); setMotion(v); }} /><ObjectsEditor near request={input} onChange={v => { invalidate(); setInput(v); }} disabledIds={disabled} onDisabledIds={v => { invalidate(); setDisabled(v); }} motion={motion} /></div> : <div className="wb-empty-chart">等待目标输入…</div>}</div>
          <div className="wb-caption">{record?.template ? `${effective?.scene.objects.length ?? 0} 个启用对象；默认用标注中心近似。` : "此帧没有目标标签，可在“编辑输入”中手工添加。"}{record?.labels.some(l => l.radial_velocity_mps == null) && " 部分速度未知，默认以 0 占位。"}</div>
        </section>
        <section className="wb-panel"><div className="wb-panel-head"><h2>仿真 RD</h2><span className="wb-muted" data-testid="comparison-status" data-job-id={runner.job?.job_id ?? ""}>{runner.submitting ? "提交中" : runner.job?.status === "running" || runner.job?.status === "queued" ? "计算中" : runner.job?.status === "complete" ? `已完成 · ${format(runner.job.frames[0]?.elapsed_s ?? 0)} s` : "等待当前输入"}</span></div><Heatmap matrix={sim} x={record?.range_m} y={record?.velocity_mps} title="当前目标输入的仿真 RD" height={360} floorDb={floor} empty={runner.busy ? "正在用统一框架生成回波…" : effective?.scene.objects.length ? "修改输入后运行仿真" : "请先添加或启用仿真目标"} /><div className="wb-caption">实测波形、阵列与处理网格固定；只修改仿真目标，不移动真实图去对齐。</div></section>
      </div>
      <div className="wb-bottom-note">{record?.amplitude_note ?? "仿真采用相对反射幅度"}。公开标签和相机不包含完整三维/材料真值，真实背景与目标散射尚未完整复现。</div>
    </>}
  </div>;
}
