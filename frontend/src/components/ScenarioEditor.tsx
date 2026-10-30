import { Plus, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { clone, objectNames, objectLabel, type Features, type SceneObject, type SceneRequest, type Terrain, type Vec3 } from "../workbenchTypes";
import { format } from "./WorkbenchCharts";

export function NumberInput({ label, displayLabel, value, onChange, unit, min = -1e9, max = 1e9, step = 1, disabled = false, testId }: {
  label: string; displayLabel?: string; value: number; onChange: (v: number) => void; unit?: string; min?: number; max?: number; step?: number; disabled?: boolean; testId?: string;
}) {
  const [text, setText] = useState(String(value));
  const dirty = useRef(false);
  useEffect(() => { dirty.current = false; setText(Number.isFinite(value) ? String(Number(value.toPrecision(6))) : ""); }, [value]);
  function commit(resetInvalid = true) { if (!dirty.current) return; const next = Number(text); if (!text.trim() || !Number.isFinite(next) || next < min || next > max) { if (resetInvalid) { dirty.current = false; setText(String(Number(value.toPrecision(6)))); } return; } dirty.current = false; if (next !== value) onChange(next); }
  const commitLatest = useRef(commit); commitLatest.current = commit;
  useEffect(() => { if (!dirty.current) return; const timer = window.setTimeout(() => commitLatest.current(false), 400); return () => window.clearTimeout(timer); }, [text]);
  const invalid = dirty.current && (!text.trim() || !Number.isFinite(Number(text)) || Number(text) < min || Number(text) > max);
  return <label className="wb-field"><span title={label}>{displayLabel ?? label}</span><div><input aria-label={label} aria-valuemin={min} aria-valuemax={max} aria-invalid={invalid} title={`范围 ${min}–${max}${unit ? ` ${unit}` : ""}`} data-testid={testId} type="number" value={text} step={step} disabled={disabled}
    onChange={event => { dirty.current = true; setText(event.target.value); }} onBlur={() => commit()} onKeyDown={event => { if (event.key === "Enter") { commit(); event.currentTarget.blur(); } }} /><em>{unit}</em></div></label>;
}

export function Switch({ label, hint, checked, onChange, disabled = false, testId }: { label: string; hint?: string; checked: boolean; onChange: (value: boolean) => void; disabled?: boolean; testId?: string }) {
  return <label className={`wb-switch ${disabled ? "disabled" : ""}`}><span><strong>{label}</strong>{hint && <small>{hint}</small>}</span><input type="checkbox" role="switch" aria-label={label} data-testid={testId} checked={checked} disabled={disabled} onChange={event => onChange(event.target.checked)} /></label>;
}

function VectorFields({ value, onChange, prefix, velocity = false, disabled = false }: { value: Vec3; onChange: (v: Vec3) => void; prefix: string; velocity?: boolean; disabled?: boolean }) {
  const names = velocity ? ["北向速度", "东向速度", "上升速度"] : ["North", "East", "高程"];
  return <div className="wb-fields three">{names.map((name, i) => <NumberInput key={name} label={`${prefix} ${name}`} displayLabel={name} value={i === 2 ? -value[i] : value[i]} unit={velocity ? "m/s" : "m"} step={velocity ? 1 : 10} disabled={disabled}
    onChange={v => { const result = [...value] as Vec3; result[i] = i === 2 ? -v : v; onChange(result); }} />)}</div>;
}

interface ObjectsProps { request: SceneRequest; onChange: (v: SceneRequest) => void; disabledIds: string[]; onDisabledIds: (ids: string[]) => void; motion: boolean; near?: boolean }
export function ObjectsEditor({ request, onChange, disabledIds, onDisabledIds, motion, near = false }: ObjectsProps) {
  const [addKind, setAddKind] = useState<"point" | "box" | "plate">("point");
  function change(index: number, change: Partial<SceneObject>) { const next = clone(request); next.scene.objects[index] = { ...next.scene.objects[index], ...change }; onChange(next); }
  function add() {
    const next = clone(request); let suffix = 1; while (next.scene.objects.some(o => o.id === `目标${suffix}`)) suffix++;
    const p = request.platform.position_ned_m, b = request.platform.boresight_ned;
    const scale = b[2] > 1e-6 ? -p[2] / b[2] : 0;
    const pos: Vec3 = near ? [10, 0, 0] : [p[0] + b[0] * scale, p[1] + b[1] * scale, 0];
    const object: SceneObject = { id: `目标${suffix}`, kind: addKind, position_ned_m: pos, velocity_ned_mps: [0, 0, 0] };
    if (addKind === "point") { object.amplitude = near ? 20 : 1; object.phase_rad = 0; }
    else {
      object.relative_reflectivity = .1; object.scattering = "diffuse"; object.divisions = 2; object.yaw_deg = 0; object.two_sided = false;
      if (addKind === "box") { object.size_m = near ? [4, 1.8, 1.5] : [40, 30, 20]; object.position_ned_m[2] = -object.size_m[2] / 2; }
      else { object.size_m = near ? [1, 1] : [20, 20]; object.normal_ned = near ? [-1, 0, 0] : [0, 0, -1]; }
    }
    next.scene.objects.push(object); onChange(next);
  }
  return <div className="wb-object-editor">
    {request.scene.objects.map((object, index) => <details className="wb-object" key={object.id} data-edit-object={object.id}>
      <summary><input type="checkbox" aria-label={`启用 ${object.id}`} checked={!disabledIds.includes(object.id)} onClick={event => event.stopPropagation()} onChange={event => onDisabledIds(event.target.checked ? disabledIds.filter(id => id !== object.id) : [...disabledIds, object.id])} /><span title={object.id}>{objectLabel(object.id)}</span><small>{objectNames[object.kind]}</small></summary>
      <div className="wb-object-fields">
        <VectorFields prefix={object.id} value={object.position_ned_m} onChange={v => change(index, { position_ned_m: v })} />
        <VectorFields prefix={object.id} value={object.velocity_ned_mps ?? [0, 0, 0]} velocity disabled={!motion} onChange={v => change(index, { velocity_ned_mps: v })} />
        {object.kind === "point" && <NumberInput label={`${object.id} 相对幅度`} value={object.amplitude ?? 1} min={.000001} max={100000} step={.1} onChange={v => change(index, { amplitude: v })} />}
        {object.relative_reflectivity !== undefined && <NumberInput label={`${object.id} 相对反射强度`} value={object.relative_reflectivity} min={.000001} max={1000} step={.01} onChange={v => change(index, { relative_reflectivity: v })} />}
        {object.size_m && <div className="wb-fields three">{object.size_m.map((value, dimension) => <NumberInput key={dimension} label={`${object.id} ${["长度", "宽度", "高度"][dimension]}`} value={value} unit="m" min={.001} max={10000} onChange={v => { const size = [...object.size_m!]; size[dimension] = v; change(index, { size_m: size }); }} />)}</div>}
        {object.divisions != null && <div className="wb-fields"><NumberInput label={`${object.id} 面片细分`} displayLabel="面片细分" value={object.divisions} min={1} max={object.kind === "plate" ? 32 : 12} onChange={v => change(index, { divisions: Math.round(v) })} /><NumberInput label={`${object.id} 固定旋转角`} displayLabel="固定旋转角" value={object.yaw_deg ?? 0} unit="°" min={-180} max={180} onChange={v => change(index, { yaw_deg: v })} /></div>}
        <button type="button" className="wb-text-button danger" onClick={() => { const next = clone(request); next.scene.objects.splice(index, 1); onDisabledIds(disabledIds.filter(id => id !== object.id)); onChange(next); }}><Trash2 size={13} />移除目标</button>
      </div>
    </details>)}
    {!request.scene.objects.length && <p className="wb-hint">暂无目标。可以手工添加，或导入场景配置。</p>}
    <div className="wb-add-object"><select aria-label="新增目标类型" value={addKind} onChange={event => setAddKind(event.target.value as typeof addKind)}><option value="point">点目标</option><option value="box">长方体</option><option value="plate">平板</option></select><button type="button" className="wb-button secondary small" onClick={add}><Plus size={14} />添加</button></div>
    <p className="wb-hint">位置和速度使用世界坐标；幅度是相对值，不是已标定的材料 RCS。</p>
  </div>;
}

const defaultTerrain = (): Terrain => ({ id: "terrain", kind: "flat", north_bounds_m: [350, 1150], east_bounds_m: [-400, 400], spacing_m: 100, base_elevation_m: 0, relative_reflectivity: .003, scattering: "diffuse", hills: [] });
const defaultHills = () => [{ center_ne_m: [550, -160] as [number, number], height_m: 120, sigma_ne_m: [160, 180] as [number, number] }];
export function ScenarioEditor({ request, onChange, features, setFeatures, auto, setAuto, presets, preset, onPreset }: {
  request: SceneRequest; onChange: (v: SceneRequest) => void; features: Features; setFeatures: (v: Features) => void;
  auto: boolean; setAuto: (v: boolean) => void; presets: Record<string, SceneRequest>; preset: string; onPreset: (key: string) => void;
}) {
  const w = request.waveform, terrain = request.scene.terrain ?? defaultTerrain();
  const bandwidth = w.kind === "pulsed_lfm" ? w.bandwidth_hz! : w.slope_hz_s! * w.samples / w.sample_rate_hz;
  const frequency = w.kind === "pulsed_lfm" ? w.carrier_hz! : w.start_hz!;
  const dive = Math.atan2(request.platform.boresight_ned[2], Math.hypot(...request.platform.boresight_ned.slice(0, 2))) * 180 / Math.PI;
  const heading = Math.atan2(request.platform.boresight_ned[1], request.platform.boresight_ned[0]) * 180 / Math.PI;
  const update = (f: (r: SceneRequest) => void) => { const next = clone(request); f(next); onChange(next); };
  const beam = (angle: number, direction: number) => update(r => { const a = angle * Math.PI / 180, h = direction * Math.PI / 180; r.platform.boresight_ned = [Math.cos(a) * Math.cos(h), Math.cos(a) * Math.sin(h), Math.sin(a)]; r.platform.right_hint_ned = [-Math.sin(h), Math.cos(h), 0]; });
  const setTerrain = (value: Terrain) => update(r => { r.scene.terrain = value; });
  const grid = (max: number) => Array.from({ length: Math.floor(max) + 1 }, (_, i) => -max + 2 * i);
  async function importDem(file?: File) {
    if (!file) return;
    try {
      if (file.size > 4_000_000) throw new Error("DEM JSON 超过 4 MB");
      const value = JSON.parse(await file.text());
      if (!Array.isArray(value.north_m) || !Array.isArray(value.east_m) || !Array.isArray(value.elevation_up_m)) throw new Error("需要 north_m、east_m、elevation_up_m 三个数组");
      setTerrain({ ...defaultTerrain(), kind: "dem", north_m: value.north_m, east_m: value.east_m, elevation_up_m: value.elevation_up_m });
      setFeatures({ ...features, terrain: true });
    } catch (error) { window.alert(error instanceof Error ? error.message : String(error)); }
  }
  return <>
    <label className="wb-field wb-preset"><span>场景预设</span><select aria-label="场景预设" value={preset} onChange={event => onPreset(event.target.value)}>{!presets[preset] && <option value={preset}>自定义配置</option>}{Object.entries(presets).map(([key, p]) => <option value={key} key={key}>{p.name}</option>)}</select></label>
    <details className="wb-section" open><summary>雷达与采集</summary><div className="wb-fields">
      <label className="wb-field"><span>波形</span><select aria-label="波形" value={w.kind} onChange={event => update(r => { const next = presets[`mixed_${event.target.value}`]; r.waveform = clone(next.waveform); })}><option value="pulsed_lfm">脉冲 LFM</option><option value="fmcw">FMCW</option></select></label>
      <NumberInput label={w.kind === "pulsed_lfm" ? "载频" : "起始频率"} testId="carrier" value={frequency / 1e9} unit="GHz" min={.1} max={100} step={.5} onChange={v => update(r => { if (r.waveform.kind === "pulsed_lfm") r.waveform.carrier_hz = v * 1e9; else r.waveform.start_hz = v * 1e9; })} />
      <NumberInput label="采样带宽" testId="bandwidth" value={bandwidth / 1e6} unit="MHz" min={1} max={w.kind === "fmcw" ? 2000 : 500} step={5} onChange={v => update(r => { if (r.waveform.kind === "pulsed_lfm") r.waveform.bandwidth_hz = v * 1e6; else r.waveform.slope_hz_s = v * 1e6 / (r.waveform.samples / r.waveform.sample_rate_hz); })} />
      {w.kind === "pulsed_lfm" ? <NumberInput label="发射 PRF" value={1 / w.slot_period_s / 1000} unit="kHz" min={.5} max={100} step={1} onChange={v => update(r => { r.waveform.slot_period_s = 1 / (v * 1000); })} /> : <NumberInput label="发射间隔" value={w.slot_period_s * 1e6} unit="μs" min={10} max={10000} step={10} onChange={v => update(r => { r.waveform.slot_period_s = v * 1e-6; })} />}
      <NumberInput label="CPI 循环数" value={w.loops} min={4} max={512} step={16} unit="次/Tx" onChange={v => update(r => { r.waveform.loops = Math.round(v); })} />
      <NumberInput label="序列帧数" testId="frame-count" value={request.sequence.frame_count} min={1} max={100} unit="帧" onChange={v => update(r => { r.sequence.frame_count = Math.round(v); })} />
    </div><button type="button" className="wb-text-button" onClick={() => setAuto(true)}>计算并应用采样建议</button>
    {!auto && <div className="wb-fields"><NumberInput label="ADC 采样率" value={w.sample_rate_hz / 1e6} unit="MHz" min={.1} max={500} onChange={v => update(r => { r.waveform.sample_rate_hz = v * 1e6; })} /><NumberInput label="采样点数" value={w.samples} min={8} max={32768} step={128} onChange={v => update(r => { r.waveform.samples = Math.round(v); })} /><NumberInput label="ADC 开始时间" value={w.adc_start_s * 1e6} unit="μs" min={0} max={10000} onChange={v => update(r => { r.waveform.adc_start_s = v * 1e-6; })} />{w.kind === "fmcw" && <NumberInput label="调频时长" value={w.ramp_s! * 1e6} unit="μs" min={1} max={10000} onChange={v => update(r => { r.waveform.ramp_s = v * 1e-6; })} />}<NumberInput label="显示最小距离" value={request.processing.range_limits_m[0]} unit="m" min={0} onChange={v => update(r => { r.processing.range_limits_m[0] = v; })} /><NumberInput label="显示最大距离" value={request.processing.range_limits_m[1]} unit="m" min={1} onChange={v => update(r => { r.processing.range_limits_m[1] = v; })} /></div>}
    </details>
    <details className="wb-section" data-editor-section="platform" open><summary>平台与指向</summary><div className="wb-fields"><NumberInput label="平台高度" testId="height" disabled={Boolean(request.platform.trajectory)} value={-request.platform.position_ned_m[2] / 1000} unit="km" min={0} max={30} step={.1} onChange={v => update(r => { r.platform.position_ned_m[2] = -v * 1000; })} /><NumberInput label="波束半角" value={request.platform.beam_half_angle_deg} unit="°" min={.1} max={89} onChange={v => update(r => { r.platform.beam_half_angle_deg = v; })} /><NumberInput label="下视角" value={dive} unit="°" min={0} max={90} onChange={v => beam(v, heading)} /><NumberInput label="指向方位" value={heading} unit={"°"} min={-180} max={180} onChange={v => beam(dive, v)} /></div>
      <Switch label="平台运动" disabled={Boolean(request.platform.trajectory)} hint={request.platform.trajectory?"采样轨迹已覆盖常量状态，请在完整配置中编辑轨迹":undefined} checked={features.platformMotion} onChange={v => setFeatures({ ...features, platformMotion: v })} testId="platform-motion" />
      <VectorFields prefix="平台" value={request.platform.velocity_ned_mps} velocity disabled={!features.platformMotion||Boolean(request.platform.trajectory)} onChange={v => update(r => { r.platform.velocity_ned_mps = v; })} />
    </details>
    <details className="wb-section" open><summary>功能开关</summary>
      <Switch label="地形回波" hint="开关实际控制地形是否参与计算" checked={features.terrain} onChange={v => { if (v && !request.scene.terrain) setTerrain(defaultTerrain()); setFeatures({ ...features, terrain: v }); }} testId="terrain-toggle" />
      {features.terrain && <div className="wb-subsection"><label className="wb-field"><span>地形输入</span><select aria-label="地形输入" value={terrain.kind} onChange={event => { const kind = event.target.value as Terrain["kind"]; if (kind === "dem") return; setTerrain({ ...terrain, kind, hills: kind === "hills" ? defaultHills() : [], north_m: null, east_m: null, elevation_up_m: null }); }}><option value="flat">合成平地</option><option value="hills">合成山丘</option>{terrain.kind === "dem" && <option value="dem">已导入 DEM</option>}</select></label>
        <div className="wb-fields">{terrain.kind !== "dem" && <NumberInput label="地形网格间距" value={terrain.spacing_m} min={10} max={1000} unit="m" onChange={v => setTerrain({ ...terrain, spacing_m: v })} />}<NumberInput label="地形反射强度" value={terrain.relative_reflectivity} min={.000001} max={1000} step={.001} onChange={v => setTerrain({ ...terrain, relative_reflectivity: v })} />{terrain.kind === "hills" && <NumberInput label="山丘高度参数" value={terrain.hills[0]?.height_m ?? 120} min={0} max={3000} unit="m" onChange={v => setTerrain({ ...terrain, hills: terrain.hills.map((h, i) => i === 0 ? { ...h, height_m: v } : h) })} />}</div>
        {terrain.kind === "dem" && <p className="wb-hint">已载入 {terrain.north_m?.length} × {terrain.east_m?.length} 高程网格，间距以文件坐标为准。</p>}
        <label className="wb-file-button">导入 DEM 高程 JSON<input type="file" accept=".json,application/json" onChange={e => { void importDem(e.target.files?.[0]); e.currentTarget.value = ""; }} /></label><p className="wb-hint">合成地形不是实测数据；DEM 不自动包含地物材料分类。</p>
      </div>}
      <Switch label="多通道 / 阵列" checked={features.array} onChange={v => { if (v && request.array.layout === "siso") update(r => { r.array = { layout: "ula", num_tx: 2, num_rx: 4, spacing_m: null }; r.processing.azimuth_deg = grid(20); r.processing.elevation_deg = [0]; }); setFeatures({ ...features, array: v }); }} testId="array-toggle" />
      {features.array && <div className="wb-subsection"><label className="wb-field"><span>阵列布局</span><select aria-label="阵列布局" value={request.array.layout === "siso" ? "ula" : request.array.layout} onChange={event => update(r => { const layout = event.target.value as "ula" | "ura"; r.array = { layout, num_tx: r.array.num_tx, num_rx: r.array.num_rx, spacing_m: r.array.spacing_m }; r.processing.elevation_deg = layout === "ura" ? grid(20) : [0]; r.processing.azimuth_deg = grid(20); })}><option value="ula">一维 ULA · 方位</option><option value="ura">二维 URA · 方位 + 俯仰</option>{request.array.layout === "custom" && <option value="custom">导入的自定义阵列</option>}</select></label>
        <div className="wb-fields"><NumberInput label="Tx 数量" value={request.array.num_tx} min={1} max={8} disabled={request.array.layout === "custom"} onChange={v => update(r => { r.array.num_tx = Math.round(v); })} /><NumberInput label="Rx 数量" value={request.array.num_rx} min={1} max={32} disabled={request.array.layout === "custom"} onChange={v => update(r => { r.array.num_rx = Math.round(v); })} /></div>
        <Switch label="半波长阵元间距" hint="频率变化时自动调整物理间距" checked={request.array.spacing_m == null} disabled={request.array.layout === "custom"} onChange={v => update(r => { r.array.spacing_m = v ? null : 299792458 / frequency / 2; })} />
        {request.array.spacing_m != null && <NumberInput label="固定阵元间距" value={request.array.spacing_m * 1000} unit="mm" min={.01} max={10000} onChange={v => update(r => { r.array.spacing_m = v / 1000; })} />}
        <Switch label="TDM 运动相位补偿" hint="主要影响角度图，不改变通道功率平均 RD" checked={request.processing.tdm_compensation} disabled={request.array.num_tx < 2} onChange={v => update(r => { r.processing.tdm_compensation = v; })} testId="tdm-toggle" />
      </div>}
      <Switch label="目标运动" checked={features.targetMotion} onChange={v => setFeatures({ ...features, targetMotion: v })} testId="target-motion" />
    </details>
    <details className="wb-section" open><summary>仿真目标 <small>{request.scene.objects.length}</small></summary><ObjectsEditor request={request} onChange={onChange} disabledIds={features.disabledObjects} onDisabledIds={ids => setFeatures({ ...features, disabledObjects: ids })} motion={features.targetMotion} /></details>
    <details className="wb-section"><summary>其他采集参数</summary><div className="wb-fields">
      {w.kind === "pulsed_lfm" && <NumberInput label="脉宽" value={w.pulse_width_s! * 1e6} min={.1} max={100} unit="μs" step={1} onChange={v => update(r => { r.waveform.pulse_width_s = v * 1e-6; })} />}
      <NumberInput label="帧间隔" value={request.sequence.frame_interval_s * 1000} min={1} max={10000} unit="ms" step={50} onChange={v => update(r => { r.sequence.frame_interval_s = v / 1000; })} />
      <NumberInput label="相位随机种子" value={request.scene.seed} min={0} max={4294967295} onChange={v => update(r => { r.scene.seed = Math.round(v); })} />
      <NumberInput label="方位扫描范围 ±" value={Math.max(...request.processing.azimuth_deg.map(Math.abs))} min={2} max={80} unit="°" onChange={v => update(r => { r.processing.azimuth_deg = grid(v); })} />
      <NumberInput label="平台 North 位置" value={request.platform.position_ned_m[0]} unit="m" onChange={v => update(r => { r.platform.position_ned_m[0] = v; })} /><NumberInput label="平台 East 位置" value={request.platform.position_ned_m[1]} unit="m" onChange={v => update(r => { r.platform.position_ned_m[1] = v; })} />
    </div><Switch label="保存原始复数 ADC" checked={request.processing.save_adc} onChange={v => update(r => { r.processing.save_adc = v; })} /><p className="wb-hint">更多网格、阵元、散射表、轨迹与背景参数可通过完整配置编辑。</p></details>
  </>;
}
