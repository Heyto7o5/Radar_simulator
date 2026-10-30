import { useEffect, useMemo, useRef, useState } from "react";
import type { Matrix, RealLabel, SceneGeometry, SceneRequest, Vec3 } from "../workbenchTypes";
import { objectLabel } from "../workbenchTypes";

const colors = [[12, 25, 49], [28, 67, 126], [29, 132, 160], [66, 187, 167], [211, 215, 113], [240, 163, 58], [200, 65, 40]];
function colorAt(t: number) {
  const location = Math.max(0, Math.min(1, t)) * (colors.length - 1), low = Math.floor(location), high = Math.min(low + 1, colors.length - 1), f = location - low;
  return colors[low].map((v, i) => Math.round(v * (1 - f) + colors[high][i] * f));
}
export const format = (v: number, digits = 2) => Number.isFinite(v) ? v.toLocaleString("zh-CN", { maximumFractionDigits: digits }) : "—";

interface HeatmapProps {
  matrix?: Matrix | null; x?: number[]; y?: number[]; xLabel?: string; yLabel?: string;
  floorDb?: number; ceilingDb?: number; height?: number; title: string; empty?: string;
  onPick?: (column: number, row: number) => void;
}
export function Heatmap({ matrix, x = [], y = [], xLabel = "距离 / m", yLabel = "表观接近速度 / m/s", floorDb = -80, ceilingDb = 6, height = 360, title, empty = "配置参数后运行仿真", onPick }: HeatmapProps) {
  const holder = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(600), [hover, setHover] = useState<{ px: number; py: number; text: string } | null>(null);
  const margin = { left: 68, right: 68, top: 20, bottom: 54 };
  useEffect(() => {
    if (!holder.current) return;
    const observer = new ResizeObserver(entries => setWidth(Math.max(220, Math.floor(entries[0].contentRect.width))));
    observer.observe(holder.current); return () => observer.disconnect();
  }, []);
  useEffect(() => setHover(null), [matrix]);
  useEffect(() => {
    const el = canvas.current;
    if (!el || !matrix || x.length !== matrix.cols || y.length !== matrix.rows) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2); el.width = width * dpr; el.height = height * dpr;
    const ctx = el.getContext("2d"); if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    const plotWidth = Math.max(1, width - margin.left - margin.right), plotHeight = height - margin.top - margin.bottom;
    const sx = Math.max(1, Math.ceil(matrix.cols / 900)), sy = Math.max(1, Math.ceil(matrix.rows / 600));
    const cols = Math.ceil(matrix.cols / sx), rows = Math.ceil(matrix.rows / sy);
    const buffer = document.createElement("canvas"); buffer.width = cols; buffer.height = rows;
    const bctx = buffer.getContext("2d"); if (!bctx) return;
    const image = bctx.createImageData(cols, rows);
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      let value = -Infinity;
      for (let dy = row * sy; dy < Math.min((row + 1) * sy, matrix.rows); dy++) for (let dx = col * sx; dx < Math.min((col + 1) * sx, matrix.cols); dx++) value = Math.max(value, matrix.data[dy * matrix.cols + dx]);
      const rgb = colorAt((value - floorDb) / (ceilingDb - floorDb)), offset = ((rows - row - 1) * cols + col) * 4;
      image.data[offset] = rgb[0]; image.data[offset + 1] = rgb[1]; image.data[offset + 2] = rgb[2]; image.data[offset + 3] = 255;
    }
    bctx.putImageData(image, 0, 0); ctx.imageSmoothingEnabled = false; ctx.drawImage(buffer, margin.left, margin.top, plotWidth, plotHeight);
    ctx.strokeStyle = "#d8e1eb"; ctx.strokeRect(margin.left, margin.top, plotWidth, plotHeight);
    ctx.fillStyle = "#607083"; ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "middle";
    for (let tick = 0; tick <= 4; tick++) {
      const t = tick / 4; ctx.textAlign = "center";
      ctx.fillText(format(x[0] + (x[x.length - 1] - x[0]) * t, Math.abs(x[x.length - 1] - x[0]) < 10 ? 2 : 0), margin.left + t * plotWidth, margin.top + plotHeight + 18);
      ctx.textAlign = "right"; ctx.fillText(format(y[0] + (y[y.length - 1] - y[0]) * t, 1), margin.left - 9, margin.top + (1 - t) * plotHeight);
    }
    ctx.textAlign = "center"; ctx.fillText(xLabel, margin.left + plotWidth / 2, height - 11);
    ctx.save(); ctx.translate(15, margin.top + plotHeight / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
    const gradient = ctx.createLinearGradient(0, margin.top + plotHeight, 0, margin.top);
    colors.forEach((c, i) => gradient.addColorStop(i / (colors.length - 1), `rgb(${c.join(",")})`));
    ctx.fillStyle = gradient; ctx.fillRect(width - 48, margin.top, 11, plotHeight);
    ctx.fillStyle = "#607083"; ctx.textAlign = "left";
    for (let i = 0; i <= 4; i++) ctx.fillText(format(floorDb + (ceilingDb - floorDb) * i / 4, 0), width - 32, margin.top + plotHeight * (1 - i / 4));
    ctx.fillText("dB", width - 45, height - 27);
  }, [matrix, x, y, width, height, floorDb, ceilingDb, xLabel, yLabel]);
  function location(event: React.MouseEvent) {
    const rect = holder.current!.getBoundingClientRect(), px = event.clientX - rect.left, py = event.clientY - rect.top;
    const tx = (px - margin.left) / (width - margin.left - margin.right), ty = 1 - (py - margin.top) / (height - margin.top - margin.bottom);
    if (!matrix || tx < 0 || tx > 1 || ty < 0 || ty > 1) return null;
    return { px, py, col: Math.min(matrix.cols - 1, Math.floor(tx * matrix.cols)), row: Math.min(matrix.rows - 1, Math.floor(ty * matrix.rows)) };
  }
  const hasSignal = matrix?.data.some(v => v > floorDb + .01);
  return <div className="wb-heatmap" ref={holder} style={{ height }} data-testid="heatmap" data-loaded={Boolean(matrix)} data-rows={matrix?.rows ?? 0} data-cols={matrix?.cols ?? 0} data-reference={matrix?.reference ?? ""}>
    {matrix ? <><canvas ref={canvas} style={{ width, height }} role="img" aria-label={title}
      onMouseMove={event => { const point = location(event); setHover(point ? { px: point.px, py: point.py, text: `${format(x[point.col])} / ${format(y[point.row])} · ${format(matrix.data[point.row * matrix.cols + point.col], 1)} dB` } : null); }}
      onMouseLeave={() => setHover(null)} onClick={event => { const point = location(event); if (point) onPick?.(point.col, point.row); }} />
      {!hasSignal && <div className="wb-no-signal">当前色阶下没有可见回波；可增大显示范围，或检查输入与接收窗</div>}
      {hover && <div className="wb-chart-tooltip" style={{ left: Math.min(hover.px + 12, width - 225), top: hover.py - 30 }}>{hover.text}</div>}</> : <div className="wb-empty-chart">{empty}</div>}
  </div>;
}

const objectColor = (kind: string) => kind === "point" ? "#dc8434" : kind === "box" ? "#5078af" : kind === "plate" ? "#7760a8" : "#43897d";
export function SceneSketch({ request, geometry, realLabels, side = false, height = 300 }: { request?: SceneRequest | null; geometry?: SceneGeometry | null; realLabels?: RealLabel[]; side?: boolean; height?: number }) {
  const objects = request?.scene.objects ?? [];
  const terrain = geometry?.terrain;
  const positions: Vec3[] = realLabels ? realLabels.map(l => [l.y_m, l.x_m, 0]) : objects.map(o => o.position_ned_m);
  const platform = request?.platform.position_ned_m ?? [0, 0, 0];
  const transform = (p: Vec3) => side ? [p[0], -p[2]] : [p[1], p[0]];
  const bounds = useMemo(() => {
    let xy = positions.map(transform);
    const near = Boolean(realLabels) || positions.every(p => Math.hypot(...p.map((v, i) => v - platform[i])) < 100);
    if (terrain) xy = [...xy, ...terrain.north_m.flatMap(n => terrain.east_m.map(e => transform([n, e, 0])))];
    if (near || side) xy.push(transform(platform));
    if (!xy.length) xy = [[-10, 0], [10, 20]];
    let x0 = Math.min(...xy.map(p => p[0])), x1 = Math.max(...xy.map(p => p[0])), y0 = Math.min(...xy.map(p => p[1])), y1 = Math.max(...xy.map(p => p[1]));
    const dx = Math.max(x1 - x0, near ? 20 : 100), dy = Math.max(y1 - y0, near ? 20 : 100);
    x0 -= dx * .12; x1 = Math.max(x1, x0 + dx) + dx * .12; y0 -= dy * .12; y1 = Math.max(y1, y0 + dy) + dy * .12;
    return [x0, x1, y0, y1];
  }, [JSON.stringify(positions), JSON.stringify(terrain), JSON.stringify(platform), Boolean(realLabels), side]);
  const width = 600, left = 57, top = 23, pw = 510, ph = height - 67;
  const px = (x: number) => left + (x - bounds[0]) / (bounds[1] - bounds[0]) * pw;
  const py = (y: number) => top + ph - (y - bounds[2]) / (bounds[3] - bounds[2]) * ph;
  const peak = terrain ? Math.max(1, ...terrain.elevation_up_m.flat()) : 1;
  const previewObjects = geometry?.objects ?? [];
  const point = (o: SceneObjectLike, i: number) => {
    const [x, y] = transform(o.position_ned_m), color = objectColor(o.kind), descriptor = previewObjects.find(v => v.id === o.id);
    const size = o.kind === "point" ? undefined : o.size_m ?? (descriptor ? descriptor.bounds_ned_m[1].map((v, k) => v - descriptor.bounds_ned_m[0][k]) : undefined);
    const rx = size ? Math.max(4, Math.abs(px(x + (side ? size[0] : size[1] ?? size[0]) / 2) - px(x))) : 0;
    const ry = size ? Math.max(3, Math.abs(py(y + (side ? size[2] ?? 1 : size[0]) / 2) - py(y))) : 0;
    return <g key={o.id} data-object-id={o.id}><title>{o.id}: ({o.position_ned_m.map(v => format(v)).join(", ")}) m</title>
      {size && <rect x={px(x) - rx} y={py(y) - ry} width={rx * 2} height={ry * 2} fill={`${color}22`} stroke={color} />}
      <circle cx={px(x)} cy={py(y)} r={4.5} fill={color} /><text x={px(x) + 7} y={py(y) - 7 - (i % 2) * 3} fill="#30465d" fontSize="10">{objectLabel(o.id)}</text>
    </g>;
  };
  return <svg viewBox={`0 0 ${width} ${height}`} className="wb-scene-svg" role="img" aria-label={realLabels ? "公开目标标注俯视图" : side ? "输入场景侧视图" : "输入场景俯视图"} data-testid="scene-preview">
    <rect x={left} y={top} width={pw} height={ph} fill="#f5f8fb" stroke="#dce5ec" />
    {terrain && !side && terrain.elevation_up_m.slice(0, -1).flatMap((row, j) => row.slice(0, -1).map((z, i) => <rect key={`${i}-${j}`} x={px(terrain.east_m[j])} y={py(terrain.north_m[i + 1])} width={Math.max(1, px(terrain.east_m[j + 1]) - px(terrain.east_m[j]))} height={Math.max(1, py(terrain.north_m[i]) - py(terrain.north_m[i + 1]))} fill={`hsl(${164 - z / peak * 110} 29% ${88 - z / peak * 30}%)`} />))}
    {Array.from({ length: 5 }, (_, i) => { const t = i / 4; return <g key={i}><line x1={left + pw * t} x2={left + pw * t} y1={top} y2={top + ph} stroke="#cfd9e0" strokeDasharray="3 5" opacity=".5" /><line x1={left} x2={left + pw} y1={top + ph * t} y2={top + ph * t} stroke="#cfd9e0" strokeDasharray="3 5" opacity=".5" /><text x={left + pw * t} y={height - 23} textAnchor="middle" fontSize="10" fill="#65778a">{format(bounds[0] + (bounds[1] - bounds[0]) * t, 0)}</text><text x={left - 8} y={top + ph * t + 3} textAnchor="end" fontSize="10" fill="#65778a">{format(bounds[3] - (bounds[3] - bounds[2]) * t, 0)}</text></g>; })}
    {side && <line x1={left} x2={left + pw} y1={py(0)} y2={py(0)} stroke="#97ada4" />}
    {side && geometry?.positions_ned_m.map((p, i) => <circle key={i} cx={px(p[0])} cy={py(-p[2])} r="1.2" fill="#609482" opacity=".5" />)}
    {objects.map(point)}
    {realLabels?.map((label, i) => point({ id: label.uid, kind: "box", position_ned_m: [label.y_m, label.x_m, 0], size_m: [label.length_m, label.width_m, 1] }, i))}
    {(side || realLabels) && (() => { const [x, y] = transform(platform); return <g><path d={`M${px(x)} ${py(y) - 7} l-6 12 h12 Z`} fill="#1d648c" /><text x={px(x) + 8} y={py(y) + 3} fill="#1d648c" fontSize="10">雷达</text></g>; })()}
    <text x={left + pw / 2} y={height - 5} textAnchor="middle" fill="#65778a" fontSize="11">{side ? "North / m" : "横向 East / m"}</text>
    <text transform={`translate(14,${top + ph / 2}) rotate(-90)`} textAnchor="middle" fill="#65778a" fontSize="11">{side ? "高程 / m" : "前向 North / m"}</text>
    {!positions.length && !terrain && <text x="300" y={height / 2} textAnchor="middle" fill="#8090a0" fontSize="13">暂无目标输入</text>}
  </svg>;
}
interface SceneObjectLike { id: string; kind: string; position_ned_m: Vec3; size_m?: number[] }
