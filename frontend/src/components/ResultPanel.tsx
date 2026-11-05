import { memo, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { connect, type EChartsOption } from "echarts";
import { getSceneView, isAbort, downloadJSON, getDisplayData } from "../workbenchApi";
import type { Matrix, SceneFrame, SceneJob } from "../workbenchTypes";
import { format } from "./WorkbenchCharts";

export function ScientificMap({ matrix, x, y, title, xLabel, yLabel, floor=-80, ceiling=6, onPick, markers=[], unit="dB", height=335, syncGroup }: { matrix:Matrix|null; x:number[]; y:number[]; title:string; xLabel:string; yLabel:string; floor?:number; ceiling?:number; onPick?:(x:number,y:number)=>void; markers?:{x:number;y:number;name:string;kind:string}[];unit?:string;height?:number;syncGroup?:string }) {
  const chart=useRef<ReactECharts>(null);
  const dx=Math.max(1,Math.ceil((matrix?.cols??0)/700)),dy=Math.max(1,Math.ceil((matrix?.rows??0)/400));
  useEffect(()=>{const instance=chart.current?.getEchartsInstance();if(instance){instance.group=syncGroup??"";if(syncGroup)connect(syncGroup);}},[syncGroup,matrix]);
  const option=useMemo<EChartsOption>(()=>{
    const cols=matrix?.cols??0,rows=matrix?.rows??0,data:number[][]=[];
    // Presentation-only maximum pooling. Original matrix is retained for numeric export and slices.
    if(matrix)for(let j=0;j<rows;j+=dy)for(let i=0;i<cols;i+=dx){let value=-Infinity;for(let a=j;a<Math.min(rows,j+dy);a++)for(let b=i;b<Math.min(cols,i+dx);b++)value=Math.max(value,matrix.data[a*cols+b]);data.push([i/dx,j/dy,Math.max(floor,Math.min(ceiling,value)),value]);}
    return {animation:false,backgroundColor:"#fff",title:{text:title,left:18,top:8,textStyle:{fontSize:12,fontWeight:500,color:"#536578"}},
      tooltip:{position:"top",formatter:(p:unknown)=>{const a=p as {value:number[]};return `${xLabel}: ${format(x[a.value[0]*dx])}<br/>${yLabel}: ${format(y[a.value[1]*dy])}<br/>${format(a.value[3]??a.value[2],4)} ${unit}`;}},
      grid:{left:68,right:80,top:50,bottom:80},xAxis:{type:"category",data:x.filter((_,i)=>i%dx===0).map(v=>format(v,2)),name:xLabel,nameLocation:"middle",nameGap:32,axisLabel:{interval:"auto",color:"#667788"}},yAxis:{type:"category",data:y.filter((_,i)=>i%dy===0).map(v=>format(v,2)),name:yLabel,nameLocation:"middle",nameGap:47,axisLabel:{interval:"auto",color:"#667788"}},
      dataZoom:[{type:"inside",xAxisIndex:0},{type:"inside",yAxisIndex:0},{type:"slider",xAxisIndex:0,height:12,bottom:12}],
      visualMap:{dimension:2,min:floor,max:ceiling,orient:"vertical",right:8,top:65,itemHeight:190,text:[unit,""],calculable:true,inRange:{color:unit==="dB"?["#0c1931","#1c437e","#1d84a0","#42bba7","#d3d771","#f0a33a","#c84128"]:["#2a6498","#f4f6f8","#bd553e"]},seriesIndex:0},
      series:[{type:"heatmap",data,encode:{x:0,y:1,value:2},progressive:0,emphasis:{itemStyle:{borderColor:"#fff",borderWidth:1}}},...markers.map(m=>({type:"scatter" as const,symbol:m.kind==="truth"?"diamond":"circle",symbolSize:12,data:[[Math.floor(m.x/dx),Math.floor(m.y/dy)]],itemStyle:{color:"transparent",borderWidth:2,borderColor:m.kind==="truth"?"#fff":"#f9a050"},label:{show:true,formatter:m.name,position:"top" as const,color:"#172839",backgroundColor:"#ffffffdd",padding:2}}))]};
  },[matrix,x,y,title,xLabel,yLabel,floor,ceiling,markers,unit]);
  return <div data-testid="heatmap" data-loaded={Boolean(matrix)} data-reference={matrix?.reference} className="lab-scientific-map">{matrix?<><ReactECharts ref={chart} option={option} notMerge style={{height}} onEvents={{click:(p:{value:number[];seriesType:string})=>{if(p.seriesType==="heatmap")onPick?.(p.value[0]*dx,p.value[1]*dy);}}}/><div className="lab-chart-actions"><span>{dx>1||dy>1?`显示块取峰值 ${dx}×${dy}；原始数据保留`:""}</span><button onClick={()=>chart.current?.getEchartsInstance().dispatchAction({type:"dataZoom",start:0,end:100})}>恢复视图</button><button onClick={()=>{const a=document.createElement("a");a.href=chart.current!.getEchartsInstance().getDataURL({pixelRatio:2,backgroundColor:"#fff"});a.download="radar-result.png";a.click();}}>导出图像</button></div>{!matrix.data.some(v=>v>floor+.01)&&<p className="wb-hint">当前色阶下没有可见回波；请检查显示范围与接收窗。</p>}</>:<div className="wb-empty-chart" style={{height}}>尚无此帧产品，请运行仿真或打开已保存结果。</div>}</div>;
}
const nearest=(values:number[],v:number)=>values.reduce((best,x,i)=>Math.abs(x-v)<Math.abs(values[best]-v)?i:best,0);
export function ResultPanel({job,frameIndex,onChangeView,onReady}:{job:SceneJob|null;frameIndex:number;onChangeView?:(view:string)=>void;onReady?:(frame:number)=>void}) {
  const [view,setView]=useState("rd"),[channel,setChannel]=useState("combined"),[floor,setFloor]=useState(-80),[elevation,setElevation]=useState(0);
  const [slice,setSlice]=useState<{x:number;y:number}|null>(null),[matrix,setMatrix]=useState<Matrix|null>(null),[error,setError]=useState("");
  const [profileAxis,setProfileAxis]=useState("range");
  const [truth,setTruth]=useState(false),[peaks,setPeaks]=useState(false),[profile,setProfile]=useState(false);
  const targetMeta:SceneFrame|undefined=job?.frames[frameIndex];
  const [displayMeta,setDisplayMeta]=useState<SceneFrame>();
  const meta=displayMeta??targetMeta;
  const selection=useRef("");
  const cached=useRef(new Map<string,Matrix>());
  useEffect(()=>{setSlice(null);setChannel("combined");setElevation(0);setView("rd");},[job?.job_id]);
  useEffect(()=>{
    onReady?.(-1);
    if(!job||!targetMeta){setMatrix(null);setDisplayMeta(undefined);return;}
    const ctrl=new AbortController();setError("");
    const options:Record<string,string|number>={view,channel,elevation_index:elevation};
    if(view==="angle"&&slice)Object.assign(options,{range_index:slice.x,doppler_index:slice.y});
    const config=job.job_id+JSON.stringify(options),key=config+":"+frameIndex;
    if(selection.current!==config){setMatrix(null);setDisplayMeta(undefined);selection.current=config;}
    const accept=(value:Matrix)=>{if(!ctrl.signal.aborted){setMatrix(value);setDisplayMeta(targetMeta);onReady?.(frameIndex);}};
    const hit=cached.current.get(key);
    if(hit)accept(hit);
    else getSceneView(job.job_id,frameIndex,options,ctrl.signal).then(value=>{
      if(ctrl.signal.aborted)return;
      cached.current.set(key,value);
      if(cached.current.size>8)cached.current.delete(cached.current.keys().next().value!);
      accept(value);
    }).catch(e=>{if(!isAbort(e))setError(String(e));});
    return()=>ctrl.abort();
  },[job?.job_id,Boolean(targetMeta),frameIndex,view,channel,elevation,slice?.x,slice?.y,onReady]);

  const axes=meta?.axes,x=(view==="angle"?axes?.azimuth_deg:axes?.range_m)??[],y=(view==="rd"?axes?.velocity_mps:view==="ra"?axes?.azimuth_deg:axes?.elevation_deg)??[];
  const markers: {x:number;y:number;name:string;kind:string}[]=[];
  if(view==="rd"&&axes){if(peaks)meta?.peak_candidates.forEach((p,i)=>markers.push({x:p.range_bin,y:p.doppler_bin,name:`候选 ${i+1}`,kind:"peak"}));if(truth)meta?.object_truth.forEach(o=>{const r=o.range_extent_m,v=o.closing_velocity_extent_mps;if(r&&v&&r[0]>=x[0]&&r[1]<=x[x.length-1]&&v[0]>=y[0]&&v[1]<=y[y.length-1])markers.push({x:nearest(x,(r[0]+r[1])/2),y:nearest(y,(v[0]+v[1])/2),name:o.id,kind:"truth"});});}
  const reference=matrix?`P参考=${matrix.reference.toExponential(3)}`:"";
  return <><div className="wb-panel-head"><div className="wb-tabs">{[["rd","RD"],["ra","距离–方位"],["angle","方位–俯仰"]].map(([key,label])=><button key={key} className={view===key?"active":""} disabled={Boolean(meta)&& (key==="ra"&&meta!.axes.azimuth_deg.length<2||key==="angle"&&meta!.axes.elevation_deg.length<2)} onClick={()=>{setView(key);onChangeView?.(key);}}>{label}</button>)}</div></div>
    <div className="lab-plot-toolbar">{view==="rd"&&<select aria-label="仿真显示通道" value={channel} onChange={e=>setChannel(e.target.value)}><option value="combined">通道功率平均</option>{meta?.array.channel_map.map(c=><option key={c.channel} value={`tx${c.tx}_rx${c.rx}`}>Tx{c.tx+1} / Rx{c.rx+1}</option>)}</select>}{view==="ra"&&<select aria-label="俯仰切片" value={elevation} onChange={e=>setElevation(+e.target.value)}>{axes?.elevation_deg.map((v,i)=><option key={i} value={i}>俯仰 {v}°</option>)}</select>}<select aria-label="显示动态范围" value={floor} onChange={e=>setFloor(+e.target.value)}>{[-60,-80,-100].map(v=><option key={v} value={v}>{-v} dB</option>)}</select><button disabled={!meta} onClick={()=>setProfile(v=>!v)}>剖面</button></div>
    {error&&<p role="alert" className="wb-alert error">{error}</p>}
    <ScientificMap matrix={matrix} x={x} y={y} title={`${view.toUpperCase()} · t=${format(meta?.frame_start_s??0,5)} s · ${view==="rd"?channel:"阵列波束合成"} · ${reference}`} xLabel={view==="angle"?"方位角 / °":"距离 / m"} yLabel={view==="rd"?"表观接近速度 / m/s":view==="ra"?"方位角 / °":"俯仰角 / °"} floor={floor} onPick={(a,b)=>{if(view==="rd")setSlice({x:a,y:b});setProfile(true);}} markers={markers}/>
    <div className="lab-result-info"><span>固定首帧参考 · 10 log₁₀(P/P参考)</span><span>{view==="ra"?`俯仰 ${axes?.elevation_deg[elevation]??0}° · Doppler 功率累加`:view==="angle"?slice?`距离 ${format(axes?.range_m[slice.x]??0)} m / 速度 ${format(axes?.velocity_mps[slice.y]??0)} m/s` :"当前帧最大 RD 功率单元切片":"速度正值表示接近；未解模糊"}</span></div>
    {view==="rd"&&<div className="lab-plot-toolbar"><label><input type="checkbox" checked={truth} onChange={e=>setTruth(e.target.checked)}/>配置真值</label><label><input type="checkbox" checked={peaks} onChange={e=>setPeaks(e.target.checked)}/>峰候选</label><button disabled={!job||!meta} onClick={()=>{void getDisplayData(job!.job_id,frameIndex,channel).then(d=>downloadJSON(d,"rd-linear-power.json")).catch(e=>setError(String(e)));}}>导出数据与轴</button></div>}
    {truth&&<p className="wb-hint">真值范围中心投影，时刻 t={format(meta?.truth_epoch_s??0,6)} s；超出坐标范围不绘制，不与检测候选自动匹配。</p>}
    {profile&&matrix&&<><div className="lab-plot-toolbar"><label>剖面方向 <select aria-label="剖面方向" value={profileAxis} onChange={e=>setProfileAxis(e.target.value)}><option value="range">横轴剖面 / 固定纵轴</option><option value="doppler">纵轴剖面 / 固定横轴</option></select></label></div><ReactECharts style={{height:185}} option={{animation:false,title:{text:profileAxis==="range"?`固定纵轴 ${format(y[Math.min(slice?.y??Math.floor(matrix.rows/2),matrix.rows-1)])}`:`固定横轴 ${format(x[Math.min(slice?.x??Math.floor(matrix.cols/2),matrix.cols-1)])}`,textStyle:{fontSize:12}},tooltip:{trigger:"axis"},grid:{left:55,right:25,bottom:35},xAxis:{type:"category",data:(profileAxis==="range"?x:y).map(v=>format(v))},yAxis:{type:"value",name:"dB"},series:[{type:"line",symbol:"none",data:profileAxis==="range"?Array.from(matrix.data.slice(Math.min(slice?.y??Math.floor(matrix.rows/2),matrix.rows-1)*matrix.cols,(Math.min(slice?.y??Math.floor(matrix.rows/2),matrix.rows-1)+1)*matrix.cols)):Array.from({length:matrix.rows},(_,i)=>matrix.data[i*matrix.cols+Math.min(slice?.x??Math.floor(matrix.cols/2),matrix.cols-1)])}]}}/></>}

  </>;
}

export const MemoResultPanel = memo(ResultPanel);
