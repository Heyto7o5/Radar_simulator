import { useEffect, useRef, useState } from 'react';
import { API_BASE } from '../api';
import { requestJSON } from '../workbenchApi';
import type { SceneJob } from '../workbenchTypes';

const sleep = (ms:number) => new Promise(resolve=>setTimeout(resolve,ms));
function save(url:string,name:string){const a=document.createElement('a');a.href=url;a.download=name;a.click();}
function contain(ctx:CanvasRenderingContext2D,source:HTMLCanvasElement,x:number,y:number,w:number,h:number){const k=Math.min(w/source.width,h/source.height);ctx.drawImage(source,x+(w-source.width*k)/2,y+(h-source.height*k)/2,source.width*k,source.height*k);}

export function VideoExport({job,start,end,frame,setFrame,stop,reveal=0}:{job:SceneJob;start:number;end:number;frame:number;setFrame:(n:number)=>void;stop:()=>void;reveal?:number}){
  const [open,setOpen]=useState(false),[title,setTitle]=useState(job.name),[description,setDescription]=useState(''),[resolution,setResolution]=useState(1080),[seconds,setSeconds]=useState(1),[progress,setProgress]=useState(''),[busy,setBusy]=useState(false),[encoding,setEncoding]=useState(false),[error,setError]=useState(''),[output,setOutput]=useState<{video:string;cover:string}|null>(null);
  useEffect(()=>{if(reveal)setOpen(true);},[reveal]);
  const abort=useRef<AbortController|null>(null);
  useEffect(()=>()=>abort.current?.abort(),[]);
  async function generate(){
    stop();const original=frame;const ctrl=new AbortController();abort.current=ctrl;setBusy(true);setError('');setOutput(null);let key='';
    const width=resolution===1080?1920:1280,height=resolution,count=end-start+1;
    try{
      const created=await requestJSON<{id:string}>('/api/v1/presentations',{method:'POST',body:JSON.stringify({title,width,height,frames:count,seconds_per_frame:seconds}),signal:ctrl.signal});key=created.id;
      const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d')!;const k=width/1920;
      for(let i=start;i<=end;i++){
        if(ctrl.signal.aborted)throw new DOMException('已取消','AbortError');
        setProgress(`生成画面 ${i-start+1} / ${count}`);setFrame(i);
        const deadline=Date.now()+30000;
        while(true){
          if(ctrl.signal.aborted)throw new DOMException('已取消','AbortError');
          const view=document.querySelector<HTMLElement>('.lab-view-pair');
          if(view?.dataset.frame===String(i)&&view.dataset.ready==='true')break;
          if(Date.now()>deadline)throw new Error('当前帧加载超时，请检查结果后重试');
          await sleep(60);
        }
        await new Promise<void>(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve())));
        const scene=document.querySelector<HTMLCanvasElement>('.lab-three-host canvas');const layers=Array.from(document.querySelectorAll<HTMLCanvasElement>('.lab-result-panel .lab-scientific-map canvas'));const chart=document.createElement('canvas');chart.width=layers[0]?.width??0;chart.height=layers[0]?.height??0;const composite=chart.getContext('2d');if(composite)for(const layer of layers)composite.drawImage(layer,0,0,chart.width,chart.height);
        if(!scene||!layers.length)throw new Error('请先打开结果场景和结果图，再生成视频');
        ctx.fillStyle='#f3f6f9';ctx.fillRect(0,0,width,height);ctx.save();ctx.scale(k,k);
        ctx.fillStyle='#2b7d80';ctx.font='14px sans-serif';ctx.fillText('RADAR SIMULATION LAB  /  仿真回放',48,45);
        ctx.fillStyle='#22364b';ctx.font='bold 30px sans-serif';ctx.fillText(title,48,94,1824);
        ctx.fillStyle='#657b8f';ctx.font='18px sans-serif';ctx.fillText(description||'三维场景与已计算结果同步展示',48,132,1824);
        ctx.fillStyle='#0e1c2c';ctx.fillRect(48,172,900,720);ctx.fillStyle='#fff';ctx.fillRect(972,172,900,720);
        contain(ctx,scene,48,172,900,720);contain(ctx,chart,972,172,900,720);
        ctx.fillStyle='#22364b';ctx.font='20px sans-serif';ctx.fillText(`帧 ${i+1} / ${job.frames.length}  ·  仿真时间 ${job.frames[i].frame_start_s.toFixed(5)} s`,48,944);
        ctx.fillStyle='#dfe6ee';ctx.fillRect(48,976,1824,5);ctx.fillStyle='#2b7d80';ctx.fillRect(48,976,1824*(i-start+1)/count,5);
        ctx.fillStyle='#738295';ctx.font='16px sans-serif';ctx.fillText(`已计算帧回放 · 每帧展示 ${seconds} 秒 · ${job.job_id.slice(0,8)}`,48,1024);ctx.restore();
        const png=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('无法生成画面')),'image/png'));
        const res=await fetch(`${API_BASE}/api/v1/presentations/${key}/frames/${i-start}`,{method:'PUT',body:png,signal:ctrl.signal});if(!res.ok)throw new Error('上传画面失败');
      }
      setEncoding(true);setProgress('正在编码 MP4…');
      const result=await requestJSON<{video:string;cover:string}>(`/api/v1/presentations/${key}/finish`,{method:'POST',signal:ctrl.signal});setOutput(result);setProgress('视频与封面已生成');
    }catch(e){if(e instanceof DOMException&&e.name==='AbortError'){setProgress('已取消');}else setError(String(e));if(key)void fetch(`${API_BASE}/api/v1/presentations/${key}`,{method:'DELETE'});}
    finally{setBusy(false);setEncoding(false);setFrame(original);abort.current=null;}
  }
  return <section className="lab-video-export wb-panel"><button className="lab-video-heading" onClick={()=>setOpen(!open)} aria-expanded={open}>视频与封面 <span>{open?'收起':'导出 MP4'}</span></button>{open&&<div className="lab-video-body"><div className="lab-video-fields"><label>视频标题<input value={title} maxLength={160} disabled={busy} onChange={e=>setTitle(e.target.value)}/></label><label>简短说明<input value={description} maxLength={160} disabled={busy} onChange={e=>setDescription(e.target.value)}/></label><label>分辨率<select value={resolution} disabled={busy} onChange={e=>setResolution(Number(e.target.value))}><option value={1080}>1080p · 16:9</option><option value={720}>720p · 16:9</option></select></label><label>每帧展示<select value={seconds} disabled={busy} onChange={e=>setSeconds(Number(e.target.value))}>{[.1,.25,.5,1,2,3].map(v=><option key={v} value={v}>{v} 秒</option>)}</select></label></div><p>片段 {start+1}–{end+1} 帧 · 预计 {((end-start+1)*seconds).toFixed(1)} 秒 · 沿用当前视角、图表与色阶</p><div className="lab-video-actions"><button className="wb-button primary" disabled={busy||!title.trim()} onClick={()=>void generate()}>生成视频</button>{busy&&!encoding&&<button className="wb-button secondary" onClick={()=>abort.current?.abort()}>取消</button>}<span role="status">{progress}</span>{output&&<><button className="wb-button secondary" onClick={()=>save(API_BASE+output.video,'simulation.mp4')}>下载 MP4</button><button className="wb-button secondary" onClick={()=>save(API_BASE+output.cover,'cover.png')}>下载封面</button></>}</div>{error&&<p role="alert">{error}</p>}{output&&<video controls preload="metadata" src={API_BASE+output.video} poster={API_BASE+output.cover}/>}</div>}{busy&&<div className="lab-export-shield"><strong>{progress}</strong><span>正在逐帧保存展示画面，请保持此页面打开。</span>{!encoding&&<button className="wb-button secondary" onClick={()=>abort.current?.abort()}>取消导出</button>}</div>}</section>;
}
