import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { CheckReport, SceneJob, SceneRequest } from "../workbenchTypes";
import { format } from "./WorkbenchCharts";

export const presentationChapters = [
  {title:"场景与配置",subtitle:"交代演示对象与输入来源",talk:"介绍场景构成、配置入口和三维视图。说明草稿可编辑，归档结果对应运行时的配置；两者有差异时需要分别说明。",items:["场景对象树与参数编辑","三维场景、图层和视角控制","配置导入、导出与草稿保存"],boundary:"三维符号及波束为显示示意；视觉效果不等同于实测性能。"},
  {title:"运行与状态",subtitle:"展示可追溯的处理流程",talk:"说明参数检查、任务状态、帧进度和错误反馈如何衔接。已有结果直接复用；只有明确点击运行才创建新任务。",items:["参数检查与风险反馈","任务进度及停止操作","任务编号与逐帧结果记录"],boundary:"检查通过表示符合当前模型约束，不代表完成实测验证或精度认证。"},
  {title:"联动回放",subtitle:"展示场景与结果的同步关系",talk:"播放选定片段，对照同一帧的场景和结果图。可暂停、逐帧查看，使用原有图表标签切换结果；手机上可切换场景与结果。",items:["场景与结果就绪后同步推进","片段选择、倍速与循环控制","视角保持、图层及图表切换"],boundary:"播放倍速只影响展示节奏；结果仍来自已计算帧，不生成额外物理数据。"},
  {title:"成果与交付",subtitle:"汇总可展示、可保存的产物",talk:"按配置、结果、图片和视频组织结题材料。展示已有产物与复现入口，并交代尚未完成的验证，不将功能演示写成性能结论。",items:["实验归档与结果对比入口","图像、复现包及演示摘要","MP4 视频与封面导出"],boundary:"对比需要另选一份结果；视频和封面需实际生成后才能作为交付材料。"},
];

export function PresentationGuide({step,onStep,request,job,check,dirty,actions,onSummary}:{step:number;onStep:(n:number)=>void;request:SceneRequest;job:SceneJob|null;check:CheckReport|null;dirty:boolean;actions:ReactNode;onSummary:()=>void}){
  const chapter=presentationChapters[step];
  const status=job?({queued:"排队中",running:"计算中",complete:"已完成",cancelled:"已停止",failed:"失败"})[job.status]:"尚未运行";
  const frames=job?.frames??[];
  const metrics=step===0?[["讲解配置",dirty?"当前草稿（与结果不同）":"当前草稿"],["场景对象",`${request.scene.objects.length} 个`],["计划帧数",`${request.sequence.frame_count} 帧`]]:step===1?[["任务状态",status],["生成进度",`${frames.length} / ${job?.frame_count??request.sequence.frame_count} 帧`],["当前草稿检查",check?({usable:"可用",risk:"存在风险",unsupported:"不支持"})[check.status]:"尚未检查"]]:step===2?[["可回放帧",`${frames.length} 帧`],["首末时间跨度",frames.length?`${format(frames[frames.length-1].frame_start_s-frames[0].frame_start_s,3)} s`:"—"],["数据来源",job?.job_id.slice(0,8)??"尚无结果"]]:[["结果任务",job?.job_id.slice(0,8)??"尚无结果"],["任务状态",status],["材料类型","配置 / 结果 / 图像 / 视频"]];
  return <section className="lab-guide" aria-label="结题演示导航">
    <div className="lab-guide-heading"><div><span className="lab-eyebrow">PROJECT REVIEW · 结题演示</span><h2>从场景配置到成果交付</h2></div><span className="lab-guide-count">手动讲解 · {step+1} / 4</span></div>
    <nav className="lab-guide-chapters" aria-label="演示章节">{presentationChapters.map((c,i)=><button key={c.title} aria-current={step===i?"step":undefined} onClick={()=>onStep(i)}><span>0{i+1}</span><strong>{c.title}</strong><small>{c.subtitle}</small></button>)}</nav>
    <div className="lab-guide-body"><div><h3>{chapter.title}</h3><p>{chapter.talk}</p><div className="lab-guide-actions">{actions}</div>{dirty&&<p className="lab-guide-warning">草稿已修改。回放与归档仍使用已有任务结果，不能作为当前草稿的新结果。</p>}</div><div className="lab-guide-metrics">{metrics.map(([label,value])=><div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div></div>
    <details className="lab-guide-notes"><summary>讲解要点与成果边界</summary><ul>{chapter.items.map(item=><li key={item}>{item}</li>)}</ul><p>{chapter.boundary}</p>{step===3&&<p>建议材料顺序：功能概览 → 配置与任务编号 → 结果截图 → 回放视频 → 局限与后续工作。</p>}</details>
    <footer><button className="wb-button secondary" disabled={step===0} onClick={()=>onStep(step-1)}>上一步</button><button className="wb-button primary" disabled={step===3} onClick={()=>onStep(step+1)}>下一步</button><button className="wb-button secondary" onClick={onSummary}>导出演示摘要</button><Link to="/cases">切换案例</Link><span>{step===3?"讲解已到最后一章；请按需保存交付材料。":"切换章节只调整展示，不会自动运行或保存。"}</span></footer>
  </section>;
}
