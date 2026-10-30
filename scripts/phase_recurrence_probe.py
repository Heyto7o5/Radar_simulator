#!/usr/bin/env python3
"""Isolated exact-polynomial phase recurrence probe; NOT used by frozen jobs."""
import json,time
from pathlib import Path
import numpy as np


def quadratic_phasor(a,b,c,start,step,count):
    """exp(i*(a+b*t+c*t²)), shape [scatterer,sample,chirp], complex128."""
    a,b,c=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float),np.asarray(c,float))
    if a.ndim!=2 or not all(np.isfinite(x).all() for x in [a,b,c]) or count<1 or int(count)!=count or not np.isfinite([start,step]).all() or step<=0:
        raise ValueError('finite 2D polynomial coefficients and positive sampling required')
    first=np.exp(1j*(a+b*start+c*start*start))
    out=np.empty((a.shape[0],count,a.shape[1]),np.complex128);out[:,0,:]=first
    if count==1:return out
    delta=np.exp(1j*(b*step+c*(2*start*step+step*step)))
    acceleration=np.exp(2j*c*step*step)
    factors=np.empty((a.shape[0],count-1,a.shape[1]),np.complex128)
    factors[:,0,:]=1
    factors[:,1:,:]=acceleration[:,None,:]
    np.cumprod(factors,axis=1,out=factors)
    factors*=delta[:,None,:]
    np.cumprod(factors,axis=1,out=factors)
    out[:,1:,:]=first[:,None,:]*factors
    return out


def coefficients(kind,delay,delay_rate,carrier,slope,pulse_width=0,reflection_phase=0):
    if kind=='fmcw':
        return (2*np.pi*(carrier*delay-.5*slope*delay**2)-reflection_phase,
                2*np.pi*(carrier*delay_rate+slope*delay-slope*delay*delay_rate),
                2*np.pi*slope*(delay_rate-.5*delay_rate**2))
    if kind=='pulsed_lfm':
        offset=delay+pulse_width/2
        return (np.pi*slope*offset**2-2*np.pi*carrier*delay+reflection_phase,
                -2*np.pi*slope*(1-delay_rate)*offset-2*np.pi*carrier*delay_rate,
                np.pi*slope*(1-delay_rate)**2)
    raise ValueError('unknown waveform')


def direct(kind,delay,rate,carrier,slope,start,step,count,pulse_width=0,reflection_phase=0):
    reflection_phase=np.broadcast_to(np.asarray(reflection_phase,float),delay.shape)
    if kind not in ['fmcw','pulsed_lfm']:raise ValueError('unknown waveform')
    t=start+np.arange(count)*step
    tau=delay[:,None,:]+rate[:,None,:]*t[None,:,None]
    if kind=='fmcw':phase=2*np.pi*(carrier*tau+slope*t[None,:,None]*tau-.5*slope*tau**2)-np.asarray(reflection_phase)[:,None,:]
    else:phase=np.pi*slope*(t[None,:,None]-tau-pulse_width/2)**2-2*np.pi*carrier*tau+np.asarray(reflection_phase)[:,None,:]
    return np.exp(1j*phase)


def main():
    rng=np.random.default_rng(42);rows=[]
    cases=[('fmcw',77e9,63.343e12,6e-6,1/9.121e6,512,0.,(.3,5.4),1.),
           ('fmcw',9.98e9,2e11,45e-6,1/20e6,2048,0.,(4000,6000),1200.),
           ('pulsed_lfm',10e9,4e12,28e-6,1/40e6,1024,5e-6,(4000,6000),1200.)]
    for kind,fc,slope,start,step,count,width,range_bounds,speed in cases:
        chunk=max(1,min(16,1_000_000//(count*256)))
        delay=2*rng.uniform(*range_bounds,(chunk,256))/299792458
        rate=2*rng.uniform(-speed,speed,(chunk,256))/299792458
        phase=rng.uniform(-np.pi,np.pi,(chunk,256));abc=coefficients(kind,delay,rate,fc,slope,width,phase)
        reference=direct(kind,delay,rate,fc,slope,start,step,count,width,phase)
        actual=quadratic_phasor(*abc,start,step,count)
        timers={}
        for name,function in [('direct',lambda:direct(kind,delay,rate,fc,slope,start,step,count,width,phase)),
                              ('recurrence',lambda:quadratic_phasor(*coefficients(kind,delay,rate,fc,slope,width,phase),start,step,count))]:
            measurements=[]
            for _ in range(5):
                t=time.perf_counter();function();measurements.append(time.perf_counter()-t)
            timers[name]=float(np.median(measurements))
        rows.append({'kind':kind,'carrier_hz':fc,'samples':count,'array_shape':list(actual.shape),
                     'max_complex_error':float(abs(actual-reference).max()),
                     'relative_complex_l2_error':float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
                     'max_unit_magnitude_error':float(abs(abs(actual)-1).max()),
                     'median_seconds':timers,'phase_kernel_speed_ratio':timers['direct']/timers['recurrence']})
    out=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/phase_kernel_probe'
    out.mkdir(parents=True,exist_ok=True)
    report={'production_integrated':False,'frozen_world_batch_unchanged':True,
            'meaning':'same quadratic fast-time phase under existing within-slot linear path assumption; no extra physical approximation',
            'not_an_end_to_end_speedup_measurement':True,'timing_repeats':5,'seed':42,'cases':rows}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    table=''.join(f'<tr><td>{r["kind"]}</td><td>{r["carrier_hz"]/1e9:.2f} GHz</td><td>{r["samples"]}</td><td>{r["max_complex_error"]:.2e}</td><td>{r["phase_kernel_speed_ratio"]:.2f}×</td></tr>' for r in rows)
    (out/'相位递推独立试验.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1000px;margin:30px auto;font:16px/1.7 sans-serif}td,th{border:1px solid #bbb;padding:10px}table{border-collapse:collapse}</style>
<h1>同一相位公式的独立计算试验</h1><p>在线性化时隙路径假设下，快时间相位是二次多项式。用相邻采样点之间的复数递推，替代逐点计算指数；没有增加新的物理近似。</p>
<table><tr><th>波形</th><th>载频</th><th>快时间样本</th><th>最大复数相位因子误差</th><th>相位计算耗时比</th></tr>TABLE</table>
<p>每种方法重复五次取中位数，分块尺寸沿用现有引擎规则。测量时服务器同时运行冻结的几何批次，因此绝对耗时只代表当时环境。</p><p><strong>尚未接入生产链路，也没有重启或改变正在运行的任务。</strong>这里不是完整仿真的加速比：可见性、延时门控、几何和 FFT 等成本未覆盖；数值一致也不证明真实场景保真。</p><p><a href="summary.json">参数、误差与原始耗时</a></p></html>'''.replace('TABLE',table))

if __name__=='__main__':main()
