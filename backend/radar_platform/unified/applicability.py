"""Configuration screening, not a physical-fidelity certificate.

All compiled scatterers are checked at start/mid/end of every requested CPI.
This is a phase-center screen; finite-aperture and intermediate-time errors
are explicitly left to the motion/visibility diagnostics, not called passed.
"""
import numpy as np
from ..models import C0
from .geometry import build_layout
from .trajectory import TrajectoryEvaluator
from .radiation import transmit_response


def assess_scene(request, scene):
    w=request.waveform
    layout=build_layout(request.array,w,request.platform)
    start=request.sequence.start_time_s
    duration=(w.loops-1)*layout.loop_period_s+max(layout.tx_offsets_s)+w.adc_start_s+w.samples/w.sample_rate_hz
    times=np.array([start+i*request.sequence.frame_interval_s+u*duration
                    for i in range(request.sequence.frame_count) for u in (0.,.5,1.)])
    trajectory=TrajectoryEvaluator(request.platform.trajectory) if request.platform.trajectory else None
    n=len(scene.positions_ned_m)
    ranges=[];closing=[];beams=[]
    for t in times:
        if trajectory:
            position,velocity,basis,_=trajectory.at(t)
            forward=basis[:,0]
        else:
            position=np.array(request.platform.position_ned_m)+np.array(request.platform.velocity_ned_mps)*t
            velocity=np.array(request.platform.velocity_ned_mps);forward=layout.basis_ned[:,0]
        delta=scene.positions_ned_m+scene.velocities_ned_mps*t-position
        r=np.linalg.norm(delta,axis=1)
        if np.any(r<=1e-6):raise ValueError('scatterer intersects radar at an applicability sample')
        unit=delta/r[:,None]
        ranges.append(r);closing.append(np.sum((velocity-scene.velocities_ned_mps)*unit,axis=1))
        illuminated=(unit@forward)>=np.cos(np.deg2rad(request.platform.beam_half_angle_deg))
        if request.radiation is not None:
            basis=basis if trajectory else layout.basis_ned
            support=np.zeros(len(delta),bool)
            for begin in range(0,len(delta),256):
                gain=transmit_response(delta[begin:begin+256,None,:],basis,np.array([t]),request.radiation,w)
                support[begin:begin+256]=abs(gain[:,0])>0
            illuminated=support if request.radiation.replace_legacy_cone else illuminated&support
        beams.append(illuminated)
    r=np.array(ranges);v=np.array(closing);beam=np.array(beams)
    delay=2*r/C0;fd=2*v/w.wavelength_m
    slow_prf=1/layout.loop_period_s
    range_cell=C0/(2*w.sample_rate_hz) if w.kind=='pulsed_lfm' else C0*w.sample_rate_hz/(2*w.slope_hz_s*w.samples)
    gate_start=w.adc_start_s;gate_end=gate_start+w.samples/w.sample_rate_hz
    checks=[]
    def check(code,status,message,**metrics):
        checks.append(dict(code=code,status=status,message=message,metrics=metrics))
    check('schema_and_timing','usable','配置及收发时序通过结构校验。',
          tx_slot_s=w.slot_period_s,same_tx_period_s=layout.loop_period_s,adc_start_s=gate_start,adc_end_s=gate_end)
    if request.radiation is not None:
        ap=request.radiation.transmit_aperture
        changes=0
        if ap is not None:
            for i in range(request.sequence.frame_count):
                t0=start+i*request.sequence.frame_interval_s
                changes+=sum(t0<s.time_s<t0+duration for s in ap.scan_schedule)
        check('transmit_scan','risk' if changes else 'usable',
              '发射权重在 Tx 时隙更新；CPI 内切换波束会调制回波，不能再按固定波束解释整个 RD。',
              schedule_changes_inside_cpis=int(changes),legacy_cone_replaced=request.radiation.replace_legacy_cone)
    if not n:
        check('empty_scene','usable','显式空场景：无目标回波；可用于噪声基准。',scatterers=0)
    else:
        check('beam_coverage','risk' if not beam.any() else 'usable',
              '采样时刻没有散射点进入主波束。' if not beam.any() else '已检查各采样时刻的主波束照射；不代表无遮挡。',
              illuminated_samples=int(beam.sum()),total_samples=int(beam.size),
              ever_illuminated_scatterers=int(beam.any(axis=0).sum()),scatterers=n)
        if w.kind=='pulsed_lfm':
            partial=(delay<gate_start)|(delay+w.pulse_width_s>gate_end)
            outside=(delay>=gate_end)|(delay+w.pulse_width_s<=gate_start)
            too_far=delay>=w.slot_period_s
            check('pulse_receive_gate','risk' if np.any(partial&beam) else 'usable',
                  '检查整个脉冲是否进入接收窗；截断会改变距离响应。',
                  illuminated_partial_or_missing=int((partial&beam).sum()),
                  illuminated_fully_missing=int((outside&beam).sum()),range_min_m=float(r.min()),range_max_m=float(r.max()))
            check('range_ambiguity','risk' if np.any(too_far&beam) else 'usable',
                  '当前模型不合成跨 PRI 的前次发射回波；超界不应解释为正确的距离折叠仿真。',
                  max_unambiguous_range_m=C0*w.slot_period_s/2,
                  illuminated_beyond_pri=int((too_far&beam).sum()))
        else:
            partial=delay>gate_start
            beat=w.slope_hz_s*delay-fd
            out=(beat<0)|(beat>=w.sample_rate_hz)
            check('fmcw_receive_gate','risk' if np.any(partial&beam) else 'usable',
                  'ADC 起始早于本 chirp 回波到达时，部分样本会被置零；未合成前一 chirp 混频。',
                  illuminated_startup_truncation=int((partial&beam).sum()))
            check('fmcw_beat_band','risk' if np.any(out&beam) else 'usable',
                  '检查正拍频复数采样区间 [0, fs)；不是实数 ADC 的半带宽约定。',
                  beat_min_hz=float(beat.min()),beat_max_hz=float(beat.max()),
                  illuminated_out_of_band=int((out&beam).sum()))
        ambiguous=(fd>=slow_prf/2)|(fd< -slow_prf/2)
        check('doppler_ambiguity','risk' if np.any(ambiguous&beam) else 'usable',
              'Doppler 按同一 Tx 的采样周期判断；本次不执行解模糊。',
              slow_prf_hz=slow_prf,max_unambiguous_closing_speed_mps=w.wavelength_m*slow_prf/4,
              closing_speed_min_mps=float(v.min()),closing_speed_max_mps=float(v.max()),
              illuminated_aliased_samples=int((ambiguous&beam).sum()))
        movement=np.ptp(r.reshape(request.sequence.frame_count,3,n),axis=1)
        max_m=float(movement.max());bins=max_m/range_cell
        check('range_walk','risk' if bins>1 else 'usable',
              '距离走动已进入回波；超过一个原生距离单元时，固定距离 FFT 可能展宽，尚未补偿。',
              sampled_max_cpi_range_change_m=max_m,native_range_cell_m=range_cell,
              sampled_max_cpi_range_change_bins=bins)
        inside=(r>=request.processing.range_limits_m[0])&(r<=request.processing.range_limits_m[1])
        check('display_range','risk' if np.any(beam&~inside) else 'usable',
              '按几何距离检查输出区间；LFM/FMCW 距离–Doppler 耦合可能进一步移动峰值。',
              illuminated_outside_output=int((beam&~inside).sum()))
    status='risk' if any(x['status']=='risk' for x in checks) else 'usable'
    return dict(status=status,checks=checks,
        derived=dict(wavelength_m=w.wavelength_m,range_resolution_m=C0/(2*w.bandwidth_hz),
                     native_range_cell_m=range_cell,velocity_cell_mps=w.wavelength_m/(2*w.loops*layout.loop_period_s),
                     maximum_unambiguous_speed_mps=w.wavelength_m/(4*layout.loop_period_s)),
        coverage=dict(time_samples=len(times),scatterers=n,all_scatterers_checked=True,
                      all_chirps_checked=False,sampling='start/mid/end of every CPI; phase center'),
        limitations=['This is suitability for the declared ideal model, not measured-scene fidelity.',
                     'Intermediate trajectory extrema may be missed by time sampling.',
                     'Finite Tx/Rx aperture, within-slot approximation and environmental occlusion require separate checks.'])


def inspect_configuration(raw):
    """Return unsupported diagnostics without launching a simulation."""
    from .schema import Simulation
    from .preflight import preflight
    try:
        request=Simulation.model_validate(raw)
        _,estimate=preflight(request)
        return {**estimate['applicability'],'resources':{k:v for k,v in estimate.items() if k!='applicability'}}
    except (ValueError,TypeError) as exc:
        return {'status':'unsupported','checks':[{'code':'configuration_or_budget','status':'unsupported','message':str(exc)}],
                'limitations':['No simulation was launched.']}
