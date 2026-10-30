"""Compare within-slot linear paths with instantaneous sampled geometry.

Reference is exact distance evaluated on the specified trajectory, NOT an
electromagnetic truth solution: propagation retardation and unknown motion
between trajectory observations remain excluded on both sides.
"""
import numpy as np
from ..models import C0
from .geometry import build_layout
from .scene import compile_scene
from .trajectory import TrajectoryEvaluator


def motion_error_report(request, *, max_scatterers=256, max_slots=16, fast_samples=5,
                        phase_tolerance_rad=np.pi/8):
    if not 1<=max_scatterers<=4096 or not 1<=max_slots<=1024 or not 2<=fast_samples<=128:
        raise ValueError('diagnostic sampling budget exceeded')
    if not np.isfinite(phase_tolerance_rad) or phase_tolerance_rad<=0:
        raise ValueError('positive finite phase tolerance required')
    scene=compile_scene(request.scene);layout=build_layout(request.array,request.waveform,request.platform)
    w=request.waveform;p=request.platform
    count=len(scene.amplitudes)
    indices=np.unique(np.linspace(0,count-1,min(count,max_scatterers),dtype=int)) if count else np.array([],int)
    positions=scene.positions_ned_m[indices];velocities=scene.velocities_ned_mps[indices]
    trajectory=TrajectoryEvaluator(p.trajectory) if p.trajectory else None
    def pose(t):
        if trajectory:return trajectory.at(t)
        return (np.asarray(p.position_ned_m)+np.asarray(p.velocity_ned_mps)*t,
                np.asarray(p.velocity_ned_mps),layout.basis_ned,np.zeros(3))
    slots=[(request.sequence.start_time_s+frame*request.sequence.frame_interval_s+loop*layout.loop_period_s+offset,tx)
           for frame in range(request.sequence.frame_count) for loop in range(w.loops)
           for tx,offset in enumerate(layout.tx_offsets_s)]
    chosen=np.unique(np.linspace(0,len(slots)-1,min(len(slots),max_slots),dtype=int))
    fast=w.adc_start_s+np.unique(np.linspace(0,w.samples-1,min(w.samples,fast_samples),dtype=int))/w.sample_rate_hz
    peak_path=peak_phase=peak_velocity=0.;evaluated=active_count=gate_disagreement=0
    worst=None
    for si in chosen:
        t0,tx=slots[si];pos0,v0,basis0,omega0=pose(t0)
        txoff=basis0@layout.tx_local_m[tx];tx0=pos0+txoff;txv=v0+np.cross(omega0,txoff)
        targets0=positions+velocities*t0
        dt=targets0-tx0;rt=np.linalg.norm(dt,axis=1)
        if np.any(rt<=1e-6):raise ValueError('scatterer intersects Tx in motion diagnostic')
        for rx,local in enumerate(layout.rx_local_m):
            rxoff=basis0@local;rx0=pos0+rxoff;rxv=v0+np.cross(omega0,rxoff)
            dr=targets0-rx0;rr=np.linalg.norm(dr,axis=1)
            if np.any(rr<=1e-6):raise ValueError('scatterer intersects Rx in motion diagnostic')
            rate0=np.sum(dt*(velocities-txv),axis=1)/rt+np.sum(dr*(velocities-rxv),axis=1)/rr
            for f in fast:
                pos,v,basis,omega=pose(t0+f)
                target=positions+velocities*(t0+f)
                txoff=basis@layout.tx_local_m[tx];rxoff=basis@local
                a=target-pos-txoff;b=target-pos-rxoff
                ra=np.linalg.norm(a,axis=1);rb=np.linalg.norm(b,axis=1)
                if np.any(ra<=1e-6)|np.any(rb<=1e-6):raise ValueError('scatterer intersects antenna in reference geometry')
                exact=ra+rb;linear=rt+rr+rate0*f
                if np.any(linear<=0):raise ValueError('linearized path crosses antenna')
                rate=np.sum(a*(velocities-v-np.cross(omega,txoff)),axis=1)/ra
                rate+=np.sum(b*(velocities-v-np.cross(omega,rxoff)),axis=1)/rb
                te,tl=exact/C0,linear/C0;delta=te-tl
                if w.kind=='pulsed_lfm':
                    phase=np.pi*w.slope_hz_s*(-2*(f-w.pulse_width_s/2)+te+tl)*delta-2*np.pi*w.carrier_hz*delta
                    end=w.pulse_width_s
                else:
                    phase=2*np.pi*delta*(w.start_hz+w.slope_hz_s*f-.5*w.slope_hz_s*(te+tl))
                    end=w.ramp_s
                # Geometry-only error includes unilluminated cells, deliberately conservative.
                ae=(f-te>=0)&(f-te<end);al=(f-tl>=0)&(f-tl<end)
                gate_disagreement+=int((ae!=al).sum());active_count+=int((ae|al).sum());evaluated+=len(indices)
                if len(indices):
                    peak_path=max(peak_path,float(np.max(abs(exact-linear))))
                    peak_velocity=max(peak_velocity,float(np.max(abs(rate-rate0)))/2)
                    index=int(np.argmax(abs(phase)))
                    if abs(phase[index])>peak_phase:
                        peak_phase=float(abs(phase[index]));worst={'scatterer_index':int(indices[index]),
                            'slot_start_s':float(t0),'fast_time_s':float(f),'tx':int(tx),'rx':rx}
    cell=C0/(2*w.sample_rate_hz) if w.kind=='pulsed_lfm' else C0/(2*w.bandwidth_hz)
    dv=w.wavelength_m/(2*w.loops*layout.loop_period_s)
    return {'status':'risk' if peak_phase>phase_tolerance_rad or gate_disagreement else 'usable',
        'reference':'instantaneous exact distance on configured motion; not retarded electromagnetic propagation',
        'metrics':{'max_two_way_path_error_m':peak_path,'max_equivalent_range_error_m':peak_path/2,
                   'max_equivalent_range_error_bins':peak_path/(2*cell),
                   'max_unwrapped_phase_error_rad':peak_phase,
                   'max_instantaneous_closing_velocity_error_mps':peak_velocity,
                   'max_instantaneous_closing_velocity_error_bins':peak_velocity/dv,
                   'sample_gate_disagreements':gate_disagreement},
        'criterion':{'phase_tolerance_rad':float(phase_tolerance_rad),
                     'description':'Engineering screening tolerance, configurable; not an experimentally calibrated fidelity threshold.'},
        'coverage':{'scatterers_checked':len(indices),'scatterers_total':count,
                    'slots_checked':len(chosen),'slots_total':len(slots),'fast_times_per_slot':len(fast),
                    'path_evaluations':evaluated,'active_sample_pairs':active_count,
                    'includes_out_of_beam_geometry':True,'all_fast_samples_checked':len(fast)==w.samples},
        'worst_phase_sample':worst,
        'limitations':['Sampled diagnostic, not a bound on unsampled extrema; increase points/slots/fast samples to check convergence.',
                       'Only path/phase and pulse-gate approximation checked; amplitude, visibility and beam variation within a slot are not validated.',
                       'Trajectory interpolation is not independent motion truth; propagation-retardation errors are excluded.']}
