"""Batch identical geometric rays across scatterers; no angular/power pruning."""
import numpy as np
from .radiation import transmit_response


def clear_masks(request,scene,layout,visibility,starts,tx,trajectory_state=None,cancelled=None,aperture_diameter=None):
    n=len(scene.amplitudes);loops=len(starts);nr=request.array.num_rx
    # Bounded optional cache, per logical Tx. Small jobs keep the original path.
    if not (request.visibility.occlusion_enabled and request.visibility.accelerator=='bvh' and
            request.visibility.batch_bvh and n>=256 and n*loops*(1+nr)<=16_000_000):
        return None
    tx_clear=np.ones((n,loops),bool);rx_clear=np.ones((n,loops,nr),bool)
    p=request.platform;w=request.waveform
    p0=np.array(p.position_ned_m);pv=np.array(p.velocity_ned_mps)
    for k,t in enumerate(starts):
        if cancelled is not None and cancelled():
            from .engine import SimulationCancelled
            raise SimulationCancelled()
        if trajectory_state is None:
            pos=p0+pv*t;basis=layout.basis_ned
            delta=scene.positions_ned_m-p0+(scene.velocities_ned_mps-pv)*t
        else:
            pos=trajectory_state[0][k];basis=trajectory_state[1][k]
            delta=scene.positions_ned_m+scene.velocities_ned_mps*t-pos
        radius=np.linalg.norm(delta,axis=1)
        if np.any(radius<=1e-6):raise ValueError('scatterer intersects radar in visibility batching')
        beam=np.sum(delta*basis[:,0],axis=1)/radius>=np.cos(np.deg2rad(p.beam_half_angle_deg))
        txoff=basis@layout.tx_local_m[tx]
        if request.radiation is not None:
            support=np.zeros(n,bool)
            for begin in range(0,n,256):
                gain=transmit_response((delta[begin:begin+256]-txoff)[:,None,:],basis,np.array([t]),request.radiation,w,aperture_diameter)
                support[begin:begin+256]=abs(gain[:,0])>0
            beam=support if request.radiation.replace_legacy_cone else beam&support
        ids=np.flatnonzero(beam)
        if not len(ids):continue
        targets=scene.positions_ned_m[ids]+scene.velocities_ned_mps[ids]*t
        tx_clear[ids,k]=visibility.clear(pos+txoff,targets,t)
        selected=ids[tx_clear[ids,k]]
        if not len(selected):continue
        targets=scene.positions_ned_m[selected]+scene.velocities_ned_mps[selected]*t
        for rx in range(nr):
            rx_clear[selected,k,rx]=visibility.clear(pos+basis@layout.rx_local_m[rx],targets,t)
    return tx_clear,rx_clear
