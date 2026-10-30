"""Sequence-aware RAMP inputs and an explicitly documented shared RAD grid."""
from dataclasses import dataclass
from pathlib import Path
import csv,math
from collections import Counter
from decimal import Decimal,InvalidOperation
import numpy as np
from scipy.io import loadmat
from .models import C0,TargetConfig
from .fmcw import FMCWConfig
from .array_fmcw import TDMArray,process_array_adc
from .ramp_dataset import normalize_adc_axes
from .ramp_label_priors import relative_class_amplitude as _class_rcs_scale


PUBLIC_FRAME_PERIOD_S=33.33333e-3
PUBLIC_SLOPE_HZ_S=21e12


class AnnotationTracks:
    """Only camera/annotation inputs enter; radar samples are never fitted here."""
    def __init__(self,label_directory: Path,window_frames=11):
        if window_frames<3 or window_frames%2!=1:raise ValueError('odd trajectory window >=3 required')
        self.frames={};self.tracks={};self.window_frames=window_frames;self.ambiguous_frames={}
        for file in sorted(Path(label_directory).glob('*.csv')):
            index=int(file.stem);labels=[]
            with file.open(encoding='utf-8-sig',newline='') as handle:
                for row in csv.reader(handle):
                    if len(row)<6:continue
                    try:uid=format(Decimal(row[0]).normalize(),'f')
                    except InvalidOperation:uid=str(row[0]).strip()
                    class_id=int(float(row[1]));x,y,w,l=map(float,row[2:6])
                    if not np.isfinite([x,y,w,l]).all() or y<=0 or w<=0 or l<=0:continue
                    label={'uid':uid,'class_id':class_id,'x_m':x,'y_m':y,'width_m':w,'length_m':l,'range_m':math.hypot(x,y)}
                    labels.append(label)
            duplicates={uid for uid,n in Counter(v['uid'] for v in labels).items() if n>1}
            if duplicates:self.ambiguous_frames[index]=sorted(duplicates)
            for ordinal,label in enumerate(labels):
                uid=label['uid'];label['source_uid']=uid;label['tracking_ambiguous']=uid in duplicates
                if uid in duplicates:
                    # Retain the single-frame geometry, but do not invent a
                    # temporal identity or derive velocity across two objects.
                    label['uid']=f'{uid}@untracked:{index}:{ordinal}'
                else:self.tracks.setdefault(uid,{})[index]=label
            self.frames[index]=labels

    def labels(self,index,*,smooth=False):
        output=[]
        for raw in self.frames.get(index,[]):
            track={index:raw} if raw['tracking_ambiguous'] else self.tracks[raw['uid']];label=dict(raw)
            half=self.window_frames//2 if smooth else 1
            # Do not bridge annotation gaps, duplicate-ID intervals or class changes.
            support=[index]
            for direction in [-1,1]:
                for offset in range(1,half+1):
                    k=index+direction*offset
                    if k not in track or track[k]['class_id']!=raw['class_id']:break
                    support.append(k)
            support.sort()
            t=(np.asarray(support)-index)*PUBLIC_FRAME_PERIOD_S
            xy=np.array([[track[k]['x_m'],track[k]['y_m']] for k in support])
            known=len(support)>=2
            degree=min(2 if smooth else 1,len(support)-1)
            if smooth:
                design=np.column_stack([t**i for i in range(degree+1)])
                coefficients=np.linalg.lstsq(design,xy,rcond=None)[0]
                center=coefficients[0];velocity=coefficients[1] if known else np.zeros(2)
                residual=float(np.sqrt(np.mean((design@coefficients-xy)**2)))
            else:
                center=np.array([raw['x_m'],raw['y_m']]);residual=None
                velocity=(xy[-1]-xy[0])/(t[-1]-t[0]) if known else np.zeros(2)
            r=float(np.linalg.norm(center))
            if r<=0 or center[1]<=0:continue
            label.update(x_m=float(center[0]),y_m=float(center[1]),range_m=r,
                radial_velocity_mps=float(-np.dot(center,velocity)/r) if known else None,
                velocity_ned_mps=[float(velocity[1]),float(velocity[0]),0.],
                raw_position_xy_m=[raw['x_m'],raw['y_m']],
                velocity_source=('annotation_local_quadratic' if smooth else 'annotation_neighbor_difference') if known else 'unavailable',
                simulation_velocity_assumption=None if known else 'zero_placeholder_not_truth',
                trajectory_support_frames=support,trajectory_residual_m=residual,
                timestamp_s=index*PUBLIC_FRAME_PERIOD_S)
            output.append(label)
        return output


def read_ramp_adc(path):
    values=loadmat(Path(path))
    arrays=[v for v in values.values() if isinstance(v,np.ndarray) and v.ndim==4 and sorted(v.shape)==[2,4,128,255]]
    if len(arrays)!=1 or not np.iscomplexobj(arrays[0]):raise ValueError('exactly one complex RAMP ADC cube required')
    result=normalize_adc_axes(arrays[0])
    if not np.isfinite(result).all():raise ValueError('non-finite raw ADC')
    return result


def aligned_config():
    # The public profile specifies the ADC-sampled bandwidth and chirp interval,
    # not the complete analog ramp or ADC-start delay. Model only the observed
    # 32us segment. Its frequency origin uses published 77GHz start as an explicit
    # approximation; no claim of calibrated absolute phase or full ramp timing.
    duration=128/4e6
    return FMCWConfig(77e9,77e9+PUBLIC_SLOPE_HZ_S*duration,duration,60e-6,128,255,4e6)


def aligned_array(east_sign=-1):
    if east_sign not in (-1,1):raise ValueError('array handedness must be +/-1')
    # Nominal horizontal 2Tx4Rx geometry per dataset + TI MIMO diagram.
    spacing=C0/77e9/2
    tx=np.zeros((2,3));rx=np.zeros((4,3))
    tx[:,1]=east_sign*np.array([0,4])*spacing;rx[:,1]=east_sign*np.arange(4)*spacing
    return TDMArray(tx,rx,np.array([0,60e-6]),120e-6)


def process_aligned_adc(adc,*,east_sign=-1,closing_sign=-1):
    return process_array_adc(adc,aligned_config(),aligned_array(east_sign),azimuth_deg=np.linspace(-80,80,81),
        closing_doppler_sign=closing_sign,spatial_phase_sign=-1,range_window='hamming',doppler_window='hamming',spatial_window='hamming')


def track_configuration(labels,*,east_sign=-1):
    """Public labels -> reusable common configuration; no radar observations used."""
    from .unified.array_adapter import motion_array_configuration
    targets=[]
    for label in labels:
        # Fixed 10m amplitude reference; no per-frame nearest-target renormalization.
        targets.append(TargetConfig(id=str(label['uid']),position_ned_m=(label['y_m'],label['x_m'],0.),
            velocity_ned_mps=tuple(label['velocity_ned_mps']),amplitude=_class_rcs_scale(int(label['class_id']))))
    config=motion_array_configuration(aligned_config(),targets,aligned_array(east_sign),
        beam_half_angle_deg=89.,amplitude_reference_range_m=10.,allow_empty_scene=True)
    config.name='RAMP label-conditioned ideal point scene'
    config.input_provenance=[
        'Source: RAMP public annotations; x=right/east, y=forward/north converted to world NED.',
        'Objects are annotation-center point targets, not vehicle CAD or measured scattering centers.',
        'Per-class amplitudes are relative heuristic coefficients, not measured RCS.',
        'Velocity vectors must be supplied by the caller; label-derived motion is not precise independent radar truth.',
        'Waveform models only the published 32 us ADC segment; 77 GHz start is an approximation, not a verified full ramp log.',
        f'Nominal 2Tx4Rx geometry and handedness east_sign={east_sign}; not individually calibrated antenna coordinates.'
    ]
    return config


def simulate_track_adc(labels,*,east_sign=-1):
    from .unified.session import SimulationSession
    return SimulationSession(track_configuration(labels,east_sign=east_sign)).frame(0).adc


def annotation_masks(labels,ranges,velocities,angles):
    """Known geometry region, not test-image peak alignment or a tight velocity oracle.

    Annotation range is uncertain and object center need not equal its scattering
    center. Fixed 1.5m/15deg margins plus object size, declared before evaluation.
    Doppler is not gated because label-difference velocities are noisy estimates.
    """
    ra=np.zeros((len(ranges),len(angles)),dtype=bool);guard=np.zeros_like(ra)
    for label in labels:
        r=float(label['range_m']);theta=np.rad2deg(np.arctan2(label['x_m'],label['y_m']))
        extent=.5*(abs(label['x_m']/r)*label['width_m']+abs(label['y_m']/r)*label['length_m'])
        dr=1.5+extent;da=15+np.rad2deg(np.arctan2(max(label['width_m'],label['length_m'])/2,r))
        ra|=(abs(ranges-r)<=dr)[:,None]&(abs(angles-theta)<=da)[None,:]
        guard|=(abs(ranges-r)<=dr+.75)[:,None]&(abs(angles-theta)<=da+8)[None,:]
    valid=(ranges>=1)&(ranges<=24)
    target=np.broadcast_to((ra&valid[:,None])[:,None,:],(len(ranges),len(velocities),len(angles)))
    background=np.broadcast_to((~guard&valid[:,None])[:,None,:],target.shape)
    return target,background,np.broadcast_to(valid[:,None,None],target.shape)
