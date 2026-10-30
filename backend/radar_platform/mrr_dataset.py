"""ACE MRR-2 measured spectra, preserving engineering units and missing values.

The published velocity-bin spacing is retained, but zero/sign conventions are
not invented. Native spectrum bins are not falsely converted into complex ADC.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
from datetime import datetime
import h5py
import numpy as np


@dataclass
class MRRSpectra:
    power: np.ndarray  # [record,height,native spectral bin], invalid -> NaN
    height_m: np.ndarray
    utc: list[str]
    quality_percent: np.ndarray
    good_spectra_count: np.ndarray
    velocity_bin_spacing_mps: float
    metadata: dict

    def instrument_velocity_axis(self):
        """MRR-2 native [0,64*dv) fall-positive ALIASED coordinate, not true wind/drop velocity.

        Source: METEK MRR#005 FAQ and Maahn & Kollias (2012), section 2.2.
        Use file-specific spacing, not the different default in other instruments.
        """
        if self.power.shape[2]!=64:raise ValueError('documented MRR-2 velocity convention requires 64 bins')
        return np.arange(64,dtype=float)*self.velocity_bin_spacing_mps


def load_precipitation_events(path,*,event_timezone):
    """Parse source intervals without silently fixing malformed source rows.

    Caller must explicitly choose a timezone: the event CSV itself omits it.
    Absence of an annotation is NOT an annotation of dry/no-precipitation weather.
    """
    lines=Path(path).read_text(encoding='utf-8-sig').splitlines()
    data=[(i+1,s) for i,s in enumerate(lines) if s.strip() and not s.lstrip().startswith('#')]
    events=[];rejected=[]
    for line_no,line in data:
        if line.startswith('ID,'):continue
        try:
            fields=next(csv.reader([line]))
            if len(fields)!=4:raise ValueError('expected four CSV fields')
            ident,start,end,kind=fields
            if kind not in ('0','1','2'):raise ValueError('unknown hydrometeor class')
            start=datetime.strptime(start,'%Y%m%d%H%M').replace(tzinfo=event_timezone)
            end=datetime.strptime(end,'%Y%m%d%H%M').replace(tzinfo=event_timezone)
            if end<=start:raise ValueError('nonpositive event duration')
            events.append({'id':int(ident),'start':start,'end':end,'label':{'0':'snow','1':'rain','2':'hail'}[kind]})
        except (ValueError,csv.Error) as exc:rejected.append({'line':line_no,'source_text':line,'reason':str(exc)})
    return events,rejected


def precipitation_label(utc,events):
    time=datetime.fromisoformat(utc)
    if time.tzinfo is None:raise ValueError('radar time must include timezone')
    matches=[e for e in events if e['start']<=time<e['end']]
    labels=sorted({e['label'] for e in matches})
    return {'label':labels[0] if len(labels)==1 else 'unlabelled' if not labels else 'conflicting_or_mixed',
            'event_ids':[e['id'] for e in matches], 'independent_drop_geometry_truth':False}


def load_mrr_spectra(path: str|Path) -> MRRSpectra:
    with h5py.File(path,'r') as f:
        required=['Fnn','H','Time','MDQ','NSA']
        if not all(k in f for k in required):raise ValueError('MRR raw spectra variables missing')
        power=np.asarray(f['Fnn'],dtype=np.float32);height=np.asarray(f['H'],float)
        times=np.asarray(f['Time']);quality=np.asarray(f['MDQ'],float);count=np.asarray(f['NSA'],float)
        if power.ndim!=3 or height.ndim!=1 or power.shape[1]!=len(height) or times.shape[0]!=len(power) or quality.shape!=(len(power),) or count.shape!=(len(power),):
            raise ValueError('MRR dimensions inconsistent; do not guess axis order')
        if times.ndim!=2 or times.dtype.kind!='S':raise ValueError('expected character-array UTC times')
        utc=[b''.join(row).decode('ascii').strip('\x00 ') for row in times]
        spacing=np.asarray(f.attrs.get('Velocity-Resolution-value',[]),float)
        if spacing.size!=1 or not np.isfinite(spacing).all() or spacing.item()<=0:raise ValueError('documented velocity bin spacing missing')
        if not np.isfinite(height).all() or np.any(np.diff(height)<=0):raise ValueError('invalid height axis')
        invalid=~np.isfinite(power)|(power<0)
        power[invalid]=np.nan # accommodates source NaN and documented negative sentinel.
        def text(v):return v.decode() if isinstance(v,bytes) else str(v)
        units=text(f['Fnn'].attrs.get('Units',''))
        if units!='engineering-units':raise ValueError('unrecognized spectral power units')
        meta={'source_file':str(Path(path)),'is_real':True,'has_complex_adc':False,
              'axes':['record','height','native_doppler_bin'],'units':units,
              'invalid_fraction':float(invalid.mean()),'zero_height_gate_retained_for_audit':bool(np.any(height==0)),
              'velocity_axis_zero_and_sign_verified':False,'absolute_calibration_applied':False,
              'native_mrr2_coordinate_documented':power.shape[2]==64,'true_unaliased_velocity_verified':False,
              'quality_filter_applied':False,'TF_applied':False,
              'warning':'Power spectra only; native bin spacing does not establish signed-velocity origin. Ship motion/vertical wind not corrected here.'}
        return MRRSpectra(power,height,utc,quality,count,float(spacing.item()),meta)
