"""Isolate sampling-rate folding using saved SISO ADC, without resynthesis."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from radar_platform.unified.schema import Acquisition
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.processing import process_adc

p=ROOT/'.runtime/original_dem_comparison_20261023'
raw=json.loads((p/'new_config.json').read_text())
raw['waveform']['loops']=64
raw['array']['loop_period_s']=66.6e-6
q=Acquisition.model_validate({k:raw[k] for k in ['waveform','platform','array','processing']})
adc=np.fromfile(p/'new/frame_0000/adc.bin',dtype='<c8').reshape(333,128,1,1)[:,::2]
r=process_adc(adc,q,build_layout(q.array,q.waveform,q.platform))
i,j=np.unravel_index(r.rd_power.argmax(),r.rd_power.shape)
out=p/'mimo_2tx2rx'
np.savez_compressed(out/'siso_even_slots_control.npz',rd=r.rd_channels,
                    range_m=r.range_m,velocity_mps=r.velocity_mps)
record={'purpose':'Sampling-only control: take even slots of the SAME saved SISO ADC; no change to geometry or propagation.',
        'range_m':float(r.range_m[i]),'apparent_velocity_mps':float(r.velocity_mps[j]),
        'same_tx_period_s':66.6e-6,'loops':64,'source':'../new/frame_0000/adc.bin',
        'conclusion':'A negative-velocity peak appears without adding any array, demonstrating temporal aliasing rather than target reversal.'}
(out/'sampling_control.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps(record))
