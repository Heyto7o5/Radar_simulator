#!/usr/bin/env python3
"""Read-only validation of completed frame outputs while the batch continues."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from radar_platform.unified.schema import Simulation
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.processing import range_compress,doppler_channels,range_axis,selected_range_indices

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'docs/validation_diagnostics/2026-10-15/dart'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='batch_v2_world_geometry');args=parser.parse_args()
    if not args.batch.replace('_','').isalnum():raise ValueError('invalid batch tag')
    folder=ROOT/args.batch
    state=json.loads((folder/'state.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    records={row['valid_frame_index']:row for row in manifest['records']}
    drift=[name for name,value in manifest['hashes'].items() if digest(PROJECT/name)!=value]
    if drift:raise ValueError('frozen simulation source changed: '+','.join(drift))
    results=[];invariant_hashes=set()
    for key,job in sorted(state['jobs'].items(),key=lambda pair:int(pair[0])):
        if job['status']!='complete':continue
        index=int(key);result_path=Path(job['result']);result=json.loads(result_path.read_text())
        request_path=result_path.with_name(result_path.name.replace('_status.json','_request.json'))
        request=Simulation.model_validate_json(request_path.read_text());wave=request.waveform
        with np.load(result['file'],allow_pickle=False) as f:
            adc=f['adc'];saved=f['rd_channels'];ranges=f['range_m'];velocity=f['closing_velocity_mps']
        expected=(wave.samples,wave.loops,request.array.num_rx,request.array.num_tx)
        if adc.shape!=expected or not np.iscomplexobj(adc) or not np.isfinite(adc).all():raise ValueError('invalid ADC output')
        if result['raw_frame_index']!=records[index]['raw_frame_index']:raise ValueError('source frame correspondence mismatch')
        layout=build_layout(request.array,wave,request.platform)
        replay,closing,_=doppler_channels(range_compress(adc,request),request,layout)
        if replay.shape!=saved.shape or not np.isfinite(saved).all():raise ValueError('invalid RD output')
        error=float(np.max(abs(replay-saved))/max(float(np.max(abs(saved))),1e-30))
        if error>1e-6:raise ValueError('saved RD not reproducible from saved ADC')
        if not np.allclose(ranges,range_axis(request)[selected_range_indices(request)],rtol=0,atol=1e-12):raise ValueError('range axis changed')
        if not np.allclose(velocity,closing*wave.wavelength_m/2,rtol=0,atol=1e-12):raise ValueError('velocity axis changed')
        invariant={key:request.model_dump()[key] for key in ['waveform','array','scene','processing','receiver_noise','range_background']}
        invariant_hashes.add(hashlib.sha256(json.dumps(invariant,sort_keys=True).encode()).hexdigest())
        results.append({'valid_frame_index':index,'raw_frame_index':result['raw_frame_index'],'adc_shape':list(adc.shape),
                        'rd_shape':list(saved.shape),'relative_complex_replay_error':error,'output_sha256':digest(Path(result['file']))})
    if len(invariant_hashes)>1:raise ValueError('geometry/radar/processing inputs changed between frames')
    audit={'as_of_utc':datetime.now(timezone.utc).isoformat(),'observed_batch_status':state['status'],
           'planned_frames':len(records),'checked_completed_frames':len(results),
           'partial_audit':len(results)!=len(records),'source_drift':drift,'frames':results,
           'meaning':'file integrity, frame mapping, invariant inputs and ADC-to-RD numerical replay only; NOT measured fidelity'}
    (folder/'output_integrity_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
