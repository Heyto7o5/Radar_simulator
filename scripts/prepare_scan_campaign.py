#!/usr/bin/env python3
"""Paired same-time controls for scanning on every fixed real DEM site."""
import argparse,hashlib,json
from pathlib import Path
from radar_platform.unified.schema import Simulation
from radar_platform.unified.preflight import preflight


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-configs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for site in ['alpine','valley','foothill']:
        for scan in [False,True]:
            raw=json.loads((a.base_configs/f'{site}_hamming_0_s11.json').read_text())
            raw['sequence']['start_time_s']=.02
            variant='scan8_t20ms' if scan else 'hold0_t20ms'
            if scan:raw['radiation']['transmit_aperture']['scan_schedule']=[dict(time_s=.02,azimuth_deg=8,elevation_deg=0)]
            key=f'{site}_{variant}_s11';raw['name']=key
            raw['input_provenance'].append('At 20 ms, scanning is compared with a held-pointing control at the identical platform time; not with an earlier platform position.')
            config=Simulation.model_validate(raw);preflight(config);path=a.output/(key+'.json')
            path.write_text(config.model_dump_json(indent=2)+'\n')
            rows.append(dict(case=key,site=site,seed=11,variant=variant,config=path.name,
                             config_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (a.output/'plan.json').write_text(json.dumps({'purpose':'paired global-time electronic scanning controls','cases':rows,
        'simulation_times_s':[0,.02],'time_zero_reference':'hamming_0 seed 11 in the frozen main campaign',
        'independent_real_sites':3,'measured_radar_available':False},indent=2))
    print(json.dumps({'cases':len(rows)}))


if __name__=='__main__':main()
