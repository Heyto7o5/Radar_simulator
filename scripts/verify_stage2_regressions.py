#!/usr/bin/env python3
"""Check optional new models do not alter old configured ADC outputs."""
import argparse,json
from pathlib import Path
import numpy as np
from radar_platform.unified.schema import Simulation
from radar_platform.unified.session import SimulationSession

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('regression record exists')
    base=ROOT/'.runtime/consolidation_20261016/acceptance_matrix'
    paths=[base/name for name in ['high_altitude_1ghz','fmcw_siso','fmcw_simo','fmcw_mimo','moving_plate_po','terrain_flat','terrain_hills','terrain_dem']]
    paths += [ROOT/f'.runtime/real_dem_20261022/{s}_30m_10ghz_run' for s in ['alpine','valley','foothill']]
    rows=[]
    for root in paths:
        request=Simulation.model_validate_json((root/'request.json').read_text())
        current=SimulationSession(request).frame(0).adc
        meta=json.loads((root/'frame_0000/metadata.json').read_text());spec=meta['products']['adc']
        original=np.fromfile(root/'frame_0000'/spec['file'],dtype=spec['dtype']).reshape(spec['shape'])
        row={'source_run':str(root.relative_to(ROOT)),'adc_max_absolute_difference':float(np.max(abs(current-original))),
             'bitwise_equal':bool(np.array_equal(current,original))}
        rows.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2))
    if not all(r['bitwise_equal'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
