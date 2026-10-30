#!/usr/bin/env python3
"""Platform forward simulation; no measured radar samples enter this command."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from radar_platform.maroon_dataset import DEFAULT_ROOT,geometry_and_radar
from radar_platform.surface_echo import simulate_surface_spectrum


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-bounces",type=int,default=3,choices=[1,2,3])
    args=p.parse_args()
    if args.output.exists():
        raise SystemExit("choose a new output directory; existing runs are immutable")
    mesh,radar,metadata=geometry_and_radar(args.root,args.max_bounces)
    started=time.perf_counter()
    result=simulate_surface_spectrum(mesh,radar)
    elapsed=time.perf_counter()-started
    args.output.mkdir(parents=True)
    api_request={"waveform":"sfcw","coordinate_frame":"NED_m",
        "frequencies_hz":radar.frequencies_hz.tolist(),"tx_ned_m":radar.tx_ned_m.tolist(),
        "rx_ned_m":radar.rx_ned_m.tolist(),"vertices_ned_m":mesh.vertices_m.tolist(),
        "faces":mesh.faces.tolist(),"max_bounces":radar.max_bounces}
    (args.output/"api_request.json").write_text(json.dumps(api_request,indent=2)+"\n")
    np.savez_compressed(args.output/"forward.npz",signal=result.signal,
        frequencies_hz=radar.frequencies_hz,vertices_ned_m=mesh.vertices_m,faces=mesh.faces,
        tx_ned_m=radar.tx_ned_m,rx_ned_m=radar.rx_ned_m,
        **{f"bounce_{k}":v for k,v in result.by_bounce.items()},
        **{f"path_count_{k}":v for k,v in result.path_counts.items()})
    metadata.update({"method":"finite-triangle scalar PEC geometric optics, image sources",
        "elapsed_s":elapsed,"max_bounces":radar.max_bounces,"shape":list(result.signal.shape),
        "path_min_m":result.path_length_min_m,"path_max_m":result.path_length_max_m,
        "paths_by_bounce":{k:int(v.sum()) for k,v in result.path_counts.items()},
        "read_measured_signal":False,"absolute_power_calibrated":False,
        "limitations":["no polarization", "no diffraction", "no diffuse scattering", "isotropic antennas", "no slow-time/Doppler data"]})
    (args.output/"forward.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
