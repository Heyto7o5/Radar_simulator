#!/usr/bin/env python3
"""Standalone learned XYZ inference: frozen weights + RDA, no DEM/pose truth."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from xyz_model import XYZNet
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from radar_platform.xyz_features import encode_rda


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--rda',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads)
    if args.output.exists():raise FileExistsError('existing output preserved')
    protocol=json.loads((args.model_dir/'training_protocol.json').read_text())
    if json.loads((args.model_dir/'status.json').read_text())['status']!='complete':raise ValueError('training incomplete')
    with np.load(args.rda,allow_pickle=False) as observation,np.load(args.model_dir/'acquisition_axes.npz',allow_pickle=False) as axes:
        for key in axes.files:
            if not np.array_equal(axes[key],observation[key]):raise ValueError(f'{key} differs from model acquisition')
        x=torch.from_numpy(encode_rda(observation['magnitude'])[None])
    outputs=[];started=time.perf_counter()
    for seed in protocol['seeds']:
        checkpoint=torch.load(args.model_dir/f'seed_{seed}.pt',map_location='cpu',weights_only=True)
        if checkpoint['architecture']!='XYZNet_v1':raise ValueError('unsupported model architecture')
        model=XYZNet();model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.inference_mode():outputs.append(model(x).numpy()[0])
    value=np.mean(outputs,axis=0)
    result=dict(xyz_m=value.tolist(),coordinate_order=['north','east','up'],reference='DEM center',
                label_time_s=0,model_seeds=protocol['seeds'],per_seed_xyz_m=np.asarray(outputs).tolist(),
                inference_with_weight_loading_s=time.perf_counter()-started,
                input='RDA magnitude frames and acquisition axes only',
                scope='fixed calibrated scene, attitude and horizontal velocity; not flight-control guidance',
                confidence='not calibrated')
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
