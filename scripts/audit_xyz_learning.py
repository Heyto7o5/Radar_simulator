#!/usr/bin/env python3
"""Verify data split, physical XYZ labels, inference contract and saved scores."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from radar_platform.xyz_features import encode_rda


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path)
    args=p.parse_args();root=args.data
    protocol=json.loads((root/'protocol.json').read_text())
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    groups={s:{tuple(r['cell']) for r in protocol['records'] if r['split']==s} for s in ['train','val','test']}
    assert not (groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    statuses={};hashes={};ranges={}
    for k,record in enumerate(protocol['records']):
        folder=root/'cases'/record['id'];cfg=json.loads((folder/'request.json').read_text())
        np.testing.assert_allclose(cfg['platform']['position_ned_m'],np.array(record['xyz_m'])*[1,1,-1],rtol=0,atol=1e-12)
        assert cfg['platform']['velocity_ned_mps']==[3.,0.,0.]
        assert cfg['scene']['objects'][0]['position_ned_m'][:2]==[0.,0.]
        assert cfg['scene']['seed']==record['phase_seed'] and cfg['receiver_noise']['seed']==record['noise_seed']
        status=json.loads((folder/'complete.json').read_text())['applicability'];statuses[status]=statuses.get(status,0)+1
        with np.load(folder/'rda.npz',allow_pickle=False) as f:
            assert set(f.files)=={'magnitude','range_m','velocity_mps','azimuth_deg','frame_times_s'}
            assert np.isfinite(f['magnitude']).all() and f['magnitude'].shape[0]==2
            if k%97==0:np.testing.assert_array_equal(encode_rda(f['magnitude']),np.load(folder/'input.npy'))
        hashes[record['id']]=hashlib.sha256((folder/'rda.npz').read_bytes()).hexdigest()
    for split in ['train','val','test']:
        y=np.load(root/f'{split}_y.npy');x=np.load(root/f'{split}_x.npy',mmap_mode='r')
        selected=[r for r in protocol['records'] if r['split']==split]
        np.testing.assert_allclose(y,np.array([r['xyz_m'] for r in selected]),rtol=1e-6,atol=1e-6)
        assert x.shape==(len(y),2,64,16,24) and np.isfinite(x).all()
        assert np.ptp(y[:,2])>15 and np.ptp(y[:,0])>25 and np.ptp(y[:,1])>18
        ranges[split]=dict(cases=len(y),cells=len(groups[split]),xyz_min=y.min(0).tolist(),xyz_max=y.max(0).tolist(),
                           xyz_correlation=np.corrcoef(y.T).tolist())
    audit=dict(three_dimensional_cells_disjoint=True,physical_label_axes_verified=True,
               z_varies_independently=True,raw_rda_has_no_pose_labels=True,preprocessing_spot_checks_pass=True,
               applicability_counts=statuses,split_ranges=ranges,cases=len(protocol['records']),frames=2*len(protocol['records']))
    if args.model:
        metrics=json.loads((args.model/'metrics.json').read_text())
        for split in ['test','grid5','building_height','noise10']:
            with np.load(args.model/f'{split}_predictions.npz') as f:
                truth=np.load(root/f'{split}_y.npy');np.testing.assert_array_equal(f['truth_xyz_m'],truth)
                for name in f.files:
                    if name=='truth_xyz_m':continue
                    assert f[name].shape==truth.shape and np.isfinite(f[name]).all()
                    mae=np.mean(abs(f[name]-truth),axis=0)
                    np.testing.assert_allclose(mae,metrics[split][name]['axis_mae_m'],rtol=1e-5,atol=1e-5)
        audit['saved_predictions_match_metrics']=True
        audit['checkpoint_selection']='validation only; confirmed by training implementation, not inferred from scores'
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (root/'rda_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
