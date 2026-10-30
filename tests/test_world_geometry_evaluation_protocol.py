"""Synthetic end-to-end scoring fixture: checking frames must not refit gain."""
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('world_evaluation_test',SCRIPTS/'evaluate_dart_world_geometry.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_scoring_uses_calibration_gain_and_saved_region_masks(tmp_path,monkeypatch):
    base=tmp_path/'batch_v1';out=tmp_path/'batch_v2';data=tmp_path/'data'
    base.mkdir();out.mkdir();(tmp_path/'colored_background_followup').mkdir()
    cache=data/'processed/batch_v1/evaluation';cache.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps({'hashes':{},'input_sha256':{}}))
    (out/'state.json').write_text(json.dumps({'status':'complete'}))
    (base/'frozen_calibration.json').write_text(json.dumps({'gain':1.,'shared_power_reference':2.}))
    (tmp_path/'colored_background_followup/frozen_colored_model.json').write_text(json.dumps({'spectral_power':[0.]*8}))
    ranges=np.arange(8,dtype=float);velocity=np.array([-2.,-1.,0.,1.])
    mask=np.zeros((8,4),bool);mask[:,0]=True
    for index in [32,40,48,56,64,72,80,88]:
        np.savez(cache/f'f{index:04d}_power_products.npz',range_m=ranges,closing_velocity_mps=velocity,
                 primary_mask=mask,geometry_mask=mask)
    def load(index):
        adc=np.full((8,4,1,1),2.,np.complex64)
        # Cal: real=8, sim=4 => gain=2. Check: real=12 must retain error.
        real=np.full((8,4),8. if index<32 else 12.)
        real[:,1:]=1000. # excluded bins must not contaminate the primary metric.
        return SimpleNamespace(),adc,ranges,velocity,real
    monkeypatch.setattr(m,'OUT',out);monkeypatch.setattr(m,'BASE',base);monkeypatch.setattr(m,'DATA',data)
    monkeypatch.setattr(m,'load',load)
    monkeypatch.setattr(m,'align',lambda adc,request,r,v:abs(adc[:,:,0,0])**2)
    monkeypatch.setattr(sys,'argv',['evaluate_world_fixture'])
    m.main()
    calibration=json.loads((out/'frozen_calibration.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert calibration['refitted_gain']==2.
    expected=10*np.log10(12/8)
    result=summary['models']['expanded_clean_refitted']
    assert result['primary']['mean_mae_db']==pytest.approx(expected)
    assert result['geometry']['mean_mae_db']==pytest.approx(expected)
    assert result['full']['mean_mae_db']>expected
    assert summary['evaluation_frames']==8 and summary['independent_holdout_scenes']==0
