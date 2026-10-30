import numpy as np
import h5py
import pytest
from radar_platform.mrr_dataset import load_mrr_spectra

def source(path):
    with h5py.File(path,'w') as f:
        f.attrs['Velocity-Resolution-value']=[.2]
        p=f.create_dataset('Fnn',data=np.array([[[1,-99900,2,3],[2,3,4,5]]],dtype=np.float32));p.attrs['Units']=b'engineering-units'
        f['H']=[0,100];f['MDQ']=[100];f['NSA']=[58]
        f['Time']=np.array([list('2016-11-17T19:00:00+00:00')],dtype='S1')

def test_real_spectrum_mask_units_and_no_invented_complex_or_velocity_axis(tmp_path):
    p=tmp_path/'mrr.nc';source(p);a=load_mrr_spectra(p)
    assert a.power.shape==(1,2,4) and np.isnan(a.power[0,0,1])
    assert not np.iscomplexobj(a.power)
    assert a.metadata['has_complex_adc'] is False and a.metadata['is_real'] is True
    assert not a.metadata['velocity_axis_zero_and_sign_verified']
    assert a.velocity_bin_spacing_mps==.2 and a.utc==['2016-11-17T19:00:00+00:00']

def test_inconsistent_dims_are_rejected(tmp_path):
    p=tmp_path/'mrr.nc';source(p)
    with h5py.File(p,'a') as f:
        del f['H'];f['H']=[0,100,200]
    with pytest.raises(ValueError,match='dimensions'):load_mrr_spectra(p)


def test_instrument_velocity_coordinate_is_explicit_and_not_dealiased(tmp_path):
    p=tmp_path/'mrr.nc';source(p)
    a=load_mrr_spectra(p)
    with pytest.raises(ValueError,match='64 bins'):a.instrument_velocity_axis()
    a.power=np.zeros((1,2,64),np.float32)
    np.testing.assert_array_equal(a.instrument_velocity_axis(),np.arange(64)*.2)
    assert a.instrument_velocity_axis()[-1]==pytest.approx(12.6)
    # Calling the explicit instrument-coordinate function does not assert true velocity.
    assert not a.metadata['velocity_axis_zero_and_sign_verified']


def test_event_join_retains_source_error_and_unlabelled_is_not_dry(tmp_path):
    from datetime import timezone
    from radar_platform.mrr_dataset import load_precipitation_events,precipitation_label
    p=tmp_path/'events.csv'
    p.write_text('ID,start,end,rain\n1,201611171900,201611180300,1\n6,201612210000.201612232100,1\n')
    events,rejected=load_precipitation_events(p,event_timezone=timezone.utc)
    assert len(events)==1 and len(rejected)==1 and rejected[0]['source_text'].startswith('6,')
    assert precipitation_label('2016-11-17T21:00:00+00:00',events)['label']=='rain'
    assert precipitation_label('2016-11-02T21:00:00+00:00',events)['label']=='unlabelled'
