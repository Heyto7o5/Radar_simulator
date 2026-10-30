"""Read-only original-project replay, followed by common-engine DEM comparison.

Run export with the original py311 environment; run compare with the V2 venv.
Original source files and DEM are never modified. No fitted radar amplitudes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OLD = Path('/home/heyiting/projects/radar_simulator_complex')
ROOT = Path(__file__).resolve().parents[1]
DEM = OLD / 'datasets/AP_01759_FBS_F0570_RT1_3.mat'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def export(out):
    sys.path.insert(0, str(OLD))
    from radar_simulator import RadarSimulator
    from echo import gen_echo
    from signal_processing import pulse_doppler
    from scipy.spatial.transform import Rotation
    out.mkdir(parents=True, exist_ok=False)
    rs = RadarSimulator(str(OLD/'configs/config_complex_motion.yml'))
    rs.load_terrain(str(DEM))
    rs.fly_with_full_state(2)
    n = rs.parset.num_in_CPI
    ant = rs.get_antenna_pointing()
    # Generate first CPI with unchanged original echo and processing functions.
    adc = gen_echo(rs.parset, rs.terrain, rs.loc[:n], rs.vel[:n],
                   batch_size=12, shadowing=False, antenna_vec=ant[:n])
    rd = pulse_doppler(rs.parset, adc)[:, :, 0]
    xyz = rs.terrain.targets.astype(float).copy()
    xyz[:, 2] *= -1
    hit = np.zeros(len(xyz), bool)
    for p, a in zip(rs.loc[:n], ant[:n]):
        delta = xyz-p
        hit |= delta@a/np.linalg.norm(delta, axis=1) >= np.cos(rs.parset.beam_width)
    iy, ix = np.where(hit.reshape(rs.terrain.size))
    if not len(ix):
        raise RuntimeError('Original scene has no illuminated DEM nodes')
    xlo,xhi=max(0,int(ix.min())-2),min(rs.terrain.size[1],int(ix.max())+3)
    ylo,yhi=max(0,int(iy.min())-2),min(rs.terrain.size[0],int(iy.max())+3)
    # Reuse the actual old pointing and positions, including old motion defects;
    # do not attribute trajectory-generator differences to the echo engine.
    bases=[]
    for f in ant[:n+1].astype(float):
        f=f/np.linalg.norm(f);right=np.array([0.,1.,0.]);right-=f*np.dot(f,right)
        right/=np.linalg.norm(right);bases.append(np.column_stack([f,right,np.cross(f,right)]))
    np.savez_compressed(out/'original.npz',adc=adc,rd=rd,
        range_m=np.arange(rd.shape[1])*rs.parset.range_reso,
        velocity_mps=-np.fft.fftshift(np.fft.fftfreq(n,rs.parset.PRI))*rs.parset.wavelength/2,
        time_s=np.arange(n+1)*rs.parset.PRI,position=rs.loc[:n+1],velocity=rs.vel[:n+1],
        quaternion=Rotation.from_matrix(bases).as_quat(),boresight=ant[:n+1],
        north=rs.terrain.x[0,xlo:xhi],east=rs.terrain.y[ylo:yhi,0],
        elevation=rs.terrain.z[ylo:yhi,xlo:xhi],full_elevation=rs.terrain.z,
        full_north=rs.terrain.x[0],full_east=rs.terrain.y[:,0],
        illuminated_nodes=hit.reshape(rs.terrain.size))
    files=[*OLD.glob('*.py'),OLD/'configs/config_complex_motion.yml',DEM]
    dump(out/'original_provenance.json',{
        'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'config':str(OLD/'configs/config_complex_motion.yml'),'dem':str(DEM.resolve()),
        'original_grid':[1000,1000],'grid_spacing_m':float(rs.terrain.resolution),
        'cropped_indices_xy':[xlo,xhi,ylo,yhi],'cropped_shape':[yhi-ylo,xhi-xlo],
        'illuminated_original_nodes':int(hit.sum()),'first_cpi_only':True,
        'note':'Crop retains all nodes inside beam at every pulse plus two-node halo. No decimation or height scaling. Full original engine uses full DEM.',
        'speed_mps':float(np.linalg.norm(rs.vel[0])),
        'old_rd_sign':'Old exp(+j4piR/lambda): closing velocity = -FFT frequency * lambda/2',
        'old_range_scaling':'chirp energy normalized; forward FFT divides by num_fft; RD unnormalized',
        'old_num_fft':rs.parset.num_fft,'old_n_chirp':rs.parset.n_chirp})
    print(json.dumps({'exported':str(out),'crop':[yhi-ylo,xhi-xlo],'rd':rd.shape}),flush=True)


def compare(out):
    sys.path.insert(0,str(ROOT/'backend'))
    from radar_platform.unified.schema import Simulation
    from radar_platform.unified.session import SimulationSession
    d=np.load(out/'original.npz');meta=json.loads((out/'original_provenance.json').read_text())
    # Same Gaussian voltage envelope as original; half assigned to Tx and Rx.
    angles=np.linspace(0,180,721)
    pattern={'kind':'axisymmetric_table','angle_deg':angles.tolist(),
        'power_db':np.maximum(-80,-(angles/5)**2*5/np.log(10)).tolist(),
        'frequency_min_hz':.99e9,'frequency_max_hz':1.01e9,
        'evidence':'analytic','source':'Original echo.py Gaussian beam, split equally between Tx and Rx; hard 10 degree support retained'}
    inc=np.linspace(0,90,181);theta=np.deg2rad(inc)
    sigma=-85.984+99*np.exp(-.628*theta)+8.189*np.cos(3.414*theta-3.142)
    request={'name':'Original DEM/config migration comparison - first CPI',
        'input_provenance':['Original complex-motion YAML unchanged in waveform and sampled trajectory. '+str(DEM.resolve()),
            'Native 12.5 m DEM samples cropped only to all-pulse illuminated support plus halo; no height scaling.',
            'Original L-band formula is transferred as an assumed angle table, not independently measured material data.',
            'New triangles and diffuse random phases differ from original grid-node coherent phases.'],
        'waveform':{'kind':'pulsed_lfm','carrier_hz':1e9,'bandwidth_hz':5e6,'sample_rate_hz':1e7,
                    'pulse_width_s':20e-6,'slot_period_s':33.3e-6,'adc_start_s':0,'samples':333,'loops':128},
        'platform':{'position_ned_m':d['position'][0].tolist(),'velocity_ned_mps':d['velocity'][0].tolist(),
            'boresight_ned':d['boresight'][0].tolist(),'beam_half_angle_deg':10,
            'trajectory':{'time_s':d['time_s'].tolist(),'position_ned_m':d['position'].tolist(),
                'velocity_ned_mps':d['velocity'].tolist(),'quaternion_xyzw':d['quaternion'].tolist(),
                'provenance':'Exact original pulse states/pointing; Hermite/SLERP between samples; original complex-motion generator not corrected in this comparison'}},
        'array':{'layout':'siso','num_tx':1,'num_rx':1},
        'radiation':{'tx_element':pattern,'rx_element':pattern,'replace_legacy_cone':False},
        'scene':{'seed':42,'amplitude_reference_range_m':5000,'max_scatterers':131072,
            'terrain':{'id':'original_dem','kind':'dem','max_grid_vertices':65536,
                'north_m':d['north'].tolist(),'east_m':d['east'].tolist(),'elevation_up_m':d['elevation'].tolist(),
                'relative_reflectivity':1.,'scattering':'diffuse',
                'land_scattering':{'polarization':'hh','land_cover':'legacy_unspecified',
                    'tables':[{'land_cover':'legacy_unspecified','polarization':'hh','frequencies_hz':[.99e9,1.01e9],
                               'incidence_deg':inc.tolist(),'sigma0_db':[sigma.tolist(),sigma.tolist()],
                               'evidence':'assumed','source':'Unmodified L-band HH empirical expression from original echo.py; original land-class applicability unverified'}]}}},
        'processing':{'range_limits_m':[0,4980],'range_window':'rectangular','doppler_window':'rectangular',
            'azimuth_deg':[0],'elevation_deg':[0],'save_adc':True},
        'sequence':{'frame_count':1,'frame_interval_s':.05}}
    dump(out/'new_config.json',request)
    session=SimulationSession(Simulation.model_validate(request))
    session.save_inputs(out/'new')
    dump(out/'new_estimate.json',session.estimate)
    print(json.dumps({'scatterers':len(session.scene.amplitudes),'status':'new frame starting'}),flush=True)
    result=session.frame(0);session.save_frame(out/'new',result)
    np.savez_compressed(out/'comparison_products.npz',new_rd=result.products.rd_channels[:,:,0,0],
        new_range=result.products.range_m,new_velocity=result.products.velocity_mps)
    print('new frame complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['export','compare']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.output)
