"""Compact real/simulated/difference panels from existing frozen outputs."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project=Path(__file__).resolve().parents[1]
report=project/'docs/validation_diagnostics/2026-10-15/dart/batch_v2_world_geometry'
data=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab/processed/batch_v2_world_geometry/evaluation')
reference=json.loads((report/'frozen_calibration.json').read_text())['reference']
rows=list(csv.DictReader((report/'per_frame_metrics.csv').open()))
for index in [32,72]:
    with np.load(data/f'f{index:04d}.npz',allow_pickle=False) as f:
        real=10*np.log10(np.maximum(f['real_power']/reference,1e-6))
        simulated=10*np.log10(np.maximum(f['expanded_colored_refitted']/reference,1e-6))
        ranges=f['range_m'];velocity=f['velocity_mps']
    fig,axes=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    for i,(ax,values,title) in enumerate(zip(axes,[real,simulated,simulated-real],['Measured RD','Our simulated RD','Simulation minus measured'])):
        image=ax.pcolormesh(velocity,ranges,values,shading='auto',cmap='RdBu_r' if i==2 else 'viridis',vmin=-30 if i==2 else -60,vmax=30 if i==2 else 10)
        ax.set(title=title,xlabel='Closing velocity (m/s)',ylabel='Range (m)')
        fig.colorbar(image,ax=ax,label='Difference (dB)' if i==2 else 'Power (dB, same reference)')
    fig.savefig(report/'images'/f'direct_compare_{index:04d}.png',dpi=150)
    plt.close(fig)
    row=next(r for r in rows if int(r['valid_frame_index'])==index and r['model']=='expanded_colored_refitted' and r['seed']=='11')
    print(json.dumps(row))
