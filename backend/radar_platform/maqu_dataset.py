"""Keysight CITIFILE adapter for original Maqu complex S-parameters."""
from pathlib import Path
import re
import numpy as np


def load_citi(path: str|Path) -> tuple[np.ndarray,dict[str,np.ndarray]]:
    text=Path(path).read_text(encoding='ascii').replace('\r','')
    sections=text.split('CITIFILE')[1:]
    channels={};frequencies=None
    for part in sections:
        match=re.search(r'DATA S\[(\d),(\d)\] RI',part)
        if match is None:raise ValueError('only real/imaginary S-parameter CITIFILE is supported')
        fblock=re.search(r'VAR_LIST_BEGIN\n(.*?)\nVAR_LIST_END',part,re.S)
        data=re.search(r'\nBEGIN\n(.*?)\nEND',part,re.S)
        if fblock is None or data is None:raise ValueError('missing CITIFILE frequency/data block')
        f=np.fromstring(fblock.group(1),sep='\n')
        pairs=np.array([[float(x) for x in line.split(',')] for line in data.group(1).splitlines() if line.strip()])
        if pairs.shape!=(len(f),2) or not np.isfinite(pairs).all():raise ValueError('invalid complex data length/value')
        if frequencies is None:frequencies=f
        elif not np.array_equal(f,frequencies):raise ValueError('polarizations have inconsistent frequencies')
        key=match.group(1)+match.group(2)
        if key in channels:raise ValueError('duplicate S-parameter')
        channels[key]=(pairs[:,0]+1j*pairs[:,1]).astype(np.complex128)
    if frequencies is None or len(channels)!=4 or np.any(np.diff(frequencies)<=0):raise ValueError('expect four frequency-sweep channels')
    return frequencies,channels


def calibration_file(root: Path,date: str,measurement: int) -> Path:
    folder=root/'raw/original/Scatterometer_Calibration_Sky'/f'VNA_data_{date}'
    found=list(folder.glob(f'*-meas{measurement:05d}-f-*.cti'))
    if len(found)!=1:raise ValueError(f'expected one raw frequency file for {date}/{measurement}: {found}')
    return found[0]


def load_background_subtracted(root: Path,date: str,measurement: int,background: int):
    path=calibration_file(root,date,measurement);bp=calibration_file(root,date,background)
    f,s=load_citi(path);bf,b=load_citi(bp)
    if not np.array_equal(f,bf):raise ValueError('background frequency mismatch')
    return f,{k:s[k]-b[k] for k in s},[str(path),str(bp)]
