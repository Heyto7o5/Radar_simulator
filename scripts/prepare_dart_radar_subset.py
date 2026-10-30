#!/usr/bin/env python3
"""Read a bounded prefix of the official compressed radar member, never fake gaps."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
from remote_zip import HTTPRangeFile

ROOT = Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')


class PrefixFile(io.RawIOBase):
    def __init__(self, source, size, budget=320 * 1024**2):
        self.source, self.size, self.budget = source, size, budget
        self.pos = 0
        self.data = bytearray()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2): raise ValueError('invalid whence')
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if pos < 0: raise ValueError('negative seek')
        self.pos = pos
        return pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0: return b''
        end = self.pos + n
        if end > self.budget: raise ValueError('HDF5 access exceeds decompressed prefix budget')
        while len(self.data) < end:
            chunk = self.source.read(min(8 * 1024**2, self.budget - len(self.data)))
            if not chunk: raise IOError('truncated ZIP member')
            self.data.extend(chunk)
        result = bytes(self.data[self.pos:end])
        self.pos = end
        return result

    def readinto(self, b):
        view = memoryview(b).cast('B')
        data = self.read(len(view))
        view[:len(data)] = data
        return len(data)


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    entry = manifest['archive']
    report = {'source': entry['url'], 'full_archive_checksum_verified': False,
              'member_crc_verified': False, 'representation': 'processed radar bins, not raw ADC',
              'physical_power_units_verified': False,
              'alignment_source': 'https://github.com/WiseLabCMU/dart/blob/main/tools/dataset.py',
              'selection': 'first three valid trajectory entries; independent of radar values'}
    with h5py.File(ROOT / 'extracted/lab-1/trajectory.h5') as trajectory:
        mask, valid = trajectory['mask'][:], trajectory['valid'][:]
        if int(mask.sum()) != len(valid): raise ValueError('mask/trajectory length mismatch')
        aligned = np.flatnonzero(valid)[:3]
        raw = np.flatnonzero(mask)[aligned]
        report.update(aligned_indices=aligned.tolist(), raw_indices=raw.tolist(),
                      valid_count=int(valid.sum()), aligned_count=len(valid))
    prefix = None
    try:
        with zipfile.ZipFile(HTTPRangeFile(entry['url'], entry['bytes'])) as archive:
            info = archive.getinfo('lab-1/radar.h5')
            with archive.open(info) as member:
                prefix = PrefixFile(member, info.file_size)
                with h5py.File(prefix, 'r') as radar:
                    dataset = radar['rad']
                    report.update(shape=list(dataset.shape), dtype=str(dataset.dtype),
                                  chunks=dataset.chunks)
                    print(json.dumps(report), flush=True)
                    if dataset.shape[0] != len(mask): raise ValueError('radar/mask mismatch')
                    values = dataset[raw]
                    if not np.isfinite(values).all(): raise ValueError('non-finite radar subset')
        path = ROOT / 'extracted/lab-1/radar_first_valid_subset.npz'
        if path.exists(): raise FileExistsError('refusing to overwrite acquired subset')
        np.savez_compressed(path, rad=values, raw_indices=raw, aligned_indices=aligned)
        report.update(status='subset_acquired', file=str(path),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      min=float(values.min()), max=float(values.max()))
    except Exception as exc:
        report.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
    report['decompressed_prefix_bytes'] = len(prefix.data) if prefix else 0
    (ROOT / 'radar_subset_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
