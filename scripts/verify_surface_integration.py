#!/usr/bin/env python3
"""Replay an exported geometry request through the actual platform API."""
import argparse,json,os,tempfile
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from radar_platform.api import app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forward',type=Path,required=True)
    args=p.parse_args()
    request=json.loads((args.forward/'api_request.json').read_text())
    with tempfile.TemporaryDirectory(prefix='radar-surface-api-') as directory:
        original=os.environ.get('RADAR_RUNTIME_ROOT')
        os.environ['RADAR_RUNTIME_ROOT']=directory
        try:
            with TestClient(app) as client:
                response=client.post('/api/v1/surface-simulations',json=request)
                response.raise_for_status();meta=response.json()
                binary=client.get(meta['spectrum_url']);binary.raise_for_status()
                signal=np.frombuffer(binary.content,dtype='<c8').reshape(meta['shape'])
                with np.load(args.forward/'forward.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(signal,z['signal'])
                print(json.dumps({'API_status':response.status_code,'binary_bytes':len(binary.content),
                                  'shape':meta['shape'],'max_absolute_difference':0.0,'matches_standalone':True}))
        finally:
            if original is None:os.environ.pop('RADAR_RUNTIME_ROOT',None)
            else:os.environ['RADAR_RUNTIME_ROOT']=original


if __name__=='__main__':main()
