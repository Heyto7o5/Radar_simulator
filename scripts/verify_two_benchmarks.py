#!/usr/bin/env python3
"""Check actual acquired benchmark requests through both platform APIs."""
import argparse,json,os,tempfile
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from radar_platform.api import app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ipix',type=Path,required=True)
    p.add_argument('--maqu',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise SystemExit('verification output already exists')
    result={}
    with tempfile.TemporaryDirectory(prefix='radar-two-benchmark-') as temp:
        previous=os.environ.get('RADAR_RUNTIME_ROOT');os.environ['RADAR_RUNTIME_ROOT']=temp
        try:
            with TestClient(app) as client:
                recipe=json.loads((args.ipix/'recipe.json').read_text())
                response=client.post('/api/v1/clutter-simulations',json={'model':recipe['model'],'samples':65536,'seed':42})
                response.raise_for_status();meta=response.json();blob=client.get(meta['iq_url']);blob.raise_for_status()
                iq=np.frombuffer(blob.content,dtype='<c8').reshape(meta['iq_shape'])
                with np.load(args.ipix/'signal_samples.npz',allow_pickle=False) as z:np.testing.assert_array_equal(iq,z['simulated_iq'])
                result['IPIX']={'http_status':200,'shape':meta['iq_shape'],'max_abs_error':0.,'bytes':len(blob.content)}
                request=json.loads((args.maqu/'next_day_36m_horizontal_api_request.json').read_text())
                response=client.post('/api/v1/surface-simulations',json=request)
                response.raise_for_status();meta=response.json();blob=client.get(meta['spectrum_url']);blob.raise_for_status()
                spectrum=np.frombuffer(blob.content,dtype='<c8').reshape(meta['shape'])
                with np.load(args.maqu/'comparison_arrays.npz',allow_pickle=False) as z:
                    reference=z['next_day_36m_horizontal_11_model_raw'].astype(np.complex64)
                np.testing.assert_array_equal(spectrum[0,0],reference)
                result['Maqu']={'http_status':200,'shape':meta['shape'],'max_abs_error':0.,'bytes':len(blob.content),'scope':'physical model before independently fitted instrument transfer'}
        finally:
            if previous is None:os.environ.pop('RADAR_RUNTIME_ROOT',None)
            else:os.environ['RADAR_RUNTIME_ROOT']=previous
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
