"""Frozen RDA-only inference. No DEM, building geometry or trajectory reads."""
import numpy as np
import torch

from .model import RadarNavigationNet,ARCHITECTURE
from .observation import encode
from .controller import Estimate
from .io import local_path,read_json


class Predictor:
    def __init__(self,model_dir,threads=4):
        root=local_path(model_dir);torch.set_num_threads(threads)
        protocol=read_json(root/'training_protocol.json')
        self.architecture=protocol['architecture']
        self.timing=read_json(root/'acquisition_timing.json') if (root/'acquisition_timing.json').exists() else None
        if read_json(root/'status.json')['status']!='complete':raise ValueError('model training is incomplete')
        with np.load(root/'acquisition_axes.npz',allow_pickle=False) as data:
            self.axes={k:data[k] for k in data.files}
        self.calibration=read_json(root/'uncertainty.json');self.models=[]
        for seed in protocol['seeds']:
            checkpoint=torch.load(root/f'seed_{seed}.pt',map_location='cpu',weights_only=True)
            if checkpoint['architecture']!=self.architecture:raise ValueError('mixed model architectures')
            if self.architecture==ARCHITECTURE:
                model=RadarNavigationNet()
            elif self.architecture=='RadarNavigationHeatmap_v2':
                from .model_v2 import RadarNavigationHeatmap
                model=RadarNavigationHeatmap()
            elif self.architecture=='RadarNavigationHeatmap_v3':
                from .model_v3 import RadarNavigationAnchored
                model=RadarNavigationAnchored()
            else:raise ValueError('unsupported radar architecture')
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            self.models.append(model)

    @torch.inference_mode()
    def predict(self,observation):
        observation.validate()
        for name,axis in self.axes.items():
            if not np.array_equal(getattr(observation,name),axis):
                raise ValueError(f'acquisition axis changed: {name}')
        interval=float(observation.frame_start_s[1]-observation.frame_start_s[0])
        if not np.isclose(interval,.25,atol=1e-8,rtol=0):raise ValueError('model requires 0.25s frame spacing')
        if self.timing is not None and not np.allclose(observation.frame_end_s-observation.frame_start_s,
                self.timing['cpi_duration_s'],atol=1e-8,rtol=0):raise ValueError('model CPI duration differs from observation')
        if self.architecture in ['RadarNavigationHeatmap_v2','RadarNavigationHeatmap_v3']:
            from .features_v2 import encode_v2
            features,baseline=encode_v2(observation)
            auxiliary=torch.from_numpy(baseline[None])
        else:
            features=encode(observation)
            auxiliary=torch.tensor([interval],dtype=torch.float32)
        # Match the explicitly recorded float16 training-feature cache exactly.
        x=torch.from_numpy(features.astype(np.float16).astype(np.float32)[None])
        predictions=[];probabilities=[]
        for model in self.models:
            output=model(x,auxiliary);prediction,logit=output[:2]
            predictions.append(prediction.numpy()[0]);probabilities.append(float(torch.sigmoid(logit)[0]))
        prediction=np.mean(predictions,axis=0)
        spread=float(np.linalg.norm(np.std(predictions,axis=0)[:3]))
        error_scale=(spread+self.calibration['floor_m'])*self.calibration['multiplier']
        probability=float(np.mean(probabilities))
        valid=bool(probability>=self.calibration['valid_probability_threshold'] and error_scale<=self.calibration['maximum_error_scale_m'])
        estimate=Estimate(float(observation.frame_end_s[-1]),prediction[:3],prediction[3:],valid,error_scale,
                          'ok' if valid else 'weak_or_inconsistent_radar_estimate')
        diagnostic=dict(ensemble_state_body=prediction.tolist(),per_seed_state_body=np.asarray(predictions).tolist(),
            valid_probability=probability,error_scale_m=error_scale,ensemble_position_spread_m=spread,
            input='numeric radar magnitudes, axes, timestamps only',label_time_s=estimate.time_s)
        return estimate,diagnostic
