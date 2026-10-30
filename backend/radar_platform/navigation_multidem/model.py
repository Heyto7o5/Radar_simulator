"""An independent CNN; only RDA-derived tensors enter forward()."""
import numpy as np
import torch
from scipy.ndimage import zoom
from torch import nn

ARCHITECTURE = 'MultiDEM_RDA_CNN_v1'
FEATURE_SHAPE = (2, 16, 96, 21, 21)
RDA_SHAPE = (2, 150, 64, 21, 21)


def encode_rda(magnitude):
    """Fixed sampling/cadence is a dataset contract, never a per-case input."""
    a = np.asarray(magnitude, dtype=np.float32)
    if a.ndim != 5 or a.shape[0] != 2 or min(a.shape[1:]) < 2:
        raise ValueError('two numeric R-D-elevation-azimuth volumes required')
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('finite nonnegative RDA magnitudes required')
    # Divide before squaring to avoid overflow and make global gain irrelevant.
    maximum = a.max(axis=(1, 2, 3, 4), keepdims=True)
    if np.any(maximum <= 0):
        raise ValueError('empty radar frame')
    power = (a / maximum)**2
    features = np.log1p(power / power.mean(axis=(1, 2, 3, 4), keepdims=True))
    features = features.transpose(0, 2, 1, 3, 4)
    features = zoom(features, np.array(FEATURE_SHAPE) / features.shape, order=1, prefilter=False)
    return features.astype(np.float32)


class MultiDEMRadarNet(nn.Module):
    def __init__(self):
        super().__init__()
        # These are fitted on training labels only. No geometry or roof buffer.
        self.register_buffer('target_mean', torch.zeros(6))
        self.register_buffer('target_std', torch.ones(6))
        self.encoder = nn.Sequential(
            nn.Conv3d(16, 12, 3, stride=2, padding=1), nn.GroupNorm(4, 12), nn.SiLU(),
            nn.Conv3d(12, 24, 3, stride=2, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv3d(24, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Flatten(), nn.Linear(32*12*3*3, 128), nn.SiLU())
        self.temporal = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Dropout(.1), nn.Linear(128, 13))

    def forward(self, rda_features):
        if rda_features.ndim != 6 or tuple(rda_features.shape[1:]) != FEATURE_SHAPE:
            raise ValueError('fixed two-frame RDA feature shape required')
        batch = rda_features.shape[0]
        encoded = self.encoder(rda_features.reshape(batch*2, *FEATURE_SHAPE[1:]))
        raw = self.temporal(encoded.reshape(batch, 256))
        state = self.target_mean + self.target_std*raw[:, :6]
        scale = self.target_std*torch.exp(raw[:, 6:12].clamp(-5, 3))
        return state, scale, raw[:, 12]


class RDAEstimator:
    """Online adapter has no scene, command, pose, axis or timestamp arguments."""
    def __init__(self, model):
        self.model = model.eval()

    @torch.inference_mode()
    def predict(self, rda_magnitude):
        if np.shape(rda_magnitude) != RDA_SHAPE:
            raise ValueError('RDA sampling shape differs from the fixed acquisition contract')
        # Match the offline float16 cache exactly at the inference boundary.
        features = torch.from_numpy(encode_rda(rda_magnitude).astype(np.float16).astype(np.float32))[None]
        device = next(self.model.parameters()).device
        state, scale, visibility = self.model(features.to(device))
        return {'state_body': state[0].cpu().numpy(),
                'predicted_scale': scale[0].cpu().numpy(),
                'visibility_probability': float(visibility.sigmoid()[0].cpu())}
