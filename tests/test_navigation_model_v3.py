import numpy as np
import pytest
torch = pytest.importorskip('torch', reason='install navigation extra for learning tests')

from radar_platform.navigation.model_v3 import RadarNavigationAnchored
from radar_platform.navigation.features_v2 import coordinate_grid


def test_zero_observed_motion_cannot_create_velocity_from_image_embedding():
    torch.set_num_threads(2);torch.manual_seed(18)
    model=RadarNavigationAnchored(coordinate_grid([8.,120.],[-60.,60.],[-60.,60.])).eval()
    model.velocity_correction.copy_(torch.randn_like(model.velocity_correction))
    # Arbitrary spatial/spectral features, including nonzero neural biases.
    features=torch.rand(1,2,20,96,21,21)
    with torch.inference_mode():state,_,_=model(features,torch.zeros(1,2,3))
    assert torch.count_nonzero(state[:,3:])==0
    assert torch.isfinite(state).all()
