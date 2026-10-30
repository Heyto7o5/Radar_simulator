"""CPI bounds that cover both equivalent floating-point evaluation orders."""
import numpy as np
from .acquisition import cpi_duration


def covered_cpi_end(config,start_s):
    w,a=config.waveform,config.array
    period=a.loop_period_s or a.num_tx*w.slot_period_s
    offsets=a.tx_offsets_s or list(np.arange(a.num_tx)*w.slot_period_s)
    # Simulation schema expands the absolute timestamp; applicability/acquisition
    # add an already-computed duration. At t=32s these can differ by one ULP.
    expanded=start_s+(w.loops-1)*period+max(offsets)+w.adc_start_s+w.samples/w.sample_rate_hz
    duration_form=start_s+cpi_duration(config)
    return float(max(expanded,duration_form))
