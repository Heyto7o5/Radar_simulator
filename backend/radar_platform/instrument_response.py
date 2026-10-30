"""Low-dimensional frequency-response calibration from a reference target.

Six real coefficients per channel: cubic log-magnitude and linear phase.
The model cannot copy thousands of measured complex frequency samples.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SmoothInstrumentResponse:
    amplitude_db_coefficients: tuple[float,...]
    phase_rad_coefficients: tuple[float,...]
    reference_hz: float=5e9
    scale_hz: float=1e9

    def evaluate(self,frequencies_hz):
        x=(np.asarray(frequencies_hz)-self.reference_hz)/self.scale_hz
        amp=np.polyval(self.amplitude_db_coefficients,x)
        phase=np.polyval(self.phase_rad_coefficients,x)
        return 10**(amp/20)*np.exp(1j*phase)

    @classmethod
    def fit(cls,frequencies_hz,measured,simulated,calibration_mask):
        f=np.asarray(frequencies_hz);m=np.asarray(measured);s=np.asarray(simulated);mask=np.asarray(calibration_mask,dtype=bool)
        if f.shape!=m.shape or m.shape!=s.shape or mask.shape!=f.shape or mask.sum()<12:raise ValueError('invalid response calibration dimensions')
        ratio=m/s
        if not np.isfinite(ratio).all() or np.any(abs(ratio[mask])<=1e-20):raise ValueError('invalid reference response')
        x=(f-5e9)/1e9
        weight=np.sqrt(abs(m[mask])/abs(m[mask]).max())
        amp=np.polyfit(x[mask],20*np.log10(abs(ratio[mask])),3,w=weight)
        phase=np.polyfit(x[mask],np.unwrap(np.angle(ratio))[mask],1,w=weight)
        return cls(tuple(map(float,amp)),tuple(map(float,phase)))
