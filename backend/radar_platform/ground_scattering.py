"""Bounded input bridge for measured, footprint-averaged land backscatter.

This is NOT a DEM-to-RD solver. It prevents silently extrapolating a measured
frequency/incidence footprint to a different band or a near-nadir scene.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


@dataclass(frozen=True)
class Sigma0Observation:
    source: str
    frequency_min_hz: float
    frequency_max_hz: float
    polarization: str
    incidence_center_deg: float
    incidence_min_deg: float
    incidence_max_deg: float
    sigma0_db: float
    upper_uncertainty_db: float
    lower_uncertainty_db: float
    land_cover: str='alpine_meadow'

    def require_coverage(self,carrier_hz,incidence_deg,polarization,land_cover):
        if polarization!=self.polarization or land_cover!=self.land_cover:
            raise ValueError('polarization/land-cover mismatch; no transfer model has been validated')
        if not self.frequency_min_hz<=carrier_hz<=self.frequency_max_hz:
            raise ValueError('frequency outside measured coverage; extrapolation forbidden')
        if not self.incidence_min_deg<=incidence_deg<=self.incidence_max_deg:
            raise ValueError('incidence outside measured footprint; near-nadir extrapolation forbidden')

    def effective_patch_rcs_m2(self,area_m2):
        if not np.isfinite(area_m2) or area_m2<=0:raise ValueError('positive surface area required')
        return area_m2*10**(self.sigma0_db/10)


def load_maqu_sigma0(path):
    path=Path(path)
    match=re.fullmatch(r'sigma0_(vv|hh|vh|hv)_alpha0_azip00_(\d+)-(\d+)\.dat',path.name)
    if not match:raise ValueError('expected official azip00 sigma0 and incidence coverage table')
    data=np.loadtxt(path,skiprows=1,ndmin=2)
    if data.shape[1]!=6:raise ValueError('six columns expected: angle,+,-,sigma0,+,-')
    rows=[]
    for center,upper,lower,sigma,su,sl in data:
        if not np.isfinite([center,upper,lower,sigma,su,sl]).all():continue
        if not (lower<=0<=upper and sl<=0<=su):raise ValueError('uncertainty columns do not match source conventions')
        rows.append(Sigma0Observation(str(path),float(match[2])*1e6,float(match[3])*1e6,match[1],
                    float(center),float(center+lower),float(center+upper),float(sigma),float(su),float(sl)))
    return rows


@dataclass(frozen=True)
class EmpiricalBackscatterCurve:
    """Bounded empirical land model, not an inferred dielectric/roughness law.

    'interpolate' interpolates observed dB values; 'cosine_power' fits
    sigma0 = scale * cos(incidence)**exponent on calibration observations.
    Neither mode extrapolates frequency, polarization, land-cover or angle.
    """
    observations: tuple[Sigma0Observation,...]
    mode: str='interpolate'

    def __post_init__(self):
        if self.mode not in ('interpolate','cosine_power') or len(self.observations)<2:
            raise ValueError('two or more observations and known empirical mode required')
        signature={(x.frequency_min_hz,x.frequency_max_hz,x.polarization,x.land_cover) for x in self.observations}
        centers=[x.incidence_center_deg for x in self.observations]
        if len(signature)!=1 or len(set(centers))!=len(centers):raise ValueError('one band/polarization/land-cover and distinct centers required')
        if not np.isfinite(centers).all() or not np.isfinite([x.sigma0_db for x in self.observations]).all():raise ValueError('finite observations required')
        if min(centers)<0 or max(centers)>=90:raise ValueError('incidence centers in [0,90) required')

    def predict_db(self,carrier_hz,incidence_deg,polarization,land_cover='alpine_meadow'):
        first=self.observations[0];angles=np.asarray(incidence_deg,dtype=float)
        centers=np.array([x.incidence_center_deg for x in self.observations]);values=np.array([x.sigma0_db for x in self.observations])
        if not np.isfinite(carrier_hz) or not first.frequency_min_hz<=carrier_hz<=first.frequency_max_hz:raise ValueError('frequency extrapolation forbidden')
        if polarization!=first.polarization or land_cover!=first.land_cover:raise ValueError('polarization/land-cover extrapolation forbidden')
        if not np.isfinite(angles).all() or np.any(angles<centers.min()) or np.any(angles>centers.max()):raise ValueError('incidence extrapolation forbidden; use measured center coverage')
        if self.mode=='interpolate':
            order=np.argsort(centers);return np.interp(angles,centers[order],values[order])
        x=10*np.log10(np.cos(np.deg2rad(centers)))
        slope=np.sum((x-x.mean())*(values-values.mean()))/np.sum((x-x.mean())**2)
        exponent=float(np.clip(slope,0,20))
        intercept=float(np.mean(values-exponent*x))
        return intercept+exponent*10*np.log10(np.cos(np.deg2rad(angles)))

    def patch_rcs_m2(self,area_m2,carrier_hz,incidence_deg,polarization,land_cover='alpine_meadow'):
        if not np.isfinite(area_m2) or area_m2<=0:raise ValueError('positive finite patch area required')
        return area_m2*10**(self.predict_db(carrier_hz,incidence_deg,polarization,land_cover)/10)

    def metadata(self):
        return {'mode':self.mode,'sources':sorted({x.source for x in self.observations}),
                'frequency_coverage_hz':[self.observations[0].frequency_min_hz,self.observations[0].frequency_max_hz],
                'incidence_center_coverage_deg':[min(x.incidence_center_deg for x in self.observations),max(x.incidence_center_deg for x in self.observations)],
                'polarization':self.observations[0].polarization,'land_cover':self.observations[0].land_cover,
                'material_parameters_inferred':False,'time_variation_predicted':False,
                'scope':'footprint-averaged empirical sigma0; not a local DEM facet truth or a dynamic RD validation'}
