"""Bounded frequency/incidence/polarization/land-class NRCS tables.

Tables declare whether they are measured or assumed. Linear interpolation is
in frequency and incidence in dB, never extrapolation. No table is bundled as
a claim about the real DEM's unknown materials.
"""
from typing import Literal
import numpy as np
from pydantic import BaseModel,ConfigDict,Field,model_validator
from scipy.interpolate import RegularGridInterpolator


class BackscatterTable(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    land_cover: str=Field(min_length=1,max_length=100)
    polarization: Literal['vv','hh','hv','vh']
    frequencies_hz: list[float]=Field(min_length=2,max_length=64)
    incidence_deg: list[float]=Field(min_length=2,max_length=181)
    sigma0_db: list[list[float]]
    evidence: Literal['measured','assumed','analytic']
    source: str=Field(min_length=1,max_length=2000)
    area_reference: Literal['local_surface_area']='local_surface_area'

    @model_validator(mode='after')
    def grid(self):
        f,a,z=map(np.asarray,(self.frequencies_hz,self.incidence_deg,self.sigma0_db))
        if not all(np.isfinite(x).all() for x in (f,a,z)) or np.any(np.diff(f)<=0) or f[0]<=0 or np.any(np.diff(a)<=0) or a[0]<0 or a[-1]>90:
            raise ValueError('finite positive increasing frequencies and increasing incidence in [0,90] required')
        if z.shape!=(len(f),len(a)) or np.any(abs(z)>120):
            raise ValueError('sigma0_db must be [frequency,incidence] with magnitude <=120 dB')
        return self


class TabulatedBackscatterSet:
    def __init__(self,tables):
        self.tables={(t.land_cover,t.polarization):t for t in tables}
        if not self.tables or len(self.tables)!=len(tables):raise ValueError('unique nonempty land-cover/polarization tables required')
        self.interpolators={key:RegularGridInterpolator((t.frequencies_hz,t.incidence_deg),np.asarray(t.sigma0_db),bounds_error=True)
                            for key,t in self.tables.items()}

    def predict_db(self,carrier_hz,incidence_deg,polarization,land_cover):
        key=(land_cover,polarization)
        if key not in self.tables:raise ValueError('missing land-cover/polarization table; no class substitution')
        a=np.asarray(incidence_deg,float);t=self.tables[key]
        if not np.isfinite(carrier_hz) or not np.isfinite(a).all():raise ValueError('finite frequency and incidence required')
        if not t.frequencies_hz[0]<=carrier_hz<=t.frequencies_hz[-1]:raise ValueError('frequency extrapolation forbidden')
        if np.any(a<t.incidence_deg[0]) or np.any(a>t.incidence_deg[-1]):raise ValueError('incidence extrapolation forbidden')
        points=np.c_[np.full(a.size,carrier_hz),a.reshape(-1)]
        return self.interpolators[key](points).reshape(a.shape)

    def metadata(self):
        return {'model':'bounded_frequency_incidence_tables','interpolation':'bilinear in frequency and incidence, dB values',
                'area_reference':'local_surface_area','tables':[t.model_dump(mode='json',exclude={'sigma0_db'}) for t in self.tables.values()],
                'contains_assumed_parameters':any(t.evidence!='measured' for t in self.tables.values()),
                'evidence_is_input_provider_declaration_not_independent_audit':True,
                'absolute_radar_power_calibrated':False,'extrapolation_allowed':False}


def compile_land_model(spec):
    if spec.tables:return TabulatedBackscatterSet(spec.tables)
    from .ground_scattering import EmpiricalBackscatterCurve
    return EmpiricalBackscatterCurve(tuple(spec.observations),spec.mode)
