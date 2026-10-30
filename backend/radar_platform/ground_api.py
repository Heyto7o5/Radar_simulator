"""Evaluate bounded empirical ground coefficients, without inventing a full RD."""
from typing import Literal
import numpy as np
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,ConfigDict,Field
from .ground_scattering import Sigma0Observation,EmpiricalBackscatterCurve

router=APIRouter(prefix='/api/v1/scattering',tags=['bounded ground scattering'])

class GroundResponseRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    observations: list[Sigma0Observation]=Field(min_length=2,max_length=128)
    mode: Literal['interpolate','cosine_power']='interpolate'
    carrier_hz: float=Field(gt=0)
    incidence_deg: list[float]=Field(min_length=1,max_length=256)
    polarization: Literal['vv','hh','hv','vh']
    land_cover: str='alpine_meadow'
    patch_area_m2: float=Field(gt=0,le=1e8)

@router.post('/ground-response')
def predict(request:GroundResponseRequest):
    try:
        curve=EmpiricalBackscatterCurve(tuple(request.observations),request.mode)
        sigma=curve.predict_db(request.carrier_hz,request.incidence_deg,request.polarization,request.land_cover)
        rcs=curve.patch_rcs_m2(request.patch_area_m2,request.carrier_hz,request.incidence_deg,request.polarization,request.land_cover)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    return {'is_real':False,'kind':'empirical_coefficient_prediction','incidence_deg':request.incidence_deg,
            'sigma0_db':np.asarray(sigma).tolist(),'effective_patch_rcs_m2':np.asarray(rcs).tolist(),
            'metadata':curve.metadata(),'input_source_authenticity_verified_by_api':False,
            'not_a_dynamic_RD':True,'local_material_parameters_inferred':False}
