"""Independent height-grid LOS reference, without triangle/ray intersection.

Evaluate clearance at grid-line and cell-diagonal crossings. Between these
events the line and piecewise-planar DEM heights are both linear, so extrema
occur at interval endpoints. Near-tangent and out-of-coverage rays are flagged.
"""
import numpy as np


def height_at(north,east,height,n,e):
    north,east,height=map(np.asarray,(north,east,height))
    if np.any(n<north[0]) or np.any(n>north[-1]) or np.any(e<east[0]) or np.any(e>east[-1]):
        raise ValueError('height query outside grid')
    i=np.clip(np.searchsorted(north,n,side='right')-1,0,len(north)-2)
    j=np.clip(np.searchsorted(east,e,side='right')-1,0,len(east)-2)
    u=(n-north[i])/(north[i+1]-north[i]);v=(e-east[j])/(east[j+1]-east[j])
    a=height[j,i];b=height[j,i+1];c=height[j+1,i];d=height[j+1,i+1]
    return np.where(u>=v,a+u*(b-a)+v*(d-b),a+u*(d-c)+v*(c-a))


def reference_los(north,east,height,origin_ned,endpoints_ned,*,endpoint_epsilon_m=1e-4,tangent_tolerance_m=1e-7):
    north,east,height=map(lambda x:np.asarray(x,float),(north,east,height))
    origin=np.asarray(origin_ned,float);endpoints=np.asarray(endpoints_ned,float)
    if height.shape!=(len(east),len(north)) or np.any(np.diff(north)<=0) or np.any(np.diff(east)<=0):
        raise ValueError('increasing grid axes and [east,north] heights required')
    if origin.shape!=(3,) or endpoints.ndim!=2 or endpoints.shape[1]!=3:
        raise ValueError('origin [3], endpoints [N,3] required')
    if not all(np.isfinite(x).all() for x in (north,east,height,origin,endpoints)):
        raise ValueError('finite geometry required')
    result=[]
    for endpoint in endpoints:
        delta=endpoint-origin;length=np.linalg.norm(delta)
        if length<=2*endpoint_epsilon_m:raise ValueError('ray too short')
        if (min(origin[0],endpoint[0])<north[0] or max(origin[0],endpoint[0])>north[-1] or
            min(origin[1],endpoint[1])<east[0] or max(origin[1],endpoint[1])>east[-1]):
            result.append(dict(covered=False,blocked=None,near_tangent=None,min_clearance_m=None));continue
        lo=endpoint_epsilon_m/length;hi=1-lo;parts=[np.array([lo,hi])]
        for axis,grid in [(0,north),(1,east)]:
            if delta[axis]!=0:
                crossings=(grid-origin[axis])/delta[axis]
                parts.append(crossings[(crossings>lo)&(crossings<hi)])
        t=np.unique(np.concatenate(parts));middle=(t[1:]+t[:-1])/2
        n,e=origin[0]+delta[0]*middle,origin[1]+delta[1]*middle
        i=np.clip(np.searchsorted(north,n,side='right')-1,0,len(north)-2)
        j=np.clip(np.searchsorted(east,e,side='right')-1,0,len(east)-2)
        dn=north[i+1]-north[i];de=east[j+1]-east[j]
        slope=delta[0]/dn-delta[1]/de
        diagonal=np.divide((origin[1]-east[j])/de-(origin[0]-north[i])/dn,slope,
                           out=np.full_like(slope,np.nan),where=slope!=0)
        diagonal=diagonal[(diagonal>t[:-1])&(diagonal<t[1:])]
        samples=np.unique(np.r_[t,diagonal]);n=origin[0]+delta[0]*samples;e=origin[1]+delta[1]*samples
        ground=height_at(north,east,height,n,e)
        clearance=-(origin[2]+delta[2]*samples)-ground
        minimum=float(clearance.min())
        result.append(dict(covered=True,blocked=minimum< -tangent_tolerance_m,
                           near_tangent=abs(minimum)<=tangent_tolerance_m,min_clearance_m=minimum))
    return result
