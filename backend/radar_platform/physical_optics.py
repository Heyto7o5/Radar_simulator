"""Scalar physical-optics surface integral and explicit range gating.

Unlike the image-ray baseline, finite-area patches contribute a coherent
field even away from one exact specular path. This approximates illuminated
PEC surfaces, not edge currents/vector diffraction or absolute hardware gain.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from .models import C0
from .surface_scene import TriangleMesh


def rectangular_plate(range_m: float,width_m: float,height_m: float,rotation_deg: float=0) -> TriangleMesh:
    if min(range_m,width_m,height_m)<=0:raise ValueError('positive plate dimensions/range required')
    yz=np.array([[-width_m/2,-height_m/2],[width_m/2,-height_m/2],[width_m/2,height_m/2],[-width_m/2,height_m/2]])
    a=np.deg2rad(rotation_deg);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    yz=yz@rot.T
    vertices=np.c_[np.full(4,range_m),yz]
    return TriangleMesh(vertices,np.array([[0,1,2],[0,2,3]]),'PEC rectangular plate')


def triangle_quadrature(mesh: TriangleMesh,order: int=24):
    if not 2<=order<=96:raise ValueError('quadrature order 2..96 required')
    q,w=leggauss(order);q=(q+1)/2;w=w/2
    u,v=np.meshgrid(q,q,indexing='ij');wu,wv=np.meshgrid(w,w,indexing='ij')
    points=[];weights=[];normals=[]
    for t,n,area in zip(mesh.triangles,mesh.normals,mesh.areas_m2):
        x=t[0]+u[...,None]*(t[1]-t[0])+(1-u[...,None])*v[...,None]*(t[2]-t[0])
        weight=(2*area*(1-u)*wu*wv).ravel()
        points.append(x.reshape(-1,3));weights.append(weight);normals.append(np.tile(n,(order*order,1)))
    return np.concatenate(points),np.concatenate(weights),np.concatenate(normals)


def simulate_po_surface(mesh: TriangleMesh,frequencies_hz: np.ndarray,tx_ned_m=(0.,-.2,0.),rx_ned_m=(0.,.2,0.),order=24) -> np.ndarray:
    f=np.asarray(frequencies_hz,dtype=float)
    if f.ndim!=1 or not np.isfinite(f).all() or np.any(f<=0):raise ValueError('positive frequencies required')
    p,area,n=triangle_quadrature(mesh,order)
    at=np.asarray(tx_ned_m)-p;ar=np.asarray(rx_ned_m)-p
    rt=np.linalg.norm(at,axis=1);rr=np.linalg.norm(ar,axis=1)
    if np.any(rt<1e-9) or np.any(rr<1e-9):raise ValueError('antenna intersects surface')
    illumination=.5*(abs(np.sum(n*at/rt[:,None],axis=1))+abs(np.sum(n*ar/rr[:,None],axis=1)))
    weights=area*illumination/(rt*rr)
    path=rt+rr;output=np.zeros(len(f),dtype=np.complex128)
    for start in range(0,len(f),128):
        k=2*np.pi*f[start:start+128]/C0
        output[start:start+128]=1j*k/(2*np.pi)*(np.exp(-1j*k[:,None]*path)@weights)
    return output


def sfcw_range_profile(signal,frequencies_hz,nfft=16384):
    f=np.asarray(frequencies_hz);s=np.asarray(signal)
    if len(f)<2 or s.shape[-1]!=len(f) or nfft<len(f) or not np.allclose(np.diff(f),np.diff(f)[0],rtol=1e-8):raise ValueError('uniform frequency sweep and sufficient nfft required')
    w=np.hanning(len(f));r=np.arange(nfft)*C0/(2*nfft*(f[1]-f[0]))
    return np.fft.ifft(s*w,nfft,axis=-1)*nfft/w.sum(),r


def range_gate(signal,frequencies_hz,center_m,half_width_m=.5,taper_m=.15):
    """Gate without changing the spectral axis. Center is fixed by calibration geometry."""
    f=np.asarray(frequencies_hz);s=np.asarray(signal)
    if len(f)<2 or not np.allclose(np.diff(f),np.diff(f)[0],rtol=1e-8):raise ValueError('uniform frequencies required')
    if not 0<taper_m<half_width_m:raise ValueError('invalid range gate taper')
    n=len(f);r=np.arange(n)*C0/(2*n*(f[1]-f[0]));d=abs(r-center_m)
    gate=np.zeros(n);gate[d<=half_width_m-taper_m]=1
    m=(d>half_width_m-taper_m)&(d<half_width_m)
    gate[m]=.5*(1+np.cos(np.pi*(d[m]-(half_width_m-taper_m))/taper_m))
    return np.fft.fft(np.fft.ifft(s,axis=-1)*gate,axis=-1)
