"""CPU scalar geometric-optics baseline for finite planar surfaces.

Method of images finds specular paths on the actual triangles.  This is a
deliberately limited model: spherical path phase/spreading, PEC scalar
reflection and hard facet boundaries; no diffraction, polarization, diffuse
scattering, calibrated RCS or instrument gain.  Failures against data are
reported rather than absorbed into per-channel corrections.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np

from .models import C0
from .surface_scene import TriangleMesh, points_in_triangle


@dataclass(frozen=True)
class SurfaceRadar:
    frequencies_hz: np.ndarray
    tx_ned_m: np.ndarray
    rx_ned_m: np.ndarray
    max_bounces: int = 3

    def __post_init__(self) -> None:
        f=np.asarray(self.frequencies_hz,dtype=float)
        if f.ndim!=1 or len(f)<2 or not np.isfinite(f).all() or np.any(f<=0) or np.any(np.diff(f)<=0):
            raise ValueError("frequencies must be a finite increasing positive vector")
        object.__setattr__(self,"frequencies_hz",f)
        for name in ("tx_ned_m","rx_ned_m"):
            a=np.asarray(getattr(self,name),dtype=float)
            if a.ndim!=2 or a.shape[1]!=3 or not len(a) or not np.isfinite(a).all():
                raise ValueError(f"{name} must be finite [element,3]")
            object.__setattr__(self,name,a)
        if self.max_bounces not in (1,2,3):
            raise ValueError("small-surface baseline supports one to three bounces")
        if len(self.tx_ned_m)*len(self.rx_ned_m)*len(f)>10_000_000:
            raise ValueError("surface-spectrum safety limit: 10 million complex samples")


@dataclass
class SurfaceEcho:
    signal: np.ndarray
    by_bounce: dict[int,np.ndarray]
    path_counts: dict[int,np.ndarray]
    path_length_min_m: float | None
    path_length_max_m: float | None


def specular_paths(mesh: TriangleMesh, tx: np.ndarray, rx: np.ndarray, sequence: tuple[int,...], *, return_points: bool = False):
    """Return valid [path lengths, boolean mask] for paired Tx and Rx rows."""
    tri=mesh.triangles; normals=mesh.normals
    images=[tx]
    for face in sequence:
        n=normals[face]; p=tri[face,0]
        previous=images[-1]
        images.append(previous-2*((previous-p)@n)[:,None]*n)
    last=rx
    valid=np.ones(len(tx),dtype=bool)
    points=[None]*len(sequence)
    for i in range(len(sequence)-1,-1,-1):
        face=sequence[i]; n=normals[face]; p=tri[face,0]
        direction=images[i+1]-last
        denominator=direction@n
        not_parallel=np.abs(denominator)>1e-12
        fraction=np.divide((p-last)@n,denominator,out=np.zeros_like(denominator),where=not_parallel)
        point=last+fraction[:,None]*direction
        valid &= not_parallel & (fraction>1e-9) & (fraction<1-1e-9) & points_in_triangle(point,tri[face])
        points[i]=point
        last=point
    vertices=[tx,*points,rx]
    length=np.zeros(len(tx))
    for i,(start,end) in enumerate(zip(vertices[:-1],vertices[1:])):
        direction=end-start
        length += np.linalg.norm(direction,axis=1)
        # Reject a segment that crosses any other face strictly before its endpoint.
        for face in range(len(tri)):
            n=normals[face]; p=tri[face,0]
            den=direction@n
            t=np.divide((p-start)@n,den,out=np.full(len(tx),-1.),where=np.abs(den)>1e-12)
            hit=start+t[:,None]*direction
            blocked=(t>1e-7)&(t<1-1e-7)&points_in_triangle(hit,tri[face])
            valid &= ~blocked
    # Geometric reflection law and positive path length are built into unfolding.
    valid &= np.isfinite(length)&(length>1e-9)
    if return_points:
        return length,valid,np.stack(points,axis=1)
    return length,valid


def simulate_surface_spectrum(mesh: TriangleMesh, radar: SurfaceRadar) -> SurfaceEcho:
    """Forward model: only geometry + acquisition enter; no measured data."""
    if len(mesh.faces)>32:
        raise ValueError("image-path solver limited to 32 faces; large DEM needs accelerated surface integration")
    sequences=sum(len(mesh.faces)*(len(mesh.faces)-1)**(k-1) for k in range(1,radar.max_bounces+1))
    if sequences>5000:
        raise ValueError("too many reflection sequences for CPU reference solver")
    ntx,nrx,nf=len(radar.tx_ned_m),len(radar.rx_ned_m),len(radar.frequencies_hz)
    tx=np.repeat(radar.tx_ned_m,nrx,axis=0)
    rx=np.tile(radar.rx_ned_m,(ntx,1))
    by_bounce={}; counts={}; min_path=None; max_path=None
    for bounce in range(1,radar.max_bounces+1):
        values=np.zeros((len(tx),nf),dtype=np.complex128)
        count=np.zeros(len(tx),dtype=np.int32)
        seen=set()
        for seq in product(range(len(mesh.faces)),repeat=bounce):
            if any(a==b for a,b in zip(seq[:-1],seq[1:])):
                continue
            lengths,valid,points=specular_paths(mesh,tx,rx,seq,return_points=True)
            indices=[]
            for index in np.flatnonzero(valid):
                # A ray on the shared edge of coplanar mesh triangles is one
                # physical path, not two. Do not deduplicate by length alone.
                key=(int(index),*np.round(points[index].ravel(),10))
                if key not in seen:
                    seen.add(key); indices.append(index)
            indices=np.asarray(indices,dtype=int)
            if not len(indices): continue
            L=lengths[indices]
            min_path=float(L.min()) if min_path is None else min(min_path,float(L.min()))
            max_path=float(L.max()) if max_path is None else max(max_path,float(L.max()))
            # Scalar PEC image field: reflection -1 per bounce; 1/L spreading.
            # Absolute magnitude is uncalibrated; do not infer radar R^-4 from it.
            values[indices] += ((-1.)**bounce/L[:,None])*np.exp(-2j*np.pi*L[:,None]*radar.frequencies_hz[None,:]/C0)
            count[indices]+=1
        by_bounce[bounce]=values.reshape(ntx,nrx,nf).astype(np.complex64)
        counts[bounce]=count.reshape(ntx,nrx)
    total=sum(by_bounce.values()).astype(np.complex64)
    return SurfaceEcho(total,by_bounce,counts,min_path,max_path)


def range_response(signal: np.ndarray, frequencies_hz: np.ndarray, nfft: int = 512) -> tuple[np.ndarray,np.ndarray]:
    """SFCW IFFT. Range is half total bistatic path, not target Cartesian z."""
    f=np.asarray(frequencies_hz,dtype=float); x=np.asarray(signal)
    if x.shape[-1]!=len(f) or nfft<len(f):
        raise ValueError("signal frequency axis / FFT length mismatch")
    if len(f)<2 or not np.all(np.diff(f)>0) or not np.allclose(np.diff(f),np.diff(f)[0],rtol=1e-8):
        raise ValueError("range IFFT requires increasing uniform frequencies")
    w=np.hanning(len(f))
    y=np.fft.ifft(x*w,nfft,axis=-1)*(nfft/w.sum())
    ranges=np.arange(nfft)*C0/(2*nfft*(f[1]-f[0]))
    return y.astype(np.complex64),ranges
