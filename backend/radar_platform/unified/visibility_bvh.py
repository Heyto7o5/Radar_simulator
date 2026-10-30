"""Balanced triangle BVH for static visibility; exact leaf intersection is shared."""
import numpy as np


class TriangleBVH:
    def __init__(self,triangles,leaf_size=12):
        self.triangles=np.asarray(triangles,dtype=float)
        if self.triangles.ndim!=3 or self.triangles.shape[1:]!=(3,3) or not np.isfinite(self.triangles).all():
            raise ValueError('finite [triangle,3,3] geometry required')
        self.nodes=[]
        self.minimum=self.triangles.min(axis=1) if len(triangles) else np.empty((0,3))
        self.maximum=self.triangles.max(axis=1) if len(triangles) else np.empty((0,3))
        centers=(self.minimum+self.maximum)/2
        def build(indices):
            node=len(self.nodes);lo=self.minimum[indices].min(axis=0);hi=self.maximum[indices].max(axis=0)
            self.nodes.append(None)
            if len(indices)<=leaf_size:self.nodes[node]=(lo,hi,indices,None,None)
            else:
                axis=int(np.argmax(np.ptp(centers[indices],axis=0)))
                order=indices[np.argsort(centers[indices,axis],kind='stable')];mid=len(order)//2
                left,right=build(order[:mid]),build(order[mid:])
                self.nodes[node]=(lo,hi,None,left,right)
            return node
        self.root=build(np.arange(len(triangles))) if len(triangles) else None

    def blocked(self,origins,endpoints,*,endpoint_epsilon_m=1e-4,work=None,work_limit=100_000_000):
        from .visibility import blocked_segments
        endpoints=np.asarray(endpoints,dtype=float);origins=np.broadcast_to(np.asarray(origins,dtype=float),endpoints.shape)
        if endpoints.ndim!=2 or endpoints.shape[1]!=3 or not np.isfinite(endpoints).all() or not np.isfinite(origins).all():
            raise ValueError('finite ray endpoints required')
        work={'aabb_tests':0,'triangle_tests':0} if work is None else work
        result=np.zeros(len(endpoints),dtype=bool)
        if self.root is None:return result
        directions=endpoints-origins;stack=[(self.root,np.arange(len(endpoints)))]
        while stack:
            node,indices=stack.pop();indices=indices[~result[indices]]
            if not len(indices):continue
            work['aabb_tests']+=len(indices)
            if sum(work.values())>work_limit:raise ValueError('BVH visibility runtime work budget exceeded')
            lo,hi,tri,left,right=self.nodes[node]
            tmin=np.zeros(len(indices));tmax=np.ones(len(indices));valid=np.ones(len(indices),bool)
            for axis in range(3):
                o=origins[indices,axis];d=directions[indices,axis];moving=d!=0
                low=np.divide(lo[axis]-1e-9-o,d,out=np.zeros_like(d),where=moving)
                high=np.divide(hi[axis]+1e-9-o,d,out=np.ones_like(d),where=moving)
                valid&=moving|((o>=lo[axis]-1e-9)&(o<=hi[axis]+1e-9))
                tmin=np.maximum(tmin,np.minimum(low,high));tmax=np.minimum(tmax,np.maximum(low,high))
            selected=indices[valid&(tmax>=tmin)]
            if not len(selected):continue
            if tri is not None:
                work['triangle_tests']+=len(selected)*len(tri)
                if sum(work.values())>work_limit:raise ValueError('BVH visibility runtime work budget exceeded')
                result[selected]=blocked_segments(origins[selected],endpoints[selected],self.triangles[tri],endpoint_epsilon_m=endpoint_epsilon_m)
            else:stack.extend([(left,selected),(right,selected)])
        return result
