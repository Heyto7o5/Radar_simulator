"""Optional compiled traversal of the same opaque-triangle BVH geometry.

No ray grouping, coarsening, visibility interpolation, or fast-math is used.
Ray-first traversal changes work counts but uses the reference tolerances.
"""
import numpy as np
try:
    from numba import njit
except ImportError as exc:
    raise ImportError('visibility.bvh_kernel=numba requires the navigation extra') from exc


@njit(cache=True)
def _blocked(origins,endpoints,triangles,lower,upper,left,right,offset,count,indices,epsilon,limit):
    result=np.zeros(len(endpoints),np.bool_)
    aabb_tests=0;triangle_tests=0
    for ray in range(len(endpoints)):
        o=origins[ray];direction=endpoints[ray]-o
        length=np.sqrt(np.sum(direction*direction))
        if length<=2*epsilon:raise ValueError('ray too short for endpoint tolerance')
        edge_epsilon=epsilon/length
        stack=np.empty(len(lower),np.int64);stack[0]=0;top=1
        while top and not result[ray]:
            top-=1;node=stack[top];aabb_tests+=1
            if aabb_tests+triangle_tests>limit:raise ValueError('BVH visibility runtime work budget exceeded')
            tmin=0.;tmax=1.;valid=True
            for axis in range(3):
                if direction[axis]==0.:
                    if o[axis]<lower[node,axis]-1e-9 or o[axis]>upper[node,axis]+1e-9:
                        valid=False;break
                else:
                    low=(lower[node,axis]-1e-9-o[axis])/direction[axis]
                    high=(upper[node,axis]+1e-9-o[axis])/direction[axis]
                    tmin=max(tmin,min(low,high));tmax=min(tmax,max(low,high))
            if not valid or tmax<tmin:continue
            if count[node]==0:
                stack[top]=left[node];stack[top+1]=right[node];top+=2
                continue
            for k in range(offset[node],offset[node]+count[node]):
                triangle_tests+=1
                if aabb_tests+triangle_tests>limit:raise ValueError('BVH visibility runtime work budget exceeded')
                tri=triangles[indices[k]]
                e1=tri[1]-tri[0];e2=tri[2]-tri[0]
                cross=np.cross(direction,e2)
                determinant=np.sum(e1*cross)
                scale=length*np.sqrt(np.sum(e1*e1))*np.sqrt(np.sum(e2*e2))
                if abs(determinant)<=1e-12*scale:continue
                inverse=1./determinant;rel=o-tri[0]
                u=np.sum(rel*cross)*inverse
                q=np.cross(rel,e1)
                v=np.sum(direction*q)*inverse;t=np.sum(e2*q)*inverse
                if u>=-1e-10 and v>=-1e-10 and u+v<=1+1e-10 and t>edge_epsilon and t<1-edge_epsilon:
                    result[ray]=True;break
    return result,aabb_tests,triangle_tests


class CompiledBVH:
    def __init__(self,tree):
        self.triangles=tree.triangles
        self.lower=np.array([x[0] for x in tree.nodes])
        self.upper=np.array([x[1] for x in tree.nodes])
        n=len(tree.nodes)
        self.left=np.full(n,-1,np.int64);self.right=self.left.copy()
        self.offset=np.zeros(n,np.int64);self.count=self.offset.copy();all_indices=[]
        for k,(_,_,indices,left,right) in enumerate(tree.nodes):
            if indices is None:self.left[k]=left;self.right[k]=right
            else:
                self.offset[k]=len(all_indices);self.count[k]=len(indices);all_indices.extend(indices)
        self.indices=np.array(all_indices,np.int64)

    def blocked(self,origins,endpoints,*,endpoint_epsilon_m=1e-4,work=None,work_limit=100_000_000):
        endpoints=np.asarray(endpoints,dtype=float)
        origins=np.broadcast_to(np.asarray(origins,dtype=float),endpoints.shape)
        if endpoints.ndim!=2 or endpoints.shape[1]!=3 or not np.isfinite(endpoints).all() or not np.isfinite(origins).all():
            raise ValueError('finite ray endpoints required')
        if endpoint_epsilon_m<=0:raise ValueError('positive endpoint tolerance required')
        if not len(self.lower):return np.zeros(len(endpoints),dtype=bool)
        work={'aabb_tests':0,'triangle_tests':0} if work is None else work
        blocked,boxes,triangles=_blocked(np.ascontiguousarray(origins),np.ascontiguousarray(endpoints),
            self.triangles,self.lower,self.upper,self.left,self.right,self.offset,self.count,self.indices,
            endpoint_epsilon_m,work_limit-sum(work.values()))
        work['aabb_tests']+=boxes;work['triangle_tests']+=triangles
        return blocked
