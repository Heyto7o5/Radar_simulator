import numpy as np
import pytest
pytest.importorskip('numba', reason='install navigation extra for compiled visibility tests')

from radar_platform.unified.visibility import blocked_segments
from radar_platform.unified.visibility_bvh import TriangleBVH
from radar_platform.unified.visibility_bvh_numba import CompiledBVH


def test_compiled_bvh_matches_independent_vectorized_intersections():
    rng=np.random.default_rng(230)
    triangles=rng.uniform(-10,10,(100,3,3))
    origins=rng.uniform(-12,12,(1000,3));ends=rng.uniform(-12,12,(1000,3))
    expected=blocked_segments(origins,ends,triangles)
    actual=CompiledBVH(TriangleBVH(triangles)).blocked(origins,ends)
    np.testing.assert_array_equal(actual,expected)


def test_compiled_endpoint_edges_parallel_and_budget():
    triangles=np.array([[[0,-1,-1],[0,1,-1],[0,0,1]]],float)
    origins=np.array([[-1,0,0],[-1,0,0],[-1,0,0],[0,0,0],[-1,1,0]],float)
    ends=np.array([[1,0,0],[0,0,0],[-.0002,0,0],[0,0,.5],[1,1,0]],float)
    tree=CompiledBVH(TriangleBVH(triangles))
    np.testing.assert_array_equal(tree.blocked(origins,ends),blocked_segments(origins,ends,triangles))
    with pytest.raises(ValueError,match='budget'):tree.blocked(origins,ends,work_limit=1)
