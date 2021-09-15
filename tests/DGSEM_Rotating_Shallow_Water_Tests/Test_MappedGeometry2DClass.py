"""
Name: Test_MappedGeometry2DClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the two-dimensional mapped geometry class defined in
../../src/DGSEM_Rotating_Shallow_Water/MappedGeometry2DClass.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import DGNodalStorage2DClass as DGNS2D
    import GeometryBasics2DClass as GB2D
    import MappedGeometry2DClass as MG2D

    
def TestMappedGeometry2D():
    nXi = 14
    nEta = 14
    myDGNodalStorage2D = DGNS2D.DGNodalStorage2D(nXi,nEta)
    BoundaryCurve = np.empty(4,dtype=GB2D.CurveInterpolant2D)
    BoundaryCurve[0] = GB2D.CurveInterpolant2D(np.array([-1.0,1.0]),np.array([0.0,1.0]),np.array([0.0,0.0]))
    BoundaryCurve[1] = GB2D.CurveInterpolant2D(np.array([-1.0,1.0]),np.array([1.0,1.0]),np.array([0.0,1.0]))
    BoundaryCurve[2] = GB2D.CurveInterpolant2D(np.array([-1.0,1.0]),np.array([0.0,1.0]),np.array([1.0,1.0]))
    BoundaryCurve[3] = GB2D.CurveInterpolant2D(np.array([-1.0,1.0]),np.array([0.0,0.0]),np.array([0.0,1.0]))
    # Note that the boundary curves should always be parameterized in the increasing directions of xi and eta.
    myMappedGeometry2D = MG2D.MappedGeometry2D(myDGNodalStorage2D,BoundaryCurve)
    myMappedGeometry2D = MG2D.MappedGeometry2D.ConstructEmptyMappedGeometry2D(myDGNodalStorage2D,nXi,nEta)


do_TestMappedGeometry2D = False
if do_TestMappedGeometry2D:
    TestMappedGeometry2D()