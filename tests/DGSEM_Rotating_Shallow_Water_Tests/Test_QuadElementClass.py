"""
Name: Test_QuadElementClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the quadrilateral element class defined in 
../../src/DGSEM_Rotating_Shallow_Water/QuadElementClass.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import DGNodalStorage2DClass as DGNS2D
    import GeometryBasics2DClass as GB2D
    import QuadElementClass as QE

    
def TestQuadElement():
    NodeIDs = np.array([1,2,3,4])
    ElementIDX = 1
    ElementIDY = 1
    ElementID = 1
    BoundaryCurve = np.empty(4,dtype=GB2D.CurveInterpolant2D)
    BoundaryCurve[0] = GB2D.CurveInterpolant2D(np.array([-1.0,1.0]),np.array([0.0,1.0]),np.array([0.0,0.0]))
    BoundaryCurve[1] = GB2D.CurveInterpolant2D(np.array([-1.0,1.0]),np.array([1.0,1.0]),np.array([0.0,1.0]))
    BoundaryCurve[2] = GB2D.CurveInterpolant2D(np.array([-1.0,1.0]),np.array([0.0,1.0]),np.array([1.0,1.0]))
    BoundaryCurve[3] = GB2D.CurveInterpolant2D(np.array([-1.0,1.0]),np.array([0.0,0.0]),np.array([0.0,1.0]))
    # Note that the boundary curves should always be parameterized in the increasing directions of xi and eta.
    nXi = 10
    nEta = 10
    myDGNodalStorage2D = DGNS2D.DGNodalStorage2D(nXi,nEta)
    myQuadElement = QE.QuadElement(NodeIDs,ElementIDX,ElementIDY,ElementID,BoundaryCurve,myDGNodalStorage2D)
    

do_TestQuadElement = False
if do_TestQuadElement:
    TestQuadElement()