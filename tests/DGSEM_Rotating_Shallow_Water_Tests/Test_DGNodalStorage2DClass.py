"""
Name: Test_DGNodalStorage2DClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the two-dimensional Discontinuous Galerkin (DG) nodal storage class 
defined in ../../src/DGSEM_Rotating_Shallow_Water/DGNodalStorage2DClass.py.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import DGNodalStorage2DClass as DGNS2D
    

def TestDGNodalStorage2D():
    nXi = 10
    nEta = 10
    myDGNodalStorage2D = DGNS2D.DGNodalStorage2D(nXi,nEta)
    
    
do_TestDGNodalStorage2D = False
if do_TestDGNodalStorage2D:
    TestDGNodalStorage2D()