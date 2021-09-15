"""
Name: Test_DGSolution2DClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the solution class for two-dimensional discontinuous Galerkin spectral 
element methods defined in ../../src/DGSEM_Rotating_Shallow_Water/DGSolution2DClass.py.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import DGSolution2DClass
    
    
def TestDGSolution2D():
    nEquations = 3
    nXi = 10
    nEta = 10
    TimeIntegrator = 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
    myDGSolution2D = DGSolution2DClass.DGSolution2D(nEquations,nXi,nEta,TimeIntegrator)
    

do_TestDGSolution2D = False
if do_TestDGSolution2D:
    TestDGSolution2D()