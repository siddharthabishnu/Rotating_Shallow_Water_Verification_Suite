"""
Name: Test_SolutionClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the MPAS-Ocean shallow water solution class defined in 
../../src/MPAS_Ocean_Shallow_Water/SolutionClass.py.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/MPAS_Ocean_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import SolutionClass
    
    
def TestSolution():
    # The following values of nCells, nEdges, and nVertices represent the number of cells, edges, and vertices in a
    # planar hexagonal MPAS-Ocean mesh with non-periodic zonal and periodic meridional boundaries.
    nCells = 10000
    nEdges = 30200
    nVertices = 20200
    TimeIntegrator = 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
    mySolution = SolutionClass.Solution(nCells,nEdges,nVertices,TimeIntegrator)
    

do_TestSolution = False
if do_TestSolution:
    TestSolution()