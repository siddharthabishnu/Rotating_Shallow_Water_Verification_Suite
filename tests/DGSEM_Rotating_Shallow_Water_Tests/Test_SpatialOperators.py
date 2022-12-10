"""
Name: Test_SpatialOperators.py
Author: Sid Bishnu
Details: As the name implies, this script tests the various spatial operators of DGSEM computed in 
../../src/DGSEM_Rotating_Shallow_Water/SpatialOperators.py against their exact counterparts using smooth 
two-dimensional functions.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import SpatialOperators as SO
    
    
def TestConvergenceStudyOfSpatialOperators():
    nXiMinimum = 2
    nXiMaximum = nXiMinimum + 2
    WriteState = True
    PlotAgainstNumberOfCellsInZonalDirection = True
    UseBestFitLine = True
    set_xticks_manually = False
    ReadFromSELFOutputData = True
    if ReadFromSELFOutputData:
        ReadDivergenceErrorNorm = False
    else:
        ReadDivergenceErrorNorm = True
    for nXi in range(nXiMinimum,nXiMaximum+1):
        if not(ReadFromSELFOutputData):
            SO.ConvergenceStudyOfSpatialOperators(nXi,WriteState)
        SO.PlotConvergenceDataOfSpatialOperators(nXi,PlotAgainstNumberOfCellsInZonalDirection,UseBestFitLine,
                                                 set_xticks_manually,ReadFromSELFOutputData,ReadDivergenceErrorNorm)
    SO.PlotConvergenceDataOfAllSpatialOperators(nXiMinimum,PlotAgainstNumberOfCellsInZonalDirection,UseBestFitLine,
                                                set_xticks_manually,ReadFromSELFOutputData,ReadDivergenceErrorNorm)
    
    
do_TestConvergenceStudyOfSpatialOperators = False
if do_TestConvergenceStudyOfSpatialOperators:
    TestConvergenceStudyOfSpatialOperators()