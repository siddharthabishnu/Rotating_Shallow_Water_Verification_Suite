"""
Name: Test_ConvergenceOfSpatialOperators.py
Author: Sid Bishnu
Details: As the name implies, this script tests the various functions of 
../../src/DGSEM_Rotating_Shallow_Water/ConvergenceOfSpatialOperators.py.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import ConvergenceOfSpatialOperators as COSO
    
    
def TestConvergenceStudyOfSpatialOperators():
    WriteState = True
    COSO.ConvergenceStudyOfSpatialOperators(WriteState)
    PlotAgainstNumberOfCellsInZonalDirection = True
    UseBestFitLine = True
    set_xticks_manually = False
    COSO.PlotConvergenceDataOfSpatialOperators(PlotAgainstNumberOfCellsInZonalDirection,UseBestFitLine,
                                               set_xticks_manually)
    
    
do_TestConvergenceStudyOfSpatialOperators = False
if do_TestConvergenceStudyOfSpatialOperators:
    TestConvergenceStudyOfSpatialOperators()