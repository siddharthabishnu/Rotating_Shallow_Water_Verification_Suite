"""
Name: Test_DGSEM2DClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the two-dimensional discontinuous Galerkin spectral element class 
defined in ../../src/DGSEM_Rotating_Shallow_Water/DGSEM2DClass.py.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import ExactSolutionsAndSourceTerms as ESST
    import DGSEM2DClass
    
    
def TestDGSEM2D():
    ProblemType = 'Coastal_Kelvin_Wave'
    ProblemType_FileName = 'CoastalKelvinWave'
    PrintPhaseSpeedOfWaveModes = True
    PrintAmplitudesOfWaveModes = True
    TimeIntegrator = 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
    LF_TR_and_LF_AM3_with_FB_Feedback_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    Generalized_FB_with_AB2_AM3_Step_Type = 'ThirdOrderAccurate_WideStabilityRange'
    Generalized_FB_with_AB3_AM4_Step_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    nElementsX = 5
    nElementsY = 5
    nXi = 10
    nEta = 10
    nXiPlot = 20
    nEtaPlot = 20
    myDGSEM2D = DGSEM2DClass.DGSEM2D(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                     Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,
                                     nEtaPlot)
    nElements = nElementsX*nElementsY
    iTime = 0
    time = 0.0
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                x = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iXi,iEta]
                y = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iXi,iEta]
                myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[0,iXi,iEta] = (
                ESST.DetermineExactZonalVelocity(ProblemType,myDGSEM2D.myNameList.myExactSolutionParameters,x,y,time))
                myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[1,iXi,iEta] = (
                ESST.DetermineExactMeridionalVelocity(ProblemType,myDGSEM2D.myNameList.myExactSolutionParameters,x,y,
                                                      time))
                myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[2,iXi,iEta] = (
                ESST.DetermineExactSurfaceElevation(ProblemType,myDGSEM2D.myNameList.myExactSolutionParameters,x,y,
                                                    time))
        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[:,:,:] = (
        myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[:,:,:])
    FileName = ProblemType_FileName + '_ExactSolution_%3.3d' %iTime
    DGSEM2DClass.WriteInterpolatedStateDGSEM2D(myDGSEM2D,FileName)
    FileName += '.tec'
    DataTypes = ['Structured','Unstructured']
    for iDataType in range(0,len(DataTypes)):
        DataType = DataTypes[iDataType]
        DisplayTime = '0 Second'
        UseGivenColorBarLimits = False
        ComputeOnlyExactSolution = False
        SpecifyDataTypeInPlotFileName = True
        PlotNumericalSolution = True
        DGSEM2DClass.PythonPlotStateDGSEM2D(myDGSEM2D,FileName,DataType,DisplayTime,UseGivenColorBarLimits,
                                            ComputeOnlyExactSolution,SpecifyDataTypeInPlotFileName,
                                            PlotNumericalSolution)


do_TestDGSEM2D = False
if do_TestDGSEM2D:
    TestDGSEM2D()