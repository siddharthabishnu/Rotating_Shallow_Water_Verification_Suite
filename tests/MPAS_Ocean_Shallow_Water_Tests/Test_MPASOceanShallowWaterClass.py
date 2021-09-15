"""
Name: Test_MPASOceanShallowWaterClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the MPAS-Ocean shallow water class defined in 
../../src/MPAS_Ocean_Shallow_Water/MPASOceanShallowWaterClass.py.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/MPAS_Ocean_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import ExactSolutionsAndSourceTerms as ESST
    import Initialization
    import MPASOceanShallowWaterClass
    
    
def TestMPASOceanShallowWater():
    ProblemType = 'Coastal_Kelvin_Wave'
    ProblemType_EquatorialWave = Initialization.isEquatorialWave(ProblemType)
    ProblemType_FileName = 'CoastalKelvinWave'
    PrintPhaseSpeedOfWaveModes = False
    PrintAmplitudesOfWaveModes = False
    TimeIntegrator = 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
    LF_TR_and_LF_AM3_with_FB_Feedback_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    Generalized_FB_with_AB2_AM3_Step_Type = 'ThirdOrderAccurate_WideStabilityRange'
    Generalized_FB_with_AB3_AM4_Step_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    nCellsX = 100
    nCellsY = 100
    PrintBasicGeometry = False
    FixAngleEdge = True
    PrintOutput = False
    UseAveragedQuantities = False
    MeshDirectory, BaseMeshFileName, MeshFileName = (
    Initialization.SpecifyMeshDirectoryAndMeshFileNames(ProblemType,ProblemType_EquatorialWave))
    myMPASOceanShallowWater = (
    MPASOceanShallowWaterClass.MPASOceanShallowWater(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                                     TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                     Generalized_FB_with_AB2_AM3_Step_Type,
                                                     Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,
                                                     PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,
                                                     FixAngleEdge,PrintOutput,UseAveragedQuantities,CourantNumber=0.5,
                                                     UseCourantNumberToDetermineTimeStep=False))
    myMPASOceanShallowWater.DetermineCoriolisParameterAndBottomDepth()
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nCells = myMPASOceanShallowWater.myMesh.nCells
    myQuadratureOnHexagon = myMPASOceanShallowWater.myMesh.myQuadratureOnHexagon
    HexagonLength = myMPASOceanShallowWater.myMesh.HexagonLength
    iTime = 0
    time = 0.0
    for iEdge in range(0,nEdges):
        xEdge = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
        yEdge = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
        dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
        angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
        myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
        ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,xEdge,yEdge,time,UseAveragedQuantities,
                                          myQuadratureOnEdge,dvEdge,angleEdge))
    for iCell in range(0,nCells):
        x = myMPASOceanShallowWater.myMesh.xCell[iCell]
        y = myMPASOceanShallowWater.myMesh.yCell[iCell]
        myMPASOceanShallowWater.mySolution.uExact[iCell] = (
        ESST.DetermineExactZonalVelocityAtCellCenter(ProblemType,myExactSolutionParameters,x,y,time,
                                                     UseAveragedQuantities,myQuadratureOnHexagon,HexagonLength))
        myMPASOceanShallowWater.mySolution.vExact[iCell] = (
        ESST.DetermineExactMeridionalVelocityAtCellCenter(ProblemType,myExactSolutionParameters,x,y,time,
                                                          UseAveragedQuantities,myQuadratureOnHexagon,HexagonLength))
        myMPASOceanShallowWater.mySolution.sshExact[iCell] = (
        ESST.DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities,
                                            myQuadratureOnHexagon,HexagonLength))
        myMPASOceanShallowWater.mySolution.u[:] = myMPASOceanShallowWater.mySolution.uExact[:]
        myMPASOceanShallowWater.mySolution.v[:] = myMPASOceanShallowWater.mySolution.vExact[:]
        myMPASOceanShallowWater.mySolution.ssh[:] = myMPASOceanShallowWater.mySolution.sshExact[:]
        myMPASOceanShallowWater.mySolution.uError[:] = (myMPASOceanShallowWater.mySolution.u[:] 
                                                        - myMPASOceanShallowWater.mySolution.uExact[:])
        myMPASOceanShallowWater.mySolution.vError[:] = (myMPASOceanShallowWater.mySolution.v[:] 
                                                        - myMPASOceanShallowWater.mySolution.vExact[:])
        myMPASOceanShallowWater.mySolution.sshError[:] = (myMPASOceanShallowWater.mySolution.ssh[:] 
                                                          - myMPASOceanShallowWater.mySolution.sshExact[:])
    myDiagnosticVariablesToCompute = MPASOceanShallowWaterClass.DiagnosticVariablesToCompute()
    myDiagnosticVariablesToCompute.LayerThickness = True
    myDiagnosticVariablesToCompute.LayerThicknessEdge = True
    myDiagnosticVariablesToCompute.RelativeVorticityCell = True
    myDiagnosticVariablesToCompute.DivergenceKineticEnergyCell = True
    myDiagnosticVariablesToCompute.TangentialVelocity = True
    myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityVertex = True
    myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityEdge = True
    myDiagnosticVariablesToCompute.NormalizedRelativeVorticityCell = True
    myMPASOceanShallowWater.DiagnosticSolve(myMPASOceanShallowWater.mySolution.normalVelocity,
                                            myMPASOceanShallowWater.mySolution.ssh,myDiagnosticVariablesToCompute)
    FileName = 'Test_MPASOceanShallowWaterClass_' + ProblemType_FileName + '_%3.3d' %iTime
    MPASOceanShallowWaterClass.WriteStateMPASOceanShallowWater(myMPASOceanShallowWater,FileName)
    FileName += '.tec'
    DisplayTime = '0 Second'
    UseGivenColorBarLimits = False
    UseInterpolatedErrorLimits = True
    ComputeOnlyExactSolution = False
    PlotNumericalSolution = True
    MPASOceanShallowWaterClass.PythonPlotStateMPASOceanShallowWater(myMPASOceanShallowWater,FileName,
                                                                    DisplayTime,UseGivenColorBarLimits,
                                                                    UseInterpolatedErrorLimits,ComputeOnlyExactSolution,
                                                                    PlotNumericalSolution)


do_TestMPASOceanShallowWater = False
if do_TestMPASOceanShallowWater:
    TestMPASOceanShallowWater()