"""
Name: Test_Convergence.py
Author: Sid Bishnu
Details: As the name implies, this script tests the various functions of 
../../src/MPAS_Ocean_Shallow_Water/Convergence.py.
"""


import numpy as np
import os
import time
import sys
sys.path.append(os.path.realpath('../..') + '/src/MPAS_Ocean_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import Initialization
    import Main
    import Convergence


def Test_ConvergenceStudy():
    ConvergenceType = 'SpaceAndTime' # Choose ConvergenceType to be 'SpaceAndTime' or 'Space' or 'Time'.
    ProblemTypes = Convergence.SpecifyProblemTypes(ConvergenceType)
    SingleProblemType = True
    SingleProblemTypeIndex = 0
    if SingleProblemType:
        iProblemTypeLowerLimit = SingleProblemTypeIndex
        iProblemTypeUpperLimit = SingleProblemTypeIndex + 1
    else:
        iProblemTypeLowerLimit = 0
        iProblemTypeUpperLimit = len(ProblemTypes)
    SingleTimeIntegrator = True
    SingleTimeIntegratorIndex = 2
    [TimeIntegrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
     Generalized_FB_with_AB3_AM4_Step_Types] = Convergence.SpecifyTimeIntegrators()
    if SingleTimeIntegrator:
        iTimeIntegratorLowerLimit = SingleTimeIntegratorIndex
        iTimeIntegratorUpperLimit = SingleTimeIntegratorIndex + 1
    else:
        iTimeIntegratorLowerLimit = 0
        iTimeIntegratorUpperLimit = len(TimeIntegrators)
    PrintPhaseSpeedOfWaveModes = False
    PrintAmplitudesOfWaveModes = False
    PrintBasicGeometry = False
    FixAngleEdge = True
    PrintOutput = False
    UseAveragedQuantities = False
    InterpolateExactVelocitiesFromEdgesToCellCenters = True
    if ConvergenceType == 'Space':
        PerformInterpolation = True
    else:
        PerformInterpolation = False
    PerformInterpolationToCoarsestRectilinearMesh = False
    StartTime = time.time()
    for iProblemType in range(iProblemTypeLowerLimit,iProblemTypeUpperLimit):
        ProblemType = ProblemTypes[iProblemType]
        ProblemType_RossbyWave = Initialization.Specify_ProblemType_RossbyWave(ProblemType)
        ProblemType_Title, ProblemType_FileName = Initialization.SpecifyTitleAndFileNamePrefixes(ProblemType)
        if ProblemType_RossbyWave:
            EntityToBeInterpolated = 'Solution'
        else:
            EntityToBeInterpolated = 'Error'
        StartTimePerProblemType = time.time()
        for iTimeIntegrator in range(iTimeIntegratorLowerLimit,iTimeIntegratorUpperLimit):
            StartTimePerProblemTypePerTimeIntegrator = time.time()
            TimeIntegrator = TimeIntegrators[iTimeIntegrator]
            LF_TR_and_LF_AM3_with_FB_Feedback_Type = LF_TR_and_LF_AM3_with_FB_Feedback_Types[iTimeIntegrator]
            Generalized_FB_with_AB2_AM3_Step_Type = Generalized_FB_with_AB2_AM3_Step_Types[iTimeIntegrator]
            Generalized_FB_with_AB3_AM4_Step_Type = Generalized_FB_with_AB3_AM4_Step_Types[iTimeIntegrator]
            TimeIntegratorShortForm = (
            Initialization.DetermineTimeIntegratorShortForm(TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                            Generalized_FB_with_AB2_AM3_Step_Type,
                                                            Generalized_FB_with_AB3_AM4_Step_Type))
            Convergence.ConvergenceStudy(
            ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
            LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
            Generalized_FB_with_AB3_AM4_Step_Type,PrintBasicGeometry,FixAngleEdge,PrintOutput,UseAveragedQuantities,
            InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
            PerformInterpolationToCoarsestRectilinearMesh,EntityToBeInterpolated)
            EndTimePerProblemTypePerTimeIntegrator = time.time()
            ElapsedTimePerProblemTypePerTimeIntegrator = (EndTimePerProblemTypePerTimeIntegrator
                                                          - StartTimePerProblemTypePerTimeIntegrator)
            print('The time taken by the time integrator %s for the %s test case is %s.' 
                  %(TimeIntegratorShortForm,ProblemType_Title,
                    (Main.FormatSimulationTime(ElapsedTimePerProblemTypePerTimeIntegrator,
                                               non_integral_seconds=True)).lower()))
        EndTimePerProblemType = time.time()
        ElapsedTimePerProblemType = EndTimePerProblemType - StartTimePerProblemType
        print('The total time taken by the time integrators for the %s test case is %s.'
              %(ProblemType_Title,(Main.FormatSimulationTime(ElapsedTimePerProblemType,
                                                             non_integral_seconds=True)).lower()))
    EndTime = time.time()
    ElapsedTime = EndTime - StartTime
    print('The total elapsed time is %s.' 
          %(Main.FormatSimulationTime(ElapsedTime,non_integral_seconds=True)).lower())
             
             
do_Test_ConvergenceStudy = False
if do_Test_ConvergenceStudy:
    Test_ConvergenceStudy()
    
    
def Test_PlotConvergenceData():
    ConvergenceType = 'SpaceAndTime' # Choose ConvergenceType to be 'SpaceAndTime' or 'Space' or 'Time'.
    ProblemTypes = Convergence.SpecifyProblemTypes(ConvergenceType)
    SingleProblemType = True
    SingleProblemTypeIndex = 0
    if SingleProblemType:
        iProblemTypeLowerLimit = SingleProblemTypeIndex
        iProblemTypeUpperLimit = SingleProblemTypeIndex + 1
    else:
        iProblemTypeLowerLimit = 0
        iProblemTypeUpperLimit = len(ProblemTypes)
    SingleTimeIntegrator = True
    SingleTimeIntegratorIndex = 2
    PlotOnlySurfaceElevationConvergenceData = True
    PlotAgainstNumberOfCellsInZonalDirection = True
    PlotAgainstNumberOfTimeSteps = True
    UseBestFitLine = False
    set_xticks_manually = False
    # Specify set_xticks_manually as True only if nElementsX consists of powers of 2 e.g. 
    # nElementsX = np.array([32,64,128,256]) and not otherwise e.g. if nElementsX = np.array([100,110,120,130,140,150]).
    for iProblemType in range(iProblemTypeLowerLimit,iProblemTypeUpperLimit):
        ProblemType = ProblemTypes[iProblemType]
        Convergence.PlotConvergenceData(ConvergenceType,ProblemType,SingleTimeIntegrator,SingleTimeIntegratorIndex,
                                        PlotOnlySurfaceElevationConvergenceData,
                                        PlotAgainstNumberOfCellsInZonalDirection,PlotAgainstNumberOfTimeSteps,
                                        UseBestFitLine,set_xticks_manually)
    
    
do_Test_PlotConvergenceData = False
if do_Test_PlotConvergenceData:
    Test_PlotConvergenceData()
    
    
def Test_PlotAllConvergenceData():
    ConvergenceType = 'SpaceAndTime' # Choose ConvergenceType to be 'SpaceAndTime' or 'Space' or 'Time'.
    ProblemTypes = Convergence.SpecifyProblemTypes(ConvergenceType)
    SingleProblemType = True
    SingleProblemTypeIndex = 0
    if SingleProblemType:
        iProblemTypeLowerLimit = SingleProblemTypeIndex
        iProblemTypeUpperLimit = SingleProblemTypeIndex + 1
    else:
        iProblemTypeLowerLimit = 0
        iProblemTypeUpperLimit = len(ProblemTypes)
    PlotOnlySurfaceElevationConvergenceData = True
    PlotAgainstNumberOfCellsInZonalDirection = True
    PlotAgainstNumberOfTimeSteps = True
    UseBestFitLine = False
    set_xticks_manually = False
    # Specify set_xticks_manually as True only if nElementsX consists of powers of 2 e.g. 
    # nElementsX = np.array([32,64,128,256]) and not otherwise e.g. if nElementsX = np.array([100,110,120,130,140,150]).
    for iProblemType in range(iProblemTypeLowerLimit,iProblemTypeUpperLimit):
        ProblemType = ProblemTypes[iProblemType]
        Convergence.PlotAllConvergenceData(ConvergenceType,ProblemType,PlotOnlySurfaceElevationConvergenceData,
                                           PlotAgainstNumberOfCellsInZonalDirection,PlotAgainstNumberOfTimeSteps,
                                           UseBestFitLine,set_xticks_manually)
    
    
do_Test_PlotAllConvergenceData = False
if do_Test_PlotAllConvergenceData:
    Test_PlotAllConvergenceData()