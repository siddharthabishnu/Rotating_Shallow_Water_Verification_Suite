"""
Name: Test_Convergence.py
Author: Sid Bishnu
Details: As the name implies, this script tests the various functions of 
../../src/DGSEM_Rotating_Shallow_Water/Convergence.py.
"""


import numpy as np
import os
import time
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import Initialization
    import Main
    import Convergence
    
    
def Test_ConvergenceStudy():
    ConvergenceType = 'SpaceAndTime' # Choose ConvergenceType to be 'SpaceAndTime' or 'Space' or 'Time'.
    if ConvergenceType == 'SpaceAndTime':
        ProblemTypes = ['Plane_Gaussian_Wave','Coastal_Kelvin_Wave','Inertia_Gravity_Wave','Barotropic_Tide',
                        'NonLinear_Manufactured_Solution']
    else:
        ProblemTypes = ['Plane_Gaussian_Wave','Coastal_Kelvin_Wave','Inertia_Gravity_Wave',
                        'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave',
                        'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave','Barotropic_Tide',
                        'NonLinear_Manufactured_Solution']
    SingleProblemType = True
    SingleProblemTypeIndex = 1
    if SingleProblemType:
        iProblemTypeLowerLimit = SingleProblemTypeIndex
        iProblemTypeUpperLimit = SingleProblemTypeIndex + 1
    else:
        iProblemTypeLowerLimit = 0
        iProblemTypeUpperLimit = len(ProblemTypes)
    SingleTimeIntegrator = True
    SingleTimeIntegratorIndex = 1
    [TimeIntegrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
     Generalized_FB_with_AB3_AM4_Step_Types] = Convergence.SetOfTimeIntegrators()
    if SingleTimeIntegrator:
        iTimeIntegratorLowerLimit = SingleTimeIntegratorIndex
        iTimeIntegratorUpperLimit = SingleTimeIntegratorIndex + 1
    else:
        iTimeIntegratorLowerLimit = 0
        iTimeIntegratorUpperLimit = len(TimeIntegrators)
    PrintPhaseSpeedOfWaveModes = False
    PrintAmplitudesOfWaveModes = False
    if ConvergenceType == 'Space':
        PerformInterpolation = True
    else:
        PerformInterpolation = False
    StartTime = time.time()
    for iProblemType in range(iProblemTypeLowerLimit,iProblemTypeUpperLimit):
        ProblemType = ProblemTypes[iProblemType]
        ProblemType_Title, ProblemType_FileName = Initialization.SpecifyTitleAndFileNamePrefixes(ProblemType) 
        if ((ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave' 
             or ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave')):
            EntityToBeInterpolated = 'Solution'
            ReadFromSELFOutputData = True
        else:
            EntityToBeInterpolated = 'Error'
            ReadFromSELFOutputData = False
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
            Convergence.ConvergenceStudy(ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,
                                         PrintAmplitudesOfWaveModes,TimeIntegrator,
                                         LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                         Generalized_FB_with_AB3_AM4_Step_Type,
                                         PerformInterpolation,EntityToBeInterpolated,ReadFromSELFOutputData)
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
    if ConvergenceType == 'SpaceAndTime':
        ProblemTypes = ['Plane_Gaussian_Wave','Coastal_Kelvin_Wave','Inertia_Gravity_Wave','Barotropic_Tide',
                        'NonLinear_Manufactured_Solution']
    else:
        ProblemTypes = ['Plane_Gaussian_Wave','Coastal_Kelvin_Wave','Inertia_Gravity_Wave',
                        'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave',
                        'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave','Barotropic_Tide',
                        'NonLinear_Manufactured_Solution']
    SingleProblemType = True
    SingleProblemTypeIndex = 1
    if SingleProblemType:
        iProblemTypeLowerLimit = SingleProblemTypeIndex
        iProblemTypeUpperLimit = SingleProblemTypeIndex + 1
    else:
        iProblemTypeLowerLimit = 0
        iProblemTypeUpperLimit = len(ProblemTypes)
    SingleTimeIntegrator = True
    SingleTimeIntegratorIndex = 1
    PlotOnlySurfaceElevationConvergenceData = True
    PlotAgainstNumberOfCellsInZonalDirection = True
    PlotAgainstNumberOfTimeSteps = True
    UseBestFitLine = False
    set_xticks_manually = False
    # Specify set_xticks_manually as True only if nElementsX consists of powers of 2 e.g. 
    # nElementsX = np.array([32,64,128,256]) and not otherwise e.g. if nElementsX = np.array([100,110,120,130,140,150]).
    for iProblemType in range(iProblemTypeLowerLimit,iProblemTypeUpperLimit):
        ProblemType = ProblemTypes[iProblemType]
        if ((ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave' 
             or ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave')):
            ReadFromSELFOutputData = True
        else:
            ReadFromSELFOutputData = False
        Convergence.PlotConvergenceData(ConvergenceType,ProblemType,SingleTimeIntegrator,SingleTimeIntegratorIndex,
                                        PlotOnlySurfaceElevationConvergenceData,
                                        PlotAgainstNumberOfCellsInZonalDirection,PlotAgainstNumberOfTimeSteps,
                                        UseBestFitLine,set_xticks_manually,ReadFromSELFOutputData)
    
    
do_Test_PlotConvergenceData = False
if do_Test_PlotConvergenceData:
    Test_PlotConvergenceData()
    
    
def Test_PlotAllConvergenceData():
    ConvergenceType = 'SpaceAndTime' # Choose ConvergenceType to be 'SpaceAndTime' or 'Space' or 'Time'.
    if ConvergenceType == 'SpaceAndTime':
        ProblemTypes = ['Plane_Gaussian_Wave','Coastal_Kelvin_Wave','Inertia_Gravity_Wave','Barotropic_Tide',
                        'NonLinear_Manufactured_Solution']
    else:
        ProblemTypes = ['Plane_Gaussian_Wave','Coastal_Kelvin_Wave','Inertia_Gravity_Wave',
                        'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave',
                        'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave','Barotropic_Tide',
                        'NonLinear_Manufactured_Solution']
    SingleProblemType = True
    SingleProblemTypeIndex = 1
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
        if ((ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave' 
             or ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave')):
            ReadFromSELFOutputData = True
        else:
            ReadFromSELFOutputData = False
        Convergence.PlotAllConvergenceData(ConvergenceType,ProblemType,PlotOnlySurfaceElevationConvergenceData,
                                           PlotAgainstNumberOfCellsInZonalDirection,PlotAgainstNumberOfTimeSteps,
                                           UseBestFitLine,set_xticks_manually,ReadFromSELFOutputData)
    
    
do_Test_PlotAllConvergenceData = False
if do_Test_PlotAllConvergenceData:
    Test_PlotAllConvergenceData()