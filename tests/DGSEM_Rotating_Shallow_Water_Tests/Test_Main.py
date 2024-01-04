"""
Name: Test_Main.py
Author: Sid Bishnu
Details: As the name implies, this script tests the various functions of ../../src/DGSEM_Rotating_Shallow_Water/Main.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import Initialization
    import Main
    
    
def Test_FormatSimulationTime():
    times = np.zeros(5)
    times[0] = 3.0*365.0*86400.0 + 3.0*86400.0 + 3.0*3600.0 + 3.0*60.0 + 3.0
    times[1] = 3.0*86400.0 + 3.0*3600.0 + 3.0*60.0 + 3.0
    times[2] = 3.0*3600.0 + 3.0*60.0 + 3.0
    times[3] = 3.0*60.0 + 3.0
    times[4] = 3.0
    for iTime in range(0,len(times)):
        time = times[iTime]
        if iTime != 0:
            print(' ')
        print('The unformatted simulation time is %.1f seconds.'%time)
        SimulationTime = Main.FormatSimulationTime(time,display_time=True)
             

do_Test_FormatSimulationTime = False
if do_Test_FormatSimulationTime:
    Test_FormatSimulationTime()


def Test_DetermineCourantNumberForGivenTimeStepAndCheckItsValue():
    ProblemType = 'Coastal_Kelvin_Wave'
    # Choose ProblemType to be 'Plane_Gaussian_Wave' or 'Coastal_Kelvin_Wave' or 'Inertia_Gravity_Wave' 
    # or 'Manufactured_Planetary_Rossby_Wave' or 'Manufactured_Topographic_Rossby_Wave' or 'Equatorial_Kelvin_Wave' 
    # or 'Equatorial_Yanai_Wave' or 'Equatorial_Rossby_Wave' or 'Equatorial_Inertia_Gravity_Wave' or 'Barotropic_Tide' 
    # or 'Diffusion_Equation' or 'Advection_Diffusion_Equation' or 'NonLinear_Manufactured_Solution' 
    # or 'Viscous_Burgers_Equation'.
    Main.DetermineCourantNumberForGivenTimeStepAndCheckItsValue(ProblemType)
            

do_Test_DetermineCourantNumberForGivenTimeStepAndCheckItsValue = False
if do_Test_DetermineCourantNumberForGivenTimeStepAndCheckItsValue:
    Test_DetermineCourantNumberForGivenTimeStepAndCheckItsValue()


def Test_DetermineNumberOfTimeStepsForSimulation():
    ProblemType = 'Coastal_Kelvin_Wave'
    # Choose ProblemType to be 'Plane_Gaussian_Wave' or 'Coastal_Kelvin_Wave' or 'Inertia_Gravity_Wave' 
    # or 'Manufactured_Planetary_Rossby_Wave' or 'Manufactured_Topographic_Rossby_Wave' or 'Equatorial_Kelvin_Wave' 
    # or 'Equatorial_Yanai_Wave' or 'Equatorial_Rossby_Wave' or 'Equatorial_Inertia_Gravity_Wave' or 'Barotropic_Tide' 
    # or 'Diffusion_Equation' or 'Advection_Diffusion_Equation' or 'NonLinear_Manufactured_Solution' 
    # or 'Viscous_Burgers_Equation'.
    Main.DetermineNumberOfTimeStepsForSimulation(ProblemType)
            

do_Test_DetermineNumberOfTimeStepsForSimulation = False
if do_Test_DetermineNumberOfTimeStepsForSimulation:
    Test_DetermineNumberOfTimeStepsForSimulation()    


def Test_DetermineExactSolutions():
    ProblemType = 'Coastal_Kelvin_Wave'
    # Choose ProblemType to be 'Plane_Gaussian_Wave' or 'Coastal_Kelvin_Wave' or 'Inertia_Gravity_Wave' 
    # or 'Manufactured_Planetary_Rossby_Wave' or 'Manufactured_Topographic_Rossby_Wave' or 'Equatorial_Kelvin_Wave' 
    # or 'Equatorial_Yanai_Wave' or 'Equatorial_Rossby_Wave' or 'Equatorial_Inertia_Gravity_Wave' or 'Barotropic_Tide' 
    # or 'Diffusion_Equation' or 'Advection_Diffusion_Equation' or 'NonLinear_Manufactured_Solution' 
    # or 'Viscous_Burgers_Equation'.
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
    ReadFromSELFOutputData = False
    CheckStateVariableLimits = False
    PlotFigures = True
    Main.DetermineExactSolutions(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                 LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                 Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,nEtaPlot,
                                 ReadFromSELFOutputData,CheckStateVariableLimits,PlotFigures)
    
    
do_Test_DetermineExactSolutions = False
if do_Test_DetermineExactSolutions:
    Test_DetermineExactSolutions()  


def Test_DetermineExactAndNumericalSolutions():
    ProblemType = 'Coastal_Kelvin_Wave'
    # Choose ProblemType to be 'Plane_Gaussian_Wave' or 'Coastal_Kelvin_Wave' or 'Inertia_Gravity_Wave' 
    # or 'Manufactured_Planetary_Rossby_Wave' or 'Manufactured_Topographic_Rossby_Wave' or 'Planetary_Rossby_Wave' 
    # or 'Topographic_Rossby_Wave' or 'Equatorial_Kelvin_Wave' or 'Equatorial_Yanai_Wave' or 'Equatorial_Rossby_Wave' 
    # or 'Equatorial_Inertia_Gravity_Wave' or 'Barotropic_Tide' or 'Diffusion_Equation' 
    # or 'Advection_Diffusion_Equation' or 'NonLinear_Manufactured_Solution' or 'Viscous_Burgers_Equation'.
    ProblemType_RossbyWave = Initialization.Specify_ProblemType_RossbyWave(ProblemType)
    PrintPhaseSpeedOfWaveModes = True
    PrintAmplitudesOfWaveModes = True
    TimeIntegrator = 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
    LF_TR_and_LF_AM3_with_FB_Feedback_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    Generalized_FB_with_AB2_AM3_Step_Type = 'ThirdOrderAccurate_WideStabilityRange'
    Generalized_FB_with_AB3_AM4_Step_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    if ProblemType_RossbyWave:
        ReadFromSELFOutputData = True
    else:
        ReadFromSELFOutputData = False
    if ProblemType_RossbyWave:
        PlotNumericalSolution = True
    else:
        PlotNumericalSolution = False
    if ReadFromSELFOutputData and ProblemType_RossbyWave:
        nElementsX = 10
        nElementsY = 10
        nXi = 7
        nEta = 7
        nXiPlot = 14
        nEtaPlot = 14
    else:
        nElementsX = 5
        nElementsY = 5
        nXi = 10
        nEta = 10
        nXiPlot = 20
        nEtaPlot = 20
    CheckStateVariableLimits = False
    PlotFigures = True
    ComputeOnlyExactSolution = False
    Restart = False
    Restart_iTime = 0
    Restart_FileName = ''
    Main.DetermineExactAndNumericalSolutions(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                             TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                             Generalized_FB_with_AB2_AM3_Step_Type,
                                             Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,
                                             nXiPlot,nEtaPlot,CheckStateVariableLimits,PlotFigures,
                                             ComputeOnlyExactSolution,PlotNumericalSolution,Restart,Restart_iTime,
                                             Restart_FileName,ReadFromSELFOutputData)
    
    
do_Test_DetermineExactAndNumericalSolutions = False
if do_Test_DetermineExactAndNumericalSolutions:
    Test_DetermineExactAndNumericalSolutions()