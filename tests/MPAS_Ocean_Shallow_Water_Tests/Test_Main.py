"""
Name: Test_Main.py
Author: Sid Bishnu
Details: As the name implies, this script tests the various functions of ../../src/MPAS_Ocean_Shallow_Water/Main.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/MPAS_Ocean_Shallow_Water/')
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
    ProblemType_EquatorialWave = Initialization.isEquatorialWave(ProblemType)
    ProblemType_ManufacturedRossbyWave = Initialization.Specify_ProblemType_ManufacturedRossbyWave(ProblemType)
    ProblemType_RossbyWave = Initialization.Specify_ProblemType_RossbyWave(ProblemType)
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
    Initialization.SpecifyMeshDirectoryAndMeshFileNames(ProblemType,ProblemType_ManufacturedRossbyWave,
                                                        ProblemType_RossbyWave,ProblemType_EquatorialWave))
    CheckStateVariableLimits = False
    PlotFigures = True
    Main.DetermineExactSolutions(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                 LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                 Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,
                                 BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                                 CheckStateVariableLimits,PlotFigures)
    
    
do_Test_DetermineExactSolutions = False
if do_Test_DetermineExactSolutions:
    Test_DetermineExactSolutions()  
    
    
def Test_DetermineNumericalSolutions():
    ProblemType = 'Coastal_Kelvin_Wave'
    # Choose ProblemType to be 'Plane_Gaussian_Wave' or 'Coastal_Kelvin_Wave' or 'Inertia_Gravity_Wave' 
    # or 'Manufactured_Planetary_Rossby_Wave' or 'Manufactured_Topographic_Rossby_Wave' or 'Planetary_Rossby_Wave' 
    # or 'Topographic_Rossby_Wave' or 'Equatorial_Kelvin_Wave' or 'Equatorial_Yanai_Wave' or 'Equatorial_Rossby_Wave' 
    # or 'Equatorial_Inertia_Gravity_Wave' or 'Barotropic_Tide' or 'Diffusion_Equation' 
    # or 'Advection_Diffusion_Equation' or 'NonLinear_Manufactured_Solution' or 'Viscous_Burgers_Equation'.
    ProblemType_EquatorialWave = Initialization.isEquatorialWave(ProblemType)
    ProblemType_ManufacturedRossbyWave = Initialization.Specify_ProblemType_ManufacturedRossbyWave(ProblemType)
    ProblemType_RossbyWave = Initialization.Specify_ProblemType_RossbyWave(ProblemType)
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
    SpecifyNumberOfTimeStepsManually = True
    nTime = 10
    MeshDirectory, BaseMeshFileName, MeshFileName = (
    Initialization.SpecifyMeshDirectoryAndMeshFileNames(ProblemType,ProblemType_ManufacturedRossbyWave,
                                                        ProblemType_RossbyWave,ProblemType_EquatorialWave))
    nRuns = 10
    elapsed_time = np.zeros(nRuns)
    for iRun in range(0,nRuns):
        elapsed_time[iRun] = (
        Main.DetermineNumericalSolutions(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                         TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                         Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
                                         nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,
                                         FixAngleEdge,PrintOutput,UseAveragedQuantities,
                                         SpecifyNumberOfTimeStepsManually,nTime))
        print('The elapsed time for run %2d is %.2f seconds.' %(iRun+1,elapsed_time[iRun]))
    print('The average elapsed time for each run is %.2f seconds.' %np.average(elapsed_time))
            
            
do_Test_DetermineNumericalSolutions = False
if do_Test_DetermineNumericalSolutions:
    Test_DetermineNumericalSolutions()    


def Test_DetermineExactAndNumericalSolutions():
    ProblemType = 'Coastal_Kelvin_Wave'
    # Choose ProblemType to be 'Plane_Gaussian_Wave' or 'Coastal_Kelvin_Wave' or 'Inertia_Gravity_Wave' 
    # or 'Manufactured_Planetary_Rossby_Wave' or 'Manufactured_Topographic_Rossby_Wave' or 'Planetary_Rossby_Wave' 
    # or 'Topographic_Rossby_Wave' or 'Equatorial_Kelvin_Wave' or 'Equatorial_Yanai_Wave' or 'Equatorial_Rossby_Wave' 
    # or 'Equatorial_Inertia_Gravity_Wave' or 'Barotropic_Tide' or 'Diffusion_Equation' 
    # or 'Advection_Diffusion_Equation' or 'NonLinear_Manufactured_Solution' or 'Viscous_Burgers_Equation'.
    ProblemType_EquatorialWave = Initialization.isEquatorialWave(ProblemType)
    ProblemType_ManufacturedRossbyWave = Initialization.Specify_ProblemType_ManufacturedRossbyWave(ProblemType)
    ProblemType_RossbyWave = Initialization.Specify_ProblemType_RossbyWave(ProblemType)
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
    Initialization.SpecifyMeshDirectoryAndMeshFileNames(ProblemType,ProblemType_ManufacturedRossbyWave,
                                                        ProblemType_RossbyWave,ProblemType_EquatorialWave))
    CheckStateVariableLimits = False
    PlotFigures = True
    ComputeOnlyExactSolution = False
    if ProblemType_RossbyWave:
        PlotNumericalSolution = True
    else:
        PlotNumericalSolution = False
    Restart = False
    Restart_iTime = 0
    Restart_FileName_NormalVelocity = ''
    Restart_FileName_SurfaceElevation = ''
    InterpolateExactVelocitiesFromEdgesToCellCenters = True
    Main.DetermineExactAndNumericalSolutions(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
    UseAveragedQuantities,CheckStateVariableLimits,PlotFigures,InterpolateExactVelocitiesFromEdgesToCellCenters,
    ComputeOnlyExactSolution,PlotNumericalSolution,Restart,Restart_iTime,Restart_FileName_NormalVelocity,
    Restart_FileName_SurfaceElevation)
    
    
do_Test_DetermineExactAndNumericalSolutions = False
if do_Test_DetermineExactAndNumericalSolutions:
    Test_DetermineExactAndNumericalSolutions()