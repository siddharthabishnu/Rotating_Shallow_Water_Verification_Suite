"""
Name: Main.py
Author: Sid Bishnu
Details: This script contains functions for determining numerical solutions of the various test cases along with the 
numerical error.
"""


import numpy as np
import time
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR
    import ExactSolutionsAndSourceTerms as ESST
    import Initialization
    import MPASOceanShallowWaterClass
    import TimeSteppingMethods as TSM


def FormatSimulationTime(time,non_integral_seconds=False,display_time=False,
                         ProblemType_PlanetaryTopographicRossbyWave=False,ProblemType_NoExactSolution=False):
    years = np.floor(time/(86400.0*365.0))
    remainingtime = np.mod(time,86400.0*365.0)
    days = np.floor(remainingtime/86400.0)
    remainingtime = np.mod(remainingtime,86400.0)
    hours = np.floor(remainingtime/3600.0)
    remainingtime = np.mod(time,3600.0)
    minutes = np.floor(remainingtime/60.0)
    seconds = np.mod(remainingtime,60.0)
    if years <= 1.0:
        years_string = 'Year'
    else:
        years_string = 'Years'
    if days <= 1.0:
        days_string = 'Day'
    else:
        days_string = 'Days'
    if hours <= 1.0:
        hours_string = 'Hour'
    else:
        hours_string = 'Hours'
    if minutes <= 1.0:
        minutes_string = 'Minute'
    else:
        minutes_string = 'Minutes'        
    if seconds <= 1.0:
        seconds_string = 'Second'
    else:
        seconds_string = 'Seconds'
    if ProblemType_PlanetaryTopographicRossbyWave or ProblemType_NoExactSolution:
        if time >= 86400.0*365.0:
            SimulationTime = ('%d %s %d %s %2d %s %2d %s' 
                              %(years,years_string,days,days_string,hours,hours_string,minutes,minutes_string))
        elif time < 86400.0*365.0 and time >= 86400.0:
            SimulationTime = '%d %s %2d %s %2d %s' %(days,days_string,hours,hours_string,minutes,minutes_string)
        elif time < 86400.0 and time >= 3600.0:
            SimulationTime = '%2d %s %2d %s' %(hours,hours_string,minutes,minutes_string)
        elif time < 3600.0:
            SimulationTime = '%2d %s' %(minutes,minutes_string)         
    else:
        if time >= 86400.0*365.0:
            if non_integral_seconds:
                SimulationTime = ('%d %s %d %s %2d %s %2d %s %.2g %s' 
                                  %(years,years_string,days,days_string,hours,hours_string,minutes,minutes_string,
                                    seconds,seconds_string))            
            else:
                SimulationTime = ('%d %s %d %s %2d %s %2d %s %2d %s' 
                                  %(years,years_string,days,days_string,hours,hours_string,minutes,minutes_string,
                                    seconds,seconds_string))
        elif time < 86400.0*365.0 and time >= 86400.0:
            if non_integral_seconds:
                SimulationTime = ('%d %s %2d %s %2d %s %.2g %s' 
                                   %(days,days_string,hours,hours_string,minutes,minutes_string,seconds,seconds_string))
            else:
                SimulationTime = ('%d %s %2d %s %2d %s %2d %s' 
                                  %(days,days_string,hours,hours_string,minutes,minutes_string,seconds,seconds_string))
        elif time < 86400.0 and time >= 3600.0:
            if non_integral_seconds:
                SimulationTime = ('%2d %s %2d %s %.2g %s' %(hours,hours_string,minutes,minutes_string,seconds,
                                                            seconds_string))
            else:
                SimulationTime = ('%2d %s %2d %s %2d %s' %(hours,hours_string,minutes,minutes_string,seconds,
                                                           seconds_string))
        elif time < 3600.0 and time >= 60.0:
            if non_integral_seconds:
                SimulationTime = '%2d %s %.2g %s' %(minutes,minutes_string,seconds,seconds_string)
            else:
                SimulationTime = '%2d %s %2d %s' %(minutes,minutes_string,seconds,seconds_string)
        elif time < 60.0:
            if non_integral_seconds:
                SimulationTime = '%.2g %s' %(seconds,seconds_string)
            else:
                SimulationTime = '%2d %s' %(seconds,seconds_string)   
    if display_time:
        print('The formatted simulation time is %s.' %SimulationTime)
    return SimulationTime


def DetermineCourantNumberForGivenTimeStep(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                           TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                           Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
                                           nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,
                                           MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,dt,
                                           PrintCourantNumber=False):
    myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
    UseAveragedQuantities)
    dx = myMPASOceanShallowWater.myNameList.dx
    dy = myMPASOceanShallowWater.myNameList.dy
    cX1 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.cX1
    cX2 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.cX2
    cY1 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.cY1
    cY2 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.cY2
    abs_cX = max(abs(cX1),abs(cX2))
    abs_cY = max(abs(cY1),abs(cY2))
    CourantNumber = dt*(abs_cX/dx + abs_cY/dy)
    if PrintCourantNumber:
        print('The Courant number is %.6f.' %CourantNumber)
    return CourantNumber


def DetermineCourantNumberForGivenTimeStepAndCheckItsValue(ProblemType):
    PrintPhaseSpeedOfWaveModes = True
    PrintAmplitudesOfWaveModes = True
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
    ProblemType_EquatorialWave = Initialization.isEquatorialWave(ProblemType)
    MeshDirectory, BaseMeshFileName, MeshFileName = (
    Initialization.SpecifyMeshDirectoryAndMeshFileNames(ProblemType,ProblemType_EquatorialWave))
    if ProblemType == 'Plane_Gaussian_Wave':
        dt = 3.27*10.0**(-3.0)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        dt = 216.0
    elif ProblemType == 'Inertia_Gravity_Wave':
        dt = 111.0
    elif ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        dt = 230100.0
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        dt = 873.0
    elif ProblemType == 'Equatorial_Yanai_Wave':
        dt = 474.0
    elif ProblemType == 'Equatorial_Rossby_Wave':
        dt = 3144.0
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        dt = 285.0
    elif ProblemType == 'Barotropic_Tide':
        dt = 12.3
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        dt = 162.0
    CourantNumber = DetermineCourantNumberForGivenTimeStep(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
    UseAveragedQuantities,dt,PrintCourantNumber=True)
    myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
    UseAveragedQuantities,CourantNumber,UseCourantNumberToDetermineTimeStep=True)
    if ProblemType == 'Inertia_Gravity_Wave' or ProblemType == 'Planetary_Rossby_Wave':
        beta0 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.beta0
        c0 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.c0
        f0 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.f0
        kX1 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.kX1
        kX2 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.kX2
        kY1 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.kY1
        kY2 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.kY2
        lY = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.lY
        k1 = np.sqrt(kX1**2.0 + kY1**2.0)
        k2 = np.sqrt(kX2**2.0 + kY2**2.0)
        if ProblemType == 'Inertia_Gravity_Wave':
            print('For the first wave mode, the ratio of f0:ck is %.6f.' %(f0/(c0*k1)))
            print('For the second wave mode, the ratio of f0:ck is %.6f.' %(f0/(c0*k2)))
        else:
            print('With the meridional extent being %.3f km, the ratio of beta0*lY:f0 is %.6f << 1.' 
                  %(lY/1000.0,beta0*lY/f0))
            
            
def DetermineNumberOfTimeStepsForSimulation(ProblemType):
    ProblemType_EquatorialWave = Initialization.isEquatorialWave(ProblemType)
    PrintPhaseSpeedOfWaveModes = True
    PrintAmplitudesOfWaveModes = True
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
    myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
    UseAveragedQuantities)
    dt = myMPASOceanShallowWater.myNameList.dt 
    lX = myMPASOceanShallowWater.myNameList.lX
    lY = myMPASOceanShallowWater.myNameList.lY
    cX1 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.cX1
    cX2 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.cX2
    cY1 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.cY1
    cY2 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.cY2
    abs_cX = max(abs(cX1),abs(cX2))
    abs_cY = max(abs(cY1),abs(cY2))
    if abs_cX != 0.0:
        SimulationTime = lX/abs_cX 
    else:
        SimulationTime = lY/abs_cY
    # Note that for all two-dimensional dispersive waves, 
    # SimulationTime = lX/abs_cX = lX*kX/abs(omega) = lY*kY/abs(omega) = lY/abs_cY
    # where kX and kY are the zonal and meridional wavenumbers of the fast wave mode with omega being its angular 
    # frequency.
    if ProblemType == 'Plane_Gaussian_Wave':
        print('The time taken by the wave to traverse half the diagonal extent of the domain is %.3g.' %SimulationTime)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        print('The time taken by the fast wave mode to traverse the meridional extent of the domain is %.3g.' 
              %SimulationTime)
    elif ProblemType_EquatorialWave:
        print('The time taken by the fast wave mode to traverse the zonal extent of the domain is %.3g.' 
              %SimulationTime)
    elif ProblemType == 'Barotropic_Tide':
        print('The time taken by either component of the first standing wave mode to traverse the zonal extent of the '
              + 'domain is %.3g.' %SimulationTime)
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        print('The time taken by the wave to traverse half the diagonal extent of the domain is %.3g.' %SimulationTime)
    else:
        print('The time taken by the fast wave mode to traverse half the diagonal extent of the domain is %.3g.' 
              %SimulationTime)
    print('The minimum number of time steps of magnitude %.3g required to constitute this simulation time is %d.'
          %(dt,int(np.ceil(SimulationTime/dt))))


def DetermineExactSolutions(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                            LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                            Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,
                            BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                            CheckStateVariableLimits,PlotFigures,InterpolateExactVelocitiesFromEdgesToCellCenters=True):
    myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
    UseAveragedQuantities)
    ProblemType_EquatorialWave = myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave
    ProblemType_NoExactSolution = myMPASOceanShallowWater.myNameList.ProblemType_NoExactSolution
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    nTime = Initialization.SpecifyNumberOfTimeSteps(ProblemType,ProblemType_EquatorialWave,ProblemType_NoExactSolution)
    dt = myMPASOceanShallowWater.myNameList.dt
    ExactZonalVelocityLimits = myMPASOceanShallowWater.myNameList.ExactZonalVelocityLimits
    ExactMeridionalVelocityLimits = myMPASOceanShallowWater.myNameList.ExactMeridionalVelocityLimits
    ExactSurfaceElevationLimits = myMPASOceanShallowWater.myNameList.ExactSurfaceElevationLimits
    if CheckStateVariableLimits:
        print('The limits of zonal velocity are [%.6f,%.6f].' 
              %(ExactZonalVelocityLimits[0],ExactZonalVelocityLimits[1]))
        print('The limits of meridional velocity are [%.6f,%.6f].' 
              %(ExactMeridionalVelocityLimits[0],ExactMeridionalVelocityLimits[1]))
        print('The limits of surface elevation are [%.6f,%.6f].' 
              %(ExactSurfaceElevationLimits[0],ExactSurfaceElevationLimits[1]))
        return
    nCounters = 2
    dt = myMPASOceanShallowWater.myNameList.dt
    nDumpFrequency = myMPASOceanShallowWater.myNameList.nDumpFrequency
    nTime = myMPASOceanShallowWater.myNameList.nTime
    if ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        ProblemType_PlanetaryTopographicRossbyWave = True
        ExactSurfaceElevationMaximumMagnitude = ExactSurfaceElevationLimits[1]
    else:
        ProblemType_PlanetaryTopographicRossbyWave = False
    if myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave and not(ProblemType == 'Equatorial_Kelvin_Wave'):
        HermiteFunctionMaximumAmplitude = (
        ESST.DetermineHermiteFunctionMaximumAmplitudeWithMeridionalLocation(ProblemType,ReturnMeridionalLocation=False))
        etaHat1 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.etaHat1
        etaHat2 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.etaHat2
        VelocityScale = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.VelocityScale
        ExactMeridionalVelocityMaximumMagnitude = VelocityScale*HermiteFunctionMaximumAmplitude*(etaHat1 + etaHat2)
    PlotExactZonalVelocity = myMPASOceanShallowWater.myNameList.LogicalArrayPlot[0]
    PlotExactMeridionalVelocity = myMPASOceanShallowWater.myNameList.LogicalArrayPlot[1]
    PlotExactSurfaceElevation = myMPASOceanShallowWater.myNameList.LogicalArrayPlot[2]
    ProblemType_FileName = myMPASOceanShallowWater.myNameList.ProblemType_FileName
    ComputeOnlyExactSolution = True
    for iCounter in range(0,nCounters):
        for iTime in range(0,nTime):
            myMPASOceanShallowWater.iTime = iTime
            myMPASOceanShallowWater.time = float(iTime)*dt
            if np.mod(iTime,nDumpFrequency) == 0.0:
                if iCounter == 0:
                    MPASOceanShallowWaterClass.DetermineExactSolutions(myMPASOceanShallowWater,
                                                                       InterpolateExactVelocitiesFromEdgesToCellCenters)
                    ExactZonalVelocities = myMPASOceanShallowWater.mySolution.uExact[:]
                    ExactMeridionalVelocities = myMPASOceanShallowWater.mySolution.vExact[:]
                    ExactSurfaceElevations = myMPASOceanShallowWater.mySolution.sshExact[:]
                    if PlotFigures:                    
                        if not(ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Coastal_Kelvin_Wave' 
                               or ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Inertia_Gravity_Wave' 
                               or ProblemType == 'Barotropic_Tide' or ProblemType == 'NonLinear_Manufactured_Solution'):
                            if iTime == 0:
                                ExactZonalVelocityMinimum = np.min(ExactZonalVelocities)
                                ExactZonalVelocityMaximum = np.max(ExactZonalVelocities)
                                ExactMeridionalVelocityMinimum = np.min(ExactMeridionalVelocities)
                                ExactMeridionalVelocityMaximum = np.max(ExactMeridionalVelocities)
                                if not(ProblemType_PlanetaryTopographicRossbyWave):
                                    ExactSurfaceElevationMinimum = np.min(ExactSurfaceElevations)
                                    ExactSurfaceElevationMaximum = np.max(ExactSurfaceElevations)
                            else:
                                ExactZonalVelocityMinimum = min(ExactZonalVelocityMinimum,np.min(ExactZonalVelocities))
                                ExactZonalVelocityMaximum = max(ExactZonalVelocityMaximum,np.max(ExactZonalVelocities))
                                ExactMeridionalVelocityMinimum = (
                                min(ExactMeridionalVelocityMinimum,np.min(ExactMeridionalVelocities)))
                                ExactMeridionalVelocityMaximum = (
                                max(ExactMeridionalVelocityMaximum,np.max(ExactMeridionalVelocities)))    
                                if not(ProblemType_PlanetaryTopographicRossbyWave):
                                    ExactSurfaceElevationMinimum = min(ExactSurfaceElevationMinimum,
                                                                       np.min(ExactSurfaceElevations))
                                    ExactSurfaceElevationMaximum = max(ExactSurfaceElevationMaximum,
                                                                       np.max(ExactSurfaceElevations))
                            if iTime == nTime - 1:                        
                                ExactZonalVelocityMaximumMagnitude = max(abs(ExactZonalVelocityMinimum),
                                                                         abs(ExactZonalVelocityMaximum)) 
                                if not(myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave 
                                       and not(ProblemType == 'Equatorial_Kelvin_Wave')):
                                    ExactMeridionalVelocityMaximumMagnitude = (
                                    max(abs(ExactMeridionalVelocityMinimum),abs(ExactMeridionalVelocityMaximum)))
                                if not(ProblemType_PlanetaryTopographicRossbyWave):
                                    ExactSurfaceElevationMaximumMagnitude = max(abs(ExactSurfaceElevationMinimum),
                                                                                abs(ExactSurfaceElevationMaximum))
                                ExactZonalVelocityLimits = [-ExactZonalVelocityMaximumMagnitude,
                                                            ExactZonalVelocityMaximumMagnitude]
                                ExactMeridionalVelocityLimits = [-ExactMeridionalVelocityMaximumMagnitude,
                                                                 ExactMeridionalVelocityMaximumMagnitude]
                                ExactSurfaceElevationLimits = [-ExactSurfaceElevationMaximumMagnitude,
                                                               ExactSurfaceElevationMaximumMagnitude]
                        FileName = ProblemType_FileName + '_ExactSolution_%3.3d' %iTime
                        MPASOceanShallowWaterClass.WriteStateMPASOceanShallowWater(myMPASOceanShallowWater,FileName,
                                                                                   ComputeOnlyExactSolution)
                        if iTime == nTime - 1:
                            if PlotExactZonalVelocity:
                                FileName = ProblemType_FileName + '_ExactZonalVelocityLimits'
                                CR.WriteStateVariableLimitsToFile(myMPASOceanShallowWater.OutputDirectory,
                                                                  ExactZonalVelocityLimits,FileName)
                            if PlotExactMeridionalVelocity:
                                FileName = ProblemType_FileName + '_ExactMeridionalVelocityLimits'
                                CR.WriteStateVariableLimitsToFile(myMPASOceanShallowWater.OutputDirectory,
                                                                  ExactMeridionalVelocityLimits,FileName)
                            if PlotExactSurfaceElevation:
                                FileName = ProblemType_FileName + '_ExactSurfaceElevationLimits'
                                CR.WriteStateVariableLimitsToFile(myMPASOceanShallowWater.OutputDirectory,
                                                                  ExactSurfaceElevationLimits,FileName)    
                else: # if iCounter == 1:
                    if PlotFigures:
                        FileName = ProblemType_FileName + '_ExactSolution_%3.3d' %iTime + '.tec'
                        if ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Barotropic_Tide':
                            non_integral_seconds = True
                        else:
                            non_integral_seconds = False
                        DisplayTime = FormatSimulationTime(myMPASOceanShallowWater.time,
                                                           non_integral_seconds=non_integral_seconds,
                                                           display_time=False,ProblemType_PlanetaryTopographicRossbyWave
                                                           =ProblemType_PlanetaryTopographicRossbyWave)
                        UseGivenColorBarLimits = True
                        UseInterpolatedErrorLimits = True
                        MPASOceanShallowWaterClass.PythonPlotStateMPASOceanShallowWater(
                        myMPASOceanShallowWater,FileName,DisplayTime,UseGivenColorBarLimits,UseInterpolatedErrorLimits,
                        ComputeOnlyExactSolution)

                        
def DetermineNumericalSolutions(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,
                                BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                                SpecifyNumberOfTimeStepsManually=True,nTime=10):
    myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
    UseAveragedQuantities)
    ProblemType_EquatorialWave = myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave
    ProblemType_NoExactSolution = myMPASOceanShallowWater.myNameList.ProblemType_NoExactSolution
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nCells = myMPASOceanShallowWater.myMesh.nCells
    myQuadratureOnHexagon = myMPASOceanShallowWater.myMesh.myQuadratureOnHexagon
    HexagonLength = myMPASOceanShallowWater.myMesh.HexagonLength
    if not(SpecifyNumberOfTimeStepsManually):
        nTime = Initialization.SpecifyNumberOfTimeSteps(ProblemType,ProblemType_EquatorialWave,
                                                        ProblemType_NoExactSolution)
    dt = myMPASOceanShallowWater.myNameList.dt
    for iTime in range(0,nTime):
        myMPASOceanShallowWater.iTime = iTime
        myMPASOceanShallowWater.time = float(iTime)*dt
        if iTime == 0:
            # Specify the initial conditions.
            for iEdge in range(0,nEdges):
                xEdge = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
                yEdge = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
                dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
                angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,xEdge,yEdge,
                                                  myMPASOceanShallowWater.time,UseAveragedQuantities,myQuadratureOnEdge,
                                                  dvEdge,angleEdge))
            for iCell in range(0,nCells):
                xCell = myMPASOceanShallowWater.myMesh.xCell[iCell]
                yCell = myMPASOceanShallowWater.myMesh.yCell[iCell]
                myMPASOceanShallowWater.mySolution.ssh[iCell] = (
                ESST.DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,xCell,yCell,
                                                    myMPASOceanShallowWater.time,UseAveragedQuantities,
                                                    myQuadratureOnHexagon,HexagonLength))
            start_time = time.time()
        if iTime <= nTime - 1:
            TSM.TimeIntegration(myMPASOceanShallowWater)
        if iTime == nTime - 1:
            end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

                        
def DetermineExactAndNumericalSolutions(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                        TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                        Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
                                        nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,
                                        FixAngleEdge,PrintOutput,UseAveragedQuantities,CheckStateVariableLimits,
                                        PlotFigures,InterpolateExactVelocitiesFromEdgesToCellCenters=True,
                                        ComputeOnlyExactSolution=False,PlotNumericalSolution=False,Restart=False,
                                        Restart_iTime=0,Restart_FileName_NormalVelocity='',
                                        Restart_FileName_SurfaceElevation=''):
    myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
    UseAveragedQuantities)
    ProblemType_EquatorialWave = myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave
    ProblemType_NoExactSolution = myMPASOceanShallowWater.myNameList.ProblemType_NoExactSolution
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    nTime = Initialization.SpecifyNumberOfTimeSteps(ProblemType,ProblemType_EquatorialWave,ProblemType_NoExactSolution)
    dt = myMPASOceanShallowWater.myNameList.dt
    ExactZonalVelocityLimits = myMPASOceanShallowWater.myNameList.ExactZonalVelocityLimits
    ExactMeridionalVelocityLimits = myMPASOceanShallowWater.myNameList.ExactMeridionalVelocityLimits
    ExactSurfaceElevationLimits = myMPASOceanShallowWater.myNameList.ExactSurfaceElevationLimits
    if CheckStateVariableLimits:
        print('The limits of zonal velocity are [%.6f,%.6f].' 
              %(ExactZonalVelocityLimits[0],ExactZonalVelocityLimits[1]))
        print('The limits of meridional velocity are [%.6f,%.6f].' 
              %(ExactMeridionalVelocityLimits[0],ExactMeridionalVelocityLimits[1]))
        print('The limits of surface elevation are [%.6f,%.6f].' 
              %(ExactSurfaceElevationLimits[0],ExactSurfaceElevationLimits[1]))
        return
    nCounters = 2
    dt = myMPASOceanShallowWater.myNameList.dt
    nDumpFrequency = myMPASOceanShallowWater.myNameList.nDumpFrequency
    nRestartFrequency = myMPASOceanShallowWater.myNameList.nRestartFrequency
    nTime = myMPASOceanShallowWater.myNameList.nTime
    if ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        ProblemType_PlanetaryTopographicRossbyWave = True
        ExactSurfaceElevationMaximumMagnitude = ExactSurfaceElevationLimits[1]
    else:
        ProblemType_PlanetaryTopographicRossbyWave = False
    if myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave and not(ProblemType == 'Equatorial_Kelvin_Wave'):
        HermiteFunctionMaximumAmplitude = (
        ESST.DetermineHermiteFunctionMaximumAmplitudeWithMeridionalLocation(ProblemType,ReturnMeridionalLocation=False))
        etaHat1 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.etaHat1
        etaHat2 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.etaHat2
        VelocityScale = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.VelocityScale
        ExactMeridionalVelocityMaximumMagnitude = VelocityScale*HermiteFunctionMaximumAmplitude*(etaHat1 + etaHat2)
    PlotExactZonalVelocity = myMPASOceanShallowWater.myNameList.LogicalArrayPlot[0]
    PlotExactMeridionalVelocity = myMPASOceanShallowWater.myNameList.LogicalArrayPlot[1]
    PlotExactSurfaceElevation = myMPASOceanShallowWater.myNameList.LogicalArrayPlot[2]
    ProblemType_FileName = myMPASOceanShallowWater.myNameList.ProblemType_FileName
    TimeIntegratorShortForm = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.TimeIntegratorShortForm
    iTimeFormat = '%3.3d'
    if Restart:
        iTime_Start = Restart_iTime
    else:
        iTime_Start = 0
    for iCounter in range(0,nCounters):
        for iTime in range(iTime_Start,nTime):
            myMPASOceanShallowWater.iTime = iTime
            myMPASOceanShallowWater.time = float(iTime)*dt
            if iCounter == 0: 
                if np.mod(iTime,nDumpFrequency) == 0.0:
                    MPASOceanShallowWaterClass.DetermineExactSolutions(myMPASOceanShallowWater,
                                                                       InterpolateExactVelocitiesFromEdgesToCellCenters)
                    ExactZonalVelocities = myMPASOceanShallowWater.mySolution.uExact[:]
                    ExactMeridionalVelocities = myMPASOceanShallowWater.mySolution.vExact[:]
                    ExactSurfaceElevations = myMPASOceanShallowWater.mySolution.sshExact[:]
                    if not(ComputeOnlyExactSolution):
                        if iTime == iTime_Start:
                            if Restart:
                                normalVelocityRestart, sshRestart = (
                                MPASOceanShallowWaterClass.ReadStateMPASOceanShallowWater(  
                                myMPASOceanShallowWater,Restart_FileName_NormalVelocity+'.tec',
                                Restart_FileName_SurfaceElevation+'.tec'))
                                MPASOceanShallowWaterClass.SpecifyRestartConditions(
                                myMPASOceanShallowWater,normalVelocityRestart,sshRestart)
                            else:                 
                                MPASOceanShallowWaterClass.SpecifyInitialConditions(myMPASOceanShallowWater)
                        MPASOceanShallowWaterClass.ComputeError(myMPASOceanShallowWater)
                        ZonalVelocityError = myMPASOceanShallowWater.mySolution.uError[:]
                        MeridionalVelocityError = myMPASOceanShallowWater.mySolution.vError[:]
                        SurfaceElevationError = myMPASOceanShallowWater.mySolution.sshError[:]
                    if PlotFigures:                    
                        if not(ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Coastal_Kelvin_Wave' 
                               or ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Inertia_Gravity_Wave' 
                               or ProblemType == 'Barotropic_Tide' or ProblemType == 'NonLinear_Manufactured_Solution'):
                            if iTime == iTime_Start:
                                ExactZonalVelocityMinimum = np.min(ExactZonalVelocities)
                                ExactZonalVelocityMaximum = np.max(ExactZonalVelocities)
                                ExactMeridionalVelocityMinimum = np.min(ExactMeridionalVelocities)
                                ExactMeridionalVelocityMaximum = np.max(ExactMeridionalVelocities)
                                if not(ProblemType_PlanetaryTopographicRossbyWave):
                                    ExactSurfaceElevationMinimum = np.min(ExactSurfaceElevations)
                                    ExactSurfaceElevationMaximum = np.max(ExactSurfaceElevations)
                            else:
                                ExactZonalVelocityMinimum = min(ExactZonalVelocityMinimum,np.min(ExactZonalVelocities))
                                ExactZonalVelocityMaximum = max(ExactZonalVelocityMaximum,np.max(ExactZonalVelocities))
                                ExactMeridionalVelocityMinimum = min(ExactMeridionalVelocityMinimum,
                                                                     np.min(ExactMeridionalVelocities))
                                ExactMeridionalVelocityMaximum = max(ExactMeridionalVelocityMaximum,
                                                                     np.max(ExactMeridionalVelocities)) 
                                if not(ProblemType_PlanetaryTopographicRossbyWave):
                                    ExactSurfaceElevationMinimum = min(ExactSurfaceElevationMinimum,
                                                                       np.min(ExactSurfaceElevations))
                                    ExactSurfaceElevationMaximum = max(ExactSurfaceElevationMaximum,
                                                                       np.max(ExactSurfaceElevations))
                            if iTime == nTime - 1:                        
                                ExactZonalVelocityMaximumMagnitude = max(abs(ExactZonalVelocityMinimum),
                                                                         abs(ExactZonalVelocityMaximum)) 
                                if not(myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave 
                                       and not(ProblemType == 'Equatorial_Kelvin_Wave')):
                                    ExactMeridionalVelocityMaximumMagnitude = max(abs(ExactMeridionalVelocityMinimum),
                                                                                  abs(ExactMeridionalVelocityMaximum))
                                if not(ProblemType_PlanetaryTopographicRossbyWave):
                                    ExactSurfaceElevationMaximumMagnitude = max(abs(ExactSurfaceElevationMinimum),
                                                                                abs(ExactSurfaceElevationMaximum))   
                                if ProblemType_NoExactSolution:
                                    ExactZonalVelocityLimits = [ExactZonalVelocityMinimum,ExactZonalVelocityMaximum]
                                    ExactMeridionalVelocityLimits = [ExactMeridionalVelocityMinimum,
                                                                     ExactMeridionalVelocityMaximum]
                                    ExactSurfaceElevationLimits = [ExactSurfaceElevationMinimum,
                                                                   ExactSurfaceElevationMaximum]            
                                else:
                                    ExactZonalVelocityLimits = [-ExactZonalVelocityMaximumMagnitude,
                                                                ExactZonalVelocityMaximumMagnitude]
                                    ExactMeridionalVelocityLimits = [-ExactMeridionalVelocityMaximumMagnitude,
                                                                    ExactMeridionalVelocityMaximumMagnitude]
                                    ExactSurfaceElevationLimits = [-ExactSurfaceElevationMaximumMagnitude,
                                                                   ExactSurfaceElevationMaximumMagnitude]
                        if not(ComputeOnlyExactSolution):
                            if iTime == iTime_Start:
                                ZonalVelocityErrorMinimum = np.min(ZonalVelocityError)
                                ZonalVelocityErrorMaximum = np.max(ZonalVelocityError)
                                MeridionalVelocityErrorMinimum = np.min(MeridionalVelocityError)
                                MeridionalVelocityErrorMaximum = np.max(MeridionalVelocityError)
                                SurfaceElevationErrorMinimum = np.min(SurfaceElevationError)
                                SurfaceElevationErrorMaximum = np.max(SurfaceElevationError)
                            else:
                                ZonalVelocityErrorMinimum = min(ZonalVelocityErrorMinimum,
                                                                np.min(ZonalVelocityError))
                                ZonalVelocityErrorMaximum = max(ZonalVelocityErrorMaximum,
                                                                np.max(ZonalVelocityError))
                                MeridionalVelocityErrorMinimum = min(MeridionalVelocityErrorMinimum,
                                                                     np.min(MeridionalVelocityError))
                                MeridionalVelocityErrorMaximum = max(MeridionalVelocityErrorMaximum,
                                                                     np.max(MeridionalVelocityError))
                                SurfaceElevationErrorMinimum = min(SurfaceElevationErrorMinimum,
                                                                   np.min(SurfaceElevationError))
                                SurfaceElevationErrorMaximum = max(SurfaceElevationErrorMaximum,
                                                                   np.max(SurfaceElevationError))
                            if iTime == nTime - 1:
                                ZonalVelocityErrorMaximumMagnitude = max(abs(ZonalVelocityErrorMinimum),
                                                                         ZonalVelocityErrorMaximum)
                                MeridionalVelocityErrorMaximumMagnitude = max(abs(MeridionalVelocityErrorMinimum),
                                                                              MeridionalVelocityErrorMaximum)
                                SurfaceElevationErrorMaximumMagnitude = max(abs(SurfaceElevationErrorMinimum),
                                                                            SurfaceElevationErrorMaximum)
                                if ProblemType_NoExactSolution:
                                    ZonalVelocityErrorLimits = [ZonalVelocityErrorMinimum,ZonalVelocityErrorMaximum]
                                    MeridionalVelocityErrorLimits = [MeridionalVelocityErrorMinimum,
                                                                     MeridionalVelocityErrorMaximum]
                                    SurfaceElevationErrorLimits = [SurfaceElevationErrorMinimum,
                                                                   SurfaceElevationErrorMaximum]            
                                else:
                                    ZonalVelocityErrorLimits = [-ZonalVelocityErrorMaximumMagnitude,
                                                                ZonalVelocityErrorMaximumMagnitude]
                                    MeridionalVelocityErrorLimits = [-MeridionalVelocityErrorMaximumMagnitude,
                                                                     MeridionalVelocityErrorMaximumMagnitude]
                                    SurfaceElevationErrorLimits = [-SurfaceElevationErrorMaximumMagnitude,
                                                                   SurfaceElevationErrorMaximumMagnitude]
                        if ComputeOnlyExactSolution:
                            FileName = ProblemType_FileName + '_State_' + iTimeFormat %iTime
                        else:
                            FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_State_' 
                                        + iTimeFormat %iTime)
                        MPASOceanShallowWaterClass.WriteStateMPASOceanShallowWater(myMPASOceanShallowWater,FileName,
                                                                                   ComputeOnlyExactSolution)
                        if not(ComputeOnlyExactSolution) and np.mod(iTime,nRestartFrequency) == 0.0:
                            RestartFileName_NormalVelocity = (
                            (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_RestartState_NormalVelocity_' 
                             + iTimeFormat %iTime))
                            RestartFileName_SurfaceElevation = (
                            (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_RestartState_SurfaceElevation_' 
                             + iTimeFormat %iTime))
                            MPASOceanShallowWaterClass.WriteRestartStateMPASOceanShallowWater(
                            myMPASOceanShallowWater,RestartFileName_NormalVelocity,RestartFileName_SurfaceElevation)
                        if iTime == nTime - 1:
                            if PlotExactZonalVelocity:
                                FileName = ProblemType_FileName + '_ExactZonalVelocityLimits'
                                CR.WriteStateVariableLimitsToFile(myMPASOceanShallowWater.OutputDirectory,
                                                                  ExactZonalVelocityLimits,FileName)
                                if not(ComputeOnlyExactSolution):
                                    FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                                + '_ZonalVelocityErrorLimits')
                                    CR.WriteStateVariableLimitsToFile(myMPASOceanShallowWater.OutputDirectory,
                                                                      ZonalVelocityErrorLimits,FileName)
                            if PlotExactMeridionalVelocity:
                                FileName = ProblemType_FileName + '_ExactMeridionalVelocityLimits'
                                CR.WriteStateVariableLimitsToFile(myMPASOceanShallowWater.OutputDirectory,
                                                                  ExactMeridionalVelocityLimits,FileName)
                                if not(ComputeOnlyExactSolution):
                                    FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm
                                                + '_MeridionalVelocityErrorLimits')
                                    CR.WriteStateVariableLimitsToFile(myMPASOceanShallowWater.OutputDirectory,
                                                                      MeridionalVelocityErrorLimits,FileName)
                            if PlotExactSurfaceElevation:
                                FileName = ProblemType_FileName + '_ExactSurfaceElevationLimits'
                                CR.WriteStateVariableLimitsToFile(myMPASOceanShallowWater.OutputDirectory,
                                                                  ExactSurfaceElevationLimits,FileName)    
                                if not(ComputeOnlyExactSolution):
                                    FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                                + '_SurfaceElevationErrorLimits')
                                    CR.WriteStateVariableLimitsToFile(myMPASOceanShallowWater.OutputDirectory,
                                                                      SurfaceElevationErrorLimits,FileName)
                if not(ComputeOnlyExactSolution) and iTime < nTime - 1:
                    TSM.TimeIntegration(myMPASOceanShallowWater)
            else: # if iCounter == 1:
                if np.mod(iTime,nDumpFrequency) == 0.0:
                    if PlotFigures:
                        if ComputeOnlyExactSolution:
                            FileName = ProblemType_FileName + '_State_' + iTimeFormat %iTime + '.tec'
                        else:
                            FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_State_' 
                                        + iTimeFormat %iTime + '.tec')
                        if ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Barotropic_Tide':
                            non_integral_seconds = True
                        else:
                            non_integral_seconds = False
                        DisplayTime = FormatSimulationTime(myMPASOceanShallowWater.time,
                                                           non_integral_seconds=non_integral_seconds,
                                                           display_time=False,ProblemType_PlanetaryTopographicRossbyWave
                                                           =ProblemType_PlanetaryTopographicRossbyWave,
                                                           ProblemType_NoExactSolution=ProblemType_NoExactSolution)
                        UseGivenColorBarLimits = True
                        UseInterpolatedErrorLimits = True
                        MPASOceanShallowWaterClass.PythonPlotStateMPASOceanShallowWater(
                        myMPASOceanShallowWater,FileName,DisplayTime,UseGivenColorBarLimits,UseInterpolatedErrorLimits,
                        ComputeOnlyExactSolution,PlotNumericalSolution)