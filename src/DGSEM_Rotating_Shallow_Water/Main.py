"""
Name: Main.py
Author: Sid Bishnu
Details: This script contains functions for determining numerical solutions of the various test cases along with the 
numerical error.
"""


import numpy as np
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR
    import ExactSolutionsAndSourceTerms as ESST
    import DGSEM2DClass
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
                                           nElementsX,nElementsY,nXi,nEta,nXiPlot,nEtaPlot,dt,PrintCourantNumber=False):
    myDGSEM2D = DGSEM2DClass.DGSEM2D(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                     Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,
                                     nEtaPlot)
    dx = myDGSEM2D.myNameList.dx
    dy = myDGSEM2D.myNameList.dy
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    cX1 = myDGSEM2D.myNameList.myExactSolutionParameters.cX1
    cX2 = myDGSEM2D.myNameList.myExactSolutionParameters.cX2
    cY1 = myDGSEM2D.myNameList.myExactSolutionParameters.cY1
    cY2 = myDGSEM2D.myNameList.myExactSolutionParameters.cY2
    abs_cX = max(abs(cX1),abs(cX2))
    abs_cY = max(abs(cY1),abs(cY2))
    CourantNumber = dt*(abs_cX/(dx/float(nXi**2)) + abs_cY/(dy/float(nEta**2)))
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
    nElementsX = 5
    nElementsY = 5
    nXi = 10
    nEta = 10
    nXiPlot = 20
    nEtaPlot = 20
    if ProblemType == 'Plane_Gaussian_Wave':
        dt = 7.0*10.0**(-4.0)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        dt = 50.0
    elif ProblemType == 'Inertia_Gravity_Wave':
        dt = 23.0
    elif ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        dt = 39000.0
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        dt = 330.0
    elif ProblemType == 'Equatorial_Yanai_Wave':
        dt = 180.0
    elif ProblemType == 'Equatorial_Rossby_Wave':
        dt = 1200.0
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        dt = 108.0
    elif ProblemType == 'Barotropic_Tide':
        dt = 2.4
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        dt = 66.0
    CourantNumber = DetermineCourantNumberForGivenTimeStep(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nElementsX,nElementsY,nXi,nEta,nXiPlot,nEtaPlot,dt,PrintCourantNumber=True)
    myDGSEM2D = DGSEM2DClass.DGSEM2D(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                     Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,
                                     nEtaPlot,CourantNumber,UseCourantNumberToDetermineTimeStep=True)
    if ProblemType == 'Inertia_Gravity_Wave' or ProblemType == 'Planetary_Rossby_Wave':
        beta0 = myDGSEM2D.myNameList.myExactSolutionParameters.beta0
        c0 = myDGSEM2D.myNameList.myExactSolutionParameters.c0
        f0 = myDGSEM2D.myNameList.myExactSolutionParameters.f0
        kX1 = myDGSEM2D.myNameList.myExactSolutionParameters.kX1
        kX2 = myDGSEM2D.myNameList.myExactSolutionParameters.kX2
        kY1 = myDGSEM2D.myNameList.myExactSolutionParameters.kY1
        kY2 = myDGSEM2D.myNameList.myExactSolutionParameters.kY2
        lY = myDGSEM2D.myNameList.myExactSolutionParameters.lY
        k1 = np.sqrt(kX1**2.0 + kY1**2.0)
        k2 = np.sqrt(kX2**2.0 + kY2**2.0)
        if ProblemType == 'Inertia_Gravity_Wave':
            print('For the first wave mode, the ratio of f0:ck is %.6f.' %(f0/(c0*k1)))
            print('For the second wave mode, the ratio of f0:ck is %.6f.' %(f0/(c0*k2)))
        else:
            print('With the meridional extent being %.3f km, the ratio of beta0*lY:f0 is %.6f << 1.' 
                  %(lY/1000.0,beta0*lY/f0))
            
            
def DetermineNumberOfTimeStepsForSimulation(ProblemType):
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
    ProblemType_EquatorialWave = myDGSEM2D.myNameList.ProblemType_EquatorialWave
    dt = myDGSEM2D.myNameList.dt 
    lX = myDGSEM2D.myNameList.lX
    lY = myDGSEM2D.myNameList.lY
    cX1 = myDGSEM2D.myNameList.myExactSolutionParameters.cX1
    cX2 = myDGSEM2D.myNameList.myExactSolutionParameters.cX2
    cY1 = myDGSEM2D.myNameList.myExactSolutionParameters.cY1
    cY2 = myDGSEM2D.myNameList.myExactSolutionParameters.cY2
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
                            Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,nEtaPlot,
                            CheckStateVariableLimits,PlotFigures):
    myDGSEM2D = DGSEM2DClass.DGSEM2D(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                     Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,
                                     nEtaPlot)
    ExactZonalVelocityLimits = myDGSEM2D.myNameList.ExactZonalVelocityLimits
    ExactMeridionalVelocityLimits = myDGSEM2D.myNameList.ExactMeridionalVelocityLimits
    ExactSurfaceElevationLimits = myDGSEM2D.myNameList.ExactSurfaceElevationLimits
    if CheckStateVariableLimits:
        print('The limits of zonal velocity are [%.6f,%.6f].' 
              %(ExactZonalVelocityLimits[0],ExactZonalVelocityLimits[1]))
        print('The limits of meridional velocity are [%.6f,%.6f].' 
              %(ExactMeridionalVelocityLimits[0],ExactMeridionalVelocityLimits[1]))
        print('The limits of surface elevation are [%.6f,%.6f].' 
              %(ExactSurfaceElevationLimits[0],ExactSurfaceElevationLimits[1]))
        return
    nCounters = 2
    dt = myDGSEM2D.myNameList.dt
    nDumpFrequency = myDGSEM2D.myNameList.nDumpFrequency
    nTime = myDGSEM2D.myNameList.nTime
    if ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        ProblemType_PlanetaryTopographicRossbyWave = True
        ExactSurfaceElevationMaximumMagnitude = ExactSurfaceElevationLimits[1]
    else:
        ProblemType_PlanetaryTopographicRossbyWave = False
    if myDGSEM2D.myNameList.ProblemType_EquatorialWave and not(ProblemType == 'Equatorial_Kelvin_Wave'):
        HermiteFunctionMaximumAmplitude = (
        ESST.DetermineHermiteFunctionMaximumAmplitudeWithMeridionalLocation(ProblemType,ReturnMeridionalLocation=False))
        etaHat1 = myDGSEM2D.myNameList.myExactSolutionParameters.etaHat1
        etaHat2 = myDGSEM2D.myNameList.myExactSolutionParameters.etaHat2
        VelocityScale = myDGSEM2D.myNameList.myExactSolutionParameters.VelocityScale
        ExactMeridionalVelocityMaximumMagnitude = VelocityScale*HermiteFunctionMaximumAmplitude*(etaHat1 + etaHat2)
    PlotExactZonalVelocity = myDGSEM2D.myNameList.LogicalArrayPlot[0]
    PlotExactMeridionalVelocity = myDGSEM2D.myNameList.LogicalArrayPlot[1]
    PlotExactSurfaceElevation = myDGSEM2D.myNameList.LogicalArrayPlot[2]
    ProblemType_FileName = myDGSEM2D.myNameList.ProblemType_FileName
    for iCounter in range(0,nCounters):
        for iTime in range(0,nTime):
            myDGSEM2D.iTime = iTime
            myDGSEM2D.time = float(iTime)*dt
            if np.mod(iTime,nDumpFrequency) == 0.0:
                if iCounter == 0:                    
                    DGSEM2DClass.DetermineExactSolutionAtInteriorNodes(myDGSEM2D)
                    ExactZonalVelocities, ExactMeridionalVelocities, ExactSurfaceElevations = (
                    DGSEM2DClass.ExpressStateAtInteriorNodesAsArrays(myDGSEM2D,'Exact'))
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
                                if not(myDGSEM2D.myNameList.ProblemType_EquatorialWave 
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
                        DGSEM2DClass.WriteInterpolatedStateDGSEM2D(myDGSEM2D,FileName,ComputeOnlyExactSolution=True)
                        if iTime == nTime - 1:
                            if PlotExactZonalVelocity:
                                FileName = ProblemType_FileName + '_ExactZonalVelocityLimits'
                                CR.WriteStateVariableLimitsToFile(myDGSEM2D.OutputDirectory,ExactZonalVelocityLimits,
                                                                  FileName)
                            if PlotExactMeridionalVelocity:
                                FileName = ProblemType_FileName + '_ExactMeridionalVelocityLimits'
                                CR.WriteStateVariableLimitsToFile(myDGSEM2D.OutputDirectory,
                                                                  ExactMeridionalVelocityLimits,FileName)
                            if PlotExactSurfaceElevation:
                                FileName = ProblemType_FileName + '_ExactSurfaceElevationLimits'
                                CR.WriteStateVariableLimitsToFile(myDGSEM2D.OutputDirectory,ExactSurfaceElevationLimits,
                                                                  FileName)    
                else: # if iCounter == 1:
                    if PlotFigures:
                        FileName = ProblemType_FileName + '_ExactSolution_%3.3d' %iTime + '.tec'
                        DataType = 'Structured'
                        if ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Barotropic_Tide':
                            non_integral_seconds = True
                        else:
                            non_integral_seconds = False
                        DisplayTime = FormatSimulationTime(myDGSEM2D.time,non_integral_seconds=non_integral_seconds,
                                                           display_time=False,ProblemType_PlanetaryTopographicRossbyWave
                                                           =ProblemType_PlanetaryTopographicRossbyWave)
                        UseGivenColorBarLimits = True
                        ComputeOnlyExactSolution = True
                        SpecifyDataTypeInPlotFileName = False
                        DGSEM2DClass.PythonPlotStateDGSEM2D(myDGSEM2D,FileName,DataType,DisplayTime,
                                                            UseGivenColorBarLimits,ComputeOnlyExactSolution,
                                                            SpecifyDataTypeInPlotFileName)
                        
                        
def DetermineExactAndNumericalSolutions(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                        TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                        Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
                                        nElementsX,nElementsY,nXi,nEta,nXiPlot,nEtaPlot,CheckStateVariableLimits,
                                        PlotFigures,ComputeOnlyExactSolution=False,PlotNumericalSolution=False,
                                        Restart=False,Restart_iTime=0,Restart_FileName='',ReadFromSELFOutputData=False):
    myDGSEM2D = DGSEM2DClass.DGSEM2D(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                     Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,
                                     nEtaPlot,ReadFromSELFOutputData=ReadFromSELFOutputData)
    ProblemType_NoExactSolution = myDGSEM2D.myDGSEM2DParameters.ProblemType_NoExactSolution
    ExactZonalVelocityLimits = myDGSEM2D.myNameList.ExactZonalVelocityLimits
    ExactMeridionalVelocityLimits = myDGSEM2D.myNameList.ExactMeridionalVelocityLimits
    ExactSurfaceElevationLimits = myDGSEM2D.myNameList.ExactSurfaceElevationLimits
    if CheckStateVariableLimits:
        print('The limits of zonal velocity are [%.6f,%.6f].' 
              %(ExactZonalVelocityLimits[0],ExactZonalVelocityLimits[1]))
        print('The limits of meridional velocity are [%.6f,%.6f].' 
              %(ExactMeridionalVelocityLimits[0],ExactMeridionalVelocityLimits[1]))
        print('The limits of surface elevation are [%.6f,%.6f].' 
              %(ExactSurfaceElevationLimits[0],ExactSurfaceElevationLimits[1]))
        return
    nCounters = 2
    dt = myDGSEM2D.myNameList.dt
    nDumpFrequency = myDGSEM2D.myNameList.nDumpFrequency
    nRestartFrequency = myDGSEM2D.myNameList.nRestartFrequency
    nTime = myDGSEM2D.myNameList.nTime
    if ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        ProblemType_PlanetaryTopographicRossbyWave = True
        ExactSurfaceElevationMaximumMagnitude = ExactSurfaceElevationLimits[1]
    else:
        ProblemType_PlanetaryTopographicRossbyWave = False
    if myDGSEM2D.myNameList.ProblemType_EquatorialWave and not(ProblemType == 'Equatorial_Kelvin_Wave'):
        HermiteFunctionMaximumAmplitude = (
        ESST.DetermineHermiteFunctionMaximumAmplitudeWithMeridionalLocation(ProblemType,ReturnMeridionalLocation=False))
        etaHat1 = myDGSEM2D.myNameList.myExactSolutionParameters.etaHat1
        etaHat2 = myDGSEM2D.myNameList.myExactSolutionParameters.etaHat2
        VelocityScale = myDGSEM2D.myNameList.myExactSolutionParameters.VelocityScale
        ExactMeridionalVelocityMaximumMagnitude = VelocityScale*HermiteFunctionMaximumAmplitude*(etaHat1 + etaHat2)
    PlotExactZonalVelocity = myDGSEM2D.myNameList.LogicalArrayPlot[0]
    PlotExactMeridionalVelocity = myDGSEM2D.myNameList.LogicalArrayPlot[1]
    PlotExactSurfaceElevation = myDGSEM2D.myNameList.LogicalArrayPlot[2]
    ProblemType_FileName = myDGSEM2D.myNameList.ProblemType_FileName
    TimeIntegratorShortForm = myDGSEM2D.myNameList.myTimeSteppingParameters.TimeIntegratorShortForm
    if ReadFromSELFOutputData:
        iTimeFormat = '%8.8d'
    else:
        iTimeFormat = '%3.3d'
    if Restart:
        iTime_Start = Restart_iTime
    else:
        iTime_Start = 0
    for iCounter in range(0,nCounters):
        for iTime in range(iTime_Start,nTime):
            myDGSEM2D.iTime = iTime
            myDGSEM2D.time = float(iTime)*dt
            if iCounter == 0: 
                if np.mod(iTime,nDumpFrequency) == 0.0:    
                    DGSEM2DClass.DetermineExactSolutionAtInteriorNodes(myDGSEM2D)
                    ExactZonalVelocities, ExactMeridionalVelocities, ExactSurfaceElevations = (
                    DGSEM2DClass.ExpressStateAtInteriorNodesAsArrays(myDGSEM2D,'Exact'))
                    if not(ComputeOnlyExactSolution):
                        if iTime == iTime_Start:
                            if Restart:
                                u, v, eta = DGSEM2DClass.ReadStateDGSEM2D(myDGSEM2D,Restart_FileName+'.tec')
                                DGSEM2DClass.SpecifyRestartConditions(myDGSEM2D,u,v,eta)
                            else:
                                DGSEM2DClass.SpecifyInitialConditions(myDGSEM2D)
                        DGSEM2DClass.ComputeError(myDGSEM2D)
                        ZonalVelocityError, MeridionalVelocityError, SurfaceElevationError = (
                        DGSEM2DClass.ExpressStateAtInteriorNodesAsArrays(myDGSEM2D,'Error'))
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
                                if not(myDGSEM2D.myNameList.ProblemType_EquatorialWave 
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
                        DGSEM2DClass.WriteInterpolatedStateDGSEM2D(myDGSEM2D,FileName,ComputeOnlyExactSolution)
                        if not(ComputeOnlyExactSolution) and np.mod(iTime,nRestartFrequency) == 0.0:
                            FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                        + '_RestartState_' + iTimeFormat %iTime)
                            DGSEM2DClass.WriteStateDGSEM2D(myDGSEM2D,FileName)
                        if iTime == nTime - 1:
                            if PlotExactZonalVelocity:
                                FileName = ProblemType_FileName + '_ExactZonalVelocityLimits'
                                CR.WriteStateVariableLimitsToFile(myDGSEM2D.OutputDirectory,ExactZonalVelocityLimits,
                                                                  FileName)
                                if not(ComputeOnlyExactSolution):
                                    FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                                + '_ZonalVelocityErrorLimits')
                                    CR.WriteStateVariableLimitsToFile(myDGSEM2D.OutputDirectory,
                                                                      ZonalVelocityErrorLimits,FileName)
                            if PlotExactMeridionalVelocity:
                                FileName = ProblemType_FileName + '_ExactMeridionalVelocityLimits'
                                CR.WriteStateVariableLimitsToFile(myDGSEM2D.OutputDirectory,
                                                                  ExactMeridionalVelocityLimits,FileName)
                                if not(ComputeOnlyExactSolution):
                                    FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm
                                                + '_MeridionalVelocityErrorLimits')
                                    CR.WriteStateVariableLimitsToFile(myDGSEM2D.OutputDirectory,
                                                                      MeridionalVelocityErrorLimits,FileName)
                            if PlotExactSurfaceElevation:
                                FileName = ProblemType_FileName + '_ExactSurfaceElevationLimits'
                                CR.WriteStateVariableLimitsToFile(myDGSEM2D.OutputDirectory,ExactSurfaceElevationLimits,
                                                                  FileName)    
                                if not(ComputeOnlyExactSolution):
                                    FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                                + '_SurfaceElevationErrorLimits')
                                    CR.WriteStateVariableLimitsToFile(myDGSEM2D.OutputDirectory,
                                                                      SurfaceElevationErrorLimits,FileName)
                if not(ComputeOnlyExactSolution) and iTime < nTime - 1:
                    TSM.TimeIntegration(myDGSEM2D)
            else: # if iCounter == 1:
                if np.mod(iTime,nDumpFrequency) == 0.0:
                    if PlotFigures:
                        if ComputeOnlyExactSolution:
                            FileName = ProblemType_FileName + '_State_' + iTimeFormat %iTime + '.tec'
                        else:
                            FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_State_' 
                                        + iTimeFormat %iTime + '.tec')
                        if ReadFromSELFOutputData:
                            DataType = 'Unstructured'
                        else:
                            DataType = 'Structured'
                        if ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Barotropic_Tide':
                            non_integral_seconds = True
                        else:
                            non_integral_seconds = False
                        DisplayTime = FormatSimulationTime(myDGSEM2D.time,non_integral_seconds=non_integral_seconds,
                                                           display_time=False,ProblemType_PlanetaryTopographicRossbyWave
                                                           =ProblemType_PlanetaryTopographicRossbyWave,
                                                           ProblemType_NoExactSolution=ProblemType_NoExactSolution)
                        UseGivenColorBarLimits = True
                        SpecifyDataTypeInPlotFileName = False
                        DGSEM2DClass.PythonPlotStateDGSEM2D(myDGSEM2D,FileName,DataType,DisplayTime,
                                                            UseGivenColorBarLimits,ComputeOnlyExactSolution,
                                                            SpecifyDataTypeInPlotFileName,PlotNumericalSolution)