"""
Name: TimeSteppingMethods.py
Author: Siddhartha Bishnu
Details: This script contains various time-stepping methods for advancing the two-dimensional rotating shallow water 
equations.
"""


import numpy as np
from IPython.utils import io
with io.capture_output() as captured:
    import ExactSolutionsAndSourceTerms as ESST
    
    
def CaptureSolutions(myMPASOceanShallowWater,PrognosticVariables='NormalVelocityAndSurfaceElevation'):
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    nCells = myMPASOceanShallowWater.myMesh.nCells
    if PrognosticVariables == 'NormalVelocityAndSurfaceElevation' or PrognosticVariables == 'NormalVelocity':
        NormalVelocities = np.zeros(nEdges)
        for iEdge in range(0,nEdges):
            NormalVelocities[iEdge] = myMPASOceanShallowWater.mySolution.normalVelocity[iEdge]
    if PrognosticVariables == 'NormalVelocityAndSurfaceElevation' or PrognosticVariables == 'SurfaceElevation':
        SurfaceElevations = np.zeros(nCells)
        for iCell in range(0,nCells):
            SurfaceElevations[iCell] = (
            myMPASOceanShallowWater.mySolution.ssh[iCell])   
    if PrognosticVariables == 'NormalVelocityAndSurfaceElevation':
        return NormalVelocities, SurfaceElevations
    elif PrognosticVariables == 'NormalVelocity':
        return NormalVelocities    
    elif PrognosticVariables == 'SurfaceElevation':
        return SurfaceElevations
    

def CaptureTendencies(myMPASOceanShallowWater,PrognosticVariables='NormalVelocityAndSurfaceElevation',
                      TimeLevel='Current'):
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    nCells = myMPASOceanShallowWater.myMesh.nCells
    if PrognosticVariables == 'NormalVelocityAndSurfaceElevation' or PrognosticVariables == 'NormalVelocity':
        NormalVelocityTendencies = np.zeros(nEdges)
        for iEdge in range(0,nEdges):
            if TimeLevel == 'Current':
                NormalVelocityTendencies[iEdge] = myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge]
            elif TimeLevel == 'Last':
                NormalVelocityTendencies[iEdge]  = myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge]
            elif TimeLevel == 'SecondLast':
                NormalVelocityTendencies[iEdge]  = (
                myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[iEdge])
    if PrognosticVariables == 'NormalVelocityAndSurfaceElevation' or PrognosticVariables == 'SurfaceElevation':
        SurfaceElevationTendencies = np.zeros(nCells)
        for iCell in range(0,nCells):
            if TimeLevel == 'Current':
                SurfaceElevationTendencies[iCell] = myMPASOceanShallowWater.mySolution.sshTendency[iCell]
            elif TimeLevel == 'Last':
                SurfaceElevationTendencies[iCell] = myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell]
            elif TimeLevel == 'SecondLast':
                SurfaceElevationTendencies[iCell] = myMPASOceanShallowWater.mySolution.sshTendencySecondLast[iCell]
    if PrognosticVariables == 'NormalVelocityAndSurfaceElevation':
        return NormalVelocityTendencies, SurfaceElevationTendencies
    elif PrognosticVariables == 'NormalVelocity':
        return NormalVelocityTendencies    
    elif PrognosticVariables == 'SurfaceElevation':
        return SurfaceElevationTendencies
    

def ShiftTendencies(myMPASOceanShallowWater):
    TimeIntegrator = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.TimeIntegrator
    if TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
        myMPASOceanShallowWater.mySolution.normalVelocityTendencyThirdLast[:] = (
        myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[:])
        myMPASOceanShallowWater.mySolution.sshTendencyThirdLast[:] = (
        myMPASOceanShallowWater.mySolution.sshTendencySecondLast[:])
    if TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' or TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
        myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[:] = (
        myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[:])
        myMPASOceanShallowWater.mySolution.sshTendencySecondLast[:] = (
        myMPASOceanShallowWater.mySolution.sshTendencyLast[:])
    if (TimeIntegrator == 'SecondOrderAdamsBashforthMethod' or TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' 
        or TimeIntegrator == 'FourthOrderAdamsBashforthMethod'):
        myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[:] = (
        myMPASOceanShallowWater.mySolution.normalVelocityTendency[:])
        myMPASOceanShallowWater.mySolution.sshTendencyLast[:] = myMPASOceanShallowWater.mySolution.sshTendency[:]


def ForwardEulerMethod(myMPASOceanShallowWater):
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    nCells = myMPASOceanShallowWater.myMesh.nCells
    dt = myMPASOceanShallowWater.myNameList.dt
    time = myMPASOceanShallowWater.time
    myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
    myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
    for iEdge in range(0,nEdges):
        x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
        y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
        dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
        angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
        if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
            # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact counterparts.
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
            ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,UseAveragedQuantities,
                                              myQuadratureOnEdge,dvEdge,angleEdge))
        else:
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += (
            dt*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
    for iCell in range(0,nCells):
        myMPASOceanShallowWater.mySolution.ssh[iCell] += dt*myMPASOceanShallowWater.mySolution.sshTendency[iCell]


def ExplicitMidpointMethod(myMPASOceanShallowWater,TimeIntegrator='ExplicitMidpointMethod'):
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    nCells = myMPASOceanShallowWater.myMesh.nCells
    dt = myMPASOceanShallowWater.myNameList.dt
    NormalVelocitiesCurrent, SurfaceElevationsCurrent = CaptureSolutions(myMPASOceanShallowWater)
    time = myMPASOceanShallowWater.time
    myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
    myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
    for iEdge in range(0,nEdges):
        if (TimeIntegrator == 'LeapfrogTrapezoidalMethod' or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback'):
            myMPASOceanShallowWater.mySolution.normalVelocityLast[iEdge] = NormalVelocitiesCurrent[iEdge]
        if (TimeIntegrator == 'SecondOrderAdamsBashforthMethod' 
            or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback'):
            myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
            myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
        elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step':
            if myMPASOceanShallowWater.iTime == 0:
                myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[iEdge] = (
                myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
            elif myMPASOceanShallowWater.iTime == 1:
                myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
                myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
        x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
        y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
        dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
        angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
        if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
            # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact counterparts.
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
            ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+0.5*dt,
                                              UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
        else:
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += (
            0.5*dt*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
    for iCell in range(0,nCells):
        if (TimeIntegrator == 'LeapfrogTrapezoidalMethod' or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback'):
            myMPASOceanShallowWater.mySolution.sshLast[iCell] = SurfaceElevationsCurrent[iCell]
        if (TimeIntegrator == 'SecondOrderAdamsBashforthMethod' 
            or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback'):
            myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = (
            myMPASOceanShallowWater.mySolution.sshTendency[iCell])
        elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step':
            if myMPASOceanShallowWater.iTime == 0:
                myMPASOceanShallowWater.mySolution.sshTendencySecondLast[iCell] = (
                myMPASOceanShallowWater.mySolution.sshTendency[iCell])
            elif myMPASOceanShallowWater.iTime == 1:
                myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = (
                myMPASOceanShallowWater.mySolution.sshTendency[iCell])
        myMPASOceanShallowWater.mySolution.ssh[iCell] += 0.5*dt*myMPASOceanShallowWater.mySolution.sshTendency[iCell]
    myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time+0.5*dt)
    myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+0.5*dt)
    for iEdge in range(0,nEdges):
        x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
        y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
        dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
        angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
        if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
            # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact counterparts.
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
            ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                              UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
        else:
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
            NormalVelocitiesCurrent[iEdge] + dt*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
    for iCell in range(0,nCells):
        myMPASOceanShallowWater.mySolution.ssh[iCell] = (SurfaceElevationsCurrent[iCell] 
                                                         + dt*myMPASOceanShallowWater.mySolution.sshTendency[iCell])


def WilliamsonLowStorageThirdOrderRungeKuttaMethod(
myMPASOceanShallowWater,TimeIntegrator='WilliamsonLowStorageThirdOrderRungeKuttaMethod'):
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    nCells = myMPASOceanShallowWater.myMesh.nCells
    dt = myMPASOceanShallowWater.myNameList.dt
    nStepsRK3 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.nStepsRK3
    aRK3 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.aRK3
    bRK3 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.bRK3
    bRK3Next = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.bRK3Next
    gRK3 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.gRK3
    TemporaryNormalVelocityTendency = np.zeros(nEdges)
    TemporarySurfaceElevationTendency = np.zeros(nCells)
    for iStep in range(0,nStepsRK3):
        time = myMPASOceanShallowWater.time + bRK3[iStep]*dt
        timeNext = myMPASOceanShallowWater.time + bRK3Next[iStep]*dt
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
        for iEdge in range(0,nEdges):
            if TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' and iStep == 0:
                myMPASOceanShallowWater.mySolution.normalVelocityLast[iEdge] = (
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge])
            if ((TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' 
                 or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step') and iStep == 0):
                myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
                myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
            elif ((TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' 
                   or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step') and iStep == 0):
                if myMPASOceanShallowWater.iTime == 0:
                    myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[iEdge] = (
                    myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
                elif myMPASOceanShallowWater.iTime == 1:
                    myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
                    myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])   
            TemporaryNormalVelocityTendency[iEdge] = (
            (aRK3[iStep]*TemporaryNormalVelocityTendency[iEdge] 
             + myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge]))
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,timeNext,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += (
                gRK3[iStep]*dt*TemporaryNormalVelocityTendency[iEdge])
        for iCell in range(0,nCells):
            if TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' and iStep == 0:
                myMPASOceanShallowWater.mySolution.sshLast[iCell] = myMPASOceanShallowWater.mySolution.ssh[iCell]
            if ((TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' 
                 or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step') and iStep == 0):
                myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = (
                myMPASOceanShallowWater.mySolution.sshTendency[iCell])
            elif ((TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' 
                   or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step') and iStep == 0):
                if myMPASOceanShallowWater.iTime == 0:
                    myMPASOceanShallowWater.mySolution.sshTendencySecondLast[iCell] = (
                    myMPASOceanShallowWater.mySolution.sshTendency[iCell])
                elif myMPASOceanShallowWater.iTime == 1:
                    myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = (
                    myMPASOceanShallowWater.mySolution.sshTendency[iCell])   
            TemporarySurfaceElevationTendency[iCell] = (aRK3[iStep]*TemporarySurfaceElevationTendency[iCell] 
                                                        + myMPASOceanShallowWater.mySolution.sshTendency[iCell])
            myMPASOceanShallowWater.mySolution.ssh[iCell] += gRK3[iStep]*dt*TemporarySurfaceElevationTendency[iCell]
            
            
def CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(
myMPASOceanShallowWater,TimeIntegrator='CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod'):
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    nCells = myMPASOceanShallowWater.myMesh.nCells
    dt = myMPASOceanShallowWater.myNameList.dt
    nStepsRK4 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.nStepsRK4
    aRK4 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.aRK4
    bRK4 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.bRK4
    bRK4Next = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.bRK4Next
    gRK4 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.gRK4
    TemporaryNormalVelocityTendency = np.zeros(nEdges)
    TemporarySurfaceElevationTendency = np.zeros(nCells)
    for iStep in range(0,nStepsRK4):
        time = myMPASOceanShallowWater.time + bRK4[iStep]*dt
        timeNext = myMPASOceanShallowWater.time + bRK4Next[iStep]*dt
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
        for iEdge in range(0,nEdges):
            if TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' and iStep == 0:
                myMPASOceanShallowWater.mySolution.normalVelocityLast[iEdge] = (
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge])
            if ((TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' 
                 or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step') and iStep == 0):
                myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
                myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
            elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step' and iStep == 0:
                if myMPASOceanShallowWater.iTime == 0:
                    myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[iEdge] = (
                    myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
                elif myMPASOceanShallowWater.iTime == 1:
                    myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
                    myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
            elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod' and iStep == 0:
                if myMPASOceanShallowWater.iTime == 0:
                    myMPASOceanShallowWater.mySolution.normalVelocityTendencyThirdLast[iEdge] = (
                    myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
                elif myMPASOceanShallowWater.iTime == 1:
                    myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[iEdge] = (
                    myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
                elif myMPASOceanShallowWater.iTime == 2:
                    myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
                    myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
            TemporaryNormalVelocityTendency[iEdge] = (
            (aRK4[iStep]*TemporaryNormalVelocityTendency[iEdge] 
             + myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge]))
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,timeNext,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += (
                gRK4[iStep]*dt*TemporaryNormalVelocityTendency[iEdge])
        for iCell in range(0,nCells):
            if TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' and iStep == 0:
                myMPASOceanShallowWater.mySolution.sshLast[iCell] = myMPASOceanShallowWater.mySolution.ssh[iCell]
            if ((TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' 
                 or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step') and iStep == 0):
                myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = (
                myMPASOceanShallowWater.mySolution.sshTendency[iCell])
            elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step' and iStep == 0:
                if myMPASOceanShallowWater.iTime == 0:
                    myMPASOceanShallowWater.mySolution.sshTendencySecondLast[iCell] = (
                    myMPASOceanShallowWater.mySolution.sshTendency[iCell])
                elif myMPASOceanShallowWater.iTime == 1:
                    myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = (
                    myMPASOceanShallowWater.mySolution.sshTendency[iCell])
            elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod' and iStep == 0:
                if myMPASOceanShallowWater.iTime == 0:
                    myMPASOceanShallowWater.mySolution.sshTendencyThirdLast[iCell] = (
                    myMPASOceanShallowWater.mySolution.sshTendency[iCell])
                elif myMPASOceanShallowWater.iTime == 1:
                    myMPASOceanShallowWater.mySolution.sshTendencySecondLast[iCell] = (
                    myMPASOceanShallowWater.mySolution.sshTendency[iCell])
                elif myMPASOceanShallowWater.iTime == 2:
                    myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = (
                    myMPASOceanShallowWater.mySolution.sshTendency[iCell])
            TemporarySurfaceElevationTendency[iCell] = (aRK4[iStep]*TemporarySurfaceElevationTendency[iCell] 
                                                        + myMPASOceanShallowWater.mySolution.sshTendency[iCell])
            myMPASOceanShallowWater.mySolution.ssh[iCell] += gRK4[iStep]*dt*TemporarySurfaceElevationTendency[iCell]

        
def SecondOrderAdamsBashforthMethod(myMPASOceanShallowWater):
    if myMPASOceanShallowWater.iTime == 0:
        ExplicitMidpointMethod(myMPASOceanShallowWater,TimeIntegrator='SecondOrderAdamsBashforthMethod')
    else:
        AB2 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.AB2
        ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
        myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
        UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
        myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
        nEdges = myMPASOceanShallowWater.myMesh.nEdges
        nCells = myMPASOceanShallowWater.myMesh.nCells
        dt = myMPASOceanShallowWater.myNameList.dt
        time = myMPASOceanShallowWater.time
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
        for iEdge in range(0,nEdges):
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += (
                dt*(AB2[0]*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge]
                    + AB2[1]*myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge]))
        for iCell in range(0,nCells):
            myMPASOceanShallowWater.mySolution.ssh[iCell] += (
            dt*(AB2[0]*myMPASOceanShallowWater.mySolution.sshTendency[iCell]
                + AB2[1]*myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell]))       
        ShiftTendencies(myMPASOceanShallowWater)
    

def ThirdOrderAdamsBashforthMethod(myMPASOceanShallowWater):
    if myMPASOceanShallowWater.iTime == 0 or myMPASOceanShallowWater.iTime == 1:
        WilliamsonLowStorageThirdOrderRungeKuttaMethod(myMPASOceanShallowWater,
                                                       TimeIntegrator='ThirdOrderAdamsBashforthMethod')
    else:
        AB3 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.AB3
        ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
        myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
        UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
        myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
        nEdges = myMPASOceanShallowWater.myMesh.nEdges
        nCells = myMPASOceanShallowWater.myMesh.nCells
        dt = myMPASOceanShallowWater.myNameList.dt
        time = myMPASOceanShallowWater.time
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
        for iEdge in range(0,nEdges):
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += (
                dt*(AB3[0]*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge]
                    + AB3[1]*myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge]
                    + AB3[2]*myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[iEdge]))
        for iCell in range(0,nCells):
            myMPASOceanShallowWater.mySolution.ssh[iCell] += (
            dt*(AB3[0]*myMPASOceanShallowWater.mySolution.sshTendency[iCell]
                + AB3[1]*myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell]
                + AB3[2]*myMPASOceanShallowWater.mySolution.sshTendencySecondLast[iCell]))
        ShiftTendencies(myMPASOceanShallowWater)
                        
                        
def FourthOrderAdamsBashforthMethod(myMPASOceanShallowWater):
    if myMPASOceanShallowWater.iTime == 0 or myMPASOceanShallowWater.iTime == 1 or myMPASOceanShallowWater.iTime == 2:
        CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(myMPASOceanShallowWater,
                                                              TimeIntegrator='FourthOrderAdamsBashforthMethod')
    else:
        AB4 = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.AB4
        ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
        myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
        UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
        myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
        nEdges = myMPASOceanShallowWater.myMesh.nEdges
        nCells = myMPASOceanShallowWater.myMesh.nCells
        dt = myMPASOceanShallowWater.myNameList.dt
        time = myMPASOceanShallowWater.time
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
        for iEdge in range(0,nEdges):
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += (
                dt*(AB4[0]*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge]
                    + AB4[1]*myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge]
                    + AB4[2]*myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[iEdge]
                    + AB4[3]*myMPASOceanShallowWater.mySolution.normalVelocityTendencyThirdLast[iEdge]))
        for iCell in range(0,nCells):
            myMPASOceanShallowWater.mySolution.ssh[iCell] += (
            dt*(AB4[0]*myMPASOceanShallowWater.mySolution.sshTendency[iCell]
                + AB4[1]*myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell]
                + AB4[2]*myMPASOceanShallowWater.mySolution.sshTendencySecondLast[iCell]
                + AB4[3]*myMPASOceanShallowWater.mySolution.sshTendencyThirdLast[iCell]))
        ShiftTendencies(myMPASOceanShallowWater)

        
def LeapfrogTrapezoidalMethod(myMPASOceanShallowWater):
    if myMPASOceanShallowWater.iTime == 0:
        ExplicitMidpointMethod(myMPASOceanShallowWater,TimeIntegrator='LeapfrogTrapezoidalMethod')
    else:
        ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
        myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
        UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
        myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
        nEdges = myMPASOceanShallowWater.myMesh.nEdges
        nCells = myMPASOceanShallowWater.myMesh.nCells
        dt = myMPASOceanShallowWater.myNameList.dt
        NormalVelocitiesCurrent, SurfaceElevationsCurrent = CaptureSolutions(myMPASOceanShallowWater)
        time = myMPASOceanShallowWater.time
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
        NormalVelocityTendenciesCurrent, SurfaceElevationTendenciesCurrent = CaptureTendencies(myMPASOceanShallowWater)
        for iEdge in range(0,nEdges):
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                (myMPASOceanShallowWater.mySolution.normalVelocityLast[iEdge]
                 + 2.0*dt*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge]))
                myMPASOceanShallowWater.mySolution.normalVelocityLast[iEdge] = NormalVelocitiesCurrent[iEdge]
        for iCell in range(0,nCells):
            myMPASOceanShallowWater.mySolution.ssh[iCell] = (
            (myMPASOceanShallowWater.mySolution.sshLast[iCell] 
             + 2.0*dt*myMPASOceanShallowWater.mySolution.sshTendency[iCell]))
            myMPASOceanShallowWater.mySolution.sshLast[iCell] = SurfaceElevationsCurrent[iCell]
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time+dt)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt)
        NormalVelocityTendenciesNew, SurfaceElevationTendenciesNew = CaptureTendencies(myMPASOceanShallowWater)
        NormalVelocityTendencies = 0.5*(NormalVelocityTendenciesCurrent + NormalVelocityTendenciesNew)
        SurfaceElevationTendencies = 0.5*(SurfaceElevationTendenciesCurrent + SurfaceElevationTendenciesNew) 
        for iEdge in range(0,nEdges):
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                NormalVelocitiesCurrent[iEdge] + dt*NormalVelocityTendencies[iEdge])
        for iCell in range(0,nCells):
            myMPASOceanShallowWater.mySolution.ssh[iCell] = (
            SurfaceElevationsCurrent[iCell] + dt*SurfaceElevationTendencies[iCell])
                        
                        
def LFTRAndLFAM3MethodWithFBFeedback(myMPASOceanShallowWater):
    if myMPASOceanShallowWater.iTime == 0:
        if (myMPASOceanShallowWater.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
            == 'SecondOrderAccurate_LF_TR'):
            ExplicitMidpointMethod(myMPASOceanShallowWater,
                                   TimeIntegrator='LFTRAndLFAM3MethodWithFBFeedback')            
        elif ((myMPASOceanShallowWater.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
               == 'ThirdOrderAccurate_LF_AM3')
              or (myMPASOceanShallowWater.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
                  == 'ThirdOrderAccurate_MaximumStabilityRange')):
            WilliamsonLowStorageThirdOrderRungeKuttaMethod(myMPASOceanShallowWater,
                                                           TimeIntegrator='LFTRAndLFAM3MethodWithFBFeedback')  
        elif ((myMPASOceanShallowWater.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
               == 'FourthOrderAccurate_MinimumTruncationError')
              or (myMPASOceanShallowWater.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
                  == 'FourthOrderAccurate_MaximumStabilityRange')):
            CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(myMPASOceanShallowWater,
                                                                  TimeIntegrator='LFTRAndLFAM3MethodWithFBFeedback')
    else:
        beta = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                .LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta)
        gamma = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                 .LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma)
        epsilon = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                   .LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon)
        ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
        myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
        UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
        myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
        nEdges = myMPASOceanShallowWater.myMesh.nEdges
        nCells = myMPASOceanShallowWater.myMesh.nCells
        dt = myMPASOceanShallowWater.myNameList.dt
        NormalVelocityTendenciesLast, SurfaceElevationTendenciesLast = CaptureTendencies(myMPASOceanShallowWater,
                                                                                         TimeLevel='Last')
        NormalVelocitiesCurrent, SurfaceElevationsCurrent = CaptureSolutions(myMPASOceanShallowWater)
        time = myMPASOceanShallowWater.time
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
        NormalVelocityTendenciesCurrent, SurfaceElevationTendenciesCurrent = CaptureTendencies(myMPASOceanShallowWater)
        for iEdge in range(0,nEdges):
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                (myMPASOceanShallowWater.mySolution.normalVelocityLast[iEdge]
                 + 2.0*dt*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge]))
                myMPASOceanShallowWater.mySolution.normalVelocityLast[iEdge] = NormalVelocitiesCurrent[iEdge]
                myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
                NormalVelocityTendenciesCurrent[iEdge])
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt)
        SurfaceElevationTendencies_3 = CaptureTendencies(myMPASOceanShallowWater,
                                                         PrognosticVariables='SurfaceElevation') 
        SurfaceElevationTendencies_2 = SurfaceElevationTendenciesCurrent
        SurfaceElevationTendencies_1 = SurfaceElevationTendenciesLast
        SurfaceElevationTendencies = ((1.0 - 2.0*beta)*SurfaceElevationTendencies_2
                                      + beta*(SurfaceElevationTendencies_3 + SurfaceElevationTendencies_1))
        for iCell in range(0,nCells):
            myMPASOceanShallowWater.mySolution.ssh[iCell] = (
            (myMPASOceanShallowWater.mySolution.sshLast[iCell] 
             + 2.0*dt*myMPASOceanShallowWater.mySolution.sshTendency[iCell]))
            myMPASOceanShallowWater.mySolution.sshLast[iCell] = SurfaceElevationsCurrent[iCell]
            myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = SurfaceElevationTendenciesCurrent[iCell]
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time+dt)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt)
        NormalVelocityTendencies_3, SurfaceElevationTendencies_3 = CaptureTendencies(myMPASOceanShallowWater)
        NormalVelocityTendencies_2 = NormalVelocityTendenciesCurrent
        NormalVelocityTendencies_1 = NormalVelocityTendenciesLast
        NormalVelocityTendencies = (-gamma*NormalVelocityTendencies_1 + (0.5 + 2.0*gamma)*NormalVelocityTendencies_2 
                                    + (0.5 - gamma)*NormalVelocityTendencies_3)
        for iEdge in range(0,nEdges):
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                NormalVelocitiesCurrent[iEdge] + dt*NormalVelocityTendencies[iEdge])
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt)
        SurfaceElevationTendencies_4 = CaptureTendencies(myMPASOceanShallowWater,PrognosticVariables='SurfaceElevation')
        SurfaceElevationTendencies = (
        ((0.5 - gamma)*(epsilon*SurfaceElevationTendencies_4 + (1.0 - epsilon)*SurfaceElevationTendencies_3) 
         + (0.5 + 2.0*gamma)*SurfaceElevationTendencies_2 - gamma*SurfaceElevationTendencies_1))
        for iCell in range(0,nCells):
            myMPASOceanShallowWater.mySolution.ssh[iCell] = (
            SurfaceElevationsCurrent[iCell] + dt*SurfaceElevationTendencies[iCell])

        
def ForwardBackwardMethod(myMPASOceanShallowWater):
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    nCells = myMPASOceanShallowWater.myMesh.nCells
    dt = myMPASOceanShallowWater.myNameList.dt
    time = myMPASOceanShallowWater.time
    myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
    for iEdge in range(0,nEdges):
        x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
        y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
        dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
        angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
        if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
            # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact counterparts.
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
            ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,UseAveragedQuantities,
                                              myQuadratureOnEdge,dvEdge,angleEdge))
        else:
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += (
            dt*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
    myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt)
    for iCell in range(0,nCells):
        myMPASOceanShallowWater.mySolution.ssh[iCell] += dt*myMPASOceanShallowWater.mySolution.sshTendency[iCell]
                    
                    
def ForwardBackwardMethodWithRK2Feedback(myMPASOceanShallowWater):
    beta = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.Forward_Backward_with_RK2_Feedback_parameter_beta
    epsilon = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
               .Forward_Backward_with_RK2_Feedback_parameter_epsilon)
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    nCells = myMPASOceanShallowWater.myMesh.nCells
    NormalVelocities_1, SurfaceElevations_1 = CaptureSolutions(myMPASOceanShallowWater)
    dt = myMPASOceanShallowWater.myNameList.dt
    time = myMPASOceanShallowWater.time
    myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
    myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
    NormalVelocityTendencies_1, SurfaceElevationTendencies_1 = CaptureTendencies(myMPASOceanShallowWater)
    for iEdge in range(0,nEdges):
        x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
        y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
        dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
        angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
        if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
            # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact counterparts.
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
            ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,UseAveragedQuantities,
                                              myQuadratureOnEdge,dvEdge,angleEdge))
        else:
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += (
            dt*myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge])
    myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt)
    SurfaceElevationTendencies_2 = CaptureTendencies(myMPASOceanShallowWater,
                                                     PrognosticVariables='SurfaceElevation')           
    SurfaceElevationTendencies = (1.0 - beta)*SurfaceElevationTendencies_1 + beta*SurfaceElevationTendencies_2
    for iCell in range(0,nCells):
        myMPASOceanShallowWater.mySolution.ssh[iCell] += dt*SurfaceElevationTendencies[iCell]
    myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time+dt)
    myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt)
    NormalVelocityTendencies_2, SurfaceElevationTendencies_2 = CaptureTendencies(myMPASOceanShallowWater)           
    NormalVelocityTendencies = 0.5*(NormalVelocityTendencies_1 + NormalVelocityTendencies_2)    
    for iEdge in range(0,nEdges):
        x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
        y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
        dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
        angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
        if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
            # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact counterparts.
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
            ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,UseAveragedQuantities,
                                              myQuadratureOnEdge,dvEdge,angleEdge))
        else:
            myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
            NormalVelocities_1[iEdge] + dt*NormalVelocityTendencies[iEdge])
    myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt)
    SurfaceElevationTendencies_3 = CaptureTendencies(myMPASOceanShallowWater,PrognosticVariables='SurfaceElevation')
    SurfaceElevationTendencies = 0.5*(SurfaceElevationTendencies_1 + (1.0 - epsilon)*SurfaceElevationTendencies_2 
                                      + epsilon*SurfaceElevationTendencies_3)           
    for iCell in range(0,nCells):
        myMPASOceanShallowWater.mySolution.ssh[iCell] = SurfaceElevations_1 + dt*SurfaceElevationTendencies[iCell]

    
def GeneralizedForwardBackwardMethodWithAB2AM3Step(myMPASOceanShallowWater):
    if myMPASOceanShallowWater.iTime == 0:
        if ((myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_Type 
             == 'ThirdOrderAccurate_WideStabilityRange')
              or (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_Type 
                  == 'ThirdOrderAccurate_WeakAsymptoticInstabilityOfPhysicalModes')):
            WilliamsonLowStorageThirdOrderRungeKuttaMethod(
            myMPASOceanShallowWater,TimeIntegrator='GeneralizedForwardBackwardMethodWithAB2AM3Step')
        elif (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_Type 
              == 'FourthOrderAccurate'):
            CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(
            myMPASOceanShallowWater,TimeIntegrator='GeneralizedForwardBackwardMethodWithAB2AM3Step')
    else:
        beta = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                .Generalized_FB_with_AB2_AM3_Step_parameter_beta)
        gamma = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                 .Generalized_FB_with_AB2_AM3_Step_parameter_gamma)
        epsilon = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                   .Generalized_FB_with_AB2_AM3_Step_parameter_epsilon)
        ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
        myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
        UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
        myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
        nEdges = myMPASOceanShallowWater.myMesh.nEdges
        nCells = myMPASOceanShallowWater.myMesh.nCells
        dt = myMPASOceanShallowWater.myNameList.dt
        NormalVelocityTendenciesLast, SurfaceElevationTendenciesLast = CaptureTendencies(myMPASOceanShallowWater,
                                                                                         TimeLevel='Last')
        time = myMPASOceanShallowWater.time
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
        NormalVelocityTendenciesCurrent, SurfaceElevationTendenciesCurrent = CaptureTendencies(myMPASOceanShallowWater)
        NormalVelocityTendencies_2 = NormalVelocityTendenciesCurrent
        NormalVelocityTendencies_1 = NormalVelocityTendenciesLast
        NormalVelocityTendencies = -beta*NormalVelocityTendencies_1 + (1.0 + beta)*NormalVelocityTendencies_2
        for iEdge in range(0,nEdges):
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += dt*NormalVelocityTendencies[iEdge]
                myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
                NormalVelocityTendenciesCurrent[iEdge])
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt) 
        SurfaceElevationTendencies_3 = CaptureTendencies(myMPASOceanShallowWater,PrognosticVariables='SurfaceElevation')
        SurfaceElevationTendencies_2 = SurfaceElevationTendenciesCurrent
        SurfaceElevationTendencies_1 = SurfaceElevationTendenciesLast
        SurfaceElevationTendencies = (epsilon*SurfaceElevationTendencies_1 + gamma*SurfaceElevationTendencies_2 
                                      + (1.0 - gamma - epsilon)*SurfaceElevationTendencies_3)
        for iCell in range(0,nCells):
            myMPASOceanShallowWater.mySolution.ssh[iCell] += dt*SurfaceElevationTendencies[iCell]
            myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = SurfaceElevationTendenciesCurrent[iCell]
   
                        
def GeneralizedForwardBackwardMethodWithAB3AM4Step(myMPASOceanShallowWater):
    if myMPASOceanShallowWater.iTime == 0 or myMPASOceanShallowWater.iTime == 1:
        if (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
            == 'SecondOrderAccurate_OptimumChoice_ROMS'):
            ExplicitMidpointMethod(myMPASOceanShallowWater,
                                   TimeIntegrator='GeneralizedForwardBackwardMethodWithAB3AM4Step')
        elif ((myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
               == 'ThirdOrderAccurate_AB3_AM4')
              or (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
                  == 'ThirdOrderAccurate_MaximumStabilityRange')
              or (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
                  == 'ThirdOrderAccurate_OptimumChoice')):
            WilliamsonLowStorageThirdOrderRungeKuttaMethod(
            myMPASOceanShallowWater,TimeIntegrator='GeneralizedForwardBackwardMethodWithAB3AM4Step')
        elif (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
              == 'FourthOrderAccurate_MaximumStabilityRange'):
            CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(
            myMPASOceanShallowWater,TimeIntegrator='GeneralizedForwardBackwardMethodWithAB3AM4Step')
    else:        
        beta = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                .Generalized_FB_with_AB3_AM4_Step_parameter_beta)
        gamma = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                 .Generalized_FB_with_AB3_AM4_Step_parameter_gamma)
        epsilon = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                   .Generalized_FB_with_AB3_AM4_Step_parameter_epsilon)
        delta = (myMPASOceanShallowWater.myNameList.myTimeSteppingParameters
                 .Generalized_FB_with_AB3_AM4_Step_parameter_delta)
        ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
        myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
        UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
        myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
        nEdges = myMPASOceanShallowWater.myMesh.nEdges
        nCells = myMPASOceanShallowWater.myMesh.nCells
        dt = myMPASOceanShallowWater.myNameList.dt
        NormalVelocityTendenciesSecondLast, SurfaceElevationTendenciesSecondLast = (
        CaptureTendencies(myMPASOceanShallowWater,TimeLevel='SecondLast'))
        NormalVelocityTendenciesLast, SurfaceElevationTendenciesLast = CaptureTendencies(myMPASOceanShallowWater,
                                                                                         TimeLevel='Last')
        time = myMPASOceanShallowWater.time
        myMPASOceanShallowWater.ComputeNormalVelocityTendencies(time)
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time)
        NormalVelocityTendenciesCurrent, SurfaceElevationTendenciesCurrent = CaptureTendencies(myMPASOceanShallowWater)
        NormalVelocityTendencies_3 = NormalVelocityTendenciesCurrent
        NormalVelocityTendencies_2 = NormalVelocityTendenciesLast
        NormalVelocityTendencies_1 = NormalVelocityTendenciesSecondLast
        NormalVelocityTendencies = (beta*NormalVelocityTendencies_1 - (0.5 + 2.0*beta)*NormalVelocityTendencies_2
                                    + (1.5 + beta)*NormalVelocityTendencies_3)
        for iEdge in range(0,nEdges):
            x = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            y = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                # Specify the normal velocities at the non-periodic boundary edges to be equal to their exact 
                # counterparts.
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
                ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time+dt,
                                                  UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            else:
                myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] += dt*NormalVelocityTendencies[iEdge]
                myMPASOceanShallowWater.mySolution.normalVelocityTendencySecondLast[iEdge] = (
                NormalVelocityTendenciesLast[iEdge])
                myMPASOceanShallowWater.mySolution.normalVelocityTendencyLast[iEdge] = (
                NormalVelocityTendenciesCurrent[iEdge])
        myMPASOceanShallowWater.ComputeSurfaceElevationTendencies(time+dt) 
        SurfaceElevationTendencies_4 = CaptureTendencies(myMPASOceanShallowWater,
                                                         PrognosticVariables='SurfaceElevation') 
        SurfaceElevationTendencies_3 = SurfaceElevationTendenciesCurrent
        SurfaceElevationTendencies_2 = SurfaceElevationTendenciesLast
        SurfaceElevationTendencies_1 = SurfaceElevationTendenciesSecondLast
        SurfaceElevationTendencies = (epsilon*SurfaceElevationTendencies_1 + gamma*SurfaceElevationTendencies_2 
                                      + (1.0 - gamma - delta - epsilon)*SurfaceElevationTendencies_3
                                      + delta*SurfaceElevationTendencies_4)
        for iCell in range(0,nCells):
            myMPASOceanShallowWater.mySolution.ssh[iCell] += dt*SurfaceElevationTendencies[iCell]
            myMPASOceanShallowWater.mySolution.sshTendencySecondLast[iCell] = SurfaceElevationTendenciesLast[iCell]
            myMPASOceanShallowWater.mySolution.sshTendencyLast[iCell] = SurfaceElevationTendenciesCurrent[iCell]
                        
                        
def TimeIntegration(myMPASOceanShallowWater):
    TimeIntegrator = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.TimeIntegrator
    if TimeIntegrator == 'ForwardEulerMethod':
        ForwardEulerMethod(myMPASOceanShallowWater)
    elif TimeIntegrator == 'ExplicitMidpointMethod':
        ExplicitMidpointMethod(myMPASOceanShallowWater)
    elif TimeIntegrator == 'WilliamsonLowStorageThirdOrderRungeKuttaMethod':
        WilliamsonLowStorageThirdOrderRungeKuttaMethod(myMPASOceanShallowWater)
    elif TimeIntegrator == 'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod':
        CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(myMPASOceanShallowWater)   
    elif TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
        SecondOrderAdamsBashforthMethod(myMPASOceanShallowWater)
    elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
        ThirdOrderAdamsBashforthMethod(myMPASOceanShallowWater)
    elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
        FourthOrderAdamsBashforthMethod(myMPASOceanShallowWater)
    elif TimeIntegrator == 'LeapfrogTrapezoidalMethod':
        LeapfrogTrapezoidalMethod(myMPASOceanShallowWater)
    elif TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback':
        LFTRAndLFAM3MethodWithFBFeedback(myMPASOceanShallowWater)
    elif TimeIntegrator == 'ForwardBackwardMethod':
        ForwardBackwardMethod(myMPASOceanShallowWater)
    elif TimeIntegrator == 'ForwardBackwardMethodWithRK2Feedback':
        ForwardBackwardMethodWithRK2Feedback(myMPASOceanShallowWater)
    elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step':
        GeneralizedForwardBackwardMethodWithAB2AM3Step(myMPASOceanShallowWater)
    elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step':
        GeneralizedForwardBackwardMethodWithAB3AM4Step(myMPASOceanShallowWater)