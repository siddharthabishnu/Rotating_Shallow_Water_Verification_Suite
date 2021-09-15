"""
Name: TimeSteppingMethods.py
Author: Siddhartha Bishnu
Details: This script contains various time-stepping methods for advancing the two-dimensional rotating shallow water 
equations.
"""


import numpy as np
from IPython.utils import io
with io.capture_output() as captured:
    import DGSEM2DClass
    
    
def CaptureSolutions(myDGSEM2D,PrognosticVariables='VelocitiesAndSurfaceElevation'):
    nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
        VelocitiesAndSurfaceElevations = np.zeros((nEquations,nElements,nXi+1,nEta+1))
    elif PrognosticVariables == 'Velocities':
        Velocities = np.zeros((2,nElements,nXi+1,nEta+1))
    elif PrognosticVariables == 'SurfaceElevation':
        SurfaceElevations = np.zeros((nElements,nXi+1,nEta+1))
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
        for iEquation in range(0,nEquations):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        VelocitiesAndSurfaceElevations[iEquation,iElement,iXi,iEta] = (
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta])
    elif PrognosticVariables == 'Velocities':
        iEquation_Start = 0
        iEquation_End = 2
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        Velocities[iEquation,iElement,iXi,iEta] = (
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta])
    elif PrognosticVariables == 'SurfaceElevation':
        iEquation_Start = 2
        iEquation_End = 3
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        SurfaceElevations[iElement,iXi,iEta] = (
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta])   
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
        return VelocitiesAndSurfaceElevations
    elif PrognosticVariables == 'Velocities':
        return Velocities    
    elif PrognosticVariables == 'SurfaceElevation':
        return SurfaceElevations
    

def CaptureTendencies(myDGSEM2D,PrognosticVariables='VelocitiesAndSurfaceElevation',TimeLevel='Current'):
    nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'Velocities':
        VelocityTendencies = np.zeros((2,nElements,nXi+1,nEta+1))
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'SurfaceElevation':
        SurfaceElevationTendencies = np.zeros((nElements,nXi+1,nEta+1))
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
        for iEquation in range(0,nEquations):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        if iEquation == 0 or iEquation == 1:
                            if TimeLevel == 'Current':
                                VelocityTendencies[iEquation,iElement,iXi,iEta] = (
                                myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                            elif TimeLevel == 'Last':
                                VelocityTendencies[iEquation,iElement,iXi,iEta] = (
                                myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta])
                            elif TimeLevel == 'SecondLast':
                                VelocityTendencies[iEquation,iElement,iXi,iEta] = (
                                (myDGSEM2D.myDGSolution2D[iElement]
                                 .SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta]))
                        elif iEquation == 2:
                            if TimeLevel == 'Current':
                                SurfaceElevationTendencies[iElement,iXi,iEta] = (
                                myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                            elif TimeLevel == 'Last':
                                SurfaceElevationTendencies[iElement,iXi,iEta] = (
                                myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta])
                            elif TimeLevel == 'SecondLast':
                                SurfaceElevationTendencies[iElement,iXi,iEta] = (
                                (myDGSEM2D.myDGSolution2D[iElement]
                                 .SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta]))
    elif PrognosticVariables == 'Velocities':
        iEquation_Start = 0
        iEquation_End = 2
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        if TimeLevel == 'Current':
                            VelocityTendencies[iEquation,iElement,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                        elif TimeLevel == 'Last':
                            VelocityTendencies[iEquation,iElement,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta])
                        elif TimeLevel == 'SecondLast':
                            VelocityTendencies[iEquation,iElement,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta])
    elif PrognosticVariables == 'SurfaceElevation':
        iEquation_Start = 2
        iEquation_End = 3
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        if TimeLevel == 'Current':
                            SurfaceElevationTendencies[iElement,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                        elif TimeLevel == 'Last':
                            SurfaceElevationTendencies[iElement,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta])
                        elif TimeLevel == 'SecondLast':
                            SurfaceElevationTendencies[iElement,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta])
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
        return VelocityTendencies, SurfaceElevationTendencies
    elif PrognosticVariables == 'Velocities':
        return VelocityTendencies    
    elif PrognosticVariables == 'SurfaceElevation':
        return SurfaceElevationTendencies
    

def ShiftTendencies(myDGSEM2D):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    TimeIntegrator = myDGSEM2D.myNameList.myTimeSteppingParameters.TimeIntegrator
    for iElement in range(0,nElements):
        if TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
            myDGSEM2D.myDGSolution2D[iElement].ThirdLastTendencyAtInteriorNodes[:,:,:] = (
            myDGSEM2D.myDGSolution2D[iElement].SecondLastTendencyAtInteriorNodes[:,:,:])
        if TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' or TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
            myDGSEM2D.myDGSolution2D[iElement].SecondLastTendencyAtInteriorNodes[:,:,:] = (
            myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[:,:,:])
        if (TimeIntegrator == 'SecondOrderAdamsBashforthMethod' or TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' 
            or TimeIntegrator == 'FourthOrderAdamsBashforthMethod'):
            myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[:,:,:] = (
            myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[:,:,:])


def ForwardEulerMethod(myDGSEM2D):
    nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    dt = myDGSEM2D.myNameList.dt
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
    for iEquation in range(0,nEquations):
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                    dt*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
    

def ExplicitMidpointMethod(myDGSEM2D,TimeIntegrator='ExplicitMidpointMethod'):
    nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    dt = myDGSEM2D.myNameList.dt
    VelocitiesAndSurfaceElevationsCurrent = CaptureSolutions(myDGSEM2D)
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
    for iEquation in range(0,nEquations):
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):    
                    if (TimeIntegrator == 'LeapfrogTrapezoidalMethod' 
                        or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback'):
                        myDGSEM2D.myDGSolution2D[iElement].LastSolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                        VelocitiesAndSurfaceElevationsCurrent[iEquation,iElement,iXi,iEta])
                    if (TimeIntegrator == 'SecondOrderAdamsBashforthMethod' 
                        or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback'):
                        myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                        myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                    elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step':
                        if myDGSEM2D.iTime == 0:
                            myDGSEM2D.myDGSolution2D[iElement].SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                        elif myDGSEM2D.iTime == 1:
                            myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                    0.5*dt*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])        
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+0.5*dt)
    for iEquation in range(0,nEquations):
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                    (VelocitiesAndSurfaceElevationsCurrent[iEquation,iElement,iXi,iEta] 
                     + dt*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta]))    


def WilliamsonLowStorageThirdOrderRungeKuttaMethod(
myDGSEM2D,TimeIntegrator='WilliamsonLowStorageThirdOrderRungeKuttaMethod'):
    nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    dt = myDGSEM2D.myNameList.dt
    nStepsRK3 = myDGSEM2D.myNameList.myTimeSteppingParameters.nStepsRK3
    aRK3 = myDGSEM2D.myNameList.myTimeSteppingParameters.aRK3
    bRK3 = myDGSEM2D.myNameList.myTimeSteppingParameters.bRK3
    gRK3 = myDGSEM2D.myNameList.myTimeSteppingParameters.gRK3
    TemporaryTendencyAtInteriorNodes = np.zeros((nEquations,nElements,nXi+1,nEta+1))
    for iStep in range(0,nStepsRK3):
        time = myDGSEM2D.time + bRK3[iStep]*dt
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,time)
        for iEquation in range(0,nEquations):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        if TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' and iStep == 0:
                            myDGSEM2D.myDGSolution2D[iElement].LastSolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta])  
                        if ((TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' 
                             or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step') and iStep == 0):
                            myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                        elif ((TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' 
                               or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step') and iStep == 0):
                            if myDGSEM2D.iTime == 0:
                                (myDGSEM2D.myDGSolution2D[iElement]
                                 .SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta]) = (
                                myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                            elif myDGSEM2D.iTime == 1:
                                myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                                myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                        TemporaryTendencyAtInteriorNodes[iEquation,iElement,iXi,iEta] = (
                        (aRK3[iStep]*TemporaryTendencyAtInteriorNodes[iEquation,iElement,iXi,iEta]
                         + myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta]))
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                        gRK3[iStep]*dt*TemporaryTendencyAtInteriorNodes[iEquation,iElement,iXi,iEta])
            
            
def CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(
myDGSEM2D,TimeIntegrator='CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod'):
    nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    dt = myDGSEM2D.myNameList.dt
    nStepsRK4 = myDGSEM2D.myNameList.myTimeSteppingParameters.nStepsRK4
    aRK4 = myDGSEM2D.myNameList.myTimeSteppingParameters.aRK4
    bRK4 = myDGSEM2D.myNameList.myTimeSteppingParameters.bRK4
    gRK4 = myDGSEM2D.myNameList.myTimeSteppingParameters.gRK4
    TemporaryTendencyAtInteriorNodes = np.zeros((nEquations,nElements,nXi+1,nEta+1))
    for iStep in range(0,nStepsRK4):
        time = myDGSEM2D.time + bRK4[iStep]*dt
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,time)
        for iEquation in range(0,nEquations):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        if TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' and iStep == 0:
                            myDGSEM2D.myDGSolution2D[iElement].LastSolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta])     
                        if ((TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback' 
                             or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step') and iStep == 0):
                            myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                            myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                        elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step' and iStep == 0:
                            if myDGSEM2D.iTime == 0:
                                (myDGSEM2D.myDGSolution2D[iElement]
                                 .SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta]) = (
                                myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                            elif myDGSEM2D.iTime == 1:
                                myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                                myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                        elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod' and iStep == 0:
                            if myDGSEM2D.iTime == 0:
                                (myDGSEM2D.myDGSolution2D[iElement]
                                 .ThirdLastTendencyAtInteriorNodes[iEquation,iXi,iEta]) = (
                                myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                            elif myDGSEM2D.iTime == 1:
                                (myDGSEM2D.myDGSolution2D[iElement]
                                 .SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta]) = (
                                myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])   
                            elif myDGSEM2D.iTime == 2:
                                myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                                myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                        TemporaryTendencyAtInteriorNodes[iEquation,iElement,iXi,iEta] = (
                        (aRK4[iStep]*TemporaryTendencyAtInteriorNodes[iEquation,iElement,iXi,iEta]
                         + myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta]))
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                        gRK4[iStep]*dt*TemporaryTendencyAtInteriorNodes[iEquation,iElement,iXi,iEta])
        
        
def SecondOrderAdamsBashforthMethod(myDGSEM2D):
    if myDGSEM2D.iTime == 0:
        ExplicitMidpointMethod(myDGSEM2D,TimeIntegrator='SecondOrderAdamsBashforthMethod')
    else:
        nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
        nElements = myDGSEM2D.myDGSEM2DParameters.nElements
        nXi = myDGSEM2D.myDGSEM2DParameters.nXi
        nEta = myDGSEM2D.myDGSEM2DParameters.nEta
        dt = myDGSEM2D.myNameList.dt
        AB2 = myDGSEM2D.myNameList.myTimeSteppingParameters.AB2
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
        for iEquation in range(0,nEquations):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                        dt*(AB2[0]*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta]
                            + AB2[1]*(myDGSEM2D.myDGSolution2D[iElement]
                                      .LastTendencyAtInteriorNodes[iEquation,iXi,iEta])))                
        ShiftTendencies(myDGSEM2D)
    

def ThirdOrderAdamsBashforthMethod(myDGSEM2D):
    if myDGSEM2D.iTime == 0 or myDGSEM2D.iTime == 1:
        WilliamsonLowStorageThirdOrderRungeKuttaMethod(myDGSEM2D,TimeIntegrator='ThirdOrderAdamsBashforthMethod')
    else:
        nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
        nElements = myDGSEM2D.myDGSEM2DParameters.nElements
        nXi = myDGSEM2D.myDGSEM2DParameters.nXi
        nEta = myDGSEM2D.myDGSEM2DParameters.nEta
        dt = myDGSEM2D.myNameList.dt
        AB3 = myDGSEM2D.myNameList.myTimeSteppingParameters.AB3
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
        for iEquation in range(0,nEquations):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                        dt*(AB3[0]*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta]
                            + AB3[1]*(myDGSEM2D.myDGSolution2D[iElement]
                                      .LastTendencyAtInteriorNodes[iEquation,iXi,iEta])
                            + AB3[2]*(myDGSEM2D.myDGSolution2D[iElement]
                                      .SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta])))
        ShiftTendencies(myDGSEM2D)
                        
                        
def FourthOrderAdamsBashforthMethod(myDGSEM2D):
    if myDGSEM2D.iTime == 0 or myDGSEM2D.iTime == 1 or myDGSEM2D.iTime == 2:
        CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(myDGSEM2D,
                                                              TimeIntegrator='FourthOrderAdamsBashforthMethod')
    else:
        nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
        nElements = myDGSEM2D.myDGSEM2DParameters.nElements
        nXi = myDGSEM2D.myDGSEM2DParameters.nXi
        nEta = myDGSEM2D.myDGSEM2DParameters.nEta
        dt = myDGSEM2D.myNameList.dt
        AB4 = myDGSEM2D.myNameList.myTimeSteppingParameters.AB4
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
        for iEquation in range(0,nEquations):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                        dt*(AB4[0]*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta]
                            + AB4[1]*(myDGSEM2D.myDGSolution2D[iElement]
                                      .LastTendencyAtInteriorNodes[iEquation,iXi,iEta])
                            + AB4[2]*(myDGSEM2D.myDGSolution2D[iElement]
                                      .SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta])
                            + AB4[3]*(myDGSEM2D.myDGSolution2D[iElement]
                                      .ThirdLastTendencyAtInteriorNodes[iEquation,iXi,iEta])))
        ShiftTendencies(myDGSEM2D)
        

def LeapfrogTrapezoidalMethod(myDGSEM2D):
    if myDGSEM2D.iTime == 0:
        ExplicitMidpointMethod(myDGSEM2D,TimeIntegrator='LeapfrogTrapezoidalMethod')
    else:
        nEquations = myDGSEM2D.myDGSEM2DParameters.nEquations
        nElements = myDGSEM2D.myDGSEM2DParameters.nElements
        nXi = myDGSEM2D.myDGSEM2DParameters.nXi
        nEta = myDGSEM2D.myDGSEM2DParameters.nEta
        dt = myDGSEM2D.myNameList.dt
        VelocitiesAndSurfaceElevationsCurrent = CaptureSolutions(myDGSEM2D)
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
        VelocityTendenciesCurrent, SurfaceElevationTendenciesCurrent = CaptureTendencies(myDGSEM2D)
        for iEquation in range(0,nEquations):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                        (myDGSEM2D.myDGSolution2D[iElement].LastSolutionAtInteriorNodes[iEquation,iXi,iEta]
                         + 2.0*dt*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta]))
                        myDGSEM2D.myDGSolution2D[iElement].LastSolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                        VelocitiesAndSurfaceElevationsCurrent[iEquation,iElement,iXi,iEta])
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt)
        VelocityTendenciesNew, SurfaceElevationTendenciesNew = CaptureTendencies(myDGSEM2D)
        VelocityTendencies = 0.5*(VelocityTendenciesCurrent + VelocityTendenciesNew)
        SurfaceElevationTendencies = 0.5*(SurfaceElevationTendenciesCurrent + SurfaceElevationTendenciesNew)            
        for iEquation in range(0,nEquations):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        if iEquation == 0 or iEquation == 1:
                            myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                            (VelocitiesAndSurfaceElevationsCurrent[iEquation,iElement,iXi,iEta] 
                             + dt*VelocityTendencies[iEquation,iElement,iXi,iEta]))
                        elif iEquation == 2:
                            myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                            (VelocitiesAndSurfaceElevationsCurrent[iEquation,iElement,iXi,iEta] 
                             + dt*SurfaceElevationTendencies[iEquation,iElement,iXi,iEta]))
                        
                        
def LFTRAndLFAM3MethodWithFBFeedback(myDGSEM2D):
    if myDGSEM2D.iTime == 0:
        if (myDGSEM2D.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
            == 'SecondOrderAccurate_LF_TR'):
            ExplicitMidpointMethod(myDGSEM2D,TimeIntegrator='LFTRAndLFAM3MethodWithFBFeedback')            
        elif ((myDGSEM2D.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
               == 'ThirdOrderAccurate_LF_AM3')
              or (myDGSEM2D.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
                  == 'ThirdOrderAccurate_MaximumStabilityRange')):
            WilliamsonLowStorageThirdOrderRungeKuttaMethod(myDGSEM2D,
                                                           TimeIntegrator='LFTRAndLFAM3MethodWithFBFeedback')  
        elif ((myDGSEM2D.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
               == 'FourthOrderAccurate_MinimumTruncationError')
              or (myDGSEM2D.myNamelist.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type 
                  == 'FourthOrderAccurate_MaximumStabilityRange')):
            CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(myDGSEM2D,
                                                                  TimeIntegrator='LFTRAndLFAM3MethodWithFBFeedback')
    else:
        nElements = myDGSEM2D.myDGSEM2DParameters.nElements
        nXi = myDGSEM2D.myDGSEM2DParameters.nXi
        nEta = myDGSEM2D.myDGSEM2DParameters.nEta
        dt = myDGSEM2D.myNameList.dt        
        beta = myDGSEM2D.myNameList.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta
        gamma = myDGSEM2D.myNameList.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma
        epsilon = myDGSEM2D.myNameList.myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon
        VelocityTendenciesLast, SurfaceElevationTendenciesLast = CaptureTendencies(myDGSEM2D,TimeLevel='Last')
        VelocitiesAndSurfaceElevationsCurrent = CaptureSolutions(myDGSEM2D)
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
        VelocityTendenciesCurrent, SurfaceElevationTendenciesCurrent = CaptureTendencies(myDGSEM2D)
        iEquation_Start = 0
        iEquation_End = 2
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                        (myDGSEM2D.myDGSolution2D[iElement].LastSolutionAtInteriorNodes[iEquation,iXi,iEta]
                         + 2.0*dt*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta]))
                        myDGSEM2D.myDGSolution2D[iElement].LastSolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                        VelocitiesAndSurfaceElevationsCurrent[iEquation,iElement,iXi,iEta])
                        myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                        VelocityTendenciesCurrent[iEquation,iElement,iXi,iEta])
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt,PrognosticVariables='SurfaceElevation',
                                          ComputeExternalSurfaceElevationOneTimeStepEarlier=True)       
        SurfaceElevationTendencies_3 = CaptureTendencies(myDGSEM2D,PrognosticVariables='SurfaceElevation') 
        SurfaceElevationTendencies_2 = SurfaceElevationTendenciesCurrent
        SurfaceElevationTendencies_1 = SurfaceElevationTendenciesLast
        SurfaceElevationTendencies = ((1.0 - 2.0*beta)*SurfaceElevationTendencies_2
                                      + beta*(SurfaceElevationTendencies_3 + SurfaceElevationTendencies_1))
        iEquation_Start = 2
        iEquation_End = 3
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = ( 
                        (myDGSEM2D.myDGSolution2D[iElement].LastSolutionAtInteriorNodes[iEquation,iXi,iEta]
                         + 2.0*dt*SurfaceElevationTendencies[iElement,iXi,iEta]))
                        myDGSEM2D.myDGSolution2D[iElement].LastSolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                        VelocitiesAndSurfaceElevationsCurrent[iEquation,iElement,iXi,iEta])
                        myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                        SurfaceElevationTendenciesCurrent[iElement,iXi,iEta])     
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt)
        VelocityTendencies_3, SurfaceElevationTendencies_3 = CaptureTendencies(myDGSEM2D)
        VelocityTendencies_2 = VelocityTendenciesCurrent
        VelocityTendencies_1 = VelocityTendenciesLast
        VelocityTendencies = (-gamma*VelocityTendencies_1 + (0.5 + 2.0*gamma)*VelocityTendencies_2 
                              + (0.5 - gamma)*VelocityTendencies_3)
        iEquation_Start = 0
        iEquation_End = 2
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                        (VelocitiesAndSurfaceElevationsCurrent[iEquation,iElement,iXi,iEta] 
                         + dt*VelocityTendencies[iEquation,iElement,iXi,iEta]))
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt,PrognosticVariables='SurfaceElevation')       
        SurfaceElevationTendencies_4 = CaptureTendencies(myDGSEM2D,PrognosticVariables='SurfaceElevation')
        SurfaceElevationTendencies = (
        ((0.5 - gamma)*(epsilon*SurfaceElevationTendencies_4 + (1.0 - epsilon)*SurfaceElevationTendencies_3) 
         + (0.5 + 2.0*gamma)*SurfaceElevationTendencies_2 - gamma*SurfaceElevationTendencies_1))         
        iEquation_Start = 2
        iEquation_End = 3
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = ( 
                        (VelocitiesAndSurfaceElevationsCurrent[iEquation,iElement,iXi,iEta] 
                         + dt*SurfaceElevationTendencies[iElement,iXi,iEta]))

        
def ForwardBackwardMethod(myDGSEM2D):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    dt = myDGSEM2D.myNameList.dt
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time,PrognosticVariables='Velocities')
    iEquation_Start = 0
    iEquation_End = 2
    for iEquation in range(iEquation_Start,iEquation_End):
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                    dt*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt,PrognosticVariables='SurfaceElevation',
                                      ComputeExternalSurfaceElevationOneTimeStepEarlier=True)
    iEquation_Start = 2
    iEquation_End = 3
    for iEquation in range(iEquation_Start,iEquation_End):
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                    dt*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
                    
                    
def ForwardBackwardMethodWithRK2Feedback(myDGSEM2D):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    dt = myDGSEM2D.myNameList.dt
    beta = myDGSEM2D.myNameList.myTimeSteppingParameters.Forward_Backward_with_RK2_Feedback_parameter_beta
    epsilon = myDGSEM2D.myNameList.myTimeSteppingParameters.Forward_Backward_with_RK2_Feedback_parameter_epsilon
    VelocitiesAndSurfaceElevations_1 = CaptureSolutions(myDGSEM2D)
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
    VelocityTendencies_1, SurfaceElevationTendencies_1 = CaptureTendencies(myDGSEM2D)
    iEquation_Start = 0
    iEquation_End = 2
    for iEquation in range(iEquation_Start,iEquation_End):
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                    dt*myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[iEquation,iXi,iEta])
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt,PrognosticVariables='SurfaceElevation',
                                      ComputeExternalSurfaceElevationOneTimeStepEarlier=True)            
    SurfaceElevationTendencies_2 = CaptureTendencies(myDGSEM2D,PrognosticVariables='SurfaceElevation')           
    SurfaceElevationTendencies = (1.0 - beta)*SurfaceElevationTendencies_1 + beta*SurfaceElevationTendencies_2 
    iEquation_Start = 2
    iEquation_End = 3
    for iEquation in range(iEquation_Start,iEquation_End):
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                    dt*SurfaceElevationTendencies[iElement,iXi,iEta])
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt)
    VelocityTendencies_2, SurfaceElevationTendencies_2 = CaptureTendencies(myDGSEM2D)           
    VelocityTendencies = 0.5*(VelocityTendencies_1 + VelocityTendencies_2)         
    iEquation_Start = 0
    iEquation_End = 2
    for iEquation in range(iEquation_Start,iEquation_End):
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                    (VelocitiesAndSurfaceElevations_1[iEquation,iElement,iXi,iEta] 
                     + dt*VelocityTendencies[iEquation,iElement,iXi,iEta]))
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt,PrognosticVariables='SurfaceElevation')
    SurfaceElevationTendencies_3 = CaptureTendencies(myDGSEM2D,PrognosticVariables='SurfaceElevation')
    SurfaceElevationTendencies = 0.5*(SurfaceElevationTendencies_1 + (1.0 - epsilon)*SurfaceElevationTendencies_2 
                                      + epsilon*SurfaceElevationTendencies_3)           
    iEquation_Start = 2
    iEquation_End = 3
    for iEquation in range(iEquation_Start,iEquation_End):
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] = (
                    (VelocitiesAndSurfaceElevations_1[iEquation,iElement,iXi,iEta] 
                     + dt*SurfaceElevationTendencies[iElement,iXi,iEta]))

    
def GeneralizedForwardBackwardMethodWithAB2AM3Step(myDGSEM2D):
    if myDGSEM2D.iTime == 0:
        if ((myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_Type 
             == 'ThirdOrderAccurate_WideStabilityRange')
              or (myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_Type 
                  == 'ThirdOrderAccurate_WeakAsymptoticInstabilityOfPhysicalModes')):
            WilliamsonLowStorageThirdOrderRungeKuttaMethod(
            myDGSEM2D,TimeIntegrator='GeneralizedForwardBackwardMethodWithAB2AM3Step')
        elif (myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_Type 
              == 'FourthOrderAccurate'):
            CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(
            myDGSEM2D,TimeIntegrator='GeneralizedForwardBackwardMethodWithAB2AM3Step')
    else:  
        nElements = myDGSEM2D.myDGSEM2DParameters.nElements
        nXi = myDGSEM2D.myDGSEM2DParameters.nXi
        nEta = myDGSEM2D.myDGSEM2DParameters.nEta
        dt = myDGSEM2D.myNameList.dt        
        beta = myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_parameter_beta
        gamma = myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_parameter_gamma
        epsilon = myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_parameter_epsilon
        VelocityTendenciesLast, SurfaceElevationTendenciesLast = CaptureTendencies(myDGSEM2D,TimeLevel='Last')
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
        VelocityTendenciesCurrent, SurfaceElevationTendenciesCurrent = CaptureTendencies(myDGSEM2D)
        VelocityTendencies_2 = VelocityTendenciesCurrent
        VelocityTendencies_1 = VelocityTendenciesLast
        VelocityTendencies = -beta*VelocityTendencies_1 + (1.0 + beta)*VelocityTendencies_2
        iEquation_Start = 0
        iEquation_End = 2
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                        dt*VelocityTendencies[iEquation,iElement,iXi,iEta])      
                        myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                        VelocityTendenciesCurrent[iEquation,iElement,iXi,iEta])
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt,PrognosticVariables='SurfaceElevation',
                                          ComputeExternalSurfaceElevationOneTimeStepEarlier=True)       
        SurfaceElevationTendencies_3 = CaptureTendencies(myDGSEM2D,PrognosticVariables='SurfaceElevation') 
        SurfaceElevationTendencies_2 = SurfaceElevationTendenciesCurrent
        SurfaceElevationTendencies_1 = SurfaceElevationTendenciesLast
        SurfaceElevationTendencies = (epsilon*SurfaceElevationTendencies_1 + gamma*SurfaceElevationTendencies_2 
                                      + (1.0 - gamma - epsilon)*SurfaceElevationTendencies_3)
        iEquation_Start = 2
        iEquation_End = 3
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += ( 
                        dt*SurfaceElevationTendencies[iElement,iXi,iEta])
                        myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                        SurfaceElevationTendenciesCurrent[iElement,iXi,iEta])
                        
                        
def GeneralizedForwardBackwardMethodWithAB3AM4Step(myDGSEM2D):
    if myDGSEM2D.iTime == 0 or myDGSEM2D.iTime == 1:
        if (myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
            == 'SecondOrderAccurate_OptimumChoice_ROMS'):
            ExplicitMidpointMethod(myDGSEM2D,TimeIntegrator='GeneralizedForwardBackwardMethodWithAB3AM4Step')
        elif ((myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
               == 'ThirdOrderAccurate_AB3_AM4')
              or (myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
                  == 'ThirdOrderAccurate_MaximumStabilityRange')
              or (myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
                  == 'ThirdOrderAccurate_OptimumChoice')):
            WilliamsonLowStorageThirdOrderRungeKuttaMethod(
            myDGSEM2D,TimeIntegrator='GeneralizedForwardBackwardMethodWithAB3AM4Step')
        elif (myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type 
              == 'FourthOrderAccurate_MaximumStabilityRange'):
            CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(
            myDGSEM2D,TimeIntegrator='GeneralizedForwardBackwardMethodWithAB3AM4Step')
    else:   
        nElements = myDGSEM2D.myDGSEM2DParameters.nElements
        nXi = myDGSEM2D.myDGSEM2DParameters.nXi
        nEta = myDGSEM2D.myDGSEM2DParameters.nEta
        dt = myDGSEM2D.myNameList.dt        
        beta = myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_parameter_beta
        gamma = myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_parameter_gamma
        epsilon = myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_parameter_epsilon
        delta = myDGSEM2D.myNameList.myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_parameter_delta
        VelocityTendenciesSecondLast, SurfaceElevationTendenciesSecondLast = (
        CaptureTendencies(myDGSEM2D,TimeLevel='SecondLast'))
        VelocityTendenciesLast, SurfaceElevationTendenciesLast = CaptureTendencies(myDGSEM2D,TimeLevel='Last')
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
        VelocityTendenciesCurrent, SurfaceElevationTendenciesCurrent = CaptureTendencies(myDGSEM2D)
        VelocityTendencies_3 = VelocityTendenciesCurrent
        VelocityTendencies_2 = VelocityTendenciesLast
        VelocityTendencies_1 = VelocityTendenciesSecondLast
        VelocityTendencies = (beta*VelocityTendencies_1 - (0.5 + 2.0*beta)*VelocityTendencies_2
                              + (1.5 + beta)*VelocityTendencies_3)
        iEquation_Start = 0
        iEquation_End = 2
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += (
                        dt*VelocityTendencies[iEquation,iElement,iXi,iEta])
                        myDGSEM2D.myDGSolution2D[iElement].SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                        VelocityTendenciesLast[iEquation,iElement,iXi,iEta])
                        myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                        VelocityTendenciesCurrent[iEquation,iElement,iXi,iEta])
        DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time+dt,PrognosticVariables='SurfaceElevation',
                                          ComputeExternalSurfaceElevationOneTimeStepEarlier=True)       
        SurfaceElevationTendencies_4 = CaptureTendencies(myDGSEM2D,PrognosticVariables='SurfaceElevation') 
        SurfaceElevationTendencies_3 = SurfaceElevationTendenciesCurrent
        SurfaceElevationTendencies_2 = SurfaceElevationTendenciesLast
        SurfaceElevationTendencies_1 = SurfaceElevationTendenciesSecondLast
        SurfaceElevationTendencies = (epsilon*SurfaceElevationTendencies_1 + gamma*SurfaceElevationTendencies_2 
                                      + (1.0 - gamma - delta - epsilon)*SurfaceElevationTendencies_3
                                      + delta*SurfaceElevationTendencies_4)
        iEquation_Start = 2
        iEquation_End = 3
        for iEquation in range(iEquation_Start,iEquation_End):
            for iElement in range(0,nElements):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[iEquation,iXi,iEta] += ( 
                        dt*SurfaceElevationTendencies[iElement,iXi,iEta])
                        myDGSEM2D.myDGSolution2D[iElement].SecondLastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                        SurfaceElevationTendenciesLast[iElement,iXi,iEta])
                        myDGSEM2D.myDGSolution2D[iElement].LastTendencyAtInteriorNodes[iEquation,iXi,iEta] = (
                        SurfaceElevationTendenciesCurrent[iElement,iXi,iEta])
                        
                        
def TimeIntegration(myDGSEM2D):
    TimeIntegrator = myDGSEM2D.myNameList.myTimeSteppingParameters.TimeIntegrator
    if TimeIntegrator == 'ForwardEulerMethod':
        ForwardEulerMethod(myDGSEM2D)
    elif TimeIntegrator == 'ExplicitMidpointMethod':
        ExplicitMidpointMethod(myDGSEM2D)
    elif TimeIntegrator == 'WilliamsonLowStorageThirdOrderRungeKuttaMethod':
        WilliamsonLowStorageThirdOrderRungeKuttaMethod(myDGSEM2D)
    elif TimeIntegrator == 'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod':
        CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod(myDGSEM2D)   
    elif TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
        SecondOrderAdamsBashforthMethod(myDGSEM2D)
    elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
        ThirdOrderAdamsBashforthMethod(myDGSEM2D)
    elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
        FourthOrderAdamsBashforthMethod(myDGSEM2D)
    elif TimeIntegrator == 'LeapfrogTrapezoidalMethod':
        LeapfrogTrapezoidalMethod(myDGSEM2D)
    elif TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback':
        LFTRAndLFAM3MethodWithFBFeedback(myDGSEM2D)
    elif TimeIntegrator == 'ForwardBackwardMethod':
        ForwardBackwardMethod(myDGSEM2D)
    elif TimeIntegrator == 'ForwardBackwardMethodWithRK2Feedback':
        ForwardBackwardMethodWithRK2Feedback(myDGSEM2D)
    elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step':
        GeneralizedForwardBackwardMethodWithAB2AM3Step(myDGSEM2D)
    elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step':
        GeneralizedForwardBackwardMethodWithAB3AM4Step(myDGSEM2D)