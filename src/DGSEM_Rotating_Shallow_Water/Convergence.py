"""
Name: Convergence.py
Author: Sid Bishnu
Details: This script contains functions for performing convergence studies of the various test cases with refinement in
both space and time, only in space, and only in time.
"""


import numpy as np
AnalyzeSELFOutputData = False
if AnalyzeSELFOutputData:
    import self.model2d as model2d
from IPython.utils import io
import os
with io.capture_output() as captured:
    import CommonRoutines as CR
    import LagrangeInterpolation2DClass as LI2D
    import Initialization
    import DGSEM2DClass
    import TimeSteppingMethods as TSM


def DetermineEnclosingElementID(nElementsX,dx,dy,x,y):
    ElementIDX = int(np.ceil(x/dx))
    ElementIDY = int(np.ceil(y/dy))
    ElementID = (ElementIDY - 1)*nElementsX + ElementIDX
    return ElementID


def InterpolateDGSolution2DToCoarsestMesh(myDGSEM2D,myCoarsestMesh,EntityToBeInterpolated='Error'):
    lX = myDGSEM2D.myNameList.lX
    lY = myDGSEM2D.myNameList.lY
    nElementsX = myDGSEM2D.myQuadMesh.nElementsX
    nElementsY = myDGSEM2D.myQuadMesh.nElementsY
    dx = lX/float(nElementsX)
    dy = lY/float(nElementsY)
    xi = myDGSEM2D.myDGNodalStorage2D.myLegendreGaussQuadrature1DX.x
    eta = myDGSEM2D.myDGNodalStorage2D.myLegendreGaussQuadrature1DY.x
    myLagrangeInterpolation2D = LI2D.LagrangeInterpolation2D(xi,eta)
    nElements_CoarsestMesh = myCoarsestMesh.nElements
    nXi_CoarsestMesh = myCoarsestMesh.myQuadElements[0].myMappedGeometry2D.nXi
    nEta_CoarsestMesh = myCoarsestMesh.myQuadElements[0].myMappedGeometry2D.nEta
    myDGSolution2DOnCoarsestMesh = np.zeros((nElements_CoarsestMesh,nXi_CoarsestMesh+1,nEta_CoarsestMesh+1,3))
    for iElement_CoarsestMesh in range(0,nElements_CoarsestMesh):
        for iXi_CoarsestMesh in range(0,nXi_CoarsestMesh+1):
            for iEta_CoarsestMesh in range(0,nEta_CoarsestMesh+1):
                x = myCoarsestMesh.myQuadElements[iElement_CoarsestMesh].myMappedGeometry2D.x[iXi_CoarsestMesh,
                                                                                              iEta_CoarsestMesh]
                y = myCoarsestMesh.myQuadElements[iElement_CoarsestMesh].myMappedGeometry2D.y[iXi_CoarsestMesh,
                                                                                              iEta_CoarsestMesh]
                ElementID = DetermineEnclosingElementID(nElementsX,dx,dy,x,y)
                iElement = ElementID - 1
                if EntityToBeInterpolated == 'Error':
                    myDGSolution2D = myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[:,:,:]
                elif EntityToBeInterpolated == 'Solution':
                    myDGSolution2D = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[:,:,:]
                success, iIterationFinal, res, xi, eta = (
                myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.DetermineComputationalCoordinates(
                myLagrangeInterpolation2D,x,y))
                for iVariable in range(0,3):
                    myDGSolution2DOnCoarsestMesh[iElement_CoarsestMesh,iXi_CoarsestMesh,iEta_CoarsestMesh,iVariable] = (
                    myLagrangeInterpolation2D.EvaluateLagrangeInterpolant2D(myDGSolution2D[iVariable,:,:],xi,eta))
    return myDGSolution2DOnCoarsestMesh


def ObtainNumericalSolutionFromSELFOutputData(myDGSEM2D,TimeIntegratorShortForm):
    ProblemType_FileName = myDGSEM2D.myNameList.ProblemType_FileName
    lX = myDGSEM2D.myNameList.lX
    lY = myDGSEM2D.myNameList.lY
    dx = myDGSEM2D.myNameList.dx
    dy = myDGSEM2D.myNameList.dy
    nElementsX = myDGSEM2D.myDGSEM2DParameters.nElementsX
    nElementsY = myDGSEM2D.myDGSEM2DParameters.nElementsY
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    dt = myDGSEM2D.myNameList.dt
    OutputDirectory = (myDGSEM2D.OutputDirectory + '/' + ProblemType_FileName + '_ConvergenceData/%dx%d/' 
                       %(nElementsX,nElementsY) + TimeIntegratorShortForm.lower() + '/%.2f' %dt)
    OutputFile = 'solution.pickup.h5'
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    # Initialize your model.
    model = model2d.model()
    # Load in the model data from the output file.
    model.load(OutputFile)
    PhysicalCoordinates = model.geom.x # Physical Coordinates
    nElementsSELF = PhysicalCoordinates.shape[0]
    for iElementSELF in range(0,nElementsSELF):
        xPhysicalCoordinatesInElement = PhysicalCoordinates[iElementSELF,0,:,:,0].compute() + 0.5*lX
        yPhysicalCoordinatesInElement = PhysicalCoordinates[iElementSELF,0,:,:,1].compute() + 0.5*lY
        ZonalVelocitiesInElement = model.solution[iElementSELF,0,:,:].compute()
        MeridionalVelocitiesInElement = model.solution[iElementSELF,1,:,:].compute()
        SurfaceElevationsInElement = model.solution[iElementSELF,2,:,:].compute()
        iElementX = int(np.min(xPhysicalCoordinatesInElement)/dx)
        iElementY = int(np.min(yPhysicalCoordinatesInElement)/dy)
        iElement = iElementY*nElementsX + iElementX
        if iElementX < nElementsX and iElementY < nElementsY:
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,iXi,iEta] = (
                    ZonalVelocitiesInElement[iEta,iXi])
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,iXi,iEta] = (
                    MeridionalVelocitiesInElement[iEta,iXi])
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,iXi,iEta] = (
                    SurfaceElevationsInElement[iEta,iXi])
    os.chdir(cwd)


def DetermineNumericalSolutionAndError(
ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber_Advection,
CourantNumber_Diffusion,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,nEtaPlot,dt,nTime,
PerformInterpolation=True,isCoarsestMesh=True,myDGNodalStorage2DOnCoarsestMesh=[],myCoarsestMesh=[],
EntityToBeInterpolated='Error',ReadFromSELFOutputData=False,TimeIntegratorShortForm='RK3'):
    UseCourantNumberToDetermineTimeStep = False 
    myDGSEM2D = DGSEM2DClass.DGSEM2D(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                     Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,
                                     nEtaPlot,CourantNumber_Advection,CourantNumber_Diffusion,
                                     UseCourantNumberToDetermineTimeStep,ReadFromSELFOutputData)
    myDGSEM2D.myNameList.dt = dt
    myDGSEM2D.myNameList.nTime = nTime
    print('The number of time steps for the %3d x %3d mesh is %3d.' %(nElementsX,nElementsY,nTime))
    if ReadFromSELFOutputData:
        iTimeLowerLimit = nTime
    else:
        iTimeLowerLimit = 0
    DisplayProgress = True
    for iTime in range(iTimeLowerLimit,nTime+1):
        if DisplayProgress:
            print('Displaying Progress: iTime = %3d.' %iTime)
        myDGSEM2D.iTime = iTime
        myDGSEM2D.time = float(iTime)*dt
        if iTime == 0 or iTime == nTime:   
            DGSEM2DClass.DetermineExactSolutionAtInteriorNodes(myDGSEM2D)
        if iTime == 0:
            DGSEM2DClass.SpecifyInitialConditions(myDGSEM2D)
        if iTime == nTime:
            if ReadFromSELFOutputData:
                ObtainNumericalSolutionFromSELFOutputData(myDGSEM2D,TimeIntegratorShortForm)
            print('The final time for the %3d x %3d mesh is %.6f seconds.' %(nElementsX,nElementsY,myDGSEM2D.time))
            if not(myDGSEM2D.myDGSEM2DParameters.ProblemType_RossbyWave):
                DGSEM2DClass.ComputeError(myDGSEM2D)
            if ConvergenceType == 'Space' or ConvergenceType == 'Time':
                if EntityToBeInterpolated == 'Solution':
                    State = 'Numerical'
                else: # if EntityToBeInterpolated == 'Error':
                    State = 'Error'
            if ConvergenceType == 'SpaceAndTime':
                if not(PerformInterpolation) or (PerformInterpolation and isCoarsestMesh):
                    L2ErrorNorm = DGSEM2DClass.ComputeErrorNorm(myDGSEM2D)
                else:
                    myDGSolution2DOnCoarsestMesh = InterpolateDGSolution2DToCoarsestMesh(myDGSEM2D,myCoarsestMesh,
                                                                                         EntityToBeInterpolated)
                    L2ErrorNorm = (
                    DGSEM2DClass.ComputeErrorNormOnCoarsestMesh(myDGNodalStorage2DOnCoarsestMesh,myCoarsestMesh,
                                                                myDGSolution2DOnCoarsestMesh))
            elif ConvergenceType == 'Space':
                if isCoarsestMesh:
                    myDGSolution2DOnCoarsestMesh = (
                    DGSEM2DClass.ExpressStateAtInteriorNodesAsArrays(myDGSEM2D,State,
                                                                     ReturnStateAsOneMultiDimensionalArray=True))
                else:
                    myDGSolution2DOnCoarsestMesh = InterpolateDGSolution2DToCoarsestMesh(myDGSEM2D,myCoarsestMesh,
                                                                                         EntityToBeInterpolated)
            elif ConvergenceType == 'Time':
                myDGSolution2D = (
                DGSEM2DClass.ExpressStateAtInteriorNodesAsArrays(myDGSEM2D,State,
                                                                 ReturnStateAsOneMultiDimensionalArray=True))
        if iTime < nTime:
            TSM.TimeIntegration(myDGSEM2D)
    if ConvergenceType == 'SpaceAndTime':
        if PerformInterpolation and isCoarsestMesh:
            return myDGSEM2D.myDGNodalStorage2D, myDGSEM2D.myQuadMesh, L2ErrorNorm
        else: # if not(PerformInterpolation) or (PerformInterpolation and not(isCoarsestMesh)):
            return L2ErrorNorm
    elif ConvergenceType == 'Space':
        if isCoarsestMesh:
            return myDGSEM2D.myDGNodalStorage2D, myDGSEM2D.myQuadMesh, myDGSolution2DOnCoarsestMesh
        else:
            return myDGSolution2DOnCoarsestMesh
    elif ConvergenceType == 'Time':
        return myDGSolution2D

        
def WriteL2ErrorNorm(OutputDirectory,nIntervals,Intervals,L2ErrorNorm,FileName):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    nCases = len(nIntervals)
    FileName += '.curve'
    OutputFile = open(FileName,'w')
    OutputFile.write('#phi\n')
    for iCase in range(0,nCases):
        OutputFile.write('%.15g %.15g %.15g %.15g %.15g\n' 
                         %(nIntervals[iCase],Intervals[iCase],L2ErrorNorm[0,iCase],L2ErrorNorm[1,iCase],
                           L2ErrorNorm[2,iCase]))
    OutputFile.close()
    os.chdir(cwd)
    
    
def ReadL2ErrorNorm(OutputDirectory,FileName):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    data = []
    count = 0
    with open(FileName,'r') as InputFile:
        for line in InputFile:
            if count != 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    nCases = data.shape[0]
    nIntervals = np.zeros(nCases)
    Intervals = np.zeros(nCases)
    L2ErrorNorm = np.zeros((3,nCases))
    for iCase in range(0,nCases):
        nIntervals[iCase] = data[iCase,0]
        Intervals[iCase] = data[iCase,1]
        L2ErrorNorm[:,iCase] = data[iCase,2:5]
    os.chdir(cwd)
    return nIntervals, Intervals, L2ErrorNorm
        
        
def SpecifyPolynomialOrderAndMinimumNumberOfElements(ConvergenceType,ProblemType,TimeIntegrator):
    if ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space':
        nElementsX_Minimum = 4
        nElementsXInEachMesh = 0
        nXi = 3
    else: # if ConvergenceType == 'Time':
        nElementsX_Minimum = 0
        nElementsXInEachMesh = 8
        if (ProblemType == 'NonLinear_Manufactured_Solution' 
            and (TimeIntegrator == 'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod' 
                 or TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' 
                 or TimeIntegrator == 'FourthOrderAdamsBashforthMethod')):
            nXi = 2
        else:
            nXi = 3
    return nElementsX_Minimum, nElementsXInEachMesh, nXi


def SpecifyTimeStepAndNumberOfTimeStepsForRossbyWaves(ConvergenceType,TimeIntegrator):
    nCases = 5
    if ConvergenceType == 'Space':
        if (TimeIntegrator == 'ExplicitMidpointMethod' 
            or TimeIntegrator == 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
            or TimeIntegrator == 'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod'):
            dt = 0.24*np.ones(nCases)
            nTime = 720000*np.ones(nCases,dtype=int)
        elif TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
            dt = 0.12*np.ones(nCases)
            nTime = 1440000*np.ones(nCases,dtype=int)
        elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
            dt = 0.06*np.ones(nCases)
            nTime = 2880000*np.ones(nCases,dtype=int)
        elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
            dt = 0.03*np.ones(nCases)
            nTime = 5760000*np.ones(nCases,dtype=int)
    elif ConvergenceType == 'Time':
        dt = np.zeros(nCases)
        nTime = np.zeros(nCases,dtype=int)
        if (TimeIntegrator == 'ExplicitMidpointMethod' 
            or TimeIntegrator == 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
            or TimeIntegrator == 'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod'):
            dt_Maximum = 1.92
            nTime_Minimum = 90000
        elif TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
            dt_Maximum = 0.96
            nTime_Minimum = 180000
        elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
            dt_Maximum = 0.48
            nTime_Minimum = 360000
        elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
            dt_Maximum = 0.24
            nTime_Minimum = 720000
        for iCase in range(0,nCases):
            if iCase == 0:
                dt[iCase] = dt_Maximum
                nTime[iCase] = nTime_Minimum
            else:
                dt[iCase] = dt[iCase-1]*0.5
                nTime[iCase] = nTime[iCase-1]*2
    FinalTime = 172800.0  
    return dt, nTime, FinalTime    


def SpecifyTimeIntegratorShortForm(TimeIntegrator):
    if TimeIntegrator == 'ExplicitMidpointMethod':
        TimeIntegratorShortForm = 'RK2'
    elif TimeIntegrator == 'WilliamsonLowStorageThirdOrderRungeKuttaMethod':
        TimeIntegratorShortForm = 'RK3'
    elif TimeIntegrator == 'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod':
        TimeIntegratorShortForm = 'RK4'    
    elif TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
        TimeIntegratorShortForm = 'AB2'
    elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
        TimeIntegratorShortForm = 'AB3'
    elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
        TimeIntegratorShortForm = 'AB4'
    return TimeIntegratorShortForm


def ConvergenceStudy(ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                     Generalized_FB_with_AB3_AM4_Step_Type,PerformInterpolation,EntityToBeInterpolated,
                     ReadFromSELFOutputData):
    nCases = 5
    nElementsX = np.zeros(nCases,dtype=int)
    nElementsX_Minimum, nElementsXInEachMesh, nXi = (
    SpecifyPolynomialOrderAndMinimumNumberOfElements(ConvergenceType,ProblemType,TimeIntegrator))
    nEta = nXi
    if ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space':
        for iCase in range(0,nCases):
            if iCase == 0:
                nElementsX[iCase] = nElementsX_Minimum
            else:
                nElementsX[iCase] = nElementsX[iCase-1]*2
    elif ConvergenceType == 'Time':
        nElementsX = np.ones(nCases,dtype=int)*nElementsXInEachMesh   
    nElementsY = nElementsX
    nXiPlot = 10
    nEtaPlot = 10
    if ConvergenceType == 'Time':
        CourantNumber_Advection = 0.5
    else:
        CourantNumber_Advection = 0.25
    CourantNumber_Diffusion = 0.5
    UseCourantNumberToDetermineTimeStep = True
    myDGSEM2D = DGSEM2DClass.DGSEM2D(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                     Generalized_FB_with_AB3_AM4_Step_Type,max(nElementsX),max(nElementsY),nXi,nEta,
                                     nXiPlot,nEtaPlot,CourantNumber_Advection,CourantNumber_Diffusion,
                                     UseCourantNumberToDetermineTimeStep,ReadFromSELFOutputData)
    ProblemType_RossbyWave = myDGSEM2D.myDGSEM2DParameters.ProblemType_RossbyWave
    dx = myDGSEM2D.myNameList.lX/nElementsX
    if ProblemType_RossbyWave and ReadFromSELFOutputData:
        dt, nTime, FinalTime = SpecifyTimeStepAndNumberOfTimeStepsForRossbyWaves(ConvergenceType,TimeIntegrator)
    else:
        dt = np.zeros(nCases)
        nTime = np.zeros(nCases,dtype=int)
        nTime_Minimum = 50
        if ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space':
            if ProblemType_RossbyWave:
                dt_Minimum = 0.12
            else:
                dt_Minimum = myDGSEM2D.myNameList.dt
            if TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
                dt_Minimum *= 0.5
            elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
                if ProblemType_RossbyWave:
                    dt_Minimum *= 0.5
                else:
                    dt_Minimum *= 0.25
            elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
                if ProblemType_RossbyWave:
                    dt_Minimum *= 0.5
                else:
                    dt_Minimum *= 0.125
            dt[nCases-1] = dt_Minimum
            for iCase in reversed(range(0,nCases-1)):
                dt[iCase] = 2.0*dt[iCase+1]
            dt_Maximum = dt[0]
            FinalTime = nTime_Minimum*dt_Maximum
            if ConvergenceType == 'Space':
                dt[:] = dt_Minimum
        else: # if ConvergenceType == 'Time':
            if ProblemType_RossbyWave:
                dt[0] = 1.92
            else:
                dt[0] = myDGSEM2D.myNameList.dt
            if TimeIntegrator == 'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod':
                if ProblemType == 'Coastal_Kelvin_Wave' or ProblemType == 'Barotropic_Tide':
                    dt[0] *= 0.5
            elif TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
                dt[0] *= 0.5
            elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
                if ProblemType_RossbyWave:
                    dt[0] *= 0.5
                else:
                    dt[0] *= 0.25
            elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
                if ProblemType_RossbyWave:
                    dt[0] *= 0.5
                elif ProblemType == 'NonLinear_Manufactured_Solution':
                    dt[0] *= 0.25
                else:
                    dt[0] *= 0.125
            for iCase in range(1,nCases):
                dt[iCase] = 0.5*dt[iCase-1]
            dt_Maximum = dt[0]
            FinalTime = nTime_Minimum*dt_Maximum
    L2ErrorNorm = np.zeros((3,nCases))
    if ConvergenceType == 'Space':
        myDGSolution2DOnCoarsestMesh = np.zeros((nElementsX[0]*nElementsY[0],nXi+1,nEta+1,3,nCases))
    if ConvergenceType == 'Time':
        myDGSolution2D = np.zeros((nElementsX[0]*nElementsY[0],nXi+1,nEta+1,3,nCases))
    TimeIntegratorShortForm = SpecifyTimeIntegratorShortForm(TimeIntegrator)
    for iCase in range(0,nCases):
        nTime[iCase] = int(round(FinalTime/dt[iCase]))
        if iCase == 0:
            isCoarsestMesh = True
        else:
            isCoarsestMesh = False
        if ConvergenceType == 'SpaceAndTime':
            if PerformInterpolation and isCoarsestMesh:
                myDGNodalStorage2DOnCoarsestMesh, myCoarsestMesh, L2ErrorNorm[:,iCase] = (
                DetermineNumericalSolutionAndError(
                ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                CourantNumber_Advection,CourantNumber_Diffusion,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,nElementsX[iCase],
                nElementsY[iCase],nXi,nEta,nXiPlot,nEtaPlot,dt[iCase],nTime[iCase],PerformInterpolation,isCoarsestMesh))
            else: # if not(PerformInterpolation) or (PerformInterpolation and not(isCoarsestMesh)):
                if not(PerformInterpolation):
                    myDGNodalStorage2DOnCoarsestMesh = []
                    myCoarsestMesh = []
                L2ErrorNorm[:,iCase] = DetermineNumericalSolutionAndError(
                ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                CourantNumber_Advection,CourantNumber_Diffusion,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,nElementsX[iCase],
                nElementsY[iCase],nXi,nEta,nXiPlot,nEtaPlot,dt[iCase],nTime[iCase],PerformInterpolation,isCoarsestMesh,
                myDGNodalStorage2DOnCoarsestMesh,myCoarsestMesh,EntityToBeInterpolated)
        elif ConvergenceType == 'Space':
            if isCoarsestMesh:
                myDGNodalStorage2DOnCoarsestMesh, myCoarsestMesh, myDGSolution2DOnCoarsestMesh[:,:,:,:,iCase] = (
                DetermineNumericalSolutionAndError(
                ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                CourantNumber_Advection,CourantNumber_Diffusion,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,nElementsX[iCase],
                nElementsY[iCase],nXi,nEta,nXiPlot,nEtaPlot,dt[iCase],nTime[iCase],PerformInterpolation,isCoarsestMesh,
                ReadFromSELFOutputData=ReadFromSELFOutputData,TimeIntegratorShortForm=TimeIntegratorShortForm))
            else:
                myDGSolution2DOnCoarsestMesh[:,:,:,:,iCase] = DetermineNumericalSolutionAndError(
                ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                CourantNumber_Advection,CourantNumber_Diffusion,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,nElementsX[iCase],
                nElementsY[iCase],nXi,nEta,nXiPlot,nEtaPlot,dt[iCase],nTime[iCase],PerformInterpolation,isCoarsestMesh,
                myDGNodalStorage2DOnCoarsestMesh,myCoarsestMesh,EntityToBeInterpolated,ReadFromSELFOutputData,
                TimeIntegratorShortForm)
                myDGSolution2DOnCoarsestMeshDifference = (myDGSolution2DOnCoarsestMesh[:,:,:,:,iCase] 
                                                          - myDGSolution2DOnCoarsestMesh[:,:,:,:,iCase-1])
                L2ErrorNorm[:,iCase] = (
                DGSEM2DClass.ComputeErrorNormOnCoarsestMesh(myDGNodalStorage2DOnCoarsestMesh,myCoarsestMesh,
                                                            myDGSolution2DOnCoarsestMeshDifference))
        elif ConvergenceType == 'Time':
            myDGSolution2D[:,:,:,:,iCase] = DetermineNumericalSolutionAndError(
            ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
            CourantNumber_Advection,CourantNumber_Diffusion,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
            Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,nElementsX[iCase],
            nElementsY[iCase],nXi,nEta,nXiPlot,nEtaPlot,dt[iCase],nTime[iCase],
            EntityToBeInterpolated=EntityToBeInterpolated,ReadFromSELFOutputData=ReadFromSELFOutputData,
            TimeIntegratorShortForm=TimeIntegratorShortForm)
            if not(isCoarsestMesh):
                myDGSolution2DDifference = myDGSolution2D[:,:,:,:,iCase] - myDGSolution2D[:,:,:,:,iCase-1]
                L2ErrorNorm[:,iCase] = DGSEM2DClass.ComputeErrorNorm(myDGSEM2D,True,myDGSolution2DDifference)
    FileName = (myDGSEM2D.myNameList.ProblemType_FileName + '_'
                + myDGSEM2D.myNameList.myTimeSteppingParameters.TimeIntegratorShortForm + '_ConvergencePlot_'
                + ConvergenceType + '_L2ErrorNorm')
    if ConvergenceType == 'SpaceAndTime':
        nIntervals = nElementsX
        Intervals = dx
    elif ConvergenceType == 'Space':
        nIntervals = nElementsX[1:]
        Intervals = dx[1:]
        L2ErrorNorm = L2ErrorNorm[:,1:]
    else: # if ConvergenceType == 'Time':
        nIntervals = nTime[1:]
        Intervals = dt[1:]
        L2ErrorNorm = L2ErrorNorm[:,1:]
    WriteL2ErrorNorm(myDGSEM2D.OutputDirectory,nIntervals,Intervals,L2ErrorNorm,FileName)
    

def SpecifyProblemTypes(ConvergenceType):
    if ConvergenceType == 'SpaceAndTime':
        ProblemTypes = ['Coastal_Kelvin_Wave','Inertia_Gravity_Wave','Equatorial_Kelvin_Wave','Equatorial_Yanai_Wave',
                        'Equatorial_Rossby_Wave','Equatorial_Inertia_Gravity_Wave','Barotropic_Tide',
                        'NonLinear_Manufactured_Solution','Viscous_Burgers_Equation']
    elif ConvergenceType == 'Space' or ConvergenceType == 'Time':
        ProblemTypes = ['Coastal_Kelvin_Wave','Inertia_Gravity_Wave','Planetary_Rossby_Wave','Topographic_Rossby_Wave',
                        'Equatorial_Kelvin_Wave','Equatorial_Yanai_Wave','Equatorial_Rossby_Wave',
                        'Equatorial_Inertia_Gravity_Wave','Barotropic_Tide','NonLinear_Manufactured_Solution',
                        'Viscous_Burgers_Equation']
    return ProblemTypes


def SpecifyTimeIntegrators():
    TimeIntegrators = ['ExplicitMidpointMethod','SecondOrderAdamsBashforthMethod',
                       'WilliamsonLowStorageThirdOrderRungeKuttaMethod','ThirdOrderAdamsBashforthMethod',
                       'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod','FourthOrderAdamsBashforthMethod']
    LF_TR_and_LF_AM3_with_FB_Feedback_Types = ['' for x in range(0,len(TimeIntegrators))]
    Generalized_FB_with_AB2_AM3_Step_Types = ['' for x in range(0,len(TimeIntegrators))]
    Generalized_FB_with_AB3_AM4_Step_Types = ['' for x in range(0,len(TimeIntegrators))]
    return [TimeIntegrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
            Generalized_FB_with_AB3_AM4_Step_Types]

   
def SpecifyAsymptoticPointsForSlopeComputation(ConvergenceType,ProblemType,TimeIntegrator):
    if ConvergenceType == 'SpaceAndTime':
        iPointLowerLimit = 0
        iPointUpperLimit = 4
        if ProblemType == 'Coastal_Kelvin_Wave':
            if TimeIntegrator == 'ExplicitMidpointMethod':
                iPointLowerLimit = 2
                iPointUpperLimit = 4
            elif TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
                iPointLowerLimit = 3
                iPointUpperLimit = 4
        elif ProblemType == 'Inertia_Gravity_Wave':
            if TimeIntegrator == 'ExplicitMidpointMethod' or TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
                iPointLowerLimit = 3
                iPointUpperLimit = 4
            elif (TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' 
                  or TimeIntegrator == 'FourthOrderAdamsBashforthMethod'):
                iPointLowerLimit = 2
                iPointUpperLimit = 4
        elif ProblemType == 'Barotropic_Tide':
            if TimeIntegrator == 'ExplicitMidpointMethod' or TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
                iPointLowerLimit = 2
                iPointUpperLimit = 4
            elif TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
                iPointLowerLimit = 3
                iPointUpperLimit = 4
        elif ProblemType == 'NonLinear_Manufactured_Solution':
            if (TimeIntegrator == 'ExplicitMidpointMethod' or TimeIntegrator == 'SecondOrderAdamsBashforthMethod'
                or TimeIntegrator == 'FourthOrderAdamsBashforthMethod'):
                iPointLowerLimit = 3
                iPointUpperLimit = 4
            elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
                iPointLowerLimit = 2
                iPointUpperLimit = 4
    elif ConvergenceType == 'Space':
        iPointLowerLimit = 0
        iPointUpperLimit = 3
        if ProblemType == 'Coastal_Kelvin_Wave':
            if TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
                iPointLowerLimit = 1
        elif ProblemType == 'Barotropic_Tide' or ProblemType == 'NonLinear_Manufactured_Solution':
            if TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
                iPointLowerLimit = 2
    elif ConvergenceType == 'Time':
        if (ProblemType == 'Barotropic_Tide' 
            and TimeIntegrator == 'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod'):
            iPointLowerLimit = 1
        else:
            iPointLowerLimit = 0
        if ProblemType == 'NonLinear_Manufactured_Solution' and TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
            iPointUpperLimit = 2
        else:
            iPointUpperLimit = 3
    return iPointLowerLimit, iPointUpperLimit

   
def PlotConvergenceData(ConvergenceType,ProblemType,SingleTimeIntegrator=True,SingleTimeIntegratorIndex=1,
                        PlotOnlySurfaceElevationConvergenceData=True,PlotAgainstNumberOfCellsInZonalDirection=True,
                        PlotAgainstNumberOfTimeSteps=True,UseBestFitLine=False,set_xticks_manually=False,
                        ReadFromSELFOutputData=False):
    ProblemType_Title, ProblemType_FileName = Initialization.SpecifyTitleAndFileNamePrefixes(ProblemType)
    OutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output/' + ProblemType
    if ReadFromSELFOutputData:
        OutputDirectory += '_SELF_Output_Data'
    linewidth = 2.0
    linewidths = [2.0,2.0]
    linestyle = '-'
    linestyles  = [' ','-']
    color = 'k'
    colors = ['k','k']
    marker = True
    markers = [True,False]
    markertype = 's'
    markertypes = ['s','s']
    markersize = 10.0
    markersizes = [10.0,10.0]
    if ConvergenceType == 'Time':
        if PlotAgainstNumberOfTimeSteps:
            xLabel = 'Number of time steps'
        else:
            xLabel = 'Time step'
    elif (ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space'):
        if PlotAgainstNumberOfCellsInZonalDirection:
            xLabel = 'Number of cells in zonal direction'
        else:
            xLabel = 'Cell width'    
    labelfontsizes = [22.5,22.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    legendfontsize = 22.5
    if (((ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space') and PlotAgainstNumberOfCellsInZonalDirection)
        or (ConvergenceType == 'Time' and PlotAgainstNumberOfTimeSteps)):
        legendposition = 'upper right'
    else:
        legendposition = 'upper left'
        set_xticks_manually = False
    Title = ProblemType_Title
    if ConvergenceType == 'Space':
        Title += '\nRefinement in Space'
    elif ConvergenceType == 'Time':
        Title += '\nRefinement in Time'
    elif ConvergenceType == 'SpaceAndTime':
        Title += '\nRefinement in Space and Time'
    Title_Common = Title
    titlefontsize = 27.5 
    PlotConvergenceData_LogicalArray = np.ones(3,dtype=bool)
    if PlotOnlySurfaceElevationConvergenceData:
        PlotConvergenceData_LogicalArray[0:2] = False
    # PlotConvergenceData_LogicalArray[0] = PlotZonalVelocityConvergenceData
    # PlotConvergenceData_LogicalArray[1] = PlotMeridionalVelocityConvergenceData
    # PlotConvergenceData_LogicalArray[2] = PlotSurfaceElevationConvergenceData
    [TimeIntegrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
     Generalized_FB_with_AB3_AM4_Step_Types] = SpecifyTimeIntegrators()
    if SingleTimeIntegrator:
        iTimeIntegratorLowerLimit = SingleTimeIntegratorIndex
        iTimeIntegratorUpperLimit = SingleTimeIntegratorIndex + 1
    else:
        iTimeIntegratorLowerLimit = 0
        iTimeIntegratorUpperLimit = len(TimeIntegrators)
    for iTimeIntegrator in range(iTimeIntegratorLowerLimit,iTimeIntegratorUpperLimit):
        TimeIntegrator = TimeIntegrators[iTimeIntegrator]
        LF_TR_and_LF_AM3_with_FB_Feedback_Type = LF_TR_and_LF_AM3_with_FB_Feedback_Types[iTimeIntegrator]
        Generalized_FB_with_AB2_AM3_Step_Type = Generalized_FB_with_AB2_AM3_Step_Types[iTimeIntegrator]
        Generalized_FB_with_AB3_AM4_Step_Type = Generalized_FB_with_AB3_AM4_Step_Types[iTimeIntegrator]
        TimeIntegratorShortForm = (
        Initialization.DetermineTimeIntegratorShortForm(TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                        Generalized_FB_with_AB2_AM3_Step_Type,
                                                        Generalized_FB_with_AB3_AM4_Step_Type))
        FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_ConvergencePlot_' + ConvergenceType 
                    + '_L2ErrorNorm') 
        nIntervals, Intervals, L2ErrorNorm = ReadL2ErrorNorm(OutputDirectory,FileName+'.curve')
        if ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space':
            if PlotAgainstNumberOfCellsInZonalDirection:
                dx = nIntervals
            else:
                dx = Intervals
        elif ConvergenceType == 'Time':
            if PlotAgainstNumberOfTimeSteps:
                dx = nIntervals
            else:
                dx = Intervals
        iPointLowerLimit, iPointUpperLimit = SpecifyAsymptoticPointsForSlopeComputation(ConvergenceType,ProblemType,
                                                                                        TimeIntegrator)
        dx_SlopeComputation = dx[iPointLowerLimit:iPointUpperLimit+1]
        PlotZonalVelocityConvergenceData = PlotConvergenceData_LogicalArray[0]
        if PlotZonalVelocityConvergenceData: 
            yLabel = 'L$^2$ error norm of zonal velocity'
            labels = [xLabel,yLabel]
            ZonalVelocityL2ErrorNorm = L2ErrorNorm[0,:]
            if set_xticks_manually:
                xticks_set_manually = dx 
            else:
                xticks_set_manually = []
            A = np.vstack([np.log10(dx_SlopeComputation),np.ones(len(dx_SlopeComputation))]).T
            ZonalVelocityL2ErrorNorm_SlopeComputation = ZonalVelocityL2ErrorNorm[iPointLowerLimit:iPointUpperLimit+1]
            m, c = np.linalg.lstsq(A,np.log10(ZonalVelocityL2ErrorNorm_SlopeComputation),rcond=None)[0]
            if UseBestFitLine:
                ZonalVelocityL2ErrorNorm_BestFitLine = m*(np.log10(dx)) + c
                ZonalVelocityL2ErrorNorm_BestFitLine = 10.0**ZonalVelocityL2ErrorNorm_BestFitLine
                FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_ZonalVelocityConvergencePlot_' 
                            + ConvergenceType + '_L2ErrorNorm_BestFitLine')
                legends = ['L$^2$ error norm of\nzonal velocity','Best fit line: slope is %.2f' %m]
                CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,'log-log',dx,ZonalVelocityL2ErrorNorm,
                                                    ZonalVelocityL2ErrorNorm_BestFitLine,linewidths,linestyles,colors,
                                                    markers,markertypes,markersizes,labels,labelfontsizes,labelpads,
                                                    tickfontsizes,legends,legendfontsize,legendposition,Title,
                                                    titlefontsize,True,FileName,False,drawMajorGrid=True,
                                                    drawMinorGrid=True,legendWithinBox=True,
                                                    set_xticks_manually=set_xticks_manually,
                                                    xticks_set_manually=xticks_set_manually,FileFormat='pdf')
            else:
                FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_ZonalVelocityConvergencePlot_' 
                            + ConvergenceType + '_L2ErrorNorm')
                Title = Title_Common + ': Slope is %.2f' %m
                CR.PythonPlot1DSaveAsPDF(OutputDirectory,'log-log',dx,ZonalVelocityL2ErrorNorm,linewidth,linestyle,
                                         color,marker,markertype,markersize,labels,labelfontsizes,labelpads,
                                         tickfontsizes,Title,titlefontsize,True,FileName,False,fig_size=[9.25,9.25],
                                         useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                         drawMinorGrid=True,set_xticks_manually=set_xticks_manually,
                                         xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        PlotMeridionalVelocityConvergenceData = PlotConvergenceData_LogicalArray[1]
        if PlotMeridionalVelocityConvergenceData:            
            yLabel = 'L$^2$ error norm of meridional velocity'
            labels = [xLabel,yLabel]
            MeridionalVelocityL2ErrorNorm = L2ErrorNorm[1,:]
            if set_xticks_manually:
                xticks_set_manually = dx
            else:
                xticks_set_manually = []
            A = np.vstack([np.log10(dx_SlopeComputation),np.ones(len(dx_SlopeComputation))]).T
            MeridionalVelocityL2ErrorNorm_SlopeComputation = (
            MeridionalVelocityL2ErrorNorm[iPointLowerLimit:iPointUpperLimit+1])
            m, c = np.linalg.lstsq(A,np.log10(MeridionalVelocityL2ErrorNorm_SlopeComputation),rcond=None)[0]
            if UseBestFitLine:
                MeridionalVelocityL2ErrorNorm_BestFitLine = m*(np.log10(dx)) + c
                MeridionalVelocityL2ErrorNorm_BestFitLine = 10.0**MeridionalVelocityL2ErrorNorm_BestFitLine    
                FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm 
                            + '_MeridionalVelocityConvergencePlot_' + ConvergenceType + '_L2ErrorNorm_BestFitLine')
                legends = ['L$^2$ error norm of\nmeridional velocity','Best fit line: slope is %.2f' %m]
                CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,'log-log',dx,MeridionalVelocityL2ErrorNorm,
                                                    MeridionalVelocityL2ErrorNorm_BestFitLine,linewidths,linestyles,
                                                    colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                                    labelpads,tickfontsizes,legends,legendfontsize,legendposition,Title,
                                                    titlefontsize,True,FileName,False,drawMajorGrid=True,
                                                    drawMinorGrid=True,legendWithinBox=True,
                                                    set_xticks_manually=set_xticks_manually,
                                                    xticks_set_manually=xticks_set_manually,FileFormat='pdf')
            else:
                FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm 
                            + '_MeridionalVelocityConvergencePlot_' + ConvergenceType + '_L2ErrorNorm')
                Title = Title_Common + ': Slope is %.2f' %m
                CR.PythonPlot1DSaveAsPDF(OutputDirectory,'log-log',dx,MeridionalVelocityL2ErrorNorm,linewidth,linestyle,
                                         color,marker,markertype,markersize,labels,labelfontsizes,labelpads,
                                         tickfontsizes,Title,titlefontsize,True,FileName,False,fig_size=[9.25,9.25],
                                         useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                         drawMinorGrid=True,set_xticks_manually=set_xticks_manually,
                                         xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        PlotSurfaceElevationConvergenceData = PlotConvergenceData_LogicalArray[2]
        if PlotSurfaceElevationConvergenceData:  
            yLabel = 'L$^2$ error norm of SSH'
            labels = [xLabel,yLabel]
            # Note that yLabel was initially specified as 'L$^2$ Error Norm of Numerical Surface Elevation'.
            SurfaceElevationL2ErrorNorm = L2ErrorNorm[2,:]
            if set_xticks_manually:
                xticks_set_manually = dx
            else:
                xticks_set_manually = []
            A = np.vstack([np.log10(dx_SlopeComputation),np.ones(len(dx_SlopeComputation))]).T
            SurfaceElevationL2ErrorNorm_SlopeComputation = (
            SurfaceElevationL2ErrorNorm[iPointLowerLimit:iPointUpperLimit+1])
            m, c = np.linalg.lstsq(A,np.log10(SurfaceElevationL2ErrorNorm_SlopeComputation),rcond=None)[0]
            if UseBestFitLine:
                SurfaceElevationL2ErrorNorm_BestFitLine = m*(np.log10(dx)) + c
                SurfaceElevationL2ErrorNorm_BestFitLine = 10.0**SurfaceElevationL2ErrorNorm_BestFitLine  
                FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_SurfaceElevationConvergencePlot_' 
                            + ConvergenceType + '_L2ErrorNorm_BestFitLine')
                legends = ['L$^2$ error norm of SSH','Best fit line: slope is %.2f' %m]
                CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,'log-log',dx,SurfaceElevationL2ErrorNorm,
                                                    SurfaceElevationL2ErrorNorm_BestFitLine,linewidths,linestyles,
                                                    colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                                    labelpads,tickfontsizes,legends,legendfontsize,legendposition,Title,
                                                    titlefontsize,True,FileName,False,drawMajorGrid=True,
                                                    drawMinorGrid=True,legendWithinBox=True,
                                                    set_xticks_manually=set_xticks_manually,
                                                    xticks_set_manually=xticks_set_manually,FileFormat='pdf')
            else:
                FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_SurfaceElevationConvergencePlot_' 
                            + ConvergenceType + '_L2ErrorNorm')
                Title = Title_Common + ': Slope is %.2f' %m
                CR.PythonPlot1DSaveAsPDF(OutputDirectory,'log-log',dx,SurfaceElevationL2ErrorNorm,linewidth,linestyle,
                                         color,marker,markertype,markersize,labels,labelfontsizes,labelpads,
                                         tickfontsizes,Title,titlefontsize,True,FileName,False,fig_size=[9.25,9.25],
                                         useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                         drawMinorGrid=True,set_xticks_manually=set_xticks_manually,
                                         xticks_set_manually=xticks_set_manually,FileFormat='pdf')
                

def SpecifyShadowAndFrameAlpha():
    shadow = False
    framealpha = 0.75
    return shadow, framealpha
                
                
def SpecifyLegends(slopes):
    legends = ['RK2','AB2','RK3','AB3','RK4','AB4']            
    for i_legend in range(0,len(legends)):
        legends[i_legend] += ': s = %.2f' %slopes[i_legend]
    legendfontsize = 14.0
    legendpads = [0.0,0.0]
    return legends, legendfontsize, legendpads
                
                
def PlotAllConvergenceData(ConvergenceType,ProblemType,PlotOnlySurfaceElevationConvergenceData=True,
                           PlotAgainstNumberOfCellsInZonalDirection=True,PlotAgainstNumberOfTimeSteps=True,
                           UseBestFitLine=False,set_xticks_manually=False,ReadFromSELFOutputData=False):
    ProblemType_Title, ProblemType_FileName = Initialization.SpecifyTitleAndFileNamePrefixes(ProblemType)
    OutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output/' + ProblemType
    if ReadFromSELFOutputData:
        OutputDirectory += '_SELF_Output_Data'
    PlotConvergenceData_LogicalArray = np.ones(3,dtype=bool)
    if PlotOnlySurfaceElevationConvergenceData:
        PlotConvergenceData_LogicalArray[0:2] = False
    # PlotConvergenceData_LogicalArray[0] = PlotZonalVelocityConvergenceData
    # PlotConvergenceData_LogicalArray[1] = PlotMeridionalVelocityConvergenceData
    # PlotConvergenceData_LogicalArray[2] = PlotSurfaceElevationConvergenceData
    [TimeIntegrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
     Generalized_FB_with_AB3_AM4_Step_Types] = SpecifyTimeIntegrators()
    nTimeIntegrators = len(TimeIntegrators)
    TimeIntegrator = TimeIntegrators[0]
    LF_TR_and_LF_AM3_with_FB_Feedback_Type = LF_TR_and_LF_AM3_with_FB_Feedback_Types[0]
    Generalized_FB_with_AB2_AM3_Step_Type = Generalized_FB_with_AB2_AM3_Step_Types[0]
    Generalized_FB_with_AB3_AM4_Step_Type = Generalized_FB_with_AB3_AM4_Step_Types[0]
    TimeIntegratorShortForm = (
    Initialization.DetermineTimeIntegratorShortForm(TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                    Generalized_FB_with_AB2_AM3_Step_Type,
                                                    Generalized_FB_with_AB3_AM4_Step_Type))
    FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_ConvergencePlot_' + ConvergenceType 
                + '_L2ErrorNorm') 
    nIntervals, Intervals, L2ErrorNorm = ReadL2ErrorNorm(OutputDirectory,FileName+'.curve')
    if ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space':
        if PlotAgainstNumberOfCellsInZonalDirection:
            dx = nIntervals
        else:
            dx = Intervals
    elif ConvergenceType == 'Time':
        if PlotAgainstNumberOfTimeSteps:
            dx = nIntervals
        else:
            dx = Intervals
    nCases = len(dx)
    nSubplots = 2
    PlotZonalVelocityConvergenceData = PlotConvergenceData_LogicalArray[0]
    PlotMeridionalVelocityConvergenceData = PlotConvergenceData_LogicalArray[1]
    PlotSurfaceElevationConvergenceData = PlotConvergenceData_LogicalArray[2]
    L2ErrorNorm = np.zeros((nTimeIntegrators,3,nCases))
    if PlotZonalVelocityConvergenceData: 
        ZonalVelocityL2ErrorNorm = np.zeros((nTimeIntegrators,nCases))           
        if UseBestFitLine:
            ZonalVelocityL2ErrorNormWithBestFitLine = np.zeros((nTimeIntegrators,nSubplots,nCases))    
        mZonalVelocity = np.zeros(nTimeIntegrators)
    if PlotMeridionalVelocityConvergenceData: 
        MeridionalVelocityL2ErrorNorm = np.zeros((nTimeIntegrators,nCases))           
        if UseBestFitLine:
            MeridionalVelocityL2ErrorNormWithBestFitLine = np.zeros((nTimeIntegrators,nSubplots,nCases))    
        mMeridionalVelocity = np.zeros(nTimeIntegrators)
    if PlotSurfaceElevationConvergenceData:
        SurfaceElevationL2ErrorNorm = np.zeros((nTimeIntegrators,nCases))           
        if UseBestFitLine:
            SurfaceElevationL2ErrorNormWithBestFitLine = np.zeros((nTimeIntegrators,nSubplots,nCases))    
        mSurfaceElevation = np.zeros(nTimeIntegrators)
    for iTimeIntegrator in range(0,nTimeIntegrators):
        TimeIntegrator = TimeIntegrators[iTimeIntegrator]
        LF_TR_and_LF_AM3_with_FB_Feedback_Type = LF_TR_and_LF_AM3_with_FB_Feedback_Types[iTimeIntegrator]
        Generalized_FB_with_AB2_AM3_Step_Type = Generalized_FB_with_AB2_AM3_Step_Types[iTimeIntegrator]
        Generalized_FB_with_AB3_AM4_Step_Type = Generalized_FB_with_AB3_AM4_Step_Types[iTimeIntegrator]
        TimeIntegratorShortForm = (
        Initialization.DetermineTimeIntegratorShortForm(TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                        Generalized_FB_with_AB2_AM3_Step_Type,
                                                        Generalized_FB_with_AB3_AM4_Step_Type))
        FileName = (ProblemType_FileName + '_' + TimeIntegratorShortForm + '_ConvergencePlot_' + ConvergenceType 
                    + '_L2ErrorNorm') 
        nIntervals, Intervals, L2ErrorNorm[iTimeIntegrator,:,:] = ReadL2ErrorNorm(OutputDirectory,FileName+'.curve')
        iPointLowerLimit, iPointUpperLimit = SpecifyAsymptoticPointsForSlopeComputation(ConvergenceType,ProblemType,
                                                                                        TimeIntegrator)
        dx_SlopeComputation = dx[iPointLowerLimit:iPointUpperLimit+1]
        if PlotZonalVelocityConvergenceData:
            ZonalVelocityL2ErrorNorm[iTimeIntegrator,:] = L2ErrorNorm[iTimeIntegrator,0,:]
            if set_xticks_manually:
                xticks_set_manually = dx
            else:
                xticks_set_manually = []
            A = np.vstack([np.log10(dx_SlopeComputation),np.ones(len(dx_SlopeComputation))]).T
            ZonalVelocityL2ErrorNorm_SlopeComputation = ZonalVelocityL2ErrorNorm[iTimeIntegrator,
                                                                                 iPointLowerLimit:iPointUpperLimit+1]
            mZonalVelocity[iTimeIntegrator], c = (
            np.linalg.lstsq(A,np.log10(ZonalVelocityL2ErrorNorm_SlopeComputation),rcond=None)[0])
            if UseBestFitLine:
                y = mZonalVelocity[iTimeIntegrator]*(np.log10(dx)) + c
                y = 10.0**y
                ZonalVelocityL2ErrorNormWithBestFitLine[iTimeIntegrator,0,:] = (
                ZonalVelocityL2ErrorNorm[iTimeIntegrator,:])
                ZonalVelocityL2ErrorNormWithBestFitLine[iTimeIntegrator,1,:] = y
        if PlotMeridionalVelocityConvergenceData:
            MeridionalVelocityL2ErrorNorm[iTimeIntegrator,:] = L2ErrorNorm[iTimeIntegrator,1,:]
            if set_xticks_manually:
                xticks_set_manually = dx
            else:
                xticks_set_manually = []
            A = np.vstack([np.log10(dx_SlopeComputation),np.ones(len(dx_SlopeComputation))]).T
            MeridionalVelocityL2ErrorNorm_SlopeComputation = (
            MeridionalVelocityL2ErrorNorm[iTimeIntegrator,iPointLowerLimit:iPointUpperLimit+1])
            mMeridionalVelocity[iTimeIntegrator], c = (
            np.linalg.lstsq(A,np.log10(MeridionalVelocityL2ErrorNorm_SlopeComputation),rcond=None)[0])
            if UseBestFitLine:
                y = mMeridionalVelocity[iTimeIntegrator]*(np.log10(dx)) + c
                y = 10.0**y
                MeridionalVelocityL2ErrorNormWithBestFitLine[iTimeIntegrator,0,:] = (
                MeridionalVelocityL2ErrorNorm[iTimeIntegrator,:])
                MeridionalVelocityL2ErrorNormWithBestFitLine[iTimeIntegrator,1,:] = y          
        if PlotSurfaceElevationConvergenceData:
            SurfaceElevationL2ErrorNorm[iTimeIntegrator,:] = L2ErrorNorm[iTimeIntegrator,2,:]
            if set_xticks_manually:
                xticks_set_manually = dx
            else:
                xticks_set_manually = []
            A = np.vstack([np.log10(dx_SlopeComputation),np.ones(len(dx_SlopeComputation))]).T
            SurfaceElevationL2ErrorNorm_SlopeComputation = (
            SurfaceElevationL2ErrorNorm[iTimeIntegrator,iPointLowerLimit:iPointUpperLimit+1])
            mSurfaceElevation[iTimeIntegrator], c = (
            np.linalg.lstsq(A,np.log10(SurfaceElevationL2ErrorNorm_SlopeComputation),rcond=None)[0])
            if UseBestFitLine:
                y = mSurfaceElevation[iTimeIntegrator]*(np.log10(dx)) + c
                y = 10.0**y
                SurfaceElevationL2ErrorNormWithBestFitLine[iTimeIntegrator,0,:] = (
                SurfaceElevationL2ErrorNorm[iTimeIntegrator,:])
                SurfaceElevationL2ErrorNormWithBestFitLine[iTimeIntegrator,1,:] = y
    linewidths = 2.0*np.ones(nTimeIntegrators)
    if (ConvergenceType == 'Space' 
        and (ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave')):
        linestyles  = ['-','-','--','--',':',':']
    else:
        linestyles  = ['-','-','-','-','-','-']
    colors = ['indigo','darkviolet','blue','green','gold','red']
    markers = np.ones(nTimeIntegrators,dtype=bool)
    markertypes = ['o','H','h','s','D','^']
    markersizes = np.array([12.5,12.5,12.5,10.0,10.0,11.25])
    if ConvergenceType == 'Time':
        if PlotAgainstNumberOfTimeSteps:
            xLabel = 'Number of time steps'
        else:
            xLabel = 'Time step'
    elif (ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space'):
        xLabel = 'Number of cells in zonal direction'
    labelfontsizes = [22.5,22.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    Title = ProblemType_Title
    if ConvergenceType == 'Space':
        Title += '\nRefinement in Space'
    elif ConvergenceType == 'Time':
        Title += '\nRefinement in Time'
    elif ConvergenceType == 'SpaceAndTime':
        Title += '\nRefinement in Space and Time'  
    titlefontsize = 27.5      
    SaveAsPDF = True
    Show = False
    shadow, framealpha = SpecifyShadowAndFrameAlpha()
    if PlotZonalVelocityConvergenceData:
        if ConvergenceType == 'Space':
            yLabel = 'L$^2$ norm of difference in numerical solution\nof zonal velocity with refinement in space'
        elif ConvergenceType == 'Time':
            yLabel = 'L$^2$ norm of difference in numerical solution\nof zonal velocity with refinement in time'
        elif ConvergenceType == 'SpaceAndTime':
            yLabel = 'L$^2$ error norm of zonal velocity'
        labels = [xLabel,yLabel]
        legends, legendfontsize, legendpads = SpecifyLegends(mZonalVelocity)
        if (((ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space') 
             and PlotAgainstNumberOfCellsInZonalDirection) 
            or (ConvergenceType == 'Time' and PlotAgainstNumberOfTimeSteps)):
            legendposition = 'lower left'
        else:
            legendposition = 'lower right'
            set_xticks_manually = False
        FileName = (ProblemType_FileName + '_ZonalVelocityConvergencePlot_' + ConvergenceType + '_L2ErrorNorm')
        if UseBestFitLine:
            FileName += '_BestFitLine'
            CR.PythonConvergencePlots1DSaveAsPDF(OutputDirectory,'log-log',dx,
                                                 ZonalVelocityL2ErrorNormWithBestFitLine,linewidths,linestyles,
                                                 colors,markertypes,markersizes,labels,labelfontsizes,labelpads,
                                                 tickfontsizes,legends,legendfontsize,legendposition,Title,
                                                 titlefontsize,SaveAsPDF,FileName,Show,fig_size=[9.25,9.25],
                                                 useDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=True,
                                                 drawMinorGrid=True,legendWithinBox=False,legendpads=legendpads,
                                                 shadow=shadow,framealpha=framealpha,titlepad=1.035,
                                                 set_xticks_manually=set_xticks_manually,
                                                 xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        else:
            CR.PythonPlots1DSaveAsPDF(OutputDirectory,'log-log',dx,ZonalVelocityL2ErrorNorm,linewidths,linestyles,
                                      colors,markers,markertypes,markersizes,labels,labelfontsizes,labelpads,
                                      tickfontsizes,legends,legendfontsize,legendposition,Title,titlefontsize,SaveAsPDF,
                                      FileName,Show,fig_size=[9.25,9.25],useDefaultMethodToSpecifyTickFontSize=True,
                                      drawMajorGrid=True,drawMinorGrid=True,setXAxisLimits=[False,False],
                                      xAxisLimits=[0.0,0.0],setYAxisLimits=[False,False],yAxisLimits=[0.0,0.0],
                                      legendWithinBox=True,legendpads=legendpads,shadow=shadow,framealpha=framealpha,
                                      titlepad=1.035,set_xticks_manually=set_xticks_manually,
                                      xticks_set_manually=xticks_set_manually,FileFormat='pdf')
    if PlotMeridionalVelocityConvergenceData:
        if ConvergenceType == 'Space':
            yLabel = (
            'L$^2$ norm of difference in numerical solution\nof meridional velocity with refinement in space')
        elif ConvergenceType == 'Time':
            yLabel = (
            'L$^2$ norm of difference in numerical solution\nof meridional velocity with refinement in time')
        elif ConvergenceType == 'SpaceAndTime':
            yLabel = 'L$^2$ error norm of meridional velocity'
        labels = [xLabel,yLabel]
        legends, legendfontsize, legendpads = SpecifyLegends(mMeridionalVelocity)
        if (((ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space') 
             and PlotAgainstNumberOfCellsInZonalDirection) 
            or (ConvergenceType == 'Time' and PlotAgainstNumberOfTimeSteps)):
            legendposition = 'lower left'
        else:
            legendposition = 'lower right'
            set_xticks_manually = False
        FileName = (ProblemType_FileName + '_MeridionalVelocityConvergencePlot_' + ConvergenceType + '_L2ErrorNorm')
        if UseBestFitLine:
            FileName += '_BestFitLine'
            CR.PythonConvergencePlots1DSaveAsPDF(OutputDirectory,'log-log',dx,
                                                 MeridionalVelocityL2ErrorNormWithBestFitLine,linewidths,
                                                 linestyles,colors,markertypes,markersizes,labels,labelfontsizes,
                                                 labelpads,tickfontsizes,legends,legendfontsize,legendposition,
                                                 Title,titlefontsize,SaveAsPDF,FileName,Show,
                                                 fig_size=[9.25,9.25],useDefaultMethodToSpecifyTickFontSize=True,
                                                 drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=False,
                                                 legendpads=legendpads,shadow=shadow,framealpha=framealpha,
                                                 titlepad=1.035,set_xticks_manually=set_xticks_manually,
                                                 xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        else:
            CR.PythonPlots1DSaveAsPDF(OutputDirectory,'log-log',dx,MeridionalVelocityL2ErrorNorm,linewidths,linestyles,
                                      colors,markers,markertypes,markersizes,labels,labelfontsizes,labelpads,
                                      tickfontsizes,legends,legendfontsize,legendposition,Title,titlefontsize,SaveAsPDF,
                                      FileName,Show,fig_size=[9.25,9.25],useDefaultMethodToSpecifyTickFontSize=True,
                                      drawMajorGrid=True,drawMinorGrid=True,setXAxisLimits=[False,False],
                                      xAxisLimits=[0.0,0.0],setYAxisLimits=[False,False],yAxisLimits=[0.0,0.0],
                                      legendWithinBox=True,legendpads=legendpads,shadow=shadow,framealpha=framealpha,
                                      titlepad=1.035,set_xticks_manually=set_xticks_manually,
                                      xticks_set_manually=xticks_set_manually,FileFormat='pdf')
    if PlotSurfaceElevationConvergenceData:
        if ConvergenceType == 'Space':
            yLabel = 'L$^2$ norm of difference in numerical solution\nof SSH with refinement in space'
        elif ConvergenceType == 'Time':
            yLabel = 'L$^2$ norm of difference in numerical solution\nof SSH with refinement in time'
        elif ConvergenceType == 'SpaceAndTime':
            yLabel = 'L$^2$ error norm of SSH'
        labels = [xLabel,yLabel]
        legends, legendfontsize, legendpads = SpecifyLegends(mSurfaceElevation)
        if (((ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space') 
             and PlotAgainstNumberOfCellsInZonalDirection)
            or (ConvergenceType == 'Time' and PlotAgainstNumberOfTimeSteps)):
            legendposition = 'lower left'
        else:
            legendposition = 'lower right'
            set_xticks_manually = False
        FileName = (ProblemType_FileName + '_SurfaceElevationConvergencePlot_' + ConvergenceType + '_L2ErrorNorm')
        if UseBestFitLine:
            FileName += '_BestFitLine'
            CR.PythonConvergencePlots1DSaveAsPDF(OutputDirectory,'log-log',dx,
                                                 SurfaceElevationL2ErrorNormWithBestFitLine,linewidths,linestyles,
                                                 colors,markertypes,markersizes,labels,labelfontsizes,labelpads,
                                                 tickfontsizes,legends,legendfontsize,legendposition,Title,
                                                 titlefontsize,SaveAsPDF,FileName,Show,fig_size=[9.25,9.25],
                                                 useDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=True,
                                                 drawMinorGrid=True,legendWithinBox=False,legendpads=legendpads,
                                                 shadow=shadow,framealpha=framealpha,titlepad=1.035,
                                                 set_xticks_manually=set_xticks_manually,
                                                 xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        else:
            CR.PythonPlots1DSaveAsPDF(OutputDirectory,'log-log',dx,SurfaceElevationL2ErrorNorm,linewidths,linestyles,
                                      colors,markers,markertypes,markersizes,labels,labelfontsizes,labelpads,
                                      tickfontsizes,legends,legendfontsize,legendposition,Title,titlefontsize,SaveAsPDF,
                                      FileName,Show,fig_size=[9.25,9.25],useDefaultMethodToSpecifyTickFontSize=True,
                                      drawMajorGrid=True,drawMinorGrid=True,setXAxisLimits=[False,False],
                                      xAxisLimits=[0.0,0.0],setYAxisLimits=[False,False],yAxisLimits=[0.0,0.0],
                                      legendWithinBox=True,legendpads=legendpads,shadow=shadow,framealpha=framealpha,
                                      titlepad=1.035,set_xticks_manually=set_xticks_manually,
                                      xticks_set_manually=xticks_set_manually,FileFormat='pdf')