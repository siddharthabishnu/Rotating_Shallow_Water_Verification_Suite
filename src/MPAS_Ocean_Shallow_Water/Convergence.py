"""
Name: Main.py
Author: Sid Bishnu
Details: This script contains functions for performing convergence studies of the various test cases with refinement in
both space and time, only in space, and only in time.
"""


import numpy as np
from IPython.utils import io
import os
with io.capture_output() as captured:
    import CommonRoutines as CR
    import Initialization
    import MeshClass
    import MPASOceanShallowWaterClass
    import TimeSteppingMethods as TSM


def InterpolateSolutionToRectilinearMesh(myMPASOceanShallowWater,State):
    BoundaryCondition = myMPASOceanShallowWater.myNameList.BoundaryCondition
    myMesh = myMPASOceanShallowWater.myMesh
    nCells = myMPASOceanShallowWater.myMesh.nCells
    xCellOnRectilinearMesh, yCellOnRectilinearMesh = myMesh.GenerateRectilinearMPASOceanMesh(BoundaryCondition)
    mySolutionOnMesh = (
    MPASOceanShallowWaterClass.ExpressStateAtCellCentersAsOneMultiDimensionalArray(myMPASOceanShallowWater,State))
    nVariables = 3
    mySolutionOnRectilinearMesh = np.zeros((nCells,3))
    for iVariable in range(0,nVariables):
        mySolutionOnRectilinearMesh[:,iVariable] = (
        myMesh.InterpolateSolutionFromMPASOceanMeshToRectilinearMPASOceanMesh(BoundaryCondition,
                                                                              mySolutionOnMesh[:,iVariable]))
    return xCellOnRectilinearMesh, yCellOnRectilinearMesh, mySolutionOnRectilinearMesh


def InterpolateSolutionToCoarsestRectilinearMesh(myMPASOceanShallowWater,xCellOnCoarsestRectilinearMesh, 
                                                 yCellOnCoarsestRectilinearMesh,State):
    BoundaryCondition = myMPASOceanShallowWater.myNameList.BoundaryCondition
    nCellsOnCoarsestMesh = len(xCellOnCoarsestRectilinearMesh)
    myFineMesh = myMPASOceanShallowWater.myMesh
    nCellsOnFineMesh = myFineMesh.nCells
    xCellOnFineRectilinearMesh, yCellOnFineRectilinearMesh = (
    myMPASOceanShallowWater.myMesh.GenerateRectilinearMPASOceanMesh(BoundaryCondition))
    mySolutionOnFineMesh = (
    MPASOceanShallowWaterClass.ExpressStateAtCellCentersAsOneMultiDimensionalArray(myMPASOceanShallowWater,State))
    nVariables = 3
    mySolutionOnFineRectilinearMesh = np.zeros((nCellsOnFineMesh,3))
    mySolutionInterpolatedToCoarsestRectilinearMesh = np.zeros((nCellsOnCoarsestMesh,3))
    for iVariable in range(0,nVariables):
        mySolutionOnFineRectilinearMesh[:,iVariable] = (
        myFineMesh.InterpolateSolutionFromMPASOceanMeshToRectilinearMPASOceanMesh(
        BoundaryCondition,mySolutionOnFineMesh[:,iVariable]))
        mySolutionInterpolatedToCoarsestRectilinearMesh[:,iVariable] = (
        MeshClass.InterpolateSolutionToCoarsestRectilinearMPASOceanMesh(
        myFineMesh.dx,xCellOnFineRectilinearMesh,yCellOnFineRectilinearMesh,
        mySolutionOnFineRectilinearMesh[:,iVariable],xCellOnCoarsestRectilinearMesh,yCellOnCoarsestRectilinearMesh))
    return mySolutionInterpolatedToCoarsestRectilinearMesh


def InterpolateSolutionToCoarsestMesh(myMPASOceanShallowWater,myCoarsestMesh,CellsOnCoarsestMeshToBeConsidered,State):
    BoundaryCondition = myMPASOceanShallowWater.myNameList.BoundaryCondition
    nCellsOnCoarsestMesh = myCoarsestMesh.nCells
    xCellOnCoarsestMesh = myCoarsestMesh.xCell
    yCellOnCoarsestMesh = myCoarsestMesh.yCell
    myFineMesh = myMPASOceanShallowWater.myMesh
    nCellsOnFineMesh = myFineMesh.nCells
    xCellOnFineRectilinearMesh, yCellOnFineRectilinearMesh = (
    myMPASOceanShallowWater.myMesh.GenerateRectilinearMPASOceanMesh(BoundaryCondition))
    mySolutionOnFineMesh = (
    MPASOceanShallowWaterClass.ExpressStateAtCellCentersAsOneMultiDimensionalArray(myMPASOceanShallowWater,State))
    nVariables = 3
    mySolutionOnFineRectilinearMesh = np.zeros((nCellsOnFineMesh,3))
    mySolutionInterpolatedToCoarsestRectilinearMesh = np.zeros((nCellsOnCoarsestMesh,3))
    for iVariable in range(0,nVariables):
        mySolutionOnFineRectilinearMesh[:,iVariable] = (
        myFineMesh.InterpolateSolutionFromMPASOceanMeshToRectilinearMPASOceanMesh(
        BoundaryCondition,mySolutionOnFineMesh[:,iVariable]))
        mySolutionInterpolatedToCoarsestRectilinearMesh[:,iVariable] = (
        MeshClass.InterpolateSolutionToCoarsestMPASOceanMesh(
        myFineMesh.dx,xCellOnFineRectilinearMesh,yCellOnFineRectilinearMesh,
        mySolutionOnFineRectilinearMesh[:,iVariable],xCellOnCoarsestMesh,yCellOnCoarsestMesh,
        CellsOnCoarsestMeshToBeConsidered))
    return mySolutionInterpolatedToCoarsestRectilinearMesh


def DetermineNumericalSolutionAndError(ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,
                                       PrintAmplitudesOfWaveModes,CourantNumber,TimeIntegrator,
                                       LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                       Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,
                                       MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
                                       UseAveragedQuantities,dt,nTime,InterpolateExactVelocitiesFromEdgesToCellCenters,
                                       PerformInterpolation=True,PerformInterpolationToCoarsestRectilinearMesh=False,
                                       isCoarsestMesh=True,xCellOnCoarsestRectilinearMesh=[],
                                       yCellOnCoarsestRectilinearMesh=[],myCoarsestMesh=[],
                                       EntityToBeInterpolated='Error'):
    UseCourantNumberToDetermineTimeStep = False 
    myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,
    UseAveragedQuantities,CourantNumber,UseCourantNumberToDetermineTimeStep)
    BoundaryCondition = myMPASOceanShallowWater.myNameList.BoundaryCondition
    myMPASOceanShallowWater.myNameList.dt = dt
    myMPASOceanShallowWater.myNameList.nTime = nTime
    print('The number of time steps for the %3d x %3d mesh is %3d.' %(nCellsX,nCellsY,nTime))
    DisplayProgress = True
    for iTime in range(0,nTime+1):
        if DisplayProgress:
            print('Displaying Progress: iTime = %3d.' %iTime)
        myMPASOceanShallowWater.iTime = iTime
        myMPASOceanShallowWater.time = float(iTime)*dt
        if iTime == 0 or iTime == nTime:   
            MPASOceanShallowWaterClass.DetermineExactSolutions(myMPASOceanShallowWater,
                                                               InterpolateExactVelocitiesFromEdgesToCellCenters)
        if iTime == 0:
            MPASOceanShallowWaterClass.SpecifyInitialConditions(myMPASOceanShallowWater)
        if iTime == nTime:
            print('The final time for the %3d x %3d mesh is %.6f seconds.' %(nCellsX,nCellsY,
                                                                             myMPASOceanShallowWater.time))
            if not(myMPASOceanShallowWater.myNameList.ProblemType_NoExactSolution):
                MPASOceanShallowWaterClass.ComputeError(myMPASOceanShallowWater)
            if EntityToBeInterpolated == 'Solution':
                State = 'Numerical'
            else: # if EntityToBeInterpolated == 'Error':
                State = 'Error'
            if ConvergenceType == 'SpaceAndTime':
                if not(PerformInterpolation):
                    L2ErrorNorm = MPASOceanShallowWaterClass.ComputeErrorNorm(myMPASOceanShallowWater)
                else: # if PerformInterpolation:
                    if PerformInterpolationToCoarsestRectilinearMesh:
                        if isCoarsestMesh:
                            nCellsOnCoarsestRectilinearMesh = myMPASOceanShallowWater.myMesh.nCells
                            (xCellOnCoarsestRectilinearMesh, yCellOnCoarsestRectilinearMesh,
                             mySolutionOnCoarsestRectilinearMesh) = (
                            InterpolateSolutionToRectilinearMesh(myMPASOceanShallowWater,State))
                        else:
                            nCellsOnCoarsestRectilinearMesh = len(xCellOnCoarsestRectilinearMesh)
                            mySolutionOnCoarsestRectilinearMesh = (
                            InterpolateSolutionToCoarsestRectilinearMesh(myMPASOceanShallowWater,
                                                                         xCellOnCoarsestRectilinearMesh,
                                                                         yCellOnCoarsestRectilinearMesh,State))
                        L2ErrorNorm = (
                        MPASOceanShallowWaterClass.ComputeErrorNormOnCoarsestRectilinearMesh(
                        nCellsOnCoarsestRectilinearMesh,mySolutionOnCoarsestRectilinearMesh))
                    else:
                        if isCoarsestMesh:
                            myCoarsestMesh = myMPASOceanShallowWater.myMesh
                            nCellsOnCoarsestMeshToBeConsidered, CellsOnCoarsestMeshToBeConsidered = (
                            myCoarsestMesh.DetermineCellsToBeConsideredForErrorComputation(BoundaryCondition))
                            mySolutionOnCoarsestMesh = (
                            MPASOceanShallowWaterClass.ExpressStateAtCellCentersAsOneMultiDimensionalArray(
                            myMPASOceanShallowWater,State))
                        else:
                            nCellsOnCoarsestMeshToBeConsidered, CellsOnCoarsestMeshToBeConsidered = (
                            myCoarsestMesh.DetermineCellsToBeConsideredForErrorComputation(BoundaryCondition))
                            mySolutionOnCoarsestMesh = (
                            InterpolateSolutionToCoarsestMesh(myMPASOceanShallowWater,myCoarsestMesh,
                                                              CellsOnCoarsestMeshToBeConsidered,State))
                        L2ErrorNorm = (
                        MPASOceanShallowWaterClass.ComputeErrorNormOnCoarsestMesh(
                        myCoarsestMesh,nCellsOnCoarsestMeshToBeConsidered,CellsOnCoarsestMeshToBeConsidered,
                        mySolutionOnCoarsestMesh))
            elif ConvergenceType == 'Space':
                if PerformInterpolationToCoarsestRectilinearMesh:
                    if isCoarsestMesh:
                        (xCellOnCoarsestRectilinearMesh, yCellOnCoarsestRectilinearMesh,
                         mySolutionOnCoarsestRectilinearMesh) = (
                        InterpolateSolutionToRectilinearMesh(myMPASOceanShallowWater,State))
                    else:
                        mySolutionOnCoarsestRectilinearMesh = (
                        InterpolateSolutionToCoarsestRectilinearMesh(myMPASOceanShallowWater,
                                                                     xCellOnCoarsestRectilinearMesh,
                                                                     yCellOnCoarsestRectilinearMesh,State))
                else:
                    if isCoarsestMesh:
                        mySolutionOnCoarsestMesh = (
                        MPASOceanShallowWaterClass.ExpressStateAtCellCentersAsOneMultiDimensionalArray(
                        myMPASOceanShallowWater,State))
                    else:
                        nCellsOnCoarsestMeshToBeConsidered, CellsOnCoarsestMeshToBeConsidered = (
                        myCoarsestMesh.DetermineCellsToBeConsideredForErrorComputation(BoundaryCondition))
                        mySolutionOnCoarsestMesh = (
                        InterpolateSolutionToCoarsestMesh(myMPASOceanShallowWater,myCoarsestMesh,
                                                          CellsOnCoarsestMeshToBeConsidered,State))
            elif ConvergenceType == 'Time':
                mySolution = MPASOceanShallowWaterClass.ExpressStateAtCellCentersAsOneMultiDimensionalArray(
                myMPASOceanShallowWater,State)
        if iTime < nTime:
            TSM.TimeIntegration(myMPASOceanShallowWater)
    if ConvergenceType == 'SpaceAndTime':
        if not(PerformInterpolation):
            return L2ErrorNorm
        else: # if PerformInterpolation:
            if PerformInterpolationToCoarsestRectilinearMesh:
                if isCoarsestMesh:
                    return xCellOnCoarsestRectilinearMesh, yCellOnCoarsestRectilinearMesh, L2ErrorNorm
                else:
                    return L2ErrorNorm
            else:
                if isCoarsestMesh:
                    return myMPASOceanShallowWater.myMesh, L2ErrorNorm
                else:
                    return L2ErrorNorm
    elif ConvergenceType == 'Space':
        if PerformInterpolationToCoarsestRectilinearMesh:
            if isCoarsestMesh:
                return (xCellOnCoarsestRectilinearMesh, yCellOnCoarsestRectilinearMesh, 
                        mySolutionOnCoarsestRectilinearMesh)
            else:
                return mySolutionOnCoarsestRectilinearMesh
        else:
            if isCoarsestMesh:
                return myMPASOceanShallowWater.myMesh, mySolutionOnCoarsestMesh
            else:
                return nCellsOnCoarsestMeshToBeConsidered, CellsOnCoarsestMeshToBeConsidered, mySolutionOnCoarsestMesh
    elif ConvergenceType == 'Time':
        return mySolution

        
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
        
        
def SpecifyNumberOfCells(ConvergenceType,ReturnSubscripts=False):
    if ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space':
        nCellsX = np.array([64,96,144,216,324])
        Subscripts = ['64x64','96x96','144x144','216x216','324x324']
    else: # if ConvergenceType == 'Time':
        nCellsX = np.array([64,64,64,64,64])
        Subscripts = ['64x64','64x64','64x64','64x64','64x64']
    if ReturnSubscripts:
        return nCellsX, Subscripts
    else:
        return nCellsX
        
        
def SpecifyMeshDirectoryAndMeshFileNamesForConvergenceStudy(ConvergenceType,ProblemType,ProblemType_EquatorialWave):
    MeshDirectory = Initialization.SpecifyMeshDirectoryAndMeshFileNames(ProblemType,ProblemType_EquatorialWave,
                                                                        ReturnOnlyMeshDirectory=True)
    MeshDirectory += '/ConvergenceStudyMeshes'
    nCellsX, Subscripts = SpecifyNumberOfCells(ConvergenceType,ReturnSubscripts=True)
    nCases = len(nCellsX)
    BaseMeshFileNames = ['' for x in range(0,nCases)]
    MeshFileNames = ['' for x in range(0,len(nCellsX))]
    if (ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Inertia_Gravity_Wave' 
        or ProblemType == 'NonLinear_Manufactured_Solution'):
        BaseMeshFileNameRoot = 'base_mesh'
    else:
        BaseMeshFileNameRoot = 'culled_mesh'
    MeshFileNameRoot = 'mesh'
    for iCase in range(0,nCases):
        BaseMeshFileNames[iCase] = BaseMeshFileNameRoot + '_%s.nc' %Subscripts[iCase]
        MeshFileNames[iCase] = MeshFileNameRoot + '_%s.nc' %Subscripts[iCase]
    return MeshDirectory, BaseMeshFileNames, MeshFileNames
    
    
def ConvergenceStudy(ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                     Generalized_FB_with_AB3_AM4_Step_Type,PrintBasicGeometry,FixAngleEdge,PrintOutput,
                     UseAveragedQuantities,InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                     PerformInterpolationToCoarsestRectilinearMesh,EntityToBeInterpolated):
    nCellsX = SpecifyNumberOfCells(ConvergenceType)
    nCellsY = nCellsX
    ProblemType_EquatorialWave = Initialization.isEquatorialWave(ProblemType)
    MeshDirectory, BaseMeshFileNames, MeshFileNames = (
    SpecifyMeshDirectoryAndMeshFileNamesForConvergenceStudy(ConvergenceType,ProblemType,ProblemType_EquatorialWave))
    nCases = len(nCellsX)
    if ConvergenceType == 'Time':
        CourantNumber = 0.5
    else:
        CourantNumber = 0.25
    UseCourantNumberToDetermineTimeStep = True
    BaseMeshFileName = BaseMeshFileNames[nCases-1]
    MeshFileName = MeshFileNames[nCases-1]
    myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
    ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
    nCellsX[nCases-1],nCellsY[nCases-1],PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,
    PrintOutput,UseAveragedQuantities,CourantNumber,UseCourantNumberToDetermineTimeStep)
    dx = myMPASOceanShallowWater.myNameList.lX/nCellsX
    dt = np.zeros(nCases)
    nTime = np.zeros(nCases,dtype=int)
    nTime_Minimum = 32
    if ConvergenceType == 'SpaceAndTime' or ConvergenceType == 'Space':
        if myMPASOceanShallowWater.myNameList.ProblemType_NoExactSolution:
            dt_Minimum = 0.125
        else:
            dt_Minimum = myMPASOceanShallowWater.myNameList.dt
        if TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
            dt_Minimum *= 0.5
        elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
            dt_Minimum *= 0.25
        elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
            dt_Minimum *= 0.125
        dt[nCases-1] = dt_Minimum
        for iCase in reversed(range(0,nCases-1)):
            dt[iCase] = 1.5*dt[iCase+1]
        dt_Maximum = dt[0]
        FinalTime = nTime_Minimum*dt_Maximum
        if ConvergenceType == 'Space':
            dt[:] = dt_Minimum
    else: # if ConvergenceType == 'Time':
        if myMPASOceanShallowWater.myNameList.ProblemType_NoExactSolution:
            dt[0] = 0.5
        else:
            dt[0] = myMPASOceanShallowWater.myNameList.dt
        if TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
            dt[0] *= 0.5
        elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
            dt[0] *= 0.25
        elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
            if (ProblemType == 'Coastal_Kelvin_Wave' or ProblemType == 'Inertia_Gravity_Wave' 
                or ProblemType == 'Barotropic_Tide'):
                dt[0] *= 0.25
            else:
                dt[0] *= 0.125
        for iCase in range(1,nCases):
            dt[iCase] = 0.5*dt[iCase-1]
        dt_Maximum = dt[0]
        FinalTime = nTime_Minimum*dt_Maximum
    L2ErrorNorm = np.zeros((3,nCases))
    if ConvergenceType == 'Space':
        if PerformInterpolationToCoarsestRectilinearMesh:
            mySolutionOnCoarsestRectilinearMesh = np.zeros((nCellsX[0]*nCellsY[0],3,nCases))
        else:
            mySolutionOnCoarsestMesh = np.zeros((nCellsX[0]*nCellsY[0],3,nCases))
    if ConvergenceType == 'Time':
        mySolution = np.zeros((nCellsX[0]*nCellsY[0],3,nCases))
    for iCase in range(0,nCases):
        BaseMeshFileName = BaseMeshFileNames[iCase]
        MeshFileName = MeshFileNames[iCase]
        nTime[iCase] = int(round(FinalTime/dt[iCase]))
        if iCase == 0:
            isCoarsestMesh = True
        else:
            isCoarsestMesh = False
        if ConvergenceType == 'SpaceAndTime':
            if not(PerformInterpolation):
                L2ErrorNorm[:,iCase] = (
                DetermineNumericalSolutionAndError(
                ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
                TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,MeshDirectory,
                BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,dt[iCase],nTime[iCase],
                InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                EntityToBeInterpolated=EntityToBeInterpolated))
            else: # if PerformInterpolation:
                if PerformInterpolationToCoarsestRectilinearMesh:
                    if isCoarsestMesh:
                        xCellOnCoarsestRectilinearMesh, yCellOnCoarsestRectilinearMesh, L2ErrorNorm[:,iCase] = (
                        DetermineNumericalSolutionAndError(
                        ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
                        TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                        Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,
                        MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                        dt[iCase],nTime[iCase],InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                        PerformInterpolationToCoarsestRectilinearMesh,isCoarsestMesh,
                        EntityToBeInterpolated=EntityToBeInterpolated))
                    else:
                        L2ErrorNorm[:,iCase] = (
                        DetermineNumericalSolutionAndError(
                        ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
                        TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                        Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,
                        MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                        dt[iCase],nTime[iCase],InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                        PerformInterpolationToCoarsestRectilinearMesh,isCoarsestMesh,xCellOnCoarsestRectilinearMesh,
                        yCellOnCoarsestRectilinearMesh,myCoarsestMesh=[],EntityToBeInterpolated=EntityToBeInterpolated))
                else:
                    if isCoarsestMesh:
                        myCoarsestMesh, L2ErrorNorm[:,iCase] = (
                        DetermineNumericalSolutionAndError(
                        ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
                        TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                        Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,
                        MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                        dt[iCase],nTime[iCase],InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                        PerformInterpolationToCoarsestRectilinearMesh,isCoarsestMesh,
                        EntityToBeInterpolated=EntityToBeInterpolated))
                    else:
                        L2ErrorNorm[:,iCase] = (
                        DetermineNumericalSolutionAndError(
                        ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
                        TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                        Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,
                        MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                        dt[iCase],nTime[iCase],InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                        PerformInterpolationToCoarsestRectilinearMesh,isCoarsestMesh,xCellOnCoarsestRectilinearMesh=[],
                        yCellOnCoarsestRectilinearMesh=[],myCoarsestMesh=myCoarsestMesh,
                        EntityToBeInterpolated=EntityToBeInterpolated))
        elif ConvergenceType == 'Space':
            if PerformInterpolationToCoarsestRectilinearMesh:
                if isCoarsestMesh:
                    (xCellOnCoarsestRectilinearMesh, yCellOnCoarsestRectilinearMesh, 
                     mySolutionOnCoarsestRectilinearMesh[:,:,iCase]) = (
                    DetermineNumericalSolutionAndError(
                    ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
                    TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                    Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,
                    MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                    dt[iCase],nTime[iCase],InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                    PerformInterpolationToCoarsestRectilinearMesh,isCoarsestMesh,
                    EntityToBeInterpolated=EntityToBeInterpolated))
                else:
                    mySolutionOnCoarsestRectilinearMesh[:,:,iCase] = (
                    DetermineNumericalSolutionAndError(
                    ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
                    TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                    Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,
                    MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                    dt[iCase],nTime[iCase],InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                    PerformInterpolationToCoarsestRectilinearMesh,isCoarsestMesh,xCellOnCoarsestRectilinearMesh,
                    yCellOnCoarsestRectilinearMesh,myCoarsestMesh=[],EntityToBeInterpolated=EntityToBeInterpolated))
                    nCellsOnCoarsestRectilinearMesh = len(xCellOnCoarsestRectilinearMesh)
                    mySolutionOnCoarsestRectilinearMeshDifference = (mySolutionOnCoarsestRectilinearMesh[:,:,iCase] 
                                                                     - mySolutionOnCoarsestRectilinearMesh[:,:,iCase-1])
                    L2ErrorNorm[:,iCase] = (
                    MPASOceanShallowWaterClass.ComputeErrorNormOnCoarsestRectilinearMesh(
                    nCellsOnCoarsestRectilinearMesh,mySolutionOnCoarsestRectilinearMeshDifference))
            else:
                if isCoarsestMesh:
                    myCoarsestMesh, mySolutionOnCoarsestMesh[:,:,iCase] = (
                    DetermineNumericalSolutionAndError(
                    ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
                    TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                    Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,
                    MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                    dt[iCase],nTime[iCase],InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                    PerformInterpolationToCoarsestRectilinearMesh,isCoarsestMesh,
                    EntityToBeInterpolated=EntityToBeInterpolated))
                else:
                    (nCellsOnCoarsestMeshToBeConsidered, CellsOnCoarsestMeshToBeConsidered, 
                     mySolutionOnCoarsestMesh[:,:,iCase]) = DetermineNumericalSolutionAndError(
                    ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
                    TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                    Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,
                    MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                    dt[iCase],nTime[iCase],InterpolateExactVelocitiesFromEdgesToCellCenters,PerformInterpolation,
                    PerformInterpolationToCoarsestRectilinearMesh,isCoarsestMesh,xCellOnCoarsestRectilinearMesh=[],
                    yCellOnCoarsestRectilinearMesh=[],myCoarsestMesh=myCoarsestMesh,
                    EntityToBeInterpolated=EntityToBeInterpolated)
                    mySolutionOnCoarsestMeshDifference = (mySolutionOnCoarsestMesh[:,:,iCase] 
                                                          - mySolutionOnCoarsestMesh[:,:,iCase-1])
                    L2ErrorNorm[:,iCase] = (
                    MPASOceanShallowWaterClass.ComputeErrorNormOnCoarsestMesh(
                    myCoarsestMesh,nCellsOnCoarsestMeshToBeConsidered,CellsOnCoarsestMeshToBeConsidered,
                    mySolutionOnCoarsestMeshDifference))
        elif ConvergenceType == 'Time':           
            mySolution[:,:,iCase] = DetermineNumericalSolutionAndError(
            ConvergenceType,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber,
            TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
            Generalized_FB_with_AB3_AM4_Step_Type,nCellsX[iCase],nCellsY[iCase],PrintBasicGeometry,MeshDirectory,
            BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,dt[iCase],nTime[iCase],
            InterpolateExactVelocitiesFromEdgesToCellCenters,EntityToBeInterpolated=EntityToBeInterpolated)
            if not(isCoarsestMesh):
                mySolutionDifference = mySolution[:,:,iCase] - mySolution[:,:,iCase-1]
                L2ErrorNorm[:,iCase] = MPASOceanShallowWaterClass.ComputeErrorNorm(myMPASOceanShallowWater,True,
                                                                                   mySolutionDifference)
    FileName = (myMPASOceanShallowWater.myNameList.ProblemType_FileName + '_'
                + myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.TimeIntegratorShortForm 
                + '_ConvergencePlot_' + ConvergenceType + '_L2ErrorNorm')
    if ConvergenceType == 'SpaceAndTime':
        nIntervals = nCellsX
        Intervals = dx
    elif ConvergenceType == 'Space':
        nIntervals = nCellsX[1:]
        Intervals = dx[1:]
        L2ErrorNorm = L2ErrorNorm[:,1:]
    else: # if ConvergenceType == 'Time':
        nIntervals = nTime[1:]
        Intervals = dt[1:]
        L2ErrorNorm = L2ErrorNorm[:,1:]
    WriteL2ErrorNorm(myMPASOceanShallowWater.OutputDirectory,nIntervals,Intervals,L2ErrorNorm,FileName)
    

def SetOfTimeIntegrators():
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
            and TimeIntegrator == 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'):
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
                        PlotAgainstNumberOfTimeSteps=True,UseBestFitLine=False,set_xticks_manually=False):
    ProblemType_Title, ProblemType_FileName = Initialization.SpecifyTitleAndFileNamePrefixes(ProblemType)
    OutputDirectory = '../../output/MPAS_Ocean_Shallow_Water_Output/' + ProblemType
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
    TitleOnCommon = Title
    titlefontsize = 27.5 
    PlotConvergenceData_LogicalArray = np.ones(3,dtype=bool)
    if PlotOnlySurfaceElevationConvergenceData:
        PlotConvergenceData_LogicalArray[0:2] = False
    # PlotConvergenceData_LogicalArray[0] = PlotZonalVelocityConvergenceData
    # PlotConvergenceData_LogicalArray[1] = PlotMeridionalVelocityConvergenceData
    # PlotConvergenceData_LogicalArray[2] = PlotSurfaceElevationConvergenceData
    [TimeIntegrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
     Generalized_FB_with_AB3_AM4_Step_Types] = SetOfTimeIntegrators()
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
                Title = TitleOnCommon + ': Slope is %.2f' %m
                CR.PythonPlot1DSaveAsPDF(OutputDirectory,'log-log',dx,ZonalVelocityL2ErrorNorm,linewidth,linestyle,
                                         color,marker,markertype,markersize,labels,labelfontsizes,labelpads,
                                         tickfontsizes,Title,titlefontsize,True,FileName,False,fig_size=[9.25,9.25],
                                         UseDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
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
                Title = TitleOnCommon + ': Slope is %.2f' %m
                CR.PythonPlot1DSaveAsPDF(OutputDirectory,'log-log',dx,MeridionalVelocityL2ErrorNorm,linewidth,linestyle,
                                         color,marker,markertype,markersize,labels,labelfontsizes,labelpads,
                                         tickfontsizes,Title,titlefontsize,True,FileName,False,fig_size=[9.25,9.25],
                                         UseDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
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
                Title = TitleOnCommon + ': Slope is %.2f' %m
                CR.PythonPlot1DSaveAsPDF(OutputDirectory,'log-log',dx,SurfaceElevationL2ErrorNorm,linewidth,linestyle,
                                         color,marker,markertype,markersize,labels,labelfontsizes,labelpads,
                                         tickfontsizes,Title,titlefontsize,True,FileName,False,fig_size=[9.25,9.25],
                                         UseDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                         drawMinorGrid=True,set_xticks_manually=set_xticks_manually,
                                         xticks_set_manually=xticks_set_manually,FileFormat='pdf')
                

def SpecifyLineStyles(ConvergenceType,ProblemType):
    if ConvergenceType == 'SpaceAndTime':
        if ProblemType == 'Coastal_Kelvin_Wave' or ProblemType == 'Barotropic_Tide':
            linestyles  = ['-','-','-','-','--','-']
        elif ProblemType == 'Inertia_Gravity_Wave':
            linestyles  = ['-','-','--','-',':','-']
    elif ConvergenceType == 'Space':
        if ProblemType == 'Coastal_Kelvin_Wave' or ProblemType == 'Barotropic_Tide':
            linestyles  = ['-','-','--','-',':','-']
    else:
        linestyles  = ['-','-','-','-','-','-']
    return linestyles


def SpecifyShadowAndFrameAlpha():
    shadow = False
    framealpha = 0.75
    return shadow, framealpha
                
                
def SetOfLegends(slopes):
    legends = ['RK2','AB2','RK3','AB3','RK4','AB4']            
    for i_legend in range(0,len(legends)):
        legends[i_legend] += ': s = %.2f' %slopes[i_legend]
    legendfontsize = 14.0
    legendpads = [0.0,0.0]
    return legends, legendfontsize, legendpads
                
                
def PlotAllConvergenceData(ConvergenceType,ProblemType,PlotOnlySurfaceElevationConvergenceData=True,
                           PlotAgainstNumberOfCellsInZonalDirection=True,PlotAgainstNumberOfTimeSteps=True,
                           UseBestFitLine=False,set_xticks_manually=False):
    ProblemType_Title, ProblemType_FileName = Initialization.SpecifyTitleAndFileNamePrefixes(ProblemType)
    OutputDirectory = '../../output/MPAS_Ocean_Shallow_Water_Output/' + ProblemType
    PlotConvergenceData_LogicalArray = np.ones(3,dtype=bool)
    if PlotOnlySurfaceElevationConvergenceData:
        PlotConvergenceData_LogicalArray[0:2] = False
    # PlotConvergenceData_LogicalArray[0] = PlotZonalVelocityConvergenceData
    # PlotConvergenceData_LogicalArray[1] = PlotMeridionalVelocityConvergenceData
    # PlotConvergenceData_LogicalArray[2] = PlotSurfaceElevationConvergenceData
    [TimeIntegrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
     Generalized_FB_with_AB3_AM4_Step_Types] = SetOfTimeIntegrators()
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
    linestyles = SpecifyLineStyles(ConvergenceType,ProblemType)
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
        xLabel = 'Number of cells'
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
        legends, legendfontsize, legendpads = SetOfLegends(mZonalVelocity)
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
                                                 UseDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=True,
                                                 drawMinorGrid=True,legendWithinBox=False,legendpads=legendpads,
                                                 shadow=shadow,framealpha=framealpha,titlepad=1.035,
                                                 set_xticks_manually=set_xticks_manually,
                                                 xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        else:
            CR.PythonPlots1DSaveAsPDF(OutputDirectory,'log-log',dx,ZonalVelocityL2ErrorNorm,linewidths,linestyles,
                                      colors,markers,markertypes,markersizes,labels,labelfontsizes,labelpads,
                                      tickfontsizes,legends,legendfontsize,legendposition,Title,titlefontsize,SaveAsPDF,
                                      FileName,Show,fig_size=[9.25,9.25],UseDefaultMethodToSpecifyTickFontSize=True,
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
        legends, legendfontsize, legendpads = SetOfLegends(mMeridionalVelocity)
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
                                                 fig_size=[9.25,9.25],UseDefaultMethodToSpecifyTickFontSize=True,
                                                 drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=False,
                                                 legendpads=legendpads,shadow=shadow,framealpha=framealpha,
                                                 titlepad=1.035,set_xticks_manually=set_xticks_manually,
                                                 xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        else:
            CR.PythonPlots1DSaveAsPDF(OutputDirectory,'log-log',dx,MeridionalVelocityL2ErrorNorm,linewidths,linestyles,
                                      colors,markers,markertypes,markersizes,labels,labelfontsizes,labelpads,
                                      tickfontsizes,legends,legendfontsize,legendposition,Title,titlefontsize,SaveAsPDF,
                                      FileName,Show,fig_size=[9.25,9.25],UseDefaultMethodToSpecifyTickFontSize=True,
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
        legends, legendfontsize, legendpads = SetOfLegends(mSurfaceElevation)
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
                                                 UseDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=True,
                                                 drawMinorGrid=True,legendWithinBox=False,legendpads=legendpads,
                                                 shadow=shadow,framealpha=framealpha,titlepad=1.035,
                                                 set_xticks_manually=set_xticks_manually,
                                                 xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        else:
            CR.PythonPlots1DSaveAsPDF(OutputDirectory,'log-log',dx,SurfaceElevationL2ErrorNorm,linewidths,linestyles,
                                      colors,markers,markertypes,markersizes,labels,labelfontsizes,labelpads,
                                      tickfontsizes,legends,legendfontsize,legendposition,Title,titlefontsize,SaveAsPDF,
                                      FileName,Show,fig_size=[9.25,9.25],UseDefaultMethodToSpecifyTickFontSize=True,
                                      drawMajorGrid=True,drawMinorGrid=True,setXAxisLimits=[False,False],
                                      xAxisLimits=[0.0,0.0],setYAxisLimits=[False,False],yAxisLimits=[0.0,0.0],
                                      legendWithinBox=True,legendpads=legendpads,shadow=shadow,framealpha=framealpha,
                                      titlepad=1.035,set_xticks_manually=set_xticks_manually,
                                      xticks_set_manually=xticks_set_manually,FileFormat='pdf')