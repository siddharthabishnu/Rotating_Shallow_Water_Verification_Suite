"""
Name: DGSEM2DClass.py
Author: Sid Bishnu
Details: This script defines the two-dimensional discontinuous Galerkin spectral element class.
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR
    import LagrangeInterpolation2DClass as LI2D
    import DGNodalStorage2DClass as DGNS2D
    import QuadMeshClass as QM
    import DGSolution2DClass as DGS2D
    import ExactSolutionsAndSourceTerms as ESST
    import Initialization
    

class DGSEM2DParameters:
    
    def __init__(myDGSEM2DParameters,ProblemType,ProblemType_NoExactSolution,Problem_is_Linear,BoundaryCondition,
                 NonTrivialSourceTerms,nEquations,nElementsX,nElementsY,nXi,nEta,nXiPlot,nEtaPlot):
        myDGSEM2DParameters.ProblemType = ProblemType
        myDGSEM2DParameters.ProblemType_NoExactSolution = ProblemType_NoExactSolution
        myDGSEM2DParameters.Problem_is_Linear = Problem_is_Linear
        myDGSEM2DParameters.BoundaryCondition = BoundaryCondition
        myDGSEM2DParameters.NonTrivialSourceTerms = NonTrivialSourceTerms
        myDGSEM2DParameters.nEquations = nEquations
        myDGSEM2DParameters.nElementsX = nElementsX
        myDGSEM2DParameters.nElementsY = nElementsY
        myDGSEM2DParameters.nElements = nElementsX*nElementsY
        myDGSEM2DParameters.nXi = nXi
        myDGSEM2DParameters.nEta = nEta
        myDGSEM2DParameters.nXiPlot = nXiPlot
        myDGSEM2DParameters.nEtaPlot = nEtaPlot
        

class DGSEM2D:
    
    def __init__(myDGSEM2D,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                 LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                 Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,nEtaPlot,
                 CourantNumber=0.5,UseCourantNumberToDetermineTimeStep=False,ReadFromSELFOutputData=False,
                 BoundaryConditionAndDomainExtentsSpecified=False,BoundaryCondition='Periodic',lX=0.0,lY=0.0,
                 SpecifyRiemannSolver=False,RiemannSolver='LocalLaxFriedrichs'):
        myDGSEM2D.myNameList = (
        Initialization.NameList(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,CourantNumber,
                                UseCourantNumberToDetermineTimeStep,ReadFromSELFOutputData))
        if BoundaryConditionAndDomainExtentsSpecified:
            myDGSEM2D.myNameList.ModifyNameList(
            PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,nXi,nEta,CourantNumber,
            UseCourantNumberToDetermineTimeStep,BoundaryCondition,lX,lY)
        myDGSEM2D.myDGSEM2DParameters = (
        DGSEM2DParameters(ProblemType,myDGSEM2D.myNameList.ProblemType_NoExactSolution,
                          myDGSEM2D.myNameList.Problem_is_Linear,myDGSEM2D.myNameList.BoundaryCondition,
                          myDGSEM2D.myNameList.NonTrivialSourceTerms,myDGSEM2D.myNameList.nEquations,nElementsX,
                          nElementsY,nXi,nEta,nXiPlot,nEtaPlot))
        myDGSEM2D.myDGNodalStorage2D = DGNS2D.DGNodalStorage2D(nXi,nEta)
        myDGSEM2D.myQuadMesh = QM.QuadMesh(myDGSEM2D.myNameList.lX,myDGSEM2D.myNameList.lY,nElementsX,nElementsY,
                                           myDGSEM2D.myDGNodalStorage2D,ProblemType,
                                           myDGSEM2D.myNameList.ProblemType_EquatorialWave)
        nElements = nElementsX*nElementsY
        myDGSEM2D.myDGSolution2D = np.empty(nElements,dtype=DGS2D.DGSolution2D)
        for iElement in range(0,nElements):
            myDGSEM2D.myDGSolution2D[iElement] = DGS2D.DGSolution2D(myDGSEM2D.myNameList.nEquations,nXi,nEta,
                                                                    TimeIntegrator)
        myDGSEM2D.DetermineCoriolisParameterAndBottomDepth()
        xiPlot = np.linspace(-1.0,1.0,nXiPlot+1)
        etaPlot = np.linspace(-1.0,1.0,nEtaPlot+1)
        myDGSEM2D.myLagrangeInterpolation2DPlot = LI2D.LagrangeInterpolation2D(xiPlot,etaPlot)
        myDGSEM2D.myInterpolationMatrixXPlot, myDGSEM2D.myInterpolationMatrixYPlot = (
        myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.EvaluateLagrangePolynomialInterpolationMatrix2D(
        myDGSEM2D.myLagrangeInterpolation2DPlot))
        myDGSEM2D.RootOutputDirectory, myDGSEM2D.OutputDirectory = MakeOutputDirectories(ProblemType,
                                                                                         ReadFromSELFOutputData)
        myDGSEM2D.iTime = 0
        myDGSEM2D.time = 0.0
        myDGSEM2D.SpecifyRiemannSolver = SpecifyRiemannSolver
        myDGSEM2D.RiemannSolver = RiemannSolver
            
    def DetermineCoriolisParameterAndBottomDepth(myDGSEM2D):
        alpha0 = myDGSEM2D.myNameList.myExactSolutionParameters.alpha0
        beta0 = myDGSEM2D.myNameList.myExactSolutionParameters.beta0
        f0 = myDGSEM2D.myNameList.myExactSolutionParameters.f0
        g = myDGSEM2D.myNameList.myExactSolutionParameters.g
        H0 = myDGSEM2D.myNameList.myExactSolutionParameters.H0
        nElements = myDGSEM2D.myQuadMesh.nElements
        nXi = myDGSEM2D.myDGNodalStorage2D.nXi
        nEta = myDGSEM2D.myDGNodalStorage2D.nEta
        for iElement in range(0,nElements):
            for iXi in range(0,nXi+1):
                for iEta in range(0,nEta+1):
                    y = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iXi,iEta]
                    if (myDGSEM2D.myNameList.ProblemType == 'Planetary_Rossby_Wave' 
                        or myDGSEM2D.myNameList.ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave'
                        or myDGSEM2D.myNameList.ProblemType_EquatorialWave):
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.f[iXi,iEta] = f0 + beta0*y
                    else:
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.f[iXi,iEta] = f0
                    if (myDGSEM2D.myNameList.ProblemType == 'Topographic_Rossby_Wave'
                        or (myDGSEM2D.myNameList.ProblemType 
                            == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave')):
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta] = H0 + alpha0*y
                    else:
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta] = H0
                    myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.c[iXi,iEta] = (
                    np.sqrt(g*myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta]))
            for iXi in range(0,max(nXi+1,nEta+1)):
                for iSide in range(0,4):
                    yBoundary = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.yBoundary[iXi,iSide]
                    if (myDGSEM2D.myNameList.ProblemType == 'Topographic_Rossby_Wave'
                        or (myDGSEM2D.myNameList.ProblemType 
                            == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave')):
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.HBoundary[iXi,iSide] = (
                        H0 + alpha0*yBoundary)
                    else:
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.HBoundary[iXi,iSide] = H0
                    myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.cBoundary[iXi,iSide] = (
                    np.sqrt(g*myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.HBoundary[iXi,iSide]))


def MakeOutputDirectories(ProblemType,ReadFromSELFOutputData):
    cwd = os.getcwd()
    RootOutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output'
    RootOutputDirectoryPath = cwd + '/' + RootOutputDirectory + '/'
    if not os.path.exists(RootOutputDirectoryPath):
        os.mkdir(RootOutputDirectoryPath) # os.makedir(RootOutputDirectoryPath)      
    os.chdir(RootOutputDirectoryPath)
    OutputDirectory = RootOutputDirectory + '/' + ProblemType
    if ReadFromSELFOutputData:
        OutputDirectory += '_SELFOutputData'
    OutputDirectoryPath = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(OutputDirectoryPath):
        os.mkdir(OutputDirectoryPath) # os.makedir(OutputDirectoryPath)   
    os.chdir(cwd)
    return RootOutputDirectory, OutputDirectory


def DetermineExactSolutionAtInteriorNodes(myDGSEM2D):
    ProblemType = myDGSEM2D.myDGSEM2DParameters.ProblemType
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    time = myDGSEM2D.time
    myExactSolutionParameters = myDGSEM2D.myNameList.myExactSolutionParameters
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                x = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iXi,iEta]
                y = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iXi,iEta]
                myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[0,iXi,iEta] = (
                ESST.DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,x,y,time))
                myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[1,iXi,iEta] = (
                ESST.DetermineExactMeridionalVelocity(ProblemType,myExactSolutionParameters,x,y,time))                
                myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[2,iXi,iEta] = (
                ESST.DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,x,y,time))
                
                
def SpecifyInitialConditions(myDGSEM2D):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                H = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta]
                if myDGSEM2D.myNameList.Problem_is_Linear:
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[:,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[:,iXi,iEta])
                else:
                    u = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[0,iXi,iEta]
                    v = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[1,iXi,iEta]
                    eta = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[2,iXi,iEta]
                    h = H + eta
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,iXi,iEta] = h*u
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,iXi,iEta] = h*v
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,iXi,iEta] = h
                
                
def SpecifyRestartConditions(myDGSEM2D,u,v,eta):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                H = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta]
                if myDGSEM2D.myNameList.Problem_is_Linear:
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,iXi,iEta] = u[iElement,iXi,iEta]
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,iXi,iEta] = v[iElement,iXi,iEta]
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,iXi,iEta] = eta[iElement,iXi,iEta]
                else:
                    h = H + eta[iElement,iXi,iEta]
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,iXi,iEta] = h*u[iElement,iXi,iEta]
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,iXi,iEta] = h*v[iElement,iXi,iEta]
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,iXi,iEta] = h
                

def ExtractPrognosticVariablesFromVelocityAndSurfaceElevation(myDGSEM2D,u,v,eta): 
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                H = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta]
                h = H + eta[iElement,iXi,iEta]
                myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,iXi,iEta] = h*u[iElement,iXi,iEta]
                myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,iXi,iEta] = h*v[iElement,iXi,iEta]
                myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,iXi,iEta] = h
    return u, v, eta


def ExtractVelocitiesAndSurfaceElevationFromPrognosticVariables(myDGSEM2D): 
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    u = np.zeros((nElements,nXi+1,nEta+1))
    v = np.zeros((nElements,nXi+1,nEta+1))
    eta = np.zeros((nElements,nXi+1,nEta+1))
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                H = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta]
                hu = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,iXi,iEta]
                hv = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,iXi,iEta]
                h = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,iXi,iEta]
                u[iElement,iXi,iEta] = hu/h
                v[iElement,iXi,iEta] = hv/h
                eta[iElement,iXi,iEta] = h - H
    return u, v, eta


def ExpressStateAtInteriorNodesAsArrays(myDGSEM2D,State,ReturnStateAsOneMultiDimensionalArray=False):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    ZonalVelocityState = np.zeros((nElements,nXi+1,nEta+1))
    MeridionalVelocityState = np.zeros((nElements,nXi+1,nEta+1))
    SurfaceElevationState = np.zeros((nElements,nXi+1,nEta+1))
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                if State == 'Exact':
                    ZonalVelocityState[iElement,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[0,iXi,iEta])
                    MeridionalVelocityState[iElement,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[1,iXi,iEta])
                    SurfaceElevationState[iElement,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[2,iXi,iEta])                   
                elif State == 'Numerical':
                    ZonalVelocityState[iElement,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,iXi,iEta])
                    MeridionalVelocityState[iElement,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,iXi,iEta])
                    SurfaceElevationState[iElement,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,iXi,iEta])       
                elif State == 'Error':
                    ZonalVelocityState[iElement,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[0,iXi,iEta])
                    MeridionalVelocityState[iElement,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[1,iXi,iEta])
                    SurfaceElevationState[iElement,iXi,iEta] = (
                    myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[2,iXi,iEta])
    if ReturnStateAsOneMultiDimensionalArray:
        StateAsOneMultiDimensionalArray = np.zeros((nElements,nXi+1,nEta+1,3))
        StateAsOneMultiDimensionalArray[:,:,:,0] = ZonalVelocityState[:,:,:]
        StateAsOneMultiDimensionalArray[:,:,:,1] = MeridionalVelocityState[:,:,:]
        StateAsOneMultiDimensionalArray[:,:,:,2] = SurfaceElevationState[:,:,:]
        return StateAsOneMultiDimensionalArray
    else:
        return ZonalVelocityState, MeridionalVelocityState, SurfaceElevationState


def ComputeError(myDGSEM2D):
    Problem_is_Linear = myDGSEM2D.myDGSEM2DParameters.Problem_is_Linear
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    if not(Problem_is_Linear):
        u, v, eta = ExtractVelocitiesAndSurfaceElevationFromPrognosticVariables(myDGSEM2D)
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                if Problem_is_Linear:
                    myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[:,iXi,iEta] = (
                    (myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[:,iXi,iEta] 
                     - myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[:,iXi,iEta]))
                else:
                    myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[0,iXi,iEta] = (
                    u[iElement,iXi,iEta] - myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[0,iXi,iEta])
                    myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[1,iXi,iEta] = (
                    v[iElement,iXi,iEta] - myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[1,iXi,iEta])
                    myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[2,iXi,iEta] = (
                    (eta[iElement,iXi,iEta] 
                     - myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[2,iXi,iEta]))
    
    
def ComputeErrorNorm(myDGSEM2D,Error_to_be_Specified=False,SpecifiedError=[]):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    wXi = myDGSEM2D.myDGNodalStorage2D.myGaussQuadratureWeightX
    wEta = myDGSEM2D.myDGNodalStorage2D.myGaussQuadratureWeightY
    L2ErrorNorm = np.zeros(3)
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                if Error_to_be_Specified:
                    L2ErrorNorm[:] += SpecifiedError[iElement,iXi,iEta,:]**2.0*wXi[iXi]*wEta[iEta]
                else:
                    L2ErrorNorm[:] += (
                    (myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[:,iXi,iEta])**2.0*wXi[iXi]*wEta[iEta])
    L2ErrorNorm = 0.5*np.sqrt(L2ErrorNorm/float(nElements))
    return L2ErrorNorm


def ComputeErrorNormOnCoarsestMesh(myDGNodalStorage2DOnCoarsestMesh,myCoarsestMesh,myDGSolution2DOnCoarsestMesh):
    nElements = myCoarsestMesh.nElements
    nXi = myDGNodalStorage2DOnCoarsestMesh.nXi
    nEta = myDGNodalStorage2DOnCoarsestMesh.nEta
    wXi = myDGNodalStorage2DOnCoarsestMesh.myGaussQuadratureWeightX
    wEta = myDGNodalStorage2DOnCoarsestMesh.myGaussQuadratureWeightY
    L2ErrorNorm = np.zeros(3)
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                L2ErrorNorm[:] += (myDGSolution2DOnCoarsestMesh[iElement,iXi,iEta,:])**2.0*wXi[iXi]*wEta[iEta]
    L2ErrorNorm = 0.5*np.sqrt(L2ErrorNorm/float(nElements))
    return L2ErrorNorm
    

def InterpolateSolutionToBoundaries(myDGSEM2DParameters,myDGNodalStorage2D,myDGSolution2D):
    nEquations = myDGSEM2DParameters.nEquations
    nXi = myDGSEM2DParameters.nXi
    nEta = myDGSEM2DParameters.nEta
    for iEquation in range(0,nEquations):
        # Interpolate the solution to the south and north boundaries.
        for iXi in range(0,nXi+1):
            myDGSolution2D.SolutionAtBoundaries[iEquation,iXi,0] = (
            np.dot(myDGSolution2D.SolutionAtInteriorNodes[iEquation,iXi,:],
                   myDGNodalStorage2D.LagrangePolynomialsAlongSouthBoundary[:]))
            myDGSolution2D.SolutionAtBoundaries[iEquation,iXi,2] = (
            np.dot(myDGSolution2D.SolutionAtInteriorNodes[iEquation,iXi,:],
                   myDGNodalStorage2D.LagrangePolynomialsAlongNorthBoundary[:]))
        # Interpolate the solution to the east and west boundaries.
        for iEta in range(0,nEta+1):
            myDGSolution2D.SolutionAtBoundaries[iEquation,iEta,1] = (
            np.dot(myDGSolution2D.SolutionAtInteriorNodes[iEquation,:,iEta],
                   myDGNodalStorage2D.LagrangePolynomialsAlongEastBoundary[:]))
            myDGSolution2D.SolutionAtBoundaries[iEquation,iEta,3] = (
            np.dot(myDGSolution2D.SolutionAtInteriorNodes[iEquation,:,iEta],
                   myDGNodalStorage2D.LagrangePolynomialsAlongWestBoundary[:]))  


def ComputeFlux(g,H,State,Problem_is_Linear,PrognosticVariables='VelocitiesAndSurfaceElevation'):
    FluxX = np.zeros(3)
    FluxY = np.zeros(3)
    if Problem_is_Linear:
        u = State[0]
        v = State[1]
        eta = State[2]
        if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'Velocities':
            FluxX[0] = g*eta
            FluxX[1] = 0.0
            FluxY[0] = 0.0
            FluxY[1] = g*eta
        if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'SurfaceElevation':
            FluxX[2] = H*u
            FluxY[2] = H*v   
    else:
        h = State[2]
        u = State[0]/h 
        v = State[1]/h
        if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'Velocities':
            FluxX[0] = h*u**2.0 + 0.5*g*h**2.0
            FluxX[1] = h*u*v            
            FluxY[0] = h*u*v
            FluxY[1] = h*v**2.0 + 0.5*g*h**2.0            
        if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'SurfaceElevation':
            FluxX[2] = h*u
            FluxY[2] = h*v     
    return FluxX, FluxY


def ComputeFluxNormalToEdge(g,H,nHatX,nHatY,State,Problem_is_Linear,
                            PrognosticVariables='VelocitiesAndSurfaceElevation'):
    FluxX, FluxY = ComputeFlux(g,H,State,Problem_is_Linear,PrognosticVariables)
    NormalFlux = nHatX*FluxX + nHatY*FluxY
    return NormalFlux
    

def ComputeEigenvaluesNormalToEdge(g,c,nHatX,nHatY,State,Problem_is_Linear):
    Eigenvalues = np.zeros(3)
    if Problem_is_Linear:
        Eigenvalues[1] = c
        Eigenvalues[2] = -c
    else:
        h = State[2]
        u = State[0]/h 
        v = State[1]/h
        c = np.sqrt(g*h)
        VelocityNormalToEdge = nHatX*u + nHatY*v
        Eigenvalues[0] = VelocityNormalToEdge
        Eigenvalues[1] = VelocityNormalToEdge + c
        Eigenvalues[2] = VelocityNormalToEdge - c
    return Eigenvalues        


def DetermineCharacteristicVariables(g,c,nHatX,nHatY,q):
    u = q[0]
    v = q[1]
    eta = q[2]
    w = np.zeros(3)
    w[0] = nHatY*u - nHatX*v
    w[1] = c/g*(nHatX*u + nHatY*v) + eta
    w[2] = -c/g*(nHatX*u + nHatY*v) + eta
    return w   


def ExactRiemannSolver(g,c,nHatX,nHatY,qInternalState,qExternalState,
                       PrognosticVariables='VelocitiesAndSurfaceElevation'):
    wInternalState = DetermineCharacteristicVariables(g,c,nHatX,nHatY,qInternalState)
    wExternalState = DetermineCharacteristicVariables(g,c,nHatX,nHatY,qExternalState)
    NumericalFlux = np.zeros(3)
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'Velocities':
        NumericalFlux[0] = 0.5*nHatX*g*(wInternalState[1] + wExternalState[2])
        NumericalFlux[1] = 0.5*nHatY*g*(wInternalState[1] + wExternalState[2])
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'SurfaceElevation':
        NumericalFlux[2] = 0.5*c*(wInternalState[1] - wExternalState[2])
    return NumericalFlux


def BassiRebayRiemannSolver(g,H,nHatX,nHatY,InternalState,ExternalState,Problem_is_Linear,
                            PrognosticVariables='VelocitiesAndSurfaceElevation'):
    InternalFluxNormalToEdge = ComputeFluxNormalToEdge(g,H,nHatX,nHatY,InternalState,Problem_is_Linear,
                                                       PrognosticVariables)
    ExternalFluxNormalToEdge = ComputeFluxNormalToEdge(g,H,nHatX,nHatY,ExternalState,Problem_is_Linear,
                                                       PrognosticVariables)
    NumericalFlux = np.zeros(3)
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
        iEquation_Start = 0
        iEquation_End = 3
    elif PrognosticVariables == 'Velocities':
        iEquation_Start = 0
        iEquation_End = 2
    elif PrognosticVariables == 'SurfaceElevation':
        iEquation_Start = 2
        iEquation_End = 3
    for iEquation in range(iEquation_Start,iEquation_End):
        NumericalFlux[iEquation] = 0.5*(InternalFluxNormalToEdge[iEquation] + ExternalFluxNormalToEdge[iEquation])
    return NumericalFlux


def LocalLaxFriedrichsRiemannSolver(g,H,c,nHatX,nHatY,InternalState,ExternalState,Problem_is_Linear,
                                    PrognosticVariables='VelocitiesAndSurfaceElevation'):
    InternalFluxNormalToEdge = ComputeFluxNormalToEdge(g,H,nHatX,nHatY,InternalState,Problem_is_Linear,
                                                       PrognosticVariables)
    ExternalFluxNormalToEdge = ComputeFluxNormalToEdge(g,H,nHatX,nHatY,ExternalState,Problem_is_Linear,
                                                       PrognosticVariables)
    InternalEigenvaluesNormalToEdge = ComputeEigenvaluesNormalToEdge(g,c,nHatX,nHatY,InternalState,Problem_is_Linear)
    ExternalEigenvaluesNormalToEdge = ComputeEigenvaluesNormalToEdge(g,c,nHatX,nHatY,ExternalState,Problem_is_Linear)
    NumericalFlux = np.zeros(3)
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
        iEquation_Start = 0
        iEquation_End = 3
    elif PrognosticVariables == 'Velocities':
        iEquation_Start = 0
        iEquation_End = 2
    elif PrognosticVariables == 'SurfaceElevation':
        iEquation_Start = 2
        iEquation_End = 3
    for iEquation in range(iEquation_Start,iEquation_End):
        NumericalFlux[iEquation] = 0.5*(InternalFluxNormalToEdge[iEquation] + ExternalFluxNormalToEdge[iEquation]
                                        - (max(np.max(abs(InternalEigenvaluesNormalToEdge)),
                                               np.max(abs(ExternalEigenvaluesNormalToEdge)))
                                           *(ExternalState[iEquation] - InternalState[iEquation])))
    return NumericalFlux


def ComputeNumericalFlux(myEdge,myExactSolutionParameters,myQuadMeshParameters,myDGSEM2DParameters,myQuadElements,
                         myDGSolution2D,time,dt,PrognosticVariables='VelocitiesAndSurfaceElevation',
                         ComputeExternalSurfaceElevationOneTimeStepEarlier=False,SpecifyRiemannSolver=False,
                         RiemannSolver='LocalLaxFriedrichs'):
    ProblemType = myDGSEM2DParameters.ProblemType
    Problem_is_Linear = myDGSEM2DParameters.Problem_is_Linear
    BoundaryCondition = myDGSEM2DParameters.BoundaryCondition
    nElementsX = myDGSEM2DParameters.nElementsX
    nElementsY = myDGSEM2DParameters.nElementsY
    nXi = myDGSEM2DParameters.nXi
    nEta = myDGSEM2DParameters.nEta
    nXi = max(nXi,nEta)
    g = myExactSolutionParameters.g
    ElementID1 = myEdge.ElementIDs[0]
    ElementSide1 = myEdge.ElementSides[0]
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
        iEquation_Start = 0
        iEquation_End = 3
    elif PrognosticVariables == 'Velocities':
        iEquation_Start = 0
        iEquation_End = 2
    elif PrognosticVariables == 'SurfaceElevation':
        iEquation_Start = 2
        iEquation_End = 3
    time_ZonalVelocity = time
    time_MeridionalVelocity = time
    if ComputeExternalSurfaceElevationOneTimeStepEarlier:
        time_SurfaceElevation = time - dt
    else:
        time_SurfaceElevation = time
    if myEdge.EdgeType == myQuadMeshParameters.INTERIOR:
        ElementID2 = myEdge.ElementIDs[1]
        ElementSide2 = abs(myEdge.ElementSides[1])
        kXi = myEdge.start - myEdge.increment
        for jXi in range(0,nXi+1):
            H = myQuadElements[ElementID1-1].myMappedGeometry2D.HBoundary[jXi,ElementSide1-1]
            c = myQuadElements[ElementID1-1].myMappedGeometry2D.cBoundary[jXi,ElementSide1-1]
            nHat = myQuadElements[ElementID1-1].myMappedGeometry2D.nHat[jXi,ElementSide1-1]
            nHatX = nHat.Components[0]
            nHatY = nHat.Components[1]
            InternalState = myDGSolution2D[ElementID1-1].SolutionAtBoundaries[:,jXi,ElementSide1-1]
            ExternalState = myDGSolution2D[ElementID2-1].SolutionAtBoundaries[:,kXi,ElementSide2-1]
            if Problem_is_Linear and not(SpecifyRiemannSolver):
                NumericalFlux = ExactRiemannSolver(g,c,nHatX,nHatY,InternalState,ExternalState,PrognosticVariables)
            else:
                if RiemannSolver == 'LocalLaxFriedrichs':
                    NumericalFlux = LocalLaxFriedrichsRiemannSolver(g,H,c,nHatX,nHatY,InternalState,ExternalState,
                                                                    Problem_is_Linear,PrognosticVariables)
                else: # if RiemannSolver == 'BassiRebay':
                    NumericalFlux = BassiRebayRiemannSolver(g,H,nHatX,nHatY,InternalState,ExternalState,
                                                            Problem_is_Linear,PrognosticVariables)
            for iEquation in range(iEquation_Start,iEquation_End):
                myDGSolution2D[ElementID1-1].FluxAtBoundaries[iEquation,jXi,ElementSide1-1] = (
                (NumericalFlux[iEquation]
                 *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))
                myDGSolution2D[ElementID2-1].FluxAtBoundaries[iEquation,kXi,ElementSide2-1] = (
                (-NumericalFlux[iEquation]
                 *myQuadElements[ElementID2-1].myMappedGeometry2D.ScalingFactors[kXi,ElementSide2-1]))
            kXi += myEdge.increment
    else:
        ExternalState = np.zeros((3,nXi+1))
        if BoundaryCondition == 'Periodic':
            if myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_SOUTH:
                ElementID2 = ElementID1 + nElementsX*(nElementsY - 1)
                ElementSide2 = 3
            elif myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_EAST:
                ElementID2 = ElementID1 - (nElementsX - 1)
                ElementSide2 = 4
            elif myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_NORTH:
                ElementID2 = ElementID1 - nElementsX*(nElementsY - 1)
                ElementSide2 = 1
            elif myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_WEST:
                ElementID2 = ElementID1 + (nElementsX - 1)
                ElementSide2 = 2
            for jXi in range(0,nXi+1):  
                ExternalState[:,jXi] = myDGSolution2D[ElementID2-1].SolutionAtBoundaries[:,jXi,ElementSide2-1]
        elif BoundaryCondition == 'NonPeriodic_x':
            if (myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_SOUTH 
                or myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_NORTH):
                if myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_SOUTH:
                    ElementID2 = ElementID1 + nElementsX*(nElementsY - 1)
                    ElementSide2 = 3
                else: # if myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_NORTH:
                    ElementID2 = ElementID1 - nElementsX*(nElementsY - 1)
                    ElementSide2 = 1
                for jXi in range(0,nXi+1):  
                    ExternalState[:,jXi] = myDGSolution2D[ElementID2-1].SolutionAtBoundaries[:,jXi,ElementSide2-1]
            else: # if (myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_EAST 
                  #     or myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_WEST):
                ElementID2 = ElementID1
                ElementSide2 = ElementSide1
                for jXi in range(0,nXi+1):          
                    x = myQuadElements[ElementID2-1].myMappedGeometry2D.xBoundary[jXi,ElementSide2-1]
                    y = myQuadElements[ElementID2-1].myMappedGeometry2D.yBoundary[jXi,ElementSide2-1]
                    ExternalState[0,jXi] = ESST.DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,x,y,
                                                                            time_ZonalVelocity)
                    ExternalState[1,jXi] = ESST.DetermineExactMeridionalVelocity(ProblemType,myExactSolutionParameters,
                                                                                 x,y,time_MeridionalVelocity)
                    ExternalState[2,jXi] = ESST.DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,x,
                                                                               y,time_SurfaceElevation)
        elif BoundaryCondition == 'NonPeriodic_y':
            if (myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_EAST 
                or myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_WEST):
                if myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_EAST:
                    ElementID2 = ElementID1 - (nElementsX - 1)
                    ElementSide2 = 4
                else: # if myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_WEST:
                    ElementID2 = ElementID1 + (nElementsX - 1)
                    ElementSide2 = 2
                for jXi in range(0,nXi+1):  
                    ExternalState[:,jXi] = myDGSolution2D[ElementID2-1].SolutionAtBoundaries[:,jXi,ElementSide2-1]
            else: # if (myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_SOUTH 
                  #     or myEdge.EdgeType == myQuadMeshParameters.DIRICHLET_NORTH):
                ElementID2 = ElementID1
                ElementSide2 = ElementSide1
                for jXi in range(0,nXi+1):          
                    x = myQuadElements[ElementID2-1].myMappedGeometry2D.xBoundary[jXi,ElementSide2-1]
                    y = myQuadElements[ElementID2-1].myMappedGeometry2D.yBoundary[jXi,ElementSide2-1]
                    ExternalState[0,jXi] = ESST.DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,x,y,
                                                                            time_ZonalVelocity)
                    ExternalState[1,jXi] = ESST.DetermineExactMeridionalVelocity(ProblemType,myExactSolutionParameters,
                                                                                 x,y,time_MeridionalVelocity)
                    ExternalState[2,jXi] = ESST.DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,x,
                                                                               y,time_SurfaceElevation)
        elif BoundaryCondition == 'NonPeriodic_xy':
            ElementID2 = ElementID1
            ElementSide2 = ElementSide1
            for jXi in range(0,nXi+1):
                x = myQuadElements[ElementID2-1].myMappedGeometry2D.xBoundary[jXi,ElementSide2-1]
                y = myQuadElements[ElementID2-1].myMappedGeometry2D.yBoundary[jXi,ElementSide2-1]
                ExternalState[0,jXi] = ESST.DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,x,y,
                                                                        time_ZonalVelocity)
                ExternalState[1,jXi] = ESST.DetermineExactMeridionalVelocity(ProblemType,myExactSolutionParameters,x,y,
                                                                             time_MeridionalVelocity)
                ExternalState[2,jXi] = ESST.DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,x,y,
                                                                           time_SurfaceElevation)
        for jXi in range(0,nXi+1):
            H = myQuadElements[ElementID1-1].myMappedGeometry2D.HBoundary[jXi,ElementSide1-1]
            c = myQuadElements[ElementID1-1].myMappedGeometry2D.cBoundary[jXi,ElementSide1-1]
            nHat = myQuadElements[ElementID1-1].myMappedGeometry2D.nHat[jXi,ElementSide1-1]
            nHatX = nHat.Components[0]
            nHatY = nHat.Components[1]
            InternalState = myDGSolution2D[ElementID1-1].SolutionAtBoundaries[:,jXi,ElementSide1-1]
            if BoundaryCondition == 'Reflection':
                ExternalState[0,jXi] = -2.0*nHatX*nHatY*InternalState[1] - (nHatX**2.0 - nHatY**2.0)*InternalState[0]
                ExternalState[1,jXi] = -2.0*nHatX*nHatY*InternalState[0] + (nHatX**2.0 - nHatY**2.0)*InternalState[1]
                ExternalState[2,jXi] = InternalState[2]
            if Problem_is_Linear and not(SpecifyRiemannSolver):
                NumericalFlux = ExactRiemannSolver(g,c,nHatX,nHatY,InternalState,ExternalState[:,jXi],
                                                   PrognosticVariables)
            else:
                if RiemannSolver == 'LocalLaxFriedrichs':
                    NumericalFlux = LocalLaxFriedrichsRiemannSolver(g,H,c,nHatX,nHatY,InternalState,
                                                                    ExternalState[:,jXi],Problem_is_Linear,
                                                                    PrognosticVariables)
                else: # if RiemannSolver == 'BassiRebay':
                    NumericalFlux = BassiRebayRiemannSolver(g,H,nHatX,nHatY,InternalState,ExternalState[:,jXi],
                                                            Problem_is_Linear,PrognosticVariables)
            for iEquation in range(iEquation_Start,iEquation_End):
                myDGSolution2D[ElementID1-1].FluxAtBoundaries[iEquation,jXi,ElementSide1-1] = (
                (NumericalFlux[iEquation]
                 *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))


def DGSystemDerivative(nEquations,nXi,DGDerivativeMatrix,QuadratureWeights,NumericalFluxAtLeftBoundary,
                       NumericalFluxAtRightBoundary,InteriorFlux,LagrangePolynomialsAtLeftBoundary,
                       LagrangePolynomialsAtRightBoundary,PrognosticVariables='VelocitiesAndSurfaceElevation'):
    FluxDerivative = np.zeros((nEquations,nXi+1))
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
        iEquation_Start = 0
        iEquation_End = 3
    elif PrognosticVariables == 'Velocities':
        iEquation_Start = 0
        iEquation_End = 2
    elif PrognosticVariables == 'SurfaceElevation':
        iEquation_Start = 2
        iEquation_End = 3
    for iEquation in range(iEquation_Start,iEquation_End):
        FluxDerivative[iEquation,:] = np.matmul(DGDerivativeMatrix,InteriorFlux[iEquation,:])
    for iEquation in range(iEquation_Start,iEquation_End):
        for iXi in range(0,nXi+1):
            FluxDerivative[iEquation,iXi] += (
            (NumericalFluxAtRightBoundary[iEquation]*LagrangePolynomialsAtRightBoundary[iXi] 
             + NumericalFluxAtLeftBoundary[iEquation]*LagrangePolynomialsAtLeftBoundary[iXi])/QuadratureWeights[iXi])
    return FluxDerivative


def DGTimeDerivative(myExactSolutionParameters,myDGSEM2DParameters,myDGNodalStorage2D,myMappedGeometry2D,myDGSolution2D,
                     time,PrognosticVariables='VelocitiesAndSurfaceElevation'):
    ProblemType = myDGSEM2DParameters.ProblemType
    Problem_is_Linear = myDGSEM2DParameters.Problem_is_Linear
    NonTrivialSourceTerms = myDGSEM2DParameters.NonTrivialSourceTerms
    nEquations = myDGSEM2DParameters.nEquations
    nXi = myDGSEM2DParameters.nXi
    nEta = myDGSEM2DParameters.nEta
    g = myExactSolutionParameters.g
    dXdXi = myMappedGeometry2D.dXdXi
    dXdEta = myMappedGeometry2D.dXdEta
    dYdXi = myMappedGeometry2D.dYdXi
    dYdEta = myMappedGeometry2D.dYdEta                   
    Jacobian = myMappedGeometry2D.Jacobian
    InteriorFluxX = np.zeros((nEquations,nXi+1,nEta+1))
    InteriorFluxY = np.zeros((nEquations,nXi+1,nEta+1))
    InteriorFluxDerivativeX = np.zeros((nEquations,nXi+1,nEta+1))
    InteriorFluxDerivativeY = np.zeros((nEquations,nXi+1,nEta+1))
    for iXi in range(0,nXi+1):
        for iEta in range(0,nEta+1):
            H = myMappedGeometry2D.H[iXi,iEta]
            SolutionAtInteriorNode = myDGSolution2D.SolutionAtInteriorNodes[:,iXi,iEta]
            FluxX, FluxY = ComputeFlux(g,H,SolutionAtInteriorNode,Problem_is_Linear,PrognosticVariables)
            InteriorFluxX[:,iXi,iEta] = dYdEta[iXi,iEta]*FluxX[:] - dXdEta[iXi,iEta]*FluxY[:]
            InteriorFluxY[:,iXi,iEta] = -dYdXi[iXi,iEta]*FluxX[:] + dXdXi[iXi,iEta]*FluxY[:]
    FluxAtWestBoundary = myDGSolution2D.FluxAtBoundaries[:,:,3]   
    FluxAtEastBoundary = myDGSolution2D.FluxAtBoundaries[:,:,1]
    for iEta in range(0,nEta+1):
        InteriorFluxDerivativeX[:,:,iEta] = (
        DGSystemDerivative(nEquations,nXi,myDGNodalStorage2D.DGDerivativeMatrixX,
                           myDGNodalStorage2D.myGaussQuadratureWeightX,FluxAtWestBoundary[:,iEta],
                           FluxAtEastBoundary[:,iEta],InteriorFluxX[:,:,iEta],
                           myDGNodalStorage2D.LagrangePolynomialsAlongWestBoundary,
                           myDGNodalStorage2D.LagrangePolynomialsAlongEastBoundary,PrognosticVariables))
    FluxAtSouthBoundary = myDGSolution2D.FluxAtBoundaries[:,:,0]
    FluxAtNorthBoundary = myDGSolution2D.FluxAtBoundaries[:,:,2]
    for iXi in range(0,nXi+1):
        InteriorFluxDerivativeY[:,iXi,:] = (
        DGSystemDerivative(nEquations,nEta,myDGNodalStorage2D.DGDerivativeMatrixY,
                           myDGNodalStorage2D.myGaussQuadratureWeightY,FluxAtSouthBoundary[:,iXi],
                           FluxAtNorthBoundary[:,iXi],InteriorFluxY[:,iXi,:],
                           myDGNodalStorage2D.LagrangePolynomialsAlongSouthBoundary,
                           myDGNodalStorage2D.LagrangePolynomialsAlongNorthBoundary,PrognosticVariables))
    for iXi in range(0,nXi+1):
        for iEta in range(0,nEta+1):
            TendencyAtInteriorNode = -(InteriorFluxDerivativeX[:,iXi,iEta] 
                                       + InteriorFluxDerivativeY[:,iXi,iEta])/Jacobian[iXi,iEta]
            f = myMappedGeometry2D.f[iXi,iEta]
            if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'Velocities':
                TendencyAtInteriorNode[0] += f*myDGSolution2D.SolutionAtInteriorNodes[1,iXi,iEta]
                TendencyAtInteriorNode[1] -= f*myDGSolution2D.SolutionAtInteriorNodes[0,iXi,iEta]
            if NonTrivialSourceTerms:
                x = myMappedGeometry2D.x[iXi,iEta]
                y = myMappedGeometry2D.y[iXi,iEta]
                SourceTermAtInteriorNode = np.zeros(3)
                if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'Velocities':
                    SourceTermAtInteriorNode[0] = (
                    ESST.DetermineZonalVelocitySourceTerm(ProblemType,myExactSolutionParameters,x,y,time))
                    SourceTermAtInteriorNode[1] = (
                    ESST.DetermineMeridionalVelocitySourceTerm(ProblemType,myExactSolutionParameters,x,y,time))
                if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'SurfaceElevation':
                    SourceTermAtInteriorNode[2] = (
                    ESST.DetermineSurfaceElevationSourceTerm(ProblemType,myExactSolutionParameters,x,y,time))
                TendencyAtInteriorNode += SourceTermAtInteriorNode
            myDGSolution2D.TendencyAtInteriorNodes[:,iXi,iEta] = TendencyAtInteriorNode
            

def GlobalTimeDerivative(myDGSEM2D,time,PrognosticVariables='VelocitiesAndSurfaceElevation',
                         ComputeExternalSurfaceElevationOneTimeStepEarlier=False):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    dt = myDGSEM2D.myNameList.dt
    for iElement in range(0,nElements):
        InterpolateSolutionToBoundaries(myDGSEM2D.myDGSEM2DParameters,myDGSEM2D.myDGNodalStorage2D,
                                        myDGSEM2D.myDGSolution2D[iElement])
    for iEdge in range(0,myDGSEM2D.myQuadMesh.nEdges):
        ComputeNumericalFlux(myDGSEM2D.myQuadMesh.myEdges[iEdge],myDGSEM2D.myNameList.myExactSolutionParameters,
                             myDGSEM2D.myQuadMesh.myQuadMeshParameters,myDGSEM2D.myDGSEM2DParameters,
                             myDGSEM2D.myQuadMesh.myQuadElements,myDGSEM2D.myDGSolution2D,time,dt,PrognosticVariables,
                             ComputeExternalSurfaceElevationOneTimeStepEarlier,myDGSEM2D.SpecifyRiemannSolver,
                             myDGSEM2D.RiemannSolver)
    for iElement in range(0,nElements):
        DGTimeDerivative(myDGSEM2D.myNameList.myExactSolutionParameters,myDGSEM2D.myDGSEM2DParameters,
                         myDGSEM2D.myDGNodalStorage2D,myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D,
                         myDGSEM2D.myDGSolution2D[iElement],time,PrognosticVariables)


def WriteStateDGSEM2D(myDGSEM2D,filename):
    Problem_is_Linear = myDGSEM2D.myDGSEM2DParameters.Problem_is_Linear
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    cwd = os.getcwd()
    path = cwd + '/' + myDGSEM2D.OutputDirectory + '/'
    os.chdir(path)
    filename += '.tec'
    outputfile = open(filename,'w')
    outputfile.write('VARIABLES = "X", "Y", "Jacobian", "u", "v", "eta"\n')
    if not(Problem_is_Linear):
        u_AllElements, v_AllElements, eta_AllElements = (
        ExtractVelocitiesAndSurfaceElevationFromPrognosticVariables(myDGSEM2D))
    for iElement in range(0,nElements):
        x = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[:,:]
        y = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[:,:]
        Jacobian = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.Jacobian[:,:]
        if Problem_is_Linear:
            u = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,:,:]
            v = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,:,:]
            eta = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,:,:]
        else:
            u = u_AllElements[iElement,:,:]
            v = v_AllElements[iElement,:,:]
            eta = eta_AllElements[iElement,:,:]         
        ZoneID = myDGSEM2D.myQuadMesh.myQuadElements[iElement].ElementID
        ZoneIDString = 'Element' + '%7.7d' %ZoneID
        outputfile.write('ZONE T="%s", I=%d, J=%d, F=POINT\n' %(ZoneIDString,nXi+1,nEta+1))
        for iEta in range(0,nEta+1):
            for iXi in range(0,nXi+1):
                outputfile.write('%.15g %.15g %.15g %.15g %.15g %.15g\n' 
                                 %(x[iXi,iEta],y[iXi,iEta],Jacobian[iXi,iEta],u[iXi,iEta],v[iXi,iEta],eta[iXi,iEta]))
    outputfile.close()
    os.chdir(cwd)  
    
    
def ReadStateDGSEM2D(myDGSEM2D,filename):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    cwd = os.getcwd()
    path = cwd + '/' + myDGSEM2D.OutputDirectory + '/'
    os.chdir(path)
    data = []
    count = 0
    with open(filename,'r') as infile:
        for line in infile:
            if count != 0 and np.mod(count-1,((nXi+1)*(nEta+1)+1)) != 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    u = np.zeros((nElements,nXi+1,nEta+1))
    v = np.zeros((nElements,nXi+1,nEta+1))
    eta = np.zeros((nElements,nXi+1,nEta+1))
    for iElement in range(0,nElements):
        for iEta in range(0,nEta+1):
            for iXi in range(0,nXi+1):
                i = iElement*(nXi + 1)*(nEta + 1) + iEta*(nXi + 1) + iXi
                u[iElement,iXi,iEta] = data[i,3]
                v[iElement,iXi,iEta] = data[i,4]
                eta[iElement,iXi,iEta] = data[i,5]  
    os.chdir(cwd)
    return u, v, eta


def WriteInterpolatedStateDGSEM2D(myDGSEM2D,filename,ComputeOnlyExactSolution=False):
    Problem_is_Linear = myDGSEM2D.myDGSEM2DParameters.Problem_is_Linear
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXiPlot = myDGSEM2D.myDGSEM2DParameters.nXiPlot
    nEtaPlot = myDGSEM2D.myDGSEM2DParameters.nEtaPlot
    cwd = os.getcwd()
    path = cwd + '/' + myDGSEM2D.OutputDirectory + '/'
    os.chdir(path)
    filename += '.tec'
    outputfile = open(filename,'w')
    if ComputeOnlyExactSolution:
        outputfile.write('VARIABLES = "X", "Y", "Jacobian", "uExact", "vExact", "etaExact"\n')       
    else:
        outputfile.write('VARIABLES = "X", "Y", "Jacobian", "uExact", "vExact", "etaExact", "u", "v", "eta", "uError", '
                         + '"vError", "etaError"\n')
    if not(Problem_is_Linear):
        u_AllElements, v_AllElements, eta_AllElements = (
        ExtractVelocitiesAndSurfaceElevationFromPrognosticVariables(myDGSEM2D))
    for iElement in range(0,nElements):
        x = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[:,:]
        y = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[:,:]
        Jacobian = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.Jacobian[:,:]
        uExact = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[0,:,:]
        vExact = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[1,:,:]
        etaExact = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[2,:,:]
        if Problem_is_Linear:
            u = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,:,:]
            v = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,:,:]
            eta = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,:,:]
        else:
            u = u_AllElements[iElement,:,:]
            v = v_AllElements[iElement,:,:]
            eta = eta_AllElements[iElement,:,:]
        uError = myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[0,:,:]
        vError = myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[1,:,:]
        etaError = myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[2,:,:]
        xPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
        myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
        myDGSEM2D.myInterpolationMatrixYPlot,x)
        yPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
        myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
        myDGSEM2D.myInterpolationMatrixYPlot,y)
        JacobianPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
        myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
        myDGSEM2D.myInterpolationMatrixYPlot,Jacobian)
        uExactPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
        myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
        myDGSEM2D.myInterpolationMatrixYPlot,uExact)
        vExactPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
        myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
        myDGSEM2D.myInterpolationMatrixYPlot,vExact)
        etaExactPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
        myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
        myDGSEM2D.myInterpolationMatrixYPlot,etaExact)
        if not(ComputeOnlyExactSolution):
            uPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
            myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
            myDGSEM2D.myInterpolationMatrixYPlot,u)
            vPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
            myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
            myDGSEM2D.myInterpolationMatrixYPlot,v)
            etaPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
            myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
            myDGSEM2D.myInterpolationMatrixYPlot,eta)
            uErrorPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
            myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
            myDGSEM2D.myInterpolationMatrixYPlot,uError)
            vErrorPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
            myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
            myDGSEM2D.myInterpolationMatrixYPlot, vError)
            etaErrorPlot = myDGSEM2D.myDGNodalStorage2D.myLagrangeInterpolation2D.InterpolateToNewPoints2D(
            myDGSEM2D.myLagrangeInterpolation2DPlot,myDGSEM2D.myInterpolationMatrixXPlot,
            myDGSEM2D.myInterpolationMatrixYPlot,etaError)
        ZoneID = myDGSEM2D.myQuadMesh.myQuadElements[iElement].ElementID
        ZoneIDString = 'Element' + '%7.7d' %ZoneID
        outputfile.write('ZONE T="%s", I=%d, J=%d, F=POINT\n' %(ZoneIDString,nXiPlot+1,nEtaPlot+1))
        for iEtaPlot in range(0,nEtaPlot+1):
            for iXiPlot in range(0,nXiPlot+1):
                if ComputeOnlyExactSolution:
                    outputfile.write('%.15g %.15g %.15g %.15g %.15g %.15g\n' 
                                     %(xPlot[iXiPlot,iEtaPlot],yPlot[iXiPlot,iEtaPlot],JacobianPlot[iXiPlot,iEtaPlot],
                                       uExactPlot[iXiPlot,iEtaPlot],vExactPlot[iXiPlot,iEtaPlot],
                                       etaExactPlot[iXiPlot,iEtaPlot]))
                else:
                    outputfile.write('%.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n' 
                                     %(xPlot[iXiPlot,iEtaPlot],yPlot[iXiPlot,iEtaPlot],JacobianPlot[iXiPlot,iEtaPlot],
                                       uExactPlot[iXiPlot,iEtaPlot],vExactPlot[iXiPlot,iEtaPlot],
                                       etaExactPlot[iXiPlot,iEtaPlot],uPlot[iXiPlot,iEtaPlot],vPlot[iXiPlot,iEtaPlot],
                                       etaPlot[iXiPlot,iEtaPlot],uErrorPlot[iXiPlot,iEtaPlot],
                                       vErrorPlot[iXiPlot,iEtaPlot],etaErrorPlot[iXiPlot,iEtaPlot]))
    outputfile.close()
    os.chdir(cwd)
    
    
def PythonPlotStateDGSEM2D(myDGSEM2D,filename,DataType,DisplayTime,UseGivenColorBarLimits=True,
                           ComputeOnlyExactSolution=False,SpecifyDataTypeInPlotFileName=False,
                           PlotNumericalSolution=False):
    ProblemType_NoExactSolution = myDGSEM2D.myDGSEM2DParameters.ProblemType_NoExactSolution
    ProblemType = myDGSEM2D.myDGSEM2DParameters.ProblemType
    nElementsX = myDGSEM2D.myDGSEM2DParameters.nElementsX
    nElementsY = myDGSEM2D.myDGSEM2DParameters.nElementsY
    nXiPlot = myDGSEM2D.myDGSEM2DParameters.nXiPlot
    nEtaPlot = myDGSEM2D.myDGSEM2DParameters.nEtaPlot
    ProblemType_EquatorialWave = myDGSEM2D.myNameList.ProblemType_EquatorialWave
    PlotZonalVelocity = myDGSEM2D.myNameList.LogicalArrayPlot[0]
    PlotMeridionalVelocity = myDGSEM2D.myNameList.LogicalArrayPlot[1]
    PlotSurfaceElevation = myDGSEM2D.myNameList.LogicalArrayPlot[2]
    cwd = os.getcwd()
    path = cwd + '/' + myDGSEM2D.OutputDirectory + '/'
    os.chdir(path)
    data = []
    count = 0
    with open(filename,'r') as infile:
        for line in infile:
            if count != 0 and np.mod(count-1,((nXiPlot+1)*(nEtaPlot+1)+1)) != 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    nPointsX = nElementsX*nXiPlot + 1
    nPointsY = nElementsY*nEtaPlot + 1
    nPoints = nPointsX*nPointsY
    if DataType == 'Structured':
        x = np.zeros(nPointsX)
        y = np.zeros(nPointsY)
        uExact = np.zeros((nPointsY,nPointsX))
        vExact = np.zeros((nPointsY,nPointsX))
        etaExact = np.zeros((nPointsY,nPointsX))
        if not(ComputeOnlyExactSolution):
            if PlotNumericalSolution:
                u = np.zeros((nPointsY,nPointsX))
                v = np.zeros((nPointsY,nPointsX))
                eta = np.zeros((nPointsY,nPointsX))  
            uError = np.zeros((nPointsY,nPointsX))
            vError = np.zeros((nPointsY,nPointsX))
            etaError = np.zeros((nPointsY,nPointsX))
    elif DataType == 'Unstructured':
        x = np.zeros(nPoints)
        y = np.zeros(nPoints)
        uExact = np.zeros(nPoints)
        vExact = np.zeros(nPoints)
        etaExact = np.zeros(nPoints)
        if not(ComputeOnlyExactSolution):
            if PlotNumericalSolution:
                u = np.zeros(nPoints)
                v = np.zeros(nPoints)
                eta = np.zeros(nPoints)
            uError = np.zeros(nPoints)
            vError = np.zeros(nPoints)
            etaError = np.zeros(nPoints)
    iPointY = -1
    for iElementY in range(0,nElementsY):
        if iElementY == nElementsY - 1:
            nEtaPlotLast = nEtaPlot + 1
        else:
            nEtaPlotLast = nEtaPlot
        for iEtaPlot in range(0,nEtaPlotLast):
            iPointY += 1
            iPointX = -1
            for iElementX in range(0,nElementsX):
                iElement = iElementY*nElementsX + iElementX
                if iElementX == nElementsX - 1:
                    nXiPlotLast = nXiPlot + 1
                else:
                    nXiPlotLast = nXiPlot
                for iXiPlot in range(0,nXiPlotLast):
                    iPointX += 1
                    i = iElement*(nXiPlot + 1)*(nEtaPlot + 1) + iEtaPlot*(nXiPlot + 1) + iXiPlot
                    if DataType == 'Structured':
                        if myDGSEM2D.myNameList.ReadFromSELFOutputData:
                            x[iPointX] = data[i,0] + 0.5*myDGSEM2D.myNameList.lX                        
                            y[iPointY] = data[i,1] + 0.5*myDGSEM2D.myNameList.lY 
                        else:
                            x[iPointX] = data[i,0]                        
                            y[iPointY] = data[i,1]
                        if myDGSEM2D.myNameList.ReadFromSELFOutputData:
                            u[iPointY,iPointX] = data[i,2]
                            v[iPointY,iPointX] = data[i,3]
                            eta[iPointY,iPointX] = data[i,4]                            
                        else:
                            uExact[iPointY,iPointX] = data[i,3]
                            vExact[iPointY,iPointX] = data[i,4]
                            etaExact[iPointY,iPointX] = data[i,5]
                            if not(ComputeOnlyExactSolution):     
                                if PlotNumericalSolution:                   
                                    u[iPointY,iPointX] = data[i,6]
                                    v[iPointY,iPointX] = data[i,7]
                                    eta[iPointY,iPointX] = data[i,8]
                                uError[iPointY,iPointX] = data[i,9]
                                vError[iPointY,iPointX] = data[i,10]
                                etaError[iPointY,iPointX] = data[i,11]
                    elif DataType == 'Unstructured':
                        iPoint = iPointY*nPointsX + iPointX
                        if myDGSEM2D.myNameList.ReadFromSELFOutputData:
                            x[iPoint] = data[i,0] + 0.5*myDGSEM2D.myNameList.lX                        
                            y[iPoint] = data[i,1] + 0.5*myDGSEM2D.myNameList.lY 
                        else:
                            x[iPoint] = data[i,0]                        
                            y[iPoint] = data[i,1]
                        if myDGSEM2D.myNameList.ReadFromSELFOutputData:
                            u[iPoint] = data[i,2]
                            v[iPoint] = data[i,3]
                            eta[iPoint] = data[i,4]
                        else:
                            uExact[iPoint] = data[i,3]
                            vExact[iPoint] = data[i,4]
                            etaExact[iPoint] = data[i,5]
                            if not(ComputeOnlyExactSolution):
                                if PlotNumericalSolution:
                                    u[iPoint] = data[i,6]
                                    v[iPoint] = data[i,7]
                                    eta[iPoint] = data[i,8]
                                uError[iPoint] = data[i,9]
                                vError[iPoint] = data[i,10]
                                etaError[iPoint] = data[i,11]
    os.chdir(cwd)
    titleroot = myDGSEM2D.myNameList.ProblemType_Title
    if SpecifyDataTypeInPlotFileName:
        PlotFileNameRoot = myDGSEM2D.myNameList.ProblemType_FileName + '_' + DataType
    else:
        PlotFileNameRoot = myDGSEM2D.myNameList.ProblemType_FileName
    TimeIntegratorShortForm = myDGSEM2D.myNameList.myTimeSteppingParameters.TimeIntegratorShortForm
    if ProblemType == 'Plane_Gaussian_Wave':
        xlabel = 'Zonal Distance (m)'
        ylabel = 'Meridional Distance (m)'
    else:
        x /= 1000.0
        y /= 1000.0
        xlabel = 'Zonal Distance (km)'
        ylabel = 'Meridional Distance (km)'
    nContours = 300
    labels = [xlabel,ylabel]
    labelfontsizes = [22.5,22.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    nColorBarTicks = 6
    titlefontsize = 25.0
    SaveAsPDF = True
    Show = False
    colormap = plt.cm.seismic
    if myDGSEM2D.myNameList.ReadFromSELFOutputData:
        iTimeFormat = '%8.8d'
    else:
        iTimeFormat = '%3.3d' 
    if ProblemType_EquatorialWave:
        specify_n_ticks = True
        n_ticks = [6,6]
    else:
        specify_n_ticks = False
        n_ticks = [0,0]
    if PlotZonalVelocity:
        if UseGivenColorBarLimits:
            FileName = myDGSEM2D.myNameList.ProblemType_FileName + '_ExactZonalVelocityLimits'
            ExactZonalVelocityLimits = CR.ReadStateVariableLimitsFromFile(myDGSEM2D.OutputDirectory,FileName+'.curve')
        else:
            ExactZonalVelocityLimits = [0.0,0.0]
        if not(ProblemType_NoExactSolution):
            title = titleroot + ':\nExact Zonal Velocity after\n' + DisplayTime
            PlotFileName = PlotFileNameRoot + '_ExactZonalVelocity_' + iTimeFormat %myDGSEM2D.iTime
            CR.PythonFilledContourPlot2DSaveAsPDF(myDGSEM2D.OutputDirectory,x,y,uExact,nContours,labels,labelfontsizes,
                                                  labelpads,tickfontsizes,UseGivenColorBarLimits,
                                                  ExactZonalVelocityLimits,nColorBarTicks,title,titlefontsize,SaveAsPDF,
                                                  PlotFileName,Show,DataType=DataType,colormap=colormap,
                                                  specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
        if not(ComputeOnlyExactSolution):
            if PlotNumericalSolution:
                title = titleroot + ':\nNumerical Zonal Velocity after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_NumericalZonalVelocity_' + iTimeFormat %myDGSEM2D.iTime)
                CR.PythonFilledContourPlot2DSaveAsPDF(myDGSEM2D.OutputDirectory,x,y,u,nContours,labels,labelfontsizes,
                                                      labelpads,tickfontsizes,UseGivenColorBarLimits,
                                                      ExactZonalVelocityLimits,nColorBarTicks,title,titlefontsize,
                                                      SaveAsPDF,PlotFileName,Show,DataType=DataType,colormap=colormap,
                                                      specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
            if not(ProblemType_NoExactSolution):
                if UseGivenColorBarLimits:
                    FileName = (myDGSEM2D.myNameList.ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                + '_ZonalVelocityErrorLimits')
                    ZonalVelocityErrorLimits = CR.ReadStateVariableLimitsFromFile(myDGSEM2D.OutputDirectory,
                                                                                  FileName+'.curve')
                else:
                    ZonalVelocityErrorLimits = [0.0,0.0]
                title = titleroot + ':\nZonal Velocity Error after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_ZonalVelocityError_' + iTimeFormat %myDGSEM2D.iTime)
                CR.PythonFilledContourPlot2DSaveAsPDF(myDGSEM2D.OutputDirectory,x,y,uError,nContours,labels,
                                                      labelfontsizes,labelpads,tickfontsizes,UseGivenColorBarLimits,
                                                      ZonalVelocityErrorLimits,nColorBarTicks,title,titlefontsize,
                                                      SaveAsPDF,PlotFileName,Show,DataType=DataType,colormap=colormap,
                                                      specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
    if PlotMeridionalVelocity:
        if UseGivenColorBarLimits:
            FileName = myDGSEM2D.myNameList.ProblemType_FileName + '_ExactMeridionalVelocityLimits'
            ExactMeridionalVelocityLimits = CR.ReadStateVariableLimitsFromFile(myDGSEM2D.OutputDirectory,
                                                                               FileName+'.curve')
        else:
            ExactMeridionalVelocityLimits = [0.0,0.0]
        if not(ProblemType_NoExactSolution):
            title = titleroot + ':\nExact Meridional Velocity after\n' + DisplayTime
            PlotFileName = PlotFileNameRoot + '_ExactMeridionalVelocity_' + iTimeFormat %myDGSEM2D.iTime
            CR.PythonFilledContourPlot2DSaveAsPDF(myDGSEM2D.OutputDirectory,x,y,vExact,nContours,labels,labelfontsizes,
                                                  labelpads,tickfontsizes,UseGivenColorBarLimits,
                                                  ExactMeridionalVelocityLimits,nColorBarTicks,title,titlefontsize,
                                                  SaveAsPDF,PlotFileName,Show,DataType=DataType,colormap=colormap,
                                                  specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
        if not(ComputeOnlyExactSolution):
            if PlotNumericalSolution:
                title = titleroot + ':\nNumerical Meridional Velocity after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_NumericalMeridionalVelocity_' + iTimeFormat %myDGSEM2D.iTime)
                CR.PythonFilledContourPlot2DSaveAsPDF(myDGSEM2D.OutputDirectory,x,y,v,nContours,labels,labelfontsizes,
                                                      labelpads,tickfontsizes,UseGivenColorBarLimits,
                                                      ExactMeridionalVelocityLimits,nColorBarTicks,title,titlefontsize,
                                                      SaveAsPDF,PlotFileName,Show,DataType=DataType,colormap=colormap,
                                                      specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
            if not(ProblemType_NoExactSolution):
                if UseGivenColorBarLimits:
                    FileName = (myDGSEM2D.myNameList.ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                + '_MeridionalVelocityErrorLimits')
                    MeridionalVelocityErrorLimits = CR.ReadStateVariableLimitsFromFile(myDGSEM2D.OutputDirectory,
                                                                                       FileName+'.curve')
                else:
                    MeridionalVelocityErrorLimits = [0.0,0.0]
                title = titleroot + ':\nMeridional Velocity Error after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_MeridionalVelocityError_' + iTimeFormat %myDGSEM2D.iTime)
                CR.PythonFilledContourPlot2DSaveAsPDF(myDGSEM2D.OutputDirectory,x,y,vError,nContours,labels,
                                                      labelfontsizes,labelpads,tickfontsizes,UseGivenColorBarLimits,
                                                      MeridionalVelocityErrorLimits,nColorBarTicks,title,titlefontsize,
                                                      SaveAsPDF,PlotFileName,Show,DataType=DataType,colormap=colormap,
                                                      specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
    if PlotSurfaceElevation:
        if UseGivenColorBarLimits:
            FileName = myDGSEM2D.myNameList.ProblemType_FileName + '_ExactSurfaceElevationLimits'
            ExactSurfaceElevationLimits = CR.ReadStateVariableLimitsFromFile(myDGSEM2D.OutputDirectory,
                                                                             FileName+'.curve')
        else:
            ExactSurfaceElevationLimits = [0.0,0.0]
        if not(ProblemType_NoExactSolution):
            title = titleroot + ':\nExact Surface Elevation after\n' + DisplayTime
            PlotFileName = PlotFileNameRoot + '_ExactSurfaceElevation_' + iTimeFormat %myDGSEM2D.iTime
            CR.PythonFilledContourPlot2DSaveAsPDF(myDGSEM2D.OutputDirectory,x,y,etaExact,nContours,labels,
                                                  labelfontsizes,labelpads,tickfontsizes,UseGivenColorBarLimits,
                                                  ExactSurfaceElevationLimits,nColorBarTicks,title,titlefontsize,
                                                  SaveAsPDF,PlotFileName,Show,DataType=DataType,colormap=colormap,
                                                  specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
        if not(ComputeOnlyExactSolution):
            if PlotNumericalSolution:
                title = titleroot + ':\nNumerical Surface Elevation after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_NumericalSurfaceElevation_' + iTimeFormat %myDGSEM2D.iTime)
                CR.PythonFilledContourPlot2DSaveAsPDF(myDGSEM2D.OutputDirectory,x,y,eta,nContours,labels,labelfontsizes,
                                                      labelpads,tickfontsizes,UseGivenColorBarLimits,
                                                      ExactSurfaceElevationLimits,nColorBarTicks,title,titlefontsize,
                                                      SaveAsPDF,PlotFileName,Show,DataType=DataType,colormap=colormap,
                                                      specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
            if not(ProblemType_NoExactSolution):
                if UseGivenColorBarLimits:
                    FileName = (myDGSEM2D.myNameList.ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                + '_SurfaceElevationErrorLimits')
                    SurfaceElevationErrorLimits = CR.ReadStateVariableLimitsFromFile(myDGSEM2D.OutputDirectory,
                                                                                     FileName+'.curve')
                else:
                    SurfaceElevationErrorLimits = [0.0,0.0]
                title = titleroot + ':\nSurface Elevation Error after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_SurfaceElevationError_' + iTimeFormat %myDGSEM2D.iTime)
                CR.PythonFilledContourPlot2DSaveAsPDF(myDGSEM2D.OutputDirectory,x,y,etaError,nContours,labels,
                                                      labelfontsizes,labelpads,tickfontsizes,UseGivenColorBarLimits,
                                                      SurfaceElevationErrorLimits,nColorBarTicks,title,titlefontsize,
                                                      SaveAsPDF,PlotFileName,Show,DataType=DataType,colormap=colormap,
                                                      specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)