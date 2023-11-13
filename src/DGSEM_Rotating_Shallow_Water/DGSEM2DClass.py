"""
Name: DGSEM2DClass.py
Author: Sid Bishnu
Details: This script defines the two-dimensional discontinuous Galerkin spectral element class.
"""


import numpy as np
import matplotlib.pyplot as plt
import self.model1d as model1d
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
    
    def __init__(myDGSEM2DParameters,ProblemType,ProblemType_RossbyWave,Problem_is_Linear,BoundaryCondition,
                 NonTrivialSourceTerms,NonTrivialDiffusionTerms,nEquations,nElementsX,nElementsY,nXi,nEta,nXiPlot,
                 nEtaPlot):
        myDGSEM2DParameters.ProblemType = ProblemType
        myDGSEM2DParameters.ProblemType_RossbyWave = ProblemType_RossbyWave
        myDGSEM2DParameters.Problem_is_Linear = Problem_is_Linear
        myDGSEM2DParameters.BoundaryCondition = BoundaryCondition
        myDGSEM2DParameters.NonTrivialSourceTerms = NonTrivialSourceTerms
        myDGSEM2DParameters.NonTrivialDiffusionTerms = NonTrivialDiffusionTerms
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
                 CourantNumber_Advection=0.5,CourantNumber_Diffusion=0.5,UseCourantNumberToDetermineTimeStep=False,
                 ReadFromSELFOutputData=False,BoundaryConditionAndDomainExtentsSpecified=False,
                 BoundaryCondition='Periodic',lX=0.0,lY=0.0,RiemannSolver='LocalLaxFriedrichs'):
        myDGSEM2D.myNameList = (
        Initialization.NameList(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,
                                CourantNumber_Advection,CourantNumber_Diffusion,UseCourantNumberToDetermineTimeStep,
                                ReadFromSELFOutputData))
        if BoundaryConditionAndDomainExtentsSpecified:
            myDGSEM2D.myNameList.ModifyNameList(
            PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,nXi,nEta,CourantNumber_Advection,
            CourantNumber_Diffusion,UseCourantNumberToDetermineTimeStep,BoundaryCondition,lX,lY)
        myDGSEM2D.myDGSEM2DParameters = (
        DGSEM2DParameters(ProblemType,myDGSEM2D.myNameList.ProblemType_RossbyWave,
                          myDGSEM2D.myNameList.Problem_is_Linear,myDGSEM2D.myNameList.BoundaryCondition,
                          myDGSEM2D.myNameList.NonTrivialSourceTerms,myDGSEM2D.myNameList.NonTrivialDiffusionTerms,
                          myDGSEM2D.myNameList.nEquations,nElementsX,nElementsY,nXi,nEta,nXiPlot,nEtaPlot))
        myDGSEM2D.myDGNodalStorage2D = DGNS2D.DGNodalStorage2D(nXi,nEta)
        myDGSEM2D.myQuadMesh = QM.QuadMesh(myDGSEM2D.myNameList.lX,myDGSEM2D.myNameList.lY,nElementsX,nElementsY,
                                           myDGSEM2D.myDGNodalStorage2D,myDGSEM2D.myNameList.ProblemType_EquatorialWave,
                                           myDGSEM2D.myNameList.QuadElementType)
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
        if ProblemType == 'Convergence_of_Spatial_Operators':
            myDGSEM2D.SpecifyRiemannSolver = True
        else:
            if myDGSEM2D.myNameList.Problem_is_Linear and not(ProblemType == 'Diffusion_Equation' 
                                                              or ProblemType == 'Advection_Diffusion_Equation'):
                myDGSEM2D.SpecifyRiemannSolver = False
            else:
                myDGSEM2D.SpecifyRiemannSolver = True
        myDGSEM2D.RiemannSolver = RiemannSolver
        myDGSEM2D.iTime = 0
        myDGSEM2D.time = 0.0
        
            
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
                    if (myDGSEM2D.myNameList.ProblemType == 'Manufactured_Planetary_Rossby_Wave' 
                        or myDGSEM2D.myNameList.ProblemType == 'Planetary_Rossby_Wave'
                        or myDGSEM2D.myNameList.ProblemType_EquatorialWave):
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.f[iXi,iEta] = f0 + beta0*y
                    else:
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.f[iXi,iEta] = f0
                    if (myDGSEM2D.myNameList.ProblemType == 'Manufactured_Topographic_Rossby_Wave'
                        or myDGSEM2D.myNameList.ProblemType == 'Topographic_Rossby_Wave'):
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta] = H0 + alpha0*y
                    else:
                        myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta] = H0
                    myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.c[iXi,iEta] = (
                    np.sqrt(g*myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta]))
            for iXi in range(0,max(nXi+1,nEta+1)):
                for iSide in range(0,4):
                    yBoundary = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.yBoundary[iXi,iSide]
                    if (myDGSEM2D.myNameList.ProblemType == 'Manufactured_Topographic_Rossby_Wave'
                        or myDGSEM2D.myNameList.ProblemType == 'Topographic_Rossby_Wave'):
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
        OutputDirectory += '_SELF_Output_Data'
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
                if ProblemType == 'NonLinear_Manufactured_Solution':
                    myDGSEM2D.myDGSolution2D[iElement].SurfaceElevationSourceTermAtInteriorNodes[iXi,iEta] = (
                    ESST.DetermineSurfaceElevationSourceTerm(ProblemType,myExactSolutionParameters,x,y,time))
                
                
def SpecifyInitialConditions(myDGSEM2D):
    ProblemType = myDGSEM2D.myDGSEM2DParameters.ProblemType
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                H = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta]
                if myDGSEM2D.myNameList.Problem_is_Linear or ProblemType == 'Viscous_Burgers_Equation':
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
    ProblemType = myDGSEM2D.myDGSEM2DParameters.ProblemType
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                H = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.H[iXi,iEta]
                if myDGSEM2D.myNameList.Problem_is_Linear or ProblemType == 'Viscous_Burgers_Equation':
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
    ProblemType = myDGSEM2D.myDGSEM2DParameters.ProblemType
    Problem_is_Linear = myDGSEM2D.myDGSEM2DParameters.Problem_is_Linear
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    if not(Problem_is_Linear or ProblemType == 'Viscous_Burgers_Equation'):
        u, v, eta = ExtractVelocitiesAndSurfaceElevationFromPrognosticVariables(myDGSEM2D)
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                if Problem_is_Linear or ProblemType == 'Viscous_Burgers_Equation':
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
            
            
def InterpolateSolutionGradientToBoundaries(myDGSEM2DParameters,myDGNodalStorage2D,myDGSolution2D):
    nEquations = myDGSEM2DParameters.nEquations
    nXi = myDGSEM2DParameters.nXi
    nEta = myDGSEM2DParameters.nEta
    for iEquation in range(0,nEquations):
        # Interpolate the solution derivatives to the south and north boundaries.
        for iXi in range(0,nXi+1):
            myDGSolution2D.SolutionGradientAtBoundaries[iEquation,0,iXi,0] = (
            np.dot(myDGSolution2D.SolutionGradientAtInteriorNodes[iEquation,0,iXi,:],
                   myDGNodalStorage2D.LagrangePolynomialsAlongSouthBoundary[:]))
            myDGSolution2D.SolutionGradientAtBoundaries[iEquation,1,iXi,0] = (
            np.dot(myDGSolution2D.SolutionGradientAtInteriorNodes[iEquation,1,iXi,:],
                   myDGNodalStorage2D.LagrangePolynomialsAlongSouthBoundary[:]))
            myDGSolution2D.SolutionGradientAtBoundaries[iEquation,0,iXi,2] = (
            np.dot(myDGSolution2D.SolutionGradientAtInteriorNodes[iEquation,0,iXi,:],
                   myDGNodalStorage2D.LagrangePolynomialsAlongNorthBoundary[:]))
            myDGSolution2D.SolutionGradientAtBoundaries[iEquation,1,iXi,2] = (
            np.dot(myDGSolution2D.SolutionGradientAtInteriorNodes[iEquation,1,iXi,:],
                   myDGNodalStorage2D.LagrangePolynomialsAlongNorthBoundary[:]))
        # Interpolate the solution derivatives to the east and west boundaries.
        for iEta in range(0,nEta+1):
            myDGSolution2D.SolutionGradientAtBoundaries[iEquation,0,iEta,1] = (
            np.dot(myDGSolution2D.SolutionGradientAtInteriorNodes[iEquation,0,:,iEta],
                   myDGNodalStorage2D.LagrangePolynomialsAlongEastBoundary[:]))
            myDGSolution2D.SolutionGradientAtBoundaries[iEquation,1,iEta,1] = (
            np.dot(myDGSolution2D.SolutionGradientAtInteriorNodes[iEquation,1,:,iEta],
                   myDGNodalStorage2D.LagrangePolynomialsAlongEastBoundary[:]))            
            myDGSolution2D.SolutionGradientAtBoundaries[iEquation,0,iEta,3] = (
            np.dot(myDGSolution2D.SolutionGradientAtInteriorNodes[iEquation,0,:,iEta],
                   myDGNodalStorage2D.LagrangePolynomialsAlongWestBoundary[:])) 
            myDGSolution2D.SolutionGradientAtBoundaries[iEquation,1,iEta,3] = (
            np.dot(myDGSolution2D.SolutionGradientAtInteriorNodes[iEquation,1,:,iEta],
                   myDGNodalStorage2D.LagrangePolynomialsAlongWestBoundary[:])) 
       

class FluxParameters:
    
    def __init__(myFluxParameters,c,g,H,nu,u0,v0):
        myFluxParameters.c = c
        myFluxParameters.g = g
        myFluxParameters.H = H
        myFluxParameters.nu = nu
        myFluxParameters.u0 = u0
        myFluxParameters.v0 = v0
        

def ComputeFlux(FluxType,myFluxParameters,State,StateGradient,ProblemType,Problem_is_Linear,
                PrognosticVariables='VelocitiesAndSurfaceElevation',AdvectiveFluxStage=1,
                WeakDerivativeDirection='X'):
    FluxX = np.zeros(3)
    FluxY = np.zeros(3)
    g = myFluxParameters.g
    H = myFluxParameters.H
    nu = myFluxParameters.nu
    u0 = myFluxParameters.u0
    v0 = myFluxParameters.v0
    u_x = StateGradient[0,0]
    u_y = StateGradient[0,1]
    eta_x = StateGradient[2,0]
    eta_y = StateGradient[2,1]
    if FluxType == 'Advective':
        if Problem_is_Linear:
            u = State[0]
            v = State[1]
            eta = State[2]
            if ProblemType == 'Diffusion_Equation':
                if AdvectiveFluxStage == 1:
                    if WeakDerivativeDirection == 'X':
                        FluxX[0] = u
                    elif WeakDerivativeDirection == 'Y':
                        FluxY[0] = u
            elif ProblemType == 'Advection_Diffusion_Equation':
                if AdvectiveFluxStage == 1:
                    if WeakDerivativeDirection == 'X':
                        FluxX[2] = eta
                    elif WeakDerivativeDirection == 'Y':
                        FluxY[2] = eta
                elif AdvectiveFluxStage == 2:
                    FluxX[2] = u0*eta
                    FluxY[2] = v0*eta
            else:
                if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'Velocities':
                    FluxX[0] = g*eta
                    FluxY[1] = g*eta
                if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'SurfaceElevation':
                    FluxX[2] = H*u
                    FluxY[2] = H*v   
        else:
            if ProblemType == 'Viscous_Burgers_Equation':
                u = State[0]
                if AdvectiveFluxStage == 1:
                    if WeakDerivativeDirection == 'X':
                        FluxX[0] = u
                elif AdvectiveFluxStage == 2:
                    FluxX[0] = 0.5*u**2.0
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
    elif FluxType == 'Diffusive':
        if ProblemType == 'Diffusion_Equation':
            FluxX[0] = nu*u_x
            FluxY[0] = nu*u_y
        elif ProblemType == 'Advection_Diffusion_Equation':
            FluxX[2] = nu*eta_x
            FluxY[2] = nu*eta_y
        elif ProblemType == 'Viscous_Burgers_Equation':
            FluxX[0] = nu*u_x                
    return FluxX, FluxY


def ComputeFluxNormalToEdge(FluxType,myFluxParameters,State,StateGradient,ProblemType,Problem_is_Linear,nHatX,nHatY,
                            PrognosticVariables='VelocitiesAndSurfaceElevation',AdvectiveFluxStage=1,
                            WeakDerivativeDirection='X'):
    FluxX, FluxY = ComputeFlux(FluxType,myFluxParameters,State,StateGradient,ProblemType,Problem_is_Linear,
                               PrognosticVariables,AdvectiveFluxStage,WeakDerivativeDirection)
    NormalFlux = nHatX*FluxX + nHatY*FluxY
    return NormalFlux
    

def ComputeEigenvaluesNormalToEdge(myFluxParameters,State,ProblemType,Problem_is_Linear,nHatX,nHatY):
    c = myFluxParameters.c
    g = myFluxParameters.g
    u0 = myFluxParameters.u0
    v0 = myFluxParameters.v0
    Eigenvalues = np.zeros(3)
    if Problem_is_Linear:
        if ProblemType == 'Advection_Diffusion_Equation':
            Eigenvalues[2] = nHatX*u0 + nHatY*v0
        else:
            Eigenvalues[1] = c
            Eigenvalues[2] = -c
    else:
        if ProblemType == 'Viscous_Burgers_Equation':
            u = State[0]
            Eigenvalues[0] = nHatX*u
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


def DetermineCharacteristicVariables(myFluxParameters,State,nHatX,nHatY):
    c = myFluxParameters.c
    g = myFluxParameters.g
    u = State[0]
    v = State[1]
    eta = State[2]
    CharacteristicState = np.zeros(3)
    CharacteristicState[0] = nHatY*u - nHatX*v
    CharacteristicState[1] = c/g*(nHatX*u + nHatY*v) + eta
    CharacteristicState[2] = -c/g*(nHatX*u + nHatY*v) + eta
    return CharacteristicState  


def ExactRiemannSolver(myFluxParameters,InternalState,ExternalState,nHatX,nHatY,
                       PrognosticVariables='VelocitiesAndSurfaceElevation'):
    c = myFluxParameters.c
    g = myFluxParameters.g
    InternalCharacteristicState = DetermineCharacteristicVariables(myFluxParameters,InternalState,nHatX,nHatY)
    ExternalCharacteristicState = DetermineCharacteristicVariables(myFluxParameters,ExternalState,nHatX,nHatY)
    NumericalFlux = np.zeros(3)
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'Velocities':
        NumericalFlux[0] = 0.5*nHatX*g*(InternalCharacteristicState[1] + ExternalCharacteristicState[2])
        NumericalFlux[1] = 0.5*nHatY*g*(InternalCharacteristicState[1] + ExternalCharacteristicState[2])
    if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'SurfaceElevation':
        NumericalFlux[2] = 0.5*c*(InternalCharacteristicState[1] - ExternalCharacteristicState[2])
    return NumericalFlux


def SpecifyPrognosticVariablesToAdvance(ProblemType,PrognosticVariables):
    if ProblemType == 'Diffusion_Equation':
        iEquation_Start = 0
        iEquation_End = 1
    elif ProblemType == 'Advection_Diffusion_Equation':
        iEquation_Start = 2
        iEquation_End = 3
    else:
        if PrognosticVariables == 'VelocitiesAndSurfaceElevation':
            iEquation_Start = 0
            iEquation_End = 3
        elif PrognosticVariables == 'Velocities':
            iEquation_Start = 0
            iEquation_End = 2
        elif PrognosticVariables == 'SurfaceElevation':
            iEquation_Start = 2
            iEquation_End = 3
    return iEquation_Start, iEquation_End
        

def BassiRebayRiemannSolver(FluxType,myFluxParameters,InternalState,InternalStateGradient,ExternalState,
                            ExternalStateGradient,ProblemType,Problem_is_Linear,nHatX,nHatY,
                            PrognosticVariables='VelocitiesAndSurfaceElevation',AdvectiveFluxStage=1,
                            WeakDerivativeDirection='X'):
    InternalFluxNormalToEdge = (
    ComputeFluxNormalToEdge(FluxType,myFluxParameters,InternalState,InternalStateGradient,ProblemType,
                            Problem_is_Linear,nHatX,nHatY,PrognosticVariables,AdvectiveFluxStage,
                            WeakDerivativeDirection))
    ExternalFluxNormalToEdge = (
    ComputeFluxNormalToEdge(FluxType,myFluxParameters,ExternalState,ExternalStateGradient,ProblemType,
                            Problem_is_Linear,nHatX,nHatY,PrognosticVariables,AdvectiveFluxStage,
                            WeakDerivativeDirection))
    NumericalFlux = np.zeros(3)
    iEquation_Start, iEquation_End = SpecifyPrognosticVariablesToAdvance(ProblemType,PrognosticVariables)
    for iEquation in range(iEquation_Start,iEquation_End):
        NumericalFlux[iEquation] = 0.5*(InternalFluxNormalToEdge[iEquation] + ExternalFluxNormalToEdge[iEquation])   
    return NumericalFlux


def LocalLaxFriedrichsRiemannSolver(FluxType,myFluxParameters,InternalState,InternalStateGradient,ExternalState,
                                    ExternalStateGradient,ProblemType,Problem_is_Linear,nHatX,nHatY,
                                    PrognosticVariables='VelocitiesAndSurfaceElevation',AdvectiveFluxStage=1):
    InternalFluxNormalToEdge = ComputeFluxNormalToEdge(FluxType,myFluxParameters,InternalState,InternalStateGradient,
                                                       ProblemType,Problem_is_Linear,nHatX,nHatY,PrognosticVariables,
                                                       AdvectiveFluxStage)
    ExternalFluxNormalToEdge = ComputeFluxNormalToEdge(FluxType,myFluxParameters,ExternalState,ExternalStateGradient,
                                                       ProblemType,Problem_is_Linear,nHatX,nHatY,PrognosticVariables,
                                                       AdvectiveFluxStage)
    InternalEigenvaluesNormalToEdge = ComputeEigenvaluesNormalToEdge(myFluxParameters,InternalState,ProblemType,
                                                                     Problem_is_Linear,nHatX,nHatY)
    ExternalEigenvaluesNormalToEdge = ComputeEigenvaluesNormalToEdge(myFluxParameters,ExternalState,ProblemType,
                                                                     Problem_is_Linear,nHatX,nHatY)
    NumericalFlux = np.zeros(3)
    iEquation_Start, iEquation_End = SpecifyPrognosticVariablesToAdvance(ProblemType,PrognosticVariables)
    for iEquation in range(iEquation_Start,iEquation_End):
        NumericalFlux[iEquation] = 0.5*(InternalFluxNormalToEdge[iEquation] + ExternalFluxNormalToEdge[iEquation]
                                        - (max(np.max(abs(InternalEigenvaluesNormalToEdge)),
                                               np.max(abs(ExternalEigenvaluesNormalToEdge)))
                                           *(ExternalState[iEquation] - InternalState[iEquation])))
    return NumericalFlux


def ComputeNumericalFlux(FluxType,myEdge,myExactSolutionParameters,myQuadMeshParameters,myDGSEM2DParameters,
                         myQuadElements,myDGSolution2D,time,dt,PrognosticVariables='VelocitiesAndSurfaceElevation',
                         ComputeExternalSurfaceElevationOneTimeStepEarlier=False,SpecifyRiemannSolver=True,
                         RiemannSolver='LocalLaxFriedrichs',AdvectiveFluxStage=1,WeakDerivativeDirection='X'):
    ProblemType = myDGSEM2DParameters.ProblemType
    Problem_is_Linear = myDGSEM2DParameters.Problem_is_Linear
    BoundaryCondition = myDGSEM2DParameters.BoundaryCondition
    nElementsX = myDGSEM2DParameters.nElementsX
    nElementsY = myDGSEM2DParameters.nElementsY
    nXi = myDGSEM2DParameters.nXi
    nEta = myDGSEM2DParameters.nEta
    nXi = max(nXi,nEta)
    g = myExactSolutionParameters.g
    nu = myExactSolutionParameters.nu
    u0 = myExactSolutionParameters.u0
    v0 = myExactSolutionParameters.v0
    ElementID1 = myEdge.ElementIDs[0]
    ElementSide1 = myEdge.ElementSides[0]
    iEquation_Start, iEquation_End = SpecifyPrognosticVariablesToAdvance(ProblemType,PrognosticVariables)
    time_ZonalVelocity = time
    time_MeridionalVelocity = time
    if ComputeExternalSurfaceElevationOneTimeStepEarlier:
        time_SurfaceElevation = time - dt
    else:
        time_SurfaceElevation = time
    myFluxParameters = FluxParameters(0.0,g,0.0,nu,u0,v0) 
    if myEdge.EdgeType == myQuadMeshParameters.INTERIOR:
        ElementID2 = myEdge.ElementIDs[1]
        ElementSide2 = abs(myEdge.ElementSides[1])
        kXi = myEdge.start - myEdge.increment
        for jXi in range(0,nXi+1):
            c = myQuadElements[ElementID1-1].myMappedGeometry2D.cBoundary[jXi,ElementSide1-1]
            H = myQuadElements[ElementID1-1].myMappedGeometry2D.HBoundary[jXi,ElementSide1-1]
            myFluxParameters.c = c
            myFluxParameters.H = H
            InternalState = myDGSolution2D[ElementID1-1].SolutionAtBoundaries[:,jXi,ElementSide1-1]
            ExternalState = myDGSolution2D[ElementID2-1].SolutionAtBoundaries[:,kXi,ElementSide2-1]
            InternalStateGradient = myDGSolution2D[ElementID1-1].SolutionGradientAtBoundaries[:,:,jXi,ElementSide1-1]
            ExternalStateGradient = myDGSolution2D[ElementID2-1].SolutionGradientAtBoundaries[:,:,kXi,ElementSide2-1]
            nHat = myQuadElements[ElementID1-1].myMappedGeometry2D.nHat[jXi,ElementSide1-1]
            nHatX = nHat.Components[0]
            nHatY = nHat.Components[1]
            if not(SpecifyRiemannSolver):
                NumericalFlux = ExactRiemannSolver(myFluxParameters,InternalState,ExternalState,nHatX,nHatY,
                                                   PrognosticVariables)
            else:
                if RiemannSolver == 'LocalLaxFriedrichs':
                    NumericalFlux = (
                    LocalLaxFriedrichsRiemannSolver(FluxType,myFluxParameters,InternalState,InternalStateGradient,
                                                    ExternalState,ExternalStateGradient,ProblemType,Problem_is_Linear,
                                                    nHatX,nHatY,PrognosticVariables,AdvectiveFluxStage))
                elif RiemannSolver == 'BassiRebay':
                    NumericalFlux = (
                    BassiRebayRiemannSolver(FluxType,myFluxParameters,InternalState,InternalStateGradient,ExternalState,
                                            ExternalStateGradient,ProblemType,Problem_is_Linear,nHatX,nHatY,
                                            PrognosticVariables,AdvectiveFluxStage,WeakDerivativeDirection))
            for iEquation in range(iEquation_Start,iEquation_End):
                if FluxType == 'Advective' and AdvectiveFluxStage == 1:
                    if WeakDerivativeDirection == 'X':
                        myDGSolution2D[ElementID1-1].ScaledNumericalWeakDerivativeFluxXAtBoundaries[iEquation,jXi,
                                                                                                    ElementSide1-1] = (
                        (NumericalFlux[iEquation]
                         *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))
                        myDGSolution2D[ElementID2-1].ScaledNumericalWeakDerivativeFluxXAtBoundaries[iEquation,kXi,
                                                                                                    ElementSide2-1] = (
                        (-NumericalFlux[iEquation]
                         *myQuadElements[ElementID2-1].myMappedGeometry2D.ScalingFactors[kXi,ElementSide2-1]))
                    elif WeakDerivativeDirection == 'Y':
                        myDGSolution2D[ElementID1-1].ScaledNumericalWeakDerivativeFluxYAtBoundaries[iEquation,jXi,
                                                                                                    ElementSide1-1] = (
                        (NumericalFlux[iEquation]
                         *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))
                        myDGSolution2D[ElementID2-1].ScaledNumericalWeakDerivativeFluxYAtBoundaries[iEquation,kXi,
                                                                                                    ElementSide2-1] = (
                        (-NumericalFlux[iEquation]
                         *myQuadElements[ElementID2-1].myMappedGeometry2D.ScalingFactors[kXi,ElementSide2-1]))
                elif FluxType == 'Advective' and AdvectiveFluxStage == 2:
                    myDGSolution2D[ElementID1-1].ScaledNumericalAdvectiveFluxAtBoundaries[iEquation,jXi,
                                                                                          ElementSide1-1] = (
                    (NumericalFlux[iEquation]
                     *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))
                    myDGSolution2D[ElementID2-1].ScaledNumericalAdvectiveFluxAtBoundaries[iEquation,kXi,
                                                                                          ElementSide2-1] = (
                    (-NumericalFlux[iEquation]
                     *myQuadElements[ElementID2-1].myMappedGeometry2D.ScalingFactors[kXi,ElementSide2-1]))
                elif FluxType == 'Diffusive':
                    myDGSolution2D[ElementID1-1].ScaledNumericalDiffusiveFluxAtBoundaries[iEquation,jXi,
                                                                                          ElementSide1-1] = (
                    (NumericalFlux[iEquation]
                     *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))
                    myDGSolution2D[ElementID2-1].ScaledNumericalDiffusiveFluxAtBoundaries[iEquation,kXi,
                                                                                          ElementSide2-1] = (
                    (-NumericalFlux[iEquation]
                     *myQuadElements[ElementID2-1].myMappedGeometry2D.ScalingFactors[kXi,ElementSide2-1]))
            kXi += myEdge.increment
    else:
        ExternalState = np.zeros((3,nXi+1))
        ExternalStateGradient = np.zeros((3,2,nXi+1))
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
                ExternalStateGradient[:,:,jXi] = (
                myDGSolution2D[ElementID2-1].SolutionGradientAtBoundaries[:,:,jXi,ElementSide2-1])
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
                    ExternalStateGradient[:,:,jXi] = (
                    myDGSolution2D[ElementID2-1].SolutionGradientAtBoundaries[:,:,jXi,ElementSide2-1])
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
                    if FluxType == 'Diffusive':
                        ExternalStateGradient[0,0,jXi] = (
                        ESST.DetermineExactZonalVelocityZonalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                      time_ZonalVelocity))
                        ExternalStateGradient[0,1,jXi] = (
                        ESST.DetermineExactZonalVelocityMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                           time_ZonalVelocity))
                        ExternalStateGradient[1,0,jXi] = (
                        ESST.DetermineExactMeridionalVelocityZonalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                           time_MeridionalVelocity))
                        ExternalStateGradient[1,1,jXi] = (
                        ESST.DetermineExactMeridionalVelocityMeridionalGradient(ProblemType,myExactSolutionParameters,x,
                                                                                y,time_MeridionalVelocity))                        
                        ExternalStateGradient[2,0,jXi] = (
                        ESST.DetermineExactSurfaceElevationZonalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                         time_SurfaceElevation))
                        ExternalStateGradient[2,1,jXi] = (
                        ESST.DetermineExactSurfaceElevationMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                              time_SurfaceElevation))
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
                    ExternalStateGradient[:,:,jXi] = (
                    myDGSolution2D[ElementID2-1].SolutionGradientAtBoundaries[:,:,jXi,ElementSide2-1])
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
                    if FluxType == 'Diffusive':
                        ExternalStateGradient[0,0,jXi] = (
                        ESST.DetermineExactZonalVelocityZonalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                      time_ZonalVelocity))
                        ExternalStateGradient[0,1,jXi] = (
                        ESST.DetermineExactZonalVelocityMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                           time_ZonalVelocity))
                        ExternalStateGradient[1,0,jXi] = (
                        ESST.DetermineExactMeridionalVelocityZonalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                           time_MeridionalVelocity))
                        ExternalStateGradient[1,1,jXi] = (
                        ESST.DetermineExactMeridionalVelocityMeridionalGradient(ProblemType,myExactSolutionParameters,x,
                                                                                y,time_MeridionalVelocity))                        
                        ExternalStateGradient[2,0,jXi] = (
                        ESST.DetermineExactSurfaceElevationZonalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                         time_SurfaceElevation))
                        ExternalStateGradient[2,1,jXi] = (
                        ESST.DetermineExactSurfaceElevationMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                              time_SurfaceElevation))
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
                if FluxType == 'Diffusive':
                    ExternalStateGradient[0,0,jXi] = (
                    ESST.DetermineExactZonalVelocityZonalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                  time_ZonalVelocity))
                    ExternalStateGradient[0,1,jXi] = (
                    ESST.DetermineExactZonalVelocityMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                       time_ZonalVelocity))
                    ExternalStateGradient[1,0,jXi] = (
                    ESST.DetermineExactMeridionalVelocityZonalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                       time_MeridionalVelocity))
                    ExternalStateGradient[1,1,jXi] = (
                    ESST.DetermineExactMeridionalVelocityMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                            time_MeridionalVelocity))                        
                    ExternalStateGradient[2,0,jXi] = (
                    ESST.DetermineExactSurfaceElevationZonalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                     time_SurfaceElevation))
                    ExternalStateGradient[2,1,jXi] = (
                    ESST.DetermineExactSurfaceElevationMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,
                                                                          time_SurfaceElevation))
        elif BoundaryCondition == 'Radiation' and not(Problem_is_Linear):
            for jXi in range(0,nXi+1):
                H = myQuadElements[ElementID1-1].myMappedGeometry2D.HBoundary[jXi,ElementSide1-1]
                ExternalState[2,jXi] = H
        for jXi in range(0,nXi+1):
            c = myQuadElements[ElementID1-1].myMappedGeometry2D.cBoundary[jXi,ElementSide1-1]
            H = myQuadElements[ElementID1-1].myMappedGeometry2D.HBoundary[jXi,ElementSide1-1]
            myFluxParameters.c = c
            myFluxParameters.H = H
            nHat = myQuadElements[ElementID1-1].myMappedGeometry2D.nHat[jXi,ElementSide1-1]
            nHatX = nHat.Components[0]
            nHatY = nHat.Components[1]
            InternalState = myDGSolution2D[ElementID1-1].SolutionAtBoundaries[:,jXi,ElementSide1-1]
            InternalStateGradient = myDGSolution2D[ElementID1-1].SolutionGradientAtBoundaries[:,:,jXi,ElementSide1-1]
            if BoundaryCondition == 'Reflection':
                ExternalState[0,jXi] = -2.0*nHatX*nHatY*InternalState[1] - (nHatX**2.0 - nHatY**2.0)*InternalState[0]
                ExternalState[1,jXi] = -2.0*nHatX*nHatY*InternalState[0] + (nHatX**2.0 - nHatY**2.0)*InternalState[1]
                ExternalState[2,jXi] = InternalState[2]
            if not(SpecifyRiemannSolver):
                NumericalFlux = ExactRiemannSolver(myFluxParameters,InternalState,ExternalState[:,jXi],nHatX,nHatY,
                                                   PrognosticVariables)
            else:
                if RiemannSolver == 'LocalLaxFriedrichs':
                    NumericalFlux = (
                    LocalLaxFriedrichsRiemannSolver(FluxType,myFluxParameters,InternalState,InternalStateGradient,
                                                    ExternalState[:,jXi],ExternalStateGradient[:,:,jXi],ProblemType,
                                                    Problem_is_Linear,nHatX,nHatY,PrognosticVariables,
                                                    AdvectiveFluxStage))
                elif RiemannSolver == 'BassiRebay':
                    NumericalFlux = (
                    BassiRebayRiemannSolver(FluxType,myFluxParameters,InternalState,InternalStateGradient,
                                            ExternalState[:,jXi],ExternalStateGradient[:,:,jXi],ProblemType,
                                            Problem_is_Linear,nHatX,nHatY,PrognosticVariables,AdvectiveFluxStage,
                                            WeakDerivativeDirection))
            for iEquation in range(iEquation_Start,iEquation_End): 
                if FluxType == 'Advective' and AdvectiveFluxStage == 1:   
                    if WeakDerivativeDirection == 'X':       
                        myDGSolution2D[ElementID1-1].ScaledNumericalWeakDerivativeFluxXAtBoundaries[iEquation,jXi,
                                                                                                    ElementSide1-1] = (
                        (NumericalFlux[iEquation]
                         *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))
                    elif WeakDerivativeDirection == 'Y':
                        myDGSolution2D[ElementID1-1].ScaledNumericalWeakDerivativeFluxYAtBoundaries[iEquation,jXi,
                                                                                                    ElementSide1-1] = (
                        (NumericalFlux[iEquation]
                         *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))
                elif FluxType == 'Advective' and AdvectiveFluxStage == 2:          
                    myDGSolution2D[ElementID1-1].ScaledNumericalAdvectiveFluxAtBoundaries[iEquation,jXi,
                                                                                          ElementSide1-1] = (
                    (NumericalFlux[iEquation]
                     *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))
                elif FluxType == 'Diffusive':   
                    myDGSolution2D[ElementID1-1].ScaledNumericalDiffusiveFluxAtBoundaries[iEquation,jXi,
                                                                                          ElementSide1-1] = (
                    (NumericalFlux[iEquation]
                     *myQuadElements[ElementID1-1].myMappedGeometry2D.ScalingFactors[jXi,ElementSide1-1]))


def DGSystemDerivative(ProblemType,nEquations,nXi,DGDerivativeMatrix,QuadratureWeights,
                       ScaledNumericalFluxAtLeftBoundary,ScaledNumericalFluxAtRightBoundary,InteriorFlux,
                       LagrangePolynomialsAtLeftBoundary,LagrangePolynomialsAtRightBoundary,
                       PrognosticVariables='VelocitiesAndSurfaceElevation'):
    FluxDerivative = np.zeros((nEquations,nXi+1))
    iEquation_Start, iEquation_End = SpecifyPrognosticVariablesToAdvance(ProblemType,PrognosticVariables)
    for iEquation in range(iEquation_Start,iEquation_End):
        FluxDerivative[iEquation,:] = np.matmul(DGDerivativeMatrix,InteriorFlux[iEquation,:])
    for iEquation in range(iEquation_Start,iEquation_End):
        for iXi in range(0,nXi+1):
            FluxDerivative[iEquation,iXi] += (
            ((ScaledNumericalFluxAtRightBoundary[iEquation]*LagrangePolynomialsAtRightBoundary[iXi] 
              + ScaledNumericalFluxAtLeftBoundary[iEquation]*LagrangePolynomialsAtLeftBoundary[iXi])
             /QuadratureWeights[iXi]))
    return FluxDerivative


def ComputeContravariantFluxDerivatives(
FluxType,myDGSEM2DParameters,myExactSolutionParameters,myDGNodalStorage2D,myMappedGeometry2D,myDGSolution2D,
PrognosticVariables='VelocitiesAndSurfaceElevation',AdvectiveFluxStage=1,WeakDerivativeDirection='X'):
    ProblemType = myDGSEM2DParameters.ProblemType
    Problem_is_Linear = myDGSEM2DParameters.Problem_is_Linear
    nEquations = myDGSEM2DParameters.nEquations
    nXi = myDGSEM2DParameters.nXi
    nEta = myDGSEM2DParameters.nEta
    g = myExactSolutionParameters.g
    nu = myExactSolutionParameters.nu
    u0 = myExactSolutionParameters.u0
    v0 = myExactSolutionParameters.v0
    dXdXi = myMappedGeometry2D.dXdXi
    dXdEta = myMappedGeometry2D.dXdEta
    dYdXi = myMappedGeometry2D.dYdXi
    dYdEta = myMappedGeometry2D.dYdEta
    ContravariantFluxX = np.zeros((nEquations,nXi+1,nEta+1))
    ContravariantFluxY = np.zeros((nEquations,nXi+1,nEta+1))
    ContravariantFluxDerivativeX = np.zeros((nEquations,nXi+1,nEta+1))
    ContravariantFluxDerivativeY = np.zeros((nEquations,nXi+1,nEta+1))              
    myFluxParameters = FluxParameters(0.0,g,0.0,nu,u0,v0)     
    SolutionGradientAtInteriorNodes = myDGSolution2D.SolutionGradientAtInteriorNodes[:,:,:,:]
    for iXi in range(0,nXi+1):
        for iEta in range(0,nEta+1):
            c = myMappedGeometry2D.c[iXi,iEta]
            H = myMappedGeometry2D.H[iXi,iEta]
            myFluxParameters.c = c
            myFluxParameters.H = H
            SolutionAtInteriorNode = myDGSolution2D.SolutionAtInteriorNodes[:,iXi,iEta]
            SolutionGradientAtInteriorNode = SolutionGradientAtInteriorNodes[:,:,iXi,iEta]
            FluxX, FluxY = ComputeFlux(FluxType,myFluxParameters,SolutionAtInteriorNode,SolutionGradientAtInteriorNode,
                                       ProblemType,Problem_is_Linear,PrognosticVariables,AdvectiveFluxStage,
                                       WeakDerivativeDirection)
            ContravariantFluxX[:,iXi,iEta] = dYdEta[iXi,iEta]*FluxX[:] - dXdEta[iXi,iEta]*FluxY[:]
            ContravariantFluxY[:,iXi,iEta] = -dYdXi[iXi,iEta]*FluxX[:] + dXdXi[iXi,iEta]*FluxY[:]
    if FluxType == 'Advective': 
        if AdvectiveFluxStage == 1:
            if WeakDerivativeDirection == 'X':
                ScaledNumericalFluxAtWestBoundary = myDGSolution2D.ScaledNumericalWeakDerivativeFluxXAtBoundaries[:,:,3]   
                ScaledNumericalFluxAtEastBoundary = myDGSolution2D.ScaledNumericalWeakDerivativeFluxXAtBoundaries[:,:,1]
                ScaledNumericalFluxAtSouthBoundary = (
                myDGSolution2D.ScaledNumericalWeakDerivativeFluxXAtBoundaries[:,:,0])
                ScaledNumericalFluxAtNorthBoundary = (
                myDGSolution2D.ScaledNumericalWeakDerivativeFluxXAtBoundaries[:,:,2])       
            elif WeakDerivativeDirection == 'Y':
                ScaledNumericalFluxAtWestBoundary = myDGSolution2D.ScaledNumericalWeakDerivativeFluxYAtBoundaries[:,:,3]   
                ScaledNumericalFluxAtEastBoundary = myDGSolution2D.ScaledNumericalWeakDerivativeFluxYAtBoundaries[:,:,1]
                ScaledNumericalFluxAtSouthBoundary = (
                myDGSolution2D.ScaledNumericalWeakDerivativeFluxYAtBoundaries[:,:,0])
                ScaledNumericalFluxAtNorthBoundary = (
                myDGSolution2D.ScaledNumericalWeakDerivativeFluxYAtBoundaries[:,:,2])
        elif AdvectiveFluxStage == 2:
            ScaledNumericalFluxAtWestBoundary = myDGSolution2D.ScaledNumericalAdvectiveFluxAtBoundaries[:,:,3]   
            ScaledNumericalFluxAtEastBoundary = myDGSolution2D.ScaledNumericalAdvectiveFluxAtBoundaries[:,:,1]
            ScaledNumericalFluxAtSouthBoundary = myDGSolution2D.ScaledNumericalAdvectiveFluxAtBoundaries[:,:,0]
            ScaledNumericalFluxAtNorthBoundary = myDGSolution2D.ScaledNumericalAdvectiveFluxAtBoundaries[:,:,2]
    elif FluxType == 'Diffusive':
        ScaledNumericalFluxAtWestBoundary = myDGSolution2D.ScaledNumericalDiffusiveFluxAtBoundaries[:,:,3]   
        ScaledNumericalFluxAtEastBoundary = myDGSolution2D.ScaledNumericalDiffusiveFluxAtBoundaries[:,:,1]
        ScaledNumericalFluxAtSouthBoundary = myDGSolution2D.ScaledNumericalDiffusiveFluxAtBoundaries[:,:,0]
        ScaledNumericalFluxAtNorthBoundary = myDGSolution2D.ScaledNumericalDiffusiveFluxAtBoundaries[:,:,2]
    for iEta in range(0,nEta+1):
        ContravariantFluxDerivativeX[:,:,iEta] = (
        DGSystemDerivative(ProblemType,nEquations,nXi,myDGNodalStorage2D.DGDerivativeMatrixX,
                           myDGNodalStorage2D.myGaussQuadratureWeightX,ScaledNumericalFluxAtWestBoundary[:,iEta],
                           ScaledNumericalFluxAtEastBoundary[:,iEta],ContravariantFluxX[:,:,iEta],
                           myDGNodalStorage2D.LagrangePolynomialsAlongWestBoundary,
                           myDGNodalStorage2D.LagrangePolynomialsAlongEastBoundary,PrognosticVariables))
    for iXi in range(0,nXi+1):
        ContravariantFluxDerivativeY[:,iXi,:] = (
        DGSystemDerivative(ProblemType,nEquations,nEta,myDGNodalStorage2D.DGDerivativeMatrixY,
                           myDGNodalStorage2D.myGaussQuadratureWeightY,ScaledNumericalFluxAtSouthBoundary[:,iXi],
                           ScaledNumericalFluxAtNorthBoundary[:,iXi],ContravariantFluxY[:,iXi,:],
                           myDGNodalStorage2D.LagrangePolynomialsAlongSouthBoundary,
                           myDGNodalStorage2D.LagrangePolynomialsAlongNorthBoundary,PrognosticVariables))
    return ContravariantFluxDerivativeX, ContravariantFluxDerivativeY


def DGWeakDerivatives(myExactSolutionParameters,myDGSEM2DParameters,myDGNodalStorage2D,myMappedGeometry2D,
                      myDGSolution2D,PrognosticVariables='VelocitiesAndSurfaceElevation'):
    nXi = myDGSEM2DParameters.nXi
    nEta = myDGSEM2DParameters.nEta
    Jacobian = myMappedGeometry2D.Jacobian
    FluxType = 'Advective'
    AdvectiveFluxStage = 1
    WeakDerivativeDirection = 'X'
    X_ContravariantAdvectiveFluxDerivativeX, X_ContravariantAdvectiveFluxDerivativeY = (
    ComputeContravariantFluxDerivatives(FluxType,myDGSEM2DParameters,myExactSolutionParameters,myDGNodalStorage2D,
                                        myMappedGeometry2D,myDGSolution2D,PrognosticVariables,AdvectiveFluxStage,
                                        WeakDerivativeDirection))
    WeakDerivativeDirection = 'Y'
    Y_ContravariantAdvectiveFluxDerivativeX, Y_ContravariantAdvectiveFluxDerivativeY = (
    ComputeContravariantFluxDerivatives(FluxType,myDGSEM2DParameters,myExactSolutionParameters,myDGNodalStorage2D,
                                        myMappedGeometry2D,myDGSolution2D,PrognosticVariables,AdvectiveFluxStage,
                                        WeakDerivativeDirection))
    for iXi in range(0,nXi+1):
        for iEta in range(0,nEta+1):
            myDGSolution2D.SolutionGradientAtInteriorNodes[:,0,iXi,iEta] = (
            (X_ContravariantAdvectiveFluxDerivativeX[:,iXi,iEta] 
             + X_ContravariantAdvectiveFluxDerivativeY[:,iXi,iEta])/Jacobian[iXi,iEta])
            myDGSolution2D.SolutionGradientAtInteriorNodes[:,1,iXi,iEta] = (
            (Y_ContravariantAdvectiveFluxDerivativeX[:,iXi,iEta] 
             + Y_ContravariantAdvectiveFluxDerivativeY[:,iXi,iEta])/Jacobian[iXi,iEta])     
            

def GlobalWeakDerivatives(myDGSEM2D,time,PrognosticVariables='VelocitiesAndSurfaceElevation',
                          ComputeExternalSurfaceElevationOneTimeStepEarlier=False):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    dt = myDGSEM2D.myNameList.dt
    FluxType = 'Advective'
    SpecifyRiemannSolver = True
    RiemannSolver = 'BassiRebay'
    AdvectiveFluxStage = 1
    for iEdge in range(0,myDGSEM2D.myQuadMesh.nEdges):
        WeakDerivativeDirection = 'X'
        ComputeNumericalFlux(
        FluxType,myDGSEM2D.myQuadMesh.myEdges[iEdge],myDGSEM2D.myNameList.myExactSolutionParameters,
        myDGSEM2D.myQuadMesh.myQuadMeshParameters,myDGSEM2D.myDGSEM2DParameters,myDGSEM2D.myQuadMesh.myQuadElements,
        myDGSEM2D.myDGSolution2D,time,dt,PrognosticVariables,ComputeExternalSurfaceElevationOneTimeStepEarlier,
        SpecifyRiemannSolver,RiemannSolver,AdvectiveFluxStage,WeakDerivativeDirection)
        WeakDerivativeDirection = 'Y'
        ComputeNumericalFlux(
        FluxType,myDGSEM2D.myQuadMesh.myEdges[iEdge],myDGSEM2D.myNameList.myExactSolutionParameters,
        myDGSEM2D.myQuadMesh.myQuadMeshParameters,myDGSEM2D.myDGSEM2DParameters,myDGSEM2D.myQuadMesh.myQuadElements,
        myDGSEM2D.myDGSolution2D,time,dt,PrognosticVariables,ComputeExternalSurfaceElevationOneTimeStepEarlier,
        SpecifyRiemannSolver,RiemannSolver,AdvectiveFluxStage,WeakDerivativeDirection)       
    for iElement in range(0,nElements):
        DGWeakDerivatives(myDGSEM2D.myNameList.myExactSolutionParameters,myDGSEM2D.myDGSEM2DParameters,
                          myDGSEM2D.myDGNodalStorage2D,myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D,
                          myDGSEM2D.myDGSolution2D[iElement],PrognosticVariables)
        InterpolateSolutionGradientToBoundaries(myDGSEM2D.myDGSEM2DParameters,myDGSEM2D.myDGNodalStorage2D,
                                                myDGSEM2D.myDGSolution2D[iElement])


def DGTimeDerivative(myExactSolutionParameters,myDGSEM2DParameters,myDGNodalStorage2D,myMappedGeometry2D,myDGSolution2D,
                     time,PrognosticVariables='VelocitiesAndSurfaceElevation'):
    alpha0 = myExactSolutionParameters.alpha0
    g = myExactSolutionParameters.g
    ProblemType = myDGSEM2DParameters.ProblemType
    NonTrivialSourceTerms = myDGSEM2DParameters.NonTrivialSourceTerms
    NonTrivialDiffusionTerms = myDGSEM2DParameters.NonTrivialDiffusionTerms
    nXi = myDGSEM2DParameters.nXi
    nEta = myDGSEM2DParameters.nEta                 
    Jacobian = myMappedGeometry2D.Jacobian
    if not(ProblemType == 'Diffusion_Equation'):
        FluxType = 'Advective'
        AdvectiveFluxStage = 2
        ContravariantAdvectiveFluxDerivativeX, ContravariantAdvectiveFluxDerivativeY = (
        ComputeContravariantFluxDerivatives(FluxType,myDGSEM2DParameters,myExactSolutionParameters,myDGNodalStorage2D,
                                            myMappedGeometry2D,myDGSolution2D,PrognosticVariables,AdvectiveFluxStage))
    if NonTrivialDiffusionTerms:
        FluxType = 'Diffusive'
        ContravariantDiffusiveFluxDerivativeX, ContravariantDiffusiveFluxDerivativeY = (
        ComputeContravariantFluxDerivatives(FluxType,myDGSEM2DParameters,myExactSolutionParameters,myDGNodalStorage2D,
                                            myMappedGeometry2D,myDGSolution2D,PrognosticVariables))
    for iXi in range(0,nXi+1):
        for iEta in range(0,nEta+1):
            TendencyAtInteriorNode = np.zeros(3)
            if not(ProblemType == 'Diffusion_Equation'):
                TendencyAtInteriorNode[:] = -(ContravariantAdvectiveFluxDerivativeX[:,iXi,iEta] 
                                              + ContravariantAdvectiveFluxDerivativeY[:,iXi,iEta])/Jacobian[iXi,iEta]
            if NonTrivialDiffusionTerms:
                TendencyAtInteriorNode[:] += (ContravariantDiffusiveFluxDerivativeX[:,iXi,iEta]
                                              + ContravariantDiffusiveFluxDerivativeY[:,iXi,iEta])/Jacobian[iXi,iEta]
            f = myMappedGeometry2D.f[iXi,iEta]
            if PrognosticVariables == 'VelocitiesAndSurfaceElevation' or PrognosticVariables == 'Velocities':
                TendencyAtInteriorNode[0] += f*myDGSolution2D.SolutionAtInteriorNodes[1,iXi,iEta]
                TendencyAtInteriorNode[1] -= f*myDGSolution2D.SolutionAtInteriorNodes[0,iXi,iEta]
                if (not(myDGSEM2DParameters.Problem_is_Linear) 
                    and myDGSEM2DParameters.ProblemType == 'Topographic_Rossby_Wave'):
                    TendencyAtInteriorNode[1] -= g*alpha0*myDGSolution2D.SolutionAtInteriorNodes[2,iXi,iEta]
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
                TendencyAtInteriorNode[:] += SourceTermAtInteriorNode[:]  
            myDGSolution2D.TendencyAtInteriorNodes[:,iXi,iEta] = TendencyAtInteriorNode[:]


def GlobalTimeDerivative(myDGSEM2D,time,PrognosticVariables='VelocitiesAndSurfaceElevation',
                         ComputeExternalSurfaceElevationOneTimeStepEarlier=False):
    NonTrivialDiffusionTerms = myDGSEM2D.myDGSEM2DParameters.NonTrivialDiffusionTerms
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    dt = myDGSEM2D.myNameList.dt
    for iElement in range(0,nElements):
        InterpolateSolutionToBoundaries(myDGSEM2D.myDGSEM2DParameters,myDGSEM2D.myDGNodalStorage2D,
                                        myDGSEM2D.myDGSolution2D[iElement])   
    if NonTrivialDiffusionTerms:
        GlobalWeakDerivatives(myDGSEM2D,time,PrognosticVariables,ComputeExternalSurfaceElevationOneTimeStepEarlier) 
    FluxType = 'Advective'
    AdvectiveFluxStage = 2                                                       
    for iEdge in range(0,myDGSEM2D.myQuadMesh.nEdges):
        ComputeNumericalFlux(
        FluxType,myDGSEM2D.myQuadMesh.myEdges[iEdge],myDGSEM2D.myNameList.myExactSolutionParameters,
        myDGSEM2D.myQuadMesh.myQuadMeshParameters,myDGSEM2D.myDGSEM2DParameters,myDGSEM2D.myQuadMesh.myQuadElements,
        myDGSEM2D.myDGSolution2D,time,dt,PrognosticVariables,ComputeExternalSurfaceElevationOneTimeStepEarlier,
        myDGSEM2D.SpecifyRiemannSolver,myDGSEM2D.RiemannSolver,AdvectiveFluxStage)
    FluxType = 'Diffusive'
    SpecifyRiemannSolver = True
    RiemannSolver = 'BassiRebay'
    for iEdge in range(0,myDGSEM2D.myQuadMesh.nEdges):
        ComputeNumericalFlux(FluxType,myDGSEM2D.myQuadMesh.myEdges[iEdge],
                             myDGSEM2D.myNameList.myExactSolutionParameters,myDGSEM2D.myQuadMesh.myQuadMeshParameters,
                             myDGSEM2D.myDGSEM2DParameters,myDGSEM2D.myQuadMesh.myQuadElements,myDGSEM2D.myDGSolution2D,
                             time,dt,PrognosticVariables,ComputeExternalSurfaceElevationOneTimeStepEarlier,
                             SpecifyRiemannSolver,RiemannSolver)       
    for iElement in range(0,nElements):
        DGTimeDerivative(myDGSEM2D.myNameList.myExactSolutionParameters,myDGSEM2D.myDGSEM2DParameters,
                         myDGSEM2D.myDGNodalStorage2D,myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D,
                         myDGSEM2D.myDGSolution2D[iElement],time,PrognosticVariables)


def WriteStateDGSEM2D(myDGSEM2D,filename):
    ProblemType = myDGSEM2D.myDGSEM2DParameters.ProblemType
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
    if not(Problem_is_Linear or ProblemType == 'Viscous_Burgers_Equation'):
        u_AllElements, v_AllElements, eta_AllElements = (
        ExtractVelocitiesAndSurfaceElevationFromPrognosticVariables(myDGSEM2D))
    for iElement in range(0,nElements):
        x = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[:,:]
        y = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[:,:]
        Jacobian = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.Jacobian[:,:]
        if Problem_is_Linear or ProblemType == 'Viscous_Burgers_Equation':
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
    ProblemType = myDGSEM2D.myDGSEM2DParameters.ProblemType
    Problem_is_Linear = myDGSEM2D.myDGSEM2DParameters.Problem_is_Linear
    nElementsX = myDGSEM2D.myDGSEM2DParameters.nElementsX
    nElementsY = myDGSEM2D.myDGSEM2DParameters.nElementsY
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    nXiPlot = myDGSEM2D.myDGSEM2DParameters.nXiPlot
    nEtaPlot = myDGSEM2D.myDGSEM2DParameters.nEtaPlot
    cwd = os.getcwd()
    path = cwd + '/' + myDGSEM2D.OutputDirectory + '/'
    os.chdir(path)
    if (not(ComputeOnlyExactSolution) and myDGSEM2D.myNameList.ReadFromSELFOutputData 
        and ProblemType == 'Viscous_Burgers_Equation'):
        myExactSolutionParameters = myDGSEM2D.myNameList.myExactSolutionParameters
        time = myDGSEM2D.time
        # Initialize the pyself 1D model.
        pyself_model = model1d.model() 
        # Load in the model data from file.
        pyself_model.load(filename + '.h5')
        u_pyself = pyself_model.solution[:,0,:].compute()
        u_AllElements = np.zeros((nElements,nXi+1,nEta+1))
        for iElementY in range(0,nElementsY):
            for iElementX in range(0,nElementsX):
                iElement = iElementY*nElementsX + iElementX
                for iEta in range(0,nEta+1):
                    u_AllElements[iElement,:,iEta] = u_pyself[iElementX,:]    
    filename += '.tec'
    outputfile = open(filename,'w')
    if ComputeOnlyExactSolution:
        outputfile.write('VARIABLES = "X", "Y", "Jacobian", "uExact", "vExact", "etaExact"\n')       
    else:
        outputfile.write('VARIABLES = "X", "Y", "Jacobian", "uExact", "vExact", "etaExact", "u", "v", "eta", "uError", '
                         + '"vError", "etaError"\n')
    if not(Problem_is_Linear or ProblemType == 'Viscous_Burgers_Equation'):
        u_AllElements, v_AllElements, eta_AllElements = (
        ExtractVelocitiesAndSurfaceElevationFromPrognosticVariables(myDGSEM2D))
    for iElement in range(0,nElements):
        x = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[:,:]
        y = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[:,:]
        Jacobian = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.Jacobian[:,:]
        uExact = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[0,:,:]
        vExact = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[1,:,:]
        etaExact = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[2,:,:]
        if Problem_is_Linear or ProblemType == 'Viscous_Burgers_Equation':
            if (not(ComputeOnlyExactSolution) and myDGSEM2D.myNameList.ReadFromSELFOutputData 
                and ProblemType == 'Viscous_Burgers_Equation'):
                for iXi in range(0,nXi+1):
                    for iEta in range(0,nEta+1):
                        uExact[iXi,iEta] = (
                        ESST.DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,x[iXi,iEta],y[iXi,iEta],
                                                         time))
                u = u_AllElements[iElement,:,:]
            else:
                u = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,:,:]
            v = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,:,:]
            eta = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,:,:]
        else:
            u = u_AllElements[iElement,:,:]
            v = v_AllElements[iElement,:,:]
            eta = eta_AllElements[iElement,:,:]
        if (not(ComputeOnlyExactSolution) and myDGSEM2D.myNameList.ReadFromSELFOutputData 
            and ProblemType == 'Viscous_Burgers_Equation'):
            uError = u - uExact
        else:
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


def PlotRossbyWaveAlongSection(lY,nPointsX,x,y,eta,OutputDirectory,labelfontsizes,labelpads,tickfontsizes,etaLimits,
                               title,titlefontsize,SaveAsPDF,PlotFileName,Show):
    nPoints = len(x)
    xPlotAlongSection = np.zeros(nPointsX)
    etaPlotAlongSection = np.zeros(nPointsX)
    yTolerance = 10.0**(-6.0)
    iPlotAlongSection = -1
    for iPoint in range(0,nPoints):
        if abs(y[iPoint]*1000.0 - 0.5*lY) < yTolerance and x[iPoint] not in xPlotAlongSection:
            iPlotAlongSection += 1
            xPlotAlongSection[iPlotAlongSection] = x[iPoint]
            etaPlotAlongSection[iPlotAlongSection] = eta[iPoint]
    xPlotAlongSection_Final = np.zeros(iPlotAlongSection+1)
    etaPlotAlongSection_Final = np.zeros(iPlotAlongSection+1)
    xPlotAlongSection_Final[:] = xPlotAlongSection[0:iPlotAlongSection+1]
    etaPlotAlongSection_Final[:] = etaPlotAlongSection[0:iPlotAlongSection+1]
    # Obtain the sorted indices of xPlotAlongSection_Final.
    sorted_indices = np.argsort(xPlotAlongSection_Final)
    # Apply these sorted indices to xPlotAlongSection_Final and etaPlotAlongSection_Final.
    xPlotAlongSection_Final_Sorted = xPlotAlongSection_Final[sorted_indices]
    etaPlotAlongSection_Final_Sorted = etaPlotAlongSection_Final[sorted_indices]
    ToleranceAsPercentage = 12.0
    etaDifference = etaLimits[1] - etaLimits[0]
    etaLimitsAlongSection = np.zeros(2)
    etaLimitsAlongSection[0] = etaLimits[0] - 0.5*ToleranceAsPercentage/100.0*etaDifference
    etaLimitsAlongSection[1] = etaLimits[1] + 0.5*ToleranceAsPercentage/100.0*etaDifference
    xLabel = 'Distance (km) along Zonal Section\nthrough Center of the Domain'
    yLabel = 'Surface elevation (m)'
    labels = [xLabel,yLabel]
    CR.PythonPlot1DSaveAsPDF(OutputDirectory,'regular',xPlotAlongSection_Final_Sorted,etaPlotAlongSection_Final_Sorted,
                             2.0,'-','k',False,'s',10.0,labels,labelfontsizes,labelpads,tickfontsizes,title,
                             titlefontsize,SaveAsPDF,PlotFileName,Show,setYAxisLimits=[True,True],
                             yAxisLimits=[etaLimitsAlongSection[0],etaLimitsAlongSection[1]])
    
    
def PythonPlotStateDGSEM2D(myDGSEM2D,filename,DataType,DisplayTime,UseGivenColorBarLimits=True,
                           ComputeOnlyExactSolution=False,SpecifyDataTypeInPlotFileName=False,
                           PlotNumericalSolution=False):
    ProblemType_RossbyWave = myDGSEM2D.myDGSEM2DParameters.ProblemType_RossbyWave
    ProblemType = myDGSEM2D.myDGSEM2DParameters.ProblemType
    nElementsX = myDGSEM2D.myDGSEM2DParameters.nElementsX
    nElementsY = myDGSEM2D.myDGSEM2DParameters.nElementsY
    nXiPlot = myDGSEM2D.myDGSEM2DParameters.nXiPlot
    nEtaPlot = myDGSEM2D.myDGSEM2DParameters.nEtaPlot
    ProblemType_EquatorialWave = myDGSEM2D.myNameList.ProblemType_EquatorialWave
    PlotZonalVelocity = myDGSEM2D.myNameList.LogicalArrayPlot[0]
    PlotMeridionalVelocity = myDGSEM2D.myNameList.LogicalArrayPlot[1]
    PlotSurfaceElevation = myDGSEM2D.myNameList.LogicalArrayPlot[2]
    if myDGSEM2D.myNameList.ReadFromSELFOutputData and ProblemType == 'Viscous_Burgers_Equation':
        WriteInterpolatedStateDGSEM2D(myDGSEM2D,filename.split(".")[0],ComputeOnlyExactSolution)
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
                        if myDGSEM2D.myNameList.ReadFromSELFOutputData and ProblemType_RossbyWave:
                            x[iPointX] = data[i,0] + 0.5*myDGSEM2D.myNameList.lX                        
                            y[iPointY] = data[i,1] + 0.5*myDGSEM2D.myNameList.lY 
                        else:
                            x[iPointX] = data[i,0]                        
                            y[iPointY] = data[i,1]
                        if myDGSEM2D.myNameList.ReadFromSELFOutputData and ProblemType_RossbyWave:
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
                        if myDGSEM2D.myNameList.ReadFromSELFOutputData and ProblemType_RossbyWave:
                            x[iPoint] = data[i,0] + 0.5*myDGSEM2D.myNameList.lX                        
                            y[iPoint] = data[i,1] + 0.5*myDGSEM2D.myNameList.lY 
                        else:
                            x[iPoint] = data[i,0]                        
                            y[iPoint] = data[i,1]
                        if myDGSEM2D.myNameList.ReadFromSELFOutputData and ProblemType_RossbyWave:
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
    if myDGSEM2D.myNameList.ProblemType == 'Advection_Diffusion_Equation':
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
    if (ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Advection_Diffusion_Equation' 
        or ProblemType == 'Viscous_Burgers_Equation'):
        colormap = plt.cm.YlOrRd
    else:   
        colormap = plt.cm.seismic
    colormap_error = plt.cm.seismic
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
        if not(ProblemType_RossbyWave):
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
            if not(ProblemType_RossbyWave):
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
                                                      SaveAsPDF,PlotFileName,Show,DataType=DataType,
                                                      colormap=colormap_error,specify_n_ticks=specify_n_ticks,
                                                      n_ticks=n_ticks)
    if PlotMeridionalVelocity:
        if UseGivenColorBarLimits:
            FileName = myDGSEM2D.myNameList.ProblemType_FileName + '_ExactMeridionalVelocityLimits'
            ExactMeridionalVelocityLimits = CR.ReadStateVariableLimitsFromFile(myDGSEM2D.OutputDirectory,
                                                                               FileName+'.curve')
        else:
            ExactMeridionalVelocityLimits = [0.0,0.0]
        if not(ProblemType_RossbyWave):
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
            if not(ProblemType_RossbyWave):
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
                                                      SaveAsPDF,PlotFileName,Show,DataType=DataType,
                                                      colormap=colormap_error,specify_n_ticks=specify_n_ticks,
                                                      n_ticks=n_ticks)
    if PlotSurfaceElevation:
        if UseGivenColorBarLimits:
            FileName = myDGSEM2D.myNameList.ProblemType_FileName + '_ExactSurfaceElevationLimits'
            ExactSurfaceElevationLimits = CR.ReadStateVariableLimitsFromFile(myDGSEM2D.OutputDirectory,
                                                                             FileName+'.curve')
        else:
            ExactSurfaceElevationLimits = [0.0,0.0]
        if not(ProblemType_RossbyWave):
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
                if myDGSEM2D.myNameList.ReadFromSELFOutputData and ProblemType_RossbyWave:
                    PlotFileNameAlongSection = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                                + '_NumericalSurfaceElevationAlongSection_'
                                                + iTimeFormat %myDGSEM2D.iTime)
                    PlotRossbyWaveAlongSection(myDGSEM2D.myNameList.lY,nPointsX,x,y,eta,myDGSEM2D.OutputDirectory,
                                               labelfontsizes,labelpads,tickfontsizes,ExactSurfaceElevationLimits,title,
                                               titlefontsize,SaveAsPDF,PlotFileNameAlongSection,Show)
            if not(ProblemType_RossbyWave):
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
                                                      SaveAsPDF,PlotFileName,Show,DataType=DataType,
                                                      colormap=colormap_error,specify_n_ticks=specify_n_ticks,
                                                      n_ticks=n_ticks)