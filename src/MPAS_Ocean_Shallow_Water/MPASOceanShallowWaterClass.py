"""
Name: MPASOceanShallowWaterClass.py
Author: Sid Bishnu
Details: This script defines the MPAS-Ocean shallow water class discretized in space with the TRiSK-based mimetic 
finite volume method used in MPAS-Ocean.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.ticker import FuncFormatter
import os
import sys
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR
    import ExactSolutionsAndSourceTerms as ESST
    import Initialization
    import MeshClass
    import SolutionClass
    
        
class DiagnosticVariablesToCompute:
    
    def __init__(myDiagnosticVariablesToCompute):
        myDiagnosticVariablesToCompute.LayerThickness = False
        myDiagnosticVariablesToCompute.LayerThicknessEdge = False
        myDiagnosticVariablesToCompute.RelativeVorticityCell = False
        myDiagnosticVariablesToCompute.DivergenceKineticEnergyCell = False
        myDiagnosticVariablesToCompute.TangentialVelocity = False
        myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityVertex = False
        myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityEdge = False
        myDiagnosticVariablesToCompute.NormalizedRelativeVorticityCell = False


class MPASOceanShallowWater:
    
    def __init__(myMPASOceanShallowWater,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                 TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                 Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,
                 BaseMeshFileName,MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,CourantNumber=0.5,
                 UseCourantNumberToDetermineTimeStep=False,SpecifyBoundaryCondition=False,BoundaryCondition='Periodic',
                 ReadDomainExtentsfromMeshFile=False,DebugVersion=False):
        myMPASOceanShallowWater.myNameList = (
        Initialization.NameList(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,CourantNumber,
                                UseCourantNumberToDetermineTimeStep))
        myMPASOceanShallowWater.myMesh = MeshClass.Mesh(myMPASOceanShallowWater.myNameList,PrintBasicGeometry,
                                                        MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,
                                                        PrintOutput,UseAveragedQuantities,SpecifyBoundaryCondition,
                                                        BoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion)
        if ReadDomainExtentsfromMeshFile:
            lX = myMPASOceanShallowWater.myMesh.lX
            lY = myMPASOceanShallowWater.myMesh.lY
            dx = myMPASOceanShallowWater.myMesh.dx
            dy = myMPASOceanShallowWater.myMesh.dy
            myMPASOceanShallowWater.myNameList.ModifyNameList(PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                                              CourantNumber,UseCourantNumberToDetermineTimeStep,
                                                              BoundaryCondition,lX,lY,dx,dy)
        nCells = myMPASOceanShallowWater.myMesh.nCells
        nEdges = myMPASOceanShallowWater.myMesh.nEdges
        nVertices = myMPASOceanShallowWater.myMesh.nVertices
        myMPASOceanShallowWater.mySolution = SolutionClass.Solution(nCells,nEdges,nVertices,TimeIntegrator)
        myMPASOceanShallowWater.DetermineCoriolisParameterAndBottomDepth()
        myMPASOceanShallowWater.RootOutputDirectory, myMPASOceanShallowWater.OutputDirectory = (
        MakeOutputDirectories(ProblemType))
        myMPASOceanShallowWater.iTime = 0
        myMPASOceanShallowWater.time = 0.0
            
    def DetermineCoriolisParameterAndBottomDepth(myMPASOceanShallowWater):
        alpha0 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.alpha0
        beta0 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.beta0
        f0 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.f0
        H0 = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.H0
        if (myMPASOceanShallowWater.myNameList.ProblemType == 'Planetary_Rossby_Wave' 
            or myMPASOceanShallowWater.myNameList.ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave'
            or myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave):
            if myMPASOceanShallowWater.myMesh.UseAveragedQuantities:
                for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
                    yQP = (myMPASOceanShallowWater.myMesh.yEdge[iEdge] 
                           + (0.5*myMPASOceanShallowWater.myMesh.HexagonLength
                              *np.sin(myMPASOceanShallowWater.myMesh.angleEdge[iEdge])
                              *myMPASOceanShallowWater.myMesh.myQuadratureOnEdge.x))
                    fQP = f0 + beta0*yQP
                    myMPASOceanShallowWater.myMesh.fEdge[iEdge] = (
                    (np.dot(0.5*myMPASOceanShallowWater.myMesh.HexagonLength*fQP,
                            myMPASOceanShallowWater.myMesh.myQuadratureOnEdge.w)
                     /myMPASOceanShallowWater.myMesh.HexagonLength))
            else:
                myMPASOceanShallowWater.myMesh.fEdge[:] = f0 + beta0*myMPASOceanShallowWater.myMesh.yEdge[:]
            myMPASOceanShallowWater.myMesh.fVertex[:] = f0 + beta0*myMPASOceanShallowWater.myMesh.yVertex[:]
        else:
            myMPASOceanShallowWater.myMesh.fEdge[:] = f0
            myMPASOceanShallowWater.myMesh.fVertex[:] = f0
        if (myMPASOceanShallowWater.myNameList.ProblemType == 'Topographic_Rossby_Wave'
            or (myMPASOceanShallowWater.myNameList.ProblemType 
                == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave')):
            if myMPASOceanShallowWater.myMesh.UseAveragedQuantities:
                for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
                    yQP = (myMPASOceanShallowWater.myMesh.yCell[iCell] 
                           + (myMPASOceanShallowWater.myMesh.myQuadratureOnHexagon.y[:]
                              *myMPASOceanShallowWater.myMesh.HexagonLength))
                    bottomDepthQP = H0 + alpha0*yQP
                    myMPASOceanShallowWater.myMesh.bottomDepth[iCell] = (
                    np.dot(bottomDepthQP,myMPASOceanShallowWater.myMesh.myQuadratureOnHexagon.w))
                for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
                    yQP = (myMPASOceanShallowWater.myMesh.yEdge[iEdge] 
                           + (0.5*myMPASOceanShallowWater.myMesh.HexagonLength
                              *np.sin(myMPASOceanShallowWater.myMesh.angleEdge[iEdge])
                              *myMPASOceanShallowWater.myMesh.myQuadratureOnEdge.x))
                    bottomDepthQP = H0 + alpha0*yQP
                    myMPASOceanShallowWater.myMesh.bottomDepthEdge[iEdge] = (
                    (np.dot(0.5*myMPASOceanShallowWater.myMesh.HexagonLength*bottomDepthQP,
                            myMPASOceanShallowWater.myMesh.myQuadratureOnEdge.w)
                     /myMPASOceanShallowWater.myMesh.HexagonLength))
            else:
                myMPASOceanShallowWater.myMesh.bottomDepth[:] = H0 + alpha0*myMPASOceanShallowWater.myMesh.yCell[:]
                myMPASOceanShallowWater.myMesh.bottomDepthEdge[:] = H0 + alpha0*myMPASOceanShallowWater.myMesh.yEdge[:]
        else:
            myMPASOceanShallowWater.myMesh.bottomDepth[:] = H0
            myMPASOceanShallowWater.myMesh.bottomDepthEdge[:] = H0
                    
    def ComputeRelativeVorticityAndCirculation(myMPASOceanShallowWater,NormalVelocity):
        for iVertex in range(0,myMPASOceanShallowWater.myMesh.nVertices):
            myMPASOceanShallowWater.mySolution.circulation[iVertex] = 0.0
            myMPASOceanShallowWater.mySolution.relativeVorticity[iVertex] = 0.0
            InverseAreaTriangle = 1.0/myMPASOceanShallowWater.myMesh.areaTriangle[iVertex]
            for iVertexDegree in range(0,myMPASOceanShallowWater.myMesh.vertexDegree):
                EdgeID = myMPASOceanShallowWater.myMesh.edgesOnVertex[iVertex,iVertexDegree]
                iEdge = EdgeID - 1
                r_tmp = myMPASOceanShallowWater.myMesh.dcEdge[iEdge]*NormalVelocity[iEdge]
                myMPASOceanShallowWater.mySolution.circulation[iVertex] += (
                myMPASOceanShallowWater.myMesh.edgeSignOnVertex[iVertex,iVertexDegree]*r_tmp)
                myMPASOceanShallowWater.mySolution.relativeVorticity[iVertex] += (
                myMPASOceanShallowWater.myMesh.edgeSignOnVertex[iVertex,iVertexDegree]*r_tmp*InverseAreaTriangle)
                    
    def DiagnosticSolve(myMPASOceanShallowWater,NormalVelocity,SurfaceElevation,myDiagnosticVariablesToCompute,
                        NormalVelocityTendencyComputation=False):
        if (myDiagnosticVariablesToCompute.LayerThickness or myDiagnosticVariablesToCompute.LayerThicknessEdge 
            or myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityVertex):
            for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
                myMPASOceanShallowWater.mySolution.layerThickness[iCell] = (
                SurfaceElevation[iCell] + myMPASOceanShallowWater.myMesh.bottomDepth[iCell])    
        if myDiagnosticVariablesToCompute.LayerThicknessEdge:
            if (not(myMPASOceanShallowWater.myNameList.UseWettingDrying) 
                or (myMPASOceanShallowWater.myNameList.UseWettingDrying 
                    and myMPASOceanShallowWater.myNameList.ThicknessFluxType == 'centered')):
                for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
                    CellID1 = myMPASOceanShallowWater.myMesh.cellsOnEdge[iEdge,0]
                    iCell1 = CellID1 - 1
                    CellID2 = myMPASOceanShallowWater.myMesh.cellsOnEdge[iEdge,1]
                    iCell2 = CellID2 - 1
                    # Centered difference
                    myMPASOceanShallowWater.mySolution.layerThicknessEdge[iEdge] = (
                    (0.5*(SurfaceElevation[iCell1] + SurfaceElevation[iCell2]) 
                     + myMPASOceanShallowWater.myMesh.bottomDepthEdge[iEdge]))
            else:
                if (myMPASOceanShallowWater.myNameList.UseWettingDrying
                    and myMPASOceanShallowWater.myNameList.ThicknessFluxType != 'centered'):
                    if myMPASOceanShallowWater.myNameList.ThicknessFluxType == 'upwind':
                        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
                            CellID1 = myMPASOceanShallowWater.myMesh.cellsOnEdge[iEdge,0]
                            iCell1 = CellID1 - 1
                            CellID2 = myMPASOceanShallowWater.myMesh.cellsOnEdge[iEdge,1]
                            iCell2 = CellID2 - 1
                            # Upwind
                            if NormalVelocity[iEdge] > 0.0:
                                myMPASOceanShallowWater.mySolution.layerThicknessEdge[iEdge] = (
                                myMPASOceanShallowWater.mySolution.layerThickness[iCell1])
                            elif NormalVelocity[iEdge] < 0.0:
                                myMPASOceanShallowWater.mySolution.layerThicknessEdge[iEdge] = (
                                myMPASOceanShallowWater.mySolution.layerThickness[iCell2])
                            else:
                                myMPASOceanShallowWater.mySolution.layerThicknessEdge[iEdge] = (
                                max(myMPASOceanShallowWater.mySolution.layerThickness[iCell1],
                                    myMPASOceanShallowWater.mySolution.layerThickness[iCell2]))
                else:
                    print('Thickness flux option %s is not known!' 
                          %(myMPASOceanShallowWater.myNameList.ThicknessFluxType))
                    sys.exit()
        if (myDiagnosticVariablesToCompute.RelativeVorticityCell 
            or myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityVertex 
            or myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityEdge 
            or myDiagnosticVariablesToCompute.NormalizedRelativeVorticityCell):
            myMPASOceanShallowWater.ComputeRelativeVorticityAndCirculation(NormalVelocity)
        if myDiagnosticVariablesToCompute.RelativeVorticityCell:
            for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
                myMPASOceanShallowWater.mySolution.relativeVorticityCell[iCell] = 0.0
                InverseAreaCell = 1.0/myMPASOceanShallowWater.myMesh.areaCell[iCell]
                for iEdgeOnCell in range(0,myMPASOceanShallowWater.myMesh.nEdgesOnCell[iCell]):
                    jID = myMPASOceanShallowWater.myMesh.kiteIndexOnCell[iCell,iEdgeOnCell]
                    j = jID - 1
                    VertexID = myMPASOceanShallowWater.myMesh.verticesOnCell[iCell,iEdgeOnCell]
                    iVertex = VertexID - 1
                    myMPASOceanShallowWater.mySolution.relativeVorticityCell[iCell] += (
                    (myMPASOceanShallowWater.myMesh.kiteAreasOnVertex[iVertex,j]
                     *myMPASOceanShallowWater.mySolution.relativeVorticity[iVertex]*InverseAreaCell))
        if myDiagnosticVariablesToCompute.DivergenceKineticEnergyCell:
            for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
                myMPASOceanShallowWater.mySolution.divergence[iCell] = 0.0
                myMPASOceanShallowWater.mySolution.kineticEnergyCell[iCell] = 0.0
                InverseAreaCell = 1.0/myMPASOceanShallowWater.myMesh.areaCell[iCell]
                for iEdgeOnCell in range(0,myMPASOceanShallowWater.myMesh.nEdgesOnCell[iCell]):
                    EdgeID = myMPASOceanShallowWater.myMesh.edgesOnCell[iCell,iEdgeOnCell]
                    iEdge = EdgeID - 1
                    edgeSignOnCell_temp = myMPASOceanShallowWater.myMesh.edgeSignOnCell[iCell,iEdgeOnCell]
                    dcEdge_temp = myMPASOceanShallowWater.myMesh.dcEdge[iEdge]
                    dvEdge_temp = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
                    r_tmp = dvEdge_temp*NormalVelocity[iEdge]*InverseAreaCell
                    myMPASOceanShallowWater.mySolution.divergence[iCell] -= edgeSignOnCell_temp*r_tmp
                    myMPASOceanShallowWater.mySolution.kineticEnergyCell[iCell] += (
                    0.25*r_tmp*dcEdge_temp*NormalVelocity[iEdge])
        if myDiagnosticVariablesToCompute.TangentialVelocity:
            for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
                if ((NormalVelocityTendencyComputation and myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 0.0)
                    or not(NormalVelocityTendencyComputation)): 
                    # i.e. if the edge is an interior one in case of normal velocity tendency computation
                    myMPASOceanShallowWater.mySolution.tangentialVelocity[iEdge] = 0.0
                    # Compute tangential velocities.
                    for iEdgeOnEdge in range(0,myMPASOceanShallowWater.myMesh.nEdgesOnEdge[iEdge]):
                        eoeID = myMPASOceanShallowWater.myMesh.edgesOnEdge[iEdge,iEdgeOnEdge]
                        eoe = eoeID - 1
                        edgeWeight = myMPASOceanShallowWater.myMesh.weightsOnEdge[iEdge,iEdgeOnEdge]
                        myMPASOceanShallowWater.mySolution.tangentialVelocity[iEdge] += edgeWeight*NormalVelocity[eoe]
        if (myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityVertex 
            or myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityEdge 
            or myDiagnosticVariablesToCompute.NormalizedRelativeVorticityCell):
            for iVertex in range(0,myMPASOceanShallowWater.myMesh.nVertices):
                InverseAreaTriangle = 1.0/myMPASOceanShallowWater.myMesh.areaTriangle[iVertex]
                layerThicknessVertex = 0.0
                for iVertexDegree in range(0,myMPASOceanShallowWater.myMesh.vertexDegree):
                    CellID = myMPASOceanShallowWater.myMesh.cellsOnVertex[iVertex,iVertexDegree]
                    iCell = CellID - 1
                    layerThicknessVertex += (myMPASOceanShallowWater.mySolution.layerThickness[iCell]
                                             *myMPASOceanShallowWater.myMesh.kiteAreasOnVertex[iVertex,iVertexDegree])
                layerThicknessVertex *= InverseAreaTriangle   
                myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityVertex[iVertex] = (
                myMPASOceanShallowWater.mySolution.relativeVorticity[iVertex]/layerThicknessVertex)
                myMPASOceanShallowWater.mySolution.normalizedPlanetaryVorticityVertex[iVertex] = (
                myMPASOceanShallowWater.myMesh.fVertex[iVertex]/layerThicknessVertex)   
        if myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityEdge:
            for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
                myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityEdge[iEdge] = 0.0
                myMPASOceanShallowWater.mySolution.normalizedPlanetaryVorticityEdge[iEdge] = 0.0
                VertexID1 = myMPASOceanShallowWater.myMesh.verticesOnEdge[iEdge,0]
                iVertex1 = VertexID1 - 1
                VertexID2 = myMPASOceanShallowWater.myMesh.verticesOnEdge[iEdge,1]
                iVertex2 = VertexID2 - 1
                myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityEdge[iEdge] = (
                0.5*(myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityVertex[iVertex1]
                     + myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityVertex[iVertex2]))
                myMPASOceanShallowWater.mySolution.normalizedPlanetaryVorticityEdge[iEdge] = (
                0.5*(myMPASOceanShallowWater.mySolution.normalizedPlanetaryVorticityVertex[iVertex1] 
                     + myMPASOceanShallowWater.mySolution.normalizedPlanetaryVorticityVertex[iVertex2]))
        if myDiagnosticVariablesToCompute.NormalizedRelativeVorticityCell:
            for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
                myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityCell[iCell] = 0.0
                InverseAreaCell = 1.0/myMPASOceanShallowWater.myMesh.areaCell[iCell]
                for iEdgeOnCell in range(0,myMPASOceanShallowWater.myMesh.nEdgesOnCell[iCell]):
                    jID = myMPASOceanShallowWater.myMesh.kiteIndexOnCell[iCell,iEdgeOnCell]
                    j = jID - 1
                    VertexID = myMPASOceanShallowWater.myMesh.verticesOnCell[iCell,iEdgeOnCell]
                    iVertex = VertexID - 1
                    myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityCell[iCell] += (
                    (myMPASOceanShallowWater.myMesh.kiteAreasOnVertex[iVertex,j]
                     *myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityVertex[iVertex]*InverseAreaCell))
                    
    def ComputeNormalVelocityTendencies(myMPASOceanShallowWater,time):
        ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
        myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
        UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
        myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
        g = myMPASOceanShallowWater.myNameList.myExactSolutionParameters.g
        if not(myMPASOceanShallowWater.myNameList.Problem_is_Linear):
            myDiagnosticVariablesToCompute = DiagnosticVariablesToCompute()
            myDiagnosticVariablesToCompute.LayerThickness = True
            myDiagnosticVariablesToCompute.LayerThicknessEdge = True
            myDiagnosticVariablesToCompute.DivergenceKineticEnergyCell = True
            myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityVertex = True
            myDiagnosticVariablesToCompute.NormalizedRelativeAndPlanetaryVorticityEdge = True
            myMPASOceanShallowWater.DiagnosticSolve(myMPASOceanShallowWater.mySolution.normalVelocity,
                                                    myMPASOceanShallowWater.mySolution.ssh,
                                                    myDiagnosticVariablesToCompute,
                                                    NormalVelocityTendencyComputation=True)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 0.0: # i.e. if the edge is an interior one
                if myMPASOceanShallowWater.myNameList.NonTrivialSourceTerms:
                    xEdge = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
                    yEdge = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
                    dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
                    angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
                    normalVelocitySourceTerm = (
                    ESST.DetermineNormalVelocitySourceTerm(ProblemType,myExactSolutionParameters,xEdge,yEdge,time,
                                                           UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
                else:
                    normalVelocitySourceTerm = 0.0
                CellID1 = myMPASOceanShallowWater.myMesh.cellsOnEdge[iEdge,0]
                iCell1 = CellID1 - 1
                CellID2 = myMPASOceanShallowWater.myMesh.cellsOnEdge[iEdge,1]
                iCell2 = CellID2 - 1
                dcEdge = myMPASOceanShallowWater.myMesh.dcEdge[iEdge]
                qArr = 0.0
                CoriolisTerm = 0.0
                for iEdgeOnEdge in range(0,myMPASOceanShallowWater.myMesh.nEdgesOnEdge[iEdge]):
                    eoeID = myMPASOceanShallowWater.myMesh.edgesOnEdge[iEdge,iEdgeOnEdge]
                    eoe = eoeID - 1
                    edgeWeight = myMPASOceanShallowWater.myMesh.weightsOnEdge[iEdge,iEdgeOnEdge]
                    if not(myMPASOceanShallowWater.myNameList.Problem_is_Linear):
                        workVorticity = 0.5*(myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityEdge[iEdge] 
                                             + myMPASOceanShallowWater.mySolution.normalizedRelativeVorticityEdge[eoe])
                        qArr += (edgeWeight*myMPASOceanShallowWater.mySolution.normalVelocity[eoe]*workVorticity
                                 *myMPASOceanShallowWater.mySolution.layerThicknessEdge[eoe])
                    CoriolisTerm += (edgeWeight*myMPASOceanShallowWater.mySolution.normalVelocity[eoe]
                                     *myMPASOceanShallowWater.myMesh.fEdge[eoe])
                if not(myMPASOceanShallowWater.myNameList.Problem_is_Linear):
                    myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge] = (
                    (CoriolisTerm 
                     - g*(myMPASOceanShallowWater.mySolution.ssh[iCell2] 
                          - myMPASOceanShallowWater.mySolution.ssh[iCell1])/dcEdge
                     + qArr - (myMPASOceanShallowWater.mySolution.kineticEnergyCell[iCell2] 
                               - myMPASOceanShallowWater.mySolution.kineticEnergyCell[iCell1])/dcEdge
                     + normalVelocitySourceTerm))            
                else:
                    myMPASOceanShallowWater.mySolution.normalVelocityTendency[iEdge] = (
                    (CoriolisTerm - g*(myMPASOceanShallowWater.mySolution.ssh[iCell2] 
                                       - myMPASOceanShallowWater.mySolution.ssh[iCell1])/dcEdge 
                     + normalVelocitySourceTerm))

    def ComputeSurfaceElevationTendencies(myMPASOceanShallowWater,time):
        ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
        myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
        UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
        myQuadratureOnHexagon = myMPASOceanShallowWater.myMesh.myQuadratureOnHexagon
        HexagonLength = myMPASOceanShallowWater.myMesh.HexagonLength
        if not(myMPASOceanShallowWater.myNameList.Problem_is_Linear):
            myDiagnosticVariablesToCompute = DiagnosticVariablesToCompute()
            myDiagnosticVariablesToCompute.LayerThickness = True
            myDiagnosticVariablesToCompute.LayerThicknessEdge = True
            myMPASOceanShallowWater.DiagnosticSolve(myMPASOceanShallowWater.mySolution.normalVelocity,
                                                    myMPASOceanShallowWater.mySolution.ssh,
                                                    myDiagnosticVariablesToCompute)
        for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
            if myMPASOceanShallowWater.myNameList.NonTrivialSourceTerms:
                xCell = myMPASOceanShallowWater.myMesh.xCell[iCell]
                yCell = myMPASOceanShallowWater.myMesh.yCell[iCell]
                SurfaceElevationSourceTerm = (
                ESST.DetermineSurfaceElevationSourceTerm(ProblemType,myExactSolutionParameters,xCell,yCell,time,
                                                         UseAveragedQuantities,myQuadratureOnHexagon,HexagonLength))
            else:
                SurfaceElevationSourceTerm = 0.0
            for iEdgeOnCell in range(0,myMPASOceanShallowWater.myMesh.nEdgesOnCell[iCell]):
                iEdgeID = myMPASOceanShallowWater.myMesh.edgesOnCell[iCell,iEdgeOnCell]
                iEdge = iEdgeID - 1
                if myMPASOceanShallowWater.myNameList.Problem_is_Linear:
                    if myMPASOceanShallowWater.myNameList.ProblemType == 'Topographic_Rossby_Wave':
                        flux = (myMPASOceanShallowWater.mySolution.normalVelocity[iEdge]
                                *myMPASOceanShallowWater.myMesh.bottomDepthEdge[iEdge])
                    else:
                        flux = (myMPASOceanShallowWater.mySolution.normalVelocity[iEdge]
                                *myMPASOceanShallowWater.myNameList.myExactSolutionParameters.H0)
                else:
                    flux = (myMPASOceanShallowWater.mySolution.normalVelocity[iEdge]
                            *myMPASOceanShallowWater.mySolution.layerThicknessEdge[iEdge])
                myMPASOceanShallowWater.mySolution.sshTendency[iCell] += (
                (myMPASOceanShallowWater.myMesh.edgeSignOnCell[iCell,iEdgeOnCell]*flux
                 *myMPASOceanShallowWater.myMesh.dvEdge[iEdge]))
            myMPASOceanShallowWater.mySolution.sshTendency[iCell] /= myMPASOceanShallowWater.myMesh.areaCell[iCell]
            myMPASOceanShallowWater.mySolution.sshTendency[iCell] += SurfaceElevationSourceTerm


def MakeOutputDirectories(ProblemType):
    cwd = os.getcwd()
    RootOutputDirectory = '../../output/MPAS_Ocean_Shallow_Water_Output'
    RootOutputDirectoryPath = cwd + '/' + RootOutputDirectory + '/'
    if not os.path.exists(RootOutputDirectoryPath):
        os.mkdir(RootOutputDirectoryPath) # os.makedir(RootOutputDirectoryPath)      
    os.chdir(RootOutputDirectoryPath)
    OutputDirectory = RootOutputDirectory + '/' + ProblemType
    OutputDirectoryPath = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(OutputDirectoryPath):
        os.mkdir(OutputDirectoryPath) # os.makedir(OutputDirectoryPath)   
    os.chdir(cwd)
    return RootOutputDirectory, OutputDirectory


def DetermineExactSolutions(myMPASOceanShallowWater,InterpolateExactVelocitiesFromEdgesToCellCenters):
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nCells = myMPASOceanShallowWater.myMesh.nCells
    myQuadratureOnHexagon = myMPASOceanShallowWater.myMesh.myQuadratureOnHexagon
    HexagonLength = myMPASOceanShallowWater.myMesh.HexagonLength
    time = myMPASOceanShallowWater.time
    if InterpolateExactVelocitiesFromEdgesToCellCenters:
        uExactEdge = np.zeros(nEdges)
        vExactEdge = np.zeros(nEdges)
    if InterpolateExactVelocitiesFromEdgesToCellCenters:
        for iEdge in range(0,nEdges):
            xEdge = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            yEdge = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            uExactEdge[iEdge] = (
            ESST.DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,xEdge,yEdge,time,
                                             UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
            vExactEdge[iEdge] = (
            ESST.DetermineExactMeridionalVelocity(ProblemType,myExactSolutionParameters,xEdge,yEdge,
                                                  time,UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge))
        myMPASOceanShallowWater.mySolution.uExact = (
        myMPASOceanShallowWater.myMesh.InterpolateSolutionFromEdgesToCellCenters(uExactEdge))
        myMPASOceanShallowWater.mySolution.vExact = (
        myMPASOceanShallowWater.myMesh.InterpolateSolutionFromEdgesToCellCenters(vExactEdge))
    else:
        for iCell in range(0,nCells):
            xCell = myMPASOceanShallowWater.myMesh.xCell[iCell]
            yCell = myMPASOceanShallowWater.myMesh.yCell[iCell]
            myMPASOceanShallowWater.mySolution.uExact[iCell] = (
            ESST.DetermineExactZonalVelocityAtCellCenter(ProblemType,myExactSolutionParameters,xCell,yCell,time,
                                                         UseAveragedQuantities,myQuadratureOnHexagon,HexagonLength))
            myMPASOceanShallowWater.mySolution.vExact[iCell] = (
            ESST.DetermineExactMeridionalVelocityAtCellCenter(ProblemType,myExactSolutionParameters,xCell,yCell,
                                                              time,UseAveragedQuantities,myQuadratureOnHexagon,
                                                              HexagonLength))
    for iCell in range(0,nCells):
        xCell = myMPASOceanShallowWater.myMesh.xCell[iCell]
        yCell = myMPASOceanShallowWater.myMesh.yCell[iCell]
        myMPASOceanShallowWater.mySolution.sshExact[iCell] = (
        ESST.DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,xCell,yCell,time,
                                            UseAveragedQuantities,myQuadratureOnHexagon,HexagonLength))
        

def SpecifyInitialConditions(myMPASOceanShallowWater):
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    myExactSolutionParameters = myMPASOceanShallowWater.myNameList.myExactSolutionParameters
    UseAveragedQuantities = myMPASOceanShallowWater.myMesh.UseAveragedQuantities
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    myQuadratureOnEdge = myMPASOceanShallowWater.myMesh.myQuadratureOnEdge
    nCells = myMPASOceanShallowWater.myMesh.nCells
    myQuadratureOnHexagon = myMPASOceanShallowWater.myMesh.myQuadratureOnHexagon
    HexagonLength = myMPASOceanShallowWater.myMesh.HexagonLength
    # Specify the initial conditions.
    for iEdge in range(0,nEdges):
        xEdge = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
        yEdge = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
        dvEdge = myMPASOceanShallowWater.myMesh.dvEdge[iEdge]
        angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
        myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = (
        ESST.DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,xEdge,yEdge,
                                          myMPASOceanShallowWater.time,UseAveragedQuantities,myQuadratureOnEdge,dvEdge,
                                          angleEdge))
    for iCell in range(0,nCells):
        xCell = myMPASOceanShallowWater.myMesh.xCell[iCell]
        yCell = myMPASOceanShallowWater.myMesh.yCell[iCell]
        myMPASOceanShallowWater.mySolution.ssh[iCell] = (
        ESST.DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,xCell,yCell,
                                            myMPASOceanShallowWater.time,UseAveragedQuantities,myQuadratureOnHexagon,
                                            HexagonLength))
                
                
def SpecifyRestartConditions(myMPASOceanShallowWater,normalVelocityRestart,sshRestart):
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    nCells = myMPASOceanShallowWater.myMesh.nCells
    for iEdge in range(0,nEdges):
        myMPASOceanShallowWater.mySolution.normalVelocity[iEdge] = normalVelocityRestart[iEdge]
    for iCell in range(0,nCells):
        myMPASOceanShallowWater.mySolution.ssh[iCell] = sshRestart[iCell]
        
        
def ExpressStateAtCellCentersAsOneMultiDimensionalArray(myMPASOceanShallowWater,State):
    nCells = myMPASOceanShallowWater.myMesh.nCells
    StateAsOneMultiDimensionalArray = np.zeros((nCells,3))
    for iCell in range(0,nCells):
        if State == 'Exact':
            StateAsOneMultiDimensionalArray[iCell,0] = myMPASOceanShallowWater.mySolution.uExact[iCell]
            StateAsOneMultiDimensionalArray[iCell,1] = myMPASOceanShallowWater.mySolution.vExact[iCell]
            StateAsOneMultiDimensionalArray[iCell,2] = myMPASOceanShallowWater.mySolution.sshExact[iCell]
        elif State == 'Numerical':
            StateAsOneMultiDimensionalArray[iCell,0] = myMPASOceanShallowWater.mySolution.u[iCell]
            StateAsOneMultiDimensionalArray[iCell,1] = myMPASOceanShallowWater.mySolution.v[iCell]
            StateAsOneMultiDimensionalArray[iCell,2] = myMPASOceanShallowWater.mySolution.ssh[iCell]     
        elif State == 'Error':
            StateAsOneMultiDimensionalArray[iCell,0] = myMPASOceanShallowWater.mySolution.uError[iCell]
            StateAsOneMultiDimensionalArray[iCell,1] = myMPASOceanShallowWater.mySolution.vError[iCell]
            StateAsOneMultiDimensionalArray[iCell,2] = myMPASOceanShallowWater.mySolution.sshError[iCell]
    return StateAsOneMultiDimensionalArray

        
def ComputeError(myMPASOceanShallowWater):
    myDiagnosticVariablesToCompute = DiagnosticVariablesToCompute()
    myDiagnosticVariablesToCompute.TangentialVelocity = True
    myMPASOceanShallowWater.DiagnosticSolve(myMPASOceanShallowWater.mySolution.normalVelocity,
                                            myMPASOceanShallowWater.mySolution.ssh,
                                            myDiagnosticVariablesToCompute,NormalVelocityTendencyComputation=False)
    uEdge = ESST.DetermineZonalComponentsFromNormalAndTangentialComponents(
    myMPASOceanShallowWater.mySolution.normalVelocity,myMPASOceanShallowWater.mySolution.tangentialVelocity,
    myMPASOceanShallowWater.myMesh.angleEdge)
    vEdge = ESST.DetermineMeridionalComponentsFromNormalAndTangentialComponents(
    myMPASOceanShallowWater.mySolution.normalVelocity,myMPASOceanShallowWater.mySolution.tangentialVelocity,
    myMPASOceanShallowWater.myMesh.angleEdge)
    myMPASOceanShallowWater.mySolution.u = (
    myMPASOceanShallowWater.myMesh.InterpolateSolutionFromEdgesToCellCenters(uEdge))
    myMPASOceanShallowWater.mySolution.v = (
    myMPASOceanShallowWater.myMesh.InterpolateSolutionFromEdgesToCellCenters(vEdge))
    myMPASOceanShallowWater.mySolution.uError[:] = (myMPASOceanShallowWater.mySolution.u[:] 
                                                    - myMPASOceanShallowWater.mySolution.uExact[:])
    myMPASOceanShallowWater.mySolution.vError[:] = (myMPASOceanShallowWater.mySolution.v[:] 
                                                    - myMPASOceanShallowWater.mySolution.vExact[:])
    myMPASOceanShallowWater.mySolution.sshError[:] = (myMPASOceanShallowWater.mySolution.ssh[:] 
                                                      - myMPASOceanShallowWater.mySolution.sshExact[:])
    
    
def ComputeErrorNorm(myMPASOceanShallowWater,ErrorToBeSpecified=False,SpecifiedError=[]):
    nCells = myMPASOceanShallowWater.myMesh.nCells
    L2ErrorNorm = np.zeros(3)
    for iCell in range(0,nCells):
        if ErrorToBeSpecified:
            L2ErrorNorm[:] += SpecifiedError[iCell,:]**2.0
        else:
            L2ErrorNorm[0] += myMPASOceanShallowWater.mySolution.uError[iCell]**2.0
            L2ErrorNorm[1] += myMPASOceanShallowWater.mySolution.vError[iCell]**2.0
            L2ErrorNorm[2] += myMPASOceanShallowWater.mySolution.sshError[iCell]**2.0
    L2ErrorNorm = np.sqrt(L2ErrorNorm/float(nCells))
    return L2ErrorNorm


def ComputeErrorNormOnCoarsestRectilinearMesh(nCellsOnCoarsestRectilinearMesh,mySolutionOnCoarsestRectilinearMesh):
    L2ErrorNorm = np.zeros(3)
    for iCell in range(0,nCellsOnCoarsestRectilinearMesh):
        L2ErrorNorm[:] += mySolutionOnCoarsestRectilinearMesh[iCell,:]**2.0
    L2ErrorNorm = np.sqrt(L2ErrorNorm/float(nCellsOnCoarsestRectilinearMesh))
    return L2ErrorNorm


def ComputeErrorNormOnCoarsestMesh(myCoarsestMesh,nCellsOnCoarsestMeshToBeConsidered,CellsOnCoarsestMeshToBeConsidered,
                                   mySolutionOnCoarsestMesh):
    L2ErrorNorm = np.zeros(3)
    for iCell in range(0,myCoarsestMesh.nCells):
        if CellsOnCoarsestMeshToBeConsidered[iCell] == 1.0:
            L2ErrorNorm[:] += mySolutionOnCoarsestMesh[iCell,:]**2.0
    L2ErrorNorm = np.sqrt(L2ErrorNorm/float(nCellsOnCoarsestMeshToBeConsidered))
    return L2ErrorNorm
    

def WriteStateMPASOceanShallowWater(myMPASOceanShallowWater,filename,ComputeOnlyExactSolution=False):
    nCells = myMPASOceanShallowWater.myMesh.nCells
    cwd = os.getcwd()
    path = cwd + '/' + myMPASOceanShallowWater.OutputDirectory + '/'
    os.chdir(path)
    filename += '.tec'
    outputfile = open(filename,'w')
    if ComputeOnlyExactSolution:
        outputfile.write('VARIABLES = "X", "Y", "uExact", "vExact", "sshExact"\n')       
    else:
        outputfile.write('VARIABLES = "X", "Y", "uExact", "vExact", "sshExact", "u", "v", "ssh", "uError", "vError", '
                         + '"sshError"\n')
    for iCell in range(0,nCells):
        xCell = myMPASOceanShallowWater.myMesh.xCell[iCell]
        yCell = myMPASOceanShallowWater.myMesh.yCell[iCell]
        uExact = myMPASOceanShallowWater.mySolution.uExact[iCell]
        vExact = myMPASOceanShallowWater.mySolution.vExact[iCell]
        sshExact = myMPASOceanShallowWater.mySolution.sshExact[iCell]
        u = myMPASOceanShallowWater.mySolution.u[iCell]
        v = myMPASOceanShallowWater.mySolution.v[iCell]
        ssh = myMPASOceanShallowWater.mySolution.ssh[iCell]
        uError = myMPASOceanShallowWater.mySolution.uError[iCell]
        vError = myMPASOceanShallowWater.mySolution.vError[iCell]
        sshError = myMPASOceanShallowWater.mySolution.sshError[iCell]
        ZoneIDString = 'Element' + '%7.7d' %(iCell + 1)
        outputfile.write('ZONE T="%s", I=1, J=1, F=BLOCK\n' %ZoneIDString)
        if ComputeOnlyExactSolution:
            outputfile.write('%.15g %.15g %.15g %.15g %.15g\n' %(xCell,yCell,uExact,vExact,sshExact))
        else:
            outputfile.write('%.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n' 
                             %(xCell,yCell,uExact,vExact,sshExact,u,v,ssh,uError,vError,sshError))
    outputfile.close()
    os.chdir(cwd)
    
    
def WriteRestartStateMPASOceanShallowWater(myMPASOceanShallowWater,filename_normalVelocity,filename_ssh):
    cwd = os.getcwd()
    path = cwd + '/' + myMPASOceanShallowWater.OutputDirectory + '/'
    os.chdir(path)
    filename_normalVelocity += '.tec'
    outputfile = open(filename_normalVelocity,'w')
    outputfile.write('VARIABLES = "X", "Y", "normalVelocity"\n')       
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    for iEdge in range(0,nEdges):
        xEdge = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
        yEdge = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
        normalVelocity = myMPASOceanShallowWater.mySolution.normalVelocity[iEdge]
        ZoneIDString = 'Element' + '%7.7d' %(iEdge + 1)
        outputfile.write('ZONE T="%s", I=1, J=1, F=BLOCK\n' %ZoneIDString)
        outputfile.write('%.15g %.15g %.15g\n' %(xEdge,yEdge,normalVelocity))
    outputfile.close()
    filename_ssh += '.tec'
    outputfile = open(filename_ssh,'w')
    outputfile.write('VARIABLES = "X", "Y", "ssh"\n')       
    nCells = myMPASOceanShallowWater.myMesh.nCells
    for iCell in range(0,nCells):
        xCell = myMPASOceanShallowWater.myMesh.xCell[iCell]
        yCell = myMPASOceanShallowWater.myMesh.yCell[iCell]
        ssh = myMPASOceanShallowWater.mySolution.ssh[iCell]
        ZoneIDString = 'Element' + '%7.7d' %(iCell + 1)
        outputfile.write('ZONE T="%s", I=1, J=1, F=BLOCK\n' %ZoneIDString)
        outputfile.write('%.15g %.15g %.15g\n' %(xCell,yCell,ssh))
    outputfile.close()
    os.chdir(cwd)
    
    
def ReadStateMPASOceanShallowWater(myMPASOceanShallowWater,filename_normalVelocity,filename_ssh):
    cwd = os.getcwd()
    path = cwd + '/' + myMPASOceanShallowWater.OutputDirectory + '/'
    os.chdir(path)
    data = []
    count = 0
    with open(filename_normalVelocity,'r') as infile:
        for line in infile:
            if count != 0 and count % 2 == 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    nEdges = myMPASOceanShallowWater.myMesh.nEdges
    xEdge = np.zeros(nEdges)
    yEdge = np.zeros(nEdges)
    normalVelocity = np.zeros(nEdges)
    for iEdge in range(0,nEdges):
        xEdge[iEdge] = data[iEdge,0]
        yEdge[iEdge] = data[iEdge,1]
        normalVelocity[iEdge] = data[iEdge,2]
    data = []
    count = 0
    with open(filename_ssh,'r') as infile:
        for line in infile:
            if count != 0 and count % 2 == 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    nCells = myMPASOceanShallowWater.myMesh.nCells
    xCell = np.zeros(nCells)
    yCell = np.zeros(nCells)
    ssh = np.zeros(nCells)
    for iCell in range(0,nCells):
        xCell[iCell] = data[iCell,0]
        yCell[iCell] = data[iCell,1]
        ssh[iCell] = data[iCell,2]
    os.chdir(cwd)
    return normalVelocity, ssh

    
def FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,phi,useGivenColorBarLimits,ColorBarLimits,
                                                nColorBarTicks,colormap,colorbarfontsize,labels,labelfontsizes,
                                                labelpads,tickfontsizes,title,titlefontsize,SaveAsPDF,FileName,Show,
                                                fig_size=[10.0,10.0],cbarlabelformat='%.2g',FigureFormat='pdf',
                                                specify_n_ticks=False,n_ticks=[0,0]):
    cwd = os.getcwd()
    path = cwd + '/' + myMPASOceanShallowWater.OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    set_aspect_equal = False
    if set_aspect_equal:
        ax.set_aspect('equal')
    else:
        xMin = 0.0 
        xMax = myMPASOceanShallowWater.myMesh.lX + myMPASOceanShallowWater.myMesh.dx/2.0
        yMin = 0.0 
        yMax = myMPASOceanShallowWater.myMesh.lY + myMPASOceanShallowWater.myMesh.dx/(2.0*np.sqrt(3.0))   
        aspect_ratio = (xMax - xMin)/(yMax - yMin)
        ax.set_aspect(aspect_ratio,adjustable='box')
    if useGivenColorBarLimits:
        cbar_min = ColorBarLimits[0]
        cbar_max = ColorBarLimits[1]
    else:
        cbar_min = np.min(phi)
        cbar_max = np.max(phi)
    n_cbar_ticks = nColorBarTicks
    cbarlabels = np.linspace(cbar_min,cbar_max,num=n_cbar_ticks,endpoint=True)
    patches = []
    ComparisonTolerance = 10.0**(-5.0)
    for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
        nVerticesOnCell = myMPASOceanShallowWater.myMesh.nEdgesOnCell[iCell] 
        vertexIndices = np.zeros(nVerticesOnCell,dtype=int)
        vertexIndices[:] = myMPASOceanShallowWater.myMesh.verticesOnCell[iCell,:]
        vertexIndices -= 1
        vertices = np.zeros((nVerticesOnCell,2))
        xCell = myMPASOceanShallowWater.myMesh.xCell[iCell]
        yCell = myMPASOceanShallowWater.myMesh.yCell[iCell]
        for iVertexOnCell in range(0,nVerticesOnCell):
            xVertex = myMPASOceanShallowWater.myMesh.xVertex[vertexIndices[iVertexOnCell]]
            yVertex = myMPASOceanShallowWater.myMesh.yVertex[vertexIndices[iVertexOnCell]]
            if abs(yVertex - yCell) > (2.0/np.sqrt(3.0))*myMPASOceanShallowWater.myMesh.dx and yVertex < yCell:
                yVertex = yCell + myMPASOceanShallowWater.myMesh.dx/np.sqrt(3.0)  
            if abs(yVertex - yCell) > (2.0/np.sqrt(3.0))*myMPASOceanShallowWater.myMesh.dx and yVertex > yCell:
                yVertex = yCell - myMPASOceanShallowWater.myMesh.dx/np.sqrt(3.0)                 
            if abs(xVertex - xCell) > myMPASOceanShallowWater.myMesh.dx and xVertex < xCell:
                if abs(yVertex - (yCell + myMPASOceanShallowWater.myMesh.dx/np.sqrt(3.0))) < ComparisonTolerance:
                    xVertex = xCell
                elif abs(yVertex - (yCell - myMPASOceanShallowWater.myMesh.dx/np.sqrt(3.0))) < ComparisonTolerance:
                    xVertex = xCell
                else:                
                    xVertex = xCell + 0.5*myMPASOceanShallowWater.myMesh.dx
            vertices[iVertexOnCell,0] = xVertex
            vertices[iVertexOnCell,1] = yVertex
        polygon = Polygon(vertices,True)
        patches.append(polygon)    
    localPatches = PatchCollection(patches,cmap=colormap,alpha=1.0) 
    localPatches.set_array(phi)
    ax.add_collection(localPatches)
    localPatches.set_clim([cbar_min,cbar_max])
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    ProblemType_EquatorialWave = myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave
    if ProblemType_EquatorialWave:
        yMin = -0.5*yMax
        yMax *= 0.5
    plt.axis([xMin,xMax,yMin,yMax])
    plt.title(title,fontsize=titlefontsize,fontweight='bold',y=1.035)
    cbarShrinkRatio = 0.825
    m = plt.cm.ScalarMappable(cmap=colormap)
    m.set_array(phi)
    m.set_clim(cbar_min,cbar_max)
    make_colorbar_boundaries_discrete = False
    if make_colorbar_boundaries_discrete:
        cbar = plt.colorbar(m,boundaries=cbarlabels,shrink=cbarShrinkRatio)
    else:
        cbar = plt.colorbar(m,shrink=cbarShrinkRatio)
    cbar.set_ticks(cbarlabels)
    cbar.set_ticklabels(cbarlabels)
    cbar.ax.set_yticklabels([cbarlabelformat %x for x in cbarlabels], fontsize=colorbarfontsize)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    plt.xticks(fontsize=tickfontsizes[0])
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/1000.0), '')))
    plt.yticks(fontsize=tickfontsizes[1])
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda y, p: format(int(y/1000.0), '')))
    if (ProblemType_EquatorialWave or ProblemType == 'Inertia_Gravity_Wave' 
        or ProblemType == 'NonLinear_Manufactured_Solution'):
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
        ax.yaxis.set_major_locator(plt.MaxNLocator(7))
    if specify_n_ticks:
        n_xticks = n_ticks[0]
        n_yticks = n_ticks[1]
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_xticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(n_yticks))
    if SaveAsPDF:
        plt.savefig(FileName+'.'+FigureFormat,format=FigureFormat,bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)
    

def PythonPlotStateMPASOceanShallowWater(myMPASOceanShallowWater,filename,DisplayTime,UseGivenColorBarLimits=True,
                                         ComputeOnlyExactSolution=False,PlotNumericalSolution=False,
                                         PlotOnMPASOceanMesh=True):
    ProblemType_NoExactSolution = myMPASOceanShallowWater.myNameList.ProblemType_NoExactSolution
    ProblemType_EquatorialWave = myMPASOceanShallowWater.myNameList.ProblemType_EquatorialWave
    ProblemType = myMPASOceanShallowWater.myNameList.ProblemType
    PlotZonalVelocity = myMPASOceanShallowWater.myNameList.LogicalArrayPlot[0]
    PlotMeridionalVelocity = myMPASOceanShallowWater.myNameList.LogicalArrayPlot[1]
    PlotSurfaceElevation = myMPASOceanShallowWater.myNameList.LogicalArrayPlot[2]
    cwd = os.getcwd()
    path = cwd + '/' + myMPASOceanShallowWater.OutputDirectory + '/'
    os.chdir(path)
    data = []
    count = 0
    with open(filename,'r') as infile:
        for line in infile:
            if count != 0 and count % 2 == 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    nCells = data.shape[0]
    x = np.zeros(nCells)
    y = np.zeros(nCells)
    uExact = np.zeros(nCells)
    vExact = np.zeros(nCells)
    sshExact = np.zeros(nCells)
    if not(ComputeOnlyExactSolution):
        if PlotNumericalSolution:
            u = np.zeros(nCells)
            v = np.zeros(nCells)
            ssh = np.zeros(nCells)
        uError = np.zeros(nCells)
        vError = np.zeros(nCells)
        sshError = np.zeros(nCells)
    for iCell in range(0,nCells):
        x[iCell] = data[iCell,0]                        
        y[iCell] = data[iCell,1]
        uExact[iCell] = data[iCell,2]
        vExact[iCell] = data[iCell,3]
        sshExact[iCell] = data[iCell,4]
        if not(ComputeOnlyExactSolution):
            if PlotNumericalSolution:
                u[iCell] = data[iCell,5]
                v[iCell] = data[iCell,6]
                ssh[iCell] = data[iCell,7]
            uError[iCell] = data[iCell,8]
            vError[iCell] = data[iCell,9]
            sshError[iCell] = data[iCell,10]            
    os.chdir(cwd)
    titleroot = myMPASOceanShallowWater.myNameList.ProblemType_Title
    PlotFileNameRoot = myMPASOceanShallowWater.myNameList.ProblemType_FileName
    TimeIntegratorShortForm = myMPASOceanShallowWater.myNameList.myTimeSteppingParameters.TimeIntegratorShortForm
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
    DataType = 'Unstructured'
    colormap = plt.cm.seismic
    colorbarfontsize = 15.0
    iTimeFormat = '%3.3d' 
    if ProblemType_EquatorialWave:
        specify_n_ticks = True
        n_ticks = [6,6]
    else:
        specify_n_ticks = False
        n_ticks = [0,0]
    if PlotZonalVelocity:
        if UseGivenColorBarLimits:
            FileName = myMPASOceanShallowWater.myNameList.ProblemType_FileName + '_ExactZonalVelocityLimits'
            ExactZonalVelocityLimits = CR.ReadStateVariableLimitsFromFile(myMPASOceanShallowWater.OutputDirectory,
                                                                          FileName+'.curve')
        else:
            ExactZonalVelocityLimits = [0.0,0.0]
        if not(ProblemType_NoExactSolution):
            title = titleroot + ':\nExact Zonal Velocity after\n' + DisplayTime
            PlotFileName = PlotFileNameRoot + '_ExactZonalVelocity_' + iTimeFormat %myMPASOceanShallowWater.iTime
            if PlotOnMPASOceanMesh:
                FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,uExact,UseGivenColorBarLimits,
                                                            ExactZonalVelocityLimits,nColorBarTicks,colormap,
                                                            colorbarfontsize,labels,labelfontsizes,labelpads,
                                                            tickfontsizes,title,titlefontsize,SaveAsPDF,PlotFileName,
                                                            Show,specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
            else:
                CR.PythonFilledContourPlot2DSaveAsPDF(myMPASOceanShallowWater.OutputDirectory,x,y,uExact,nContours,
                                                      labels,labelfontsizes,labelpads,tickfontsizes,
                                                      UseGivenColorBarLimits,ExactZonalVelocityLimits,nColorBarTicks,
                                                      title,titlefontsize,SaveAsPDF,PlotFileName,Show,DataType=DataType,
                                                      colormap=colormap,specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
        if not(ComputeOnlyExactSolution):
            if PlotNumericalSolution:
                title = titleroot + ':\nNumerical Zonal Velocity after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_NumericalZonalVelocity_' + iTimeFormat %myMPASOceanShallowWater.iTime)
                if PlotOnMPASOceanMesh:
                    FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,u,UseGivenColorBarLimits,
                                                                ExactZonalVelocityLimits,nColorBarTicks,colormap,
                                                                colorbarfontsize,labels,labelfontsizes,labelpads,
                                                                tickfontsizes,title,titlefontsize,SaveAsPDF,
                                                                PlotFileName,Show,specify_n_ticks=specify_n_ticks,
                                                                n_ticks=n_ticks)
                else:
                    CR.PythonFilledContourPlot2DSaveAsPDF(myMPASOceanShallowWater.OutputDirectory,x,y,u,nContours,
                                                          labels,labelfontsizes,labelpads,tickfontsizes,
                                                          UseGivenColorBarLimits,ExactZonalVelocityLimits,
                                                          nColorBarTicks,title,titlefontsize,SaveAsPDF,PlotFileName,
                                                          Show,DataType=DataType,colormap=colormap,
                                                          specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
            if not(ProblemType_NoExactSolution):
                if UseGivenColorBarLimits:
                    FileName = (myMPASOceanShallowWater.myNameList.ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                + '_ZonalVelocityErrorLimits')
                    ZonalVelocityErrorLimits = (
                    CR.ReadStateVariableLimitsFromFile(myMPASOceanShallowWater.OutputDirectory,FileName+'.curve'))
                else:
                    ZonalVelocityErrorLimits = [0.0,0.0]
                title = titleroot + ':\nZonal Velocity Error after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_ZonalVelocityError_' + iTimeFormat %myMPASOceanShallowWater.iTime)
                if PlotOnMPASOceanMesh:
                    FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,uError,UseGivenColorBarLimits,
                                                                ZonalVelocityErrorLimits,nColorBarTicks,colormap,
                                                                colorbarfontsize,labels,labelfontsizes,labelpads,
                                                                tickfontsizes,title,titlefontsize,SaveAsPDF,
                                                                PlotFileName,Show,specify_n_ticks=specify_n_ticks,
                                                                n_ticks=n_ticks)
                else:
                    CR.PythonFilledContourPlot2DSaveAsPDF(myMPASOceanShallowWater.OutputDirectory,x,y,uError,nContours,
                                                          labels,labelfontsizes,labelpads,tickfontsizes,
                                                          UseGivenColorBarLimits,ZonalVelocityErrorLimits,
                                                          nColorBarTicks,title,titlefontsize,SaveAsPDF,PlotFileName,
                                                          Show,DataType=DataType,colormap=colormap,
                                                          specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
    if PlotMeridionalVelocity:
        if UseGivenColorBarLimits:
            FileName = myMPASOceanShallowWater.myNameList.ProblemType_FileName + '_ExactMeridionalVelocityLimits'
            ExactMeridionalVelocityLimits = CR.ReadStateVariableLimitsFromFile(myMPASOceanShallowWater.OutputDirectory,
                                                                               FileName+'.curve')
        else:
            ExactMeridionalVelocityLimits = [0.0,0.0]
        if not(ProblemType_NoExactSolution):
            title = titleroot + ':\nExact Meridional Velocity after\n' + DisplayTime
            PlotFileName = PlotFileNameRoot + '_ExactMeridionalVelocity_' + iTimeFormat %myMPASOceanShallowWater.iTime
            if PlotOnMPASOceanMesh:
                FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,vExact,UseGivenColorBarLimits,
                                                            ExactMeridionalVelocityLimits,nColorBarTicks,colormap,
                                                            colorbarfontsize,labels,labelfontsizes,labelpads,
                                                            tickfontsizes,title,titlefontsize,SaveAsPDF,PlotFileName,
                                                            Show,specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
            else:
                CR.PythonFilledContourPlot2DSaveAsPDF(myMPASOceanShallowWater.OutputDirectory,x,y,vExact,nContours,
                                                      labels,labelfontsizes,labelpads,tickfontsizes,
                                                      UseGivenColorBarLimits,ExactMeridionalVelocityLimits,
                                                      nColorBarTicks,title,titlefontsize,SaveAsPDF,PlotFileName,Show,
                                                      DataType=DataType,colormap=colormap,
                                                      specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
        if not(ComputeOnlyExactSolution):
            if PlotNumericalSolution:
                title = titleroot + ':\nNumerical Meridional Velocity after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_NumericalMeridionalVelocity_' + iTimeFormat %myMPASOceanShallowWater.iTime)
                if PlotOnMPASOceanMesh:
                    FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,v,UseGivenColorBarLimits,
                                                                ExactMeridionalVelocityLimits,nColorBarTicks,colormap,
                                                                colorbarfontsize,labels,labelfontsizes,labelpads,
                                                                tickfontsizes,title,titlefontsize,SaveAsPDF,
                                                                PlotFileName,Show,specify_n_ticks=specify_n_ticks,
                                                                n_ticks=n_ticks)
                else:
                    CR.PythonFilledContourPlot2DSaveAsPDF(myMPASOceanShallowWater.OutputDirectory,x,y,v,nContours,
                                                          labels,labelfontsizes,labelpads,tickfontsizes,
                                                          UseGivenColorBarLimits,ExactMeridionalVelocityLimits,
                                                          nColorBarTicks,title,titlefontsize,SaveAsPDF,PlotFileName,
                                                          Show,DataType=DataType,colormap=colormap,
                                                          specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
            if not(ProblemType_NoExactSolution):
                if UseGivenColorBarLimits:
                    FileName = (myMPASOceanShallowWater.myNameList.ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                + '_MeridionalVelocityErrorLimits')
                    MeridionalVelocityErrorLimits = (
                    CR.ReadStateVariableLimitsFromFile(myMPASOceanShallowWater.OutputDirectory,FileName+'.curve'))
                else:
                    MeridionalVelocityErrorLimits = [0.0,0.0]
                title = titleroot + ':\nMeridional Velocity Error after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_MeridionalVelocityError_' + iTimeFormat %myMPASOceanShallowWater.iTime)
                if PlotOnMPASOceanMesh:
                    FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,vError,UseGivenColorBarLimits,
                                                                MeridionalVelocityErrorLimits,nColorBarTicks,colormap,
                                                                colorbarfontsize,labels,labelfontsizes,labelpads,
                                                                tickfontsizes,title,titlefontsize,SaveAsPDF,
                                                                PlotFileName,Show,specify_n_ticks=specify_n_ticks,
                                                                n_ticks=n_ticks)
                else:
                    CR.PythonFilledContourPlot2DSaveAsPDF(myMPASOceanShallowWater.OutputDirectory,x,y,vError,nContours,
                                                          labels,labelfontsizes,labelpads,tickfontsizes,
                                                          UseGivenColorBarLimits,MeridionalVelocityErrorLimits,
                                                          nColorBarTicks,title,titlefontsize,SaveAsPDF,PlotFileName,
                                                          Show,DataType=DataType,colormap=colormap,
                                                          specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
    if PlotSurfaceElevation:
        if UseGivenColorBarLimits:
            FileName = myMPASOceanShallowWater.myNameList.ProblemType_FileName + '_ExactSurfaceElevationLimits'
            ExactSurfaceElevationLimits = CR.ReadStateVariableLimitsFromFile(myMPASOceanShallowWater.OutputDirectory,
                                                                             FileName+'.curve')
        else:
            ExactSurfaceElevationLimits = [0.0,0.0]
        if not(ProblemType_NoExactSolution):
            title = titleroot + ':\nExact Surface Elevation after\n' + DisplayTime
            PlotFileName = PlotFileNameRoot + '_ExactSurfaceElevation_' + iTimeFormat %myMPASOceanShallowWater.iTime
            if PlotOnMPASOceanMesh:
                FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,sshExact,UseGivenColorBarLimits,
                                                            ExactSurfaceElevationLimits,nColorBarTicks,colormap,
                                                            colorbarfontsize,labels,labelfontsizes,labelpads,
                                                            tickfontsizes,title,titlefontsize,SaveAsPDF,PlotFileName,
                                                            Show,specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
            else:
                CR.PythonFilledContourPlot2DSaveAsPDF(myMPASOceanShallowWater.OutputDirectory,x,y,sshExact,nContours,
                                                      labels,labelfontsizes,labelpads,tickfontsizes,
                                                      UseGivenColorBarLimits,ExactSurfaceElevationLimits,
                                                      nColorBarTicks,title,titlefontsize,SaveAsPDF,PlotFileName,Show,
                                                      DataType=DataType,colormap=colormap,
                                                      specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
        if not(ComputeOnlyExactSolution):
            if PlotNumericalSolution:
                title = titleroot + ':\nNumerical Surface Elevation after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_NumericalSurfaceElevation_' + iTimeFormat %myMPASOceanShallowWater.iTime)
                if PlotOnMPASOceanMesh:
                    FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,ssh,UseGivenColorBarLimits,
                                                                ExactSurfaceElevationLimits,nColorBarTicks,colormap,
                                                                colorbarfontsize,labels,labelfontsizes,labelpads,
                                                                tickfontsizes,title,titlefontsize,SaveAsPDF,
                                                                PlotFileName,Show,specify_n_ticks=specify_n_ticks,
                                                                n_ticks=n_ticks)
                else:
                    CR.PythonFilledContourPlot2DSaveAsPDF(myMPASOceanShallowWater.OutputDirectory,x,y,ssh,nContours,
                                                          labels,labelfontsizes,labelpads,tickfontsizes,
                                                          UseGivenColorBarLimits,ExactSurfaceElevationLimits,
                                                          nColorBarTicks,title,titlefontsize,SaveAsPDF,PlotFileName,
                                                          Show,DataType=DataType,colormap=colormap,
                                                          specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)
            if not(ProblemType_NoExactSolution):
                if UseGivenColorBarLimits:
                    FileName = (myMPASOceanShallowWater.myNameList.ProblemType_FileName + '_' + TimeIntegratorShortForm 
                                + '_SurfaceElevationErrorLimits')
                    SurfaceElevationErrorLimits = (
                    CR.ReadStateVariableLimitsFromFile(myMPASOceanShallowWater.OutputDirectory,FileName+'.curve'))
                else:
                    SurfaceElevationErrorLimits = [0.0,0.0]
                title = titleroot + ':\nSurface Elevation Error after\n' + DisplayTime
                PlotFileName = (PlotFileNameRoot + '_' + TimeIntegratorShortForm 
                                + '_SurfaceElevationError_' + iTimeFormat %myMPASOceanShallowWater.iTime)
                if PlotOnMPASOceanMesh:
                    FilledContourPlot2DSaveAsPDFOnMPASOceanMesh(myMPASOceanShallowWater,sshError,UseGivenColorBarLimits,
                                                                SurfaceElevationErrorLimits,nColorBarTicks,colormap,
                                                                colorbarfontsize,labels,labelfontsizes,labelpads,
                                                                tickfontsizes,title,titlefontsize,SaveAsPDF,
                                                                PlotFileName,Show,specify_n_ticks=specify_n_ticks,
                                                                n_ticks=n_ticks)
                else:
                    CR.PythonFilledContourPlot2DSaveAsPDF(myMPASOceanShallowWater.OutputDirectory,x,y,sshError,
                                                          nContours,labels,labelfontsizes,labelpads,tickfontsizes,
                                                          UseGivenColorBarLimits,SurfaceElevationErrorLimits,
                                                          nColorBarTicks,title,titlefontsize,SaveAsPDF,PlotFileName,
                                                          Show,DataType=DataType,colormap=colormap,
                                                          specify_n_ticks=specify_n_ticks,n_ticks=n_ticks)