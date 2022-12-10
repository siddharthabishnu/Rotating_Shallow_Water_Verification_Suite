"""
Name: Test_SpatialOperators.py
Author: Sid Bishnu
Details: As the name implies, this script tests the various spatial operators of the TRiSK-based mimetic finite volume 
method computed in ../../src/MPAS_Ocean_Shallow_Water/SpatialOperators.py against their exact counterparts using smooth 
two-dimensional functions.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/MPAS_Ocean_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR
    import SpatialOperators as SO
    import MPASOceanShallowWaterClass
    

def SpecifyInitializationParameters(ConvergenceStudy=False,nCellsX=0,nCellsY=0):
    PrintPhaseSpeedOfWaveModes = False
    PrintAmplitudesOfWaveModes = False
    TimeIntegrator = 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
    LF_TR_and_LF_AM3_with_FB_Feedback_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    Generalized_FB_with_AB2_AM3_Step_Type = 'ThirdOrderAccurate_WideStabilityRange'
    Generalized_FB_with_AB3_AM4_Step_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    CourantNumber = 0.5
    UseCourantNumberToDetermineTimeStep = True
    PrintBasicGeometry = False
    FixAngleEdge = True
    PrintOutput = False
    UseAveragedQuantities = False
    SpecifyBoundaryCondition = True
    ReadDomainExtentsfromMeshFile = True
    DebugVersion = False
    if ConvergenceStudy:
        MeshDirectoryRoot = (
        '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_Convergence_Study')
    else:
        MeshDirectoryRoot = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_50x50_Cells'
    ProblemTypes = ['Inertia_Gravity_Wave','Coastal_Kelvin_Wave','Planetary_Rossby_Wave',
                    'NonLinear_Manufactured_Solution']
    if ConvergenceStudy:
        BaseMeshFileName = 'base_mesh_%dx%d.nc' %(nCellsX,nCellsY)
        CulledMeshFileName = 'culled_mesh_%dx%d.nc' %(nCellsX,nCellsY)
        BaseMeshFileNames = [BaseMeshFileName,CulledMeshFileName,CulledMeshFileName,CulledMeshFileName]
    else:
        BaseMeshFileNames = ['base_mesh_Periodic.nc','culled_mesh_NonPeriodic_x.nc','culled_mesh_NonPeriodic_y.nc',
                             'culled_mesh_NonPeriodic_xy.nc']
    if ConvergenceStudy:
        MeshFileName = 'mesh_%dx%d.nc' %(nCellsX,nCellsY)
        MeshFileNames = [MeshFileName]*len(ProblemTypes)
    else:
        MeshFileNames = ['mesh_Periodic.nc','mesh_NonPeriodic_x.nc','mesh_NonPeriodic_y.nc','mesh_NonPeriodic_xy.nc']
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    if not(ConvergenceStudy):
        nCellsX = 50
        nCellsY = nCellsX
    return [PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
            Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,CourantNumber,
            UseCourantNumberToDetermineTimeStep,PrintBasicGeometry,FixAngleEdge,PrintOutput,UseAveragedQuantities,
            SpecifyBoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion,MeshDirectoryRoot,ProblemTypes,
            BaseMeshFileNames,MeshFileNames,BoundaryConditions,nCellsX,nCellsY]
    

def SpecifyPlotParameters(MeshDirectory):
    OutputDirectory = MeshDirectory
    nContours = 300
    useGivenColorBarLimits = False
    ColorBarLimits = [0.0,0.0]
    nColorBarTicks = 6
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    labels = [xLabel,yLabel]
    labelfontsizes = [22.5,22.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    titlefontsize = 27.5
    SaveAsPDF = True
    Show = False
    return [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,labels,labelfontsizes,
            labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show]
    

def SpecifyConvergencePlotParameters(MeshDirectory,SpatialOperator,PlotAgainstNumberOfCellsInZonalDirection,slope,
                                     ErrorNormType):
    OutputDirectory = MeshDirectory
    PlotType = 'log-log'
    linewidths = [2.0,2.0]
    linestyles = [' ','-']
    colors = ['k','k']
    markers = [True,False]
    markertypes = ['s','s']
    markersizes = [10.0,10.0]
    if PlotAgainstNumberOfCellsInZonalDirection:
        xLabel = 'Number of cells in zonal direction'
    else:
        xLabel = 'Cell width'
    if SpatialOperator == 'Gradient':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of gradient operator'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of gradient operator'
        title = 'Convergence of Gradient Operator\nNormal to Edges'
        FileName = 'ConvergencePlot_GradientOperator_%sErrorNorm' %ErrorNormType
    elif SpatialOperator == 'Divergence':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of divergence operator'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of divergence operator'
        title = 'Convergence of Divergence Operator\nat Cell Centers'    
        FileName = 'ConvergencePlot_DivergenceOperator_%sErrorNorm' %ErrorNormType
    elif SpatialOperator == 'CurlAtVertices':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of curl operator'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of curl operator'
        title = 'Convergence of Curl Operator\nat Vertices'    
        FileName = 'ConvergencePlot_CurlOperatorAtVertices_%sErrorNorm' %ErrorNormType
    elif SpatialOperator == 'CurlAtCellCenters':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of curl operator'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of curl operator'
        title = 'Convergence of Curl Operator\nInterpolated to Cell Centers'    
        FileName = 'ConvergencePlot_CurlOperatorAtCellCenters_%sErrorNorm' %ErrorNormType
    elif SpatialOperator == 'TangentialVelocity':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of tangential velocity'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of tangential velocity'
        title = 'Convergence of Tangential Velocity\nalong Edges'
        FileName = 'ConvergencePlot_TangentialVelocity_%sErrorNorm' %ErrorNormType
    labels = [xLabel,yLabel]
    labelfontsizes = [22.5,22.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    if ErrorNormType == 'Max':
        legends = ['Maximum error norm','Best fit line: slope is %.2f' %slope]
    elif ErrorNormType == 'L2':
        legends = ['L$^2$ error norm','Best fit line: slope is %.2f' %slope]
    legendfontsize = 22.5
    if PlotAgainstNumberOfCellsInZonalDirection:
        legendposition = 'upper right'
    else:
        legendposition = 'upper left'
    titlefontsize = 27.5
    SaveAsPDF = True
    Show = False
    FigureSize = [9.25,9.25]
    drawMajorGrid = True
    drawMinorGrid = True
    legendWithinBox = True
    return [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
            FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox]
    

def TestSurfaceElevationNormalVelocity(PlotFigures=True,PlotNormalVelocity=True):
    [PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
     Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,CourantNumber,
     UseCourantNumberToDetermineTimeStep,PrintBasicGeometry,FixAngleEdge,PrintOutput,UseAveragedQuantities,
     SpecifyBoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion,MeshDirectoryRoot,ProblemTypes,
     BaseMeshFileNames,MeshFileNames,BoundaryConditions,nCellsX,nCellsY] = SpecifyInitializationParameters()
    for iProblemType in range(0,len(ProblemTypes)):
        ProblemType = ProblemTypes[iProblemType]
        BaseMeshFileName = BaseMeshFileNames[iProblemType]
        MeshFileName = MeshFileNames[iProblemType]
        BoundaryCondition = BoundaryConditions[iProblemType]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
        ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
        LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
        Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,
        MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,CourantNumber,UseCourantNumberToDetermineTimeStep,
        SpecifyBoundaryCondition,BoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion)
        lX = myMPASOceanShallowWater.myMesh.lX
        lY = myMPASOceanShallowWater.myMesh.lY
        if BoundaryCondition == 'NonPeriodic_x':
            iEdgeStartingIndex = 1
        else:
            iEdgeStartingIndex = 0
        prefix = SO.ProblemSpecificPrefix()
        mySurfaceElevation = np.zeros(myMPASOceanShallowWater.myMesh.nCells)
        for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
            mySurfaceElevation[iCell] = (
            SO.SurfaceElevation(lX,lY,myMPASOceanShallowWater.myMesh.xCell[iCell],
                                myMPASOceanShallowWater.myMesh.yCell[iCell]))
        myZonalVelocity = np.zeros(myMPASOceanShallowWater.myMesh.nEdges) 
        myMeridionalVelocity = np.zeros(myMPASOceanShallowWater.myMesh.nEdges) 
        myResultantVelocity = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)   
        myNormalVelocity = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            u, v = SO.Velocity(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],
                               myMPASOceanShallowWater.myMesh.yEdge[iEdge])
            myZonalVelocity[iEdge] = u
            myMeridionalVelocity[iEdge] = v
            myResultantVelocity[iEdge] = np.sqrt(u**2.0 + v**2.0)
            myNormalVelocity[iEdge] = (u*np.cos(myMPASOceanShallowWater.myMesh.angleEdge[iEdge]) 
                                       + v*np.sin(myMPASOceanShallowWater.myMesh.angleEdge[iEdge]))
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,labels,labelfontsizes,
             labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show] = SpecifyPlotParameters(MeshDirectory)
            Title = 'Surface Elevation'
            FileName = prefix + 'SurfaceElevation'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            mySurfaceElevation,nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
            ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured')
            Title = 'Zonal Velocity'
            FileName = prefix + 'ZonalVelocity'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,myZonalVelocity[iEdgeStartingIndex:],
            nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
            nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured')
            Title = 'Meridional Velocity'
            FileName = prefix + 'MeridionalVelocity'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,myMeridionalVelocity[iEdgeStartingIndex:],
            nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
            nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured')
            Title = 'Resultant Velocity'
            FileName = prefix + 'ResultantVelocity'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,myResultantVelocity[iEdgeStartingIndex:],
            nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
            nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured')
            if PlotNormalVelocity:
                Title = 'Normal Velocity'
                FileName = prefix + 'NormalVelocity'
                CR.PythonFilledContourPlot2DSaveAsPDF(
                OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
                myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,myNormalVelocity[iEdgeStartingIndex:],
                nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
                nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured')
                
                
do_TestSurfaceElevationNormalVelocity = False
if do_TestSurfaceElevationNormalVelocity:
    TestSurfaceElevationNormalVelocity()
    
    
def TestNumericalGradientOperator(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
    [PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
     Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,CourantNumber,
     UseCourantNumberToDetermineTimeStep,PrintBasicGeometry,FixAngleEdge,PrintOutput,UseAveragedQuantities,
     SpecifyBoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion,MeshDirectoryRoot,ProblemTypes,
     BaseMeshFileNames,MeshFileNames,BoundaryConditions,nCellsX,nCellsY] = (
    SpecifyInitializationParameters(ConvergenceStudy,nCellsX,nCellsY))
    MaxErrorNorm = np.zeros(4)
    L2ErrorNorm = np.zeros(4)
    for iProblemType in range(0,len(ProblemTypes)):
        ProblemType = ProblemTypes[iProblemType]
        BaseMeshFileName = BaseMeshFileNames[iProblemType]
        MeshFileName = MeshFileNames[iProblemType]
        BoundaryCondition = BoundaryConditions[iProblemType]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
        ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
        LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
        Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,
        MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,CourantNumber,UseCourantNumberToDetermineTimeStep,
        SpecifyBoundaryCondition,BoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion)
        lX = myMPASOceanShallowWater.myMesh.lX
        lY = myMPASOceanShallowWater.myMesh.lY
        dx = myMPASOceanShallowWater.myMesh.dx
        if BoundaryCondition == 'NonPeriodic_x':
            iEdgeStartingIndex = 1
        else:
            iEdgeStartingIndex = 0
        prefix = SO.ProblemSpecificPrefix()
        mySurfaceElevation = np.zeros(myMPASOceanShallowWater.myMesh.nCells)
        for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
            mySurfaceElevation[iCell] = (
            SO.SurfaceElevation(lX,lY,myMPASOceanShallowWater.myMesh.xCell[iCell],
                                myMPASOceanShallowWater.myMesh.yCell[iCell]))
        mySurfaceElevationGradientAtEdge = np.zeros((myMPASOceanShallowWater.myMesh.nEdges,2))
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            mySurfaceElevationGradientAtEdge[iEdge,0], mySurfaceElevationGradientAtEdge[iEdge,1] = (
            SO.SurfaceElevationGradient(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],
                                        myMPASOceanShallowWater.myMesh.yEdge[iEdge]))
        myAnalyticalSurfaceElevationGradientNormalToEdge = (
        SO.AnalyticalGradientOperator(mySurfaceElevationGradientAtEdge,myMPASOceanShallowWater.myMesh.angleEdge))
        myNumericalSurfaceElevationGradientNormalToEdge = (
        SO.NumericalGradientOperator(myMPASOceanShallowWater.myMesh,mySurfaceElevation,BoundaryCondition))
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            if ((BoundaryCondition == 'NonPeriodic_x' or BoundaryCondition == 'NonPeriodic_y' 
                 or BoundaryCondition == 'NonPeriodic_xy') 
                and myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0):
                myNumericalSurfaceElevationGradientNormalToEdge[iEdge] = (
                myAnalyticalSurfaceElevationGradientNormalToEdge[iEdge])  
        mySurfaceElevationGradientNormalToEdgeError = (
        myNumericalSurfaceElevationGradientNormalToEdge - myAnalyticalSurfaceElevationGradientNormalToEdge)
        MaxErrorNorm[iProblemType] = np.linalg.norm(mySurfaceElevationGradientNormalToEdgeError,np.inf)
        L2ErrorNorm[iProblemType]  = (
        (np.linalg.norm(mySurfaceElevationGradientNormalToEdgeError)
         /np.sqrt(float(myMPASOceanShallowWater.myMesh.nEdges 
                        - myMPASOceanShallowWater.myMesh.nNonPeriodicBoundaryEdges))))
        print('The maximum error norm of the surface elevation gradient normal to edges is %.2g.' 
              %MaxErrorNorm[iProblemType])
        print('The L2 error norm of the surface elevation gradient normal to edges is %.2g.' 
              %L2ErrorNorm[iProblemType])
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,labels,labelfontsizes,
             labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show] = SpecifyPlotParameters(MeshDirectory)
            Title = 'Analytical Surface Elevation Gradient\nNormal to Edge'
            FileName = prefix + 'SurfaceElevationGradientNormalToEdge_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myAnalyticalSurfaceElevationGradientNormalToEdge[iEdgeStartingIndex:],nContours,labels,labelfontsizes,
            labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,
            FileName,Show,DataType='Unstructured')
            Title = 'Numerical Surface Elevation Gradient\nNormal to Edge'
            FileName = prefix + 'SurfaceElevationGradientNormalToEdge_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myNumericalSurfaceElevationGradientNormalToEdge[iEdgeStartingIndex:],nContours,labels,labelfontsizes,
            labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,
            FileName,Show,DataType='Unstructured')
            Title = 'Error of Surface Elevation Gradient\nNormal to Edge'
            FileName = prefix + 'SurfaceElevationGradientNormalToEdge_Error'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            mySurfaceElevationGradientNormalToEdgeError[iEdgeStartingIndex:],nContours,labels,labelfontsizes,
            labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,
            FileName,Show,DataType='Unstructured')
    if ConvergenceStudy:
        return MeshDirectoryRoot, lX, dx, MaxErrorNorm, L2ErrorNorm


do_TestNumericalGradientOperator = False
if do_TestNumericalGradientOperator:
    TestNumericalGradientOperator()
    
    
def TestConvergenceOfNumericalGradientOperator(PlotAgainstNumberOfCellsInZonalDirection=True):
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    nBoundaryConditions = len(BoundaryConditions)
    nCellsXArray = np.array([64,96,144,216,324])
    nCases = len(nCellsXArray)
    dc = np.zeros(nCases)
    MaxErrorNorm = np.zeros((nBoundaryConditions,nCases))
    L2ErrorNorm = np.zeros((nBoundaryConditions,nCases))
    prefix = SO.ProblemSpecificPrefix()
    for iCase in range(0,nCases):
        nCellsX = nCellsXArray[iCase]
        nCellsY = nCellsX
        MeshDirectoryRoot, lX, dc[iCase], MaxErrorNorm[:,iCase], L2ErrorNorm[:,iCase] = (
        TestNumericalGradientOperator(ConvergenceStudy=True,PlotFigures=False,nCellsX=nCellsX,nCellsY=nCellsY))
        if PlotAgainstNumberOfCellsInZonalDirection:
            dc[iCase] = lX/dc[iCase]
    for iBoundaryCondition in range(0,nBoundaryConditions):
        BoundaryCondition = BoundaryConditions[iBoundaryCondition]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
        m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'Gradient',PlotAgainstNumberOfCellsInZonalDirection,m,'Max'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNorm[iBoundaryCondition,:],FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, MaxErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'_'+BoundaryCondition+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,MaxErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'Gradient',PlotAgainstNumberOfCellsInZonalDirection,m,'L2'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNorm[iBoundaryCondition,:],FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, L2ErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'_'+BoundaryCondition+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalGradientOperator = False
if do_TestConvergenceOfNumericalGradientOperator:
    TestConvergenceOfNumericalGradientOperator()
    
    
def TestNumericalDivergenceOperator(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
    [PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
     Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,CourantNumber,
     UseCourantNumberToDetermineTimeStep,PrintBasicGeometry,FixAngleEdge,PrintOutput,UseAveragedQuantities,
     SpecifyBoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion,MeshDirectoryRoot,ProblemTypes,
     BaseMeshFileNames,MeshFileNames,BoundaryConditions,nCellsX,nCellsY] = (
    SpecifyInitializationParameters(ConvergenceStudy,nCellsX,nCellsY))
    MaxErrorNorm = np.zeros(4)
    L2ErrorNorm = np.zeros(4)
    for iProblemType in range(0,len(ProblemTypes)):
        ProblemType = ProblemTypes[iProblemType]
        BaseMeshFileName = BaseMeshFileNames[iProblemType]
        MeshFileName = MeshFileNames[iProblemType]
        BoundaryCondition = BoundaryConditions[iProblemType]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
        ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
        LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
        Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,
        MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,CourantNumber,UseCourantNumberToDetermineTimeStep,
        SpecifyBoundaryCondition,BoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion)
        lX = myMPASOceanShallowWater.myMesh.lX
        lY = myMPASOceanShallowWater.myMesh.lY
        dx = myMPASOceanShallowWater.myMesh.dx
        prefix = SO.ProblemSpecificPrefix()
        myAnalyticalSurfaceElevationLaplacian = np.zeros(myMPASOceanShallowWater.myMesh.nCells)
        for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
            myAnalyticalSurfaceElevationLaplacian[iCell] = (
            SO.SurfaceElevationLaplacian(lX,lY,myMPASOceanShallowWater.myMesh.xCell[iCell],
                                         myMPASOceanShallowWater.myMesh.yCell[iCell]))
        mySurfaceElevationGradientAtEdge = np.zeros((myMPASOceanShallowWater.myMesh.nEdges,2))
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            mySurfaceElevationGradientAtEdge[iEdge,0], mySurfaceElevationGradientAtEdge[iEdge,1] = (
            SO.SurfaceElevationGradient(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],
                                        myMPASOceanShallowWater.myMesh.yEdge[iEdge]))
        myAnalyticalSurfaceElevationGradientNormalToEdge = (
        SO.AnalyticalGradientOperator(mySurfaceElevationGradientAtEdge,myMPASOceanShallowWater.myMesh.angleEdge))
        myNumericalSurfaceElevationLaplacian = (
        SO.NumericalDivergenceOperator(myMPASOceanShallowWater.myMesh,myAnalyticalSurfaceElevationGradientNormalToEdge))
        mySurfaceElevationLaplacianError = myNumericalSurfaceElevationLaplacian - myAnalyticalSurfaceElevationLaplacian
        MaxErrorNorm[iProblemType] = np.linalg.norm(mySurfaceElevationLaplacianError,np.inf)
        L2ErrorNorm[iProblemType] = (
        np.linalg.norm(mySurfaceElevationLaplacianError)/np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))   
        print('The maximum error norm of the SurfaceElevation laplacian is %.2g.' %MaxErrorNorm[iProblemType])
        print('The L2 error norm of the SurfaceElevation laplacian is %.2g.' %L2ErrorNorm[iProblemType])
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,labels,labelfontsizes,
             labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show] = SpecifyPlotParameters(MeshDirectory)
            Title = 'Analytical Surface Elevation Laplacian'
            FileName = prefix + 'SurfaceElevationLaplacian_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myAnalyticalSurfaceElevationLaplacian,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured')
            Title = 'Numerical Surface Elevation Laplacian'
            FileName = prefix + 'SurfaceElevationLaplacian_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myNumericalSurfaceElevationLaplacian,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured')            
            Title = 'Error of Surface Elevation Laplacian'
            FileName = prefix + 'SurfaceElevationLaplacian_Error'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            mySurfaceElevationLaplacianError,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured')
    if ConvergenceStudy:
        return MeshDirectoryRoot, lX, dx, MaxErrorNorm, L2ErrorNorm


do_TestNumericalDivergenceOperator = False
if do_TestNumericalDivergenceOperator:
    TestNumericalDivergenceOperator()
    
    
def TestConvergenceOfNumericalDivergenceOperator(PlotAgainstNumberOfCellsInZonalDirection=True):
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    nBoundaryConditions = len(BoundaryConditions)
    nCellsXArray = np.array([64,96,144,216,324])
    nCases = len(nCellsXArray)
    dc = np.zeros(nCases)
    MaxErrorNorm = np.zeros((nBoundaryConditions,nCases))
    L2ErrorNorm = np.zeros((nBoundaryConditions,nCases))
    prefix = SO.ProblemSpecificPrefix()
    for iCase in range(0,nCases):
        nCellsX = nCellsXArray[iCase]
        nCellsY = nCellsX
        MeshDirectoryRoot, lX, dc[iCase], MaxErrorNorm[:,iCase], L2ErrorNorm[:,iCase] = (
        TestNumericalDivergenceOperator(ConvergenceStudy=True,PlotFigures=False,nCellsX=nCellsX,nCellsY=nCellsY))
        if PlotAgainstNumberOfCellsInZonalDirection:
            dc[iCase] = lX/dc[iCase]
    for iBoundaryCondition in range(0,nBoundaryConditions):
        BoundaryCondition = BoundaryConditions[iBoundaryCondition]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
        m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'Divergence',PlotAgainstNumberOfCellsInZonalDirection,m,'Max'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNorm[iBoundaryCondition,:],FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, MaxErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'_'+BoundaryCondition+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,MaxErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'Divergence',PlotAgainstNumberOfCellsInZonalDirection,m,'L2'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNorm[iBoundaryCondition,:],FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, L2ErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'_'+BoundaryCondition+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalDivergenceOperator = False
if do_TestConvergenceOfNumericalDivergenceOperator:
    TestConvergenceOfNumericalDivergenceOperator()
    
    
def TestNumericalCurlOperator(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
    [PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
     Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,CourantNumber,
     UseCourantNumberToDetermineTimeStep,PrintBasicGeometry,FixAngleEdge,PrintOutput,UseAveragedQuantities,
     SpecifyBoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion,MeshDirectoryRoot,ProblemTypes,
     BaseMeshFileNames,MeshFileNames,BoundaryConditions,nCellsX,nCellsY] = (
    SpecifyInitializationParameters(ConvergenceStudy,nCellsX,nCellsY))
    MaxErrorNormAtVertices = np.zeros(4)
    L2ErrorNormAtVertices = np.zeros(4)
    MaxErrorNormAtCellCenters = np.zeros(4)
    L2ErrorNormAtCellCenters = np.zeros(4)
    for iProblemType in range(0,len(ProblemTypes)):
        ProblemType = ProblemTypes[iProblemType]
        BaseMeshFileName = BaseMeshFileNames[iProblemType]
        MeshFileName = MeshFileNames[iProblemType]
        BoundaryCondition = BoundaryConditions[iProblemType]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
        ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
        LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
        Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,
        MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,CourantNumber,UseCourantNumberToDetermineTimeStep,
        SpecifyBoundaryCondition,BoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion)
        lX = myMPASOceanShallowWater.myMesh.lX
        lY = myMPASOceanShallowWater.myMesh.lY
        dx = myMPASOceanShallowWater.myMesh.dx
        prefix = SO.ProblemSpecificPrefix()
        myAnalyticalVelocityCurlAtVertex = np.zeros(myMPASOceanShallowWater.myMesh.nVertices)
        for iVertex in range(0,myMPASOceanShallowWater.myMesh.nVertices):
            myAnalyticalVelocityCurlAtVertex[iVertex] = (
            SO.VelocityCurl(lX,lY,myMPASOceanShallowWater.myMesh.xVertex[iVertex],
                            myMPASOceanShallowWater.myMesh.yVertex[iVertex]))
        myAnalyticalVelocityCurlAtCellCenter = np.zeros(myMPASOceanShallowWater.myMesh.nCells)
        for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
            myAnalyticalVelocityCurlAtCellCenter[iCell] = (
            SO.VelocityCurl(lX,lY,myMPASOceanShallowWater.myMesh.xCell[iCell],
                            myMPASOceanShallowWater.myMesh.yCell[iCell]))  
        myAnalyticalVelocityComponentsAtEdge = np.zeros((myMPASOceanShallowWater.myMesh.nEdges,2))
        myAnalyticalVelocityNormalToEdge = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            myAnalyticalVelocityComponentsAtEdge[iEdge,0], myAnalyticalVelocityComponentsAtEdge[iEdge,1] = (
            SO.Velocity(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],myMPASOceanShallowWater.myMesh.yEdge[iEdge]))
        myAnalyticalVelocityNormalToEdge = (
        SO.ComputeNormalAndTangentialComponentsAtEdge(myAnalyticalVelocityComponentsAtEdge,
                                                      myMPASOceanShallowWater.myMesh.angleEdge,'normal'))
        myNumericalVelocityCurlAtVertex, myNumericalVelocityCurlAtCellCenter = (
        SO.NumericalCurlOperator(myMPASOceanShallowWater.myMesh,myAnalyticalVelocityNormalToEdge,BoundaryCondition))
        myVelocityCurlAtVertexError = myNumericalVelocityCurlAtVertex - myAnalyticalVelocityCurlAtVertex
        MaxErrorNormAtVertices[iProblemType] = np.linalg.norm(myVelocityCurlAtVertexError,np.inf)
        L2ErrorNormAtVertices[iProblemType] = (
        (np.linalg.norm(myVelocityCurlAtVertexError)
         /np.sqrt(float(myMPASOceanShallowWater.myMesh.nVertices 
                        - myMPASOceanShallowWater.myMesh.nNonPeriodicBoundaryVertices))))
        print('The maximum error norm of the velocity curl at vertices is %.2g.' %MaxErrorNormAtVertices[iProblemType])
        print('The L2 error norm of the velocity curl at vertices is %.2g.' %L2ErrorNormAtVertices[iProblemType])
        myVelocityCurlAtCellCenterError = myNumericalVelocityCurlAtCellCenter - myAnalyticalVelocityCurlAtCellCenter
        MaxErrorNormAtCellCenters[iProblemType] = np.linalg.norm(myVelocityCurlAtCellCenterError,np.inf)
        L2ErrorNormAtCellCenters[iProblemType] = (
        np.linalg.norm(myVelocityCurlAtCellCenterError)/np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))
        print('The maximum error norm of the velocity curl at cell centers is %.2g.' 
              %MaxErrorNormAtCellCenters[iProblemType])
        print('The L2 error norm of the velocity curl at cell centers is %.2g.' %L2ErrorNormAtCellCenters[iProblemType])
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,labels,labelfontsizes,
             labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show] = SpecifyPlotParameters(MeshDirectory)
            Title = 'Analytical Velocity Curl at Vertices'
            FileName = prefix + 'VelocityCurlAtVertices_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xVertex/1000.0,
            myMPASOceanShallowWater.myMesh.yVertex/1000.0,myAnalyticalVelocityCurlAtVertex,nContours,labels,
            labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,
            titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured')      
            Title = 'Numerical Velocity Curl at Vertices'
            FileName = prefix + 'VelocityCurlAtVertices_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xVertex/1000.0,
            myMPASOceanShallowWater.myMesh.yVertex/1000.0,myNumericalVelocityCurlAtVertex,nContours,labels,
            labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,
            titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured')        
            Title = 'Error of Velocity Curl at Vertices'
            FileName = prefix + 'VelocityCurlAtVertices_Error'      
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xVertex/1000.0,
            myMPASOceanShallowWater.myMesh.yVertex/1000.0,myVelocityCurlAtVertexError,nContours,labels,labelfontsizes,
            labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,
            FileName,Show,DataType='Unstructured') 
            Title = 'Analytical Velocity Curl at Cell Centers'
            FileName = prefix + 'VelocityCurlAtCellCenters_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myAnalyticalVelocityCurlAtCellCenter,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured')       
            Title = 'Numerical Velocity Curl at Cell Centers'
            FileName = prefix + 'VelocityCurlAtCellCenters_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myNumericalVelocityCurlAtCellCenter,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured')  
            Title = 'Error of Velocity Curl at Cell Centers'
            FileName = prefix + 'VelocityCurlAtCellCenters_Error'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myVelocityCurlAtCellCenterError,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured')  
    if ConvergenceStudy:
        return [MeshDirectoryRoot, lX, dx, MaxErrorNormAtVertices, L2ErrorNormAtVertices, MaxErrorNormAtCellCenters, 
                L2ErrorNormAtCellCenters]
    
    
do_TestNumericalCurlOperator = False
if do_TestNumericalCurlOperator:
    TestNumericalCurlOperator()
    
    
def TestConvergenceOfNumericalCurlOperator(PlotAgainstNumberOfCellsInZonalDirection=True):
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    nBoundaryConditions = len(BoundaryConditions)
    nCellsXArray = np.array([64,96,144,216,324])
    nCases = len(nCellsXArray)
    dc = np.zeros(nCases)
    MaxErrorNormAtVertices = np.zeros((nBoundaryConditions,nCases))
    L2ErrorNormAtVertices = np.zeros((nBoundaryConditions,nCases))
    MaxErrorNormAtCellCenters = np.zeros((nBoundaryConditions,nCases))
    L2ErrorNormAtCellCenters = np.zeros((nBoundaryConditions,nCases))
    prefix = SO.ProblemSpecificPrefix()
    for iCase in range(0,nCases):
        nCellsX = nCellsXArray[iCase]
        nCellsY = nCellsX
        [MeshDirectoryRoot, lX, dc[iCase], MaxErrorNormAtVertices[:,iCase], L2ErrorNormAtVertices[:,iCase], 
         MaxErrorNormAtCellCenters[:,iCase], L2ErrorNormAtCellCenters[:,iCase]] = (
        TestNumericalCurlOperator(ConvergenceStudy=True,PlotFigures=False,nCellsX=nCellsX,nCellsY=nCellsY))
        if PlotAgainstNumberOfCellsInZonalDirection:
            dc[iCase] = lX/dc[iCase]
    for iBoundaryCondition in range(0,nBoundaryConditions):
        BoundaryCondition = BoundaryConditions[iBoundaryCondition]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
        m, c = np.linalg.lstsq(A,np.log10(MaxErrorNormAtVertices[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'CurlAtVertices',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'Max'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNormAtVertices[iBoundaryCondition,:],FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, MaxErrorNormAtVertices[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,
                                                                          FileName+'_'+BoundaryCondition+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,MaxErrorNormAtVertices[iBoundaryCondition,:],y,
                                            linewidths,linestyles,colors,markers,markertypes,markersizes,labels,
                                            labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
                                            legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,FigureSize,
                                            drawMajorGrid,drawMinorGrid,legendWithinBox)
        m, c = np.linalg.lstsq(A,np.log10(L2ErrorNormAtVertices[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'CurlAtVertices',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'L2'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNormAtVertices[iBoundaryCondition,:],FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, L2ErrorNormAtVertices[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,
                                                                         FileName+'_'+BoundaryCondition+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNormAtVertices[iBoundaryCondition,:],y,
                                            linewidths,linestyles,colors,markers,markertypes,markersizes,labels,
                                            labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
                                            legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,FigureSize,
                                            drawMajorGrid,drawMinorGrid,legendWithinBox)
        m, c = np.linalg.lstsq(A,np.log10(MaxErrorNormAtCellCenters[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'CurlAtCellCenters',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'Max'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNormAtCellCenters[iBoundaryCondition,:],
                        FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, MaxErrorNormAtCellCenters[iBoundaryCondition,:] = (
        CR.ReadCurve1D(OutputDirectory,FileName+'_'+BoundaryCondition+'.curve'))
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,MaxErrorNormAtCellCenters[iBoundaryCondition,:],
                                            y,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,
                                            labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
                                            legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,FigureSize,
                                            drawMajorGrid,drawMinorGrid,legendWithinBox)
        m, c = np.linalg.lstsq(A,np.log10(L2ErrorNormAtCellCenters[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'CurlAtCellCenters',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'L2'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNormAtCellCenters[iBoundaryCondition,:],
                        FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, L2ErrorNormAtCellCenters[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,
                                                                            FileName+'_'+BoundaryCondition+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNormAtCellCenters[iBoundaryCondition,:],
                                            y,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,
                                            labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
                                            legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,FigureSize,
                                            drawMajorGrid,drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalCurlOperator = False
if do_TestConvergenceOfNumericalCurlOperator:
    TestConvergenceOfNumericalCurlOperator()
    
    
def TestNumericalTangentialVelocity(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
    [PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
     Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,CourantNumber,
     UseCourantNumberToDetermineTimeStep,PrintBasicGeometry,FixAngleEdge,PrintOutput,UseAveragedQuantities,
     SpecifyBoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion,MeshDirectoryRoot,ProblemTypes,
     BaseMeshFileNames,MeshFileNames,BoundaryConditions,nCellsX,nCellsY] = (
    SpecifyInitializationParameters(ConvergenceStudy,nCellsX,nCellsY))
    MaxErrorNorm = np.zeros(4)
    L2ErrorNorm = np.zeros(4)
    for iProblemType in range(0,len(ProblemTypes)):
        ProblemType = ProblemTypes[iProblemType]
        BaseMeshFileName = BaseMeshFileNames[iProblemType]
        MeshFileName = MeshFileNames[iProblemType]
        BoundaryCondition = BoundaryConditions[iProblemType]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        myMPASOceanShallowWater = MPASOceanShallowWaterClass.MPASOceanShallowWater(
        ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
        LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
        Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,
        MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,CourantNumber,UseCourantNumberToDetermineTimeStep,
        SpecifyBoundaryCondition,BoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion)
        lX = myMPASOceanShallowWater.myMesh.lX
        lY = myMPASOceanShallowWater.myMesh.lY
        dx = myMPASOceanShallowWater.myMesh.dx
        if BoundaryCondition == 'NonPeriodic_x':
            iEdgeStartingIndex = 1
        else:
            iEdgeStartingIndex = 0
        prefix = SO.ProblemSpecificPrefix()
        myAnalyticalVelocityComponentsAtEdge = np.zeros((myMPASOceanShallowWater.myMesh.nEdges,2))
        myAnalyticalNormalVelocity = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        myAnalyticalTangentialVelocity = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            myAnalyticalVelocityComponentsAtEdge[iEdge,0], myAnalyticalVelocityComponentsAtEdge[iEdge,1] = (
            SO.Velocity(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],myMPASOceanShallowWater.myMesh.yEdge[iEdge]))
        myAnalyticalNormalVelocity, myAnalyticalTangentialVelocity = (
        SO.ComputeNormalAndTangentialComponentsAtEdge(myAnalyticalVelocityComponentsAtEdge,
                                                      myMPASOceanShallowWater.myMesh.angleEdge,'both'))
        myNumericalTangentialVelocity = (
        SO.NumericalTangentialVelocity(myMPASOceanShallowWater.myMesh,myAnalyticalNormalVelocity,BoundaryCondition))
        myTangentialVelocityError = myNumericalTangentialVelocity - myAnalyticalTangentialVelocity
        MaxErrorNorm[iProblemType] = np.linalg.norm(myTangentialVelocityError,np.inf)
        L2ErrorNorm[iProblemType] = (np.linalg.norm(myTangentialVelocityError)
                                     /np.sqrt(float(myMPASOceanShallowWater.myMesh.nEdges 
                                                    - myMPASOceanShallowWater.myMesh.nNonPeriodicBoundaryEdges)))
        print('The maximum error norm of the tangential velocity is %.2g.' %MaxErrorNorm[iProblemType])
        print('The L2 error norm of the tangential velocity is %.2g.' %L2ErrorNorm[iProblemType])
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,labels,labelfontsizes,
             labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show] = SpecifyPlotParameters(MeshDirectory)
            Title = 'Analytical Tangential Velocity'
            FileName = prefix + 'TangentialVelocity_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myAnalyticalTangentialVelocity[iEdgeStartingIndex:],nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured')
            Title = 'Numerical Tangential Velocity'
            FileName = prefix + 'TangentialVelocity_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myNumericalTangentialVelocity[iEdgeStartingIndex:],nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured')
            Title = 'Tangential Velocity Error'
            FileName = prefix + 'TangentialVelocity_Error'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myTangentialVelocityError[iEdgeStartingIndex:],nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured')
    if ConvergenceStudy:
        return MeshDirectoryRoot, lX, dx, MaxErrorNorm, L2ErrorNorm
    
    
do_TestNumericalTangentialVelocity = False
if do_TestNumericalTangentialVelocity:
    TestNumericalTangentialVelocity()
    
    
def TestConvergenceOfNumericalTangentialVelocity(PlotAgainstNumberOfCellsInZonalDirection=True):
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    nBoundaryConditions = len(BoundaryConditions)
    nCellsXArray = np.array([64,96,144,216,324])
    nCases = len(nCellsXArray)
    dc = np.zeros(nCases)
    MaxErrorNorm = np.zeros((nBoundaryConditions,nCases))
    L2ErrorNorm = np.zeros((nBoundaryConditions,nCases))
    prefix = SO.ProblemSpecificPrefix()
    for iCase in range(0,nCases):
        nCellsX = nCellsXArray[iCase]
        nCellsY = nCellsX
        MeshDirectoryRoot, lX, dc[iCase], MaxErrorNorm[:,iCase], L2ErrorNorm[:,iCase] = (
        TestNumericalTangentialVelocity(ConvergenceStudy=True,PlotFigures=False,nCellsX=nCellsX,nCellsY=nCellsY))
        if PlotAgainstNumberOfCellsInZonalDirection:
            dc[iCase] = lX/dc[iCase]
    for iBoundaryCondition in range(0,nBoundaryConditions):
        BoundaryCondition = BoundaryConditions[iBoundaryCondition]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
        m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'TangentialVelocity',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'Max'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNorm[iBoundaryCondition,:],FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, MaxErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'_'+BoundaryCondition+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,MaxErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'TangentialVelocity',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'L2'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNorm[iBoundaryCondition,:],FileName+'_'+BoundaryCondition)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine_'+BoundaryCondition)
        dc, L2ErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'_'+BoundaryCondition+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine_'+BoundaryCondition+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalTangentialVelocity = False
if do_TestConvergenceOfNumericalTangentialVelocity:
    TestConvergenceOfNumericalTangentialVelocity()