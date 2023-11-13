"""
Name: Test_SpatialOperators.py
Author: Sid Bishnu
Details: As the name implies, this script tests the various spatial operators of the TRiSK-based mimetic finite volume 
method computed in ../../src/MPAS_Ocean_Shallow_Water/SpatialOperators.py against their exact counterparts using smooth 
two-dimensional functions.
"""


import numpy as np
import matplotlib.pyplot as plt
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
    marker = 's'
    markersize = 7.5
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    labels = [xLabel,yLabel]
    labelfontsizes = [22.5,22.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    titlefontsize = 27.5
    SaveAsPDF = True
    Show = False
    ColorMap = plt.cm.seismic
    return [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,marker,markersize,labels,
            labelfontsizes,labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show,ColorMap]
    

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
    elif SpatialOperator == 'CurlAtEdges':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of curl operator'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of curl operator'
        title = 'Convergence of Curl Operator\nInterpolated to Edges'    
        FileName = 'ConvergencePlot_CurlOperatorAtEdges_%sErrorNorm' %ErrorNormType  
    elif SpatialOperator == 'CurlAtCellCenters':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of curl operator'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of curl operator'
        title = 'Convergence of Curl Operator\nInterpolated to Cell Centers'    
        FileName = 'ConvergencePlot_CurlOperatorAtCellCenters_%sErrorNorm' %ErrorNormType
    elif SpatialOperator == 'TangentialOperator':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of tangential velocity'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of tangential velocity'
        title = 'Convergence of Tangential Operator\nalong Edges'
        FileName = 'ConvergencePlot_TangentialOperator_%sErrorNorm' %ErrorNormType
    elif SpatialOperator == 'EnergyOperator':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of kinetic energy'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of kinetic energy'
        title = 'Convergence of Kinetic Energy\nat Cell Centers'
        FileName = 'ConvergencePlot_KineticEnergy_%sErrorNorm' %ErrorNormType    
    elif SpatialOperator == 'LaplacianOperator':
        if ErrorNormType == 'Max':
            yLabel = 'Maximum error norm of normal velocity Laplacian'
        elif ErrorNormType == 'L2':
            yLabel = 'L$^2$ error norm of normal velocity Laplacian'
        title = 'Convergence of Normal Velocity Laplacian\nat Edges'
        FileName = 'ConvergencePlot_NormalVelocityLaplacian_%sErrorNorm' %ErrorNormType
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
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,marker,markersize,labels,
             labelfontsizes,labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show,ColorMap] = (
            SpecifyPlotParameters(MeshDirectory))
            Title = 'Surface Elevation'
            FileName = prefix + 'SurfaceElevation'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            mySurfaceElevation,nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
            ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured',
            colormap=ColorMap)
            Title = 'Zonal Velocity'
            FileName = prefix + 'ZonalVelocity'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,myZonalVelocity[iEdgeStartingIndex:],
            nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
            nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured',colormap=ColorMap)
            Title = 'Meridional Velocity'
            FileName = prefix + 'MeridionalVelocity'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,myMeridionalVelocity[iEdgeStartingIndex:],
            nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
            nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured',colormap=ColorMap)
            Title = 'Resultant Velocity'
            FileName = prefix + 'ResultantVelocity'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,myResultantVelocity[iEdgeStartingIndex:],
            nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
            nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured',colormap=ColorMap)
            if PlotNormalVelocity:
                Title = 'Normal Velocity'
                FileName = prefix + 'NormalVelocity'
                CR.ScatterPlotWithColorBar(
                OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
                myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,myNormalVelocity[iEdgeStartingIndex:],
                marker,markersize,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
                nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,colormap=ColorMap)
                
                
do_TestSurfaceElevationNormalVelocity = False
if do_TestSurfaceElevationNormalVelocity:
    TestSurfaceElevationNormalVelocity()
    
    
def TestNumericalGradientOperatorAtEdge_NormalComponent(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
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
        myAnalyticalSurfaceElevationGradientAtEdge_NormalComponent = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            myAnalyticalSurfaceElevationGradientAtEdge_NormalComponent[iEdge] = (
            SO.SurfaceElevationGradientAtEdge_NormalComponent(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],
                                                              myMPASOceanShallowWater.myMesh.yEdge[iEdge],
                                                              myMPASOceanShallowWater.myMesh.angleEdge[iEdge]))
        myNumericalSurfaceElevationGradientAtEdge_NormalComponent = (
        SO.NumericalGradientOperatorAtEdge_NormalComponent(myMPASOceanShallowWater.myMesh,mySurfaceElevation,
                                                           SO.SurfaceElevationGradientAtEdge_NormalComponent))
        mySurfaceElevationGradientAtEdge_NormalComponent_Error = (
        (myNumericalSurfaceElevationGradientAtEdge_NormalComponent 
         - myAnalyticalSurfaceElevationGradientAtEdge_NormalComponent))
        MaxErrorNorm[iProblemType] = np.linalg.norm(mySurfaceElevationGradientAtEdge_NormalComponent_Error,np.inf)
        L2ErrorNorm[iProblemType] = (np.linalg.norm(mySurfaceElevationGradientAtEdge_NormalComponent_Error)
                                     /np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))
        print('The maximum error norm of the surface elevation gradient normal to edges is %.2g.' 
              %MaxErrorNorm[iProblemType])
        print('The L2 error norm of the surface elevation gradient normal to edges is %.2g.' 
              %L2ErrorNorm[iProblemType])
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,marker,markersize,labels,
             labelfontsizes,labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show,ColorMap] = (
            SpecifyPlotParameters(MeshDirectory))
            Title = 'Analytical Surface Elevation Gradient\nNormal to Edge'
            FileName = prefix + 'SurfaceElevationGradientAtEdge_NormalComponent_Analytical'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myAnalyticalSurfaceElevationGradientAtEdge_NormalComponent[iEdgeStartingIndex:],marker,markersize,labels,
            labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,
            titlefontsize,SaveAsPDF,FileName,Show,colormap=ColorMap)
            Title = 'Numerical Surface Elevation Gradient\nNormal to Edge'
            FileName = prefix + 'SurfaceElevationGradientAtEdge_NormalComponent_Numerical'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myNumericalSurfaceElevationGradientAtEdge_NormalComponent[iEdgeStartingIndex:],marker,markersize,labels,
            labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,
            titlefontsize,SaveAsPDF,FileName,Show,colormap=ColorMap)
            useGivenColorBarLimits_Error = True
            ColorBarLimits_Error = np.array([-MaxErrorNorm[iProblemType],MaxErrorNorm[iProblemType]])
            Title = 'Error of Surface Elevation Gradient\nNormal to Edge'
            FileName = prefix + 'SurfaceElevationGradientAtEdge_NormalComponent_Error'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            mySurfaceElevationGradientAtEdge_NormalComponent_Error[iEdgeStartingIndex:],marker,markersize,labels,
            labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits_Error,ColorBarLimits_Error,nColorBarTicks,
            Title,titlefontsize,SaveAsPDF,FileName,Show,colormap=ColorMap)
    if ConvergenceStudy:
        return MeshDirectoryRoot, lX, dx, MaxErrorNorm, L2ErrorNorm


do_TestNumericalGradientOperatorAtEdge_NormalComponent = False
if do_TestNumericalGradientOperatorAtEdge_NormalComponent:
    TestNumericalGradientOperatorAtEdge_NormalComponent()
    
    
def TestConvergenceOfNumericalGradientOperatorAtEdge_NormalComponent(PlotAgainstNumberOfCellsInZonalDirection=True):
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
        TestNumericalGradientOperatorAtEdge_NormalComponent(ConvergenceStudy=True,PlotFigures=False,nCellsX=nCellsX,
                                                            nCellsY=nCellsY))
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
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, MaxErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
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
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, L2ErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalGradientOperatorAtEdge_NormalComponent = False
if do_TestConvergenceOfNumericalGradientOperatorAtEdge_NormalComponent:
    TestConvergenceOfNumericalGradientOperatorAtEdge_NormalComponent()
    

def TestNumericalDivergenceOperatorAtCellCenter(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
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
        myAnalyticalSurfaceElevationGradientAtEdge_NormalComponent = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            myAnalyticalSurfaceElevationGradientAtEdge_NormalComponent[iEdge] = (
            SO.SurfaceElevationGradientAtEdge_NormalComponent(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],
                                                              myMPASOceanShallowWater.myMesh.yEdge[iEdge],
                                                              myMPASOceanShallowWater.myMesh.angleEdge[iEdge]))
        myNumericalSurfaceElevationLaplacian = (
        SO.NumericalDivergenceOperatorAtCellCenter(myMPASOceanShallowWater.myMesh,
        myAnalyticalSurfaceElevationGradientAtEdge_NormalComponent))
        mySurfaceElevationLaplacianError = myNumericalSurfaceElevationLaplacian - myAnalyticalSurfaceElevationLaplacian
        MaxErrorNorm[iProblemType] = np.linalg.norm(mySurfaceElevationLaplacianError,np.inf)
        L2ErrorNorm[iProblemType] = (
        np.linalg.norm(mySurfaceElevationLaplacianError)/np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))   
        print('The maximum error norm of the surface elevation Laplacian is %.2g.' %MaxErrorNorm[iProblemType])
        print('The L2 error norm of the surface elevation Laplacian is %.2g.' %L2ErrorNorm[iProblemType])
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,marker,markersize,labels,
             labelfontsizes,labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show,ColorMap] = (
            SpecifyPlotParameters(MeshDirectory))
            Title = 'Analytical Surface Elevation Laplacian'
            FileName = prefix + 'SurfaceElevationLaplacian_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myAnalyticalSurfaceElevationLaplacian,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured',colormap=ColorMap)
            Title = 'Numerical Surface Elevation Laplacian'
            FileName = prefix + 'SurfaceElevationLaplacian_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myNumericalSurfaceElevationLaplacian,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured',colormap=ColorMap)  
            useGivenColorBarLimits_Error = True
            ColorBarLimits_Error = np.array([-MaxErrorNorm[iProblemType],MaxErrorNorm[iProblemType]])
            Title = 'Error of Surface Elevation Laplacian'
            FileName = prefix + 'SurfaceElevationLaplacian_Error'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            mySurfaceElevationLaplacianError,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits_Error,ColorBarLimits_Error,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
            Show,DataType='Unstructured',colormap=ColorMap)
    if ConvergenceStudy:
        return MeshDirectoryRoot, lX, dx, MaxErrorNorm, L2ErrorNorm


do_TestNumericalDivergenceOperatorAtCellCenter = False
if do_TestNumericalDivergenceOperatorAtCellCenter:
    TestNumericalDivergenceOperatorAtCellCenter()
    

def TestConvergenceOfNumericalDivergenceOperatorAtCellCenter(PlotAgainstNumberOfCellsInZonalDirection=True):
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
        TestNumericalDivergenceOperatorAtCellCenter(ConvergenceStudy=True,PlotFigures=False,nCellsX=nCellsX,
                                                    nCellsY=nCellsY))
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
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, MaxErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
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
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, L2ErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalDivergenceOperatorAtCellCenter = False
if do_TestConvergenceOfNumericalDivergenceOperatorAtCellCenter:
    TestConvergenceOfNumericalDivergenceOperatorAtCellCenter()

    
def TestNumericalCurlOperator(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
    [PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
     Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,CourantNumber,
     UseCourantNumberToDetermineTimeStep,PrintBasicGeometry,FixAngleEdge,PrintOutput,UseAveragedQuantities,
     SpecifyBoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion,MeshDirectoryRoot,ProblemTypes,
     BaseMeshFileNames,MeshFileNames,BoundaryConditions,nCellsX,nCellsY] = (
    SpecifyInitializationParameters(ConvergenceStudy,nCellsX,nCellsY))
    MaxErrorNormAtVertices = np.zeros(4)
    L2ErrorNormAtVertices = np.zeros(4)
    MaxErrorNormAtEdges = np.zeros(4)
    L2ErrorNormAtEdges = np.zeros(4)
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
        myAnalyticalVelocityCurlAtEdge = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            myAnalyticalVelocityCurlAtEdge[iEdge] = (
            SO.VelocityCurl(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],
                            myMPASOceanShallowWater.myMesh.yEdge[iEdge]))
        myAnalyticalVelocityCurlAtCellCenter = np.zeros(myMPASOceanShallowWater.myMesh.nCells)
        for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
            myAnalyticalVelocityCurlAtCellCenter[iCell] = (
            SO.VelocityCurl(lX,lY,myMPASOceanShallowWater.myMesh.xCell[iCell],
                            myMPASOceanShallowWater.myMesh.yCell[iCell]))  
        myAnalyticalVelocityAtEdge_NormalComponent = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            myAnalyticalVelocityAtEdge_NormalComponent[iEdge] = (
            SO.VelocityAtEdge_NormalComponent(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],
                                              myMPASOceanShallowWater.myMesh.yEdge[iEdge],
                                              myMPASOceanShallowWater.myMesh.angleEdge[iEdge]))
        myNumericalVelocityCurlAtVertex, myNumericalVelocityCurlAtEdge, myNumericalVelocityCurlAtCellCenter = (
        SO.NumericalCurlOperator(myMPASOceanShallowWater.myMesh,myAnalyticalVelocityAtEdge_NormalComponent,
                                 SO.VelocityCurl))
        myVelocityCurlAtVertexError = myNumericalVelocityCurlAtVertex - myAnalyticalVelocityCurlAtVertex
        MaxErrorNormAtVertices[iProblemType] = np.linalg.norm(myVelocityCurlAtVertexError,np.inf)
        L2ErrorNormAtVertices[iProblemType] = (
        np.linalg.norm(myVelocityCurlAtVertexError)/np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))
        print('The maximum error norm of the velocity curl at vertices is %.2g.' %MaxErrorNormAtVertices[iProblemType])
        print('The L2 error norm of the velocity curl at vertices is %.2g.' %L2ErrorNormAtVertices[iProblemType])
        myVelocityCurlAtEdgeError = myNumericalVelocityCurlAtEdge - myAnalyticalVelocityCurlAtEdge
        MaxErrorNormAtEdges[iProblemType] = np.linalg.norm(myVelocityCurlAtEdgeError,np.inf)
        L2ErrorNormAtEdges[iProblemType] = (
        np.linalg.norm(myVelocityCurlAtEdgeError)/np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))
        print('The maximum error norm of the velocity curl at edges is %.2g.' %MaxErrorNormAtEdges[iProblemType])
        print('The L2 error norm of the velocity curl at edges is %.2g.' %L2ErrorNormAtEdges[iProblemType])
        myVelocityCurlAtCellCenterError = myNumericalVelocityCurlAtCellCenter - myAnalyticalVelocityCurlAtCellCenter
        MaxErrorNormAtCellCenters[iProblemType] = np.linalg.norm(myVelocityCurlAtCellCenterError,np.inf)
        L2ErrorNormAtCellCenters[iProblemType] = (
        np.linalg.norm(myVelocityCurlAtCellCenterError)/np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))
        print('The maximum error norm of the velocity curl at cell centers is %.2g.' 
              %MaxErrorNormAtCellCenters[iProblemType])
        print('The L2 error norm of the velocity curl at cell centers is %.2g.' %L2ErrorNormAtCellCenters[iProblemType])
        CheckMaxErrorNormAtEdgesAndVertices = False
        if CheckMaxErrorNormAtEdgesAndVertices and not(BoundaryCondition == 'Periodic'):
            nBoundaryEdges = myMPASOceanShallowWater.myMesh.nBoundaryEdges
            nBoundaryVertices = myMPASOceanShallowWater.myMesh.nBoundaryVertices
            myVelocityCurlErrorAtBoundaryEdges = np.zeros(nBoundaryEdges)
            myVelocityCurlErrorAtBoundaryVertices = np.zeros(nBoundaryVertices)
            iBoundaryEdge = -1
            for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
                if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                    iBoundaryEdge += 1
                    myVelocityCurlErrorAtBoundaryEdges[iBoundaryEdge] = myVelocityCurlAtEdgeError[iEdge]
            iBoundaryVertex = -1
            for iVertex in range(0,myMPASOceanShallowWater.myMesh.nVertices):
                if myMPASOceanShallowWater.myMesh.boundaryVertex[iVertex] == 1:
                    iBoundaryVertex += 1
                    myVelocityCurlErrorAtBoundaryVertices[iBoundaryVertex] = myVelocityCurlAtVertexError[iVertex]
            print('The expected maximum error norm of the velocity curl at the boundary edges is 0.0.')
            print('The computed maximum error norm of the velocity curl at the boundary edges is %.2g.' 
                  %np.linalg.norm(myVelocityCurlErrorAtBoundaryEdges,np.inf))
            print('The expected maximum error norm of the velocity curl at the boundary vertices is 0.0.')
            print('The computed maximum error norm of the velocity curl at the boundary vertices is %.2g.'
                  %np.linalg.norm(myVelocityCurlErrorAtBoundaryVertices,np.inf))
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,marker,markersize,labels,
             labelfontsizes,labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show,ColorMap] = (
            SpecifyPlotParameters(MeshDirectory))
            Title = 'Analytical Velocity Curl at Vertices'
            FileName = prefix + 'VelocityCurlAtVertices_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xVertex/1000.0,
            myMPASOceanShallowWater.myMesh.yVertex/1000.0,myAnalyticalVelocityCurlAtVertex,nContours,labels,
            labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,
            titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured',colormap=ColorMap)      
            Title = 'Numerical Velocity Curl at Vertices'
            FileName = prefix + 'VelocityCurlAtVertices_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xVertex/1000.0,
            myMPASOceanShallowWater.myMesh.yVertex/1000.0,myNumericalVelocityCurlAtVertex,nContours,labels,
            labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,
            titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured',colormap=ColorMap)         
            useGivenColorBarLimits_Error = True
            ColorBarLimits_Error = np.array([-MaxErrorNormAtVertices[iProblemType],
                                             MaxErrorNormAtVertices[iProblemType]])
            Title = 'Error of Velocity Curl at Vertices'
            FileName = prefix + 'VelocityCurlAtVertices_Error'      
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xVertex/1000.0,
            myMPASOceanShallowWater.myMesh.yVertex/1000.0,myVelocityCurlAtVertexError,marker,markersize,labels,
            labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits_Error,ColorBarLimits_Error,nColorBarTicks,
            Title,titlefontsize,SaveAsPDF,FileName,Show,colormap=ColorMap)
            Title = 'Analytical Velocity Curl at Edges'
            FileName = prefix + 'VelocityCurlAtEdges_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge/1000.0,myMPASOceanShallowWater.myMesh.yEdge/1000.0,
            myAnalyticalVelocityCurlAtEdge,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured',colormap=ColorMap)       
            Title = 'Numerical Velocity Curl at Edges'
            FileName = prefix + 'VelocityCurlAtEdges_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge/1000.0,myMPASOceanShallowWater.myMesh.yEdge/1000.0,
            myNumericalVelocityCurlAtEdge,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured',colormap=ColorMap)  
            useGivenColorBarLimits_Error = True
            ColorBarLimits_Error = np.array([-MaxErrorNormAtEdges[iProblemType],MaxErrorNormAtEdges[iProblemType]])
            Title = 'Error of Velocity Curl at Edges'
            FileName = prefix + 'VelocityCurlAtEdges_Error'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge/1000.0,myMPASOceanShallowWater.myMesh.yEdge/1000.0,
            myVelocityCurlAtEdgeError,marker,markersize,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits_Error,ColorBarLimits_Error,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
            Show,colormap=ColorMap)
            Title = 'Analytical Velocity Curl at Cell Centers'
            FileName = prefix + 'VelocityCurlAtCellCenters_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myAnalyticalVelocityCurlAtCellCenter,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured',colormap=ColorMap)       
            Title = 'Numerical Velocity Curl at Cell Centers'
            FileName = prefix + 'VelocityCurlAtCellCenters_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myNumericalVelocityCurlAtCellCenter,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
            DataType='Unstructured',colormap=ColorMap)  
            useGivenColorBarLimits_Error = True
            ColorBarLimits_Error = np.array([-MaxErrorNormAtCellCenters[iProblemType],
                                             MaxErrorNormAtCellCenters[iProblemType]])
            Title = 'Error of Velocity Curl at Cell Centers'
            FileName = prefix + 'VelocityCurlAtCellCenters_Error'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myVelocityCurlAtCellCenterError,marker,markersize,labels,labelfontsizes,labelpads,tickfontsizes,
            useGivenColorBarLimits_Error,ColorBarLimits_Error,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
            Show,colormap=ColorMap)  
    if ConvergenceStudy:
        return [MeshDirectoryRoot, lX, dx, MaxErrorNormAtVertices, L2ErrorNormAtVertices, MaxErrorNormAtEdges,
                L2ErrorNormAtEdges, MaxErrorNormAtCellCenters, L2ErrorNormAtCellCenters]
    
    
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
    MaxErrorNormAtEdges = np.zeros((nBoundaryConditions,nCases))
    L2ErrorNormAtEdges = np.zeros((nBoundaryConditions,nCases))
    MaxErrorNormAtCellCenters = np.zeros((nBoundaryConditions,nCases))
    L2ErrorNormAtCellCenters = np.zeros((nBoundaryConditions,nCases))
    prefix = SO.ProblemSpecificPrefix()
    for iCase in range(0,nCases):
        nCellsX = nCellsXArray[iCase]
        nCellsY = nCellsX
        [MeshDirectoryRoot, lX, dc[iCase], MaxErrorNormAtVertices[:,iCase], L2ErrorNormAtVertices[:,iCase], 
         MaxErrorNormAtEdges[:,iCase], L2ErrorNormAtEdges[:,iCase], MaxErrorNormAtCellCenters[:,iCase], 
         L2ErrorNormAtCellCenters[:,iCase]] = TestNumericalCurlOperator(ConvergenceStudy=True,PlotFigures=False,
                                                                        nCellsX=nCellsX,nCellsY=nCellsY)
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
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNormAtVertices[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, MaxErrorNormAtVertices[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
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
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNormAtVertices[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, L2ErrorNormAtVertices[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNormAtVertices[iBoundaryCondition,:],y,
                                            linewidths,linestyles,colors,markers,markertypes,markersizes,labels,
                                            labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
                                            legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,FigureSize,
                                            drawMajorGrid,drawMinorGrid,legendWithinBox)
        m, c = np.linalg.lstsq(A,np.log10(MaxErrorNormAtEdges[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'CurlAtEdges',PlotAgainstNumberOfCellsInZonalDirection,m,'Max'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNormAtEdges[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, MaxErrorNormAtEdges[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,MaxErrorNormAtEdges[iBoundaryCondition,:],y,
                                            linewidths,linestyles,colors,markers,markertypes,markersizes,labels,
                                            labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
                                            legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,FigureSize,
                                            drawMajorGrid,drawMinorGrid,legendWithinBox)
        m, c = np.linalg.lstsq(A,np.log10(L2ErrorNormAtEdges[iBoundaryCondition,:]),rcond=None)[0]
        y = m*(np.log10(dc)) + c
        y = 10.0**y
        [OutputDirectory,PlotType,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
        labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
        FigureSize,drawMajorGrid,drawMinorGrid,legendWithinBox] = (
        SpecifyConvergencePlotParameters(MeshDirectory,'CurlAtEdges',PlotAgainstNumberOfCellsInZonalDirection,m,'L2'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNormAtEdges[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, L2ErrorNormAtEdges[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNormAtEdges[iBoundaryCondition,:],y,
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
                        FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, MaxErrorNormAtCellCenters[iBoundaryCondition,:] = (
        CR.ReadCurve1D(OutputDirectory,FileName+'.curve'))
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
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
                        FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, L2ErrorNormAtCellCenters[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNormAtCellCenters[iBoundaryCondition,:],
                                            y,linewidths,linestyles,colors,markers,markertypes,markersizes,labels,
                                            labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
                                            legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,FigureSize,
                                            drawMajorGrid,drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalCurlOperator = False
if do_TestConvergenceOfNumericalCurlOperator:
    TestConvergenceOfNumericalCurlOperator()
    

def TestNumericalTangentialOperatorAlongEdge(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
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
        myAnalyticalNormalVelocity = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        myAnalyticalTangentialVelocity = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            xEdge = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            yEdge = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            myAnalyticalNormalVelocity[iEdge] = SO.VelocityAtEdge_NormalComponent(lX,lY,xEdge,yEdge,angleEdge)
            myAnalyticalTangentialVelocity[iEdge] = SO.VelocityAtEdge_TangentialComponent(lX,lY,xEdge,yEdge,angleEdge)
        myNumericalTangentialVelocity = (
        SO.NumericalTangentialOperatorAlongEdge(myMPASOceanShallowWater.myMesh,myAnalyticalNormalVelocity,
                                                SO.VelocityAtEdge_TangentialComponent))
        myTangentialVelocityError = myNumericalTangentialVelocity - myAnalyticalTangentialVelocity
        MaxErrorNorm[iProblemType] = np.linalg.norm(myTangentialVelocityError,np.inf)
        L2ErrorNorm[iProblemType] = (np.linalg.norm(myTangentialVelocityError)
                                     /np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))
        print('The maximum error norm of the tangential velocity is %.2g.' %MaxErrorNorm[iProblemType])
        print('The L2 error norm of the tangential velocity is %.2g.' %L2ErrorNorm[iProblemType])
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,marker,markersize,labels,
             labelfontsizes,labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show,ColorMap] = (
            SpecifyPlotParameters(MeshDirectory))
            Title = 'Analytical Tangential Velocity'
            FileName = prefix + 'TangentialVelocity_Analytical'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myAnalyticalTangentialVelocity[iEdgeStartingIndex:],marker,markersize,labels,labelfontsizes,labelpads,
            tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
            Show,colormap=ColorMap)
            Title = 'Numerical Tangential Velocity'
            FileName = prefix + 'TangentialVelocity_Numerical'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myNumericalTangentialVelocity[iEdgeStartingIndex:],marker,markersize,labels,labelfontsizes,labelpads,
            tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
            Show,colormap=ColorMap)
            useGivenColorBarLimits_Error = True
            ColorBarLimits_Error = np.array([-MaxErrorNorm[iProblemType],MaxErrorNorm[iProblemType]])
            Title = 'Tangential Velocity Error'
            FileName = prefix + 'TangentialVelocity_Error'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myTangentialVelocityError[iEdgeStartingIndex:],marker,markersize,labels,labelfontsizes,labelpads,
            tickfontsizes,useGivenColorBarLimits_Error,ColorBarLimits_Error,nColorBarTicks,Title,titlefontsize,
            SaveAsPDF,FileName,Show,colormap=ColorMap)
    if ConvergenceStudy:
        return MeshDirectoryRoot, lX, dx, MaxErrorNorm, L2ErrorNorm
    
    
do_TestNumericalTangentialOperatorAlongEdge = False
if do_TestNumericalTangentialOperatorAlongEdge:
    TestNumericalTangentialOperatorAlongEdge()
    
    
def TestConvergenceOfNumericalTangentialOperatorAlongEdge(PlotAgainstNumberOfCellsInZonalDirection=True):
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
        TestNumericalTangentialOperatorAlongEdge(ConvergenceStudy=True,PlotFigures=False,nCellsX=nCellsX,
                                                 nCellsY=nCellsY))
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
        SpecifyConvergencePlotParameters(MeshDirectory,'TangentialOperator',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'Max'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, MaxErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
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
        SpecifyConvergencePlotParameters(MeshDirectory,'TangentialOperator',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'L2'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, L2ErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalTangentialOperatorAlongEdge = False
if do_TestConvergenceOfNumericalTangentialOperatorAlongEdge:
    TestConvergenceOfNumericalTangentialOperatorAlongEdge()
    
    
def TestNumericalEnergyOperatorAtCellCenter(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
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
        myAnalyticalKineticEnergy = np.zeros(myMPASOceanShallowWater.myMesh.nCells)
        for iCell in range(0,myMPASOceanShallowWater.myMesh.nCells):
            myAnalyticalKineticEnergy[iCell] = (
            SO.KineticEnergy(lX,lY,myMPASOceanShallowWater.myMesh.xCell[iCell],
                             myMPASOceanShallowWater.myMesh.yCell[iCell]))
        myVelocityVectorAtEdge_NormalComponent = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            myVelocityVectorAtEdge_NormalComponent[iEdge] = (
            SO.VelocityAtEdge_NormalComponent(lX,lY,myMPASOceanShallowWater.myMesh.xEdge[iEdge],
                                              myMPASOceanShallowWater.myMesh.yEdge[iEdge],
                                              myMPASOceanShallowWater.myMesh.angleEdge[iEdge]))
        myNumericalKineticEnergy = (
        SO.NumericalEnergyOperatorAtCellCenter(myMPASOceanShallowWater.myMesh,myVelocityVectorAtEdge_NormalComponent))
        myKineticEnergyError = myNumericalKineticEnergy - myAnalyticalKineticEnergy
        MaxErrorNorm[iProblemType] = np.linalg.norm(myKineticEnergyError,np.inf)
        L2ErrorNorm[iProblemType] = (
        np.linalg.norm(myKineticEnergyError)/np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))   
        print('The maximum error norm of the kinetic energy is %.2g.' %MaxErrorNorm[iProblemType])
        print('The L2 error norm of the kinetic energy is %.2g.' %L2ErrorNorm[iProblemType])
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,marker,markersize,labels,
             labelfontsizes,labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show,ColorMap] = (
            SpecifyPlotParameters(MeshDirectory))
            Title = 'Analytical Kinetic Energy'
            FileName = prefix + 'KineticEnergy_Analytical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myAnalyticalKineticEnergy,nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
            ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured',
            colormap=ColorMap)
            Title = 'Numerical Kinetic Energy'
            FileName = prefix + 'KineticEnergy_Numerical'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myNumericalKineticEnergy,nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
            ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured',
            colormap=ColorMap)  
            useGivenColorBarLimits_Error = True
            ColorBarLimits_Error = np.array([-MaxErrorNorm[iProblemType],MaxErrorNorm[iProblemType]])          
            Title = 'Error of Kinetic Energy'
            FileName = prefix + 'KineticEnergy_Error'
            CR.PythonFilledContourPlot2DSaveAsPDF(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xCell/1000.0,myMPASOceanShallowWater.myMesh.yCell/1000.0,
            myKineticEnergyError,nContours,labels,labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits_Error,
            ColorBarLimits_Error,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,DataType='Unstructured',
            colormap=ColorMap)
    if ConvergenceStudy:
        return MeshDirectoryRoot, lX, dx, MaxErrorNorm, L2ErrorNorm


do_TestNumericalEnergyOperatorAtCellCenter = False
if do_TestNumericalEnergyOperatorAtCellCenter:
    TestNumericalEnergyOperatorAtCellCenter()
    

def TestConvergenceOfNumericalEnergyOperatorAtCellCenter(PlotAgainstNumberOfCellsInZonalDirection=True):
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
        TestNumericalEnergyOperatorAtCellCenter(ConvergenceStudy=True,PlotFigures=False,nCellsX=nCellsX,
                                                nCellsY=nCellsY))
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
        SpecifyConvergencePlotParameters(MeshDirectory,'EnergyOperator',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'Max'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, MaxErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
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
        SpecifyConvergencePlotParameters(MeshDirectory,'EnergyOperator',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'L2'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, L2ErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalEnergyOperatorAtCellCenter = False
if do_TestConvergenceOfNumericalEnergyOperatorAtCellCenter:
    TestConvergenceOfNumericalEnergyOperatorAtCellCenter()
    
    
def TestNumericalLaplacianOperatorAtEdge(ConvergenceStudy=False,PlotFigures=True,nCellsX=0,nCellsY=0):
    Method = 2
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
        myAnalyticalNormalVelocity = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        myAnalyticalNormalVelocityLaplacian = np.zeros(myMPASOceanShallowWater.myMesh.nEdges)
        for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
            xEdge = myMPASOceanShallowWater.myMesh.xEdge[iEdge]
            yEdge = myMPASOceanShallowWater.myMesh.yEdge[iEdge]
            angleEdge = myMPASOceanShallowWater.myMesh.angleEdge[iEdge]
            myAnalyticalNormalVelocity[iEdge] = SO.VelocityAtEdge_NormalComponent(lX,lY,xEdge,yEdge,angleEdge)
            myAnalyticalNormalVelocityLaplacian[iEdge] = SO.VelocityLaplacianAtEdge_NormalComponent(lX,lY,xEdge,yEdge,
                                                                                                    angleEdge)
        if Method == 1:
            myNumericalNormalVelocityLaplacian = (
            SO.NumericalLaplacianOperatorAtEdge_Method_1(
            myMPASOceanShallowWater.myMesh,myAnalyticalNormalVelocity,SO.VelocityAtEdge_TangentialComponent,
            SO.ZonalVelocityGradientAtEdge_NormalComponent,SO.MeridionalVelocityGradientAtEdge_NormalComponent,
            SO.VelocityLaplacianAtEdge_NormalComponent))       
        else: # if Method == 2:
            myNumericalNormalVelocityLaplacian = (
            SO.NumericalLaplacianOperatorAtEdge_Method_2(
            myMPASOceanShallowWater.myMesh,myAnalyticalNormalVelocity,SO.VelocityCurl,
            SO.VelocityGradientOfDivergenceAtEdge_NormalComponent,SO.VelocityLaplacianAtEdge_NormalComponent))
        myNormalVelocityLaplacianError = myNumericalNormalVelocityLaplacian - myAnalyticalNormalVelocityLaplacian
        MaxErrorNorm[iProblemType] = np.linalg.norm(myNormalVelocityLaplacianError,np.inf)
        L2ErrorNorm[iProblemType] = (np.linalg.norm(myNormalVelocityLaplacianError)
                                     /np.sqrt(float(myMPASOceanShallowWater.myMesh.nCells)))
        print('The maximum error norm of the normal velocity Laplacian is %.2g.' %MaxErrorNorm[iProblemType])
        print('The L2 error norm of the normal velocity Laplacian is %.2g.' %L2ErrorNorm[iProblemType])
        CheckMaxErrorNorm = False
        if CheckMaxErrorNorm and not(BoundaryCondition == 'Periodic'):
            nBoundaryEdges = myMPASOceanShallowWater.myMesh.nBoundaryEdges
            myNormalVelocityLaplacianErrorAtBoundaryEdges = np.zeros(nBoundaryEdges)
            iBoundaryEdge = -1
            for iEdge in range(0,myMPASOceanShallowWater.myMesh.nEdges):
                if myMPASOceanShallowWater.myMesh.boundaryEdge[iEdge] == 1.0:
                    iBoundaryEdge += 1
                    myNormalVelocityLaplacianErrorAtBoundaryEdges[iBoundaryEdge] = myNormalVelocityLaplacianError[iEdge]
            print('The expected maximum error norm of the normal velocity Laplacian at the boundary edges is 0.0.')
            print('The computed maximum error norm of the normal velocity Laplacian at the boundary edges is %.2g.' 
                  %np.linalg.norm(myNormalVelocityLaplacianErrorAtBoundaryEdges,np.inf))
        if PlotFigures:
            [OutputDirectory,nContours,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,marker,markersize,labels,
             labelfontsizes,labelpads,tickfontsizes,titlefontsize,SaveAsPDF,Show,ColorMap] = (
            SpecifyPlotParameters(MeshDirectory))
            Title = 'Analytical Laplacian of Velocity'
            FileName = prefix + 'VelocityLaplacian_Analytical'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myAnalyticalNormalVelocityLaplacian[iEdgeStartingIndex:],marker,markersize,labels,labelfontsizes,labelpads,
            tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
            Show,colormap=ColorMap)
            Title = 'Numerical Laplacian of Velocity'
            FileName = prefix + 'VelocityLaplacian_Numerical'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myNumericalNormalVelocityLaplacian[iEdgeStartingIndex:],marker,markersize,labels,labelfontsizes,labelpads,
            tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
            Show,colormap=ColorMap)
            useGivenColorBarLimits_Error = True
            ColorBarLimits_Error = np.array([-MaxErrorNorm[iProblemType],MaxErrorNorm[iProblemType]])
            Title = 'Laplacian of Velocity Error'
            FileName = prefix + 'VelocityLaplacian_Error'
            CR.ScatterPlotWithColorBar(
            OutputDirectory,myMPASOceanShallowWater.myMesh.xEdge[iEdgeStartingIndex:]/1000.0,
            myMPASOceanShallowWater.myMesh.yEdge[iEdgeStartingIndex:]/1000.0,
            myNormalVelocityLaplacianError[iEdgeStartingIndex:],marker,markersize,labels,labelfontsizes,labelpads,
            tickfontsizes,useGivenColorBarLimits_Error,ColorBarLimits_Error,nColorBarTicks,Title,titlefontsize,
            SaveAsPDF,FileName,Show,colormap=ColorMap)
    if ConvergenceStudy:
        return MeshDirectoryRoot, lX, dx, MaxErrorNorm, L2ErrorNorm
    
    
do_TestNumericalLaplacianOperatorAtEdge = False
if do_TestNumericalLaplacianOperatorAtEdge:
    TestNumericalLaplacianOperatorAtEdge()
    
    
def TestConvergenceOfNumericalLaplacianOperatorAtEdge(PlotAgainstNumberOfCellsInZonalDirection=True):
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
        TestNumericalLaplacianOperatorAtEdge(ConvergenceStudy=True,PlotFigures=False,nCellsX=nCellsX,nCellsY=nCellsY))
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
        SpecifyConvergencePlotParameters(MeshDirectory,'LaplacianOperator',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'Max'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,MaxErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, MaxErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
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
        SpecifyConvergencePlotParameters(MeshDirectory,'LaplacianOperator',PlotAgainstNumberOfCellsInZonalDirection,m,
                                         'L2'))
        FileName = prefix + FileName
        CR.WriteCurve1D(OutputDirectory,dc,L2ErrorNorm[iBoundaryCondition,:],FileName)
        CR.WriteCurve1D(OutputDirectory,dc,y,FileName+'_BestFitLine')
        dc, L2ErrorNorm[iBoundaryCondition,:] = CR.ReadCurve1D(OutputDirectory,FileName+'.curve')
        dc, y = CR.ReadCurve1D(OutputDirectory,FileName+'_BestFitLine'+'.curve')
        CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,PlotType,dc,L2ErrorNorm[iBoundaryCondition,:],y,linewidths,
                                            linestyles,colors,markers,markertypes,markersizes,labels,labelfontsizes,
                                            labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
                                            titlefontsize,SaveAsPDF,FileName,Show,FigureSize,drawMajorGrid,
                                            drawMinorGrid,legendWithinBox)
        
        
do_TestConvergenceOfNumericalLaplacianOperatorAtEdge = False
if do_TestConvergenceOfNumericalLaplacianOperatorAtEdge:
    TestConvergenceOfNumericalLaplacianOperatorAtEdge()