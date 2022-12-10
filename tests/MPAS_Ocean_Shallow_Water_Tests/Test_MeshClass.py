"""
Name: Test_MeshClass.py
Author: Siddhartha Bishnu
Details: As the name implies, this script tests the planar hexagonal mesh class defined in 
../../src/MPAS_Ocean_Shallow_Water/MeshClass.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/MPAS_Ocean_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR
    import Initialization
    import MeshClass
    
    
def TestMesh():
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
    DebugVersions = [True,False]
    nCellsXArray = np.array([4,50])
    PlotCellCentersArray = [True,False]
    CellCenterMarkerSizes = np.array([7.5,2.5])
    PlotEdgeCentersArray = [True,True]
    EdgeCenterMarkerSizes = np.array([7.5,2.5])
    PlotVerticesArray = [True,False]
    VertexMarkerSizes = np.array([7.5,2.5])
    PlotUnitVectorsAlongAngleEdgesArray = [True,False]
    VectorScales = np.array([3.5,3.5])
    MeshDirectories = ['../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_4x4_Cells',
                       '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_50x50_Cells']
    ProblemTypes = ['Inertia_Gravity_Wave','Coastal_Kelvin_Wave','Planetary_Rossby_Wave',
                    'NonLinear_Manufactured_Solution']
    BaseMeshFileNames = ['base_mesh_Periodic.nc','culled_mesh_NonPeriodic_x.nc','culled_mesh_NonPeriodic_y.nc',
                         'culled_mesh_NonPeriodic_xy.nc']
    MeshFileNames = ['mesh_Periodic.nc','mesh_NonPeriodic_x.nc','mesh_NonPeriodic_y.nc','mesh_NonPeriodic_xy.nc']
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    MeshPlotTitles = ['Planar Hexagonal MPAS-Ocean Mesh\nPeriodic along Zonal and Meridional Boundaries',
                      ('Planar Hexagonal MPAS-Ocean Mesh'
                       + '\nNon-Periodic along Zonal and Periodic along Meridional Boundaries'),
                      ('Planar Hexagonal MPAS-Ocean Mesh'
                       + '\nPeriodic along Zonal and Non-Periodic along Meridional Boundaries'),
                      'Planar Hexagonal MPAS-Ocean Mesh\nNon-Periodic along Zonal and Meridional Boundaries']
    CellCentersPlotTitles = [('Planar Hexagonal MPAS-Ocean Mesh Cell Centers'
                              + '\nPeriodic along Zonal and Meridional Boundaries'),
                             ('Planar Hexagonal MPAS-Ocean Mesh Cell Centers'
                              + '\nNon-Periodic along Zonal and Periodic along Meridional Boundaries'),
                             ('Planar Hexagonal MPAS-Ocean Mesh Cell Centers'
                              + '\nPeriodic along Zonal and Non-Periodic along Meridional Boundaries'),
                             ('Planar Hexagonal MPAS-Ocean Mesh Cell Centers'
                              + '\nNon-Periodic along Zonal and Meridional Boundaries')]
    ShiftedCellCentersPlotTitles = [('Planar Hexagonal MPAS-Ocean Mesh Shifted Cell Centers'
                                     + '\nPeriodic along Zonal and Meridional Boundaries'),
                                    ('Planar Hexagonal MPAS-Ocean Mesh Shifted Cell Centers'
                                     + '\nNon-Periodic along Zonal and Periodic along Meridional Boundaries'),
                                    ('Planar Hexagonal MPAS-Ocean Mesh Shifted Cell Centers'
                                     + '\nPeriodic along Zonal and Non-Periodic along Meridional Boundaries'),
                                    ('Planar Hexagonal MPAS-Ocean Mesh Shifted Cell Centers'
                                     + '\nNon-Periodic along Zonal and Meridional Boundaries')]
    MeshPlotFileNames = ['MPAS_Ocean_Mesh_Periodic','MPAS_Ocean_Mesh_NonPeriodic_x','MPAS_Ocean_Mesh_NonPeriodic_y',
                         'MPAS_Ocean_Mesh_NonPeriodic_xy']
    CellCentersPlotFileNames = ['MPAS_Ocean_Mesh_CellCenters_Periodic','MPAS_Ocean_Mesh_CellCenters_NonPeriodic_x',
                                'MPAS_Ocean_Mesh_CellCenters_NonPeriodic_y',
                                'MPAS_Ocean_Mesh_CellCenters_NonPeriodic_xy']
    ShiftedCellCentersPlotFileNames = ['MPAS_Ocean_Mesh_ShiftedCellCenters_Periodic',
                                       'MPAS_Ocean_Mesh_ShiftedCellCenters_NonPeriodic_x',
                                       'MPAS_Ocean_Mesh_ShiftedCellCenters_NonPeriodic_y',
                                       'MPAS_Ocean_Mesh_ShiftedCellCenters_NonPeriodic_xy']
    for iMeshDirectory in range(0,len(MeshDirectories)):
        nCellsX = nCellsXArray[iMeshDirectory]
        nCellsY = nCellsX
        DebugVersion = DebugVersions[iMeshDirectory]
        PlotCellCenters = PlotCellCentersArray[iMeshDirectory]
        CellCenterMarkerSize = CellCenterMarkerSizes[iMeshDirectory]
        PlotEdgeCenters = PlotEdgeCentersArray[iMeshDirectory]
        EdgeCenterMarkerSize = EdgeCenterMarkerSizes[iMeshDirectory]
        PlotVertices = PlotVerticesArray[iMeshDirectory]
        VertexMarkerSize = VertexMarkerSizes[iMeshDirectory]
        PlotUnitVectorsAlongAngleEdges = PlotUnitVectorsAlongAngleEdgesArray[iMeshDirectory]
        VectorScale = VectorScales[iMeshDirectory]
        for iProblemType in range(0,len(ProblemTypes)):
            ProblemType = ProblemTypes[iProblemType]
            myNameList = Initialization.NameList(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                                 TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                 Generalized_FB_with_AB2_AM3_Step_Type,
                                                 Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,CourantNumber,
                                                 UseCourantNumberToDetermineTimeStep)
            BaseMeshFileName = BaseMeshFileNames[iProblemType]
            MeshFileName = MeshFileNames[iProblemType]
            BoundaryCondition = BoundaryConditions[iProblemType]
            MeshDirectory = MeshDirectories[iMeshDirectory] + '/' + BoundaryCondition
            myMesh = MeshClass.Mesh(myNameList,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,
                                    FixAngleEdge,PrintOutput,UseAveragedQuantities,SpecifyBoundaryCondition,
                                    BoundaryCondition,ReadDomainExtentsfromMeshFile,DebugVersion)
            OutputDirectory = MeshDirectory
            linewidth = 2.0
            linestyle = '-'
            color = 'k'
            markertype = 's'
            xLabel = 'Zonal Distance (km)'
            yLabel = 'Meridional Distance (km)'
            labels = [xLabel,yLabel]
            labelfontsizes = [22.5,22.5]
            labelpads = [10.0,10.0]
            tickfontsizes = [15.0,15.0]
            MeshPlotTitle = MeshPlotTitles[iProblemType]
            CellCentersPlotTitle = CellCentersPlotTitles[iProblemType]
            ShiftedCellCentersPlotTitle = ShiftedCellCentersPlotTitles[iProblemType]
            titlefontsize = 27.5
            SaveAsPDF = True
            MeshPlotFileName = MeshPlotFileNames[iProblemType]
            CellCentersPlotFileName = CellCentersPlotFileNames[iProblemType]
            ShiftedCellCentersPlotFileName = ShiftedCellCentersPlotFileNames[iProblemType]
            Show = False
            myMesh.PlotMesh(OutputDirectory,linewidth,linestyle,color,labels,labelfontsizes,labelpads,tickfontsizes,
                            MeshPlotTitle,titlefontsize,SaveAsPDF,MeshPlotFileName,Show,fig_size=[9.5,9.5],
                            UseDefaultMethodToSpecifyTickFontSize=True,PlotCellCenters=PlotCellCenters,
                            CellCenterMarkerSize=CellCenterMarkerSize,PlotEdgeCenters=PlotEdgeCenters,
                            EdgeCenterMarkerSize=EdgeCenterMarkerSize,PlotVertices=PlotVertices,
                            VertexMarkerSize=VertexMarkerSize,
                            PlotUnitVectorsAlongAngleEdges=PlotUnitVectorsAlongAngleEdges,VectorScale=VectorScale)
            # Plot the cell centers.
            CR.ScatterPlot(OutputDirectory,myMesh.xCell/1000.0,myMesh.yCell/1000.0,color,markertype,
                           CellCenterMarkerSize,labels,labelfontsizes,labelpads,tickfontsizes,CellCentersPlotTitle,
                           titlefontsize,SaveAsPDF,CellCentersPlotFileName,Show,fig_size=[9.5,9.5],
                           UseDefaultMethodToSpecifyTickFontSize=True,titlepad=1.035,FileFormat='pdf')
            xCell, yCell = myMesh.GenerateRectilinearMPASOceanMesh(BoundaryCondition)
            # Plot the shifted cell centers.
            CR.ScatterPlot(OutputDirectory,xCell/1000.0,yCell/1000.0,color,markertype,CellCenterMarkerSize,labels,
                           labelfontsizes,labelpads,tickfontsizes,ShiftedCellCentersPlotTitle,titlefontsize,SaveAsPDF,
                           ShiftedCellCentersPlotFileName,Show,fig_size=[9.5,9.5],
                           UseDefaultMethodToSpecifyTickFontSize=True,titlepad=1.035,FileFormat='pdf')
            xCellStructured, yCellStructured = (
            MeshClass.GenerateStructuredRectinilearMeshCoordinateArrays1D(myMesh.xCell,myMesh.yCell,
                                                                          PrintMeshCoordinateArrays1D=True))


do_TestMesh = False
if do_TestMesh:
    TestMesh()
    
    
def TestPlotMeshes():
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
    nCellsXArray = np.array([4,8])
    CellCenterMarkerTypes = ['s','s']
    CellCenterMarkerSizes = np.array([10.0,5.0])
    MeshDirectories = ['../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_4x4_Cells',
                       '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_8x8_Cells']
    ProblemTypes = ['Inertia_Gravity_Wave','Coastal_Kelvin_Wave','Planetary_Rossby_Wave',
                    'NonLinear_Manufactured_Solution']
    BaseMeshFileNames = ['base_mesh_Periodic.nc','culled_mesh_NonPeriodic_x.nc','culled_mesh_NonPeriodic_y.nc',
                         'culled_mesh_NonPeriodic_xy.nc']
    MeshFileNames = ['mesh_Periodic.nc','mesh_NonPeriodic_x.nc','mesh_NonPeriodic_y.nc','mesh_NonPeriodic_xy.nc']
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    for iProblemType in range(0,len(ProblemTypes)):
        ProblemType = ProblemTypes[iProblemType]
        for iMeshDirectory in range(0,len(MeshDirectories)):
            nCellsX = nCellsXArray[iMeshDirectory]
            nCellsY = nCellsX
            myNameList = Initialization.NameList(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                                 TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                 Generalized_FB_with_AB2_AM3_Step_Type,
                                                 Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,CourantNumber,
                                                 UseCourantNumberToDetermineTimeStep)
            BaseMeshFileName = BaseMeshFileNames[iProblemType]
            MeshFileName = MeshFileNames[iProblemType]
            BoundaryCondition = BoundaryConditions[iProblemType]
            MeshDirectory = MeshDirectories[iMeshDirectory] + '/' + BoundaryCondition
            myMesh = MeshClass.Mesh(myNameList,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,
                                    FixAngleEdge,PrintOutput,UseAveragedQuantities,SpecifyBoundaryCondition,
                                    BoundaryCondition,ReadDomainExtentsfromMeshFile)
            if iMeshDirectory == 0:
                myCoarseMesh = myMesh
            else:
                myFineMesh = myMesh
                OutputDirectory = MeshDirectory
                linewidths = [2.5,2.5]
                linestyles = ['-','--']
                colors = ['r','b']
                xLabel = 'Zonal Distance (km)'
                yLabel = 'Meridional Distance (km)'
                labels = [xLabel,yLabel]
                labelfontsizes = [22.5,22.5]
                labelpads = [10.0,10.0]
                tickfontsizes = [15.0,15.0]
                titlefontsize = 27.5
                title = 'Planar Hexagonal MPAS-Ocean Meshes'
                SaveAsPDF = True
                FileName = 'MPASOceanMeshes_' + BoundaryCondition
                Show = False
                fig_size = [9.5,9.5]
                UseDefaultMethodToSpecifyTickFontSize = True
                MeshClass.PlotMeshes(myCoarseMesh,myFineMesh,OutputDirectory,linewidths,linestyles,colors,labels,
                                     labelfontsizes,labelpads,tickfontsizes,title,titlefontsize,SaveAsPDF,FileName,Show,
                                     fig_size,UseDefaultMethodToSpecifyTickFontSize,CellCenterMarkerTypes,
                                     CellCenterMarkerSizes)
            
            
do_TestPlotMeshes = False
if do_TestPlotMeshes:
    TestPlotMeshes()
    
    
def SurfaceElevation(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    return eta


def TestInterpolateSolutionFromVerticesAndEdgesToCellCenters(PlotFigures=True):
    PrintPhaseSpeedOfWaveModes = False
    PrintAmplitudesOfWaveModes = False
    TimeIntegrator = 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
    LF_TR_and_LF_AM3_with_FB_Feedback_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    Generalized_FB_with_AB2_AM3_Step_Type = 'ThirdOrderAccurate_WideStabilityRange'
    Generalized_FB_with_AB3_AM4_Step_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    MeshDirectoryRoot = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_50x50_Cells'
    nCellsX = 50
    nCellsY = nCellsX
    CourantNumber = 0.5
    UseCourantNumberToDetermineTimeStep = True
    PrintBasicGeometry = False
    FixAngleEdge = True
    PrintOutput = False
    UseAveragedQuantities = False
    SpecifyBoundaryCondition = True
    ReadDomainExtentsfromMeshFile = True
    ProblemTypes = ['Inertia_Gravity_Wave','Coastal_Kelvin_Wave','Planetary_Rossby_Wave',
                    'NonLinear_Manufactured_Solution']
    BaseMeshFileNames = ['base_mesh_Periodic.nc','culled_mesh_NonPeriodic_x.nc','culled_mesh_NonPeriodic_y.nc',
                         'culled_mesh_NonPeriodic_xy.nc']
    MeshFileNames = ['mesh_Periodic.nc','mesh_NonPeriodic_x.nc','mesh_NonPeriodic_y.nc','mesh_NonPeriodic_xy.nc']
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    for iProblemType in range(0,len(ProblemTypes)):
        ProblemType = ProblemTypes[iProblemType]
        myNameList = Initialization.NameList(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                             TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                             Generalized_FB_with_AB2_AM3_Step_Type,
                                             Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,CourantNumber,
                                             UseCourantNumberToDetermineTimeStep)
        BaseMeshFileName = BaseMeshFileNames[iProblemType]
        MeshFileName = MeshFileNames[iProblemType]
        BoundaryCondition = BoundaryConditions[iProblemType]
        MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
        myMesh = MeshClass.Mesh(myNameList,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,FixAngleEdge,
                                PrintOutput,UseAveragedQuantities,SpecifyBoundaryCondition,BoundaryCondition,
                                ReadDomainExtentsfromMeshFile)
        ExactSSHAtVertices = np.zeros(myMesh.nVertices)
        ExactSSHAtEdges = np.zeros(myMesh.nEdges)
        ExactSSHAtCellCenters = np.zeros(myMesh.nCells)
        InterpolatedSSHFromVerticesToCellCenters = np.zeros(myMesh.nCells)
        SSHInterpolationErrorFromVerticesToCellCenters = np.zeros(myMesh.nCells)
        InterpolatedSSHFromEdgesToCellCenters = np.zeros(myMesh.nCells)
        SSHInterpolationErrorFromEdgesToCellCenters = np.zeros(myMesh.nCells)
        for iVertex in range(0,myMesh.nVertices):
            ExactSSHAtVertices[iVertex] = SurfaceElevation(myMesh.lX,myMesh.lY,myMesh.xVertex[iVertex],
                                                           myMesh.yVertex[iVertex])
        for iEdge in range(0,myMesh.nEdges):
            ExactSSHAtEdges[iEdge] = SurfaceElevation(myMesh.lX,myMesh.lY,myMesh.xEdge[iEdge],myMesh.yEdge[iEdge])
        for iCell in range(0,myMesh.nCells):
            ExactSSHAtCellCenters[iCell] = SurfaceElevation(myMesh.lX,myMesh.lY,myMesh.xCell[iCell],
                                                            myMesh.yCell[iCell])          
        InterpolatedSSHFromVerticesToCellCenters = (
        myMesh.InterpolateSolutionFromVerticesToCellCenters(ExactSSHAtVertices))
        SSHInterpolationErrorFromVerticesToCellCenters = (
        InterpolatedSSHFromVerticesToCellCenters - ExactSSHAtCellCenters)
        InterpolatedSSHFromEdgesToCellCenters = myMesh.InterpolateSolutionFromEdgesToCellCenters(ExactSSHAtEdges)
        SSHInterpolationErrorFromEdgesToCellCenters = InterpolatedSSHFromEdgesToCellCenters - ExactSSHAtCellCenters
        if PlotFigures:
            OutputDirectory = MeshDirectory
            nContours = 300
            xLabel = 'Zonal Distance (km)'
            yLabel = 'Meridional Distance (km)'
            labels = [xLabel,yLabel]
            labelfontsizes = [22.5,22.5]
            labelpads = [10.0,10.0]
            tickfontsizes = [15.0,15.0]
            useGivenColorBarLimits = False
            ColorBarLimits = [0.0,0.0]
            nColorBarTicks = 6
            xLabel = 'Zonal Distance (km)'
            yLabel = 'Meridional Distance (km)'
            labels = [xLabel,yLabel]
            titlefontsize = 27.5
            SaveAsPDF = True
            Show = False
            DataType = 'Unstructured'
            Title = 'Exact Surface Elevation At Vertices'
            FileName = 'ExactSurfaceElevationAtVertices_' + BoundaryCondition
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,myMesh.xVertex/1000.0,myMesh.yVertex/1000.0,
                                                  ExactSSHAtVertices,nContours,labels,labelfontsizes,labelpads,
                                                  tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,
                                                  Title,titlefontsize,SaveAsPDF,FileName,Show,DataType=DataType)
            Title = 'Exact Surface Elevation At Edges'
            FileName = 'ExactSurfaceElevationAtEdges_' + BoundaryCondition
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,myMesh.xEdge/1000.0,myMesh.yEdge/1000.0,
                                                  ExactSSHAtEdges,nContours,labels,labelfontsizes,labelpads,
                                                  tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,
                                                  Title,titlefontsize,SaveAsPDF,FileName,Show,DataType=DataType)  
            Title = 'Exact Surface Elevation At Cell Centers'
            FileName = 'ExactSurfaceElevationAtCellCenters_' + BoundaryCondition 
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,myMesh.xCell/1000.0,myMesh.yCell/1000.0,
                                                  ExactSSHAtCellCenters,nContours,labels,labelfontsizes,labelpads,
                                                  tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,
                                                  Title,titlefontsize,SaveAsPDF,FileName,Show,DataType=DataType)       
            Title = 'Interpolated Surface Elevation\nfrom Vertices to Cell Centers'
            FileName = 'InterpolatedSurfaceElevationFromVerticesToCellCenters_' + BoundaryCondition 
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,myMesh.xCell/1000.0,myMesh.yCell/1000.0,
                                                  InterpolatedSSHFromVerticesToCellCenters,nContours,labels,
                                                  labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
                                                  ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
                                                  Show,DataType=DataType)   
            Title = 'Surface Elevation Interpolation Error\nfrom Vertices to Cell Centers'
            FileName = 'SurfaceElevationInterpolationErrorFromVerticesToCellCenters_' + BoundaryCondition 
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,myMesh.xCell/1000.0,myMesh.yCell/1000.0,
                                                  SSHInterpolationErrorFromVerticesToCellCenters,nContours,labels,
                                                  labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
                                                  ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
                                                  Show,DataType=DataType,cbarlabelformat='%.2e')              
            Title = 'Interpolated Surface Elevation\nfrom Edges to Cell Centers'
            FileName = 'InterpolatedSurfaceElevationFromEdgesToCellCenters_' + BoundaryCondition 
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,myMesh.xCell/1000.0,myMesh.yCell/1000.0,
                                                  InterpolatedSSHFromEdgesToCellCenters,nContours,labels,
                                                  labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
                                                  ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
                                                  Show,DataType=DataType,cbarlabelformat='%.2e')           
            Title = 'Surface Elevation Interpolation Error\nfrom Edges to Cell Centers'
            FileName = 'SurfaceElevationInterpolationErrorFromEdgesToCellCenters_' + BoundaryCondition 
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,myMesh.xCell/1000.0,myMesh.yCell/1000.0,
                                                  SSHInterpolationErrorFromEdgesToCellCenters,nContours,labels,
                                                  labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
                                                  ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
                                                  Show,DataType=DataType,cbarlabelformat='%.2e')


do_TestInterpolateSolutionFromVerticesAndEdgesToCellCenters = False
if do_TestInterpolateSolutionFromVerticesAndEdgesToCellCenters:
    TestInterpolateSolutionFromVerticesAndEdgesToCellCenters()
    
    
def TestInterpolateSolutionToCoarsestRectilinearMPASOceanMesh(PlotFigures=True):
    PrintPhaseSpeedOfWaveModes = False
    PrintAmplitudesOfWaveModes = False
    TimeIntegrator = 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
    LF_TR_and_LF_AM3_with_FB_Feedback_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    Generalized_FB_with_AB2_AM3_Step_Type = 'ThirdOrderAccurate_WideStabilityRange'
    Generalized_FB_with_AB3_AM4_Step_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    MeshDirectoryRoot = (
    '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_Interpolation_Study')
    CourantNumber = 0.5
    UseCourantNumberToDetermineTimeStep = True
    PrintBasicGeometry = False
    FixAngleEdge = True
    PrintOutput = False
    UseAveragedQuantities = False
    SpecifyBoundaryCondition = True
    ReadDomainExtentsfromMeshFile = True
    ProblemTypes = ['Inertia_Gravity_Wave','Coastal_Kelvin_Wave','Planetary_Rossby_Wave',
                    'NonLinear_Manufactured_Solution']
    BaseMeshFileNames = ['base_mesh_Periodic','culled_mesh_NonPeriodic_x','culled_mesh_NonPeriodic_y',
                         'culled_mesh_NonPeriodic_xy']
    MeshFileNames = ['mesh_Periodic','mesh_NonPeriodic_x','mesh_NonPeriodic_y','mesh_NonPeriodic_xy']
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    nCases = 2
    for iProblemType in range(0,len(ProblemTypes)):
        ProblemType = ProblemTypes[iProblemType]
        for iCase in range(0,nCases):
            if iCase == 0:
                nCellsX = 32
            else: # if iCase == 1:
                nCellsX = 64
            nCellsY = nCellsX
            myNameList = Initialization.NameList(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                                 TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                 Generalized_FB_with_AB2_AM3_Step_Type,
                                                 Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,CourantNumber,
                                                 UseCourantNumberToDetermineTimeStep)
            BaseMeshFileName = BaseMeshFileNames[iProblemType] + '_%sx%s.nc' %(nCellsX,nCellsX)
            MeshFileName = MeshFileNames[iProblemType] + '_%sx%s.nc' %(nCellsX,nCellsX)
            BoundaryCondition = BoundaryConditions[iProblemType]
            MeshDirectory = MeshDirectoryRoot + '/' + BoundaryCondition
            if iCase == 0:
                myCoarseMesh = MeshClass.Mesh(myNameList,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,
                                              MeshFileName,FixAngleEdge,PrintOutput,UseAveragedQuantities,
                                              SpecifyBoundaryCondition,BoundaryCondition,ReadDomainExtentsfromMeshFile)
            else: # if iCase == 1:
                myFineMesh = MeshClass.Mesh(myNameList,PrintBasicGeometry,MeshDirectory,BaseMeshFileName,MeshFileName,
                                            FixAngleEdge,PrintOutput,UseAveragedQuantities,SpecifyBoundaryCondition,
                                            BoundaryCondition,ReadDomainExtentsfromMeshFile)
        sshOnCoarseMesh = np.zeros(myCoarseMesh.nCells)
        sshOnFineMesh = np.zeros(myFineMesh.nCells)
        lX = max(myCoarseMesh.lX,myFineMesh.lX)
        lY = max(myCoarseMesh.lY,myFineMesh.lY)
        for iCell in range(0,myCoarseMesh.nCells):
            sshOnCoarseMesh[iCell] = SurfaceElevation(lX,lY,myCoarseMesh.xCell[iCell],myCoarseMesh.yCell[iCell])
        for iCell in range(0,myFineMesh.nCells):
            sshOnFineMesh[iCell] = SurfaceElevation(lX,lY,myFineMesh.xCell[iCell],myFineMesh.yCell[iCell])   
        xCellOnCoarseRectilinearMesh, yCellOnCoarseRectilinearMesh = (
        myCoarseMesh.GenerateRectilinearMPASOceanMesh(BoundaryCondition))
        InterpolatedSSHOnCoarseRectilinearMesh = (
        myCoarseMesh.InterpolateSolutionFromMPASOceanMeshToRectilinearMPASOceanMesh(BoundaryCondition,
                                                                                    sshOnCoarseMesh))    
        exactSSHOnCoarseRectilinearMesh = np.zeros(myCoarseMesh.nCells)
        for iCell in range(0,myCoarseMesh.nCells):
            exactSSHOnCoarseRectilinearMesh[iCell] = SurfaceElevation(lX,lY,xCellOnCoarseRectilinearMesh[iCell],
                                                                      yCellOnCoarseRectilinearMesh[iCell])
        xCellOnFineRectilinearMesh, yCellOnFineRectilinearMesh = (
        myFineMesh.GenerateRectilinearMPASOceanMesh(BoundaryCondition))
        InterpolatedSSHOnFineRectilinearMesh = (
        myFineMesh.InterpolateSolutionFromMPASOceanMeshToRectilinearMPASOceanMesh(BoundaryCondition,
                                                                                  sshOnFineMesh))
        exactSSHOnFineRectilinearMesh = np.zeros(myFineMesh.nCells)
        for iCell in range(0,myFineMesh.nCells):
            exactSSHOnFineRectilinearMesh[iCell] = SurfaceElevation(lX,lY,xCellOnFineRectilinearMesh[iCell],
                                                                    yCellOnFineRectilinearMesh[iCell])    
        FineRectilinearMeshSSHInterpolatedToCoarseRectilinearMesh = (
        MeshClass.InterpolateSolutionToCoarsestRectilinearMPASOceanMesh(
        myFineMesh.dx,xCellOnFineRectilinearMesh,yCellOnFineRectilinearMesh,InterpolatedSSHOnFineRectilinearMesh,
        xCellOnCoarseRectilinearMesh,yCellOnCoarseRectilinearMesh))
        L2InterpolationErrorOnCoarseMeshToCoarseRectilinearMesh = (
        InterpolatedSSHOnCoarseRectilinearMesh - exactSSHOnCoarseRectilinearMesh)
        L2InterpolationErrorNormOnCoarseMeshToCoarseRectilinearMesh = (
        np.linalg.norm(L2InterpolationErrorOnCoarseMeshToCoarseRectilinearMesh)/np.sqrt(float(myCoarseMesh.nCells)))
        print('The L2 interpolation error norm from the coarse mesh to the coarse rectilinear mesh is %.15f.' 
              %L2InterpolationErrorNormOnCoarseMeshToCoarseRectilinearMesh)
        L2InterpolationErrorOnFineMeshToFineRectilinearMesh = (
        InterpolatedSSHOnFineRectilinearMesh - exactSSHOnFineRectilinearMesh)
        L2InterpolationErrorNormOnFineMeshToFineRectilinearMesh = (
        np.linalg.norm(L2InterpolationErrorOnFineMeshToFineRectilinearMesh)/np.sqrt(float(myFineMesh.nCells)))
        print('The L2 interpolation error norm from the fine mesh to the fine rectilinear mesh is %.15f.' 
              %L2InterpolationErrorNormOnFineMeshToFineRectilinearMesh)    
        L2InterpolationErrorOnFineRectilinearMeshToCoarseRectilinearMesh = (
        FineRectilinearMeshSSHInterpolatedToCoarseRectilinearMesh - exactSSHOnCoarseRectilinearMesh)
        L2InterpolationErrorNormOnFineRectilinearMeshToCoarseRectilinearMesh = (
        (np.linalg.norm(L2InterpolationErrorOnFineRectilinearMeshToCoarseRectilinearMesh)
         /np.sqrt(float(myCoarseMesh.nCells))))
        print('The L2 interpolation error norm from the fine rectilinear mesh to the coarse rectilinear mesh is %.15f.'
              %L2InterpolationErrorNormOnFineRectilinearMeshToCoarseRectilinearMesh)    
        if PlotFigures:
            OutputDirectory = MeshDirectory
            nContours = 300
            xLabel = 'Zonal Distance (km)'
            yLabel = 'Meridional Distance (km)'
            labels = [xLabel,yLabel]
            labelfontsizes = [22.5,22.5]
            labelpads = [10.0,10.0]
            tickfontsizes = [15.0,15.0]
            useGivenColorBarLimits = False
            ColorBarLimits = [0.0,0.0]
            nColorBarTicks = 6
            xLabel = 'Zonal Distance (km)'
            yLabel = 'Meridional Distance (km)'
            labels = [xLabel,yLabel]
            titlefontsize = 27.5
            SaveAsPDF = True
            Show = False
            DataType = 'Unstructured'
            Title = 'Surface Elevation on Coarse Mesh'
            FileName = 'SurfaceElevationOnCoarseMesh_' + BoundaryCondition
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,myCoarseMesh.xCell/1000.0,myCoarseMesh.yCell/1000.0,
                                                  sshOnCoarseMesh,nContours,labels,labelfontsizes,labelpads,
                                                  tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,
                                                  Title,titlefontsize,SaveAsPDF,FileName,Show,DataType=DataType)
            Title = 'Exact Surface Elevation\non Coarse Rectilinear Mesh'
            FileName = 'ExactSurfaceElevationOnCoarseRectilinearMesh_' + BoundaryCondition
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,xCellOnCoarseRectilinearMesh/1000.0,
                                                  yCellOnCoarseRectilinearMesh/1000.0,exactSSHOnCoarseRectilinearMesh,
                                                  nContours,labels,labelfontsizes,labelpads,tickfontsizes,
                                                  useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,
                                                  titlefontsize,SaveAsPDF,FileName,Show,DataType=DataType)
            Title = 'Interpolated Surface Elevation\non Coarse Rectilinear Mesh'
            FileName = 'InterpolatedSurfaceElevationOnCoarseRectilinearMesh_' + BoundaryCondition
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,xCellOnCoarseRectilinearMesh/1000.0,
                                                  yCellOnCoarseRectilinearMesh/1000.0,
                                                  InterpolatedSSHOnCoarseRectilinearMesh,nContours,labels,
                                                  labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
                                                  ColorBarLimits,nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,
                                                  Show,DataType=DataType) 
            Title = 'Surface Elevation on Fine Mesh'
            FileName = 'SurfaceElevationOnFineMesh'
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,myFineMesh.xCell/1000.0,myFineMesh.yCell/1000.0,
                                                  sshOnFineMesh,nContours,labels,labelfontsizes,labelpads,
                                                  tickfontsizes,useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,
                                                  Title,titlefontsize,SaveAsPDF,FileName,Show,DataType=DataType)
            Title = 'Interpolated Surface Elevation\non Fine Rectilinear Mesh'
            FileName = 'InterpolatedSurfaceElevationOnFineRectilinearMesh_' + BoundaryCondition
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,xCellOnFineRectilinearMesh/1000.0,
                                                  yCellOnFineRectilinearMesh/1000.0,
                                                  InterpolatedSSHOnFineRectilinearMesh,nContours,labels,labelfontsizes,
                                                  labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
                                                  nColorBarTicks,Title,titlefontsize,SaveAsPDF,FileName,Show,
                                                  DataType=DataType)
            Title = 'Fine Rectilinear Mesh Surface Elevation\nInterpolated to Coarse Rectilinear Mesh'
            FileName = 'FineRectilinearMeshSurfaceElevationInterpolatedToCoarseRectilinearMesh_' + BoundaryCondition
            CR.PythonFilledContourPlot2DSaveAsPDF(OutputDirectory,xCellOnCoarseRectilinearMesh/1000.0,
                                                  yCellOnCoarseRectilinearMesh/1000.0,
                                                  FineRectilinearMeshSSHInterpolatedToCoarseRectilinearMesh,
                                                  nContours,labels,labelfontsizes,labelpads,tickfontsizes,
                                                  useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,Title,
                                                  titlefontsize,SaveAsPDF,FileName,Show,DataType=DataType)
        
        
do_TestInterpolateSolutionToCoarsestRectilinearMPASOceanMesh = False
if do_TestInterpolateSolutionToCoarsestRectilinearMPASOceanMesh:
    TestInterpolateSolutionToCoarsestRectilinearMPASOceanMesh()