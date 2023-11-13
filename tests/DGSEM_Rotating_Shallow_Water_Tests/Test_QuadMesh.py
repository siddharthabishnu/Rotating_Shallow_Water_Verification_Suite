"""
Name: Test_QuadMeshClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the quadrilateral mesh class defined in 
../../src/DGSEM_Rotating_Shallow_Water/QuadMeshClass.py.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import DGNodalStorage2DClass as DGNS2D
    import QuadMeshClass as QM
    
    
def TestQuadMesh():
    lX = 40.0*10**3.0
    lY = 40.0*10**3.0
    nElementsX = 2
    nElementsY = 2
    nXi = 4
    nEta = 4
    myDGNodalStorage2D = DGNS2D.DGNodalStorage2D(nXi,nEta)
    ProblemType_EquatorialWave = False
    QuadElementType = 'CurvedSidedQuadrilateral'
    PrintEdgeProperties = True
    myQuadMesh = QM.QuadMesh(lX,lY,nElementsX,nElementsY,myDGNodalStorage2D,ProblemType_EquatorialWave,QuadElementType,
                             PrintEdgeProperties)
    print('Checking correct assignment of element IDs and element sides to the edges after specifying boundary edges!')
    print('iEdge EdgeID NodeIDs ElementIDs ElementSides')
    for iEdge in range(0,myQuadMesh.nEdges):
        print('%2d %2d [%2d %2d] [%11d %11d] [%11d %11d]' 
              %(iEdge,myQuadMesh.myEdges[iEdge].EdgeID,myQuadMesh.myEdges[iEdge].NodeIDs[0],
                myQuadMesh.myEdges[iEdge].NodeIDs[1],myQuadMesh.myEdges[iEdge].ElementIDs[0],
                myQuadMesh.myEdges[iEdge].ElementIDs[1],myQuadMesh.myEdges[iEdge].ElementSides[0],
                myQuadMesh.myEdges[iEdge].ElementSides[1]))
    OutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output/'
    FileName = 'myQuadMesh'
    myQuadMesh.WriteQuadMesh(OutputDirectory,FileName)
    linewidth = 2.5
    linestyle = '-'
    color = 'k'
    marker = 's'
    markersize = 5.0
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    labels = [xLabel,yLabel]
    labelfontsizes = [22.5,22.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    title = 'Quadrilateral Mesh for DGSEM'
    titlefontsize = 27.5
    SaveAsPDF = True
    FileName = 'DGSEMQuadMesh'
    Show = False
    myQuadMesh.PlotQuadMesh(OutputDirectory,linewidth,linestyle,color,marker,markersize,labels,labelfontsizes,labelpads,
                            tickfontsizes,title,titlefontsize,SaveAsPDF,FileName,Show)
    nXi = 3
    nEta = 3
    myDGNodalStorage2D = DGNS2D.DGNodalStorage2D(nXi,nEta)
    PrintEdgeProperties = False
    myCoarseQuadMesh = QM.QuadMesh(lX,lY,nElementsX,nElementsY,myDGNodalStorage2D,ProblemType_EquatorialWave,
                                   QuadElementType,PrintEdgeProperties)
    nElementsX = 4
    nElementsY = 4
    myFineQuadMesh = QM.QuadMesh(lX,lY,nElementsX,nElementsY,myDGNodalStorage2D,ProblemType_EquatorialWave,
                                 QuadElementType,PrintEdgeProperties)
    linewidths = [2.5,2.5]
    linestyles = ['-','--']
    colors = ['r','b']
    markers = ['s','s']
    markersizes = [7.5,5.0]
    title = 'Quadrilateral Meshes for DGSEM'
    FileName = 'DGSEMQuadMeshes'
    QM.PlotQuadMeshes(myCoarseQuadMesh,myFineQuadMesh,OutputDirectory,linewidths,linestyles,colors,markers,markersizes,
                      labels,labelfontsizes,labelpads,tickfontsizes,title,titlefontsize,SaveAsPDF,FileName,Show)
    
    
do_TestQuadMesh = False
if do_TestQuadMesh:
    TestQuadMesh()