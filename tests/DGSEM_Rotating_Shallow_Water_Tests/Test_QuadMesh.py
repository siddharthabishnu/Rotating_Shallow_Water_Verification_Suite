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
    lX = 1.0
    lY = 1.0
    nElementsX = 2
    nElementsY = 2
    nXi = 10
    nEta = 10
    myDGNodalStorage2D = DGNS2D.DGNodalStorage2D(nXi,nEta)
    ProblemType = 'Coastal_Kelvin_Wave'
    ProblemType_EquatorialWave = False
    PrintEdgeProperties = True
    myQuadMesh = QM.QuadMesh(lX,lY,nElementsX,nElementsY,myDGNodalStorage2D,ProblemType,ProblemType_EquatorialWave,
                             PrintEdgeProperties)
    print('Checking correct assignment of element IDs and element sides to the edges after specifying boundary edges!')
    print('iEdge EdgeID NodeIDs ElementIDs ElementSides')
    for iEdge in range(0,myQuadMesh.nEdges):
        print('%2d %2d [%2d %2d] [%11d %11d] [%11d %11d]' 
              %(iEdge,myQuadMesh.myEdges[iEdge].EdgeID,myQuadMesh.myEdges[iEdge].NodeIDs[0],
                myQuadMesh.myEdges[iEdge].NodeIDs[1],myQuadMesh.myEdges[iEdge].ElementIDs[0],
                myQuadMesh.myEdges[iEdge].ElementIDs[1],myQuadMesh.myEdges[iEdge].ElementSides[0],
                myQuadMesh.myEdges[iEdge].ElementSides[1]))
    output_directory = '../../output/DGSEM_Rotating_Shallow_Water_Output/'
    filename = 'myQuadMesh'
    myQuadMesh.WriteQuadMesh(output_directory,filename)
    

do_TestQuadMesh = False
if do_TestQuadMesh:
    TestQuadMesh()