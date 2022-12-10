"""
Name: QuadMeshClass.py
Author: Sid Bishnu
Details: This script defines the quadrilateral mesh class for two-dimensional spectral element methods.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import sys
from IPython.utils import io
with io.capture_output() as captured:
    import HashTableClass as HT
    import GeometryBasics2DClass as GB2D
    import CornerNode2DClass as CN2D
    import QuadElementClass as QE
    import EdgeClass as EC


class QuadMeshParameters:
    
    def __init__(myQuadMeshParameters):
        myQuadMeshParameters.NONE = 10**5
        myQuadMeshParameters.INTERIOR = 10**6
        myQuadMeshParameters.DIRICHLET_SOUTH = 10**7
        myQuadMeshParameters.DIRICHLET_EAST = 10**8
        myQuadMeshParameters.DIRICHLET_NORTH = 10**9
        myQuadMeshParameters.DIRICHLET_WEST = 10**10


class QuadMesh:
    
    def __init__(myQuadMesh,lX,lY,nElementsX,nElementsY,myDGNodalStorage2D,ProblemType,ProblemType_EquatorialWave=False,
                 PrintEdgeProperties=False):
        myQuadMesh.myQuadMeshParameters = QuadMeshParameters()
        myQuadMesh.lX = lX
        myQuadMesh.lY = lY
        myQuadMesh.nElementsX = nElementsX
        myQuadMesh.nElementsY = nElementsY
        nElements = nElementsX*nElementsY
        myQuadMesh.nElements = nElements
        myQuadMesh.dx = lX/float(nElementsX)
        myQuadMesh.dy = lY/float(nElementsY)
        myQuadMesh.myQuadElements = np.empty(nElements,dtype=QE.QuadElement) 
        nNodes = (nElementsX + 1)*(nElementsY + 1)
        myQuadMesh.nNodes = nNodes
        myQuadMesh.myCornerNodes = np.empty(nNodes,dtype=CN2D.CornerNode)
        nEdgesByFormula = nElements + nNodes - 1 # This is true for a structured mesh with no holes as in our case.
        myQuadMesh.myEdges = np.empty(nEdgesByFormula,dtype=EC.Edge)
        nXi = myDGNodalStorage2D.nXi
        nEta = myDGNodalStorage2D.nEta
        myQuadMesh.SideMap = np.array([0,nXi,nEta,0],dtype=int)
        # The sidemap of a side is nothing but the coordinate of every point on that side which remains constant.
        myQuadMesh.CornerMap = np.zeros((2,4),dtype=int)
        myQuadMesh.CornerMap[0,:] = np.array([0,nXi,nXi,0],dtype=int)
        myQuadMesh.CornerMap[1,:] = np.array([0,0,nEta,nEta],dtype=int)
        # The CornerMaps are nothing but the coordinates of each corner viz. {(0,0), (nXi,0), (nXi,nEta) and (0,nEta)}.
        myQuadMesh.EdgeMap = np.zeros((2,4),dtype=int)
        myQuadMesh.EdgeMap[0,:] = np.array([1,2,4,1],dtype=int)
        myQuadMesh.EdgeMap[1,:] = np.array([2,3,3,4],dtype=int)
        # The EdgeMap is just the set of local NodeIDs connected by each edge viz. {(1,2), (2,3), (4,3) and (1,4)} 
        # where the ordering of the local NodeIDs are along the positive xi and eta directions.
        # Construct the corner nodes with their x and y coordinates.          
        myQuadMesh.BuildCornerNodes(ProblemType,ProblemType_EquatorialWave)
        # Construct the elements with elementID, the global NodeIDs of the four corner nodes, boundary curves, and 
        # geometry.
        myQuadMesh.BuildQuadElements(myDGNodalStorage2D)
        myQuadMesh.GetNodeConnectivityAndConstructEdges(PrintEdgeProperties)

    def BuildCornerNodes(myQuadMesh,ProblemType,ProblemType_EquatorialWave=False):
        lX = myQuadMesh.lX
        lY = myQuadMesh.lY
        nElementsX = myQuadMesh.nElementsX
        nElementsY = myQuadMesh.nElementsY
        dx = myQuadMesh.dx
        dy = myQuadMesh.dy
        for iNodeY in range(0,nElementsY+1):
            for iNodeX in range(0,nElementsX+1):
                GlobalNodeID = iNodeY*(nElementsX+1) + iNodeX + 1
                if ProblemType == 'Plane_Gaussian_Wave':
                    xCoordinate = float(iNodeX)*dx - 0.5*lX
                else:
                    xCoordinate = float(iNodeX)*dx
                if ProblemType == 'Plane_Gaussian_Wave' or ProblemType_EquatorialWave:
                    yCoordinate = float(iNodeY)*dy - 0.5*lY
                else:
                    yCoordinate = float(iNodeY)*dy
                myQuadMesh.myCornerNodes[GlobalNodeID-1] = CN2D.CornerNode(xCoordinate,yCoordinate)
                
    def BuildQuadElements(myQuadMesh,myDGNodalStorage2D):
        nElementsX = myQuadMesh.nElementsX
        nElementsY = myQuadMesh.nElementsY
        xCoordinate = np.zeros(4)
        yCoordinate = np.zeros(4)
        nParametricNodes = 2
        ParametricNodes = np.array([-1.0,1.0])
        BoundaryCurve11 = np.zeros(2)
        BoundaryCurve12 = np.zeros(2)
        BoundaryCurve21 = np.zeros(2)
        BoundaryCurve22 = np.zeros(2)
        BoundaryCurve31 = np.zeros(2)
        BoundaryCurve32 = np.zeros(2)
        BoundaryCurve41 = np.zeros(2)
        BoundaryCurve42 = np.zeros(2)
        BoundaryCurve = np.empty(4,dtype=GB2D.CurveInterpolant2D)
        for iElementY in range(0,nElementsY):
            ElementIDY = iElementY + 1
            for iElementX in range(0,nElementsX):
                ElementIDX = iElementX + 1
                ElementID = (ElementIDY - 1)*nElementsX + ElementIDX
                NodeIDs = np.zeros(4,dtype=int)
                NodeIDs[0] = (ElementIDY - 1)*(nElementsX + 1) + ElementIDX
                NodeIDs[1] = (ElementIDY - 1)*(nElementsX + 1) + ElementIDX + 1
                NodeIDs[2] = ElementIDY*(nElementsX + 1) + ElementIDX + 1
                NodeIDs[3] = ElementIDY*(nElementsX + 1) + ElementIDX               
                xCoordinate[0] = myQuadMesh.myCornerNodes[NodeIDs[0]-1].x
                yCoordinate[0] = myQuadMesh.myCornerNodes[NodeIDs[0]-1].y               
                xCoordinate[1] = myQuadMesh.myCornerNodes[NodeIDs[1]-1].x
                yCoordinate[1] = myQuadMesh.myCornerNodes[NodeIDs[1]-1].y
                xCoordinate[2] = myQuadMesh.myCornerNodes[NodeIDs[2]-1].x
                yCoordinate[2] = myQuadMesh.myCornerNodes[NodeIDs[2]-1].y
                xCoordinate[3] = myQuadMesh.myCornerNodes[NodeIDs[3]-1].x
                yCoordinate[3] = myQuadMesh.myCornerNodes[NodeIDs[3]-1].y
                for iParametricNode in range(0,nParametricNodes):
                    BoundaryCurve11[iParametricNode] = (
                    xCoordinate[0] + 0.5*(xCoordinate[1] - xCoordinate[0])*(ParametricNodes[iParametricNode] + 1.0))
                    BoundaryCurve12[iParametricNode] = (
                    yCoordinate[0] + 0.5*(yCoordinate[1] - yCoordinate[0])*(ParametricNodes[iParametricNode] + 1.0))
                    BoundaryCurve21[iParametricNode] = (
                    xCoordinate[1] + 0.5*(xCoordinate[2] - xCoordinate[1])*(ParametricNodes[iParametricNode] + 1.0))
                    BoundaryCurve22[iParametricNode] = (
                    yCoordinate[1] + 0.5*(yCoordinate[2] - yCoordinate[1])*(ParametricNodes[iParametricNode] + 1.0))
                    BoundaryCurve31[iParametricNode] = (
                    xCoordinate[3] + 0.5*(xCoordinate[2] - xCoordinate[3])*(ParametricNodes[iParametricNode] + 1.0))
                    BoundaryCurve32[iParametricNode] = (
                    yCoordinate[3] + 0.5*(yCoordinate[2] - yCoordinate[3])*(ParametricNodes[iParametricNode] + 1.0))
                    BoundaryCurve41[iParametricNode] = (
                    xCoordinate[0] + 0.5*(xCoordinate[3] - xCoordinate[0])*(ParametricNodes[iParametricNode] + 1.0))
                    BoundaryCurve42[iParametricNode] = (
                    yCoordinate[0] + 0.5*(yCoordinate[3] - yCoordinate[0])*(ParametricNodes[iParametricNode] + 1.0))
                BoundaryCurve[0] = GB2D.CurveInterpolant2D(ParametricNodes,BoundaryCurve11,BoundaryCurve12)
                BoundaryCurve[1] = GB2D.CurveInterpolant2D(ParametricNodes,BoundaryCurve21,BoundaryCurve22)
                BoundaryCurve[2] = GB2D.CurveInterpolant2D(ParametricNodes,BoundaryCurve31,BoundaryCurve32)
                BoundaryCurve[3] = GB2D.CurveInterpolant2D(ParametricNodes,BoundaryCurve41,BoundaryCurve42)
                myQuadMesh.myQuadElements[ElementID-1] = QE.QuadElement(NodeIDs,ElementIDX,ElementIDY,ElementID,
                                                                        BoundaryCurve,myDGNodalStorage2D)
                
    def GetNodeToElement(myQuadMesh):
        nElements = myQuadMesh.nElements
        for iElement in range(0,nElements): # Loop over the elements in the mesh.
            ElementID = iElement + 1
            for iNode in range(0,4): # Loop over the nodes.
                # Add element to the corner node connectivity list.
                NodeID = iNode + 1
                GlobalNodeID = myQuadMesh.myQuadElements[iElement].NodeIDs[iNode]
                myQuadMesh.myCornerNodes[GlobalNodeID-1].NodeToElement.AddToList(ElementID,NodeID)
                # The key is the local node ID, the data is the element ID.

    def ConstructMeshEdges(myQuadMesh,PrintEdgeProperties):
        """
        This function takes in the mesh of quadrilaterals which has not filled in the edge information and finds all of 
        the unique edges in the mesh. The space for the edges is reallocated with the correct number of edges.
        """
        nNodes = myQuadMesh.nNodes
        nElements = myQuadMesh.nElements
        nEdges = 0
        # First just count the number of edges.
        EdgeTable = HT.HashTable(nNodes)
        for iElement in range(0,nElements): # Loop over the elements in the mesh.
            for iSide in range(0,4): # Loop over the sides of each element.
                l1 = myQuadMesh.EdgeMap[0,iSide] # Starting local node for this edge.
                l2 = myQuadMesh.EdgeMap[1,iSide] # Ending local node for this edge.
                startID = myQuadMesh.myQuadElements[iElement].NodeIDs[l1-1]
                endID = myQuadMesh.myQuadElements[iElement].NodeIDs[l2-1]
                key1 = np.min([startID,endID])
                key2 = np.max([startID,endID])
                if not(EdgeTable.ContainsKeys(key1,key2)): # This is a new edge. Add the edge to the list.
                    nEdges += 1
                    EdgeTable.AddDataForKeys(nEdges,key1,key2)         
        EdgeTable.DestructHashTable() # Trash the EdgeTable.
        EdgeTable = HT.HashTable(nNodes) # And rebuild it.
        NONE = myQuadMesh.myQuadMeshParameters.NONE
        # Reallocate space for the mesh edges.
        myQuadMesh.myEdges = np.empty(nEdges,dtype=EC.Edge)
        for iEdge in range(0,nEdges):
            myQuadMesh.myEdges[iEdge] = EC.Edge()
        nEdges = 0 # Restart the edge counting.
        for iElement in range(0,nElements): # Loop over the elements in the mesh.
            for iSide in range(0,4): # Loop over the sides of each element.
                l1 = myQuadMesh.EdgeMap[0,iSide] # Starting local node for this edge.
                l2 = myQuadMesh.EdgeMap[1,iSide] # Ending local node for this edge.
                startID = myQuadMesh.myQuadElements[iElement].NodeIDs[l1-1]               
                endID = myQuadMesh.myQuadElements[iElement].NodeIDs[l2-1]
                key1 = np.min([startID,endID])
                key2 = np.max([startID,endID])
                if EdgeTable.ContainsKeys(key1,key2): # This edge already exists. Get the EdgeID.
                    EdgeID = EdgeTable.GetDataForKeys(key1,key2)
                    iEdge = EdgeID - 1
                    # Find the primary element and the starting node for this element's edge. This is compared with the
                    # secondary element's starting node to infer the relative orientation of the two elements.
                    e1 = myQuadMesh.myEdges[iEdge].ElementIDs[0]
                    s1 = myQuadMesh.myEdges[iEdge].ElementSides[0]
                    l1 = myQuadMesh.EdgeMap[0,s1-1]
                    n1 = myQuadMesh.myQuadElements[e1-1].NodeIDs[l1-1]
                    # Set the secondary element information.
                    myQuadMesh.myEdges[iEdge].ElementIDs[1] = iElement + 1
                    if startID == n1: # The elements are oriented the same direction.
                        myQuadMesh.myEdges[iEdge].ElementSides[1] = iSide + 1
                    else:
                        # The elements are oriented in the opposite direction. For these edges, we mark the side ID as 
                        # negative. A post-process step will mark:
                        # myQuadMesh.myEdges[iEdge].start = nXi - 1, myQuadMesh.myEdges[iEdge].increment = -1.
                        myQuadMesh.myEdges[iEdge].ElementSides[1] = -(iSide + 1)       
                else: 
                    nEdges += 1
                    EdgeID = nEdges
                    iEdge = EdgeID - 1
                    NodeIDs = np.zeros(2,dtype=int)
                    NodeIDs[0] = startID
                    NodeIDs[1] = endID
                    myQuadMesh.myEdges[iEdge].OverwriteEdgeProperties(
                    EdgeID,NodeIDs,np.array([iElement+1,NONE],dtype=int),np.array([iSide+1,NONE],dtype=int))
                    EdgeTable.AddDataForKeys(EdgeID,key1,key2)
        for iElement in range(0,nElements): # Loop over the elements in the mesh.
            for iSide in range(0,4): # Loop over the sides of each element.
                l1 = myQuadMesh.EdgeMap[0,iSide] # Starting local node for this edge.
                l2 = myQuadMesh.EdgeMap[1,iSide] # Ending local node for this edge.
                startID = myQuadMesh.myQuadElements[iElement].NodeIDs[l1-1]
                endID = myQuadMesh.myQuadElements[iElement].NodeIDs[l2-1]
                key1 = np.min([startID,endID])
                key2 = np.max([startID,endID])
        myQuadMesh.nEdges = nEdges
        EdgeTable.DestructHashTable()
        if PrintEdgeProperties:
            print('Checking correct assignment of element IDs and element sides to the edges ' 
                  + 'before specifying boundary edges!')
            print('iEdge EdgeID NodeIDs ElementIDs ElementSides')
            for iEdge in range(0,myQuadMesh.nEdges):
                print('%2d %2d [%2d %2d] [%6d %6d] [%6d %6d]' 
                    %(iEdge,myQuadMesh.myEdges[iEdge].EdgeID,myQuadMesh.myEdges[iEdge].NodeIDs[0],
                      myQuadMesh.myEdges[iEdge].NodeIDs[1],myQuadMesh.myEdges[iEdge].ElementIDs[0],
                      myQuadMesh.myEdges[iEdge].ElementIDs[1],myQuadMesh.myEdges[iEdge].ElementSides[0],
                      myQuadMesh.myEdges[iEdge].ElementSides[1]))

    def GenerateMeshEdges(myQuadMesh,PrintEdgeProperties):
        nNodes = myQuadMesh.nNodes
        nElements = myQuadMesh.nElements
        nEdges = nElements + nNodes - 1 # This is true for a structured mesh with no holes, as in our case.
        myQuadMesh.nEdges = nEdges
        NONE = myQuadMesh.myQuadMeshParameters.NONE
        E = np.ones((nNodes,nNodes),dtype=int)*NONE
        EdgeID = 0
        for iEdge in range(0,nEdges):
            myQuadMesh.myEdges[iEdge] = EC.Edge()
            myQuadMesh.myEdges[iEdge].ElementIDs = np.array([NONE,NONE],dtype=int)
            myQuadMesh.myEdges[iEdge].ElementSides = np.array([NONE,NONE],dtype=int)
        for iElement in range(0,nElements):
            for iSide in range(0,4):
                LocalNodeIDs = myQuadMesh.EdgeMap[:,iSide]
                NodeIDs = np.array([myQuadMesh.myQuadElements[iElement].NodeIDs[LocalNodeIDs[0]-1],
                                    myQuadMesh.myQuadElements[iElement].NodeIDs[LocalNodeIDs[1]-1]],dtype=int)
                key1 = np.min(NodeIDs)
                key2 = np.max(NodeIDs)
                if E[key1-1,key2-1] == NONE:
                    EdgeID += 1
                    iEdge = EdgeID - 1
                    myQuadMesh.myEdges[iEdge].EdgeID = EdgeID
                    myQuadMesh.myEdges[iEdge].NodeIDs = np.array([key1,key2],dtype=int)
                    myQuadMesh.myEdges[iEdge].ElementIDs[0] = iElement + 1
                    myQuadMesh.myEdges[iEdge].ElementSides[0] = iSide + 1
                    E[key1-1,key2-1] = EdgeID
                else:
                    EID = E[key1-1,key2-1]
                    iEdge = EID - 1
                    myQuadMesh.myEdges[iEdge].ElementIDs[1] = iElement + 1
                    myQuadMesh.myEdges[iEdge].ElementSides[1] = iSide + 1
        if PrintEdgeProperties:
            print('Checking correct assignment of element IDs and element sides to the edges ' 
                  + 'before specifying boundary edges!')
            print('iEdge EdgeID NodeIDs ElementIDs ElementSides')
            for iEdge in range(0,myQuadMesh.nEdges):
                print('%2d %2d [%2d %2d] [%6d %6d] [%6d %6d]' 
                    %(iEdge,myQuadMesh.myEdges[iEdge].EdgeID,myQuadMesh.myEdges[iEdge].NodeIDs[0],
                      myQuadMesh.myEdges[iEdge].NodeIDs[1],myQuadMesh.myEdges[iEdge].ElementIDs[0],
                      myQuadMesh.myEdges[iEdge].ElementIDs[1],myQuadMesh.myEdges[iEdge].ElementSides[0],
                      myQuadMesh.myEdges[iEdge].ElementSides[1]))
         
    def GetNodeConnectivityAndConstructEdges(myQuadMesh,PrintEdgeProperties):
        nXi = myQuadMesh.myQuadElements[0].myMappedGeometry2D.nXi
        myQuadMesh.GetNodeToElement()
        DefaultOption = True
        if DefaultOption:
            myQuadMesh.ConstructMeshEdges(PrintEdgeProperties)
        else: 
            myQuadMesh.GenerateMeshEdges(PrintEdgeProperties)
        NONE = myQuadMesh.myQuadMeshParameters.NONE
        INTERIOR = myQuadMesh.myQuadMeshParameters.INTERIOR
        DIRICHLET_WEST = myQuadMesh.myQuadMeshParameters.DIRICHLET_WEST
        DIRICHLET_EAST = myQuadMesh.myQuadMeshParameters.DIRICHLET_EAST
        DIRICHLET_SOUTH = myQuadMesh.myQuadMeshParameters.DIRICHLET_SOUTH
        DIRICHLET_NORTH = myQuadMesh.myQuadMeshParameters.DIRICHLET_NORTH
        for iEdge in range(0,myQuadMesh.nEdges): # Loop over the edges.
            myQuadMesh.myEdges[iEdge].EdgeType = INTERIOR
        # Specify the types of boundary conditions.
        for iEdge in range(0,myQuadMesh.nEdges): # Loop over the edges.
            ElementSide1 = myQuadMesh.myEdges[iEdge].ElementSides[0]
            ElementID2 = myQuadMesh.myEdges[iEdge].ElementIDs[1]
            ElementSide2 = myQuadMesh.myEdges[iEdge].ElementSides[1]
            # Specify the boundary conditions.
            if ElementID2 == NONE: # This edge is a physical boundary.
                NodeID1 = myQuadMesh.myEdges[iEdge].NodeIDs[0]
                NodeID2 = myQuadMesh.myEdges[iEdge].NodeIDs[1]
                iNode1 = NodeID1 - 1
                iNode2 = NodeID2 - 1
                if ElementSide1 == 1:
                    myQuadMesh.myEdges[iEdge].ElementIDs[1] = DIRICHLET_SOUTH
                    myQuadMesh.myEdges[iEdge].ElementSides[1] = DIRICHLET_SOUTH
                    myQuadMesh.myEdges[iEdge].EdgeType = DIRICHLET_SOUTH
                    myQuadMesh.myCornerNodes[iNode1].NodeType = DIRICHLET_SOUTH
                    myQuadMesh.myCornerNodes[iNode2].NodeType = DIRICHLET_SOUTH
                elif ElementSide1 == 2:
                    myQuadMesh.myEdges[iEdge].ElementIDs[1] = DIRICHLET_EAST
                    myQuadMesh.myEdges[iEdge].ElementSides[1] = DIRICHLET_EAST
                    myQuadMesh.myEdges[iEdge].EdgeType = DIRICHLET_EAST
                    myQuadMesh.myCornerNodes[iNode1].NodeType = DIRICHLET_EAST
                    myQuadMesh.myCornerNodes[iNode2].NodeType = DIRICHLET_EAST
                elif ElementSide1 == 3:
                    myQuadMesh.myEdges[iEdge].ElementIDs[1] = DIRICHLET_NORTH
                    myQuadMesh.myEdges[iEdge].ElementSides[1] = DIRICHLET_NORTH
                    myQuadMesh.myEdges[iEdge].EdgeType = DIRICHLET_NORTH
                    myQuadMesh.myCornerNodes[iNode1].NodeType = DIRICHLET_NORTH
                    myQuadMesh.myCornerNodes[iNode2].NodeType = DIRICHLET_NORTH
                elif ElementSide1 == 4:
                    myQuadMesh.myEdges[iEdge].ElementIDs[1] = DIRICHLET_WEST
                    myQuadMesh.myEdges[iEdge].ElementSides[1] = DIRICHLET_WEST
                    myQuadMesh.myEdges[iEdge].EdgeType = DIRICHLET_WEST
                    myQuadMesh.myCornerNodes[iNode1].NodeType = DIRICHLET_WEST
                    myQuadMesh.myCornerNodes[iNode2].NodeType = DIRICHLET_WEST
                myQuadMesh.myEdges[iEdge].start = 1
                myQuadMesh.myEdges[iEdge].increment = 1  
            else: # We need to exchange information.
                if ElementSide2 > 0:
                    myQuadMesh.myEdges[iEdge].start = 1
                    myQuadMesh.myEdges[iEdge].increment = 1
                else:           
                    myQuadMesh.myEdges[iEdge].start = nXi - 1
                    myQuadMesh.myEdges[iEdge].increment = -1
        for iEdge in range(0,myQuadMesh.nEdges): # Loop over the edges.
            if myQuadMesh.myEdges[iEdge].ElementIDs[1] == NONE:
                print('Not all boundary elements have been labelled yet! Stopping!')
                sys.exit()
    
    def WriteQuadMesh(myQuadMesh,OutputDirectory,filename):
        cwd = os.getcwd()
        path = cwd + '/' + OutputDirectory + '/'
        if not os.path.exists(path):
            os.mkdir(path) # os.makedir(path)
        os.chdir(path)
        filename = filename + '.tec'
        outputfile = open(filename,'w')
        outputfile.write('VARIABLES = "X", "Y", "Jacobian"\n')
        for iElement in range(0,myQuadMesh.nElements):
            nXi = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.nXi
            nEta = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.nEta
            ZoneID = myQuadMesh.myQuadElements[iElement].ElementID
            ZoneIDString = 'Element' + '%7.7d' %ZoneID
            outputfile.write('ZONE T="%s", I=%d, J=%d, F=POINT\n' %(ZoneIDString,nXi+1,nEta+1))
            for iY in range(0,nEta+1):
                for iX in range(0,nXi+1):
                    outputfile.write('%.15g %.15g %.15g\n' 
                                     %(myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iX,iY],
                                       myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iX,iY],
                                       myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.Jacobian[iX,iY]))
        outputfile.close()
        os.chdir(cwd)
        
    def PlotQuadMesh(myQuadMesh,OutputDirectory,linewidth,linestyle,color,marker,markersize,labels,labelfontsizes,
                     labelpads,tickfontsizes,title,titlefontsize,SaveAsPDF,FileName,Show,fig_size=[9.5,9.5],
                     UseDefaultMethodToSpecifyTickFontSize=True,titlepad=1.035):
        cwd = os.getcwd()
        path = cwd + '/' + OutputDirectory + '/'
        if not os.path.exists(path):
            os.mkdir(path) # os.makedir(path)
        os.chdir(path)
        fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
        ax = fig.add_subplot(111) # Create an axes object in the figure
        nElementsX = myQuadMesh.nElementsX
        nElementsY = myQuadMesh.nElementsY
        for iNodeY in range(0,nElementsY+1):
            for iNodeX in range(0,nElementsX+1):
                GlobalNodeID = iNodeY*(nElementsX+1) + iNodeX + 1
                iGlobalNode = GlobalNodeID - 1
                x1 = myQuadMesh.myCornerNodes[iGlobalNode].x
                y1 = myQuadMesh.myCornerNodes[iGlobalNode].y
                GlobalNodeIDOnRight = GlobalNodeID + 1
                iGlobalNodeOnRight = GlobalNodeIDOnRight - 1
                GlobalNodeIDOnTop = GlobalNodeID + nElementsX + 1
                iGlobalNodeOnTop = GlobalNodeIDOnTop - 1
                if GlobalNodeIDOnRight <= (nElementsX+1)*(nElementsY+1):
                    x2 = myQuadMesh.myCornerNodes[iGlobalNodeOnRight].x
                    y2 = myQuadMesh.myCornerNodes[iGlobalNodeOnRight].y
                    if x2 > x1 and y2 == y1:
                        x = np.array([x1,x2])
                        y = np.array([y1,y2])
                        plt.plot(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=None)
                if GlobalNodeIDOnTop <= (nElementsX+1)*(nElementsY+1):
                    x2 = myQuadMesh.myCornerNodes[iGlobalNodeOnTop].x
                    y2 = myQuadMesh.myCornerNodes[iGlobalNodeOnTop].y
                    if x2 == x1 and y2 > y1:
                        x = np.array([x1,x2])
                        y = np.array([y1,y2])
                        plt.plot(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=None)
        for iElement in range(0,myQuadMesh.nElements):
            nXi = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.nXi
            nEta = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.nEta
            for iY in range(0,nEta+1):
                for iX in range(0,nXi+1):
                    x1 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iX,iY]
                    y1 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iX,iY]
                    plt.plot([x1],[y1],color=color,marker=marker,markersize=markersize)
                    if iX < nXi:
                        x2 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iX+1,iY]
                        y2 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iX+1,iY]
                        x = np.array([x1,x2])
                        y = np.array([y1,y2])
                        plt.plot(x,y,linewidth=0.5*linewidth,linestyle=linestyle,color=color,marker=None)
                    if iY < nEta:
                        x2 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iX,iY+1]
                        y2 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iX,iY+1]
                        x = np.array([x1,x2])
                        y = np.array([y1,y2])
                        plt.plot(x,y,linewidth=0.5*linewidth,linestyle=linestyle,color=color,marker=None)
        plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
        plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
        if UseDefaultMethodToSpecifyTickFontSize:
            plt.xticks(fontsize=tickfontsizes[0])
            plt.yticks(fontsize=tickfontsizes[1])
        else:
            ax.tick_params(axis='x',labelsize=tickfontsizes[0])
            ax.tick_params(axis='y',labelsize=tickfontsizes[1])
        plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/1000.0), '')))
        plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda y, p: format(int(y/1000.0), '')))
        ax.set_title(title,fontsize=titlefontsize,fontweight='bold',y=titlepad)
        if SaveAsPDF:
            plt.savefig(FileName+'.pdf',format='pdf',bbox_inches='tight')
        if Show:
            plt.show()
        plt.close()
        os.chdir(cwd)

        
def PlotQuadMeshes(myCoarseQuadMesh,myFineQuadMesh,OutputDirectory,linewidths,linestyles,colors,markers,markersizes,
                   labels,labelfontsizes,labelpads,tickfontsizes,title,titlefontsize,SaveAsPDF,FileName,Show,
                   fig_size=[9.5,9.5],UseDefaultMethodToSpecifyTickFontSize=True,titlepad=1.035):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    myQuadMeshes = [myCoarseQuadMesh,myFineQuadMesh]
    for iQuadMesh in range(0,2):
        myQuadMesh = myQuadMeshes[iQuadMesh]
        nElementsX = myQuadMesh.nElementsX
        nElementsY = myQuadMesh.nElementsY
        linewidth = linewidths[iQuadMesh]
        linestyle = linestyles[iQuadMesh]
        color = colors[iQuadMesh]
        marker = markers[iQuadMesh]
        markersize = markersizes[iQuadMesh]
        for iNodeY in range(0,nElementsY+1):
            for iNodeX in range(0,nElementsX+1):
                GlobalNodeID = iNodeY*(nElementsX+1) + iNodeX + 1
                iGlobalNode = GlobalNodeID - 1
                x1 = myQuadMesh.myCornerNodes[iGlobalNode].x
                y1 = myQuadMesh.myCornerNodes[iGlobalNode].y
                GlobalNodeIDOnRight = GlobalNodeID + 1
                iGlobalNodeOnRight = GlobalNodeIDOnRight - 1
                GlobalNodeIDOnTop = GlobalNodeID + nElementsX + 1
                iGlobalNodeOnTop = GlobalNodeIDOnTop - 1
                if GlobalNodeIDOnRight <= (nElementsX+1)*(nElementsY+1):
                    x2 = myQuadMesh.myCornerNodes[iGlobalNodeOnRight].x
                    y2 = myQuadMesh.myCornerNodes[iGlobalNodeOnRight].y
                    if x2 > x1 and y2 == y1:
                        x = np.array([x1,x2])
                        y = np.array([y1,y2])
                        plt.plot(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=None)
                if GlobalNodeIDOnTop <= (nElementsX+1)*(nElementsY+1):
                    x2 = myQuadMesh.myCornerNodes[iGlobalNodeOnTop].x
                    y2 = myQuadMesh.myCornerNodes[iGlobalNodeOnTop].y
                    if x2 == x1 and y2 > y1:
                        x = np.array([x1,x2])
                        y = np.array([y1,y2])
                        plt.plot(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=None)
        for iElement in range(0,myQuadMesh.nElements):
            nXi = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.nXi
            nEta = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.nEta
            for iY in range(0,nEta+1):
                for iX in range(0,nXi+1):
                    x1 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iX,iY]
                    y1 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iX,iY]
                    plt.plot([x1],[y1],color=color,marker=marker,markersize=markersize)
                    if iX < nXi:
                        x2 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iX+1,iY]
                        y2 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iX+1,iY]
                        x = np.array([x1,x2])
                        y = np.array([y1,y2])
                        plt.plot(x,y,linewidth=0.5*linewidth,linestyle=linestyle,color=color,marker=None)
                    if iY < nEta:
                        x2 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iX,iY+1]
                        y2 = myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iX,iY+1]
                        x = np.array([x1,x2])
                        y = np.array([y1,y2])
                        plt.plot(x,y,linewidth=0.5*linewidth,linestyle=linestyle,color=color,marker=None)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    if UseDefaultMethodToSpecifyTickFontSize:
        plt.xticks(fontsize=tickfontsizes[0])
        plt.yticks(fontsize=tickfontsizes[1])
    else:
        ax.tick_params(axis='x',labelsize=tickfontsizes[0])
        ax.tick_params(axis='y',labelsize=tickfontsizes[1])
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/1000.0), '')))
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda y, p: format(int(y/1000.0), '')))
    ax.set_title(title,fontsize=titlefontsize,fontweight='bold',y=titlepad)
    if SaveAsPDF:
        plt.savefig(FileName+'.pdf',format='pdf',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)