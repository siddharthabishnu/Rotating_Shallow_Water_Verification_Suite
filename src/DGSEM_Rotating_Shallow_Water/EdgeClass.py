"""
Name: EdgeClass.py
Author: Sid Bishnu
Details: This script defines the edge class for two-dimensional spectral element methods.
"""


import numpy as np


class Edge:
    
    def __init__(myEdge,EdgeID=0,NodeIDs=np.array([0,0],dtype=int),ElementIDs=np.array([0,0],dtype=int),
                 ElementSides=np.array([0,0],dtype=int)):
        myEdge.EdgeType = 0
        myEdge.EdgeID = EdgeID
        myEdge.NodeIDs = NodeIDs
        myEdge.ElementIDs = ElementIDs
        myEdge.ElementSides = ElementSides
        myEdge.start = 0
        myEdge.increment = 0
        
    def OverwriteEdgeProperties(myEdge,EdgeID,NodeIDs,ElementIDs,ElementSides):
        myEdge.EdgeID = EdgeID
        myEdge.NodeIDs = NodeIDs
        myEdge.ElementIDs = ElementIDs
        myEdge.ElementSides = ElementSides