"""
Name: QuadElementClass.py
Author: Sid Bishnu
Details: This script defines the quadrilateral element class for two-dimensional spectral element methods.
"""


import numpy as np
from IPython.utils import io
with io.capture_output() as captured:
    import MappedGeometry2DClass as MG2D


class QuadElement:
    
    def __init__(myQuadElement,NodeIDs,ElementIDX,ElementIDY,ElementID,BoundaryCurve,myDGNodalStorage2D,
                 QuadElementType='CurvedSidedQuadrilateral',xCoordinate=np.zeros(4),yCoordinate=np.zeros(4)):
        myQuadElement.NodeIDs = NodeIDs
        myQuadElement.ElementIDX = ElementIDX
        myQuadElement.ElementIDY = ElementIDY
        myQuadElement.ElementID = ElementID
        myQuadElement.myMappedGeometry2D = MG2D.MappedGeometry2D(myDGNodalStorage2D,BoundaryCurve,QuadElementType,
                                                                 xCoordinate,yCoordinate)