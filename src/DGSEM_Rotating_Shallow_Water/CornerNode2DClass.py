"""
Name: CornerNode2DClass.py
Author: Sid Bishnu
Details: This script defines the corner node class for two-dimensional spectral element methods.
"""


from IPython.utils import io
with io.capture_output() as captured:
    import LinkedListClass as LL


class CornerNode:
    
    def __init__(myCornerNode,xCoordinate,yCoordinate):
        myCornerNode.NodeType = 'Interior' # Default the node type to 'Interior'.
        myCornerNode.x = xCoordinate
        myCornerNode.y = yCoordinate
        # Initialize the NodeToElement linked-list so that it initially points to None.
        myCornerNode.NodeToElement = LL.LinkedList()
        
    def ConstructEmptyCornerNode2D(myCornerNode):
        myCornerNode.NodeType = 'Interior' # Default the node type to 'Interior'.
        myCornerNode.x = 0.0
        myCornerNode.y = 0.0
        # Initialize the NodeToElement linked-list so that it initially points to None.
        myCornerNode.NodeToElement = LL.LinkedList()