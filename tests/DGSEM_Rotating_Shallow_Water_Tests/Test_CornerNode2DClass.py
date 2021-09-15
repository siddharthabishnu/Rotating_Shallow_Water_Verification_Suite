"""
Name: Test_CornerNode2DClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the corner node class defined in 
../../src/DGSEM_Rotating_Shallow_Water/.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import CornerNode2DClass


def TestCornerNode():
    myCornerNode = CornerNode2DClass.CornerNode(1.0,1.0)
    myCornerNode.ConstructEmptyCornerNode2D()
    
    
do_TestCornerNode = False
if do_TestCornerNode:
    TestCornerNode()