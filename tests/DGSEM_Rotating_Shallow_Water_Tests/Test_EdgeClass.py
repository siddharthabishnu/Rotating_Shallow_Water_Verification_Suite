"""
Name: Test_EdgeClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the edge class defined in ../../src/DGSEM_Rotating_Shallow_Water/EdgeClass.py.
"""


import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import EdgeClass


def TestEdge():
    myEdge = EdgeClass.Edge(1,[1,2],[1,2],[1,2])
    myEdge.OverwriteEdgeProperties(1,[1,2],[1,2],[1,2])
    
    
do_TestEdge = False
if do_TestEdge:
    TestEdge()