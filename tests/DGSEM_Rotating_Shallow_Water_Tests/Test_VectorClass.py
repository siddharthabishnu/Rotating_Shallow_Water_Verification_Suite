"""
Name: Test_VectorClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the vector class defined in 
../../src/DGSEM_Rotating_Shallow_Water/VectorClass.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import VectorClass


def TestVector():
    print('The normalized first vector:\n')
    Components = np.array([3.0,4.0])
    myVector1 = VectorClass.Vector(Components)
    myVector1.NormalizeVector()
    myVector1.PrintVector()
    print('\nThe normalized second vector:\n')
    Components = np.array([5.0,12.0])
    myVector2 = VectorClass.Vector(Components)
    myVector2.NormalizeVector()
    myVector2.PrintVector()
    print('\nAdding the normalized first and second vectors and normalizing the result:\n')
    myVector3 = VectorClass.AddVectors(myVector1,myVector2)
    myVector3.NormalizeVector()
    myVector3.PrintVector()
    myVector1_dot_myVector2 = VectorClass.DotVectors(myVector1,myVector2)
    print('\nThe dot product of the normalized first and second vectors is %.6f.' %myVector1_dot_myVector2)
    
    
do_TestVector = False
if do_TestVector:
    TestVector()