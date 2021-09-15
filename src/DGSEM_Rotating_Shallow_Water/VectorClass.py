"""
Name: VectorClass.py
Author: Sid Bishnu
Details: This script contains functions for forming a vector class and performing vector operations.
"""


import numpy as np


class Vector:
    
    def __init__(myVector,Components):
        nDimensions = len(Components)
        myVector.nDimensions = nDimensions
        myVector.Components = Components
        myVector.Length = np.sqrt(DotVectors(myVector,myVector))
        
    def NormalizeVector(myVector):
        myVector.Components[:] /= myVector.Length
        myVector.Length = 1.0
        
    def PrintVector(myVector):
        print('The number of dimensions of the vector is %d.' %myVector.nDimensions)
        print('The components of the vector are', myVector.Components)
        print('The length of the vector is %.6f.' %myVector.Length)


def AddVectors(myVector1,myVector2):
    nDimensions = myVector1.nDimensions
    ResultantComponents = np.zeros(nDimensions)
    for iDimension in range(0,nDimensions):
        ResultantComponents[iDimension] = myVector1.Components[iDimension] + myVector2.Components[iDimension]    
    myVector3 = Vector(ResultantComponents)
    return myVector3


def DotVectors(myVector1,myVector2):
    nDimensions = myVector1.nDimensions
    myVector1_dot_myVector2 = 0.0
    for iDimension in range(0,nDimensions):
        myVector1_dot_myVector2 += myVector1.Components[iDimension]*myVector2.Components[iDimension]
    return myVector1_dot_myVector2