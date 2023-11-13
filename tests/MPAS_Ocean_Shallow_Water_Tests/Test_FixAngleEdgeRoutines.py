"""
Name: Test_FixAngleEdgeRoutines.py
Author: Siddhartha Bishnu
Details: As the name implies, this script tests various functions of 
../../src/MPAS_Ocean_Shallow_Water/FixAngleEdgeRoutines.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/MPAS_Ocean_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import FixAngleEdgeRoutines as FAER
    
    
def TestReturnTanInverseInProperQuadrant():
    PrintAngle = True 
    ReturnAngle = False
    DeltaX = np.sqrt(3.0)
    DeltaY = 1.0
    FAER.ReturnTanInverseInProperQuadrant(DeltaX,DeltaY,PrintAngle,ReturnAngle)
    print(' ')
    DeltaX = -np.sqrt(3.0)
    DeltaY = 1.0
    FAER.ReturnTanInverseInProperQuadrant(DeltaX,DeltaY,PrintAngle,ReturnAngle)
    print(' ')
    DeltaX = -np.sqrt(3.0)
    DeltaY = -1.0
    FAER.ReturnTanInverseInProperQuadrant(DeltaX,DeltaY,PrintAngle,ReturnAngle)
    print(' ')
    DeltaX = np.sqrt(3.0)
    DeltaY = -1.0
    FAER.ReturnTanInverseInProperQuadrant(DeltaX,DeltaY,PrintAngle,ReturnAngle)
    print(' ')
    DeltaX = 0.0
    DeltaY = 1.0
    FAER.ReturnTanInverseInProperQuadrant(DeltaX,DeltaY,PrintAngle,ReturnAngle)
    print(' ')
    DeltaX = 0.0
    DeltaY = -1.0
    FAER.ReturnTanInverseInProperQuadrant(DeltaX,DeltaY,PrintAngle,ReturnAngle)
    print(' ')
    DeltaX = 0.0
    DeltaY = 0.0
    FAER.ReturnTanInverseInProperQuadrant(DeltaX,DeltaY,PrintAngle,ReturnAngle)
    
    
do_TestReturnTanInverseInProperQuadrant = False
if do_TestReturnTanInverseInProperQuadrant:
    TestReturnTanInverseInProperQuadrant()
    
    
def TestFixAngleEdge():
    mesh_directory_root = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_4x4_Cells'
    PrintOutput = False
    PrintRelevantMeshData = False
    ReturnComputedAngleEdge = False
    BoundaryConditions = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    for iBoundaryCondition in range(0,len(BoundaryConditions)):
        BoundaryCondition = BoundaryConditions[iBoundaryCondition]
        mesh_directory = mesh_directory_root + '/' + BoundaryCondition
        if BoundaryCondition == 'Periodic':
            DetermineYCellAlongLatitude = True
        else:
            DetermineYCellAlongLatitude = False
        mesh_file_name = 'base_mesh_' + BoundaryCondition + '.nc'
        FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                          ReturnComputedAngleEdge)
        if not(BoundaryCondition == 'Periodic'):
            DetermineYCellAlongLatitude = True
            mesh_file_name = 'culled_mesh_' + BoundaryCondition + '.nc'
            FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,
                              PrintRelevantMeshData,ReturnComputedAngleEdge)        
        # Note that DetermineYCellAlongLatitude is now specified as True.
        mesh_file_name = 'mesh_' + BoundaryCondition + '.nc'
        FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                          ReturnComputedAngleEdge)    
    
    
do_TestFixAngleEdge = False
if do_TestFixAngleEdge:
    TestFixAngleEdge()