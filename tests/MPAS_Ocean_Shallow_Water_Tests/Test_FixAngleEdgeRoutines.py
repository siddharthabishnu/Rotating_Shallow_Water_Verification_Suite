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
    mesh_directory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/MPAS_Ocean_Shallow_Water_Meshes_4x4_Cells'
    PrintOutput = False
    PrintRelevantMeshData = False
    ReturnComputedAngleEdge = False
    mesh_file_name = 'base_mesh_P.nc'
    DetermineYCellAlongLatitude = True
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    mesh_file_name = 'mesh_P.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    DetermineYCellAlongLatitude = False
    mesh_file_name = 'base_mesh_NP_x.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    DetermineYCellAlongLatitude = True
    mesh_file_name = 'culled_mesh_NP_x.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    mesh_file_name = 'mesh_NP_x.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    DetermineYCellAlongLatitude = False
    mesh_file_name = 'base_mesh_NP_y.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    DetermineYCellAlongLatitude = True
    mesh_file_name = 'culled_mesh_NP_y.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    mesh_file_name = 'mesh_NP_y.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    DetermineYCellAlongLatitude = False
    mesh_file_name = 'base_mesh_NP_xy.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    DetermineYCellAlongLatitude = True
    mesh_file_name = 'culled_mesh_NP_xy.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    mesh_file_name = 'mesh_NP_xy.nc'
    FAER.FixAngleEdge(mesh_directory,mesh_file_name,DetermineYCellAlongLatitude,PrintOutput,PrintRelevantMeshData,
                      ReturnComputedAngleEdge)
    
    
do_TestFixAngleEdge = False
if do_TestFixAngleEdge:
    TestFixAngleEdge()