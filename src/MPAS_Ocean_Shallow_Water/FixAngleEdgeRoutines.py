"""
Name: FixAngleEdgeRoutines.py
Author: Siddhartha Bishnu
Details: This script fixes the values of the angleEdge, defined by the angle made by the vector directed from the
center of cellsOnEdge[0] to that of cellsOnEdge[1], along the boundaries of the domain.
"""


import numpy as np
import os
from netCDF4 import Dataset


def ReturnTanInverseInProperQuadrant(DeltaX,DeltaY,PrintAngle=False,ReturnAngle=True):
    if DeltaX != 0.0:
        if DeltaX > 0.0 and DeltaY > 0.0: # First Quadrant
            angle = np.arctan(DeltaY/DeltaX)
        elif DeltaX < 0.0 and DeltaY > 0.0: # Second Quadrant
            angle = np.pi + np.arctan(DeltaY/DeltaX) 
        elif DeltaX < 0.0 and DeltaY < 0.0: # Third Quadrant
            angle = np.pi + np.arctan(DeltaY/DeltaX) 
        elif DeltaX > 0.0 and DeltaY < 0.0: # Fourth Quadrant
            angle = 2.0*np.pi + np.arctan(DeltaY/DeltaX) 
        elif DeltaX > 0.0 and DeltaY == 0.0:
            angle = 0.0
        elif DeltaX < 0.0 and DeltaY == 0.0:
            angle = np.pi   
    else:
        if DeltaY > 0.0:
            angle = np.pi/2.0
        elif DeltaY < 0.0:
            angle = -np.pi/2.0
        else:
            print('DeltaX = 0 and DeltaY = 0! Stopping!')
            return
    if PrintAngle:
        if DeltaX != 0.0:
            print('DeltaY/DeltaX = %.15f.' %(DeltaY/DeltaX))
        print('The angle in radians is %.15f.' %angle)
        print('The angle in degrees is %.15f.' %(angle*180.0/np.pi))
        if DeltaX != 0.0:
            print('The trigonometric tangent of the angle is %.15f.' %np.tan(angle))
    if ReturnAngle:
        return angle


def FixAngleEdge(MeshDirectory,myMeshFileName,DetermineYCellAlongLatitude=True,PrintOutput=False,
                 PrintRelevantMeshData=False,ReturnComputedAngleEdge=True):
    cwd = os.getcwd()
    path = cwd + '/' + MeshDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    mesh_file = Dataset(myMeshFileName, "r", format='NETCDF4_CLASSIC')
    dcEdge = max(mesh_file.variables['dcEdge'][:])
    DeltaXMax = max(mesh_file.variables['dcEdge'][:])
    xCell = mesh_file.variables['xCell'][:]        
    yCell = mesh_file.variables['yCell'][:]  
    nCells = np.size(yCell)
    # The determination of yCellAlongLatitude in the following lines only holds for rectangular structured meshes with 
    # equal number of cells in each direction. However, for a problem with non-periodic boundary conditions, it will 
    # work for the culled mesh and the final mesh, but not the base mesh.
    if DetermineYCellAlongLatitude:
        nY = int(np.sqrt(nCells))
        yCellAlongLatitude = np.zeros(nY)
        iYAlongLatitude = 0
        for iY in range(0,nCells):
            if np.mod(float(iY),float(nY)) == 0.0:
                yCellAlongLatitude[iYAlongLatitude] = yCell[iY]
                iYAlongLatitude += 1
        DeltaYMax = max(np.diff(yCellAlongLatitude))
    else:
        DeltaYMax = DeltaXMax*np.sqrt(3.0)/2.0
    xEdge = mesh_file.variables['xEdge'][:]        
    yEdge = mesh_file.variables['yEdge'][:] 
    angleEdge = mesh_file.variables['angleEdge'][:]
    cellsOnEdge = mesh_file.variables['cellsOnEdge'][:]
    nEdges = np.size(angleEdge)
    ComputedAngleEdge = np.zeros(nEdges)
    tolerance = 10.0**(-6.0)*DeltaYMax
    if PrintOutput and PrintRelevantMeshData:
        print('The relevant mesh data is:')
    for iEdge in range(0,nEdges):
        thisXEdge = xEdge[iEdge]
        thisYEdge = yEdge[iEdge]
        cellID1 = cellsOnEdge[iEdge,0]
        iCell1 = cellID1 - 1
        cellID2 = cellsOnEdge[iEdge,1]
        iCell2 = cellID2 - 1
        xCell1 = xCell[iCell1]
        xCell2 = xCell[iCell2]
        DeltaX = xCell2 - xCell1
        yCell1 = yCell[iCell1]
        yCell2 = yCell[iCell2]
        DeltaY = yCell2 - yCell1
        if cellID2 == 0:
            if thisXEdge > xCell1 and abs(thisYEdge - yCell1) < tolerance:
                DeltaX = dcEdge
                DeltaY = 0.0
            elif thisXEdge > xCell1 and thisYEdge > yCell1:
                DeltaX = dcEdge/2.0
                DeltaY = np.sqrt(3.0)/2.0*dcEdge   
            elif thisXEdge > xCell1 and thisYEdge < yCell1:
                DeltaX = dcEdge/2.0
                DeltaY = -np.sqrt(3.0)/2.0*dcEdge                   
            elif thisXEdge < xCell1 and abs(thisYEdge - yCell1) < tolerance:
                DeltaX = -dcEdge
                DeltaY = 0.0                
            elif thisXEdge < xCell1 and thisYEdge > yCell1:
                DeltaX = -dcEdge/2.0
                DeltaY = np.sqrt(3.0)/2.0*dcEdge
            elif thisXEdge < xCell1 and thisYEdge < yCell1:
                DeltaX = -dcEdge/2.0
                DeltaY = -np.sqrt(3.0)/2.0*dcEdge                 
        else:
            if abs(DeltaY) < tolerance and DeltaX < 0.0 and abs(DeltaX) > DeltaXMax:
            # cells [{4,1},{8,5},{12,9},{16,13}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge
            elif abs(DeltaY) < tolerance and DeltaX > 0.0 and abs(DeltaX) > DeltaXMax:
            # cells [{1,4},{5,8},{9,12},{13,16}] for a regular structured 4 x 4 mesh
                DeltaX = -dcEdge
            elif DeltaX < 0.0 and DeltaY < 0.0 and abs(DeltaX) > DeltaXMax and abs(DeltaY) > DeltaYMax:
            # cells [{16,1}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge/2.0
                DeltaY = np.sqrt(3.0)/2.0*dcEdge
            elif DeltaX < 0.0 and DeltaY < 0.0 and abs(DeltaX) > DeltaXMax and abs(DeltaY) <= DeltaYMax:
            # cells [{8,1},{16,9}] for a regular structured 4 x 4 mesh    
                DeltaX = dcEdge/2.0
                DeltaY = -np.sqrt(3.0)/2.0*dcEdge                   
            elif DeltaX < 0.0 and DeltaY < 0.0 and abs(DeltaX) < DeltaXMax and abs(DeltaY) > DeltaYMax:
            # cells [{13,1},{14,2},{15,3},{16,4}] for a regular structured 4 x 4 mesh
                DeltaX = -dcEdge/2.0
                DeltaY = np.sqrt(3.0)/2.0*dcEdge
            elif DeltaX < 0.0 and DeltaY > 0.0 and abs(DeltaX) < DeltaXMax and abs(DeltaY) > DeltaYMax:
            # cells [{2,13},{3,14},{4,15}] for a regular structured 4 x 4 mesh    
                DeltaX = -dcEdge/2.0
                DeltaY = -np.sqrt(3.0)/2.0*dcEdge       
            elif DeltaX < 0.0 and DeltaY > 0.0 and abs(DeltaX) > DeltaXMax and abs(DeltaY) <= DeltaYMax:
            # cells [{8,9}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge/2.0
                DeltaY = np.sqrt(3.0)/2.0*dcEdge                   
            elif DeltaX > 0.0 and DeltaY < 0.0 and abs(DeltaX) > DeltaXMax and abs(DeltaY) <= DeltaYMax:
            # cells [{9,8}] for a regular structured 4 x 4 mesh
                DeltaX = -dcEdge/2.0
                DeltaY = -np.sqrt(3.0)/2.0*dcEdge                   
            elif DeltaX > 0.0 and DeltaY < 0.0 and abs(DeltaX) < DeltaXMax and abs(DeltaY) > DeltaYMax:
            # cells [{13,2},{14,3},{15,4}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge/2.0
                DeltaY = np.sqrt(3.0)/2.0*dcEdge                
            elif DeltaX > 0.0 and DeltaY > 0.0 and abs(DeltaX) < DeltaXMax and abs(DeltaY) > DeltaYMax:
            # cells [{1,13},{2,14},{3,15},{4,16}] for a regular structured 4 x 4 mesh            
                DeltaX = dcEdge/2.0
                DeltaY = -np.sqrt(3.0)/2.0*dcEdge    
            elif DeltaX > 0.0 and DeltaY > 0.0 and abs(DeltaX) > DeltaXMax and abs(DeltaY) <= DeltaYMax:
            # cells [{1,8},{9,16}] for a regular structured 4 x 4 mesh    
                DeltaX = -dcEdge/2.0
                DeltaY = np.sqrt(3.0)/2.0*dcEdge    
            elif DeltaX > 0.0 and DeltaY > 0.0 and abs(DeltaX) > DeltaXMax and abs(DeltaY) > DeltaYMax:
            # cells [{1,16}] for a regular structured 4 x 4 mesh    
                DeltaX = -dcEdge/2.0
                DeltaY = -np.sqrt(3.0)/2.0*dcEdge
        ComputedAngleEdge[iEdge] = ReturnTanInverseInProperQuadrant(DeltaX,DeltaY)
        if PrintOutput:
            # PrintOutput should be specified as True only for small meshes consisting of 4 x 4 cells.
            if PrintRelevantMeshData: 
            # PrintRelevantMeshData should be specified as True only for small meshes consisting of 4 x 4 cells.
                print('%2d [%2d %2d] %+9.2f [%+9.2f %+9.2f] %+9.2f %+9.2f [%+9.2f %+9.2f] %+8.2f [%+5.2f %+5.2f]'
                      %(iEdge,cellID1,cellID2,thisXEdge,xCell1,xCell2,DeltaX,thisYEdge,yCell1,yCell2,DeltaY,
                        angleEdge[iEdge],ComputedAngleEdge[iEdge]))
            else:
                print('For edge %2d with cellsOnEdge = {%2d,%2d}, {angleEdge, ComputedAngleEdge} is {%.2f, %.2f}.'
                      %(iEdge+1,cellID1,cellID2,angleEdge[iEdge],ComputedAngleEdge[iEdge]))
    os.chdir(cwd)
    if ReturnComputedAngleEdge:
        return ComputedAngleEdge