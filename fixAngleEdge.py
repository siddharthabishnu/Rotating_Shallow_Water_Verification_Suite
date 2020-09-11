
# coding: utf-8

# Name: fixAngleEdge.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code fixes the values of the angleEdge, defined by the angle made by the vector directed from the center of cellsOnEdge[0] to that of cellsOnEdge[1] with the positive eastward direction, along the boundaries of the domain. <br/>

# In[1]:

import numpy as np
import os
import netCDF4 as nc
from netCDF4 import Dataset


# In[2]:

def returnTanInverseInProperQuadrant(DeltaX,DeltaY,printAngle=False):
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
    if printAngle:
        if DeltaX != 0.0:
            print('DeltaY/DeltaX = %.15f.' %(DeltaY/DeltaX))
        print('The angle in radians is %.15f.' %angle)
        print('The angle in degrees is %.15f.' %(angle*180.0/np.pi))
        if DeltaX != 0.0:
            print('The trigonometric tangent of the angle is %.15f.' %np.tan(angle))
    return angle


# In[3]:

angle = returnTanInverseInProperQuadrant(np.sqrt(3.0),1.0,True)


# In[4]:

angle = returnTanInverseInProperQuadrant(-np.sqrt(3.0),1.0,True)


# In[5]:

angle = returnTanInverseInProperQuadrant(-np.sqrt(3.0),-1.0,True)


# In[6]:

angle = returnTanInverseInProperQuadrant(np.sqrt(3.0),-1.0,True)


# In[7]:

angle = returnTanInverseInProperQuadrant(0.0,1.0,True)


# In[8]:

angle = returnTanInverseInProperQuadrant(0.0,-1.0,True)


# In[9]:

returnTanInverseInProperQuadrant(0.0,0.0,True)


# In[10]:

def fix_angleEdge(mesh_directory,mesh_file_name,determineYCellAlongLatitude=True,printOutput=False,
                  printRelevantMeshData=False):
    cwd = os.getcwd()
    path = cwd + '/' + mesh_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    mesh_file = Dataset(mesh_file_name, "r", format='NETCDF4_CLASSIC')
    dcEdge = round(max(mesh_file.variables['dcEdge'][:]))  
    DeltaXMax = max(mesh_file.variables['dcEdge'][:])
    xCell = mesh_file.variables['xCell'][:]        
    yCell = mesh_file.variables['yCell'][:]  
    nCells = np.size(yCell)
    # The determination of yCellAlongLatitude in the following lines only holds for rectangular structured meshes 
    # with equal number of cells in each direction. However, for a problem with non-periodic boundary conditions,
    # it will work for the culled mesh and the final mesh, but not the base mesh.
    if determineYCellAlongLatitude:
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
    computed_angleEdge = np.zeros(nEdges)
    tolerance = 10.0**(-3.0) 
    if printOutput and printRelevantMeshData:
        print('The relevant mesh data is:')
    for iEdge in range(0,nEdges):
        thisXEdge = xEdge[iEdge]
        thisYEdge = yEdge[iEdge]
        cellID1 = cellsOnEdge[iEdge,0]
        cell1 = cellID1 - 1
        cellID2 = cellsOnEdge[iEdge,1]
        cell2 = cellID2 - 1
        xCell1 = xCell[cell1]
        xCell2 = xCell[cell2]
        DeltaX = xCell2 - xCell1
        yCell1 = yCell[cell1]
        yCell2 = yCell[cell2]
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
        computed_angleEdge[iEdge] = returnTanInverseInProperQuadrant(DeltaX,DeltaY)
        if printOutput:
            # printOutput should be specified as True only for small meshes consisting of 4 x 4 cells.
            if printRelevantMeshData: 
            # printRelevantMeshData should be specified as True only for small meshes consisting of 4 x 4 cells.
                print('%2d [%2d %2d] %+9.2f [%+9.2f %+9.2f] %+9.2f %+9.2f [%+9.2f %+9.2f] %+8.2f [%+5.2f %+5.2f]'
                      %(iEdge,cellID1,cellID2,thisXEdge,xCell1,xCell2,DeltaX,thisYEdge,yCell1,yCell2,DeltaY,
                        angleEdge[iEdge],computed_angleEdge[iEdge]))
            else:
                print(
                'For edge %2d with cellsOnEdge = {%2d,%2d}, {angleEdge, computed_angleEdge} is {%.2f, %.2f}.'
                %(iEdge+1,cellID1,cellID2,angleEdge[iEdge],computed_angleEdge[iEdge]))
    os.chdir(cwd)
    return computed_angleEdge


# In[11]:

do_fix_angleEdge_base_mesh_P = False
if do_fix_angleEdge_base_mesh_P:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'base_mesh_P.nc',determineYCellAlongLatitude=True,printOutput=False,
                                       printRelevantMeshData=False)


# In[12]:

do_fix_angleEdge_mesh_P = False
if do_fix_angleEdge_mesh_P:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'mesh_P.nc',determineYCellAlongLatitude=True,printOutput=False,
                                       printRelevantMeshData=False)


# In[13]:

do_fix_angleEdge_base_mesh_NP_x = False
if do_fix_angleEdge_base_mesh_NP_x:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'base_mesh_NP_x.nc',determineYCellAlongLatitude=False,printOutput=False,
                                       printRelevantMeshData=False)


# In[14]:

do_fix_angleEdge_culled_mesh_NP_x = False
if do_fix_angleEdge_culled_mesh_NP_x:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'culled_mesh_NP_x.nc',determineYCellAlongLatitude=True,printOutput=False,
                                       printRelevantMeshData=False)


# In[15]:

do_fix_angleEdge_mesh_NP_x = False
if do_fix_angleEdge_mesh_NP_x:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'mesh_NP_x.nc',determineYCellAlongLatitude=True,printOutput=False,
                                       printRelevantMeshData=False)


# In[16]:

do_fix_angleEdge_base_mesh_NP_y = False
if do_fix_angleEdge_base_mesh_NP_y:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'base_mesh_NP_y.nc',determineYCellAlongLatitude=False,printOutput=False,
                                       printRelevantMeshData=False)


# In[17]:

do_fix_angleEdge_culled_mesh_NP_y = False
if do_fix_angleEdge_culled_mesh_NP_y:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'culled_mesh_NP_y.nc',determineYCellAlongLatitude=True,printOutput=False,
                                       printRelevantMeshData=False)


# In[18]:

do_fix_angleEdge_mesh_NP_y = False
if do_fix_angleEdge_mesh_NP_y:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'mesh_NP_y.nc',determineYCellAlongLatitude=True,printOutput=False,
                                       printRelevantMeshData=False)


# In[19]:

do_fix_angleEdge_base_mesh_NP_xy = False
if do_fix_angleEdge_base_mesh_NP_xy:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'base_mesh_NP_xy.nc',determineYCellAlongLatitude=False,printOutput=False,
                                       printRelevantMeshData=False)


# In[20]:

do_fix_angleEdge_culled_mesh_NP_xy = False
if do_fix_angleEdge_culled_mesh_NP_xy:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'culled_mesh_NP_xy.nc',determineYCellAlongLatitude=True,printOutput=False,
                                       printRelevantMeshData=False)


# In[21]:

do_fix_angleEdge_mesh_NP_xy = False
if do_fix_angleEdge_mesh_NP_xy:
    computed_angleEdge = fix_angleEdge('MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                                       'mesh_NP_xy.nc',determineYCellAlongLatitude=True,printOutput=False,
                                       printRelevantMeshData=False)