
# coding: utf-8

# Name: Initialization.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains routines for creating the 'init.nc' file containing the mesh parameters copied from the 'mesh.nc' file as well as the initial values of the relevant variables. <br/>

# In[1]:

import numpy as np
import netCDF4 as nc
import os
from netCDF4 import Dataset
from lxml import etree


# In[2]:

def StringToBoolean(x):
    return x.lower() in ("TRUE", "T", "True", "true", "t")


# In[3]:

def CommonData(problem_type='default'):
    global nVertLevelsParameter, CoriolisParameter, NormalVelocityParameter, LayerThicknessParameter
    global BottomDepthParameter, MaxLevelCellParameter, EdgeMaskParameter
    if problem_type == 'default' or problem_type == 'Coastal_Kelvin_Wave':
        nVertLevelsParameter = 1
        CoriolisParameter = 10.0**(-4.0)
        NormalVelocityParameter = 0.0
        LayerThicknessParameter = 1000.0
        BottomDepthParameter = 1000.0
        MaxLevelCellParameter = 0
        EdgeMaskParameter = 0


# In[4]:

def SpecifyInitialConditions(mesh_directory='Mesh+Initial_Condition+Registry_Files/Periodic',
                             mesh_file_name='mesh.nc',init_file_name='init.nc',problem_type='default'):
    CommonData(problem_type)
    cwd = os.getcwd()
    path = cwd + '/' + mesh_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path) 
    # Source file
    src = Dataset(mesh_file_name, "r", format='NETCDF4_CLASSIC')
    # Destination file
    if os.path.exists(init_file_name):
        os.remove(init_file_name)
    dst = Dataset(init_file_name, "w", format="NETCDF4_CLASSIC")
    # Copy attributes
    for name in src.ncattrs():
        dst.setncattr(name, src.getncattr(name))
    # Copy dimensions
    for name, dimension in src.dimensions.items():
        dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
    # Add dimensions
    dst.createDimension('nVertLevels', nVertLevelsParameter)
    srcList = list()
    # Copy variables
    for name, variable in src.variables.items():
        srcList.append(src.variables[name].name)
        x = dst.createVariable(name, variable.datatype, variable.dimensions)
        dst.variables[name][:] = src.variables[name][:]
    # Parse the Registry file 
    Registry = etree.parse("Registry.xml")
    # You can only use the above line of code after copying or linking the Registry file to this folder. 
    # Alternatively, you can use the exact path location of the Registry file and modify the above line of code as
    # Registry = etree.parse("/path/to/Registry.xml")
    # Copy the units and long names of the common variables from the parsed Registry file 
    for name, variable in dst.variables.items():
        for var_struct in Registry.xpath('//var_struct'):
            if var_struct.attrib['name'] == 'mesh':
                for var in var_struct.getchildren():
                    if var.attrib['name'] == name:
                        dst.variables[name].units = var.attrib['units']
                        dst.variables[name].long_name = var.attrib['description']
    # Get values of the dimensions
    nCells = len(dst.dimensions['nCells'])
    nEdges = len(dst.dimensions['nEdges'])
    nVertices = len(dst.dimensions['nVertices'])
    nVertLevels = len(dst.dimensions['nVertLevels'])
    # Create new output variables and specify their dimensions
    fVertex = dst.createVariable('fVertex', np.float64, ('nVertices',))
    fCell = dst.createVariable('fCell', np.float64, ('nCells',))
    fEdge = dst.createVariable('fEdge', np.float64, ('nEdges',))
    normalVelocity = dst.createVariable('normalVelocity', np.float64, ('Time','nEdges','nVertLevels',))
    layerThickness = dst.createVariable('layerThickness', np.float64, ('Time','nCells','nVertLevels',))
    bottomDepth = dst.createVariable('bottomDepth', np.float64, ('nCells',))
    maxLevelCell = dst.createVariable('maxLevelCell', np.int32, ('nCells',))
    edgeMask = dst.createVariable('edgeMask', np.int32, ('nEdges','nVertLevels',))      
    # Copy the units and long names of these new output variables from the parsed Registry file 
    for name, variable in dst.variables.items():
        if name not in srcList:
            for var in Registry.xpath('/registry/var_struct/var'):
                if var.attrib['name'] == name:
                    dst.variables[name].units = var.attrib['units']
                    dst.variables[name].long_name = var.attrib['description']  
    # Initialize these new output variables  
    fVertex[:] = CoriolisParameter
    fCell[:] = CoriolisParameter
    fEdge[:] = CoriolisParameter
    normalVelocity[:] = NormalVelocityParameter
    layerThickness[0,:,:] = LayerThicknessParameter
    bottomDepth[:] = BottomDepthParameter
    maxLevelCell[:] = MaxLevelCellParameter
    edgeMask[:] = EdgeMaskParameter      
    # Close the destination file
    dst.close()
    os.chdir(cwd)


# In[5]:

do_SpecifyInitialConditions_1 = False
if do_SpecifyInitialConditions_1:
    SpecifyInitialConditions()


# In[6]:

do_SpecifyInitialConditions_2 = False
if do_SpecifyInitialConditions_2:
    SpecifyInitialConditions(mesh_directory='Mesh+Initial_Condition+Registry_Files/NonPeriodic_x',
                             mesh_file_name='mesh.nc',init_file_name='init.nc',problem_type='default')


# In[7]:

do_SpecifyInitialConditions_3 = False
if do_SpecifyInitialConditions_3:
    SpecifyInitialConditions(mesh_directory='MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                             mesh_file_name='mesh_P.nc',init_file_name='init_P.nc',
                             problem_type='Coastal_Kelvin_Wave')


# In[8]:

do_SpecifyInitialConditions_4 = False
if do_SpecifyInitialConditions_4:
    SpecifyInitialConditions(mesh_directory='MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh',
                             mesh_file_name='mesh_NP.nc',init_file_name='init_NP.nc',
                             problem_type='Coastal_Kelvin_Wave')