
# coding: utf-8

# Name: MPAS_O_Mode_Init.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code initializes the MPAS_O class after acquiring the relevant information from the mesh and initial condition files. <br/>

# In[1]:

import numpy as np
import io as inputoutput
import os
from IPython.utils import io
import netCDF4 as nc
from netCDF4 import Dataset
with io.capture_output() as captured: 
    import fixAngleEdge


# In[2]:

class Namelist:
    
    def __init__(myNamelist,problem_type='default',problem_is_linear=True):
        myNamelist.config_problem_type = problem_type
        myNamelist.config_problem_is_linear = problem_is_linear
        if problem_is_linear:
            myNamelist.config_linearity_prefactor = 0.0
        else:
            myNamelist.config_linearity_prefactor = 1.0        
        if problem_type == 'default' or problem_type == 'Coastal_Kelvin_Wave':
            myNamelist.config_time_integrator = 'forward_backward_predictor'
            myNamelist.config_dt = 180.0
            myNamelist.config_forward_backward_predictor_parameter_gamma = 1.0
            myNamelist.config_gravity = 10.0
            myNamelist.config_mean_depth = 1000.0
            myNamelist.config_thickness_flux_type = 'centered'
            myNamelist.config_use_wetting_drying = False
            # Derived Parameters
            myNamelist.config_wave_speed = np.sqrt(myNamelist.config_gravity*myNamelist.config_mean_depth)


# In[3]:

def CheckDimensions(myVariableName):
    cwd = os.getcwd()
    path = cwd + '/Mesh+Initial_Condition+Registry_Files/Periodic'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path) 
    mesh_file = Dataset("mesh.nc", "r", format='NETCDF4_CLASSIC')
    myVariable = mesh_file.variables[myVariableName][:]
    print('The shape of the given variable is ')
    print(np.shape(myVariable))
    os.chdir(cwd)


# In[4]:

do_CheckDimensions_kiteAreasOnVertex = False
if do_CheckDimensions_kiteAreasOnVertex:
    CheckDimensions('kiteAreasOnVertex')


# In[5]:

def DetermineCoriolisParameterAndBottomDepth(myMPAS_O,problem_type='default'):
    CoriolisParameter = 10.0**(-4.0)
    BottomDepthParameter = 1000.0
    if problem_type == 'default' or problem_type == 'Coastal_Kelvin_Wave':
        myMPAS_O.fCell[:] = CoriolisParameter        
        myMPAS_O.fEdge[:] = CoriolisParameter
        myMPAS_O.fVertex[:] = CoriolisParameter
        myMPAS_O.bottomDepth[:] = BottomDepthParameter


# In[6]:

class MPAS_O:
    
    def __init__(myMPAS_O,print_basic_geometry,mesh_directory='Mesh+Initial_Condition+Registry_Files/Periodic',
                 base_mesh_file_name='base_mesh.nc',mesh_file_name='mesh.nc',problem_type='default',
                 problem_is_linear=True,do_fixAngleEdge=True,print_Output=False):
        myMPAS_O.myNamelist = Namelist(problem_type,problem_is_linear)
        cwd = os.getcwd()
        path = cwd + '/' + mesh_directory + '/'
        if not os.path.exists(path):
            os.mkdir(path) # os.makedir(path)
        os.chdir(path)
        # base mesh file
        base_mesh_file = Dataset(base_mesh_file_name, "r", format='NETCDF4_CLASSIC')
        # mesh file
        mesh_file = Dataset(mesh_file_name, "r", format='NETCDF4_CLASSIC')
        # Get values of the dimensions
        myMPAS_O.nCells = len(mesh_file.dimensions['nCells'])
        myMPAS_O.nEdges = len(mesh_file.dimensions['nEdges'])
        myMPAS_O.nVertices = len(mesh_file.dimensions['nVertices'])
        myMPAS_O.maxEdges = len(mesh_file.dimensions['maxEdges'])
        myMPAS_O.maxEdges2 = len(mesh_file.dimensions['maxEdges2'])
        myMPAS_O.vertexDegree = len(mesh_file.dimensions['vertexDegree'])
        myMPAS_O.nVertLevels = 1
        nCells = myMPAS_O.nCells
        nEdges = myMPAS_O.nEdges
        nVertices = myMPAS_O.nVertices
        maxEdges = myMPAS_O.maxEdges
        maxEdges2 = myMPAS_O.maxEdges2
        vertexDegree = myMPAS_O.vertexDegree
        nVertLevels = myMPAS_O.nVertLevels
        if print_basic_geometry:
            print('The number of cells is %d.' %nCells)
            print('The number of edges is %d.' %nEdges)
            print('The number of vertices is %d.' %nVertices)
            print('The number of vertical levels is %d.' %nVertLevels)
        # Get values of the variables
        myMPAS_O.latCell = base_mesh_file.variables['latCell'][:]
        myMPAS_O.lonCell = base_mesh_file.variables['lonCell'][:]
        myMPAS_O.xCell = base_mesh_file.variables['xCell'][:]        
        myMPAS_O.yCell = base_mesh_file.variables['yCell'][:]      
        myMPAS_O.zCell = base_mesh_file.variables['zCell'][:]        
        myMPAS_O.indexToCellID = base_mesh_file.variables['indexToCellID'][:]   
        myMPAS_O.indexToEdgeID = base_mesh_file.variables['indexToEdgeID'][:]
        myMPAS_O.latVertex = base_mesh_file.variables['latVertex'][:]
        myMPAS_O.lonVertex = base_mesh_file.variables['lonVertex'][:]
        myMPAS_O.xVertex = base_mesh_file.variables['xVertex'][:]
        myMPAS_O.yVertex = base_mesh_file.variables['yVertex'][:]
        myMPAS_O.zVertex = base_mesh_file.variables['zVertex'][:]  
        myMPAS_O.indexToVertexID = base_mesh_file.variables['indexToVertexID'][:]
        myMPAS_O.nEdgesOnCell = base_mesh_file.variables['nEdgesOnCell'][:]
        # Choose my_mesh_file_name to be either base_mesh_file_name or mesh_file_name
        my_mesh_file_name = mesh_file_name
        # Choose my_mesh_file to be either base_mesh_file or mesh_file
        my_mesh_file = mesh_file
        # For a regular mesh, areaCell, dcEdge, dvEdge and areaTriangle should remain constant. I have to report 
        # to Xylar that the mpas_mesh_converter.cpp file (very) slightly modifies the values of these arrays while
        # generating the mesh file from the base mesh (or culled mesh) file. Even though the first incorrect digit
        # may appear after a few places of the decimal point, I think it is better to get it fixed so that it does
        # not influence any convergence study involving error analysis of simplified test cases. In case of a 
        # global simulation involving a global mesh with variable resolution capability, this (very) minor error 
        # will most definitely be overshadowed by other errors of larger magnitude in which case we do not need to
        # fix it.        
        myMPAS_O.areaCell = my_mesh_file.variables['areaCell'][:]
        myMPAS_O.dcEdge = my_mesh_file.variables['dcEdge'][:]
        myMPAS_O.dvEdge = my_mesh_file.variables['dvEdge'][:]
        myMPAS_O.areaTriangle = my_mesh_file.variables['areaTriangle'][:]
        # Even though the following arrays are contained within the base mesh (or culled mesh) file, the local and
        # global ordering of the edges and hence the sequence of elements of these arrays are modified by the 
        # mpas_mesh_converter.cpp file which generates the mesh file using the base mesh (or culled mesh) file.
        # The implementation of non-periodic boundary conditions also modifies some of these arrays within the 
        # culled mesh file. That is why we are reading all of these arrays from the mesh file.
        myMPAS_O.latEdge = my_mesh_file.variables['latEdge'][:]
        myMPAS_O.lonEdge = my_mesh_file.variables['lonEdge'][:]
        myMPAS_O.xEdge = my_mesh_file.variables['xEdge'][:]
        myMPAS_O.yEdge = my_mesh_file.variables['yEdge'][:]
        myMPAS_O.zEdge = my_mesh_file.variables['zEdge'][:]
        myMPAS_O.cellsOnCell = my_mesh_file.variables['cellsOnCell'][:]
        myMPAS_O.edgesOnCell = my_mesh_file.variables['edgesOnCell'][:]
        myMPAS_O.verticesOnCell = my_mesh_file.variables['verticesOnCell'][:]
        myMPAS_O.edgesOnEdge = my_mesh_file.variables['edgesOnEdge'][:]
        myMPAS_O.cellsOnEdge = my_mesh_file.variables['cellsOnEdge'][:]
        myMPAS_O.verticesOnEdge = my_mesh_file.variables['verticesOnEdge'][:] 
        myMPAS_O.nEdgesOnEdge = my_mesh_file.variables['nEdgesOnEdge'][:] 
        myMPAS_O.cellsOnVertex = my_mesh_file.variables['cellsOnVertex'][:]
        myMPAS_O.edgesOnVertex = my_mesh_file.variables['edgesOnVertex'][:]
        myMPAS_O.angleEdge = my_mesh_file.variables['angleEdge'][:]
        myMPAS_O.weightsOnEdge = my_mesh_file.variables['weightsOnEdge'][:]
        myMPAS_O.kiteAreasOnVertex = my_mesh_file.variables['kiteAreasOnVertex'][:]
        # The following two arrays are contained only within the mesh file.
        myMPAS_O.boundaryVertex = mesh_file.variables['boundaryVertex'][:]
        myMPAS_O.gridSpacing = mesh_file.variables['gridSpacing'][:]
        # Define and initialize the following arrays not contained within either the base mesh file or the mesh 
        # file.
        myMPAS_O.fVertex = np.zeros(nVertices)
        myMPAS_O.fCell = np.zeros(nCells)
        myMPAS_O.fEdge = np.zeros(nEdges)
        myMPAS_O.bottomDepth = np.zeros(nCells)
        DetermineCoriolisParameterAndBottomDepth(myMPAS_O,problem_type='default')
        myMPAS_O.lX = max(myMPAS_O.xCell)
        myMPAS_O.lY = max(myMPAS_O.yVertex)
        myMPAS_O.gridSpacingMagnitude = myMPAS_O.xCell[1] - myMPAS_O.xCell[0]
        myMPAS_O.boundaryCell = np.zeros((nCells,nVertLevels))
        myMPAS_O.boundaryEdge = np.zeros((nEdges,nVertLevels))
        myMPAS_O.boundaryVertex = np.zeros((nVertices,nVertLevels))        
        myMPAS_O.cellMask = np.zeros((nCells,nVertLevels))
        myMPAS_O.circulation = np.zeros((nVertices,nVertLevels))
        myMPAS_O.divergence = np.zeros((nCells,nVertLevels))
        myMPAS_O.edgeMask = np.zeros((nEdges,nVertLevels),dtype=int)
        myMPAS_O.edgeSignOnCell = np.zeros((nCells,maxEdges))
        myMPAS_O.edgeSignOnVertex = np.zeros((nVertices,maxEdges))
        myMPAS_O.kineticEnergyCell = np.zeros((nCells,nVertLevels))
        myMPAS_O.kiteIndexOnCell = np.zeros((nCells,maxEdges),dtype=int)
        myMPAS_O.layerThicknessCurrent = np.zeros((nCells,nVertLevels))
        myMPAS_O.layerThicknessEdge = np.zeros((nEdges,nVertLevels))
        myMPAS_O.maxLevelCell = np.zeros(nCells,dtype=int)
        myMPAS_O.maxLevelEdgeBot = np.zeros(nEdges,dtype=int)
        myMPAS_O.maxLevelEdgeTop = np.zeros(nEdges,dtype=int)
        myMPAS_O.maxLevelVertexBot = np.zeros(nVertices,dtype=int)
        myMPAS_O.maxLevelVertexTop = np.zeros(nVertices,dtype=int)
        myMPAS_O.normalizedPlanetaryVorticityEdge = np.zeros((nEdges,nVertLevels))
        myMPAS_O.normalizedPlanetaryVorticityVertex = np.zeros((nVertices,nVertLevels))
        myMPAS_O.normalizedRelativeVorticityCell = np.zeros((nCells,nVertLevels))
        myMPAS_O.normalizedRelativeVorticityEdge = np.zeros((nEdges,nVertLevels))
        myMPAS_O.normalizedRelativeVorticityVertex = np.zeros((nVertices,nVertLevels))
        myMPAS_O.normalVelocityCurrent = np.zeros((nEdges,nVertLevels))
        myMPAS_O.normalVelocityExact = np.zeros((nEdges,nVertLevels))
        myMPAS_O.normalVelocityNew = np.zeros((nEdges,nVertLevels))
        myMPAS_O.relativeVorticity = np.zeros((nVertices,nVertLevels))
        myMPAS_O.relativeVorticityCell = np.zeros((nCells,nVertLevels))
        myMPAS_O.sshCurrent = np.zeros(nCells)
        myMPAS_O.sshExact = np.zeros(nCells)
        myMPAS_O.sshNew = np.zeros(nCells)
        myMPAS_O.tangentialVelocity = np.zeros((nEdges,nVertLevels))
        myMPAS_O.vertexMask = np.zeros((nVertices,nVertLevels))
        # Remember that if the dimension of a variable x defined in the Registry file is "nX nY nZ" then should be
        # specified as "nZ nY nX" due to the column-major vs row-major ordering of arrays in Fortran vs Python.
        myMPAS_O.time = 0.0
        os.chdir(cwd)
        if do_fixAngleEdge:
            print(' ')
            myMPAS_O.angleEdge[:] = (
            fixAngleEdge.fix_angleEdge(mesh_directory,my_mesh_file_name,determineYCellAlongLatitude=True,
                                       printOutput=print_Output,printRelevantMeshData=False))


# In[7]:

test_MPAS_O_1 = False
if test_MPAS_O_1:
    myMPAS_O = MPAS_O(True,print_Output=False)


# In[8]:

test_MPAS_O_2 = False
if test_MPAS_O_2:
    print_basic_geometry = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    # If you specify the base_mesh_file_name to be base_mesh.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,problem_type,
                      problem_is_linear,do_fixAngleEdge=True,print_Output=False)


# In[9]:

test_MPAS_O_3 = False
if test_MPAS_O_3:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,problem_type,
                      problem_is_linear,do_fixAngleEdge=True,print_Output=True)


# In[10]:

test_MPAS_O_4 = False
if test_MPAS_O_4:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    # If you specify the base_mesh_file_name to be base_mesh_NP.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh_NP.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,problem_type,
                      problem_is_linear,do_fixAngleEdge=True,print_Output=True)