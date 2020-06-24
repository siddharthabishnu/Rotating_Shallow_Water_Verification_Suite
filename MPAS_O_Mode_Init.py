
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
import matplotlib.pyplot as plt
with io.capture_output() as captured:
    import Common_Routines as CR
    import fixAngleEdge


# In[2]:

class Namelist:
    
    def __init__(myNamelist,mesh_type='uniform',problem_type='default',problem_is_linear=True,
                 periodicity='Periodic',time_integrator='forward_backward_predictor'):
        myNamelist.config_mesh_type = mesh_type
        myNamelist.config_problem_type = problem_type
        myNamelist.config_problem_is_linear = problem_is_linear
        myNamelist.config_periodicity = periodicity
        myNamelist.config_time_integrator = time_integrator
        if problem_type == 'default' or problem_type == 'Coastal_Kelvin_Wave':
            myNamelist.config_dt = 180.0
            # The default timestep is dictated by a Courant Number of 0.36 for a wave speed of 100 m/s and a 
            # uniform grid spacing of 50 km.
            myNamelist.config_forward_backward_predictor_parameter_gamma = 1.0
            myNamelist.config_gravity = 10.0
            myNamelist.config_mean_depth = 1000.0
            myNamelist.config_thickness_flux_type = 'centered'
            myNamelist.config_use_wetting_drying = False
            # Derived Parameters
            myNamelist.config_wave_speed = np.sqrt(myNamelist.config_gravity*myNamelist.config_mean_depth)
        if problem_is_linear:
            myNamelist.config_linearity_prefactor = 0.0
        else:
            myNamelist.config_linearity_prefactor = 1.0 


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

def DetermineCoriolisParameterAndBottomDepth(myMPAS_O):
    CoriolisParameter = 10.0**(-4.0)
    BottomDepthParameter = 1000.0
    problem_type = myMPAS_O.myNamelist.config_problem_type
    if problem_type == 'default' or problem_type == 'Coastal_Kelvin_Wave':
        myMPAS_O.fCell[:] = CoriolisParameter        
        myMPAS_O.fEdge[:] = CoriolisParameter
        myMPAS_O.fVertex[:] = CoriolisParameter
        myMPAS_O.bottomDepth[:] = BottomDepthParameter


# In[6]:

class MPAS_O:
    
    def __init__(myMPAS_O,print_basic_geometry,mesh_directory='Mesh+Initial_Condition+Registry_Files/Periodic',
                 base_mesh_file_name='base_mesh.nc',mesh_file_name='mesh.nc',mesh_type='uniform',
                 problem_type='default',problem_is_linear=True,periodicity='Periodic',do_fixAngleEdge=True,
                 print_Output=False,CourantNumber=0.5,useCourantNumberToDetermineTimeStep=False,
                 time_integrator='forward_backward_predictor',
                 specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False):
        myMPAS_O.myNamelist = Namelist(mesh_type,problem_type,problem_is_linear,periodicity,time_integrator)
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
        myMPAS_O.nNonPeriodicBoundaryEdges = 0
        myMPAS_O.nNonPeriodicBoundaryCells = 0
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
        # Define the grid spacing magnitude for a uniform mesh.
        useGridSpacingMagnitudeDefaultDefinition = True
        if useGridSpacingMagnitudeDefaultDefinition:
            myMPAS_O.gridSpacingMagnitude = myMPAS_O.gridSpacing[0]
        else:
            myMPAS_O.gridSpacingMagnitude = myMPAS_O.xCell[1] - myMPAS_O.xCell[0]
        # For a mesh with non-periodic zonal boundaries, adjust the zonal coordinates of the cell centers, 
        # vertices and edges.
        if periodicity == 'NonPeriodic_x':
            myMPAS_O.xCell[:] -= myMPAS_O.gridSpacingMagnitude
            myMPAS_O.xVertex[:] -= myMPAS_O.gridSpacingMagnitude
            myMPAS_O.xEdge[:] -= myMPAS_O.gridSpacingMagnitude
        # Specify the zonal and meridional extents of the domain.
        myMPAS_O.lX = max(myMPAS_O.xCell)
        myMPAS_O.lY = max(myMPAS_O.yVertex)
        # Define and initialize the following arrays not contained within either the base mesh file or the mesh 
        # file.
        myMPAS_O.fVertex = np.zeros(nVertices)
        myMPAS_O.fCell = np.zeros(nCells)
        myMPAS_O.fEdge = np.zeros(nEdges)
        myMPAS_O.bottomDepth = np.zeros(nCells)
        DetermineCoriolisParameterAndBottomDepth(myMPAS_O)
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
        myMPAS_O.normalVelocityNew = np.zeros((nEdges,nVertLevels))
        myMPAS_O.relativeVorticity = np.zeros((nVertices,nVertLevels))
        myMPAS_O.relativeVorticityCell = np.zeros((nCells,nVertLevels))
        myMPAS_O.sshCurrent = np.zeros(nCells)
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
        if useCourantNumberToDetermineTimeStep:
            dx = myMPAS_O.gridSpacingMagnitude # i.e. dx = myMPAS_O.dcEdge[0]
            WaveSpeed = myMPAS_O.myNamelist.config_wave_speed
            myMPAS_O.myNamelist.config_dt = CourantNumber*dx/WaveSpeed
            print('The timestep for Courant number %.2f is %.2f seconds.' 
                  %(CourantNumber,myMPAS_O.myNamelist.config_dt))
        myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells = (
        specifyExactSurfaceElevationAtNonPeriodicBoundaryCells)


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
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)


# In[9]:

test_MPAS_O_3 = False
if test_MPAS_O_3:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'Periodic'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=True)


# In[10]:

test_MPAS_O_4 = False
if test_MPAS_O_4:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    # If you specify the base_mesh_file_name to be base_mesh_NP.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh_NP.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=True)


# In[11]:

def Plot_MPAS_O_Mesh(myMPAS_O,output_directory,linewidth,linestyle,color,labels,labelfontsizes,labelpads,
                     tickfontsizes,title,titlefontsize,SaveAsPNG,FigureTitle,Show,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)    
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    dvEdge = max(myMPAS_O.dvEdge[:])
    for iEdge in range(0,myMPAS_O.nEdges):
        vertexID1 = myMPAS_O.verticesOnEdge[iEdge,0] - 1
        vertexID2 = myMPAS_O.verticesOnEdge[iEdge,1] - 1
        x1 = myMPAS_O.xVertex[vertexID1]
        x2 = myMPAS_O.xVertex[vertexID2]
        y1 = myMPAS_O.yVertex[vertexID1]
        y2 = myMPAS_O.yVertex[vertexID2]
        edgeLength = np.sqrt((x2 - x1)**2.0 + (y2 - y1)**2.0)
        if edgeLength <= dvEdge:
            plt.plot([x1,x2],[y1,y2],linewidth=linewidth,linestyle=linestyle,color=color)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    if useDefaultMethodToSpecifyTickFontSize:
        plt.xticks(fontsize=tickfontsizes[0])
        plt.yticks(fontsize=tickfontsizes[1])
    else:
        ax.tick_params(axis='x',labelsize=tickfontsizes[0])
        ax.tick_params(axis='y',labelsize=tickfontsizes[1])
    ax.set_title(title,fontsize=titlefontsize,y=1.035)
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[12]:

test_PlotMesh_1 = False
if test_PlotMesh_1:
    print_basic_geometry = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/Periodic'
    base_mesh_file_name = 'base_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'Periodic'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',['Latitude','Longitude'],[17.5,17.5],[10.0,10.0],
                     [15.0,15.0],'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[13]:

test_PlotMesh_2 = False
if test_PlotMesh_2:
    print_basic_geometry = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    # If you specify the base_mesh_file_name to be base_mesh.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',['Latitude','Longitude'],[17.5,17.5],[10.0,10.0],
                     [15.0,15.0],'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[14]:

test_PlotMesh_3 = False
if test_PlotMesh_3:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'Periodic'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',['Latitude','Longitude'],[17.5,17.5],[10.0,10.0],
                     [15.0,15.0],'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh_P',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[15]:

test_PlotMesh_4 = False
if test_PlotMesh_4:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    # If you specify the base_mesh_file_name to be base_mesh_NP.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh_NP.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',['Latitude','Longitude'],[17.5,17.5],[10.0,10.0],
                     [15.0,15.0],'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh_NP',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)