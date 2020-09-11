
# coding: utf-8

# Name: MPAS_O_Mode_Forward.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains routines for computing the tendencies of the progostic variables and advancing them through one timestep. <br/>

# In[1]:

import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
from IPython.utils import io
with io.capture_output() as captured:
    import GeophysicalWaves_ExactSolutions_SourceTerms as GWESST
    import MPAS_O_Mode_Init
    import MPAS_O_Shared


# In[2]:

def ComputeNormalVelocityTendency(myMPAS_O,normalVelocity,ssh):
    gravity = myMPAS_O.myNamelist.config_gravity
    qArr = np.zeros(myMPAS_O.nVertLevels)
    CoriolisTerm = np.zeros(myMPAS_O.nVertLevels)
    normalVelocityTendency = np.zeros((myMPAS_O.nEdges,myMPAS_O.nVertLevels))
    compute_these_variables = np.zeros(8,dtype=bool)
    if not(myMPAS_O.myNamelist.config_problem_is_linear):
        compute_these_variables[0] = True # compute_layerThickness = True
        compute_these_variables[1] = True # compute_layerThicknessEdge = True
        compute_these_variables[3] = True # compute_divergence_kineticEnergyCell = True
        compute_these_variables[5] = True 
        # compute_normalizedRelativeVorticityVertex_normalizedPlanetaryVorticityVertex = True
        compute_these_variables[6] = True 
        # compute_normalizedRelativeVorticityEdge_normalizedPlanetaryVorticityEdge = True
        MPAS_O_Shared.ocn_diagnostic_solve(myMPAS_O,normalVelocity,ssh,compute_these_variables)
    LPF = myMPAS_O.myNamelist.config_linearity_prefactor
    for iEdge in range(0,myMPAS_O.nEdges):
        if myMPAS_O.boundaryEdge[iEdge] == 0.0: # i.e. if the edge is an interior one
            cellID1 = myMPAS_O.cellsOnEdge[iEdge,0]
            cell1 = cellID1 - 1
            cellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
            cell2 = cellID2 - 1
            invLength = 1.0/myMPAS_O.dcEdge[iEdge]
            for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                qArr[k] = 0.0
                CoriolisTerm[k] = 0.0
            for j in range(0,myMPAS_O.nEdgesOnEdge[iEdge]):
                eoeID = myMPAS_O.edgesOnEdge[iEdge,j]
                eoe = eoeID - 1
                edgeWeight = myMPAS_O.weightsOnEdge[iEdge,j]
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    if not(myMPAS_O.myNamelist.config_problem_is_linear):
                        workVorticity = 0.5*(myMPAS_O.normalizedRelativeVorticityEdge[iEdge,k] 
                                             + myMPAS_O.normalizedRelativeVorticityEdge[eoe,k])
                        qArr[k] += (edgeWeight*normalVelocity[eoe,k]*workVorticity
                                    *myMPAS_O.layerThicknessEdge[eoe,k])
                    CoriolisTerm[k] += edgeWeight*normalVelocity[eoe,k]*myMPAS_O.fEdge[eoe]
            for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                normalVelocityTendency[iEdge,k] = (myMPAS_O.edgeMask[iEdge,k]
                                                   *(CoriolisTerm[k] 
                                                     - gravity*(ssh[cell2] - ssh[cell1])/myMPAS_O.dcEdge[iEdge] 
                                                     + LPF*(qArr[k] 
                                                            - (myMPAS_O.kineticEnergyCell[cell2,k]
                                                               - myMPAS_O.kineticEnergyCell[cell1,k])*invLength)))
    return normalVelocityTendency


# In[3]:

test_ComputeNormalVelocityTendency_11 = False
if test_ComputeNormalVelocityTendency_11:
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,periodicity='Periodic')
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[4]:

test_ComputeNormalVelocityTendency_12 = False
if test_ComputeNormalVelocityTendency_12:
    print_basic_geometry = False
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[5]:

test_ComputeNormalVelocityTendency_13 = False
if test_ComputeNormalVelocityTendency_13:
    print_basic_geometry = False
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_y'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_y'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[6]:

test_ComputeNormalVelocityTendency_14 = False
if test_ComputeNormalVelocityTendency_14:
    print_basic_geometry = False
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_xy'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_xy'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[7]:

test_ComputeNormalVelocityTendency_21 = False
if test_ComputeNormalVelocityTendency_21:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'Periodic'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[8]:

test_ComputeNormalVelocityTendency_22 = False
if test_ComputeNormalVelocityTendency_22:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_x.nc'
    mesh_file_name = 'mesh_NP_x.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[9]:

test_ComputeNormalVelocityTendency_23 = False
if test_ComputeNormalVelocityTendency_23:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_y.nc'
    mesh_file_name = 'mesh_NP_y.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_y'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[10]:

test_ComputeNormalVelocityTendency_24 = False
if test_ComputeNormalVelocityTendency_24:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_xy.nc'
    mesh_file_name = 'mesh_NP_xy.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_xy'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[11]:

def ComputeSSHTendency(myMPAS_O,normalVelocity,ssh):
    sshTendency = np.zeros(myMPAS_O.nCells)
    compute_these_variables = np.zeros(8,dtype=bool)
    if not(myMPAS_O.myNamelist.config_problem_is_linear):
        compute_these_variables[0:2] = True
        MPAS_O_Shared.ocn_diagnostic_solve(myMPAS_O,normalVelocity,ssh,compute_these_variables)
    for iCell in range(0,myMPAS_O.nCells):
        for i in range(0,myMPAS_O.nEdgesOnCell[iCell]):
            iEdgeID = myMPAS_O.edgesOnCell[iCell,i]
            iEdge = iEdgeID - 1
            if myMPAS_O.myNamelist.config_problem_is_linear:
                flux = normalVelocity[iEdge,0]*myMPAS_O.myNamelist.config_mean_depth
            else:
                flux = normalVelocity[iEdge,0]*myMPAS_O.layerThicknessEdge[iEdge,0]
            sshTendency[iCell] += myMPAS_O.edgeSignOnCell[iCell,i]*flux*myMPAS_O.dvEdge[iEdge]
        sshTendency[iCell] /= myMPAS_O.areaCell[iCell]
    return sshTendency


# In[12]:

test_ComputeSSHTendency_11 = False
if test_ComputeSSHTendency_11:
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,periodicity='Periodic')
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[13]:

test_ComputeSSHTendency_12 = False
if test_ComputeSSHTendency_12:
    print_basic_geometry = False
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[14]:

test_ComputeSSHTendency_13 = False
if test_ComputeSSHTendency_13:
    print_basic_geometry = False
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_y'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_y'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[15]:

test_ComputeSSHTendency_14 = False
if test_ComputeSSHTendency_14:
    print_basic_geometry = False
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_xy'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_xy'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[16]:

test_ComputeSSHTendency_21 = False
if test_ComputeSSHTendency_21:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'Periodic'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[17]:

test_ComputeSSHTendency_22 = False
if test_ComputeSSHTendency_22:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_x.nc'
    mesh_file_name = 'mesh_NP_x.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[18]:

test_ComputeSSHTendency_23 = False
if test_ComputeSSHTendency_23:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_y.nc'
    mesh_file_name = 'mesh_NP_y.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_y'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[19]:

test_ComputeSSHTendency_24 = False
if test_ComputeSSHTendency_24:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_xy.nc'
    mesh_file_name = 'mesh_NP_xy.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_xy'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[20]:

def ocn_shift_time_levels(myMPAS_O):
    if myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Fourth_Order' and myMPAS_O.iTime > 1:
        myMPAS_O.normalVelocityTendencyThirdLast[:,:] = myMPAS_O.normalVelocityTendencySecondLast[:,:]
        myMPAS_O.sshTendencyThirdLast[:] = myMPAS_O.sshTendencySecondLast[:]
    if ((myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Third_Order'
         or myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Fourth_Order'
         or myMPAS_O.myNamelist.config_time_integrator == 'Generalized_FB_with_AB3_AM4_Step') 
        and myMPAS_O.iTime > 0):
        myMPAS_O.normalVelocityTendencySecondLast[:,:] = myMPAS_O.normalVelocityTendencyLast[:,:]
        myMPAS_O.sshTendencySecondLast[:] = myMPAS_O.sshTendencyLast[:]    
    if (myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Second_Order' 
        or myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Third_Order' 
        or myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Fourth_Order'
        or myMPAS_O.myNamelist.config_time_integrator == 'LF_TR_and_LF_AM3_with_FB_Feedback'
        or myMPAS_O.myNamelist.config_time_integrator == 'Generalized_FB_with_AB2_AM3_Step' 
        or myMPAS_O.myNamelist.config_time_integrator == 'Generalized_FB_with_AB3_AM4_Step'):
        myMPAS_O.normalVelocityTendencyLast[:,:] = myMPAS_O.normalVelocityTendencyCurrent[:,:]
        myMPAS_O.sshTendencyLast[:] = myMPAS_O.sshTendencyCurrent[:]    
    if (myMPAS_O.myNamelist.config_time_integrator == 'Leapfrog_Trapezoidal'
        or myMPAS_O.myNamelist.config_time_integrator == 'LF_TR_and_LF_AM3_with_FB_Feedback'):
        myMPAS_O.normalVelocityLast[:,:] = myMPAS_O.normalVelocityCurrent[:,:]
        myMPAS_O.sshLast[:] = myMPAS_O.sshCurrent[:] 
    myMPAS_O.normalVelocityCurrent[:,:] = myMPAS_O.normalVelocityNew[:,:]
    myMPAS_O.sshCurrent[:] = myMPAS_O.sshNew[:]


# In[21]:

def ocn_time_integration_Forward_Euler_Geophysical_Wave(myMPAS_O):
    dt = myMPAS_O.myNamelist.config_dt
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)
    for iEdge in range(0,myMPAS_O.nEdges): 
        if myMPAS_O.boundaryEdge[iEdge] == 1.0:
            # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary edges
            # to be equal to their exact analytical counterparts.          
            GeophysicalWaveExactZonalVelocity = (
            GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                              myMPAS_O.ExactSolutionParameters,
                                                              myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                              myMPAS_O.time+dt))           
            GeophysicalWaveExactMeridionalVelocity = (
            GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                   myMPAS_O.ExactSolutionParameters,
                                                                   myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                   myMPAS_O.time+dt))
            myMPAS_O.normalVelocityNew[iEdge,0] = (
            (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
             + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
        else:
            for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                myMPAS_O.normalVelocityNew[iEdge,k] = (myMPAS_O.normalVelocityCurrent[iEdge,k] 
                                                       + dt*normalVelocityTendency[iEdge,k])
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)
    for iCell in range(0,myMPAS_O.nCells):
        if myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells and myMPAS_O.boundaryCell[iCell] == 1.0:
            # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
            # non-periodic boundary cells to be equal to their exact analytical counterparts.         
            myMPAS_O.sshNew[iCell] = (
            GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                 myMPAS_O.ExactSolutionParameters,
                                                                 myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                 myMPAS_O.time+dt))      
        else:
            myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency[iCell]


# In[22]:

def ocn_time_integration_Forward_Backward_Geophysical_Wave(myMPAS_O):
    dt = myMPAS_O.myNamelist.config_dt
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)
    for iEdge in range(0,myMPAS_O.nEdges): 
        if myMPAS_O.boundaryEdge[iEdge] == 1.0:
            # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary edges
            # to be equal to their exact analytical counterparts.            
            GeophysicalWaveExactZonalVelocity = (
            GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                              myMPAS_O.ExactSolutionParameters,
                                                              myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                              myMPAS_O.time+dt))              
            GeophysicalWaveExactMeridionalVelocity = (
            GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                   myMPAS_O.ExactSolutionParameters,
                                                                   myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                   myMPAS_O.time+dt))            
            myMPAS_O.normalVelocityNew[iEdge,0] = (
            (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
             + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
        else:
            for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                myMPAS_O.normalVelocityNew[iEdge,k] = (myMPAS_O.normalVelocityCurrent[iEdge,k] 
                                                       + dt*normalVelocityTendency[iEdge,k])
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityNew,myMPAS_O.sshCurrent)
    for iCell in range(0,myMPAS_O.nCells):
        if myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells and myMPAS_O.boundaryCell[iCell] == 1.0:
            # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
            # non-periodic boundary cells to be equal to their exact analytical counterparts.      
            myMPAS_O.sshNew[iCell] = (
            GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                 myMPAS_O.ExactSolutionParameters,
                                                                 myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                 myMPAS_O.time+dt)) 
        else:
            myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency[iCell]


# In[23]:

def ocn_time_integration_Explicit_Midpoint_Method_Geophysical_Wave(myMPAS_O):
    dt = myMPAS_O.myNamelist.config_dt
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)
    normalVelocityAfterHalfTimeStep = np.zeros((myMPAS_O.nEdges,myMPAS_O.nVertLevels))
    sshAfterHalfTimeStep = np.zeros(myMPAS_O.nCells)
    for iEdge in range(0,myMPAS_O.nEdges): 
        if myMPAS_O.boundaryEdge[iEdge] == 1.0:
            # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary edges
            # to be equal to their exact analytical counterparts.
            GeophysicalWaveExactZonalVelocity = (
            GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                              myMPAS_O.ExactSolutionParameters,
                                                              myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                              myMPAS_O.time+0.5*dt))           
            GeophysicalWaveExactMeridionalVelocity = (
            GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                   myMPAS_O.ExactSolutionParameters,
                                                                   myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                   myMPAS_O.time+0.5*dt))
            normalVelocityAfterHalfTimeStep[iEdge,0] = (
            (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
             + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
        else:
            for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                normalVelocityAfterHalfTimeStep[iEdge,k] = (myMPAS_O.normalVelocityCurrent[iEdge,k] 
                                                            + 0.5*dt*normalVelocityTendency[iEdge,k])     
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)
    for iCell in range(0,myMPAS_O.nCells):
        if myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells and myMPAS_O.boundaryCell[iCell] == 1.0:
            # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
            # non-periodic boundary cells to be equal to their exact analytical counterparts.   
            sshAfterHalfTimeStep[iCell] = (
            GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                 myMPAS_O.ExactSolutionParameters,
                                                                 myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                 myMPAS_O.time+0.5*dt))                
        else:
            sshAfterHalfTimeStep[iCell] = myMPAS_O.sshCurrent[iCell] + 0.5*dt*sshTendency[iCell]
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,normalVelocityAfterHalfTimeStep,
                                                           sshAfterHalfTimeStep)
    for iEdge in range(0,myMPAS_O.nEdges): 
        if myMPAS_O.boundaryEdge[iEdge] == 1.0:
            # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary edges
            # to be equal to their exact analytical counterparts.
            GeophysicalWaveExactZonalVelocity = (
            GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                              myMPAS_O.ExactSolutionParameters,
                                                              myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                              myMPAS_O.time+dt))           
            GeophysicalWaveExactMeridionalVelocity = (
            GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                   myMPAS_O.ExactSolutionParameters,
                                                                   myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                   myMPAS_O.time+dt))            
            myMPAS_O.normalVelocityNew[iEdge,0] = (
            (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
             + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
        else:
            for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                myMPAS_O.normalVelocityNew[iEdge,k] = (myMPAS_O.normalVelocityCurrent[iEdge,k] 
                                                       + dt*normalVelocityTendency[iEdge,k])     
    sshTendency = ComputeSSHTendency(myMPAS_O,normalVelocityAfterHalfTimeStep,sshAfterHalfTimeStep)
    for iCell in range(0,myMPAS_O.nCells):
        if myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells and myMPAS_O.boundaryCell[iCell] == 1.0:
            # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
            # non-periodic boundary cells to be equal to their exact analytical counterparts. 
            myMPAS_O.sshNew[iCell] = (
            GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                 myMPAS_O.ExactSolutionParameters,
                                                                 myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                 myMPAS_O.time+dt))      
        else:
            myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency[iCell]            


# In[24]:

def ocn_time_integration_Williamson_Low_Storage_Runge_Kutta_Third_Order_Geophysical_Wave(myMPAS_O):
    dt = myMPAS_O.myNamelist.config_dt
    nStepsRK3 = 3
    aRK = np.array([0.0,-5.0/9.0,-153.0/128.0])
    bRK = np.array([0.0,1.0/3.0,3.0/4.0])
    next_bRK = np.array([1.0/3.0,3.0/4.0,1.0])
    gRK = np.array([1.0/3.0,15.0/16.0,8.0/15.0])
    temporary_normalVelocityTendency = np.zeros((myMPAS_O.nEdges,myMPAS_O.nVertLevels))
    temporary_sshTendency = np.zeros(myMPAS_O.nCells)
    myMPAS_O.normalVelocityNew[:,:] = myMPAS_O.normalVelocityCurrent[:,:]
    myMPAS_O.sshNew[:] = myMPAS_O.sshCurrent[:]
    normalVelocityNew = np.zeros((myMPAS_O.nEdges,myMPAS_O.nVertLevels))
    for iStep in range(0,nStepsRK3):
        normalVelocityNew[:,:] = myMPAS_O.normalVelocityNew[:,:]
        time = myMPAS_O.time + bRK[iStep]*dt
        next_time = myMPAS_O.time + next_bRK[iStep]*dt
        normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityNew,myMPAS_O.sshNew)
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.                
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  next_time))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                       next_time))
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    temporary_normalVelocityTendency[iEdge,k] = (
                    aRK[iStep]*temporary_normalVelocityTendency[iEdge,k] + normalVelocityTendency[iEdge,k])
                    myMPAS_O.normalVelocityNew[iEdge,k] += gRK[iStep]*dt*temporary_normalVelocityTendency[iEdge,k]
        sshTendency = ComputeSSHTendency(myMPAS_O,normalVelocityNew,myMPAS_O.sshNew)
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.  
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     next_time))                        
            else:
                temporary_sshTendency[iCell] = aRK[iStep]*temporary_sshTendency[iCell] + sshTendency[iCell]  
                myMPAS_O.sshNew[iCell] += gRK[iStep]*dt*temporary_sshTendency[iCell]


# In[25]:

def ocn_time_integration_Low_Storage_Runge_Kutta_Fourth_Order_Geophysical_Wave(myMPAS_O):
    dt = myMPAS_O.myNamelist.config_dt
    nStepsRK4 = 5
    aRK = np.zeros(5)
    aRK[1] = -1.0
    aRK[2] = -1.0/3.0 + 2.0**(2.0/3.0)/6.0 - 2.0*2.0**(1.0/3.0)/3.0
    aRK[3] = -2.0**(1.0/3.0) - 2.0**(2.0/3.0) - 2.0
    aRK[4] = -1.0 + 2.0**(1.0/3.0)
    bRK = np.zeros(5)
    bRK[1] = 2.0/3.0 + 2.0**(1.0/3.0)/3.0 + 2.0**(2.0/3.0)/6.0
    bRK[2] = 2.0/3.0 + 2.0**(1.0/3.0)/3.0 + 2.0**(2.0/3.0)/6.0
    bRK[3] = 1.0/3.0 - 2.0**(1.0/3.0)/3.0 - 2.0**(2.0/3.0)/6.0
    bRK[4] = 1.0
    next_bRK = np.zeros(5)
    for i in range(0,4):
        next_bRK[i] = bRK[i+1]
    next_bRK[4] = 1.0
    gRK = np.zeros(5)
    gRK[0] = 2.0/3.0 + 2.0**(1.0/3.0)/3.0 + 2.0**(2.0/3.0)/6.0
    gRK[1] = -2.0**(2.0/3.0)/6.0 + 1.0/6.0
    gRK[2] = -1.0/3.0 - 2.0*2.0**(1.0/3.0)/3.0 - 2.0**(2.0/3.0)/3.0
    gRK[3] = 1.0/3.0 - 2.0**(1.0/3.0)/3.0 - 2.0**(2.0/3.0)/6.0
    gRK[4] = 1.0/3.0 + 2.0**(1.0/3.0)/6.0 + 2.0**(2.0/3.0)/12.0    
    temporary_normalVelocityTendency = np.zeros((myMPAS_O.nEdges,myMPAS_O.nVertLevels))
    temporary_sshTendency = np.zeros(myMPAS_O.nCells)
    myMPAS_O.normalVelocityNew[:,:] = myMPAS_O.normalVelocityCurrent[:,:]
    myMPAS_O.sshNew[:] = myMPAS_O.sshCurrent[:]
    normalVelocityNew = np.zeros((myMPAS_O.nEdges,myMPAS_O.nVertLevels))
    for iStep in range(0,nStepsRK4):
        normalVelocityNew[:,:] = myMPAS_O.normalVelocityNew[:,:]
        time = myMPAS_O.time + bRK[iStep]*dt
        next_time = myMPAS_O.time + next_bRK[iStep]*dt
        normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityNew,myMPAS_O.sshNew)
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.                
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  next_time))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                       next_time))                
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    temporary_normalVelocityTendency[iEdge,k] = (
                    aRK[iStep]*temporary_normalVelocityTendency[iEdge,k] + normalVelocityTendency[iEdge,k])
                    myMPAS_O.normalVelocityNew[iEdge,k] += gRK[iStep]*dt*temporary_normalVelocityTendency[iEdge,k]
        sshTendency = ComputeSSHTendency(myMPAS_O,normalVelocityNew,myMPAS_O.sshNew)
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.                  
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     next_time))         
            else:
                temporary_sshTendency[iCell] = aRK[iStep]*temporary_sshTendency[iCell] + sshTendency[iCell]  
                myMPAS_O.sshNew[iCell] += gRK[iStep]*dt*temporary_sshTendency[iCell]


# In[26]:

def ocn_time_integration_Adams_Bashforth_Second_Order_Geophysical_Wave(myMPAS_O):
    if myMPAS_O.iTime == 0:
        ocn_time_integration_Explicit_Midpoint_Method_Geophysical_Wave(myMPAS_O)
    else:
        AB2 = np.array([1.5,-0.5])
        dt = myMPAS_O.myNamelist.config_dt
        myMPAS_O.normalVelocityTendencyCurrent = (
        ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent))
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  myMPAS_O.time+dt))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                       myMPAS_O.time+dt))
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    myMPAS_O.normalVelocityNew[iEdge,k] = (
                    (myMPAS_O.normalVelocityCurrent[iEdge,k]
                     + dt*(AB2[0]*myMPAS_O.normalVelocityTendencyCurrent[iEdge,k] 
                           + AB2[1]*myMPAS_O.normalVelocityTendencyLast[iEdge,k])))
        myMPAS_O.sshTendencyCurrent = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                         myMPAS_O.sshCurrent)
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.                
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     myMPAS_O.time+dt))        
            else:
                myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*(AB2[0]*myMPAS_O.sshTendencyCurrent[iCell]
                                                                          + AB2[1]*myMPAS_O.sshTendencyLast[iCell])


# In[27]:

def ocn_time_integration_Adams_Bashforth_Third_Order_Geophysical_Wave(myMPAS_O):
    if myMPAS_O.iTime == 0 or myMPAS_O.iTime == 1:
        ocn_time_integration_Williamson_Low_Storage_Runge_Kutta_Third_Order_Geophysical_Wave(myMPAS_O)
    else:
        AB3 = np.array([23.0/12.0,-4.0/3.0,5.0/12.0])
        dt = myMPAS_O.myNamelist.config_dt
        myMPAS_O.normalVelocityTendencyCurrent = (
        ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent))
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.                
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  myMPAS_O.time+dt))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                       myMPAS_O.time+dt))                
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    myMPAS_O.normalVelocityNew[iEdge,k] = (
                    (myMPAS_O.normalVelocityCurrent[iEdge,k]
                     + dt*(AB3[0]*myMPAS_O.normalVelocityTendencyCurrent[iEdge,k] 
                           + AB3[1]*myMPAS_O.normalVelocityTendencyLast[iEdge,k]
                           + AB3[2]*myMPAS_O.normalVelocityTendencySecondLast[iEdge,k])))
        myMPAS_O.sshTendencyCurrent = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                         myMPAS_O.sshCurrent)
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.             
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     myMPAS_O.time+dt))                          
            else:
                myMPAS_O.sshNew[iCell] = (
                myMPAS_O.sshCurrent[iCell] + dt*(AB3[0]*myMPAS_O.sshTendencyCurrent[iCell]
                                                 + AB3[1]*myMPAS_O.sshTendencyLast[iCell]
                                                 + AB3[2]*myMPAS_O.sshTendencySecondLast[iCell]))


# In[28]:

def ocn_time_integration_Adams_Bashforth_Fourth_Order_Geophysical_Wave(myMPAS_O):
    if myMPAS_O.iTime == 0 or myMPAS_O.iTime == 1 or myMPAS_O.iTime == 2:
        ocn_time_integration_Low_Storage_Runge_Kutta_Fourth_Order_Geophysical_Wave(myMPAS_O)
    else:
        AB4 = np.array([55.0/24.0,-59.0/24.0,37.0/24.0,-3.0/8.0])
        dt = myMPAS_O.myNamelist.config_dt
        myMPAS_O.normalVelocityTendencyCurrent = (
        ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent))
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.                
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  myMPAS_O.time+dt))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                       myMPAS_O.time+dt))
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    myMPAS_O.normalVelocityNew[iEdge,k] = (
                    (myMPAS_O.normalVelocityCurrent[iEdge,k]
                     + dt*(AB4[0]*myMPAS_O.normalVelocityTendencyCurrent[iEdge,k] 
                           + AB4[1]*myMPAS_O.normalVelocityTendencyLast[iEdge,k]
                           + AB4[2]*myMPAS_O.normalVelocityTendencySecondLast[iEdge,k]
                           + AB4[3]*myMPAS_O.normalVelocityTendencyThirdLast[iEdge,k])))
        myMPAS_O.sshTendencyCurrent = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                         myMPAS_O.sshCurrent)
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.                 
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     myMPAS_O.time+dt))                  
            else:
                myMPAS_O.sshNew[iCell] = (
                myMPAS_O.sshCurrent[iCell] + dt*(AB4[0]*myMPAS_O.sshTendencyCurrent[iCell]
                                                 + AB4[1]*myMPAS_O.sshTendencyLast[iCell]
                                                 + AB4[2]*myMPAS_O.sshTendencySecondLast[iCell]
                                                 + AB4[3]*myMPAS_O.sshTendencyThirdLast[iCell]))


# In[29]:

def ocn_time_integration_Leapfrog_Trapezoidal_Geophysical_Wave(myMPAS_O):
    if myMPAS_O.iTime == 0:
        ocn_time_integration_Explicit_Midpoint_Method_Geophysical_Wave(myMPAS_O)
    else:    
        dt = myMPAS_O.myNamelist.config_dt
        myMPAS_O.normalVelocityTendencyCurrent = (
        ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent))
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  myMPAS_O.time+dt))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                       myMPAS_O.time+dt))
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    myMPAS_O.normalVelocityNew[iEdge,k] = (
                    myMPAS_O.normalVelocityLast[iEdge,k] + 2.0*dt*myMPAS_O.normalVelocityTendencyCurrent[iEdge,k])
        sshTendencyCurrent = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.               
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     myMPAS_O.time+dt))       
            else:
                myMPAS_O.sshNew[iCell] = myMPAS_O.sshLast[iCell] + 2.0*dt*sshTendencyCurrent[iCell]       
        normalVelocityTendencyNew = (
        ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityNew,myMPAS_O.sshNew))        
        sshTendencyNew = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityNew,myMPAS_O.sshNew)        
        normalVelocityTendency = 0.5*(myMPAS_O.normalVelocityTendencyCurrent + normalVelocityTendencyNew)    
        sshTendency = 0.5*(sshTendencyCurrent + sshTendencyNew)
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  myMPAS_O.time+dt))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                       myMPAS_O.time+dt))
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    myMPAS_O.normalVelocityNew[iEdge,k] = (
                    myMPAS_O.normalVelocityCurrent[iEdge,k] + dt*normalVelocityTendency[iEdge,k])
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.                
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     myMPAS_O.time+dt))
            else:
                myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency[iCell]        


# In[30]:

def ocn_time_integration_LF_TR_and_LF_AM3_with_FB_Feedback_Geophysical_Wave(myMPAS_O):
    if myMPAS_O.iTime == 0:
        if myMPAS_O.myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'SecondOrderAccurate_LF_TR':
            ocn_time_integration_Explicit_Midpoint_Method_Geophysical_Wave(myMPAS_O)            
        elif (myMPAS_O.myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'ThirdOrderAccurate_LF_AM3'
              or (myMPAS_O.myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_Type 
                  == 'ThirdOrderAccurate_MaximumStabilityRange')):
            ocn_time_integration_Williamson_Low_Storage_Runge_Kutta_Third_Order_Geophysical_Wave(myMPAS_O)  
        elif ((myMPAS_O.myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_Type 
               == 'FourthOrderAccurate_MinimumTruncationError')
              or (myMPAS_O.myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_Type 
                  == 'FourthOrderAccurate_MaximumStabilityRange')):
            ocn_time_integration_Low_Storage_Runge_Kutta_Fourth_Order_Geophysical_Wave(myMPAS_O)         
    else:    
        beta = myMPAS_O.myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta
        gamma = myMPAS_O.myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma
        epsilon = myMPAS_O.myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon
        dt = myMPAS_O.myNamelist.config_dt
        normalVelocityPredictor = np.zeros((myMPAS_O.nEdges,myMPAS_O.nVertLevels))
        sshPredictor = np.zeros(myMPAS_O.nCells)
        myMPAS_O.normalVelocityTendencyCurrent = (
        ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent))
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.                
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  myMPAS_O.time+dt))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],
                                                                       myMPAS_O.yEdge[iEdge],myMPAS_O.time+dt))
                normalVelocityPredictor[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    normalVelocityPredictor[iEdge,k] = (
                    myMPAS_O.normalVelocityLast[iEdge,k] + 2.0*dt*myMPAS_O.normalVelocityTendencyCurrent[iEdge,k])
        myMPAS_O.sshTendencyCurrent = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                         myMPAS_O.sshCurrent)   
        sshTendency1 = myMPAS_O.sshTendencyLast    
        sshTendency2 = myMPAS_O.sshTendencyCurrent
        sshTendency3 = ComputeSSHTendency(myMPAS_O,normalVelocityPredictor,myMPAS_O.sshCurrent)
        sshTendency = (1.0 - 2.0*beta)*sshTendency2 + beta*(sshTendency3 + sshTendency1)
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.       
                sshPredictor[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     myMPAS_O.time+dt))       
            else:
                sshPredictor[iCell] = myMPAS_O.sshLast[iCell] + 2.0*dt*sshTendency[iCell]    
        normalVelocityTendency1 = myMPAS_O.normalVelocityTendencyLast  
        normalVelocityTendency2 = myMPAS_O.normalVelocityTendencyCurrent
        normalVelocityTendency3 = ComputeNormalVelocityTendency(myMPAS_O,normalVelocityPredictor,sshPredictor)
        normalVelocityTendency = (-gamma*normalVelocityTendency1 + (0.5 + 2.0*gamma)*normalVelocityTendency2 
                                  + (0.5 - gamma)*normalVelocityTendency3)
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.           
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  myMPAS_O.time+dt))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],
                                                                       myMPAS_O.yEdge[iEdge],myMPAS_O.time+dt))
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    myMPAS_O.normalVelocityNew[iEdge,k] = (
                    myMPAS_O.normalVelocityCurrent[iEdge,k] + dt*normalVelocityTendency[iEdge,k])
        sshTendency1 = myMPAS_O.sshTendencyLast    
        sshTendency2 = myMPAS_O.sshTendencyCurrent
        sshTendency3 = ComputeSSHTendency(myMPAS_O,normalVelocityPredictor,sshPredictor)     
        sshTendency4 = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityNew,sshPredictor)  
        sshTendency = ((0.5 - gamma)*(epsilon*sshTendency4 + (1.0 - epsilon)*sshTendency3) 
                       + (0.5 + 2.0*gamma)*sshTendency2 - gamma*sshTendency1)
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.                
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     myMPAS_O.time+dt))
            else:
                myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency[iCell] 


# In[31]:

def ocn_time_integration_Forward_Backward_with_RK2_Feedback_Geophysical_Wave(myMPAS_O):
    beta = myMPAS_O.myNamelist.config_Forward_Backward_with_RK2_Feedback_parameter_beta
    epsilon = myMPAS_O.myNamelist.config_Forward_Backward_with_RK2_Feedback_parameter_epsilon
    dt = myMPAS_O.myNamelist.config_dt
    normalVelocityPredictor = np.zeros((myMPAS_O.nEdges,myMPAS_O.nVertLevels))
    sshPredictor = np.zeros(myMPAS_O.nCells)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)
    for iEdge in range(0,myMPAS_O.nEdges): 
        if myMPAS_O.boundaryEdge[iEdge] == 1.0:
            # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary edges
            # to be equal to their exact analytical counterparts.            
            GeophysicalWaveExactZonalVelocity = (
            GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                              myMPAS_O.ExactSolutionParameters,
                                                              myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                              myMPAS_O.time+dt))           
            GeophysicalWaveExactMeridionalVelocity = (
            GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                   myMPAS_O.ExactSolutionParameters,
                                                                   myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                   myMPAS_O.time+dt))
            normalVelocityPredictor[iEdge,0] = (
            (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
             + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
        else:
            for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                normalVelocityPredictor[iEdge,k] = (myMPAS_O.normalVelocityCurrent[iEdge,k] 
                                                    + dt*normalVelocityTendency[iEdge,k])
    sshTendency1 = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)
    sshTendency2 = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityNew,myMPAS_O.sshCurrent)
    sshTendency = (1.0 - beta)*sshTendency1 + beta*sshTendency2
    for iCell in range(0,myMPAS_O.nCells):
        if myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells and myMPAS_O.boundaryCell[iCell] == 1.0:
            # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
            # non-periodic boundary cells to be equal to their exact analytical counterparts.
            sshPredictor[iCell] = (
            GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                 myMPAS_O.ExactSolutionParameters,
                                                                 myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                 myMPAS_O.time+dt))  
        else:
            sshPredictor[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency[iCell]
    normalVelocityTendency1 = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                            myMPAS_O.sshCurrent)
    normalVelocityTendency2 = ComputeNormalVelocityTendency(myMPAS_O,normalVelocityPredictor,sshPredictor)
    normalVelocityTendency = 0.5*(normalVelocityTendency1 + normalVelocityTendency2)
    for iEdge in range(0,myMPAS_O.nEdges): 
        if myMPAS_O.boundaryEdge[iEdge] == 1.0:
            # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary edges
            # to be equal to their exact analytical counterparts.
            GeophysicalWaveExactZonalVelocity = (
            GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                              myMPAS_O.ExactSolutionParameters,
                                                              myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                              myMPAS_O.time+dt))           
            GeophysicalWaveExactMeridionalVelocity = (
            GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                   myMPAS_O.ExactSolutionParameters,
                                                                   myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                   myMPAS_O.time+dt))
            myMPAS_O.normalVelocityNew[iEdge,0] = (
            (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
             + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
        else:
            for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                myMPAS_O.normalVelocityNew[iEdge,k] = (myMPAS_O.normalVelocityCurrent[iEdge,k] 
                                                       + dt*normalVelocityTendency[iEdge,k])        
    sshTendency1 = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)
    sshTendency2 = ComputeSSHTendency(myMPAS_O,normalVelocityPredictor,sshPredictor)
    sshTendency3 = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityNew,sshPredictor)
    sshTendency = 0.5*(sshTendency1 + (1.0 - epsilon)*sshTendency2 + epsilon*sshTendency3)
    for iCell in range(0,myMPAS_O.nCells):
        if myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells and myMPAS_O.boundaryCell[iCell] == 1.0:
            # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
            # non-periodic boundary cells to be equal to their exact analytical counterparts.      
            myMPAS_O.sshNew[iCell] = (
            GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                 myMPAS_O.ExactSolutionParameters,
                                                                 myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                 myMPAS_O.time+dt))       
        else:
            myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency[iCell]    


# In[32]:

def ocn_time_integration_Generalized_FB_with_AB2_AM3_Step_Geophysical_Wave(myMPAS_O):
    if myMPAS_O.iTime == 0:
        if ((myMPAS_O.myNamelist.config_Generalized_FB_with_AB2_AM3_Step_Type 
             == 'ThirdOrderAccurate_WideStabilityRange')
              or (myMPAS_O.myNamelist.config_Generalized_FB_with_AB2_AM3_Step_Type 
                  == 'ThirdOrderAccurate_WeakAsymptoticInstabilityOfPhysicalModes')):
            ocn_time_integration_Williamson_Low_Storage_Runge_Kutta_Third_Order_Geophysical_Wave(myMPAS_O)    
        elif myMPAS_O.myNamelist.config_Generalized_FB_with_AB2_AM3_Step_Type == 'FourthOrderAccurate':
            ocn_time_integration_Low_Storage_Runge_Kutta_Fourth_Order_Geophysical_Wave(myMPAS_O)  
    else:   
        beta = myMPAS_O.myNamelist.config_Generalized_FB_with_AB2_AM3_Step_parameter_beta
        gamma = myMPAS_O.myNamelist.config_Generalized_FB_with_AB2_AM3_Step_parameter_gamma
        epsilon = myMPAS_O.myNamelist.config_Generalized_FB_with_AB2_AM3_Step_parameter_epsilon
        dt = myMPAS_O.myNamelist.config_dt
        myMPAS_O.normalVelocityTendencyCurrent = (
        ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent))
        normalVelocityTendency1 = myMPAS_O.normalVelocityTendencyLast
        normalVelocityTendency2 = myMPAS_O.normalVelocityTendencyCurrent
        normalVelocityTendency = -beta*normalVelocityTendency1 + (1.0 + beta)*normalVelocityTendency2
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.                
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  myMPAS_O.time+dt))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                       myMPAS_O.time+dt))
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    myMPAS_O.normalVelocityNew[iEdge,k] = (myMPAS_O.normalVelocityCurrent[iEdge,k] 
                                                           + dt*normalVelocityTendency[iEdge,k])    
        myMPAS_O.sshTendencyCurrent = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                         myMPAS_O.sshCurrent)                    
        sshTendency1 = myMPAS_O.sshTendencyLast    
        sshTendency2 = myMPAS_O.sshTendencyCurrent
        sshTendency3 = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityNew,myMPAS_O.sshCurrent)  
        sshTendency = epsilon*sshTendency1 + gamma*sshTendency2 + (1.0 - gamma - epsilon)*sshTendency3
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.                
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     myMPAS_O.time+dt))
            else:
                myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency[iCell]                     


# In[33]:

def ocn_time_integration_Generalized_FB_with_AB3_AM4_Step_Geophysical_Wave(myMPAS_O):
    if myMPAS_O.iTime == 0 or myMPAS_O.iTime == 1:
        if (myMPAS_O.myNamelist.config_Generalized_FB_with_AB3_AM4_Step_Type 
            == 'SecondOrderAccurate_OptimumChoice_ROMS'):
            ocn_time_integration_Explicit_Midpoint_Method_Geophysical_Wave(myMPAS_O)            
        elif (myMPAS_O.myNamelist.config_Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_AB3_AM4'
              or (myMPAS_O.myNamelist.config_Generalized_FB_with_AB3_AM4_Step_Type 
                  == 'ThirdOrderAccurate_MaximumStabilityRange')
              or (myMPAS_O.myNamelist.config_Generalized_FB_with_AB3_AM4_Step_Type 
                  == 'ThirdOrderAccurate_OptimumChoice')):
            ocn_time_integration_Williamson_Low_Storage_Runge_Kutta_Third_Order_Geophysical_Wave(myMPAS_O)   
        elif (myMPAS_O.myNamelist.config_Generalized_FB_with_AB3_AM4_Step_Type 
              == 'FourthOrderAccurate_MaximumStabilityRange'):
            ocn_time_integration_Low_Storage_Runge_Kutta_Fourth_Order_Geophysical_Wave(myMPAS_O)         
    else:   
        beta = myMPAS_O.myNamelist.config_Generalized_FB_with_AB3_AM4_Step_parameter_beta
        gamma = myMPAS_O.myNamelist.config_Generalized_FB_with_AB3_AM4_Step_parameter_gamma
        epsilon = myMPAS_O.myNamelist.config_Generalized_FB_with_AB3_AM4_Step_parameter_epsilon
        delta = myMPAS_O.myNamelist.config_Generalized_FB_with_AB3_AM4_Step_parameter_delta
        dt = myMPAS_O.myNamelist.config_dt
        myMPAS_O.normalVelocityTendencyCurrent = (
        ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent))
        normalVelocityTendency1 = myMPAS_O.normalVelocityTendencySecondLast
        normalVelocityTendency2 = myMPAS_O.normalVelocityTendencyLast
        normalVelocityTendency3 = myMPAS_O.normalVelocityTendencyCurrent
        normalVelocityTendency = (beta*normalVelocityTendency1 - (0.5 + 2.0*beta)*normalVelocityTendency2
                                  + (1.5 + beta)*normalVelocityTendency3)
        for iEdge in range(0,myMPAS_O.nEdges): 
            if myMPAS_O.boundaryEdge[iEdge] == 1.0:
                # Specify the boundary conditions by enforcing the normal velocities at the non-periodic boundary 
                # edges to be equal to their exact analytical counterparts.                
                GeophysicalWaveExactZonalVelocity = (
                GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                  myMPAS_O.ExactSolutionParameters,
                                                                  myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                  myMPAS_O.time+dt))           
                GeophysicalWaveExactMeridionalVelocity = (
                GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                                       myMPAS_O.ExactSolutionParameters,
                                                                       myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge],
                                                                       myMPAS_O.time+dt))
                myMPAS_O.normalVelocityNew[iEdge,0] = (
                (GeophysicalWaveExactZonalVelocity*np.cos(myMPAS_O.angleEdge[iEdge])
                 + GeophysicalWaveExactMeridionalVelocity*np.sin(myMPAS_O.angleEdge[iEdge])))
            else:
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    myMPAS_O.normalVelocityNew[iEdge,k] = (myMPAS_O.normalVelocityCurrent[iEdge,k] 
                                                           + dt*normalVelocityTendency[iEdge,k])    
        myMPAS_O.sshTendencyCurrent = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                         myMPAS_O.sshCurrent)                    
        sshTendency1 = myMPAS_O.sshTendencySecondLast    
        sshTendency2 = myMPAS_O.sshTendencyLast
        sshTendency3 = myMPAS_O.sshTendencyCurrent
        sshTendency4 = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityNew,myMPAS_O.sshCurrent)  
        sshTendency = (epsilon*sshTendency1 + gamma*sshTendency2 + (1.0 - gamma - delta - epsilon)*sshTendency3
                       + delta*sshTendency4)
        for iCell in range(0,myMPAS_O.nCells):
            if (myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells 
                and myMPAS_O.boundaryCell[iCell] == 1.0):
                # Specify the boundary conditions by enforcing the surface elevations at the centers of the 
                # non-periodic boundary cells to be equal to their exact analytical counterparts.                
                myMPAS_O.sshNew[iCell] = (
                GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                     myMPAS_O.ExactSolutionParameters,
                                                                     myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell],
                                                                     myMPAS_O.time+dt))
            else:
                myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency[iCell]  


# In[34]:

def ocn_time_integration_Geophysical_Wave(myMPAS_O):
    if myMPAS_O.myNamelist.config_time_integrator == 'Forward_Euler':
        ocn_time_integration_Forward_Euler_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Forward_Backward':
        ocn_time_integration_Forward_Backward_Geophysical_Wave(myMPAS_O)        
    elif myMPAS_O.myNamelist.config_time_integrator == 'Explicit_Midpoint_Method':
        ocn_time_integration_Explicit_Midpoint_Method_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Williamson_Low_Storage_Runge_Kutta_Third_Order':
        ocn_time_integration_Williamson_Low_Storage_Runge_Kutta_Third_Order_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Low_Storage_Runge_Kutta_Fourth_Order':
        ocn_time_integration_Low_Storage_Runge_Kutta_Fourth_Order_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Second_Order':
        ocn_time_integration_Adams_Bashforth_Second_Order_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Third_Order':
        ocn_time_integration_Adams_Bashforth_Third_Order_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Fourth_Order':
        ocn_time_integration_Adams_Bashforth_Fourth_Order_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Leapfrog_Trapezoidal':
        ocn_time_integration_Leapfrog_Trapezoidal_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'LF_TR_and_LF_AM3_with_FB_Feedback':
        ocn_time_integration_LF_TR_and_LF_AM3_with_FB_Feedback_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Forward_Backward_with_RK2_Feedback':
        ocn_time_integration_Forward_Backward_with_RK2_Feedback_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Generalized_FB_with_AB2_AM3_Step':
        ocn_time_integration_Generalized_FB_with_AB2_AM3_Step_Geophysical_Wave(myMPAS_O)
    elif myMPAS_O.myNamelist.config_time_integrator == 'Generalized_FB_with_AB3_AM4_Step':
        ocn_time_integration_Generalized_FB_with_AB3_AM4_Step_Geophysical_Wave(myMPAS_O)