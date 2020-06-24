
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
    import MPAS_O_Mode_Init
    import MPAS_O_Shared


# In[2]:

def ComputeNormalVelocityTendency(myMPAS_O,normalVelocity,ssh):
    gravity = myMPAS_O.myNamelist.config_gravity
    qArr = np.zeros(myMPAS_O.nVertLevels)
    CoriolisTerm = np.zeros(myMPAS_O.nVertLevels)
    normalVelocityTendency = np.zeros((myMPAS_O.nEdges,myMPAS_O.nVertLevels))
    compute_layerThickness = True
    compute_layerThicknessEdge = True
    compute_relativeVorticityCell = False
    compute_divergence_kineticEnergyCell = True
    compute_tangentialVelocity = False
    compute_normalizedRelativeVorticityVertex_normalizedPlanetaryVorticityVertex = True
    compute_normalizedRelativeVorticityEdge_normalizedPlanetaryVorticityEdge = True
    compute_normalizedRelativeVorticityCell = False
    compute_these_variables = [compute_layerThickness,compute_layerThicknessEdge,compute_relativeVorticityCell,
                               compute_divergence_kineticEnergyCell,compute_tangentialVelocity,
                               compute_normalizedRelativeVorticityVertex_normalizedPlanetaryVorticityVertex,
                               compute_normalizedRelativeVorticityEdge_normalizedPlanetaryVorticityEdge,
                               compute_normalizedRelativeVorticityCell]
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

test_ComputeNormalVelocityTendency_1 = False
if test_ComputeNormalVelocityTendency_1:
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[4]:

test_ComputeNormalVelocityTendency_2 = False
if test_ComputeNormalVelocityTendency_2:
    print_basic_geometry = False
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[5]:

test_ComputeNormalVelocityTendency_3 = False
if test_ComputeNormalVelocityTendency_3:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[6]:

test_ComputeNormalVelocityTendency_4 = False
if test_ComputeNormalVelocityTendency_4:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear)
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)


# In[7]:

def ComputeSSHTendency(myMPAS_O,normalVelocity,ssh):
    sshTendency = np.zeros(myMPAS_O.nCells)
    compute_these_variables = np.zeros(8,dtype=bool)
    compute_these_variables[:] = False
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


# In[8]:

test_ComputeSSHTendency_1 = False
if test_ComputeSSHTendency_1:
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[9]:

test_ComputeSSHTendency_2 = False
if test_ComputeSSHTendency_2:
    print_basic_geometry = False
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[10]:

test_ComputeSSHTendency_3 = False
if test_ComputeSSHTendency_3:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# In[11]:

test_ComputeSSHTendency_4 = False
if test_ComputeSSHTendency_4:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear)
    sshTendency = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)


# # mpas_ocn_time_integration_forward_backward_predictor

# In[12]:

def ocn_time_integration_forward_backward_predictor(myMPAS_O):
    gamma = myMPAS_O.myNamelist.config_forward_backward_predictor_parameter_gamma
    dt = myMPAS_O.myNamelist.config_dt
    normalVelocityTendency = ComputeNormalVelocityTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                           myMPAS_O.sshCurrent)
    for iEdge in range(0,myMPAS_O.nEdges): 
        for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
            myMPAS_O.normalVelocityNew[iEdge,k] = (myMPAS_O.normalVelocityCurrent[iEdge,k] 
                                                   + dt*normalVelocityTendency[iEdge,k])
    sshTendency1 = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent)
    sshTendency2 = ComputeSSHTendency(myMPAS_O,myMPAS_O.normalVelocityNew,myMPAS_O.sshCurrent)
    for iCell in range(0,myMPAS_O.nCells):
        sshTendency = (1.0 - gamma)*sshTendency1[iCell] + gamma*sshTendency2[iCell]
        myMPAS_O.sshNew[iCell] = myMPAS_O.sshCurrent[iCell] + dt*sshTendency


# In[13]:

def ocn_shift_time_levels(myMPAS_O):
    myMPAS_O.normalVelocityCurrent[:,:] = myMPAS_O.normalVelocityNew[:,:]
    myMPAS_O.sshCurrent[:] = myMPAS_O.sshNew[:] 


# In[14]:

test_ocn_time_integration_forward_backward_predictor_1 = False
if test_ocn_time_integration_forward_backward_predictor_1:
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    nTimeSteps = 1
    for iTimeStep in range(0,nTimeSteps):
        ocn_time_integration_forward_backward_predictor(myMPAS_O)
        ocn_shift_time_levels(myMPAS_O)


# In[15]:

test_ocn_time_integration_forward_backward_predictor_2 = False
if test_ocn_time_integration_forward_backward_predictor_2:
    print_basic_geometry = False
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear)
    nTimeSteps = 1
    for iTimeStep in range(0,nTimeSteps):
        ocn_time_integration_forward_backward_predictor(myMPAS_O)
        ocn_shift_time_levels(myMPAS_O)


# In[16]:

test_ocn_time_integration_forward_backward_predictor_3 = False
if test_ocn_time_integration_forward_backward_predictor_3:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear)
    nTimeSteps = 1
    for iTimeStep in range(0,nTimeSteps):
        ocn_time_integration_forward_backward_predictor(myMPAS_O)
        ocn_shift_time_levels(myMPAS_O)


# In[17]:

test_ocn_time_integration_forward_backward_predictor_4 = False
if test_ocn_time_integration_forward_backward_predictor_4:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    problem_type = 'default'
    problem_is_linear = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear)
    nTimeSteps = 1
    for iTimeStep in range(0,nTimeSteps):
        ocn_time_integration_forward_backward_predictor(myMPAS_O)
        ocn_shift_time_levels(myMPAS_O)