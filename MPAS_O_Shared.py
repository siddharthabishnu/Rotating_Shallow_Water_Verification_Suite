
# coding: utf-8

# Name: MPAS_O_Shared.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains routines for setting up some (a) MPAS-Ocean mesh parameters using the initial condition data, and (b) diagnostic variables.

# In[1]:

import numpy as np
import io as inputoutput
import sys
from IPython.utils import io
with io.capture_output() as captured: 
    import MPAS_O_Mode_Init


# In[2]:

def test_function():
    print('Hello my human programmer!')

def suppress_output_of_function_call(suppressOutput): # Note that this routine is not perfect
    if suppressOutput:
        # Create a text trap and redirect stdout
        text_trap = inputoutput.StringIO()
        sys.stdout = text_trap
        test_function()
        # Now restore stdout function
        sys.stdout = sys.__stdout__        
    else:
        test_function()
        
suppress_output_of_function_call(False)


# # mpas_ocn_init_routines

# In[3]:

def ocn_init_routines_compute_max_level(myMPAS_O,printValues=False):
    for iEdge in range(0,myMPAS_O.nEdges):
        iCellID1 = myMPAS_O.cellsOnEdge[iEdge,0]
        iCell1 = iCellID1 - 1
        iCellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
        iCell2 = iCellID2 - 1 
        if iCellID1 == 0 or iCellID2 == 0: # i.e. if the edge is along a non-periodic boundary of the domain
            myMPAS_O.boundaryEdge[iEdge,:] = 1.0
            myMPAS_O.maxLevelEdgeTop[iEdge] = -1
    # maxLevelEdgeTop is the minimum (shallowest) of the surrounding cells
    for iEdge in range(0,myMPAS_O.nEdges):
        iCellID1 = myMPAS_O.cellsOnEdge[iEdge,0]
        iCell1 = iCellID1 - 1
        iCellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
        iCell2 = iCellID2 - 1        
        if not(iCellID1 == 0 or iCellID2 == 0):
            myMPAS_O.maxLevelEdgeTop[iEdge] = min(myMPAS_O.maxLevelCell[iCell1],myMPAS_O.maxLevelCell[iCell2]) 
    # maxLevelEdgeBot is the maximum (deepest) of the surrounding cells
    for iEdge in range(0,myMPAS_O.nEdges):
        iCellID1 = myMPAS_O.cellsOnEdge[iEdge,0]
        iCell1 = iCellID1 - 1
        iCellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
        iCell2 = iCellID2 - 1     
        if iCellID1 == 0:
            myMPAS_O.maxLevelEdgeBot[iEdge] = myMPAS_O.maxLevelCell[iCell2]
        elif iCellID2 == 0:
            myMPAS_O.maxLevelEdgeBot[iEdge] = myMPAS_O.maxLevelCell[iCell1]
        else:
            myMPAS_O.maxLevelEdgeBot[iEdge] = max(myMPAS_O.maxLevelCell[iCell1],myMPAS_O.maxLevelCell[iCell2]) 
    # maxLevelVertexBot is the maximum (deepest) of the surrounding cells
    for iVertex in range(0,myMPAS_O.nVertices):
        iCellID1 = myMPAS_O.cellsOnVertex[iVertex,0]
        iCell1 = iCellID1 - 1
        if iCellID1 == 0:
            myMPAS_O.maxLevelVertexBot[iVertex] = -1
        else:
            myMPAS_O.maxLevelVertexBot[iVertex] = myMPAS_O.maxLevelCell[iCell1]
        for i in range(1,myMPAS_O.vertexDegree):
            iCellID = myMPAS_O.cellsOnVertex[iVertex,i]
            iCell = iCellID - 1
            if iCellID == 0:
                myMPAS_O.maxLevelVertexBot[iVertex] = max(myMPAS_O.maxLevelVertexBot[iVertex],-1)
            else:
                myMPAS_O.maxLevelVertexBot[iVertex] = max(myMPAS_O.maxLevelVertexBot[iVertex], 
                                                          myMPAS_O.maxLevelCell[iCell])
    # maxLevelVertexTop is the minimum (shallowest) of the surrounding cells
    for iVertex in range(0,myMPAS_O.nVertices):
        iCellID1 = myMPAS_O.cellsOnVertex[iVertex,0]
        iCell1 = iCellID1 - 1
        if iCellID1 == 0:
            myMPAS_O.maxLevelVertexTop[iVertex] = -1
        else:
            myMPAS_O.maxLevelVertexTop[iVertex] = myMPAS_O.maxLevelCell[iCell1]
        for i in range(1,myMPAS_O.vertexDegree):
            iCellID = myMPAS_O.cellsOnVertex[iVertex,i]
            iCell = iCellID - 1
            if iCellID == 0:
                myMPAS_O.maxLevelVertexTop[iVertex] = min(myMPAS_O.maxLevelVertexTop[iVertex],-1)
            else:
                myMPAS_O.maxLevelVertexTop[iVertex] = min(myMPAS_O.maxLevelVertexTop[iVertex], 
                                                          myMPAS_O.maxLevelCell[iCell])
    determine_boundaryEdge_Generalized_Method = True
    # Set boundary edge      
    if determine_boundaryEdge_Generalized_Method:
        myMPAS_O.boundaryEdge[:,:] = 1
    myMPAS_O.edgeMask[:,:] = 0
    # In order to assign the same value to all the elements of an array in Python in a 'vectorized' fashion, 
    # always use the above syntax instead of 
    #
    # myMPAS_O.boundaryEdge = 1
    # myMPAS_O.edgeMask = 0
    #
    # In the latter case, the type of the variable itself is changed from an array to a float. So, as you can see,
    # the convenience of not having to define the variables in a Python code comes with a catch.
    for iEdge in range(0,myMPAS_O.nEdges):
        index_UpperLimit = myMPAS_O.maxLevelEdgeTop[iEdge]
        if index_UpperLimit > -1:
            if determine_boundaryEdge_Generalized_Method:
                myMPAS_O.boundaryEdge[iEdge,0:index_UpperLimit+1] = 0
            myMPAS_O.edgeMask[iEdge,0:index_UpperLimit+1] = 1
    # Find cells and vertices that have an edge on the boundary
    for iEdge in range(0,myMPAS_O.nEdges):
        iCellID1 = myMPAS_O.cellsOnEdge[iEdge,0]
        iCell1 = iCellID1 - 1
        iCellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
        iCell2 = iCellID2 - 1   
        iVertexID1 = myMPAS_O.verticesOnEdge[iEdge,0]
        iVertex1 = iVertexID1 - 1
        iVertexID2 = myMPAS_O.verticesOnEdge[iEdge,1]
        iVertex2 = iVertexID2 - 1                  
        for k in range(0,myMPAS_O.nVertLevels):
            if myMPAS_O.boundaryEdge[iEdge,k] == 1:
                if iCellID1 != 0:
                    myMPAS_O.boundaryCell[iCell1,k] = 1
                if iCellID2 != 0:    
                    myMPAS_O.boundaryCell[iCell2,k] = 1
                myMPAS_O.boundaryVertex[iVertex1,k] = 1
                myMPAS_O.boundaryVertex[iVertex2,k] = 1
    for iCell in range(0,myMPAS_O.nCells):
        for k in range(0,myMPAS_O.nVertLevels):
            if myMPAS_O.maxLevelCell[iCell] >= k:
                myMPAS_O.cellMask[iCell,k] = 1
    for iVertex in range(0,myMPAS_O.nVertices):
        for k in range(0,myMPAS_O.nVertLevels):
            if myMPAS_O.maxLevelVertexBot[iVertex] >= k:
                myMPAS_O.vertexMask[iVertex,k] = 1
    for iEdge in range(0,myMPAS_O.nEdges):
        if myMPAS_O.boundaryEdge[iEdge,0] == 1.0:
            myMPAS_O.nNonPeriodicBoundaryEdges += 1
    for iCell in range(0,myMPAS_O.nCells):
        if myMPAS_O.boundaryCell[iCell] == 1.0:
            myMPAS_O.nNonPeriodicBoundaryCells += 1
    if printValues:
        print('maxLevelEdgeTop is:')
        print(myMPAS_O.maxLevelEdgeTop)
        print('maxLevelEdgeBot is:')
        print(myMPAS_O.maxLevelEdgeBot)
        print('maxLevelVertexBot is:')
        print(myMPAS_O.maxLevelVertexBot)
        print('maxLevelVertexTop is:')
        print(myMPAS_O.maxLevelVertexTop)
        print('boundaryEdge is:')
        print(myMPAS_O.boundaryEdge[:,0])
        print('The number of non-periodic boundary edges is %d.' %(myMPAS_O.nNonPeriodicBoundaryEdges))
        print('edgeMask is:')
        print(myMPAS_O.edgeMask[:,0])
        print('boundaryCell is:')
        print(myMPAS_O.boundaryCell[:,0])
        print('The number of non-periodic boundary cells is %d.' %(myMPAS_O.nNonPeriodicBoundaryCells))
        print('boundaryVertex is:')
        print(myMPAS_O.boundaryVertex[:,0])
        print('cellMask is:')
        print(myMPAS_O.cellMask[:,0])
        print('vertexMask is:')
        print(myMPAS_O.vertexMask[:,0])


# In[4]:

test_ocn_init_routines_compute_max_level_1 = False
if test_ocn_init_routines_compute_max_level_1:
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    ocn_init_routines_compute_max_level(myMPAS_O)


# In[5]:

test_ocn_init_routines_compute_max_level_2 = False
if test_ocn_init_routines_compute_max_level_2:
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
    ocn_init_routines_compute_max_level(myMPAS_O)


# In[6]:

test_ocn_init_routines_compute_max_level_3 = False
if test_ocn_init_routines_compute_max_level_3:
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
    ocn_init_routines_compute_max_level(myMPAS_O,printValues=True)


# In[7]:

test_ocn_init_routines_compute_max_level_4 = False
if test_ocn_init_routines_compute_max_level_4:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    ocn_init_routines_compute_max_level(myMPAS_O,printValues=True)


# In[8]:

def ocn_init_routines_setup_sign_and_index_fields(myMPAS_O):
    for iCell in range(0,myMPAS_O.nCells):
        iCellID = iCell + 1
        for i in range(0,myMPAS_O.nEdgesOnCell[iCell]):
            iEdgeID = myMPAS_O.edgesOnCell[iCell,i]
            iEdge = iEdgeID - 1
            iVertexID = myMPAS_O.verticesOnCell[iCell,i]
            iVertex = iVertexID - 1
            # Vector points from cell 1 to cell 2            
            if myMPAS_O.cellsOnEdge[iEdge,0] == iCellID:
                myMPAS_O.edgeSignOnCell[iCell,i] = -1
            else:
                myMPAS_O.edgeSignOnCell[iCell,i] = 1                
            for j in range(0,myMPAS_O.vertexDegree):
                if myMPAS_O.cellsOnVertex[iVertex,j] == iCellID:
                    myMPAS_O.kiteIndexOnCell[iCell,i] = j + 1
    printKiteIndexOnCell = False
    if printKiteIndexOnCell:
        for iCell in range(0,myMPAS_O.nCells):
            for i in range(0,myMPAS_O.nEdgesOnCell[iCell]):
                print('On edge %d of cell %2d, kiteIndexOnCell is %d.' 
                      %(i, iCell, myMPAS_O.kiteIndexOnCell[iCell,i]))
    for iVertex in range(0,myMPAS_O.nVertices):
        iVertexID = iVertex + 1
        for i in range(0,myMPAS_O.vertexDegree):
            iEdgeID = myMPAS_O.edgesOnVertex[iVertex,i]
            iEdge = iEdgeID - 1
            # Vector points from vertex 1 to vertex 2
            if myMPAS_O.verticesOnEdge[iEdge,0] == iVertexID:
                myMPAS_O.edgeSignOnVertex[iVertex,i] = -1
            else:
                myMPAS_O.edgeSignOnVertex[iVertex,i] = 1


# In[9]:

test_ocn_init_routines_setup_sign_and_index_fields_1 = False
if test_ocn_init_routines_setup_sign_and_index_fields_1:
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)


# In[10]:

test_ocn_init_routines_setup_sign_and_index_fields_2 = False
if test_ocn_init_routines_setup_sign_and_index_fields_2:
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
    ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)


# In[11]:

test_ocn_init_routines_setup_sign_and_index_fields_3 = False
if test_ocn_init_routines_setup_sign_and_index_fields_3:
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
    ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)


# In[12]:

test_ocn_init_routines_setup_sign_and_index_fields_4 = False
if test_ocn_init_routines_setup_sign_and_index_fields_4:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)


# # mpas_ocn_diagnostics_routines

# In[13]:

def ocn_relativeVorticity_circulation(myMPAS_O,normalVelocity):
    for iVertex in range(0,myMPAS_O.nVertices):
        myMPAS_O.circulation[iVertex,:] = 0.0
        myMPAS_O.relativeVorticity[iVertex,:] = 0.0
        invAreaTri1 = 1.0/myMPAS_O.areaTriangle[iVertex]
        for i in range(0,myMPAS_O.vertexDegree):
            iEdgeID = myMPAS_O.edgesOnVertex[iVertex,i]
            iEdge = iEdgeID - 1
            for k in range(0,myMPAS_O.maxLevelVertexBot[iVertex]+1):
                r_tmp = myMPAS_O.dcEdge[iEdge]*normalVelocity[iEdge,k]
                myMPAS_O.circulation[iVertex,k] += myMPAS_O.edgeSignOnVertex[iVertex,i]*r_tmp
                myMPAS_O.relativeVorticity[iVertex,k] += myMPAS_O.edgeSignOnVertex[iVertex,i]*r_tmp*invAreaTri1 


# In[14]:

test_ocn_relativeVorticity_circulation_1 = False
if test_ocn_relativeVorticity_circulation_1:
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    ocn_relativeVorticity_circulation(myMPAS_O,myMPAS_O.normalVelocityCurrent)


# In[15]:

test_ocn_relativeVorticity_circulation_2 = False
if test_ocn_relativeVorticity_circulation_2:
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
    ocn_relativeVorticity_circulation(myMPAS_O,myMPAS_O.normalVelocityCurrent)


# In[16]:

test_ocn_relativeVorticity_circulation_3 = False
if test_ocn_relativeVorticity_circulation_3:
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
    ocn_relativeVorticity_circulation(myMPAS_O,myMPAS_O.normalVelocityCurrent)


# In[17]:

test_ocn_relativeVorticity_circulation_4 = False
if test_ocn_relativeVorticity_circulation_4:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    ocn_relativeVorticity_circulation(myMPAS_O,myMPAS_O.normalVelocityCurrent)


# # mpas_ocn_diagnostics

# In[18]:

def ocn_diagnostic_solve(myMPAS_O,normalVelocity,ssh,compute_these_variables):
    compute_layerThickness = compute_these_variables[0]
    compute_layerThicknessEdge = compute_these_variables[1]
    compute_relativeVorticityCell = compute_these_variables[2]
    compute_divergence_kineticEnergyCell = compute_these_variables[3]
    compute_tangentialVelocity = compute_these_variables[4]
    compute_normalizedRelativeVorticityVertex_normalizedPlanetaryVorticityVertex = compute_these_variables[5]
    compute_normalizedRelativeVorticityEdge_normalizedPlanetaryVorticityEdge = compute_these_variables[6]
    compute_normalizedRelativeVorticityCell = compute_these_variables[7]
    if (compute_layerThickness or compute_layerThicknessEdge or 
        compute_normalizedRelativeVorticityVertex_normalizedPlanetaryVorticityVertex):
        for iCell in range(0,myMPAS_O.nCells):
            myMPAS_O.layerThicknessCurrent[iCell] = ssh[iCell] + myMPAS_O.bottomDepth[iCell]    
    if compute_layerThicknessEdge:
        if (not(myMPAS_O.myNamelist.config_use_wetting_drying) 
            or (myMPAS_O.myNamelist.config_use_wetting_drying 
                and myMPAS_O.myNamelist.config_thickness_flux_type == 'centered')):
            for iEdge in range(0,myMPAS_O.nEdges):
                cellID1 = myMPAS_O.cellsOnEdge[iEdge,0]
                cell1 = cellID1 - 1
                cellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
                cell2 = cellID2 - 1
                for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                    # Central-differenced
                    myMPAS_O.layerThicknessEdge[iEdge,k] = 0.5*(myMPAS_O.layerThicknessCurrent[cell1,k] 
                                                                + myMPAS_O.layerThicknessCurrent[cell2,k])
        else:
            if (myMPAS_O.myNamelist.config_use_wetting_drying
                and myMPAS_O.myNamelist.config_thickness_flux_type != 'centered'):
                if myMPAS_O.myNamelist.config_thickness_flux_type == 'upwind':
                    for iEdge in range(0,myMPAS_O.nEdges):
                        cellID1 = myMPAS_O.cellsOnEdge[iEdge,0]
                        cell1 = cellID1 - 1
                        cellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
                        cell2 = cellID2 - 1
                        for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                            # Upwind
                            if normalVelocity[iEdge,k] > 0.0:
                                myMPAS_O.layerThicknessEdge[iEdge,k] = myMPAS_O.layerThicknessCurrent[cell1,k]
                            elif normalVelocity[iEdge,k] < 0.0:
                                myMPAS_O.layerThicknessEdge[iEdge,k] = myMPAS_O.layerThicknessCurrent[cell2,k]
                            else:
                                myMPAS_O.layerThicknessEdge[iEdge,k] = (
                                max(myMPAS_O.layerThicknessCurrent[cell1,k],
                                    myMPAS_O.layerThicknessCurrent[cell2,k]))
            else:
                print('Thickness flux option %s is not known!' %(myMPAS_O.myNamelist.config_thickness_flux_type))
                sys.exit()
    if (compute_relativeVorticityCell or
        compute_normalizedRelativeVorticityVertex_normalizedPlanetaryVorticityVertex or
        compute_normalizedRelativeVorticityEdge_normalizedPlanetaryVorticityEdge or
        compute_normalizedRelativeVorticityCell):
        ocn_relativeVorticity_circulation(myMPAS_O,normalVelocity)
    if compute_relativeVorticityCell:
        for iCell in range(0,myMPAS_O.nCells):
            myMPAS_O.relativeVorticityCell[iCell,:] = 0.0
            invAreaCell1 = 1.0/myMPAS_O.areaCell[iCell]
            for i in range(0,myMPAS_O.nEdgesOnCell[iCell]):
                jID = myMPAS_O.kiteIndexOnCell[iCell,i]
                j = jID - 1
                iVertexID = myMPAS_O.verticesOnCell[iCell,i]
                iVertex = iVertexID - 1
                for k in range(0,myMPAS_O.maxLevelCell[iCell]+1):
                    myMPAS_O.relativeVorticityCell[iCell,k] += (
                    myMPAS_O.kiteAreasOnVertex[iVertex,j]*myMPAS_O.relativeVorticity[iVertex,k]*invAreaCell1)
    if compute_divergence_kineticEnergyCell:
        for iCell in range(0,myMPAS_O.nCells):
            myMPAS_O.divergence[iCell,:] = 0.0
            myMPAS_O.kineticEnergyCell[iCell,:] = 0.0
            invAreaCell1 = 1.0/myMPAS_O.areaCell[iCell]
            for i in range(0,myMPAS_O.nEdgesOnCell[iCell]):
                iEdgeID = myMPAS_O.edgesOnCell[iCell,i]
                iEdge = iEdgeID - 1
                edgeSignOnCell_temp = myMPAS_O.edgeSignOnCell[iCell,i]
                dcEdge_temp = myMPAS_O.dcEdge[iEdge]
                dvEdge_temp = myMPAS_O.dvEdge[iEdge]
                for k in range(0,myMPAS_O.maxLevelCell[iCell]+1):
                    r_tmp = dvEdge_temp*normalVelocity[iEdge,k]*invAreaCell1
                    myMPAS_O.divergence[iCell,k] -= edgeSignOnCell_temp*r_tmp
                    myMPAS_O.kineticEnergyCell[iCell,k] += 0.25*r_tmp*dcEdge_temp*normalVelocity[iEdge,k]
    if compute_tangentialVelocity:
        for iEdge in range(0,myMPAS_O.nEdges):
            if myMPAS_O.boundaryEdge[iEdge] == 0: # i.e. if the edge is an interior one
                myMPAS_O.tangentialVelocity[iEdge,:] = 0.0
                # Compute tangential velocities
                for i in range(0,myMPAS_O.nEdgesOnEdge[iEdge]):
                    eoeID = myMPAS_O.edgesOnEdge[iEdge,i]
                    eoe = eoeID - 1
                    weightsOnEdge_temp = myMPAS_O.weightsOnEdge[iEdge,i]
                    for k in range(0,myMPAS_O.maxLevelEdgeTop[iEdge]+1):
                        myMPAS_O.tangentialVelocity[iEdge,k] += weightsOnEdge_temp*normalVelocity[eoe,k]
    if (compute_normalizedRelativeVorticityVertex_normalizedPlanetaryVorticityVertex or
        compute_normalizedRelativeVorticityEdge_normalizedPlanetaryVorticityEdge or
        compute_normalizedRelativeVorticityCell):
        for iVertex in range(0,myMPAS_O.nVertices):
            invAreaTri1 = 1.0/myMPAS_O.areaTriangle[iVertex]
            for k in range(0,myMPAS_O.maxLevelVertexBot[iVertex]+1):
                layerThicknessVertex = 0.0
                for i in range(0,myMPAS_O.vertexDegree):
                    iCellID = myMPAS_O.cellsOnVertex[iVertex,i]
                    iCell = iCellID - 1
                    layerThicknessVertex += (myMPAS_O.layerThicknessCurrent[iCell,k]
                                             *myMPAS_O.kiteAreasOnVertex[iVertex,i])
                layerThicknessVertex *= invAreaTri1   
                myMPAS_O.normalizedRelativeVorticityVertex[iVertex,k] = (myMPAS_O.relativeVorticity[iVertex,k]
                                                                         /layerThicknessVertex)
                myMPAS_O.normalizedPlanetaryVorticityVertex[iVertex,k] = (myMPAS_O.fVertex[iVertex]
                                                                          /layerThicknessVertex)
    if compute_normalizedRelativeVorticityEdge_normalizedPlanetaryVorticityEdge:
        for iEdge in range(0,myMPAS_O.nEdges):
            myMPAS_O.normalizedRelativeVorticityEdge[iEdge,:] = 0.0
            myMPAS_O.normalizedPlanetaryVorticityEdge[iEdge,:] = 0.0
            vertexID1 = myMPAS_O.verticesOnEdge[iEdge,0]
            vertex1 = vertexID1 - 1
            vertexID2 = myMPAS_O.verticesOnEdge[iEdge,1]
            vertex2 = vertexID2 - 1
            for k in range(0,myMPAS_O.maxLevelEdgeBot[iEdge]+1):
                myMPAS_O.normalizedRelativeVorticityEdge[iEdge,k] = (
                0.5*(myMPAS_O.normalizedRelativeVorticityVertex[vertex1,k]
                     + myMPAS_O.normalizedRelativeVorticityVertex[vertex2,k]))
                myMPAS_O.normalizedPlanetaryVorticityEdge[iEdge,k] = (
                0.5*(myMPAS_O.normalizedPlanetaryVorticityVertex[vertex1,k] 
                     + myMPAS_O.normalizedPlanetaryVorticityVertex[vertex2,k]))
    if compute_normalizedRelativeVorticityCell:
        for iCell in range(0,myMPAS_O.nCells):
            myMPAS_O.normalizedRelativeVorticityCell[iCell,:] = 0.0
            invAreaCell1 = 1.0/myMPAS_O.areaCell[iCell]
            for i in range(0,myMPAS_O.nEdgesOnCell[iCell]):
                jID = myMPAS_O.kiteIndexOnCell[iCell,i]
                j = jID - 1
                iVertexID = myMPAS_O.verticesOnCell[iCell,i]
                iVertex = iVertexID - 1
                for k in range(0,myMPAS_O.maxLevelCell[iCell]+1):
                    myMPAS_O.normalizedRelativeVorticityCell[iCell,k] += (
                    (myMPAS_O.kiteAreasOnVertex[iVertex,j]*myMPAS_O.normalizedRelativeVorticityVertex[iVertex,k]
                     *invAreaCell1))


# In[19]:

test_ocn_diagnostic_solve_1 = False
if test_ocn_diagnostic_solve_1:
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    compute_these_variables = np.ones(8,dtype=bool)
    ocn_diagnostic_solve(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent,compute_these_variables)


# In[20]:

test_ocn_diagnostic_solve_2 = False
if test_ocn_diagnostic_solve_2:
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
    compute_these_variables = np.ones(8,dtype=bool)
    ocn_diagnostic_solve(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent,compute_these_variables)


# In[21]:

test_ocn_diagnostic_solve_3 = False
if test_ocn_diagnostic_solve_3:
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
    compute_these_variables = np.ones(8,dtype=bool)
    ocn_diagnostic_solve(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent,compute_these_variables)


# In[22]:

test_ocn_diagnostic_solve_4 = False
if test_ocn_diagnostic_solve_4:
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    compute_these_variables = np.ones(8,dtype=bool)
    ocn_diagnostic_solve(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent,compute_these_variables)