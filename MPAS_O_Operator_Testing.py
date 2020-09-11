
# coding: utf-8

# Name: MPAS_O_Operator_Testing.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains routines for determining and testing the various spatial operators on a MPAS-Ocean mesh e.g. gradient, divergence, curl etc.

# In[1]:

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import io as inputoutput
import os
import sys
from IPython.utils import io
with io.capture_output() as captured: 
    import Common_Routines as CR
    import MPAS_O_Mode_Init
    import MPAS_O_Shared


# In[2]:

def surface_elevation_1(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    return eta


# In[3]:

def surface_elevation_gradient_1(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_x = eta0*(2.0*np.pi/lX)*np.cos(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_y = eta0*(2.0*np.pi/lY)*np.sin(2.0*np.pi*x/lX)*np.cos(2.0*np.pi*y/lY)
    return eta_x, eta_y


# In[4]:

def surface_elevation_laplacian_1(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_xx = -(2.0*np.pi/lX)**2.0*eta
    eta_yy = -(2.0*np.pi/lY)**2.0*eta
    laplacian = eta_xx + eta_yy
    return laplacian


# In[5]:

def problem_specific_prefix_1():
    prefix = 'Expt1_'
    return prefix


# In[6]:

def surface_elevation_2(lX,lY,x,y):
    xCenter = 0.5*lX
    yCenter = 0.5*lY
    RangeX = 0.2*lX
    RangeY = 0.2*lY
    eta0 = 0.1
    eta = eta0*np.exp(-((np.sin(2.0*np.pi*x/lX))**2.0 + (np.sin(2.0*np.pi*y/lY))**2.0))
    return eta


# In[7]:

def surface_elevation_gradient_2_functional_form():
    lX = sp.Symbol('lX')
    lY = sp.Symbol('lY')
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    eta0 = 0.1
    eta = eta0*sp.exp(-((sp.sin(2.0*sp.pi*x/lX))**2.0 + (sp.sin(2.0*sp.pi*y/lY))**2.0))
    eta_x = sp.diff(eta,x)
    eta_y = sp.diff(eta,y)
    eta_x_np = lambdify((lX,lY,x,y), eta_x, modules=["numpy","sympy"])
    eta_y_np = lambdify((lX,lY,x,y), eta_y, modules=["numpy","sympy"])
    # Note that np in eta_x_np and eta_y_np stands for numpy.
    return eta_x_np, eta_y_np


# In[8]:

eta_x_np, eta_y_np = surface_elevation_gradient_2_functional_form()


# In[9]:

def surface_elevation_gradient_2(lX,lY,x,y):
    eta_x = eta_x_np(lX,lY,x,y)
    eta_y = eta_y_np(lX,lY,x,y)
    return eta_x, eta_y


# In[10]:

def surface_elevation_laplacian_2_functional_form():
    lX = sp.Symbol('lX')
    lY = sp.Symbol('lY')
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    eta0 = 0.1
    eta = eta0*sp.exp(-((sp.sin(2.0*sp.pi*x/lX))**2.0 + (sp.sin(2.0*sp.pi*y/lY))**2.0))
    eta_xx = sp.diff(eta,x,x)
    eta_yy = sp.diff(eta,y,y)
    laplacian = eta_xx + eta_yy
    laplacian_np = lambdify((lX,lY,x,y), laplacian, modules=["numpy","sympy"])
    # Note that np in laplacian_np stands for numpy.
    return laplacian_np


# In[11]:

surface_elevation_laplacian_2 = surface_elevation_laplacian_2_functional_form()


# In[12]:

def problem_specific_prefix_2():
    prefix = 'Expt2_'
    return prefix


# In[13]:

surface_elevation = surface_elevation_1


# In[14]:

surface_elevation_gradient = surface_elevation_gradient_1


# In[15]:

surface_elevation_laplacian = surface_elevation_laplacian_1


# In[16]:

problem_specific_prefix = problem_specific_prefix_1


# In[17]:

def velocity(lX,lY,x,y):
    eta_x, eta_y = surface_elevation_gradient(lX,lY,x,y) 
    f = 10.0**(-4.0)
    g = 10.0
    v = g*eta_x/f
    u = -g*eta_y/f
    return u, v


# In[18]:

def velocity_curl(lX,lY,x,y):
    f = 10.0**(-4.0)
    g = 10.0
    zeta = g/f*surface_elevation_laplacian(lX,lY,x,y)
    return zeta


# In[19]:

def plot_SurfaceElevation_NormalVelocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                         mesh_file_name,periodicity,output_directory):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,periodicity=periodicity)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,
                                           periodicity=periodicity)
    if periodicity == 'NonPeriodic_x':
        iEdgeStartingIndex = 1
    else:
        iEdgeStartingIndex = 0
    prefix = problem_specific_prefix()
    
    mySurfaceElevation = np.zeros(myMPAS_O.nCells)
    for iCell in range(0,myMPAS_O.nCells):
        mySurfaceElevation[iCell] = (
        surface_elevation(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell]))
    myZonalVelocity = np.zeros(myMPAS_O.nEdges) 
    myMeridionalVelocity = np.zeros(myMPAS_O.nEdges) 
    myResultantVelocity = np.zeros(myMPAS_O.nEdges)   
    myNormalVelocity = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        u, v = velocity(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge])
        myZonalVelocity[iEdge] = u
        myMeridionalVelocity[iEdge] = v
        myResultantVelocity[iEdge] = np.sqrt(u**2.0 + v**2.0)
        myNormalVelocity[iEdge] = u*np.cos(myMPAS_O.angleEdge[iEdge]) + v*np.sin(myMPAS_O.angleEdge[iEdge])
    if plotFigures:
        xLabel = 'Zonal Distance (km)'
        yLabel = 'Meridional Distance (km)'
        Title = 'Surface Elevation'
        FigureTitle = prefix + 'SurfaceElevation'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          mySurfaceElevation,300,False,[0.0,0.0],6,plt.cm.jet,
                                                          13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],
                                                          [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                          fig_size=[10.0,10.0],tick_units_in_km=True,
                                                          cbarlabelformat='%.2f')
        Title = 'Zonal Velocity'
        FigureTitle = prefix + 'ZonalVelocity'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],
                                                          myMPAS_O.yEdge[iEdgeStartingIndex:],
                                                          myZonalVelocity[iEdgeStartingIndex:],300,False,[0.0,0.0],
                                                          6,plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],
                                                          [10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,
                                                          False,fig_size=[10.0,10.0],tick_units_in_km=True,
                                                          cbarlabelformat='%.2f')          
        Title = 'Meridional Velocity'
        FigureTitle = prefix + 'MeridionalVelocity'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],
                                                          myMPAS_O.yEdge[iEdgeStartingIndex:],
                                                          myMeridionalVelocity[iEdgeStartingIndex:],300,False,
                                                          [0.0,0.0],6,plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],
                                                          [10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,
                                                          False,fig_size=[10.0,10.0],tick_units_in_km=True,
                                                          cbarlabelformat='%.2f')    
        Title = 'Resultant Velocity'
        FigureTitle = prefix + 'ResultantVelocity'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],
                                                          myMPAS_O.yEdge[iEdgeStartingIndex:],
                                                          myResultantVelocity[iEdgeStartingIndex:],300,False,
                                                          [0.0,0.0],6,plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],
                                                          [10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,
                                                          False,fig_size=[10.0,10.0],tick_units_in_km=True,
                                                          cbarlabelformat='%.2f')   
        Title = 'Normal Velocity'
        FigureTitle = prefix + 'NormalVelocity'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],
                                                          myMPAS_O.yEdge[iEdgeStartingIndex:],
                                                          myNormalVelocity[iEdgeStartingIndex:],300,False,
                                                          [0.0,0.0],6,plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],
                                                          [10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,
                                                          False,fig_size=[10.0,10.0],tick_units_in_km=True,
                                                          cbarlabelformat='%.2f')


# In[20]:

do_plot_SurfaceElevation_NormalVelocity_11 = False
if do_plot_SurfaceElevation_NormalVelocity_11:
    periodicity = 'Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    plot_SurfaceElevation_NormalVelocity(True,True,' ',' ',' ',periodicity,output_directory)


# In[21]:

do_plot_SurfaceElevation_NormalVelocity_12 = False
if do_plot_SurfaceElevation_NormalVelocity_12:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    plot_SurfaceElevation_NormalVelocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                         mesh_file_name,periodicity,output_directory)


# In[22]:

do_plot_SurfaceElevation_NormalVelocity_13 = False
if do_plot_SurfaceElevation_NormalVelocity_13:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_y'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_y'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_y'
    plot_SurfaceElevation_NormalVelocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                         mesh_file_name,periodicity,output_directory)


# In[23]:

do_plot_SurfaceElevation_NormalVelocity_14 = False
if do_plot_SurfaceElevation_NormalVelocity_14:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_xy'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_xy'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_xy'
    plot_SurfaceElevation_NormalVelocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                         mesh_file_name,periodicity,output_directory)


# In[24]:

def ComputeNormalAndTangentialComponentsAtEdge(myVectorQuantityAtEdge,angleEdge,returningComponent):
    nEdges = len(angleEdge)
    if returningComponent == 'normal' or returningComponent == 'both':
        myVectorQuantityAtEdgeNormalComponent = np.zeros(nEdges)
    if returningComponent == 'tangential' or returningComponent == 'both':
        myVectorQuantityAtEdgeTangentialComponent = np.zeros(nEdges)    
    for iEdge in range(0,nEdges):
        xComponent = myVectorQuantityAtEdge[iEdge,0]
        yComponent = myVectorQuantityAtEdge[iEdge,1]
        if returningComponent == 'normal' or returningComponent == 'both':
            myVectorQuantityAtEdgeNormalComponent[iEdge] = (xComponent*np.cos(angleEdge[iEdge]) 
                                                            + yComponent*np.sin(angleEdge[iEdge]))
        if returningComponent == 'tangential' or returningComponent == 'both':
            myVectorQuantityAtEdgeTangentialComponent[iEdge] = (yComponent*np.cos(angleEdge[iEdge]) 
                                                                - xComponent*np.sin(angleEdge[iEdge]))
    if returningComponent == 'normal':
        return myVectorQuantityAtEdgeNormalComponent
    elif returningComponent == 'tangential':
        return myVectorQuantityAtEdgeTangentialComponent
    else: # if returningComponent == 'both':
        return myVectorQuantityAtEdgeNormalComponent, myVectorQuantityAtEdgeTangentialComponent


# In[25]:

def analytical_gradient_operator(nEdges,myScalarQuantityGradientComponentsAtEdge,angleEdge):
    myScalarQuantityGradientNormalToEdge = (
    ComputeNormalAndTangentialComponentsAtEdge(myScalarQuantityGradientComponentsAtEdge,angleEdge,'normal'))
    return myScalarQuantityGradientNormalToEdge


# In[26]:

def numerical_gradient_operator(myMPAS_O,myScalarQuantity,periodicity='Periodic'):
    if periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy':
        MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    myScalarQuantityGradientNormalToEdge = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        if not((periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' 
                or periodicity == 'NonPeriodic_xy') and myMPAS_O.boundaryEdge[iEdge] == 1.0):
            cellID1 = myMPAS_O.cellsOnEdge[iEdge,0]
            cell1 = cellID1 - 1
            cellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
            cell2 = cellID2 - 1
            invLength = 1.0/myMPAS_O.dcEdge[iEdge]    
            myScalarQuantityGradientNormalToEdge[iEdge] = (
            (myScalarQuantity[cell2] - myScalarQuantity[cell1])*invLength)
    return myScalarQuantityGradientNormalToEdge


# In[27]:

def test_numerical_gradient_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     periodicity,output_directory):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,periodicity=periodicity)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,
                                           periodicity=periodicity)
    if periodicity == 'NonPeriodic_x':
        iEdgeStartingIndex = 1
    else:
        iEdgeStartingIndex = 0
    prefix = problem_specific_prefix()
    mySurfaceElevation = np.zeros(myMPAS_O.nCells)
    for iCell in range(0,myMPAS_O.nCells):
        mySurfaceElevation[iCell] = surface_elevation(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xCell[iCell],
                                                      myMPAS_O.yCell[iCell])
    mySurfaceElevationGradientAtEdge = np.zeros((myMPAS_O.nEdges,2))
    for iEdge in range(0,myMPAS_O.nEdges):
        mySurfaceElevationGradientAtEdge[iEdge,0], mySurfaceElevationGradientAtEdge[iEdge,1] = (
        surface_elevation_gradient(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge]))
    myAnalyticalSurfaceElevationGradientNormalToEdge = (
    analytical_gradient_operator(myMPAS_O.nEdges,mySurfaceElevationGradientAtEdge,myMPAS_O.angleEdge))
    myNumericalSurfaceElevationGradientNormalToEdge = (
    numerical_gradient_operator(myMPAS_O,mySurfaceElevation,periodicity))
    # Reduce the numerical normal gradient at the midpoint of every non-periodic boundary edge to be equal to its
    # exact analytical counterpart. Note that due to the staggered nature of the MPAS-O grid, we do not even need 
    # to compute the gradient of any variable at the midpoint of a non-periodic boundary edge. For instance, in 
    # order to update the normal velocity defined at the midpoint of an edge, we need to determine the gradient of 
    # the surface elevation there. However, if the edge is aligned along a non-periodic boundary, the normal 
    # velocity at its midpoint is specified to be zero at all times and the entire exercise of updating it every
    # time step is not carried out.
    for iEdge in range(0,myMPAS_O.nEdges):
        if ((periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy') 
            and myMPAS_O.boundaryEdge[iEdge] == 1.0):
            myNumericalSurfaceElevationGradientNormalToEdge[iEdge] = (
            myAnalyticalSurfaceElevationGradientNormalToEdge[iEdge])  
    mySurfaceElevationGradientNormalToEdgeError = (
    myNumericalSurfaceElevationGradientNormalToEdge - myAnalyticalSurfaceElevationGradientNormalToEdge)
    MaxErrorNorm = np.linalg.norm(mySurfaceElevationGradientNormalToEdgeError,np.inf)
    L2ErrorNorm = (np.linalg.norm(mySurfaceElevationGradientNormalToEdgeError)
                   /np.sqrt(float(myMPAS_O.nEdges - myMPAS_O.nNonPeriodicBoundaryEdges)))
    print('The maximum error norm of the surface elevation gradient normal to edges is %.3g.' %MaxErrorNorm)
    print('The L2 error norm of the surface elevation gradient normal to edges is %.3g.' %L2ErrorNorm)
    if plotFigures:
        xLabel = 'Zonal Distance (km)'
        yLabel = 'Meridional Distance (km)'
        Title = 'Analytical Surface Elevation Gradient Normal to Edge'
        FigureTitle = prefix + 'SurfaceElevationGradientNormalToEdge_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],myMPAS_O.yEdge[iEdgeStartingIndex:],
        myAnalyticalSurfaceElevationGradientNormalToEdge[iEdgeStartingIndex:],300,False,[0.0,0.0],6,plt.cm.jet,
        13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')              
        Title = 'Numerical Surface Elevation Gradient Normal to Edge'
        FigureTitle = prefix + 'SurfaceElevationGradientNormalToEdge_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],myMPAS_O.yEdge[iEdgeStartingIndex:],
        myNumericalSurfaceElevationGradientNormalToEdge[iEdgeStartingIndex:],300,False,[0.0,0.0],6,plt.cm.jet,
        13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')
        Title = 'Error of the Surface Elevation Gradient Normal to Edge'
        FigureTitle = prefix + 'SurfaceElevationGradientNormalToEdge_Error'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(            
        output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],myMPAS_O.yEdge[iEdgeStartingIndex:],
        mySurfaceElevationGradientNormalToEdgeError[iEdgeStartingIndex:],300,False,[0.0,0.0],6,plt.cm.jet,
        13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')   
    return myMPAS_O.gridSpacingMagnitude, MaxErrorNorm, L2ErrorNorm


# In[28]:

do_test_numerical_gradient_operator_11 = False
if do_test_numerical_gradient_operator_11:
    periodicity = 'Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    dc, MaxErrorNorm, L2ErrorNorm = test_numerical_gradient_operator(True,True,' ',' ',' ',periodicity,
                                                                     output_directory)


# In[29]:

do_test_numerical_gradient_operator_12 = False
if do_test_numerical_gradient_operator_12:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_gradient_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     periodicity,output_directory))


# In[30]:

do_test_numerical_gradient_operator_13 = False
if do_test_numerical_gradient_operator_13:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_y'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_y'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_y'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_gradient_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     periodicity,output_directory))


# In[31]:

do_test_numerical_gradient_operator_14 = False
if do_test_numerical_gradient_operator_14:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_xy'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_xy'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_xy'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_gradient_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     periodicity,output_directory))


# In[32]:

def convergence_test_numerical_gradient_operator(periodicity,mesh_directory,output_directory):
    nCases = 5
    dc = np.zeros(nCases)
    MaxErrorNorm = np.zeros(nCases)
    L2ErrorNorm = np.zeros(nCases)
    prefix = problem_specific_prefix()
    for iCase in range(0,nCases):
        if periodicity == 'Periodic':
            base_mesh_file_name = 'base_mesh_%s.nc' %(iCase+1)
        elif periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy':
            base_mesh_file_name = 'culled_mesh_%s.nc' %(iCase+1)
        mesh_file_name = 'mesh_%s.nc' %(iCase+1)
        dc[iCase], MaxErrorNorm[iCase], L2ErrorNorm[iCase] = (
        test_numerical_gradient_operator(False,False,mesh_directory,base_mesh_file_name,mesh_file_name,periodicity,
                                         output_directory))
    A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm))[0]
    y = m*(np.log10(dc)) + c
    y = 10.0**y
    xLabel = 'Cell Width'
    yLabel = 'Maximum Error Norm of Numerical Gradient Operator'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalGradientOperator_MaxErrorNorm'
    CR.WriteCurve1D(output_directory,dc,MaxErrorNorm,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,MaxErrorNorm,y,[2.0,2.0],[' ','-'],['k','k'],
                                        [True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],[10.0,10.0],
                                        [15.0,15.0],legends,17.5,'upper left',Title,20.0,True,FigureTitle,False,
                                        drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm))[0]
    y = m*(np.log10(dc)) + c
    y = 10.0**y
    legends = ['L2 Error Norm','Best Fit Straight Line']
    xLabel = 'Cell Width'
    yLabel = 'L2 Error Norm of Numerical Gradient Operator'
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalGradientOperator_L2ErrorNorm'   
    CR.WriteCurve1D(output_directory,dc,L2ErrorNorm,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,L2ErrorNorm,y,[2.0,2.0],[' ','-'],['k','k'],
                                        [True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],[10.0,10.0],
                                        [15.0,15.0],legends,17.5,'upper left',Title,20.0,True,FigureTitle,False,
                                        drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True)


# In[33]:

do_convergence_test_numerical_gradient_operator_11 = False
if do_convergence_test_numerical_gradient_operator_11:
    periodicity = 'Periodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    convergence_test_numerical_gradient_operator(periodicity,mesh_directory,output_directory)


# In[34]:

do_convergence_test_numerical_gradient_operator_12 = False
if do_convergence_test_numerical_gradient_operator_12:
    periodicity = 'NonPeriodic_x'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    convergence_test_numerical_gradient_operator(periodicity,mesh_directory,output_directory)


# In[35]:

do_convergence_test_numerical_gradient_operator_13 = False
if do_convergence_test_numerical_gradient_operator_13:
    periodicity = 'NonPeriodic_y'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_y'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_y'
    convergence_test_numerical_gradient_operator(periodicity,mesh_directory,output_directory)


# In[36]:

do_convergence_test_numerical_gradient_operator_14 = False
if do_convergence_test_numerical_gradient_operator_14:
    periodicity = 'NonPeriodic_xy'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_xy'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_xy'
    convergence_test_numerical_gradient_operator(periodicity,mesh_directory,output_directory)


# In[37]:

def numerical_divergence_operator(myMPAS_O,myVectorQuantityNormalToEdge):
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    myVectorQuantityDivergence = np.zeros(myMPAS_O.nCells)
    for iCell in range(0,myMPAS_O.nCells):
        invAreaCell1 = 1.0/myMPAS_O.areaCell[iCell]
        for i in range(0,myMPAS_O.nEdgesOnCell[iCell]):
            iEdgeID = myMPAS_O.edgesOnCell[iCell,i]
            iEdge = iEdgeID - 1
            edgeSignOnCell_temp = myMPAS_O.edgeSignOnCell[iCell,i]
            dvEdge_temp = myMPAS_O.dvEdge[iEdge]
            r_tmp = dvEdge_temp*myVectorQuantityNormalToEdge[iEdge]*invAreaCell1
            myVectorQuantityDivergence[iCell] -= edgeSignOnCell_temp*r_tmp
    return myVectorQuantityDivergence


# In[38]:

def test_numerical_divergence_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                       mesh_file_name,periodicity,output_directory):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,periodicity=periodicity)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,
                                           periodicity=periodicity)
    prefix = problem_specific_prefix()
    myAnalyticalSurfaceElevationLaplacian = np.zeros(myMPAS_O.nCells)
    for iCell in range(0,myMPAS_O.nCells):
        myAnalyticalSurfaceElevationLaplacian[iCell] = (
        surface_elevation_laplacian(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell]))
    mySurfaceElevationGradientAtEdge = np.zeros((myMPAS_O.nEdges,2))
    for iEdge in range(0,myMPAS_O.nEdges):
        mySurfaceElevationGradientAtEdge[iEdge,0], mySurfaceElevationGradientAtEdge[iEdge,1] = (
        surface_elevation_gradient(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge]))
    myAnalyticalSurfaceElevationGradientNormalToEdge = (
    analytical_gradient_operator(myMPAS_O.nEdges,mySurfaceElevationGradientAtEdge,myMPAS_O.angleEdge))
    myNumericalSurfaceElevationLaplacian = (
    numerical_divergence_operator(myMPAS_O,myAnalyticalSurfaceElevationGradientNormalToEdge))
    mySurfaceElevationLaplacianError = myNumericalSurfaceElevationLaplacian - myAnalyticalSurfaceElevationLaplacian
    MaxErrorNorm = np.linalg.norm(mySurfaceElevationLaplacianError,np.inf)
    L2ErrorNorm = np.linalg.norm(mySurfaceElevationLaplacianError)/np.sqrt(float(myMPAS_O.nCells))    
    print('The maximum error norm of the SurfaceElevation laplacian is %.3g.' %MaxErrorNorm)
    print('The L2 error norm of the SurfaceElevation laplacian is %.3g.' %L2ErrorNorm)  
    if plotFigures:
        xLabel = 'Zonal Distance (km)'
        yLabel = 'Meridional Distance (km)'
        Title = 'Analytical Laplacian of the Surface Elevation'
        FigureTitle = prefix + 'SurfaceElevationLaplacian_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xCell,myMPAS_O.yCell,myAnalyticalSurfaceElevationLaplacian,300,False,[0.0,0.0],6,
        plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')          
        Title = 'Numerical Laplacian of the Surface Elevation'
        FigureTitle = prefix + 'SurfaceElevationLaplacian_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xCell,myMPAS_O.yCell,myNumericalSurfaceElevationLaplacian,300,False,[0.0,0.0],6,
        plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')
        Title = 'Error of the Surface Elevation Laplacian'
        FigureTitle = prefix + 'SurfaceElevationLaplacian_Error'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xCell,myMPAS_O.yCell,mySurfaceElevationLaplacianError,300,False,[0.0,0.0],6,
        plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')      
    return myMPAS_O.gridSpacingMagnitude, MaxErrorNorm, L2ErrorNorm


# In[39]:

do_test_numerical_divergence_operator_11 = False
if do_test_numerical_divergence_operator_11:
    periodicity = 'Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    dc, MaxErrorNorm, L2ErrorNorm = test_numerical_divergence_operator(True,True,' ',' ',' ',periodicity,
                                                                       output_directory)


# In[40]:

do_test_numerical_divergence_operator_12 = False
if do_test_numerical_divergence_operator_12:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_divergence_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                       mesh_file_name,periodicity,output_directory))


# In[41]:

do_test_numerical_divergence_operator_13 = False
if do_test_numerical_divergence_operator_13:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_y'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_y'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_y'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_divergence_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                       mesh_file_name,periodicity,output_directory))


# In[42]:

do_test_numerical_divergence_operator_14 = False
if do_test_numerical_divergence_operator_14:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_xy'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_xy'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_xy'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_divergence_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                       mesh_file_name,periodicity,output_directory))


# In[43]:

def convergence_test_numerical_divergence_operator(periodicity,mesh_directory,output_directory):
    nCases = 5
    dc = np.zeros(nCases)
    MaxErrorNorm = np.zeros(nCases)
    L2ErrorNorm = np.zeros(nCases)
    prefix = problem_specific_prefix()
    for iCase in range(0,nCases):
        if periodicity == 'Periodic':
            base_mesh_file_name = 'base_mesh_%s.nc' %(iCase+1)
        elif periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy':
            base_mesh_file_name = 'culled_mesh_%s.nc' %(iCase+1)
        mesh_file_name = 'mesh_%s.nc' %(iCase+1)
        dc[iCase], MaxErrorNorm[iCase], L2ErrorNorm[iCase] = (
        test_numerical_divergence_operator(False,False,mesh_directory,base_mesh_file_name,mesh_file_name,
                                           periodicity,output_directory))
    A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm))[0]
    y = m*(np.log10(dc)) + c
    y = 10.0**y
    xLabel = 'Cell Width'
    yLabel = 'Maximum Error Norm of Numerical Divergence Operator'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalDivergenceOperator_MaxErrorNorm'
    CR.WriteCurve1D(output_directory,dc,MaxErrorNorm,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,MaxErrorNorm,y,[2.0,2.0],[' ','-'],['k','k'],
                                        [True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],[10.0,10.0],
                                        [15.0,15.0],legends,17.5,'upper left',Title,20.0,True,FigureTitle,False,
                                        drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True)    
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm))[0]                     
    y = m*(np.log10(dc)) + c
    y = 10.0**y                                                               
    xLabel = 'Cell Width'
    yLabel = 'L2 Error Norm of Numerical Divergence Operator'
    legends = ['L2 Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalDivergenceOperator_L2ErrorNorm' 
    CR.WriteCurve1D(output_directory,dc,L2ErrorNorm,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,L2ErrorNorm,y,[2.0,2.0],[' ','-'],['k','k'],
                                        [True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],[10.0,10.0],
                                        [15.0,15.0],legends,17.5,'upper left',Title,20.0,True,FigureTitle,False,
                                        drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True)    


# In[44]:

do_convergence_test_numerical_divergence_operator_11 = False
if do_convergence_test_numerical_divergence_operator_11:
    periodicity = 'Periodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    convergence_test_numerical_divergence_operator(periodicity,mesh_directory,output_directory)


# In[45]:

do_convergence_test_numerical_divergence_operator_12 = False
if do_convergence_test_numerical_divergence_operator_12:
    periodicity = 'NonPeriodic_x'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    convergence_test_numerical_divergence_operator(periodicity,mesh_directory,output_directory)


# In[46]:

do_convergence_test_numerical_divergence_operator_13 = False
if do_convergence_test_numerical_divergence_operator_13:
    periodicity = 'NonPeriodic_y'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_y'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_y'
    convergence_test_numerical_divergence_operator(periodicity,mesh_directory,output_directory)


# In[47]:

do_convergence_test_numerical_divergence_operator_14 = False
if do_convergence_test_numerical_divergence_operator_14:
    periodicity = 'NonPeriodic_xy'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_xy'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_xy'
    convergence_test_numerical_divergence_operator(periodicity,mesh_directory,output_directory)


# In[48]:

def numerical_curl_operator(myMPAS_O,myVectorQuantityNormalToEdge,periodicity='Periodic'):
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    if periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy':
        MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    myVectorQuantityCurlAtVertex = np.zeros(myMPAS_O.nVertices)
    myVectorQuantityCurlAtCellCenter = np.zeros(myMPAS_O.nCells)
    for iVertex in range(0,myMPAS_O.nVertices):
        if ((periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy') 
            and myMPAS_O.boundaryVertex[iVertex] == 1.0):
            myVectorQuantityCurlAtVertex[iVertex] = (
            velocity_curl(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xVertex[iVertex],myMPAS_O.yVertex[iVertex]))
        else:
            invAreaTri1 = 1.0/myMPAS_O.areaTriangle[iVertex]
            for i in range(0,myMPAS_O.vertexDegree):
                iEdgeID = myMPAS_O.edgesOnVertex[iVertex,i]
                iEdge = iEdgeID - 1
                r_tmp = myMPAS_O.dcEdge[iEdge]*myVectorQuantityNormalToEdge[iEdge]
                myVectorQuantityCurlAtVertex[iVertex] += myMPAS_O.edgeSignOnVertex[iVertex,i]*r_tmp*invAreaTri1
    for iCell in range(0,myMPAS_O.nCells):
        invAreaCell1 = 1.0/myMPAS_O.areaCell[iCell]
        for i in range(0,myMPAS_O.nEdgesOnCell[iCell]):
            jID = myMPAS_O.kiteIndexOnCell[iCell,i]
            j = jID - 1
            iVertexID = myMPAS_O.verticesOnCell[iCell,i]
            iVertex = iVertexID - 1
            myVectorQuantityCurlAtCellCenter[iCell] += (
            myMPAS_O.kiteAreasOnVertex[iVertex,j]*myVectorQuantityCurlAtVertex[iVertex]*invAreaCell1)
    return myVectorQuantityCurlAtVertex, myVectorQuantityCurlAtCellCenter


# In[49]:

def test_numerical_curl_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                 periodicity,output_directory):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,periodicity=periodicity)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,
                                           periodicity=periodicity)
    prefix = problem_specific_prefix()
    myAnalyticalVelocityCurlAtVertex = np.zeros(myMPAS_O.nVertices)
    for iVertex in range(0,myMPAS_O.nVertices):
        myAnalyticalVelocityCurlAtVertex[iVertex] = (
        velocity_curl(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xVertex[iVertex],myMPAS_O.yVertex[iVertex]))
    myAnalyticalVelocityCurlAtCellCenter = np.zeros(myMPAS_O.nCells)
    for iCell in range(0,myMPAS_O.nCells):
        myAnalyticalVelocityCurlAtCellCenter[iCell] = (
        velocity_curl(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell]))  
    myAnalyticalVelocityComponentsAtEdge = np.zeros((myMPAS_O.nEdges,2))
    myAnalyticalVelocityNormalToEdge = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        myAnalyticalVelocityComponentsAtEdge[iEdge,0], myAnalyticalVelocityComponentsAtEdge[iEdge,1] = (
        velocity(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge]))
    myAnalyticalVelocityNormalToEdge = (
    ComputeNormalAndTangentialComponentsAtEdge(myAnalyticalVelocityComponentsAtEdge,myMPAS_O.angleEdge,'normal'))
    myNumericalVelocityCurlAtVertex, myNumericalVelocityCurlAtCellCenter = (
    numerical_curl_operator(myMPAS_O,myAnalyticalVelocityNormalToEdge,periodicity))
    myVelocityCurlAtVertexError = myNumericalVelocityCurlAtVertex - myAnalyticalVelocityCurlAtVertex
    MaxErrorNorm_Vertex = np.linalg.norm(myVelocityCurlAtVertexError,np.inf)
    L2ErrorNorm_Vertex = (np.linalg.norm(myVelocityCurlAtVertexError)
                          /np.sqrt(float(myMPAS_O.nVertices - myMPAS_O.nNonPeriodicBoundaryVertices)))
    print('The maximum error norm of the velocity curl at vertices is %.3g.' %MaxErrorNorm_Vertex)
    print('The L2 error norm of the velocity curl at vertices is %.3g.' %L2ErrorNorm_Vertex)   
    myVelocityCurlAtCellCenterError = myNumericalVelocityCurlAtCellCenter - myAnalyticalVelocityCurlAtCellCenter
    MaxErrorNorm_CellCenter = np.linalg.norm(myVelocityCurlAtCellCenterError,np.inf)
    L2ErrorNorm_CellCenter = np.linalg.norm(myVelocityCurlAtCellCenterError)/np.sqrt(float(myMPAS_O.nCells))
    print('The maximum error norm of the velocity curl at cell centers is %.3g.' %MaxErrorNorm_CellCenter)
    print('The L2 error norm of the velocity curl at cell centers is %.3g.' %L2ErrorNorm_CellCenter) 
    if plotFigures:
        xLabel = 'Zonal Distance (km)'
        yLabel = 'Meridional Distance (km)'
        Title = 'Analytical Velocity Curl At Vertex'
        FigureTitle = prefix + 'VelocityCurlAtVertex_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xVertex,myMPAS_O.yVertex,myAnalyticalVelocityCurlAtVertex,300,False,[0.0,0.0],6,
        plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g') 
        Title = 'Analytical Velocity Curl At Cell Center'
        FigureTitle = prefix + 'VelocityCurlAtCellCenter_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xCell,myMPAS_O.yCell,myAnalyticalVelocityCurlAtCellCenter,300,False,[0.0,0.0],6,
        plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')        
        Title = 'Numerical Velocity Curl At Vertex'
        FigureTitle = prefix + 'VelocityCurlAtVertex_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xVertex,myMPAS_O.yVertex,myNumericalVelocityCurlAtVertex,300,False,[0.0,0.0],6,
        plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g') 
        Title = 'Numerical Velocity Curl At Cell Center'
        FigureTitle = prefix + 'VelocityCurlAtCellCenter_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xCell,myMPAS_O.yCell,myNumericalVelocityCurlAtCellCenter,300,False,[0.0,0.0],6,
        plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')               
        Title = 'Velocity Curl Error at Vertices'
        FigureTitle = prefix + 'VelocityCurlAtVertex_Error'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xVertex,myMPAS_O.yVertex,myVelocityCurlAtVertexError,300,False,[0.0,0.0],6,
        plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')             
        Title = 'Velocity Curl Error at Cell Centers'
        FigureTitle = prefix + 'VelocityCurlAtCellCenter_Error'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xCell,myMPAS_O.yCell,myVelocityCurlAtCellCenterError,300,False,[0.0,0.0],6,
        plt.cm.jet,13.75,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,
        fig_size=[10.0,10.0],tick_units_in_km=True,cbarlabelformat='%.2g')          
    returningArguments = [myMPAS_O.gridSpacingMagnitude, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, 
                          MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter]
    return returningArguments


# In[50]:

do_test_numerical_curl_operator_11 = False
if do_test_numerical_curl_operator_11:
    periodicity='Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    [dc, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter] = (
    test_numerical_curl_operator(True,True,' ',' ',' ',periodicity,output_directory))


# In[51]:

do_test_numerical_curl_operator_12 = False
if do_test_numerical_curl_operator_12:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    [dc, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter] = (
    test_numerical_curl_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                 periodicity,output_directory))


# In[52]:

do_test_numerical_curl_operator_13 = False
if do_test_numerical_curl_operator_13:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_y'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_y'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_y'
    [dc, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter] = (
    test_numerical_curl_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                 periodicity,output_directory))


# In[53]:

do_test_numerical_curl_operator_14 = False
if do_test_numerical_curl_operator_14:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_xy'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_xy'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_xy'
    [dc, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter] = (
    test_numerical_curl_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                 periodicity,output_directory))


# In[54]:

def convergence_test_numerical_curl_operator(periodicity,mesh_directory,output_directory):
    nCases = 5
    dc = np.zeros(nCases)
    MaxErrorNorm_Vertex = np.zeros(nCases)
    L2ErrorNorm_Vertex = np.zeros(nCases)
    MaxErrorNorm_CellCenter = np.zeros(nCases)
    L2ErrorNorm_CellCenter = np.zeros(nCases)    
    prefix = problem_specific_prefix()
    for iCase in range(0,nCases):
        if periodicity == 'Periodic':
            base_mesh_file_name = 'base_mesh_%s.nc' %(iCase+1)
        elif periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy':
            base_mesh_file_name = 'culled_mesh_%s.nc' %(iCase+1)
        mesh_file_name = 'mesh_%s.nc' %(iCase+1)
        [dc[iCase], MaxErrorNorm_Vertex[iCase], L2ErrorNorm_Vertex[iCase], MaxErrorNorm_CellCenter[iCase], 
         L2ErrorNorm_CellCenter[iCase]] = test_numerical_curl_operator(False,False,mesh_directory,
                                                                       base_mesh_file_name,mesh_file_name,
                                                                       periodicity,output_directory)
    A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm_Vertex))[0]
    y = m*(np.log10(dc)) + c
    y = 10.0**y
    xLabel = 'Cell Width'
    yLabel = 'Maximum Error Norm of Numerical Curl Operator'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalCurlOperator_Vertex_MaxErrorNorm'
    CR.WriteCurve1D(output_directory,dc,MaxErrorNorm_Vertex,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,MaxErrorNorm_Vertex,y,[2.0,2.0],[' ','-'],
                                        ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],
                                        [10.0,10.0],[15.0,15.0],legends,17.5,'upper left',Title,20.0,True,
                                        FigureTitle,False,drawMajorGrid=True,drawMinorGrid=True,
                                        legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm_Vertex))[0]
    y = m*(np.log10(dc)) + c
    y = 10.0**y
    xLabel = 'Cell Width'
    yLabel = 'L2 Error Norm of Numerical Curl Operator'
    legends = ['L2 Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalCurlOperator_Vertex_L2ErrorNorm'    
    CR.WriteCurve1D(output_directory,dc,L2ErrorNorm_Vertex,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,L2ErrorNorm_Vertex,y,[2.0,2.0],[' ','-'],
                                        ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],
                                        [10.0,10.0],[15.0,15.0],legends,17.5,'upper left',Title,20.0,True,
                                        FigureTitle,False,drawMajorGrid=True,drawMinorGrid=True,
                                        legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm_CellCenter))[0]
    y = m*(np.log10(dc)) + c
    y = 10.0**y
    xLabel = 'Cell Width'
    yLabel = 'Maximum Error Norm of Numerical Curl Operator'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalCurlOperator_CellCenter_MaxErrorNorm'
    CR.WriteCurve1D(output_directory,dc,MaxErrorNorm_CellCenter,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,MaxErrorNorm_CellCenter,y,[2.0,2.0],
                                        [' ','-'],['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],
                                        [17.5,17.5],[10.0,10.0],[15.0,15.0],legends,17.5,'upper left',Title,20.0,
                                        True,FigureTitle,False,drawMajorGrid=True,drawMinorGrid=True,
                                        legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm_CellCenter))[0]
    y = m*(np.log10(dc)) + c
    y = 10.0**y
    xLabel = 'Cell Width'
    yLabel = 'L2 Error Norm of Numerical Curl Operator'
    legends = ['L2 Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalCurlOperator_CellCenter_L2ErrorNorm'
    CR.WriteCurve1D(output_directory,dc,L2ErrorNorm_CellCenter,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,L2ErrorNorm_CellCenter,y,[2.0,2.0],[' ','-'],
                                        ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],
                                        [10.0,10.0],[15.0,15.0],legends,17.5,'upper left',Title,20.0,True,
                                        FigureTitle,False,drawMajorGrid=True,drawMinorGrid=True,
                                        legendWithinBox=True)


# In[55]:

do_convergence_test_numerical_curl_operator_11 = False
if do_convergence_test_numerical_curl_operator_11:
    periodicity = 'Periodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    convergence_test_numerical_curl_operator(periodicity,mesh_directory,output_directory)


# In[56]:

do_convergence_test_numerical_curl_operator_12 = False
if do_convergence_test_numerical_curl_operator_12:
    periodicity = 'NonPeriodic_x'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    convergence_test_numerical_curl_operator(periodicity,mesh_directory,output_directory)


# In[57]:

do_convergence_test_numerical_curl_operator_13 = False
if do_convergence_test_numerical_curl_operator_13:
    periodicity = 'NonPeriodic_y'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_y'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_y'
    convergence_test_numerical_curl_operator(periodicity,mesh_directory,output_directory)


# In[58]:

do_convergence_test_numerical_curl_operator_14 = False
if do_convergence_test_numerical_curl_operator_14:
    periodicity = 'NonPeriodic_xy'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_xy'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_xy'
    convergence_test_numerical_curl_operator(periodicity,mesh_directory,output_directory)


# In[59]:

def numerical_tangential_velocity(myMPAS_O,myNormalVelocity,periodicity='Periodic'):
    if periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy':
        MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    myTangentialVelocity = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        if ((periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy') 
            and myMPAS_O.boundaryEdge[iEdge] == 1.0):
            u, v = velocity(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge])
            myTangentialVelocity[iEdge] = v*np.cos(myMPAS_O.angleEdge[iEdge]) - u*np.sin(myMPAS_O.angleEdge[iEdge])
        else:
            myTangentialVelocity[iEdge] = 0.0
            # Compute tangential velocities
            for i in range(0,myMPAS_O.nEdgesOnEdge[iEdge]):
                eoeID = myMPAS_O.edgesOnEdge[iEdge,i]
                eoe = eoeID - 1
                weightsOnEdge_temp = myMPAS_O.weightsOnEdge[iEdge,i]
                myTangentialVelocity[iEdge] += weightsOnEdge_temp*myNormalVelocity[eoe]
    return myTangentialVelocity


# In[60]:

def test_tangential_velocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                             periodicity,output_directory):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,periodicity=periodicity)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,
                                           periodicity=periodicity)
    if periodicity == 'NonPeriodic_x':
        iEdgeStartingIndex = 1
    else:
        iEdgeStartingIndex = 0
    prefix = problem_specific_prefix()
    myAnalyticalVelocityComponentsAtEdge = np.zeros((myMPAS_O.nEdges,2))
    myAnalyticalNormalVelocity = np.zeros(myMPAS_O.nEdges)
    myAnalyticalTangentialVelocity = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        myAnalyticalVelocityComponentsAtEdge[iEdge,0], myAnalyticalVelocityComponentsAtEdge[iEdge,1] = (
        velocity(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge]))
    myAnalyticalNormalVelocity, myAnalyticalTangentialVelocity = (
    ComputeNormalAndTangentialComponentsAtEdge(myAnalyticalVelocityComponentsAtEdge,myMPAS_O.angleEdge,'both'))
    myNumericalTangentialVelocity = (
    numerical_tangential_velocity(myMPAS_O,myAnalyticalNormalVelocity,periodicity))
    myTangentialVelocityError = myNumericalTangentialVelocity - myAnalyticalTangentialVelocity
    MaxErrorNorm = np.linalg.norm(myTangentialVelocityError,np.inf)
    L2ErrorNorm = (np.linalg.norm(myTangentialVelocityError)
                   /np.sqrt(float(myMPAS_O.nEdges - myMPAS_O.nNonPeriodicBoundaryEdges)))
    print('The maximum error norm of the tangential velocity is %.3g.' %MaxErrorNorm)
    print('The L2 error norm of the tangential velocity is %.3g.' %L2ErrorNorm)    
    if plotFigures:
        xLabel = 'Zonal Distance (km)'
        yLabel = 'Meridional Distance (km)'
        Title = 'Analytical Tangential Velocity'
        FigureTitle = prefix + 'TangentialVelocity_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],myMPAS_O.yEdge[iEdgeStartingIndex:],
        myAnalyticalTangentialVelocity[iEdgeStartingIndex:],300,False,[0.0,0.0],6,plt.cm.jet,13.75,[xLabel,yLabel],
        [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,fig_size=[10.0,10.0],
        tick_units_in_km=True,cbarlabelformat='%.2f') 
        Title = 'Numerical Tangential Velocity'
        FigureTitle = prefix + 'TangentialVelocity_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],myMPAS_O.yEdge[iEdgeStartingIndex:],
        myNumericalTangentialVelocity[iEdgeStartingIndex:],300,False,[0.0,0.0],6,plt.cm.jet,13.75,[xLabel,yLabel],
        [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,fig_size=[10.0,10.0],
        tick_units_in_km=True,cbarlabelformat='%.2f')         
        Title = 'Tangential Velocity Error'
        FigureTitle = prefix + 'TangentialVelocity_Error'        
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(
        output_directory,myMPAS_O.xEdge[iEdgeStartingIndex:],myMPAS_O.yEdge[iEdgeStartingIndex:],
        myTangentialVelocityError[iEdgeStartingIndex:],300,False,[0.0,0.0],6,plt.cm.jet,13.75,[xLabel,yLabel],
        [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FigureTitle,False,fig_size=[10.0,10.0],
        tick_units_in_km=True,cbarlabelformat='%.2g') 
    return myMPAS_O.gridSpacingMagnitude, MaxErrorNorm, L2ErrorNorm


# In[61]:

do_test_tangential_velocity_11 = False
if do_test_tangential_velocity_11:
    periodicity = 'Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    dc, MaxErrorNorm, L2ErrorNorm = test_tangential_velocity(True,True,' ',' ',' ',periodicity,output_directory)


# In[62]:

do_test_tangential_velocity_12 = False
if do_test_tangential_velocity_12:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_tangential_velocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                             periodicity,output_directory))


# In[63]:

do_test_tangential_velocity_13 = False
if do_test_tangential_velocity_13:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_y'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_y'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_y'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_tangential_velocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                             periodicity,output_directory))


# In[64]:

do_test_tangential_velocity_14 = False
if do_test_tangential_velocity_14:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_xy'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    periodicity = 'NonPeriodic_xy'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_xy'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_tangential_velocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                             periodicity,output_directory))


# In[65]:

def convergence_test_tangential_velocity(periodicity,mesh_directory,output_directory):
    nCases = 5
    dc = np.zeros(nCases)
    MaxErrorNorm = np.zeros(nCases)
    L2ErrorNorm = np.zeros(nCases)
    prefix = problem_specific_prefix()
    for iCase in range(0,nCases):
        if periodicity == 'Periodic':
            base_mesh_file_name = 'base_mesh_%s.nc' %(iCase+1)
        elif periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy':
            base_mesh_file_name = 'culled_mesh_%s.nc' %(iCase+1)
        mesh_file_name = 'mesh_%s.nc' %(iCase+1)
        dc[iCase], MaxErrorNorm[iCase], L2ErrorNorm[iCase] = (
        test_tangential_velocity(False,False,mesh_directory,base_mesh_file_name,mesh_file_name,periodicity,
                                 output_directory))
    A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm))[0]
    y = m*(np.log10(dc)) + c
    y = 10.0**y
    xLabel = 'Cell Width'
    yLabel = 'Maximum Error Norm of Numerical Tangential Velocity'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalTangentialVelocity_MaxErrorNorm'
    CR.WriteCurve1D(output_directory,dc,MaxErrorNorm,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,MaxErrorNorm,y,[2.0,2.0],[' ','-'],['k','k'],
                                        [True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],[10.0,10.0],
                                        [15.0,15.0],legends,17.5,'upper left',Title,20.0,True,FigureTitle,False,
                                        drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm))[0]
    y = m*(np.log10(dc)) + c
    y = 10.0**y
    xLabel = 'Cell Width'
    yLabel = 'L2 Error Norm of Numerical Tangential Velocity'
    legends = ['L2 Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'ConvergencePlot_NumericalTangentialVelocity_L2ErrorNorm' 
    CR.WriteCurve1D(output_directory,dc,L2ErrorNorm,FigureTitle+'_'+periodicity)
    CR.WriteCurve1D(output_directory,dc,y,FigureTitle+'_BestFitStraightLine_'+periodicity)
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,L2ErrorNorm,y,[2.0,2.0],[' ','-'],['k','k'],
                                        [True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],[10.0,10.0],
                                        [15.0,15.0],legends,17.5,'upper left',Title,20.0,True,FigureTitle,False,
                                        drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True)


# In[66]:

do_convergence_test_tangential_velocity_11 = False
if do_convergence_test_tangential_velocity_11:
    periodicity = 'Periodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    convergence_test_tangential_velocity(periodicity,mesh_directory,output_directory)


# In[67]:

do_convergence_test_tangential_velocity_12 = False
if do_convergence_test_tangential_velocity_12:
    periodicity = 'NonPeriodic_x'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    convergence_test_tangential_velocity(periodicity,mesh_directory,output_directory)


# In[68]:

do_convergence_test_tangential_velocity_13 = False
if do_convergence_test_tangential_velocity_13:
    periodicity = 'NonPeriodic_y'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_y'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_y'
    convergence_test_tangential_velocity(periodicity,mesh_directory,output_directory)


# In[69]:

do_convergence_test_tangential_velocity_14 = False
if do_convergence_test_tangential_velocity_14:
    periodicity = 'NonPeriodic_xy'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_xy'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_xy'
    convergence_test_tangential_velocity(periodicity,mesh_directory,output_directory)


# In[70]:

def MPAS_O_Operator_Testing_Convergence_Plots(output_directory,x,y,linewidths,linestyles,colors,markers,
                                              markersizes,labels,labelfontsizes,labelpads,tickfontsizes,legends,
                                              legendfontsize,legendposition,title,titlefontsize,SaveAsPNG,
                                              FigureTitle,Show,fig_size=[9.25,9.25],
                                              useDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=False,
                                              drawMinorGrid=False,legendWithinBox=False,legendpads=[1.0,0.5],
                                              titlepad=1.035):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)   
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    nPoints = len(x)
    # Note that the shape of y is [nPlots,nSubplots,nPoints]
    nPlots = y.shape[0]
    nSubplots = 2
    for iPlot in range(0,nPlots):
        for iSubplot in range(0,nSubplots):
            if iSubplot == 0:
                mylinestyle = ""
                mylabel = legends[iPlot]
                mymarker = markers[iPlot]
            else:
                mylinestyle = linestyles[iPlot]
                mylabel = ""
                mymarker = ""
            ax.loglog(x,y[iPlot,iSubplot,:],linewidth=linewidths[iPlot],linestyle=mylinestyle,
                      color=colors[iPlot],marker=mymarker,markersize=markersizes[iPlot],label=mylabel)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    if useDefaultMethodToSpecifyTickFontSize:
        plt.xticks(fontsize=tickfontsizes[0])
        plt.yticks(fontsize=tickfontsizes[1])
    else:
        ax.tick_params(axis='x',labelsize=tickfontsizes[0])
        ax.tick_params(axis='y',labelsize=tickfontsizes[1])
    if legendWithinBox:
        ax.legend(fontsize=legendfontsize,loc=legendposition,shadow=True) 
    else:
        ax.legend(fontsize=legendfontsize,loc=legendposition,bbox_to_anchor=(legendpads[0],legendpads[1]),
                  shadow=True) 
    ax.set_title(title,fontsize=titlefontsize,y=titlepad)
    if drawMajorGrid and not(drawMinorGrid):
        plt.grid(which='major')
    elif not(drawMajorGrid) and drawMinorGrid:
        plt.grid(which='minor')       
    elif drawMajorGrid and drawMinorGrid:
        plt.grid(which='both') 
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[71]:

def Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle):
    # Before running this routine, please move the output files containing the convergence data to
    # <pwd>/MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/ConvergencePlots/<Expt#>/
    # where <pwd> is the path to working directory, and <Expt#> can be Expt1 or Expt2.
    periodicities = ['Periodic','NonPeriodic_x','NonPeriodic_y','NonPeriodic_xy']
    nPlots = len(periodicities)
    nSubplots = 2
    nCases = 5
    yAll = np.zeros((nPlots,nSubplots,nCases))
    mAll = np.zeros(nPlots)
    for iPlot in range(0,nPlots):
        output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/ConvergencePlots/Expt'
                            + str(ExperimentNumber))
        filename = ('Expt' + str(ExperimentNumber) + '_ConvergencePlot_' + NumericalOperator + '_L2ErrorNorm_' 
                    + periodicities[iPlot] + '.curve')
        dc, yAll[iPlot,0,:] = CR.ReadCurve1D(output_directory,filename)
        filename = ('Expt' + str(ExperimentNumber) + '_ConvergencePlot_' + NumericalOperator 
                    + '_L2ErrorNorm_BestFitStraightLine_' + periodicities[iPlot] + '.curve')
        dc, yAll[iPlot,1,:] = CR.ReadCurve1D(output_directory,filename)
        mAll[iPlot] = ((np.log10(yAll[iPlot,1,1]) - np.log10(yAll[iPlot,1,0]))
                       /(np.log10(dc[1]) - np.log10(dc[0])))
    linewidths = 2.0*np.ones(nPlots)
    linestyles  = ['-','-','-','-']
    colors = ['darkviolet','lawngreen','gold','red']
    markers = ['o','v','^','P']
    markersizes = 12.5*np.ones(nPlots)
    labels = ['Cell Width','L2 Error Norm of '+NumericalOperatorTitle]
    labelfontsizes = [17.5,17.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    legend1 = 'Periodic in x and y: Slope of Best Fit Straight Line is %.2f' %mAll[0]
    legend2 = 'Non-Periodic in x and Periodic in y: Slope of Best Fit Straight Line is %.2f' %mAll[1]
    legend3 = 'Periodic in x and Non-Periodic in y: Slope of Best Fit Straight Line is %.2f' %mAll[2]
    legend4 = 'Non-Periodic in x and y: Slope of Best Fit Straight Line is %.2f' %mAll[3]
    legends = [legend1,legend2,legend3,legend4]
    legendfontsize = 15.0
    legendposition = 'lower center'
    title = 'Convergence Plot w.r.t. L2 Error Norm of ' + NumericalOperatorTitle
    titlefontsize = 20.0
    SaveAsPNG = True
    FigureTitle = 'Expt' + str(ExperimentNumber) + '_ConvergencePlots_' + NumericalOperator + '_L2ErrorNorm'
    Show = False
    MPAS_O_Operator_Testing_Convergence_Plots(output_directory,dc,yAll,linewidths,linestyles,colors,markers,
                                              markersizes,labels,labelfontsizes,labelpads,tickfontsizes,legends,
                                              legendfontsize,legendposition,title,titlefontsize,SaveAsPNG,
                                              FigureTitle,Show,fig_size=[9.25,9.25],
                                              useDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=True,
                                              drawMinorGrid=True,legendWithinBox=False,legendpads=[0.5,-0.37],
                                              titlepad=1.035)


# In[72]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Gradient_Operator_Expt1 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Gradient_Operator_Expt1:
    ExperimentNumber = 1
    NumericalOperator = 'NumericalGradientOperator'
    NumericalOperatorTitle = 'Numerical Gradient Operator'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)


# In[73]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Gradient_Operator_Expt2 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Gradient_Operator_Expt2:
    ExperimentNumber = 2
    NumericalOperator = 'NumericalGradientOperator'
    NumericalOperatorTitle = 'Numerical Gradient Operator'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)


# In[74]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Divergence_Operator_Expt1 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Divergence_Operator_Expt1:
    ExperimentNumber = 1
    NumericalOperator = 'NumericalDivergenceOperator'
    NumericalOperatorTitle = 'Numerical Divergence Operator'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)


# In[75]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Divergence_Operator_Expt2 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Divergence_Operator_Expt2:
    ExperimentNumber = 2
    NumericalOperator = 'NumericalDivergenceOperator'
    NumericalOperatorTitle = 'Numerical Divergence Operator'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)


# In[76]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Curl_Operator_Vertex_Expt1 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Curl_Operator_Vertex_Expt1:
    ExperimentNumber = 1
    NumericalOperator = 'NumericalCurlOperator_Vertex'
    NumericalOperatorTitle = 'Numerical Curl Operator'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)


# In[77]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Curl_Operator_Vertex_Expt2 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Curl_Operator_Vertex_Expt2:
    ExperimentNumber = 2
    NumericalOperator = 'NumericalCurlOperator_Vertex'
    NumericalOperatorTitle = 'Numerical Curl Operator'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)


# In[78]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Curl_Operator_CellCenter_Expt1 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Curl_Operator_CellCenter_Expt1:
    ExperimentNumber = 1
    NumericalOperator = 'NumericalCurlOperator_CellCenter'
    NumericalOperatorTitle = 'Numerical Curl Operator'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)


# In[79]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Curl_Operator_CellCenter_Expt2 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Curl_Operator_CellCenter_Expt2:
    ExperimentNumber = 2
    NumericalOperator = 'NumericalCurlOperator_CellCenter'
    NumericalOperatorTitle = 'Numerical Curl Operator'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)


# In[80]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Tangential_Velocity_Expt1 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Tangential_Velocity_Expt1:
    ExperimentNumber = 1
    NumericalOperator = 'NumericalTangentialVelocity'
    NumericalOperatorTitle = 'Numerical Tangential Velocity'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)


# In[81]:

do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Tangential_Velocity_Expt2 = False
if do_Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots_Numerical_Tangential_Velocity_Expt2:
    ExperimentNumber = 2
    NumericalOperator = 'NumericalTangentialVelocity'
    NumericalOperatorTitle = 'Numerical Tangential Velocity'
    Read_Files_and_Make_MPAS_O_Operator_Testing_Convergence_Plots(ExperimentNumber,NumericalOperator,
                                                                  NumericalOperatorTitle)