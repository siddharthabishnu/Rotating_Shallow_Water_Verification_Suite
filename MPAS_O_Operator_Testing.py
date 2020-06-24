
# coding: utf-8

# Name: MPAS_O_Operator_Testing.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains routines for determining and testing the various spatial operators on a MPAS-Ocean mesh e.g. gradient, divergence, curl etc.

# In[1]:

import numpy as np
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
    xCenter = 0.5*lX
    yCenter = 0.5*lY
    RangeX = 0.2*lX
    RangeY = 0.2*lY
    eta0 = 0.1
    eta = eta0*np.exp(-(((x - xCenter)/RangeX)**2.0 + ((y - yCenter)/RangeY)**2.0))
    return eta


# In[3]:

def surface_elevation_gradient_1(lX,lY,x,y):
    eta = surface_elevation(lX,lY,x,y)
    xCenter = 0.5*lX
    yCenter = 0.5*lY
    RangeX = 0.2*lX
    RangeY = 0.2*lY
    eta_x = -2.0*(x - xCenter)*eta/(RangeX**2.0)
    eta_y = -2.0*(y - yCenter)*eta/(RangeY**2.0)
    return eta_x, eta_y


# In[4]:

def surface_elevation_laplacian_1(lX,lY,x,y):
    eta = surface_elevation(lX,lY,x,y)
    xCenter = 0.5*lX
    yCenter = 0.5*lY
    RangeX = 0.2*lX
    RangeY = 0.2*lY
    eta_x = -2.0*(x - xCenter)*eta/(RangeX**2.0)
    eta_y = -2.0*(y - yCenter)*eta/(RangeY**2.0)
    eta_xx = -2.0/(RangeX**2.0)*((x - xCenter)*eta_x + eta)
    eta_yy = -2.0/(RangeY**2.0)*((y - yCenter)*eta_y + eta)
    laplacian = eta_xx + eta_yy
    return laplacian


# In[5]:

def problem_specific_prefix_1():
    prefix = 'Expt1_'
    return prefix


# In[6]:

def surface_elevation_2(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    return eta


# In[7]:

def surface_elevation_gradient_2(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_x = eta0*(2.0*np.pi/lX)*np.cos(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_y = eta0*(2.0*np.pi/lY)*np.sin(2.0*np.pi*x/lX)*np.cos(2.0*np.pi*y/lY)
    return eta_x, eta_y


# In[8]:

def surface_elevation_laplacian_2(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_xx = -(2.0*np.pi/lX)**2.0*eta
    eta_yy = -(2.0*np.pi/lY)**2.0*eta
    laplacian = eta_xx + eta_yy
    return laplacian


# In[9]:

def problem_specific_prefix_2():
    prefix = 'Expt2_'
    return prefix


# In[10]:

surface_elevation = surface_elevation_2


# In[11]:

surface_elevation_gradient = surface_elevation_gradient_2


# In[12]:

surface_elevation_laplacian = surface_elevation_laplacian_2


# In[13]:

problem_specific_prefix = problem_specific_prefix_2


# In[14]:

def velocity(lX,lY,x,y):
    eta_x, eta_y = surface_elevation_gradient(lX,lY,x,y) 
    f = 10.0**(-4.0)
    g = 10.0
    v = g*eta_x/f
    u = -g*eta_y/f
    return u, v


# In[15]:

def velocity_curl(lX,lY,x,y):
    f = 10.0**(-4.0)
    g = 10.0
    zeta = g/f*surface_elevation_laplacian(lX,lY,x,y)
    return zeta


# In[16]:

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


# In[17]:

def analytical_gradient_operator(nEdges,myScalarQuantityGradientComponentsAtEdge,angleEdge):
    myScalarQuantityGradientNormalToEdge = (
    ComputeNormalAndTangentialComponentsAtEdge(myScalarQuantityGradientComponentsAtEdge,angleEdge,'normal'))
    return myScalarQuantityGradientNormalToEdge


# In[18]:

def numerical_gradient_operator(myMPAS_O,myScalarQuantity):
    myScalarQuantityGradientNormalToEdge = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        cellID1 = myMPAS_O.cellsOnEdge[iEdge,0]
        cell1 = cellID1 - 1
        cellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
        cell2 = cellID2 - 1
        invLength = 1.0/myMPAS_O.dcEdge[iEdge]    
        myScalarQuantityGradientNormalToEdge[iEdge] = (myScalarQuantity[cell2] - myScalarQuantity[cell1])*invLength
    return myScalarQuantityGradientNormalToEdge


# In[19]:

def plot_ssh_velocity_normalVelocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     output_directory):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name)
    prefix = problem_specific_prefix()
    mySSH = np.zeros(myMPAS_O.nCells)
    for iCell in range(0,myMPAS_O.nCells):
        mySSH[iCell] = surface_elevation(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell])
    myVelocity = np.zeros(myMPAS_O.nEdges)   
    myNormalVelocity = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        u, v = velocity(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge])
        myVelocity[iEdge] = np.sqrt(u**2.0 + v**2.0)
        myNormalVelocity[iEdge] = u*np.cos(myMPAS_O.angleEdge[iEdge]) + v*np.sin(myMPAS_O.angleEdge[iEdge])
    if plotFigures:
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,mySSH,300,
                                                          False,[0.0,0.0],'x',10,'y',10,'SSH',True,prefix+'SSH',
                                                          False)        
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                          myVelocity,300,False,[0.0,0.0],'x',10,'y',10,'velocity',
                                                          True,prefix+'velocity',False)                
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                          myNormalVelocity,300,False,[0.0,0.0],'x',10,'y',10,
                                                          'normalVelocity',True,prefix+'normalVelocity',False)


# In[20]:

do_plot_ssh_velocity_normalVelocity_1 = False
if do_plot_ssh_velocity_normalVelocity_1:
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    plot_ssh_velocity_normalVelocity(True,True,' ',' ',' ',output_directory)


# In[21]:

do_plot_ssh_velocity_normalVelocity_2 = False
if do_plot_ssh_velocity_normalVelocity_2:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    plot_ssh_velocity_normalVelocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     output_directory)


# In[22]:

def test_numerical_gradient_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     output_directory):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name)
    prefix = problem_specific_prefix()
    ssh = np.zeros(myMPAS_O.nCells)
    for iCell in range(0,myMPAS_O.nCells):
        ssh[iCell] = surface_elevation(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xCell[iCell],myMPAS_O.yCell[iCell])
    sshGradientEdge = np.zeros((myMPAS_O.nEdges,2))
    for iEdge in range(0,myMPAS_O.nEdges):
        sshGradientEdge[iEdge,0], sshGradientEdge[iEdge,1] = (
        surface_elevation_gradient(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge]))
    analyticalSSHGradientNormalToEdge = analytical_gradient_operator(myMPAS_O.nEdges,sshGradientEdge,
                                                                     myMPAS_O.angleEdge)
    # Reduce the normal gradient at the midpoint of the non-periodic boundary edges to be zero. Due to the 
    # staggered nature of the MPAS-O grid, we do not need to compute the gradient of any variable at the midpoint 
    # of a non-periodic boundary edge anyway.
    for iEdge in range(0,myMPAS_O.nEdges):
        cellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
        if cellID2 == 0:
            analyticalSSHGradientNormalToEdge[iEdge] = 0.0
    if plotFigures:
        Title = 'Analytical SSH Gradient normal to Edge'
        FigureTitle = prefix + 'SSHGradientNormalToEdge_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                          analyticalSSHGradientNormalToEdge,300,False,[0.0,0.0],
                                                          'x',10,'y',10,Title,True,FigureTitle,False)
    numericalSSHGradientNormalToEdge = numerical_gradient_operator(myMPAS_O,ssh)
    # Reduce the normal gradient at the midpoint of the non-periodic boundary edges to be zero. Due to the 
    # staggered nature of the MPAS-O grid, we do not need to compute the gradient of any variable at the midpoint 
    # of a non-periodic boundary edge anyway.
    for iEdge in range(0,myMPAS_O.nEdges):
        cellID2 = myMPAS_O.cellsOnEdge[iEdge,1]
        if cellID2 == 0:
            numericalSSHGradientNormalToEdge[iEdge] = 0.0
    if plotFigures:
        Title = 'Numerical SSH Gradient normal to Edge'
        FigureTitle = prefix + 'SSHGradientNormalToEdge_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                          numericalSSHGradientNormalToEdge,300,False,[0.0,0.0],'x',
                                                          10,'y',10,Title,True,FigureTitle,False)  
    normalSSHGradientError = numericalSSHGradientNormalToEdge - analyticalSSHGradientNormalToEdge
    MaxErrorNorm = np.linalg.norm(normalSSHGradientError,np.inf)
    L2ErrorNorm = np.linalg.norm(normalSSHGradientError)/np.sqrt(myMPAS_O.nEdges)
    print('The maximum error norm of the normal ssh gradient is %.3g.' %MaxErrorNorm)
    print('The L2 error norm of the normal ssh gradient is %.3g.' %L2ErrorNorm)
    if plotFigures:
        Title = 'normal SSH Gradient Error'
        FigureTitle = prefix + 'SSHGradientNormalToEdge_Error'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                          normalSSHGradientError,300,False,[0.0,0.0],'x',10,'y',10,
                                                          Title,True,FigureTitle,False)    
    return myMPAS_O.gridSpacingMagnitude, MaxErrorNorm, L2ErrorNorm


# In[23]:

do_test_numerical_gradient_operator_1 = False
if do_test_numerical_gradient_operator_1:
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    dc, MaxErrorNorm, L2ErrorNorm = test_numerical_gradient_operator(True,True,' ',' ',' ',output_directory)


# In[24]:

do_test_numerical_gradient_operator_2 = False
if do_test_numerical_gradient_operator_2:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_gradient_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     output_directory))


# In[25]:

do_test_numerical_gradient_operator_3 = False
if do_test_numerical_gradient_operator_3:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_gradient_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     output_directory))


# In[26]:

do_test_numerical_gradient_operator_4 = False
if do_test_numerical_gradient_operator_4:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_gradient_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                     output_directory))


# In[27]:

def convergence_test_numerical_gradient_operator(problem_type,mesh_directory,output_directory):
    nCases = 5
    dc = np.zeros(nCases)
    MaxErrorNorm = np.zeros(nCases)
    L2ErrorNorm = np.zeros(nCases)
    prefix = problem_specific_prefix()
    for iCase in range(0,nCases):
        if problem_type == 'Periodic':
            base_mesh_file_name = 'base_mesh_%s.nc' %(iCase+1)
        elif problem_type == 'NonPeriodic':
            base_mesh_file_name = 'culled_mesh_%s.nc' %(iCase+1)
        mesh_file_name = 'mesh_%s.nc' %(iCase+1)
        dc[iCase], MaxErrorNorm[iCase], L2ErrorNorm[iCase] = (
        test_numerical_gradient_operator(False,False,mesh_directory,base_mesh_file_name,mesh_file_name,
                                         output_directory))
    A = np.vstack([np.log10(1.0/dc),np.ones(len(dc))]).T
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm))[0]
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'Maximum Error Norm of Numerical Gradient Operator'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalGradientOperatorConvergencePlot_MaxErrorNorm'
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,MaxErrorNorm,y,[2.0,2.0],[' ','-'],
                                        ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],
                                        [10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,True,
                                        FigureTitle,False,drawGrid=True,legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm))[0]
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y
    legends = ['L2 Error Norm','Best Fit Straight Line']
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'L2 Error Norm of Numerical Gradient Operator'
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalGradientOperatorConvergencePlot_L2ErrorNorm'   
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,L2ErrorNorm,y,[2.0,2.0],[' ','-'],
                                        ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],
                                        [10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,True,
                                        FigureTitle,False,drawGrid=True,legendWithinBox=True)


# In[28]:

do_convergence_test_numerical_gradient_operator_1 = False
if do_convergence_test_numerical_gradient_operator_1:
    problem_type = 'Periodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    convergence_test_numerical_gradient_operator(problem_type,mesh_directory,output_directory)


# In[29]:

do_convergence_test_numerical_gradient_operator_2 = False
if do_convergence_test_numerical_gradient_operator_2:
    problem_type = 'NonPeriodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    convergence_test_numerical_gradient_operator(problem_type,mesh_directory,output_directory)


# In[30]:

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


# In[31]:

def test_numerical_divergence_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                       mesh_file_name,output_directory):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name)
    prefix = problem_specific_prefix()
    analyticalSSHLaplacian = np.zeros(myMPAS_O.nCells)
    for iCell in range(0,myMPAS_O.nCells):
        analyticalSSHLaplacian[iCell] = surface_elevation_laplacian(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xCell[iCell],
                                                                    myMPAS_O.yCell[iCell])
    if plotFigures:
        Title = 'Analytical SSH Laplacian'
        FigureTitle = prefix + 'SSHLaplacian_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          analyticalSSHLaplacian,300,False,[0.0,0.0],'x',10,'y',10,
                                                          Title,True,FigureTitle,False)        
    sshGradientEdge = np.zeros((myMPAS_O.nEdges,2))
    for iEdge in range(0,myMPAS_O.nEdges):
        sshGradientEdge[iEdge,0], sshGradientEdge[iEdge,1] = (
        surface_elevation_gradient(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge]))
    analyticalSSHGradientNormalToEdge = analytical_gradient_operator(myMPAS_O.nEdges,sshGradientEdge,
                                                                     myMPAS_O.angleEdge)
    numericalSSHLaplacian = numerical_divergence_operator(myMPAS_O,analyticalSSHGradientNormalToEdge)
    if plotFigures:
        Title = 'Numerical SSH Laplacian'
        FigureTitle = prefix + 'SSHLaplacian_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          numericalSSHLaplacian,300,False,[0.0,0.0],'x',10,'y',10,
                                                          Title,True,FigureTitle,False)  
    SSHLaplacianError = numericalSSHLaplacian - analyticalSSHLaplacian
    if plotFigures:
        Title = 'SSH Laplacian Error'
        FigureTitle = prefix + 'SSHLaplacian_Error'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          SSHLaplacianError,300,False,[0.0,0.0],'x',10,'y',10,
                                                          Title,True,FigureTitle,False) 
    MaxErrorNorm = np.linalg.norm(SSHLaplacianError,np.inf)
    L2ErrorNorm = np.linalg.norm(SSHLaplacianError)/np.sqrt(myMPAS_O.nCells)    
    print('The maximum error norm of the ssh laplacian is %.3g.' %MaxErrorNorm)
    print('The L2 error norm of the ssh laplacian is %.3g.' %L2ErrorNorm)  
    return myMPAS_O.gridSpacingMagnitude, MaxErrorNorm, L2ErrorNorm


# In[32]:

do_test_numerical_divergence_operator_1 = False
if do_test_numerical_divergence_operator_1:
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    dc, MaxErrorNorm, L2ErrorNorm = test_numerical_divergence_operator(True,True,' ',' ',' ',output_directory)


# In[33]:

do_test_numerical_divergence_operator_2 = False
if do_test_numerical_divergence_operator_2:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_divergence_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                       mesh_file_name,output_directory))


# In[34]:

do_test_numerical_divergence_operator_3 = False
if do_test_numerical_divergence_operator_3:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_divergence_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                       mesh_file_name,output_directory))


# In[35]:

do_test_numerical_divergence_operator_4 = False
if do_test_numerical_divergence_operator_4:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_numerical_divergence_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,
                                       mesh_file_name,output_directory))


# In[36]:

def convergence_test_numerical_divergence_operator(problem_type,mesh_directory,output_directory):
    nCases = 5
    dc = np.zeros(nCases)
    MaxErrorNorm = np.zeros(nCases)
    L2ErrorNorm = np.zeros(nCases)
    prefix = problem_specific_prefix()
    for iCase in range(0,nCases):
        if problem_type == 'Periodic':
            base_mesh_file_name = 'base_mesh_%s.nc' %(iCase+1)
        elif problem_type == 'NonPeriodic':
            base_mesh_file_name = 'culled_mesh_%s.nc' %(iCase+1)
        mesh_file_name = 'mesh_%s.nc' %(iCase+1)
        dc[iCase], MaxErrorNorm[iCase], L2ErrorNorm[iCase] = (
        test_numerical_divergence_operator(False,False,mesh_directory,base_mesh_file_name,mesh_file_name,
                                           output_directory))
    A = np.vstack([np.log10(1.0/dc),np.ones(len(dc))]).T
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm))[0]
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'Maximum Error Norm of Numerical Divergence Operator'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalDivergenceOperatorConvergencePlot_MaxErrorNorm'
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,MaxErrorNorm,y,[2.0,2.0],[' ','-'],
                                        ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],
                                        [10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,True,
                                        FigureTitle,False,drawGrid=True,legendWithinBox=True)    
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm))[0]                     
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y                                                               
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'L2 Error Norm of Numerical Divergence Operator'
    legends = ['L2 Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalDivergenceOperatorConvergencePlot_L2ErrorNorm'    
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,L2ErrorNorm,y,[2.0,2.0],[' ','-'],
                                        ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],
                                        [10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,True,
                                        FigureTitle,False,drawGrid=True,legendWithinBox=True)    


# In[37]:

do_convergence_test_numerical_divergence_operator_1 = False
if do_convergence_test_numerical_divergence_operator_1:
    problem_type = 'Periodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    convergence_test_numerical_divergence_operator(problem_type,mesh_directory,output_directory)


# In[38]:

do_convergence_test_numerical_divergence_operator_2 = False
if do_convergence_test_numerical_divergence_operator_2:
    problem_type = 'NonPeriodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    convergence_test_numerical_divergence_operator(problem_type,mesh_directory,output_directory)


# In[39]:

def numerical_curl_operator(myMPAS_O,myVectorQuantityNormalToEdge,problem_type='Periodic'):
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    if problem_type == 'NonPeriodic':
        MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    myVectorQuantityCurlAtVertex = np.zeros(myMPAS_O.nVertices)
    myVectorQuantityCurlAtCellCenter = np.zeros(myMPAS_O.nCells)
    for iVertex in range(0,myMPAS_O.nVertices):
        if problem_type == 'NonPeriodic' and myMPAS_O.boundaryVertex[iVertex] == 1.0:
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


# In[40]:

def test_numerical_curl_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                 output_directory,problem_type='Periodic'):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name)
    prefix = problem_specific_prefix()
    analyticalVelocityCurlAtVertex = np.zeros(myMPAS_O.nVertices)
    for iVertex in range(0,myMPAS_O.nVertices):
        analyticalVelocityCurlAtVertex[iVertex] = velocity_curl(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xVertex[iVertex],
                                                                myMPAS_O.yVertex[iVertex])
    if plotFigures:
        Title = 'Analytical Velocity Curl At Vertex'
        FigureTitle = prefix + 'VelocityCurlAtVertex_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xVertex,myMPAS_O.yVertex,
                                                          analyticalVelocityCurlAtVertex,300,False,[0.0,0.0],'x',
                                                          10,'y',10,Title,True,FigureTitle,False) 
    analyticalVelocityCurlAtCellCenter = np.zeros(myMPAS_O.nCells)
    for iCell in range(0,myMPAS_O.nCells):
        analyticalVelocityCurlAtCellCenter[iCell] = velocity_curl(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xCell[iCell],
                                                                  myMPAS_O.yCell[iCell])    
    if plotFigures:
        Title = 'Analytical Velocity Curl At Cell Center'
        FigureTitle = prefix + 'VelocityCurlAtCellCenter_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          analyticalVelocityCurlAtCellCenter,300,False,[0.0,0.0],
                                                          'x',10,'y',10,Title,True,FigureTitle,False)        
    analyticalVelocityComponentsAtEdge = np.zeros((myMPAS_O.nEdges,2))
    analyticalVelocityNormalToEdge = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        analyticalVelocityComponentsAtEdge[iEdge,0], analyticalVelocityComponentsAtEdge[iEdge,1] = (
        velocity(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge]))
    analyticalVelocityNormalToEdge = (
    ComputeNormalAndTangentialComponentsAtEdge(analyticalVelocityComponentsAtEdge,myMPAS_O.angleEdge,'normal'))
    numericalVelocityCurlAtVertex, numericalVelocityCurlAtCellCenter = (
    numerical_curl_operator(myMPAS_O,analyticalVelocityNormalToEdge,problem_type))
    if plotFigures:
        Title = 'Numerical Velocity Curl At Vertex'
        FigureTitle = prefix + 'VelocityCurlAtVertex_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xVertex,myMPAS_O.yVertex,
                                                          numericalVelocityCurlAtVertex,300,False,[0.0,0.0],'x',10,
                                                          'y',10,Title,True,FigureTitle,False) 
        Title = 'Numerical Velocity Curl At Cell Center'
        FigureTitle = prefix + 'VelocityCurlAtCellCenter_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          numericalVelocityCurlAtCellCenter,300,False,[0.0,0.0],
                                                          'x',10,'y',10,Title,True,FigureTitle,False)
    VelocityCurlAtVertexError = numericalVelocityCurlAtVertex - analyticalVelocityCurlAtVertex
    if plotFigures:
        Title = 'Velocity Curl Error at Vertices'
        FigureTitle = prefix + 'VelocityCurlAtVertex_Error'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xVertex,myMPAS_O.yVertex,
                                                          VelocityCurlAtVertexError,300,False,[0.0,0.0],'x',10,'y',
                                                          10,Title,True,FigureTitle,False) 
    MaxErrorNorm_Vertex = np.linalg.norm(VelocityCurlAtVertexError,np.inf)
    L2ErrorNorm_Vertex = np.linalg.norm(VelocityCurlAtVertexError)/np.sqrt(myMPAS_O.nVertices)
    print('The maximum error norm of the velocity curl at vertices is %.3g.' %MaxErrorNorm_Vertex)
    print('The L2 error norm of the velocity curl at vertices is %.3g.' %L2ErrorNorm_Vertex)   
    VelocityCurlAtCellCenterError = numericalVelocityCurlAtCellCenter - analyticalVelocityCurlAtCellCenter
    if plotFigures:
        Title = 'Velocity Curl Error at Cell Centers'
        FigureTitle = prefix + 'VelocityCurlAtCellCenter_Error'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          VelocityCurlAtCellCenterError,300,False,[0.0,0.0],'x',10,
                                                          'y',10,Title,True,FigureTitle,False) 
    MaxErrorNorm_CellCenter = np.linalg.norm(VelocityCurlAtCellCenterError,np.inf)
    L2ErrorNorm_CellCenter = np.linalg.norm(VelocityCurlAtCellCenterError)/np.sqrt(myMPAS_O.nCells)
    print('The maximum error norm of the velocity curl at cell centers is %.3g.' %MaxErrorNorm_CellCenter)
    print('The L2 error norm of the velocity curl at cell centers is %.3g.' %L2ErrorNorm_CellCenter)  
    returningArguments = [myMPAS_O.gridSpacingMagnitude, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, 
                          MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter]
    return returningArguments


# In[41]:

do_test_numerical_curl_operator_1 = False
if do_test_numerical_curl_operator_1:
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    [dc, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter] = (
    test_numerical_curl_operator(True,True,' ',' ',' ',output_directory))


# In[42]:

do_test_numerical_curl_operator_2 = False
if do_test_numerical_curl_operator_2:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    problem_type = 'NonPeriodic'
    [dc, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter] = (
    test_numerical_curl_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                 output_directory,problem_type))


# In[43]:

do_test_numerical_curl_operator_3 = False
if do_test_numerical_curl_operator_3:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    [dc, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter] = (
    test_numerical_curl_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                 output_directory))


# In[44]:

do_test_numerical_curl_operator_4 = False
if do_test_numerical_curl_operator_4:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    problem_type = 'NonPeriodic'
    [dc, MaxErrorNorm_Vertex, L2ErrorNorm_Vertex, MaxErrorNorm_CellCenter, L2ErrorNorm_CellCenter] = (
    test_numerical_curl_operator(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                                 output_directory,problem_type))


# In[45]:

def convergence_test_numerical_curl_operator(problem_type,mesh_directory,output_directory):
    nCases = 5
    dc = np.zeros(nCases)
    MaxErrorNorm_Vertex = np.zeros(nCases)
    L2ErrorNorm_Vertex = np.zeros(nCases)
    MaxErrorNorm_CellCenter = np.zeros(nCases)
    L2ErrorNorm_CellCenter = np.zeros(nCases)    
    prefix = problem_specific_prefix()
    for iCase in range(0,nCases):
        if problem_type == 'Periodic':
            base_mesh_file_name = 'base_mesh_%s.nc' %(iCase+1)
        elif problem_type == 'NonPeriodic':
            base_mesh_file_name = 'culled_mesh_%s.nc' %(iCase+1)
        mesh_file_name = 'mesh_%s.nc' %(iCase+1)
        [dc[iCase], MaxErrorNorm_Vertex[iCase], L2ErrorNorm_Vertex[iCase], MaxErrorNorm_CellCenter[iCase], 
         L2ErrorNorm_CellCenter[iCase]] = test_numerical_curl_operator(False,False,mesh_directory,
                                                                       base_mesh_file_name,mesh_file_name,
                                                                       output_directory,problem_type)
    A = np.vstack([np.log10(1.0/dc),np.ones(len(dc))]).T
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm_Vertex))[0]
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'Maximum Error Norm of Numerical Curl Operator'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalCurlOperator_Vertex_ConvergencePlot_MaxErrorNorm'
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,MaxErrorNorm_Vertex,y,[2.0,2.0],
                                        [' ','-'],['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],
                                        [17.5,17.5],[10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,
                                        True,FigureTitle,False,drawGrid=True,legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm_Vertex))[0]
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'L2 Error Norm of Numerical Curl Operator'
    legends = ['L2 Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalCurlOperator_Vertex_ConvergencePlot_L2ErrorNorm'    
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,L2ErrorNorm_Vertex,y,[2.0,2.0],
                                        [' ','-'],['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],
                                        [17.5,17.5],[10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,
                                        True,FigureTitle,False,drawGrid=True,legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm_CellCenter))[0]
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'Maximum Error Norm of Numerical Curl Operator'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalCurlOperator_CellCenter_ConvergencePlot_MaxErrorNorm'
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,MaxErrorNorm_CellCenter,y,[2.0,2.0],
                                        [' ','-'],['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],
                                        [17.5,17.5],[10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,
                                        True,FigureTitle,False,drawGrid=True,legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm_CellCenter))[0]
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'L2 Error Norm of Numerical Curl Operator'
    legends = ['L2 Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalCurlOperator_CellCenter_ConvergencePlot_L2ErrorNorm'  
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,L2ErrorNorm_CellCenter,y,[2.0,2.0],
                                        [' ','-'],['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],
                                        [17.5,17.5],[10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,
                                        True,FigureTitle,False,drawGrid=True,legendWithinBox=True)


# In[46]:

do_convergence_test_numerical_curl_operator_1 = False
if do_convergence_test_numerical_curl_operator_1:
    problem_type = 'Periodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    convergence_test_numerical_curl_operator(problem_type,mesh_directory,output_directory)


# In[47]:

do_convergence_test_numerical_curl_operator_2 = False
if do_convergence_test_numerical_curl_operator_2:
    problem_type = 'NonPeriodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    convergence_test_numerical_curl_operator(problem_type,mesh_directory,output_directory)


# In[48]:

def numerical_tangential_velocity(myMPAS_O,myNormalVelocity,problem_type='Periodic'):
    if problem_type == 'NonPeriodic':
        MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    myTangentialVelocity = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        if problem_type == 'NonPeriodic' and myMPAS_O.boundaryEdge[iEdge] == 1.0:
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


# In[49]:

def test_tangential_velocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                             output_directory,problem_type='Periodic'):
    if useDefaultMesh:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False)
    else:
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name)
    prefix = problem_specific_prefix()
    analyticalVelocityComponentsAtEdge = np.zeros((myMPAS_O.nEdges,2))
    analyticalNormalVelocity = np.zeros(myMPAS_O.nEdges)
    analyticalTangentialVelocity = np.zeros(myMPAS_O.nEdges)
    for iEdge in range(0,myMPAS_O.nEdges):
        analyticalVelocityComponentsAtEdge[iEdge,0], analyticalVelocityComponentsAtEdge[iEdge,1] = (
        velocity(myMPAS_O.lX,myMPAS_O.lY,myMPAS_O.xEdge[iEdge],myMPAS_O.yEdge[iEdge]))
    analyticalNormalVelocity, analyticalTangentialVelocity = (
    ComputeNormalAndTangentialComponentsAtEdge(analyticalVelocityComponentsAtEdge,myMPAS_O.angleEdge,'both'))
    if plotFigures:
        Title = 'Analytical Tangential Velocity'
        FigureTitle = prefix + 'TangentialVelocity_Analytical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                          analyticalTangentialVelocity,300,False,[0.0,0.0],'x',10,
                                                          'y',10,Title,True,FigureTitle,False) 
    numericalTangentialVelocity = numerical_tangential_velocity(myMPAS_O,analyticalNormalVelocity,problem_type)
    if plotFigures:
        Title = 'Numerical Tangential Velocity'
        FigureTitle = prefix + 'TangentialVelocity_Numerical'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                          numericalTangentialVelocity,300,False,[0.0,0.0],'x',10,
                                                          'y',10,Title,True,FigureTitle,False)     
    TangentialVelocityError = numericalTangentialVelocity - analyticalTangentialVelocity
    if plotFigures:
        Title = 'Tangential Velocity Error'
        FigureTitle = prefix + 'TangentialVelocity_Error'
        CR.PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                          TangentialVelocityError,300,False,[0.0,0.0],'x',10,'y',
                                                          10,Title,True,FigureTitle,False) 
    MaxErrorNorm = np.linalg.norm(TangentialVelocityError,np.inf)
    L2ErrorNorm = np.linalg.norm(TangentialVelocityError)/np.sqrt(myMPAS_O.nEdges)
    print('The maximum error norm of the tangential velocity is %.3g.' %MaxErrorNorm)
    print('The L2 error norm of the tangential velocity is %.3g.' %L2ErrorNorm)       
    return myMPAS_O.gridSpacingMagnitude, MaxErrorNorm, L2ErrorNorm


# In[50]:

do_test_tangential_velocity_1 = False
if do_test_tangential_velocity_1:
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    dc, MaxErrorNorm, L2ErrorNorm = test_tangential_velocity(True,True,' ',' ',' ',output_directory)


# In[51]:

do_test_tangential_velocity_2 = False
if do_test_tangential_velocity_2:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_x'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    problem_type = 'NonPeriodic'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_tangential_velocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                             output_directory,problem_type))


# In[52]:

do_test_tangential_velocity_3 = False
if do_test_tangential_velocity_3:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_tangential_velocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                             output_directory))


# In[53]:

do_test_tangential_velocity_4 = False
if do_test_tangential_velocity_4:
    useDefaultMesh = False
    plotFigures = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP.nc'
    mesh_file_name = 'mesh_NP.nc'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    problem_type = 'NonPeriodic'
    dc, MaxErrorNorm, L2ErrorNorm = (
    test_tangential_velocity(useDefaultMesh,plotFigures,mesh_directory,base_mesh_file_name,mesh_file_name,
                             output_directory,problem_type))


# In[54]:

def convergence_test_tangential_velocity(problem_type,mesh_directory,output_directory):
    nCases = 5
    dc = np.zeros(nCases)
    MaxErrorNorm = np.zeros(nCases)
    L2ErrorNorm = np.zeros(nCases)
    prefix = problem_specific_prefix()
    for iCase in range(0,nCases):
        if problem_type == 'Periodic':
            base_mesh_file_name = 'base_mesh_%s.nc' %(iCase+1)
        elif problem_type == 'NonPeriodic':
            base_mesh_file_name = 'culled_mesh_%s.nc' %(iCase+1)
        mesh_file_name = 'mesh_%s.nc' %(iCase+1)
        dc[iCase], MaxErrorNorm[iCase], L2ErrorNorm[iCase] = (
        test_tangential_velocity(False,False,mesh_directory,base_mesh_file_name,mesh_file_name,output_directory,
                                 problem_type))
    A = np.vstack([np.log10(1.0/dc),np.ones(len(dc))]).T
    m, c = np.linalg.lstsq(A,np.log10(MaxErrorNorm))[0]
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'Maximum Error Norm of Numerical Tangential Velocity'
    legends = ['Maximum Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. Maximum Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalTangentialVelocityConvergencePlot_MaxErrorNorm'
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,MaxErrorNorm,y,[2.0,2.0],[' ','-'],
                                        ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],
                                        [10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,True,
                                        FigureTitle,False,drawGrid=True,legendWithinBox=True)
    m, c = np.linalg.lstsq(A,np.log10(L2ErrorNorm))[0]
    y = m*(np.log10(1.0/dc)) + c
    y = 10.0**y
    xLabel = 'Grid Spacing Inverse'
    yLabel = 'L2 Error Norm of Numerical Tangential Velocity'
    legends = ['L2 Error Norm','Best Fit Straight Line']
    Title = 'Convergence Plot w.r.t. L2 Error Norm: Slope is %.3g' %m
    FigureTitle = prefix + 'NumericalTangentialVelocityConvergencePlot_L2ErrorNorm'    
    CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,L2ErrorNorm,y,[2.0,2.0],[' ','-'],
                                        ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],[17.5,17.5],
                                        [10.0,10.0],[15.0,15.0],legends,17.5,'upper right',Title,20.0,True,
                                        FigureTitle,False,drawGrid=True,legendWithinBox=True)


# In[55]:

do_convergence_test_tangential_velocity_1 = False
if do_convergence_test_tangential_velocity_1:
    problem_type = 'Periodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/Periodic'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/Periodic'
    convergence_test_tangential_velocity(problem_type,mesh_directory,output_directory)


# In[56]:

do_convergence_test_tangential_velocity_2 = False
if do_convergence_test_tangential_velocity_2:
    problem_type = 'NonPeriodic'
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/ConvergenceStudyMeshes/NonPeriodic_x'
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Operator_Testing_Figures/NonPeriodic_x'
    convergence_test_tangential_velocity(problem_type,mesh_directory,output_directory)