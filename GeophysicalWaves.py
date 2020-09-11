
# coding: utf-8

# Name: GeophysicalWaves.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains routines for verifying the barotropic solver of MPAS-Ocean against test cases involving geophysical waves. <br/>

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.ticker import FuncFormatter
import io as inputoutput
import os
import sys
import time
from IPython.utils import io
with io.capture_output() as captured: 
    import Common_Routines as CR
    import GeophysicalWaves_ExactSolutions_SourceTerms as GWESST
    import MPAS_O_Mode_Init
    import MPAS_O_Shared
    import MPAS_O_Mesh_Interpolation_Routines as MOMIR
    import MPAS_O_Mode_Forward


# In[2]:

def DetermineCoastalKelvinWaveSurfaceElevationAmplitudeThroughCentersOfCellsAlongCoast():
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                       problem_type,problem_is_linear,periodicity)    
    etaHat1 = myMPAS_O.ExactSolutionParameters[10]
    H = myMPAS_O.ExactSolutionParameters[14]
    R = myMPAS_O.ExactSolutionParameters[23]
    xCell = 0.5*myMPAS_O.gridSpacingMagnitude
    CoastalKelvinWaveSurfaceElevationAmplitudeAtCentersOfCellsAlongCoast = (
    -(H*etaHat1*float(GWESST.DetermineKelvinWaveAmplitude())*np.exp(-xCell/R)))
    PrintStatement1 = 'The amplitude of the section of the coastal Kelvin wave surface elevation '
    PrintStatement2 = ('passing through the centers of the cells along the coast is %.6f.' 
                       %abs(CoastalKelvinWaveSurfaceElevationAmplitudeAtCentersOfCellsAlongCoast))
    print(PrintStatement1 + PrintStatement2)


# In[3]:

do_DetermineCoastalKelvinWaveSurfaceElevationAmplitudeThroughCentersOfCellsAlongCoast = False
if do_DetermineCoastalKelvinWaveSurfaceElevationAmplitudeThroughCentersOfCellsAlongCoast:
    DetermineCoastalKelvinWaveSurfaceElevationAmplitudeThroughCentersOfCellsAlongCoast()


# In[4]:

def DetermineEquatorialKelvinWaveSurfaceElevationAmplitudeAlongEquator():
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'Equatorial_Kelvin_Wave'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                       problem_type,problem_is_linear,periodicity)    
    etaHat1 = myMPAS_O.ExactSolutionParameters[10]
    H = myMPAS_O.ExactSolutionParameters[14]
    Req = myMPAS_O.ExactSolutionParameters[24]
    yCell = 0.0 
    EquatorialKelvinWaveSurfaceElevationAmplitudeAlongEquator = (
    H*etaHat1*float(GWESST.DetermineKelvinWaveAmplitude())*np.exp(-0.5*(yCell/Req)**2.0))
    print('The amplitude of the equatorial Kelvin wave surface elevation along the equator is %.6f.'
          %abs(EquatorialKelvinWaveSurfaceElevationAmplitudeAlongEquator))


# In[5]:

do_DetermineEquatorialKelvinWaveSurfaceElevationAmplitudeAlongEquator = False
if do_DetermineEquatorialKelvinWaveSurfaceElevationAmplitudeAlongEquator:
    DetermineEquatorialKelvinWaveSurfaceElevationAmplitudeAlongEquator()


# In[6]:

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


# In[7]:

def PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,x,y,phi,nContours,useGivenColorBarLimits,
                                                        ColorBarLimits,nColorBarTicks,colormap,
                                                        colorbarfontsize,labels,labelfontsizes,labelpads,
                                                        tickfontsizes,title,titlefontsize,SaveAsPDF,FigureTitle,
                                                        Show,fig_size=[10.0,10.0],cbarlabelformat='%.2g'):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    set_aspect_equal = False
    if set_aspect_equal:
        ax.set_aspect('equal')
    else:
        xMin = min(x[:])
        xMax = max(x[:])
        yMin = min(y[:])
        yMax = max(y[:])        
        aspect_ratio = (xMax - xMin)/(yMax - yMin)
        ax.set_aspect(aspect_ratio,adjustable='box')
    if useGivenColorBarLimits:
        cbar_min = ColorBarLimits[0]
        cbar_max = ColorBarLimits[1]
    else:
        cbar_min = np.min(phi)
        cbar_max = np.max(phi)
    n_cbar_ticks = nColorBarTicks
    cbarlabels = np.linspace(cbar_min,cbar_max,num=n_cbar_ticks,endpoint=True)
    FCP = plt.tricontourf(x,y,phi,nContours,vmin=cbar_min,vmax=cbar_max,cmap=colormap) 
    # FCP stands for filled contour plot
    plt.title(title,fontsize=titlefontsize,y=1.035)
    cbarShrinkRatio = 0.825
    m = plt.cm.ScalarMappable(cmap=colormap)
    m.set_array(phi)
    m.set_clim(cbar_min,cbar_max)
    make_colorbar_boundaries_discrete = False
    if make_colorbar_boundaries_discrete:
        cbar = plt.colorbar(m,boundaries=cbarlabels,shrink=cbarShrinkRatio)
    else:
        cbar = plt.colorbar(m,shrink=cbarShrinkRatio)
    cbar.set_ticks(cbarlabels)
    cbar.set_ticklabels(cbarlabels)
    cbar.ax.set_yticklabels([cbarlabelformat %x for x in cbarlabels], fontsize=colorbarfontsize)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    plt.xticks(fontsize=tickfontsizes[0])
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/1000.0), '')))
    plt.yticks(fontsize=tickfontsizes[1])
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda y, p: format(int(y/1000.0), '')))
    if SaveAsPDF:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[8]:

def PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,phi,nContours,
                                                        useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,
                                                        colormap,colorbarfontsize,labels,labelfontsizes,labelpads,
                                                        tickfontsizes,title,titlefontsize,SaveAsPDF,FigureTitle,
                                                        Show,fig_size=[10.0,10.0],cbarlabelformat='%.2g'):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    set_aspect_equal = False
    if set_aspect_equal:
        ax.set_aspect('equal')
    else:
        xMin = 0.0 
        xMax = myMPAS_O.lX + myMPAS_O.gridSpacingMagnitude/2.0
        yMin = 0.0 
        yMax = myMPAS_O.lY + myMPAS_O.gridSpacingMagnitude/(2.0*np.sqrt(3.0))   
        aspect_ratio = (xMax - xMin)/(yMax - yMin)
        ax.set_aspect(aspect_ratio,adjustable='box')
    if useGivenColorBarLimits:
        cbar_min = ColorBarLimits[0]
        cbar_max = ColorBarLimits[1]
    else:
        cbar_min = np.min(phi)
        cbar_max = np.max(phi)
    n_cbar_ticks = nColorBarTicks
    cbarlabels = np.linspace(cbar_min,cbar_max,num=n_cbar_ticks,endpoint=True)
    patches = []
    ComparisonTolerance = 10.0**(-10.0)
    for iCell in range(0,myMPAS_O.nCells):
        nVerticesOnCell = myMPAS_O.nEdgesOnCell[iCell] 
        vertexIndices = np.zeros(nVerticesOnCell,dtype=int)
        vertexIndices[:] = myMPAS_O.verticesOnCell[iCell,:]
        vertexIndices -= 1
        vertices = np.zeros((nVerticesOnCell,2))
        xCell = myMPAS_O.xCell[iCell]
        yCell = myMPAS_O.yCell[iCell]
        for iVertexOnCell in range(0,nVerticesOnCell):
            xVertex = myMPAS_O.xVertex[vertexIndices[iVertexOnCell]]
            yVertex = myMPAS_O.yVertex[vertexIndices[iVertexOnCell]]
            if abs(yVertex - yCell) > (2.0/np.sqrt(3.0))*myMPAS_O.gridSpacingMagnitude and yVertex < yCell:
                yVertex = yCell + myMPAS_O.gridSpacingMagnitude/np.sqrt(3.0)  
            if abs(yVertex - yCell) > (2.0/np.sqrt(3.0))*myMPAS_O.gridSpacingMagnitude and yVertex > yCell:
                yVertex = yCell - myMPAS_O.gridSpacingMagnitude/np.sqrt(3.0)                 
            if abs(xVertex - xCell) > myMPAS_O.gridSpacingMagnitude and xVertex < xCell:
                if abs(yVertex - (yCell + myMPAS_O.gridSpacingMagnitude/np.sqrt(3.0))) < ComparisonTolerance:
                    xVertex = xCell
                elif abs(yVertex - (yCell - myMPAS_O.gridSpacingMagnitude/np.sqrt(3.0))) < ComparisonTolerance:
                    xVertex = xCell
                else:                
                    xVertex = xCell + 0.5*myMPAS_O.gridSpacingMagnitude      
            vertices[iVertexOnCell,0] = xVertex
            vertices[iVertexOnCell,1] = yVertex
        polygon = Polygon(vertices,True)
        patches.append(polygon)    
    localPatches = PatchCollection(patches,cmap=colormap,alpha=1.0) 
    localPatches.set_array(phi)
    ax.add_collection(localPatches)
    localPatches.set_clim([cbar_min,cbar_max])
    problem_type = myMPAS_O.myNamelist.config_problem_type
    if myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave:
        yMin = -0.5*yMax
        yMax *= 0.5
    plt.axis([xMin,xMax,yMin,yMax])
    plt.title(title,fontsize=titlefontsize,y=1.035)
    cbarShrinkRatio = 0.825
    m = plt.cm.ScalarMappable(cmap=colormap)
    m.set_array(phi)
    m.set_clim(cbar_min,cbar_max)
    make_colorbar_boundaries_discrete = False
    if make_colorbar_boundaries_discrete:
        cbar = plt.colorbar(m,boundaries=cbarlabels,shrink=cbarShrinkRatio)
    else:
        cbar = plt.colorbar(m,shrink=cbarShrinkRatio)
    cbar.set_ticks(cbarlabels)
    cbar.set_ticklabels(cbarlabels)
    cbar.ax.set_yticklabels([cbarlabelformat %x for x in cbarlabels], fontsize=colorbarfontsize)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    plt.xticks(fontsize=tickfontsizes[0])
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/1000.0), '')))
    plt.yticks(fontsize=tickfontsizes[1])
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda y, p: format(int(y/1000.0), '')))
    if (myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave 
        or myMPAS_O.myNamelist.config_problem_type == 'Inertia_Gravity_Wave'):
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
        ax.yaxis.set_major_locator(plt.MaxNLocator(7))
    if SaveAsPDF:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[9]:

def PrintDisplayTime(time,non_integral_seconds=False,display_time=False):
    years = np.floor(time/(86400.0*365.0))
    remainingtime = np.mod(time,86400.0*365.0)
    days = np.floor(remainingtime/86400.0)
    remainingtime = np.mod(remainingtime,86400.0)
    hours = np.floor(remainingtime/3600.0)
    remainingtime = np.mod(time,3600.0)
    minutes = np.floor(remainingtime/60.0)
    seconds = np.mod(remainingtime,60.0)
    if years <= 1.0:
        years_string = 'Year'
    else:
        years_string = 'Years'
    if days <= 1.0:
        days_string = 'Day'
    else:
        days_string = 'Days'
    if hours <= 1.0:
        hours_string = 'Hour'
    else:
        hours_string = 'Hours'
    if minutes <= 1.0:
        minutes_string = 'Minute'
    else:
        minutes_string = 'Minutes'        
    if seconds <= 1.0:
        seconds_string = 'Second'
    else:
        seconds_string = 'Seconds'
    if time >= 86400.0*365.0:
        if non_integral_seconds:
            DisplayTime = ('%d %s %d %s %2d %s %2d %s %.2f %s' 
                           %(years,years_string,days,days_string,hours,hours_string,minutes,minutes_string,
                             seconds,seconds_string))            
        else:
            DisplayTime = ('%d %s %d %s %2d %s %2d %s %2d %s' 
                           %(years,years_string,days,days_string,hours,hours_string,minutes,minutes_string,
                             seconds,seconds_string))
    elif time < 86400.0*365.0 and time >= 86400.0:
        if non_integral_seconds:
            DisplayTime = ('%d %s %2d %s %2d %s %.2f %s' 
                           %(days,days_string,hours,hours_string,minutes,minutes_string,seconds,seconds_string))
        else:
            DisplayTime = ('%d %s %2d %s %2d %s %2d %s' 
                           %(days,days_string,hours,hours_string,minutes,minutes_string,seconds,seconds_string))
    elif time < 86400.0 and time >= 3600.0:
        if non_integral_seconds:
            DisplayTime = ('%2d %s %2d %s %.2f %s' %(hours,hours_string,minutes,minutes_string,seconds,
                                                     seconds_string))
        else:
            DisplayTime = ('%2d %s %2d %s %2d %s' %(hours,hours_string,minutes,minutes_string,seconds,
                                                    seconds_string))
    elif time < 3600.0 and time >= 60.0:
        if non_integral_seconds:
            DisplayTime = ('%2d %s %.2f %s' %(minutes,minutes_string,seconds,seconds_string))
        else:
            DisplayTime = ('%2d %s %2d %s' %(minutes,minutes_string,seconds,seconds_string))
    elif time < 60.0:
        if non_integral_seconds:
            DisplayTime = ('%.2f %s' %(seconds,seconds_string))
        else:
            DisplayTime = ('%2d %s' %(seconds,seconds_string))
    if display_time:
        print('The Display Time is %s.' %DisplayTime)
    return DisplayTime


# In[10]:

do_PrintDisplayTime_1 = False
if do_PrintDisplayTime_1:
    time = 3.0*365.0*86400.0 + 3.0*86400.0 + 3.0*3600.0 + 3.0*60.0 + 3.0
    display_time = True
    DisplayTime = PrintDisplayTime(time,display_time)


# In[11]:

do_PrintDisplayTime_2 = False
if do_PrintDisplayTime_2:
    time = 3.0*86400.0 + 3.0*3600.0 + 3.0*60.0 + 3.0
    display_time = True
    DisplayTime = PrintDisplayTime(time,display_time)


# In[12]:

do_PrintDisplayTime_3 = False
if do_PrintDisplayTime_3:
    time = 3.0*3600.0 + 3.0*60.0 + 3.0
    display_time = True
    DisplayTime = PrintDisplayTime(time,display_time)


# In[13]:

do_PrintDisplayTime_4 = False
if do_PrintDisplayTime_4:
    time = 3.0*60.0 + 3.0
    display_time = True
    DisplayTime = PrintDisplayTime(time,display_time)


# In[14]:

do_PrintDisplayTime_5 = False
if do_PrintDisplayTime_5:
    time = 3.0
    display_time = True
    DisplayTime = PrintDisplayTime(time,display_time)


# In[15]:

def DetermineCourantNumberForGivenTimeStep(mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                           problem_type,problem_is_linear,periodicity,dt,printCourantNumber=False):
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                       problem_type,problem_is_linear,periodicity)
    dx = myMPAS_O.ExactSolutionParameters[8]
    dy = myMPAS_O.ExactSolutionParameters[9]
    if myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave or problem_type == 'Barotropic_Tide':
        cX1 = myMPAS_O.ExactSolutionParameters[4]
        cX2 = myMPAS_O.ExactSolutionParameters[5]
        cY1 = myMPAS_O.ExactSolutionParameters[6]
        cY2 = myMPAS_O.ExactSolutionParameters[7]
        abs_cX = max(abs(cX1),abs(cX2))
        abs_cY = max(abs(cY1),abs(cY2))
        CourantNumber = dt*(abs_cX/dx + abs_cY/dy)
        if printCourantNumber:
            print('The Courant number is %.6f.' %CourantNumber)
    elif problem_type == 'Diffusion_Equation':
        kappaX = myMPAS_O.ExactSolutionParameters[29]
        kappaY = myMPAS_O.ExactSolutionParameters[30]
        CourantNumber = dt*(kappaX/dx**2.0 + kappaY/dy**2.0)
        if printCourantNumber:
            print('The stability coefficient is %.6f.' %CourantNumber)
    elif problem_type == 'Viscous_Burgers_Equation':
        s = myMPAS_O.ExactSolutionParameters[35]
        CourantNumber = s*dt/dx
        if printCourantNumber:
            print('The Courant number is %.6f.' %CourantNumber)
    return CourantNumber


# In[16]:

def DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type):
    if problem_type == 'Coastal_Kelvin_Wave':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_x'
        dt = 180.0
        printPhaseSpeedOfWaveModes = True
    elif problem_type == 'Inertia_Gravity_Wave':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/InertiaGravityWaveMesh'
        base_mesh_file_name = 'base_mesh.nc'
        periodicity = 'Periodic'
        dt = 96.0
        printPhaseSpeedOfWaveModes = True
    elif problem_type == 'Planetary_Rossby_Wave' or problem_type == 'Topographic_Rossby_Wave':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/PlanetaryRossbyWaveMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_y'
        dt = 195000.0   
        printPhaseSpeedOfWaveModes = True
    elif (problem_type == 'Equatorial_Kelvin_Wave' or problem_type == 'Equatorial_Yanai_Wave' 
          or problem_type == 'Equatorial_Rossby_Wave' or problem_type == 'Equatorial_Inertia_Gravity_Wave'):
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/EquatorialWaveMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_y'
        if problem_type == 'Equatorial_Kelvin_Wave':
            dt = 750.0
        elif problem_type == 'Equatorial_Yanai_Wave':
            dt = 390.0
        elif problem_type == 'Equatorial_Rossby_Wave':
            dt = 2700.0
        elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
            dt = 420.0
        printPhaseSpeedOfWaveModes = True
    elif problem_type == 'Barotropic_Tide':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/BarotropicTideMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_xy'  
        dt = 10.0
        printPhaseSpeedOfWaveModes = True
    elif problem_type == 'Diffusion_Equation':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/BarotropicTideMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_xy'
        dt = 2260.0
        printPhaseSpeedOfWaveModes = False 
    elif problem_type == 'Viscous_Burgers_Equation':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/BarotropicTideMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_xy'  
        dt = 2100.0
        printPhaseSpeedOfWaveModes = False         
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_is_linear = True
    do_fixAngleEdge = True
    print_Output = False
    CourantNumber = DetermineCourantNumberForGivenTimeStep(mesh_directory,base_mesh_file_name,mesh_file_name,
                                                           mesh_type,problem_type,problem_is_linear,periodicity,dt,
                                                           printCourantNumber=True)
    useCourantNumberToDetermineTimeStep = True
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                       problem_type,problem_is_linear,periodicity,do_fixAngleEdge,print_Output,
                                       CourantNumber,useCourantNumberToDetermineTimeStep,
                                       printPhaseSpeedOfWaveModes=printPhaseSpeedOfWaveModes)
    if problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Planetary_Rossby_Wave':
        beta0 = myMPAS_O.ExactSolutionParameters[2]
        c = myMPAS_O.ExactSolutionParameters[3]
        f0 = myMPAS_O.ExactSolutionParameters[12]
        kX1 = myMPAS_O.ExactSolutionParameters[15] 
        kX2 = myMPAS_O.ExactSolutionParameters[16]
        kY1 = myMPAS_O.ExactSolutionParameters[17] 
        kY2 = myMPAS_O.ExactSolutionParameters[18]
        lY = myMPAS_O.ExactSolutionParameters[20]
        k1 = np.sqrt(kX1**2.0 + kY1**2.0)
        k2 = np.sqrt(kX2**2.0 + kY2**2.0)
        if problem_type == 'Inertia_Gravity_Wave':
            print('For the first wave mode, the ratio of f0:ck is %.6f.' %(f0/(c*k1)))
            print('For the second wave mode, the ratio of f0:ck is %.6f.' %(f0/(c*k2)))
        else:
            print('With the meridional extent being %.3f km, the ratio of beta0*lY:f0 is %.6f << 1.' 
                  %(lY/1000.0,beta0*lY/f0))


# In[17]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_CoastalKelvinWave = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_CoastalKelvinWave:
    problem_type = 'Coastal_Kelvin_Wave'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[18]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_InertiaGravityWave = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_InertiaGravityWave:
    problem_type = 'Inertia_Gravity_Wave'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[19]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_PlanetaryRossbyWave = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_PlanetaryRossbyWave:
    problem_type = 'Planetary_Rossby_Wave'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[20]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_TopographicRossbyWave = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_TopographicRossbyWave:
    problem_type = 'Topographic_Rossby_Wave'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[21]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_EquatorialKelvinWave = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_EquatorialKelvinWave:
    problem_type = 'Equatorial_Kelvin_Wave'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[22]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_EquatorialYanaiWave = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_EquatorialYanaiWave:
    problem_type = 'Equatorial_Yanai_Wave'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[23]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_EquatorialRossbyWave = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_EquatorialRossbyWave:
    problem_type = 'Equatorial_Rossby_Wave'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[24]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_EquatorialInertiaGravityWave = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_EquatorialInertiaGravityWave:
    problem_type = 'Equatorial_Inertia_Gravity_Wave'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[25]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_BarotropicTide = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_BarotropicTide:
    problem_type = 'Barotropic_Tide'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[26]:

do_DetermineStabilityCoefficientForGivenTimeStepAndCheckIfItIsLessThanHalf_DiffusionEquation = False
if do_DetermineStabilityCoefficientForGivenTimeStepAndCheckIfItIsLessThanHalf_DiffusionEquation:
    problem_type = 'Diffusion_Equation'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[27]:

do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_ViscousBurgersEquation = False
if do_DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne_ViscousBurgersEquation:
    problem_type = 'Viscous_Burgers_Equation'
    DetermineCourantNumberForGivenTimeStepAndCheckIfItIsLessThanOne(problem_type)


# In[28]:

def DetermineGeophysicalWaveExactSolutionsAtCellCenters(myMPAS_O):
    time = myMPAS_O.time
    GeophysicalWaveExactSurfaceElevations = np.zeros(myMPAS_O.nCells) 
    GeophysicalWaveExactZonalVelocities = np.zeros(myMPAS_O.nCells)
    GeophysicalWaveExactMeridionalVelocities = np.zeros(myMPAS_O.nCells) 
    for iCell in range(0,myMPAS_O.nCells):
        xCell = myMPAS_O.xCell[iCell]
        yCell = myMPAS_O.yCell[iCell]
        GeophysicalWaveExactSurfaceElevations[iCell] = (
        GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                             myMPAS_O.ExactSolutionParameters,xCell,yCell,time))
        GeophysicalWaveExactZonalVelocities[iCell] = (
        GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                          myMPAS_O.ExactSolutionParameters,xCell,yCell,time))
        GeophysicalWaveExactMeridionalVelocities[iCell] = (
        GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                               myMPAS_O.ExactSolutionParameters,xCell,yCell,time))
    return [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocities, 
            GeophysicalWaveExactMeridionalVelocities]


# In[29]:

def DetermineGeophysicalWaveExactSolutions(myMPAS_O,DetermineGeophysicalWaveExactSurfaceElevations=True):
    time = myMPAS_O.time
    if DetermineGeophysicalWaveExactSurfaceElevations:
        GeophysicalWaveExactSurfaceElevations = np.zeros(myMPAS_O.nCells) 
    GeophysicalWaveExactZonalVelocities = np.zeros(myMPAS_O.nEdges)
    GeophysicalWaveExactMeridionalVelocities = np.zeros(myMPAS_O.nEdges) 
    if DetermineGeophysicalWaveExactSurfaceElevations:
        for iCell in range(0,myMPAS_O.nCells):
            xCell = myMPAS_O.xCell[iCell]
            yCell = myMPAS_O.yCell[iCell]
            GeophysicalWaveExactSurfaceElevations[iCell] = (
            GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                                 myMPAS_O.ExactSolutionParameters,xCell,yCell,
                                                                 time))
    for iEdge in range(0,myMPAS_O.nEdges):
        xEdge = myMPAS_O.xEdge[iEdge]
        yEdge = myMPAS_O.yEdge[iEdge]                                                
        GeophysicalWaveExactZonalVelocities[iEdge] = (
        GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                          myMPAS_O.ExactSolutionParameters,xEdge,yEdge,time))
        GeophysicalWaveExactMeridionalVelocities[iEdge] = (
        GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                               myMPAS_O.ExactSolutionParameters,xEdge,yEdge,time))
    if DetermineGeophysicalWaveExactSurfaceElevations:
        return [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocities, 
                GeophysicalWaveExactMeridionalVelocities]
    else:
        return [GeophysicalWaveExactZonalVelocities, GeophysicalWaveExactMeridionalVelocities]


# In[30]:

def DetermineGeophysicalWaveExactSolutionsOnCoarsestRectilinearMesh(myMPAS_O,xCell_CoarsestRectilinearMesh,
                                                                    yCell_CoarsestRectilinearMesh):
    time = myMPAS_O.time
    GeophysicalWaveExactSurfaceElevations = np.zeros(myMPAS_O.nCells) 
    GeophysicalWaveExactZonalVelocitiesAtCellCenters = np.zeros(myMPAS_O.nCells)
    GeophysicalWaveExactMeridionalVelocitiesAtCellCenters = np.zeros(myMPAS_O.nCells) 
    for iCell in range(0,myMPAS_O.nCells):
        xCell = xCell_CoarsestRectilinearMesh[iCell]
        yCell = yCell_CoarsestRectilinearMesh[iCell]
        GeophysicalWaveExactSurfaceElevations[iCell] = (
        GWESST.DetermineGeophysicalWaveExactSurfaceElevation(myMPAS_O.myNamelist.config_problem_type,
                                                             myMPAS_O.ExactSolutionParameters,xCell,yCell,time))
        GeophysicalWaveExactZonalVelocitiesAtCellCenters[iCell] = (
        GWESST.DetermineGeophysicalWaveExactZonalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                          myMPAS_O.ExactSolutionParameters,xCell,yCell,time))    
        GeophysicalWaveExactMeridionalVelocitiesAtCellCenters[iCell] = (
        GWESST.DetermineGeophysicalWaveExactMeridionalVelocity(myMPAS_O.myNamelist.config_problem_type,
                                                               myMPAS_O.ExactSolutionParameters,xCell,yCell,time))
    return [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocitiesAtCellCenters, 
            GeophysicalWaveExactMeridionalVelocitiesAtCellCenters]


# In[31]:

def SpecifyZonalExtent(problem_type):
    if problem_type == 'Coastal_Kelvin_Wave':
        ZonalExtent = 100.0*50000.0
    elif problem_type == 'Inertia_Gravity_Wave':
        ZonalExtent = 100.0*200000.0
    elif problem_type == 'Planetary_Rossby_Wave' or problem_type == 'Topographic_Rossby_Wave':    
        ZonalExtent = 100.0*500.0
    elif (problem_type == 'Equatorial_Kelvin_Wave' or problem_type == 'Equatorial_Yanai_Wave'
          or problem_type == 'Equatorial_Rossby_Wave' or problem_type == 'Equatorial_Inertia_Gravity_Wave'):     
        ZonalExtent = 100.0*175000.0
    elif (problem_type == 'Barotropic_Tide' or problem_type == 'Diffusion_Equation' 
          or problem_type == 'Viscous_Burgers_Equation'):
        ZonalExtent = 100.0*2500.0
    return ZonalExtent  


# In[32]:

def SpecifyNumberOfTimeSteps(problem_type):
    if problem_type == 'Coastal_Kelvin_Wave':
        nTime = 242 + 1
    elif problem_type == 'Inertia_Gravity_Wave':
        nTime = 476 + 1 
    elif problem_type == 'Planetary_Rossby_Wave' or problem_type == 'Topographic_Rossby_Wave':    
        nTime = 472 + 1
    elif (problem_type == 'Equatorial_Kelvin_Wave' or problem_type == 'Equatorial_Rossby_Wave' 
          or problem_type == 'Equatorial_Inertia_Gravity_Wave'): 
        nTime = 236 + 1
    elif problem_type == 'Equatorial_Yanai_Wave':
        nTime = 244 + 1
    elif problem_type == 'Barotropic_Tide':
        nTime = 250 + 1
    elif problem_type == 'Diffusion_Equation':
        nTime = 840 + 1
    elif problem_type == 'Viscous_Burgers_Equation':
        nTime = 120 + 1
    # Note that nTime is the minimum integer such that nTime - 1 is a multiple of nDumpFrequency and (nTime - 1) 
    # times the time step is greater than or equal to the time period of the geophysical wave.
    return nTime


# In[33]:

def SpecifyDumpFrequency(problem_type):
    if (problem_type == 'Coastal_Kelvin_Wave' or problem_type == 'Equatorial_Kelvin_Wave' 
        or problem_type == 'Equatorial_Yanai_Wave' or problem_type == 'Equatorial_Rossby_Wave' 
        or problem_type == 'Equatorial_Inertia_Gravity_Wave' or problem_type == 'Barotropic_Tide'):
        nDumpFrequency = 2
    elif (problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Planetary_Rossby_Wave' 
          or problem_type == 'Topographic_Rossby_Wave' or problem_type == 'Equatorial_Yanai_Wave'):
        nDumpFrequency = 4
    elif problem_type == 'Diffusion_Equation':
        nDumpFrequency = 7
    elif problem_type == 'Viscous_Burgers_Equation':
        nDumpFrequency = 1
    # Note that nDumpFrequency is chosen in such a way that we end up with about 115 to 125 output files for a 
    # simulation time of one time period of the geophysical wave. 
    return nDumpFrequency


# In[34]:

def SpecifyPlotStateVariables(problem_type):
    if problem_type == 'Diffusion_Equation' or problem_type == 'Viscous_Burgers_Equation':
        plotSurfaceElevation = False
    else:
        plotSurfaceElevation = True
    if problem_type == 'Coastal_Kelvin_Wave':
        plotZonalVelocity = False
    else:
        plotZonalVelocity = True
    if (problem_type == 'Equatorial_Kelvin_Wave' or problem_type == 'Diffusion_Equation' 
        or problem_type == 'Viscous_Burgers_Equation'):
        plotMeridionalVelocity = False
    else:
        plotMeridionalVelocity = True
    return plotSurfaceElevation, plotZonalVelocity, plotMeridionalVelocity


# In[35]:

def DetermineFigureAndImageTitles(problem_type):
    if problem_type == 'Coastal_Kelvin_Wave':
        wave_type_title = 'Coastal Kelvin Wave'
        wave_type_figure_title = 'CoastalKelvinWave'
    elif problem_type == 'Inertia_Gravity_Wave':
        wave_type_title = 'Inertia Gravity Wave'
        wave_type_figure_title = 'InertiaGravityWave'
    elif problem_type == 'Planetary_Rossby_Wave':
        wave_type_title = 'Planetary Rossby Wave'
        wave_type_figure_title = 'PlanetaryRossbyWave'
    elif problem_type == 'Topographic_Rossby_Wave':
        wave_type_title = 'Topographic Rossby Wave'
        wave_type_figure_title = 'TopographicRossbyWave'
    elif problem_type == 'Equatorial_Kelvin_Wave':
        wave_type_title = 'Equatorial Kelvin Wave'
        wave_type_figure_title = 'EquatorialKelvinWave' 
    elif problem_type == 'Equatorial_Yanai_Wave':
        wave_type_title = 'Equatorial Yanai Wave'
        wave_type_figure_title = 'EquatorialYanaiWave'   
    elif problem_type == 'Equatorial_Rossby_Wave':
        wave_type_title = 'Equatorial Rossby Wave'
        wave_type_figure_title = 'EquatorialRossbyWave' 
    elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
        wave_type_title = 'Equatorial Inertia Gravity Wave'
        wave_type_figure_title = 'EquatorialInertiaGravityWave' 
    elif problem_type == 'Barotropic_Tide':
        wave_type_title = 'Barotropic Tide'
        wave_type_figure_title = 'BarotropicTide'
    elif problem_type == 'Diffusion_Equation':
        wave_type_title = 'Diffusion Equation'
        wave_type_figure_title = 'DiffusionEquation'        
    elif problem_type == 'Viscous_Burgers_Equation':
        wave_type_title = 'Viscous Burgers Equation'
        wave_type_figure_title = 'ViscousBurgerEquation'         
    return wave_type_title, wave_type_figure_title


# In[36]:

def SpecifyLegendsAndLegendPadsForGeophysicalWaveExactStateVariablesAlongSection(problem_type,phase_speeds,
                                                                                 decay_scales,
                                                                                 amplitudes=[1.0,2.0]):
    cX1 = phase_speeds[0]
    cX2 = phase_speeds[1]
    cY1 = phase_speeds[2]
    cY2 = phase_speeds[3]
    kappa1 = decay_scales[0]
    kappa2 = decay_scales[1]
    if problem_type == 'Coastal_Kelvin_Wave':
        legends = [('First Coastal Kelvin Wave Mode with Surface Elevation Amplitude = %d m,\n' %amplitudes[0]
                    + 'Wavelength = %s%% of Meridional Extent and Southward Phase Speed = %d m/s' 
                    %('100',int(-cY1))),
                   ('Second Coastal Kelvin Wave Mode with Surface Elevation Amplitude = %d m,\n' %amplitudes[1]
                    + ('Wavelength = %s%% of Meridional Extent and Southward Phase Speed = %d m/s' 
                       %('50',int(-cY2)))),
                   'Resultant Coastal Kelvin Wave Solution'] 
        legendpads = [0.5,-0.3725]
    elif problem_type == 'Inertia_Gravity_Wave':
        legends = [('First Inertia Gravity Wave Mode with Surface Elevation Amplitude = %d m,\n' %amplitudes[0]
                    + 'Wavelengths = %s%% of Domain Extents and Phase Speeds = {%.2f,%.2f} m/s' %('100',cX1,cY1)),
                   ('Second Inertia Gravity Wave Mode with Surface Elevation Amplitude = %d m,\n' %amplitudes[1]
                    + 'Wavelengths = %s%% of Domain Extents and Phase Speeds = {%.2f,%.2f} m/s' %('50',cX2,cY2)),
                   'Resultant Inertia Gravity Wave Solution']      
        legendpads = [0.5,-0.3725]
    elif problem_type == 'Planetary_Rossby_Wave':
        legends = [('First Planetary Rossby Wave Mode with Surface Elevation Amplitude = %d m,\n' %amplitudes[0]
                    + 'Wavelengths = %s%% of Domain Extents and Phase Speeds = {%.2e,%.2e} m/s' %('100',cX1,cY1)),
                   ('Second Planetary Rossby Wave Mode with Surface Elevation Amplitude = %d m,\n' %amplitudes[1]
                    + 'Wavelengths = %s%% of Domain Extents and Phase Speeds = {%.2e,%.2e} m/s' %('50',cX2,cY2)),
                   'Resultant Planetary Rossby Wave Solution']   
        legendpads = [0.5,-0.3725]
    elif problem_type == 'Topographic_Rossby_Wave':
        legends = [('First Topographic Rossby Wave Mode with Surface Elevation Amplitude = %d m,\n' %amplitudes[0]
                    + 'Wavelengths = %s%% of Domain Extents and Phase Speeds = {%.2e,%.2e} m/s' %('100',cX1,cY1)),
                   (('Second Topographic Rossby Wave Mode with Surface Elevation Amplitude = %d m,\n' 
                     %amplitudes[1])
                    + 'Wavelengths = %s%% of Domain Extents and Phase Speeds = {%.2e,%.2e} m/s' %('50',cX2,cY2)),
                   'Resultant Topographic Rossby Wave Solution'] 
        legendpads = [0.5,-0.3725]
    elif problem_type == 'Equatorial_Kelvin_Wave':
        legends = [('First Equatorial Kelvin Wave Mode with Surface Elevation Amplitude = %d m,\n' %amplitudes[0]
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %d m/s' %('100',int(cX1))),
                   ('Second Equatorial Kelvin Wave Mode with Surface Elevation Amplitude = %d m,\n' %amplitudes[1]
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %d m/s' %('50',int(cX2))),
                   'Resultant Equatorial Kelvin Wave Solution'] 
        legendpads = [0.5,-0.3725]
    elif problem_type == 'Equatorial_Yanai_Wave':
        legends = [(('First Equatorial Yanai Wave Mode with Meridional Velocity Amplitude = %.2f m,\n' 
                     %amplitudes[0])
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %.2f m/s' %('100',cX1)),
                   (('Second Equatorial Yanai Wave Mode with Meridional Velocity Amplitude = %.2f m,\n' 
                     %amplitudes[1])
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %.2f m/s' %('50',cX2)),
                   'Resultant Equatorial Yanai Wave Solution'] 
        legendpads = [0.5,-0.3725]
    elif problem_type == 'Equatorial_Rossby_Wave':
        legends = [(('First Equatorial Rossby Wave Mode with Meridional Velocity Amplitude = %.2f m,\n' 
                     %amplitudes[0])
                    + 'Wavelength = %s%% of Zonal Extent and Westward Phase Speed = %.2f m/s' %('100',-cX1)),
                   (('Second Equatorial Rossby Wave Mode with Meridional Velocity Amplitude = %.2f m,\n' 
                     %amplitudes[1])
                    + 'Wavelength = %s%% of Zonal Extent and Westward Phase Speed = %.2f m/s' %('50',-cX2)),
                   'Resultant Equatorial Rossby Wave Solution'] 
        legendpads = [0.5,-0.3725]        
    elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
        legends = [(('First Equatorial Inertia Gravity Wave Mode with Meridional Velocity Amplitude = %.2f m,\n' 
                     %amplitudes[0])
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %.2f m/s' %('100',cX1)),
                   (('Second Equatorial Inertia Gravity Wave Mode with Meridional Velocity Amplitude = %.2f m,\n' 
                     %amplitudes[1])
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %.2f m/s' %('50',cX2)),
                   'Resultant Equatorial Inertia Gravity Wave Solution'] 
        legendpads = [0.5,-0.3725]
    elif problem_type == 'Barotropic_Tide':
        legends = [('First Component of First Barotropic Tidal Mode '
                    + 'with Surface Elevation Amplitude = %.2f m,\n' %amplitudes[0]
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %.2f m/s' %('100',-abs(cX1))),
                   ('Second Component of First Barotropic Tidal Mode '
                    + 'with Surface Elevation Amplitude = %.2f m,\n' %amplitudes[0]
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %.2f m/s' %('100',abs(cX1))),
                   'First Barotropic Tidal Mode',
                   ('First Component of Second Barotropic Tidal Mode ' 
                    + 'with Surface Elevation Amplitude = %.2f m,\n' %amplitudes[1]
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %.2f m/s' %('50',-abs(cX2))),
                   ('Second Component of Second Barotropic Tidal Mode '
                    + 'with Surface Elevation Amplitude = %.2f m,\n' %amplitudes[1]
                    + 'Wavelength = %s%% of Zonal Extent and Eastward Phase Speed = %.2f m/s' %('50',abs(cX2))),
                   'Second Barotropic Tidal Mode',
                   'Resultant Barotropic Tidal Solution'] 
        legendpads = [0.5,-0.625] 
    elif problem_type == 'Diffusion_Equation':
        legends = [('Linearly Independent Solution Component with Decay Scale of\n%s' 
                    %PrintDisplayTime(1.0/kappa1,non_integral_seconds=True)),
                   ('Linearly Independent Solution Component with Decay Scale of\n%s' 
                    %PrintDisplayTime(1.0/kappa2,non_integral_seconds=True)),
                   'Resultant Solution'] 
        legendpads = [0.5,-0.375]
    else:
        legends = []
        legendpads = []
    return legends, legendpads


# In[37]:

def SpecifyStateVariableLimitsAlongSection(StateVariableLimits,ToleranceAsPercentage):
    StateVariableDifference = StateVariableLimits[1] - StateVariableLimits[0]
    StateVariableLimitsAlongSection = np.zeros(2)
    StateVariableLimitsAlongSection[0] = (
    StateVariableLimits[0] - 0.5*ToleranceAsPercentage/100.0*StateVariableDifference)
    StateVariableLimitsAlongSection[1] = (
    StateVariableLimits[1] + 0.5*ToleranceAsPercentage/100.0*StateVariableDifference)    
    return StateVariableLimitsAlongSection    


# In[38]:

def WriteStateVariableLimitsToFile(output_directory,StateVariableLimitsLimits,filename):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    filename = filename + '.curve'
    outputfile = open(filename,'w')
    outputfile.write('%.15g %.15g\n' %(StateVariableLimitsLimits[0],StateVariableLimitsLimits[1]))
    outputfile.close()
    os.chdir(cwd)


# In[39]:

def ReadStateVariableLimitsFromFile(output_directory,filename):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    data = [];
    with open(filename,'r') as infile:
        for line in infile:
            data.append(line)
    data = np.loadtxt(data)
    StateVariableLimitsLimits = [data[0],data[1]]
    os.chdir(cwd)
    return StateVariableLimitsLimits


# In[40]:

def testDetermineGeophysicalWaveExactSolutions(mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                               problem_type,problem_is_linear,periodicity,CourantNumber,
                                               printPhaseSpeedOfWaveModes,plotFigures,colormap=plt.cm.seismic,
                                               determineStateVariableLimits=False):
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                       problem_type,problem_is_linear,periodicity=periodicity,
                                       CourantNumber=CourantNumber,useCourantNumberToDetermineTimeStep=True,
                                       printPhaseSpeedOfWaveModes=printPhaseSpeedOfWaveModes)
    myMPAS_O.myNamelist.config_dt = round(myMPAS_O.myNamelist.config_dt)
    dt = myMPAS_O.myNamelist.config_dt 
    lX = myMPAS_O.lX
    lY = myMPAS_O.lY
    cX1 = myMPAS_O.ExactSolutionParameters[4]
    cX2 = myMPAS_O.ExactSolutionParameters[5]
    cY1 = myMPAS_O.ExactSolutionParameters[6]
    cY2 = myMPAS_O.ExactSolutionParameters[7]
    etaHat1 = myMPAS_O.ExactSolutionParameters[10]
    etaHat2 = myMPAS_O.ExactSolutionParameters[11]
    H = myMPAS_O.ExactSolutionParameters[14]
    VelocityScale = myMPAS_O.ExactSolutionParameters[27]
    kappa1 = myMPAS_O.ExactSolutionParameters[31]
    kappa2 = myMPAS_O.ExactSolutionParameters[32]
    s = myMPAS_O.ExactSolutionParameters[35]
    if myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave or problem_type == 'Barotropic_Tide':
        abs_cX = max(abs(cX1),abs(cX2))
        abs_cY = max(abs(cY1),abs(cY2))
        if abs_cX != 0.0:
            TimePeriodOfFastWaveMode = lX/abs_cX 
        else:
            TimePeriodOfFastWaveMode = lY/abs_cY
        # Note that for all two-dimensional dispersive waves, 
        # TimePeriodOfFastWaveMode = lX/abs_cX = lX*kX/abs(omega) = lY*kY/abs(omega) = lY/abs_cY
        # where kX and kY are the zonal and meridional wavenumbers of the fast wave mode with omega being its 
        # angular frequency.
        print('The time period of the fast wave mode is %.6f.' %TimePeriodOfFastWaveMode)
        print('The minimum number of time steps of magnitude %.2f required to constitute the time period is %d.'
              %(dt,int(np.ceil(TimePeriodOfFastWaveMode/dt))))
    elif problem_type == 'Diffusion_Equation':
        kappa = min(kappa1,kappa2) # i.e. kappa = kappa1
        FinalTime = np.log(4.0)/kappa
        print('The time taken by the solution magnitude to drop to %d%% is %.6f.' %(25,FinalTime))
        PrintStatement1 = 'The minimum number of time steps of magnitude %.2f ' %dt
        PrintStatement2 = ('required for the solution magnitude to drop to %d%% is %d.' 
                           %(25,int(np.ceil(FinalTime/dt))))
        print(PrintStatement1 + PrintStatement2)
    elif problem_type == 'Viscous_Burgers_Equation':
        FinalTime = (0.75 - 0.25)*myMPAS_O.lX/s
        print('The time taken by the shock wave to traverse %d%% of the zonal extent is %.6f.' %(50,FinalTime))
        PrintStatement1 = 'The minimum number of time steps of magnitude %.2f ' %dt
        PrintStatement2 = ('required for the shock wave to traverse %d%% of the zonal extent is %d.' 
                           %(50,int(np.ceil(FinalTime/dt))))
        print(PrintStatement1 + PrintStatement2)        
    nTime = SpecifyNumberOfTimeSteps(problem_type)
    nDumpFrequency = SpecifyDumpFrequency(problem_type)
    output_directory_root = 'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/'
    plotExactSurfaceElevation, plotExactZonalVelocity, plotExactMeridionalVelocity = (
    SpecifyPlotStateVariables(problem_type))
    if (problem_type == 'Coastal_Kelvin_Wave' or problem_type == 'Equatorial_Kelvin_Wave'
        or problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Barotropic_Tide'
        or problem_type == 'Viscous_Burgers_Equation'):
        ExactSurfaceElevationLimits, ExactZonalVelocityLimits, ExactMeridionalVelocityLimits = (
        GWESST.DetermineGeophysicalWaveExactSolutionLimits(problem_type,myMPAS_O.ExactSolutionParameters))
    if determineStateVariableLimits:
        print(' ')
        print('The limits of surface elevation are [%.6f,%.6f].' 
              %(ExactSurfaceElevationLimits[0],ExactSurfaceElevationLimits[1]))
        print('The limits of zonal velocity are [%.6f,%.6f].' 
              %(ExactZonalVelocityLimits[0],ExactZonalVelocityLimits[1]))
        print('The limits of meridional velocity are [%.6f,%.6f].' 
              %(ExactMeridionalVelocityLimits[0],ExactMeridionalVelocityLimits[1]))
        return
    wave_type_title, wave_type_figure_title = DetermineFigureAndImageTitles(problem_type)
    if (myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave 
        and not(problem_type == 'Equatorial_Kelvin_Wave')):
        yMaximumAmplitude, HermiteFunctionMaximumAmplitude = (
        GWESST.DetermineHermiteFunctionMaximumAmplitudeWithMeridionalLocation(problem_type))
        amplitudes = VelocityScale*HermiteFunctionMaximumAmplitude*np.array([etaHat1,etaHat2])
        ExactMeridionalVelocityMaximumMagnitude = np.sum(amplitudes)
        Req = myMPAS_O.ExactSolutionParameters[24]
        yLatitude = Req*yMaximumAmplitude
        Latitude = GWESST.DetermineLatitude(yLatitude)
        state_variable_title = 'Meridional Velocity'
        state_variable_figure_title = 'MeridionalVelocity'
        exact_state_variable_title = 'Exact Meridional Velocity'
        exact_state_variable_figure_title = 'ExactMeridionalVelocity'    
    elif problem_type == 'Diffusion_Equation' or problem_type == 'Viscous_Burgers_Equation':        
        amplitudes = [0.0,0.0]
        if problem_type == 'Diffusion_Equation':
            yLatitude = 1.0/8.0*myMPAS_O.lY
            Latitude = GWESST.DetermineLatitude(yLatitude)
        else: # if problem_type == 'Viscous_Burgers_Equation':
            yLatitude = 0.0
            Latitude = 0.0
        state_variable_title = 'Zonal Velocity'
        state_variable_figure_title = 'ZonalVelocity'
        exact_state_variable_title = 'Exact Zonal Velocity'
        exact_state_variable_figure_title = 'ExactZonalVelocity'
    else:
        if problem_type == 'Coastal_Kelvin_Wave' or problem_type == 'Equatorial_Kelvin_Wave':
            amplitudes = H*np.array([etaHat1,etaHat2])
        else:
            amplitudes = np.array([etaHat1,etaHat2])       
        if problem_type == 'Planetary_Rossby_Wave' or problem_type == 'Topographic_Rossby_Wave':
            ExactSurfaceElevationMaximumMagnitude = np.sum(amplitudes)
        yLatitude = 0.0
        Latitude = 0.0
        state_variable_title = 'Surface Elevation'
        state_variable_figure_title = 'SurfaceElevation'
        exact_state_variable_title = 'Exact Surface Elevation'
        exact_state_variable_figure_title = 'ExactSurfaceElevation'
    nPlotAlongSection = 150
    rPlotAlongSection, xPlotAlongSection, yPlotAlongSection = (
    GWESST.DetermineCoordinatesAlongSection(problem_type,myMPAS_O.lX,myMPAS_O.lY,nPlotAlongSection,yLatitude))
    if problem_type == 'Coastal_Kelvin_Wave':
        xLabel = 'Distance Along Coastline from South to North (km)'
    elif (problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Planetary_Rossby_Wave' 
          or problem_type == 'Topographic_Rossby_Wave'):
        xLabel = 'Distance Along Domain Diagonal from South-West to North-East (km)'    
    elif myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave:
        if problem_type == 'Equatorial_Kelvin_Wave' or problem_type == 'Equatorial_Yanai_Wave':
            xLabel = 'Distance Along the Equator (km)'
        else:
            xLabel = 'Distance Along %.2f\u00b0 N Latitude (km)' %Latitude
    elif problem_type == 'Barotropic_Tide' or problem_type == 'Viscous_Burgers_Equation':
        xLabel = 'Distance Along Any Zonal Section (km)'
    elif problem_type == 'Diffusion_Equation':
        xLabel = 'Distance Along %.2f\u00b0 N Latitude (km)' %Latitude    
    if (myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave 
        and not(problem_type == 'Equatorial_Kelvin_Wave')):
        yLabel = 'Meridional Velocity (m/s)'
    elif problem_type == 'Diffusion_Equation' or problem_type == 'Viscous_Burgers_Equation':
        yLabel = 'Zonal Velocity (m/s)'
    else:
        yLabel = 'Surface Elevation (m)'
    phase_speeds = [cX1,cX2,cY1,cY2]
    decay_scales = [kappa1,kappa2]
    legends, legend_pads = (
    SpecifyLegendsAndLegendPadsForGeophysicalWaveExactStateVariablesAlongSection(problem_type,phase_speeds,
                                                                                 decay_scales,amplitudes))    
    nCounters = 2
    for iCounter in range(0,nCounters):
        for iTime in range(0,nTime):
            myMPAS_O.iTime = iTime
            myMPAS_O.time = float(iTime)*dt
            if np.mod(iTime,nDumpFrequency) == 0.0:
                if iCounter == 0:
                    [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocities, 
                     GeophysicalWaveExactMeridionalVelocities] = (
                    DetermineGeophysicalWaveExactSolutionsAtCellCenters(myMPAS_O))
                    GeophysicalWaveExactStateVariablesAlongSection = (
                    GWESST.ComputeGeophysicalWaveExactStateVariablesAlongSection(
                    problem_type,myMPAS_O.ExactSolutionParameters,xPlotAlongSection,yPlotAlongSection,
                    myMPAS_O.time))
                    if plotFigures:                    
                        if not(problem_type == 'Coastal_Kelvin_Wave' or problem_type == 'Equatorial_Kelvin_Wave'
                               or problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Barotropic_Tide'
                               or problem_type == 'Viscous_Burgers_Equation'):
                            if iTime == 0:
                                if not(problem_type == 'Planetary_Rossby_Wave' 
                                       or problem_type == 'Topographic_Rossby_Wave'):
                                    ExactSurfaceElevationMinimum = min(GeophysicalWaveExactSurfaceElevations)
                                    ExactSurfaceElevationMaximum = max(GeophysicalWaveExactSurfaceElevations)
                                ExactZonalVelocityMinimum = min(GeophysicalWaveExactZonalVelocities)
                                ExactZonalVelocityMaximum = max(GeophysicalWaveExactZonalVelocities)
                                ExactMeridionalVelocityMinimum = min(GeophysicalWaveExactMeridionalVelocities)
                                ExactMeridionalVelocityMaximum = max(GeophysicalWaveExactMeridionalVelocities)
                            else:
                                if not(problem_type == 'Planetary_Rossby_Wave' 
                                       or problem_type == 'Topographic_Rossby_Wave'):
                                    ExactSurfaceElevationMinimum = min(ExactSurfaceElevationMinimum,
                                                                       min(GeophysicalWaveExactSurfaceElevations))
                                    ExactSurfaceElevationMaximum = max(ExactSurfaceElevationMaximum,
                                                                       max(GeophysicalWaveExactSurfaceElevations))
                                ExactZonalVelocityMinimum = min(ExactZonalVelocityMinimum,
                                                                min(GeophysicalWaveExactZonalVelocities))
                                ExactZonalVelocityMaximum = max(ExactZonalVelocityMaximum,
                                                                max(GeophysicalWaveExactZonalVelocities))
                                ExactMeridionalVelocityMinimum = (
                                min(ExactMeridionalVelocityMinimum,min(GeophysicalWaveExactMeridionalVelocities)))
                                ExactMeridionalVelocityMaximum = (
                                max(ExactMeridionalVelocityMaximum,max(GeophysicalWaveExactMeridionalVelocities)))
                            if iTime == nTime - 1:                        
                                if not(problem_type == 'Planetary_Rossby_Wave' 
                                       or problem_type == 'Topographic_Rossby_Wave'):
                                    ExactSurfaceElevationMaximumMagnitude = max(abs(ExactSurfaceElevationMinimum),
                                                                                ExactSurfaceElevationMaximum)
                                ExactZonalVelocityMaximumMagnitude = max(abs(ExactZonalVelocityMinimum),
                                                                         ExactZonalVelocityMaximum) 
                                if not(myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave 
                                       and not(problem_type == 'Equatorial_Kelvin_Wave')):
                                    ExactMeridionalVelocityMaximumMagnitude = (
                                    max(abs(ExactMeridionalVelocityMinimum),ExactMeridionalVelocityMaximum))
                                ExactSurfaceElevationLimits = [-ExactSurfaceElevationMaximumMagnitude,
                                                               ExactSurfaceElevationMaximumMagnitude]
                                ExactZonalVelocityLimits = [-ExactZonalVelocityMaximumMagnitude,
                                                            ExactZonalVelocityMaximumMagnitude]
                                ExactMeridionalVelocityLimits = [-ExactMeridionalVelocityMaximumMagnitude,
                                                                 ExactMeridionalVelocityMaximumMagnitude]
                        if plotExactSurfaceElevation:                        
                            output_directory = (
                            output_directory_root + wave_type_figure_title + '_ExactSurfaceElevation')
                            FileName = wave_type_figure_title + '_ExactSurfaceElevation_' + '%3.3d' %iTime
                            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          GeophysicalWaveExactSurfaceElevations,FileName)
                            if iTime == nTime - 1:
                                FileName = wave_type_figure_title + '_ExactSurfaceElevationLimits'
                                WriteStateVariableLimitsToFile(output_directory,ExactSurfaceElevationLimits,
                                                               FileName)
                        if plotExactZonalVelocity:
                            output_directory = (
                            output_directory_root + wave_type_figure_title + '_ExactZonalVelocity')
                            FileName = wave_type_figure_title + '_ExactZonalVelocity_' + '%3.3d' %iTime
                            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          GeophysicalWaveExactZonalVelocities,FileName)
                            if iTime == nTime - 1:
                                FileName = wave_type_figure_title + '_ExactZonalVelocityLimits'
                                WriteStateVariableLimitsToFile(output_directory,ExactZonalVelocityLimits,FileName)
                        if plotExactMeridionalVelocity:
                            output_directory = (
                            output_directory_root + wave_type_figure_title + '_ExactMeridionalVelocity')
                            FileName = wave_type_figure_title + '_ExactMeridionalVelocity_' + '%3.3d' %iTime
                            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          GeophysicalWaveExactMeridionalVelocities,FileName)
                            if iTime == nTime - 1:
                                FileName = wave_type_figure_title + '_ExactMeridionalVelocityLimits'
                                WriteStateVariableLimitsToFile(output_directory,ExactMeridionalVelocityLimits,
                                                               FileName)
                        output_directory = (output_directory_root + wave_type_figure_title + '_' 
                                            + exact_state_variable_figure_title)
                        FileName = (wave_type_figure_title + '_' + state_variable_figure_title 
                                    + 'AlongSection_' + '%3.3d' %iTime)            
                        GWESST.WriteGeophysicalWaveExactStateVariablesAlongSectionToFile(
                        problem_type,output_directory,rPlotAlongSection,
                        GeophysicalWaveExactStateVariablesAlongSection,FileName)
                else: # if iCounter == 1:                    
                    if plotFigures:   
                        DisplayTime = PrintDisplayTime(myMPAS_O.time)   
                        xlabel = 'Zonal Distance (km)'
                        ylabel = 'Meridional Distance (km)'
                        if plotExactSurfaceElevation:
                            output_directory = (
                            output_directory_root + wave_type_figure_title + '_ExactSurfaceElevation')    
                            FileName = wave_type_figure_title + '_ExactSurfaceElevationLimits'
                            ExactSurfaceElevationLimits = ReadStateVariableLimitsFromFile(output_directory,
                                                                                          FileName+'.curve')
                            FileName = wave_type_figure_title + '_ExactSurfaceElevation_' + '%3.3d' %iTime
                            GeophysicalWaveExactSurfaceElevations = (
                            CR.ReadTecPlot2DUnstructured(output_directory,FileName+'.tec',
                                                         returnIndependentVariables=False))
                            Title = wave_type_title + ': Exact Surface Elevation after\n' + DisplayTime
                            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                            myMPAS_O,output_directory,GeophysicalWaveExactSurfaceElevations,300,True,
                            ExactSurfaceElevationLimits,6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],
                            [15.0,15.0],Title,20.0,True,FileName,False,fig_size=[9.25,9.25],
                            cbarlabelformat='%.5f')                               
                        if plotExactZonalVelocity:                        
                            output_directory = (
                            output_directory_root + wave_type_figure_title + '_ExactZonalVelocity')
                            FileName = wave_type_figure_title + '_ExactZonalVelocityLimits'
                            ExactZonalVelocityLimits = ReadStateVariableLimitsFromFile(output_directory,
                                                                                       FileName+'.curve')
                            FileName = wave_type_figure_title + '_ExactZonalVelocity_' + '%3.3d' %iTime
                            GeophysicalWaveExactZonalVelocities = (
                            CR.ReadTecPlot2DUnstructured(output_directory,FileName+'.tec',
                                                         returnIndependentVariables=False))
                            Title = wave_type_title + ': Exact Zonal Velocity after\n' + DisplayTime
                            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                            myMPAS_O,output_directory,GeophysicalWaveExactZonalVelocities,300,True,
                            ExactZonalVelocityLimits,6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],
                            [15.0,15.0],Title,20.0,True,FileName,False,cbarlabelformat='%.5f')         
                        if plotExactMeridionalVelocity:                        
                            output_directory = (
                            output_directory_root + wave_type_figure_title + '_ExactMeridionalVelocity')
                            FileName = wave_type_figure_title + '_ExactMeridionalVelocityLimits'
                            ExactMeridionalVelocityLimits = ReadStateVariableLimitsFromFile(output_directory,
                                                                                            FileName+'.curve')
                            FileName = wave_type_figure_title + '_ExactMeridionalVelocity_' + '%3.3d' %iTime
                            GeophysicalWaveExactMeridionalVelocities = (
                            CR.ReadTecPlot2DUnstructured(output_directory,FileName+'.tec',
                                                         returnIndependentVariables=False))
                            Title = wave_type_title + ': Exact Meridional Velocity after\n' + DisplayTime
                            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                            myMPAS_O,output_directory,GeophysicalWaveExactMeridionalVelocities,300,True,
                            ExactMeridionalVelocityLimits,6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],
                            [10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,False,cbarlabelformat='%.5f') 
                        output_directory = (output_directory_root + wave_type_figure_title + '_' 
                                            + exact_state_variable_figure_title)
                        FileName = (wave_type_figure_title + '_' + state_variable_figure_title 
                                    + 'AlongSection_' + '%3.3d' %iTime)
                        GeophysicalWaveExactStateVariablesAlongSection = (
                        GWESST.ReadGeophysicalWaveExactStateVariablesAlongSectionFromFile(
                        problem_type,output_directory,FileName+'.curve'))
                        ToleranceAsPercentage = 12.0
                        if (myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave 
                            and not(problem_type == 'Equatorial_Kelvin_Wave')):
                            StateVariableLimitsAlongSection = (
                            SpecifyStateVariableLimitsAlongSection(ExactMeridionalVelocityLimits,
                                                                   ToleranceAsPercentage))   
                        elif problem_type == 'Diffusion_Equation' or problem_type == 'Viscous_Burgers_Equation':
                            StateVariableLimitsAlongSection = (
                            SpecifyStateVariableLimitsAlongSection(ExactZonalVelocityLimits,
                                                                   ToleranceAsPercentage))
                        else:
                            StateVariableLimitsAlongSection = (
                            SpecifyStateVariableLimitsAlongSection(ExactSurfaceElevationLimits,
                                                                   ToleranceAsPercentage))
                        Title = wave_type_title + ': ' + state_variable_title + ' after\n' + DisplayTime
                        if (myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave
                            or problem_type == 'Diffusion_Equation'):
                            linewidths = [2.5,3.0,0.5]
                            linestyles = ['--',':','-']
                            colors = ['k','k','k']
                        elif problem_type == 'Barotropic_Tide':
                            linewidths = [2.5,3.0,1.0,2.5,3.0,1.0,1.0]
                            linestyles = ['--',':','-','--',':','-','-']
                            colors = ['r','r','r','b','b','b','gold']
                        else:
                            linewidths = [2.0]
                            linestyles = ['-']
                            colors = ['k']
                        GWESST.PlotGeophysicalWaveExactStateVariablesAlongSectionSaveAsPNG(
                        problem_type,output_directory,rPlotAlongSection,
                        GeophysicalWaveExactStateVariablesAlongSection,StateVariableLimitsAlongSection,
                        linewidths,linestyles,colors,[xLabel,yLabel],[20.0,20.0],[10.0,10.0],[15.0,15.0],legends,
                        15.0,'lower center',Title,20.0,True,FileName,False,fig_size=[9.25,9.25],
                        legendWithinBox=False,legendpads=legend_pads,titlepad=1.035,
                        problem_type_Equatorial_Wave=myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave)


# In[41]:

def do_testDetermineGeophysicalWaveExactSolutions(problem_type,determineStateVariableLimits=False):    
    if problem_type == 'Coastal_Kelvin_Wave':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_x'
        dt = 180.0
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif problem_type == 'Inertia_Gravity_Wave':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/InertiaGravityWaveMesh'
        base_mesh_file_name = 'base_mesh.nc'
        periodicity = 'Periodic'
        dt = 96.0 
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif problem_type == 'Planetary_Rossby_Wave' or problem_type == 'Topographic_Rossby_Wave':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/PlanetaryRossbyWaveMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_y'
        dt = 195000.0
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif (problem_type == 'Equatorial_Kelvin_Wave' or problem_type == 'Equatorial_Yanai_Wave' 
          or problem_type == 'Equatorial_Rossby_Wave' or problem_type == 'Equatorial_Inertia_Gravity_Wave'):
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/EquatorialWaveMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_y'        
        if problem_type == 'Equatorial_Kelvin_Wave':
            dt = 750.00 
        elif problem_type == 'Equatorial_Yanai_Wave':
            dt = 390.0
        elif problem_type == 'Equatorial_Rossby_Wave':
            dt = 2700.0
        elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
            dt = 420.0
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif problem_type == 'Barotropic_Tide':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/BarotropicTideMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_xy'  
        dt = 10.0
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif problem_type == 'Diffusion_Equation':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/BarotropicTideMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_xy'
        dt = 2260.0
        printPhaseSpeedOfWaveModes = False
        problem_is_linear = True
    elif problem_type == 'Viscous_Burgers_Equation':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/BarotropicTideMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_xy'  
        dt = 2100.0
        printPhaseSpeedOfWaveModes = False  
        problem_is_linear = False
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    CourantNumber = DetermineCourantNumberForGivenTimeStep(mesh_directory,base_mesh_file_name,mesh_file_name,
                                                           mesh_type,problem_type,problem_is_linear,periodicity,dt,
                                                           printCourantNumber=False)
    plotFigures = True
    colormap = plt.cm.seismic
    testDetermineGeophysicalWaveExactSolutions(mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                               problem_type,problem_is_linear,periodicity,CourantNumber,
                                               printPhaseSpeedOfWaveModes,plotFigures,colormap,
                                               determineStateVariableLimits)


# In[42]:

do_testDetermineGeophysicalWaveExactSolutions_CoastalKelvinWave_determineStateVariableLimits = False
if do_testDetermineGeophysicalWaveExactSolutions_CoastalKelvinWave_determineStateVariableLimits:
    problem_type = 'Coastal_Kelvin_Wave'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type,determineStateVariableLimits=True)


# In[43]:

do_testDetermineGeophysicalWaveExactSolutions_CoastalKelvinWave = False
if do_testDetermineGeophysicalWaveExactSolutions_CoastalKelvinWave:
    problem_type = 'Coastal_Kelvin_Wave'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[44]:

do_testDetermineGeophysicalWaveExactSolutions_InertiaGravityWave = False
if do_testDetermineGeophysicalWaveExactSolutions_InertiaGravityWave:
    problem_type = 'Inertia_Gravity_Wave'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[45]:

do_testDetermineGeophysicalWaveExactSolutions_PlanetaryRossbyWave = False
if do_testDetermineGeophysicalWaveExactSolutions_PlanetaryRossbyWave:
    problem_type = 'Planetary_Rossby_Wave'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[46]:

do_testDetermineGeophysicalWaveExactSolutions_TopographicRossbyWave = False 
if do_testDetermineGeophysicalWaveExactSolutions_TopographicRossbyWave:
    problem_type = 'Topographic_Rossby_Wave'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[47]:

do_testDetermineGeophysicalWaveExactSolutions_EquatorialKelvinWave = False
if do_testDetermineGeophysicalWaveExactSolutions_EquatorialKelvinWave:
    problem_type = 'Equatorial_Kelvin_Wave'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[48]:

do_testDetermineGeophysicalWaveExactSolutions_EquatorialYanaiWave = False
if do_testDetermineGeophysicalWaveExactSolutions_EquatorialYanaiWave:
    problem_type = 'Equatorial_Yanai_Wave'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[49]:

do_testDetermineGeophysicalWaveExactSolutions_EquatorialRossbyWave = False 
if do_testDetermineGeophysicalWaveExactSolutions_EquatorialRossbyWave:
    problem_type = 'Equatorial_Rossby_Wave'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[50]:

do_testDetermineGeophysicalWaveExactSolutions_EquatorialInertiaGravityWave = False 
if do_testDetermineGeophysicalWaveExactSolutions_EquatorialInertiaGravityWave:
    problem_type = 'Equatorial_Inertia_Gravity_Wave'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[51]:

do_testDetermineGeophysicalWaveExactSolutions_BarotropicTide = False 
if do_testDetermineGeophysicalWaveExactSolutions_BarotropicTide:
    problem_type = 'Barotropic_Tide'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[52]:

do_testDetermineGeophysicalWaveExactSolutions_DiffusionEquation = False 
if do_testDetermineGeophysicalWaveExactSolutions_DiffusionEquation:
    problem_type = 'Diffusion_Equation'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[53]:

do_testDetermineGeophysicalWaveExactSolutions_ViscousBurgersEquation = False 
if do_testDetermineGeophysicalWaveExactSolutions_ViscousBurgersEquation:
    problem_type = 'Viscous_Burgers_Equation'
    do_testDetermineGeophysicalWaveExactSolutions(problem_type)


# In[54]:

def DetermineTimeIntegratorShortForm(
time_integrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type='FourthOrderAccurate_MaximumStabilityRange',
Generalized_FB_with_AB2_AM3_Step_Type='FourthOrderAccurate',
Generalized_FB_with_AB3_AM4_Step_Type='FourthOrderAccurate_MaximumStabilityRange'):
    if time_integrator == 'Forward_Euler':
        time_integrator_short_form = 'FE'
    elif time_integrator == 'Forward_Backward':
        time_integrator_short_form = 'FB'
    elif time_integrator == 'Explicit_Midpoint_Method':
        time_integrator_short_form = 'EMM'
    elif time_integrator == 'Williamson_Low_Storage_Runge_Kutta_Third_Order':
        time_integrator_short_form = 'WLSRK3'
    elif time_integrator == 'Low_Storage_Runge_Kutta_Fourth_Order':
        time_integrator_short_form = 'LSRK4'    
    elif time_integrator == 'Adams_Bashforth_Second_Order':
        time_integrator_short_form = 'AB2'
    elif time_integrator == 'Adams_Bashforth_Third_Order':
        time_integrator_short_form = 'AB3'
    elif time_integrator == 'Adams_Bashforth_Fourth_Order':
        time_integrator_short_form = 'AB4'
    elif time_integrator == 'Leapfrog_Trapezoidal':
        time_integrator_short_form = 'LFTR_Odr2'
    elif time_integrator == 'LF_TR_and_LF_AM3_with_FB_Feedback':
        if LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'SecondOrderAccurate_LF_TR':
            time_integrator_short_form = 'LFTR_Odr2'
        elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'ThirdOrderAccurate_LF_AM3':
            time_integrator_short_form = 'LFAM_Odr3'        
        elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'ThirdOrderAccurate_MaximumStabilityRange':
            time_integrator_short_form = 'LFAM_Odr3_MaxStabRng'
        elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'FourthOrderAccurate_MinimumTruncationError':
            time_integrator_short_form = 'LFAM_Odr4_MinTruncErr'
        elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'FourthOrderAccurate_MaximumStabilityRange':
            time_integrator_short_form = 'LFAM_Odr4_MaxStabRng'
    elif time_integrator == 'Forward_Backward_with_RK2_Feedback':
        time_integrator_short_form = 'FB_RK2Fdbk'
    elif time_integrator == 'Generalized_FB_with_AB2_AM3_Step':
        if Generalized_FB_with_AB2_AM3_Step_Type == 'ThirdOrderAccurate_WideStabilityRange':
            time_integrator_short_form = 'GenFB_AB2AM3_Ordr3_WideStabRng'
        elif (Generalized_FB_with_AB2_AM3_Step_Type 
              == 'ThirdOrderAccurate_WeakAsymptoticInstabilityOfPhysicalModes'):
            time_integrator_short_form = 'GenFB_AB2AM3_Ordr3_WeakAsympInstab'
        elif Generalized_FB_with_AB2_AM3_Step_Type == 'FourthOrderAccurate':
            time_integrator_short_form = 'GenFB_AB2AM3_Ordr4' 
    elif time_integrator == 'Generalized_FB_with_AB3_AM4_Step':
        if Generalized_FB_with_AB3_AM4_Step_Type == 'SecondOrderAccurate_OptimumChoice_ROMS':
            time_integrator_short_form = 'GenFB_AB3AM4_Ordr2_Optm_ROMS'
        elif Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_AB3_AM4':
            time_integrator_short_form = 'GenFB_AB3AM4_Ordr3'
        elif Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_MaximumStabilityRange':
            time_integrator_short_form = 'GenFB_AB3AM4_Ordr3_MaxStabRng'
        elif Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_OptimumChoice':
            time_integrator_short_form = 'GenFB_AB3AM4_Ordr3_Optm'
        elif Generalized_FB_with_AB3_AM4_Step_Type == 'FourthOrderAccurate_MaximumStabilityRange':
            time_integrator_short_form = 'GenFB_AB3AM4_Ordr4_MaxStabRng'
    return time_integrator_short_form


# In[55]:

def Main(mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,problem_is_linear,periodicity,
         CourantNumber,time_integrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
         Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,printPhaseSpeedOfWaveModes,
         plotFigures,CheckNumericalSurfaceElevationError=False,colormap=plt.cm.seismic):
    myMPAS_O = (
    MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,
                            problem_is_linear,periodicity=periodicity,CourantNumber=CourantNumber,
                            useCourantNumberToDetermineTimeStep=True,time_integrator=time_integrator,
                            LF_TR_and_LF_AM3_with_FB_Feedback_Type=LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                            Generalized_FB_with_AB2_AM3_Step_Type=Generalized_FB_with_AB2_AM3_Step_Type,
                            Generalized_FB_with_AB3_AM4_Step_Type=Generalized_FB_with_AB3_AM4_Step_Type,
                            printPhaseSpeedOfWaveModes=printPhaseSpeedOfWaveModes,
                            specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False))
    myMPAS_O.myNamelist.config_dt = round(myMPAS_O.myNamelist.config_dt)
    dt = myMPAS_O.myNamelist.config_dt 
    lX = myMPAS_O.lX
    lY = myMPAS_O.lY
    cX1 = myMPAS_O.ExactSolutionParameters[4]
    cX2 = myMPAS_O.ExactSolutionParameters[5]
    cY1 = myMPAS_O.ExactSolutionParameters[6]
    cY2 = myMPAS_O.ExactSolutionParameters[7]
    etaHat1 = myMPAS_O.ExactSolutionParameters[10]
    etaHat2 = myMPAS_O.ExactSolutionParameters[11]
    VelocityScale = myMPAS_O.ExactSolutionParameters[27]
    kappa1 = myMPAS_O.ExactSolutionParameters[31]
    kappa2 = myMPAS_O.ExactSolutionParameters[32]
    s = myMPAS_O.ExactSolutionParameters[35]
    if myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave or problem_type == 'Barotropic_Tide':
        abs_cX = max(abs(cX1),abs(cX2))
        abs_cY = max(abs(cY1),abs(cY2))
        if abs_cX != 0.0:
            TimePeriodOfFastWaveMode = lX/abs_cX 
        else:
            TimePeriodOfFastWaveMode = lY/abs_cY
        # Note that for all two-dimensional dispersive waves, 
        # TimePeriodOfFastWaveMode = lX/abs_cX = lX*kX/abs(omega) = lY*kY/abs(omega) = lY/abs_cY
        # where kX and kY are the zonal and meridional wavenumbers of the fast wave mode with omega being its 
        # angular frequency.
        print('The time period of the fast wave mode is %.6f.' %TimePeriodOfFastWaveMode)
        print('The minimum number of time steps of magnitude %.2f required to constitute the time period is %d.'
              %(dt,int(np.ceil(TimePeriodOfFastWaveMode/dt))))
    elif problem_type == 'Diffusion_Equation':
        kappa = min(kappa1,kappa2) # i.e. kappa = kappa1
        FinalTime = np.log(4.0)/kappa
        print('The time taken by the solution magnitude to drop to %d%% is %.6f.' %(25,FinalTime))
        PrintStatement1 = 'The minimum number of time steps of magnitude %.2f ' %dt
        PrintStatement2 = ('required for the solution magnitude to drop to %d%% is %d.' 
                           %(25,int(np.ceil(FinalTime/dt))))
        print(PrintStatement1 + PrintStatement2)
    elif problem_type == 'Viscous_Burgers_Equation':
        FinalTime = (0.75 - 0.25)*myMPAS_O.lX/s
        print('The time taken by the shock wave to traverse %d%% of the zonal extent is %.6f.' %(50,FinalTime))
        PrintStatement1 = 'The minimum number of time steps of magnitude %.2f ' %dt
        PrintStatement2 = ('required for the shock wave to traverse %d%% of the zonal extent is %d.' 
                           %(50,int(np.ceil(FinalTime/dt))))
        print(PrintStatement1 + PrintStatement2)  
    nTime = SpecifyNumberOfTimeSteps(problem_type)
    nDumpFrequency = SpecifyDumpFrequency(problem_type)
    output_directory_root = 'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/'
    plotSurfaceElevation, plotZonalVelocity, plotMeridionalVelocity = SpecifyPlotStateVariables(problem_type)
    plotSurfaceElevationError, plotZonalVelocityError, plotMeridionalVelocityError = (
    SpecifyPlotStateVariables(problem_type))
    if (problem_type == 'Coastal_Kelvin_Wave' or problem_type == 'Equatorial_Kelvin_Wave'
        or problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Barotropic_Tide'
        or problem_type == 'Viscous_Burgers_Equation'):
        ExactSurfaceElevationLimits, ExactZonalVelocityLimits, ExactMeridionalVelocityLimits = (
        GWESST.DetermineGeophysicalWaveExactSolutionLimits(problem_type,myMPAS_O.ExactSolutionParameters)) 
    if problem_type == 'Planetary_Rossby_Wave' or problem_type == 'Topographic_Rossby_Wave':
        ExactSurfaceElevationMaximumMagnitude = etaHat1 + etaHat2 
    if (myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave 
        and not(problem_type == 'Equatorial_Kelvin_Wave')):
        HermiteFunctionMaximumAmplitude = (
        GWESST.DetermineHermiteFunctionMaximumAmplitudeWithMeridionalLocation(problem_type,
                                                                              returnMeridionalLocation=False))
        ExactMeridionalVelocityMaximumMagnitude = (
        VelocityScale*HermiteFunctionMaximumAmplitude*(etaHat1 + etaHat2))   
    wave_type_title, wave_type_figure_title = DetermineFigureAndImageTitles(problem_type) 
    time_integrator_short_form = (
    DetermineTimeIntegratorShortForm(time_integrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                     Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type))
    MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    compute_these_variables = np.zeros(8,dtype=bool)
    compute_these_variables[4] = True # compute_these_variables[4] = compute_tangentialVelocity
    GeophysicalWaveNumericalZonalVelocities = np.zeros(myMPAS_O.nEdges)
    GeophysicalWaveNumericalMeridionalVelocities = np.zeros(myMPAS_O.nEdges)
    GeophysicalWaveExactVelocities = np.zeros((myMPAS_O.nEdges,2))
    nCounters = 2
    for iCounter in range(0,nCounters):
        for iTime in range(0,nTime):
            myMPAS_O.iTime = iTime
            myMPAS_O.time = float(iTime)*dt
            if iCounter == 0:
                if np.mod(iTime,nDumpFrequency) == 0.0:
                    [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocitiesAtCellCenters, 
                     GeophysicalWaveExactMeridionalVelocitiesAtCellCenters] = (
                    DetermineGeophysicalWaveExactSolutionsAtCellCenters(myMPAS_O))
                    [GeophysicalWaveExactZonalVelocities, GeophysicalWaveExactMeridionalVelocities] = (
                    DetermineGeophysicalWaveExactSolutions(myMPAS_O,
                                                           DetermineGeophysicalWaveExactSurfaceElevations=False))
                    GeophysicalWaveExactVelocities[:,0] = GeophysicalWaveExactZonalVelocities[:] 
                    GeophysicalWaveExactVelocities[:,1] = GeophysicalWaveExactMeridionalVelocities[:] 
                    GeophysicalWaveExactNormalVelocities, GeophysicalWaveExactTangentialVelocities = (
                    ComputeNormalAndTangentialComponentsAtEdge(GeophysicalWaveExactVelocities,myMPAS_O.angleEdge,
                                                               'both'))                    
                    if iTime == 0.0: # Specify initial conditions
                        myMPAS_O.sshCurrent[:] = GeophysicalWaveExactSurfaceElevations[:]
                        myMPAS_O.normalVelocityCurrent[:,0] = GeophysicalWaveExactNormalVelocities      
                    MPAS_O_Shared.ocn_diagnostic_solve(myMPAS_O,myMPAS_O.normalVelocityCurrent,
                                                       myMPAS_O.sshCurrent,compute_these_variables)
                    for iEdge in range(0,myMPAS_O.nEdges):
                        if myMPAS_O.boundaryEdge[iEdge] == 1.0: 
                        # i.e. if the edge is along a non-periodic boundary
                            myMPAS_O.tangentialVelocity[iEdge,0] = GeophysicalWaveExactTangentialVelocities[iEdge]
                            GeophysicalWaveNumericalZonalVelocities[iEdge] = (
                            GeophysicalWaveExactZonalVelocities[iEdge])
                            GeophysicalWaveNumericalMeridionalVelocities[iEdge] = (
                            GeophysicalWaveExactMeridionalVelocities[iEdge])
                        else:
                            GeophysicalWaveNumericalZonalVelocities[iEdge] = (
                            (myMPAS_O.normalVelocityCurrent[iEdge,0]*np.cos(myMPAS_O.angleEdge[iEdge])
                             - myMPAS_O.tangentialVelocity[iEdge,0]*np.sin(myMPAS_O.angleEdge[iEdge])))
                            GeophysicalWaveNumericalMeridionalVelocities[iEdge] = (
                            (myMPAS_O.normalVelocityCurrent[iEdge,0]*np.sin(myMPAS_O.angleEdge[iEdge])
                             + myMPAS_O.tangentialVelocity[iEdge,0]*np.cos(myMPAS_O.angleEdge[iEdge])))
                    GeophysicalWaveNumericalZonalVelocitiesAtCellCenters = (
                    MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                    myMPAS_O,GeophysicalWaveNumericalZonalVelocities))
                    GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters = (
                    MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                    myMPAS_O,GeophysicalWaveNumericalMeridionalVelocities))
                    GeophysicalWaveSurfaceElevationError = (
                    myMPAS_O.sshCurrent - GeophysicalWaveExactSurfaceElevations)
                    GeophysicalWaveZonalVelocityErrorAtCellCenters = (
                    (GeophysicalWaveExactZonalVelocitiesAtCellCenters 
                     - GeophysicalWaveNumericalZonalVelocitiesAtCellCenters))
                    GeophysicalWaveMeridionalVelocityErrorAtCellCenters = (
                    (GeophysicalWaveExactMeridionalVelocitiesAtCellCenters
                     - GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters))
                    if plotFigures:
                        if not(problem_type == 'Coastal_Kelvin_Wave' or problem_type == 'Equatorial_Kelvin_Wave'
                               or problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Barotropic_Tide'
                               or problem_type == 'Viscous_Burgers_Equation'):
                            if iTime == 0:
                                if not(problem_type == 'Planetary_Rossby_Wave' 
                                       or problem_type == 'Topographic_Rossby_Wave'):
                                    ExactSurfaceElevationMinimum = min(GeophysicalWaveExactSurfaceElevations)
                                    ExactSurfaceElevationMaximum = max(GeophysicalWaveExactSurfaceElevations)
                                ExactZonalVelocityMinimum = min(GeophysicalWaveExactZonalVelocitiesAtCellCenters)
                                ExactZonalVelocityMaximum = max(GeophysicalWaveExactZonalVelocitiesAtCellCenters)
                                ExactMeridionalVelocityMinimum = (
                                min(GeophysicalWaveExactMeridionalVelocitiesAtCellCenters))
                                ExactMeridionalVelocityMaximum = (
                                max(GeophysicalWaveExactMeridionalVelocitiesAtCellCenters))    
                            else:
                                if not(problem_type == 'Planetary_Rossby_Wave' 
                                       or problem_type == 'Topographic_Rossby_Wave'):
                                    ExactSurfaceElevationMinimum = min(ExactSurfaceElevationMinimum,
                                                                       min(GeophysicalWaveExactSurfaceElevations))
                                    ExactSurfaceElevationMaximum = max(ExactSurfaceElevationMaximum,
                                                                       max(GeophysicalWaveExactSurfaceElevations))
                                ExactZonalVelocityMinimum = (
                                min(ExactZonalVelocityMinimum,
                                min(GeophysicalWaveExactZonalVelocitiesAtCellCenters)))
                                ExactZonalVelocityMaximum = (
                                max(ExactZonalVelocityMaximum,
                                max(GeophysicalWaveExactZonalVelocitiesAtCellCenters)))
                                ExactMeridionalVelocityMinimum = (
                                min(ExactMeridionalVelocityMinimum,
                                min(GeophysicalWaveExactMeridionalVelocitiesAtCellCenters)))
                                ExactMeridionalVelocityMaximum = (
                                max(ExactMeridionalVelocityMaximum,
                                max(GeophysicalWaveExactMeridionalVelocitiesAtCellCenters)))
                            if iTime == nTime - 1:                        
                                if not(problem_type == 'Planetary_Rossby_Wave' 
                                       or problem_type == 'Topographic_Rossby_Wave'):
                                    ExactSurfaceElevationMaximumMagnitude = max(abs(ExactSurfaceElevationMinimum),
                                                                                ExactSurfaceElevationMaximum)
                                ExactZonalVelocityMaximumMagnitude = max(abs(ExactZonalVelocityMinimum),
                                                                         ExactZonalVelocityMaximum) 
                                if not(myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave 
                                       and not(problem_type == 'Equatorial_Kelvin_Wave')):
                                    ExactMeridionalVelocityMaximumMagnitude = (
                                    max(abs(ExactMeridionalVelocityMinimum),ExactMeridionalVelocityMaximum))
                                ExactSurfaceElevationLimits = [-ExactSurfaceElevationMaximumMagnitude,
                                                               ExactSurfaceElevationMaximumMagnitude]
                                ExactZonalVelocityLimits = [-ExactZonalVelocityMaximumMagnitude,
                                                            ExactZonalVelocityMaximumMagnitude]
                                ExactMeridionalVelocityLimits = [-ExactMeridionalVelocityMaximumMagnitude,
                                                                 ExactMeridionalVelocityMaximumMagnitude]
                        if iTime == 0:
                            SurfaceElevationErrorMinimum = min(GeophysicalWaveSurfaceElevationError)
                            SurfaceElevationErrorMaximum = max(GeophysicalWaveSurfaceElevationError)
                            ZonalVelocityErrorMinimum = min(GeophysicalWaveZonalVelocityErrorAtCellCenters)
                            ZonalVelocityErrorMaximum = max(GeophysicalWaveZonalVelocityErrorAtCellCenters)
                            MeridionalVelocityErrorMinimum = (
                            min(GeophysicalWaveMeridionalVelocityErrorAtCellCenters))
                            MeridionalVelocityErrorMaximum = (
                            max(GeophysicalWaveMeridionalVelocityErrorAtCellCenters))
                        else:
                            SurfaceElevationErrorMinimum = min(SurfaceElevationErrorMinimum,
                                                               min(GeophysicalWaveSurfaceElevationError))
                            SurfaceElevationErrorMaximum = max(SurfaceElevationErrorMaximum,
                                                               max(GeophysicalWaveSurfaceElevationError)) 
                            ZonalVelocityErrorMinimum = (
                            min(ZonalVelocityErrorMinimum,min(GeophysicalWaveZonalVelocityErrorAtCellCenters)))
                            ZonalVelocityErrorMaximum = (
                            max(ZonalVelocityErrorMaximum,max(GeophysicalWaveZonalVelocityErrorAtCellCenters)))
                            MeridionalVelocityErrorMinimum = (
                            min(MeridionalVelocityErrorMinimum,
                                min(GeophysicalWaveMeridionalVelocityErrorAtCellCenters)))
                            MeridionalVelocityErrorMaximum = (
                            max(MeridionalVelocityErrorMaximum,
                                max(GeophysicalWaveMeridionalVelocityErrorAtCellCenters)))          
                        if CheckNumericalSurfaceElevationError:
                            PrintStatement1 = 'After %3d timesteps, ' %iTime 
                            PrintStatement2 = (
                            'the minimum and maximum surface elevation errors are {%+18.15f,%+18.15f}.'
                            %(SurfaceElevationErrorMinimum,SurfaceElevationErrorMaximum))
                            print(PrintStatement1 + PrintStatement2) 
                        if iTime == nTime - 1:                            
                            if (myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave 
                                or problem_type == 'Barotropic_Tide'):
                                SurfaceElevationErrorMaximumMagnitude = max(abs(SurfaceElevationErrorMinimum),
                                                                                SurfaceElevationErrorMaximum)
                                ZonalVelocityErrorMaximumMagnitude = max(abs(ZonalVelocityErrorMinimum),
                                                                         ZonalVelocityErrorMaximum)
                                MeridionalVelocityErrorMaximumMagnitude = max(abs(MeridionalVelocityErrorMinimum),
                                                                              MeridionalVelocityErrorMaximum)
                                SurfaceElevationErrorLimits = [-SurfaceElevationErrorMaximumMagnitude,
                                                               SurfaceElevationErrorMaximumMagnitude]
                                ZonalVelocityErrorLimits = [-ZonalVelocityErrorMaximumMagnitude,
                                                            ZonalVelocityErrorMaximumMagnitude]
                                MeridionalVelocityErrorLimits = [-MeridionalVelocityErrorMaximumMagnitude,
                                                                 MeridionalVelocityErrorMaximumMagnitude]
                            else:                            
                                SurfaceElevationErrorLimits = [SurfaceElevationErrorMinimum,
                                                               SurfaceElevationErrorMaximum]
                                ZonalVelocityErrorLimits = [ZonalVelocityErrorMinimum,ZonalVelocityErrorMaximum]
                                MeridionalVelocityErrorLimits = [MeridionalVelocityErrorMinimum,
                                                                 MeridionalVelocityErrorMaximum]
                        if plotSurfaceElevation and not(CheckNumericalSurfaceElevationError):
                            if iTime == nTime - 1:
                                output_directory = (
                                output_directory_root + wave_type_figure_title + '_ExactSurfaceElevation')
                                FileName = wave_type_figure_title + '_ExactSurfaceElevationLimits'
                                WriteStateVariableLimitsToFile(output_directory,ExactSurfaceElevationLimits,
                                                               FileName)
                                output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                    + time_integrator_short_form + '_SurfaceElevationError')
                                FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                            + '_SurfaceElevationErrorLimits')
                                WriteStateVariableLimitsToFile(output_directory,SurfaceElevationErrorLimits,
                                                               FileName)
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_NumericalSurfaceElevation')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_NumericalSurfaceElevation_' + '%3.3d' %iTime)
                            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          myMPAS_O.sshCurrent,FileName)         
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_SurfaceElevationError')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_SurfaceElevationError_' + '%3.3d' %iTime)
                            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          GeophysicalWaveSurfaceElevationError,FileName)  
                        if plotZonalVelocity and not(CheckNumericalSurfaceElevationError):
                            if iTime == nTime - 1:
                                output_directory = (
                                output_directory_root + wave_type_figure_title + '_ExactZonalVelocity')
                                FileName = wave_type_figure_title + '_ExactZonalVelocityLimits'
                                WriteStateVariableLimitsToFile(output_directory,ExactZonalVelocityLimits,FileName)
                                output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                    + time_integrator_short_form + '_ZonalVelocityError')
                                FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                            + '_ZonalVelocityErrorLimits')
                                WriteStateVariableLimitsToFile(output_directory,ZonalVelocityErrorLimits,
                                                               FileName)    
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_NumericalZonalVelocity')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_NumericalZonalVelocity_' + '%3.3d' %iTime)
                            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          GeophysicalWaveNumericalZonalVelocitiesAtCellCenters,
                                                          FileName)                            
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_ZonalVelocityError') 
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_ZonalVelocityError_' + '%3.3d' %iTime)
                            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          GeophysicalWaveZonalVelocityErrorAtCellCenters,FileName)
                        if plotMeridionalVelocity and not(CheckNumericalSurfaceElevationError):
                            if iTime == nTime - 1:
                                output_directory = (
                                output_directory_root + wave_type_figure_title + '_ExactMeridionalVelocity')
                                FileName = wave_type_figure_title + '_ExactMeridionalVelocityLimits'
                                WriteStateVariableLimitsToFile(output_directory,ExactMeridionalVelocityLimits,
                                                               FileName)                                  
                                output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                    + time_integrator_short_form + '_MeridionalVelocityError')
                                FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                            + '_MeridionalVelocityErrorLimits')
                                WriteStateVariableLimitsToFile(output_directory,MeridionalVelocityErrorLimits,
                                                               FileName)   
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_NumericalMeridionalVelocity')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_NumericalMeridionalVelocity_' + '%3.3d' %iTime)
                            CR.WriteTecPlot2DUnstructured(
                            output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                            GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters,FileName)
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_MeridionalVelocityError') 
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_MeridionalVelocityError_' + '%3.3d' %iTime)
                            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                          GeophysicalWaveMeridionalVelocityErrorAtCellCenters,
                                                          FileName)
                if iTime < nTime - 1:
                    MPAS_O_Mode_Forward.ocn_time_integration_Geophysical_Wave(myMPAS_O)
                    MPAS_O_Mode_Forward.ocn_shift_time_levels(myMPAS_O) 
            else: # if iCounter == 1:                        
                if np.mod(iTime,nDumpFrequency) == 0.0:
                    if plotFigures:   
                        DisplayTime = PrintDisplayTime(myMPAS_O.time)
                        xlabel = 'Zonal Distance (km)'
                        ylabel = 'Meridional Distance (km)'
                        if plotSurfaceElevation and not(CheckNumericalSurfaceElevationError):
                            if iTime == 0:                            
                                output_directory = (
                                output_directory_root + wave_type_figure_title + '_ExactSurfaceElevation')    
                                FileName = wave_type_figure_title + '_ExactSurfaceElevationLimits'
                                ExactSurfaceElevationLimits = ReadStateVariableLimitsFromFile(output_directory,
                                                                                              FileName+'.curve')
                                output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                    + time_integrator_short_form + '_SurfaceElevationError')
                                FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                            + '_SurfaceElevationErrorLimits')
                                SurfaceElevationErrorLimits = ReadStateVariableLimitsFromFile(output_directory,
                                                                                              FileName+'.curve')
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_NumericalSurfaceElevation')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_NumericalSurfaceElevation_' + '%3.3d' %iTime)
                            GeophysicalWaveNumericalSurfaceElevations = (
                            CR.ReadTecPlot2DUnstructured(output_directory,FileName+'.tec',
                                                         returnIndependentVariables=False)) 
                            Title = wave_type_title + ': Numerical Surface Elevation after\n' + DisplayTime
                            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                            myMPAS_O,output_directory,GeophysicalWaveNumericalSurfaceElevations,300,True,
                            ExactSurfaceElevationLimits,6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],
                            [15.0,15.0],Title,20.0,True,FileName,False,fig_size=[9.25,9.25],
                            cbarlabelformat='%.5f')                                 
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_SurfaceElevationError')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_SurfaceElevationError_' + '%3.3d' %iTime)
                            GeophysicalWaveSurfaceElevationError = (
                            CR.ReadTecPlot2DUnstructured(output_directory,FileName+'.tec',
                                                         returnIndependentVariables=False)) 
                            Title = wave_type_title + ': Surface Elevation Error after\n' + DisplayTime
                            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                            myMPAS_O,output_directory,GeophysicalWaveSurfaceElevationError,300,True,
                            SurfaceElevationErrorLimits,6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],
                            [15.0,15.0],Title,20.0,True,FileName,False,fig_size=[9.25,9.25],
                            cbarlabelformat='%.5f')
                        if plotZonalVelocity and not(CheckNumericalSurfaceElevationError):
                            if iTime == 0:
                                output_directory = (
                                output_directory_root + wave_type_figure_title + '_ExactZonalVelocity')
                                FileName = wave_type_figure_title + '_ExactZonalVelocityLimits'
                                ExactZonalVelocityLimits = ReadStateVariableLimitsFromFile(output_directory,
                                                                                           FileName+'.curve')
                                output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                    + time_integrator_short_form + '_ZonalVelocityError')
                                FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                            + '_ZonalVelocityErrorLimits')
                                ZonalVelocityErrorLimits = ReadStateVariableLimitsFromFile(output_directory,
                                                                                           FileName+'.curve')
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_NumericalZonalVelocity')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_NumericalZonalVelocity_' + '%3.3d' %iTime)
                            GeophysicalWaveNumericalZonalVelocities = (
                            CR.ReadTecPlot2DUnstructured(output_directory,FileName+'.tec',
                                                         returnIndependentVariables=False))
                            Title = wave_type_title + ': Numerical Zonal Velocity after\n' + DisplayTime
                            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                            myMPAS_O,output_directory,GeophysicalWaveNumericalZonalVelocities,300,True,
                            ExactZonalVelocityLimits,6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],
                            [15.0,15.0],Title,20.0,True,FileName,False,cbarlabelformat='%.5f')
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_ZonalVelocityError')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_ZonalVelocityError_' + '%3.3d' %iTime)
                            GeophysicalWaveZonalVelocityError = (
                            CR.ReadTecPlot2DUnstructured(output_directory,FileName+'.tec',
                                                         returnIndependentVariables=False)) 
                            Title = wave_type_title + ': Zonal Velocity Error after\n' + DisplayTime
                            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                            myMPAS_O,output_directory,GeophysicalWaveZonalVelocityError,300,True,
                            ZonalVelocityErrorLimits,6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],
                            [15.0,15.0],Title,20.0,True,FileName,False,fig_size=[9.25,9.25],
                            cbarlabelformat='%.5f')                              
                        if plotMeridionalVelocity and not(CheckNumericalSurfaceElevationError):  
                            if iTime == 0:
                                output_directory = (
                                output_directory_root + wave_type_figure_title + '_ExactMeridionalVelocity')
                                FileName = wave_type_figure_title + '_ExactMeridionalVelocityLimits'
                                ExactMeridionalVelocityLimits = ReadStateVariableLimitsFromFile(output_directory,
                                                                                                FileName+'.curve')
                                output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                    + time_integrator_short_form + '_MeridionalVelocityError')
                                FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                            + '_MeridionalVelocityErrorLimits')
                                MeridionalVelocityErrorLimits = ReadStateVariableLimitsFromFile(output_directory,
                                                                                                FileName+'.curve')
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_NumericalMeridionalVelocity')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_NumericalMeridionalVelocity_' + '%3.3d' %iTime)
                            GeophysicalWaveNumericalMeridionalVelocities = (
                            CR.ReadTecPlot2DUnstructured(output_directory,FileName+'.tec',
                                                         returnIndependentVariables=False))
                            Title = wave_type_title + ': Numerical Meridional Velocity after\n' + DisplayTime
                            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                            myMPAS_O,output_directory,GeophysicalWaveNumericalMeridionalVelocities,300,True,
                            ExactMeridionalVelocityLimits,6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],
                            [10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,False,cbarlabelformat='%.5f')
                            output_directory = (output_directory_root + wave_type_figure_title + '_'
                                                + time_integrator_short_form + '_MeridionalVelocityError')
                            FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                        + '_MeridionalVelocityError_' + '%3.3d' %iTime)
                            GeophysicalWaveMeridionalVelocityError = (
                            CR.ReadTecPlot2DUnstructured(output_directory,FileName+'.tec',
                                                         returnIndependentVariables=False)) 
                            Title = wave_type_title + ': Meridional Velocity Error after\n' + DisplayTime
                            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                            myMPAS_O,output_directory,GeophysicalWaveMeridionalVelocityError,300,True,
                            MeridionalVelocityErrorLimits,6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],
                            [10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,False,fig_size=[9.25,9.25],
                            cbarlabelformat='%.5f')


# In[56]:

def set_of_time_integrators(time_integrator_part=1):
    if time_integrator_part == 1:
        time_integrators = ['Forward_Backward','Generalized_FB_with_AB2_AM3_Step',
                            'Generalized_FB_with_AB2_AM3_Step','Generalized_FB_with_AB3_AM4_Step',
                            'Generalized_FB_with_AB3_AM4_Step','Generalized_FB_with_AB3_AM4_Step']
        LF_TR_and_LF_AM3_with_FB_Feedback_Types = ['' for x in range(0,len(time_integrators))]
        Generalized_FB_with_AB2_AM3_Step_Types = ['' for x in range(0,len(time_integrators))]
        Generalized_FB_with_AB2_AM3_Step_Types[1] = 'ThirdOrderAccurate_WideStabilityRange'
        Generalized_FB_with_AB2_AM3_Step_Types[2] = 'FourthOrderAccurate'
        Generalized_FB_with_AB3_AM4_Step_Types = ['' for x in range(0,len(time_integrators))]  
        Generalized_FB_with_AB3_AM4_Step_Types[3] = 'SecondOrderAccurate_OptimumChoice_ROMS'
        Generalized_FB_with_AB3_AM4_Step_Types[4] = 'ThirdOrderAccurate_MaximumStabilityRange'
        Generalized_FB_with_AB3_AM4_Step_Types[5] = 'FourthOrderAccurate_MaximumStabilityRange'
    if time_integrator_part == 2:
        time_integrators = ['Explicit_Midpoint_Method','Williamson_Low_Storage_Runge_Kutta_Third_Order',
                            'Low_Storage_Runge_Kutta_Fourth_Order','Leapfrog_Trapezoidal',
                            'LF_TR_and_LF_AM3_with_FB_Feedback','LF_TR_and_LF_AM3_with_FB_Feedback',
                            'Forward_Backward_with_RK2_Feedback','Generalized_FB_with_AB2_AM3_Step']       
        LF_TR_and_LF_AM3_with_FB_Feedback_Types = ['' for x in range(0,len(time_integrators))]
        LF_TR_and_LF_AM3_with_FB_Feedback_Types[4] = 'ThirdOrderAccurate_MaximumStabilityRange'
        LF_TR_and_LF_AM3_with_FB_Feedback_Types[5] = 'FourthOrderAccurate_MaximumStabilityRange'
        Generalized_FB_with_AB2_AM3_Step_Types = ['' for x in range(0,len(time_integrators))]
        Generalized_FB_with_AB2_AM3_Step_Types[7] = 'ThirdOrderAccurate_WeakAsymptoticInstabilityOfPhysicalModes'
        Generalized_FB_with_AB3_AM4_Step_Types = ['' for x in range(0,len(time_integrators))] 
    elif time_integrator_part == 3:
        time_integrators = ['Forward_Euler','Adams_Bashforth_Second_Order','Adams_Bashforth_Third_Order',
                            'Adams_Bashforth_Fourth_Order','LF_TR_and_LF_AM3_with_FB_Feedback',
                            'LF_TR_and_LF_AM3_with_FB_Feedback','Generalized_FB_with_AB3_AM4_Step',
                            'Generalized_FB_with_AB3_AM4_Step']
        LF_TR_and_LF_AM3_with_FB_Feedback_Types = ['' for x in range(0,len(time_integrators))]
        LF_TR_and_LF_AM3_with_FB_Feedback_Types[4] = 'ThirdOrderAccurate_LF_AM3'
        LF_TR_and_LF_AM3_with_FB_Feedback_Types[5] = 'FourthOrderAccurate_MinimumTruncationError'
        Generalized_FB_with_AB2_AM3_Step_Types = ['' for x in range(0,len(time_integrators))]
        Generalized_FB_with_AB3_AM4_Step_Types = ['' for x in range(0,len(time_integrators))]
        Generalized_FB_with_AB3_AM4_Step_Types[6] = 'ThirdOrderAccurate_AB3_AM4'
        Generalized_FB_with_AB3_AM4_Step_Types[7] = 'ThirdOrderAccurate_OptimumChoice'
    return [time_integrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
            Generalized_FB_with_AB3_AM4_Step_Types]


# In[57]:

def runMain(problem_type,time_integrator_part=1,single_time_integrator=True,single_time_integrator_index=5):    
    if problem_type == 'Coastal_Kelvin_Wave':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_x'
        dt = 180.0
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif problem_type == 'Inertia_Gravity_Wave':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/InertiaGravityWaveMesh'
        base_mesh_file_name = 'base_mesh.nc'
        periodicity = 'Periodic'
        dt = 96.0
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif problem_type == 'Planetary_Rossby_Wave' or problem_type == 'Topographic_Rossby_Wave':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/PlanetaryRossbyWaveMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_y'
        dt = 195000.0
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif (problem_type == 'Equatorial_Kelvin_Wave' or problem_type == 'Equatorial_Yanai_Wave' 
          or problem_type == 'Equatorial_Rossby_Wave' or problem_type == 'Equatorial_Inertia_Gravity_Wave'):
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/EquatorialWaveMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_y'        
        if problem_type == 'Equatorial_Kelvin_Wave':
            dt = 750.00 
        elif problem_type == 'Equatorial_Yanai_Wave':
            dt = 390.0
        elif problem_type == 'Equatorial_Rossby_Wave':
            dt = 2700.0
        elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
            dt = 420.0
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif problem_type == 'Barotropic_Tide':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/BarotropicTideMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_xy'  
        dt = 10.0
        printPhaseSpeedOfWaveModes = True
        problem_is_linear = True
    elif problem_type == 'Diffusion_Equation':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/BarotropicTideMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_xy'
        dt = 2260.0 
        printPhaseSpeedOfWaveModes = False 
        problem_is_linear = True
    elif problem_type == 'Viscous_Burgers_Equation':
        mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/BarotropicTideMesh'
        base_mesh_file_name = 'culled_mesh.nc'
        periodicity = 'NonPeriodic_xy'  
        dt = 2100.0
        printPhaseSpeedOfWaveModes = False   
        problem_is_linear = False
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    CourantNumber = DetermineCourantNumberForGivenTimeStep(mesh_directory,base_mesh_file_name,mesh_file_name,
                                                           mesh_type,problem_type,problem_is_linear,periodicity,dt,
                                                           printCourantNumber=False)
    print('The Courant Number is %.6f.' %CourantNumber)
    plotFigures = True
    [time_integrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
     Generalized_FB_with_AB3_AM4_Step_Types] = set_of_time_integrators(time_integrator_part)
    if single_time_integrator:
        i_time_integrator_lower_limit = single_time_integrator_index
        i_time_integrator_upper_limit = single_time_integrator_index + 1
    else:
        i_time_integrator_lower_limit = 0
        i_time_integrator_upper_limit = len(time_integrators)
    for i_time_integrator in range(i_time_integrator_lower_limit,i_time_integrator_upper_limit):
        time_integrator = time_integrators[i_time_integrator]
        LF_TR_and_LF_AM3_with_FB_Feedback_Type = LF_TR_and_LF_AM3_with_FB_Feedback_Types[i_time_integrator]
        Generalized_FB_with_AB2_AM3_Step_Type = Generalized_FB_with_AB2_AM3_Step_Types[i_time_integrator]
        Generalized_FB_with_AB3_AM4_Step_Type = Generalized_FB_with_AB3_AM4_Step_Types[i_time_integrator]
        Main(mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,problem_is_linear,
             periodicity,CourantNumber,time_integrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
             Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
             printPhaseSpeedOfWaveModes,plotFigures,CheckNumericalSurfaceElevationError=False,
             colormap=plt.cm.seismic)


# In[58]:

do_runMain_CoastalKelvinWave = False
if do_runMain_CoastalKelvinWave:
    problem_type = 'Coastal_Kelvin_Wave'
    runMain(problem_type,time_integrator_part=1,single_time_integrator=True,single_time_integrator_index=5)


# In[59]:

do_runMain_InertiaGravityWave = False
if do_runMain_InertiaGravityWave:
    problem_type = 'Inertia_Gravity_Wave'
    runMain(problem_type,time_integrator_part=1,single_time_integrator=True,single_time_integrator_index=5)


# In[60]:

def ComputeL2NormOfStateVariableDefinedAtEdges(nEdges,nNonPeriodicBoundaryEdges,boundaryEdge,
                                               StateVariableDefinedAtEdges):
    L2Norm = 0.0
    for iEdge in range(0,nEdges):
        if boundaryEdge[iEdge] == 0.0:
            L2Norm += (StateVariableDefinedAtEdges[iEdge])**2.0
    L2Norm = np.sqrt(L2Norm/float(nNonPeriodicBoundaryEdges))
    return L2Norm


# In[61]:

def ComputeL2NormOfStateVariableDefinedAtCellCenters(nCells,nNonPeriodicBoundaryCells,boundaryCell,
                                                     StateVariableDefinedAtCellCenters):
    L2Norm = 0.0
    for iCell in range(0,nCells):
        if boundaryCell[iCell] == 0.0:
            L2Norm += (StateVariableDefinedAtCellCenters[iCell])**2.0
    L2Norm = np.sqrt(L2Norm/float(nNonPeriodicBoundaryCells))
    return L2Norm


# In[62]:

def Specify_Ratio_of_FinalTime_to_TimePeriod(problem_type):
    if (problem_type == 'Coastal_Kelvin_Wave' or problem_type == 'Equatorial_Kelvin_Wave' 
        or problem_type == 'Equatorial_Yanai_Wave' or problem_type == 'Equatorial_Rossby_Wave'
        or problem_type == 'Equatorial_Inertia_Gravity_Wave' or problem_type == 'Barotropic_Tide'):
        Ratio_of_FinalTime_to_TimePeriod = 0.5
    elif (problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Planetary_Rossby_Wave'
          or problem_type == 'Topographic_Rossby_Wave' or problem_type == 'Diffusion_Equation'):    
        Ratio_of_FinalTime_to_TimePeriod = 0.25 
    elif problem_type == 'Viscous_Burgers_Equation':
        Ratio_of_FinalTime_to_TimePeriod = 1.0
    return Ratio_of_FinalTime_to_TimePeriod


# In[63]:

def Main_ConvergenceTest_GeophysicalWave(
convergence_type,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,problem_is_linear,
periodicity,CourantNumber,time_integrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,Ratio_of_FinalTime_to_TimePeriod,
specified_lY,specified_dt,wave_type_title,wave_type_figure_title,time_integrator_short_form,
plotNumericalSolution=False,coarsestMesh=False,plotFigures=False,xCell_CoarsestRectilinearMesh=[],
yCell_CoarsestRectilinearMesh=[],GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh=[],
GeophysicalWaveExactZonalVelocitiesAtCellCenters_CoarsestRectilinearMesh=[],
GeophysicalWaveExactMeridionalVelocitiesAtCellCenters_CoarsestRectilinearMesh=[]):   
    if convergence_type == 'Space':
        useCourantNumberToDetermineTimeStep = False
    else:
        useCourantNumberToDetermineTimeStep = True    
    myMPAS_O = (
    MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,
                            problem_is_linear,periodicity=periodicity,CourantNumber=CourantNumber,
                            useCourantNumberToDetermineTimeStep=useCourantNumberToDetermineTimeStep,
                            time_integrator=time_integrator,
                            LF_TR_and_LF_AM3_with_FB_Feedback_Type=LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                            Generalized_FB_with_AB2_AM3_Step_Type=Generalized_FB_with_AB2_AM3_Step_Type,
                            Generalized_FB_with_AB3_AM4_Step_Type=Generalized_FB_with_AB3_AM4_Step_Type,
                            printPhaseSpeedOfWaveModes=False,
                            specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False))
    myMPAS_O.myNamelist.config_dt = round(myMPAS_O.myNamelist.config_dt)
    if (convergence_type == 'Space' or convergence_type == 'SpaceAndTime'
        or convergence_type == 'SpaceAndTime_Short'):
        myMPAS_O.lY = specified_lY
    if convergence_type == 'Space' or convergence_type == 'Time':
        myMPAS_O.myNamelist.config_dt = specified_dt
    if convergence_type == 'Space':
        print('The time step for Courant number %.2f is %.2f seconds.' %(CourantNumber, 
                                                                         myMPAS_O.myNamelist.config_dt))     
    dt = myMPAS_O.myNamelist.config_dt 
    lX = myMPAS_O.lX
    lY = myMPAS_O.lY
    cX1 = myMPAS_O.ExactSolutionParameters[4]
    cX2 = myMPAS_O.ExactSolutionParameters[5]
    cY1 = myMPAS_O.ExactSolutionParameters[6]
    cY2 = myMPAS_O.ExactSolutionParameters[7]
    etaHat1 = myMPAS_O.ExactSolutionParameters[10]
    etaHat2 = myMPAS_O.ExactSolutionParameters[11]
    VelocityScale = myMPAS_O.ExactSolutionParameters[27]
    kappa1 = myMPAS_O.ExactSolutionParameters[31]
    kappa2 = myMPAS_O.ExactSolutionParameters[32]
    s = myMPAS_O.ExactSolutionParameters[35]
    if myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave or problem_type == 'Barotropic_Tide':
        abs_cX = max(abs(cX1),abs(cX2))
        abs_cY = max(abs(cY1),abs(cY2))
        if abs_cX != 0.0:
            TimePeriodOfFastWaveMode = lX/abs_cX 
        else:
            TimePeriodOfFastWaveMode = lY/abs_cY
        # Note that for all two-dimensional dispersive waves, 
        # TimePeriodOfFastWaveMode = lX/abs_cX = lX*kX/abs(omega) = lY*kY/abs(omega) = lY/abs_cY
        # where kX and kY are the zonal and meridional wavenumbers of the fast wave mode with omega being its 
        # angular frequency.
        print('The zonal extent of the domain is %.6f m.' %lX)
        print('The meridional extent of the domain is %.6f m.' %lY)
        print('The time period of the fast wave mode is %.6f.' %TimePeriodOfFastWaveMode)
        print('The minimum number of time steps of magnitude %.2f required to constitute the time period is %d.'
              %(dt,int(np.ceil(TimePeriodOfFastWaveMode/dt))))
        FinalTime = Ratio_of_FinalTime_to_TimePeriod*TimePeriodOfFastWaveMode
    elif problem_type == 'Diffusion_Equation':
        kappa = min(kappa1,kappa2) # i.e. kappa = kappa1
        FinalTime = np.log(4.0)/kappa
        print('The time taken by the solution magnitude to drop to %d%% is %.6f.' %(25,FinalTime))
        PrintStatement1 = 'The minimum number of time steps of magnitude %.2f ' %dt
        PrintStatement2 = ('required for the solution magnitude to drop to %d%% is %d.' 
                           %(25,int(np.ceil(FinalTime/dt))))
        print(PrintStatement1 + PrintStatement2)
        FinalTime *= Ratio_of_FinalTime_to_TimePeriod
    elif problem_type == 'Viscous_Burgers_Equation':
        FinalTime = (0.75 - 0.25)*myMPAS_O.lX/s
        print('The time taken by the shock wave to traverse %d%% of the zonal extent is %.6f.' %(50,FinalTime))
        PrintStatement1 = 'The minimum number of time steps of magnitude %.2f ' %dt
        PrintStatement2 = ('required for the shock wave to traverse %d%% of the zonal extent is %d.' 
                           %(50,int(np.ceil(FinalTime/dt))))
        print(PrintStatement1 + PrintStatement2)  
        FinalTime *= Ratio_of_FinalTime_to_TimePeriod
    nTime = int(FinalTime/dt)
    if convergence_type == 'SpaceAndTime' or convergence_type == 'SpaceAndTime_Short':
        print('The final time for the convergence study at constant ratio of time step to grid spacing is %.6f.' 
              %FinalTime)
        print('For number of cells in each direction = %3d, number of time steps = %3d.' 
              %(int(np.sqrt(float(myMPAS_O.nCells))),nTime))
        myMPAS_O.myNamelist.config_dt = FinalTime/float(nTime)
        ModifiedCourantNumber = CourantNumber*myMPAS_O.myNamelist.config_dt/dt
        print('The final time step for the modified Courant number of %.6f is %.6f seconds.' 
              %(ModifiedCourantNumber,myMPAS_O.myNamelist.config_dt))
        print('The increase in time step is %.6f seconds and that in the Courant Number is %.6f.'
              %(myMPAS_O.myNamelist.config_dt-dt,ModifiedCourantNumber-CourantNumber))
    MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    compute_these_variables = np.zeros(8,dtype=bool)
    compute_these_variables[4] = True # compute_these_variables[4] = compute_tangentialVelocity
    GeophysicalWaveNumericalZonalVelocities = np.zeros(myMPAS_O.nEdges)
    GeophysicalWaveNumericalMeridionalVelocities = np.zeros(myMPAS_O.nEdges)
    GeophysicalWaveExactVelocities = np.zeros((myMPAS_O.nEdges,2))
    for iTime in range(0,nTime+1):
        myMPAS_O.iTime = iTime
        myMPAS_O.time = float(iTime)*myMPAS_O.myNamelist.config_dt
        if iTime == 0 or iTime == nTime:
            [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocitiesAtCellCenters, 
             GeophysicalWaveExactMeridionalVelocitiesAtCellCenters] = (
            DetermineGeophysicalWaveExactSolutionsAtCellCenters(myMPAS_O))
            [GeophysicalWaveExactZonalVelocities, GeophysicalWaveExactMeridionalVelocities] = (
            DetermineGeophysicalWaveExactSolutions(myMPAS_O,DetermineGeophysicalWaveExactSurfaceElevations=False))
            GeophysicalWaveExactVelocities[:,0] = GeophysicalWaveExactZonalVelocities[:] 
            GeophysicalWaveExactVelocities[:,1] = GeophysicalWaveExactMeridionalVelocities[:] 
            GeophysicalWaveExactNormalVelocities, GeophysicalWaveExactTangentialVelocities = (
            ComputeNormalAndTangentialComponentsAtEdge(GeophysicalWaveExactVelocities,myMPAS_O.angleEdge,'both'))
        if iTime == 0.0: # Specify initial conditions
            myMPAS_O.sshCurrent[:] = GeophysicalWaveExactSurfaceElevations[:]
            myMPAS_O.normalVelocityCurrent[:,0] = GeophysicalWaveExactNormalVelocities      
        if iTime == nTime:
            print('The final time for the %3d x %3d mesh is %.6f seconds.' 
                  %(int(np.sqrt(float(myMPAS_O.nCells))),int(np.sqrt(float(myMPAS_O.nCells))),myMPAS_O.time))
            MPAS_O_Shared.ocn_diagnostic_solve(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent,
                                               compute_these_variables)     
            for iEdge in range(0,myMPAS_O.nEdges):
                if myMPAS_O.boundaryEdge[iEdge] == 1.0: 
                # i.e. if the edge is along a non-periodic boundary
                    myMPAS_O.tangentialVelocity[iEdge,0] = GeophysicalWaveExactTangentialVelocities[iEdge]
                    GeophysicalWaveNumericalZonalVelocities[iEdge] = GeophysicalWaveExactZonalVelocities[iEdge]
                    GeophysicalWaveNumericalMeridionalVelocities[iEdge] = (
                    GeophysicalWaveExactMeridionalVelocities[iEdge])
                else:
                    GeophysicalWaveNumericalZonalVelocities[iEdge] = (
                    (myMPAS_O.normalVelocityCurrent[iEdge,0]*np.cos(myMPAS_O.angleEdge[iEdge])
                     - myMPAS_O.tangentialVelocity[iEdge,0]*np.sin(myMPAS_O.angleEdge[iEdge])))
                    GeophysicalWaveNumericalMeridionalVelocities[iEdge] = (
                    (myMPAS_O.normalVelocityCurrent[iEdge,0]*np.sin(myMPAS_O.angleEdge[iEdge])
                     + myMPAS_O.tangentialVelocity[iEdge,0]*np.cos(myMPAS_O.angleEdge[iEdge])))            
            GeophysicalWaveNumericalZonalVelocitiesAtCellCenters = (
            MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
            myMPAS_O,GeophysicalWaveNumericalZonalVelocities))
            GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters = (
            MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
            myMPAS_O,GeophysicalWaveNumericalMeridionalVelocities))
            GeophysicalWaveSurfaceElevationError = myMPAS_O.sshCurrent - GeophysicalWaveExactSurfaceElevations
            GeophysicalWaveZonalVelocityErrorAtCellCenters = (
            (GeophysicalWaveExactZonalVelocitiesAtCellCenters 
             - GeophysicalWaveNumericalZonalVelocitiesAtCellCenters))
            GeophysicalWaveMeridionalVelocityErrorAtCellCenters = (
            (GeophysicalWaveExactMeridionalVelocitiesAtCellCenters
             - GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters))
            if plotFigures:
                DisplayTime = PrintDisplayTime(myMPAS_O.time)
                xlabel = 'Zonal Distance (km)'
                ylabel = 'Meridional Distance (km)'
                if plotSurfaceElevation:
                    output_directory = output_directory_root + wave_type_figure_title + '_ExactSurfaceElevation'
                    FileName = (wave_type_figure_title + '_ExactSurfaceElevation_nCellsInEachHorizontalDirection_'
                                + '%3.3d' %int(np.sqrt(float(myMPAS_O.nCells))))
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveExactSurfaceElevations,FileName)
                    Title = wave_type_title + ': Exact Surface Elevation after\n' + DisplayTime
                    PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                    myMPAS_O,output_directory,GeophysicalWaveExactSurfaceElevations,300,False,[0.0,0.0],6,
                    colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,
                    False,fig_size=[9.25,9.25],cbarlabelformat='%.5f')                       
                    output_directory = (output_directory_root + wave_type_figure_title + '_'
                                        + time_integrator_short_form + '_NumericalSurfaceElevation')
                    FileName = (wave_type_figure_title + '_' + time_integrator_short_form
                                + '_NumericalSurfaceElevation_nCellsInEachHorizontalDirection_' + '%3.3d' 
                                %int(np.sqrt(float(myMPAS_O.nCells))))
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  myMPAS_O.sshCurrent,FileName)     
                    Title = wave_type_title + ': Numerical Surface Elevation after\n' + DisplayTime
                    PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                    myMPAS_O,output_directory,myMPAS_O.sshCurrent,300,False,[0.0,0.0],6,colormap,13.75,
                    [xlabel,ylabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,False,
                    fig_size=[9.25,9.25],cbarlabelformat='%.5f')
                    output_directory = (output_directory_root + wave_type_figure_title + '_'
                                        + time_integrator_short_form + '_SurfaceElevationError')
                    FileName = (wave_type_figure_title + '_' + time_integrator_short_form
                                + '_SurfaceElevationError_nCellsInEachHorizontalDirection_' + '%3.3d' 
                                %int(np.sqrt(float(myMPAS_O.nCells))))
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveSurfaceElevationError,FileName)
                    Title = wave_type_title + ': Surface Elevation Error after\n' + DisplayTime
                    PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                    myMPAS_O,output_directory,GeophysicalWaveSurfaceElevationError,300,False,[0.0,0.0],6,colormap,
                    13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,False,
                    fig_size=[9.25,9.25],cbarlabelformat='%.5f')
                if plotZonalVelocity:
                    output_directory = output_directory_root + wave_type_figure_title + '_ExactZonalVelocity'
                    FileName = (wave_type_figure_title + '_ExactZonalVelocity_nCellsInEachHorizontalDirection_'
                                + '%3.3d' %int(np.sqrt(float(myMPAS_O.nCells))))
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveExactZonalVelocitiesAtCellCenters,FileName)
                    Title = wave_type_title + ': Exact Zonal Velocity after\n' + DisplayTime
                    PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                    myMPAS_O,output_directory,GeophysicalWaveExactZonalVelocitiesAtCellCenters,300,False,
                    [0.0,0.0],6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                    True,FileName,False,fig_size=[9.25,9.25],cbarlabelformat='%.5f')      
                    output_directory = (output_directory_root + wave_type_figure_title + '_'
                                        + time_integrator_short_form + '_NumericalZonalVelocity')
                    FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                + '_NumericalZonalVelocity_nCellsInEachHorizontalDirection_' + '%3.3d' 
                                %int(np.sqrt(float(myMPAS_O.nCells))))
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveNumericalZonalVelocitiesAtCellCenters,FileName)
                    Title = wave_type_title + ': Numerical Zonal Velocity after\n' + DisplayTime
                    PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                    myMPAS_O,output_directory,GeophysicalWaveNumericalZonalVelocitiesAtCellCenters,300,False,
                    [0.0,0.0],6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                    True,FileName,False,fig_size=[9.25,9.25],cbarlabelformat='%.5f')
                    output_directory = (output_directory_root + wave_type_figure_title + '_'
                                        + time_integrator_short_form + '_ZonalVelocityError')                    
                    FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                + '_ZonalVelocityError_nCellsInEachHorizontalDirection_' + '%3.3d' 
                                %int(np.sqrt(float(myMPAS_O.nCells))))
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveZonalVelocityErrorAtCellCenters,FileName)
                    Title = wave_type_title + ': Zonal Velocity Error after\n' + DisplayTime
                    PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                    myMPAS_O,output_directory,GeophysicalWaveZonalVelocityErrorAtCellCenters,300,False,[0.0,0.0],
                    6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,
                    False,fig_size=[9.25,9.25],cbarlabelformat='%.5f')
                if plotMeridionalVelocity:
                    output_directory = output_directory_root + wave_type_figure_title + '_ExactMeridionalVelocity'
                    FileName = (wave_type_figure_title 
                                + '_ExactMeridionalVelocity_nCellsInEachHorizontalDirection_'
                                + '%3.3d' %int(np.sqrt(float(myMPAS_O.nCells))))
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveExactMeridionalVelocitiesAtCellCenters,FileName)
                    Title = wave_type_title + ': Exact Meridional Velocity after\n' + DisplayTime
                    PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                    myMPAS_O,output_directory,GeophysicalWaveExactMeridionalVelocitiesAtCellCenters,300,False,
                    [0.0,0.0],6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                    True,FileName,False,fig_size=[9.25,9.25],cbarlabelformat='%.5f')      
                    output_directory = (output_directory_root + wave_type_figure_title + '_'
                                        + time_integrator_short_form + '_NumericalMeridionalVelocity')
                    FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                + '_NumericalMeridionalVelocity_nCellsInEachHorizontalDirection_' + '%3.3d' 
                                %int(np.sqrt(float(myMPAS_O.nCells))))
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters,
                                                  FileName)
                    Title = wave_type_title + ': Numerical Meridional Velocity after\n' + DisplayTime
                    PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                    myMPAS_O,output_directory,GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters,300,False,
                    [0.0,0.0],6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                    True,FileName,False,fig_size=[9.25,9.25],cbarlabelformat='%.5f')
                    output_directory = (output_directory_root + wave_type_figure_title + '_'
                                        + time_integrator_short_form + '_MeridionalVelocityError')
                    FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                                + '_MeridionalVelocityError_nCellsInEachHorizontalDirection_' + '%3.3d' 
                                %int(np.sqrt(float(myMPAS_O.nCells))))
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveMeridionalVelocityErrorAtCellCenters,FileName)
                    Title = wave_type_title + ': Meridional Velocity Error after\n' + DisplayTime
                    PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(
                    myMPAS_O,output_directory,GeophysicalWaveMeridionalVelocityErrorAtCellCenters,300,False,
                    [0.0,0.0],6,colormap,13.75,[xlabel,ylabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                    True,FileName,False,fig_size=[9.25,9.25],cbarlabelformat='%.5f')      
            if convergence_type == 'Time':
                if myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells:
                    GeophysicalWaveSurfaceElevationL2ErrorNorm = ComputeL2NormOfStateVariableDefinedAtCellCenters(
                    myMPAS_O.nCells,myMPAS_O.nNonPeriodicBoundaryCells,myMPAS_O.boundaryCell,
                    GeophysicalWaveSurfaceElevationError)
                else:
                    GeophysicalWaveSurfaceElevationL2ErrorNorm = (
                    np.linalg.norm(GeophysicalWaveSurfaceElevationError)/np.sqrt(myMPAS_O.nCells))
                GeophysicalWaveZonalVelocityL2ErrorNorm = (
                np.linalg.norm(GeophysicalWaveZonalVelocityErrorAtCellCenters)/np.sqrt(myMPAS_O.nCells))   
                GeophysicalWaveWaveMeridionalVelocityL2ErrorNorm = (
                np.linalg.norm(GeophysicalWaveWaveMeridionalVelocityErrorAtCellCenters)/np.sqrt(myMPAS_O.nCells))
            else:
                if coarsestMesh:
                    xCell_CoarsestRectilinearMesh, yCell_CoarsestRectilinearMesh = (
                    MOMIR.Generate_Rectilinear_MPAS_O_Mesh(myMPAS_O))
                    [GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh, 
                     GeophysicalWaveExactZonalVelocitiesAtCellCenters_CoarsestRectilinearMesh, 
                     GeophysicalWaveExactMeridionalVelocitiesAtCellCenters_CoarsestRectilinearMesh] = (
                    DetermineGeophysicalWaveExactSolutionsOnCoarsestRectilinearMesh(
                    myMPAS_O,xCell_CoarsestRectilinearMesh,yCell_CoarsestRectilinearMesh))
                    GeophysicalWaveNumericalSurfaceElevations = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,myMPAS_O.sshCurrent)) 
                    GeophysicalWaveSurfaceElevationError = (
                    (GeophysicalWaveNumericalSurfaceElevations 
                     - GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh))
                    GeophysicalWaveSurfaceElevationL2ErrorNorm = (
                    np.linalg.norm(GeophysicalWaveSurfaceElevationError)/np.sqrt(float(myMPAS_O.nCells))) 
                    GeophysicalWaveNumericalZonalVelocitiesAtCellCenters = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,GeophysicalWaveNumericalZonalVelocitiesAtCellCenters))         
                    GeophysicalWaveZonalVelocityErrorAtCellCenters = (
                    (GeophysicalWaveNumericalZonalVelocitiesAtCellCenters 
                     - GeophysicalWaveExactZonalVelocitiesAtCellCenters_CoarsestRectilinearMesh))
                    GeophysicalWaveZonalVelocityL2ErrorNorm = (
                    (np.linalg.norm(GeophysicalWaveZonalVelocityErrorAtCellCenters)
                     /np.sqrt(float(myMPAS_O.nCells))))          
                    GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters))                   
                    GeophysicalWaveMeridionalVelocityErrorAtCellCenters = (
                    (GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters
                     - GeophysicalWaveExactMeridionalVelocitiesAtCellCenters_CoarsestRectilinearMesh))
                    GeophysicalWaveMeridionalVelocityL2ErrorNorm = (
                    (np.linalg.norm(GeophysicalWaveMeridionalVelocityErrorAtCellCenters)
                     /np.sqrt(float(myMPAS_O.nCells))))
                else:
                    nCells_CoarsestRectilinearMesh = len(xCell_CoarsestRectilinearMesh)
                    xCell_FineRectilinearMesh, yCell_FineRectilinearMesh = (
                    MOMIR.Generate_Rectilinear_MPAS_O_Mesh(myMPAS_O))                
                    GeophysicalWaveNumericalSurfaceElevations = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,myMPAS_O.sshCurrent))
                    GeophysicalWaveNumericalSurfaceElevations = (
                    MOMIR.Interpolate_Solution_To_Coarsest_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O.gridSpacingMagnitude,xCell_FineRectilinearMesh,yCell_FineRectilinearMesh,
                    GeophysicalWaveNumericalSurfaceElevations,xCell_CoarsestRectilinearMesh,
                    yCell_CoarsestRectilinearMesh))
                    GeophysicalWaveSurfaceElevationError = (
                    (GeophysicalWaveNumericalSurfaceElevations 
                     - GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh))
                    GeophysicalWaveSurfaceElevationL2ErrorNorm = (
                    np.linalg.norm(GeophysicalWaveSurfaceElevationError)/np.sqrt(float(myMPAS_O.nCells))) 
                    GeophysicalWaveNumericalZonalVelocitiesAtCellCenters = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,GeophysicalWaveNumericalZonalVelocitiesAtCellCenters)) 
                    GeophysicalWaveNumericalZonalVelocitiesAtCellCenters = (
                    MOMIR.Interpolate_Solution_To_Coarsest_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O.gridSpacingMagnitude,xCell_FineRectilinearMesh,yCell_FineRectilinearMesh,
                    GeophysicalWaveNumericalZonalVelocitiesAtCellCenters,xCell_CoarsestRectilinearMesh,
                    yCell_CoarsestRectilinearMesh))                    
                    GeophysicalWaveZonalVelocityErrorAtCellCenters = (
                    (GeophysicalWaveNumericalZonalVelocitiesAtCellCenters 
                     - GeophysicalWaveExactZonalVelocitiesAtCellCenters_CoarsestRectilinearMesh)) 
                    GeophysicalWaveZonalVelocityL2ErrorNorm = (
                    (np.linalg.norm(GeophysicalWaveZonalVelocityErrorAtCellCenters)
                     /np.sqrt(float(nCells_CoarsestRectilinearMesh))))
                    GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters))
                    GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters = (
                    MOMIR.Interpolate_Solution_To_Coarsest_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O.gridSpacingMagnitude,xCell_FineRectilinearMesh,yCell_FineRectilinearMesh,
                    GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters,xCell_CoarsestRectilinearMesh,
                    yCell_CoarsestRectilinearMesh))
                    GeophysicalWaveMeridionalVelocityErrorAtCellCenters = (
                    (GeophysicalWaveNumericalMeridionalVelocitiesAtCellCenters 
                     - GeophysicalWaveExactMeridionalVelocitiesAtCellCenters_CoarsestRectilinearMesh))
                    GeophysicalWaveMeridionalVelocityL2ErrorNorm = (
                    (np.linalg.norm(GeophysicalWaveMeridionalVelocityErrorAtCellCenters)
                     /np.sqrt(float(nCells_CoarsestRectilinearMesh))))
        if iTime < nTime:
            MPAS_O_Mode_Forward.ocn_time_integration_Geophysical_Wave(myMPAS_O)
            MPAS_O_Mode_Forward.ocn_shift_time_levels(myMPAS_O)        
    if not(convergence_type == 'Time') and coarsestMesh:
        return [myMPAS_O.gridSpacingMagnitude, GeophysicalWaveSurfaceElevationL2ErrorNorm,
                GeophysicalWaveZonalVelocityL2ErrorNorm, GeophysicalWaveMeridionalVelocityL2ErrorNorm, 
                xCell_CoarsestRectilinearMesh, yCell_CoarsestRectilinearMesh,
                GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh,
                GeophysicalWaveExactZonalVelocitiesAtCellCenters_CoarsestRectilinearMesh,
                GeophysicalWaveExactMeridionalVelocitiesAtCellCenters_CoarsestRectilinearMesh]
    else:
        return [myMPAS_O.gridSpacingMagnitude, GeophysicalWaveSurfaceElevationL2ErrorNorm, 
                GeophysicalWaveZonalVelocityL2ErrorNorm, GeophysicalWaveMeridionalVelocityL2ErrorNorm]


# In[64]:

def ConvergenceTest_GeophysicalWave(convergence_type,problem_type,CourantNumber,time_integrator,
                                    LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                    Generalized_FB_with_AB3_AM4_Step_Type,Ratio_of_FinalTime_to_TimePeriod,
                                    output_these_variables):
    if convergence_type == 'Space':
        nCellsX = np.array([16,32,64,128,256])
        nCases = len(nCellsX)
    elif convergence_type == 'Time':
        CourantNumberArray = np.array([0.25,0.30,0.35,0.40,0.45])
        nCases = len(CourantNumberArray)
        nCellsX = np.ones(nCases,dtype=int)*100   
    elif convergence_type == 'SpaceAndTime':
        nCellsXMin = 50
        nCellsXMax = 250
        d_nCellsX = 10
        nCases = int((nCellsXMax - nCellsXMin)/d_nCellsX) + 1 # i.e. nCases = (250 - 50)/10 + 1 = 21
        nCellsX = np.linspace(nCellsXMin,nCellsXMax,nCases,dtype=int)
    elif convergence_type == 'SpaceAndTime_Short':
        nCellsXMin = 100
        nCellsXMax = 150
        d_nCellsX = 10
        nCases = int((nCellsXMax - nCellsXMin)/d_nCellsX) + 1 # i.e. nCases = (150 - 100)/10 + 1 = 6
        nCellsX = np.linspace(nCellsXMin,nCellsXMax,nCases,dtype=int)
    dc = np.zeros(nCases)
    GeophysicalWaveSurfaceElevationL2ErrorNorm = np.zeros(nCases)
    GeophysicalWaveZonalVelocityL2ErrorNorm = np.zeros(nCases)
    GeophysicalWaveMeridionalVelocityL2ErrorNorm = np.zeros(nCases)
    if (convergence_type == 'Space' or convergence_type == 'SpaceAndTime'
        or convergence_type == 'SpaceAndTime_Short'):
        if problem_type == 'Coastal_Kelvin_Wave':
            mesh_directory = (
            'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/ConvergenceStudyMeshes')
            base_mesh_file_name = 'culled_mesh_%s.nc' %(nCellsX[nCases-1])
            mesh_file_name = 'mesh_%s.nc' %(nCellsX[nCases-1])
            periodicity = 'NonPeriodic_x'
            problem_is_linear = True
        elif problem_type == 'Inertia_Gravity_Wave':
            mesh_directory = (
            'MPAS_O_Shallow_Water_Mesh_Generation/InertiaGravityWaveMesh/ConvergenceStudyMeshes')
            base_mesh_file_name = 'base_mesh_%s.nc' %(nCellsX[nCases-1])
            mesh_file_name = 'mesh_%s.nc' %(nCellsX[nCases-1])
            periodicity = 'Periodic'
            problem_is_linear = True            
        mesh_type = 'uniform'
        myMPAS_O = (
        MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,
                                problem_is_linear,periodicity=periodicity,CourantNumber=CourantNumber,
                                useCourantNumberToDetermineTimeStep=True,time_integrator=time_integrator,
                                LF_TR_and_LF_AM3_with_FB_Feedback_Type=LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                Generalized_FB_with_AB2_AM3_Step_Type=Generalized_FB_with_AB2_AM3_Step_Type,
                                Generalized_FB_with_AB3_AM4_Step_Type=Generalized_FB_with_AB3_AM4_Step_Type,
                                printPhaseSpeedOfWaveModes=False,
                                specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False))
        specified_lY = myMPAS_O.lY 
    if convergence_type == 'Space':   
        lX = myMPAS_O.lX
        lY = myMPAS_O.lY
        cX1 = myMPAS_O.ExactSolutionParameters[4]
        cX2 = myMPAS_O.ExactSolutionParameters[5]
        cY1 = myMPAS_O.ExactSolutionParameters[6]
        cY2 = myMPAS_O.ExactSolutionParameters[7]
        kappa1 = myMPAS_O.ExactSolutionParameters[31]
        kappa2 = myMPAS_O.ExactSolutionParameters[32]
        s = myMPAS_O.ExactSolutionParameters[35]
        if myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave or problem_type == 'Barotropic_Tide':
            abs_cX = max(abs(cX1),abs(cX2))
            abs_cY = max(abs(cY1),abs(cY2))
            if abs_cX != 0.0:
                TimePeriodOfFastWaveMode = lX/abs_cX 
            else:
                TimePeriodOfFastWaveMode = lY/abs_cY
            FinalTime = Ratio_of_FinalTime_to_TimePeriod*TimePeriodOfFastWaveMode
        elif problem_type == 'Diffusion_Equation':
            kappa = min(kappa1,kappa2) # i.e. kappa = kappa1
            FinalTime = np.log(4.0)/kappa
            FinalTime *= Ratio_of_FinalTime_to_TimePeriod
        elif problem_type == 'Viscous_Burgers_Equation':
            FinalTime = (0.75 - 0.25)*myMPAS_O.lX/s  
            FinalTime *= Ratio_of_FinalTime_to_TimePeriod
        specified_dt = myMPAS_O.myNamelist.config_dt
        nTime = int(FinalTime/specified_dt)
        specified_dt = FinalTime/float(nTime)   
    elif convergence_type == 'Time':
        if problem_type == 'Coastal_Kelvin_Wave':
            mesh_directory = (
            'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/ConvergenceStudyMeshes')
            base_mesh_file_name = 'culled_mesh_%s.nc' %(nCellsX[0])
            mesh_file_name = 'mesh_%s.nc' %(nCellsX[0])
            periodicity = 'NonPeriodic_x'
            problem_is_linear = True
        elif problem_type == 'Inertia_Gravity_Wave':
            mesh_directory = (
            'MPAS_O_Shallow_Water_Mesh_Generation/InertiaGravityWaveMesh/ConvergenceStudyMeshes')
            base_mesh_file_name = 'base_mesh_%s.nc' %(nCellsX[0])
            mesh_file_name = 'mesh_%s.nc' %(nCellsX[0])
            periodicity = 'Periodic'
            problem_is_linear = True            
        mesh_type = 'uniform'
        myMPAS_O = (
        MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,
                                problem_is_linear,periodicity=periodicity,CourantNumber=CourantNumber,
                                useCourantNumberToDetermineTimeStep=False,time_integrator=time_integrator,
                                LF_TR_and_LF_AM3_with_FB_Feedback_Type=LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                Generalized_FB_with_AB2_AM3_Step_Type=Generalized_FB_with_AB2_AM3_Step_Type,
                                Generalized_FB_with_AB3_AM4_Step_Type=Generalized_FB_with_AB3_AM4_Step_Type,
                                printPhaseSpeedOfWaveModes=False,
                                specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False))   
        lX = myMPAS_O.lX
        lY = myMPAS_O.lY
        cX1 = myMPAS_O.ExactSolutionParameters[4]
        cX2 = myMPAS_O.ExactSolutionParameters[5]
        cY1 = myMPAS_O.ExactSolutionParameters[6]
        cY2 = myMPAS_O.ExactSolutionParameters[7]
        kappa1 = myMPAS_O.ExactSolutionParameters[31]
        kappa2 = myMPAS_O.ExactSolutionParameters[32]
        s = myMPAS_O.ExactSolutionParameters[35]
        if myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave or problem_type == 'Barotropic_Tide':
            abs_cX = max(abs(cX1),abs(cX2))
            abs_cY = max(abs(cY1),abs(cY2))
            if abs_cX != 0.0:
                TimePeriodOfFastWaveMode = lX/abs_cX 
            else:
                TimePeriodOfFastWaveMode = lY/abs_cY
            FinalTime = Ratio_of_FinalTime_to_TimePeriod*TimePeriodOfFastWaveMode
        elif problem_type == 'Diffusion_Equation':
            kappa = min(kappa1,kappa2) # i.e. kappa = kappa1
            FinalTime = np.log(4.0)/kappa
            FinalTime *= Ratio_of_FinalTime_to_TimePeriod
        elif problem_type == 'Viscous_Burgers_Equation':
            FinalTime = (0.75 - 0.25)*myMPAS_O.lX/s  
            FinalTime *= Ratio_of_FinalTime_to_TimePeriod
        specified_lY = 0.0
        dx = myMPAS_O.gridSpacingMagnitude
    elif convergence_type == 'SpaceAndTime' or convergence_type == 'SpaceAndTime_Short':
        specified_dt = 0.0
    wave_type_title, wave_type_figure_title = DetermineFigureAndImageTitles(problem_type) 
    time_integrator_short_form = (
    DetermineTimeIntegratorShortForm(time_integrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                     Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type))
    for iCase in range(0,nCases):
        if convergence_type == 'Time':
            CourantNumber = CourantNumberArray[iCase]
            dt = CourantNumber*dx/WaveSpeed
            nTime = int(FinalTime/dt)
            specified_dt = FinalTime/float(nTime)
        if problem_type == 'Coastal_Kelvin_Wave':
            mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/ConvergenceStudyMeshes'
            base_mesh_file_name = 'culled_mesh_%s.nc' %(nCellsX[iCase])
            mesh_file_name = 'mesh_%s.nc' %(nCellsX[iCase])             
        elif problem_type == 'Inertia_Gravity_Wave':
            mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/InertiaGravityWaveMesh/ConvergenceStudyMeshes'
            base_mesh_file_name = 'base_mesh_%s.nc' %(nCellsX[iCase])
            mesh_file_name = 'mesh_%s.nc' %(nCellsX[iCase])
        if convergence_type == 'Time':        
            [dc[iCase], GeophysicalWaveSurfaceElevationL2ErrorNorm[iCase], 
             GeophysicalWaveZonalVelocityL2ErrorNorm[iCase], 
             GeophysicalWaveMeridionalVelocityL2ErrorNorm[iCase]] = (
            Main_ConvergenceTest_GeophysicalWave(convergence_type,mesh_directory,base_mesh_file_name,
                                                 mesh_file_name,mesh_type,problem_type,problem_is_linear,
                                                 periodicity,CourantNumber,time_integrator,
                                                 LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                 Generalized_FB_with_AB2_AM3_Step_Type,
                                                 Generalized_FB_with_AB3_AM4_Step_Type,
                                                 Ratio_of_FinalTime_to_TimePeriod,specified_lY,specified_dt,
                                                 wave_type_title,wave_type_figure_title,
                                                 time_integrator_short_form,plotNumericalSolution=False)) 
        else:
            if iCase == 0:
                [dc[iCase], GeophysicalWaveSurfaceElevationL2ErrorNorm[iCase], 
                 GeophysicalWaveZonalVelocityL2ErrorNorm[iCase], 
                 GeophysicalWaveMeridionalVelocityL2ErrorNorm[iCase], xCell_CoarsestRectilinearMesh, 
                 yCell_CoarsestRectilinearMesh, GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh,
                 GeophysicalWaveExactZonalVelocitiesAtCellCenters_CoarsestRectilinearMesh,
                 GeophysicalWaveExactMeridionalVelocitiesAtCellCenters_CoarsestRectilinearMesh] = (
                Main_ConvergenceTest_GeophysicalWave(convergence_type,mesh_directory,base_mesh_file_name,
                                                     mesh_file_name,mesh_type,problem_type,problem_is_linear,
                                                     periodicity,CourantNumber,time_integrator,
                                                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                                     Generalized_FB_with_AB2_AM3_Step_Type,
                                                     Generalized_FB_with_AB3_AM4_Step_Type,
                                                     Ratio_of_FinalTime_to_TimePeriod,specified_lY,specified_dt,
                                                     wave_type_title,wave_type_figure_title,
                                                     time_integrator_short_form,plotNumericalSolution=False,
                                                     coarsestMesh=True,plotFigures=False)) 
            else:    
                [dc[iCase], GeophysicalWaveSurfaceElevationL2ErrorNorm[iCase], 
                 GeophysicalWaveZonalVelocityL2ErrorNorm[iCase], 
                 GeophysicalWaveMeridionalVelocityL2ErrorNorm[iCase]] = (
                Main_ConvergenceTest_GeophysicalWave(
                convergence_type,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,
                problem_is_linear,periodicity,CourantNumber,time_integrator,
                LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                Generalized_FB_with_AB3_AM4_Step_Type,Ratio_of_FinalTime_to_TimePeriod,specified_lY,specified_dt,
                wave_type_title,wave_type_figure_title,time_integrator_short_form,False,False,False,
                xCell_CoarsestRectilinearMesh,yCell_CoarsestRectilinearMesh,
                GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh,
                GeophysicalWaveExactZonalVelocitiesAtCellCenters_CoarsestRectilinearMesh,
                GeophysicalWaveExactMeridionalVelocitiesAtCellCenters_CoarsestRectilinearMesh))
        if convergence_type == 'Time':
            dc[iCase] = specified_dt
    output_directory = (
    'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' + wave_type_figure_title + '_ConvergenceStudy')
    output_surface_elevation_convergence_data = output_these_variables[0]
    if output_surface_elevation_convergence_data:
        FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                    + '_NumericalSurfaceElevationConvergencePlot_' + convergence_type + '_L2ErrorNorm')  
        CR.WriteCurve1D(output_directory,dc,GeophysicalWaveSurfaceElevationL2ErrorNorm,FileName)
    output_zonal_velocity_convergence_data = output_these_variables[1]
    if output_zonal_velocity_convergence_data:        
        FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                    + '_NumericalZonalVelocityConvergencePlot_' + convergence_type + '_L2ErrorNorm')  
        CR.WriteCurve1D(output_directory,dc,GeophysicalWaveZonalVelocityL2ErrorNorm,FileName)
    output_meridional_velocity_convergence_data = output_these_variables[2]
    if output_meridional_velocity_convergence_data:              
        FileName = (wave_type_figure_title + '_' + time_integrator_short_form 
                    + '_NumericalMeridionalVelocityConvergencePlot_' + convergence_type + '_L2ErrorNorm')     
        CR.WriteCurve1D(output_directory,dc,GeophysicalWaveMeridionalVelocityL2ErrorNorm,FileName)


# In[65]:

def run_ConvergenceTest(convergence_type,problem_type,time_integrator_part=1,single_time_integrator=True,
                        single_time_integrator_index=5):
    CourantNumber = 0.45
    Ratio_of_FinalTime_to_TimePeriod = Specify_Ratio_of_FinalTime_to_TimePeriod(problem_type)
    output_these_variables = np.ones(3,dtype=bool)
    # output_these_variables[0] = output_surface_elevation_convergence_data
    # output_these_variables[1] = output_zonal_velocity_convergence_data
    # output_these_variables[2] = output_meridional_velocity_convergence_data
    [time_integrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
     Generalized_FB_with_AB3_AM4_Step_Types] = set_of_time_integrators(time_integrator_part)
    if single_time_integrator:
        i_time_integrator_lower_limit = single_time_integrator_index
        i_time_integrator_upper_limit = single_time_integrator_index + 1
    else:
        i_time_integrator_lower_limit = 0
        i_time_integrator_upper_limit = len(time_integrators)
    start_time = time.time()
    for i_time_integrator in range(i_time_integrator_lower_limit,i_time_integrator_upper_limit):
        start_time_within_loop = time.time()
        time_integrator = time_integrators[i_time_integrator]
        LF_TR_and_LF_AM3_with_FB_Feedback_Type = LF_TR_and_LF_AM3_with_FB_Feedback_Types[i_time_integrator]
        Generalized_FB_with_AB2_AM3_Step_Type = Generalized_FB_with_AB2_AM3_Step_Types[i_time_integrator]
        Generalized_FB_with_AB3_AM4_Step_Type = Generalized_FB_with_AB3_AM4_Step_Types[i_time_integrator]
        time_integrator_short_form = (
        DetermineTimeIntegratorShortForm(time_integrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                         Generalized_FB_with_AB2_AM3_Step_Type,
                                         Generalized_FB_with_AB3_AM4_Step_Type))
        ConvergenceTest_GeophysicalWave(convergence_type,problem_type,CourantNumber,time_integrator,
                                        LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                        Generalized_FB_with_AB2_AM3_Step_Type,
                                        Generalized_FB_with_AB3_AM4_Step_Type,Ratio_of_FinalTime_to_TimePeriod,
                                        output_these_variables)
        end_time_within_loop = time.time()
        elapsed_time = end_time_within_loop - start_time_within_loop
        print('The time taken by the time integrator %s is %s.' 
              %(time_integrator_short_form,(PrintDisplayTime(elapsed_time,non_integral_seconds=True)).lower()))
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print('The total elapsed time is %s.' 
          %(PrintDisplayTime(total_elapsed_time,non_integral_seconds=True)).lower())


# In[66]:

run_ConvergenceTest_CoastalKelvinWave = False
if run_ConvergenceTest_CoastalKelvinWave:
    convergence_type = 'SpaceAndTime_Short' 
    # Note that convergence_type can be 'SpaceAndTime_Short' or 'SpaceAndTime' or 'Space' or 'Time'.
    problem_type = 'Coastal_Kelvin_Wave'
    time_integrator_part = 1
    single_time_integrator = True
    single_time_integrator_index = 5
    run_ConvergenceTest(convergence_type,problem_type,time_integrator_part,single_time_integrator,
                        single_time_integrator_index)


# In[67]:

run_ConvergenceTest_InertiaGravityWave = False
if run_ConvergenceTest_InertiaGravityWave:
    convergence_type = 'SpaceAndTime_Short' 
    # Note that convergence_type can be 'SpaceAndTime_Short' or 'SpaceAndTime' or 'Space' or 'Time'.
    problem_type = 'Inertia_Gravity_Wave'
    time_integrator_part = 1
    single_time_integrator = True
    single_time_integrator_index = 5
    run_ConvergenceTest(convergence_type,problem_type,time_integrator_part,single_time_integrator,
                        single_time_integrator_index)


# In[68]:

def plot_ConvergenceData(convergence_type,problem_type,time_integrator_part=1,single_time_integrator=True,
                         single_time_integrator_index=5,plot_only_surface_elevation_data=True,
                         plot_against_numbers_of_cells_in_zonal_direction=True,usePlotly=False,
                         useMeanCurve=False):
    wave_type_title, wave_type_figure_title = DetermineFigureAndImageTitles(problem_type)
    output_directory = (
    'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' + wave_type_figure_title + '_ConvergenceStudy')
    legendposition = 'upper left'
    if convergence_type == 'Time':
        xLabel = 'Time Step'
    elif (convergence_type == 'Space' or convergence_type == 'SpaceAndTime'
          or convergence_type == 'SpaceAndTime_Short'):
        if plot_against_numbers_of_cells_in_zonal_direction:
            xLabel = 'Number of Cells in Zonal Direction'
            legendposition = 'upper right'
        else:
            xLabel = 'Cell Width'
    if convergence_type == 'Space':
        Title = wave_type_title + ': Convergence Plot in Space'
    elif convergence_type == 'Time':
        Title = wave_type_title + ': Convergence Plot in Time'
    elif convergence_type == 'SpaceAndTime' or convergence_type == 'SpaceAndTime_Short':
        Title = wave_type_title + ': Convergence Plot in Space and Time'
    plot_these_variables = np.ones(3,dtype=bool)
    if plot_only_surface_elevation_data:
        plot_these_variables[1:] = False
    # plot_these_variables[0] = plot_surface_elevation_convergence_data
    # plot_these_variables[1] = plot_zonal_velocity_convergence_data
    # plot_these_variables[2] = plot_meridional_velocity_convergence_data
    [time_integrators, LF_TR_and_LF_AM3_with_FB_Feedback_Types, Generalized_FB_with_AB2_AM3_Step_Types,
     Generalized_FB_with_AB3_AM4_Step_Types] = set_of_time_integrators(time_integrator_part)
    if single_time_integrator:
        i_time_integrator_lower_limit = single_time_integrator_index
        i_time_integrator_upper_limit = single_time_integrator_index + 1
    else:
        i_time_integrator_lower_limit = 0
        i_time_integrator_upper_limit = len(time_integrators)
    for i_time_integrator in range(i_time_integrator_lower_limit,i_time_integrator_upper_limit):
        time_integrator = time_integrators[i_time_integrator]
        LF_TR_and_LF_AM3_with_FB_Feedback_Type = LF_TR_and_LF_AM3_with_FB_Feedback_Types[i_time_integrator]
        Generalized_FB_with_AB2_AM3_Step_Type = Generalized_FB_with_AB2_AM3_Step_Types[i_time_integrator]
        Generalized_FB_with_AB3_AM4_Step_Type = Generalized_FB_with_AB3_AM4_Step_Types[i_time_integrator]
        time_integrator_short_form = (
        DetermineTimeIntegratorShortForm(time_integrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                         Generalized_FB_with_AB2_AM3_Step_Type,
                                         Generalized_FB_with_AB3_AM4_Step_Type))
        plot_surface_elevation_convergence_data = plot_these_variables[0]
        if plot_surface_elevation_convergence_data:  
            yLabel = 'L2 Error Norm of SSH'
            # Note that yLabel was initially specified as 'L2 Error Norm of Numerical Surface Elevation'.
            FigureTitle = (wave_type_figure_title + '_' + time_integrator_short_form
                           + '_NumericalSurfaceElevationConvergencePlot_' + convergence_type + '_L2ErrorNorm') 
            dc, SurfaceElevationL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
            if ((convergence_type == 'Space' or convergence_type == 'SpaceAndTime'
                 or convergence_type == 'SpaceAndTime_Short') 
                and plot_against_numbers_of_cells_in_zonal_direction):
                dc = CR.RoundArray(SpecifyZonalExtent(problem_type)/dc)
            if convergence_type == 'Space':
                yLabel = 'Difference in ' + yLabel
                nCases = len(dc)
                dc = dc[1:nCases]
                SurfaceElevationL2ErrorNormDifference = np.zeros(nCases-1)
                for iCase in range(0,nCases-1):
                    SurfaceElevationL2ErrorNormDifference[iCase] = (
                    SurfaceElevationL2ErrorNorm[iCase] - SurfaceElevationL2ErrorNorm[iCase+1])
                SurfaceElevationL2ErrorNorm = SurfaceElevationL2ErrorNormDifference
            if useMeanCurve:
                A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
                m, c = np.linalg.lstsq(A,np.log10(SurfaceElevationL2ErrorNorm))[0]
                SurfaceElevationL2ErrorNorm_MeanCurve = m*(np.log10(dc)) + c
                SurfaceElevationL2ErrorNorm_MeanCurve = 10.0**SurfaceElevationL2ErrorNorm_MeanCurve    
                FigureTitle += '_MeanCurve' 
                legends = ['L2 Error Norm of SSH','Best Fit Line: Slope is %.2f' %m]
                CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,SurfaceElevationL2ErrorNorm,
                                                    SurfaceElevationL2ErrorNorm_MeanCurve,[2.0,2.0],[' ','-'],
                                                    ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],
                                                    [17.5,17.5],[10.0,10.0],[15.0,15.0],legends,17.5,
                                                    legendposition,Title,20.0,True,FigureTitle,False,
                                                    drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True,
                                                    FigureFormat='pdf')
            else:
                if not(usePlotly):
                    CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',dc,SurfaceElevationL2ErrorNorm,2.0,'-',
                                             'k','s',10.0,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                                             Title,20.0,True,FigureTitle,False,fig_size=[9.25,9.25],
                                             useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                             drawMinorGrid=True,FigureFormat='pdf') 
                else:
                    CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',dc,SurfaceElevationL2ErrorNorm,'black',
                                               2.0,'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,
                                               25.0,True,FigureTitle,False,fig_size=[700.0,700.0],
                                               FigureFormat='pdf')  
        plot_zonal_velocity_convergence_data = plot_these_variables[1]
        if plot_zonal_velocity_convergence_data: 
            yLabel = 'L2 Error Norm of Numerical Zonal Velocity'
            FigureTitle = (wave_type_figure_title + '_' + time_integrator_short_form 
                           + '_NumericalZonalVelocityConvergencePlot_' + convergence_type + '_L2ErrorNorm') 
            dc, ZonalVelocityL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
            if ((convergence_type == 'Space' or convergence_type == 'SpaceAndTime'
                 or convergence_type == 'SpaceAndTime_Short') 
                and plot_against_numbers_of_cells_in_zonal_direction):
                dc = CR.RoundArray(SpecifyZonalExtent(problem_type)/dc)
            if convergence_type == 'Space':
                yLabel = 'Difference in ' + yLabel
                nCases = len(dc)
                dc = dc[1:nCases]
                ZonalVelocityL2ErrorNormDifference = np.zeros(nCases-1)
                for iCase in range(0,nCases-1):
                    ZonalVelocityL2ErrorNormDifference[iCase] = (
                    ZonalVelocityL2ErrorNorm[iCase] - ZonalVelocityL2ErrorNorm[iCase+1])
                ZonalVelocityL2ErrorNorm = ZonalVelocityL2ErrorNormDifference
            if useMeanCurve:
                A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
                m, c = np.linalg.lstsq(A,np.log10(ZonalVelocityL2ErrorNorm))[0]
                ZonalVelocityL2ErrorNorm_MeanCurve = m*(np.log10(dc)) + c
                ZonalVelocityL2ErrorNorm_MeanCurve = 10.0**ZonalVelocityL2ErrorNorm_MeanCurve    
                FigureTitle += '_MeanCurve' 
                legends = ['L2 Error Norm of Zonal Velocity','Best Fit Line: Slope is %.2f' %m]
                CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,ZonalVelocityL2ErrorNorm,
                                                    ZonalVelocityL2ErrorNorm_MeanCurve,[2.0,2.0],[' ','-'],
                                                    ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],
                                                    [17.5,17.5],[10.0,10.0],[15.0,15.0],legends,17.5,
                                                    legendposition,Title,20.0,True,FigureTitle,False,
                                                    drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True,
                                                    FigureFormat='pdf')
            else:
                if not(usePlotly):
                    CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',dc,ZonalVelocityL2ErrorNorm,2.0,'-','k',
                                             's',10.0,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],myTitle,
                                             20.0,True,FigureTitle,False,fig_size=[9.25,9.25],
                                             useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                             drawMinorGrid=True,FigureFormat='pdf') 
                else:
                    CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',dc,ZonalVelocityL2ErrorNorm,'black',2.0,
                                               'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,25.0,
                                               True,FigureTitle,False,fig_size=[700.0,700.0],FigureFormat='pdf')
        plot_meridional_velocity_convergence_data = plot_these_variables[2]
        if plot_meridional_velocity_convergence_data:            
            yLabel = 'L2 Error Norm of Numerical Meridional Velocity'
            FigureTitle = (wave_type_figure_title + '_' + time_integrator_short_form 
                           + '_NumericalMeridionalVelocityConvergencePlot_' + convergence_type + '_L2ErrorNorm')
            dc, MeridionalVelocityL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')            
            if ((convergence_type == 'Space' or convergence_type == 'SpaceAndTime'
                 or convergence_type == 'SpaceAndTime_Short') 
                and plot_against_numbers_of_cells_in_zonal_direction):
                dc = CR.RoundArray(SpecifyZonalExtent(problem_type)/dc)
            if convergence_type == 'Space':
                yLabel = 'Difference in ' + yLabel
                nCases = len(dc)
                dc = dc[1:nCases]
                MeridionalVelocityL2ErrorNormDifference = np.zeros(nCases-1)
                for iCase in range(0,nCases-1):
                    MeridionalVelocityL2ErrorNormDifference[iCase] = (
                    MeridionalVelocityL2ErrorNorm[iCase] - MeridionalVelocityL2ErrorNorm[iCase+1])
                MeridionalVelocityL2ErrorNorm = MeridionalVelocityL2ErrorNormDifference                
            if useMeanCurve:
                A = np.vstack([np.log10(dc),np.ones(len(dc))]).T
                m, c = np.linalg.lstsq(A,np.log10(MeridionalVelocityL2ErrorNorm))[0]
                MeridionalVelocityL2ErrorNorm_MeanCurve = m*(np.log10(dc)) + c
                MeridionalVelocityL2ErrorNorm_MeanCurve = 10.0**MeridionalVelocityL2ErrorNorm_MeanCurve    
                FigureTitle += '_MeanCurve' 
                legends = ['L2 Error Norm of Meridional Velocity','Best Fit Line: Slope is %.2f' %m]
                CR.PythonConvergencePlot1DSaveAsPNG(output_directory,'log-log',dc,MeridionalVelocityL2ErrorNorm,
                                                    MeridionalVelocityL2ErrorNorm_MeanCurve,[2.0,2.0],[' ','-'],
                                                    ['k','k'],[True,False],['s','s'],[10.0,10.0],[xLabel,yLabel],
                                                    [17.5,17.5],[10.0,10.0],[15.0,15.0],legends,17.5,
                                                    legendposition,Title,20.0,True,FigureTitle,False,
                                                    drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True,
                                                    FigureFormat='pdf')
            else:
                if not(usePlotly):
                    CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',dc,MeridionalVelocityL2ErrorNorm,2.0,'-',
                                             'k','s',10.0,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                                             myTitle,20.0,True,FigureTitle,False,fig_size=[9.25,9.25],
                                             useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                             drawMinorGrid=True,FigureFormat='pdf')
                else:
                    CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',dc,MeridionalVelocityL2ErrorNorm,
                                               'black',2.0,'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],
                                               Title,25.0,True,FigureTitle,False,fig_size=[700.0,700.0],
                                               FigureFormat='pdf')


# In[69]:

plot_ConvergenceData_CoastalKelvinWave = False
if plot_ConvergenceData_CoastalKelvinWave:
    convergence_type = 'SpaceAndTime_Short' 
    # Note that convergence_type can be 'SpaceAndTime' or 'SpaceAndTime_Short' or 'Space' or 'Time'.
    problem_type = 'Coastal_Kelvin_Wave'
    time_integrator_part = 1
    single_time_integrator = True
    single_time_integrator_index = 5
    plot_only_surface_elevation_data = True
    plot_against_numbers_of_cells_in_zonal_direction = True
    usePlotly = False
    useMeanCurve = True
    plot_ConvergenceData(convergence_type,problem_type,time_integrator_part,single_time_integrator,
                         single_time_integrator_index,plot_only_surface_elevation_data,
                         plot_against_numbers_of_cells_in_zonal_direction,usePlotly,useMeanCurve)


# In[70]:

plot_ConvergenceData_InertiaGravityWave = False
if plot_ConvergenceData_InertiaGravityWave:
    convergence_type = 'SpaceAndTime_Short' 
    # Note that convergence_type can be 'SpaceAndTime' or 'SpaceAndTime_Short' or 'Space' or 'Time'.
    problem_type = 'Inertia_Gravity_Wave'
    time_integrator_part = 1
    single_time_integrator = True
    single_time_integrator_index = 5
    plot_only_surface_elevation_data = True
    plot_against_numbers_of_cells_in_zonal_direction = True
    usePlotly = False
    useMeanCurve = True
    plot_ConvergenceData(convergence_type,problem_type,time_integrator_part,single_time_integrator,
                         single_time_integrator_index,plot_only_surface_elevation_data,
                         plot_against_numbers_of_cells_in_zonal_direction,usePlotly,useMeanCurve)