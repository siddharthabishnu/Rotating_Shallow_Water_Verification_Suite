
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
from IPython.utils import io
with io.capture_output() as captured: 
    import Common_Routines as CR
    import MPAS_O_Mode_Init
    import MPAS_O_Shared
    import MPAS_O_Mesh_Interpolation_Routines as MOMIR
    import MPAS_O_Mode_Forward


# In[2]:

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


# In[3]:

def PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,x,y,phi,nContours,useGivenColorBarLimits,
                                                        ColorBarLimits,nColorBarTicks,colormap,
                                                        colorbarfontsize,labels,labelfontsizes,labelpads,
                                                        tickfontsizes,title,titlefontsize,SaveAsPDF,FigureTitle,
                                                        Show,cbarlabelformat='%.2g'):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    fig = plt.figure(figsize=(10,10)) # Create a figure object
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
        plt.savefig(FigureTitle+'.png',format='png')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[4]:

def PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,phi,nContours,
                                                        useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,
                                                        colormap,colorbarfontsize,labels,labelfontsizes,labelpads,
                                                        tickfontsizes,title,titlefontsize,SaveAsPDF,FigureTitle,
                                                        Show,cbarlabelformat='%.2g'):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    fig = plt.figure(figsize=(10,10)) # Create a figure object
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
    if SaveAsPDF:
        plt.savefig(FigureTitle+'.png',format='png')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[5]:

def PrintDisplayTime(time,displaytime):
    hours = np.floor(time/3600.0)
    remainingtime = np.mod(time,3600.0)
    minutes = np.floor(remainingtime/60.0)
    seconds = np.mod(remainingtime,60.0)
    if displaytime:
        print('The display time is %1d hours %2d minutes %2d seconds.' %(hours,minutes,seconds))
    return hours, minutes, seconds


# In[6]:

hours, minutes, seconds = PrintDisplayTime(10983.0,True)


# In[7]:

def CoastalKelvinWaveFunctionalForm(lY,y,returnAmplitude=False):
    yCenter = 0.5*lY
    etaHat = 0.001
    eta = etaHat*np.sin(2.0*np.pi*y/lY)
    if returnAmplitude:
        return etaHat
    else:
        return eta


# In[8]:

def DetermineCoastalKelvinWaveExactSurfaceElevation(H,c,R,lY,x,y,time):
    CoastalKelvinWaveExactSurfaceElevation = -H*CoastalKelvinWaveFunctionalForm(lY,y+c*time)*np.exp(-x/R)
    return CoastalKelvinWaveExactSurfaceElevation


# In[9]:

def DetermineCoastalKelvinWaveExactZonalVelocity():
    CoastalKelvinWaveExactZonalVelocity = 0.0
    return CoastalKelvinWaveExactZonalVelocity


# In[10]:

def DetermineCoastalKelvinWaveExactMeridionalVelocity(c,R,lY,x,y,time):
    CoastalKelvinWaveExactMeridionalVelocity = c*CoastalKelvinWaveFunctionalForm(lY,y+c*time)*np.exp(-x/R)
    return CoastalKelvinWaveExactMeridionalVelocity


# In[11]:

def DetermineTimeStepForGivenCourantNumber(CourantNumber,dx,WaveSpeed):
    dt = CourantNumber*dx/WaveSpeed
    return dt


# In[12]:

do_DetermineTimeStepForGivenCourantNumber = False
if do_DetermineTimeStepForGivenCourantNumber:    
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       mesh_type,problem_type,problem_is_linear,periodicity)
    CourantNumber = 0.36
    dx = myMPAS_O.dcEdge[0]
    WaveSpeed = myMPAS_O.myNamelist.config_wave_speed
    dt = DetermineTimeStepForGivenCourantNumber(CourantNumber,dx,WaveSpeed)
    print('The timestep for Courant Number %.2f is %.2f seconds.' %(CourantNumber,dt))


# In[13]:

def DetermineGeophysicalWaveExactSolutions(myMPAS_O,DetermineGeophysicalWaveExactSurfaceElevation,
                                           DetermineGeophysicalWaveExactZonalVelocity,
                                           DetermineGeophysicalWaveExactMeridionalVelocity):
    H = myMPAS_O.myNamelist.config_mean_depth
    c = myMPAS_O.myNamelist.config_wave_speed
    lY = myMPAS_O.lY
    time = myMPAS_O.time
    GeophysicalWaveExactSurfaceElevations = np.zeros(myMPAS_O.nCells) 
    GeophysicalWaveExactZonalVelocities = np.zeros(myMPAS_O.nEdges)
    GeophysicalWaveExactMeridionalVelocities = np.zeros(myMPAS_O.nEdges) 
    for iCell in range(0,myMPAS_O.nCells):
        fCell = myMPAS_O.fCell[iCell]
        RCell = c/fCell
        xCell = myMPAS_O.xCell[iCell]
        yCell = myMPAS_O.yCell[iCell]
        if myMPAS_O.myNamelist.config_problem_type == 'Coastal_Kelvin_Wave':
            GeophysicalWaveExactSurfaceElevations[iCell] = (
            DetermineGeophysicalWaveExactSurfaceElevation(H,c,RCell,lY,xCell,yCell,time))
    for iEdge in range(0,myMPAS_O.nEdges):
        fEdge = myMPAS_O.fEdge[iEdge]
        REdge = c/fEdge
        xEdge = myMPAS_O.xEdge[iEdge]
        yEdge = myMPAS_O.yEdge[iEdge]
        if myMPAS_O.myNamelist.config_problem_type == 'Coastal_Kelvin_Wave':
            GeophysicalWaveExactZonalVelocities[iEdge] = DetermineGeophysicalWaveExactZonalVelocity()
            GeophysicalWaveExactMeridionalVelocities[iEdge] = (
            DetermineCoastalKelvinWaveExactMeridionalVelocity(c,REdge,lY,xEdge,yEdge,time))
    return [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocities, 
            GeophysicalWaveExactMeridionalVelocities]


# In[14]:

def DetermineGeophysicalWaveExactSolutionsOnCoarsestRectilinearMesh(
myMPAS_O,xCell_CoarsestRectilinearMesh,yCell_CoarsestRectilinearMesh,DetermineGeophysicalWaveExactSurfaceElevation,
DetermineGeophysicalWaveExactZonalVelocity,DetermineGeophysicalWaveExactMeridionalVelocity):
    H = myMPAS_O.myNamelist.config_mean_depth
    c = myMPAS_O.myNamelist.config_wave_speed
    lY = myMPAS_O.lY
    time = myMPAS_O.time
    GeophysicalWaveExactSurfaceElevations = np.zeros(myMPAS_O.nCells) 
    GeophysicalWaveExactZonalVelocities = np.zeros(myMPAS_O.nCells)
    GeophysicalWaveExactMeridionalVelocities = np.zeros(myMPAS_O.nCells) 
    for iCell in range(0,myMPAS_O.nCells):
        fCell = myMPAS_O.fCell[iCell]
        RCell = c/fCell
        xCell = xCell_CoarsestRectilinearMesh[iCell]
        yCell = yCell_CoarsestRectilinearMesh[iCell]
        if myMPAS_O.myNamelist.config_problem_type == 'Coastal_Kelvin_Wave':
            GeophysicalWaveExactSurfaceElevations[iCell] = (
            DetermineGeophysicalWaveExactSurfaceElevation(H,c,RCell,lY,xCell,yCell,time))
            GeophysicalWaveExactZonalVelocities[iCell] = DetermineGeophysicalWaveExactZonalVelocity()
            GeophysicalWaveExactMeridionalVelocities[iCell] = (
            DetermineCoastalKelvinWaveExactMeridionalVelocity(c,RCell,lY,xCell,yCell,time))
    return [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocities, 
            GeophysicalWaveExactMeridionalVelocities]


# In[15]:

def DetermineCoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters():
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    periodicity='NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                       problem_type,problem_is_linear,periodicity)    
    H = myMPAS_O.myNamelist.config_mean_depth
    lY = myMPAS_O.lY
    yCell = 0.0
    c = myMPAS_O.myNamelist.config_wave_speed
    time = 0.0
    xCell = 0.5*myMPAS_O.gridSpacingMagnitude
    fCell = myMPAS_O.fCell[0]
    RCell = c/fCell
    CoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters = (
    -H*CoastalKelvinWaveFunctionalForm(lY,yCell+c*time,returnAmplitude=True)*np.exp(-xCell/RCell))
    print('The amplitude of the coastal Kelvin wave surface elevation at cell centers is %.6f.'
          %abs(CoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters))


# In[16]:

do_DetermineCoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters = False
if do_DetermineCoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters:
    DetermineCoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters()


# In[17]:

def testDetermineGeophysicalWaveExactSolutions(mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                               problem_type,problem_is_linear,CourantNumber,plotFigures):
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                       problem_type,problem_is_linear,periodicity='NonPeriodic_x',
                                       CourantNumber=CourantNumber,useCourantNumberToDetermineTimeStep=True)
    myMPAS_O.myNamelist.config_dt = round(myMPAS_O.myNamelist.config_dt)
    dt = myMPAS_O.myNamelist.config_dt 
    nTime = 250 + 1
    nDumpFrequency = 10
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves'
    for iTime in range(0,nTime):
        myMPAS_O.time = float(iTime)*dt
        if np.mod(iTime,nDumpFrequency) == 0.0 and plotFigures:
            if problem_type == 'Coastal_Kelvin_Wave':
                [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocities, 
                 GeophysicalWaveExactMeridionalVelocities] = (
                DetermineGeophysicalWaveExactSolutions(myMPAS_O,DetermineCoastalKelvinWaveExactSurfaceElevation,
                                                       DetermineCoastalKelvinWaveExactZonalVelocity,
                                                       DetermineCoastalKelvinWaveExactMeridionalVelocity))
                wave_type_title = 'Coastal Kelvin Wave'
                wave_type_figure_title = 'CoastalKelvinWave'
                plotExactZonalVelocity = False
            else:
                plotExactZonalVelocity = True
            hours, minutes, seconds = PrintDisplayTime(myMPAS_O.time,False)
            if problem_type == 'Coastal_Kelvin_Wave':
                ColorBarLimits = [-0.97531,0.97531]
            Title = (wave_type_title + ': Exact Surface Elevation after\n%d Hours %2d Minutes %2d Seconds'
                     %(hours,minutes,seconds))
            FigureTitle = wave_type_figure_title + '_ExactSurfaceElevation_' + '%3.3d' %iTime
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                GeophysicalWaveExactSurfaceElevations,300,True,
                                                                ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                cbarlabelformat='%.5f')
            if plotExactZonalVelocity:               
                GeophysicalWaveExactZonalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                myMPAS_O,GeophysicalWaveExactZonalVelocities)                
                Title = (wave_type_title + ': Exact Zonal Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds))
                FigureTitle = wave_type_figure_title + '_ExactZonalVelocity_' + '%3.3d' %iTime
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                    GeophysicalWaveExactZonalVelocities,300,False,
                                                                    [0.0,0.0],6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
            if problem_type == 'Coastal_Kelvin_Wave':
                ColorBarLimits = [-0.1,0.1] 
            GeophysicalWaveExactMeridionalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
            myMPAS_O,GeophysicalWaveExactMeridionalVelocities)
            Title = (wave_type_title + ': Exact Meridional Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                     %(hours,minutes,seconds))
            FigureTitle = wave_type_figure_title + '_ExactMeridionalVelocity_' + '%3.3d' %iTime
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                GeophysicalWaveExactMeridionalVelocities,300,True,
                                                                ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                cbarlabelformat='%.5f')


# In[18]:

do_testDetermineGeophysicalWaveExactSolutions = False
if do_testDetermineGeophysicalWaveExactSolutions:    
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    CourantNumber = 0.36
    plotFigures = True    
    testDetermineGeophysicalWaveExactSolutions(mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                               problem_type,problem_is_linear,CourantNumber,plotFigures)


# In[19]:

def Main(mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,problem_is_linear,CourantNumber,
         time_integrator,output_these_variables,plot_these_variables,display_range_of_variables):
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                       problem_type,problem_is_linear,periodicity='NonPeriodic_x',
                                       CourantNumber=CourantNumber,useCourantNumberToDetermineTimeStep=True,
                                       time_integrator=time_integrator,
                                       specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False)
    myMPAS_O.myNamelist.config_dt = round(myMPAS_O.myNamelist.config_dt)
    MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    dt = myMPAS_O.myNamelist.config_dt 
    nTime = 250 + 1
    nDumpFrequency = 10
    compute_these_variables = np.zeros(8,dtype=bool)
    compute_these_variables[4] = True # compute_these_variables[4] = compute_tangentialVelocity
    if problem_type == 'Coastal_Kelvin_Wave':
        wave_type_title = 'Coastal Kelvin Wave'
        wave_type_figure_title = 'CoastalKelvinWave'
    if problem_is_linear:
        wave_nature = 'Linear'
    else: 
        wave_nature = 'NonLinear'
    if time_integrator == 'forward_backward_predictor':    
        time_integrator_short_form = 'FBP'
    max_yEdge_index = np.argmax(myMPAS_O.yEdge)
    xEdge_Plot = CR.RemoveElementFrom1DArray(myMPAS_O.xEdge,max_yEdge_index)
    yEdge_Plot = CR.RemoveElementFrom1DArray(myMPAS_O.yEdge,max_yEdge_index)
    for iTime in range(0,nTime):
        myMPAS_O.time = float(iTime)*dt
        hours, minutes, seconds = PrintDisplayTime(myMPAS_O.time,False)
        printProgress = False
        if printProgress:
            print('Computing Numerical Solution after %2d Hours %2d Minutes %2d Seconds!'
                  %(hours,minutes,seconds))        
        if np.mod(iTime,nDumpFrequency) == 0.0:
            numericalZonalVelocities = np.zeros(myMPAS_O.nEdges)
            numericalMeridionalVelocities = np.zeros(myMPAS_O.nEdges)
            if problem_type == 'Coastal_Kelvin_Wave':
                [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocities,
                 GeophysicalWaveExactMeridionalVelocities] = (
                DetermineGeophysicalWaveExactSolutions(myMPAS_O,DetermineCoastalKelvinWaveExactSurfaceElevation,
                                                       DetermineCoastalKelvinWaveExactZonalVelocity,
                                                       DetermineCoastalKelvinWaveExactMeridionalVelocity))
            GeophysicalWaveExactVelocities = np.zeros((myMPAS_O.nEdges,2))
            GeophysicalWaveExactVelocities[:,0] = GeophysicalWaveExactZonalVelocities[:] 
            GeophysicalWaveExactVelocities[:,1] = GeophysicalWaveExactMeridionalVelocities[:]   
            GeophysicalWaveExactNormalVelocities, GeophysicalWaveExactTangentialVelocities = (
            ComputeNormalAndTangentialComponentsAtEdge(GeophysicalWaveExactVelocities,myMPAS_O.angleEdge,'both'))
            if iTime == 0.0: # Specify initial conditions
                myMPAS_O.sshCurrent[:] = GeophysicalWaveExactSurfaceElevations[:]
                myMPAS_O.normalVelocityCurrent[:,0] = GeophysicalWaveExactNormalVelocities
            MPAS_O_Shared.ocn_diagnostic_solve(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent,
                                               compute_these_variables)
            for iEdge in range(0,myMPAS_O.nEdges):
                if myMPAS_O.boundaryEdge[iEdge] == 1.0: # i.e. if the edge is along a non-periodic boundary
                    myMPAS_O.tangentialVelocity[iEdge,0] = GeophysicalWaveExactTangentialVelocities[iEdge]   
                    numericalZonalVelocities[iEdge] = GeophysicalWaveExactZonalVelocities[iEdge]
                    numericalMeridionalVelocities[iEdge] = GeophysicalWaveExactMeridionalVelocities[iEdge]
                else:
                    numericalZonalVelocities[iEdge] = (
                    (myMPAS_O.normalVelocityCurrent[iEdge,0]*np.cos(myMPAS_O.angleEdge[iEdge])
                     - myMPAS_O.tangentialVelocity[iEdge,0]*np.sin(myMPAS_O.angleEdge[iEdge])))
                    numericalMeridionalVelocities[iEdge] = (
                    (myMPAS_O.normalVelocityCurrent[iEdge,0]*np.sin(myMPAS_O.angleEdge[iEdge])
                     + myMPAS_O.tangentialVelocity[iEdge,0]*np.cos(myMPAS_O.angleEdge[iEdge])))
            if plot_these_variables[0]: # if plotExactSurfaceElevation:
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_ExactSurfaceElevation')
                if problem_type == 'Coastal_Kelvin_Wave':
                    ColorBarLimits = [-0.97531,0.97531]
                Title = (wave_type_title + ': Exact Surface Elevation after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))         
                FigureTitle = wave_type_figure_title + '_ExactSurfaceElevation_' + '%3.3d' %iTime     
                if output_these_variables[0]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveExactSurfaceElevations,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                    GeophysicalWaveExactSurfaceElevations,300,
                                                                    True,ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,
                                                                    False,cbarlabelformat='%.5f')
            if plot_these_variables[1]: # if plotExactZonalVelocity:
                GeophysicalWaveExactZonalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                myMPAS_O,GeophysicalWaveExactZonalVelocities)
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_ExactZonalVelocity')                
                Title = (wave_type_title + ': Exact Zonal Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds)) 
                FigureTitle = wave_type_figure_title + '_ExactZonalVelocity_' + '%3.3d' %iTime
                if output_these_variables[1]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveExactZonalVelocities,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                    GeophysicalWaveExactZonalVelocities,300,False,
                                                                    [0.0,0.0],6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
            if plot_these_variables[2]: # if plotExactMeridionalVelocity:
                GeophysicalWaveExactMeridionalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                myMPAS_O,GeophysicalWaveExactMeridionalVelocities)
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_ExactMeridionalVelocity')
                if problem_type == 'Coastal_Kelvin_Wave':
                    ColorBarLimits = [-0.1,0.1]
                Title = (wave_type_title + ': Exact Meridional Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds))                 
                FigureTitle = wave_type_figure_title + '_ExactMeridionalVelocity_' + '%3.3d' %iTime       
                if output_these_variables[2]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  GeophysicalWaveExactMeridionalVelocities,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                    GeophysicalWaveExactMeridionalVelocities,300,
                                                                    True,ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
            if plot_these_variables[3]: # if plotExactNormalVelocity:
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_ExactNormalVelocity')                
                if problem_type == 'Coastal_Kelvin_Wave':
                    ColorBarLimits = [-0.08553,0.08553]
                Title = (wave_type_title + ': Exact Normal Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds)) 
                FigureTitle = wave_type_figure_title + '_ExactNormalVelocity_' + '%3.3d' %iTime
                if output_these_variables[3]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  GeophysicalWaveExactNormalVelocities,FigureTitle)
                GeophysicalWaveExactNormalVelocities_Plot = (
                CR.RemoveElementFrom1DArray(GeophysicalWaveExactNormalVelocities,max_yEdge_index))
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,xEdge_Plot,yEdge_Plot,
                                                                    GeophysicalWaveExactNormalVelocities_Plot,300,
                                                                    True,ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == 0:
                        ExactNormalVelocity_Min = min(GeophysicalWaveExactNormalVelocities)
                        ExactNormalVelocity_Max = max(GeophysicalWaveExactNormalVelocities)
                    else:
                        ExactNormalVelocity_Min = min(ExactNormalVelocity_Min,
                                                      min(GeophysicalWaveExactNormalVelocities))
                        ExactNormalVelocity_Max = max(ExactNormalVelocity_Max,
                                                      max(GeophysicalWaveExactNormalVelocities))
                    if iTime == nTime - 1:
                        print('The range of exact normal velocity is [%.6f,%.6f].'
                              %(ExactNormalVelocity_Min,ExactNormalVelocity_Max))                
            if plot_these_variables[4]: # if plotExactTangentialVelocity:
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_ExactTangentialVelocity')                
                if problem_type == 'Coastal_Kelvin_Wave':
                    ColorBarLimits = [-0.1,0.1]
                Title = (wave_type_title + ': Exact Tangential Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds)) 
                FigureTitle = wave_type_figure_title + '_ExactTangentialVelocity_' + '%3.3d' %iTime
                if output_these_variables[4]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  GeophysicalWaveExactTangentialVelocities,FigureTitle)
                GeophysicalWaveExactTangentialVelocities_Plot = (
                CR.RemoveElementFrom1DArray(GeophysicalWaveExactTangentialVelocities,max_yEdge_index))
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,xEdge_Plot,yEdge_Plot,
                                                                    GeophysicalWaveExactTangentialVelocities_Plot,
                                                                    300,True,ColorBarLimits,6,plt.cm.seismic,
                                                                    13.75,['x (km)','y (km)'],[17.5,17.5],
                                                                    [10.0,10.0],[15.0,15.0],Title,20.0,True,
                                                                    FigureTitle,False,cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == 0:
                        ExactTangentialVelocity_Min = min(GeophysicalWaveExactTangentialVelocities)
                        ExactTangentialVelocity_Max = max(GeophysicalWaveExactTangentialVelocities)
                    else:
                        ExactTangentialVelocity_Min = min(ExactTangentialVelocity_Min,
                                                          min(GeophysicalWaveExactTangentialVelocities))
                        ExactTangentialVelocity_Max = max(ExactTangentialVelocity_Max,
                                                          max(GeophysicalWaveExactTangentialVelocities))
                    if iTime == nTime - 1:
                        print('The range of exact tangential velocity is [%.6f,%.6f].'
                              %(ExactTangentialVelocity_Min,ExactTangentialVelocity_Max))
            if plot_these_variables[5]: # if plotNumericalSurfaceElevation:
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_NumericalSurfaceElevation') 
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.9751,0.9751]
                Title = (wave_type_title + ': Numerical Surface Elevation after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NumericalSurfaceElevation_' + '%3.3d' %iTime)    
                if output_these_variables[5]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  myMPAS_O.sshCurrent,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,myMPAS_O.sshCurrent,
                                                                    300,True,ColorBarLimits,6,plt.cm.seismic,
                                                                    13.75,['x (km)','y (km)'],[17.5,17.5],
                                                                    [10.0,10.0],[15.0,15.0],Title,20.0,True,
                                                                    FigureTitle,False,cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == 0:
                        NumericalSurfaceElevation_Min = min(myMPAS_O.sshCurrent)
                        NumericalSurfaceElevation_Max = max(myMPAS_O.sshCurrent)
                    else:
                        NumericalSurfaceElevation_Min = min(NumericalSurfaceElevation_Min,
                                                            min(myMPAS_O.sshCurrent))
                        NumericalSurfaceElevation_Max = max(NumericalSurfaceElevation_Max,
                                                            max(myMPAS_O.sshCurrent))
                    if iTime == nTime - 1:
                        print('The range of numerical surface elevation is [%.6f,%.6f].'
                              %(NumericalSurfaceElevation_Min,NumericalSurfaceElevation_Max))
            if plot_these_variables[6]: # if plotNumericalZonalVelocity:
                numericalZonalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                myMPAS_O,numericalZonalVelocities)
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_NumericalZonalVelocity')
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.00073,0.00073]
                Title = (wave_type_title + ': Numerical Zonal Velocity after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NumericalZonalVelocity_' + '%3.3d' %iTime)        
                if output_these_variables[6]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  numericalZonalVelocities,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                    numericalZonalVelocities,300,True,
                                                                    ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == 0:
                        NumericalZonalVelocity_Min = min(numericalZonalVelocities)
                        NumericalZonalVelocity_Max = max(numericalZonalVelocities)
                    else:
                        NumericalZonalVelocity_Min = min(NumericalZonalVelocity_Min,min(numericalZonalVelocities))
                        NumericalZonalVelocity_Max = max(NumericalZonalVelocity_Max,max(numericalZonalVelocities))
                    if iTime == nTime - 1:
                        print('The range of numerical zonal velocity is [%.6f,%.6f].'
                              %(NumericalZonalVelocity_Min,NumericalZonalVelocity_Max))
            if plot_these_variables[7]: # if plotNumericalMeridionalVelocity:  
                numericalMeridionalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                myMPAS_O,numericalMeridionalVelocities)
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_NumericalMeridionalVelocity')
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.1,0.1]
                Title = (wave_type_title  
                         + ': Numerical Meridional Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds))
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NumericalMeridionalVelocity_' + '%3.3d' %iTime)
                if output_these_variables[7]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  numericalMeridionalVelocities,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                    numericalMeridionalVelocities,300,True,
                                                                    ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == 0:
                        NumericalMeridionalVelocity_Min = min(numericalMeridionalVelocities)
                        NumericalMeridionalVelocity_Max = max(numericalMeridionalVelocities)
                    else:
                        NumericalMeridionalVelocity_Min = min(NumericalMeridionalVelocity_Min,
                                                              min(numericalMeridionalVelocities))
                        NumericalMeridionalVelocity_Max = max(NumericalMeridionalVelocity_Max,
                                                              max(numericalMeridionalVelocities))
                    if iTime == nTime - 1:
                        print('The range of numerical meridional velocity is [%.6f,%.6f].'
                              %(NumericalMeridionalVelocity_Min,NumericalMeridionalVelocity_Max))
            if plot_these_variables[8]: # if plotNumericalNormalVelocity:
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_NumericalNormalVelocity')
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.08553,0.08553]
                Title = (wave_type_title + ': Numerical Normal Velocity after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NumericalNormalVelocity_' + '%3.3d' %iTime)
                if output_these_variables[8]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  myMPAS_O.normalVelocityCurrent[:,0],FigureTitle)
                numericalNormalVelocities_Plot = (
                CR.RemoveElementFrom1DArray(myMPAS_O.normalVelocityCurrent[:,0],max_yEdge_index))
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,xEdge_Plot,yEdge_Plot,
                                                                    numericalNormalVelocities_Plot,300,True,
                                                                    ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == 0:
                        NumericalNormalVelocity_Min = min(myMPAS_O.normalVelocityCurrent[:,0])
                        NumericalNormalVelocity_Max = max(myMPAS_O.normalVelocityCurrent[:,0])
                    else:
                        NumericalNormalVelocity_Min = min(NumericalNormalVelocity_Min,
                                                          min(myMPAS_O.normalVelocityCurrent[:,0]))
                        NumericalNormalVelocity_Max = max(NumericalNormalVelocity_Max,
                                                          max(myMPAS_O.normalVelocityCurrent[:,0]))
                    if iTime == nTime - 1:
                        print('The range of numerical normal velocity is [%.6f,%.6f].'
                              %(NumericalNormalVelocity_Min,NumericalNormalVelocity_Max))                 
            if plot_these_variables[9]: # if plotNumericalTangentialVelocity:
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_NumericalTangentialVelocity')
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.1,0.1]
                Title = (wave_type_title
                         + ': Numerical Tangential Velocity after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NumericalTangentialVelocity_' + '%3.3d' %iTime)
                if output_these_variables[9]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  myMPAS_O.tangentialVelocity[:,0],FigureTitle)
                numericalTangentialVelocities_Plot = (
                CR.RemoveElementFrom1DArray(myMPAS_O.tangentialVelocity[:,0],max_yEdge_index))
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,xEdge_Plot,yEdge_Plot,
                                                                    numericalTangentialVelocities_Plot,300,True,
                                                                    ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')    
                if display_range_of_variables:
                    if iTime == 0:
                        NumericalTangentialVelocity_Min = min(myMPAS_O.tangentialVelocity[:,0])
                        NumericalTangentialVelocity_Max = max(myMPAS_O.tangentialVelocity[:,0])
                    else:
                        NumericalTangentialVelocity_Min = min(NumericalTangentialVelocity_Min,
                                                              min(myMPAS_O.tangentialVelocity[:,0]))
                        NumericalTangentialVelocity_Max = max(NumericalTangentialVelocity_Max,
                                                              max(myMPAS_O.tangentialVelocity[:,0]))
                    if iTime == nTime - 1:
                        print('The range of numerical tangential velocity is [%.6f,%.6f].'
                              %(NumericalTangentialVelocity_Min,NumericalTangentialVelocity_Max))    
            if plot_these_variables[10] and myMPAS_O.time > 0.0: 
            # if plotSurfaceElevationError and myMPAS_O.time > 0.0:
                SurfaceElevationError = myMPAS_O.sshCurrent - GeophysicalWaveExactSurfaceElevations
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_SurfaceElevationError')                
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.00877,0.00877]
                Title = (wave_type_title + ': Surface Elevation Error after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds))                          
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_SurfaceElevationError_' + '%3.3d' %iTime) 
                if output_these_variables[10]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  SurfaceElevationError,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                    SurfaceElevationError,300,True,ColorBarLimits,
                                                                    6,plt.cm.seismic,13.75,['x (km)','y (km)'],
                                                                    [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,
                                                                    20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == nDumpFrequency:
                        SurfaceElevationError_Min = min(SurfaceElevationError)
                        SurfaceElevationError_Max = max(SurfaceElevationError)
                    else:
                        SurfaceElevationError_Min = min(SurfaceElevationError_Min,min(SurfaceElevationError))
                        SurfaceElevationError_Max = max(SurfaceElevationError_Max,max(SurfaceElevationError))  
                    if iTime == nTime - 1:
                        print('The range of surface elevation error is [%.6f,%.6f].'
                              %(SurfaceElevationError_Min,SurfaceElevationError_Max))    
            if plot_these_variables[11] and myMPAS_O.time > 0.0: 
            # if plotZonalVelocityError and myMPAS_O.time > 0.0:                    
                ZonalVelocityError = numericalZonalVelocities - GeophysicalWaveExactZonalVelocities               
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_ZonalVelocityError')                
                Title = (wave_type_title + ': Zonal Velocity Error after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_ZonalVelocityError_' + '%3.3d' %iTime)   
                if output_these_variables[11]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  ZonalVelocityError,FigureTitle)                
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,ZonalVelocityError,
                                                                    300,False,[0.0,0.0],6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == nDumpFrequency:
                        ZonalVelocityError_Min = min(ZonalVelocityError)
                        ZonalVelocityError_Max = max(ZonalVelocityError)
                    else:
                        ZonalVelocityError_Min = min(ZonalVelocityError_Min,min(ZonalVelocityError))
                        ZonalVelocityError_Max = max(ZonalVelocityError_Max,max(ZonalVelocityError))
                    if iTime == nTime - 1:
                        print('The range of zonal velocity error is [%.6f,%.6f].'
                              %(ZonalVelocityError_Min,ZonalVelocityError_Max))   
            if plot_these_variables[12] and myMPAS_O.time > 0.0:
            # if plotMeridionalVelocityError and myMPAS_O.time > 0.0:
                MeridionalVelocityError = numericalMeridionalVelocities - GeophysicalWaveExactMeridionalVelocities
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_MeridionalVelocityError')                    
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.00213,0.00213]
                Title = (wave_type_title + ': Meridional Velocity Error after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))               
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_MeridionalVelocityError_' + '%3.3d' %iTime)    
                if output_these_variables[12]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  MeridionalVelocityError,FigureTitle) 
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_2(myMPAS_O,output_directory,
                                                                    MeridionalVelocityError,300,True,
                                                                    ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == nDumpFrequency:
                        MeridionalVelocityError_Min = min(MeridionalVelocityError)
                        MeridionalVelocityError_Max = max(MeridionalVelocityError)
                    else:
                        MeridionalVelocityError_Min = min(MeridionalVelocityError_Min,
                                                          min(MeridionalVelocityError))
                        MeridionalVelocityError_Max = max(MeridionalVelocityError_Max,
                                                          max(MeridionalVelocityError))              
                    if iTime == nTime - 1:
                        print('The range of meridional velocity error is [%.6f,%.6f].'
                              %(MeridionalVelocityError_Min,MeridionalVelocityError_Max))                     
            if plot_these_variables[13] and myMPAS_O.time > 0.0:
            # if plotNormalVelocityError and myMPAS_O.time > 0.0:   
                NormalVelocityError = (
                myMPAS_O.normalVelocityCurrent[:,0] - GeophysicalWaveExactNormalVelocities[:])
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_NormalVelocityError')
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.00193,0.00193]               
                Title = (wave_type_title + ': Normal Velocity Error after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))               
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NormalVelocityError_' + '%3.3d' %iTime)    
                if output_these_variables[13]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  NormalVelocityError,FigureTitle)
                NormalVelocityError_Plot = CR.RemoveElementFrom1DArray(NormalVelocityError,max_yEdge_index)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,xEdge_Plot,yEdge_Plot,
                                                                    NormalVelocityError_Plot,300,True,
                                                                    ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == nDumpFrequency:
                        NormalVelocityError_Min = min(NormalVelocityError)
                        NormalVelocityError_Max = max(NormalVelocityError)
                    else:
                        NormalVelocityError_Min = min(NormalVelocityError_Min,min(NormalVelocityError))
                        NormalVelocityError_Max = max(NormalVelocityError_Max,max(NormalVelocityError))
                    if iTime == nTime - 1:
                        print('The range of normal velocity error is [%.6f,%.6f].'
                              %(NormalVelocityError_Min,NormalVelocityError_Max))   
            if plot_these_variables[14] and myMPAS_O.time > 0.0:
            # if plotTangentialVelocityError and myMPAS_O.time > 0.0:   
                TangentialVelocityError = (
                myMPAS_O.tangentialVelocity[:,0] - GeophysicalWaveExactTangentialVelocities[:])
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_TangentialVelocityError')                    
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.00206,0.00206]
                Title = (wave_type_title + ': Tangential Velocity Error after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))               
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_TangentialVelocityError_' + '%3.3d' %iTime)    
                if output_these_variables[14]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  TangentialVelocityError,FigureTitle) 
                TangentialVelocityError_Plot = (
                CR.RemoveElementFrom1DArray(TangentialVelocityError,max_yEdge_index))
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,xEdge_Plot,yEdge_Plot,
                                                                    TangentialVelocityError_Plot,300,True,
                                                                    ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                if display_range_of_variables:
                    if iTime == nDumpFrequency:
                        TangentialVelocityError_Min = min(TangentialVelocityError)
                        TangentialVelocityError_Max = max(TangentialVelocityError)
                    else:
                        TangentialVelocityError_Min = min(TangentialVelocityError_Min,
                                                          min(TangentialVelocityError))
                        TangentialVelocityError_Max = max(TangentialVelocityError_Max,
                                                          max(TangentialVelocityError))              
                    if iTime == nTime - 1:
                        print('The range of tangential velocity error is [%.6f,%.6f].'
                              %(TangentialVelocityError_Min,TangentialVelocityError_Max))    
        if iTime < nTime - 1 and myMPAS_O.myNamelist.config_time_integrator == 'forward_backward_predictor':
            if problem_type == 'Coastal_Kelvin_Wave':
                MPAS_O_Mode_Forward.ocn_time_integration_forward_backward_predictor_Geophysical_Wave(
                myMPAS_O,DetermineCoastalKelvinWaveExactSurfaceElevation,
                DetermineCoastalKelvinWaveExactZonalVelocity,DetermineCoastalKelvinWaveExactMeridionalVelocity)
            MPAS_O_Mode_Forward.ocn_shift_time_levels(myMPAS_O)


# In[20]:

runMain_LinearCoastalKelvinWave_FBP = False
if runMain_LinearCoastalKelvinWave_FBP:    
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    CourantNumber = 0.36
    time_integrator = 'forward_backward_predictor'
    output_these_variables = np.zeros(15,dtype=bool)
    plotExactSurfaceElevation = True
    plotExactZonalVelocity = False
    plotExactMeridionalVelocity = True
    plotExactNormalVelocity = True
    plotExactTangentialVelocity = True    
    plotNumericalSurfaceElevation = True
    plotNumericalZonalVelocity = True
    plotNumericalMeridionalVelocity = True
    plotNumericalNormalVelocity = True
    plotNumericalTangentialVelocity = True  
    plotSurfaceElevationError = True
    plotZonalVelocityError = False
    plotMeridionalVelocityError = True
    plotNormalVelocityError = True
    plotTangentialVelocityError = True    
    plot_these_variables = [plotExactSurfaceElevation,plotExactZonalVelocity,plotExactMeridionalVelocity,
                            plotExactNormalVelocity,plotExactTangentialVelocity,plotNumericalSurfaceElevation,
                            plotNumericalZonalVelocity,plotNumericalMeridionalVelocity,plotNumericalNormalVelocity,
                            plotNumericalTangentialVelocity,plotSurfaceElevationError,plotZonalVelocityError,
                            plotMeridionalVelocityError,plotNormalVelocityError,plotTangentialVelocityError]
    display_range_of_variables = False
    Main(mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,problem_is_linear,CourantNumber,
         time_integrator,output_these_variables,plot_these_variables,display_range_of_variables)


# In[21]:

def ComputeL2NormOfStateVariableDefinedAtEdges(nEdges,nNonPeriodicBoundaryEdges,boundaryEdge,
                                               StateVariableDefinedAtEdges):
    L2Norm = 0.0
    for iEdge in range(0,nEdges):
        if boundaryEdge[iEdge] == 0.0:
            L2Norm += (StateVariableDefinedAtEdges[iEdge])**2.0
    L2Norm = np.sqrt(L2Norm/float(nNonPeriodicBoundaryEdges))
    return L2Norm


# In[22]:

def ComputeL2NormOfStateVariableDefinedAtCellCenters(nCells,nNonPeriodicBoundaryCells,boundaryCell,
                                                     StateVariableDefinedAtCellCenters):
    L2Norm = 0.0
    for iCell in range(0,nCells):
        if boundaryCell[iCell] == 0.0:
            L2Norm += (StateVariableDefinedAtCellCenters[iCell])**2.0
    L2Norm = np.sqrt(L2Norm/float(nNonPeriodicBoundaryCells))
    return L2Norm


# In[23]:

def Main_ConvergenceTest_GeophysicalWave(convergence_type,mesh_directory,base_mesh_file_name,mesh_file_name,
                                         mesh_type,problem_type,problem_is_linear,CourantNumber,time_integrator,
                                         Ratio_of_FinalTime_to_TimePeriod,specified_lY,specified_dt,
                                         plotNumericalSolution=False,coarsestMesh=False,
                                         xCell_CoarsestRectilinearMesh=[],yCell_CoarsestRectilinearMesh=[],
                                         GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh=[],
                                         GeophysicalWaveExactZonalVelocities_CoarsestRectilinearMesh=[],
                                         GeophysicalWaveExactMeridionalVelocities_CoarsestRectilinearMesh=[]):    
    if convergence_type == 'Space':
        uCNTDTS = False
    else:
        uCNTDTS = True    
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                       problem_type,problem_is_linear,periodicity='NonPeriodic_x',
                                       CourantNumber=CourantNumber,useCourantNumberToDetermineTimeStep=uCNTDTS,
                                       time_integrator=time_integrator,
                                       specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False)
    MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    if convergence_type == 'Space' or convergence_type == 'SpaceAndTime':
        myMPAS_O.lY = specified_lY
    if convergence_type == 'Space' or convergence_type == 'Time':
        myMPAS_O.myNamelist.config_dt = specified_dt
    if convergence_type == 'Space':
        print('The timestep for Courant number %.2f is %.2f seconds.' %(CourantNumber,
                                                                        myMPAS_O.myNamelist.config_dt))        
    WaveSpeed = myMPAS_O.myNamelist.config_wave_speed   
    TimePeriod = myMPAS_O.lY/WaveSpeed
    FinalTime = Ratio_of_FinalTime_to_TimePeriod*TimePeriod
    dt = myMPAS_O.myNamelist.config_dt
    nTime = int(FinalTime/dt)
    if convergence_type == 'SpaceAndTime':
        print('TimePeriod is %.6f and FinalTime is %.6f.' %(TimePeriod,FinalTime))
        print('For nCells in each direction = %3d, nTime = %3d.' %(int(np.sqrt(float(myMPAS_O.nCells))),nTime))
        myMPAS_O.myNamelist.config_dt = FinalTime/float(nTime)
        ModifiedCourantNumber = CourantNumber*myMPAS_O.myNamelist.config_dt/dt
        print('The final timestep for the modified Courant number of %.6f is %.6f seconds.' 
              %(ModifiedCourantNumber,myMPAS_O.myNamelist.config_dt))
        print('The increase in timestep is %.6f seconds and that in the Courant Number is %.6f.'
              %(myMPAS_O.myNamelist.config_dt-dt,ModifiedCourantNumber-CourantNumber))
    compute_these_variables = np.zeros(8,dtype=bool)
    compute_these_variables[4] = True # compute_these_variables[4] = compute_tangentialVelocity
    numericalZonalVelocities = np.zeros(myMPAS_O.nEdges)
    numericalMeridionalVelocities = np.zeros(myMPAS_O.nEdges)
    if problem_type == 'Coastal_Kelvin_Wave':
        wave_type_title = 'Coastal Kelvin Wave'
        wave_type_figure_title = 'CoastalKelvinWave'
    if problem_is_linear:
        wave_nature = 'Linear'
    else: 
        wave_nature = 'NonLinear'
    if time_integrator == 'forward_backward_predictor':    
        time_integrator_short_form = 'FBP'    
    for iTime in range(0,nTime+1):
        myMPAS_O.time = float(iTime)*myMPAS_O.myNamelist.config_dt
        if iTime == 0 or iTime == nTime:
            if problem_type == 'Coastal_Kelvin_Wave':
                [GeophysicalWaveExactSurfaceElevations, GeophysicalWaveExactZonalVelocities, 
                 GeophysicalWaveExactMeridionalVelocities] = (
                DetermineGeophysicalWaveExactSolutions(myMPAS_O,DetermineCoastalKelvinWaveExactSurfaceElevation,
                                                       DetermineCoastalKelvinWaveExactZonalVelocity,
                                                       DetermineCoastalKelvinWaveExactMeridionalVelocity))
            GeophysicalWaveExactVelocities = np.zeros((myMPAS_O.nEdges,2))
            GeophysicalWaveExactVelocities[:,0] = GeophysicalWaveExactZonalVelocities[:]
            GeophysicalWaveExactVelocities[:,1] = GeophysicalWaveExactMeridionalVelocities[:]
            GeophysicalWaveExactNormalVelocities, GeophysicalWaveExactTangentialVelocities = (
            ComputeNormalAndTangentialComponentsAtEdge(GeophysicalWaveExactVelocities,myMPAS_O.angleEdge,'both'))
        if iTime == 0.0: # Specify initial conditions
            myMPAS_O.sshCurrent[:] = GeophysicalWaveExactSurfaceElevations[:]
            myMPAS_O.normalVelocityCurrent[:,0] = GeophysicalWaveExactNormalVelocities[:]
        if iTime == nTime:
            print('The final time for the %3d x %3d mesh is %.6f seconds.' 
                  %(int(np.sqrt(float(myMPAS_O.nCells))),int(np.sqrt(float(myMPAS_O.nCells))),myMPAS_O.time))
            MPAS_O_Shared.ocn_diagnostic_solve(myMPAS_O,myMPAS_O.normalVelocityCurrent,myMPAS_O.sshCurrent,
                                               compute_these_variables)
            for iEdge in range(0,myMPAS_O.nEdges):
                if myMPAS_O.boundaryEdge[iEdge] == 1.0: # i.e. if the edge is along a non-periodic boundary
                    myMPAS_O.tangentialVelocity[iEdge,0] = GeophysicalWaveExactTangentialVelocities[iEdge]
                    numericalZonalVelocities[iEdge] = GeophysicalWaveExactZonalVelocities[iEdge]
                    numericalMeridionalVelocities[iEdge] = GeophysicalWaveExactMeridionalVelocities[iEdge]
                else:
                    numericalZonalVelocities[iEdge] = (
                    (myMPAS_O.normalVelocityCurrent[iEdge,0]*np.cos(myMPAS_O.angleEdge[iEdge])
                     - myMPAS_O.tangentialVelocity[iEdge,0]*np.sin(myMPAS_O.angleEdge[iEdge])))
                    numericalMeridionalVelocities[iEdge] = (
                    (myMPAS_O.normalVelocityCurrent[iEdge,0]*np.sin(myMPAS_O.angleEdge[iEdge])
                     + myMPAS_O.tangentialVelocity[iEdge,0]*np.cos(myMPAS_O.angleEdge[iEdge])))
            SurfaceElevationError = myMPAS_O.sshCurrent - GeophysicalWaveExactSurfaceElevations
            ZonalVelocityError = numericalZonalVelocities - GeophysicalWaveExactZonalVelocities
            MeridionalVelocityError = numericalMeridionalVelocities - GeophysicalWaveExactMeridionalVelocities
            NormalVelocityError = myMPAS_O.normalVelocityCurrent[:,0] - GeophysicalWaveExactNormalVelocities[:]
            TangentialVelocityError = (
            myMPAS_O.tangentialVelocity[:,0] - GeophysicalWaveExactTangentialVelocities[:])
            if plotNumericalSolution:
                max_yEdge_index = np.argmax(myMPAS_O.yEdge)
                xEdge_Plot = CR.RemoveElementFrom1DArray(myMPAS_O.xEdge,max_yEdge_index)
                yEdge_Plot = CR.RemoveElementFrom1DArray(myMPAS_O.yEdge,max_yEdge_index)
                hours, minutes, seconds = PrintDisplayTime(myMPAS_O.time,False)
                output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves'      
                Title = (wave_type_title + ': Exact Surface Elevation after\n%d Hours %2d Minutes %.6f Seconds'
                         %(hours,minutes,seconds))                          
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_ExactSurfaceElevation_nCellsInEachHorizontalDirection_' + '%3.3d' 
                               %int(np.sqrt(float(myMPAS_O.nCells))))
                CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                              GeophysicalWaveExactSurfaceElevations,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,myMPAS_O.xCell,
                                                                    myMPAS_O.yCell,
                                                                    GeophysicalWaveExactSurfaceElevations,300,
                                                                    False,[0.0,0.0],6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                Title = (
                wave_type_title + ': Numerical Surface Elevation after\n%d Hours %2d Minutes %.6f Seconds'
                %(hours,minutes,seconds))                          
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NumericalSurfaceElevation_nCellsInEachHorizontalDirection_' + '%3.3d' 
                               %int(np.sqrt(float(myMPAS_O.nCells))))
                CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,myMPAS_O.sshCurrent,
                                              FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,myMPAS_O.xCell,
                                                                    myMPAS_O.yCell,myMPAS_O.sshCurrent,300,False,
                                                                    [0.0,0.0],6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                Title = (wave_type_title + ': Surface Elevation Error after\n%d Hours %2d Minutes %.6f Seconds' 
                         %(hours,minutes,seconds))                          
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_SurfaceElevationError_nCellsInEachHorizontalDirection_' + '%3.3d' 
                               %int(np.sqrt(float(myMPAS_O.nCells))))
                CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                              SurfaceElevationError,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,myMPAS_O.xCell,
                                                                    myMPAS_O.yCell,SurfaceElevationError,300,
                                                                    False,[0.0,0.0],6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                Title = (wave_type_title + ': Exact Normal Velocity after\n%d Hours %2d Minutes %.6f Seconds' 
                         %(hours,minutes,seconds))                          
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_ExactNormalVelocity_nCellsInEachHorizontalDirection_' + '%3.3d' 
                               %int(np.sqrt(float(myMPAS_O.nCells)))) 
                CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                              GeophysicalWaveExactNormalVelocities,FigureTitle)                
                GeophysicalWaveExactNormalVelocities_Plot = (
                CR.RemoveElementFrom1DArray(GeophysicalWaveExactNormalVelocities,max_yEdge_index))                
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,xEdge_Plot,yEdge_Plot,
                                                                    GeophysicalWaveExactNormalVelocities_Plot,300,
                                                                    False,[0.0,0.0],6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                Title = (wave_type_title + ': Numerical Normal Velocity after\n%d Hours %2d Minutes %.6f Seconds' 
                         %(hours,minutes,seconds))                          
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NumericalNormalVelocity_nCellsInEachHorizontalDirection_' + '%3.3d' 
                               %int(np.sqrt(float(myMPAS_O.nCells)))) 
                CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                              myMPAS_O.normalVelocityCurrent[:,0],FigureTitle)
                numericalNormalVelocities_Plot = (
                CR.RemoveElementFrom1DArray(myMPAS_O.normalVelocityCurrent[:,0],max_yEdge_index))
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,xEdge_Plot,yEdge_Plot,
                                                                    numericalNormalVelocities_Plot,300,False,
                                                                    [0.0,0.0],6,plt.cm.seismic,13.75,
                                                                    ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                    [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')
                Title = (wave_type_title + ': Normal Velocity Error after\n%d Hours %2d Minutes %.6f Seconds' 
                         %(hours,minutes,seconds))                          
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NormalVelocityError_nCellsInEachHorizontalDirection_' + '%3.3d' 
                               %int(np.sqrt(float(myMPAS_O.nCells)))) 
                CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,NormalVelocityError,
                                              FigureTitle)
                NormalVelocityError_Plot = CR.RemoveElementFrom1DArray(NormalVelocityError,max_yEdge_index)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW_1(output_directory,xEdge_Plot,yEdge_Plot,
                                                                    NormalVelocityError_Plot,300,False,[0.0,0.0],
                                                                    6,plt.cm.seismic,13.75,['x (km)','y (km)'],
                                                                    [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,
                                                                    20.0,True,FigureTitle,False,
                                                                    cbarlabelformat='%.5f')            
            if convergence_type == 'Time':
                if myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells:
                    SurfaceElevationL2ErrorNorm = (
                    ComputeL2NormOfStateVariableDefinedAtCellCenters(myMPAS_O.nCells,
                                                                     myMPAS_O.nNonPeriodicBoundaryCells,
                                                                     myMPAS_O.boundaryCell,SurfaceElevationError))
                else:
                    SurfaceElevationL2ErrorNorm = np.linalg.norm(SurfaceElevationError)/np.sqrt(myMPAS_O.nCells)
                ZonalVelocityL2ErrorNorm = (
                ComputeL2NormOfStateVariableDefinedAtEdges(myMPAS_O.nEdges,myMPAS_O.nNonPeriodicBoundaryEdges,
                                                           myMPAS_O.boundaryEdge,ZonalVelocityError))
                MeridionalVelocityL2ErrorNorm = (
                ComputeL2NormOfStateVariableDefinedAtEdges(myMPAS_O.nEdges,myMPAS_O.nNonPeriodicBoundaryEdges,
                                                           myMPAS_O.boundaryEdge,MeridionalVelocityError))
            else:
                if coarsestMesh:
                    xCell_CoarsestRectilinearMesh, yCell_CoarsestRectilinearMesh = (
                    MOMIR.Generate_Rectilinear_MPAS_O_Mesh(myMPAS_O))     
                    if problem_type == 'Coastal_Kelvin_Wave':
                        [GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh, 
                         GeophysicalWaveExactZonalVelocities_CoarsestRectilinearMesh, 
                         GeophysicalWaveExactMeridionalVelocities_CoarsestRectilinearMesh] = (
                        DetermineGeophysicalWaveExactSolutionsOnCoarsestRectilinearMesh(
                        myMPAS_O,xCell_CoarsestRectilinearMesh,yCell_CoarsestRectilinearMesh,
                        DetermineCoastalKelvinWaveExactSurfaceElevation,
                        DetermineCoastalKelvinWaveExactZonalVelocity,
                            DetermineCoastalKelvinWaveExactMeridionalVelocity))
                    numericalSurfaceElevations = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,myMPAS_O.sshCurrent))                
                    SurfaceElevationError = (
                    numericalSurfaceElevations - GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh)
                    SurfaceElevationL2ErrorNorm = (
                    np.linalg.norm(SurfaceElevationError)/np.sqrt(float(myMPAS_O.nCells)))                
                    numericalZonalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                    myMPAS_O,numericalZonalVelocities)                
                    numericalZonalVelocities = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,numericalZonalVelocities))                   
                    ZonalVelocityError = (
                    numericalZonalVelocities - GeophysicalWaveExactZonalVelocities_CoarsestRectilinearMesh)
                    ZonalVelocityL2ErrorNorm = np.linalg.norm(ZonalVelocityError)/np.sqrt(float(myMPAS_O.nCells))
                    numericalMeridionalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                    myMPAS_O,numericalMeridionalVelocities)                
                    numericalMeridionalVelocities = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,numericalMeridionalVelocities))                   
                    MeridionalVelocityError = (
                    (numericalMeridionalVelocities 
                     - GeophysicalWaveExactMeridionalVelocities_CoarsestRectilinearMesh))
                    MeridionalVelocityL2ErrorNorm = (
                    np.linalg.norm(MeridionalVelocityError)/np.sqrt(float(myMPAS_O.nCells)))                
                else:
                    nCells_CoarsestRectilinearMesh = len(xCell_CoarsestRectilinearMesh)
                    xCell_FineRectilinearMesh, yCell_FineRectilinearMesh = (
                    MOMIR.Generate_Rectilinear_MPAS_O_Mesh(myMPAS_O))                
                    numericalSurfaceElevations = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,myMPAS_O.sshCurrent))
                    numericalSurfaceElevations = (
                    MOMIR.Interpolate_Solution_To_Coarsest_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O.gridSpacingMagnitude,xCell_FineRectilinearMesh,yCell_FineRectilinearMesh,
                    numericalSurfaceElevations,xCell_CoarsestRectilinearMesh,yCell_CoarsestRectilinearMesh))
                    SurfaceElevationError = (
                    numericalSurfaceElevations - GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh)
                    SurfaceElevationL2ErrorNorm = (
                    np.linalg.norm(SurfaceElevationError)/np.sqrt(float(myMPAS_O.nCells)))                
                    numericalZonalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                    myMPAS_O,numericalZonalVelocities)
                    numericalZonalVelocities = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,numericalZonalVelocities))                
                    numericalZonalVelocities = (
                    MOMIR.Interpolate_Solution_To_Coarsest_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O.gridSpacingMagnitude,xCell_FineRectilinearMesh,yCell_FineRectilinearMesh,
                    numericalZonalVelocities,xCell_CoarsestRectilinearMesh,yCell_CoarsestRectilinearMesh))
                    ZonalVelocityError = (
                    numericalZonalVelocities - GeophysicalWaveExactZonalVelocities_CoarsestRectilinearMesh)
                    ZonalVelocityL2ErrorNorm = (
                    np.linalg.norm(ZonalVelocityError)/np.sqrt(float(nCells_CoarsestRectilinearMesh)))  
                    numericalMeridionalVelocities = MOMIR.Interpolate_Solution_From_Edges_To_Cell_Centers(
                    myMPAS_O,numericalMeridionalVelocities)
                    numericalMeridionalVelocities = (
                    MOMIR.Interpolate_Solution_From_MPAS_O_Mesh_To_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O,numericalMeridionalVelocities))                
                    numericalMeridionalVelocities = (
                    MOMIR.Interpolate_Solution_To_Coarsest_Rectilinear_MPAS_O_Mesh(
                    myMPAS_O.gridSpacingMagnitude,xCell_FineRectilinearMesh,yCell_FineRectilinearMesh,
                    numericalMeridionalVelocities,xCell_CoarsestRectilinearMesh,yCell_CoarsestRectilinearMesh))
                    MeridionalVelocityError = (
                    (numericalMeridionalVelocities 
                     - GeophysicalWaveExactMeridionalVelocities_CoarsestRectilinearMesh))
                    MeridionalVelocityL2ErrorNorm = (
                    np.linalg.norm(MeridionalVelocityError)/np.sqrt(float(nCells_CoarsestRectilinearMesh)))
        if iTime < nTime and time_integrator == 'forward_backward_predictor':
            if problem_type == 'Coastal_Kelvin_Wave':
                MPAS_O_Mode_Forward.ocn_time_integration_forward_backward_predictor_Geophysical_Wave(
                myMPAS_O,DetermineCoastalKelvinWaveExactSurfaceElevation,
                DetermineCoastalKelvinWaveExactZonalVelocity,DetermineCoastalKelvinWaveExactMeridionalVelocity)
            MPAS_O_Mode_Forward.ocn_shift_time_levels(myMPAS_O)
    if not(convergence_type == 'Time') and coarsestMesh:
        return [myMPAS_O.gridSpacingMagnitude, SurfaceElevationL2ErrorNorm, ZonalVelocityL2ErrorNorm, 
                MeridionalVelocityL2ErrorNorm, xCell_CoarsestRectilinearMesh, yCell_CoarsestRectilinearMesh,
                GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh,
                GeophysicalWaveExactZonalVelocities_CoarsestRectilinearMesh,
                GeophysicalWaveExactMeridionalVelocities_CoarsestRectilinearMesh]
    else:
        return [myMPAS_O.gridSpacingMagnitude, SurfaceElevationL2ErrorNorm, ZonalVelocityL2ErrorNorm, 
                MeridionalVelocityL2ErrorNorm]


# In[24]:

def ConvergenceTest_GeophysicalWave(convergence_type,mesh_type,problem_type,problem_is_linear,CourantNumber,
                                    time_integrator,Ratio_of_FinalTime_to_TimePeriod,output_these_variables):  
    if convergence_type == 'Space':
        nCellsX = np.array([16,32,64,128,256])
        nCases = len(nCellsX)
    elif convergence_type == 'Time':
        CourantNumberMin = 0.25
        CourantNumberMax = 0.75
        nCases = 21
        CourantNumberArray = np.linspace(CourantNumberMin,CourantNumberMax,nCases)
        nCellsX = np.ones(nCases,dtype=int)*150
    elif convergence_type == 'SpaceAndTime':
        nCellsXMin = 50
        nCellsXMax = 250
        d_nCellsX = 10
        nCases = int((nCellsXMax - nCellsXMin)/d_nCellsX) + 1
        nCellsX = np.linspace(nCellsXMin,nCellsXMax,nCases,dtype=int)
    dc = np.zeros(nCases)
    SurfaceElevationL2ErrorNorm = np.zeros(nCases)
    ZonalVelocityL2ErrorNorm = np.zeros(nCases)
    MeridionalVelocityL2ErrorNorm = np.zeros(nCases)
    NormalVelocityL2ErrorNorm = np.zeros(nCases)
    TangentialVelocityL2ErrorNorm = np.zeros(nCases)
    if problem_type == 'Coastal_Kelvin_Wave':
        wave_type_file_name = 'CoastalKelvinWave'
    if problem_is_linear:
        wave_nature = 'Linear'
    else: 
        wave_nature = 'NonLinear'
    if time_integrator == 'forward_backward_predictor':    
        time_integrator_short_form = 'FBP'
    if convergence_type == 'Space' or convergence_type == 'SpaceAndTime':
        if problem_type == 'Coastal_Kelvin_Wave':
            mesh_directory = (
            'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/ConvergenceStudyMeshes')
            base_mesh_file_name = 'culled_mesh_%s.nc' %(nCellsX[nCases-1])
            mesh_file_name = 'mesh_%s.nc' %(nCellsX[nCases-1])
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                           problem_type,problem_is_linear,periodicity='NonPeriodic_x',
                                           CourantNumber=CourantNumber,useCourantNumberToDetermineTimeStep=True,
                                           time_integrator=time_integrator,
                                           specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False)
        specified_lY = myMPAS_O.lY 
    if convergence_type == 'Space':    
        WaveSpeed = myMPAS_O.myNamelist.config_wave_speed
        TimePeriod = myMPAS_O.lY/WaveSpeed
        FinalTime = Ratio_of_FinalTime_to_TimePeriod*TimePeriod
        specified_dt = myMPAS_O.myNamelist.config_dt
        nTime = int(FinalTime/specified_dt)
        specified_dt = FinalTime/float(nTime)
    elif convergence_type == 'Time':
        if problem_type == 'Coastal_Kelvin_Wave':
            mesh_directory = (
            'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/ConvergenceStudyMeshes')
            base_mesh_file_name = 'culled_mesh_%s.nc' %(nCellsX[0])
            mesh_file_name = 'mesh_%s.nc' %(nCellsX[0])
        myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                                           problem_type,problem_is_linear,periodicity='NonPeriodic_x',
                                           CourantNumber=CourantNumber,useCourantNumberToDetermineTimeStep=False,
                                           time_integrator=time_integrator,
                                           specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False)
        specified_lY = 0.0
        dx = myMPAS_O.gridSpacingMagnitude
        WaveSpeed = myMPAS_O.myNamelist.config_wave_speed
        TimePeriod = myMPAS_O.lY/WaveSpeed
        FinalTime = Ratio_of_FinalTime_to_TimePeriod*TimePeriod
    elif convergence_type == 'SpaceAndTime':
        specified_dt = 1.0
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
        if convergence_type == 'Time':        
            [dc[iCase], SurfaceElevationL2ErrorNorm[iCase], ZonalVelocityL2ErrorNorm[iCase], 
             MeridionalVelocityL2ErrorNorm[iCase]] = (
            Main_ConvergenceTest_GeophysicalWave(convergence_type,mesh_directory,base_mesh_file_name,
                                                 mesh_file_name,mesh_type,problem_type,problem_is_linear,
                                                 CourantNumber,time_integrator,Ratio_of_FinalTime_to_TimePeriod,
                                                 specified_lY,specified_dt,plotNumericalSolution=False))
        else:
            if iCase == 0:
                [dc[iCase], SurfaceElevationL2ErrorNorm[iCase], ZonalVelocityL2ErrorNorm[iCase], 
                 MeridionalVelocityL2ErrorNorm[iCase], xCell_CoarsestRectilinearMesh, 
                 yCell_CoarsestRectilinearMesh, GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh,
                 GeophysicalWaveExactZonalVelocities_CoarsestRectilinearMesh,
                 GeophysicalWaveExactMeridionalVelocities_CoarsestRectilinearMesh] = (
                Main_ConvergenceTest_GeophysicalWave(convergence_type,mesh_directory,base_mesh_file_name,
                                                     mesh_file_name,mesh_type,problem_type,problem_is_linear,
                                                     CourantNumber,time_integrator,
                                                     Ratio_of_FinalTime_to_TimePeriod,specified_lY,specified_dt,
                                                     plotNumericalSolution=False,coarsestMesh=True))
            else:    
                [dc[iCase], SurfaceElevationL2ErrorNorm[iCase], ZonalVelocityL2ErrorNorm[iCase], 
                 MeridionalVelocityL2ErrorNorm[iCase]] = (
                Main_ConvergenceTest_GeophysicalWave(
                convergence_type,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,problem_type,
                problem_is_linear,CourantNumber,time_integrator,Ratio_of_FinalTime_to_TimePeriod,specified_lY,
                specified_dt,False,False,xCell_CoarsestRectilinearMesh,yCell_CoarsestRectilinearMesh,
                GeophysicalWaveExactSurfaceElevations_CoarsestRectilinearMesh,
                GeophysicalWaveExactZonalVelocities_CoarsestRectilinearMesh,
                GeophysicalWaveExactMeridionalVelocities_CoarsestRectilinearMesh))
        if convergence_type == 'Time':
            dc[iCase] = specified_dt
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves'
    output_surface_elevation_convergence_data = output_these_variables[0]
    if output_surface_elevation_convergence_data:
        FileName = (wave_type_file_name + '_' + wave_nature + '_' + time_integrator_short_form
                    + '_NumericalSurfaceElevationConvergencePlot_' + convergence_type + '_L2ErrorNorm')  
        CR.WriteCurve1D(output_directory,dc,SurfaceElevationL2ErrorNorm,FileName)
    output_zonal_velocity_convergence_data = output_these_variables[1]
    if output_zonal_velocity_convergence_data:        
        FileName = (wave_type_file_name + '_' + wave_nature + '_' + time_integrator_short_form
                    + '_NumericalZonalVelocityConvergencePlot_' + convergence_type + '_L2ErrorNorm')  
        CR.WriteCurve1D(output_directory,dc,ZonalVelocityL2ErrorNorm,FileName)
    output_meridional_velocity_convergence_data = output_these_variables[2]
    if output_meridional_velocity_convergence_data:              
        FileName = (wave_type_file_name + '_' + wave_nature + '_' + time_integrator_short_form
                    + '_NumericalMeridionalVelocityConvergencePlot_' + convergence_type + '_L2ErrorNorm')     
        CR.WriteCurve1D(output_directory,dc,MeridionalVelocityL2ErrorNorm,FileName)


# In[25]:

def run_ConvergenceTest_LinearCoastalKelvinWave(convergence_type):
    mesh_type = 'uniform'
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    CourantNumber = 0.75
    time_integrator = 'forward_backward_predictor'
    Ratio_of_FinalTime_to_TimePeriod = 0.5
    output_these_variables = np.zeros(3,dtype=bool)
    output_these_variables[0] = True
    # output_these_variables[0] = output_surface_elevation_convergence_data
    # output_these_variables[1] = output_zonal_velocity_convergence_data
    # output_these_variables[2] = output_meridional_velocity_convergence_data
    ConvergenceTest_GeophysicalWave(convergence_type,mesh_type,problem_type,problem_is_linear,CourantNumber,
                                    time_integrator,Ratio_of_FinalTime_to_TimePeriod,output_these_variables)


# In[26]:

run_ConvergenceTest_Space_LinearCoastalKelvinWave = False
if run_ConvergenceTest_Space_LinearCoastalKelvinWave:
    run_ConvergenceTest_LinearCoastalKelvinWave('Space')


# In[27]:

run_ConvergenceTest_Time_LinearCoastalKelvinWave = False
if run_ConvergenceTest_Time_LinearCoastalKelvinWave:
    run_ConvergenceTest_LinearCoastalKelvinWave('Time')


# In[28]:

run_ConvergenceTest_SpaceAndTime_LinearCoastalKelvinWave = False
if run_ConvergenceTest_SpaceAndTime_LinearCoastalKelvinWave:
    run_ConvergenceTest_LinearCoastalKelvinWave('SpaceAndTime')


# In[29]:

def PlotConvergenceData(convergence_type,problem_type,problem_is_linear,CourantNumber,time_integrator,
                        plot_these_variables,plotAgainstCellWidthInverse=True,usePlotly=False,useMeanCurve=False):
    if problem_type == 'Coastal_Kelvin_Wave':
        wave_type_figure_title = 'CoastalKelvinWave'
    if problem_is_linear:
        wave_nature = 'Linear'
    else: 
        wave_nature = 'NonLinear'
    if time_integrator == 'forward_backward_predictor':    
        time_integrator_short_form = 'FBP'        
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves'
    if convergence_type == 'Time':
        xLabel = 'Inverse of Time Step'
    else: # if convergence_type == 'Space' or convergence_type == 'SpaceAndTime':
        xLabel = 'Inverse of Cell Width'
    if convergence_type == 'Space':
        Title = 'Convergence in Space w.r.t. L2 Error Norm'
    elif convergence_type == 'Time':
        Title = 'Convergence in Time w.r.t. L2 Error Norm'
    elif convergence_type == 'SpaceAndTime':
        Title = 'Convergence at Constant Courant Number of %.2f w.r.t. L2 Error Norm' %CourantNumber
    myTitle = Title
    plot_surface_elevation_convergence_data = plot_these_variables[0]
    if plot_surface_elevation_convergence_data:  
        yLabel = 'L2 Error Norm of Numerical Surface Elevation'         
        FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                       + '_NumericalSurfaceElevationConvergencePlot_' + convergence_type + '_L2ErrorNorm') 
        dc, SurfaceElevationL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
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
            A = np.vstack([np.log10(1.0/dc),np.ones(len(dc))]).T
            m, c = np.linalg.lstsq(A,np.log10(SurfaceElevationL2ErrorNorm))[0]
            SurfaceElevationL2ErrorNorm = m*(np.log10(1.0/dc)) + c
            SurfaceElevationL2ErrorNorm = 10.0**SurfaceElevationL2ErrorNorm    
            myTitle = Title + ': Slope is %.3g' %m
            FigureTitle += '_MeanCurve'
        if not(plotAgainstCellWidthInverse):
            dc = 1.0/dc  
        if not(usePlotly):
            CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,SurfaceElevationL2ErrorNorm,2.0,'-','k',
                                     's',7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],myTitle,20.0,
                                     True,FigureTitle,False,fig_size=[9.25,9.25],
                                     useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                     drawMinorGrid=True) 
        else:
            CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',1.0/dc,SurfaceElevationL2ErrorNorm,'black',2.0,
                                       'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,25.0,True,
                                       FigureTitle,False,fig_size=[700.0,700.0])  
    plot_zonal_velocity_convergence_data = plot_these_variables[1]
    if plot_zonal_velocity_convergence_data: 
        yLabel = 'L2 Error Norm of Numerical Zonal Velocity'
        FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                       + '_NumericalZonalVelocityConvergencePlot_' + convergence_type + '_L2ErrorNorm') 
        dc, ZonalVelocityL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
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
            A = np.vstack([np.log10(1.0/dc),np.ones(len(dc))]).T
            m, c = np.linalg.lstsq(A,np.log10(ZonalVelocityL2ErrorNorm))[0]
            ZonalVelocityL2ErrorNorm = m*(np.log10(1.0/dc)) + c
            ZonalVelocityL2ErrorNorm = 10.0**ZonalVelocityL2ErrorNorm  
            myTitle = Title + ': Slope is %.3g' %m
            FigureTitle += '_MeanCurve'
        if not(plotAgainstCellWidthInverse):
            dc = 1.0/dc 
        if not(usePlotly):
            CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,ZonalVelocityL2ErrorNorm,2.0,'-','k','s',
                                     7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],myTitle,20.0,True,
                                     FigureTitle,False,fig_size=[9.25,9.25],
                                     useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                     drawMinorGrid=True) 
        else:
            CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',1.0/dc,ZonalVelocityL2ErrorNorm,'black',2.0,
                                       'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,25.0,True,
                                       FigureTitle,False,fig_size=[700.0,700.0])  
    plot_meridional_velocity_convergence_data = plot_these_variables[2]
    if plot_meridional_velocity_convergence_data:            
        yLabel = 'L2 Error Norm of Numerical Meridional Velocity'
        FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                       + '_NumericalMeridionalVelocityConvergencePlot_' + convergence_type + '_L2ErrorNorm')     
        dc, MeridionalVelocityL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
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
            A = np.vstack([np.log10(1.0/dc),np.ones(len(dc))]).T
            m, c = np.linalg.lstsq(A,np.log10(MeridionalVelocityL2ErrorNorm))[0]
            MeridionalVelocityL2ErrorNorm = m*(np.log10(1.0/dc)) + c
            MeridionalVelocityL2ErrorNorm = 10.0**MeridionalVelocityL2ErrorNorm  
            myTitle = Title + ': Slope is %.3g' %m
            FigureTitle += '_MeanCurve'
        if not(plotAgainstCellWidthInverse):
            dc = 1.0/dc 
        if not(usePlotly):
            CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,MeridionalVelocityL2ErrorNorm,2.0,'-','k',
                                     's',7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],myTitle,20.0,
                                     True,FigureTitle,False,fig_size=[9.25,9.25],
                                     useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,
                                     drawMinorGrid=True) 
        else:
            CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',1.0/dc,MeridionalVelocityL2ErrorNorm,'black',
                                       2.0,'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,25.0,True,
                                       FigureTitle,False,fig_size=[700.0,700.0])


# In[30]:

def do_PlotConvergenceData_LinearCoastalKelvinWave(convergence_type):
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    CourantNumber = 0.75
    time_integrator = 'forward_backward_predictor'
    plot_these_variables = np.zeros(3,dtype=bool)
    plot_these_variables[0] = True
    # plot_these_variables[0] = plot_surface_elevation_convergence_data
    # plot_these_variables[1] = plot_zonal_velocity_convergence_data
    # plot_these_variables[2] = plot_meridional_velocity_convergence_data
    plotAgainstCellWidthInverse = True
    usePlotly = False
    useMeanCurve = True
    PlotConvergenceData(convergence_type,problem_type,problem_is_linear,CourantNumber,time_integrator,
                        plot_these_variables,plotAgainstCellWidthInverse,usePlotly,useMeanCurve)


# In[31]:

do_PlotConvergenceData_Space_LinearCoastalKelvinWave = False
if do_PlotConvergenceData_Space_LinearCoastalKelvinWave:
    do_PlotConvergenceData_LinearCoastalKelvinWave('Space')


# In[32]:

do_PlotConvergenceData_Time_LinearCoastalKelvinWave = False
if do_PlotConvergenceData_Time_LinearCoastalKelvinWave:
    do_PlotConvergenceData_LinearCoastalKelvinWave('Time')


# In[33]:

do_PlotConvergenceData_SpaceAndTime_LinearCoastalKelvinWave = False
if do_PlotConvergenceData_SpaceAndTime_LinearCoastalKelvinWave:
    do_PlotConvergenceData_LinearCoastalKelvinWave('SpaceAndTime')