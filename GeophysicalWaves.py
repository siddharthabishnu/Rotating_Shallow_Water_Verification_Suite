
# coding: utf-8

# Name: GeophysicalWaves.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains routines for verifying the barotropic solver of MPAS-Ocean against test cases involving geophysical waves. <br/>

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import io as inputoutput
import os
import sys
from IPython.utils import io
with io.capture_output() as captured: 
    import Common_Routines as CR
    import MPAS_O_Mode_Init
    import MPAS_O_Shared
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

def PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,x,y,phi,nContours,useGivenColorBarLimits,
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

def PrintDisplayTime(time,displaytime):
    hours = np.floor(time/3600.0)
    remainingtime = np.mod(time,3600.0)
    minutes = np.floor(remainingtime/60.0)
    seconds = np.mod(remainingtime,60.0)
    if displaytime:
        print('The display time is %1d hours %2d minutes %2d seconds.' %(hours,minutes,seconds))
    return hours, minutes, seconds


# In[5]:

hours, minutes, seconds = PrintDisplayTime(10983.0,True)


# In[6]:

def CoastalKelvinWaveFunctionalForm(lY,y,returnAmplitude=False):
    yCenter = 0.5*lY
    etaHat = 0.001
    eta = etaHat*np.sin(2.0*np.pi*y/lY)
    if returnAmplitude:
        return etaHat
    else:
        return eta


# In[7]:

def DetermineCoastalKelvinWaveExactSurfaceElevation(H,c,R,lY,x,y,time):
    CoastalKelvinWaveExactSurfaceElevation = -H*CoastalKelvinWaveFunctionalForm(lY,y+c*time)*np.exp(-x/R)
    return CoastalKelvinWaveExactSurfaceElevation


# In[8]:

def DetermineCoastalKelvinWaveExactZonalVelocity():
    CoastalKelvinWaveExactZonalVelocity = 0.0
    return CoastalKelvinWaveExactZonalVelocity


# In[9]:

def DetermineCoastalKelvinWaveExactMeridionalVelocity(c,R,lY,x,y,time):
    CoastalKelvinWaveExactMeridionalVelocity = c*CoastalKelvinWaveFunctionalForm(lY,y+c*time)*np.exp(-x/R)
    return CoastalKelvinWaveExactMeridionalVelocity


# In[10]:

def DetermineTimeStepForGivenCourantNumber(CourantNumber,dx,WaveSpeed):
    dt = CourantNumber*dx/WaveSpeed
    return dt


# In[11]:

do_DetermineTimeStepForGivenCourantNumber = False
if do_DetermineTimeStepForGivenCourantNumber:    
    print_basic_geometry = False
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,
                                       problem_type,problem_is_linear,periodicity)
    CourantNumber = 0.36
    dx = myMPAS_O.dcEdge[0]
    WaveSpeed = myMPAS_O.myNamelist.config_wave_speed
    dt = DetermineTimeStepForGivenCourantNumber(CourantNumber,dx,WaveSpeed)
    print('The timestep for Courant Number %.2f is %.2f seconds.' %(CourantNumber,dt))


# In[12]:

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


# In[13]:

def DetermineCoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters():
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    periodicity='NonPeriodic_x'
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,problem_type,
                                       problem_is_linear,periodicity)    
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


# In[14]:

do_DetermineCoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters = False
if do_DetermineCoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters:
    DetermineCoastalKelvinWaveSurfaceElevationAmplitudeAtCellCenters()


# In[15]:

def testDetermineGeophysicalWaveExactSolutions(mesh_directory,base_mesh_file_name,mesh_file_name,
                                               problem_type,problem_is_linear,CourantNumber,plotFigures):
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,problem_type,
                                       problem_is_linear,periodicity='NonPeriodic_x',CourantNumber=CourantNumber,
                                       useCourantNumberToDetermineTimeStep=True)
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
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                              GeophysicalWaveExactSurfaceElevations,300,True,
                                                              ColorBarLimits,6,plt.cm.seismic,13.75,
                                                              ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                              [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                              cbarlabelformat='%.5f') 
            if plotExactZonalVelocity:
                Title = (wave_type_title + ': Exact Zonal Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds))
                FigureTitle = wave_type_figure_title + '_ExactZonalVelocity_' + '%3.3d' %iTime
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                                  GeophysicalWaveExactZonalVelocities,300,False,
                                                                  [0.0,0.0],6,plt.cm.seismic,13.75,
                                                                  ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                  [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                  cbarlabelformat='%.5f')
            if problem_type == 'Coastal_Kelvin_Wave':
                ColorBarLimits = [-0.1,0.1]
            Title = (wave_type_title + ': Exact Meridional Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                     %(hours,minutes,seconds))
            FigureTitle = wave_type_figure_title + '_ExactMeridionalVelocity_' + '%3.3d' %iTime
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                              GeophysicalWaveExactMeridionalVelocities,300,True,
                                                              ColorBarLimits,6,plt.cm.seismic,13.75,
                                                              ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                              [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                              cbarlabelformat='%.5f')


# In[16]:

do_testDetermineGeophysicalWaveExactSolutions = False
if do_testDetermineGeophysicalWaveExactSolutions:    
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    CourantNumber = 0.36
    plotFigures = True
    testDetermineGeophysicalWaveExactSolutions(mesh_directory,base_mesh_file_name,mesh_file_name,
                                               problem_type,problem_is_linear,CourantNumber,plotFigures)


# In[17]:

def Main(mesh_directory,base_mesh_file_name,mesh_file_name,problem_type,problem_is_linear,CourantNumber,
         time_integrator,output_these_variables,plot_these_variables,display_range_of_variables):
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,problem_type,
                                       problem_is_linear,periodicity='NonPeriodic_x',CourantNumber=CourantNumber,
                                       useCourantNumberToDetermineTimeStep=True,time_integrator=time_integrator,
                                       specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False)
    myMPAS_O.myNamelist.config_dt = round(myMPAS_O.myNamelist.config_dt)
    MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    dt = myMPAS_O.myNamelist.config_dt 
    nTime = 250 + 1
    nDumpFrequency = 10
    compute_these_variables = np.zeros(8,dtype=bool)
    compute_these_variables[:] = False
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
    for iTime in range(0,nTime):
        myMPAS_O.time = float(iTime)*dt
        hours, minutes, seconds = PrintDisplayTime(myMPAS_O.time,False)
        printProgress = False
        if printProgress:
            print('Computing Numerical Solution after %2d Hours %2d Minutes %2d Seconds!'
                  %(hours,minutes,seconds))        
        if np.mod(iTime,nDumpFrequency) == 0.0: 
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
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                                  GeophysicalWaveExactSurfaceElevations,300,True,
                                                                  ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                  ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                  [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                  cbarlabelformat='%.5f')
            if plot_these_variables[1]: # if plotExactZonalVelocity:
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_ExactZonalVelocity')                
                Title = (wave_type_title + ': Exact Zonal Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds)) 
                FigureTitle = wave_type_figure_title + '_ExactZonalVelocity_' + '%3.3d' %iTime
                if output_these_variables[1]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  GeophysicalWaveExactZonalVelocities,FigureTitle)                
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                                  GeophysicalWaveExactZonalVelocities,300,False,
                                                                  [0.0,0.0],6,plt.cm.seismic,13.75,
                                                                  ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                  [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                  cbarlabelformat='%.5f')
            if plot_these_variables[2]: # if plotExactMeridionalVelocity:
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_ExactMeridionalVelocity')
                if problem_type == 'Coastal_Kelvin_Wave':
                    ColorBarLimits = [-0.1,0.1]
                Title = (wave_type_title + ': Exact Meridional Velocity after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds))                 
                FigureTitle = wave_type_figure_title + '_ExactMeridionalVelocity_' + '%3.3d' %iTime       
                if output_these_variables[2]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  GeophysicalWaveExactMeridionalVelocities,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
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
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                                  GeophysicalWaveExactNormalVelocities,300,True,
                                                                  ColorBarLimits,6,plt.cm.seismic,13.75,
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
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                                  GeophysicalWaveExactTangentialVelocities,300,
                                                                  True,ColorBarLimits,6,plt.cm.seismic,13.75,
                                                                  ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                                  [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                                  cbarlabelformat='%.5f')        
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
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                                  myMPAS_O.sshCurrent,300,True,ColorBarLimits,6,
                                                                  plt.cm.seismic,13.75,['x (km)','y (km)'],
                                                                  [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                                                                  True,FigureTitle,False,cbarlabelformat='%.5f')
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
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  numericalZonalVelocities,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
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
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  numericalMeridionalVelocities,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
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
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                                  myMPAS_O.normalVelocityCurrent[:,0],300,True,
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
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                                  myMPAS_O.tangentialVelocity[:,0],300,True,
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
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_SurfaceElevationError')                
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.00877,0.00877]
                SurfaceElevationError = myMPAS_O.sshCurrent - GeophysicalWaveExactSurfaceElevations
                Title = (wave_type_title + ': Surface Elevation Error after\n%d Hours %2d Minutes %2d Seconds' 
                         %(hours,minutes,seconds))                          
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_SurfaceElevationError_' + '%3.3d' %iTime) 
                if output_these_variables[10]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                  SurfaceElevationError,FigureTitle)
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                                  SurfaceElevationError,300,True,ColorBarLimits,
                                                                  6,plt.cm.seismic,13.75,['x (km)','y (km)'],
                                                                  [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                                                                  True,FigureTitle,False,cbarlabelformat='%.5f')
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
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_ZonalVelocityError')                  
                ZonalVelocityError = numericalZonalVelocities - GeophysicalWaveExactZonalVelocities
                Title = (wave_type_title + ': Zonal Velocity Error after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_ZonalVelocityError_' + '%3.3d' %iTime)   
                if output_these_variables[11]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  ZonalVelocityError,FigureTitle)                
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                                  ZonalVelocityError,300,False,[0.0,0.0],6,
                                                                  plt.cm.seismic,13.75,['x (km)','y (km)'],
                                                                  [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                                                                  True,FigureTitle,False,cbarlabelformat='%.5f')
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
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_MeridionalVelocityError')                    
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.00213,0.00213]
                MeridionalVelocityError = numericalMeridionalVelocities - GeophysicalWaveExactMeridionalVelocities
                Title = (wave_type_title + ': Meridional Velocity Error after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))               
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_MeridionalVelocityError_' + '%3.3d' %iTime)    
                if output_these_variables[12]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  MeridionalVelocityError,FigureTitle) 
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
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
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_NormalVelocityError')
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.00193,0.00193]
                NormalVelocityError = myMPAS_O.normalVelocityCurrent[:,0] - GeophysicalWaveExactNormalVelocities
                Title = (wave_type_title + ': Normal Velocity Error after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))               
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_NormalVelocityError_' + '%3.3d' %iTime)    
                if output_these_variables[13]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  NormalVelocityError,FigureTitle) 
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                                  NormalVelocityError,300,True,ColorBarLimits,6,
                                                                  plt.cm.seismic,13.75,['x (km)','y (km)'],
                                                                  [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                                                                  True,FigureTitle,False,cbarlabelformat='%.5f')
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
                output_directory = ('MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves/' 
                                    + wave_type_figure_title + '_' + wave_nature + '_' 
                                    + time_integrator_short_form + '_TangentialVelocityError')                    
                if problem_type == 'Coastal_Kelvin_Wave' and time_integrator == 'forward_backward_predictor':
                    ColorBarLimits = [-0.00206,0.00206]
                TangentialVelocityError = (
                myMPAS_O.tangentialVelocity[:,0] - GeophysicalWaveExactTangentialVelocities[:])
                Title = (wave_type_title + ': Tangential Velocity Error after\n%d Hours %2d Minutes %2d Seconds'
                         %(hours,minutes,seconds))               
                FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                               + '_TangentialVelocityError_' + '%3.3d' %iTime)    
                if output_these_variables[14]:
                    CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                  TangentialVelocityError,FigureTitle) 
                PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                                  TangentialVelocityError,300,True,
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


# In[18]:

runMain_LinearCoastalKelvinWave_FBP = False
if runMain_LinearCoastalKelvinWave_FBP:    
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh'
    base_mesh_file_name = 'culled_mesh.nc'
    mesh_file_name = 'mesh.nc'
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
    Main(mesh_directory,base_mesh_file_name,mesh_file_name,problem_type,problem_is_linear,CourantNumber,
         time_integrator,output_these_variables,plot_these_variables,display_range_of_variables)


# In[19]:

def ComputeL2NormOfStateVariableDefinedAtEdges(nEdges,nNonPeriodicBoundaryEdges,boundaryEdge,
                                               StateVariableDefinedAtEdges):
    L2Norm = 0.0
    for iEdge in range(0,nEdges):
        if boundaryEdge[iEdge] == 0.0:
            L2Norm += (StateVariableDefinedAtEdges[iEdge])**2.0
    L2Norm = np.sqrt(L2Norm/nNonPeriodicBoundaryEdges)
    return L2Norm


# In[20]:

def ComputeL2NormOfStateVariableDefinedAtCellCenters(nCells,nNonPeriodicBoundaryCells,boundaryCell,
                                                     StateVariableDefinedAtCellCenters):
    L2Norm = 0.0
    for iCell in range(0,nCells):
        if boundaryCell[iCell] == 0.0:
            L2Norm += (StateVariableDefinedAtCellCenters[iCell])**2.0
    L2Norm = np.sqrt(L2Norm/nNonPeriodicBoundaryCells)
    return L2Norm


# In[21]:

def Main_ConvergenceTest_SpaceAndTime_GeophysicalWave(mesh_directory,base_mesh_file_name,mesh_file_name,
                                                      problem_type,problem_is_linear,CourantNumber,
                                                      time_integrator,plotNumericalSolution=False):
    myMPAS_O = MPAS_O_Mode_Init.MPAS_O(False,mesh_directory,base_mesh_file_name,mesh_file_name,problem_type,
                                       problem_is_linear,periodicity='NonPeriodic_x',CourantNumber=CourantNumber,
                                       useCourantNumberToDetermineTimeStep=True,time_integrator=time_integrator,
                                       specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False)
    MPAS_O_Shared.ocn_init_routines_compute_max_level(myMPAS_O)
    MPAS_O_Shared.ocn_init_routines_setup_sign_and_index_fields(myMPAS_O)
    WaveSpeed = myMPAS_O.myNamelist.config_wave_speed
    TimePeriod = myMPAS_O.lY/WaveSpeed
    FinalTime = 0.5*TimePeriod
    dt = myMPAS_O.myNamelist.config_dt
    nTime = int(FinalTime/dt)
    print('TimePeriod is %.6f and FinalTime is %.6f.' %(TimePeriod,FinalTime))
    print('For nCells in each direction = %3d, nTime = %3d.' %(int(np.sqrt(myMPAS_O.nCells)),nTime))
    myMPAS_O.myNamelist.config_dt = FinalTime/float(nTime)
    ModifiedCourantNumber = CourantNumber*myMPAS_O.myNamelist.config_dt/dt
    print('The final timestep for the modified Courant number of %.6f is %.6f seconds.' 
          %(ModifiedCourantNumber,myMPAS_O.myNamelist.config_dt))
    compute_these_variables = np.zeros(8,dtype=bool)
    compute_these_variables[:] = False
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
                  %(int(np.sqrt(myMPAS_O.nCells)),int(np.sqrt(myMPAS_O.nCells)),myMPAS_O.time))
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
            if myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells:
                SurfaceElevationL2ErrorNorm = (
                ComputeL2NormOfStateVariableDefinedAtCellCenters(myMPAS_O.nCells,
                                                                 myMPAS_O.nNonPeriodicBoundaryCells,
                                                                 myMPAS_O.boundaryCell,SurfaceElevationError))
            else:
                SurfaceElevationL2ErrorNorm = np.linalg.norm(SurfaceElevationError)/np.sqrt(myMPAS_O.nCells)
            ZonalVelocityError = numericalZonalVelocities - GeophysicalWaveExactZonalVelocities
            ZonalVelocityL2ErrorNorm = (
            ComputeL2NormOfStateVariableDefinedAtEdges(myMPAS_O.nEdges,myMPAS_O.nNonPeriodicBoundaryEdges,
                                                       myMPAS_O.boundaryEdge,ZonalVelocityError))
            MeridionalVelocityError = numericalMeridionalVelocities - GeophysicalWaveExactMeridionalVelocities
            MeridionalVelocityL2ErrorNorm = (
            ComputeL2NormOfStateVariableDefinedAtEdges(myMPAS_O.nEdges,myMPAS_O.nNonPeriodicBoundaryEdges,
                                                       myMPAS_O.boundaryEdge,MeridionalVelocityError))
            NormalVelocityError = myMPAS_O.normalVelocityCurrent[:,0] - GeophysicalWaveExactNormalVelocities[:]
            NormalVelocityL2ErrorNorm = (
            ComputeL2NormOfStateVariableDefinedAtEdges(myMPAS_O.nEdges,myMPAS_O.nNonPeriodicBoundaryEdges,
                                                       myMPAS_O.boundaryEdge,NormalVelocityError))
            TangentialVelocityError = (
            myMPAS_O.tangentialVelocity[:,0] - GeophysicalWaveExactTangentialVelocities[:])
            TangentialVelocityL2ErrorNorm = (
            ComputeL2NormOfStateVariableDefinedAtEdges(myMPAS_O.nEdges,myMPAS_O.nNonPeriodicBoundaryEdges,
                                                       myMPAS_O.boundaryEdge,TangentialVelocityError))
        if plotNumericalSolution and iTime == nTime:
            hours, minutes, seconds = PrintDisplayTime(myMPAS_O.time,False)
            output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves'      
            Title = (wave_type_title + ': Exact Surface Elevation after\n%d Hours %2d Minutes %.6f Seconds' 
                     %(hours,minutes,seconds))                          
            FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                           + '_ExactSurfaceElevation_nCellsInEachHorizontalDirection_' + '%3.3d' 
                           %int(np.sqrt(myMPAS_O.nCells)))
            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                          GeophysicalWaveExactSurfaceElevations,FigureTitle)
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                              GeophysicalWaveExactSurfaceElevations,300,False,
                                                              [0.0,0.0],6,plt.cm.seismic,13.75,
                                                              ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                              [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                              cbarlabelformat='%.5f')
            Title = (wave_type_title + ': Numerical Surface Elevation after\n%d Hours %2d Minutes %.6f Seconds' 
                     %(hours,minutes,seconds))                          
            FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                           + '_NumericalSurfaceElevation_nCellsInEachHorizontalDirection_' + '%3.3d' 
                           %int(np.sqrt(myMPAS_O.nCells)))
            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,myMPAS_O.sshCurrent,
                                          FigureTitle)
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                              myMPAS_O.sshCurrent,300,False,[0.0,0.0],6,
                                                              plt.cm.seismic,13.75,['x (km)','y (km)'],
                                                              [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,
                                                              FigureTitle,False,cbarlabelformat='%.5f')
            Title = (wave_type_title + ': Surface Elevation Error after\n%d Hours %2d Minutes %.6f Seconds' 
                     %(hours,minutes,seconds))                          
            FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                           + '_SurfaceElevationError_nCellsInEachHorizontalDirection_' + '%3.3d' 
                           %int(np.sqrt(myMPAS_O.nCells)))
            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,SurfaceElevationError,
                                          FigureTitle)
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xCell,myMPAS_O.yCell,
                                                              SurfaceElevationError,300,False,[0.0,0.0],6,
                                                              plt.cm.seismic,13.75,['x (km)','y (km)'],
                                                              [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                                                              True,FigureTitle,False,cbarlabelformat='%.5f')
            Title = (wave_type_title + ': Exact Normal Velocity after\n%d Hours %2d Minutes %.6f Seconds' 
                     %(hours,minutes,seconds))                          
            FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                           + '_ExactNormalVelocity_nCellsInEachHorizontalDirection_' + '%3.3d' 
                           %int(np.sqrt(myMPAS_O.nCells))) 
            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                          GeophysicalWaveExactNormalVelocities,FigureTitle)
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                              GeophysicalWaveExactNormalVelocities,300,False,
                                                              [0.0,0.0],6,plt.cm.seismic,13.75,
                                                              ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                              [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                              cbarlabelformat='%.5f')
            Title = (wave_type_title + ': Numerical Normal Velocity after\n%d Hours %2d Minutes %.6f Seconds' 
                     %(hours,minutes,seconds))                          
            FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                           + '_NumericalNormalVelocity_nCellsInEachHorizontalDirection_' + '%3.3d' 
                           %int(np.sqrt(myMPAS_O.nCells))) 
            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                          myMPAS_O.normalVelocityCurrent[:,0],FigureTitle)
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                              myMPAS_O.normalVelocityCurrent[:,0],300,False,
                                                              [0.0,0.0],6,plt.cm.seismic,13.75,
                                                              ['x (km)','y (km)'],[17.5,17.5],[10.0,10.0],
                                                              [15.0,15.0],Title,20.0,True,FigureTitle,False,
                                                              cbarlabelformat='%.5f')
            Title = (wave_type_title + ': Normal Velocity Error after\n%d Hours %2d Minutes %.6f Seconds' 
                     %(hours,minutes,seconds))                          
            FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                           + '_NormalVelocityError_nCellsInEachHorizontalDirection_' + '%3.3d' 
                           %int(np.sqrt(myMPAS_O.nCells))) 
            CR.WriteTecPlot2DUnstructured(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,NormalVelocityError,
                                          FigureTitle)
            PythonFilledUnstructuredContourPlot2DSaveAsPNG_GW(output_directory,myMPAS_O.xEdge,myMPAS_O.yEdge,
                                                              NormalVelocityError,300,False,[0.0,0.0],6,
                                                              plt.cm.seismic,13.75,['x (km)','y (km)'],
                                                              [17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                                                              True,FigureTitle,False,cbarlabelformat='%.5f') 
        if iTime < nTime and time_integrator == 'forward_backward_predictor':
            if problem_type == 'Coastal_Kelvin_Wave':
                MPAS_O_Mode_Forward.ocn_time_integration_forward_backward_predictor_Geophysical_Wave(
                myMPAS_O,DetermineCoastalKelvinWaveExactSurfaceElevation,
                DetermineCoastalKelvinWaveExactZonalVelocity,DetermineCoastalKelvinWaveExactMeridionalVelocity)   
            MPAS_O_Mode_Forward.ocn_shift_time_levels(myMPAS_O)
    return [myMPAS_O.gridSpacingMagnitude, SurfaceElevationL2ErrorNorm, ZonalVelocityL2ErrorNorm, 
            MeridionalVelocityL2ErrorNorm, NormalVelocityL2ErrorNorm, TangentialVelocityL2ErrorNorm]


# In[22]:

def ConvergenceTest_SpaceAndTime_GeophysicalWave(problem_type,problem_is_linear,CourantNumber,time_integrator,
                                                 output_these_variables):
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
    for iCase in range(0,nCases):
        if problem_type == 'Coastal_Kelvin_Wave':
            mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/ConvergenceStudyMeshes'
            base_mesh_file_name = 'culled_mesh_%s.nc' %(nCellsX[iCase])
            mesh_file_name = 'mesh_%s.nc' %(nCellsX[iCase])
        [dc[iCase], SurfaceElevationL2ErrorNorm[iCase], ZonalVelocityL2ErrorNorm[iCase], 
         MeridionalVelocityL2ErrorNorm[iCase], NormalVelocityL2ErrorNorm[iCase], 
         TangentialVelocityL2ErrorNorm[iCase]] = (
        Main_ConvergenceTest_SpaceAndTime_GeophysicalWave(mesh_directory,base_mesh_file_name,mesh_file_name,
                                                          problem_type,problem_is_linear,CourantNumber,
                                                          time_integrator,plotNumericalSolution=False))
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves'
    output_surface_elevation_convergence_data = output_these_variables[0]
    if output_surface_elevation_convergence_data:
        FileName = (wave_type_file_name + '_' + wave_nature + '_' + time_integrator_short_form
                    + '_NumericalSurfaceElevationConvergencePlot_SpaceAndTime_L2ErrorNorm')  
        CR.WriteCurve1D(output_directory,dc,SurfaceElevationL2ErrorNorm,FileName)
    output_zonal_velocity_convergence_data = output_these_variables[1]
    if output_zonal_velocity_convergence_data:        
        FileName = (wave_type_file_name + '_' + wave_nature + '_' + time_integrator_short_form
                    + '_NumericalZonalVelocityConvergencePlot_SpaceAndTime_L2ErrorNorm')  
        CR.WriteCurve1D(output_directory,dc,ZonalVelocityL2ErrorNorm,FileName)
    output_meridional_velocity_convergence_data = output_these_variables[2]
    if output_meridional_velocity_convergence_data:              
        FileName = (wave_type_file_name + '_' + wave_nature + '_' + time_integrator_short_form
                    + '_NumericalMeridionalVelocityConvergencePlot_SpaceAndTime_L2ErrorNorm')     
        CR.WriteCurve1D(output_directory,dc,MeridionalVelocityL2ErrorNorm,FileName)
    output_normal_velocity_convergence_data = output_these_variables[3]
    if output_normal_velocity_convergence_data:          
        FileName = (wave_type_file_name + '_' + wave_nature + '_' + time_integrator_short_form
                    + '_NumericalNormalVelocityConvergencePlot_SpaceAndTime_L2ErrorNorm')        
        CR.WriteCurve1D(output_directory,dc,NormalVelocityL2ErrorNorm,FileName)
    output_tangential_velocity_convergence_data = output_these_variables[4]
    if output_tangential_velocity_convergence_data:              
        FileName = (wave_type_file_name + '_' + wave_nature + '_' + time_integrator_short_form
                    + '_NumericalTangentialVelocityConvergencePlot_SpaceAndTime_L2ErrorNorm') 
        CR.WriteCurve1D(output_directory,dc,TangentialVelocityL2ErrorNorm,FileName)


# In[23]:

run_ConvergenceTest_SpaceAndTime_LinearCoastalKelvinWave = False
if run_ConvergenceTest_SpaceAndTime_LinearCoastalKelvinWave:
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    CourantNumber = 0.75
    time_integrator = 'forward_backward_predictor'
    output_these_variables = np.zeros(8,dtype=bool)
    output_these_variables[0] = True # output_these_variables[0] = output_surface_elevation_convergence_data
    output_these_variables[3] = True # output_these_variables[3] = output_normal_velocity_convergence_data
    ConvergenceTest_SpaceAndTime_GeophysicalWave(problem_type,problem_is_linear,CourantNumber,time_integrator,
                                                 output_these_variables)


# In[24]:

def PlotConvergenceData(problem_type,problem_is_linear,time_integrator,plot_these_variables,
                        plotAgainstCellWidthInverse=True,usePlotly=False):
    if problem_type == 'Coastal_Kelvin_Wave':
        wave_type_figure_title = 'CoastalKelvinWave'
    if problem_is_linear:
        wave_nature = 'Linear'
    else: 
        wave_nature = 'NonLinear'
    if time_integrator == 'forward_backward_predictor':    
        time_integrator_short_form = 'FBP'        
    output_directory = 'MPAS_O_Shallow_Water_Output/MPAS_O_Geophysical_Waves'
    xLabel = 'Grid Spacing Inverse'
    plot_surface_elevation_convergence_data = plot_these_variables[0]
    if plot_surface_elevation_convergence_data:  
        yLabel = 'L2 Error Norm of Numerical Surface Elevation'
        Title = 'Convergence Plot w.r.t. L2 Error Norm' 
        FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                       + '_NumericalSurfaceElevationConvergencePlot_SpaceAndTime_L2ErrorNorm') 
        dc, SurfaceElevationL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
        if not(plotAgainstCellWidthInverse):
            dc = 1.0/dc  
        if not(usePlotly):
            CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,SurfaceElevationL2ErrorNorm,2.0,'-','k',
                                     's',7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,
                                     FigureTitle,False,fig_size=[9.25,9.25],
                                     useDefaultMethodToSpecifyTickFontSize=False,drawGrid=True) 
        else:
            CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',1.0/dc,SurfaceElevationL2ErrorNorm,'black',2.0,
                                       'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,25.0,True,
                                       FigureTitle,False,fig_size=[700.0,700.0])  
    plot_zonal_velocity_convergence_data = plot_these_variables[1]
    if plot_zonal_velocity_convergence_data: 
        yLabel = 'L2 Error Norm of Numerical Zonal Velocity'
        Title = 'Convergence Plot w.r.t. L2 Error Norm' 
        FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                       + '_NumericalZonalVelocityConvergencePlot_SpaceAndTime_L2ErrorNorm') 
        dc, ZonalVelocityL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
        if not(plotAgainstCellWidthInverse):
            dc = 1.0/dc 
        if not(usePlotly):
            CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,ZonalVelocityL2ErrorNorm,2.0,'-','k','s',
                                     7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,
                                     FigureTitle,False,fig_size=[9.25,9.25],
                                     useDefaultMethodToSpecifyTickFontSize=False,drawGrid=True) 
        else:
            CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',1.0/dc,ZonalVelocityL2ErrorNorm,'black',2.0,
                                       'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,25.0,True,
                                       FigureTitle,False,fig_size=[700.0,700.0])  
    plot_meridional_velocity_convergence_data = plot_these_variables[2]
    if plot_meridional_velocity_convergence_data:            
        yLabel = 'L2 Error Norm of Numerical Meridional Velocity'
        Title = 'Convergence Plot w.r.t. L2 Error Norm' 
        FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                       + '_NumericalMeridionalVelocityConvergencePlot_SpaceAndTime_L2ErrorNorm')     
        dc, MeridionalVelocityL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
        if not(plotAgainstCellWidthInverse):
            dc = 1.0/dc 
        if not(usePlotly):
            CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,MeridionalVelocityL2ErrorNorm,2.0,'-','k',
                                     's',7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,
                                     FigureTitle,False,fig_size=[9.25,9.25],
                                     useDefaultMethodToSpecifyTickFontSize=False,drawGrid=True) 
        else:
            CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',1.0/dc,MeridionalVelocityL2ErrorNorm,'black',
                                       2.0,'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,25.0,True,
                                       FigureTitle,False,fig_size=[700.0,700.0])  
    plot_normal_velocity_convergence_data = plot_these_variables[3]
    if plot_normal_velocity_convergence_data:          
        yLabel = 'L2 Error Norm of Numerical Normal Velocity'
        Title = 'Convergence Plot w.r.t. L2 Error Norm' 
        FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                       + '_NumericalNormalVelocityConvergencePlot_SpaceAndTime_L2ErrorNorm')        
        dc, NormalVelocityL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
        if not(plotAgainstCellWidthInverse):
            dc = 1.0/dc 
        if not(usePlotly):
            CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,NormalVelocityL2ErrorNorm,2.0,'-','k','s',
                                     7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,
                                     FigureTitle,False,fig_size=[9.25,9.25],
                                     useDefaultMethodToSpecifyTickFontSize=False,drawGrid=True) 
        else:
            CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',1.0/dc,NormalVelocityL2ErrorNorm,'black',2.0,
                                       'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,25.0,True,
                                       FigureTitle,False,fig_size=[700.0,700.0])  
    plot_tangential_velocity_convergence_data = plot_these_variables[4]
    if plot_tangential_velocity_convergence_data:     
        yLabel = 'L2 Error Norm of Numerical Tangential Velocity'
        Title = 'Convergence Plot w.r.t. L2 Error Norm'
        FigureTitle = (wave_type_figure_title + '_' + wave_nature + '_' + time_integrator_short_form
                       + '_NumericalTangentialVelocityConvergencePlot_SpaceAndTime_L2ErrorNorm')      
        dc, TangentialVelocityL2ErrorNorm = CR.ReadCurve1D(output_directory,FigureTitle+'.curve')
        if not(plotAgainstCellWidthInverse):
            dc = 1.0/dc 
        if not(usePlotly):
            CR.PythonPlot1DSaveAsPNG(output_directory,'log-log',1.0/dc,TangentialVelocityL2ErrorNorm,2.0,'-','k',
                                     's',7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,
                                     FigureTitle,False,fig_size=[9.25,9.25],
                                     useDefaultMethodToSpecifyTickFontSize=False,drawGrid=True)
        else:
            CR.PythonPlotly1DSaveAsPNG(output_directory,'log-log',1.0/dc,TangentialVelocityL2ErrorNorm,'black',
                                       2.0,'square',10.0,[xLabel,yLabel],[20.0,20.0],[17.5,17.5],Title,25.0,True,
                                       FigureTitle,False,fig_size=[700.0,700.0])


# In[25]:

do_PlotConvergenceData_LinearCoastalKelvinWave = False
if do_PlotConvergenceData_LinearCoastalKelvinWave:
    problem_type = 'Coastal_Kelvin_Wave'
    problem_is_linear = True
    time_integrator = 'forward_backward_predictor'
    plot_these_variables = np.zeros(8,dtype=bool)
    plot_these_variables[0] = True # plot_these_variables[0] = plot_surface_elevation_convergence_data
    plot_these_variables[3] = True # plot_these_variables[3] = plot_normal_velocity_convergence_data
    plotAgainstCellWidthInverse = True
    usePlotly = False
    PlotConvergenceData(problem_type,problem_is_linear,time_integrator,plot_these_variables,
                        plotAgainstCellWidthInverse,usePlotly)