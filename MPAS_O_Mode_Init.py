
# coding: utf-8

# Name: MPAS_O_Mode_Init.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code initializes the MPAS_O class after acquiring the relevant information from the mesh and initial condition files. <br/>

# In[1]:

import numpy as np
import sympy as sp
import io as inputoutput
import os
from IPython.utils import io
import netCDF4 as nc
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
with io.capture_output() as captured:
    import Common_Routines as CR
    import GeophysicalWaves_ExactSolutions_SourceTerms as GWESST
    import fixAngleEdge


# In[2]:

class Namelist:
    
    def __init__(myNamelist,mesh_type='uniform',problem_type='default',problem_is_linear=True,
                 periodicity='Periodic',time_integrator='Forward_Backward',
                 LF_TR_and_LF_AM3_with_FB_Feedback_Type='FourthOrderAccurate_MaximumStabilityRange',
                 Generalized_FB_with_AB2_AM3_Step_Type='FourthOrderAccurate',
                 Generalized_FB_with_AB3_AM4_Step_Type='FourthOrderAccurate_MaximumStabilityRange'):
        myNamelist.config_mesh_type = mesh_type
        myNamelist.config_problem_type = problem_type
        if (problem_type == 'Coastal_Kelvin_Wave' or problem_type == 'Inertia_Gravity_Wave'
            or problem_type == 'Inertia_Gravity_Waves' or problem_type == 'Planetary_Rossby_Wave' 
            or problem_type == 'Topographic_Rossby_Wave' or problem_type == 'Equatorial_Kelvin_Wave' 
            or problem_type == 'Equatorial_Yanai_Wave' or problem_type == 'Equatorial_Rossby_Wave'
            or problem_type == 'Equatorial_Inertia_Gravity_Wave'):
            myNamelist.config_problem_type_Geophysical_Wave = True
        else:
            myNamelist.config_problem_type_Geophysical_Wave = False            
        if (problem_type == 'Equatorial_Kelvin_Wave' or problem_type == 'Equatorial_Yanai_Wave' 
            or problem_type == 'Equatorial_Rossby_Wave' or problem_type == 'Equatorial_Inertia_Gravity_Wave'):
            myNamelist.config_problem_type_Equatorial_Wave = True
        else:
            myNamelist.config_problem_type_Equatorial_Wave = False  
        myNamelist.config_problem_is_linear = problem_is_linear
        myNamelist.config_periodicity = periodicity
        myNamelist.config_time_integrator = time_integrator
        if time_integrator == 'Forward_Backward_with_RK2_Feedback':
            myNamelist.config_Forward_Backward_with_RK2_Feedback_parameter_beta = 1.0/3.0
            myNamelist.config_Forward_Backward_with_RK2_Feedback_parameter_epsilon = 2.0/3.0
        elif time_integrator == 'LF_TR_and_LF_AM3_with_FB_Feedback':
            myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_Type = LF_TR_and_LF_AM3_with_FB_Feedback_Type
            if LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'SecondOrderAccurate_LF_TR':
                beta = 0.0
                gamma = 0.0
                epsilon = 0.0
            elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'ThirdOrderAccurate_LF_AM3': 
                beta = 0.0
                gamma = 1.0/12.0
                epsilon = 0.0
            elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'ThirdOrderAccurate_MaximumStabilityRange': 
                beta = 0.126
                gamma = 1.0/12.0
                epsilon = 0.83
            elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'FourthOrderAccurate_MinimumTruncationError': 
                beta = 17.0/120.0
                gamma = 1.0/12.0
                epsilon = 11.0/20.0 
            elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'FourthOrderAccurate_MaximumStabilityRange': 
                epsilon = 0.7166
                beta = 7.0/30.0 - epsilon/6.0
                gamma = 1.0/12.0
            myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta = beta
            myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma = gamma
            myNamelist.config_LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon = epsilon
        elif time_integrator == 'Generalized_FB_with_AB2_AM3_Step':
            myNamelist.config_Generalized_FB_with_AB2_AM3_Step_Type = Generalized_FB_with_AB2_AM3_Step_Type  
            if Generalized_FB_with_AB2_AM3_Step_Type == 'ThirdOrderAccurate_WideStabilityRange':
                beta = 0.0
            elif (Generalized_FB_with_AB2_AM3_Step_Type 
                  == 'ThirdOrderAccurate_WeakAsymptoticInstabilityOfPhysicalModes'):
                beta = 0.5    
            elif Generalized_FB_with_AB2_AM3_Step_Type == 'FourthOrderAccurate':
                useSymPyToDetermineBeta = False
                # Note that if useSymPyToDetermineBeta is specified as True, every time beta is used, the SymPy 
                # polynomial equation solver will be executed, resulting in immense slowdown of the code.
                if useSymPyToDetermineBeta:
                    symbolic_beta = sp.Symbol('beta')
                    beta_roots = sp.solve(-symbolic_beta**3.0 - symbolic_beta/12.0 + 1.0/12.0, symbolic_beta)
                    beta = beta_roots[0]   
                else:
                    beta = 0.373707625197906
            gamma = beta - 2.0*beta**2.0 - 1.0/6.0
            epsilon = beta**2.0 + 1.0/12.0
            myNamelist.config_Generalized_FB_with_AB2_AM3_Step_parameter_beta = beta
            myNamelist.config_Generalized_FB_with_AB2_AM3_Step_parameter_gamma = gamma
            myNamelist.config_Generalized_FB_with_AB2_AM3_Step_parameter_epsilon = epsilon
        if time_integrator == 'Generalized_FB_with_AB3_AM4_Step':
            myNamelist.config_Generalized_FB_with_AB3_AM4_Step_Type = Generalized_FB_with_AB3_AM4_Step_Type
            if Generalized_FB_with_AB3_AM4_Step_Type == 'SecondOrderAccurate_OptimumChoice_ROMS':   
                beta = 0.281105
                gamma = 0.088
                epsilon = 0.013     
            elif Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_AB3_AM4':
                beta = 5.0/12.0     
                gamma = -1.0/12.0
                epsilon = 0.0
            elif Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_MaximumStabilityRange':
                beta = 0.232
                epsilon = 0.00525
                gamma = 1.0/3.0 - beta - 3.0*epsilon
            elif Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_OptimumChoice':   
                beta = 0.21
                epsilon = 0.0115
                gamma = 1.0/3.0 - beta - 3.0*epsilon             
            elif Generalized_FB_with_AB3_AM4_Step_Type == 'FourthOrderAccurate_MaximumStabilityRange':
                epsilon = 0.083
                gamma = 0.25 - 2.0*epsilon
                beta = 1.0/12.0 - epsilon            
            delta = 0.5 + gamma + 2.0*epsilon
            myNamelist.config_Generalized_FB_with_AB3_AM4_Step_parameter_beta = beta
            myNamelist.config_Generalized_FB_with_AB3_AM4_Step_parameter_gamma = gamma
            myNamelist.config_Generalized_FB_with_AB3_AM4_Step_parameter_epsilon = epsilon    
            myNamelist.config_Generalized_FB_with_AB3_AM4_Step_parameter_delta = delta 
        myNamelist.config_bottom_depth_parameter = 1000.0
        myNamelist.config_bottom_slope = 0.0
        myNamelist.config_Coriolis_parameter = 10.0**(-4.0)
        myNamelist.config_meridional_gradient_of_Coriolis_parameter = 2.0*10.0**(-11.0)
        myNamelist.config_gravity = 10.0
        myNamelist.config_mean_depth = 1000.0
        if (problem_type == 'default' or problem_type == 'Coastal_Kelvin_Wave'
            or myNamelist.config_problem_type_Equatorial_Wave):
            myNamelist.config_surface_elevation_amplitude = 0.001
        elif (problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Planetary_Rossby_Wave' 
              or problem_type == 'Topographic_Rossby_Wave' or problem_type == 'Diffusion_Equation'):
            myNamelist.config_surface_elevation_amplitude = 1.0
        elif myNamelist.config_problem_type == 'Barotropic_Tide':
            myNamelist.config_surface_elevation_amplitude = 2.0
        elif myNamelist.config_problem_type == 'Viscous_Burgers_Equation':
            myNamelist.config_surface_elevation_amplitude = 0.0
        myNamelist.config_thickness_flux_type = 'centered'
        myNamelist.config_use_wetting_drying = False
        # Derived Parameters
        myNamelist.config_phase_speed_of_coastal_Kelvin_wave = (
        np.sqrt(myNamelist.config_gravity*myNamelist.config_mean_depth))
        myNamelist.config_radius_of_deformation = (
        myNamelist.config_phase_speed_of_coastal_Kelvin_wave/myNamelist.config_Coriolis_parameter)
        myNamelist.config_equatorial_radius_of_deformation = (
        np.sqrt(myNamelist.config_phase_speed_of_coastal_Kelvin_wave
                /myNamelist.config_meridional_gradient_of_Coriolis_parameter))
        if problem_type == 'default' or problem_type == 'Coastal_Kelvin_Wave':
            myNamelist.config_dt = 180.0
        elif problem_type == 'Inertia_Gravity_Wave':
            myNamelist.config_dt = 96.0
        elif problem_type == 'Planetary_Rossby_Wave' or problem_type == 'Topographic_Rossby_Wave':
            myNamelist.config_dt = 195000.0
        elif problem_type == 'Equatorial_Kelvin_Wave':
            myNamelist.config_dt = 750.00 
        elif problem_type == 'Equatorial_Yanai_Wave':
            myNamelist.config_dt = 390.00   
        elif problem_type == 'Equatorial_Rossby_Wave':
            myNamelist.config_dt = 2700.00
        elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
            myNamelist.config_dt = 420.00
        elif problem_type == 'Barotropic_Tide':
            myNamelist.config_dt = 10.00    
        elif problem_type == 'Diffusion_Equation':
            myNamelist.config_dt = 2260.0 
        elif problem_type == 'Viscous_Burgers_Equation':
            myNamelist.config_dt = 2100.00  
        if problem_is_linear:
            myNamelist.config_linearity_prefactor = 0.0
        else:
            myNamelist.config_linearity_prefactor = 1.0 
        if problem_type == 'Viscous_Burgers_Equation':
            myNamelist.config_zonal_diffusivity = 5000.0
        else:
            myNamelist.config_zonal_diffusivity = 500.0
        myNamelist.config_meridional_diffusivity = 500.0
        myNamelist.config_viscous_Burgers_zonal_velocity_left = 1.0
        myNamelist.config_viscous_Burgers_zonal_velocity_right = 0.0
        myNamelist.config_viscous_Burgers_shock_speed = (
        0.5*(myNamelist.config_viscous_Burgers_zonal_velocity_left
             + myNamelist.config_viscous_Burgers_zonal_velocity_right))
        myNamelist.config_viscous_Burgers_zonal_offset = 0.0


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

def DetermineExactSolutionParameters(myMPAS_O,printPhaseSpeedOfWaveModes):
    angleEdge = 0.0
    beta0 = myMPAS_O.myNamelist.config_meridional_gradient_of_Coriolis_parameter
    c = myMPAS_O.myNamelist.config_phase_speed_of_coastal_Kelvin_wave
    dx = myMPAS_O.gridSpacingMagnitude # i.e. dx = myMPAS_O.dcEdge[0]
    dy = np.sqrt(3.0)/2.0*dx
    etaHat1 = myMPAS_O.myNamelist.config_surface_elevation_amplitude
    if myMPAS_O.myNamelist.config_problem_type == 'Viscous_Burgers_Equation':
        etaHat2 = 0.0
    else:
        etaHat2 = 2.0*etaHat1
    f0 = myMPAS_O.myNamelist.config_Coriolis_parameter
    g = myMPAS_O.myNamelist.config_gravity
    H = myMPAS_O.myNamelist.config_mean_depth
    if myMPAS_O.myNamelist.config_problem_type == 'Topographic_Rossby_Wave':
        alpha0 = beta0*H/f0
        # In the Northern Hemisphere where f0 > 0, the topographic Rossby wave travels with the shallower water on
        # its right. Hence if alpha0 > 0 i.e. the ocean depth increases northward, the topographic Rossby wave 
        # will propagate eastward else it will propagate westward.
        myMPAS_O.myNamelist.config_bottom_slope = alpha0
    else:
        alpha0 = myMPAS_O.myNamelist.config_bottom_slope # alpha0 = 0.0
    if myMPAS_O.myNamelist.config_problem_type == 'Barotropic_Tide':
        kX1 = 2.5*np.pi/myMPAS_O.lX
        kY1 = 0.0
        kX2 = 4.5*np.pi/myMPAS_O.lX
        kY2 = 0.0 
    elif myMPAS_O.myNamelist.config_problem_type == 'Viscous_Burgers_Equation':    
        kX1 = 0.0
        kY1 = 0.0
        kX2 = 0.0
        kY2 = 0.0
    else:
        kX1 = 2.0*np.pi/myMPAS_O.lX
        kY1 = 2.0*np.pi/myMPAS_O.lY
        kX2 = 2.0*kX1
        kY2 = 2.0*kY1
    lX = myMPAS_O.lX
    lY = myMPAS_O.lY
    R = myMPAS_O.myNamelist.config_radius_of_deformation
    Req = myMPAS_O.myNamelist.config_equatorial_radius_of_deformation
    LengthScale = np.sqrt(c/beta0)
    TimeScale = 1.0/np.sqrt(beta0*c)
    VelocityScale = c
    SurfaceElevationScale = c**2.0/g
    if (myMPAS_O.myNamelist.config_problem_type == 'default' 
        or myMPAS_O.myNamelist.config_problem_type == 'Coastal_Kelvin_Wave'):
        omega1 = -c*kY1
        omega2 = -c*kY2
    elif myMPAS_O.myNamelist.config_problem_type == 'Inertia_Gravity_Wave':
        omega1 = np.sqrt(g*H*(kX1**2.0 + kY1**2.0) + f0**2.0)
        omega2 = np.sqrt(g*H*(kX2**2.0 + kY2**2.0) + f0**2.0)
    elif myMPAS_O.myNamelist.config_problem_type == 'Planetary_Rossby_Wave':
        omega1 = -beta0*R**2.0*kX1/(1.0 + R**2.0*(kX1**2.0 + kY1**2.0))
        omega2 = -beta0*R**2.0*kX2/(1.0 + R**2.0*(kX2**2.0 + kY2**2.0))
    elif myMPAS_O.myNamelist.config_problem_type == 'Topographic_Rossby_Wave':
        omega1 = alpha0*g/f0*kX1/(1.0 + R**2.0*(kX1**2.0 + kY1**2.0))
        omega2 = alpha0*g/f0*kX2/(1.0 + R**2.0*(kX2**2.0 + kY2**2.0))    
    elif myMPAS_O.myNamelist.config_problem_type == 'Equatorial_Kelvin_Wave':
        omega1 = c*kX1
        omega2 = c*kX2
    elif myMPAS_O.myNamelist.config_problem_type == 'Equatorial_Yanai_Wave':
        omega1 = GWESST.DetermineEquatorialYanaiWaveNonDimensionalAngularFrequency(LengthScale*kX1)/TimeScale
        omega2 = GWESST.DetermineEquatorialYanaiWaveNonDimensionalAngularFrequency(LengthScale*kX2)/TimeScale
    elif myMPAS_O.myNamelist.config_problem_type == 'Equatorial_Rossby_Wave':
        omega1 = GWESST.DetermineEquatorialRossbyAndInertiaGravityWaveNonDimensionalAngularFrequency(
                 'Equatorial_Rossby_Wave',LengthScale*kX1,m=1)/TimeScale
        omega2 = GWESST.DetermineEquatorialRossbyAndInertiaGravityWaveNonDimensionalAngularFrequency(
                 'Equatorial_Rossby_Wave',LengthScale*kX2,m=1)/TimeScale
    elif myMPAS_O.myNamelist.config_problem_type == 'Equatorial_Inertia_Gravity_Wave':
        omega1 = GWESST.DetermineEquatorialRossbyAndInertiaGravityWaveNonDimensionalAngularFrequency(
                 'Equatorial_Inertia_Gravity_Wave',LengthScale*kX1,m=2)/TimeScale
        omega2 = GWESST.DetermineEquatorialRossbyAndInertiaGravityWaveNonDimensionalAngularFrequency(
                 'Equatorial_Inertia_Gravity_Wave',LengthScale*kX2,m=2)/TimeScale        
    elif myMPAS_O.myNamelist.config_problem_type == 'Barotropic_Tide':
        omega1 = np.sqrt(g*H*kX1**2.0 + f0**2.0)
        omega2 = np.sqrt(g*H*kX2**2.0 + f0**2.0)
    elif (myMPAS_O.myNamelist.config_problem_type == 'Diffusion_Equation'
          or myMPAS_O.myNamelist.config_problem_type == 'Viscous_Burgers_Equation'):
        omega1 = 0.0
        omega2 = 0.0    
    if (myMPAS_O.myNamelist.config_problem_type == 'default' 
        or myMPAS_O.myNamelist.config_problem_type == 'Coastal_Kelvin_Wave'):  
        cX1 = 0.0
        cY1 = -c # cY1 = omega1/kY1 = -c
        cX2 = 0.0
        cY2 = -c # cY2 = omega2/kY2 = -c
    elif myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave:
        if myMPAS_O.myNamelist.config_problem_type == 'Equatorial_Kelvin_Wave':
            cX1 = c
            cY1 = 0.0
            cX2 = c
            cY2 = 0.0
        else:
            cX1 = omega1/kX1
            cY1 = 0.0
            cX2 = omega2/kX2
            cY2 = 0.0              
    elif myMPAS_O.myNamelist.config_problem_type == 'Barotropic_Tide':
        cX1 = omega1/kX1
        cY1 = 0.0
        cX2 = omega2/kX2
        cY2 = 0.0
    elif (myMPAS_O.myNamelist.config_problem_type == 'Diffusion_Equation' 
          or myMPAS_O.myNamelist.config_problem_type == 'Viscous_Burgers_Equation'):
        cX1 = 0.0
        cY1 = 0.0
        cX2 = 0.0
        cY2 = 0.0
    else:
        cX1 = omega1/kX1
        cY1 = omega1/kY1
        cX2 = omega2/kX2
        cY2 = omega2/kY2         
    if ((myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave 
         or myMPAS_O.myNamelist.config_problem_type == 'Barotropic_Tide') and printPhaseSpeedOfWaveModes):
        print('The zonal component of the phase speed of the first wave mode is %.4g.' %cX1)
        print('The meridional component of the phase speed of the first wave mode is %.4g.' %cY1)
        print('The zonal component of the phase speed of the second wave mode is %.4g.' %cX2)
        print('The meridional component of the phase speed of the second wave mode is %.4g.' %cY2)
    kappaX = myMPAS_O.myNamelist.config_zonal_diffusivity
    kappaY = myMPAS_O.myNamelist.config_meridional_diffusivity
    kappa1 = kappaX*kX1**2.0 + kappaY*kY1**2.0
    kappa2 = kappaX*kX2**2.0 + kappaY*kY2**2.0
    uL = myMPAS_O.myNamelist.config_viscous_Burgers_zonal_velocity_left
    uR = myMPAS_O.myNamelist.config_viscous_Burgers_zonal_velocity_right
    s = myMPAS_O.myNamelist.config_viscous_Burgers_shock_speed
    myMPAS_O.myNamelist.config_viscous_Burgers_zonal_offset = 0.25*lX
    x0 = myMPAS_O.myNamelist.config_viscous_Burgers_zonal_offset
    ExactSolutionParameters = [alpha0,angleEdge,beta0,c,cX1,cX2,cY1,cY2,dx,dy,etaHat1,etaHat2,f0,g,H,kX1,kX2,kY1,
                               kY2,lX,lY,omega1,omega2,R,Req,LengthScale,TimeScale,VelocityScale,
                               SurfaceElevationScale,kappaX,kappaY,kappa1,kappa2,uL,uR,s,x0]  
    return ExactSolutionParameters


# In[6]:

def DetermineCoriolisParameterAndBottomDepth(myMPAS_O):
    alpha0 = myMPAS_O.ExactSolutionParameters[0]
    beta0 = myMPAS_O.ExactSolutionParameters[2]
    f0 = myMPAS_O.ExactSolutionParameters[12]
    H = myMPAS_O.ExactSolutionParameters[14]
    if (myMPAS_O.myNamelist.config_problem_type == 'default' 
        or myMPAS_O.myNamelist.config_problem_type == 'Barotropic_Tide' 
        or myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave):
        if myMPAS_O.myNamelist.config_problem_type == 'Planetary_Rossby_Wave':
            myMPAS_O.fCell[:] = f0 + beta0*myMPAS_O.yCell[:]        
            myMPAS_O.fEdge[:] = f0 + beta0*myMPAS_O.yEdge[:]
            myMPAS_O.fVertex[:] = f0 + beta0*myMPAS_O.yVertex[:]
        else:
            myMPAS_O.fCell[:] = f0      
            myMPAS_O.fEdge[:] = f0
            myMPAS_O.fVertex[:] = f0
        if myMPAS_O.myNamelist.config_problem_type == 'Topographic_Rossby_Wave':
            myMPAS_O.bottomDepth[:] = H + alpha0*yCell[:]
        else:
            myMPAS_O.bottomDepth[:] = H


# In[7]:

class MPAS_O:
    
    def __init__(myMPAS_O,print_basic_geometry,mesh_directory='Mesh+Initial_Condition+Registry_Files/Periodic',
                 base_mesh_file_name='base_mesh.nc',mesh_file_name='mesh.nc',mesh_type='uniform',
                 problem_type='default',problem_is_linear=True,periodicity='Periodic',do_fixAngleEdge=True,
                 print_Output=False,CourantNumber=0.5,useCourantNumberToDetermineTimeStep=False,
                 time_integrator='Forward_Backward',
                 LF_TR_and_LF_AM3_with_FB_Feedback_Type='FourthOrderAccurate_MaximumStabilityRange',
                 Generalized_FB_with_AB2_AM3_Step_Type='FourthOrderAccurate',
                 Generalized_FB_with_AB3_AM4_Step_Type='FourthOrderAccurate_MaximumStabilityRange',
                 printPhaseSpeedOfWaveModes=False,specifyExactSurfaceElevationAtNonPeriodicBoundaryCells=False):
        myMPAS_O.myNamelist = Namelist(mesh_type,problem_type,problem_is_linear,periodicity,time_integrator,
                                       LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                       Generalized_FB_with_AB2_AM3_Step_Type,
                                       Generalized_FB_with_AB3_AM4_Step_Type)
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
        myMPAS_O.nNonPeriodicBoundaryVertices = 0
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
        dx = myMPAS_O.gridSpacingMagnitude # i.e. dx = myMPAS_O.dcEdge[0]
        dy = np.sqrt(3.0)/2.0*dx
        if periodicity == 'NonPeriodic_x' or periodicity == 'NonPeriodic_xy':
            myMPAS_O.xCell[:] -= dx
            myMPAS_O.xVertex[:] -= dx
            myMPAS_O.xEdge[:] -= dx
        if periodicity == 'NonPeriodic_y' or periodicity == 'NonPeriodic_xy':
            myMPAS_O.yCell[:] -= dy
            myMPAS_O.yVertex[:] -= dy
            myMPAS_O.yEdge[:] -= dy
        # Specify the zonal and meridional extents of the domain.
        myMPAS_O.lX = round(max(myMPAS_O.xCell)) # i.e. myMPAS_O.lX = np.sqrt(float(myMPAS_O.nCells))*dx
        # Please note that for all of our test problems, the MPAS-O mesh is generated in such a way that 
        # myMPAS_O.lX i.e. the number of cells in the x (or y) direction times dx has zero fractional part in 
        # units of m, for which we can afford to round it to attain perfection.
        myMPAS_O.lY = np.sqrt(3.0)/2.0*myMPAS_O.lX # i.e. myMPAS_O.lY = max(myMPAS_O.yVertex)
        if myMPAS_O.myNamelist.config_problem_type_Equatorial_Wave and periodicity == 'NonPeriodic_y':
            myMPAS_O.yCell[:] -= 0.5*myMPAS_O.lY
            myMPAS_O.yVertex[:] -= 0.5*myMPAS_O.lY
            myMPAS_O.yEdge[:] -= 0.5*myMPAS_O.lY             
        myMPAS_O.ExactSolutionParameters = DetermineExactSolutionParameters(myMPAS_O,printPhaseSpeedOfWaveModes)
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
        if (myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Second_Order' or
            myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Third_Order' or
            myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Fourth_Order' or
            myMPAS_O.myNamelist.config_time_integrator == 'LF_TR_and_LF_AM3_with_FB_Feedback' or
            myMPAS_O.myNamelist.config_time_integrator == 'Generalized_FB_with_AB2_AM3_Step' or
            myMPAS_O.myNamelist.config_time_integrator == 'Generalized_FB_with_AB3_AM4_Step'):
            myMPAS_O.normalVelocityTendencyCurrent = np.zeros((nEdges,nVertLevels))
            myMPAS_O.sshTendencyCurrent = np.zeros(nCells)
            myMPAS_O.normalVelocityTendencyLast = np.zeros((nEdges,nVertLevels))
            myMPAS_O.sshTendencyLast = np.zeros(nCells)
        if (myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Third_Order' or
            myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Fourth_Order' or
            myMPAS_O.myNamelist.config_time_integrator == 'Generalized_FB_with_AB3_AM4_Step'):
            myMPAS_O.normalVelocityTendencySecondLast = np.zeros((nEdges,nVertLevels))
            myMPAS_O.sshTendencySecondLast = np.zeros(nCells)
        if myMPAS_O.myNamelist.config_time_integrator == 'Adams_Bashforth_Fourth_Order':            
            myMPAS_O.normalVelocityTendencyThirdLast = np.zeros((nEdges,nVertLevels))
            myMPAS_O.sshTendencyThirdLast = np.zeros(nCells)
        if (myMPAS_O.myNamelist.config_time_integrator == 'Leapfrog_Trapezoidal' 
            or myMPAS_O.myNamelist.config_time_integrator == 'LF_TR_and_LF_AM3_with_FB_Feedback'):
            myMPAS_O.normalVelocityLast = np.zeros((nEdges,nVertLevels))
            myMPAS_O.sshLast = np.zeros(nCells)
        # Remember that if the dimension of a variable x defined in the Registry file is "nX nY nZ" then should be
        # specified as "nZ nY nX" due to the column-major vs row-major ordering of arrays in Fortran vs Python.
        myMPAS_O.iTime = 0
        myMPAS_O.time = 0.0
        os.chdir(cwd)
        if do_fixAngleEdge:
            if print_Output:
                print(' ')
            myMPAS_O.angleEdge[:] = (
            fixAngleEdge.fix_angleEdge(mesh_directory,my_mesh_file_name,determineYCellAlongLatitude=True,
                                       printOutput=print_Output,printRelevantMeshData=False))
        if useCourantNumberToDetermineTimeStep: 
            dx = myMPAS_O.ExactSolutionParameters[8]
            dy = myMPAS_O.ExactSolutionParameters[9]
            if myMPAS_O.myNamelist.config_problem_type_Geophysical_Wave or problem_type == 'Barotropic_Tide': 
                cX1 = myMPAS_O.ExactSolutionParameters[4]
                cX2 = myMPAS_O.ExactSolutionParameters[5]
                cY1 = myMPAS_O.ExactSolutionParameters[6]
                cY2 = myMPAS_O.ExactSolutionParameters[7]
                abs_cX = max(abs(cX1),abs(cX2))
                abs_cY = max(abs(cY1),abs(cY2))
                myMPAS_O.myNamelist.config_dt = CourantNumber/(abs_cX/dx + abs_cY/dy)   
                # The time step for a given Courant number is obtained using the maximum magnitudes of the zonal 
                # and meridional phase speeds of both wave modes, which results in a smaller i.e. more restrictive
                # time step.
                print('The time step for Courant number %.6f is %.2f seconds.' 
                      %(CourantNumber,myMPAS_O.myNamelist.config_dt))
            elif problem_type == 'Diffusion_Equation':                 
                kappaX = myMPAS_O.ExactSolutionParameters[29]
                kappaY = myMPAS_O.ExactSolutionParameters[30]
                myMPAS_O.myNamelist.config_dt = CourantNumber/(kappaX/dx**2.0 + kappaY/dy**2.0)
                print('The time step for stability coefficient %.6f is %.2f seconds.' 
                      %(CourantNumber,myMPAS_O.myNamelist.config_dt))                   
            elif problem_type == 'Viscous_Burgers_Equation':     
                s = myMPAS_O.ExactSolutionParameters[35]
                myMPAS_O.myNamelist.config_dt = CourantNumber*dx/s
                print('The time step for Courant number %.6f is %.2f seconds.' 
                      %(CourantNumber,myMPAS_O.myNamelist.config_dt))               
        myMPAS_O.specifyExactSurfaceElevationAtNonPeriodicBoundaryCells = (
        specifyExactSurfaceElevationAtNonPeriodicBoundaryCells)


# In[8]:

test_MPAS_O_11 = False
if test_MPAS_O_11:
    myMPAS_O = MPAS_O(True,print_Output=False)


# In[9]:

test_MPAS_O_12 = False
if test_MPAS_O_12:
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


# In[10]:

test_MPAS_O_13 = False
if test_MPAS_O_13:
    print_basic_geometry = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_y'
    base_mesh_file_name = 'culled_mesh.nc'
    # If you specify the base_mesh_file_name to be base_mesh.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_y'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)


# In[11]:

test_MPAS_O_14 = False
if test_MPAS_O_14:
    print_basic_geometry = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_xy'
    base_mesh_file_name = 'culled_mesh.nc'
    # If you specify the base_mesh_file_name to be base_mesh.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_xy'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)


# In[12]:

test_MPAS_O_21 = False
if test_MPAS_O_21:
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


# In[13]:

test_MPAS_O_22 = False
if test_MPAS_O_22:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_x.nc'
    # If you specify the base_mesh_file_name to be base_mesh_NP.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh_NP_x.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=True)


# In[14]:

test_MPAS_O_23 = False
if test_MPAS_O_23:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_y.nc'
    # If you specify the base_mesh_file_name to be base_mesh_NP.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh_NP_y.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_y'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=True)


# In[15]:

test_MPAS_O_24 = False
if test_MPAS_O_24:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_xy.nc'
    # If you specify the base_mesh_file_name to be base_mesh_NP.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh_NP_xy.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_xy'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=True)


# In[16]:

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
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/1000.0), '')))
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda y, p: format(int(y/1000.0), '')))
    ax.set_title(title,fontsize=titlefontsize,y=1.035)
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[17]:

test_PlotMesh_11 = False
if test_PlotMesh_11:
    print_basic_geometry = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/Periodic'
    base_mesh_file_name = 'base_mesh.nc'
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'Periodic'
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                     'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[18]:

test_PlotMesh_12 = False
if test_PlotMesh_12:
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
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                     'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[19]:

test_PlotMesh_13 = False
if test_PlotMesh_13:
    print_basic_geometry = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_y'
    base_mesh_file_name = 'culled_mesh.nc'
    # If you specify the base_mesh_file_name to be base_mesh.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_y'
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                     'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[20]:

test_PlotMesh_14 = False
if test_PlotMesh_14:
    print_basic_geometry = True
    mesh_directory = 'Mesh+Initial_Condition+Registry_Files/NonPeriodic_xy'
    base_mesh_file_name = 'culled_mesh.nc'
    # If you specify the base_mesh_file_name to be base_mesh.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_xy'
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                     'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[21]:

test_PlotMesh_21 = False
if test_PlotMesh_21:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'base_mesh_P.nc'
    mesh_file_name = 'mesh_P.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'Periodic'
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                     'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh_P',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[22]:

test_PlotMesh_22 = False
if test_PlotMesh_22:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_x.nc'
    # If you specify the base_mesh_file_name to be base_mesh_NP.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh_NP_x.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_x'
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                     'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh_NP_x',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[23]:

test_PlotMesh_23 = False
if test_PlotMesh_23:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_y.nc'
    # If you specify the base_mesh_file_name to be base_mesh_NP.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh_NP_y.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_y'
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                     'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh_NP_y',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)


# In[24]:

test_PlotMesh_24 = False
if test_PlotMesh_24:
    print_basic_geometry = True
    mesh_directory = 'MPAS_O_Shallow_Water_Mesh_Generation/CoastalKelvinWaveMesh/PlotMesh'
    base_mesh_file_name = 'culled_mesh_NP_xy.nc'
    # If you specify the base_mesh_file_name to be base_mesh_NP.nc and my_mesh_file_name to be base_mesh_file_name,
    # the fixAngle routine will not work unless you also specify determineYCellAlongLatitude to be False.
    mesh_file_name = 'mesh_NP_xy.nc'
    mesh_type = 'uniform'
    problem_type = 'default'
    problem_is_linear = True
    periodicity = 'NonPeriodic_xy'
    xLabel = 'Zonal Distance (km)'
    yLabel = 'Meridional Distance (km)'
    myMPAS_O = MPAS_O(print_basic_geometry,mesh_directory,base_mesh_file_name,mesh_file_name,mesh_type,
                      problem_type,problem_is_linear,periodicity,do_fixAngleEdge=True,print_Output=False)
    Plot_MPAS_O_Mesh(myMPAS_O,mesh_directory,2.0,'-','k',[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                     'MPAS-O Mesh',20.0,True,'MPAS_O_Mesh_NP_xy',False,fig_size=[9.25,9.25],
                     useDefaultMethodToSpecifyTickFontSize=True)