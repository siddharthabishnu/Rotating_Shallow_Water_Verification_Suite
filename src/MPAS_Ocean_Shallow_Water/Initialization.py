"""
Name: Initialization.py
Author: Sid Bishnu
Details: This script specifies the parameters required to initialize the MPAS-Ocean shallow water class.
"""


import numpy as np
import sympy as sp
from IPython.utils import io
with io.capture_output() as captured:
    import ExactSolutionsAndSourceTerms as ESST
    
    
def isGeophysicalWave(ProblemType):
    if (ProblemType == 'Coastal_Kelvin_Wave' or ProblemType == 'Inertia_Gravity_Wave' 
        or ProblemType == 'Manufactured_Planetary_Rossby_Wave' or ProblemType == 'Manufactured_Topographic_Rossby_Wave'
        or ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Equatorial_Yanai_Wave' 
        or ProblemType == 'Equatorial_Rossby_Wave' or ProblemType == 'Equatorial_Inertia_Gravity_Wave'):
        ProblemType_GeophysicalWave = True
    else:
        ProblemType_GeophysicalWave = False
    return ProblemType_GeophysicalWave
    
    
def isEquatorialWave(ProblemType):
    if (ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Equatorial_Yanai_Wave' 
        or ProblemType == 'Equatorial_Rossby_Wave' or ProblemType == 'Equatorial_Inertia_Gravity_Wave'):
        ProblemType_EquatorialWave = True
    else:
        ProblemType_EquatorialWave = False
    return ProblemType_EquatorialWave


def Specify_ProblemType_ManufacturedRossbyWave(ProblemType):
    if ProblemType == 'Manufactured_Planetary_Rossby_Wave' or ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        ProblemType_ManufacturedRossbyWave = True
    else:
        ProblemType_ManufacturedRossbyWave = False
    return ProblemType_ManufacturedRossbyWave


def Specify_ProblemType_RossbyWave(ProblemType):
    if ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        ProblemType_RossbyWave = True
    else:
        ProblemType_RossbyWave = False
    return ProblemType_RossbyWave


def SpecifyExactSolutionLimits(ProblemType):
    if (ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Coastal_Kelvin_Wave' 
        or ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Inertia_Gravity_Wave' 
        or ProblemType == 'Barotropic_Tide' or ProblemType == 'Diffusion_Equation' 
        or ProblemType == 'Advection_Diffusion_Equation' or ProblemType == 'NonLinear_Manufactured_Solution' 
        or ProblemType == 'Viscous_Burgers_Equation'):
        ProblemType_SpecifyExactSolutionLimits = True
    else:
        ProblemType_SpecifyExactSolutionLimits = False
    return ProblemType_SpecifyExactSolutionLimits


def SpecifyBoundaryCondition(ProblemType,ProblemType_ManufacturedRossbyWave,ProblemType_RossbyWave,
                             ProblemType_EquatorialWave):
    if (ProblemType == 'Inertia_Gravity_Wave' or ProblemType_RossbyWave or ProblemType == 'Diffusion_Equation' 
        or ProblemType == 'NonLinear_Manufactured_Solution'):
        BoundaryCondition = 'Periodic'
    elif (ProblemType == 'Coastal_Kelvin_Wave' or ProblemType == 'Barotropic_Tide' 
          or ProblemType == 'Viscous_Burgers_Equation'):
        BoundaryCondition = 'NonPeriodic_x'
    elif ProblemType_ManufacturedRossbyWave or ProblemType_EquatorialWave:
        BoundaryCondition = 'NonPeriodic_y'
    else:
        BoundaryCondition = 'NonPeriodic_xy'
    return BoundaryCondition
    
    
def SpecifyMeshDirectoryAndMeshFileNames(ProblemType,ProblemType_ManufacturedRossbyWave,ProblemType_RossbyWave,
                                         ProblemType_EquatorialWave,ReturnOnlyMeshDirectory=False):
    if ProblemType == 'Plane_Gaussian_Wave':
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/PlaneGaussianWaveMesh'
    elif ProblemType == 'Coastal_Kelvin_Wave':
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/CoastalKelvinWaveMesh'
    elif ProblemType == 'Inertia_Gravity_Wave' or ProblemType == 'NonLinear_Manufactured_Solution':
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/InertiaGravityWaveMesh'
    elif ProblemType == ProblemType_ManufacturedRossbyWave:
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/ManufacturedRossbyWaveMesh'
    elif ProblemType_RossbyWave:
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/RossbyWaveMesh'
    elif ProblemType_EquatorialWave:
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/EquatorialWaveMesh'
    elif ProblemType == 'Barotropic_Tide':
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/BarotropicTideMesh'
    elif ProblemType == 'Diffusion_Equation':
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/DiffusionEquationMesh'
    elif ProblemType == 'Advection_Diffusion_Equation':
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/AdvectionDiffusionEquationMesh'
    elif ProblemType == 'Viscous_Burgers_Equation':
        MeshDirectory = '../../meshes/MPAS_Ocean_Shallow_Water_Meshes/ViscousBurgersEquationMesh'
    if (ProblemType == 'Inertia_Gravity_Wave' or ProblemType_RossbyWave or ProblemType == 'Diffusion_Equation'
        or ProblemType == 'NonLinear_Manufactured_Solution'):
        BaseMeshFileName = 'base_mesh.nc'
    else:
        BaseMeshFileName = 'culled_mesh.nc'
    MeshFileName = 'mesh.nc'
    if ReturnOnlyMeshDirectory:
        return MeshDirectory
    else:
        return MeshDirectory, BaseMeshFileName, MeshFileName
    
    
def SpecifyAmplitudes(ProblemType,ProblemType_ManufacturedRossbyWave,ProblemType_RossbyWave,ProblemType_EquatorialWave):
    if (ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Diffusion_Equation' 
        or ProblemType == 'Advection_Diffusion_Equation' or ProblemType == 'Viscous_Burgers_Equation'):
        etaHat1 = 0.0
    elif ProblemType == 'Coastal_Kelvin_Wave' or ProblemType_EquatorialWave:
        etaHat1 = 0.0001
    elif ProblemType == 'Inertia_Gravity_Wave':           
        etaHat1 = 0.1
    elif ProblemType_ManufacturedRossbyWave or ProblemType_RossbyWave:
        etaHat1 = 0.01
    elif ProblemType == 'Barotropic_Tide':
        etaHat1 = 0.2
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        etaHat1 = 0.01
    if ProblemType == 'NonLinear_Manufactured_Solution':
        etaHat2 = 0.0 
    else:
        etaHat2 = 2.0*etaHat1
    return etaHat1, etaHat2


def SpecifyDomainExtents(ProblemType,ProblemType_ManufacturedRossbyWave,ProblemType_RossbyWave,
                         ProblemType_EquatorialWave):
    if (ProblemType == 'Plane_Gaussian_Wave' or ProblemType_RossbyWave or ProblemType == 'Diffusion_Equation' 
        or ProblemType == 'Viscous_Burgers_Equation'):
        lX = 1000.0*1000.0
    elif ProblemType == 'Coastal_Kelvin_Wave':
        lX = 5000.0*1000.0
    elif ProblemType == 'Inertia_Gravity_Wave' or ProblemType == 'NonLinear_Manufactured_Solution':
        lX = 10000.0*1000.0
    elif ProblemType_ManufacturedRossbyWave:    
        lX = 50.0*1000.0
    elif ProblemType_EquatorialWave:
        lX = 17500.0*1000.0
    elif ProblemType == 'Barotropic_Tide':
        lX = 250.0*1000.0
    elif ProblemType == 'Advection_Diffusion_Equation':
        lX = 2.0
    lY = np.sqrt(3.0)/2.0*lX
    return lX, lY


def SpecifyWaveNumbers(ProblemType,lX,lY):
    if ProblemType == 'Plane_Gaussian_Wave':
        kX1 = 1.0/np.sqrt(2.0)
        kY1 = 1.0/np.sqrt(2.0)
        kX2 = 0.0
        kY2 = 0.0
    elif ProblemType == 'Barotropic_Tide':
        kX1 = 2.5*np.pi/lX
        kY1 = 0.0
        kX2 = 4.5*np.pi/lX
        kY2 = 0.0
    elif ProblemType == 'Advection_Diffusion_Equation':
        kX1 = 1.0
        kY1 = 1.0
        kX2 = 0.0
        kY2 = 0.0
    elif ProblemType == 'Viscous_Burgers_Equation':
        kX1 = 0.0
        kY1 = 0.0
        kX2 = 0.0
        kY2 = 0.0
    else:
        kX1 = 2.0*np.pi/lX
        kY1 = 2.0*np.pi/lY
        kX2 = 2.0*kX1
        kY2 = 2.0*kY1
    return kX1, kY1, kX2, kY2


def SpecifyAngularFrequencies(ProblemType,ProblemType_RossbyWave,alpha0,beta0,c0,f0,g,H0,kX1,kY1,kX2,kY2,R,LengthScale,
                              TimeScale):
    if ProblemType == 'Plane_Gaussian_Wave':
        omega1 = c0
        omega2 = 0.0
    elif ProblemType == 'Coastal_Kelvin_Wave':
        omega1 = -c0*kY1
        omega2 = -c0*kY2
    elif ProblemType == 'Inertia_Gravity_Wave':
        omega1 = np.sqrt(g*H0*(kX1**2.0 + kY1**2.0) + f0**2.0)
        omega2 = np.sqrt(g*H0*(kX2**2.0 + kY2**2.0) + f0**2.0)
    elif ProblemType == 'Manufactured_Planetary_Rossby_Wave':
        omega1 = -beta0*R**2.0*kX1/(1.0 + R**2.0*(kX1**2.0 + kY1**2.0))
        omega2 = -beta0*R**2.0*kX2/(1.0 + R**2.0*(kX2**2.0 + kY2**2.0))
    elif ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        omega1 = alpha0*g/f0*kX1/(1.0 + R**2.0*(kX1**2.0 + kY1**2.0))
        omega2 = alpha0*g/f0*kX2/(1.0 + R**2.0*(kX2**2.0 + kY2**2.0))
    elif ProblemType_RossbyWave:
        omega1 = 0.0
        omega2 = 0.0
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        omega1 = c0*kX1
        omega2 = c0*kX2
    elif ProblemType == 'Equatorial_Yanai_Wave':
        omega1 = ESST.DetermineEquatorialYanaiWaveNonDimensionalAngularFrequency(LengthScale*kX1)/TimeScale
        omega2 = ESST.DetermineEquatorialYanaiWaveNonDimensionalAngularFrequency(LengthScale*kX2)/TimeScale
    elif ProblemType == 'Equatorial_Rossby_Wave':
        omega1 = ESST.DetermineEquatorialRossbyAndInertiaGravityWaveNonDimensionalAngularFrequency(
                 'Equatorial_Rossby_Wave',LengthScale*kX1,m=1)/TimeScale
        omega2 = ESST.DetermineEquatorialRossbyAndInertiaGravityWaveNonDimensionalAngularFrequency(
                 'Equatorial_Rossby_Wave',LengthScale*kX2,m=1)/TimeScale
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        omega1 = ESST.DetermineEquatorialRossbyAndInertiaGravityWaveNonDimensionalAngularFrequency(
                 'Equatorial_Inertia_Gravity_Wave',LengthScale*kX1,m=2)/TimeScale
        omega2 = ESST.DetermineEquatorialRossbyAndInertiaGravityWaveNonDimensionalAngularFrequency(
                 'Equatorial_Inertia_Gravity_Wave',LengthScale*kX2,m=2)/TimeScale        
    elif ProblemType == 'Barotropic_Tide':
        omega1 = np.sqrt(g*H0*kX1**2.0 + f0**2.0)
        omega2 = np.sqrt(g*H0*kX2**2.0 + f0**2.0) 
    elif (ProblemType == 'Diffusion_Equation' or ProblemType == 'Advection_Diffusion_Equation' 
          or ProblemType == 'Viscous_Burgers_Equation'):
        omega1 = 0.0
        omega2 = 0.0
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        omega1 = np.sqrt(g*H0*(kX1**2.0 + kY1**2.0))
        omega2 = 0.0
    return omega1, omega2


def SpecifyPhaseSpeeds(ProblemType,ProblemType_RossbyWave,ProblemType_EquatorialWave,c0,kX1,kY1,kX2,kY2,omega1,omega2,
                       u0,v0,s):
    if ProblemType == 'Plane_Gaussian_Wave':
        cX1 = omega1/kX1
        cY1 = omega1/kY1
        cX2 = 0.0
        cY2 = 0.0
    elif ProblemType == 'Coastal_Kelvin_Wave':  
        cX1 = 0.0
        cY1 = -c0 # cY1 = omega1/kY1 = -c0.
        cX2 = 0.0
        cY2 = -c0 # cY2 = omega2/kY2 = -c0.
    elif ProblemType == 'Diffusion_Equation' or ProblemType_RossbyWave:
        cX1 = 0.0
        cY1 = 0.0
        cX2 = 0.0
        cY2 = 0.0
    elif ProblemType == 'Advection_Diffusion_Equation':
        cX1 = u0
        cY1 = v0
        cX2 = 0.0
        cY2 = 0.0
    elif ProblemType == 'Viscous_Burgers_Equation':
        cX1 = s
        cX2 = 0.0
        cY1 = 0.0
        cY2 = 0.0
    elif ProblemType_EquatorialWave:
        if ProblemType == 'Equatorial_Kelvin_Wave':
            cX1 = c0
            cY1 = 0.0
            cX2 = c0
            cY2 = 0.0
        else:
            cX1 = omega1/kX1
            cY1 = 0.0
            cX2 = omega2/kX2
            cY2 = 0.0              
    elif ProblemType == 'Barotropic_Tide':
        cX1 = omega1/kX1
        cY1 = 0.0
        cX2 = omega2/kX2
        cY2 = 0.0
    else:
        cX1 = omega1/kX1
        cY1 = omega1/kY1
        cX2 = omega2/kX2
        cY2 = omega2/kY2    
    return cX1, cY1, cX2, cY2


class ExactSolutionParameters:
    
    def __init__(myExactSolutionParameters,alpha0,beta0,c0,cX1,cX2,cY1,cY2,etaHat1,etaHat2,f0_MidLatitude,f0,g,H0,
                 kappa1,kappa2,kX1,kX2,kY1,kY2,lX,lY,nu,omega1,omega2,x0,y0,R0,R0x,R0y,s,uL,uR,u0,v0,R,Req,LengthScale,
                 TimeScale,VelocityScale,SurfaceElevationScale):
        myExactSolutionParameters.alpha0 = alpha0
        myExactSolutionParameters.beta0 = beta0
        myExactSolutionParameters.c0 = c0
        myExactSolutionParameters.cX1 = cX1
        myExactSolutionParameters.cX2 = cX2
        myExactSolutionParameters.cY1 = cY1
        myExactSolutionParameters.cY2 = cY2
        myExactSolutionParameters.etaHat1 = etaHat1
        myExactSolutionParameters.etaHat2 = etaHat2
        myExactSolutionParameters.f0_MidLatitude = f0_MidLatitude
        myExactSolutionParameters.f0 = f0
        myExactSolutionParameters.g = g
        myExactSolutionParameters.H0 = H0
        myExactSolutionParameters.kappa1 = kappa1
        myExactSolutionParameters.kappa2 = kappa2
        myExactSolutionParameters.kX1 = kX1
        myExactSolutionParameters.kX2 = kX2
        myExactSolutionParameters.kY1 = kY1
        myExactSolutionParameters.kY2 = kY2
        myExactSolutionParameters.lX = lX
        myExactSolutionParameters.lY = lY
        myExactSolutionParameters.nu = nu
        myExactSolutionParameters.omega1 = omega1
        myExactSolutionParameters.omega2 = omega2
        myExactSolutionParameters.x0 = x0
        myExactSolutionParameters.y0 = y0
        myExactSolutionParameters.R0 = R0
        myExactSolutionParameters.R0x = R0x
        myExactSolutionParameters.R0y = R0y
        myExactSolutionParameters.s = s
        myExactSolutionParameters.uL = uL
        myExactSolutionParameters.uR = uR
        myExactSolutionParameters.u0 = u0
        myExactSolutionParameters.v0 = v0
        myExactSolutionParameters.R = R
        myExactSolutionParameters.Req = Req
        myExactSolutionParameters.LengthScale = LengthScale
        myExactSolutionParameters.TimeScale = TimeScale
        myExactSolutionParameters.VelocityScale = VelocityScale
        myExactSolutionParameters.SurfaceElevationScale = SurfaceElevationScale
        

def SpecifyExactSolutionParameters(ProblemType,ProblemType_GeophysicalWave,ProblemType_ManufacturedRossbyWave,
                                   ProblemType_RossbyWave,ProblemType_EquatorialWave,PrintPhaseSpeedOfWaveModes,
                                   PrintAmplitudesOfWaveModes,ReadDomainExtentsFromMeshFile=False,lX=0.0,lY=0.0):
    beta0 = 2.0*10.0**(-11.0)
    if ProblemType == 'Viscous_Burgers_Equation':
        g = 0.0
        H0 = 1.0
    else:
        g = 10.0
        H0 = 1000.0    
    c0 = np.sqrt(g*H0)
    etaHat1, etaHat2 = SpecifyAmplitudes(ProblemType,ProblemType_ManufacturedRossbyWave,ProblemType_RossbyWave,
                                         ProblemType_EquatorialWave)
    f0_MidLatitude = 10.0**(-4.0)
    if (ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Manufactured_Topographic_Rossby_Wave' 
        or ProblemType == 'Topographic_Rossby_Wave' or ProblemType == 'Diffusion_Equation' 
        or ProblemType == 'Advection_Diffusion_Equation' or ProblemType == 'Viscous_Burgers_Equation' 
        or ProblemType_EquatorialWave):
        f0 = 0.0
    else:
        f0 = f0_MidLatitude
    if ProblemType == 'Manufactured_Topographic_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        alpha0 = beta0*H0/f0_MidLatitude
        # In the Northern Hemisphere where f0 > 0, the topographic Rossby wave travels with the shallower water on its 
        # right. Hence if alpha0 > 0 i.e. the ocean depth increases northward, the topographic Rossby wave will 
        # propagate eastward else it will propagate westward.
    else:
        alpha0 = 0.0
    if not(ReadDomainExtentsFromMeshFile):
        lX, lY = SpecifyDomainExtents(ProblemType,ProblemType_ManufacturedRossbyWave,ProblemType_RossbyWave,
                                      ProblemType_EquatorialWave)
    kX1, kY1, kX2, kY2 = SpecifyWaveNumbers(ProblemType,lX,lY)
    if ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Advection_Diffusion_Equation':
        x0 = 0.25*lX
        y0 = 0.25*lY
    elif ProblemType_RossbyWave:
        x0 = 0.5*lX
        y0 = 0.5*lY
    elif ProblemType == 'Viscous_Burgers_Equation':
        x0 = 0.25*lX # Zonal offset
        y0 = 0.0
    else:
        x0 = 0.0
        y0 = 0.0
    w = 0.2*(0.5*10.0**6.0)
    if ProblemType == 'Plane_Gaussian_Wave':
        R0 = w/(2.0*np.sqrt(np.log(2.0)))
    elif ProblemType == 'Advection_Diffusion_Equation':
        R0 = 0.05*lX
    else:
        R0 = 0.0
    R0x = 10.0**5.0
    R0y = 10.0**5.0
    u0 = 1.0
    v0 = u0*np.sqrt(3.0)/2.0
    uL = 1.0 # Speed at the left boundary
    uR = 0.0 # Speed at the right boundary
    s = 0.5*(uL + uR) # Speed of the shock
    if ProblemType == 'Diffusion_Equation' or ProblemType == 'Viscous_Burgers_Equation':
        nu = 25000.0 # Kinematic viscosity
    elif ProblemType == 'Advection_Diffusion_Equation':
        nu = R0**2.0
    else:
        nu = 0.0
    kappa1 = nu*(kX1**2.0 + kY1**2.0)
    kappa2 = nu*(kX2**2.0 + kY2**2.0)
    if not(ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Manufactured_Topographic_Rossby_Wave' 
           or ProblemType == 'Topographic_Rossby_Wave' or ProblemType == 'Diffusion_Equation' 
           or ProblemType == 'Advection_Diffusion_Equation' or ProblemType == 'Viscous_Burgers_Equation' 
           or ProblemType_EquatorialWave):
        R = c0/f0
    else:
        R = 0.0
    Req = np.sqrt(c0/beta0)
    if ProblemType == 'Viscous_Burgers_Equation':
        LengthScale = 1.0
        TimeScale = 1.0
        VelocityScale = 1.0
        SurfaceElevationScale = 1.0
    else:
        LengthScale = np.sqrt(c0/beta0)
        TimeScale = 1.0/np.sqrt(beta0*c0)
        VelocityScale = c0
        SurfaceElevationScale = c0**2.0/g
    omega1, omega2 = SpecifyAngularFrequencies(ProblemType,ProblemType_RossbyWave,alpha0,beta0,c0,f0_MidLatitude,g,H0,
                                               kX1,kY1,kX2,kY2,R,LengthScale,TimeScale)
    cX1, cY1, cX2, cY2 = SpecifyPhaseSpeeds(ProblemType,ProblemType_RossbyWave,ProblemType_EquatorialWave,c0,kX1,kY1,
                                            kX2,kY2,omega1,omega2,u0,v0,s)
    if (ProblemType_GeophysicalWave or ProblemType == 'Barotropic_Tide') and PrintPhaseSpeedOfWaveModes:
        print('The zonal component of the phase speed of the first wave mode is %.4g.' %cX1)
        print('The meridional component of the phase speed of the first wave mode is %.4g.' %cY1)
        print('The zonal component of the phase speed of the second wave mode is %.4g.' %cX2)
        print('The meridional component of the phase speed of the second wave mode is %.4g.' %cY2)
    myExactSolutionParameters = ExactSolutionParameters(alpha0,beta0,c0,cX1,cX2,cY1,cY2,etaHat1,etaHat2,f0_MidLatitude,
                                                        f0,g,H0,kappa1,kappa2,kX1,kX2,kY1,kY2,lX,lY,nu,omega1,omega2,x0,
                                                        y0,R0,R0x,R0y,s,uL,uR,u0,v0,R,Req,LengthScale,TimeScale,
                                                        VelocityScale,SurfaceElevationScale)
    if (((ProblemType_GeophysicalWave and not(ProblemType_EquatorialWave)) or ProblemType == 'Barotropic_Tide') 
        and PrintAmplitudesOfWaveModes):        
        SurfaceElevationAmplitude, ZonalVelocityAmplitude, MeridionalVelocityAmplitude = (
        ESST.DetermineSolutionAmplitude(ProblemType,myExactSolutionParameters))
        print('The amplitude of the surface elevation of the first wave mode is %.4g.' %SurfaceElevationAmplitude[0])
        print('The amplitude of the surface elevation of the second wave mode is %.4g.' %SurfaceElevationAmplitude[1])
        print('The amplitude of the zonal velocity of the first wave mode is %.4g.' %ZonalVelocityAmplitude[0])
        print('The amplitude of the zonal velocity of the second wave mode is %.4g.' %ZonalVelocityAmplitude[1])   
        print('The amplitude of the meridional velocity of the first wave mode is %.4g.' 
              %MeridionalVelocityAmplitude[0])
        print('The amplitude of the meridional velocity of the second wave mode is %.4g.' 
              %MeridionalVelocityAmplitude[1])
    return myExactSolutionParameters


class TimeSteppingParameters:

    def __init__(myTimeSteppingParameters,TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                 Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type,
                 Forward_Backward_with_RK2_Feedback_parameter_beta,Forward_Backward_with_RK2_Feedback_parameter_epsilon,
                 LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta,LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma,
                 LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon,Generalized_FB_with_AB2_AM3_Step_parameter_beta,
                 Generalized_FB_with_AB2_AM3_Step_parameter_gamma,Generalized_FB_with_AB2_AM3_Step_parameter_epsilon,
                 Generalized_FB_with_AB3_AM4_Step_parameter_beta,Generalized_FB_with_AB3_AM4_Step_parameter_gamma,
                 Generalized_FB_with_AB3_AM4_Step_parameter_epsilon,Generalized_FB_with_AB3_AM4_Step_parameter_delta):
        myTimeSteppingParameters.nStepsRK3 = 3
        myTimeSteppingParameters.aRK3 = np.array([0.0,-5.0/9.0,-153.0/128.0])
        myTimeSteppingParameters.bRK3 = np.array([0.0,1.0/3.0,3.0/4.0])
        myTimeSteppingParameters.bRK3Next = np.array([1.0/3.0,3.0/4.0,1.0])
        myTimeSteppingParameters.gRK3 = np.array([1.0/3.0,15.0/16.0,8.0/15.0])
        myTimeSteppingParameters.nStepsRK4 = 5
        myTimeSteppingParameters.aRK4 = np.zeros(5)
        myTimeSteppingParameters.aRK4[1] = -1.0
        myTimeSteppingParameters.aRK4[2] = -1.0/3.0 + 2.0**(2.0/3.0)/6.0 - 2.0*2.0**(1.0/3.0)/3.0
        myTimeSteppingParameters.aRK4[3] = -2.0**(1.0/3.0) - 2.0**(2.0/3.0) - 2.0
        myTimeSteppingParameters.aRK4[4] = -1.0 + 2.0**(1.0/3.0)
        myTimeSteppingParameters.bRK4 = np.zeros(5)
        myTimeSteppingParameters.bRK4[1] = 2.0/3.0 + 2.0**(1.0/3.0)/3.0 + 2.0**(2.0/3.0)/6.0
        myTimeSteppingParameters.bRK4[2] = 2.0/3.0 + 2.0**(1.0/3.0)/3.0 + 2.0**(2.0/3.0)/6.0
        myTimeSteppingParameters.bRK4[3] = 1.0/3.0 - 2.0**(1.0/3.0)/3.0 - 2.0**(2.0/3.0)/6.0
        myTimeSteppingParameters.bRK4[4] = 1.0
        myTimeSteppingParameters.bRK4Next = np.ones(5)
        myTimeSteppingParameters.bRK4Next[0:4] = myTimeSteppingParameters.bRK4[1:5]
        myTimeSteppingParameters.gRK4 = np.zeros(5)
        myTimeSteppingParameters.gRK4[0] = 2.0/3.0 + 2.0**(1.0/3.0)/3.0 + 2.0**(2.0/3.0)/6.0
        myTimeSteppingParameters.gRK4[1] = -2.0**(2.0/3.0)/6.0 + 1.0/6.0
        myTimeSteppingParameters.gRK4[2] = -1.0/3.0 - 2.0*2.0**(1.0/3.0)/3.0 - 2.0**(2.0/3.0)/3.0
        myTimeSteppingParameters.gRK4[3] = 1.0/3.0 - 2.0**(1.0/3.0)/3.0 - 2.0**(2.0/3.0)/6.0
        myTimeSteppingParameters.gRK4[4] = 1.0/3.0 + 2.0**(1.0/3.0)/6.0 + 2.0**(2.0/3.0)/12.0
        myTimeSteppingParameters.AB2 = np.array([1.5,-0.5])
        myTimeSteppingParameters.AB3 = np.array([23.0/12.0,-4.0/3.0,5.0/12.0])
        myTimeSteppingParameters.AB4 = np.array([55.0/24.0,-59.0/24.0,37.0/24.0,-3.0/8.0])
        myTimeSteppingParameters.TimeIntegrator = TimeIntegrator
        myTimeSteppingParameters.TimeIntegratorShortForm = (
        DetermineTimeIntegratorShortForm(TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                         Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type))
        myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_Type = LF_TR_and_LF_AM3_with_FB_Feedback_Type
        myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_Type = Generalized_FB_with_AB2_AM3_Step_Type
        myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_Type = Generalized_FB_with_AB3_AM4_Step_Type
        myTimeSteppingParameters.Forward_Backward_with_RK2_Feedback_parameter_beta = (
        Forward_Backward_with_RK2_Feedback_parameter_beta)
        myTimeSteppingParameters.Forward_Backward_with_RK2_Feedback_parameter_epsilon = (
        Forward_Backward_with_RK2_Feedback_parameter_epsilon)
        myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta = (
        LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta)
        myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma = (
        LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma)
        myTimeSteppingParameters.LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon = (
        LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon)
        myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_parameter_beta = (
        Generalized_FB_with_AB2_AM3_Step_parameter_beta)
        myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_parameter_gamma = (
        Generalized_FB_with_AB2_AM3_Step_parameter_gamma)
        myTimeSteppingParameters.Generalized_FB_with_AB2_AM3_Step_parameter_epsilon = (
        Generalized_FB_with_AB2_AM3_Step_parameter_epsilon)
        myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_parameter_beta = (
        Generalized_FB_with_AB3_AM4_Step_parameter_beta)
        myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_parameter_gamma = (
        Generalized_FB_with_AB3_AM4_Step_parameter_gamma)
        myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_parameter_epsilon = (
        Generalized_FB_with_AB3_AM4_Step_parameter_epsilon)
        myTimeSteppingParameters.Generalized_FB_with_AB3_AM4_Step_parameter_delta = (
        Generalized_FB_with_AB3_AM4_Step_parameter_delta)


def DetermineTimeIntegratorShortForm(TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                     Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type):
    if TimeIntegrator == 'ForwardEulerMethod':
        TimeIntegratorShortForm = 'FE'
    elif TimeIntegrator == 'ForwardBackwardMethod':
        TimeIntegratorShortForm = 'FB'
    elif TimeIntegrator == 'ExplicitMidpointMethod':
        TimeIntegratorShortForm = 'EMM'
    elif TimeIntegrator == 'WilliamsonLowStorageThirdOrderRungeKuttaMethod':
        TimeIntegratorShortForm = 'WLSRK3'
    elif TimeIntegrator == 'CarpenterKennedyLowStorageFourthOrderRungeKuttaMethod':
        TimeIntegratorShortForm = 'CKLSRK4'    
    elif TimeIntegrator == 'SecondOrderAdamsBashforthMethod':
        TimeIntegratorShortForm = 'AB2'
    elif TimeIntegrator == 'ThirdOrderAdamsBashforthMethod':
        TimeIntegratorShortForm = 'AB3'
    elif TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
        TimeIntegratorShortForm = 'AB4'
    elif TimeIntegrator == 'LeapfrogTrapezoidalMethod':
        TimeIntegratorShortForm = 'LFTR_Odr2'
    elif TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback':
        if LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'SecondOrderAccurate_LF_TR':
            TimeIntegratorShortForm = 'LFTR_Odr2'
        elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'ThirdOrderAccurate_LF_AM3':
            TimeIntegratorShortForm = 'LFAM_Odr3'        
        elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'ThirdOrderAccurate_MaximumStabilityRange':
            TimeIntegratorShortForm = 'LFAM_Odr3_MaxStabRng'
        elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'FourthOrderAccurate_MinimumTruncationError':
            TimeIntegratorShortForm = 'LFAM_Odr4_MinTruncErr'
        elif LF_TR_and_LF_AM3_with_FB_Feedback_Type == 'FourthOrderAccurate_MaximumStabilityRange':
            TimeIntegratorShortForm = 'LFAM_Odr4_MaxStabRng'
    elif TimeIntegrator == 'ForwardBackwardMethodWithRK2Feedback':
        TimeIntegratorShortForm = 'FB_RK2Fdbk'
    elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step':
        if Generalized_FB_with_AB2_AM3_Step_Type == 'ThirdOrderAccurate_WideStabilityRange':
            TimeIntegratorShortForm = 'GenFB_AB2AM3_Ordr3_WideStabRng'
        elif Generalized_FB_with_AB2_AM3_Step_Type == 'ThirdOrderAccurate_WeakAsymptoticInstabilityOfPhysicalModes':
            TimeIntegratorShortForm = 'GenFB_AB2AM3_Ordr3_WeakAsympInstab'
        elif Generalized_FB_with_AB2_AM3_Step_Type == 'FourthOrderAccurate':
            TimeIntegratorShortForm = 'GenFB_AB2AM3_Ordr4' 
    elif TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step':
        if Generalized_FB_with_AB3_AM4_Step_Type == 'SecondOrderAccurate_OptimumChoice_ROMS':
            TimeIntegratorShortForm = 'GenFB_AB3AM4_Ordr2_Optm_ROMS'
        elif Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_AB3_AM4':
            TimeIntegratorShortForm = 'GenFB_AB3AM4_Ordr3'
        elif Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_MaximumStabilityRange':
            TimeIntegratorShortForm = 'GenFB_AB3AM4_Ordr3_MaxStabRng'
        elif Generalized_FB_with_AB3_AM4_Step_Type == 'ThirdOrderAccurate_OptimumChoice':
            TimeIntegratorShortForm = 'GenFB_AB3AM4_Ordr3_Optm'
        elif Generalized_FB_with_AB3_AM4_Step_Type == 'FourthOrderAccurate_MaximumStabilityRange':
            TimeIntegratorShortForm = 'GenFB_AB3AM4_Ordr4_MaxStabRng'
    return TimeIntegratorShortForm


def SpecifyTimeSteppingParameters(TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                  Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type):
    Forward_Backward_with_RK2_Feedback_parameter_beta = 0.0
    Forward_Backward_with_RK2_Feedback_parameter_epsilon = 0.0
    LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta = 0.0
    LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma = 0.0
    LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon = 0.0
    Generalized_FB_with_AB2_AM3_Step_parameter_beta = 0.0
    Generalized_FB_with_AB2_AM3_Step_parameter_gamma = 0.0
    Generalized_FB_with_AB2_AM3_Step_parameter_epsilon = 0.0
    Generalized_FB_with_AB3_AM4_Step_parameter_beta = 0.0
    Generalized_FB_with_AB3_AM4_Step_parameter_gamma = 0.0
    Generalized_FB_with_AB3_AM4_Step_parameter_epsilon = 0.0    
    Generalized_FB_with_AB3_AM4_Step_parameter_delta = 0.0
    if TimeIntegrator == 'Forward_Backward_with_RK2_Feedback':
        Forward_Backward_with_RK2_Feedback_parameter_beta = 1.0/3.0
        Forward_Backward_with_RK2_Feedback_parameter_epsilon = 2.0/3.0
    elif TimeIntegrator == 'LF_TR_and_LF_AM3_with_FB_Feedback':
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
        LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta = beta
        LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma = gamma
        LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon = epsilon
    elif TimeIntegrator == 'Generalized_FB_with_AB2_AM3_Step':
        if Generalized_FB_with_AB2_AM3_Step_Type == 'ThirdOrderAccurate_WideStabilityRange':
            beta = 0.0
        elif Generalized_FB_with_AB2_AM3_Step_Type == 'ThirdOrderAccurate_WeakAsymptoticInstabilityOfPhysicalModes':
            beta = 0.5    
        elif Generalized_FB_with_AB2_AM3_Step_Type == 'FourthOrderAccurate':
            useSymPyToDetermineBeta = False
            # Note that if useSymPyToDetermineBeta is specified as True, every time beta is used, the SymPy polynomial
            # equation solver will be executed, resulting in an immense slowdown of the code.
            if useSymPyToDetermineBeta:
                symbolic_beta = sp.Symbol('beta')
                beta_roots = sp.solve(-symbolic_beta**3.0 - symbolic_beta/12.0 + 1.0/12.0, symbolic_beta)
                beta = beta_roots[0]   
            else:
                beta = 0.373707625197906
        gamma = beta - 2.0*beta**2.0 - 1.0/6.0
        epsilon = beta**2.0 + 1.0/12.0
        Generalized_FB_with_AB2_AM3_Step_parameter_beta = beta
        Generalized_FB_with_AB2_AM3_Step_parameter_gamma = gamma
        Generalized_FB_with_AB2_AM3_Step_parameter_epsilon = epsilon
    elif TimeIntegrator == 'Generalized_FB_with_AB3_AM4_Step':
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
        Generalized_FB_with_AB3_AM4_Step_parameter_beta = beta
        Generalized_FB_with_AB3_AM4_Step_parameter_gamma = gamma
        Generalized_FB_with_AB3_AM4_Step_parameter_epsilon = epsilon   
        Generalized_FB_with_AB3_AM4_Step_parameter_delta = delta
    myTimeSteppingParameters = TimeSteppingParameters(
    TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
    Generalized_FB_with_AB3_AM4_Step_Type,Forward_Backward_with_RK2_Feedback_parameter_beta,
    Forward_Backward_with_RK2_Feedback_parameter_epsilon,LF_TR_and_LF_AM3_with_FB_Feedback_parameter_beta,
    LF_TR_and_LF_AM3_with_FB_Feedback_parameter_gamma,LF_TR_and_LF_AM3_with_FB_Feedback_parameter_epsilon,
    Generalized_FB_with_AB2_AM3_Step_parameter_beta,Generalized_FB_with_AB2_AM3_Step_parameter_gamma,
    Generalized_FB_with_AB2_AM3_Step_parameter_epsilon,Generalized_FB_with_AB3_AM4_Step_parameter_beta,
    Generalized_FB_with_AB3_AM4_Step_parameter_gamma,Generalized_FB_with_AB3_AM4_Step_parameter_epsilon,
    Generalized_FB_with_AB3_AM4_Step_parameter_delta)
    return myTimeSteppingParameters 


def SpecifyTimeStep(ProblemType,ProblemType_ManufacturedRossbyWave,ProblemType_RossbyWave):
    if ProblemType == 'Plane_Gaussian_Wave':
        dt = 16.0
    elif ProblemType == 'Coastal_Kelvin_Wave':
        dt = 216.0
    elif ProblemType == 'Inertia_Gravity_Wave':
        dt = 111.0
    elif ProblemType_ManufacturedRossbyWave:
        dt = 230100.0
    elif ProblemType_RossbyWave:
        dt = 0.5
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        dt = 873.0
    elif ProblemType == 'Equatorial_Yanai_Wave':
        dt = 474.0
    elif ProblemType == 'Equatorial_Rossby_Wave':
        dt = 1200.0
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        dt = 285.0
    elif ProblemType == 'Barotropic_Tide':
        dt = 12.3
    elif ProblemType == 'Diffusion_Equation' or ProblemType == 'Viscous_Burgers_Equation':
        dt = 750.0
    elif ProblemType == 'Advection_Diffusion_Equation':
        dt = 5.0*10.0**(-3.0)
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        dt = 162.0
    return dt


def SpecifyDumpFrequency(ProblemType,ProblemType_ManufacturedRossbyWave,ProblemType_RossbyWave,
                         ProblemType_EquatorialWave):
    if (ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Inertia_Gravity_Wave' 
        or ProblemType_ManufacturedRossbyWave or ProblemType == 'Advection_Diffusion_Equation'):
        nDumpFrequency = 4
    elif ((ProblemType == 'Coastal_Kelvin_Wave' 
           or (ProblemType_EquatorialWave and not(ProblemType == 'Equatorial_Rossby_Wave')) 
           or ProblemType == 'Barotropic_Tide')):
        nDumpFrequency = 2
    elif ProblemType_RossbyWave:
        nDumpFrequency = 1440
    elif ProblemType == 'Equatorial_Rossby_Wave':
        nDumpFrequency = 5
    elif ProblemType == 'Diffusion_Equation':
        nDumpFrequency = 8
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        nDumpFrequency = 3
    elif ProblemType == 'Viscous_Burgers_Equation':
        nDumpFrequency = 10
    # Note that nDumpFrequency is chosen in such a way that we end up with approximately 100 output files for the entire
    # simulation time.
    return nDumpFrequency


def SpecifyNumberOfTimeSteps(ProblemType,ProblemType_ManufacturedRossbyWave,ProblemType_RossbyWave,
                             ProblemType_EquatorialWave):
    if ProblemType == 'Plane_Gaussian_Wave':
        nTime_Minimum = 442 + 1
        nTime = 444 + 1
    elif ProblemType == 'Coastal_Kelvin_Wave' or ProblemType_EquatorialWave:
        if ProblemType == 'Equatorial_Inertia_Gravity_Wave':
            nTime_Minimum = 202 + 1
        elif ProblemType == 'Equatorial_Rossby_Wave':
            nTime_Minimum = 525 + 1
        else:
            nTime_Minimum = 201 + 1
        if ProblemType == 'Equatorial_Rossby_Wave':
            nTime = 525 + 1
        else:
            nTime = 202 + 1
    elif ProblemType == 'Inertia_Gravity_Wave':
        nTime_Minimum = 409 + 1
        nTime = 412 + 1
    elif ProblemType_ManufacturedRossbyWave:   
        nTime_Minimum = 401 + 1
        nTime = 404 + 1
    elif ProblemType_RossbyWave:
        nTime_Minimum = 86400*2 + 1
        nTime = 86400*2 + 1
    elif ProblemType == 'Barotropic_Tide':
        nTime_Minimum = 204 + 1
        nTime = 204 + 1
    elif ProblemType == 'Diffusion_Equation':
        nTime_Minimum = 803 + 1
        nTime = 808 + 1
    elif ProblemType == 'Advection_Diffusion_Equation':
        nTime_Minimum = 200 + 1
        nTime = 200 + 1
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        nTime_Minimum = 364 + 1
        nTime = 366 + 1
    elif ProblemType == 'Viscous_Burgers_Equation':
        nTime_Minimum = 1334 + 1
        nTime = 1340 + 1 
    # Note that (a) nTime_Minimum is the minimum integer such that (nTime_Minimum - 1) times the time step is greater 
    # than or equal to the simulation time, and (b) nTime is the minimum integer such that nTime >= nTime_Minimum and 
    # nTime - 1 is a multiple of nDumpFrequency.
    return nTime_Minimum, nTime


def SpecifyLogicalArrayPlot(ProblemType):
    if ProblemType == 'Coastal_Kelvin_Wave' or ProblemType == 'Advection_Diffusion_Equation':
        PlotZonalVelocity = False
    else:
        PlotZonalVelocity = True
    if (ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Diffusion_Equation' 
        or ProblemType == 'Advection_Diffusion_Equation' or ProblemType == 'Viscous_Burgers_Equation'):
        PlotMeridionalVelocity = False
    else:
        PlotMeridionalVelocity = True
    if ProblemType == 'Diffusion_Equation' or ProblemType == 'Viscous_Burgers_Equation':
        PlotSurfaceElevation = False
    else:
        PlotSurfaceElevation = True
    LogicalArrayPlot = [PlotZonalVelocity,PlotMeridionalVelocity,PlotSurfaceElevation]
    return LogicalArrayPlot


def SpecifyTitleAndFileNamePrefixes(ProblemType):
    if ProblemType == 'Plane_Gaussian_Wave':
        ProblemType_Title = 'Plane Gaussian Wave'
        ProblemType_FileName = 'PlaneGaussianWave'
    elif ProblemType == 'Coastal_Kelvin_Wave':
        ProblemType_Title = 'Coastal Kelvin Wave'
        ProblemType_FileName = 'CoastalKelvinWave'
    elif ProblemType == 'Inertia_Gravity_Wave':
        ProblemType_Title = 'Inertia Gravity Wave'
        ProblemType_FileName = 'InertiaGravityWave'
    elif ProblemType == 'Manufactured_Planetary_Rossby_Wave' or ProblemType == 'Planetary_Rossby_Wave':
        ProblemType_Title = 'Planetary Rossby Wave'
        ProblemType_FileName = 'PlanetaryRossbyWave'
    elif ProblemType == 'Manufactured_Topographic_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        ProblemType_Title = 'Topographic Rossby Wave'
        ProblemType_FileName = 'TopographicRossbyWave'
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        ProblemType_Title = 'Equatorial Kelvin Wave'
        ProblemType_FileName = 'EquatorialKelvinWave'
    elif ProblemType == 'Equatorial_Yanai_Wave':
        ProblemType_Title = 'Equatorial Yanai Wave'
        ProblemType_FileName = 'EquatorialYanaiWave'        
    elif ProblemType == 'Equatorial_Rossby_Wave':
        ProblemType_Title = 'Equatorial Rossby Wave'
        ProblemType_FileName = 'EquatorialRossbyWave'           
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        ProblemType_Title = 'Equatorial Inertia Gravity Wave'
        ProblemType_FileName = 'EquatorialInertiaGravityWave'
    elif ProblemType == 'Barotropic_Tide':
        ProblemType_Title = 'Barotropic Tide'
        ProblemType_FileName = 'BarotropicTide' 
    elif ProblemType == 'Diffusion_Equation':
        ProblemType_Title = 'Diffusion Equation'
        ProblemType_FileName = 'DiffusionEquation' 
    elif ProblemType == 'Advection_Diffusion_Equation':
        ProblemType_Title = 'Advection Diffusion Equation'
        ProblemType_FileName = 'AdvectionDiffusionEquation'
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ProblemType_Title = 'Non-Linear Manufactured Solution'
        ProblemType_FileName = 'NonLinearManufacturedSolution'
    elif ProblemType == 'Viscous_Burgers_Equation':
        ProblemType_Title = 'Viscous Burgers Equation'
        ProblemType_FileName = 'ViscousBurgersEquation'
    return ProblemType_Title, ProblemType_FileName


class NameList:
    
    def __init__(myNameList,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                 LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                 Generalized_FB_with_AB3_AM4_Step_Type,nCellsX,nCellsY,CourantNumber_Advection,CourantNumber_Diffusion,
                 UseCourantNumberToDetermineTimeStep):
        myNameList.ProblemType = ProblemType
        myNameList.ProblemType_GeophysicalWave = isGeophysicalWave(ProblemType)
        myNameList.ProblemType_EquatorialWave = isEquatorialWave(ProblemType)
        myNameList.ProblemType_ManufacturedRossbyWave = Specify_ProblemType_ManufacturedRossbyWave(ProblemType)
        myNameList.ProblemType_RossbyWave = Specify_ProblemType_RossbyWave(ProblemType)
        myNameList.ProblemType_SpecifyExactSolutionLimits = SpecifyExactSolutionLimits(ProblemType)
        if ProblemType == 'NonLinear_Manufactured_Solution' or ProblemType == 'Viscous_Burgers_Equation':
            myNameList.Problem_is_Linear = False
        else:
            myNameList.Problem_is_Linear = True
        SimulateNonLinearRossbyWave = False
        if SimulateNonLinearRossbyWave:
            if myNameList.ProblemType_RossbyWave:
                myNameList.Problem_is_Linear = False
        if myNameList.ProblemType_ManufacturedRossbyWave or ProblemType == 'NonLinear_Manufactured_Solution':
            myNameList.NonTrivialSourceTerms = True
        else:
            myNameList.NonTrivialSourceTerms = False
        if (ProblemType == 'Diffusion_Equation' or ProblemType == 'Advection_Diffusion_Equation' 
            or ProblemType == 'Viscous_Burgers_Equation'):
            myNameList.NonTrivialDiffusionTerms = True
        else:
            myNameList.NonTrivialDiffusionTerms = False
        myNameList.BoundaryCondition = (
        SpecifyBoundaryCondition(ProblemType,myNameList.ProblemType_ManufacturedRossbyWave,
                                 myNameList.ProblemType_RossbyWave,myNameList.ProblemType_EquatorialWave))
        myNameList.lX, myNameList.lY = SpecifyDomainExtents(ProblemType,myNameList.ProblemType_ManufacturedRossbyWave,
                                                            myNameList.ProblemType_RossbyWave,
                                                            myNameList.ProblemType_EquatorialWave)
        myNameList.nCellsX = nCellsX
        myNameList.nCellsY = nCellsY
        myNameList.dx = myNameList.lX/float(nCellsX)
        myNameList.dy = myNameList.lY/float(nCellsY)
        myNameList.myExactSolutionParameters = (
        SpecifyExactSolutionParameters(ProblemType,myNameList.ProblemType_GeophysicalWave,
                                       myNameList.ProblemType_ManufacturedRossbyWave,myNameList.ProblemType_RossbyWave,
                                       myNameList.ProblemType_EquatorialWave,PrintPhaseSpeedOfWaveModes,
                                       PrintAmplitudesOfWaveModes))
        (myNameList.ExactSurfaceElevationLimits, myNameList.ExactZonalVelocityLimits, 
         myNameList.ExactMeridionalVelocityLimits) = (
        ESST.DetermineExactSolutionLimits(ProblemType,myNameList.myExactSolutionParameters))
        myNameList.myTimeSteppingParameters = (
        SpecifyTimeSteppingParameters(TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                      Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type))
        if UseCourantNumberToDetermineTimeStep and not(myNameList.ProblemType_RossbyWave):
            cX1 = myNameList.myExactSolutionParameters.cX1
            cX2 = myNameList.myExactSolutionParameters.cX2
            cY1 = myNameList.myExactSolutionParameters.cY1
            cY2 = myNameList.myExactSolutionParameters.cY2
            abs_cX = max(abs(cX1),abs(cX2))
            abs_cY = max(abs(cY1),abs(cY2))
            if myNameList.NonTrivialDiffusionTerms:
                nu = myNameList.myExactSolutionParameters.nu
                dt_Diffusion = CourantNumber_Diffusion/(nu*(1.0/(myNameList.dx)**2.0 + 1.0/(myNameList.dy)**2.0))
            if ProblemType == 'Diffusion_Equation':
                myNameList.dt = dt_Diffusion
                print('The time step for diffusive Courant number %.6f is %.3g seconds.' 
                      %(CourantNumber_Diffusion,myNameList.dt))
            else:
                dt_Advection = CourantNumber_Advection/(abs_cX/(myNameList.dx) + abs_cY/(myNameList.dy))
                UseMinimumOfAdvectiveAndDiffusiveTimeSteps = False
                if myNameList.NonTrivialDiffusionTerms and UseMinimumOfAdvectiveAndDiffusiveTimeSteps:
                    myNameList.dt = min(dt_Advection,dt_Diffusion)
                    print(
                    'The time step for advective Courant number %.6f and diffusive Courant number %.6f is %.3g seconds.' 
                    %(CourantNumber_Advection,CourantNumber_Diffusion,myNameList.dt))
                else:
                    myNameList.dt = dt_Advection
                    print('The time step for advective Courant number %.6f is %.3g seconds.' 
                          %(CourantNumber_Advection,myNameList.dt))
        else:
            myNameList.dt = SpecifyTimeStep(ProblemType,myNameList.ProblemType_ManufacturedRossbyWave,
                                            myNameList.ProblemType_RossbyWave)
        myNameList.nDumpFrequency = SpecifyDumpFrequency(ProblemType,myNameList.ProblemType_ManufacturedRossbyWave,
                                                         myNameList.ProblemType_RossbyWave,
                                                         myNameList.ProblemType_EquatorialWave)
        if myNameList.ProblemType_RossbyWave:
            nRestartFrequencyBynDumpFrequency = 10
        else:
            nRestartFrequencyBynDumpFrequency = 50
        myNameList.nRestartFrequency = nRestartFrequencyBynDumpFrequency*myNameList.nDumpFrequency
        # Specify myNameList.nRestartFrequency to be an integral multiple of myNameList.nDumpFrequency.
        myNameList.nTime_Minimum, myNameList.nTime = (
        SpecifyNumberOfTimeSteps(ProblemType,myNameList.ProblemType_ManufacturedRossbyWave,
                                 myNameList.ProblemType_RossbyWave,myNameList.ProblemType_EquatorialWave))
        myNameList.LogicalArrayPlot = SpecifyLogicalArrayPlot(ProblemType)
        myNameList.ProblemType_Title, myNameList.ProblemType_FileName = SpecifyTitleAndFileNamePrefixes(ProblemType)
        
    def ModifyNameList(myNameList,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,CourantNumber_Advection,
                       CourantNumber_Diffusion,UseCourantNumberToDetermineTimeStep,BoundaryCondition,lX,lY,dx,dy):
        myNameList.BoundaryCondition = BoundaryCondition
        myNameList.lX = lX
        myNameList.lY = lY
        myNameList.dx = dx
        myNameList.dy = dy
        myNameList.myExactSolutionParameters = (
        SpecifyExactSolutionParameters(myNameList.ProblemType,myNameList.ProblemType_GeophysicalWave,
                                       myNameList.ProblemType_ManufacturedRossbyWave,myNameList.ProblemType_RossbyWave,
                                       myNameList.ProblemType_EquatorialWave,PrintPhaseSpeedOfWaveModes,
                                       PrintAmplitudesOfWaveModes,ReadDomainExtentsFromMeshFile=True,lX=lX,lY=lY))
        (myNameList.ExactSurfaceElevationLimits, myNameList.ExactZonalVelocityLimits, 
         myNameList.ExactMeridionalVelocityLimits) = (
        ESST.DetermineExactSolutionLimits(myNameList.ProblemType,myNameList.myExactSolutionParameters))
        if UseCourantNumberToDetermineTimeStep and not(myNameList.ProblemType_RossbyWave):
            cX1 = myNameList.myExactSolutionParameters.cX1
            cX2 = myNameList.myExactSolutionParameters.cX2
            cY1 = myNameList.myExactSolutionParameters.cY1
            cY2 = myNameList.myExactSolutionParameters.cY2
            abs_cX = max(abs(cX1),abs(cX2))
            abs_cY = max(abs(cY1),abs(cY2))
            if myNameList.NonTrivialDiffusionTerms:
                nu = myNameList.myExactSolutionParameters.nu
                dt_Diffusion = CourantNumber_Diffusion/(nu*(1.0/(myNameList.dx)**2.0 + 1.0/(myNameList.dy)**2.0))
            if myNameList.ProblemType == 'Diffusion_Equation':
                myNameList.dt = dt_Diffusion
                print('The time step for diffusive Courant number %.6f is %.3g seconds.' 
                      %(CourantNumber_Diffusion,myNameList.dt))
            else:
                dt_Advection = CourantNumber_Advection/(abs_cX/(myNameList.dx) + abs_cY/(myNameList.dy))
                UseMinimumOfAdvectiveAndDiffusiveTimeSteps = False
                if myNameList.NonTrivialDiffusionTerms and UseMinimumOfAdvectiveAndDiffusiveTimeSteps:
                    myNameList.dt = min(dt_Advection,dt_Diffusion)
                    print(
                    'The time step for advective Courant number %.6f and diffusive Courant number %.6f is %.3g seconds.' 
                    %(CourantNumber_Advection,CourantNumber_Diffusion,myNameList.dt))
                else:
                    myNameList.dt = dt_Advection
                    print('The time step for advective Courant number %.6f is %.3g seconds.' 
                          %(CourantNumber_Advection,myNameList.dt))
        else:
            myNameList.dt = SpecifyTimeStep(myNameList.ProblemType,myNameList.ProblemType_ManufacturedRossbyWave,
                                            myNameList.ProblemType_RossbyWave)