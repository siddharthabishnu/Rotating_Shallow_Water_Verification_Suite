"""
Name: Initialization.py
Author: Sid Bishnu
Details: This script specifies the parameters required to initialize the two-dimensional discontinuous Galerkin 
spectral element class.
"""


import numpy as np
import sympy as sp
from IPython.utils import io
with io.capture_output() as captured:
    import ExactSolutionsAndSourceTerms as ESST
    
    
def SpecifyAmplitudes(ProblemType,ProblemType_NoExactSolution,ProblemType_EquatorialWave):
    if ProblemType == 'Plane_Gaussian_Wave':
        etaHat1 = 0.0
    elif ProblemType == 'Coastal_Kelvin_Wave' or ProblemType_EquatorialWave:
        etaHat1 = 0.0001
    elif ProblemType == 'Inertia_Gravity_Wave':           
        etaHat1 = 0.1
    elif (ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave' 
          or ProblemType_NoExactSolution):
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


def SpecifyDomainExtents(ProblemType,ProblemType_NoExactSolution,ProblemType_EquatorialWave):
    if ProblemType == 'Plane_Gaussian_Wave':
        lX = 2.0
    elif ProblemType == 'Coastal_Kelvin_Wave':
        lX = 5000.0*1000.0
    elif ProblemType == 'Inertia_Gravity_Wave' or ProblemType == 'NonLinear_Manufactured_Solution':
        lX = 10000.0*1000.0
    elif ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':    
        lX = 50.0*1000.0
    elif ProblemType_NoExactSolution:
        lX = 1000.0*1000.0
    elif ProblemType_EquatorialWave:
        lX = 17500.0*1000.0
    elif ProblemType == 'Barotropic_Tide':
        lX = 250.0*1000.0
    lY = lX
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
    else:
        kX1 = 2.0*np.pi/lX
        kY1 = 2.0*np.pi/lY
        kX2 = 2.0*kX1
        kY2 = 2.0*kY1
    return kX1, kY1, kX2, kY2


def SpecifyAngularFrequencies(ProblemType,ProblemType_NoExactSolution,alpha0,beta0,c0,f0,g,H0,kX1,kY1,kX2,kY2,R,
                              LengthScale,TimeScale):
    if ProblemType == 'Plane_Gaussian_Wave':
        omega1 = c0
        omega2 = 0.0
    elif ProblemType == 'Coastal_Kelvin_Wave':
        omega1 = -c0*kY1
        omega2 = -c0*kY2
    elif ProblemType == 'Inertia_Gravity_Wave':
        omega1 = np.sqrt(g*H0*(kX1**2.0 + kY1**2.0) + f0**2.0)
        omega2 = np.sqrt(g*H0*(kX2**2.0 + kY2**2.0) + f0**2.0)
    elif ProblemType == 'Planetary_Rossby_Wave':
        omega1 = -beta0*R**2.0*kX1/(1.0 + R**2.0*(kX1**2.0 + kY1**2.0))
        omega2 = -beta0*R**2.0*kX2/(1.0 + R**2.0*(kX2**2.0 + kY2**2.0))
    elif ProblemType == 'Topographic_Rossby_Wave':
        omega1 = alpha0*g/f0*kX1/(1.0 + R**2.0*(kX1**2.0 + kY1**2.0))
        omega2 = alpha0*g/f0*kX2/(1.0 + R**2.0*(kX2**2.0 + kY2**2.0))
    elif ProblemType_NoExactSolution:
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
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        omega1 = np.sqrt(g*H0*(kX1**2.0 + kY1**2.0))
        omega2 = 0.0
    return omega1, omega2


def SpecifyPhaseSpeeds(ProblemType,ProblemType_NoExactSolution,ProblemType_EquatorialWave,c0,kX1,kY1,kX2,kY2,omega1,
                       omega2):
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
    elif ProblemType_NoExactSolution:
        cX1 = 0.0
        cY1 = 0.0
        cX2 = 0.0
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
    
    def __init__(myExactSolutionParameters,alpha0,beta0,c0,cX1,cX2,cY1,cY2,etaHat1,etaHat2,f0,g,H0,kX1,kX2,kY1,kY2,lX,
                 lY,omega1,omega2,x0,y0,R0,R0x,R0y,R,Req,LengthScale,TimeScale,VelocityScale,SurfaceElevationScale):
        myExactSolutionParameters.alpha0 = alpha0
        myExactSolutionParameters.beta0 = beta0
        myExactSolutionParameters.c0 = c0
        myExactSolutionParameters.cX1 = cX1
        myExactSolutionParameters.cX2 = cX2
        myExactSolutionParameters.cY1 = cY1
        myExactSolutionParameters.cY2 = cY2
        myExactSolutionParameters.etaHat1 = etaHat1
        myExactSolutionParameters.etaHat2 = etaHat2
        myExactSolutionParameters.f0 = f0
        myExactSolutionParameters.g = g
        myExactSolutionParameters.H0 = H0
        myExactSolutionParameters.kX1 = kX1
        myExactSolutionParameters.kX2 = kX2
        myExactSolutionParameters.kY1 = kY1
        myExactSolutionParameters.kY2 = kY2
        myExactSolutionParameters.lX = lX
        myExactSolutionParameters.lY = lY
        myExactSolutionParameters.omega1 = omega1
        myExactSolutionParameters.omega2 = omega2
        myExactSolutionParameters.x0 = x0
        myExactSolutionParameters.y0 = y0
        myExactSolutionParameters.R0 = R0
        myExactSolutionParameters.R0x = R0x
        myExactSolutionParameters.R0y = R0y
        myExactSolutionParameters.R = R
        myExactSolutionParameters.Req = Req
        myExactSolutionParameters.LengthScale = LengthScale
        myExactSolutionParameters.TimeScale = TimeScale
        myExactSolutionParameters.VelocityScale = VelocityScale
        myExactSolutionParameters.SurfaceElevationScale = SurfaceElevationScale
        

def SpecifyExactSolutionParameters(ProblemType,ProblemType_GeophysicalWave,ProblemType_NoExactSolution,
                                   ProblemType_EquatorialWave,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                   DomainExtentsSpecified=False,lX=0.0,lY=0.0):
    beta0 = 2.0*10.0**(-11.0)
    if ProblemType == 'Plane_Gaussian_Wave':
        g = 1.0
        H0 = 1.0
    else:
        g = 10.0
        H0 = 1000.0    
    c0 = np.sqrt(g*H0)
    etaHat1, etaHat2 = SpecifyAmplitudes(ProblemType,ProblemType_NoExactSolution,ProblemType_EquatorialWave)
    if ProblemType == 'Plane_Gaussian_Wave' or ProblemType_EquatorialWave:
        f0 = 0.0
    else:
        f0 = 10.0**(-4.0)
    if (ProblemType == 'Topographic_Rossby_Wave'
        or ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave'):
        alpha0 = beta0*H0/f0
        # In the Northern Hemisphere where f0 > 0, the topographic Rossby wave travels with the shallower water on its 
        # right. Hence if alpha0 > 0 i.e. the ocean depth increases northward, the topographic Rossby wave will 
        # propagate eastward else it will propagate westward.
    else:
        alpha0 = 0.0
    if not(DomainExtentsSpecified):
        lX, lY = SpecifyDomainExtents(ProblemType,ProblemType_NoExactSolution,ProblemType_EquatorialWave)
    kX1, kY1, kX2, kY2 = SpecifyWaveNumbers(ProblemType,lX,lY)
    if ProblemType == 'Plane_Gaussian_Wave':
        x0 = -0.5*(0.5*lX)
        y0 = -0.5*(0.5*lY)
    elif ProblemType_NoExactSolution:
        x0 = 0.5*lX
        y0 = 0.5*lY
    else:
        x0 = 0.0
        y0 = 0.0
    R0 = 0.2/(2.0*np.sqrt(np.log(2.0)))
    R0x = 10.0**5.0
    R0y = 10.0**5.0
    if not(ProblemType == 'Plane_Gaussian_Wave' or ProblemType_EquatorialWave):
        R = c0/f0
    else:
        R = 0.0
    Req = np.sqrt(c0/beta0)
    LengthScale = np.sqrt(c0/beta0)
    TimeScale = 1.0/np.sqrt(beta0*c0)
    VelocityScale = c0
    SurfaceElevationScale = c0**2.0/g
    omega1, omega2 = SpecifyAngularFrequencies(ProblemType,ProblemType_NoExactSolution,alpha0,beta0,c0,f0,g,H0,kX1,kY1,
                                               kX2,kY2,R,LengthScale,TimeScale)
    cX1, cY1, cX2, cY2 = SpecifyPhaseSpeeds(ProblemType,ProblemType_NoExactSolution,ProblemType_EquatorialWave,c0,kX1,
                                            kY1,kX2,kY2,omega1,omega2)
    if (ProblemType_GeophysicalWave or ProblemType == 'Barotropic_Tide') and PrintPhaseSpeedOfWaveModes:
        print('The zonal component of the phase speed of the first wave mode is %.4g.' %cX1)
        print('The meridional component of the phase speed of the first wave mode is %.4g.' %cY1)
        print('The zonal component of the phase speed of the second wave mode is %.4g.' %cX2)
        print('The meridional component of the phase speed of the second wave mode is %.4g.' %cY2)
    myExactSolutionParameters = ExactSolutionParameters(alpha0,beta0,c0,cX1,cX2,cY1,cY2,etaHat1,etaHat2,f0,g,H0,kX1,kX2,
                                                        kY1,kY2,lX,lY,omega1,omega2,x0,y0,R0,R0x,R0y,R,Req,LengthScale,
                                                        TimeScale,VelocityScale,SurfaceElevationScale)
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


def SpecifyTimeStep(ProblemType,ProblemType_NoExactSolution):
    if ProblemType == 'Plane_Gaussian_Wave':
        dt = 7.0*10.0**(-4.0)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        dt = 50.0
    elif ProblemType == 'Inertia_Gravity_Wave':
        dt = 23.0
    elif ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        dt = 39000.0
    elif ProblemType_NoExactSolution:
        dt = 0.5
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        dt = 330.0
    elif ProblemType == 'Equatorial_Yanai_Wave':
        dt = 180.0
    elif ProblemType == 'Equatorial_Rossby_Wave':
        dt = 240.0
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        dt = 108.0
    elif ProblemType == 'Barotropic_Tide':
        dt = 2.4
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        dt = 66.0
    return dt


def SpecifyDumpFrequency(ProblemType,ProblemType_NoExactSolution,ReadFromSELFOutputData):
    if ProblemType == 'Coastal_Kelvin_Wave' or ProblemType == 'Barotropic_Tide':
        nDumpFrequency = 10
    elif (ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Equatorial_Yanai_Wave' 
          or ProblemType == 'Equatorial_Inertia_Gravity_Wave'):
        nDumpFrequency = 5
    elif ProblemType == 'Equatorial_Rossby_Wave':
        nDumpFrequency = 25
    elif (ProblemType == 'Plane_Gaussian_Wave' or ProblemType == 'Inertia_Gravity_Wave' 
          or ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave' 
          or ProblemType == 'NonLinear_Manufactured_Solution'):
        nDumpFrequency = 20
    elif ProblemType_NoExactSolution:
        if ReadFromSELFOutputData:
            nDumpFrequency = 86400*2
        else:
            nDumpFrequency = 1440
    # Note that nDumpFrequency is chosen in such a way that we end up with approximately 100 output files for the entire
    # simulation time.
    return nDumpFrequency


def SpecifyNumberOfTimeSteps(ProblemType,ProblemType_NoExactSolution,ReadFromSELFOutputData):
    if ProblemType == 'Plane_Gaussian_Wave':
        nTime_Minimum = 2021 + 1
        nTime = 2040 + 1
    elif ProblemType == 'Coastal_Kelvin_Wave':
        nTime_Minimum = 1000 + 1
        nTime = 1000 + 1
    elif ProblemType == 'Inertia_Gravity_Wave':
        nTime_Minimum = 2043 + 1
        nTime = 2060 + 1
    elif ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':   
        nTime_Minimum = 2025 + 1
        nTime = 2040 + 1
    elif ProblemType_NoExactSolution:
        if ReadFromSELFOutputData:
            nTime_Minimum = 86400*60*2 + 1
            nTime = 86400*60*2 + 1
        else:
            nTime_Minimum = 86400*2 + 1
            nTime = 86400*2 + 1 
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        nTime_Minimum = 531 + 1
        nTime = 535 + 1
    elif ProblemType == 'Equatorial_Yanai_Wave':
        nTime_Minimum = 528 + 1
        nTime = 530 + 1
    elif ProblemType == 'Equatorial_Rossby_Wave':
        nTime_Minimum = 2625 + 1
        nTime = nTime_Minimum
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        nTime_Minimum = 533 + 1
        nTime = 535 + 1
    elif ProblemType == 'Barotropic_Tide':
        nTime_Minimum = 1042 + 1
        nTime = 1050 + 1
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        nTime_Minimum = 1072 + 1
        nTime = 1080 + 1
    # Note that (a) nTime_Minimum is the minimum integer such that (nTime_Minimum - 1) times the time step is greater 
    # than or equal to the simulation time, and (b) nTime is the minimum integer such that nTime >= nTime_Minimum and 
    # nTime - 1 is a multiple of nDumpFrequency.
    return nTime_Minimum, nTime


def SpecifyLogicalArrayPlot(ProblemType):
    if ProblemType == 'Coastal_Kelvin_Wave':
        PlotZonalVelocity = False
    else:
        PlotZonalVelocity = True
    if ProblemType == 'Equatorial_Kelvin_Wave':
        PlotMeridionalVelocity = False
    else:
        PlotMeridionalVelocity = True
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
    elif ProblemType == 'Planetary_Rossby_Wave':
        ProblemType_Title = 'Planetary Rossby Wave'
        ProblemType_FileName = 'PlanetaryRossbyWave'
    elif ProblemType == 'Topographic_Rossby_Wave':
        ProblemType_Title = 'Topographic Rossby Wave'
        ProblemType_FileName = 'TopographicRossbyWave'
    elif ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave':
        ProblemType_Title = 'Planetary Rossby Wave'
        ProblemType_FileName = 'CoastalKelvinInertiaGravityPlanetaryRossbyWave'
    elif ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave':
        ProblemType_Title = 'Topographic Rossby Wave'
        ProblemType_FileName = 'CoastalKelvinInertiaGravityTopographicRossbyWave'
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
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ProblemType_Title = 'Non-Linear Manufactured Solution'
        ProblemType_FileName = 'NonLinearManufacturedSolution'
    return ProblemType_Title, ProblemType_FileName


class NameList:
    
    def __init__(myNameList,ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                 LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                 Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,CourantNumber,
                 UseCourantNumberToDetermineTimeStep,ReadFromSELFOutputData):
        myNameList.ProblemType = ProblemType
        if (ProblemType == 'Coastal_Kelvin_Wave' or ProblemType == 'Inertia_Gravity_Wave' 
            or ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave'
            or ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Equatorial_Yanai_Wave' 
            or ProblemType == 'Equatorial_Rossby_Wave' or ProblemType == 'Equatorial_Inertia_Gravity_Wave'):
            myNameList.ProblemType_GeophysicalWave = True
        else:
            myNameList.ProblemType_GeophysicalWave = False
        if (ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave'
            or ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave'):
            myNameList.ProblemType_NoExactSolution = True
        else:
            myNameList.ProblemType_NoExactSolution = False
        if (ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Equatorial_Yanai_Wave' 
            or ProblemType == 'Equatorial_Rossby_Wave' or ProblemType == 'Equatorial_Inertia_Gravity_Wave'):
            myNameList.ProblemType_EquatorialWave = True
        else:
            myNameList.ProblemType_EquatorialWave = False
        if (ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave' 
            or ProblemType == 'NonLinear_Manufactured_Solution'):
            myNameList.NonTrivialSourceTerms = True
        else:
            myNameList.NonTrivialSourceTerms = False
        if ProblemType == 'NonLinear_Manufactured_Solution':
            myNameList.Problem_is_Linear = False
        else:
            myNameList.Problem_is_Linear = True
        if ProblemType == 'Coastal_Kelvin_Wave':
            myNameList.BoundaryCondition = 'NonPeriodic_x'
        elif ProblemType == 'Inertia_Gravity_Wave' or ProblemType == 'NonLinear_Manufactured_Solution':
            myNameList.BoundaryCondition = 'Periodic'
        elif (ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave' 
              or myNameList.ProblemType_EquatorialWave):
            myNameList.BoundaryCondition = 'NonPeriodic_y'
        elif myNameList.ProblemType_NoExactSolution:
            myNameList.BoundaryCondition = 'Radiation'
            # Choose the boundary condition to be 'Radiation' or 'Reflection' i.e. no normal flow at the solid boundary.
        else:
            myNameList.BoundaryCondition = 'NonPeriodic_xy'
        myNameList.lX, myNameList.lY = SpecifyDomainExtents(ProblemType,myNameList.ProblemType_NoExactSolution,
                                                            myNameList.ProblemType_EquatorialWave)
        myNameList.nElementsX = nElementsX
        myNameList.nElementsY = nElementsY
        myNameList.dx = myNameList.lX/float(nElementsX)
        myNameList.dy = myNameList.lY/float(nElementsY)
        myNameList.myExactSolutionParameters = (
        SpecifyExactSolutionParameters(ProblemType,myNameList.ProblemType_GeophysicalWave,
                                       myNameList.ProblemType_NoExactSolution,myNameList.ProblemType_EquatorialWave,
                                       PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes))
        (myNameList.ExactSurfaceElevationLimits, myNameList.ExactZonalVelocityLimits, 
         myNameList.ExactMeridionalVelocityLimits) = (
        ESST.DetermineExactSolutionLimits(ProblemType,myNameList.myExactSolutionParameters))
        myNameList.myTimeSteppingParameters = (
        SpecifyTimeSteppingParameters(TimeIntegrator,LF_TR_and_LF_AM3_with_FB_Feedback_Type,
                                      Generalized_FB_with_AB2_AM3_Step_Type,Generalized_FB_with_AB3_AM4_Step_Type))
        if UseCourantNumberToDetermineTimeStep and not(myNameList.ProblemType_NoExactSolution):
            cX1 = myNameList.myExactSolutionParameters.cX1
            cX2 = myNameList.myExactSolutionParameters.cX2
            cY1 = myNameList.myExactSolutionParameters.cY1
            cY2 = myNameList.myExactSolutionParameters.cY2
            abs_cX = max(abs(cX1),abs(cX2))
            abs_cY = max(abs(cY1),abs(cY2))
            myNameList.dt = CourantNumber/(abs_cX/(myNameList.dx/float(nXi**2)) + abs_cY/(myNameList.dy/float(nEta**2)))
            print('The time step for Courant number %.6f is %.3g seconds.' %(CourantNumber,myNameList.dt))
        else:
            myNameList.dt = SpecifyTimeStep(ProblemType,myNameList.ProblemType_NoExactSolution)
        myNameList.ReadFromSELFOutputData = ReadFromSELFOutputData
        myNameList.nDumpFrequency = SpecifyDumpFrequency(ProblemType,myNameList.ProblemType_NoExactSolution,
                                                         ReadFromSELFOutputData)
        if myNameList.ProblemType_NoExactSolution:
            nRestartFrequencyBynDumpFrequency = 10
        else:
            nRestartFrequencyBynDumpFrequency = 50
        myNameList.nRestartFrequency = nRestartFrequencyBynDumpFrequency*myNameList.nDumpFrequency
        # Specify myNameList.nRestartFrequency to be an integral multiple of myNameList.nDumpFrequency.
        myNameList.nTime_Minimum, myNameList.nTime = (
        SpecifyNumberOfTimeSteps(ProblemType,myNameList.ProblemType_NoExactSolution,ReadFromSELFOutputData))
        myNameList.LogicalArrayPlot = SpecifyLogicalArrayPlot(ProblemType)
        myNameList.ProblemType_Title, myNameList.ProblemType_FileName = SpecifyTitleAndFileNamePrefixes(ProblemType)
        myNameList.nEquations = 3
        
    def ModifyNameList(myNameList,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,nXi,nEta,CourantNumber,
                       UseCourantNumberToDetermineTimeStep,BoundaryCondition,lX,lY):
        myNameList.BoundaryCondition = BoundaryCondition
        myNameList.lX = lX
        myNameList.lY = lY
        myNameList.dx = lX/float(myNameList.nElementsX)
        myNameList.dy = lY/float(myNameList.nElementsY)
        myNameList.myExactSolutionParameters = (
        SpecifyExactSolutionParameters(myNameList.ProblemType,myNameList.ProblemType_GeophysicalWave,
                                       myNameList.ProblemType_NoExactSolution,myNameList.ProblemType_EquatorialWave,
                                       PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,
                                       DomainExtentsSpecified=True,lX=lX,lY=lY))
        (myNameList.ExactSurfaceElevationLimits, myNameList.ExactZonalVelocityLimits, 
         myNameList.ExactMeridionalVelocityLimits) = (
        ESST.DetermineExactSolutionLimits(myNameList.ProblemType,myNameList.myExactSolutionParameters))
        if UseCourantNumberToDetermineTimeStep and not(myNameList.ProblemType_NoExactSolution):
            cX1 = myNameList.myExactSolutionParameters.cX1
            cX2 = myNameList.myExactSolutionParameters.cX2
            cY1 = myNameList.myExactSolutionParameters.cY1
            cY2 = myNameList.myExactSolutionParameters.cY2
            abs_cX = max(abs(cX1),abs(cX2))
            abs_cY = max(abs(cY1),abs(cY2))
            myNameList.dt = CourantNumber/(abs_cX/(myNameList.dx/float(nXi**2)) + abs_cY/(myNameList.dy/float(nEta**2)))
            print('The time step for Courant number %.6f is %.3g seconds.' %(CourantNumber,myNameList.dt))
        else:
            myNameList.dt = SpecifyTimeStep(myNameList.ProblemType,myNameList.ProblemType_NoExactSolution)