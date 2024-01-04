"""
Name: ExactSolutionsAndSourceTerms.py
Author: Siddhartha Bishnu
Details: This script contains functions for determining (a) the exact solutions of various test cases including 
geophysical waves, and (b) the source terms of the Rossby waves and the non-linear manufactured solutions as functions 
of space of time appearing on the right hand side of the prognostic equations.
"""


import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify


def SurfacElevationTestFunction(etaHat,kX,kY,x,y):
    eta = etaHat*np.sin(kX*x)*np.sin(kY*y)
    return eta


def ZonalVelocityTestFunction(etaHat,kX,kY,x,y):
    eta_x = etaHat*kX*np.cos(kX*x)*np.sin(kY*y)
    u = eta_x
    return u


def MeridionalVelocityTestFunction(etaHat,kX,kY,x,y):
    eta_y = etaHat*kY*np.sin(kX*x)*np.cos(kY*y)
    v = eta_y
    return v


def DeterminePlaneGaussianWaveExactSurfaceElevation(c0,g,kX,kY,R0,x0,y0,x,y,time):
    eta = np.exp(-(kX*(x - x0) + kY*(y - y0) - c0*time)**2.0/R0**2.0)/g
    return eta


def DeterminePlaneGaussianWaveExactZonalVelocity(c0,kX,kY,R0,x0,y0,x,y,time):
    u = kX*np.exp(-(kX*(x - x0) + kY*(y - y0) - c0*time)**2.0/R0**2.0)/c0
    return u


def DeterminePlaneGaussianWaveExactMeridionalVelocity(c0,kX,kY,R0,x0,y0,x,y,time):
    v = kY*np.exp(-(kX*(x - x0) + kY*(y - y0) - c0*time)**2.0/R0**2.0)/c0
    return v


def DetermineKelvinWaveAmplitude():
    x = sp.Symbol('x')
    f = sp.sin(x) + 2*sp.sin(2*x)
    Amplitude = sp.calculus.util.maximum(f,x,sp.sets.Interval(0,2*sp.pi))
    return Amplitude


def CoastalKelvinWaveFunctionalForm(etaHat,kY,y): 
    eta = etaHat*np.sin(kY*y)
    return eta


def DetermineCoastalKelvinWaveExactSurfaceElevation(c0,etaHat,H0,kY,R,x,y,time):
    CoastalKelvinWaveExactSurfaceElevation = -H0*CoastalKelvinWaveFunctionalForm(etaHat,kY,y+c0*time)*np.exp(-x/R)
    return CoastalKelvinWaveExactSurfaceElevation


def DetermineCoastalKelvinWaveExactMeridionalVelocity(c0,etaHat,kY,R,x,y,time):
    CoastalKelvinWaveExactMeridionalVelocity = c0*CoastalKelvinWaveFunctionalForm(etaHat,kY,y+c0*time)*np.exp(-x/R)
    return CoastalKelvinWaveExactMeridionalVelocity


def DetermineInertiaGravityWaveExactSurfaceElevation(etaHat,kX,kY,omega,x,y,time):
    eta = etaHat*np.cos(kX*x + kY*y - omega*time)
    return eta


def DetermineInertiaGravityWaveExactZonalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    u = etaHat*(g/(omega**2.0 - f0**2.0)*(omega*kX*np.cos(kX*x + kY*y - omega*time) 
                                          - f0*kY*np.sin(kX*x + kY*y - omega*time)))
    return u


def DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    v = etaHat*(g/(omega**2.0 - f0**2.0)*(omega*kY*np.cos(kX*x + kY*y - omega*time) 
                                          + f0*kX*np.sin(kX*x + kY*y - omega*time)))
    return v


def DetermineInertiaGravityWavesExactSurfaceElevation(etaHat,kX,kY,omega,x,y,time):
    eta = etaHat*(2.0*np.cos(omega*time)*np.cos(kX*x + kY*y))
    return eta


def DetermineInertiaGravityWavesExactZonalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    u = etaHat*(2.0*g*np.sin(kX*x + kY*y)/(omega**2.0 - f0**2.0)*(omega*kX*np.sin(omega*time) 
                                                                  - f0*kY*np.cos(omega*time)))
    return u


def DetermineInertiaGravityWavesExactMeridionalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    v = etaHat*(2.0*g*np.sin(kX*x + kY*y)/(omega**2.0 - f0**2.0)*(omega*kY*np.sin(omega*time) 
                                                                  + f0*kX*np.cos(omega*time)))
    return v


def DetermineManufacturedPlanetaryRossbyWaveExactSurfaceElevation(etaHat,kX,kY,omega,x,y,time):
    eta = etaHat*np.sin(kX*x + kY*y - omega*time)
    return eta


def DetermineManufacturedPlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat,f0,g,kX,kY,omega,x,y,time):
    u = -etaHat*g*(kY*(f0 - beta0*y)*np.cos(kX*x + kY*y - omega*time) 
                   + kX*omega*np.sin(kX*x + kY*y - omega*time))/f0**2.0
    return u


def DetermineManufacturedPlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat,f0,g,kX,kY,omega,x,y,time):
    v = etaHat*g*(kX*(f0 - beta0*y)*np.cos(kX*x + kY*y - omega*time) 
                  - kY*omega*np.sin(kX*x + kY*y - omega*time))/f0**2.0
    return v


def DetermineManufacturedPlanetaryRossbyWaveZonalVelocitySourceTerm(beta0,etaHat,f0,g,kX,kY,omega,x,y,time):
    uSourceTerm = etaHat*g*(2.0*kY*omega*beta0*y*np.sin(kX*x + kY*y - omega*time)
                            + kX*(omega**2.0 + (beta0*y)**2.0)*np.cos(kX*x + kY*y - omega*time))/f0**2.0
    return uSourceTerm


def DetermineManufacturedPlanetaryRossbyWaveMeridionalVelocitySourceTerm(beta0,etaHat,f0,g,kX,kY,omega,x,y,time):
    vSourceTerm = etaHat*g*(-2.0*kX*omega*beta0*y*np.sin(kX*x + kY*y - omega*time)
                            + kY*(omega**2.0 + (beta0*y)**2.0)*np.cos(kX*x + kY*y - omega*time))/f0**2.0
    return vSourceTerm


def DetermineManufacturedTopographicRossbyWaveExactSurfaceElevation(etaHat,kX,kY,omega,x,y,time):
    eta = etaHat*np.sin(kX*x + kY*y - omega*time)
    return eta


def DetermineManufacturedTopographicRossbyWaveExactZonalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    u = -etaHat*g*(f0*kY*np.cos(kX*x + kY*y - omega*time) + omega*kX*np.sin(kX*x + kY*y - omega*time))/f0**2.0
    return u


def DetermineManufacturedTopographicRossbyWaveExactMeridionalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    v = etaHat*g*(f0*kX*np.cos(kX*x + kY*y - omega*time) - omega*kY*np.sin(kX*x + kY*y - omega*time))/f0**2.0
    return v


def DetermineManufacturedTopographicRossbyWaveZonalVelocitySourceTerm(etaHat,f0,g,kX,kY,omega,x,y,time):
    uSourceTerm = etaHat*g*kX*omega**2.0*np.cos(kX*x + kY*y - omega*time)/f0**2.0
    return uSourceTerm


def DetermineManufacturedTopographicRossbyWaveMeridionalVelocitySourceTerm(etaHat,f0,g,kX,kY,omega,x,y,time):
    vSourceTerm = etaHat*g*kY*omega**2.0*np.cos(kX*x + kY*y - omega*time)/f0**2.0
    return vSourceTerm


def DetermineManufacturedTopographicRossbyWaveSurfaceElevationSourceTerm(alpha0,etaHat,f0,g,kX,kY,omega,x,y,time):
    etaSourceTerm = -etaHat*g*alpha0*(kY*omega*np.sin(kX*x + kY*y - omega*time) 
                                      + (kX*f0 + (kX**2.0 + kY**2.0)*omega*y)*np.cos(kX*x + kY*y - omega*time))/f0**2.0
    return etaSourceTerm


def DetermineRossbyWaveInitialSurfaceElevation(etaHat,R0x,R0y,x0,y0,x,y):
    eta = etaHat*np.exp(-((x - x0)**2.0/(2.0*R0x**2.0) + (y - y0)**2.0/(2.0*R0y**2.0)))
    return eta


def DetermineRossbyWaveInitialZonalVelocity(etaHat,f0_MidLatitude,g,R0x,R0y,x0,y0,x,y):
    u = g/(f0_MidLatitude*R0y**2.0)*(y - y0)*etaHat*np.exp(-((x - x0)**2.0/(2.0*R0x**2.0) 
                                                             + (y - y0)**2.0/(2.0*R0y**2.0)))
    return u


def DetermineRossbyWaveInitialMeridionalVelocity(etaHat,f0_MidLatitude,g,R0x,R0y,x0,y0,x,y):
    v = -g/(f0_MidLatitude*R0x**2.0)*(x - x0)*etaHat*np.exp(-((x - x0)**2.0/(2.0*R0x**2.0) 
                                                              + (y - y0)**2.0/(2.0*R0y**2.0)))
    return v


def EquatorialKelvinWaveFunctionalForm(etaHat,kX,x): 
    eta = etaHat*np.sin(kX*x)
    return eta


def DetermineEquatorialKelvinWaveExactSurfaceElevation(c0,etaHat,H0,kX,Req,x,y,time):
    EquatorialKelvinWaveExactSurfaceElevation = (
    H0*EquatorialKelvinWaveFunctionalForm(etaHat,kX,x-c0*time)*np.exp(-0.5*(y/Req)**2.0))
    return EquatorialKelvinWaveExactSurfaceElevation


def DetermineEquatorialKelvinWaveExactZonalVelocity(c0,etaHat,kX,Req,x,y,time):
    EquatorialKelvinWaveExactZonalVelocity = (
    c0*EquatorialKelvinWaveFunctionalForm(etaHat,kX,x-c0*time)*np.exp(-0.5*(y/Req)**2.0))
    return EquatorialKelvinWaveExactZonalVelocity


def DetermineHermiteFunctionMaximumAmplitudeWithMeridionalLocation(ProblemType,ReturnMeridionalLocation=True):
    if ProblemType == 'Equatorial_Yanai_Wave': # i.e. if HermitePolynomialOrder == 0:
        yMaximumAmplitude = 0.0
        HermitePolynomial = 1.0
        HermiteFunctionMaximumAmplitude = np.exp(-yMaximumAmplitude**2.0/2.0)*HermitePolynomial/np.pi**0.25
    elif ProblemType == 'Equatorial_Rossby_Wave': # i.e. if HermitePolynomialOrder == 1:
        yMaximumAmplitude = 1.0
        HermitePolynomial = 2.0*yMaximumAmplitude
        HermiteFunctionMaximumAmplitude = (
        np.exp(-yMaximumAmplitude**2.0/2.0)*HermitePolynomial/np.sqrt(2.0*np.sqrt(np.pi)))
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave': # i.e. if HermitePolynomialOrder == 2:
        yMaximumAmplitude = np.sqrt(2.5)
        HermitePolynomial = 4.0*yMaximumAmplitude**2.0 - 2.0
        HermiteFunctionMaximumAmplitude = (
        np.exp(-yMaximumAmplitude**2.0/2.0)*HermitePolynomial/(2.0*np.sqrt(2.0*np.sqrt(np.pi))))
    if ReturnMeridionalLocation:
        return yMaximumAmplitude, HermiteFunctionMaximumAmplitude
    else:
        return HermiteFunctionMaximumAmplitude


def DetermineEquatorialYanaiWaveNonDimensionalAngularFrequency(kX):
    omega = 0.5*(kX + np.sqrt(kX**2.0 + 4.0))
    return omega


def DetermineEquatorialRossbyAndInertiaGravityWaveNonDimensionalAngularFrequency(EquatorialWaveType,kX,m=1,
                                                                                 PrintOutcome=False):
    if EquatorialWaveType == 'Equatorial_Rossby_Wave':
        omega = 0.25
    elif EquatorialWaveType == 'Equatorial_Inertia_Gravity_Wave':
        omega = 2.25
    nIterations = 10**6
    Tolerance = 10.0**(-12.0)
    converged = False
    for iIteration in range(0,nIterations):
        if EquatorialWaveType == 'Equatorial_Inertia_Gravity_Wave' and omega < 0.0:
            omega *= -1.0
        f = omega**3.0 - (kX**2.0 + 2.0*m + 1.0)*omega - kX
        fPrime = 3.0*omega**2.0 - (kX**2.0 + 2.0*m + 1.0)
        Delta = -f/fPrime
        omega += Delta
        if omega > 0.0 and abs(Delta/omega) <= Tolerance:
            converged = True
            iIterationFinal = iIteration
            break
    if PrintOutcome:
        if converged:
            print('The Newton Raphson solver for the Equatorial angular frequency has converged within %d iterations.' 
                  %(iIterationFinal+1))
            print('The angular frequency is %.6f.' %omega)
        else:
            print('The Newton Raphson solver for the Equatorial angular frequency has not converged within '
                  + '%d iterations.' %(iIterationFinal+1))            
    return omega


def DetermineEquatorialWaveExactNonDimensionalStateVariables(HermitePolynomialOrder):
    kX, omega, t, x, y = sp.symbols('kX, omega, t, x, y')
    if HermitePolynomialOrder == 0:
        H = 1
        Psi = sp.exp(-y**2/2)*H/sp.pi**(1/4)       
    elif HermitePolynomialOrder == 1:
        H = 2*y
        Psi = sp.exp(-y**2/2)*H/sp.sqrt(2*sp.sqrt(sp.pi))
    elif HermitePolynomialOrder == 2:  
        H = 4*y**2 - 2
        Psi = sp.exp(-y**2/2)*H/(2*sp.sqrt(2*sp.sqrt(sp.pi)))        
    Psi_y = sp.diff(Psi,y)    
    u = (omega*y*Psi - kX*Psi_y)/(kX**2 - omega**2)*sp.sin(kX*x - omega*t)
    v = Psi*sp.cos(kX*x - omega*t)
    eta = (kX*y*Psi - omega*Psi_y)/(kX**2 - omega**2)*sp.sin(kX*x - omega*t)
    DetermineEquatorialWaveExactNonDimensionalZonalVelocity = lambdify((kX,omega,x,y,t,), u, modules="numpy")
    DetermineEquatorialWaveExactNonDimensionalMeridionalVelocity = lambdify((kX,omega,x,y,t,), v, modules="numpy")
    DetermineEquatorialWaveExactNonDimensionalSurfaceElevation = lambdify((kX,omega,x,y,t,), eta, modules="numpy")
    return [DetermineEquatorialWaveExactNonDimensionalZonalVelocity,
            DetermineEquatorialWaveExactNonDimensionalMeridionalVelocity,
            DetermineEquatorialWaveExactNonDimensionalSurfaceElevation]    


[DetermineEquatorialYanaiWaveExactNonDimensionalZonalVelocity,
 DetermineEquatorialYanaiWaveExactNonDimensionalMeridionalVelocity,
 DetermineEquatorialYanaiWaveExactNonDimensionalSurfaceElevation] = (
DetermineEquatorialWaveExactNonDimensionalStateVariables(HermitePolynomialOrder=0))


def DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat,kX,omega,LengthScale,TimeScale,SurfaceElevationScale,x,y,
                                                      time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    eta = SurfaceElevationScale*DetermineEquatorialYanaiWaveExactNonDimensionalSurfaceElevation(kX,omega,x,y,time)
    eta *= etaHat
    return eta    


def DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    u = VelocityScale*DetermineEquatorialYanaiWaveExactNonDimensionalZonalVelocity(kX,omega,x,y,time)
    u *= etaHat
    return u    


def DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    v = VelocityScale*DetermineEquatorialYanaiWaveExactNonDimensionalMeridionalVelocity(kX,omega,x,y,time)
    v *= etaHat
    return v    


[DetermineEquatorialRossbyWaveExactNonDimensionalZonalVelocity,
 DetermineEquatorialRossbyWaveExactNonDimensionalMeridionalVelocity,
 DetermineEquatorialRossbyWaveExactNonDimensionalSurfaceElevation] = (
DetermineEquatorialWaveExactNonDimensionalStateVariables(HermitePolynomialOrder=1))


def DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat,kX,omega,LengthScale,TimeScale,SurfaceElevationScale,x,y,
                                                       time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    eta = SurfaceElevationScale*DetermineEquatorialRossbyWaveExactNonDimensionalSurfaceElevation(kX,omega,x,y,time)
    eta *= etaHat
    return eta    


def DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    u = VelocityScale*DetermineEquatorialRossbyWaveExactNonDimensionalZonalVelocity(kX,omega,x,y,time)
    u *= etaHat
    return u    


def DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    v = VelocityScale*DetermineEquatorialRossbyWaveExactNonDimensionalMeridionalVelocity(kX,omega,x,y,time)
    v *= etaHat
    return v 


[DetermineEquatorialInertiaGravityWaveExactNonDimensionalZonalVelocity,
 DetermineEquatorialInertiaGravityWaveExactNonDimensionalMeridionalVelocity,
 DetermineEquatorialInertiaGravityWaveExactNonDimensionalSurfaceElevation] = (
DetermineEquatorialWaveExactNonDimensionalStateVariables(HermitePolynomialOrder=2))


def DetermineEquatorialInertiaGravityWaveExactSurfaceElevation(etaHat,kX,omega,LengthScale,TimeScale,
                                                               SurfaceElevationScale,x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    eta = (SurfaceElevationScale
           *DetermineEquatorialInertiaGravityWaveExactNonDimensionalSurfaceElevation(kX,omega,x,y,time))
    eta *= etaHat
    return eta    


def DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,y,
                                                            time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    u = VelocityScale*DetermineEquatorialInertiaGravityWaveExactNonDimensionalZonalVelocity(kX,omega,x,y,time)
    u *= etaHat
    return u    


def DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,
                                                                 y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    v = VelocityScale*DetermineEquatorialInertiaGravityWaveExactNonDimensionalMeridionalVelocity(kX,omega,x,y,time)
    v *= etaHat
    return v


def DetermineBarotropicTideExactSurfaceElevation(etaHat,kX,omega,x,y,time):
    eta = etaHat*np.cos(kX*x)*np.cos(omega*time)
    # Note that eta = 0.5*etaHat(np.cos(kX*x + omega*t) + np.cos(kX*x - omega*t))
    return eta


def DetermineBarotropicTideExactZonalVelocity(etaHat,f0,g,kX,omega,x,y,time):
    u = etaHat*g*omega*kX*np.sin(kX*x)*np.sin(omega*time)/(omega**2.0 - f0**2.0)
    return u


def DetermineBarotropicTideExactMeridionalVelocity(etaHat,f0,g,kX,omega,x,y,time):
    v = etaHat*g*f0*kX*np.sin(kX*x)*np.cos(omega*time)/(omega**2.0 - f0**2.0)
    return v


def DetermineDiffusionEquationExactZonalVelocity(kappa,kX,kY,x,y,time):
    u = np.sin(kX*x)*np.sin(kY*y)*np.exp(-kappa*time) 
    return u
    
    
def DetermineDiffusionEquationExactZonalVelocityZonalGradient(kappa,kX,kY,x,y,time):
    u_x = kX*np.cos(kX*x)*np.sin(kY*y)*np.exp(-kappa*time)
    return u_x
    
    
def DetermineDiffusionEquationExactZonalVelocityMeridionalGradient(kappa,kX,kY,x,y,time):
    u_y = kY*np.sin(kX*x)*np.cos(kY*y)*np.exp(-kappa*time)
    return u_y


def DetermineAdvectionDiffusionEquationExactSurfaceElevation(nu,u0,v0,x0,y0,x,y,time):
    eta = (np.exp(-((x - u0*time - x0)**2.0 + (y - v0*time - y0)**2.0)/(nu*(4.0*time + 1.0))))/(4.0*time + 1.0)
    return eta


def DetermineAdvectionDiffusionEquationExactSurfaceElevationZonalGradient(nu,u0,v0,x0,y0,x,y,time):
    eta_x = (-(2.0*(x - u0*time - x0)
               *np.exp(-((x - u0*time - x0)**2.0 + (y - v0*time - y0)**2.0)/(nu*(4.0*time + 1.0))))
             /(nu*(4.0*time + 1.0)**2.0))
    return eta_x


def DetermineAdvectionDiffusionEquationExactSurfaceElevationMeridionalGradient(nu,u0,v0,x0,y0,x,y,time):
    eta_y = (-(2.0*(y - v0*time - y0) 
               *np.exp(-((x - u0*time - x0)**2.0 + (y - v0*time - y0)**2.0)/(nu*(4.0*time + 1.0))))
             /(nu*(4.0*time + 1.0)**2.0))
    return eta_y


def DetermineNonLinearManufacturedSolutionExactSurfaceElevation(etaHat,kX,kY,omega,x,y,time):
    eta = etaHat*np.sin(kX*x + kY*y - omega*time)
    return eta


def DetermineNonLinearManufacturedSolutionExactZonalVelocity(etaHat,kX,kY,omega,x,y,time):
    u = etaHat*np.cos(kX*x + kY*y - omega*time)
    return u


def DetermineNonLinearManufacturedSolutionExactMeridionalVelocity(etaHat,kX,kY,omega,x,y,time):
    v = etaHat*np.cos(kX*x + kY*y - omega*time)
    return v


def DetermineNonLinearManufacturedSolutionSurfaceElevationSourceTerm(etaHat,H0,kX,kY,omega,x,y,time):
    phase = kX*x + kY*y - omega*time
    etaSourceTerm = (
    etaHat**2.0*((kX + kY)*np.cos(2.0*phase)) - etaHat*((kX + kY)*H0*np.sin(phase) + omega*np.cos(phase)))
    return etaSourceTerm


def DetermineNonLinearManufacturedSolutionZonalVelocitySourceTerm(etaHat,f0,g,H0,kX,kY,omega,x,y,time):
    phase = kX*x + kY*y - omega*time
    uSourceTerm = (etaHat*H0*((g*kX - f0)*np.cos(phase) + omega*np.sin(phase)) 
                   + etaHat**2.0*(np.sin(2.0*phase)*(-H0*(kX + kY) + 0.5*(g*kX - f0)) - omega*np.cos(2.0*phase)) 
                   + etaHat**3.0*(kX + kY)*(-np.sin(2.0*phase)*np.sin(phase) + (np.cos(phase))**3.0))
    return uSourceTerm


def DetermineNonLinearManufacturedSolutionMeridionalVelocitySourceTerm(etaHat,f0,g,H0,kX,kY,omega,x,y,time):
    phase = kX*x + kY*y - omega*time
    vSourceTerm = (etaHat*H0*((g*kY + f0)*np.cos(phase) + omega*np.sin(phase))
                   + etaHat**2.0*(np.sin(2.0*phase)*(-H0*(kX + kY) + 0.5*(g*kY + f0)) - omega*np.cos(2.0*phase))
                   + etaHat**3.0*(kX + kY)*(-np.sin(2.0*phase)*np.sin(phase) + (np.cos(phase))**3.0))
    return vSourceTerm


def DetermineViscousBurgersEquationExactZonalVelocity(nu,s,uL,uR,x0,x,time):
    u = s - 0.5*(uL - uR)*np.tanh((x - x0 - s*time)*(uL - uR)/(4.0*nu))
    return u


def DetermineViscousBurgersEquationExactZonalVelocityZonalGradient(nu,s,uL,uR,x0,x,time):
    u_x = -1.0/(8.0*nu)*(uL - uR)**2.0/(np.cosh((x - x0 - s*time)*(uL - uR)/(4.0*nu)))**2.0
    return u_x


def DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,x,y,time):
    c0 = myExactSolutionParameters.c0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
    g = myExactSolutionParameters.g
    H0 = myExactSolutionParameters.H0
    kX1 = myExactSolutionParameters.kX1
    kX2 = myExactSolutionParameters.kX2
    kY1 = myExactSolutionParameters.kY1
    kY2 = myExactSolutionParameters.kY2
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    nu = myExactSolutionParameters.nu
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    x0 = myExactSolutionParameters.x0
    y0 = myExactSolutionParameters.y0
    R0 = myExactSolutionParameters.R0
    R0x = myExactSolutionParameters.R0x
    R0y = myExactSolutionParameters.R0y
    u0 = myExactSolutionParameters.u0
    v0 = myExactSolutionParameters.v0
    R = myExactSolutionParameters.R
    Req = myExactSolutionParameters.Req
    LengthScale = myExactSolutionParameters.LengthScale
    TimeScale = myExactSolutionParameters.TimeScale
    SurfaceElevationScale = myExactSolutionParameters.SurfaceElevationScale
    if ProblemType == 'Convergence_of_Spatial_Operators':
        ExactSurfaceElevation = SurfacElevationTestFunction(etaHat1,kX1,kY1,x,y)
    elif ProblemType == 'Plane_Gaussian_Wave':
        ExactSurfaceElevation = DeterminePlaneGaussianWaveExactSurfaceElevation(c0,g,kX1,kY1,R0,x0,y0,x,y,time)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        ExactSurfaceElevation = (DetermineCoastalKelvinWaveExactSurfaceElevation(c0,etaHat1,H0,kY1,R,x,y,time)
                                 + DetermineCoastalKelvinWaveExactSurfaceElevation(c0,etaHat2,H0,kY2,R,x,y,time))
    elif ProblemType == 'Inertia_Gravity_Wave':
        ExactSurfaceElevation = (DetermineInertiaGravityWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
                                 + DetermineInertiaGravityWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time))
    elif ProblemType == 'Manufactured_Planetary_Rossby_Wave':
        ExactSurfaceElevation = (
        (DetermineManufacturedPlanetaryRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
         + DetermineManufacturedPlanetaryRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        ExactSurfaceElevation = (
        (DetermineManufacturedTopographicRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
         + DetermineManufacturedTopographicRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        if time == 0.0:
            ExactSurfaceElevation = DetermineRossbyWaveInitialSurfaceElevation(etaHat1,R0x,R0y,x0,y0,x,y)
        else:
            ExactSurfaceElevation = 0.0
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        ExactSurfaceElevation = (DetermineEquatorialKelvinWaveExactSurfaceElevation(c0,etaHat1,H0,kX1,Req,x,y,time)
                                 + DetermineEquatorialKelvinWaveExactSurfaceElevation(c0,etaHat2,H0,kX2,Req,x,y,time))
    elif ProblemType == 'Equatorial_Yanai_Wave':
        ExactSurfaceElevation = (
        (DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                           SurfaceElevationScale,x,y,time)
         + DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                             SurfaceElevationScale,x,y,time)))
    elif ProblemType == 'Equatorial_Rossby_Wave':
        ExactSurfaceElevation = (
        (DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                            SurfaceElevationScale,x,y,time)
         + DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                              SurfaceElevationScale,x,y,time)))  
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        ExactSurfaceElevation = (
        (DetermineEquatorialInertiaGravityWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                    SurfaceElevationScale,x,y,time)
         + DetermineEquatorialInertiaGravityWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                      SurfaceElevationScale,x,y,time)))  
    elif ProblemType == 'Barotropic_Tide':
        ExactSurfaceElevation = (DetermineBarotropicTideExactSurfaceElevation(etaHat1,kX1,omega1,x,y,time)
                                 + DetermineBarotropicTideExactSurfaceElevation(etaHat2,kX2,omega2,x,y,time))
    elif ProblemType == 'Advection_Diffusion_Equation':
        ExactSurfaceElevation = DetermineAdvectionDiffusionEquationExactSurfaceElevation(nu,u0,v0,x0,y0,x,y,time)
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactSurfaceElevation = (
        DetermineNonLinearManufacturedSolutionExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time))
    else:
        ExactSurfaceElevation = 0.0
    return ExactSurfaceElevation


def DetermineExactSurfaceElevationZonalGradient(ProblemType,myExactSolutionParameters,x,y,time):
    nu = myExactSolutionParameters.nu
    u0 = myExactSolutionParameters.u0
    v0 = myExactSolutionParameters.v0
    x0 = myExactSolutionParameters.x0
    y0 = myExactSolutionParameters.y0
    if ProblemType == 'Advection_Diffusion_Equation':
        ExactSurfaceElevationZonalGradient = (
        DetermineAdvectionDiffusionEquationExactSurfaceElevationZonalGradient(nu,u0,v0,x0,y0,x,y,time))
    else:
        ExactSurfaceElevationZonalGradient = 0.0
    return ExactSurfaceElevationZonalGradient       
        
        
def DetermineExactSurfaceElevationMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,time):
    nu = myExactSolutionParameters.nu
    u0 = myExactSolutionParameters.u0
    v0 = myExactSolutionParameters.v0
    x0 = myExactSolutionParameters.x0
    y0 = myExactSolutionParameters.y0
    if ProblemType == 'Advection_Diffusion_Equation':
        ExactSurfaceElevationMeridionalGradient = (
        DetermineAdvectionDiffusionEquationExactSurfaceElevationMeridionalGradient(nu,u0,v0,x0,y0,x,y,time))
    else:
        ExactSurfaceElevationMeridionalGradient = 0.0
    return ExactSurfaceElevationMeridionalGradient       


def DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,x,y,time):
    beta0 = myExactSolutionParameters.beta0
    c0 = myExactSolutionParameters.c0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
    f0_MidLatitude = myExactSolutionParameters.f0_MidLatitude
    f0 = myExactSolutionParameters.f0
    g = myExactSolutionParameters.g
    kappa1 = myExactSolutionParameters.kappa1
    kX1 = myExactSolutionParameters.kX1
    kX2 = myExactSolutionParameters.kX2 
    kY1 = myExactSolutionParameters.kY1
    kY2 = myExactSolutionParameters.kY2 
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    nu = myExactSolutionParameters.nu
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    x0 = myExactSolutionParameters.x0
    y0 = myExactSolutionParameters.y0
    R0 = myExactSolutionParameters.R0
    R0x = myExactSolutionParameters.R0x
    R0y = myExactSolutionParameters.R0y
    s = myExactSolutionParameters.s
    uL = myExactSolutionParameters.uL
    uR = myExactSolutionParameters.uR
    Req = myExactSolutionParameters.Req
    LengthScale = myExactSolutionParameters.LengthScale
    TimeScale = myExactSolutionParameters.TimeScale
    VelocityScale = myExactSolutionParameters.VelocityScale
    if ProblemType == 'Convergence_of_Spatial_Operators':
        ExactZonalVelocity = ZonalVelocityTestFunction(etaHat1,kX1,kY1,x,y)
    elif ProblemType == 'Plane_Gaussian_Wave':
        ExactZonalVelocity = DeterminePlaneGaussianWaveExactZonalVelocity(c0,kX1,kY1,R0,x0,y0,x,y,time)
    elif ProblemType == 'Inertia_Gravity_Wave':
        ExactZonalVelocity = (DetermineInertiaGravityWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
                              + DetermineInertiaGravityWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))
    elif ProblemType == 'Manufactured_Planetary_Rossby_Wave':
        ExactZonalVelocity = (
        (DetermineManufacturedPlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineManufacturedPlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        ExactZonalVelocity = (
        (DetermineManufacturedTopographicRossbyWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineManufacturedTopographicRossbyWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        if time == 0.0:
            ExactZonalVelocity = DetermineRossbyWaveInitialZonalVelocity(etaHat1,f0_MidLatitude,g,R0x,R0y,x0,y0,x,y)
        else:
            ExactZonalVelocity = 0.0
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        ExactZonalVelocity = (DetermineEquatorialKelvinWaveExactZonalVelocity(c0,etaHat1,kX1,Req,x,y,time)
                              + DetermineEquatorialKelvinWaveExactZonalVelocity(c0,etaHat2,kX2,Req,x,y,time))
    elif ProblemType == 'Equatorial_Yanai_Wave':
        ExactZonalVelocity = (DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                             VelocityScale,x,y,time)
                              + DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                               VelocityScale,x,y,time))
    elif ProblemType == 'Equatorial_Rossby_Wave':
        ExactZonalVelocity = (DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                              VelocityScale,x,y,time)
                              + DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,
                                                                                TimeScale,VelocityScale,x,y,time))
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        ExactZonalVelocity = (
        (DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,
                                                                 x,y,time)
         + DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                   VelocityScale,x,y,time)))
    elif ProblemType == 'Barotropic_Tide':
        ExactZonalVelocity = (DetermineBarotropicTideExactZonalVelocity(etaHat1,f0,g,kX1,omega1,x,y,time)
                              + DetermineBarotropicTideExactZonalVelocity(etaHat2,f0,g,kX2,omega2,x,y,time))
    elif ProblemType == 'Diffusion_Equation':
        ExactZonalVelocity = DetermineDiffusionEquationExactZonalVelocity(kappa1,kX1,kY1,x,y,time)
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactZonalVelocity = DetermineNonLinearManufacturedSolutionExactZonalVelocity(etaHat1,kX1,kY1,omega1,x,y,time)
    elif ProblemType == 'Viscous_Burgers_Equation':
        ExactZonalVelocity = DetermineViscousBurgersEquationExactZonalVelocity(nu,s,uL,uR,x0,x,time)
    else:
        ExactZonalVelocity = 0.0
    return ExactZonalVelocity


def DetermineExactZonalVelocityZonalGradient(ProblemType,myExactSolutionParameters,x,y,time):
    kappa1 = myExactSolutionParameters.kappa1
    kX1 = myExactSolutionParameters.kX1
    kY1 = myExactSolutionParameters.kY1
    nu = myExactSolutionParameters.nu
    x0 = myExactSolutionParameters.x0
    s = myExactSolutionParameters.s
    uL = myExactSolutionParameters.uL
    uR = myExactSolutionParameters.uR  
    if ProblemType == 'Diffusion_Equation' or ProblemType == 'Viscous_Burgers_Equation':
        if ProblemType == 'Diffusion_Equation':
            ExactZonalVelocityZonalGradient = (
            DetermineDiffusionEquationExactZonalVelocityZonalGradient(kappa1,kX1,kY1,x,y,time))
        elif ProblemType == 'Viscous_Burgers_Equation':
            ExactZonalVelocityZonalGradient = (
            DetermineViscousBurgersEquationExactZonalVelocityZonalGradient(nu,s,uL,uR,x0,x,time))
    else:
        ExactZonalVelocityZonalGradient = 0.0
    return ExactZonalVelocityZonalGradient


def DetermineExactZonalVelocityMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,time):
    kappa1 = myExactSolutionParameters.kappa1
    kX1 = myExactSolutionParameters.kX1
    kY1 = myExactSolutionParameters.kY1
    if ProblemType == 'Diffusion_Equation':
        ExactZonalVelocityMeridionalGradient = (
        DetermineDiffusionEquationExactZonalVelocityMeridionalGradient(kappa1,kX1,kY1,x,y,time))
    else:
        ExactZonalVelocityMeridionalGradient = 0.0
    return ExactZonalVelocityMeridionalGradient


def DetermineExactMeridionalVelocity(ProblemType,myExactSolutionParameters,x,y,time):
    beta0 = myExactSolutionParameters.beta0
    c0 = myExactSolutionParameters.c0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
    f0_MidLatitude = myExactSolutionParameters.f0_MidLatitude
    f0 = myExactSolutionParameters.f0
    g = myExactSolutionParameters.g
    kX1 = myExactSolutionParameters.kX1
    kX2 = myExactSolutionParameters.kX2 
    kY1 = myExactSolutionParameters.kY1
    kY2 = myExactSolutionParameters.kY2 
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    x0 = myExactSolutionParameters.x0
    y0 = myExactSolutionParameters.y0
    R0 = myExactSolutionParameters.R0
    R0x = myExactSolutionParameters.R0x
    R0y = myExactSolutionParameters.R0y
    R = myExactSolutionParameters.R
    LengthScale = myExactSolutionParameters.LengthScale
    TimeScale = myExactSolutionParameters.TimeScale
    VelocityScale = myExactSolutionParameters.VelocityScale  
    if ProblemType == 'Convergence_of_Spatial_Operators':
        ExactMeridionalVelocity = MeridionalVelocityTestFunction(etaHat1,kX1,kY1,x,y)
    elif ProblemType == 'Plane_Gaussian_Wave':
        ExactMeridionalVelocity = DeterminePlaneGaussianWaveExactMeridionalVelocity(c0,kX1,kY1,R0,x0,y0,x,y,time)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        ExactMeridionalVelocity = (DetermineCoastalKelvinWaveExactMeridionalVelocity(c0,etaHat1,kY1,R,x,y,time)
                                   + DetermineCoastalKelvinWaveExactMeridionalVelocity(c0,etaHat2,kY2,R,x,y,time))
    elif ProblemType == 'Inertia_Gravity_Wave':
        ExactMeridionalVelocity = (DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,
                                                                                      time)
                                   + DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,
                                                                                        time))
    elif ProblemType == 'Manufactured_Planetary_Rossby_Wave':
        ExactMeridionalVelocity = (
        (DetermineManufacturedPlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineManufacturedPlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        ExactMeridionalVelocity = (
        (DetermineManufacturedTopographicRossbyWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineManufacturedTopographicRossbyWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
        if time == 0.0:
            ExactMeridionalVelocity = DetermineRossbyWaveInitialMeridionalVelocity(etaHat1,f0_MidLatitude,g,R0x,R0y,x0,
                                                                                   y0,x,y)
        else:
            ExactMeridionalVelocity = 0.0
    elif ProblemType == 'Equatorial_Yanai_Wave':
        ExactMeridionalVelocity = (
        (DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,x,y,
                                                             time)
         + DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,x,
                                                               y,time)))
    elif ProblemType == 'Equatorial_Rossby_Wave':
        ExactMeridionalVelocity = (
        (DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,x,
                                                              y,time)
         + DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,
                                                                x,y,time)))
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        ExactMeridionalVelocity = (
        (DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                      VelocityScale,x,y,time)
         + DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                        VelocityScale,x,y,time)))
    elif ProblemType == 'Barotropic_Tide':
        ExactMeridionalVelocity = (DetermineBarotropicTideExactMeridionalVelocity(etaHat1,f0,g,kX1,omega1,x,y,time)
                                   + DetermineBarotropicTideExactMeridionalVelocity(etaHat2,f0,g,kX2,omega2,x,y,time))
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactMeridionalVelocity = DetermineNonLinearManufacturedSolutionExactMeridionalVelocity(etaHat1,kX1,kY1,omega1,
                                                                                                x,y,time)
    else:
        ExactMeridionalVelocity = 0.0
    return ExactMeridionalVelocity


def DetermineExactMeridionalVelocityZonalGradient(ProblemType,myExactSolutionParameters,x,y,time):
    ExactMeridionalVelocityZonalGradient = 0.0
    return ExactMeridionalVelocityZonalGradient


def DetermineExactMeridionalVelocityMeridionalGradient(ProblemType,myExactSolutionParameters,x,y,time):
    ExactMeridionalVelocityMeridionalGradient = 0.0
    return ExactMeridionalVelocityMeridionalGradient


def DetermineSurfaceElevationSourceTerm(ProblemType,myExactSolutionParameters,x,y,time):
    alpha0 = myExactSolutionParameters.alpha0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
    f0 = myExactSolutionParameters.f0
    g = myExactSolutionParameters.g
    H0 = myExactSolutionParameters.H0
    kX1 = myExactSolutionParameters.kX1
    kX2 = myExactSolutionParameters.kX2
    kY1 = myExactSolutionParameters.kY1
    kY2 = myExactSolutionParameters.kY2
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    if ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        SurfaceElevationSourceTerm = (
        (DetermineManufacturedTopographicRossbyWaveSurfaceElevationSourceTerm(alpha0,etaHat1,f0,g,kX1,kY1,omega1,x,y,
                                                                              time)
         + DetermineManufacturedTopographicRossbyWaveSurfaceElevationSourceTerm(alpha0,etaHat2,f0,g,kX2,kY2,omega2,x,y,
                                                                                time)))
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        SurfaceElevationSourceTerm = (
        DetermineNonLinearManufacturedSolutionSurfaceElevationSourceTerm(etaHat1,H0,kX1,kY1,omega1,x,y,time))
    else:
        SurfaceElevationSourceTerm = 0.0
    return SurfaceElevationSourceTerm


def DetermineZonalVelocitySourceTerm(ProblemType,myExactSolutionParameters,x,y,time):
    beta0 = myExactSolutionParameters.beta0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
    f0 = myExactSolutionParameters.f0
    g = myExactSolutionParameters.g
    H0 = myExactSolutionParameters.H0
    kX1 = myExactSolutionParameters.kX1
    kX2 = myExactSolutionParameters.kX2
    kY1 = myExactSolutionParameters.kY1
    kY2 = myExactSolutionParameters.kY2
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    if ProblemType == 'Manufactured_Planetary_Rossby_Wave':
        ZonalVelocitySourceTerm = (
        (DetermineManufacturedPlanetaryRossbyWaveZonalVelocitySourceTerm(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineManufacturedPlanetaryRossbyWaveZonalVelocitySourceTerm(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        ZonalVelocitySourceTerm = (
        (DetermineManufacturedTopographicRossbyWaveZonalVelocitySourceTerm(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineManufacturedTopographicRossbyWaveZonalVelocitySourceTerm(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ZonalVelocitySourceTerm = (
        DetermineNonLinearManufacturedSolutionZonalVelocitySourceTerm(etaHat1,f0,g,H0,kX1,kY1,omega1,x,y,time))
    else:
        ZonalVelocitySourceTerm = 0.0
    return ZonalVelocitySourceTerm


def DetermineMeridionalVelocitySourceTerm(ProblemType,myExactSolutionParameters,x,y,time):
    beta0 = myExactSolutionParameters.beta0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
    f0 = myExactSolutionParameters.f0
    g = myExactSolutionParameters.g
    H0 = myExactSolutionParameters.H0
    kX1 = myExactSolutionParameters.kX1
    kX2 = myExactSolutionParameters.kX2
    kY1 = myExactSolutionParameters.kY1
    kY2 = myExactSolutionParameters.kY2
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    if ProblemType == 'Manufactured_Planetary_Rossby_Wave':
        MeridionalVelocitySourceTerm = (
        (DetermineManufacturedPlanetaryRossbyWaveMeridionalVelocitySourceTerm(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,
                                                                              time)
         + DetermineManufacturedPlanetaryRossbyWaveMeridionalVelocitySourceTerm(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,
                                                                                time)))
    elif ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        MeridionalVelocitySourceTerm = (
        (DetermineManufacturedTopographicRossbyWaveMeridionalVelocitySourceTerm(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineManufacturedTopographicRossbyWaveMeridionalVelocitySourceTerm(etaHat2,f0,g,kX2,kY2,omega2,x,y,
                                                                                  time)))
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        MeridionalVelocitySourceTerm = (
        DetermineNonLinearManufacturedSolutionMeridionalVelocitySourceTerm(etaHat1,f0,g,H0,kX1,kY1,omega1,x,y,time))
    else:
        MeridionalVelocitySourceTerm = 0.0
    return MeridionalVelocitySourceTerm


def DetermineSolutionAmplitude(ProblemType,myExactSolutionParameters):
    beta0 = myExactSolutionParameters.beta0
    c0 = myExactSolutionParameters.c0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
    f0 = myExactSolutionParameters.f0
    g = myExactSolutionParameters.g
    H0 = myExactSolutionParameters.H0
    kX1 = myExactSolutionParameters.kX1
    kX2 = myExactSolutionParameters.kX2 
    kY1 = myExactSolutionParameters.kY1
    kY2 = myExactSolutionParameters.kY2
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    lY = myExactSolutionParameters.lY
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    s = myExactSolutionParameters.s
    uL = myExactSolutionParameters.uL
    uR = myExactSolutionParameters.uR
    SurfaceElevationAmplitude = np.zeros(2)
    ZonalVelocityAmplitude = np.zeros(2)
    MeridionalVelocityAmplitude = np.zeros(2)
    if ProblemType == 'Coastal_Kelvin_Wave':
        SurfaceElevationAmplitude[0] = H0*etaHat1
        SurfaceElevationAmplitude[1] = H0*etaHat2
        MeridionalVelocityAmplitude[0] = c0*etaHat1
        MeridionalVelocityAmplitude[1] = c0*etaHat2
    elif ProblemType == 'Inertia_Gravity_Wave':
        SurfaceElevationAmplitude[0] = etaHat1
        SurfaceElevationAmplitude[1] = etaHat2
        ZonalVelocityAmplitude[0] = etaHat1*g/(omega1**2.0 - f0**2.0)*np.sqrt((omega1*kX1)**2.0 + (f0*kY1)**2.0)
        ZonalVelocityAmplitude[1] = etaHat2*g/(omega2**2.0 - f0**2.0)*np.sqrt((omega2*kX2)**2.0 + (f0*kY2)**2.0)
        MeridionalVelocityAmplitude[0] = (
        etaHat1*g/(omega1**2.0 - f0**2.0)*np.sqrt((omega1*kY1)**2.0 + (f0*kX1)**2.0))
        MeridionalVelocityAmplitude[1] = (
        etaHat2*g/(omega2**2.0 - f0**2.0)*np.sqrt((omega2*kY2)**2.0 + (f0*kX2)**2.0))
    elif ProblemType == 'Manufactured_Planetary_Rossby_Wave':
        SurfaceElevationAmplitude[0] = etaHat1
        SurfaceElevationAmplitude[1] = etaHat2
        ZonalVelocityAmplitude[0] = etaHat1*g*np.sqrt((beta0*kY1*lY)**2.0 + (kX1*omega1)**2.0)/f0**2.0
        ZonalVelocityAmplitude[1] = etaHat2*g*np.sqrt((beta0*kY2*lY)**2.0 + (kX2*omega2)**2.0)/f0**2.0
        MeridionalVelocityAmplitude[0] = etaHat1*g*np.sqrt((beta0*kX1*lY)**2.0 + (kY1*omega1)**2.0)/f0**2.0
        MeridionalVelocityAmplitude[1] = etaHat2*g*np.sqrt((beta0*kX2*lY)**2.0 + (kY2*omega1)**2.0)/f0**2.0
    elif ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        SurfaceElevationAmplitude[0] = etaHat1
        SurfaceElevationAmplitude[1] = etaHat2
        ZonalVelocityAmplitude[0] = etaHat1*g*kX1*omega1/f0**2.0
        ZonalVelocityAmplitude[1] = etaHat2*g*kX2*omega2/f0**2.0
        MeridionalVelocityAmplitude[0] = etaHat1*g*kY1*omega1/f0**2.0
        MeridionalVelocityAmplitude[1] = etaHat2*g*kY2*omega2/f0**2.0
    elif ProblemType == 'Barotropic_Tide':
        SurfaceElevationAmplitude[0] = etaHat1
        SurfaceElevationAmplitude[1] = etaHat2
        ZonalVelocityAmplitude[0] = etaHat1*g*omega1*kX1/(omega1**2.0 - f0**2.0)
        ZonalVelocityAmplitude[1] = etaHat2*g*omega2*kX2/(omega2**2.0 - f0**2.0)
        MeridionalVelocityAmplitude[0] = etaHat1*g*f0*kX1/(omega1**2.0 - f0**2.0)
        MeridionalVelocityAmplitude[1] = etaHat2*g*f0*kX2/(omega2**2.0 - f0**2.0)
    return SurfaceElevationAmplitude, ZonalVelocityAmplitude, MeridionalVelocityAmplitude


def DetermineExactSolutionLimits(ProblemType,myExactSolutionParameters):
    c0 = myExactSolutionParameters.c0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
    f0 = myExactSolutionParameters.f0
    g = myExactSolutionParameters.g
    H0 = myExactSolutionParameters.H0
    kX1 = myExactSolutionParameters.kX1
    kX2 = myExactSolutionParameters.kX2
    kY1 = myExactSolutionParameters.kY1
    kY2 = myExactSolutionParameters.kY2
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    s = myExactSolutionParameters.s
    uL = myExactSolutionParameters.uL
    uR = myExactSolutionParameters.uR
    ExactSurfaceElevationLimits = np.zeros(2)
    ExactZonalVelocityLimits = np.zeros(2)
    ExactMeridionalVelocityLimits = np.zeros(2)    
    if ProblemType == 'Plane_Gaussian_Wave':
        ExactSurfaceElevationLimits[1] = 1.0/g
        ExactZonalVelocityLimits[1] = kX1/c0
        ExactMeridionalVelocityLimits[1] = kY1/c0
    elif ProblemType == 'Coastal_Kelvin_Wave':
        ExactSurfaceElevationLimits[1] = abs(H0*etaHat1*float(DetermineKelvinWaveAmplitude()))
        ExactSurfaceElevationLimits[0] = -ExactSurfaceElevationLimits[1]
        ExactMeridionalVelocityLimits[1] = abs(c0*etaHat1*float(DetermineKelvinWaveAmplitude()))
        ExactMeridionalVelocityLimits[0] = -ExactMeridionalVelocityLimits[1]
    elif ProblemType == 'Inertia_Gravity_Wave':
        ExactSurfaceElevationLimits[1] = etaHat1 + etaHat2
        ExactSurfaceElevationLimits[0] = -ExactSurfaceElevationLimits[1]
        ExactZonalVelocityLimits[1] = (
        (abs(etaHat1*g/(omega1**2.0 - f0**2.0)*np.sqrt((omega1*kX1)**2.0 + (f0*kY1)**2.0))
         + abs(etaHat2*g/(omega2**2.0 - f0**2.0)*np.sqrt((omega2*kX2)**2.0 + (f0*kY2)**2.0))))
        ExactZonalVelocityLimits[0] = -ExactZonalVelocityLimits[1]
        ExactMeridionalVelocityLimits[1] = (
        (abs(etaHat1*g/(omega1**2.0 - f0**2.0)*np.sqrt((omega1*kY1)**2.0 + (f0*kX1)**2.0)) 
         + abs(etaHat2*g/(omega2**2.0 - f0**2.0)*np.sqrt((omega2*kY2)**2.0 + (f0*kX2)**2.0))))
        ExactMeridionalVelocityLimits[0] = -ExactMeridionalVelocityLimits[1]   
    elif ProblemType == 'Manufactured_Planetary_Rossby_Wave' or ProblemType == 'Manufactured_Topographic_Rossby_Wave':
        ExactSurfaceElevationLimits[1] = etaHat1 + etaHat2
        ExactSurfaceElevationLimits[0] = -ExactSurfaceElevationLimits[1]
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        ExactSurfaceElevationLimits[1] = abs(H0*etaHat1*float(DetermineKelvinWaveAmplitude()))
        ExactSurfaceElevationLimits[0] = -ExactSurfaceElevationLimits[1]
        ExactZonalVelocityLimits[1] = abs(c0*etaHat1*float(DetermineKelvinWaveAmplitude()))
        ExactZonalVelocityLimits[0] = -ExactZonalVelocityLimits[1]
    elif ProblemType == 'Barotropic_Tide':
        ExactSurfaceElevationLimits[1] = etaHat1 + etaHat2
        ExactSurfaceElevationLimits[0] = -ExactSurfaceElevationLimits[1]
        ExactZonalVelocityLimits[1] = (
        etaHat1*g*omega1*kX1/(omega1**2.0 - f0**2.0) + etaHat2*g*omega2*kX2/(omega2**2.0 - f0**2.0))
        ExactZonalVelocityLimits[0] = -ExactZonalVelocityLimits[1]
        ExactMeridionalVelocityLimits[1] = (
        etaHat1*g*f0*kX1/(omega1**2.0 - f0**2.0) + etaHat2*g*f0*kX2/(omega2**2.0 - f0**2.0))
        ExactMeridionalVelocityLimits[0] = -ExactMeridionalVelocityLimits[1]
    elif ProblemType == 'Diffusion_Equation':
        ExactZonalVelocityLimits[1] = 1.0
        ExactZonalVelocityLimits[0] = -ExactZonalVelocityLimits[1]
    elif ProblemType == 'Advection_Diffusion_Equation':
        ExactSurfaceElevationLimits[1] = 1.0
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactSurfaceElevationLimits[1] = etaHat1
        ExactSurfaceElevationLimits[0] = -ExactSurfaceElevationLimits[1]               
        ExactZonalVelocityLimits[1] = etaHat1
        ExactZonalVelocityLimits[0] = -ExactZonalVelocityLimits[1]
        ExactMeridionalVelocityLimits[1] = etaHat1
        ExactMeridionalVelocityLimits[0] = -ExactMeridionalVelocityLimits[1]
    elif ProblemType == 'Viscous_Burgers_Equation':
        ExactZonalVelocityLimits[1] = s + 0.5*(uL - uR)
        ExactZonalVelocityLimits[0] = s - 0.5*(uL - uR)
    return ExactSurfaceElevationLimits, ExactZonalVelocityLimits, ExactMeridionalVelocityLimits