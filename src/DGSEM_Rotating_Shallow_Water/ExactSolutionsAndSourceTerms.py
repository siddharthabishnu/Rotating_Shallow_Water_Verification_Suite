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


def DeterminePlanetaryRossbyWaveExactSurfaceElevation(etaHat,kX,kY,omega,x,y,time):
    eta = etaHat*np.sin(kX*x + kY*y - omega*time)
    return eta


def DeterminePlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat,f0,g,kX,kY,omega,x,y,time):
    u = -etaHat*g*(kY*(f0 - beta0*y)*np.cos(kX*x + kY*y - omega*time) 
                   + kX*omega*np.sin(kX*x + kY*y - omega*time))/f0**2.0
    return u


def DeterminePlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat,f0,g,kX,kY,omega,x,y,time):
    v = etaHat*g*(kX*(f0 - beta0*y)*np.cos(kX*x + kY*y - omega*time) 
                  - kY*omega*np.sin(kX*x + kY*y - omega*time))/f0**2.0
    return v


def DeterminePlanetaryRossbyWaveZonalVelocitySourceTerm(beta0,etaHat,f0,g,kX,kY,omega,x,y,time):
    uSourceTerm = etaHat*g*(2.0*kY*omega*beta0*y*np.sin(kX*x + kY*y - omega*time)
                            + kX*(omega**2.0 + (beta0*y)**2.0)*np.cos(kX*x + kY*y - omega*time))/f0**2.0
    return uSourceTerm


def DeterminePlanetaryRossbyWaveMeridionalVelocitySourceTerm(beta0,etaHat,f0,g,kX,kY,omega,x,y,time):
    vSourceTerm = etaHat*g*(-2.0*kX*omega*beta0*y*np.sin(kX*x + kY*y - omega*time)
                            + kY*(omega**2.0 + (beta0*y)**2.0)*np.cos(kX*x + kY*y - omega*time))/f0**2.0
    return vSourceTerm


def DetermineTopographicRossbyWaveExactSurfaceElevation(etaHat,kX,kY,omega,x,y,time):
    eta = etaHat*np.sin(kX*x + kY*y - omega*time)
    return eta


def DetermineTopographicRossbyWaveExactZonalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    u = -etaHat*g*(f0*kY*np.cos(kX*x + kY*y - omega*time) + omega*kX*np.sin(kX*x + kY*y - omega*time))/f0**2.0
    return u


def DetermineTopographicRossbyWaveExactMeridionalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    v = etaHat*g*(f0*kX*np.cos(kX*x + kY*y - omega*time) - omega*kY*np.sin(kX*x + kY*y - omega*time))/f0**2.0
    return v


def DetermineTopographicRossbyWaveZonalVelocitySourceTerm(etaHat,f0,g,kX,kY,omega,x,y,time):
    uSourceTerm = etaHat*g*kX*omega**2.0*np.cos(kX*x + kY*y - omega*time)/f0**2.0
    return uSourceTerm


def DetermineTopographicRossbyWaveMeridionalVelocitySourceTerm(etaHat,f0,g,kX,kY,omega,x,y,time):
    vSourceTerm = etaHat*g*kY*omega**2.0*np.cos(kX*x + kY*y - omega*time)/f0**2.0
    return vSourceTerm


def DetermineTopographicRossbyWaveSurfaceElevationSourceTerm(alpha0,etaHat,f0,g,kX,kY,omega,x,y,time):
    etaSourceTerm = -etaHat*g*alpha0*(kY*omega*np.sin(kX*x + kY*y - omega*time) 
                                      + (kX*f0 + (kX**2.0 + kY**2.0)*omega*y)*np.cos(kX*x + kY*y - omega*time))/f0**2.0
    return etaSourceTerm


def DetermineCoastalKelvinInertiaGravityRossbyWaveInitialSurfaceElevation(etaHat,R0x,R0y,x0,y0,x,y):
    eta = etaHat*np.exp(-((x - x0)**2.0/(2.0*R0x**2.0) + (y - y0)**2.0/(2.0*R0y**2.0)))
    return eta


def DetermineCoastalKelvinInertiaGravityRossbyWaveInitialZonalVelocity(etaHat,f0,g,R0x,R0y,x0,y0,x,y):
    u = g/(f0*R0y**2.0)*(y - y0)*etaHat*np.exp(-((x - x0)**2.0/(2.0*R0x**2.0) + (y - y0)**2.0/(2.0*R0y**2.0)))
    return u


def DetermineCoastalKelvinInertiaGravityRossbyWaveInitialMeridionalVelocity(etaHat,f0,g,R0x,R0y,x0,y0,x,y):
    v = -g/(f0*R0x**2.0)*(x - x0)*etaHat*np.exp(-((x - x0)**2.0/(2.0*R0x**2.0) + (y - y0)**2.0/(2.0*R0y**2.0)))
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
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    x0 = myExactSolutionParameters.x0
    y0 = myExactSolutionParameters.y0
    R0 = myExactSolutionParameters.R0
    R0x = myExactSolutionParameters.R0x
    R0y = myExactSolutionParameters.R0y
    R = myExactSolutionParameters.R
    Req = myExactSolutionParameters.Req
    LengthScale = myExactSolutionParameters.LengthScale
    TimeScale = myExactSolutionParameters.TimeScale
    SurfaceElevationScale = myExactSolutionParameters.SurfaceElevationScale
    if ProblemType == 'Plane_Gaussian_Wave':
        ExactSurfaceElevation = DeterminePlaneGaussianWaveExactSurfaceElevation(c0,g,kX1,kY1,R0,x0,y0,x,y,time)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        ExactSurfaceElevation = (DetermineCoastalKelvinWaveExactSurfaceElevation(c0,etaHat1,H0,kY1,R,x,y,time)
                                 + DetermineCoastalKelvinWaveExactSurfaceElevation(c0,etaHat2,H0,kY2,R,x,y,time))
    elif ProblemType == 'Inertia_Gravity_Wave':
        ExactSurfaceElevation = (DetermineInertiaGravityWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
                                 + DetermineInertiaGravityWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time))
    elif ProblemType == 'Planetary_Rossby_Wave':
        ExactSurfaceElevation = (DeterminePlanetaryRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
                                 + DeterminePlanetaryRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time))
    elif ProblemType == 'Topographic_Rossby_Wave':
        ExactSurfaceElevation = (DetermineTopographicRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
                                 + DetermineTopographicRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time))
    elif (ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave' 
          or ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave'):
        if time == 0.0:
            ExactSurfaceElevation = (
            DetermineCoastalKelvinInertiaGravityRossbyWaveInitialSurfaceElevation(etaHat1,R0x,R0y,x0,y0,x,y))
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
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactSurfaceElevation = (
        DetermineNonLinearManufacturedSolutionExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time))
    return ExactSurfaceElevation


def DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,x,y,time):
    beta0 = myExactSolutionParameters.beta0
    c0 = myExactSolutionParameters.c0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
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
    Req = myExactSolutionParameters.Req
    LengthScale = myExactSolutionParameters.LengthScale
    TimeScale = myExactSolutionParameters.TimeScale
    VelocityScale = myExactSolutionParameters.VelocityScale
    if ProblemType == 'Plane_Gaussian_Wave':
        ExactZonalVelocity = DeterminePlaneGaussianWaveExactZonalVelocity(c0,kX1,kY1,R0,x0,y0,x,y,time)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        ExactZonalVelocity = 0.0
    elif ProblemType == 'Inertia_Gravity_Wave':
        ExactZonalVelocity = (DetermineInertiaGravityWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
                              + DetermineInertiaGravityWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))
    elif ProblemType == 'Planetary_Rossby_Wave':
        ExactZonalVelocity = (DeterminePlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
                              + DeterminePlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,
                                                                               time))
    elif ProblemType == 'Topographic_Rossby_Wave':
        ExactZonalVelocity = (DetermineTopographicRossbyWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
                              + DetermineTopographicRossbyWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))
    elif (ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave' 
          or ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave'):
        if time == 0.0:
            ExactZonalVelocity = (
            DetermineCoastalKelvinInertiaGravityRossbyWaveInitialZonalVelocity(etaHat1,f0,g,R0x,R0y,x0,y0,x,y))
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
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactZonalVelocity = DetermineNonLinearManufacturedSolutionExactZonalVelocity(etaHat1,kX1,kY1,omega1,x,y,time)
    return ExactZonalVelocity


def DetermineExactMeridionalVelocity(ProblemType,myExactSolutionParameters,x,y,time):
    beta0 = myExactSolutionParameters.beta0
    c0 = myExactSolutionParameters.c0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
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
    if ProblemType == 'Plane_Gaussian_Wave':
        ExactMeridionalVelocity = DeterminePlaneGaussianWaveExactMeridionalVelocity(c0,kX1,kY1,R0,x0,y0,x,y,time)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        ExactMeridionalVelocity = (DetermineCoastalKelvinWaveExactMeridionalVelocity(c0,etaHat1,kY1,R,x,y,time)
                                   + DetermineCoastalKelvinWaveExactMeridionalVelocity(c0,etaHat2,kY2,R,x,y,time))
    if ProblemType == 'Inertia_Gravity_Wave':
        ExactMeridionalVelocity = (DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,
                                                                                      time)
                                   + DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,
                                                                                        time))
    elif ProblemType == 'Planetary_Rossby_Wave':
        ExactMeridionalVelocity = (DeterminePlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat1,f0,g,kX1,kY1,
                                                                                       omega1,x,y,time)
                                   + DeterminePlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat2,f0,g,kX2,kY2,
                                                                                         omega2,x,y,time))
    elif ProblemType == 'Topographic_Rossby_Wave':
        ExactMeridionalVelocity = (DetermineTopographicRossbyWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,
                                                                                         y,time)
                                   + DetermineTopographicRossbyWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,
                                                                                           x,y,time))
    elif (ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Planetary_Rossby_Wave' 
          or ProblemType == 'Coastal_Kelvin_Inertia_Gravity_Topographic_Rossby_Wave'):
        if time == 0.0:
            ExactMeridionalVelocity = (
            DetermineCoastalKelvinInertiaGravityRossbyWaveInitialMeridionalVelocity(etaHat1,f0,g,R0x,R0y,x0,y0,x,y))
        else:
            ExactMeridionalVelocity = 0.0
    elif ProblemType == 'Equatorial_Kelvin_Wave':
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
    return ExactMeridionalVelocity


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
    if ProblemType == 'Topographic_Rossby_Wave':
        SurfaceElevationSourceTerm = (
        (DetermineTopographicRossbyWaveSurfaceElevationSourceTerm(alpha0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineTopographicRossbyWaveSurfaceElevationSourceTerm(alpha0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
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
    if ProblemType == 'Planetary_Rossby_Wave':
        ZonalVelocitySourceTerm = (
        (DeterminePlanetaryRossbyWaveZonalVelocitySourceTerm(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DeterminePlanetaryRossbyWaveZonalVelocitySourceTerm(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'Topographic_Rossby_Wave':
        ZonalVelocitySourceTerm = (
        (DetermineTopographicRossbyWaveZonalVelocitySourceTerm(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineTopographicRossbyWaveZonalVelocitySourceTerm(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
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
    if ProblemType == 'Planetary_Rossby_Wave':
        MeridionalVelocitySourceTerm = (
        (DeterminePlanetaryRossbyWaveMeridionalVelocitySourceTerm(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DeterminePlanetaryRossbyWaveMeridionalVelocitySourceTerm(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    elif ProblemType == 'Topographic_Rossby_Wave':
        MeridionalVelocitySourceTerm = (
        (DetermineTopographicRossbyWaveMeridionalVelocitySourceTerm(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineTopographicRossbyWaveMeridionalVelocitySourceTerm(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
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
    SurfaceElevationAmplitude = np.zeros(2)
    ZonalVelocityAmplitude = np.zeros(2)
    MeridionalVelocityAmplitude = np.zeros(2)
    if ProblemType == 'Plane_Gaussian_Wave':
        SurfaceElevationAmplitude[0] = 1.0/g
        ZonalVelocityAmplitude[0] = kX1/c0
        MeridionalVelocityAmplitude[0] = kY1/c0
    elif ProblemType == 'Coastal_Kelvin_Wave':
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
    elif ProblemType == 'Planetary_Rossby_Wave':
        SurfaceElevationAmplitude[0] = etaHat1
        SurfaceElevationAmplitude[1] = etaHat2
        ZonalVelocityAmplitude[0] = etaHat1*g*np.sqrt((beta0*kY1*lY)**2.0 + (kX1*omega1)**2.0)/f0**2.0
        ZonalVelocityAmplitude[1] = etaHat2*g*np.sqrt((beta0*kY2*lY)**2.0 + (kX2*omega2)**2.0)/f0**2.0
        MeridionalVelocityAmplitude[0] = etaHat1*g*np.sqrt((beta0*kX1*lY)**2.0 + (kY1*omega1)**2.0)/f0**2.0
        MeridionalVelocityAmplitude[1] = etaHat2*g*np.sqrt((beta0*kX2*lY)**2.0 + (kY2*omega1)**2.0)/f0**2.0
    elif ProblemType == 'Topographic_Rossby_Wave':
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
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        SurfaceElevationAmplitude[0] = etaHat1
        ZonalVelocityAmplitude[0] = etaHat1
        MeridionalVelocityAmplitude[0] = etaHat1
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
    ExactSurfaceElevationLimits = np.zeros(2)
    ExactZonalVelocityLimits = np.zeros(2)
    ExactMeridionalVelocityLimits = np.zeros(2)    
    if ProblemType == 'Plane_Gaussian_Wave':
        ExactSurfaceElevationLimits[1] = 1.0/g
        ExactSurfaceElevationLimits[0] = -ExactSurfaceElevationLimits[1]
        ExactZonalVelocityLimits[1] = kX1/c0
        ExactZonalVelocityLimits[0] = -ExactZonalVelocityLimits[1]
        ExactMeridionalVelocityLimits[1] = kY1/c0
        ExactMeridionalVelocityLimits[0] = -ExactMeridionalVelocityLimits[1]
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
    elif ProblemType == 'Planetary_Rossby_Wave' or ProblemType == 'Topographic_Rossby_Wave':
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
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactSurfaceElevationLimits[1] = etaHat1
        ExactSurfaceElevationLimits[0] = -ExactSurfaceElevationLimits[1]               
        ExactZonalVelocityLimits[1] = etaHat1
        ExactZonalVelocityLimits[0] = -ExactZonalVelocityLimits[1]
        ExactMeridionalVelocityLimits[1] = etaHat1
        ExactMeridionalVelocityLimits[0] = -ExactMeridionalVelocityLimits[1]
    return ExactSurfaceElevationLimits, ExactZonalVelocityLimits, ExactMeridionalVelocityLimits