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
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os


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


def DetermineBarotropicTideExactSurfaceElevation(etaHat,kX,omega,x,time):
    eta = etaHat*np.cos(kX*x)*np.cos(omega*time)
    # Note that eta = 0.5*etaHat(np.cos(kX*x + omega*t) + np.cos(kX*x - omega*t))
    return eta


def DetermineBarotropicTideExactSurfaceElevations_ComponentWaves(etaHat,kX,omega,x,time):
    eta1 = 0.5*etaHat*np.cos(kX*x + omega*time)
    eta2 = 0.5*etaHat*np.cos(kX*x - omega*time)
    return eta1, eta2


def DetermineBarotropicTideExactZonalVelocity(etaHat,f0,g,kX,omega,x,time):
    u = etaHat*g*omega*kX*np.sin(kX*x)*np.sin(omega*time)/(omega**2.0 - f0**2.0)
    return u


def DetermineBarotropicTideExactMeridionalVelocity(etaHat,f0,g,kX,omega,x,time):
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
    etaHat*(-H0*(kX + kY)*np.sin(phase) - omega*np.cos(phase) + etaHat*(kX + kY)*np.cos(2.0*phase)))
    return etaSourceTerm


def DetermineNonLinearManufacturedSolutionZonalVelocitySourceTerm(etaHat,f0,g,kX,kY,omega,x,y,time):
    phase = kX*x + kY*y - omega*time
    uSourceTerm = etaHat*((-f0 + g*kX)*np.cos(phase) + omega*np.sin(phase) - 0.5*etaHat*(kX + kY)*np.sin(2.0*(phase)))
    return uSourceTerm


def DetermineNonLinearManufacturedSolutionMeridionalVelocitySourceTerm(etaHat,f0,g,kX,kY,omega,x,y,time):
    phase = kX*x + kY*y - omega*time
    vSourceTerm = etaHat*((f0 + g*kY)*np.cos(phase) + omega*np.sin(phase) - 0.5*etaHat*(kX + kY)*np.sin(2.0*(phase)))
    return vSourceTerm


def DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                   myQuadratureOnHexagon=[],HexagonLength=1.0):
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
    if UseAveragedQuantities:
        x += myQuadratureOnHexagon.x[:]*HexagonLength
        y += myQuadratureOnHexagon.y[:]*HexagonLength
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
            return ExactSurfaceElevation
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
        ExactSurfaceElevation = (DetermineBarotropicTideExactSurfaceElevation(etaHat1,kX1,omega1,x,time)
                                 + DetermineBarotropicTideExactSurfaceElevation(etaHat2,kX2,omega2,x,time))
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactSurfaceElevation = (
        DetermineNonLinearManufacturedSolutionExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time))
    if UseAveragedQuantities:
        ExactSurfaceElevation = np.dot(ExactSurfaceElevation,myQuadratureOnHexagon.w)
    return ExactSurfaceElevation


def DetermineExactSurfaceElevations(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                    myQuadratureOnHexagon=[],HexagonLength=1.0):
    c0 = myExactSolutionParameters.c0
    etaHat1 = myExactSolutionParameters.etaHat1
    etaHat2 = myExactSolutionParameters.etaHat2
    H0 = myExactSolutionParameters.H0
    kX1 = myExactSolutionParameters.kX1
    kX2 = myExactSolutionParameters.kX2
    kY1 = myExactSolutionParameters.kY1
    kY2 = myExactSolutionParameters.kY2
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    omega1 = myExactSolutionParameters.omega1
    omega2 = myExactSolutionParameters.omega2
    R = myExactSolutionParameters.R
    Req = myExactSolutionParameters.Req
    LengthScale = myExactSolutionParameters.LengthScale
    TimeScale = myExactSolutionParameters.TimeScale
    SurfaceElevationScale = myExactSolutionParameters.SurfaceElevationScale
    if UseAveragedQuantities:
        x += myQuadratureOnHexagon.x[:]*HexagonLength
        y += myQuadratureOnHexagon.y[:]*HexagonLength
    if ProblemType == 'Barotropic_Tide':
        ExactSurfaceElevations = np.zeros(7)
    else:
        ExactSurfaceElevations = np.zeros(3)
    if ProblemType == 'Coastal_Kelvin_Wave':
        ExactSurfaceElevation_1 = DetermineCoastalKelvinWaveExactSurfaceElevation(c0,etaHat1,H0,kY1,R,x,y,time)
        ExactSurfaceElevation_2 = DetermineCoastalKelvinWaveExactSurfaceElevation(c0,etaHat2,H0,kY2,R,x,y,time)
    elif ProblemType == 'Inertia_Gravity_Wave':
        ExactSurfaceElevation_1 = DetermineInertiaGravityWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
        ExactSurfaceElevation_2 = DetermineInertiaGravityWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time)
    elif ProblemType == 'Planetary_Rossby_Wave':
        ExactSurfaceElevation_1 = DeterminePlanetaryRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
        ExactSurfaceElevation_2 = DeterminePlanetaryRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time)
    elif ProblemType == 'Topographic_Rossby_Wave':
        ExactSurfaceElevation_1 = DetermineTopographicRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
        ExactSurfaceElevation_2 = DetermineTopographicRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time)
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        ExactSurfaceElevation_1 = DetermineEquatorialKelvinWaveExactSurfaceElevation(c0,etaHat1,H0,kX1,Req,x,y,time)
        ExactSurfaceElevation_2 = DetermineEquatorialKelvinWaveExactSurfaceElevation(c0,etaHat2,H0,kX2,Req,x,y,time)
    elif ProblemType == 'Equatorial_Yanai_Wave':
        ExactSurfaceElevation_1 = (
        DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                          SurfaceElevationScale,x,y,time))
        ExactSurfaceElevation_2 = (
        DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                          SurfaceElevationScale,x,y,time))
    elif ProblemType == 'Equatorial_Rossby_Wave':
        ExactSurfaceElevation_1 = (
        DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                           SurfaceElevationScale,x,y,time))
        ExactSurfaceElevation_2 = (
        DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                           SurfaceElevationScale,x,y,time))
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        ExactSurfaceElevation_1 = (
        DetermineEquatorialInertiaGravityWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                   SurfaceElevationScale,x,y,time))
        ExactSurfaceElevation_2 = (
        DetermineEquatorialInertiaGravityWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                   SurfaceElevationScale,x,y,time))
    elif ProblemType == 'Barotropic_Tide':
        ExactSurfaceElevation_1, ExactSurfaceElevation_2 = (
        DetermineBarotropicTideExactSurfaceElevations_ComponentWaves(etaHat1,kX1,omega1,x,time))
        ExactSurfaceElevation_3 = DetermineBarotropicTideExactSurfaceElevation(etaHat1,kX1,omega1,x,time)
        ExactSurfaceElevation_4, ExactSurfaceElevation_5 = (
        DetermineBarotropicTideExactSurfaceElevations_ComponentWaves(etaHat2,kX2,omega2,x,time))
        ExactSurfaceElevation_6 = DetermineBarotropicTideExactSurfaceElevation(etaHat2,kX2,omega2,x,time)
    if UseAveragedQuantities:
        if ProblemType == 'Barotropic_Tide':
            ExactSurfaceElevations[0] = np.dot(ExactSurfaceElevation_1,myQuadratureOnHexagon.w)
            ExactSurfaceElevations[1] = np.dot(ExactSurfaceElevation_2,myQuadratureOnHexagon.w)
            ExactSurfaceElevations[2] = np.dot(ExactSurfaceElevation_3,myQuadratureOnHexagon.w)
            ExactSurfaceElevations[3] = np.dot(ExactSurfaceElevation_4,myQuadratureOnHexagon.w)
            ExactSurfaceElevations[4] = np.dot(ExactSurfaceElevation_5,myQuadratureOnHexagon.w)
            ExactSurfaceElevations[5] = np.dot(ExactSurfaceElevation_6,myQuadratureOnHexagon.w)
            ExactSurfaceElevations[6] = ExactSurfaceElevations[2] + ExactSurfaceElevations[5]
        else:
            ExactSurfaceElevations[0] = np.dot(ExactSurfaceElevation_1,myQuadratureOnHexagon.w)
            ExactSurfaceElevations[1] = np.dot(ExactSurfaceElevation_2,myQuadratureOnHexagon.w)
            ExactSurfaceElevations[2] = ExactSurfaceElevations[0] + ExactSurfaceElevations[1]
    else:
        if ProblemType == 'Barotropic_Tide':
            ExactSurfaceElevations[0] = ExactSurfaceElevation_1
            ExactSurfaceElevations[1] = ExactSurfaceElevation_2
            ExactSurfaceElevations[2] = ExactSurfaceElevation_3
            ExactSurfaceElevations[3] = ExactSurfaceElevation_4
            ExactSurfaceElevations[4] = ExactSurfaceElevation_5
            ExactSurfaceElevations[5] = ExactSurfaceElevation_6
            ExactSurfaceElevations[6] = ExactSurfaceElevation_3 + ExactSurfaceElevation_6
        else:
            ExactSurfaceElevations[0] = ExactSurfaceElevation_1
            ExactSurfaceElevations[1] = ExactSurfaceElevation_2
            ExactSurfaceElevations[2] = ExactSurfaceElevation_1 + ExactSurfaceElevation_2
    return ExactSurfaceElevations


def DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                myQuadratureOnEdge=[],dvEdge=1.0,angleEdge=0.0):
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
    if UseAveragedQuantities:
        x += 0.5*dvEdge*np.cos(angleEdge)*myQuadratureOnEdge.x[:]
        y += 0.5*dvEdge*np.sin(angleEdge)*myQuadratureOnEdge.x[:]
    if ProblemType == 'Plane_Gaussian_Wave':
        ExactZonalVelocity = DeterminePlaneGaussianWaveExactZonalVelocity(c0,kX1,kY1,R0,x0,y0,x,y,time)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        ExactZonalVelocity = 0.0
        return ExactZonalVelocity
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
            return ExactZonalVelocity
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
        ExactZonalVelocity = (DetermineBarotropicTideExactZonalVelocity(etaHat1,f0,g,kX1,omega1,x,time)
                              + DetermineBarotropicTideExactZonalVelocity(etaHat2,f0,g,kX2,omega2,x,time))
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactZonalVelocity = DetermineNonLinearManufacturedSolutionExactZonalVelocity(etaHat1,kX1,kY1,omega1,x,y,time)
    if UseAveragedQuantities:
        ExactZonalVelocity = np.dot(0.5*dvEdge*ExactZonalVelocity,myQuadratureOnEdge.w)/dvEdge
    return ExactZonalVelocity


def DetermineExactZonalVelocities(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                  myQuadratureOnEdge=[],dvEdge=1.0,angleEdge=0.0):
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
    Req = myExactSolutionParameters.Req
    LengthScale = myExactSolutionParameters.LengthScale
    TimeScale = myExactSolutionParameters.TimeScale
    VelocityScale = myExactSolutionParameters.VelocityScale
    if UseAveragedQuantities:
        x += 0.5*dvEdge*np.cos(angleEdge)*myQuadratureOnEdge.x[:]
        y += 0.5*dvEdge*np.sin(angleEdge)*myQuadratureOnEdge.x[:]
    ExactZonalVelocities = np.zeros(3)
    if ProblemType == 'Coastal_Kelvin_Wave':
        return ExactZonalVelocities
    elif ProblemType == 'Inertia_Gravity_Wave':
        ExactZonalVelocity_1 = DetermineInertiaGravityWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
        ExactZonalVelocity_2 = DetermineInertiaGravityWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)
    elif ProblemType == 'Planetary_Rossby_Wave':
        ExactZonalVelocity_1 = DeterminePlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,
                                                                              time)
        ExactZonalVelocity_2 = DeterminePlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,
                                                                              time)
    elif ProblemType == 'Topographic_Rossby_Wave':
        ExactZonalVelocity_1 = DetermineTopographicRossbyWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
        ExactZonalVelocity_2 = DetermineTopographicRossbyWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        ExactZonalVelocity_1 = DetermineEquatorialKelvinWaveExactZonalVelocity(c0,etaHat1,kX1,Req,x,y,time)
        ExactZonalVelocity_2 = DetermineEquatorialKelvinWaveExactZonalVelocity(c0,etaHat2,kX2,Req,x,y,time)
    elif ProblemType == 'Equatorial_Yanai_Wave':
        ExactZonalVelocity_1 = DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                              VelocityScale,x,y,time)
        ExactZonalVelocity_2 = DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                              VelocityScale,x,y,time)
    elif ProblemType == 'Equatorial_Rossby_Wave':
        ExactZonalVelocity_1 = DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                               VelocityScale,x,y,time)
        ExactZonalVelocity_2 = DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                               VelocityScale,x,y,time)
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        ExactZonalVelocity_1 = (
        DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,
                                                                x,y,time))
        ExactZonalVelocity_2 = (
        DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,
                                                                x,y,time))
    elif ProblemType == 'Barotropic_Tide':
        ExactZonalVelocity_1 = DetermineBarotropicTideExactZonalVelocity(etaHat1,f0,g,kX1,omega1,x,time)
        ExactZonalVelocity_2 = DetermineBarotropicTideExactZonalVelocity(etaHat2,f0,g,kX2,omega2,x,time)
    if UseAveragedQuantities:
        ExactZonalVelocities[0] = np.dot(0.5*dvEdge*ExactZonalVelocity_1,myQuadratureOnEdge.w)/dvEdge
        ExactZonalVelocities[1] = np.dot(0.5*dvEdge*ExactZonalVelocity_2,myQuadratureOnEdge.w)/dvEdge
        ExactZonalVelocities[2] = ExactZonalVelocities[0] + ExactZonalVelocities[1]
    else:
        ExactZonalVelocities[0] = ExactZonalVelocity_1
        ExactZonalVelocities[1] = ExactZonalVelocity_2
        ExactZonalVelocities[2] = ExactZonalVelocity_1 + ExactZonalVelocity_2
    return ExactZonalVelocities


def DetermineExactZonalVelocityAtCellCenter(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                            myQuadratureOnHexagon=[],HexagonLength=1.0):
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
    if UseAveragedQuantities:
        x += myQuadratureOnHexagon.x[:]*HexagonLength
        y += myQuadratureOnHexagon.y[:]*HexagonLength
    if ProblemType == 'Plane_Gaussian_Wave':
        ExactZonalVelocity = DeterminePlaneGaussianWaveExactZonalVelocity(c0,kX1,kY1,R0,x0,y0,x,y,time)
    elif ProblemType == 'Coastal_Kelvin_Wave':
        ExactZonalVelocity = 0.0
        return ExactZonalVelocity
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
            return ExactZonalVelocity
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
        ExactZonalVelocity = (DetermineBarotropicTideExactZonalVelocity(etaHat1,f0,g,kX1,omega1,x,time)
                              + DetermineBarotropicTideExactZonalVelocity(etaHat2,f0,g,kX2,omega2,x,time))
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactZonalVelocity = DetermineNonLinearManufacturedSolutionExactZonalVelocity(etaHat1,kX1,kY1,omega1,x,y,time)
    if UseAveragedQuantities:
        ExactZonalVelocity = np.dot(ExactZonalVelocity,myQuadratureOnHexagon.w)
    return ExactZonalVelocity


def DetermineExactMeridionalVelocity(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                     myQuadratureOnEdge=[],dvEdge=1.0,angleEdge=0.0):
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
    if UseAveragedQuantities:
        x += 0.5*dvEdge*np.cos(angleEdge)*myQuadratureOnEdge.x[:]
        y += 0.5*dvEdge*np.sin(angleEdge)*myQuadratureOnEdge.x[:]
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
            return ExactMeridionalVelocity
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        ExactMeridionalVelocity = 0.0
        return ExactMeridionalVelocity
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
        ExactMeridionalVelocity = (DetermineBarotropicTideExactMeridionalVelocity(etaHat1,f0,g,kX1,omega1,x,time)
                                   + DetermineBarotropicTideExactMeridionalVelocity(etaHat2,f0,g,kX2,omega2,x,time))
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactMeridionalVelocity = DetermineNonLinearManufacturedSolutionExactMeridionalVelocity(etaHat1,kX1,kY1,omega1,
                                                                                                x,y,time)
    if UseAveragedQuantities:
        ExactMeridionalVelocity = np.dot(0.5*dvEdge*ExactMeridionalVelocity,myQuadratureOnEdge.w)/dvEdge
    return ExactMeridionalVelocity


def DetermineExactMeridionalVelocities(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                       myQuadratureOnEdge=[],dvEdge=1.0,angleEdge=0.0):
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
    R = myExactSolutionParameters.R
    LengthScale = myExactSolutionParameters.LengthScale
    TimeScale = myExactSolutionParameters.TimeScale
    VelocityScale = myExactSolutionParameters.VelocityScale  
    if UseAveragedQuantities:
        x += 0.5*dvEdge*np.cos(angleEdge)*myQuadratureOnEdge.x[:]
        y += 0.5*dvEdge*np.sin(angleEdge)*myQuadratureOnEdge.x[:]
    ExactMeridionalVelocities = np.zeros(3)
    if ProblemType == 'Coastal_Kelvin_Wave':
        ExactMeridionalVelocity_1 = DetermineCoastalKelvinWaveExactMeridionalVelocity(c0,etaHat1,kY1,R,x,y,time)
        ExactMeridionalVelocity_2 = DetermineCoastalKelvinWaveExactMeridionalVelocity(c0,etaHat2,kY2,R,x,y,time)
    if ProblemType == 'Inertia_Gravity_Wave':
        ExactMeridionalVelocity_1 = DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,
                                                                                       time)
        ExactMeridionalVelocity_2 = DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,
                                                                                       time)
    elif ProblemType == 'Planetary_Rossby_Wave':
        ExactMeridionalVelocity_1 = DeterminePlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat1,f0,g,kX1,kY1,
                                                                                        omega1,x,y,time)
        ExactMeridionalVelocity_2 = DeterminePlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat2,f0,g,kX2,kY2,
                                                                                        omega2,x,y,time)
    elif ProblemType == 'Topographic_Rossby_Wave':
        ExactMeridionalVelocity_1 = DetermineTopographicRossbyWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,
                                                                                          y,time)
        ExactMeridionalVelocity_2 = DetermineTopographicRossbyWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,
                                                                                          y,time)
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        return ExactMeridionalVelocities
    elif ProblemType == 'Equatorial_Yanai_Wave':
        ExactMeridionalVelocity_1 = (
        DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,x,y,
                                                            time))
        ExactMeridionalVelocity_2 = (
        DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,x,y,
                                                            time))
    elif ProblemType == 'Equatorial_Rossby_Wave':
        ExactMeridionalVelocity_1 = (
        DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,x,y,
                                                             time))
        ExactMeridionalVelocity_2 = (
        DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,x,y,
                                                             time))
    elif ProblemType == 'Equatorial_Inertia_Gravity_Wave':
        ExactMeridionalVelocity_1 = (
        DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                     VelocityScale,x,y,time))
        ExactMeridionalVelocity_2 = (
        DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                     VelocityScale,x,y,time))
    elif ProblemType == 'Barotropic_Tide':
        ExactMeridionalVelocity_1 = DetermineBarotropicTideExactMeridionalVelocity(etaHat1,f0,g,kX1,omega1,x,time)
        ExactMeridionalVelocity_2 = DetermineBarotropicTideExactMeridionalVelocity(etaHat2,f0,g,kX2,omega2,x,time)
    if UseAveragedQuantities:
        ExactMeridionalVelocities[0] = np.dot(0.5*dvEdge*ExactMeridionalVelocity_1,myQuadratureOnEdge.w)/dvEdge
        ExactMeridionalVelocities[1] = np.dot(0.5*dvEdge*ExactMeridionalVelocity_2,myQuadratureOnEdge.w)/dvEdge
        ExactMeridionalVelocities[2] = ExactMeridionalVelocities[0] + ExactMeridionalVelocities[1]
    else:
        ExactMeridionalVelocities[0] = ExactMeridionalVelocity_1
        ExactMeridionalVelocities[1] = ExactMeridionalVelocity_2
        ExactMeridionalVelocities[2] = ExactMeridionalVelocity_1 + ExactMeridionalVelocity_2
    return ExactMeridionalVelocities


def DetermineExactMeridionalVelocityAtCellCenter(ProblemType,myExactSolutionParameters,x,y,time,
                                                 UseAveragedQuantities=False,myQuadratureOnHexagon=[],
                                                 HexagonLength=1.0):
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
    if UseAveragedQuantities:
        x += myQuadratureOnHexagon.x[:]*HexagonLength
        y += myQuadratureOnHexagon.y[:]*HexagonLength
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
            return ExactMeridionalVelocity
    elif ProblemType == 'Equatorial_Kelvin_Wave':
        ExactMeridionalVelocity = 0.0
        return ExactMeridionalVelocity
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
        ExactMeridionalVelocity = (DetermineBarotropicTideExactMeridionalVelocity(etaHat1,f0,g,kX1,omega1,x,time)
                                   + DetermineBarotropicTideExactMeridionalVelocity(etaHat2,f0,g,kX2,omega2,x,time))
    elif ProblemType == 'NonLinear_Manufactured_Solution':
        ExactMeridionalVelocity = DetermineNonLinearManufacturedSolutionExactMeridionalVelocity(etaHat1,kX1,kY1,omega1,
                                                                                                x,y,time)
    if UseAveragedQuantities:
        ExactMeridionalVelocity = np.dot(ExactMeridionalVelocity,myQuadratureOnHexagon.w)
    return ExactMeridionalVelocity


def DetermineExactNormalVelocity(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                 myQuadratureOnEdge=[],dvEdge=1.0,angleEdge=0.0):
    ExactZonalVelocity = DetermineExactZonalVelocity(ProblemType,myExactSolutionParameters,x,y,time,
                                                     UseAveragedQuantities,myQuadratureOnEdge,dvEdge,angleEdge)
    ExactMeridionalVelocity = DetermineExactMeridionalVelocity(ProblemType,myExactSolutionParameters,x,y,time,
                                                               UseAveragedQuantities,myQuadratureOnEdge,dvEdge,
                                                               angleEdge)
    ExactNormalVelocity = ExactZonalVelocity*np.cos(angleEdge) + ExactMeridionalVelocity*np.sin(angleEdge)
    return ExactNormalVelocity


def DetermineSurfaceElevationSourceTerm(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                        myQuadratureOnHexagon=[],HexagonLength=1.0):
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
    if UseAveragedQuantities:
        x += myQuadratureOnHexagon.x[:]*HexagonLength
        y += myQuadratureOnHexagon.y[:]*HexagonLength
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
    if UseAveragedQuantities:
        SurfaceElevationSourceTerm = np.dot(SurfaceElevationSourceTerm,myQuadratureOnHexagon.w)
    return SurfaceElevationSourceTerm


def DetermineZonalVelocitySourceTerm(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                     myQuadratureOnEdge=[],dvEdge=1.0,angleEdge=0.0):
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
    if UseAveragedQuantities:
        x += 0.5*dvEdge*np.cos(angleEdge)*myQuadratureOnEdge.x[:]
        y += 0.5*dvEdge*np.sin(angleEdge)*myQuadratureOnEdge.x[:]
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
        DetermineNonLinearManufacturedSolutionZonalVelocitySourceTerm(etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
    else:
        ZonalVelocitySourceTerm = 0.0
        return ZonalVelocitySourceTerm
    if UseAveragedQuantities:
        ZonalVelocitySourceTerm = np.dot(0.5*dvEdge*ZonalVelocitySourceTerm,myQuadratureOnEdge.w)/dvEdge
    return ZonalVelocitySourceTerm


def DetermineMeridionalVelocitySourceTerm(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                          myQuadratureOnEdge=[],dvEdge=1.0,angleEdge=0.0):
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
    if UseAveragedQuantities:
        x += 0.5*dvEdge*np.cos(angleEdge)*myQuadratureOnEdge.x[:]
        y += 0.5*dvEdge*np.sin(angleEdge)*myQuadratureOnEdge.x[:]
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
        DetermineNonLinearManufacturedSolutionMeridionalVelocitySourceTerm(etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
    else:
        MeridionalVelocitySourceTerm = 0.0
        return MeridionalVelocitySourceTerm
    if UseAveragedQuantities:
        MeridionalVelocitySourceTerm = np.dot(0.5*dvEdge*MeridionalVelocitySourceTerm,myQuadratureOnEdge.w)/dvEdge
    return MeridionalVelocitySourceTerm


def DetermineNormalVelocitySourceTerm(ProblemType,myExactSolutionParameters,x,y,time,UseAveragedQuantities=False,
                                      myQuadratureOnEdge=[],dvEdge=1.0,angleEdge=0.0):
    ZonalVelocitySourceTerm = DetermineZonalVelocitySourceTerm(ProblemType,myExactSolutionParameters,x,y,time,
                                                               UseAveragedQuantities,myQuadratureOnEdge,dvEdge,
                                                               angleEdge)
    MeridionalVelocitySourceTerm = DetermineMeridionalVelocitySourceTerm(ProblemType,myExactSolutionParameters,x,y,time,
                                                                         UseAveragedQuantities,myQuadratureOnEdge,
                                                                         dvEdge,angleEdge)
    NormalVelocitySourceTerm = (ZonalVelocitySourceTerm*np.cos(angleEdge) 
                                + MeridionalVelocitySourceTerm*np.sin(angleEdge))
    return NormalVelocitySourceTerm


def DetermineNormalComponentFromZonalAndMeridionalComponents(ZonalComponent,MeridionalComponent,angleEdge):
    NormalComponent = ZonalComponent*np.cos(angleEdge) + MeridionalComponent*np.sin(angleEdge)
    return NormalComponent


def DetermineNormalComponentsFromZonalAndMeridionalComponents(ZonalComponents,MeridionalComponents,angleEdges):
    nEdges = len(angleEdges)
    NormalComponents = np.zeros(nEdges)
    for iEdge in range(0,nEdges):
        NormalComponents[iEdge] = (ZonalComponents[iEdge]*np.cos(angleEdges[iEdge]) 
                                   + MeridionalComponents[iEdge]*np.sin(angleEdges[iEdge]))
    return NormalComponents


def DetermineTangentialComponentFromZonalAndMeridionalComponents(ZonalComponent,MeridionalComponent,angleEdge):
    TangentialComponent = MeridionalComponent*np.cos(angleEdge) - ZonalComponent*np.sin(angleEdge)
    return TangentialComponent


def DetermineTangentialComponentsFromZonalAndMeridionalComponents(ZonalComponents,MeridionalComponents,angleEdges):
    nEdges = len(angleEdges)
    TangentialComponents = np.zeros(nEdges)
    for iEdge in range(0,nEdges):
        TangentialComponents[iEdge] = (MeridionalComponents[iEdge]*np.cos(angleEdges[iEdge]) 
                                       - ZonalComponents[iEdge]*np.sin(angleEdges[iEdge]))
    return TangentialComponents


def DetermineZonalComponentFromNormalAndTangentialComponents(NormalComponent,TangentialComponent,angleEdge):
    ZonalComponent = NormalComponent*np.cos(angleEdge) - TangentialComponent*np.sin(angleEdge)
    return ZonalComponent


def DetermineZonalComponentsFromNormalAndTangentialComponents(NormalComponents,TangentialComponents,angleEdges):
    nEdges = len(angleEdges)
    ZonalComponents = np.zeros(nEdges)
    for iEdge in range(0,nEdges):
        ZonalComponents[iEdge] = (NormalComponents[iEdge]*np.cos(angleEdges[iEdge]) 
                                  - TangentialComponents[iEdge]*np.sin(angleEdges[iEdge]))
    return ZonalComponents


def DetermineMeridionalComponentFromNormalAndTangentialComponents(NormalComponent,TangentialComponent,angleEdge):
    MeridionalComponent = NormalComponent*np.sin(angleEdge) + TangentialComponent*np.cos(angleEdge)
    return MeridionalComponent


def DetermineMeridionalComponentsFromNormalAndTangentialComponents(NormalComponents,TangentialComponents,angleEdges):
    nEdges = len(angleEdges)
    MeridionalComponents = np.zeros(nEdges)
    for iEdge in range(0,nEdges):
        MeridionalComponents[iEdge] = (NormalComponents[iEdge]*np.sin(angleEdges[iEdge]) 
                                       + TangentialComponents[iEdge]*np.cos(angleEdges[iEdge]))
    return MeridionalComponents


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


def DetermineCoastalCoordinates(lY,nPlotAlongCoastline):
    xPlotAlongCoastline = np.zeros(nPlotAlongCoastline+1)
    yPlotAlongCoastline = np.linspace(0.0,lY,nPlotAlongCoastline+1)
    rPlotAlongCoastline = yPlotAlongCoastline
    return rPlotAlongCoastline, xPlotAlongCoastline, yPlotAlongCoastline


def DetermineDiagonalCoordinates(DiagonalType,lX,lY,nPlotAlongDiagonal):
    l = np.sqrt(lX**2.0 + lY**2.0)
    rPlotAlongDiagonal = np.linspace(0.0,l,nPlotAlongDiagonal+1)
    theta = np.arctan(lY/lX)
    xPlotAlongDiagonal = rPlotAlongDiagonal*np.cos(theta)
    if DiagonalType == 'SouthWest-NorthEast':
        yPlotAlongDiagonal = rPlotAlongDiagonal*np.sin(theta)
    elif DiagonalType == 'NorthWest-SouthEast':
        yPlotAlongDiagonal = lY - rPlotAlongDiagonal*np.sin(theta)
    return rPlotAlongDiagonal, xPlotAlongDiagonal, yPlotAlongDiagonal


def DetermineCoordinatesAlongZonalSection(lX,nPlotAlongZonalSection,y=0.0):
    xPlotAlongZonalSection = np.linspace(0.0,lX,nPlotAlongZonalSection+1)
    yPlotAlongZonalSection = y*np.ones(nPlotAlongZonalSection+1)
    rPlotAlongZonalSection = xPlotAlongZonalSection
    return rPlotAlongZonalSection, xPlotAlongZonalSection, yPlotAlongZonalSection


def DetermineCoordinatesAlongSection(ProblemType,lX,lY,nPlotAlongSection,y=0.0):
    if ProblemType == 'Coastal_Kelvin_Wave':
        rPlotAlongSection, xPlotAlongSection, yPlotAlongSection = (
        DetermineCoastalCoordinates(lY,nPlotAlongSection))
    elif (ProblemType == 'Inertia_Gravity_Wave' or ProblemType == 'Planetary_Rossby_Wave' 
          or ProblemType == 'Topographic_Rossby_Wave' or ProblemType == 'NonLinear_Manufactured_Solution'):    
        DiagonalType = 'SouthWest-NorthEast'
        rPlotAlongSection, xPlotAlongSection, yPlotAlongSection = (
        DetermineDiagonalCoordinates(DiagonalType,lX,lY,nPlotAlongSection))        
    elif (ProblemType == 'Equatorial_Kelvin_Wave' or ProblemType == 'Equatorial_Yanai_Wave' 
          or ProblemType == 'Equatorial_Rossby_Wave' or ProblemType == 'Equatorial_Inertia_Gravity_Wave'
          or ProblemType == 'Barotropic_Tide'):
        rPlotAlongSection, xPlotAlongSection, yPlotAlongSection = (
        DetermineCoordinatesAlongZonalSection(lX,nPlotAlongSection,y))
    return rPlotAlongSection, xPlotAlongSection, yPlotAlongSection


def DetermineExactStateVariablesAlongSection(ProblemType,myExactSolutionParameters,xPlotAlongSection,yPlotAlongSection,
                                             time):
    nPlotAlongSection = len(xPlotAlongSection) - 1    
    if ProblemType == 'NonLinear_Manufactured_Solution':
        ExactStateVariablesAlongSection = np.zeros(nPlotAlongSection+1)
        for iPlotAlongSection in range(0,nPlotAlongSection+1): 
            ExactStateVariablesAlongSection[iPlotAlongSection] = (
            DetermineExactSurfaceElevation(ProblemType,myExactSolutionParameters,xPlotAlongSection[iPlotAlongSection],
                                           yPlotAlongSection[iPlotAlongSection],time))
    elif ProblemType == 'Barotropic_Tide':
        ExactStateVariablesAlongSection = np.zeros((7,nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1): 
            [ExactStateVariablesAlongSection[0,iPlotAlongSection],
             ExactStateVariablesAlongSection[1,iPlotAlongSection],
             ExactStateVariablesAlongSection[2,iPlotAlongSection],
             ExactStateVariablesAlongSection[3,iPlotAlongSection],
             ExactStateVariablesAlongSection[4,iPlotAlongSection],
             ExactStateVariablesAlongSection[5,iPlotAlongSection],
             ExactStateVariablesAlongSection[6,iPlotAlongSection]] = (
            DetermineExactSurfaceElevations(ProblemType,myExactSolutionParameters,xPlotAlongSection[iPlotAlongSection],
                                            yPlotAlongSection[iPlotAlongSection],time))
    else:
        ExactStateVariablesAlongSection = np.zeros((3,nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            if (ProblemType == 'Equatorial_Yanai_Wave' or ProblemType == 'Equatorial_Rossby_Wave' 
                or ProblemType == 'Equatorial_Inertia_Gravity_Wave'):
                [ExactStateVariablesAlongSection[0,iPlotAlongSection],
                 ExactStateVariablesAlongSection[1,iPlotAlongSection],
                 ExactStateVariablesAlongSection[2,iPlotAlongSection]] = (
                DetermineExactMeridionalVelocities(ProblemType,myExactSolutionParameters,
                                                   xPlotAlongSection[iPlotAlongSection],
                                                   yPlotAlongSection[iPlotAlongSection],time))
            else:
                [ExactStateVariablesAlongSection[0,iPlotAlongSection],
                 ExactStateVariablesAlongSection[1,iPlotAlongSection],
                 ExactStateVariablesAlongSection[2,iPlotAlongSection]] = (
                DetermineExactSurfaceElevations(ProblemType,myExactSolutionParameters,
                                                xPlotAlongSection[iPlotAlongSection],
                                                yPlotAlongSection[iPlotAlongSection],time))
    return ExactStateVariablesAlongSection


def WriteExactStateVariablesAlongSectionToFile(ProblemType,OutputDirectory,rPlotAlongSection,
                                               ExactStateVariablesAlongSection,filename):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    nPlotAlongSection = len(rPlotAlongSection) - 1
    filename = filename + '.curve'
    outputfile = open(filename,'w')
    outputfile.write('#phi\n')
    if ProblemType == 'NonLinear_Manufactured_Solution':
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            outputfile.write('%.15g %.15g\n' %(rPlotAlongSection[iPlotAlongSection],
                                               ExactStateVariablesAlongSection[iPlotAlongSection]))
    elif ProblemType == 'Barotropic_Tide':    
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            outputfile.write('%.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n' 
                             %(rPlotAlongSection[iPlotAlongSection],
                               ExactStateVariablesAlongSection[0,iPlotAlongSection],
                               ExactStateVariablesAlongSection[1,iPlotAlongSection],
                               ExactStateVariablesAlongSection[2,iPlotAlongSection],
                               ExactStateVariablesAlongSection[3,iPlotAlongSection],
                               ExactStateVariablesAlongSection[4,iPlotAlongSection],
                               ExactStateVariablesAlongSection[5,iPlotAlongSection],
                               ExactStateVariablesAlongSection[6,iPlotAlongSection]))    
    else:
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            outputfile.write('%.15g %.15g %.15g %.15g\n' 
                             %(rPlotAlongSection[iPlotAlongSection],
                               ExactStateVariablesAlongSection[0,iPlotAlongSection],
                               ExactStateVariablesAlongSection[1,iPlotAlongSection],
                               ExactStateVariablesAlongSection[2,iPlotAlongSection]))
    outputfile.close()
    os.chdir(cwd)


def ReadExactStateVariablesAlongSectionFromFile(ProblemType,OutputDirectory,filename):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    data = []
    count = 0
    with open(filename,'r') as infile:
        for line in infile:
            if count != 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    nPlotAlongSection = data.shape[0] - 1
    rPlotAlongSection = np.zeros(nPlotAlongSection+1)
    if ProblemType == 'NonLinear_Manufactured_Solution':
        ExactStateVariablesAlongSection = np.zeros((nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            rPlotAlongSection[iPlotAlongSection] = data[iPlotAlongSection,0]
            ExactStateVariablesAlongSection[iPlotAlongSection] = data[iPlotAlongSection,1]      
    elif ProblemType == 'Barotropic_Tide':    
        ExactStateVariablesAlongSection = np.zeros((7,nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            rPlotAlongSection[iPlotAlongSection] = data[iPlotAlongSection,0]
            ExactStateVariablesAlongSection[0,iPlotAlongSection] = data[iPlotAlongSection,1]
            ExactStateVariablesAlongSection[1,iPlotAlongSection] = data[iPlotAlongSection,2]
            ExactStateVariablesAlongSection[2,iPlotAlongSection] = data[iPlotAlongSection,3] 
            ExactStateVariablesAlongSection[3,iPlotAlongSection] = data[iPlotAlongSection,4]
            ExactStateVariablesAlongSection[4,iPlotAlongSection] = data[iPlotAlongSection,5]
            ExactStateVariablesAlongSection[5,iPlotAlongSection] = data[iPlotAlongSection,6]
            ExactStateVariablesAlongSection[6,iPlotAlongSection] = data[iPlotAlongSection,7]
    else:
        ExactStateVariablesAlongSection = np.zeros((3,nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            rPlotAlongSection[iPlotAlongSection] = data[iPlotAlongSection,0]
            ExactStateVariablesAlongSection[0,iPlotAlongSection] = data[iPlotAlongSection,1]
            ExactStateVariablesAlongSection[1,iPlotAlongSection] = data[iPlotAlongSection,2]
            ExactStateVariablesAlongSection[2,iPlotAlongSection] = data[iPlotAlongSection,3]
    os.chdir(cwd)
    return ExactStateVariablesAlongSection


def PlotExactStateVariablesAlongSectionSaveAsPDF(
ProblemType,OutputDirectory,rPlotAlongSection,ExactStateVariablesAlongSection,StateVariableLimitsAlongSection,
linewidths,linestyles,colors,labels,labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,legendposition,title,
titlefontsize,SaveAsPDF,FigureTitle,Show,fig_size=[9.25,9.25],legendWithinBox=False,legendpads=[1.0,0.5],shadow=True,
framealpha=1.0,titlepad=1.035,ProblemType_Equatorial_Wave=False,FigureFormat='pdf'):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)   
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    x = rPlotAlongSection
    y = ExactStateVariablesAlongSection
    if ProblemType == 'NonLinear_Manufactured_Solution':
        ax.plot(x[:],y[:],linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0])
    elif ProblemType == 'Barotropic_Tide':
        ax.plot(x[:],y[0,:],linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],label=legends[0])
        ax.plot(x[:],y[1,:],linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],label=legends[1])
        ax.plot(x[:],y[2,:],linewidth=linewidths[2],linestyle=linestyles[2],color=colors[2],label=legends[2])
        ax.plot(x[:],y[3,:],linewidth=linewidths[3],linestyle=linestyles[3],color=colors[3],label=legends[3])
        ax.plot(x[:],y[4,:],linewidth=linewidths[4],linestyle=linestyles[4],color=colors[4],label=legends[4])
        ax.plot(x[:],y[5,:],linewidth=linewidths[5],linestyle=linestyles[5],color=colors[5],label=legends[5])  
        ax.plot(x[:],y[6,:],linewidth=linewidths[6],linestyle=linestyles[6],color=colors[6],label=legends[6])
    else:
        ax.plot(x[:],y[0,:],linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],label=legends[0])
        ax.plot(x[:],y[1,:],linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],label=legends[1])
        ax.plot(x[:],y[2,:],linewidth=linewidths[2],linestyle=linestyles[2],color=colors[2],label=legends[2])
    plt.xlim(rPlotAlongSection[0],rPlotAlongSection[-1])
    plt.ylim(StateVariableLimitsAlongSection)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    plt.xticks(fontsize=tickfontsizes[0])
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/1000.0), '')))
    plt.yticks(fontsize=tickfontsizes[1])
    if not(ProblemType == 'NonLinear_Manufactured_Solution'):
        if legendWithinBox:
            ax.legend(fontsize=legendfontsize,loc=legendposition,shadow=shadow,framealpha=framealpha) 
        else:
            ax.legend(fontsize=legendfontsize,loc=legendposition,bbox_to_anchor=(legendpads[0],legendpads[1]),
                      shadow=shadow,framealpha=framealpha) 
    ax.set_title(title,fontsize=titlefontsize,fontweight='bold',y=titlepad)
    if ProblemType_Equatorial_Wave:
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    if SaveAsPDF:
        plt.savefig(FigureTitle+'.'+FigureFormat,format=FigureFormat,bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)