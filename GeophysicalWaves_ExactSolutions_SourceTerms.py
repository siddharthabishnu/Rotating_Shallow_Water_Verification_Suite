
# coding: utf-8

# Name: GeophysicalWaves_ExactSolutions_SourceTerms.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains routines for determining (a) the exact solutions of various test cases including geophysical waves, and (b) the source terms of manufactured solutions as functions of space of time appearing on the right hand side of the prognostic equations. <br/>

# In[1]:

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import io as inputoutput
from IPython.utils import io
with io.capture_output() as captured: 
    import Common_Routines as CR


# In[2]:

def DetermineMeridionalDistanceFromEquator(Latitude):
    LatitudeInRadian = Latitude*np.pi/180.0
    RadiusOfEarth = 6371.0*1000.0
    MeridionalDistanceFromEquator = RadiusOfEarth*LatitudeInRadian
    return MeridionalDistanceFromEquator


# In[3]:

def DetermineLatitude(MeridionalDistanceFromEquator):
    RadiusOfEarth = 6371.0*1000.0
    LatitudeInRadian = MeridionalDistanceFromEquator/RadiusOfEarth
    Latitude = LatitudeInRadian*180.0/np.pi
    return Latitude


# In[4]:

def DetermineKelvinWaveAmplitude():
    x = sp.Symbol('x')
    f = sp.sin(x) + 2*sp.sin(2*x)
    Amplitude = sp.calculus.util.maximum(f,x,sp.sets.Interval(0,2*sp.pi))
    return Amplitude


# In[5]:

def CoastalKelvinWaveFunctionalForm(etaHat,kY,y): 
    eta = etaHat*np.sin(kY*y)
    return eta


# In[6]:

def DetermineCoastalKelvinWaveExactSurfaceElevation(c,etaHat,H,kY,R,x,y,time):
    CoastalKelvinWaveExactSurfaceElevation = (
    -H*CoastalKelvinWaveFunctionalForm(etaHat,kY,y+c*time)*np.exp(-x/R))
    return CoastalKelvinWaveExactSurfaceElevation


# In[7]:

def DetermineCoastalKelvinWaveExactMeridionalVelocity(c,etaHat,kY,R,x,y,time):
    CoastalKelvinWaveExactMeridionalVelocity = (
    c*CoastalKelvinWaveFunctionalForm(etaHat,kY,y+c*time)*np.exp(-x/R))
    return CoastalKelvinWaveExactMeridionalVelocity


# In[8]:

def VerifyInertiaGravityWaveExactStateVariables():
    f0, g, H, kX, kY, omega, t, x, y = sp.symbols('f0, g, H, kX, kY, omega, t, x, y')
    eta = sp.cos(kX*x + kY*y - omega*t)
    u = g/(omega**2 - f0**2)*(omega*kX*sp.cos(kX*x + kY*y - omega*t) - f0*kY*sp.sin(kX*x + kY*y - omega*t))
    v = g/(omega**2 - f0**2)*(omega*kY*sp.cos(kX*x + kY*y - omega*t) + f0*kX*sp.sin(kX*x + kY*y - omega*t))
    u_t = sp.diff(u,t)
    u_x = sp.diff(u,x)
    u_y = sp.diff(u,y)
    v_t = sp.diff(v,t)
    v_x = sp.diff(v,x)
    v_y = sp.diff(v,y) 
    eta_t = sp.diff(eta,t)
    eta_x = sp.diff(eta,x)
    eta_y = sp.diff(eta,y)  
    LHS_ZonalMomentumEquation = sp.simplify(u_t - f0*v + g*eta_x)
    LHS_MeridionalMomentumEquation = sp.simplify(v_t + f0*u + g*eta_y)
    LHS_ContinuityEquation = sp.simplify(eta_t + H*(u_x + v_y))
    LHS_ContinuityEquation = sp.simplify(LHS_ContinuityEquation.subs(omega**2, g*H*(kX**2 + kY**2) + f0**2))
    print('The LHS of the zonal momentum equation reduces to %s.' %LHS_ZonalMomentumEquation)
    print('The LHS of the meridional momentum equation reduces to %s.' %LHS_MeridionalMomentumEquation)
    print('The LHS of the continuity equation reduces to %s.' %LHS_ContinuityEquation)


# In[9]:

do_VerifyInertiaGravityWaveExactStateVariables = False
if do_VerifyInertiaGravityWaveExactStateVariables:
    VerifyInertiaGravityWaveExactStateVariables()


# In[10]:

def DetermineInertiaGravityWaveExactSurfaceElevation(etaHat,kX,kY,omega,x,y,time):
    eta = np.cos(kX*x + kY*y - omega*time)
    eta *= etaHat
    return eta


# In[11]:

def DetermineInertiaGravityWaveExactZonalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    u = g/(omega**2.0 - f0**2.0)*(omega*kX*np.cos(kX*x + kY*y - omega*time) 
                                  - f0*kY*np.sin(kX*x + kY*y - omega*time))
    u *= etaHat
    return u


# In[12]:

def DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    v = g/(omega**2.0 - f0**2.0)*(omega*kY*np.cos(kX*x + kY*y - omega*time) 
                                  + f0*kX*np.sin(kX*x + kY*y - omega*time))
    v *= etaHat
    return v


# In[13]:

def VerifyInertiaGravityWavesExactStateVariables():
    f0, g, H, kX, kY, omega, t, x, y = sp.symbols('f0, g, H, kX, kY, omega, t, x, y')
    eta = 2*sp.cos(omega*t)*sp.cos(kX*x + kY*y)
    u = 2*g*sp.sin(kX*x + kY*y)/(omega**2 - f0**2)*(omega*kX*sp.sin(omega*t) - f0*kY*sp.cos(omega*t))
    v = 2*g*sp.sin(kX*x + kY*y)/(omega**2 - f0**2)*(omega*kY*sp.sin(omega*t) + f0*kX*sp.cos(omega*t))
    u_t = sp.diff(u,t)
    u_x = sp.diff(u,x)
    u_y = sp.diff(u,y)
    v_t = sp.diff(v,t)
    v_x = sp.diff(v,x)
    v_y = sp.diff(v,y) 
    eta_t = sp.diff(eta,t)
    eta_x = sp.diff(eta,x)
    eta_y = sp.diff(eta,y)  
    LHS_ZonalMomentumEquation = sp.simplify(u_t - f0*v + g*eta_x)
    LHS_MeridionalMomentumEquation = sp.simplify(v_t + f0*u + g*eta_y)
    LHS_ContinuityEquation = sp.simplify(eta_t + H*(u_x + v_y))
    LHS_ContinuityEquation = sp.simplify(LHS_ContinuityEquation.subs(omega**2, g*H*(kX**2 + kY**2) + f0**2))
    print('The LHS of the zonal momentum equation reduces to %s.' %LHS_ZonalMomentumEquation)
    print('The LHS of the meridional momentum equation reduces to %s.' %LHS_MeridionalMomentumEquation)
    print('The LHS of the continuity equation reduces to %s.' %LHS_ContinuityEquation)


# In[14]:

do_VerifyInertiaGravityWavesExactStateVariables = False
if do_VerifyInertiaGravityWavesExactStateVariables:
    VerifyInertiaGravityWavesExactStateVariables()


# In[15]:

def DetermineInertiaGravityWavesExactSurfaceElevation(etaHat,kX,kY,omega,x,y,time):
    eta = 2.0*np.cos(omega*time)*np.cos(kX*x + kY*y)
    eta *= etaHat
    return eta


# In[16]:

def DetermineInertiaGravityWavesExactZonalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    u = 2.0*g*sp.sin(kX*x + kY*y)/(omega**2.0 - f0**2.0)*(omega*kX*sp.sin(omega*time) - f0*kY*sp.cos(omega*time))
    u *= etaHat
    return u


# In[17]:

def DetermineInertiaGravityWavesExactMeridionalVelocity(etaHat,f0,g,kX,kY,omega,x,y,time):
    v = 2.0*g*sp.sin(kX*x + kY*y)/(omega**2.0 - f0**2.0)*(omega*kY*sp.sin(omega*time) + f0*kX*sp.cos(omega*time))
    v *= etaHat
    return v


# In[18]:

def DeterminePlanetaryRossbyWaveExactStateVariables():
    angleEdge, beta0, etaHat, f0, g, H, kX, kY, omega, t, x, y = (
    sp.symbols('angleEdge, beta0, etaHat, f0, g, H, kX, kY, omega, t, x, y'))
    eta = etaHat*sp.sin(kX*x + kY*y - omega*t)
    DeterminePlanetaryRossbyWaveExactSurfaceElevation = (
    lambdify((etaHat,kX,kY,omega,x,y,t,), eta, modules="numpy"))
    eta_x = sp.diff(eta,x)
    eta_y = sp.diff(eta,y)
    eta_t = sp.diff(eta,t)
    eta_tx = sp.diff(eta_t,x)
    eta_ty = sp.diff(eta_t,y)
    u = -g/f0*eta_y - g/f0**2*eta_tx + beta0*g/f0**2*y*eta_y
    v = g/f0*eta_x - g/f0**2*eta_ty - beta0*g/f0**2*y*eta_x
    u = sp.collect(sp.expand(u),[sp.sin(kX*x + kY*y - omega*t),sp.cos(kX*x + kY*y - omega*t)])
    v = sp.collect(sp.expand(v),[sp.sin(kX*x + kY*y - omega*t),sp.cos(kX*x + kY*y - omega*t)])
    DeterminePlanetaryRossbyWaveExactZonalVelocity = (
    lambdify((beta0,etaHat,f0,g,kX,kY,omega,x,y,t,), u, modules="numpy"))
    DeterminePlanetaryRossbyWaveExactMeridionalVelocity = (
    lambdify((beta0,etaHat,f0,g,kX,kY,omega,x,y,t,), v, modules="numpy"))
    u_x = sp.diff(u,x)
    u_t = sp.diff(u,t)
    v_y = sp.diff(v,y)
    v_t = sp.diff(v,t)
    etaSourceTerm = eta_t + H*(u_x + v_y)
    etaSourceTerm = sp.collect(sp.expand(etaSourceTerm),
                               [sp.sin(kX*x + kY*y - omega*t),sp.cos(kX*x + kY*y - omega*t)])
    DeterminePlanetaryRossbyWaveSurfaceElevationSourceTerm = (
    lambdify((beta0,etaHat,f0,g,H,kX,kY,omega,x,y,t,), etaSourceTerm, modules="numpy"))
    uSourceTerm = u_t - (f0 + beta0*y)*v + g*eta_x
    vSourceTerm = v_t + (f0 + beta0*y)*u + g*eta_y
    normalVelocitySourceTerm = uSourceTerm*sp.cos(angleEdge) + vSourceTerm*sp.sin(angleEdge)
    normalVelocitySourceTerm = sp.collect(sp.expand(normalVelocitySourceTerm),
                                          [sp.sin(kX*x + kY*y - omega*t),sp.cos(kX*x + kY*y - omega*t)])
    DeterminePlanetaryRossbyWaveNormalVelocitySourceTerm = (
    lambdify((angleEdge,beta0,etaHat,f0,g,H,kX,kY,omega,x,y,t,), normalVelocitySourceTerm, modules="numpy")) 
    return [DeterminePlanetaryRossbyWaveExactSurfaceElevation,DeterminePlanetaryRossbyWaveExactZonalVelocity,
            DeterminePlanetaryRossbyWaveExactMeridionalVelocity,
            DeterminePlanetaryRossbyWaveSurfaceElevationSourceTerm,
            DeterminePlanetaryRossbyWaveNormalVelocitySourceTerm]


# In[19]:

[DeterminePlanetaryRossbyWaveExactSurfaceElevation,DeterminePlanetaryRossbyWaveExactZonalVelocity,
 DeterminePlanetaryRossbyWaveExactMeridionalVelocity,
 DeterminePlanetaryRossbyWaveSurfaceElevationSourceTerm,
 DeterminePlanetaryRossbyWaveNormalVelocitySourceTerm] = DeterminePlanetaryRossbyWaveExactStateVariables()


# In[20]:

def DetermineTopographicRossbyWaveExactStateVariables():
    alpha0, angleEdge, etaHat, f0, g, H0, kX, kY, omega, t, x, y = (
    sp.symbols('alpha0, angleEdge, etaHat, f0, g, H0, kX, kY, omega, t, x, y'))
    eta = etaHat*sp.sin(kX*x + kY*y - omega*t)
    DetermineTopographicRossbyWaveExactSurfaceElevation = (
    lambdify((etaHat,kX,kY,omega,x,y,t,), eta, modules="numpy"))
    eta_x = sp.diff(eta,x)
    eta_y = sp.diff(eta,y)
    eta_t = sp.diff(eta,t)
    eta_tx = sp.diff(eta_t,x)
    eta_ty = sp.diff(eta_t,y)
    u = -g/f0*eta_y - g/f0**2*eta_tx
    v = g/f0*eta_x - g/f0**2*eta_ty
    u = sp.collect(sp.expand(u),[sp.sin(kX*x + kY*y - omega*t),sp.cos(kX*x + kY*y - omega*t)])
    v = sp.collect(sp.expand(v),[sp.sin(kX*x + kY*y - omega*t),sp.cos(kX*x + kY*y - omega*t)])    
    DetermineTopographicRossbyWaveExactZonalVelocity = (
    lambdify((etaHat,f0,g,kX,kY,omega,x,y,t,), u, modules="numpy"))
    DetermineTopographicRossbyWaveExactMeridionalVelocity = (
    lambdify((etaHat,f0,g,kX,kY,omega,x,y,t,), v, modules="numpy"))
    u_x = sp.diff(u,x)
    u_t = sp.diff(u,t)
    v_y = sp.diff(v,y)
    v_t = sp.diff(v,t)
    etaSourceTerm = eta_t + H0*(u_x + v_y) + alpha0*v
    etaSourceTerm = sp.collect(sp.expand(etaSourceTerm),
                               [sp.sin(kX*x + kY*y - omega*t),sp.cos(kX*x + kY*y - omega*t)])
    DetermineTopographicRossbyWaveSurfaceElevationSourceTerm = (
    lambdify((alpha0,etaHat,f0,g,H0,kX,kY,omega,x,y,t,), etaSourceTerm, modules="numpy"))
    uSourceTerm = u_t - f0*v + g*eta_x
    vSourceTerm = v_t + f0*u + g*eta_y
    normalVelocitySourceTerm = uSourceTerm*sp.cos(angleEdge) + vSourceTerm*sp.sin(angleEdge)
    normalVelocitySourceTerm = sp.collect(sp.expand(normalVelocitySourceTerm),
                                          [sp.sin(kX*x + kY*y - omega*t),sp.cos(kX*x + kY*y - omega*t)])
    DetermineTopographicRossbyWaveNormalVelocitySourceTerm = (
    lambdify((angleEdge,etaHat,f0,g,H0,kX,kY,omega,x,y,t,), normalVelocitySourceTerm, modules="numpy")) 
    return [DetermineTopographicRossbyWaveExactSurfaceElevation,DetermineTopographicRossbyWaveExactZonalVelocity,
            DetermineTopographicRossbyWaveExactMeridionalVelocity,
            DetermineTopographicRossbyWaveSurfaceElevationSourceTerm,
            DetermineTopographicRossbyWaveNormalVelocitySourceTerm]


# In[21]:

[DetermineTopographicRossbyWaveExactSurfaceElevation,DetermineTopographicRossbyWaveExactZonalVelocity,
 DetermineTopographicRossbyWaveExactMeridionalVelocity,
 DetermineTopographicRossbyWaveSurfaceElevationSourceTerm,
 DetermineTopographicRossbyWaveNormalVelocitySourceTerm] = DetermineTopographicRossbyWaveExactStateVariables()


# In[22]:

def EquatorialKelvinWaveFunctionalForm(etaHat,kX,x): 
    eta = etaHat*np.sin(kX*x)
    return eta


# In[23]:

def DetermineEquatorialKelvinWaveExactSurfaceElevation(c,etaHat,H,kX,Req,x,y,time):
    EquatorialKelvinWaveExactSurfaceElevation = (
    H*EquatorialKelvinWaveFunctionalForm(etaHat,kX,x-c*time)*np.exp(-0.5*(y/Req)**2.0))
    return EquatorialKelvinWaveExactSurfaceElevation


# In[24]:

def DetermineEquatorialKelvinWaveExactZonalVelocity(c,etaHat,H,kX,Req,x,y,time):
    EquatorialKelvinWaveExactZonalVelocity = (
    c*EquatorialKelvinWaveFunctionalForm(etaHat,kX,x-c*time)*np.exp(-0.5*(y/Req)**2.0))
    return EquatorialKelvinWaveExactZonalVelocity


# In[25]:

def sympy_HermiteFunctions(printSolution=False):
    y = sp.Symbol('y')
    ZerothOrderHermitePolynomial = 1
    ZerothOrderHermiteFunction = sp.exp(-y**2/2)*ZerothOrderHermitePolynomial/sp.pi**(1/4)
    ZerothOrderHermiteFunctionDerivative = sp.diff(ZerothOrderHermiteFunction,y)
    FirstOrderHermitePolynomial = 2*y
    FirstOrderHermiteFunction = sp.exp(-y**2/2)*FirstOrderHermitePolynomial/sp.sqrt(2*sp.sqrt(sp.pi))
    FirstOrderHermiteFunctionDerivative = sp.diff(FirstOrderHermiteFunction,y)    
    SecondOrderHermitePolynomial = 4*y**2 - 2
    SecondOrderHermiteFunction = sp.exp(-y**2/2)*SecondOrderHermitePolynomial/(2*sp.sqrt(2*sp.sqrt(sp.pi)))
    SecondOrderHermiteFunctionDerivative = sp.diff(SecondOrderHermiteFunction,y)       
    if printSolution:
        print('The zeroth order Hermite function is %s.' %ZerothOrderHermiteFunction)
        print('The zeroth order Hermite function derivative is %s.' %ZerothOrderHermiteFunctionDerivative)
        print('The first order Hermite function is %s.' %FirstOrderHermiteFunction)
        print('The first order Hermite function derivative is %s.' %FirstOrderHermiteFunctionDerivative)
        print('The second order Hermite function is %s.' %SecondOrderHermiteFunction)
        print('The second order Hermite function derivative is %s.' %SecondOrderHermiteFunctionDerivative)
    ZerothOrderHermiteFunction = lambdify((y,), ZerothOrderHermiteFunction, modules="numpy")
    ZerothOrderHermiteFunctionDerivative = lambdify((y,), ZerothOrderHermiteFunctionDerivative, modules="numpy")
    FirstOrderHermiteFunction = lambdify((y,), FirstOrderHermiteFunction, modules="numpy")
    FirstOrderHermiteFunctionDerivative = lambdify((y,), FirstOrderHermiteFunctionDerivative, modules="numpy")  
    SecondOrderHermiteFunction = lambdify((y,), SecondOrderHermiteFunction, modules="numpy")
    SecondOrderHermiteFunctionDerivative = lambdify((y,), SecondOrderHermiteFunctionDerivative, modules="numpy")
    return [ZerothOrderHermiteFunction, ZerothOrderHermiteFunctionDerivative, FirstOrderHermiteFunction,
            FirstOrderHermiteFunctionDerivative, SecondOrderHermiteFunction, SecondOrderHermiteFunctionDerivative]


# In[26]:

do_sympy_HermiteFunctions = False
if do_sympy_HermiteFunctions:
    [ZerothOrderHermiteFunction, ZerothOrderHermiteFunctionDerivative, FirstOrderHermiteFunction,
     FirstOrderHermiteFunctionDerivative, SecondOrderHermiteFunction, SecondOrderHermiteFunctionDerivative] = (
    sympy_HermiteFunctions(printSolution=True))


# In[27]:

[ZerothOrderHermiteFunction, ZerothOrderHermiteFunctionDerivative, FirstOrderHermiteFunction,
 FirstOrderHermiteFunctionDerivative, SecondOrderHermiteFunction, SecondOrderHermiteFunctionDerivative] = (
sympy_HermiteFunctions(printSolution=False))


# In[28]:

def DetermineHermiteFunctionMaximumAmplitudeWithMeridionalLocation(problem_type,returnMeridionalLocation=True):
    if problem_type == 'Equatorial_Yanai_Wave': # i.e. if HermitePolynomialOrder == 0:
        yMaximumAmplitude = 0.0
        HermitePolynomial = 1.0
        HermiteFunctionMaximumAmplitude = np.exp(-yMaximumAmplitude**2.0/2.0)*HermitePolynomial/np.pi**0.25
    elif problem_type == 'Equatorial_Rossby_Wave': # i.e. if HermitePolynomialOrder == 1:
        yMaximumAmplitude = 1.0
        HermitePolynomial = 2.0*yMaximumAmplitude
        HermiteFunctionMaximumAmplitude = (
        np.exp(-yMaximumAmplitude**2.0/2.0)*HermitePolynomial/np.sqrt(2.0*np.sqrt(np.pi)))
    elif problem_type == 'Equatorial_Inertia_Gravity_Wave': # i.e. if HermitePolynomialOrder == 2:
        yMaximumAmplitude = np.sqrt(2.5)
        HermitePolynomial = 4.0*yMaximumAmplitude**2.0 - 2.0
        HermiteFunctionMaximumAmplitude = (
        np.exp(-yMaximumAmplitude**2.0/2.0)*HermitePolynomial/(2.0*np.sqrt(2.0*np.sqrt(np.pi))))
    if returnMeridionalLocation:
        return yMaximumAmplitude, HermiteFunctionMaximumAmplitude
    else:
        return HermiteFunctionMaximumAmplitude


# In[29]:

def VerifyEquatorialWaveNonDimensionalStateVariables(HermitePolynomialOrder):
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
    print('Verification for Hermite Polynomial of order %d:' %HermitePolynomialOrder)
    u = (omega*y*Psi - kX*Psi_y)/(kX**2 - omega**2)*sp.sin(kX*x - omega*t)
    v = Psi*sp.cos(kX*x - omega*t)
    eta = (kX*y*Psi - omega*Psi_y)/(kX**2 - omega**2)*sp.sin(kX*x - omega*t)
    u_t = sp.diff(u,t)
    u_x = sp.diff(u,x)
    u_y = sp.diff(u,y)
    v_t = sp.diff(v,t)
    v_x = sp.diff(v,x)
    v_y = sp.diff(v,y) 
    eta_t = sp.diff(eta,t)
    eta_x = sp.diff(eta,x)
    eta_y = sp.diff(eta,y) 
    LHS_ZonalMomentumEquation = sp.simplify(u_t - y*v + eta_x)
    LHS_MeridionalMomentumEquation = sp.simplify(v_t + y*u + eta_y)
    if HermitePolynomialOrder == 0:
        LHS_MeridionalMomentumEquation = LHS_MeridionalMomentumEquation.subs(-kX*omega + omega**2 - 1, 0)
    elif HermitePolynomialOrder == 1: 
        LHS_MeridionalMomentumEquation = (
        LHS_MeridionalMomentumEquation.subs(-omega**3 + kX**2*omega + kX + 3*omega, 0))
    elif HermitePolynomialOrder == 2: 
        LHS_MeridionalMomentumEquation = (
        LHS_MeridionalMomentumEquation.subs(omega**3, kX**2*omega + kX + 5*omega))      
        LHS_MeridionalMomentumEquation = sp.simplify(LHS_MeridionalMomentumEquation)
    LHS_ContinuityEquation = sp.simplify(eta_t + u_x + v_y)
    print('The LHS of the zonal momentum equation reduces to %s.' %LHS_ZonalMomentumEquation)
    print('The LHS of the meridional momentum equation reduces to %s.' %LHS_MeridionalMomentumEquation)
    print('The LHS of the continuity equation reduces to %s.' %LHS_ContinuityEquation)


# In[30]:

do_VerifyEquatorialWaveNonDimensionalStateVariables_ZerothOrderHermitePolynomial = False
if do_VerifyEquatorialWaveNonDimensionalStateVariables_ZerothOrderHermitePolynomial:
    VerifyEquatorialWaveNonDimensionalStateVariables(HermitePolynomialOrder=0)


# In[31]:

do_VerifyEquatorialWaveNonDimensionalStateVariables_FirstOrderHermitePolynomial = False
if do_VerifyEquatorialWaveNonDimensionalStateVariables_FirstOrderHermitePolynomial:
    VerifyEquatorialWaveNonDimensionalStateVariables(HermitePolynomialOrder=1)


# In[32]:

do_VerifyEquatorialWaveNonDimensionalStateVariables_SecondOrderHermitePolynomial = False
if do_VerifyEquatorialWaveNonDimensionalStateVariables_SecondOrderHermitePolynomial:
    VerifyEquatorialWaveNonDimensionalStateVariables(HermitePolynomialOrder=2)


# In[33]:

def DetermineEquatorialYanaiWaveNonDimensionalAngularFrequency(kX):
    omega = 0.5*(kX + np.sqrt(kX**2.0 + 4.0))
    return omega


# In[34]:

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
            print('The Newton Raphson solver for the Equatorial angular frequency has converged within', 
                  '%d iterations.' %(iIterationFinal+1))
            print('The angular frequency is %.6f.' %omega)
        else:
            print('The Newton Raphson solver for the Equatorial angular frequency has not converged within', 
                  '%d iterations.' %(iIterationFinal+1))            
    return omega


# In[35]:

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


# In[36]:

[DetermineEquatorialYanaiWaveExactNonDimensionalZonalVelocity,
 DetermineEquatorialYanaiWaveExactNonDimensionalMeridionalVelocity,
 DetermineEquatorialYanaiWaveExactNonDimensionalSurfaceElevation] = (
DetermineEquatorialWaveExactNonDimensionalStateVariables(HermitePolynomialOrder=0))


# In[37]:

def DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat,kX,omega,LengthScale,TimeScale,SurfaceElevationScale,
                                                      x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    eta = SurfaceElevationScale*DetermineEquatorialYanaiWaveExactNonDimensionalSurfaceElevation(kX,omega,x,y,time)
    eta *= etaHat
    return eta    


# In[38]:

def DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    u = VelocityScale*DetermineEquatorialYanaiWaveExactNonDimensionalZonalVelocity(kX,omega,x,y,time)
    u *= etaHat
    return u    


# In[39]:

def DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,y,
                                                        time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    v = VelocityScale*DetermineEquatorialYanaiWaveExactNonDimensionalMeridionalVelocity(kX,omega,x,y,time)
    v *= etaHat
    return v    


# In[40]:

[DetermineEquatorialRossbyWaveExactNonDimensionalZonalVelocity,
 DetermineEquatorialRossbyWaveExactNonDimensionalMeridionalVelocity,
 DetermineEquatorialRossbyWaveExactNonDimensionalSurfaceElevation] = (
DetermineEquatorialWaveExactNonDimensionalStateVariables(HermitePolynomialOrder=1))


# In[41]:

def DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat,kX,omega,LengthScale,TimeScale,SurfaceElevationScale,
                                                       x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    eta = SurfaceElevationScale*DetermineEquatorialRossbyWaveExactNonDimensionalSurfaceElevation(kX,omega,x,y,time)
    eta *= etaHat
    return eta    


# In[42]:

def DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    u = VelocityScale*DetermineEquatorialRossbyWaveExactNonDimensionalZonalVelocity(kX,omega,x,y,time)
    u *= etaHat
    return u    


# In[43]:

def DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,y,
                                                         time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    v = VelocityScale*DetermineEquatorialRossbyWaveExactNonDimensionalMeridionalVelocity(kX,omega,x,y,time)
    v *= etaHat
    return v 


# In[44]:

[DetermineEquatorialInertiaGravityWaveExactNonDimensionalZonalVelocity,
 DetermineEquatorialInertiaGravityWaveExactNonDimensionalMeridionalVelocity,
 DetermineEquatorialInertiaGravityWaveExactNonDimensionalSurfaceElevation] = (
DetermineEquatorialWaveExactNonDimensionalStateVariables(HermitePolynomialOrder=2))


# In[45]:

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


# In[46]:

def DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat,kX,omega,LengthScale,TimeScale,VelocityScale,x,
                                                            y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    u = VelocityScale*DetermineEquatorialInertiaGravityWaveExactNonDimensionalZonalVelocity(kX,omega,x,y,time)
    u *= etaHat
    return u    


# In[47]:

def DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat,kX,omega,LengthScale,TimeScale,
                                                                 VelocityScale,x,y,time):
    x /= LengthScale
    y /= LengthScale
    time /= TimeScale
    kX *= LengthScale
    omega *= TimeScale
    v = VelocityScale*DetermineEquatorialInertiaGravityWaveExactNonDimensionalMeridionalVelocity(kX,omega,x,y,time)
    v *= etaHat
    return v 


# In[48]:

def VerifyBarotropicTideExactStateVariables():
    etaHat, f0, g, H, kX, omega, t, x, y = sp.symbols('etaHat, f0, g, H, kX, omega, t, x, y')
    eta = etaHat*sp.cos(kX*x)*sp.cos(omega*t)
    u = etaHat*g*omega*kX*sp.sin(kX*x)*sp.sin(omega*t)/(omega**2 - f0**2)
    v = etaHat*g*f0*kX*sp.sin(kX*x)*sp.cos(omega*t)/(omega**2 - f0**2)
    u_t = sp.diff(u,t)
    u_x = sp.diff(u,x)
    u_y = sp.diff(u,y)
    v_t = sp.diff(v,t)
    v_x = sp.diff(v,x)
    v_y = sp.diff(v,y) 
    eta_t = sp.diff(eta,t)
    eta_x = sp.diff(eta,x)
    eta_y = sp.diff(eta,y)  
    LHS_ZonalMomentumEquation = sp.simplify(u_t - f0*v + g*eta_x)
    LHS_MeridionalMomentumEquation = sp.simplify(v_t + f0*u + g*eta_y)
    LHS_ContinuityEquation = sp.simplify(eta_t + H*(u_x + v_y))
    LHS_ContinuityEquation = LHS_ContinuityEquation.subs(omega**2, g*H*kX**2 + f0**2)
    print('The LHS of the zonal momentum equation reduces to %s.' %LHS_ZonalMomentumEquation)
    print('The LHS of the meridional momentum equation reduces to %s.' %LHS_MeridionalMomentumEquation)
    print('The LHS of the continuity equation reduces to %s.' %LHS_ContinuityEquation)


# In[49]:

do_VerifyBarotropicTideExactStateVariables = False
if do_VerifyBarotropicTideExactStateVariables:
    VerifyBarotropicTideExactStateVariables()


# In[50]:

def DetermineBarotropicTideExactSurfaceElevation(etaHat,kX,omega,x,y,time):
    eta = etaHat*np.cos(kX*x)*np.cos(omega*time)
    # Note that eta = 0.5*etaHat(np.cos(kX*x + omega*t) + np.cos(kX*x - omega*t))
    return eta


# In[51]:

def DetermineBarotropicTideExactSurfaceElevations_ComponentWaves(etaHat,kX,omega,x,y,time):
    eta1 = 0.5*etaHat*np.cos(kX*x + omega*time)
    eta2 = 0.5*etaHat*np.cos(kX*x - omega*time)
    return eta1, eta2


# In[52]:

def DetermineBarotropicTideExactZonalVelocity(etaHat,f0,g,kX,omega,x,y,time):
    u = etaHat*g*omega*kX*np.sin(kX*x)*np.sin(omega*time)/(omega**2.0 - f0**2.0)
    return u


# In[53]:

def DetermineBarotropicTideExactMeridionalVelocity(etaHat,f0,g,kX,omega,x,y,time):
    v = etaHat*g*f0*kX*np.sin(kX*x)*np.cos(omega*time)/(omega**2.0 - f0**2.0)
    return v


# In[54]:

def VerifyDiffusionEquationExactZonalVelocity():
    etaHat, kX, kappaX, kappaY, kY, t, x, y = sp.symbols('etaHat, kX, kappaX, kappaY, kY, t, x, y')
    kappa = kappaX*kX**2 + kappaY*kY**2
    u = etaHat*sp.sin(kX*x)*sp.sin(kY*y)*sp.exp(-kappa*t)
    v = 0
    eta = 0
    u_t = sp.diff(u,t)
    u_xx = sp.diff(u,x,2)
    u_yy = sp.diff(u,y,2)    
    LHS_ZonalMomentumEquation = sp.simplify(u_t - kappaX*u_xx - kappaY*u_yy)
    print('The LHS of the zonal momentum equation reduces to %s.' %LHS_ZonalMomentumEquation)


# In[55]:

do_VerifyDiffusionEquationExactZonalVelocity = False
if VerifyDiffusionEquationExactZonalVelocity:
    VerifyDiffusionEquationExactZonalVelocity()


# In[56]:

def DetermineDiffusionEquationExactZonalVelocity(etaHat,kappa,kX,kY,x,y,time):
    u = etaHat*np.sin(kX*x)*np.sin(kY*y)*np.exp(-kappa*time)
    return u


# In[57]:

def VerifyViscousBurgersEquationExactZonalVelocity():
    nu, t, uL, uR, x0, x, y = sp.symbols('nu, t, uL, uR, x0, x, y')
    s = (uL + uR)/2
    u = s - (uL - uR)/2*sp.tanh((x - x0 - s*t)*(uL - uR)/(4*nu))
    u_t = sp.diff(u,t)
    u_x = sp.diff(u,x)
    u_xx = sp.diff(u,x,2)
    LHS_ZonalMomentumEquation = sp.simplify(u_t + u*u_x - nu*u_xx)
    print('The LHS of the zonal momentum equation reduces to %s.' %LHS_ZonalMomentumEquation)


# In[58]:

do_VerifyViscousBurgersEquationExactZonalVelocity = False
if do_VerifyViscousBurgersEquationExactZonalVelocity:
    VerifyViscousBurgersEquationExactZonalVelocity()


# In[59]:

def DetermineViscousBurgersEquationExactZonalVelocity(nu,s,uL,uR,x0,x,y,time):
    u = s - 0.5*(uL - uR)*np.tanh((x - x0 - s*time)*(uL - uR)/(4.0*nu))
    return u


# In[60]:

def DetermineGeophysicalWaveExactSurfaceElevation(problem_type,ExactSolutionParameters,x,y,time):
    alpha0 = ExactSolutionParameters[0]
    angleEdge = ExactSolutionParameters[1]
    beta0 = ExactSolutionParameters[2]
    c = ExactSolutionParameters[3]
    dx = ExactSolutionParameters[4]
    dy = ExactSolutionParameters[5]
    cX1 = ExactSolutionParameters[6]
    cX2 = ExactSolutionParameters[7]
    cY1 = ExactSolutionParameters[8]
    cY2 = ExactSolutionParameters[9]
    etaHat1 = ExactSolutionParameters[10]
    etaHat2 = ExactSolutionParameters[11]
    f0 = ExactSolutionParameters[12]
    g = ExactSolutionParameters[13]
    H = ExactSolutionParameters[14]
    kX1 = ExactSolutionParameters[15] 
    kX2 = ExactSolutionParameters[16] 
    kY1 = ExactSolutionParameters[17] 
    kY2 = ExactSolutionParameters[18] 
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    lX = ExactSolutionParameters[19]
    lY = ExactSolutionParameters[20]
    omega1 = ExactSolutionParameters[21]
    omega2 = ExactSolutionParameters[22]
    R = ExactSolutionParameters[23]
    Req = ExactSolutionParameters[24]
    LengthScale = ExactSolutionParameters[25]
    TimeScale = ExactSolutionParameters[26]
    VelocityScale = ExactSolutionParameters[27]
    SurfaceElevationScale = ExactSolutionParameters[28]
    if problem_type == 'Coastal_Kelvin_Wave':
        GeophysicalWaveExactSurfaceElevation = (
        (DetermineCoastalKelvinWaveExactSurfaceElevation(c,etaHat1,H,kY1,R,x,y,time)
         + DetermineCoastalKelvinWaveExactSurfaceElevation(c,etaHat2,H,kY2,R,x,y,time)))
    elif problem_type == 'Inertia_Gravity_Wave':
        GeophysicalWaveExactSurfaceElevation = (
        (DetermineInertiaGravityWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
         + DetermineInertiaGravityWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time)))
    elif problem_type == 'Inertia_Gravity_Waves':
        GeophysicalWaveExactSurfaceElevation = (
        (DetermineInertiaGravityWavesExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
         + DetermineInertiaGravityWavesExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time)))        
    elif problem_type == 'Planetary_Rossby_Wave':
        GeophysicalWaveExactSurfaceElevation = (
        (DeterminePlanetaryRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
         + DeterminePlanetaryRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time)))      
    elif problem_type == 'Topographic_Rossby_Wave':
        GeophysicalWaveExactSurfaceElevation = (
        (DetermineTopographicRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time)
         + DetermineTopographicRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time))) 
    elif problem_type == 'Equatorial_Kelvin_Wave':
        GeophysicalWaveExactSurfaceElevation = (
        (DetermineEquatorialKelvinWaveExactSurfaceElevation(c,etaHat1,H,kX1,Req,x,y,time)
         + DetermineEquatorialKelvinWaveExactSurfaceElevation(c,etaHat2,H,kX2,Req,x,y,time)))
    elif problem_type == 'Equatorial_Yanai_Wave':
        GeophysicalWaveExactSurfaceElevation = (
        (DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                           SurfaceElevationScale,x,y,time)
         + DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                             SurfaceElevationScale,x,y,time)))    
    elif problem_type == 'Equatorial_Rossby_Wave':
        GeophysicalWaveExactSurfaceElevation = (
        (DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                            SurfaceElevationScale,x,y,time)
         + DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                              SurfaceElevationScale,x,y,time)))  
    elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
        GeophysicalWaveExactSurfaceElevation = (
        (DetermineEquatorialInertiaGravityWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                    SurfaceElevationScale,x,y,time)
         + DetermineEquatorialInertiaGravityWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                      SurfaceElevationScale,x,y,time)))  
    elif problem_type == 'Barotropic_Tide':
        GeophysicalWaveExactSurfaceElevation = (
        (DetermineBarotropicTideExactSurfaceElevation(etaHat1,kX1,omega1,x,y,time)
         + DetermineBarotropicTideExactSurfaceElevation(etaHat2,kX2,omega2,x,y,time)))
    elif problem_type == 'Diffusion_Equation':
        GeophysicalWaveExactSurfaceElevation = 0.0
    elif problem_type == 'Viscous_Burgers_Equation':
        GeophysicalWaveExactSurfaceElevation = 0.0
    return GeophysicalWaveExactSurfaceElevation


# In[61]:

def DetermineGeophysicalWaveExactSurfaceElevations(problem_type,ExactSolutionParameters,x,y,time):
    alpha0 = ExactSolutionParameters[0]
    angleEdge = ExactSolutionParameters[1]
    beta0 = ExactSolutionParameters[2]
    c = ExactSolutionParameters[3]
    dx = ExactSolutionParameters[4]
    dy = ExactSolutionParameters[5]
    cX1 = ExactSolutionParameters[6]
    cX2 = ExactSolutionParameters[7]
    cY1 = ExactSolutionParameters[8]
    cY2 = ExactSolutionParameters[9]
    etaHat1 = ExactSolutionParameters[10]
    etaHat2 = ExactSolutionParameters[11]
    f0 = ExactSolutionParameters[12]
    g = ExactSolutionParameters[13]
    H = ExactSolutionParameters[14]
    kX1 = ExactSolutionParameters[15] 
    kX2 = ExactSolutionParameters[16] 
    kY1 = ExactSolutionParameters[17] 
    kY2 = ExactSolutionParameters[18] 
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical 
    # wave.
    lX = ExactSolutionParameters[19]
    lY = ExactSolutionParameters[20]
    omega1 = ExactSolutionParameters[21]
    omega2 = ExactSolutionParameters[22]
    R = ExactSolutionParameters[23]
    Req = ExactSolutionParameters[24]
    LengthScale = ExactSolutionParameters[25]
    TimeScale = ExactSolutionParameters[26]
    VelocityScale = ExactSolutionParameters[27]
    SurfaceElevationScale = ExactSolutionParameters[28]
    if problem_type == 'Barotropic_Tide':
        GeophysicalWaveExactSurfaceElevations = np.zeros(7)
    else:
        GeophysicalWaveExactSurfaceElevations = np.zeros(3)
    if problem_type == 'Coastal_Kelvin_Wave':
        GeophysicalWaveExactSurfaceElevations[0] = (
        DetermineCoastalKelvinWaveExactSurfaceElevation(c,etaHat1,H,kY1,R,x,y,time))
        GeophysicalWaveExactSurfaceElevations[1] = (
        DetermineCoastalKelvinWaveExactSurfaceElevation(c,etaHat2,H,kY2,R,x,y,time))
    elif problem_type == 'Inertia_Gravity_Wave':
        GeophysicalWaveExactSurfaceElevations[0] = (
        DetermineInertiaGravityWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactSurfaceElevations[1] = (
        DetermineInertiaGravityWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time))      
    elif problem_type == 'Inertia_Gravity_Waves':
        GeophysicalWaveExactSurfaceElevations[0] = (
        DetermineInertiaGravityWavesExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactSurfaceElevations[1] = (
        DetermineInertiaGravityWavesExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time))         
    elif problem_type == 'Planetary_Rossby_Wave':
        GeophysicalWaveExactSurfaceElevations[0] = (
        DeterminePlanetaryRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactSurfaceElevations[1] = (
        DeterminePlanetaryRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time))        
    elif problem_type == 'Topographic_Rossby_Wave':
        GeophysicalWaveExactSurfaceElevations[0] = (
        DetermineTopographicRossbyWaveExactSurfaceElevation(etaHat1,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactSurfaceElevations[1] = (
        DetermineTopographicRossbyWaveExactSurfaceElevation(etaHat2,kX2,kY2,omega2,x,y,time))       
    elif problem_type == 'Equatorial_Kelvin_Wave':
        GeophysicalWaveExactSurfaceElevations[0] = (
        DetermineEquatorialKelvinWaveExactSurfaceElevation(c,etaHat1,H,kX1,Req,x,y,time))
        GeophysicalWaveExactSurfaceElevations[1] = (
        DetermineEquatorialKelvinWaveExactSurfaceElevation(c,etaHat2,H,kX2,Req,x,y,time))         
    elif problem_type == 'Equatorial_Yanai_Wave':
        GeophysicalWaveExactSurfaceElevations[0] = (
        DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                          SurfaceElevationScale,x,y,time))
        GeophysicalWaveExactSurfaceElevations[1] = (
        DetermineEquatorialYanaiWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                          SurfaceElevationScale,x,y,time))       
    elif problem_type == 'Equatorial_Rossby_Wave':
        GeophysicalWaveExactSurfaceElevations[0] = (
        DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                           SurfaceElevationScale,x,y,time))
        GeophysicalWaveExactSurfaceElevations[1] = (
        DetermineEquatorialRossbyWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                           SurfaceElevationScale,x,y,time))        
    elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
        GeophysicalWaveExactSurfaceElevations[0] = (
        DetermineEquatorialInertiaGravityWaveExactSurfaceElevation(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                   SurfaceElevationScale,x,y,time))
        GeophysicalWaveExactSurfaceElevations[1] = (
        DetermineEquatorialInertiaGravityWaveExactSurfaceElevation(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                   SurfaceElevationScale,x,y,time))
    elif problem_type == 'Barotropic_Tide':
        GeophysicalWaveExactSurfaceElevations[0], GeophysicalWaveExactSurfaceElevations[1] = (
        DetermineBarotropicTideExactSurfaceElevations_ComponentWaves(etaHat1,kX1,omega1,x,y,time))
        GeophysicalWaveExactSurfaceElevations[2] = (
        DetermineBarotropicTideExactSurfaceElevation(etaHat1,kX1,omega1,x,y,time))
        GeophysicalWaveExactSurfaceElevations[3], GeophysicalWaveExactSurfaceElevations[4] = (
        DetermineBarotropicTideExactSurfaceElevations_ComponentWaves(etaHat2,kX2,omega2,x,y,time))
        GeophysicalWaveExactSurfaceElevations[5] = (
        DetermineBarotropicTideExactSurfaceElevation(etaHat2,kX2,omega2,x,y,time))        
    if problem_type == 'Barotropic_Tide':
        GeophysicalWaveExactSurfaceElevations[6] = (
        GeophysicalWaveExactSurfaceElevations[2] + GeophysicalWaveExactSurfaceElevations[5])
    else:
        GeophysicalWaveExactSurfaceElevations[2] = (
        GeophysicalWaveExactSurfaceElevations[0] + GeophysicalWaveExactSurfaceElevations[1])          
    return GeophysicalWaveExactSurfaceElevations


# In[62]:

def DetermineGeophysicalWaveExactZonalVelocity(problem_type,ExactSolutionParameters,x,y,time):
    alpha0 = ExactSolutionParameters[0]
    angleEdge = ExactSolutionParameters[1]
    beta0 = ExactSolutionParameters[2]
    c = ExactSolutionParameters[3]
    cX1 = ExactSolutionParameters[4]
    cX2 = ExactSolutionParameters[5]
    cY1 = ExactSolutionParameters[6]
    cY2 = ExactSolutionParameters[7]
    dx = ExactSolutionParameters[8]
    dy = ExactSolutionParameters[9]
    etaHat1 = ExactSolutionParameters[10]
    etaHat2 = ExactSolutionParameters[11]
    f0 = ExactSolutionParameters[12]
    g = ExactSolutionParameters[13]
    H = ExactSolutionParameters[14]
    kX1 = ExactSolutionParameters[15] 
    kX2 = ExactSolutionParameters[16] 
    kY1 = ExactSolutionParameters[17] 
    kY2 = ExactSolutionParameters[18] 
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    lX = ExactSolutionParameters[19]
    lY = ExactSolutionParameters[20]
    omega1 = ExactSolutionParameters[21]
    omega2 = ExactSolutionParameters[22]
    R = ExactSolutionParameters[23]
    Req = ExactSolutionParameters[24]
    LengthScale = ExactSolutionParameters[25]
    TimeScale = ExactSolutionParameters[26]
    VelocityScale = ExactSolutionParameters[27]
    SurfaceElevationScale = ExactSolutionParameters[28]  
    kappaX = ExactSolutionParameters[29]
    kappaY = ExactSolutionParameters[30]
    kappa1 = ExactSolutionParameters[31]
    kappa2 = ExactSolutionParameters[32]
    uL = ExactSolutionParameters[33]
    uR = ExactSolutionParameters[34]
    s = ExactSolutionParameters[35]
    x0 = ExactSolutionParameters[36]
    if problem_type == 'Coastal_Kelvin_Wave':
        GeophysicalWaveExactZonalVelocity = 0.0
    elif problem_type == 'Inertia_Gravity_Wave':
        GeophysicalWaveExactZonalVelocity = (
        (DetermineInertiaGravityWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineInertiaGravityWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    elif problem_type == 'Inertia_Gravity_Waves':
        GeophysicalWaveExactZonalVelocity = (
        (DetermineInertiaGravityWavesExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineInertiaGravityWavesExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))        
    elif problem_type == 'Planetary_Rossby_Wave':
        GeophysicalWaveExactZonalVelocity = (
        (DeterminePlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DeterminePlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))  
    elif problem_type == 'Topographic_Rossby_Wave':
        GeophysicalWaveExactZonalVelocity = (
        (DetermineTopographicRossbyWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineTopographicRossbyWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))) 
    elif problem_type == 'Equatorial_Kelvin_Wave':
        GeophysicalWaveExactZonalVelocity = (
        (DetermineEquatorialKelvinWaveExactZonalVelocity(c,etaHat1,H,kX1,Req,x,y,time)
         + DetermineEquatorialKelvinWaveExactZonalVelocity(c,etaHat2,H,kX2,Req,x,y,time)))
    elif problem_type == 'Equatorial_Yanai_Wave':
        GeophysicalWaveExactZonalVelocity = (
        (DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,x,y,
                                                        time)
         + DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,x,
                                                          y,time))) 
    elif problem_type == 'Equatorial_Rossby_Wave':
        GeophysicalWaveExactZonalVelocity = (
        (DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,x,
                                                         y,time)
         + DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,
                                                           x,y,time))) 
    elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
        GeophysicalWaveExactZonalVelocity = (
        (DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                 VelocityScale,x,y,time)
         + DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                   VelocityScale,x,y,time)))
    elif problem_type == 'Barotropic_Tide':
        GeophysicalWaveExactZonalVelocity = (
        (DetermineBarotropicTideExactZonalVelocity(etaHat1,f0,g,kX1,omega1,x,y,time)
         + DetermineBarotropicTideExactZonalVelocity(etaHat2,f0,g,kX2,omega2,x,y,time)))        
    elif problem_type == 'Diffusion_Equation':
        GeophysicalWaveExactZonalVelocity = (
        (DetermineDiffusionEquationExactZonalVelocity(etaHat1,kappa1,kX1,kY1,x,y,time)
         + DetermineDiffusionEquationExactZonalVelocity(etaHat2,kappa2,kX2,kY2,x,y,time)))
    elif problem_type == 'Viscous_Burgers_Equation':
        GeophysicalWaveExactZonalVelocity = (
        DetermineViscousBurgersEquationExactZonalVelocity(kappaX,s,uL,uR,x0,x,y,time))
    return GeophysicalWaveExactZonalVelocity


# In[63]:

def DetermineGeophysicalWaveExactZonalVelocities(problem_type,ExactSolutionParameters,x,y,time):
    alpha0 = ExactSolutionParameters[0]
    angleEdge = ExactSolutionParameters[1]
    beta0 = ExactSolutionParameters[2]
    c = ExactSolutionParameters[3]
    dx = ExactSolutionParameters[4]
    dy = ExactSolutionParameters[5]
    cX1 = ExactSolutionParameters[6]
    cX2 = ExactSolutionParameters[7]
    cY1 = ExactSolutionParameters[8]
    cY2 = ExactSolutionParameters[9]
    etaHat1 = ExactSolutionParameters[10]
    etaHat2 = ExactSolutionParameters[11]
    f0 = ExactSolutionParameters[12]
    g = ExactSolutionParameters[13]
    H = ExactSolutionParameters[14]
    kX1 = ExactSolutionParameters[15] 
    kX2 = ExactSolutionParameters[16] 
    kY1 = ExactSolutionParameters[17] 
    kY2 = ExactSolutionParameters[18] 
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    lX = ExactSolutionParameters[19]
    lY = ExactSolutionParameters[20]
    omega1 = ExactSolutionParameters[21]
    omega2 = ExactSolutionParameters[22]
    R = ExactSolutionParameters[23]
    Req = ExactSolutionParameters[24]
    LengthScale = ExactSolutionParameters[25]
    TimeScale = ExactSolutionParameters[26]
    VelocityScale = ExactSolutionParameters[27]
    SurfaceElevationScale = ExactSolutionParameters[28]
    kappaX = ExactSolutionParameters[29]
    kappaY = ExactSolutionParameters[30]
    kappa1 = ExactSolutionParameters[31]
    kappa2 = ExactSolutionParameters[32]
    GeophysicalWaveExactZonalVelocities = np.zeros(3)
    if problem_type == 'Inertia_Gravity_Wave':
        GeophysicalWaveExactZonalVelocities[0] = (
        DetermineInertiaGravityWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DetermineInertiaGravityWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))      
    elif problem_type == 'Inertia_Gravity_Waves':
        GeophysicalWaveExactZonalVelocities[0] = (
        DetermineInertiaGravityWavesExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DetermineInertiaGravityWavesExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))         
    elif problem_type == 'Planetary_Rossby_Wave':
        GeophysicalWaveExactZonalVelocities[0] = (
        DeterminePlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DeterminePlanetaryRossbyWaveExactZonalVelocity(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time))        
    elif problem_type == 'Topographic_Rossby_Wave':
        GeophysicalWaveExactZonalVelocities[0] = (
        DetermineTopographicRossbyWaveExactZonalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DetermineTopographicRossbyWaveExactZonalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))   
    elif problem_type == 'Equatorial_Kelvin_Wave':
        GeophysicalWaveExactZonalVelocities[0] = (
        DetermineEquatorialKelvinWaveExactZonalVelocity(c,etaHat1,H,kX1,Req,x,y,time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DetermineEquatorialKelvinWaveExactZonalVelocity(c,etaHat2,H,kX2,Req,x,y,time))         
    elif problem_type == 'Equatorial_Yanai_Wave':
        GeophysicalWaveExactZonalVelocities[0] = (
        DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,x,y,
                                                       time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DetermineEquatorialYanaiWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,x,y,
                                                       time))       
    elif problem_type == 'Equatorial_Rossby_Wave':
        GeophysicalWaveExactZonalVelocities[0] = (
        DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,x,y,
                                                        time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DetermineEquatorialRossbyWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,x,y,
                                                        time))        
    elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
        GeophysicalWaveExactZonalVelocities[0] = (
        DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                VelocityScale,x,y,time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DetermineEquatorialInertiaGravityWaveExactZonalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                VelocityScale,x,y,time))  
    elif problem_type == 'Barotropic_Tide':
        GeophysicalWaveExactZonalVelocities[0] = (
        DetermineBarotropicTideExactZonalVelocity(etaHat1,f0,g,kX1,omega1,x,y,time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DetermineBarotropicTideExactZonalVelocity(etaHat2,f0,g,kX2,omega2,x,y,time))
    elif problem_type == 'Diffusion_Equation':
        GeophysicalWaveExactZonalVelocities[0] = (
        DetermineDiffusionEquationExactZonalVelocity(etaHat1,kappa1,kX1,kY1,x,y,time))
        GeophysicalWaveExactZonalVelocities[1] = (
        DetermineDiffusionEquationExactZonalVelocity(etaHat2,kappa2,kX2,kY2,x,y,time)) 
    GeophysicalWaveExactZonalVelocities[2] = (
    GeophysicalWaveExactZonalVelocities[0] + GeophysicalWaveExactZonalVelocities[1])          
    return GeophysicalWaveExactZonalVelocities


# In[64]:

def DetermineGeophysicalWaveExactMeridionalVelocity(problem_type,ExactSolutionParameters,x,y,time):
    alpha0 = ExactSolutionParameters[0]
    angleEdge = ExactSolutionParameters[1]
    beta0 = ExactSolutionParameters[2]
    c = ExactSolutionParameters[3]
    cX1 = ExactSolutionParameters[4]
    cX2 = ExactSolutionParameters[5]
    cY1 = ExactSolutionParameters[6]
    cY2 = ExactSolutionParameters[7]
    dx = ExactSolutionParameters[8]
    dy = ExactSolutionParameters[9]
    etaHat1 = ExactSolutionParameters[10]
    etaHat2 = ExactSolutionParameters[11]
    f0 = ExactSolutionParameters[12]
    g = ExactSolutionParameters[13]
    H = ExactSolutionParameters[14]
    kX1 = ExactSolutionParameters[15] 
    kX2 = ExactSolutionParameters[16] 
    kY1 = ExactSolutionParameters[17] 
    kY2 = ExactSolutionParameters[18] 
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    lX = ExactSolutionParameters[19]
    lY = ExactSolutionParameters[20]
    omega1 = ExactSolutionParameters[21]
    omega2 = ExactSolutionParameters[22]
    R = ExactSolutionParameters[23]
    Req = ExactSolutionParameters[24]
    LengthScale = ExactSolutionParameters[25]
    TimeScale = ExactSolutionParameters[26]
    VelocityScale = ExactSolutionParameters[27]
    SurfaceElevationScale = ExactSolutionParameters[28] 
    if problem_type == 'Coastal_Kelvin_Wave':
        GeophysicalWaveExactMeridionalVelocity = (
        (DetermineCoastalKelvinWaveExactMeridionalVelocity(c,etaHat1,kY1,R,x,y,time)
         + DetermineCoastalKelvinWaveExactMeridionalVelocity(c,etaHat2,kY2,R,x,y,time)))
    if problem_type == 'Inertia_Gravity_Wave':
        GeophysicalWaveExactMeridionalVelocity = (
        (DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))
    if problem_type == 'Inertia_Gravity_Waves':
        GeophysicalWaveExactMeridionalVelocity = (
        (DetermineInertiaGravityWavesExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineInertiaGravityWavesExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))        
    elif problem_type == 'Planetary_Rossby_Wave':
        GeophysicalWaveExactMeridionalVelocity = (
        (DeterminePlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DeterminePlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time)))  
    elif problem_type == 'Topographic_Rossby_Wave':
        GeophysicalWaveExactMeridionalVelocity = (
        (DetermineTopographicRossbyWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time)
         + DetermineTopographicRossbyWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))) 
    elif problem_type == 'Equatorial_Kelvin_Wave':
        GeophysicalWaveExactMeridionalVelocity = 0.0
    elif problem_type == 'Equatorial_Yanai_Wave':
        GeophysicalWaveExactMeridionalVelocity = (
        (DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                             VelocityScale,x,y,time)
         + DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                               VelocityScale,x,y,time))) 
    elif problem_type == 'Equatorial_Rossby_Wave':
        GeophysicalWaveExactMeridionalVelocity = (
        (DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                              VelocityScale,x,y,time)
         + DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                VelocityScale,x,y,time)))
    elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
        GeophysicalWaveExactMeridionalVelocity = (
        (DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                      VelocityScale,x,y,time)
         + DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                        VelocityScale,x,y,time)))
    elif problem_type == 'Barotropic_Tide':
        GeophysicalWaveExactMeridionalVelocity = (
        (DetermineBarotropicTideExactMeridionalVelocity(etaHat1,f0,g,kX1,omega1,x,y,time)
         + DetermineBarotropicTideExactMeridionalVelocity(etaHat2,f0,g,kX2,omega2,x,y,time)))
    elif problem_type == 'Diffusion_Equation':
        GeophysicalWaveExactMeridionalVelocity = 0.0
    elif problem_type == 'Viscous_Burgers_Equation':
        GeophysicalWaveExactMeridionalVelocity = 0.0
    return GeophysicalWaveExactMeridionalVelocity


# In[65]:

def DetermineGeophysicalWaveExactMeridionalVelocities(problem_type,ExactSolutionParameters,x,y,time):
    alpha0 = ExactSolutionParameters[0]
    angleEdge = ExactSolutionParameters[1]
    beta0 = ExactSolutionParameters[2]
    c = ExactSolutionParameters[3]
    dx = ExactSolutionParameters[4]
    dy = ExactSolutionParameters[5]
    cX1 = ExactSolutionParameters[6]
    cX2 = ExactSolutionParameters[7]
    cY1 = ExactSolutionParameters[8]
    cY2 = ExactSolutionParameters[9]
    etaHat1 = ExactSolutionParameters[10]
    etaHat2 = ExactSolutionParameters[11]
    f0 = ExactSolutionParameters[12]
    g = ExactSolutionParameters[13]
    H = ExactSolutionParameters[14]
    kX1 = ExactSolutionParameters[15] 
    kX2 = ExactSolutionParameters[16] 
    kY1 = ExactSolutionParameters[17] 
    kY2 = ExactSolutionParameters[18] 
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    lX = ExactSolutionParameters[19]
    lY = ExactSolutionParameters[20]
    omega1 = ExactSolutionParameters[21]
    omega2 = ExactSolutionParameters[22]
    R = ExactSolutionParameters[23]
    Req = ExactSolutionParameters[24]
    LengthScale = ExactSolutionParameters[25]
    TimeScale = ExactSolutionParameters[26]
    VelocityScale = ExactSolutionParameters[27]
    SurfaceElevationScale = ExactSolutionParameters[28]
    GeophysicalWaveExactMeridionalVelocities = np.zeros(3)
    if problem_type == 'Coastal_Kelvin_Wave':
        GeophysicalWaveExactMeridionalVelocities[0] = (
        DetermineCoastalKelvinWaveExactMeridionalVelocity(c,etaHat1,kY1,R,x,y,time))
        GeophysicalWaveExactMeridionalVelocities[1] = (
        DetermineCoastalKelvinWaveExactMeridionalVelocity(c,etaHat2,kY2,R,x,y,time))
    elif problem_type == 'Inertia_Gravity_Wave':
        GeophysicalWaveExactMeridionalVelocities[0] = (
        DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactMeridionalVelocities[1] = (
        DetermineInertiaGravityWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))      
    elif problem_type == 'Inertia_Gravity_Waves':
        GeophysicalWaveExactMeridionalVelocities[0] = (
        DetermineInertiaGravityWavesExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactMeridionalVelocities[1] = (
        DetermineInertiaGravityWavesExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))         
    elif problem_type == 'Planetary_Rossby_Wave':
        GeophysicalWaveExactMeridionalVelocities[0] = (
        DeterminePlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactMeridionalVelocities[1] = (
        DeterminePlanetaryRossbyWaveExactMeridionalVelocity(beta0,etaHat2,f0,g,kX2,kY2,omega2,x,y,time))        
    elif problem_type == 'Topographic_Rossby_Wave':
        GeophysicalWaveExactMeridionalVelocities[0] = (
        DetermineTopographicRossbyWaveExactMeridionalVelocity(etaHat1,f0,g,kX1,kY1,omega1,x,y,time))
        GeophysicalWaveExactMeridionalVelocities[1] = (
        DetermineTopographicRossbyWaveExactMeridionalVelocity(etaHat2,f0,g,kX2,kY2,omega2,x,y,time))               
    elif problem_type == 'Equatorial_Yanai_Wave':
        GeophysicalWaveExactMeridionalVelocities[0] = (
        DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,VelocityScale,
                                                            x,y,time))
        GeophysicalWaveExactMeridionalVelocities[1] = (
        DetermineEquatorialYanaiWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,VelocityScale,
                                                            x,y,time))       
    elif problem_type == 'Equatorial_Rossby_Wave':
        GeophysicalWaveExactMeridionalVelocities[0] = (
        DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                             VelocityScale,x,y,time))
        GeophysicalWaveExactMeridionalVelocities[1] = (
        DetermineEquatorialRossbyWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                             VelocityScale,x,y,time))        
    elif problem_type == 'Equatorial_Inertia_Gravity_Wave':
        GeophysicalWaveExactMeridionalVelocities[0] = (
        DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat1,kX1,omega1,LengthScale,TimeScale,
                                                                     VelocityScale,x,y,time))
        GeophysicalWaveExactMeridionalVelocities[1] = (
        DetermineEquatorialInertiaGravityWaveExactMeridionalVelocity(etaHat2,kX2,omega2,LengthScale,TimeScale,
                                                                     VelocityScale,x,y,time))    
    elif problem_type == 'Barotropic_Tide':
        GeophysicalWaveExactMeridionalVelocities[0] = (
        DetermineBarotropicTideExactMeridionalVelocity(etaHat1,f0,g,kX1,omega1,x,y,time))
        GeophysicalWaveExactMeridionalVelocities[1] = (
        DetermineBarotropicTideExactMeridionalVelocity(etaHat2,f0,g,kX2,omega2,x,y,time))  
    GeophysicalWaveExactMeridionalVelocities[2] = (
    GeophysicalWaveExactMeridionalVelocities[0] + GeophysicalWaveExactMeridionalVelocities[1])          
    return GeophysicalWaveExactMeridionalVelocities


# In[66]:

def DetermineGeophysicalWaveExactSolutionLimits(problem_type,ExactSolutionParameters):
    alpha0 = ExactSolutionParameters[0]
    angleEdge = ExactSolutionParameters[1]
    beta0 = ExactSolutionParameters[2]
    c = ExactSolutionParameters[3]
    cX1 = ExactSolutionParameters[4]
    cX2 = ExactSolutionParameters[5]
    cY1 = ExactSolutionParameters[6]
    cY2 = ExactSolutionParameters[7]
    dx = ExactSolutionParameters[8]
    dy = ExactSolutionParameters[9]
    etaHat1 = ExactSolutionParameters[10]
    etaHat2 = ExactSolutionParameters[11]
    f0 = ExactSolutionParameters[12]
    g = ExactSolutionParameters[13]
    H = ExactSolutionParameters[14]
    kX1 = ExactSolutionParameters[15] 
    kX2 = ExactSolutionParameters[16] 
    kY1 = ExactSolutionParameters[17] 
    kY2 = ExactSolutionParameters[18] 
    # Note that kX1*lX = kY1*lY = 4.0*np.pi and kX2*lX = kY2*lY = 8.0*np.pi for the inertia-gravity wave and 
    # kX1*lX = kY1*lY = 2.0*np.pi and kX2*lX = kY2*lY = 4.0*np.pi for every other two-dimensional geophysical wave.
    lX = ExactSolutionParameters[19]
    lY = ExactSolutionParameters[20]
    omega1 = ExactSolutionParameters[21]
    omega2 = ExactSolutionParameters[22]
    R = ExactSolutionParameters[23]
    Req = ExactSolutionParameters[24]
    LengthScale = ExactSolutionParameters[25]
    TimeScale = ExactSolutionParameters[26]
    VelocityScale = ExactSolutionParameters[27]
    SurfaceElevationScale = ExactSolutionParameters[28]
    uL = ExactSolutionParameters[33]
    uR = ExactSolutionParameters[34]
    SurfaceElevationLimits = np.zeros(2)
    ZonalVelocityLimits = np.zeros(2)
    MeridionalVelocityLimits = np.zeros(2)    
    if problem_type == 'Coastal_Kelvin_Wave':
        xCell = 0.5*dx
        SurfaceElevationLimits[1] = abs(H*etaHat1*float(DetermineKelvinWaveAmplitude())*np.exp(-xCell/R))
        SurfaceElevationLimits[0] = -SurfaceElevationLimits[1]
        MeridionalVelocityLimits[1] = abs(c*etaHat1*float(DetermineKelvinWaveAmplitude())*np.exp(-xCell/R))
        MeridionalVelocityLimits[0] = -MeridionalVelocityLimits[1]
    elif problem_type == 'Equatorial_Kelvin_Wave':
        SurfaceElevationLimits[1] = abs(H*etaHat1*float(DetermineKelvinWaveAmplitude()))
        SurfaceElevationLimits[0] = -SurfaceElevationLimits[1]
        ZonalVelocityLimits[1] = abs(c*etaHat1*float(DetermineKelvinWaveAmplitude()))
        ZonalVelocityLimits[0] = -ZonalVelocityLimits[1]        
    elif problem_type == 'Inertia_Gravity_Wave':
        SurfaceElevationLimits[1] = etaHat1 + etaHat2
        SurfaceElevationLimits[0] = -SurfaceElevationLimits[1]
        ZonalVelocityLimits[1] = (abs(etaHat1*g/(omega1**2.0 - f0**2.0)*np.sqrt((omega1*kX1)**2.0 + (f0*kY1)**2.0))
                                  + abs(etaHat2*g/(omega2**2.0 - f0**2.0)
                                        *np.sqrt((omega2*kX2)**2.0 + (f0*kY2)**2.0)))
        ZonalVelocityLimits[0] = -ZonalVelocityLimits[1]
        MeridionalVelocityLimits[1] = (abs(etaHat1*g/(omega1**2.0 - f0**2.0)
                                           *np.sqrt((omega1*kY1)**2.0 + (f0*kX1)**2.0))
                                       + abs(etaHat2*g/(omega2**2.0 - f0**2.0)
                                             *np.sqrt((omega2*kY2)**2.0 + (f0*kX2)**2.0)))
        MeridionalVelocityLimits[0] = -MeridionalVelocityLimits[1]   
    elif problem_type == 'Barotropic_Tide':
        SurfaceElevationLimits[1] = etaHat1 + etaHat2
        SurfaceElevationLimits[0] = -SurfaceElevationLimits[1]
        ZonalVelocityLimits[1] = (
        etaHat1*g*omega1*kX1/(omega1**2.0 - f0**2.0) + etaHat2*g*omega2*kX2/(omega2**2.0 - f0**2.0))
        ZonalVelocityLimits[0] = -ZonalVelocityLimits[1]
        MeridionalVelocityLimits[1] = (
        etaHat1*g*f0*kX1/(omega1**2.0 - f0**2.0) + etaHat2*g*f0*kX2/(omega2**2.0 - f0**2.0))
        MeridionalVelocityLimits[0] = -MeridionalVelocityLimits[1]
    elif problem_type == 'Viscous_Burgers_Equation':
        ZonalVelocityLimits[1] = max(uL,uR)
        ZonalVelocityLimits[0] = min(uL,uR)
    return SurfaceElevationLimits, ZonalVelocityLimits, MeridionalVelocityLimits


# In[67]:

def DetermineCoastalCoordinates(lY,nPlotAlongCoastline):
    xPlotAlongCoastline = np.zeros(nPlotAlongCoastline+1)
    yPlotAlongCoastline = np.linspace(0.0,lY,nPlotAlongCoastline+1)
    rPlotAlongCoastline = yPlotAlongCoastline
    return rPlotAlongCoastline, xPlotAlongCoastline, yPlotAlongCoastline


# In[68]:

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


# In[69]:

def DetermineCoordinatesAlongZonalSection(lX,nPlotAlongZonalSection,y=0.0):
    xPlotAlongZonalSection = np.linspace(0.0,lX,nPlotAlongZonalSection+1)
    yPlotAlongZonalSection = y*np.ones(nPlotAlongZonalSection+1)
    rPlotAlongZonalSection = xPlotAlongZonalSection
    return rPlotAlongZonalSection, xPlotAlongZonalSection, yPlotAlongZonalSection


# In[70]:

def DetermineCoordinatesAlongSection(problem_type,lX,lY,nPlotAlongSection,y=0.0):
    if problem_type == 'Coastal_Kelvin_Wave':
        rPlotAlongSection, xPlotAlongSection, yPlotAlongSection = (
        DetermineCoastalCoordinates(lY,nPlotAlongSection))
    elif (problem_type == 'Inertia_Gravity_Wave' or problem_type == 'Planetary_Rossby_Wave'
          or problem_type == 'Topographic_Rossby_Wave'):    
        DiagonalType = 'SouthWest-NorthEast'
        rPlotAlongSection, xPlotAlongSection, yPlotAlongSection = (
        DetermineDiagonalCoordinates(DiagonalType,lX,lY,nPlotAlongSection))        
    elif (problem_type == 'Equatorial_Kelvin_Wave' or problem_type == 'Equatorial_Yanai_Wave' 
          or problem_type == 'Equatorial_Rossby_Wave' or problem_type == 'Equatorial_Inertia_Gravity_Wave'
          or problem_type == 'Barotropic_Tide' or problem_type == 'Diffusion_Equation' 
          or problem_type == 'Viscous_Burgers_Equation'):
        rPlotAlongSection, xPlotAlongSection, yPlotAlongSection = (
        DetermineCoordinatesAlongZonalSection(lX,nPlotAlongSection,y))
    return rPlotAlongSection, xPlotAlongSection, yPlotAlongSection


# In[71]:

def ComputeGeophysicalWaveExactStateVariablesAlongSection(problem_type,ExactSolutionParameters,xPlotAlongSection,
                                                          yPlotAlongSection,time):
    nPlotAlongSection = len(xPlotAlongSection) - 1    
    if problem_type == 'Viscous_Burgers_Equation':
        GeophysicalWaveExactStateVariablesAlongSection = np.zeros(nPlotAlongSection+1)
        for iPlotAlongSection in range(0,nPlotAlongSection+1): 
            GeophysicalWaveExactStateVariablesAlongSection[iPlotAlongSection] = (
            DetermineGeophysicalWaveExactZonalVelocity(problem_type,ExactSolutionParameters,
                                                       xPlotAlongSection[iPlotAlongSection],
                                                       yPlotAlongSection[iPlotAlongSection],time))
    elif problem_type == 'Barotropic_Tide':
        GeophysicalWaveExactStateVariablesAlongSection = np.zeros((7,nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1): 
            [GeophysicalWaveExactStateVariablesAlongSection[0,iPlotAlongSection],
             GeophysicalWaveExactStateVariablesAlongSection[1,iPlotAlongSection],
             GeophysicalWaveExactStateVariablesAlongSection[2,iPlotAlongSection],
             GeophysicalWaveExactStateVariablesAlongSection[3,iPlotAlongSection],
             GeophysicalWaveExactStateVariablesAlongSection[4,iPlotAlongSection],
             GeophysicalWaveExactStateVariablesAlongSection[5,iPlotAlongSection],
             GeophysicalWaveExactStateVariablesAlongSection[6,iPlotAlongSection]] = (
            DetermineGeophysicalWaveExactSurfaceElevations(problem_type,ExactSolutionParameters,
                                                           xPlotAlongSection[iPlotAlongSection],
                                                           yPlotAlongSection[iPlotAlongSection],time))
    else:
        GeophysicalWaveExactStateVariablesAlongSection = np.zeros((3,nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            if problem_type == 'Diffusion_Equation':
                [GeophysicalWaveExactStateVariablesAlongSection[0,iPlotAlongSection],
                 GeophysicalWaveExactStateVariablesAlongSection[1,iPlotAlongSection],
                 GeophysicalWaveExactStateVariablesAlongSection[2,iPlotAlongSection]] = (
                DetermineGeophysicalWaveExactZonalVelocities(problem_type,ExactSolutionParameters,
                                                             xPlotAlongSection[iPlotAlongSection],
                                                             yPlotAlongSection[iPlotAlongSection],time))
            elif (problem_type == 'Equatorial_Yanai_Wave' or problem_type == 'Equatorial_Rossby_Wave'
                  or problem_type == 'Equatorial_Inertia_Gravity_Wave'):
                [GeophysicalWaveExactStateVariablesAlongSection[0,iPlotAlongSection],
                 GeophysicalWaveExactStateVariablesAlongSection[1,iPlotAlongSection],
                 GeophysicalWaveExactStateVariablesAlongSection[2,iPlotAlongSection]] = (
                DetermineGeophysicalWaveExactMeridionalVelocities(problem_type,ExactSolutionParameters,
                                                                  xPlotAlongSection[iPlotAlongSection],
                                                                  yPlotAlongSection[iPlotAlongSection],time))
            else:
                [GeophysicalWaveExactStateVariablesAlongSection[0,iPlotAlongSection],
                 GeophysicalWaveExactStateVariablesAlongSection[1,iPlotAlongSection],
                 GeophysicalWaveExactStateVariablesAlongSection[2,iPlotAlongSection]] = (
                DetermineGeophysicalWaveExactSurfaceElevations(problem_type,ExactSolutionParameters,
                                                               xPlotAlongSection[iPlotAlongSection],
                                                               yPlotAlongSection[iPlotAlongSection],time))         
    return GeophysicalWaveExactStateVariablesAlongSection


# In[72]:

def WriteGeophysicalWaveExactStateVariablesAlongSectionToFile(problem_type,output_directory,rPlotAlongSection,
                                                              GeophysicalWaveExactStateVariablesAlongSection,
                                                              filename):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    nPlotAlongSection = len(rPlotAlongSection) - 1
    filename = filename + '.curve'
    outputfile = open(filename,'w')
    outputfile.write('#phi\n')
    if problem_type == 'Viscous_Burgers_Equation':
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            outputfile.write('%.15g %.15g\n' 
                             %(rPlotAlongSection[iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[iPlotAlongSection]))           
    elif problem_type == 'Barotropic_Tide':    
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            outputfile.write('%.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n' 
                             %(rPlotAlongSection[iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[0,iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[1,iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[2,iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[3,iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[4,iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[5,iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[6,iPlotAlongSection]))    
    else:
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            outputfile.write('%.15g %.15g %.15g %.15g\n' 
                             %(rPlotAlongSection[iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[0,iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[1,iPlotAlongSection],
                               GeophysicalWaveExactStateVariablesAlongSection[2,iPlotAlongSection]))
    outputfile.close()
    os.chdir(cwd)


# In[73]:

def ReadGeophysicalWaveExactStateVariablesAlongSectionFromFile(problem_type,output_directory,filename):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    data = [];
    count = 0;
    with open(filename,'r') as infile:
        for line in infile:
            if count != 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    nPlotAlongSection = data.shape[0] - 1
    rPlotAlongSection = np.zeros(nPlotAlongSection+1)
    if problem_type == 'Viscous_Burgers_Equation':
        GeophysicalWaveExactStateVariablesAlongSection = np.zeros((nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            rPlotAlongSection[iPlotAlongSection] = data[iPlotAlongSection,0]
            GeophysicalWaveExactStateVariablesAlongSection[iPlotAlongSection] = data[iPlotAlongSection,1]      
    elif problem_type == 'Barotropic_Tide':    
        GeophysicalWaveExactStateVariablesAlongSection = np.zeros((7,nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            rPlotAlongSection[iPlotAlongSection] = data[iPlotAlongSection,0]
            GeophysicalWaveExactStateVariablesAlongSection[0,iPlotAlongSection] = data[iPlotAlongSection,1]
            GeophysicalWaveExactStateVariablesAlongSection[1,iPlotAlongSection] = data[iPlotAlongSection,2]
            GeophysicalWaveExactStateVariablesAlongSection[2,iPlotAlongSection] = data[iPlotAlongSection,3] 
            GeophysicalWaveExactStateVariablesAlongSection[3,iPlotAlongSection] = data[iPlotAlongSection,4]
            GeophysicalWaveExactStateVariablesAlongSection[4,iPlotAlongSection] = data[iPlotAlongSection,5]
            GeophysicalWaveExactStateVariablesAlongSection[5,iPlotAlongSection] = data[iPlotAlongSection,6]
            GeophysicalWaveExactStateVariablesAlongSection[6,iPlotAlongSection] = data[iPlotAlongSection,7]
    else:
        GeophysicalWaveExactStateVariablesAlongSection = np.zeros((3,nPlotAlongSection+1))
        for iPlotAlongSection in range(0,nPlotAlongSection+1):
            rPlotAlongSection[iPlotAlongSection] = data[iPlotAlongSection,0]
            GeophysicalWaveExactStateVariablesAlongSection[0,iPlotAlongSection] = data[iPlotAlongSection,1]
            GeophysicalWaveExactStateVariablesAlongSection[1,iPlotAlongSection] = data[iPlotAlongSection,2]
            GeophysicalWaveExactStateVariablesAlongSection[2,iPlotAlongSection] = data[iPlotAlongSection,3]
    os.chdir(cwd)
    return GeophysicalWaveExactStateVariablesAlongSection


# In[74]:

def PlotGeophysicalWaveExactStateVariablesAlongSectionSaveAsPNG(
problem_type,output_directory,rPlotAlongSection,GeophysicalWaveExactStateVariablesAlongSection,
StateVariableLimitsAlongSection,linewidths,linestyles,colors,labels,labelfontsizes,labelpads,tickfontsizes,legends,
legendfontsize,legendposition,title,titlefontsize,SaveAsPNG,FigureTitle,Show,fig_size=[9.25,9.25],
legendWithinBox=False,legendpads=[1.0,0.5],titlepad=1.035,problem_type_Equatorial_Wave=False):
    cwd = CR.CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)   
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    x = rPlotAlongSection
    y = GeophysicalWaveExactStateVariablesAlongSection
    if problem_type == 'Viscous_Burgers_Equation':
        ax.plot(x[:],y[:],linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0])
    elif problem_type == 'Barotropic_Tide':
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
    if not(problem_type == 'Viscous_Burgers_Equation'):
        if legendWithinBox:
            ax.legend(fontsize=legendfontsize,loc=legendposition,shadow=True) 
        else:
            ax.legend(fontsize=legendfontsize,loc=legendposition,bbox_to_anchor=(legendpads[0],legendpads[1]),
                      shadow=True) 
    ax.set_title(title,fontsize=titlefontsize,y=titlepad)
    if problem_type_Equatorial_Wave:
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)