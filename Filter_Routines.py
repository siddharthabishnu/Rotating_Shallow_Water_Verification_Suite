
# coding: utf-8

# Name: Filter_Routines.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains routines for determining the primary and secondary weights of various barotropic time-averaging filters. 

# In[1]:

import numpy as np
import os
from IPython.utils import io
with io.capture_output() as captured:
    import Common_Routines as CR


# In[2]:

def ShchepetkinShapeFunction(A0,tau0,tau,p,q,r):
    A = A0*((tau/tau0)**p*(1.0 - (tau/tau0)**q) - r*(tau/tau0))
    return A


# In[3]:

def ShchepetkinShapeFunctionDerivative(A0,tau0,tau,p,q,r): 
    A_tau = A0*(p*tau**(p - 1.0)/tau0**p - (p + q)*tau**(p + q - 1.0)/tau0**(p + q) - r/tau0)
    return A_tau


# In[4]:

def GivenFunction(p,q,xVector,ShchepetkinFilterType):
    fVector = np.zeros(xVector.shape[0])
    tauStar = xVector[0]
    tau0 = xVector[1]
    A0 = xVector[2]
    r = xVector[3]
    ATauStar = ShchepetkinShapeFunction(A0,tau0,tauStar,p,q,r)
    I0 = A0*(tauStar**(p+1.0)/(tau0**p*(p+1.0)) - tauStar**(p+q+1.0)/(tau0**(p+q)*(p+q+1.0)) 
             - r*tauStar**2.0/(2.0*tau0))
    I1 = A0*(tauStar**(p+2.0)/(tau0**p*(p+2.0)) - tauStar**(p+q+2.0)/(tau0**(p+q)*(p+q+2.0)) 
             - r*tauStar**3.0/(3.0*tau0))
    I2 = A0*(tauStar**(p+3.0)/(tau0**p*(p+3.0)) - tauStar**(p+q+3.0)/(tau0**(p+q)*(p+q+3.0)) 
             - r*tauStar**4.0/(4.0*tau0))
    I3 = A0*(tauStar**(p+4.0)/(tau0**p*(p+4.0)) - tauStar**(p+q+4.0)/(tau0**(p+q)*(p+q+4.0)) 
             - r*tauStar**5.0/(5.0*tau0))
    fVector[0] = ATauStar - 0.0
    fVector[1] = I0 - 1.0
    fVector[2] = I1 - 1.0
    if ShchepetkinFilterType == 'SecondOrderAccurate':
        fVector[3] = I2 - 1.0
    else: # if ShchepetkinFilterType == 'MinimalDispersion':
        fVector[3] = I3 - 3.0*I2 + 2.0
    return fVector


# In[5]:

def GivenFunctionGradient(p,q,xVector,ShchepetkinFilterType):
    fVector = np.zeros(xVector.shape[0])
    GradFVector = np.zeros((xVector.shape[0],xVector.shape[0]))
    fVector = GivenFunction(p,q,xVector,ShchepetkinFilterType)
    tauStar = xVector[0]
    tau0 = xVector[1]
    A0 = xVector[2]
    r = xVector[3]
    I2Prime = np.zeros(4)
    I3Prime = np.zeros(4)
    GradFVector[0,0] = A0*(p*tauStar**(p-1.0)/tau0**p - (p+q)*tauStar**(p+q-1.0)/tau0**(p+q) - r/tau0)
    GradFVector[0,1] = A0*(-p*tauStar**p/tau0**(p+1.0) + (p+q)*tauStar**(p+q)/tau0**(p+q+1.0) 
                           + r*tauStar/tau0**2.0)
    GradFVector[0,2] = ShchepetkinShapeFunction(A0,tau0,tauStar,p,q,r)/A0
    GradFVector[0,3] = -A0*tauStar/tau0
    GradFVector[1,0] = ShchepetkinShapeFunction(A0,tau0,tauStar,p,q,r)
    GradFVector[1,1] = A0*(-p*tauStar**(p+1.0)/(tau0**(p+1.0)*(p+1.0)) 
                           + (p+q)*tauStar**(p+q+1.0)/(tau0**(p+q+1.0)*(p+q+1.0)) + r*tauStar**2.0/(2.0*tau0**2.0))
    GradFVector[1,2] = (tauStar**(p+1.0)/(tau0**p*(p+1.0)) - tauStar**(p+q+1.0)/(tau0**(p+q)*(p+q+1.0)) 
                        - r*tauStar**2.0/(2.0*tau0))         
    GradFVector[1,3] = -A0*tauStar**2.0/(2.0*tau0)
    GradFVector[2,0] = tauStar*ShchepetkinShapeFunction(A0,tau0,tauStar,p,q,r)
    GradFVector[2,1] = A0*(-p*tauStar**(p+2.0)/(tau0**(p+1.0)*(p+2.0)) 
                           + (p+q)*tauStar**(p+q+2.0)/(tau0**(p+q+1.0)*(p+q+2.0)) 
                           + r*tauStar**3.0/(3.0*tau0**2.0))
    GradFVector[2,2] = (tauStar**(p+2.0)/(tau0**p*(p+2.0)) - tauStar**(p+q+2.0)/(tau0**(p+q)*(p+q+2.0)) 
                        - r*tauStar**3.0/(3.0*tau0))             
    GradFVector[2,3] = -A0*tauStar**3.0/(3.0*tau0)
    I2 = A0*(tauStar**(p+3.0)/(tau0**p*(p+3.0)) - tauStar**(p+q+3.0)/(tau0**(p+q)*(p+q+3.0)) 
             - r*tauStar**4.0/(4.0*tau0))
    I3 = A0*(tauStar**(p+4.0)/(tau0**p*(p+4.0)) - tauStar**(p+q+4.0)/(tau0**(p+q)*(p+q+4.0)) 
             - r*tauStar**5.0/(5.0*tau0))    
    I2Prime[0] = tauStar**2.0*ShchepetkinShapeFunction(A0,tau0,tauStar,p,q,r)
    I2Prime[1] = A0*(-p*tauStar**(p+3.0)/(tau0**(p+1.0)*(p+3.0)) 
                     + (p+q)*tauStar**(p+q+3.0)/(tau0**(p+q+1.0)*(p+q+3.0)) + r*tauStar**4.0/(4.0*tau0**2.0))
    I2Prime[2] = (tauStar**(p+3.0)/(tau0**p*(p+3.0)) - tauStar**(p+q+3.0)/(tau0**(p+q)*(p+q+3.0))
                  - r*tauStar**4.0/(4.0*tau0))             
    I2Prime[3] = -A0*tauStar**4.0/(4.0*tau0)   
    I3Prime[0] = tauStar**3.0*ShchepetkinShapeFunction(A0,tau0,tauStar,p,q,r)
    I3Prime[1] = A0*(-p*tauStar**(p+4.0)/(tau0**(p+1.0)*(p+4.0)) 
                     + (p+q)*tauStar**(p+q+4.0)/(tau0**(p+q+1.0)*(p+q+4.0)) + r*tauStar**5.0/(5.0*tau0**2.0))
    I3Prime[2] = (tauStar**(p+4.0)/(tau0**p*(p+4.0)) - tauStar**(p+q+4.0)/(tau0**(p+q)*(p+q+4.0))
                  - r*tauStar**5.0/(5.0*tau0))             
    I3Prime[3] = -A0*tauStar**5.0/(5.0*tau0)   
    if ShchepetkinFilterType == 'SecondOrderAccurate':
        GradFVector[3,:] = I2Prime[:]
    else: # if ShchepetkinFilterType == 'MinimalDispersion':   
        GradFVector[3,:] = I3Prime[:] - 3.0*I2Prime[:]   
    return GradFVector


# In[6]:

def NewtonRaphsonSolverForDeterminingShchepetkinFilterParameters(p,q,ShchepetkinFilterType):
    nIterationsNewtonRaphson = 10**4
    NewtonRaphsonTolerance = 4.0*np.finfo(float).eps
    xVector = np.zeros(4)
    if ShchepetkinFilterType == 'SecondOrderAccurate':
        tauStar = 1.35
    else: # if ShchepetkinFilterType == 'MinimalDispersion':   
        tauStar = 1.49
    tau0 = (p + 2.0)*(p + q + 2.0)/((p + 1.0)*(p + q + 1.0))
    A0 = 1.0
    r = 0.0
    # This Newton-Raphson method is not ideal for solving this nonlinear system of equations with the variables
    # being The Shchepetkin filter parameters. This is because the problem is not well-posed. It has multiple 
    # solutions. If the initial guess is not close to our desired solution, the method may not converge at all and
    # even if it does, the converged solution will be different one. As a matter of fact, the initial guess for 
    # tauStar should be chosen very carefully since it is the most sensitive one. For the second order accurate 
    # filter, the method only works for initial guess of tauStar equal to 1.35 (not even 1.30 or 1.40). For the 
    # filter optimized for minimal numerical dispersion, the method only works for initial guess of tauStar equal 
    # to 1.49 (not even 1.48 or 1.50).
    xVector[0] = tauStar
    xVector[1] = tau0
    xVector[2] = A0
    xVector[3] = r
    converged = False
    for iIteration in range(0,nIterationsNewtonRaphson):  
        fVector = GivenFunction(p,q,xVector,ShchepetkinFilterType)
        GradFVector = GivenFunctionGradient(p,q,xVector,ShchepetkinFilterType)
        GradFVectorInverse = np.linalg.inv(GradFVector)
        DeltaX = -np.matmul(GradFVectorInverse,fVector)
        xVector += DeltaX
        DeltaXMaxNorm = np.linalg.norm(DeltaX,np.inf)
        xVectorMaxNorm = np.linalg.norm(xVector,np.inf)
        if DeltaXMaxNorm/xVectorMaxNorm <= NewtonRaphsonTolerance:
            converged = True
            nIterationsUptoConvergence = iIteration + 1
            break
    if converged:    
        print('The numerical solution for the Shchepetkin filter parameters has converged within %d iterations.' 
              %nIterationsUptoConvergence)
        print('They are as follows:')
        print('tauStar = %.6f' %xVector[0])
        print('tau0 = %.6f' %xVector[1])
        print('A0 = %.6f' %xVector[2])
        print('r = %.6f' %xVector[3])
    else:
        print('The numerical solution for the Shchepetkin filter parameters has not converged within',
              '%d iterations.' %nIterationsNewtonRaphson)
        print('The final values of the parameters are as follows:')
        print('tauStar = %.6f' %xVector[0])
        print('tau0 = %.6f' %xVector[1])
        print('A0 = %.6f' %xVector[2])
        print('r = %.6f' %xVector[3])     


# In[7]:

RunNewtonRaphsonSolverForDeterminingShchepetkinFilterParameters = False
if RunNewtonRaphsonSolverForDeterminingShchepetkinFilterParameters:
    NewtonRaphsonSolverForDeterminingShchepetkinFilterParameters(2.0,2.0,'SecondOrderAccurate')


# In[8]:

RunNewtonRaphsonSolverForDeterminingShchepetkinFilterParameters = False
if RunNewtonRaphsonSolverForDeterminingShchepetkinFilterParameters:
    NewtonRaphsonSolverForDeterminingShchepetkinFilterParameters(2.0,2.0,'MinimalDispersion')


# In[9]:

def NewtonRaphsonSolverForDeterminingShchepetkinFilterEndPoint(A0,tau0,p,q,r,tau,printResult):
    nIterationsNewtonRaphson = 10**2
    NewtonRaphsonTolerance = 4.0*np.finfo(float).eps
    converged = False
    for iIteration in range(0,nIterationsNewtonRaphson):  
        FunctionValue = ShchepetkinShapeFunction(A0,tau0,tau,p,q,r)
        DerivativeValue = ShchepetkinShapeFunctionDerivative(A0,tau0,tau,p,q,r)
        Delta = -FunctionValue/DerivativeValue
        tau += Delta
        if abs(Delta)/abs(tau) <= NewtonRaphsonTolerance:
            converged = True
            IterationsUptoConvergence = iIteration + 1
            break
    if printResult:
        if converged:    
            print('The numerical solution for the Shchepetkin filter endpoint has converged to',
                  '%.6f within %d iterations.' %(tau,IterationsUptoConvergence))
        else:
            print('The numerical solution for the Shchepetkin filter endpoint has not converged within',
                  '%d iterations.' %nIterationsNewtonRaphson)
    return converged, tau


# In[10]:

def TestNewtonRaphsonSolverForDeterminingShchepetkinFilterEndPoint(p,q):
    A0 = 1.0
    tau0 = (p + 2.0)*(p + q + 2.0)/((p + 1.0)*(p + q + 1.0))
    r = 0.25  
    tauStar = 1.5
    printResult = True
    converged, tauStar = NewtonRaphsonSolverForDeterminingShchepetkinFilterEndPoint(A0,tau0,p,q,r,tauStar,
                                                                                    printResult)


# In[11]:

Run_TestNewtonRaphsonSolverForDeterminingShchepetkinFilterEndPoint = False
if Run_TestNewtonRaphsonSolverForDeterminingShchepetkinFilterEndPoint:
    TestNewtonRaphsonSolverForDeterminingShchepetkinFilterEndPoint(2.0,2.0)


# In[12]:

def ShchepetkinFilterParameters_SecondOrderAccurate(p,q,DisplayOutputAtEachIterationLevel):
    tauStar = 1.5
    tau0 = (p + 2.0)*(p + q + 2.0)/((p + 1.0)*(p + q + 1.0))
    tau0Last = tau0
    A0 = 1.0
    r = 0.0
    rLast = r
    nIterationsNewtonRaphson = 10**3
    NewtonRaphsonTolerance = 4.0*np.finfo(float).eps 
    converged = False
    for iIteration in range(0,nIterationsNewtonRaphson): 
        convergedTauStar, tauStar = NewtonRaphsonSolverForDeterminingShchepetkinFilterEndPoint(A0,tau0,p,q,r,
                                                                                               tauStar,False)
        LHSFactor = A0*(tauStar**(p+1.0)/(tau0**p*(p+1.0)) - tauStar**(p+q+1.0)/(tau0**(p+q)*(p+q+1.0))
                        - r*tauStar**2.0/(2.0*tau0))
        A0 /= LHSFactor
        convergedTau0 = False
        for jIteration in range(0,nIterationsNewtonRaphson):  
            FunctionValueTau0 = A0*(tauStar**(p+2.0)/(tau0**p*(p+2.0)) - tauStar**(p+q+2.0)/(tau0**(p+q)*(p+q+2.0))
                                    - r*tauStar**3.0/(3.0*tau0)) - 1.0
            DerivativeValueTau0 = A0*(-p*tauStar**(p+2.0)/(tau0**(p+1.0)*(p+2.0))
                                      + (p+q)*tauStar**(p+q+2.0)/(tau0**(p+q+1.0)*(p+q+2.0))
                                      + r*tauStar**3.0/(3.0*tau0**2.0))
            DeltaTau0 = -FunctionValueTau0/DerivativeValueTau0
            tau0 += DeltaTau0
            if abs(DeltaTau0)/abs(tau0) <= NewtonRaphsonTolerance:
                convergedTau0 = True
                IterationsTau0UptoConvergence = jIteration + 1
                break
            tau0Last = tau0
        r = -(4.0*tau0/tauStar**4.0)*(1.0/A0 - tauStar**(p+3.0)/(tau0**p*(p+3.0)) 
                                      + tauStar**(p+q+3.0)/(tau0**(p+q)*(p+q+3.0)))
        DeltaTau0 = tau0 - tau0Last
        DeltaR = r - rLast
        if max(abs(DeltaTau0)/abs(tau0),abs(DeltaR)/abs(r)) <= NewtonRaphsonTolerance:
            converged = True
            nIterationsUptoConvergence = iIteration + 1
            break
        if DisplayOutputAtEachIterationLevel:
            print('The Shchepetkin filter parameters at iteration level %3d are %12.5f %12.5f %12.5f %12.5f.' 
                  %(iIteration+1,tauStar, tau0, A0, r))
        rLast = r
    if converged:    
        print('The numerical solution for the Shchepetkin filter parameters has converged within %d iterations.' 
              %nIterationsUptoConvergence)
        print('They are as follows:')
        print('tauStar = %.6f' %tauStar)
        print('tau0 = %.6f' %tau0)
        print('A0 = %.6f' %A0)
        print('r = %.6f' %r)
    else:
        print('The numerical solution for the Shchepetkin filter parameters has not converged within',
              '%d iterations.' %nIterationsNewtonRaphson)  
        print('The final values of the parameters are as follows:')
        print('tauStar = %.6f' %tauStar)
        print('tau0 = %.6f' %tau0)
        print('A0 = %.6f' %A0)
        print('r = %.6f' %r)
    return tauStar, tau0, A0, r


# In[13]:

DetermineShchepetkinFilterParameters_SecondOrderAccurate = False
if DetermineShchepetkinFilterParameters_SecondOrderAccurate:
    tauStar, tau0, A0, r = ShchepetkinFilterParameters_SecondOrderAccurate(2.0,2.0,False)


# In[14]:

def ShchepetkinFilterParameters_MinimalDispersion(p,q,DisplayOutputAtEachIterationLevel):
    tauStar = 1.5
    tau0 = (p + 2.0)*(p + q + 2.0)/((p + 1.0)*(p + q + 1.0))
    tau0Last = tau0
    A0 = 1.0
    r = 0.0
    rLast = r
    nIterationsNewtonRaphson = 10**4
    # It is observed that the iterative method for determining the Shchepetkin parameters does not converge to our
    # desired tolerance, close to machine precision. However, it does converge to a solution very close to the one
    # obtained by Shchepetkin. But the residual approximated by max(abs(DeltaTau0)/abs(tau0),abs(DeltaR)/abs(r))
    # is ~O(10.0**(-8.0)) after 1000 iterations and ~O(10.0**(-15.0)) after 10000 iterations. That is why we choose
    # nIterationsNewtonRaphson to be 10**4, not 10**3.
    NewtonRaphsonTolerance = 4.0*np.finfo(float).eps 
    converged = False
    for iIteration in range(0,nIterationsNewtonRaphson): 
        convergedTauStar, tauStar = (
        NewtonRaphsonSolverForDeterminingShchepetkinFilterEndPoint(A0,tau0,p,q,r,tauStar,False))
        LHSFactor = A0*(tauStar**(p+1.0)/(tau0**p*(p+1.0)) - tauStar**(p+q+1.0)/(tau0**(p+q)*(p+q+1.0))
                        - r*tauStar**2.0/(2.0*tau0))
        A0 /= LHSFactor
        convergedTau0 = False
        for jIteration in range(0,nIterationsNewtonRaphson):  
            FunctionValueTau0 = A0*(tauStar**(p+2.0)/(tau0**p*(p+2.0)) - tauStar**(p+q+2.0)/(tau0**(p+q)*(p+q+2.0))
                                    - r*tauStar**3.0/(3.0*tau0)) - 1.0
            DerivativeValueTau0 = A0*(-p*tauStar**(p+2.0)/(tau0**(p+1.0)*(p+2.0))
                                      + (p+q)*tauStar**(p+q+2.0)/(tau0**(p+q+1.0)*(p+q+2.0))
                                      + r*tauStar**3.0/(3.0*tau0**2.0))
            DeltaTau0 = -FunctionValueTau0/DerivativeValueTau0
            tau0 += DeltaTau0
            if abs(DeltaTau0)/abs(tau0) <= NewtonRaphsonTolerance:
                convergedTau0 = True
                IterationsTau0UptoConvergence = jIteration + 1
                break
            tau0Last = tau0
        LHS = A0*(-tauStar**5.0/(5.0*tau0) + 3.0*tauStar**4.0/(4.0*tau0))
        RHS = (3.0*A0*(tauStar**(p+3.0)/(tau0**p*(p+3.0)) - tauStar**(p+q+3.0)/(tau0**(p+q)*(p+q+3.0)))
               - A0*(tauStar**(p+4.0)/(tau0**p*(p+4.0)) - tauStar**(p+q+4.0)/(tau0**(p+q)*(p+q+4.0))) - 2.0)
        r = RHS/LHS
        DeltaTau0 = tau0 - tau0Last
        DeltaR = r - rLast
        if max(abs(DeltaTau0)/abs(tau0),abs(DeltaR)/abs(r)) <= NewtonRaphsonTolerance:
            converged = True
            nIterationsUptoConvergence = iIteration + 1
            break
        if DisplayOutputAtEachIterationLevel:
            print('The Shchepetkin filter parameters at iteration level %3d are %12.5f %12.5f %12.5f %12.5f.'
                  %(iIteration+1,tauStar, tau0, A0, r))
        rLast = r
    if converged:    
        print('The numerical solution for the Shchepetkin filter parameters has converged within %d iterations.' 
              %nIterationsUptoConvergence)
        print('They are as follows:')
        print('tauStar = %.6f' %tauStar)
        print('tau0 = %.6f' %tau0)
        print('A0 = %.6f' %A0)
        print('r = %.6f' %r)
    else:
        print('The numerical solution for the Shchepetkin filter parameters has not converged within',
              '%d iterations.' %nIterationsNewtonRaphson)  
        print('The final values of the parameters are as follows:')
        print('tauStar = %.6f' %tauStar)
        print('tau0 = %.6f' %tau0)
        print('A0 = %.6f' %A0)
        print('r = %.6f' %r)
    return tauStar, tau0, A0, r


# In[15]:

DetermineShchepetkinFilterParameters_MinimalDispersion = False
if DetermineShchepetkinFilterParameters_MinimalDispersion:
    tauStar, tau0, A0, r = ShchepetkinFilterParameters_MinimalDispersion(2.0,2.0,False)


# In[16]:

def DetermineFilterWeights(FilterType,RectangularFilterRange,pShchepetkin,qShchepetkin,nBarotropicTimeSteps):
    PrimaryWeights = np.zeros(2*nBarotropicTimeSteps)
    SecondaryWeights = np.zeros(2*nBarotropicTimeSteps)
    if FilterType == 'no_filter_type_1':
        PrimaryWeights[nBarotropicTimeSteps-1] = 1.0
    elif FilterType == 'no_filter_type_2':
        PrimaryWeights[nBarotropicTimeSteps-1] = 1.0
        SecondaryWeights[int(nBarotropicTimeSteps/2)-1] = 1.0  
        return PrimaryWeights, SecondaryWeights
    elif FilterType == 'rectangular':
        for iBarotropicTimeStep in range(0,2*nBarotropicTimeSteps):
            tau = float(iBarotropicTimeStep + 1)/float(nBarotropicTimeSteps)
            if tau > (1.0 - RectangularFilterRange) and tau <= (1.0 + RectangularFilterRange):
                PrimaryWeights[iBarotropicTimeStep] = 1.0/(2.0*RectangularFilterRange*float(nBarotropicTimeSteps))
    elif FilterType == 'ROMS_cosine':
        for iBarotropicTimeStep in range(0,2*nBarotropicTimeSteps):
            tau = float(iBarotropicTimeStep + 1)/float(nBarotropicTimeSteps)
            if tau > 0.5 and tau <= 1.5:
                PrimaryWeights[iBarotropicTimeStep] = 1.0 + np.cos(2.0*np.pi*(tau - 1.0))
        PrimaryWeights /= float(nBarotropicTimeSteps)
    elif FilterType == 'Hamming_window_cosine':
        for iBarotropicTimeStep in range(0,2*nBarotropicTimeSteps):
            tau = float(iBarotropicTimeStep + 1)/float(nBarotropicTimeSteps)
            if tau > 0.5 and tau <= 1.5:
                PrimaryWeights[iBarotropicTimeStep] = 1.0 + 0.85*np.cos(2.0*np.pi*(tau - 1.0))
        PrimaryWeights /= float(nBarotropicTimeSteps)
    elif FilterType == 'Shchepetkin_SecondOrderAccurate':
        tauStar, tau0, A0, r = ShchepetkinFilterParameters_SecondOrderAccurate(pShchepetkin,qShchepetkin,False)
        for iBarotropicTimeStep in range(0,2*nBarotropicTimeSteps):
            tau = float(iBarotropicTimeStep + 1)/float(nBarotropicTimeSteps)
            PrimaryWeights[iBarotropicTimeStep] = ShchepetkinShapeFunction(A0,tau0,tau,pShchepetkin,qShchepetkin,r)
            if tau > 1.0 and PrimaryWeights[iBarotropicTimeStep] < 0.0:
                PrimaryWeights[iBarotropicTimeStep] = 0.0
        PrimaryWeights /= float(nBarotropicTimeSteps)   
    elif FilterType == 'Shchepetkin_MinimalDispersion':
        tauStar, tau0, A0, r = ShchepetkinFilterParameters_MinimalDispersion(pShchepetkin,qShchepetkin,False)
        for iBarotropicTimeStep in range(0,2*nBarotropicTimeSteps):
            tau = float(iBarotropicTimeStep + 1)/float(nBarotropicTimeSteps)
            PrimaryWeights[iBarotropicTimeStep] = ShchepetkinShapeFunction(A0,tau0,tau,pShchepetkin,qShchepetkin,r)
            if tau > 1.0 and PrimaryWeights[iBarotropicTimeStep] < 0.0:
                PrimaryWeights[iBarotropicTimeStep] = 0.0
        PrimaryWeights /= float(nBarotropicTimeSteps)   
    for iBarotropicTimeStep in range(0,2*nBarotropicTimeSteps):
        for jBarotropicTimeStep in range(iBarotropicTimeStep,2*nBarotropicTimeSteps):
            SecondaryWeights[iBarotropicTimeStep] += PrimaryWeights[jBarotropicTimeStep]
        SecondaryWeights[iBarotropicTimeStep] /= float(nBarotropicTimeSteps)
    PrimaryWeights /= sum(PrimaryWeights)
    SecondaryWeights /= sum(SecondaryWeights)
    return PrimaryWeights, SecondaryWeights


# In[17]:

def NoFilterType1():
    FilterType = 'no_filter_type_1'
    nBarotropicTimeSteps = 30
    PrimaryWeights, SecondaryWeights = DetermineFilterWeights(FilterType,0.5,2.0,2.0,nBarotropicTimeSteps)
    print('With no filter of type 1, the sum of all primary weights is %.6f.' %(sum(PrimaryWeights)))
    print('With no filter of type 1, the sum of all secondary weights is %.6f.' %(sum(SecondaryWeights)))
    Filename = 'NoFilterType1Weights'
    CR.PythonPlots1DSaveAsPNG('MPAS_O_Shallow_Water_Output',np.linspace(1.0/float(nBarotropicTimeSteps),2.0,
                                                                        2*nBarotropicTimeSteps),PrimaryWeights,
                              SecondaryWeights,2.0,True,True,'Fraction of Baroclinic Time Step',10,'Weights',10,
                              'Primary Weights','Secondary Weights','center left',' ',True,7.5,True,Filename,True)


# In[18]:

run_NoFilterType1 = False
if run_NoFilterType1:
    NoFilterType1()


# In[19]:

def NoFilterType2():
    FilterType = 'no_filter_type_2'
    nBarotropicTimeSteps = 30
    PrimaryWeights, SecondaryWeights = DetermineFilterWeights(FilterType,0.5,2.0,2.0,nBarotropicTimeSteps)
    print('With no filter of type 2, the sum of all primary weights is %.6f.' %(sum(PrimaryWeights)))
    print('With no filter of type 2, the sum of all secondary weights is %.6f.' %(sum(SecondaryWeights)))
    Filename = 'NoFilterType2Weights'
    CR.PythonPlots1DSaveAsPNG('MPAS_O_Shallow_Water_Output',np.linspace(1.0/float(nBarotropicTimeSteps),2.0,
                                                                        2*nBarotropicTimeSteps),PrimaryWeights,
                              SecondaryWeights,2.0,True,True,'Fraction of Baroclinic Time Step',10,'Weights',10,
                              'Primary Weights','Secondary Weights','center left',' ',True,7.5,True,Filename,True)


# In[20]:

run_NoFilterType2 = False
if run_NoFilterType2:
    NoFilterType2()


# In[21]:

def RectangularFilter(RectangularFilterRange):
    FilterType = 'rectangular'
    nBarotropicTimeSteps = 30
    PrimaryWeights, SecondaryWeights = DetermineFilterWeights(FilterType,RectangularFilterRange,2.0,2.0,
                                                              nBarotropicTimeSteps)
    print('The sum of all primary weights of the rectangular filter with range %.2f is %.6f.'
          %(RectangularFilterRange,sum(PrimaryWeights)))
    print('The sum of all secondary weights of the rectangular filter with range %.2f is %.6f.'
          %(RectangularFilterRange,sum(SecondaryWeights)))
    Filename = 'RectangularFilterWeights'+str(RectangularFilterRange)
    CR.PythonPlots1DSaveAsPNG('MPAS_O_Shallow_Water_Output',np.linspace(1.0/float(nBarotropicTimeSteps),2.0,
                                                                        2*nBarotropicTimeSteps),PrimaryWeights,
                              SecondaryWeights,2.0,True,True,'Fraction of Baroclinic Time Step',10,'Weights',10,
                              'Primary Weights','Secondary Weights','center left',' ',True,7.5,True,Filename,True)


# In[22]:

run_RectangularFilterRangeOne = False
if run_RectangularFilterRangeOne:
    RectangularFilter(1.0)


# In[23]:

run_RectangularFilterRangeHalf = False
if run_RectangularFilterRangeHalf:
    RectangularFilter(0.5)


# In[24]:

def ROMSCosineFilter():
    FilterType = 'ROMS_cosine'
    nBarotropicTimeSteps = 30
    PrimaryWeights, SecondaryWeights = DetermineFilterWeights(FilterType,0.5,2.0,2.0,nBarotropicTimeSteps)
    print('The sum of all primary weights of the ROMS cosine filter is %.6f.' %sum(PrimaryWeights))
    print('The sum of all secondary weights of the ROMS cosine filter is %.6f.' %sum(SecondaryWeights))
    Filename = 'ROMSCosineFilterWeights'
    CR.PythonPlots1DSaveAsPNG('MPAS_O_Shallow_Water_Output',np.linspace(1.0/float(nBarotropicTimeSteps),2.0,
                                                                        2*nBarotropicTimeSteps),PrimaryWeights,
                              SecondaryWeights,2.0,True,True,'Fraction of Baroclinic Time Step',10,'Weights',10,
                              'Primary Weights','Secondary Weights','center left',' ',True,7.5,True,Filename,True)


# In[25]:

run_ROMSCosineFilter = False
if run_ROMSCosineFilter:
    ROMSCosineFilter()


# In[26]:

def HammingWindowCosineFilter():
    FilterType = 'Hamming_window_cosine'
    nBarotropicTimeSteps = 30
    PrimaryWeights, SecondaryWeights = DetermineFilterWeights(FilterType,0.5,2.0,2.0,nBarotropicTimeSteps)
    print('The sum of all primary weights of the Hamming window cosine filter is %.6f.' %sum(PrimaryWeights))
    print('The sum of all secondary weights of the Hamming window cosine filter is %.6f.' %sum(SecondaryWeights))
    Filename = 'HammingWindowCosineFilterWeights'
    CR.PythonPlots1DSaveAsPNG('MPAS_O_Shallow_Water_Output',np.linspace(1.0/float(nBarotropicTimeSteps),2.0,
                                                                        2*nBarotropicTimeSteps),PrimaryWeights,
                              SecondaryWeights,2.0,True,True,'Fraction of Baroclinic Time Step',10,'Weights',10,
                              'Primary Weights','Secondary Weights','center left',' ',True,7.5,True,Filename,True)


# In[27]:

run_HammingWindowCosineFilter = False
if run_HammingWindowCosineFilter:
    HammingWindowCosineFilter()


# In[28]:

def ShchepetkinFilter_SecondOrderAccurate(pShchepetkin,qShchepetkin):
    FilterType = 'Shchepetkin_SecondOrderAccurate'
    nBarotropicTimeSteps = 30
    PrimaryWeights, SecondaryWeights = DetermineFilterWeights(FilterType,0.5,pShchepetkin,qShchepetkin,
                                                              nBarotropicTimeSteps)
    print('The sum of all primary weights of the second order accurate Shchepetkin filter is %.6f.'
          %sum(PrimaryWeights))
    print('The sum of all secondary weights of the second order accurate Shchepetkin filter is %.6f.'
          %sum(SecondaryWeights))
    Filename = 'ShchepetkinFilterWeights_SecondOrderAccurate'
    CR.PythonPlots1DSaveAsPNG('MPAS_O_Shallow_Water_Output',np.linspace(1.0/float(nBarotropicTimeSteps),2.0,
                                                                        2*nBarotropicTimeSteps),PrimaryWeights,
                              SecondaryWeights,2.0,True,True,'Fraction of Baroclinic Time Step',10,'Weights',10,
                              'Primary Weights','Secondary Weights','center left',' ',True,7.5,True,Filename,True)


# In[29]:

run_ShchepetkinFilter_SecondOrderAccurate_p2_q2 = False
if run_ShchepetkinFilter_SecondOrderAccurate_p2_q2:
    ShchepetkinFilter_SecondOrderAccurate(2.0,2.0)


# In[30]:

run_ShchepetkinFilter_SecondOrderAccurate_p2_q4 = False
if run_ShchepetkinFilter_SecondOrderAccurate_p2_q4:
    ShchepetkinFilter_SecondOrderAccurate(2.0,4.0)


# In[31]:

def ShchepetkinFilter_MinimalDispersion(pShchepetkin,qShchepetkin):
    FilterType = 'Shchepetkin_MinimalDispersion'
    nBarotropicTimeSteps = 30
    PrimaryWeights, SecondaryWeights = DetermineFilterWeights(FilterType,0.5,pShchepetkin,qShchepetkin,
                                                              nBarotropicTimeSteps)
    print('The sum of all primary weights of the Shchepetkin filter optimized for minimal dispersion is %.6f.'
          %sum(PrimaryWeights))
    print('The sum of all secondary weights of the Shchepetkin filter optimized for minimal dispersion is %.6f.'
          %sum(SecondaryWeights))
    Filename = 'ShchepetkinFilterWeights_MinimalDispersion'
    CR.PythonPlots1DSaveAsPNG('MPAS_O_Shallow_Water_Output',np.linspace(1.0/float(nBarotropicTimeSteps),2.0,
                                                                        2*nBarotropicTimeSteps),PrimaryWeights,
                              SecondaryWeights,2.0,True,True,'Fraction of Baroclinic Time Step',10,'Weights',10,
                              'Primary Weights','Secondary Weights','center left',' ',True,7.5,True,Filename,True)


# In[32]:

run_ShchepetkinFilter_MinimalDispersion_p2_q2 = False
if run_ShchepetkinFilter_MinimalDispersion_p2_q2:
    ShchepetkinFilter_MinimalDispersion(2.0,2.0)


# In[33]:

run_ShchepetkinFilter_MinimalDispersion_p2_q4 = False
if run_ShchepetkinFilter_MinimalDispersion_p2_q4:
    ShchepetkinFilter_MinimalDispersion(2.0,4.0)