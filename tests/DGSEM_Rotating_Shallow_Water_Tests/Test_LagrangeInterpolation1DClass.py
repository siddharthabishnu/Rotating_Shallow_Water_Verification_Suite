"""
Name: Test_LagrangeInterpolation1DClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the one-dimensional Lagrange interpolation class defined in
../../src/DGSEM_Rotating_Shallow_Water/LagrangeInterpolation1DClass.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR
    import LegendreGaussQuadrature1DClass as LGQ1D
    import LagrangeInterpolation1DClass as LI1D


def TestLagrangeInterpolationFunction1D():
    nX = 14
    xData = np.linspace(0.0,2.0*np.pi,nX+1)
    fData = np.sin(xData)
    xInterpolatingPoint = np.pi/6.0
    fInterpolatingPoint = LI1D.LagrangeInterpolationFunction1D(xData,fData,xInterpolatingPoint)
    print('The exact solution of sin(x) at x = pi/6 is %.15f.' %np.sin(xInterpolatingPoint))
    print('The solution of sin(x) at x = pi/6 obtained using Lagrange interpolation is %.15f.' %fInterpolatingPoint)
    
    
do_TestLagrangeInterpolationFunction1D = False
if do_TestLagrangeInterpolationFunction1D:
    TestLagrangeInterpolationFunction1D()


def TestEvaluateLagrangeInterpolant1D():
    nX = 14
    xData = np.linspace(0.0,2.0*np.pi,nX+1)
    fData = np.sin(xData)
    xInterpolatingPoint = np.pi/6.0
    myLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xData)
    fInterpolatingPoint = myLagrangeInterpolation1D.EvaluateLagrangeInterpolant1D(fData,xInterpolatingPoint)
    print('The exact solution of sin(x) at x = pi/6 is %.15f.' %np.sin(xInterpolatingPoint))
    print('The solution of sin(x) at x = pi/6 obtained using Lagrange interpolation is %.15f.' %fInterpolatingPoint)


do_TestEvaluateLagrangeInterpolant1D = False
if do_TestEvaluateLagrangeInterpolant1D:
    TestEvaluateLagrangeInterpolant1D()
    
    
def TestEvaluateLagrangeInterpolatingPolynomialsAtPoint1D():
    nX = 14
    xData = np.linspace(0.0,2.0*np.pi,nX+1)
    fData = np.sin(xData)
    xInterpolatingPoint = np.pi/6.0
    myLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xData)
    myLagrangeInterpolatingPolynomialsAtPoint = (
    myLagrangeInterpolation1D.EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(xInterpolatingPoint))
    fInterpolatingPoint = np.dot(myLagrangeInterpolatingPolynomialsAtPoint,fData)
    print('The exact solution of sin(x) at x = pi/6 is %.15f.' %np.sin(xInterpolatingPoint))
    print('The solution of sin(x) at x = pi/6 obtained using Lagrange interpolation is %.15f.' %fInterpolatingPoint)


do_TestEvaluateLagrangeInterpolatingPolynomialsAtPoint1D = False
if do_TestEvaluateLagrangeInterpolatingPolynomialsAtPoint1D:
    TestEvaluateLagrangeInterpolatingPolynomialsAtPoint1D()
    
    
def TestEvaluateLagrangeInterpolatingPolynomials1D():
    x = np.array([0.0,1.0,2.0])
    myLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(x)
    LagrangeInterpolatingPolynomials = myLagrangeInterpolation1D.EvaluateLagrangeInterpolatingPolynomials1D()
    print('Exact Solution:')
    print('The Lagrange interpolating polynomial with index 1 is')
    print('0.5*x**2 - 1.5*x + 1.0')
    print('The Lagrange interpolating polynomial with index 2 is')
    print('-1.0*x**2 + 2.0*x')
    print('The Lagrange interpolating polynomial with index 3 is')
    print('0.5*x**2 - 0.5*x\n')
    print('Numerical Solution:')
    for iX in range(0,np.size(x)):
        print('The Lagrange interpolating polynomial with index %d is' %(iX+1))
        print(LagrangeInterpolatingPolynomials[iX])
    
    
do_TestEvaluateLagrangeInterpolatingPolynomials1D = False
if do_TestEvaluateLagrangeInterpolatingPolynomials1D:
    TestEvaluateLagrangeInterpolatingPolynomials1D()


def TestEvaluateLagrangeInterpolantDerivative1D():
    nX = 14
    xData = np.linspace(0.0,2.0*np.pi,nX+1)
    fData = np.sin(xData)
    xInterpolatingPoint = np.pi/3.0
    myLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xData)
    fPrimeInterpolatingPoint = (
    myLagrangeInterpolation1D.EvaluateLagrangeInterpolantDerivative1D(fData,xInterpolatingPoint))
    print('The exact solution of d/dx (sin(x)) = cos(x) at x = pi/3 is %.15f.' %np.cos(xInterpolatingPoint))
    print('The solution of d/dx (sin(x)) at x = pi/3 obtained using Lagrange interpolation is %.15f.' 
          %fPrimeInterpolatingPoint)


do_TestEvaluateLagrangeInterpolantDerivative1D = False
if do_TestEvaluateLagrangeInterpolantDerivative1D:
    TestEvaluateLagrangeInterpolantDerivative1D()


def TestEvaluateLagrangePolynomialDerivative1D():
    nX = 14
    xData = np.linspace(0.0,2.0*np.pi,nX+1)
    fData = np.sin(xData)
    fDataDerivative = np.cos(xData)
    myLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xData)
    LagrangePolynomialDerivativeMatrix = myLagrangeInterpolation1D.EvaluateLagrangePolynomialDerivativeMatrix1D()
    LagrangePolynomialDerivative = (
    myLagrangeInterpolation1D.EvaluateLagrangePolynomialDerivative1D(LagrangePolynomialDerivativeMatrix,fData))
    for iX in range(0,myLagrangeInterpolation1D.nX+1):
        print('At iX = %2d, the exact derivative is %+.9f.' %(iX,fDataDerivative[iX]))
        print('At iX = %2d, the derivative obtained using Lagrange interpolation is %+.9f.' 
              %(iX,LagrangePolynomialDerivative[iX]))


do_TestEvaluateLagrangePolynomialDerivative1D = False
if do_TestEvaluateLagrangePolynomialDerivative1D:
    TestEvaluateLagrangePolynomialDerivative1D()


def TestEvaluate_mth_Order_LagrangePolynomialDerivative1D(m=2):
    nX = 14
    xData = np.linspace(0.0,2.0*np.pi,nX+1)
    fData = np.sin(xData)
    fDataDoubleDerivative = -np.sin(xData)
    myLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xData)
    mthOrderLagrangePolynomialDerivativeMatrix = (
    myLagrangeInterpolation1D.Evaluate_mth_Order_LagrangePolynomialDerivativeMatrix1D(m))
    mthOrderLagrangePolynomialDerivative = (
    myLagrangeInterpolation1D.Evaluate_mth_Order_LagrangePolynomialDerivative1D(
    mthOrderLagrangePolynomialDerivativeMatrix,fData))
    for iX in range(0,myLagrangeInterpolation1D.nX+1):
        print('At iX = %2d, the exact double derivative is %+.9f.' %(iX,fDataDoubleDerivative[iX]))
        print('At iX = %2d, the double derivative obtained using Lagrange interpolation is %+.9f.' 
              %(iX,mthOrderLagrangePolynomialDerivative[iX]))


do_TestEvaluate_mth_Order_LagrangePolynomialDerivative1D = False
if do_TestEvaluate_mth_Order_LagrangePolynomialDerivative1D:
    TestEvaluate_mth_Order_LagrangePolynomialDerivative1D()


def TestInterpolateToNewPoints1D():
    nXOld = 8
    nXNew = 14
    xLeft = 0.0
    xRight = 2.0*np.pi
    xOld = np.linspace(xLeft,xRight,nXOld+1)
    fOld = np.sin(xOld)
    myOldLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xOld)
    xNew = np.linspace(xLeft,xRight,nXNew+1)
    fNewExact = np.sin(xNew)
    myNewLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xNew)
    myLagrangePolynomialInterpolationMatrix = (
    myOldLagrangeInterpolation1D.EvaluateLagrangePolynomialInterpolationMatrix1D(myNewLagrangeInterpolation1D))
    fNew = LI1D.InterpolateToNewPoints1D(myLagrangePolynomialInterpolationMatrix,fOld)
    for iXNew in range(0,nXNew+1):
        print('At iXNew = %2d, the exact solution is %+.9f.' %(iXNew,fNewExact[iXNew]))
        print('At iXNew = %2d, the solution obtained using Lagrange interpolation is %+.9f.' %(iXNew,fNew[iXNew]))


do_TestInterpolateToNewPoints1D = False
if do_TestInterpolateToNewPoints1D:
    TestInterpolateToNewPoints1D()
    
    
def PolynomialTestFunction(x,polynomialOrder=21,returnDerivatives=False,derivativeOrder=1):
    function = 0.0
    for i in range(0,polynomialOrder+1):
        function += x**float(i)
    return function


def PolynomialTestFunctionDerivative(x,polynomialOrder=21,derivativeOrder=1):
    functionDerivative = 0.0
    if derivativeOrder == 1:
        if polynomialOrder >= 1:
            for i in range(1,polynomialOrder+1):
                functionDerivative += float(i)*x**(float(i)-1.0)
    elif derivativeOrder == 2: 
        if polynomialOrder >= 2:
            for i in range(2,polynomialOrder+1):
                functionDerivative += float(i*(i-1))*x**(float(i)-2.0)
    return functionDerivative


def TrigonometricTestFunction(x):
    function = np.cos(np.pi*(x+1.0))
    return function


def TrigonometricTestFunctionDerivative(x,derivativeOrder=1):
    if derivativeOrder == 1:
        functionDerivative = -np.pi*np.sin(np.pi*(x+1.0))
    elif derivativeOrder == 2: 
        functionDerivative = -(np.pi)**2.0*np.cos(np.pi*(x+1.0))
    return functionDerivative


def ExponentialTestFunction(x):
    function = np.exp(x)
    return function


def ExponentialTestFunctionDerivative(x,derivativeOrder=1):
    functionDerivative = np.exp(x)
    return functionDerivative


def LogarithmicTestFunction(x,a=2.0):
    function = np.log(x+a)
    return function


def LogarithmicTestFunctionDerivative(x,a=2.0,derivativeOrder=1):
    if derivativeOrder == 1:
        functionDerivative = 1.0/(x+a)
    elif derivativeOrder == 2: 
        functionDerivative = -1.0/(x+a)**2.0
    return functionDerivative


def TestLagrangePolynomialInterpolantConvergenceStudy(functionType,nXLimits):
    if functionType == 'Polynomial':
        TestFunction = PolynomialTestFunction
        Title = ('L2 Errror Norm of Lagrange Interpolant for Polynomial Function\n' 
                 + 'f(x) = x**21 + x**20 + ... + x**2 + x + 1')
    elif functionType == 'Trigonometric':
        TestFunction = TrigonometricTestFunction
        Title = 'L2 Errror Norm of Lagrange Interpolant\nfor Trigonometric Function f(x) = cos(pi*(x+1))'
    elif functionType == 'Exponential':
        TestFunction = ExponentialTestFunction 
        Title = 'L2 Errror Norm of Lagrange Interpolant\nfor Exponential Function f(x) = exp(x)'
    elif functionType == 'Logarithmic':
        TestFunction = LogarithmicTestFunction
        Title = 'L2 Errror Norm of Lagrange Interpolant\nfor Logarithmic Function f(x) = log(x+2.0)'
    mX = 100
    myNewLegendreGaussQuadrature1D = LGQ1D.LegendreGaussQuadrature1D(mX)
    xNew = myNewLegendreGaussQuadrature1D.x
    newFunctionValuesExact = np.zeros(mX+1)
    for iX in range(0,mX+1):
        newFunctionValuesExact[iX] = TestFunction(myNewLegendreGaussQuadrature1D.x[iX])
    myNewLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xNew)
    nXMin = nXLimits[0]
    nXMax = nXLimits[1]
    n_nX = nXMax - nXMin + 1
    nX = np.linspace(nXMin,nXMax,n_nX,dtype=int)
    LagrangeInterpolantL2ErrrorNorm = np.zeros(n_nX)
    epsilon = np.finfo(float).eps
    for i_nX in range(0,n_nX):
        myOldLegendreGaussQuadrature1D = LGQ1D.LegendreGaussQuadrature1D(nX[i_nX])
        xOld = myOldLegendreGaussQuadrature1D.x
        oldFunctionValues = np.zeros(nX[i_nX]+1)
        for iX in range(0,nX[i_nX]+1):
            oldFunctionValues[iX] = TestFunction(myOldLegendreGaussQuadrature1D.x[iX])
        myOldLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xOld)
        myLagrangePolynomialInterpolationMatrix = (
        myOldLagrangeInterpolation1D.EvaluateLagrangePolynomialInterpolationMatrix1D(
        myNewLagrangeInterpolation1D))
        newFunctionValues = LI1D.InterpolateToNewPoints1D(myLagrangePolynomialInterpolationMatrix,oldFunctionValues)
        LagrangeInterpolantErrror = newFunctionValues - newFunctionValuesExact
        LagrangeInterpolantL2ErrrorNorm[i_nX] = np.sqrt(
        (myNewLegendreGaussQuadrature1D.EvaluateQuadrature(LagrangeInterpolantErrror**2.0))/float(mX+1))
        if LagrangeInterpolantL2ErrrorNorm[i_nX] < epsilon:
            LagrangeInterpolantL2ErrrorNorm[i_nX] = epsilon
    xLabel = 'Number of Legendre Gauss Nodes'
    yLabel = 'Lagrange Interpolant L2 Error Norm'
    FileName = 'ConvergencePlot_LagrangeInterpolantL2ErrorNorm_' + functionType + 'Function'
    OutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output/'
    CR.PythonPlot1DSaveAsPDF(OutputDirectory,'semi-log_y',nX+1,LagrangeInterpolantL2ErrrorNorm,2.0,'-','k',True,'s',7.5,
                             [xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,False)


do_TestLagrangePolynomialInterpolantConvergenceStudy = False
if do_TestLagrangePolynomialInterpolantConvergenceStudy:
    nXLimits = [2,30]
    TestLagrangePolynomialInterpolantConvergenceStudy('Polynomial',nXLimits)
    TestLagrangePolynomialInterpolantConvergenceStudy('Trigonometric',nXLimits)
    TestLagrangePolynomialInterpolantConvergenceStudy('Exponential',nXLimits)
    TestLagrangePolynomialInterpolantConvergenceStudy('Logarithmic',nXLimits)


def TestLagrangePolynomialInterpolantDerivativeConvergenceStudy(functionType,nXLimits,m):
    if functionType == 'Polynomial':
        TestFunction = PolynomialTestFunction
        TestFunctionDerivative = PolynomialTestFunctionDerivative
        Title = ('L2 Errror Norm of Lagrange Interpolant Derivative of Order %d\n' %m
                 + 'for Polynomial Function f(x) = x**21 + x**20 + ... + x**2 + x + 1')
    elif functionType == 'Trigonometric':
        TestFunction = TrigonometricTestFunction
        TestFunctionDerivative = TrigonometricTestFunctionDerivative
        Title = ('L2 Errror Norm of Lagrange Interpolant Derivative of Order %d\n' %m
                 + 'for Trigonometric Function f(x) = cos(pi*(x+1))')
    elif functionType == 'Exponential':
        TestFunction = ExponentialTestFunction
        TestFunctionDerivative = ExponentialTestFunctionDerivative 
        Title = ('L2 Errror Norm of Lagrange Interpolant Derivative of Order %d\n' %m 
                 + 'for Exponential Function f(x) = exp(x)')
    elif functionType == 'Logarithmic':
        TestFunction = LogarithmicTestFunction
        TestFunctionDerivative = LogarithmicTestFunctionDerivative
        Title = ('L2 Errror Norm of Lagrange Interpolant Derivative of Order %d\n' %m
                 + 'for Logarithmic Function f(x) = log(x+2.0)')
    mX = 100
    myNewLegendreGaussQuadrature1D = LGQ1D.LegendreGaussQuadrature1D(mX)
    xNew = myNewLegendreGaussQuadrature1D.x
    newFunctionDerivativeValuesExact = np.zeros(mX+1)
    for iX in range(0,mX+1):
        newFunctionDerivativeValuesExact[iX] = TestFunctionDerivative(myNewLegendreGaussQuadrature1D.x[iX],
                                                                      derivativeOrder=m)
    myNewLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xNew)
    nXMin = nXLimits[0]
    nXMax = nXLimits[1]
    n_nX = nXMax - nXMin + 1
    nX = np.linspace(nXMin,nXMax,n_nX,dtype=int)
    LagrangeInterpolantDerivativeL2ErrrorNorm = np.zeros(n_nX)
    epsilon = np.finfo(float).eps
    for i_nX in range(0,n_nX):
        myOldLegendreGaussQuadrature1D = LGQ1D.LegendreGaussQuadrature1D(nX[i_nX])
        xOld = myOldLegendreGaussQuadrature1D.x
        oldFunctionValues = np.zeros(nX[i_nX]+1)
        for iX in range(0,nX[i_nX]+1):
            oldFunctionValues[iX] = TestFunction(myOldLegendreGaussQuadrature1D.x[iX])
        myOldLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(xOld)
        mthOrderOldLagrangePolynomialDerivativeMatrix = (
        myOldLagrangeInterpolation1D.Evaluate_mth_Order_LagrangePolynomialDerivativeMatrix1D(m))
        mthOrderOldLagrangePolynomialDerivative = (
        myOldLagrangeInterpolation1D.Evaluate_mth_Order_LagrangePolynomialDerivative1D(
        mthOrderOldLagrangePolynomialDerivativeMatrix,oldFunctionValues))
        myLagrangePolynomialInterpolationMatrix = (
        myOldLagrangeInterpolation1D.EvaluateLagrangePolynomialInterpolationMatrix1D(
        myNewLagrangeInterpolation1D))
        mthOrderNewLagrangePolynomialDerivative = LI1D.InterpolateToNewPoints1D(myLagrangePolynomialInterpolationMatrix,
                                                                                mthOrderOldLagrangePolynomialDerivative)
        LagrangeInterpolantDerivativeErrror = (
        mthOrderNewLagrangePolynomialDerivative - newFunctionDerivativeValuesExact)
        LagrangeInterpolantDerivativeL2ErrrorNorm[i_nX] = np.sqrt(
        (myNewLegendreGaussQuadrature1D.EvaluateQuadrature(LagrangeInterpolantDerivativeErrror**2.0))/float(mX+1))
        if LagrangeInterpolantDerivativeL2ErrrorNorm[i_nX] < epsilon:
            LagrangeInterpolantDerivativeL2ErrrorNorm[i_nX] = epsilon
    xLabel = 'Number of Legendre Gauss Nodes'
    yLabel = 'L2 Error Norm of Lagrange Interpolant Derivative of Order %d' %m
    FileName = 'ConvergencePlot_LagrangeInterpolantDerivative_Order_%d_L2ErrorNorm_' %m + functionType + 'Function'
    OutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output/'
    CR.PythonPlot1DSaveAsPDF(OutputDirectory,'semi-log_y',nX+1,LagrangeInterpolantDerivativeL2ErrrorNorm,2.0,'-','k',
                             True,'s',7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,
                             False)


do_TestLagrangePolynomialInterpolantDerivativeConvergenceStudy = False
if do_TestLagrangePolynomialInterpolantDerivativeConvergenceStudy:
    nXLimits = [2,30]
    TestLagrangePolynomialInterpolantDerivativeConvergenceStudy('Polynomial',nXLimits,1)
    TestLagrangePolynomialInterpolantDerivativeConvergenceStudy('Polynomial',nXLimits,2)
    TestLagrangePolynomialInterpolantDerivativeConvergenceStudy('Trigonometric',nXLimits,1)
    TestLagrangePolynomialInterpolantDerivativeConvergenceStudy('Trigonometric',nXLimits,2)
    TestLagrangePolynomialInterpolantDerivativeConvergenceStudy('Exponential',nXLimits,1)
    TestLagrangePolynomialInterpolantDerivativeConvergenceStudy('Exponential',nXLimits,2)
    TestLagrangePolynomialInterpolantDerivativeConvergenceStudy('Logarithmic',nXLimits,1)
    TestLagrangePolynomialInterpolantDerivativeConvergenceStudy('Logarithmic',nXLimits,2)