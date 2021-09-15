"""
Name: Test_LagrangeInterpolation2DClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the two-dimensional Lagrange interpolation class defined in
../../src/DGSEM_Rotating_Shallow_Water/LagrangeInterpolation2DClass.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import LagrangeInterpolation2DClass as LI2D


def TestEvaluateLagrangeInterpolant2D():
    nX = 14
    nY = 17
    xData = np.linspace(0.0,2.0*np.pi,nX+1)
    yData = np.linspace(0.0,2.0*np.pi,nY+1)
    fData = np.zeros((nX+1,nY+1))
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            fData[iX,iY] = np.sin(xData[iX])*np.cos(yData[iY])
    xInterpolatingPoint = np.pi/6.0
    yInterpolatingPoint = np.pi/3.0
    myLagrangeInterpolation2D = LI2D.LagrangeInterpolation2D(xData,yData)
    fInterpolatingPoint = (
    myLagrangeInterpolation2D.EvaluateLagrangeInterpolant2D(fData,xInterpolatingPoint,yInterpolatingPoint))
    print('The exact solution of sin(x)cos(y) at (x,y) = (pi/6,pi/3) is %.15f.' 
          %(np.sin(xInterpolatingPoint)*np.cos(yInterpolatingPoint)))
    print('The solution of sin(x)cos(y) at (x,y) = (pi/6,pi/3) obtained using Lagrange interpolation is %.15f.' 
          %fInterpolatingPoint)


do_TestEvaluateLagrangeInterpolant2D = False
if do_TestEvaluateLagrangeInterpolant2D:
    TestEvaluateLagrangeInterpolant2D()


def TestEvaluateLagrangeInterpolantDerivative2D():
    nX = 14
    nY = 17
    xData = np.linspace(0.0,2.0*np.pi,nX+1)
    yData = np.linspace(0.0,2.0*np.pi,nY+1)
    fData = np.zeros((nX+1,nY+1))
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            fData[iX,iY] = np.sin(xData[iX])*np.cos(yData[iY])
    xInterpolatingPoint = np.pi/6.0
    yInterpolatingPoint = np.pi/3.0
    myLagrangeInterpolation2D = LI2D.LagrangeInterpolation2D(xData,yData)
    grad_f_InterpolatingPoint = (
    myLagrangeInterpolation2D.EvaluateLagrangeInterpolantDerivative2D(fData,xInterpolatingPoint,
                                                                      yInterpolatingPoint))
    print('The exact gradient of sin(x)cos(y) at (x,y) = (pi/6,pi/3) is (%.15f,%.15f).' 
          %(np.cos(xInterpolatingPoint)*np.cos(yInterpolatingPoint),
            -np.sin(xInterpolatingPoint)*np.sin(yInterpolatingPoint)))
    print('The gradient of sin(x)cos(y) at (x,y) = (pi/6,pi/3) obtained using Lagrange interpolation is '
          + '(%.15f,%.15f).' %(grad_f_InterpolatingPoint[0],grad_f_InterpolatingPoint[1]))


do_TestEvaluateLagrangeInterpolantDerivative2D = False
if do_TestEvaluateLagrangeInterpolantDerivative2D:
    TestEvaluateLagrangeInterpolantDerivative2D()


def TestEvaluateLagrangePolynomialDerivative2D():
    nX = 14
    nY = 17
    xData = np.linspace(0.0,2.0*np.pi,nX+1)
    yData = np.linspace(0.0,2.0*np.pi,nY+1)
    fData = np.zeros((nX+1,nY+1))
    fDataGradient = np.zeros((nX+1,nY+1,2))
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            fData[iX,iY] = np.sin(xData[iX])*np.cos(yData[iY])
            fDataGradient[iX,iY,0] = np.cos(xData[iX])*np.cos(yData[iY])
            fDataGradient[iX,iY,1] = -np.sin(xData[iX])*np.sin(yData[iY])
    myLagrangeInterpolation2D = LI2D.LagrangeInterpolation2D(xData,yData)
    LagrangePolynomialDerivativeMatrixX, LagrangePolynomialDerivativeMatrixY = (
    myLagrangeInterpolation2D.EvaluateLagrangePolynomialDerivativeMatrix2D())
    myLagrangePolynomialGradient = (
    myLagrangeInterpolation2D.EvaluateLagrangePolynomialDerivative2D(LagrangePolynomialDerivativeMatrixX,
                                                                     LagrangePolynomialDerivativeMatrixY,fData))
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            print('At (iX,iY) = (%2d,%2d), the exact gradient is (%+.9f,%+.9f).' %(iX,iY,fDataGradient[iX,iY,0],
                                                                                   fDataGradient[iX,iY,1]))
            print('At (iX,iY) = (%2d,%2d), the gradient obtained using Lagrange interpolation is (%+.9f,%+.9f).'
                  %(iX,iY,myLagrangePolynomialGradient[iX,iY,0],myLagrangePolynomialGradient[iX,iY,1]))
            
            
do_TestEvaluateLagrangePolynomialDerivative2D = False
if do_TestEvaluateLagrangePolynomialDerivative2D:
    TestEvaluateLagrangePolynomialDerivative2D()


def TestInterpolateToNewPoints2D():
    nXOld = 11
    nYOld = 14
    nXNew = 17
    nYNew = 20
    xOld = np.linspace(0.0,2.0*np.pi,nXOld+1)
    yOld = np.linspace(0.0,2.0*np.pi,nYOld+1)
    fOld = np.zeros((nXOld+1,nYOld+1))
    for iYOld in range(0,nYOld+1):
        for iXOld in range(0,nXOld+1):
            fOld[iXOld,iYOld] = np.sin(xOld[iXOld])*np.cos(yOld[iYOld])
    myOldLagrangeInterpolation2D = LI2D.LagrangeInterpolation2D(xOld,yOld)
    xNew = np.linspace(0.0,2.0*np.pi,nXNew+1)
    yNew = np.linspace(0.0,2.0*np.pi,nYNew+1)
    fNewExact = np.zeros((nXNew+1,nYNew+1))
    for iYNew in range(0,nYNew+1):
        for iXNew in range(0,nXNew+1):
            fNewExact[iXNew,iYNew] = np.sin(xNew[iXNew])*np.cos(yNew[iYNew])
    myNewLagrangeInterpolation2D = LI2D.LagrangeInterpolation2D(xNew,yNew)
    myLagrangePolynomialInterpolationMatrixX, myLagrangePolynomialInterpolationMatrixY = (
    myOldLagrangeInterpolation2D.EvaluateLagrangePolynomialInterpolationMatrix2D(myNewLagrangeInterpolation2D))
    fNew = myOldLagrangeInterpolation2D.InterpolateToNewPoints2D(myNewLagrangeInterpolation2D,
                                                                 myLagrangePolynomialInterpolationMatrixX,
                                                                 myLagrangePolynomialInterpolationMatrixY,fOld)
    for iYNew in range(0,nYNew+1):
        for iXNew in range(0,nXNew+1):
            print('At (iX,iY) = (%2d,%2d), the exact solution is %+.9f.' %(iXNew,iYNew,fNewExact[iXNew,iYNew]))
            print('At (iX,iY) = (%2d,%2d), the solution obtained using Lagrange interpolation is %+.9f.'
                  %(iXNew,iYNew,fNew[iXNew,iYNew]))
            
            
do_TestInterpolateToNewPoints2D = False
if do_TestInterpolateToNewPoints2D:
    TestInterpolateToNewPoints2D()