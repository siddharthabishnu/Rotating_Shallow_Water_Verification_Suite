"""
Name: LagrangeInterpolation2DClass.py
Author: Sid Bishnu
Details: This script defines the two-dimensional Lagrange interpolation class.
"""


import numpy as np
from IPython.utils import io
with io.capture_output() as captured:
    import LagrangeInterpolation1DClass as LI1D


class LagrangeInterpolation2D:
    
    def __init__(myLagrangeInterpolation2D,x,y):
        myLagrangeInterpolation2D.myLagrangeInterpolation1DX = LI1D.LagrangeInterpolation1D(x)
        myLagrangeInterpolation2D.myLagrangeInterpolation1DY = LI1D.LagrangeInterpolation1D(y)

    def EvaluateLagrangeInterpolant2D(myLagrangeInterpolation2D,f,xInterpolatingPoint,yInterpolatingPoint):
        LagrangeInterpolatingPolynomialsAtPointX = (
        myLagrangeInterpolation2D.myLagrangeInterpolation1DX.EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(
        xInterpolatingPoint))
        LagrangeInterpolatingPolynomialsAtPointY = (
        myLagrangeInterpolation2D.myLagrangeInterpolation1DY.EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(
        yInterpolatingPoint))
        LagrangeInterpolant = 0.0
        for iX in range(0,myLagrangeInterpolation2D.myLagrangeInterpolation1DX.nX+1):
            for iY in range(0,myLagrangeInterpolation2D.myLagrangeInterpolation1DX.nX+1):
                LagrangeInterpolant += (f[iX,iY]*LagrangeInterpolatingPolynomialsAtPointX[iX]
                                        *LagrangeInterpolatingPolynomialsAtPointY[iY])
        return LagrangeInterpolant
    
    def EvaluateLagrangeInterpolantDerivative2D(myLagrangeInterpolation2D,f,xInterpolatingPoint,yInterpolatingPoint):
        LagrangeInterpolatingPolynomialsAtPointX = (
        myLagrangeInterpolation2D.myLagrangeInterpolation1DX.EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(
        xInterpolatingPoint))
        LagrangeInterpolatingPolynomialsAtPointY = (
        myLagrangeInterpolation2D.myLagrangeInterpolation1DY.EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(
        yInterpolatingPoint))
        grad_f = np.zeros(2)
        for iY in range(0,myLagrangeInterpolation2D.myLagrangeInterpolation1DY.nX+1):
            grad_f[0] += myLagrangeInterpolation2D.myLagrangeInterpolation1DX.EvaluateLagrangeInterpolantDerivative1D(
            f[:,iY],xInterpolatingPoint)*LagrangeInterpolatingPolynomialsAtPointY[iY]
        for iX in range(0,myLagrangeInterpolation2D.myLagrangeInterpolation1DX.nX+1):
            grad_f[1] += myLagrangeInterpolation2D.myLagrangeInterpolation1DY.EvaluateLagrangeInterpolantDerivative1D(
            f[iX,:],yInterpolatingPoint)*LagrangeInterpolatingPolynomialsAtPointX[iX]
        return grad_f
    
    def EvaluateLagrangePolynomialDerivativeMatrix2D(myLagrangeInterpolation2D):
        LagrangePolynomialDerivativeMatrixX = (
        myLagrangeInterpolation2D.myLagrangeInterpolation1DX.EvaluateLagrangePolynomialDerivativeMatrix1D())
        LagrangePolynomialDerivativeMatrixY = (
        myLagrangeInterpolation2D.myLagrangeInterpolation1DY.EvaluateLagrangePolynomialDerivativeMatrix1D())  
        return LagrangePolynomialDerivativeMatrixX, LagrangePolynomialDerivativeMatrixY

    def EvaluateLagrangePolynomialDerivative2D(myLagrangeInterpolation2D,LagrangePolynomialDerivativeMatrixX,
                                               LagrangePolynomialDerivativeMatrixY,f):
        grad_f = np.zeros((myLagrangeInterpolation2D.myLagrangeInterpolation1DX.nX+1,
                           myLagrangeInterpolation2D.myLagrangeInterpolation1DY.nX+1,2))
        for iY in range(0,myLagrangeInterpolation2D.myLagrangeInterpolation1DY.nX+1):
            grad_f[:,iY,0] = (
            myLagrangeInterpolation2D.myLagrangeInterpolation1DX.EvaluateLagrangePolynomialDerivative1D(
            LagrangePolynomialDerivativeMatrixX,f[:,iY]))
        for iX in range(0,myLagrangeInterpolation2D.myLagrangeInterpolation1DX.nX+1):
            grad_f[iX,:,1] = (
            myLagrangeInterpolation2D.myLagrangeInterpolation1DY.EvaluateLagrangePolynomialDerivative1D(
            LagrangePolynomialDerivativeMatrixY,f[iX,:]))
        return grad_f
    
    def EvaluateLagrangePolynomialInterpolationMatrix2D(myOldLagrangeInterpolation2D,myNewLagrangeInterpolation2D):
        LagrangePolynomialInterpolationMatrixX = (
        myOldLagrangeInterpolation2D.myLagrangeInterpolation1DX.EvaluateLagrangePolynomialInterpolationMatrix1D(
        myNewLagrangeInterpolation2D.myLagrangeInterpolation1DX))
        LagrangePolynomialInterpolationMatrixY = (
        myOldLagrangeInterpolation2D.myLagrangeInterpolation1DY.EvaluateLagrangePolynomialInterpolationMatrix1D(
        myNewLagrangeInterpolation2D.myLagrangeInterpolation1DY))
        return LagrangePolynomialInterpolationMatrixX, LagrangePolynomialInterpolationMatrixY
            
    def InterpolateToNewPoints2D(myOldLagrangeInterpolation2D,myNewLagrangeInterpolation2D,
                                 LagrangePolynomialInterpolationMatrixX,LagrangePolynomialInterpolationMatrixY,fOld):
        nXOld = myOldLagrangeInterpolation2D.myLagrangeInterpolation1DX.nX
        nYOld = myOldLagrangeInterpolation2D.myLagrangeInterpolation1DY.nX
        nXNew = myNewLagrangeInterpolation2D.myLagrangeInterpolation1DX.nX
        nYNew = myNewLagrangeInterpolation2D.myLagrangeInterpolation1DY.nX
        fIntermediate = np.zeros((nXNew+1,nYOld+1))
        fNew = np.zeros((nXNew+1,nYNew+1))
        for iY in range(0,nYOld+1):
            fIntermediate[0:nXNew+1,iY] = LI1D.InterpolateToNewPoints1D(LagrangePolynomialInterpolationMatrixX,
                                                                        fOld[0:nXOld+1,iY])
        for iX in range(0,nXNew+1):
            fNew[iX,0:nYNew+1] = LI1D.InterpolateToNewPoints1D(LagrangePolynomialInterpolationMatrixY,
                                                               fIntermediate[iX,0:nYOld+1])
        return fNew