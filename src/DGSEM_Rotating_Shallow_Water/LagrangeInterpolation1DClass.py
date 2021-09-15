"""
Name: LagrangeInterpolation1DClass.py
Author: Sid Bishnu
Details: This script defines the one-dimensional Lagrange interpolation class.
"""


import numpy as np
import sympy as sp
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR


def LagrangeInterpolationFunction1D(xData,fData,x):
    nX = len(xData) - 1
    LagInterp1D = 0.0
    for iX in range(0,nX+1):
        LagrangeProduct = fData[iX]
        for jX in range(0,nX+1):
            if iX != jX:
                LagrangeProduct = LagrangeProduct*(x - xData[jX])/(xData[iX] - xData[jX])
        LagInterp1D = LagInterp1D + LagrangeProduct
    return LagInterp1D


class LagrangeInterpolation1D:
    
    def __init__(myLagrangeInterpolation1D,x):
        nX = np.size(x) - 1
        myLagrangeInterpolation1D.nX = nX
        myLagrangeInterpolation1D.x = x
        myLagrangeInterpolation1D.BarycentricWeights = np.ones(nX+1)
        for jX in range(1,nX+1):
            for kX in range(0,jX):
                myLagrangeInterpolation1D.BarycentricWeights[kX] *= (
                (myLagrangeInterpolation1D.x[kX] - myLagrangeInterpolation1D.x[jX]))
                myLagrangeInterpolation1D.BarycentricWeights[jX] *= (
                (myLagrangeInterpolation1D.x[jX] - myLagrangeInterpolation1D.x[kX]))
        for jX in range(0,nX+1):
            myLagrangeInterpolation1D.BarycentricWeights[jX] = 1.0/myLagrangeInterpolation1D.BarycentricWeights[jX]
            
    def EvaluateLagrangeInterpolant1D(myLagrangeInterpolation1D,f,xInterpolatingPoint):
        nX = myLagrangeInterpolation1D.nX
        numerator = 0.0
        denominator = 0.0
        for jX in range(0,nX+1):
            if CR.AlmostEqual(myLagrangeInterpolation1D.x[jX],xInterpolatingPoint):
                LagrangeInterpolant = f[jX]
                return LagrangeInterpolant
            else:
                t = (myLagrangeInterpolation1D.BarycentricWeights[jX]
                     /(xInterpolatingPoint - myLagrangeInterpolation1D.x[jX]))
                numerator += t*f[jX]
                denominator += t
        LagrangeInterpolant = numerator/denominator
        return LagrangeInterpolant
    
    def EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(myLagrangeInterpolation1D,xInterpolatingPoint):
        xInterpolatingPointMatchesNode = False
        nX = myLagrangeInterpolation1D.nX
        LagrangeInterpolatingPolynomialsAtPoint = np.zeros(nX+1)
        for jX in range(0,nX+1):
            if CR.AlmostEqual(myLagrangeInterpolation1D.x[jX],xInterpolatingPoint):
                LagrangeInterpolatingPolynomialsAtPoint[jX] = 1.0
                xInterpolatingPointMatchesNode = True
                break
        if xInterpolatingPointMatchesNode:
            return LagrangeInterpolatingPolynomialsAtPoint
        denominator = 0.0
        for jX in range(0,nX+1):
            numerator = (myLagrangeInterpolation1D.BarycentricWeights[jX]
                         /(xInterpolatingPoint - myLagrangeInterpolation1D.x[jX]))
            LagrangeInterpolatingPolynomialsAtPoint[jX] = numerator
            denominator += numerator
        for jX in range(0,nX+1):
            LagrangeInterpolatingPolynomialsAtPoint[jX] /= denominator
        return LagrangeInterpolatingPolynomialsAtPoint
    
    def EvaluateLagrangeInterpolatingPolynomials1D(myLagrangeInterpolation1D):
        nX = myLagrangeInterpolation1D.nX
        LagrangeInterpolatingPolynomials = np.zeros(nX+1)
        x = sp.Symbol('x')
        LagrangeInterpolatingPolynomials = np.zeros(nX+1,dtype=sp.Symbol)
        LagrangeInterpolatingPolynomials[:] = 1.0
        for iX in range(0,nX+1):
            for jX in range(0,nX+1):
                if iX != jX:
                    LagrangeInterpolatingPolynomials[iX] *= (
                    ((x - myLagrangeInterpolation1D.x[jX])
                     /(myLagrangeInterpolation1D.x[iX] - myLagrangeInterpolation1D.x[jX])))      
            LagrangeInterpolatingPolynomials[iX] = sp.collect(sp.expand(LagrangeInterpolatingPolynomials[iX]),x)
        return LagrangeInterpolatingPolynomials
    
    def EvaluateLagrangeInterpolantDerivative1D(myLagrangeInterpolation1D,f,xInterpolatingPoint):
        atNode = False
        numerator = 0.0
        for jX in range(0,myLagrangeInterpolation1D.nX+1):
            if CR.AlmostEqual(xInterpolatingPoint,myLagrangeInterpolation1D.x[jX]):
                atNode = True
                p = f[jX]
                denominator = - myLagrangeInterpolation1D.BarycentricWeights[jX]
                iX = jX
        if atNode:
            for jX in range(0,myLagrangeInterpolation1D.nX+1):
                if jX != iX:
                    numerator += ((myLagrangeInterpolation1D.BarycentricWeights[jX])*(p - f[jX])
                                  /(xInterpolatingPoint - myLagrangeInterpolation1D.x[jX]))
        else:
            denominator = 0.0
            p = myLagrangeInterpolation1D.EvaluateLagrangeInterpolant1D(f,xInterpolatingPoint)
            for jX in range(0,myLagrangeInterpolation1D.nX+1):
                t = ((myLagrangeInterpolation1D.BarycentricWeights[jX])
                     /(xInterpolatingPoint - myLagrangeInterpolation1D.x[jX]))
                numerator += t*(p - f[jX])/(xInterpolatingPoint - myLagrangeInterpolation1D.x[jX])
                denominator += t                
        LagrangeInterpolantDerivative = numerator/denominator 
        return LagrangeInterpolantDerivative
    
    def EvaluateLagrangePolynomialDerivativeMatrix1D(myLagrangeInterpolation1D):
        LagrangePolynomialDerivativeMatrix = np.zeros((myLagrangeInterpolation1D.nX+1,myLagrangeInterpolation1D.nX+1))
        for iX in range(0,myLagrangeInterpolation1D.nX+1):
            for jX in range(0,myLagrangeInterpolation1D.nX+1):
                if jX != iX:
                    LagrangePolynomialDerivativeMatrix[iX,jX] = (
                    ((myLagrangeInterpolation1D.BarycentricWeights[jX])
                     /(myLagrangeInterpolation1D.BarycentricWeights[iX]
                       *(myLagrangeInterpolation1D.x[iX] - myLagrangeInterpolation1D.x[jX]))))
                    LagrangePolynomialDerivativeMatrix[iX,iX] -= LagrangePolynomialDerivativeMatrix[iX,jX]
        return LagrangePolynomialDerivativeMatrix
    
    def EvaluateLagrangePolynomialDerivative1D(myLagrangeInterpolation1D,LagrangePolynomialDerivativeMatrix,f):
        LagrangePolynomialDerivative = np.matmul(LagrangePolynomialDerivativeMatrix,f)
        return LagrangePolynomialDerivative
    
    def Evaluate_mth_Order_LagrangePolynomialDerivativeMatrix1D(myLagrangeInterpolation1D,m):
        nX = myLagrangeInterpolation1D.nX
        x = myLagrangeInterpolation1D.x
        BarycentricWeights = myLagrangeInterpolation1D.BarycentricWeights
        mthOrderLagrangePolynomialDerivativeMatrix = (
        myLagrangeInterpolation1D.EvaluateLagrangePolynomialDerivativeMatrix1D())
        if m > 1:
            mMinus1thOrderLagrangePolynomialDerivativeMatrix = np.zeros((nX+1,nX+1))
            for iOrder in range(2,m+1):
                for iX in range(0,nX+1):
                    mMinus1thOrderLagrangePolynomialDerivativeMatrix[iX,iX] = (
                    mthOrderLagrangePolynomialDerivativeMatrix[iX,iX])
                    mthOrderLagrangePolynomialDerivativeMatrix[iX,iX] = 0.0
                    for jX in range(0,nX+1):
                        if jX != iX:
                            mthOrderLagrangePolynomialDerivativeMatrix[iX,jX] = (
                            float(iOrder)*((BarycentricWeights[jX]/BarycentricWeights[iX]
                                            *mMinus1thOrderLagrangePolynomialDerivativeMatrix[iX,iX])
                                           - mthOrderLagrangePolynomialDerivativeMatrix[iX,jX])/(x[iX] - x[jX]))
                            mthOrderLagrangePolynomialDerivativeMatrix[iX,iX] -= (
                            mthOrderLagrangePolynomialDerivativeMatrix[iX,jX])
        return mthOrderLagrangePolynomialDerivativeMatrix    
    
    def Evaluate_mth_Order_LagrangePolynomialDerivative1D(myLagrangeInterpolation1D,
                                                          mthOrderLagrangePolynomialDerivativeMatrix,f):
        mthOrderLagrangePolynomialDerivative = np.matmul(mthOrderLagrangePolynomialDerivativeMatrix,f)
        return mthOrderLagrangePolynomialDerivative

    def EvaluateLagrangePolynomialInterpolationMatrix1D(myOldLagrangeInterpolation1D,myNewLagrangeInterpolation1D):
        LagrangePolynomialInterpolationMatrix = np.zeros((myNewLagrangeInterpolation1D.nX+1,
                                                          myOldLagrangeInterpolation1D.nX+1))
        for kX in range(0,myNewLagrangeInterpolation1D.nX+1):
            rowHasMatch = False
            for jX in range(0,myOldLagrangeInterpolation1D.nX+1):
                LagrangePolynomialInterpolationMatrix[kX,jX] = 0.0
                if CR.AlmostEqual(myNewLagrangeInterpolation1D.x[kX],myOldLagrangeInterpolation1D.x[jX]):
                    rowHasMatch = True
                    LagrangePolynomialInterpolationMatrix[kX,jX] = 1.0
            if not(rowHasMatch):
                temporaryEntry2 = 0.0
                for jX in range(0,myOldLagrangeInterpolation1D.nX+1):
                    temporaryEntry1 = ((myOldLagrangeInterpolation1D.BarycentricWeights[jX])
                                       /(myNewLagrangeInterpolation1D.x[kX] - myOldLagrangeInterpolation1D.x[jX]))
                    LagrangePolynomialInterpolationMatrix[kX,jX] = temporaryEntry1
                    temporaryEntry2 += temporaryEntry1         
                for jX in range(0,myOldLagrangeInterpolation1D.nX+1):
                    LagrangePolynomialInterpolationMatrix[kX,jX] /= temporaryEntry2       
        return LagrangePolynomialInterpolationMatrix


def InterpolateToNewPoints1D(LagrangePolynomialInterpolationMatrix,fOld):
    fNew = np.matmul(LagrangePolynomialInterpolationMatrix,fOld)
    return fNew