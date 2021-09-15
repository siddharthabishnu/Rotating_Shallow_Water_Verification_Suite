"""
Name: DGNodalStorage2DClass.py
Author: Sid Bishnu
Details: This script contains functions for creating and storing data for the Nodal Discontinuous Galerkin (DG) 
Spectral Method.
"""


import numpy as np
from IPython.utils import io
with io.capture_output() as captured:
    import LegendreGaussQuadrature1DClass as LGQ1D
    import LagrangeInterpolation2DClass as LI2D


class DGNodalStorage2D:
    
    def __init__(myDGNodalStorage2D,nXi,nEta):
        myDGNodalStorage2D.nXi = nXi
        myDGNodalStorage2D.nEta = nEta
        myDGNodalStorage2D.myLegendreGaussQuadrature1DX = LGQ1D.LegendreGaussQuadrature1D(nXi)
        myDGNodalStorage2D.myLegendreGaussQuadrature1DY = LGQ1D.LegendreGaussQuadrature1D(nEta)
        myDGNodalStorage2D.myGaussQuadratureWeightX = myDGNodalStorage2D.myLegendreGaussQuadrature1DX.w
        myDGNodalStorage2D.myGaussQuadratureWeightY = myDGNodalStorage2D.myLegendreGaussQuadrature1DY.w
        xi = myDGNodalStorage2D.myLegendreGaussQuadrature1DX.x
        eta = myDGNodalStorage2D.myLegendreGaussQuadrature1DY.x
        myDGNodalStorage2D.myLagrangeInterpolation2D = LI2D.LagrangeInterpolation2D(xi,eta)
        myDGNodalStorage2D.LagrangePolynomialsAlongWestBoundary = (
        (myDGNodalStorage2D.myLagrangeInterpolation2D.myLagrangeInterpolation1DX
         .EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(-1.0)))
        myDGNodalStorage2D.LagrangePolynomialsAlongEastBoundary = (
        (myDGNodalStorage2D.myLagrangeInterpolation2D.myLagrangeInterpolation1DX
         .EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(1.0)))
        myDGNodalStorage2D.LagrangePolynomialsAlongSouthBoundary = (
        (myDGNodalStorage2D.myLagrangeInterpolation2D.myLagrangeInterpolation1DY
         .EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(-1.0)))
        myDGNodalStorage2D.LagrangePolynomialsAlongNorthBoundary = (
        (myDGNodalStorage2D.myLagrangeInterpolation2D.myLagrangeInterpolation1DY
         .EvaluateLagrangeInterpolatingPolynomialsAtPoint1D(1.0)))
        LagrangePolynomialDerivativeMatrixX = (myDGNodalStorage2D.myLagrangeInterpolation2D.myLagrangeInterpolation1DX
                                               .EvaluateLagrangePolynomialDerivativeMatrix1D())
        LagrangePolynomialDerivativeMatrixY = (myDGNodalStorage2D.myLagrangeInterpolation2D.myLagrangeInterpolation1DY
                                               .EvaluateLagrangePolynomialDerivativeMatrix1D())
        myDGNodalStorage2D.DGDerivativeMatrixX = np.zeros((nXi+1,nXi+1))
        myDGNodalStorage2D.DGDerivativeMatrixY = np.zeros((nEta+1,nEta+1))
        for jXi in range(0,nXi+1):
            for iXi in range(0,nXi+1):
                myDGNodalStorage2D.DGDerivativeMatrixX[iXi,jXi] = (
                -(LagrangePolynomialDerivativeMatrixX[jXi,iXi]
                  *myDGNodalStorage2D.myGaussQuadratureWeightX[jXi]/myDGNodalStorage2D.myGaussQuadratureWeightX[iXi]))
        for jEta in range(0,nEta+1):
            for iEta in range(0,nEta+1):
                myDGNodalStorage2D.DGDerivativeMatrixY[iEta,jEta] = (
                -(LagrangePolynomialDerivativeMatrixY[jEta,iEta]
                  *myDGNodalStorage2D.myGaussQuadratureWeightY[jEta]/myDGNodalStorage2D.myGaussQuadratureWeightY[iEta]))