"""
Name: MappedGeometry2DClass.py
Author: Sid Bishnu
Details: This script contains functions for managing geometry and metric terms in quadrilateral domains.
"""


import numpy as np
from IPython.utils import io
with io.capture_output() as captured:
    import GeometryBasics2DClass as GB2D
    import VectorClass


class MappedGeometry2D:
    
    def __init__(myMappedGeometry2D,myDGNodalStorage2D,BoundaryCurve):
        nXi = myDGNodalStorage2D.nXi
        nEta = myDGNodalStorage2D.nEta
        myMappedGeometry2D.nXi = nXi
        myMappedGeometry2D.nEta = nEta
        myMappedGeometry2D.x = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.y = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.xBoundary = np.zeros((max(nXi+1,nEta+1),4))
        myMappedGeometry2D.yBoundary = np.zeros((max(nXi+1,nEta+1),4))
        myMappedGeometry2D.dXdXi = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.dXdEta = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.dYdXi = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.dYdEta = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.Jacobian = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.ScalingFactors = np.zeros((max(nXi+1,nEta+1),4))
        myMappedGeometry2D.nHat = np.empty((max(nXi+1,nEta+1),4),dtype=VectorClass.Vector)        
        xi = myDGNodalStorage2D.myLegendreGaussQuadrature1DX.x
        eta = myDGNodalStorage2D.myLegendreGaussQuadrature1DY.x        
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                myMappedGeometry2D.x[iXi,iEta],myMappedGeometry2D.y[iXi,iEta] = (
                GB2D.TransfiniteQuadMap(xi[iXi],eta[iEta],BoundaryCurve))
                (myMappedGeometry2D.dXdXi[iXi,iEta],myMappedGeometry2D.dXdEta[iXi,iEta],
                 myMappedGeometry2D.dYdXi[iXi,iEta],myMappedGeometry2D.dYdEta[iXi,iEta]) = (
                GB2D.TransfiniteQuadMetrics(xi[iXi],eta[iEta],BoundaryCurve))
                myMappedGeometry2D.Jacobian[iXi,iEta] = (
                (myMappedGeometry2D.dXdXi[iXi,iEta]*myMappedGeometry2D.dYdEta[iXi,iEta] 
                 - myMappedGeometry2D.dYdXi[iXi,iEta]*myMappedGeometry2D.dXdEta[iXi,iEta]))
        # Construct the unit normal vectors on the south and north boundaries.
        for iXi in range(0,nXi+1):
            myMappedGeometry2D.xBoundary[iXi,0],myMappedGeometry2D.yBoundary[iXi,0] = (
            GB2D.TransfiniteQuadMap(xi[iXi],-1.0,BoundaryCurve))
            dXdXi, dXdEta, dYdXi, dYdEta = GB2D.TransfiniteQuadMetrics(xi[iXi],-1.0,BoundaryCurve)
            Jacobian = dXdXi*dYdEta - dYdXi*dXdEta
            Components = np.zeros(2)
            Components[0] = np.sign(Jacobian)*dYdXi
            Components[1] = -np.sign(Jacobian)*dXdXi
            myMappedGeometry2D.nHat[iXi,0] = VectorClass.Vector(Components)
            myMappedGeometry2D.ScalingFactors[iXi,0] = myMappedGeometry2D.nHat[iXi,0].Length
            myMappedGeometry2D.nHat[iXi,0].NormalizeVector()
            myMappedGeometry2D.xBoundary[iXi,2],myMappedGeometry2D.yBoundary[iXi,2] = (
            GB2D.TransfiniteQuadMap(xi[iXi],1.0,BoundaryCurve))
            dXdXi, dXdEta, dYdXi, dYdEta = GB2D.TransfiniteQuadMetrics(xi[iXi],1.0,BoundaryCurve)
            Jacobian = dXdXi*dYdEta - dYdXi*dXdEta
            Components = np.zeros(2)
            Components[0] = -np.sign(Jacobian)*dYdXi
            Components[1] = np.sign(Jacobian)*dXdXi              
            myMappedGeometry2D.nHat[iXi,2] = VectorClass.Vector(Components)
            myMappedGeometry2D.ScalingFactors[iXi,2] = myMappedGeometry2D.nHat[iXi,2].Length
            myMappedGeometry2D.nHat[iXi,2].NormalizeVector()
        # Construct the unit normal vectors on the east and west boundaries.
        for iEta in range(0,nEta+1):
            myMappedGeometry2D.xBoundary[iEta,1],myMappedGeometry2D.yBoundary[iEta,1] = (
            GB2D.TransfiniteQuadMap(1.0,eta[iEta],BoundaryCurve))
            dXdXi, dXdEta, dYdXi, dYdEta = GB2D.TransfiniteQuadMetrics(1.0,eta[iEta],BoundaryCurve)
            Jacobian = dXdXi*dYdEta - dYdXi*dXdEta
            Components = np.zeros(2)
            Components[0] = np.sign(Jacobian)*dYdEta
            Components[1] = -np.sign(Jacobian)*dXdEta            
            myMappedGeometry2D.nHat[iEta,1] = VectorClass.Vector(Components)
            myMappedGeometry2D.ScalingFactors[iEta,1] = myMappedGeometry2D.nHat[iEta,1].Length
            myMappedGeometry2D.nHat[iEta,1].NormalizeVector()
            myMappedGeometry2D.xBoundary[iEta,3],myMappedGeometry2D.yBoundary[iEta,3] = (
            GB2D.TransfiniteQuadMap(-1.0,eta[iEta],BoundaryCurve)) 
            dXdXi, dXdEta, dYdXi, dYdEta = GB2D.TransfiniteQuadMetrics(-1.0,eta[iEta],BoundaryCurve)
            Jacobian = dXdXi*dYdEta - dYdXi*dXdEta
            Components = np.zeros(2)
            Components[0] = -np.sign(Jacobian)*dYdEta
            Components[1] = np.sign(Jacobian)*dXdEta             
            myMappedGeometry2D.nHat[iEta,3] = VectorClass.Vector(Components)
            myMappedGeometry2D.ScalingFactors[iEta,3] = myMappedGeometry2D.nHat[iEta,3].Length
            myMappedGeometry2D.nHat[iEta,3].NormalizeVector()
        myMappedGeometry2D.f = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.H = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.c = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.HBoundary = np.zeros((max(nXi+1,nEta+1),4))
        myMappedGeometry2D.cBoundary = np.zeros((max(nXi+1,nEta+1),4))
        
    def ConstructEmptyMappedGeometry2D(myMappedGeometry2D,nXi,nEta):
        myMappedGeometry2D.nXi = nXi
        myMappedGeometry2D.nEta = nEta
        myMappedGeometry2D.x = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.y = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.xBoundary = np.zeros((max(nXi+1,nEta+1),4))
        myMappedGeometry2D.yBoundary = np.zeros((max(nXi+1,nEta+1),4))
        myMappedGeometry2D.dXdXi = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.dXdEta = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.dYdXi = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.dYdEta = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.Jacobian = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.ScalingFactors = np.zeros((max(nXi+1,nEta+1),4))
        myMappedGeometry2D.nHat = np.empty((max(nXi+1,nEta+1),4),dtype=VectorClass.Vector)
        for iXi in range(0,max(nXi+1,nEta+1)):
            for iSide in range(0,4):
                myMappedGeometry2D.nHat[iXi,iSide] = VectorClass.Vector(np.array([0.0,0.0]))
        myMappedGeometry2D.f = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.H = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.c = np.zeros((nXi+1,nEta+1))
        myMappedGeometry2D.HBoundary = np.zeros((max(nXi+1,nEta+1),4))
        myMappedGeometry2D.cBoundary = np.zeros((max(nXi+1,nEta+1),4))
                
    def CalculateLocation(myMappedGeometry2D,myLagrangeInterpolation2D,xi,eta):
    # This function calculates the physical coordinates (x,y) given the computational coordinates (xi,eta).
        x = myLagrangeInterpolation2D.EvaluateLagrangeInterpolant2D(myMappedGeometry2D.x,xi,eta)
        y = myLagrangeInterpolation2D.EvaluateLagrangeInterpolant2D(myMappedGeometry2D.y,xi,eta)
        return x, y
    
    def CalculateMetricTerms(myMappedGeometry2D,myLagrangeInterpolation2D,xi,eta):   
        """
        This function calculates the covariant metric terms given the computational coordinates (xi,eta).

        CovariantMetricTensor(1,1) --> dXdXi
        CovariantMetricTensor(1,2) --> dXdEta
        CovariantMetricTensor(2,1) --> dYdXi
        CovariantMetricTensor(2,2) --> dYdEta    
        """
        CovariantMetricTensor = np.zeros((2,2))
        CovariantMetricTensor[0,:] = (
        myLagrangeInterpolation2D.EvaluateLagrangeInterpolantDerivative2D(myMappedGeometry2D.x,xi,eta))
        CovariantMetricTensor[1,:] = (
        myLagrangeInterpolation2D.EvaluateLagrangeInterpolantDerivative2D(myMappedGeometry2D.y,xi,eta))
        return CovariantMetricTensor
    
    def DetermineComputationalCoordinates(myMappedGeometry2D,myLagrangeInterpolation2D,x,y):   
    # Given the physical coordinates (x*,z*), the computational coordinates are calculated using Newton's method for 
    # root finding to solve
    #     x* = x(xi,eta)
    #     z* = z(xi,eta)
    # for xi and eta.
        ComparisonTolerance = 10.0**(-15.0)
        ConvergenceTolerance = 10.0**(-15.0)
        nIterationsNewtonRaphson2D = 25
        success = False
        # Initial guess is at the origin.
        thisXi = 0.0 
        thisEta = 0.0
        dr = np.zeros(2)
        for iIteration in range(0,nIterationsNewtonRaphson2D):
            # Calculate the physical coordinates associated with the guessed computational coordinates.
            thisX, thisY = myMappedGeometry2D.CalculateLocation(myLagrangeInterpolation2D,thisXi,thisEta)
            # Calculate the residual.
            dr[0] = x - thisX
            dr[1] = y - thisY  
            res = np.sqrt(np.dot(dr,dr))
            if (res < ConvergenceTolerance and (abs(thisXi <= 1.0) or abs(thisXi) - 1.0 <= ComparisonTolerance)
                and (abs(thisEta <= 1.0) or abs(thisEta) - 1.0 <= ComparisonTolerance)):
                if (abs(thisXi) >= 1.0):
                    xi = np.sign(thisXi)
                else:
                    xi = thisXi
                if (abs(thisEta) >= 1.0):
                    eta = np.sign(thisEta)
                else:
                    eta = thisEta                
                success = True
                iIterationFinal = iIteration 
                return success, iIterationFinal, res, xi, eta
            # Calculate the covariant metric tensor.
            A = myMappedGeometry2D.CalculateMetricTerms(myLagrangeInterpolation2D,thisXi,thisEta)    
            # Invert the covariant metric tensor. The matrix is invertable as long as the Jacobian is non-zero.   
            AInverse = np.linalg.inv(A)   
            # Calculate the correction in the computational coordinates.
            ds = np.matmul(AInverse,dr)
            thisXi = thisXi + ds[0]
            thisEta = thisEta + ds[1]
        # Calculate the physical coordinates associated with the guessed computational coordinates.
        thisX, thisY = myMappedGeometry2D.CalculateLocation(myLagrangeInterpolation2D,thisXi,thisEta)
        # Calculate the residual.
        dr[0] = x - thisX
        dr[1] = y - thisY  
        res = np.sqrt(np.dot(dr,dr))
        if (abs(thisXi) >= 1.0):
            xi = np.sign(thisXi)
        else:
            xi = thisXi
        if (abs(thisEta) >= 1.0):
            eta = np.sign(thisEta)
        else:
            eta = thisEta       
        if (res < ConvergenceTolerance and (abs(thisXi <= 1.0) or abs(thisXi) - 1.0 <= ComparisonTolerance)
            and (abs(thisEta <= 1.0) or abs(thisEta) - 1.0 <= ComparisonTolerance)):
            success = True
        iIterationFinal = nIterationsNewtonRaphson2D 
        return success, iIterationFinal, res, xi, eta