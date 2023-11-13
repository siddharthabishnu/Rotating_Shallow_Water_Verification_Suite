"""
Name: GeometryBasics2DClass.py
Author: Sid Bishnu
Details: This script contains functions for approximating curved boundaries, mapping the reference square to a 
curve-sided quadrilateral, and computation of the metric terms.
"""


import numpy as np
from IPython.utils import io
with io.capture_output() as captured:
    import LagrangeInterpolation1DClass as LI1D


class CurveInterpolant2D:
    
    def __init__(myCurveInterpolant2D,ParametricNodes,x,y):
        N = len(x) - 1
        myCurveInterpolant2D.N = N
        myCurveInterpolant2D.ParametricNodes = ParametricNodes
        myCurveInterpolant2D.x = x
        myCurveInterpolant2D.y = y
        
    def EvaluateCurve(myCurveInterpolant2D,s):
        myLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(myCurveInterpolant2D.ParametricNodes)
        CurveLocation = np.zeros(2)
        CurveLocation[0] = myLagrangeInterpolation1D.EvaluateLagrangeInterpolant1D(myCurveInterpolant2D.x,s)
        CurveLocation[1] = myLagrangeInterpolation1D.EvaluateLagrangeInterpolant1D(myCurveInterpolant2D.y,s)
        return CurveLocation
    
    def EvaluateCurveDerivative(myCurveInterpolant2D,s):
        myLagrangeInterpolation1D = LI1D.LagrangeInterpolation1D(myCurveInterpolant2D.ParametricNodes)
        CurveDerivative = np.zeros(2)
        CurveDerivative[0] = myLagrangeInterpolation1D.EvaluateLagrangeInterpolantDerivative1D(myCurveInterpolant2D.x,s)
        CurveDerivative[1] = myLagrangeInterpolation1D.EvaluateLagrangeInterpolantDerivative1D(myCurveInterpolant2D.y,s)
        return CurveDerivative


def TransfiniteQuadMap(xi,eta,BoundaryCurve):
    SouthWestCorner = BoundaryCurve[0].EvaluateCurve(-1.0)
    SouthEastCorner = BoundaryCurve[0].EvaluateCurve(1.0)
    NorthEastCorner = BoundaryCurve[2].EvaluateCurve(1.0)
    NorthWestCorner = BoundaryCurve[2].EvaluateCurve(-1.0)
    BoundaryCurve1 = BoundaryCurve[0].EvaluateCurve(xi)   
    BoundaryCurve2 = BoundaryCurve[1].EvaluateCurve(eta) 
    BoundaryCurve3 = BoundaryCurve[2].EvaluateCurve(xi) 
    BoundaryCurve4 = BoundaryCurve[3].EvaluateCurve(eta)
    x = (0.5*((1.0 - xi)*BoundaryCurve4[0] + (1.0 + xi)*BoundaryCurve2[0] + (1.0 - eta)*BoundaryCurve1[0]
              + (1 + eta)*BoundaryCurve3[0])
         - 0.25*((1.0 - xi)*((1.0 - eta)*SouthWestCorner[0] + (1.0 + eta)*NorthWestCorner[0]) 
                 + (1.0 + xi)*((1.0 - eta)*SouthEastCorner[0] + (1.0 + eta)*NorthEastCorner[0])))
    y = (0.5*((1.0 - xi)*BoundaryCurve4[1] + (1.0 + xi)*BoundaryCurve2[1] + (1.0 - eta)*BoundaryCurve1[1] 
              + (1 + eta)*BoundaryCurve3[1]) 
         - 0.25*((1.0 - xi)*((1.0 - eta)*SouthWestCorner[1] + (1.0 + eta)*NorthWestCorner[1]) 
                 + (1.0 + xi)*((1.0 - eta)*SouthEastCorner[1] + (1.0 + eta)*NorthEastCorner[1])))
    return x, y


def TransfiniteQuadMap_StraightSidedQuadrilateral(xi,eta,xCoordinate,yCoordinate):
    x = 0.25*((1.0 - xi)*(1.0 - eta)*xCoordinate[0] + (1.0 + xi)*(1.0 - eta)*xCoordinate[1] 
              + (1.0 + xi)*(1.0 + eta)*xCoordinate[2] + (1.0 - xi)*(1.0 + eta)*xCoordinate[3])
    y = 0.25*((1.0 - xi)*(1.0 - eta)*yCoordinate[0] + (1.0 + xi)*(1.0 - eta)*yCoordinate[1] 
              + (1.0 + xi)*(1.0 + eta)*yCoordinate[2] + (1.0 - xi)*(1.0 + eta)*yCoordinate[3])
    return x, y


def TransfiniteQuadMetrics(xi,eta,BoundaryCurve):
    SouthWestCorner = BoundaryCurve[0].EvaluateCurve(-1.0)
    SouthEastCorner = BoundaryCurve[0].EvaluateCurve(1.0)
    NorthEastCorner = BoundaryCurve[2].EvaluateCurve(1.0)
    NorthWestCorner = BoundaryCurve[2].EvaluateCurve(-1.0)
    BoundaryCurve1 = BoundaryCurve[0].EvaluateCurve(xi)   
    BoundaryCurve2 = BoundaryCurve[1].EvaluateCurve(eta) 
    BoundaryCurve3 = BoundaryCurve[2].EvaluateCurve(xi) 
    BoundaryCurve4 = BoundaryCurve[3].EvaluateCurve(eta)
    BoundaryCurve1Derivative = BoundaryCurve[0].EvaluateCurveDerivative(xi)   
    BoundaryCurve2Derivative = BoundaryCurve[1].EvaluateCurveDerivative(eta) 
    BoundaryCurve3Derivative = BoundaryCurve[2].EvaluateCurveDerivative(xi) 
    BoundaryCurve4Derivative = BoundaryCurve[3].EvaluateCurveDerivative(eta) 
    dXdXi = (0.5*(BoundaryCurve2[0] - BoundaryCurve4[0] + (1.0 - eta)*BoundaryCurve1Derivative[0] 
                  + (1.0 + eta)*BoundaryCurve3Derivative[0]) 
             - 0.25*((1.0 - eta)*(SouthEastCorner[0] - SouthWestCorner[0]) 
                     + (1.0 + eta)*(NorthEastCorner[0] - NorthWestCorner[0])))
    dYdXi = (0.5*(BoundaryCurve2[1] - BoundaryCurve4[1] + (1.0 - eta)*BoundaryCurve1Derivative[1] 
                  + (1.0 + eta)*BoundaryCurve3Derivative[1]) 
             - 0.25*((1.0 - eta)*(SouthEastCorner[1] - SouthWestCorner[1]) 
                     + (1.0 + eta)*(NorthEastCorner[1] - NorthWestCorner[1])))
    dXdEta = (0.5*((1.0 - xi)*BoundaryCurve4Derivative[0] + (1.0 + xi)*BoundaryCurve2Derivative[0] 
                   + BoundaryCurve3[0] - BoundaryCurve1[0]) 
              - 0.25*((1.0 - xi)*(NorthWestCorner[0] - SouthWestCorner[0]) 
                      + (1.0 + xi)*(NorthEastCorner[0] - SouthEastCorner[0])))
    dYdEta = (0.5*((1.0 - xi)*BoundaryCurve4Derivative[1] + (1.0 + xi)*BoundaryCurve2Derivative[1] 
                   + BoundaryCurve3[1] - BoundaryCurve1[1]) 
              - 0.25*((1.0 - xi)*(NorthWestCorner[1] - SouthWestCorner[1]) 
                      + (1.0 + xi)*(NorthEastCorner[1] - SouthEastCorner[1])))
    return dXdXi, dXdEta, dYdXi, dYdEta


def TransfiniteQuadMetrics_StraightSidedQuadrilateral(xi,eta,BoundaryCurve):
    SouthWestCorner = BoundaryCurve[0].EvaluateCurve(-1.0)
    SouthEastCorner = BoundaryCurve[0].EvaluateCurve(1.0)
    NorthEastCorner = BoundaryCurve[2].EvaluateCurve(1.0)
    NorthWestCorner = BoundaryCurve[2].EvaluateCurve(-1.0)
    dXdXi = 0.25*((1.0 - eta)*(SouthEastCorner[0] - SouthWestCorner[0]) 
                  + (1.0 + eta)*(NorthEastCorner[0] - NorthWestCorner[0]))
    dYdXi = 0.25*((1.0 - eta)*(SouthEastCorner[1] - SouthWestCorner[1])
                  + (1.0 + eta)*(NorthEastCorner[1] - NorthWestCorner[1]))
    dXdEta = 0.25*((1.0 - xi)*(NorthWestCorner[0] - SouthWestCorner[0])
                   + (1.0 + xi)*(NorthEastCorner[0] - SouthEastCorner[0]))
    dYdEta = 0.25*((1.0 - xi)*(NorthWestCorner[1] - SouthWestCorner[1])
                   + (1.0 + xi)*(NorthEastCorner[1] - SouthEastCorner[1]))
    return dXdXi, dXdEta, dYdXi, dYdEta