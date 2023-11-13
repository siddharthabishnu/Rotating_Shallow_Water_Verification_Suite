"""
Name: CommonRoutines_CellAndEdgeAveragedQuantities.py
Author: Siddhartha Bishnu
Details: This script contains
(a) Gauss Legendre quadrature routines to evaluate integrals along straight lines e.g. edges of the MPAS-O mesh, and
(b) Lyness' quadrature routines to evaluate integrals on hexagons e.g. hexagonal cells of the MPAS-O mesh.
References: 
JN Lyness and Giovanni Monegato. Quadrature rules for regions having regular hexagonal symmetry. SIAM Journal on 
Numerical Analysis, 14(2):283-295, 1977.
(b) https://people.sc.fsu.edu/~jburkardt/m_src/hexagon_rule/hexagon_rule.html
"""


import numpy as np


def LegendrePolynomialAndDerivative(nX,x): 
# Evaluate the Legendre Polynomial of degree nX and its derivative using three term recursion.
    if nX == 0:
        L_nX = 1.0
        L_nX_prime = 0.0
    elif nX == 1.0:
        L_nX = x
        L_nX_prime = 1.0
    else:
        L_nX_minus_2 = 1.0
        L_nX_minus_1 = x
        L_nX_minus_2_prime = 0.0
        L_nX_minus_1_prime = 1.0
        for kX in range(2,nX+1):
            L_nX = (2.0*float(kX) - 1.0)*x*L_nX_minus_1/float(kX) - (float(kX) - 1.0)*L_nX_minus_2/float(kX)
            L_nX_prime = L_nX_minus_2_prime + (2.0*float(kX) - 1.0)*L_nX_minus_1
            L_nX_minus_2 = L_nX_minus_1
            L_nX_minus_1 = L_nX
            L_nX_minus_2_prime = L_nX_minus_1_prime
            L_nX_minus_1_prime = L_nX_prime
    return L_nX, L_nX_prime


class LegendreGaussQuadrature1D:
    
    def __init__(myLegendreGaussQuadrature1D,nX,PrintNodesAndWeights=False,PrintSumOfWeights=False):
        myLegendreGaussQuadrature1D.nX = nX
        myLegendreGaussQuadrature1D.x = np.zeros(nX+1)
        myLegendreGaussQuadrature1D.w = np.zeros(nX+1)
        nIterations = 10**6
        tolerance = 4.0*np.finfo(float).eps
        if nX == 0:
            myLegendreGaussQuadrature1D.x[0] = 0.0
            myLegendreGaussQuadrature1D.w[0] = 2.0
        elif nX == 1:
            myLegendreGaussQuadrature1D.x[0] = -np.sqrt(1.0/3.0)
            myLegendreGaussQuadrature1D.w[0] = 1.0
            myLegendreGaussQuadrature1D.x[1] = -myLegendreGaussQuadrature1D.x[0]
            myLegendreGaussQuadrature1D.w[1] = myLegendreGaussQuadrature1D.w[0]
        else:
            for jX in range(0,int(np.floor(float(nX+1)/2.0))):
                myLegendreGaussQuadrature1D.x[jX] = -np.cos((2.0*float(jX) + 1.0)*np.pi/(2.0*float(nX) + 2.0))
                for kX in range(0,nIterations+1):
                    L_nX_plus_1, L_nX_plus_1_prime = (
                    LegendrePolynomialAndDerivative(nX+1,myLegendreGaussQuadrature1D.x[jX]))
                    Delta = -L_nX_plus_1/L_nX_plus_1_prime
                    myLegendreGaussQuadrature1D.x[jX] += Delta
                    if abs(Delta) <= tolerance*abs(myLegendreGaussQuadrature1D.x[jX]):
                        break
                L_nX_plus_1, L_nX_plus_1_prime = (
                LegendrePolynomialAndDerivative(nX+1,myLegendreGaussQuadrature1D.x[jX]))
                myLegendreGaussQuadrature1D.x[nX-jX] = -myLegendreGaussQuadrature1D.x[jX]
                myLegendreGaussQuadrature1D.w[jX] = (
                2.0/((1.0 - (myLegendreGaussQuadrature1D.x[jX])**2.0)*L_nX_plus_1_prime**2.0))
                myLegendreGaussQuadrature1D.w[nX-jX] = myLegendreGaussQuadrature1D.w[jX]
        if np.mod(nX,2) == 0.0:
            L_nX_plus_1, L_nX_plus_1_prime = LegendrePolynomialAndDerivative(nX+1,0.0)
            myLegendreGaussQuadrature1D.x[int(nX/2)] = 0.0
            myLegendreGaussQuadrature1D.w[int(nX/2)] = 2.0/L_nX_plus_1_prime**2.0
        if PrintNodesAndWeights:
            for jX in range(0,nX+1):
                print('For the Legendre Gauss quadrature node with index %2d, (x,w) is (%+.015f,%+.015f).' 
                      %(jX,myLegendreGaussQuadrature1D.x[jX],myLegendreGaussQuadrature1D.w[jX]))
        if PrintSumOfWeights:
            if PrintNodesAndWeights:
                print(' ')
            print('The sum of the Legendre Gauss weights is %.15f.' %sum(myLegendreGaussQuadrature1D.w))
            
    def EvaluateQuadrature(myLegendreGaussQuadrature1D,f):
        Q = np.dot(f,myLegendreGaussQuadrature1D.w)
        return Q


class QuadratureOnHexagon():
    
    def __init__(myQuadratureOnHexagon,origin=[0.0,0.0],l=1.0):
        myQuadratureOnHexagon.x0 = origin[0]
        myQuadratureOnHexagon.y0 = origin[1]
        myQuadratureOnHexagon.l = l
        myQuadratureOnHexagon.a = 6.0*np.sqrt(3.0)/4.0*l**2.0
        myQuadratureOnHexagon.n = 7
        myQuadratureOnHexagon.x = np.zeros(myQuadratureOnHexagon.n)
        myQuadratureOnHexagon.y = np.zeros(myQuadratureOnHexagon.n)
        myQuadratureOnHexagon.w = np.zeros(myQuadratureOnHexagon.n)
        myQuadratureOnHexagon.w[0] = 0.2559523809523809
        myQuadratureOnHexagon.x[1] = 0.7483314773547883
        myQuadratureOnHexagon.y[1] = 0.0000000000000000
        myQuadratureOnHexagon.w[1] = 0.7440476190476190/6.0
        for i in range(2,myQuadratureOnHexagon.n):
            myQuadratureOnHexagon.x[i] = (
            np.cos(np.pi/3.0)*myQuadratureOnHexagon.x[i-1] - np.sin(np.pi/3.0)*myQuadratureOnHexagon.y[i-1])
            myQuadratureOnHexagon.y[i] = (
            np.sin(np.pi/3.0)*myQuadratureOnHexagon.x[i-1] + np.cos(np.pi/3.0)*myQuadratureOnHexagon.y[i-1])
            myQuadratureOnHexagon.w[i] = myQuadratureOnHexagon.w[i-1]
        myQuadratureOnHexagon.x[:] = myQuadratureOnHexagon.x0 + myQuadratureOnHexagon.x[:]*l
        myQuadratureOnHexagon.y[:] = myQuadratureOnHexagon.y0 + myQuadratureOnHexagon.y[:]*l      
        myQuadratureOnHexagon.s = 5
        
    def DetermineQuadrature(myQuadratureOnHexagon,f):
        Quadrature = myQuadratureOnHexagon.a*np.dot(f,myQuadratureOnHexagon.w)
        return Quadrature