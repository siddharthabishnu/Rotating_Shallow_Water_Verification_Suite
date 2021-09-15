"""
Name: LegendreGaussLobattoQuadrature1DClass.py
Author: Sid Bishnu
Details: This script defines the Legendre Gauss-Lobatto Quadrature class in one dimension.
"""


import numpy as np


def qAndLEvaluation(nX,x): # It is invoked only for nX >= 2
    L_nX_minus_2 = 1.0
    L_nX_minus_1 = x
    L_nX_minus_2_prime = 0.0
    L_nX_minus_1_prime = 1.0   
    for k in range(2,nX+1):
        L_nX = (2.0*float(k) - 1.0)*x*L_nX_minus_1/float(k) - (float(k) - 1.0)*L_nX_minus_2/float(k)
        L_nX_prime = L_nX_minus_2_prime + (2.0*float(k) - 1.0)*L_nX_minus_1
        L_nX_minus_2 = L_nX_minus_1
        L_nX_minus_1 = L_nX
        L_nX_minus_2_prime = L_nX_minus_1_prime
        L_nX_minus_1_prime = L_nX_prime        
    k = nX + 1
    L_nX_plus_1 = (2.0*float(k) - 1.0)*x*L_nX/float(k) - (float(k) - 1.0)*L_nX_minus_2/float(k)
    L_nX_plus_1_prime = L_nX_minus_2_prime + (2.0*float(k) - 1.0)*L_nX_minus_1
    q = L_nX_plus_1 - L_nX_minus_2
    q_prime = L_nX_plus_1_prime - L_nX_minus_2_prime
    return q, q_prime, L_nX


class LegendreGaussLobattoQuadrature1D:
    
    def __init__(myLegendreGaussLobattoQuadrature1D,nX,PrintNodesAndWeights=False,PrintSumOfWeights=False):
        myLegendreGaussLobattoQuadrature1D.nX = nX
        myLegendreGaussLobattoQuadrature1D.x = np.zeros(nX+1)
        myLegendreGaussLobattoQuadrature1D.w = np.zeros(nX+1)
        nIterations = 10**6
        tolerance = 4.0*np.finfo(float).eps
        if nX == 1:
            myLegendreGaussLobattoQuadrature1D.x[0] = -1.0
            myLegendreGaussLobattoQuadrature1D.w[0] = 1.0
            myLegendreGaussLobattoQuadrature1D.x[1] = 1.0
            myLegendreGaussLobattoQuadrature1D.w[1] = myLegendreGaussLobattoQuadrature1D.w[0]
        else:
            myLegendreGaussLobattoQuadrature1D.x[0] = -1.0
            myLegendreGaussLobattoQuadrature1D.w[0] = 2.0/float(nX*(nX+1))
            myLegendreGaussLobattoQuadrature1D.x[nX] = 1.0
            myLegendreGaussLobattoQuadrature1D.w[nX] = myLegendreGaussLobattoQuadrature1D.w[0]
            for jX in range(1,int(np.floor(float(nX+1)/2.0))):
                myLegendreGaussLobattoQuadrature1D.x[jX] = (
                -np.cos((float(jX) + 0.25)*np.pi/float(nX) - 3.0/(8.0*float(nX)*np.pi*(float(jX) + 0.25))))
                for kX in range(0,nIterations+1):
                    q, q_prime, L_nX = qAndLEvaluation(nX,myLegendreGaussLobattoQuadrature1D.x[jX])
                    Delta = -q/q_prime
                    myLegendreGaussLobattoQuadrature1D.x[jX] += Delta
                    if abs(Delta) <= tolerance*abs(myLegendreGaussLobattoQuadrature1D.x[jX]):
                        break
                q, q_prime, L_nX = qAndLEvaluation(nX,myLegendreGaussLobattoQuadrature1D.x[jX])
                myLegendreGaussLobattoQuadrature1D.x[nX-jX] = -myLegendreGaussLobattoQuadrature1D.x[jX]
                myLegendreGaussLobattoQuadrature1D.w[jX] = 2.0/(float(nX*(nX+1))*L_nX**2.0)
                myLegendreGaussLobattoQuadrature1D.w[nX-jX] = myLegendreGaussLobattoQuadrature1D.w[jX]
        if np.mod(nX,2) == 0.0:
            q, q_prime, L_nX = qAndLEvaluation(nX,0.0)
            myLegendreGaussLobattoQuadrature1D.w[int(float(nX)/2.0)] = 2.0/(float(nX*(nX+1))*L_nX**2.0)
        if PrintNodesAndWeights:
            for jX in range(0,nX+1):
                print('For the Legendre Gauss-Lobatto quadrature node with index %2d, (x,w) is (%+.015f,%+.015f).' 
                      %(jX,myLegendreGaussLobattoQuadrature1D.x[jX],myLegendreGaussLobattoQuadrature1D.w[jX]))
        if PrintSumOfWeights:
            if PrintNodesAndWeights:
                print(' ')
            print('The sum of the Legendre Gauss-Lobatto weights is %.15f.' 
                  %sum(myLegendreGaussLobattoQuadrature1D.w))
            
    def EvaluateQuadrature(myLegendreGaussLobattoQuadrature1D,f):
        Q = np.dot(f,myLegendreGaussLobattoQuadrature1D.w)
        return Q