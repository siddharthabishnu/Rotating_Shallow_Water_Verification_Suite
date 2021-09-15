"""
Name: DGSolution2DClass.py
Author: Sid Bishnu
Details: This script defines the solution class for two-dimensional discontinuous Galerkin spectral element methods.
"""


import numpy as np
    

class DGSolution2D:
    
    def __init__(myDGSolution2D,nEquations,nXi,nEta,TimeIntegrator):
        myDGSolution2D.nEquations = nEquations
        myDGSolution2D.ExactSolutionAtInteriorNodes = np.zeros((nEquations,nXi+1,nEta+1))
        myDGSolution2D.SolutionAtInteriorNodes = np.zeros((nEquations,nXi+1,nEta+1))
        myDGSolution2D.ErrorAtInteriorNodes = np.zeros((nEquations,nXi+1,nEta+1))
        myDGSolution2D.TendencyAtInteriorNodes = np.zeros((nEquations,nXi+1,nEta+1))
        myDGSolution2D.SolutionAtBoundaries = np.zeros((nEquations,max(nXi+1,nEta+1),4))
        myDGSolution2D.FluxAtBoundaries = np.zeros((nEquations,max(nXi+1,nEta+1),4))
        if (TimeIntegrator == 'SecondOrderAdamsBashforthMethod' or TimeIntegrator == 'ThirdOrderAdamsBashforthMethod'
            or TimeIntegrator == 'FourthOrderAdamsBashforthMethod' 
            or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback'
            or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step'
            or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step'):
            myDGSolution2D.LastTendencyAtInteriorNodes = np.zeros((nEquations,nXi+1,nEta+1))
        if (TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' or TimeIntegrator == 'FourthOrderAdamsBashforthMethod'
            or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step'):
            myDGSolution2D.SecondLastTendencyAtInteriorNodes = np.zeros((nEquations,nXi+1,nEta+1))
        if TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
            myDGSolution2D.ThirdLastTendencyAtInteriorNodes = np.zeros((nEquations,nXi+1,nEta+1))
        if TimeIntegrator == 'LeapfrogTrapezoidalMethod' or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback':
            myDGSolution2D.LastSolutionAtInteriorNodes = np.zeros((nEquations,nXi+1,nEta+1))