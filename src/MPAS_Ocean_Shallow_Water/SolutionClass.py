"""
Name: SolutionClass.py
Author: Sid Bishnu
Details: This script defines the MPAS-Ocean shallow water solution class.
"""


import numpy as np
    

class Solution:
    
    def __init__(mySolution,nCells,nEdges,nVertices,TimeIntegrator):
        mySolution.uExact = np.zeros(nCells)
        mySolution.vExact = np.zeros(nCells)
        mySolution.sshExact = np.zeros(nCells)
        mySolution.u = np.zeros(nCells)
        mySolution.v = np.zeros(nCells)
        mySolution.ssh = np.zeros(nCells)
        mySolution.uError = np.zeros(nCells)
        mySolution.vError = np.zeros(nCells)
        mySolution.sshError = np.zeros(nCells)
        mySolution.sshSourceTerm = np.zeros(nCells)
        mySolution.sshTendency = np.zeros(nCells)
        mySolution.normalVelocity = np.zeros(nEdges)
        mySolution.normalVelocitySourceTerm = np.zeros(nEdges)
        mySolution.normalVelocityTendency = np.zeros(nEdges)
        mySolution.tangentialVelocity = np.zeros(nEdges)
        if (TimeIntegrator == 'SecondOrderAdamsBashforthMethod' or TimeIntegrator == 'ThirdOrderAdamsBashforthMethod'
            or TimeIntegrator == 'FourthOrderAdamsBashforthMethod' 
            or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback'
            or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB2AM3Step'
            or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step'):
            mySolution.sshTendencyLast = np.zeros(nCells)
            mySolution.normalVelocityTendencyLast = np.zeros(nEdges)
        if (TimeIntegrator == 'ThirdOrderAdamsBashforthMethod' or TimeIntegrator == 'FourthOrderAdamsBashforthMethod'
            or TimeIntegrator == 'GeneralizedForwardBackwardMethodWithAB3AM4Step'):
            mySolution.sshTendencySecondLast = np.zeros(nCells)
            mySolution.normalVelocityTendencySecondLast = np.zeros(nEdges)
        if TimeIntegrator == 'FourthOrderAdamsBashforthMethod':
            mySolution.sshTendencyThirdLast = np.zeros(nCells)
            mySolution.normalVelocityTendencyThirdLast = np.zeros(nEdges)
        if TimeIntegrator == 'LeapfrogTrapezoidalMethod' or TimeIntegrator == 'LFTRAndLFAM3MethodWithFBFeedback':
            mySolution.sshLast = np.zeros(nCells)
            mySolution.normalVelocityLast = np.zeros(nEdges)
        mySolution.circulation = np.zeros(nVertices)
        mySolution.divergence = np.zeros(nCells)
        mySolution.kineticEnergyCell = np.zeros(nCells)
        mySolution.layerThickness = np.zeros(nCells)
        mySolution.layerThicknessEdge = np.zeros(nEdges)
        mySolution.normalizedPlanetaryVorticityEdge = np.zeros(nEdges)
        mySolution.normalizedPlanetaryVorticityVertex = np.zeros(nVertices)
        mySolution.normalizedRelativeVorticityCell = np.zeros(nCells)
        mySolution.normalizedRelativeVorticityEdge = np.zeros(nEdges)
        mySolution.normalizedRelativeVorticityVertex = np.zeros(nVertices)
        mySolution.relativeVorticity = np.zeros(nVertices)
        mySolution.relativeVorticityCell = np.zeros(nCells)