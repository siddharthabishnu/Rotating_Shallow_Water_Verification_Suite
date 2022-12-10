"""
Name: SpatialOperators.py
Author: Sid Bishnu
Details: This script contains functions for computing the spatial operators of the TRiSK-based mimetic finite volume 
method. These operators are the gradient, divergence, and curl operators as well as the interpolation or flux-mapping 
operator used to diagnostically obtain the tangential velocity from the normal velocity at the edges of the cells.
"""


import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from IPython.utils import io
with io.capture_output() as captured: 
    import CommonRoutines as CR
    
    
def ProblemSpecificPrefix_1():
    prefix = 'Expt1_'
    return prefix

    
def SurfaceElevation_1(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    return eta


def SurfaceElevationGradient_1(lX,lY,x,y):
    eta0 = 0.1
    eta_x = eta0*(2.0*np.pi/lX)*np.cos(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_y = eta0*(2.0*np.pi/lY)*np.sin(2.0*np.pi*x/lX)*np.cos(2.0*np.pi*y/lY)
    return eta_x, eta_y


def SurfaceElevationLaplacian_1(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_xx = -(2.0*np.pi/lX)**2.0*eta
    eta_yy = -(2.0*np.pi/lY)**2.0*eta
    Laplacian = eta_xx + eta_yy
    return Laplacian


def ProblemSpecificPrefix_2():
    prefix = 'Expt2_'
    return prefix


def SurfaceElevation_2(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.exp(-((np.sin(2.0*np.pi*x/lX))**2.0 + (np.sin(2.0*np.pi*y/lY))**2.0))
    return eta


def SurfaceElevationGradient_2_FunctionalForm():
    lX = sp.Symbol('lX')
    lY = sp.Symbol('lY')
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    eta0 = 0.1
    eta = eta0*sp.exp(-((sp.sin(2.0*sp.pi*x/lX))**2.0 + (sp.sin(2.0*sp.pi*y/lY))**2.0))
    eta_x = sp.diff(eta,x)
    eta_y = sp.diff(eta,y)
    eta_x_np = lambdify((lX,lY,x,y), eta_x, modules=["numpy","sympy"])
    eta_y_np = lambdify((lX,lY,x,y), eta_y, modules=["numpy","sympy"])
    # Note that np in eta_x_np and eta_y_np stands for numpy.
    return eta_x_np, eta_y_np


eta_x_np, eta_y_np = SurfaceElevationGradient_2_FunctionalForm()


def SurfaceElevationGradient_2(lX,lY,x,y):
    eta_x = eta_x_np(lX,lY,x,y)
    eta_y = eta_y_np(lX,lY,x,y)
    return eta_x, eta_y


def SurfaceElevationLaplacian_2_FunctionalForm():
    lX = sp.Symbol('lX')
    lY = sp.Symbol('lY')
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    eta0 = 0.1
    eta = eta0*sp.exp(-((sp.sin(2.0*sp.pi*x/lX))**2.0 + (sp.sin(2.0*sp.pi*y/lY))**2.0))
    eta_xx = sp.diff(eta,x,x)
    eta_yy = sp.diff(eta,y,y)
    Laplacian = eta_xx + eta_yy
    Laplacian_np = lambdify((lX,lY,x,y), Laplacian, modules=["numpy","sympy"])
    # Note that np in Laplacian_np stands for numpy.
    return Laplacian_np


SurfaceElevationLaplacian_2 = SurfaceElevationLaplacian_2_FunctionalForm()


MyExperimentNumber = 1 # Choose MyExperimentNumber to be 1 or 2.
if MyExperimentNumber == 1:
    SurfaceElevation = SurfaceElevation_1
    SurfaceElevationGradient = SurfaceElevationGradient_1
    SurfaceElevationLaplacian = SurfaceElevationLaplacian_1
    ProblemSpecificPrefix = ProblemSpecificPrefix_1
else: # if MyExperimentNumber == 2:
    SurfaceElevation = SurfaceElevation_2
    SurfaceElevationGradient = SurfaceElevationGradient_2
    SurfaceElevationLaplacian = SurfaceElevationLaplacian_2
    ProblemSpecificPrefix = ProblemSpecificPrefix_2


def Velocity(lX,lY,x,y):
    eta_x, eta_y = SurfaceElevationGradient(lX,lY,x,y) 
    f = 10.0**(-4.0)
    g = 10.0
    v = g*eta_x/f
    u = -g*eta_y/f
    return u, v


def VelocityCurl(lX,lY,x,y):
    f = 10.0**(-4.0)
    g = 10.0
    zeta = g/f*SurfaceElevationLaplacian(lX,lY,x,y)
    return zeta


def ComputeNormalAndTangentialComponentsAtEdge(myVectorQuantityAtEdge,angleEdge,returningComponent):
    nEdges = len(angleEdge)
    if returningComponent == 'normal' or returningComponent == 'both':
        myVectorQuantityAtEdgeNormalComponent = np.zeros(nEdges)
    if returningComponent == 'tangential' or returningComponent == 'both':
        myVectorQuantityAtEdgeTangentialComponent = np.zeros(nEdges)    
    for iEdge in range(0,nEdges):
        xComponent = myVectorQuantityAtEdge[iEdge,0]
        yComponent = myVectorQuantityAtEdge[iEdge,1]
        if returningComponent == 'normal' or returningComponent == 'both':
            myVectorQuantityAtEdgeNormalComponent[iEdge] = (xComponent*np.cos(angleEdge[iEdge]) 
                                                            + yComponent*np.sin(angleEdge[iEdge]))
        if returningComponent == 'tangential' or returningComponent == 'both':
            myVectorQuantityAtEdgeTangentialComponent[iEdge] = (yComponent*np.cos(angleEdge[iEdge]) 
                                                                - xComponent*np.sin(angleEdge[iEdge]))
    if returningComponent == 'normal':
        return myVectorQuantityAtEdgeNormalComponent
    elif returningComponent == 'tangential':
        return myVectorQuantityAtEdgeTangentialComponent
    else: # if returningComponent == 'both':
        return myVectorQuantityAtEdgeNormalComponent, myVectorQuantityAtEdgeTangentialComponent


def AnalyticalGradientOperator(myScalarQuantityGradientComponentsAtEdge,angleEdge):
    myScalarQuantityGradientNormalToEdge = (
    ComputeNormalAndTangentialComponentsAtEdge(myScalarQuantityGradientComponentsAtEdge,angleEdge,'normal'))
    return myScalarQuantityGradientNormalToEdge


def NumericalGradientOperator(myMesh,myScalarQuantity,BoundaryCondition='Periodic'):
    myScalarQuantityGradientNormalToEdge = np.zeros(myMesh.nEdges)
    for iEdge in range(0,myMesh.nEdges):
        if not((BoundaryCondition == 'NonPeriodic_x' or BoundaryCondition == 'NonPeriodic_y' 
                or BoundaryCondition == 'NonPeriodic_xy') and myMesh.boundaryEdge[iEdge] == 1.0):
            cellID1 = myMesh.cellsOnEdge[iEdge,0]
            cell1 = cellID1 - 1
            cellID2 = myMesh.cellsOnEdge[iEdge,1]
            cell2 = cellID2 - 1
            invLength = 1.0/myMesh.dcEdge[iEdge]    
            myScalarQuantityGradientNormalToEdge[iEdge] = (myScalarQuantity[cell2] - myScalarQuantity[cell1])*invLength
    return myScalarQuantityGradientNormalToEdge


def NumericalDivergenceOperator(myMesh,myVectorQuantityNormalToEdge):
    myVectorQuantityDivergence = np.zeros(myMesh.nCells)
    for iCell in range(0,myMesh.nCells):
        InverseAreaCell = 1.0/myMesh.areaCell[iCell]
        for iEdgeOnCell in range(0,myMesh.nEdgesOnCell[iCell]):
            iEdgeID = myMesh.edgesOnCell[iCell,iEdgeOnCell]
            iEdge = iEdgeID - 1
            edgeSignOnCell_temp = myMesh.edgeSignOnCell[iCell,iEdgeOnCell]
            dvEdge_temp = myMesh.dvEdge[iEdge]
            r_tmp = dvEdge_temp*myVectorQuantityNormalToEdge[iEdge]*InverseAreaCell
            myVectorQuantityDivergence[iCell] -= edgeSignOnCell_temp*r_tmp
    return myVectorQuantityDivergence


def NumericalCurlOperator(myMesh,myVectorQuantityNormalToEdge,BoundaryCondition='Periodic'):
    myVectorQuantityCurlAtVertex = np.zeros(myMesh.nVertices)
    myVectorQuantityCurlAtCellCenter = np.zeros(myMesh.nCells)
    for iVertex in range(0,myMesh.nVertices):
        if ((BoundaryCondition == 'NonPeriodic_x' or BoundaryCondition == 'NonPeriodic_y' 
             or BoundaryCondition == 'NonPeriodic_xy') and myMesh.boundaryVertex[iVertex] == 1.0):
            myVectorQuantityCurlAtVertex[iVertex] = (
            VelocityCurl(myMesh.lX,myMesh.lY,myMesh.xVertex[iVertex],myMesh.yVertex[iVertex]))
        else:
            inverseAreaTriangle = 1.0/myMesh.areaTriangle[iVertex]
            for iVertexDegree in range(0,myMesh.vertexDegree):
                iEdgeID = myMesh.edgesOnVertex[iVertex,iVertexDegree]
                iEdge = iEdgeID - 1
                r_tmp = myMesh.dcEdge[iEdge]*myVectorQuantityNormalToEdge[iEdge]
                myVectorQuantityCurlAtVertex[iVertex] += (myMesh.edgeSignOnVertex[iVertex,iVertexDegree]
                                                          *r_tmp*inverseAreaTriangle)
    for iCell in range(0,myMesh.nCells):
        inverseAreaCell = 1.0/myMesh.areaCell[iCell]
        for iEdgeOnCell in range(0,myMesh.nEdgesOnCell[iCell]):
            jID = myMesh.kiteIndexOnCell[iCell,iEdgeOnCell]
            j = jID - 1
            iVertexID = myMesh.verticesOnCell[iCell,iEdgeOnCell]
            iVertex = iVertexID - 1
            myVectorQuantityCurlAtCellCenter[iCell] += (
            myMesh.kiteAreasOnVertex[iVertex,j]*myVectorQuantityCurlAtVertex[iVertex]*inverseAreaCell)
    return myVectorQuantityCurlAtVertex, myVectorQuantityCurlAtCellCenter


def NumericalTangentialVelocity(myMesh,myNormalVelocity,BoundaryCondition='Periodic'):
    myTangentialVelocity = np.zeros(myMesh.nEdges)
    for iEdge in range(0,myMesh.nEdges):
        if ((BoundaryCondition == 'NonPeriodic_x' or BoundaryCondition == 'NonPeriodic_y' 
             or BoundaryCondition == 'NonPeriodic_xy') and myMesh.boundaryEdge[iEdge] == 1.0):
            u, v = Velocity(myMesh.lX,myMesh.lY,myMesh.xEdge[iEdge],myMesh.yEdge[iEdge])
            myTangentialVelocity[iEdge] = v*np.cos(myMesh.angleEdge[iEdge]) - u*np.sin(myMesh.angleEdge[iEdge])
        else:
            myTangentialVelocity[iEdge] = 0.0
            # Compute the tangential velocities.
            for iEdgeOnEdge in range(0,myMesh.nEdgesOnEdge[iEdge]):
                eoeID = myMesh.edgesOnEdge[iEdge,iEdgeOnEdge]
                eoe = eoeID - 1
                weightsOnEdge_temp = myMesh.weightsOnEdge[iEdge,iEdgeOnEdge]
                myTangentialVelocity[iEdge] += weightsOnEdge_temp*myNormalVelocity[eoe]
    return myTangentialVelocity