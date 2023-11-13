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


def SurfaceElevationDoubleDerivatives_1(lX,lY,x,y):
    eta0 = 0.1
    eta = SurfaceElevation_1(lX,lY,x,y)
    eta_xx = -(2.0*np.pi/lX)**2.0*eta
    eta_yy = -(2.0*np.pi/lY)**2.0*eta
    eta_xy = eta0*(2.0*np.pi/lX)*(2.0*np.pi/lY)*np.cos(2.0*np.pi*x/lX)*np.cos(2.0*np.pi*y/lY)
    return eta_xx, eta_xy, eta_yy


def SurfaceElevationLaplacian_1(lX,lY,x,y):
    eta = SurfaceElevation_1(lX,lY,x,y)
    eta_xx = -(2.0*np.pi/lX)**2.0*eta
    eta_yy = -(2.0*np.pi/lY)**2.0*eta
    Laplacian = eta_xx + eta_yy
    return Laplacian


def SurfaceElevationGradientOfLaplacian_1(lX,lY,x,y):
    eta0 = 0.1
    eta_x = eta0*(2.0*np.pi/lX)*np.cos(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_y = eta0*(2.0*np.pi/lY)*np.sin(2.0*np.pi*x/lX)*np.cos(2.0*np.pi*y/lY)
    Laplacian_eta_x = -((2.0*np.pi/lX)**2.0 + (2.0*np.pi/lY)**2.0)*eta_x
    Laplacian_eta_y = -((2.0*np.pi/lX)**2.0 + (2.0*np.pi/lY)**2.0)*eta_y
    return Laplacian_eta_x, Laplacian_eta_y


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


def SurfaceElevationDoubleDerivatives_2_FunctionalForm():
    lX = sp.Symbol('lX')
    lY = sp.Symbol('lY')
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    eta0 = 0.1
    eta = eta0*sp.exp(-((sp.sin(2.0*sp.pi*x/lX))**2.0 + (sp.sin(2.0*sp.pi*y/lY))**2.0))
    eta_xx = sp.diff(eta,x,x)
    eta_xy = sp.diff(eta,x,y)
    eta_yy = sp.diff(eta,y,y)
    eta_xx_np = lambdify((lX,lY,x,y), eta_xx, modules=["numpy","sympy"])
    eta_xy_np = lambdify((lX,lY,x,y), eta_xy, modules=["numpy","sympy"])
    eta_yy_np = lambdify((lX,lY,x,y), eta_yy, modules=["numpy","sympy"])
    # Note that np in Laplacian_np stands for numpy.
    return eta_xx_np, eta_xy_np, eta_yy_np


eta_xx_np, eta_xy_np, eta_yy_np = SurfaceElevationDoubleDerivatives_2_FunctionalForm()


def SurfaceElevationDoubleDerivatives_2(lX,lY,x,y):
    eta_xx = eta_xx_np(lX,lY,x,y)
    eta_xy = eta_xy_np(lX,lY,x,y)
    eta_yy = eta_yy_np(lX,lY,x,y)
    return eta_xx, eta_xy, eta_yy


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


Laplacian_np = SurfaceElevationLaplacian_2_FunctionalForm()


def SurfaceElevationLaplacian_2(lX,lY,x,y):
    Laplacian = Laplacian_np(lX,lY,x,y)
    return Laplacian


def SurfaceElevationGradientOfLaplacian_2_FunctionalForm():
    lX = sp.Symbol('lX')
    lY = sp.Symbol('lY')
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    eta0 = 0.1
    eta = eta0*sp.exp(-((sp.sin(2.0*sp.pi*x/lX))**2.0 + (sp.sin(2.0*sp.pi*y/lY))**2.0))
    eta_xx = sp.diff(eta,x,x)
    eta_yy = sp.diff(eta,y,y)
    Laplacian = eta_xx + eta_yy
    Laplacian_x = sp.diff(Laplacian,x)
    Laplacian_y = sp.diff(Laplacian,y)
    Laplacian_x_np = lambdify((lX,lY,x,y), Laplacian_x, modules=["numpy","sympy"])
    Laplacian_y_np = lambdify((lX,lY,x,y), Laplacian_y, modules=["numpy","sympy"])
    # Note that np in Laplacian_x_np and Laplacian_y_np stands for numpy.
    return Laplacian_x_np, Laplacian_y_np


Laplacian_x_np, Laplacian_y_np = SurfaceElevationGradientOfLaplacian_2_FunctionalForm()


def SurfaceElevationGradientOfLaplacian_2(lX,lY,x,y):
    Laplacian_x = Laplacian_x_np(lX,lY,x,y)
    Laplacian_y = Laplacian_y_np(lX,lY,x,y)
    return Laplacian_x, Laplacian_y


MyExperimentNumber = 1 # Choose MyExperimentNumber to be 1 or 2.
if MyExperimentNumber == 1:
    SurfaceElevation = SurfaceElevation_1
    SurfaceElevationGradient = SurfaceElevationGradient_1
    SurfaceElevationDoubleDerivatives = SurfaceElevationDoubleDerivatives_1
    SurfaceElevationLaplacian = SurfaceElevationLaplacian_1
    SurfaceElevationGradientOfLaplacian = SurfaceElevationGradientOfLaplacian_1
    ProblemSpecificPrefix = ProblemSpecificPrefix_1
else: # if MyExperimentNumber == 2:
    SurfaceElevation = SurfaceElevation_2
    SurfaceElevationGradient = SurfaceElevationGradient_2
    SurfaceElevationDoubleDerivatives = SurfaceElevationDoubleDerivatives_2
    SurfaceElevationLaplacian = SurfaceElevationLaplacian_2
    SurfaceElevationGradientOfLaplacian = SurfaceElevationGradientOfLaplacian_2
    ProblemSpecificPrefix = ProblemSpecificPrefix_2


def SurfaceElevationGradientAtEdge_NormalComponent(lX,lY,x,y,angleEdge):
    eta_x, eta_y = SurfaceElevationGradient(lX,lY,x,y)
    eta_n = eta_x*np.cos(angleEdge) + eta_y*np.sin(angleEdge)
    return eta_n


def SurfaceElevationGradientAtEdge_TangentialComponent(lX,lY,x,y,angleEdge):
    eta_x, eta_y = SurfaceElevationGradient(lX,lY,x,y)
    eta_t = -eta_x*np.sin(angleEdge) + eta_y*np.cos(angleEdge)
    return eta_t


def Velocity(lX,lY,x,y):
    eta_x, eta_y = SurfaceElevationGradient(lX,lY,x,y) 
    f = 10.0**(-4.0)
    g = 10.0
    v = g*eta_x/f
    u = -g*eta_y/f
    return u, v


def VelocityAtEdge_NormalComponent(lX,lY,x,y,angleEdge):
    u, v = Velocity(lX,lY,x,y)
    u_n = u*np.cos(angleEdge) + v*np.sin(angleEdge)
    return u_n


def VelocityAtEdge_TangentialComponent(lX,lY,x,y,angleEdge):
    u, v = Velocity(lX,lY,x,y)
    u_t = -u*np.sin(angleEdge) + v*np.cos(angleEdge)
    return u_t


def ZonalVelocityGradientAtEdge_NormalComponent(lX,lY,x,y,angleEdge):
    f = 10.0**(-4.0)
    g = 10.0
    eta_xx, eta_xy, eta_yy = SurfaceElevationDoubleDerivatives(lX,lY,x,y)
    u_x = -g/f*eta_xy
    u_y = -g/f*eta_yy
    u_n = u_x*np.cos(angleEdge) + u_y*np.sin(angleEdge)
    return u_n


def MeridionalVelocityGradientAtEdge_NormalComponent(lX,lY,x,y,angleEdge):
    f = 10.0**(-4.0)
    g = 10.0
    eta_xx, eta_xy, eta_yy = SurfaceElevationDoubleDerivatives(lX,lY,x,y)
    v_x = g/f*eta_xx
    v_y = g/f*eta_xy
    v_n = v_x*np.cos(angleEdge) + v_y*np.sin(angleEdge)
    return v_n


def VelocityCurl(lX,lY,x,y):
    f = 10.0**(-4.0)
    g = 10.0
    zeta = g/f*SurfaceElevationLaplacian(lX,lY,x,y)
    return zeta


def VelocityLaplacian(lX,lY,x,y):
    f = 10.0**(-4.0)
    g = 10.0
    Laplacian_eta_x, Laplacian_eta_y = SurfaceElevationGradientOfLaplacian(lX,lY,x,y)
    Laplacian_v = g/f*Laplacian_eta_x
    Laplacian_u = -g/f*Laplacian_eta_y
    return Laplacian_u, Laplacian_v


def VelocityLaplacianAtEdge_NormalComponent(lX,lY,x,y,angleEdge):
    Laplacian_u, Laplacian_v = VelocityLaplacian(lX,lY,x,y)
    Laplacian_u_n = Laplacian_u*np.cos(angleEdge) + Laplacian_v*np.sin(angleEdge)
    return Laplacian_u_n


def VelocityGradientOfDivergenceAtEdge_NormalComponent(lX,lY,x,y,angleEdge):
    Grad_Divergence_u_n = 0.0
    return Grad_Divergence_u_n


def KineticEnergy(lX,lY,x,y):
    u, v = Velocity(lX,lY,x,y)
    KE = 0.5*(u**2.0 + v**2.0)
    return KE


def DetermineNormalComponentFromZonalAndMeridionalComponents(ZonalComponent,MeridionalComponent,angleEdge):
    NormalComponent = ZonalComponent*np.cos(angleEdge) + MeridionalComponent*np.sin(angleEdge)
    return NormalComponent


def DetermineNormalComponentsFromZonalAndMeridionalComponents(ZonalComponents,MeridionalComponents,angleEdges):
    nEdges = len(angleEdges)
    NormalComponents = np.zeros(nEdges)
    for iEdge in range(0,nEdges):
        NormalComponents[iEdge] = (ZonalComponents[iEdge]*np.cos(angleEdges[iEdge]) 
                                   + MeridionalComponents[iEdge]*np.sin(angleEdges[iEdge]))
    return NormalComponents


def DetermineTangentialComponentFromZonalAndMeridionalComponents(ZonalComponent,MeridionalComponent,angleEdge):
    TangentialComponent = MeridionalComponent*np.cos(angleEdge) - ZonalComponent*np.sin(angleEdge)
    return TangentialComponent


def DetermineTangentialComponentsFromZonalAndMeridionalComponents(ZonalComponents,MeridionalComponents,angleEdges):
    nEdges = len(angleEdges)
    TangentialComponents = np.zeros(nEdges)
    for iEdge in range(0,nEdges):
        TangentialComponents[iEdge] = (MeridionalComponents[iEdge]*np.cos(angleEdges[iEdge]) 
                                       - ZonalComponents[iEdge]*np.sin(angleEdges[iEdge]))
    return TangentialComponents


def DetermineZonalComponentFromNormalAndTangentialComponents(NormalComponent,TangentialComponent,angleEdge):
    ZonalComponent = NormalComponent*np.cos(angleEdge) - TangentialComponent*np.sin(angleEdge)
    return ZonalComponent


def DetermineZonalComponentsFromNormalAndTangentialComponents(NormalComponents,TangentialComponents,angleEdges):
    nEdges = len(angleEdges)
    ZonalComponents = np.zeros(nEdges)
    for iEdge in range(0,nEdges):
        ZonalComponents[iEdge] = (NormalComponents[iEdge]*np.cos(angleEdges[iEdge]) 
                                  - TangentialComponents[iEdge]*np.sin(angleEdges[iEdge]))
    return ZonalComponents


def DetermineMeridionalComponentFromNormalAndTangentialComponents(NormalComponent,TangentialComponent,angleEdge):
    MeridionalComponent = NormalComponent*np.sin(angleEdge) + TangentialComponent*np.cos(angleEdge)
    return MeridionalComponent


def DetermineMeridionalComponentsFromNormalAndTangentialComponents(NormalComponents,TangentialComponents,angleEdges):
    nEdges = len(angleEdges)
    MeridionalComponents = np.zeros(nEdges)
    for iEdge in range(0,nEdges):
        MeridionalComponents[iEdge] = (NormalComponents[iEdge]*np.sin(angleEdges[iEdge]) 
                                       + TangentialComponents[iEdge]*np.cos(angleEdges[iEdge]))
    return MeridionalComponents


def NumericalGradientOperatorAtEdge_NormalComponent(myMesh,myScalar,myScalarGradientAtEdge_NormalComponent_Function):
    lX = myMesh.lX
    lY = myMesh.lY
    myScalarGradientAtEdge_NormalComponent = np.zeros(myMesh.nEdges)
    for iEdge in range(0,myMesh.nEdges):
        xEdge = myMesh.xEdge[iEdge]
        yEdge = myMesh.yEdge[iEdge]
        dcEdge = myMesh.dcEdge[iEdge]
        angleEdge = myMesh.angleEdge[iEdge]
        if myMesh.boundaryEdge[iEdge] == 1.0:
            myScalarGradientAtEdge_NormalComponent[iEdge] = (
            myScalarGradientAtEdge_NormalComponent_Function(lX,lY,xEdge,yEdge,angleEdge))
        else:
            cellID1 = myMesh.cellsOnEdge[iEdge,0]
            cell1 = cellID1 - 1
            cellID2 = myMesh.cellsOnEdge[iEdge,1]
            cell2 = cellID2 - 1
            myScalarGradientAtEdge_NormalComponent[iEdge] = (myScalar[cell2] - myScalar[cell1])/dcEdge
    return myScalarGradientAtEdge_NormalComponent


def NumericalGradientOperatorAtEdge_TangentialComponent(myMesh,myScalarAtVertex):
    myScalarGradientAlongEdge = np.zeros(myMesh.nEdges)
    for iEdge in range(0,myMesh.nEdges):
        dvEdge = myMesh.dvEdge[iEdge]
        VertexID1 = myMesh.verticesOnEdge[iEdge,0]
        VertexID2 = myMesh.verticesOnEdge[iEdge,1]
        iVertex1 = VertexID1 - 1
        iVertex2 = VertexID2 - 1
        myScalarGradientAlongEdge[iEdge] = (myScalarAtVertex[iVertex2] - myScalarAtVertex[iVertex1])/dvEdge
    return myScalarGradientAlongEdge


def NumericalDivergenceOperatorAtCellCenter(myMesh,myVectorAtEdge_NormalComponent):
    myVectorDivergence = np.zeros(myMesh.nCells)
    for iCell in range(0,myMesh.nCells):
        for iEdgeOnCell in range(0,myMesh.nEdgesOnCell[iCell]):
            iEdgeID = myMesh.edgesOnCell[iCell,iEdgeOnCell]
            iEdge = iEdgeID - 1
            edgeSignOnCell = myMesh.edgeSignOnCell[iCell,iEdgeOnCell]
            dvEdge = myMesh.dvEdge[iEdge]
            myVectorDivergence[iCell] -= edgeSignOnCell*dvEdge*myVectorAtEdge_NormalComponent[iEdge]
        myVectorDivergence[iCell] /= myMesh.areaCell[iCell]
    return myVectorDivergence


def NumericalCurlOperator(myMesh,myVectorAtEdge_NormalComponent,AnalyticalCurlFunction,ReturnOnlyCurlAtVertex=False):
    lX = myMesh.lX
    lY = myMesh.lY
    myVectorCurlAtVertex = np.zeros(myMesh.nVertices)
    myVectorCurlAtEdge = np.zeros(myMesh.nEdges)
    myVectorCurlAtCellCenter = np.zeros(myMesh.nCells)
    for iVertex in range(0,myMesh.nVertices):
        xVertex = myMesh.xVertex[iVertex]
        yVertex = myMesh.yVertex[iVertex]
        if myMesh.boundaryVertex[iVertex] == 1.0:
            myVectorCurlAtVertex[iVertex] = AnalyticalCurlFunction(lX,lY,xVertex,yVertex)
        else:
            for iVertexDegree in range(0,myMesh.vertexDegree):
                iEdgeID = myMesh.edgesOnVertex[iVertex,iVertexDegree]
                iEdge = iEdgeID - 1
                myVectorCurlAtVertex[iVertex] += (myMesh.edgeSignOnVertex[iVertex,iVertexDegree]
                                                  *myMesh.dcEdge[iEdge]*myVectorAtEdge_NormalComponent[iEdge])
            myVectorCurlAtVertex[iVertex] /= myMesh.areaTriangle[iVertex]
    myVectorCurlAtCellCenter = myMesh.InterpolateSolutionFromVerticesToCellCenters(myVectorCurlAtVertex)
    myVectorCurlAtEdge = np.zeros(myMesh.nEdges)
    for iEdge in range(0,myMesh.nEdges):
        if myMesh.boundaryEdge[iEdge] == 1.0:
            xEdge = myMesh.xEdge[iEdge]
            yEdge = myMesh.yEdge[iEdge]
            myVectorCurlAtEdge[iEdge] = AnalyticalCurlFunction(lX,lY,xEdge,yEdge)    
    myVectorCurlAtEdge = myMesh.InterpolateSolutionFromVerticesToEdges(myVectorCurlAtVertex,myVectorCurlAtEdge)
    myVectorCurlAtVertex = myMesh.InterpolateSolutionFromCellCentersToVertices(myVectorCurlAtCellCenter,
                                                                               myVectorCurlAtVertex)
    if ReturnOnlyCurlAtVertex:
        return myVectorCurlAtVertex
    else:
        return myVectorCurlAtVertex, myVectorCurlAtEdge, myVectorCurlAtCellCenter


def NumericalTangentialOperatorAlongEdge(myMesh,myVectorAtEdge_NormalComponent,
                                         myVectorAtEdge_TangentialComponent_Function):
    lX = myMesh.lX
    lY = myMesh.lY
    myVectorAtEdge_TangentialComponent = np.zeros(myMesh.nEdges)
    for iEdge in range(0,myMesh.nEdges):
        xEdge = myMesh.xEdge[iEdge]
        yEdge = myMesh.yEdge[iEdge]
        angleEdge = myMesh.angleEdge[iEdge]
        if myMesh.boundaryEdge[iEdge] == 1.0:
            myVectorAtEdge_TangentialComponent[iEdge] = myVectorAtEdge_TangentialComponent_Function(lX,lY,xEdge,yEdge,
                                                                                                    angleEdge)
        else:
            myVectorAtEdge_TangentialComponent[iEdge] = 0.0
            # Compute the tangential velocities.
            for iEdgeOnEdge in range(0,myMesh.nEdgesOnEdge[iEdge]):
                eoeID = myMesh.edgesOnEdge[iEdge,iEdgeOnEdge]
                eoe = eoeID - 1
                weightsOnEdge_temp = myMesh.weightsOnEdge[iEdge,iEdgeOnEdge]
                myVectorAtEdge_TangentialComponent[iEdge] += weightsOnEdge_temp*myVectorAtEdge_NormalComponent[eoe]
    return myVectorAtEdge_TangentialComponent


def NumericalEnergyOperatorAtCellCenter(myMesh,myVectorAtEdge_NormalComponent):
    myVectorEnergyAtCellCenter = np.zeros(myMesh.nCells)
    for iCell in range(0,myMesh.nCells):
        for iEdgeOnCell in range(0,myMesh.nEdgesOnCell[iCell]):
            EdgeID = myMesh.edgesOnCell[iCell,iEdgeOnCell]
            iEdge = EdgeID - 1
            dvEdge = myMesh.dvEdge[iEdge]
            dcEdge = myMesh.dcEdge[iEdge]
            myVectorEnergyAtCellCenter[iCell] += dcEdge*dvEdge*(myVectorAtEdge_NormalComponent[iEdge])**2.0
        myVectorEnergyAtCellCenter[iCell] *= 0.25/myMesh.areaCell[iCell]
    return myVectorEnergyAtCellCenter


def NumericalLaplacianOperatorAtEdge_Method_1(
myMesh,myVectorAtEdge_NormalComponent,myVectorAtEdge_TangentialComponent_Function,
myVector_ZonalComponent_GradientAtEdge_NormalComponent_Function,
myVector_MeridionalComponent_GradientAtEdge_NormalComponent_Function,myVector_NormalComponent_LaplacianAtEdge_Function):
    lX = myMesh.lX
    lY = myMesh.lY
    myVectorAtEdge_TangentialComponent = (
    NumericalTangentialOperatorAlongEdge(myMesh,myVectorAtEdge_NormalComponent,
                                         myVectorAtEdge_TangentialComponent_Function))
    myVector_ZonalComponent_AtEdge = DetermineZonalComponentsFromNormalAndTangentialComponents(
    myVectorAtEdge_NormalComponent,myVectorAtEdge_TangentialComponent,myMesh.angleEdge)
    myVector_MeridionalComponent_AtEdge = DetermineMeridionalComponentsFromNormalAndTangentialComponents(
    myVectorAtEdge_NormalComponent,myVectorAtEdge_TangentialComponent,myMesh.angleEdge)
    myVector_ZonalComponent_AtCellCenter = (
    myMesh.InterpolateSolutionFromEdgesToCellCenters(myVector_ZonalComponent_AtEdge))
    myVector_MeridionalComponent_AtCellCenter = (
    myMesh.InterpolateSolutionFromEdgesToCellCenters(myVector_MeridionalComponent_AtEdge))
    myVector_ZonalComponent_GradientAtEdge_NormalComponent = (
    NumericalGradientOperatorAtEdge_NormalComponent(myMesh,myVector_ZonalComponent_AtCellCenter,
                                                    myVector_ZonalComponent_GradientAtEdge_NormalComponent_Function))
    myVector_MeridionalComponent_GradientAtEdge_NormalComponent = (
    NumericalGradientOperatorAtEdge_NormalComponent(
    myMesh,myVector_MeridionalComponent_AtCellCenter,
    myVector_MeridionalComponent_GradientAtEdge_NormalComponent_Function))
    myVector_ZonalComponent_LaplacianAtCellCenter = (
    NumericalDivergenceOperatorAtCellCenter(myMesh,myVector_ZonalComponent_GradientAtEdge_NormalComponent))
    myVector_MeridionalComponent_LaplacianAtCellCenter = (
    NumericalDivergenceOperatorAtCellCenter(myMesh,myVector_MeridionalComponent_GradientAtEdge_NormalComponent))
    myVector_ZonalComponent_LaplacianAtEdge = np.zeros(myMesh.nEdges)
    myVector_ZonalComponent_LaplacianAtEdge = (
    myMesh.InterpolateSolutionFromCellCentersToEdges(myVector_ZonalComponent_LaplacianAtCellCenter,
                                                     myVector_ZonalComponent_LaplacianAtEdge))
    myVector_MeridionalComponent_LaplacianAtEdge = np.zeros(myMesh.nEdges)
    myVector_MeridionalComponent_LaplacianAtEdge = (
    myMesh.InterpolateSolutionFromCellCentersToEdges(myVector_MeridionalComponent_LaplacianAtCellCenter,
                                                     myVector_MeridionalComponent_LaplacianAtEdge))
    myVectorAtEdge_NormalComponent_Laplacian = DetermineNormalComponentFromZonalAndMeridionalComponents(
    myVector_ZonalComponent_LaplacianAtEdge,myVector_MeridionalComponent_LaplacianAtEdge,myMesh.angleEdge)
    for iEdge in range(0,myMesh.nEdges):
        if myMesh.boundaryEdge[iEdge] == 1.0:
            xEdge = myMesh.xEdge[iEdge]
            yEdge = myMesh.yEdge[iEdge]
            angleEdge = myMesh.angleEdge[iEdge]
            myVectorAtEdge_NormalComponent_Laplacian[iEdge] = (
            myVector_NormalComponent_LaplacianAtEdge_Function(lX,lY,xEdge,yEdge,angleEdge))
    return myVectorAtEdge_NormalComponent_Laplacian


def NumericalLaplacianOperatorAtEdge_Method_2(myMesh,myVectorAtEdge_NormalComponent,AnalyticalCurlFunction,
                                              myVector_GradientOfDivergenceAtEdge_NormalComponent_Function,
                                              myVector_NormalComponent_LaplacianAtEdge_Function):
    lX = myMesh.lX
    lY = myMesh.lY
    myVector_DivergenceAtCellCenter = NumericalDivergenceOperatorAtCellCenter(myMesh,myVectorAtEdge_NormalComponent)
    myVector_CurlAtVertex = NumericalCurlOperator(myMesh,myVectorAtEdge_NormalComponent,AnalyticalCurlFunction,
                                                  ReturnOnlyCurlAtVertex=True)
    myVector_GradientOfDivergenceAtEdge_NormalComponent = (
    NumericalGradientOperatorAtEdge_NormalComponent(myMesh,myVector_DivergenceAtCellCenter,
                                                    myVector_GradientOfDivergenceAtEdge_NormalComponent_Function))
    myVector_GradientOfCurlAtEdge_TangentialComponent = (
    NumericalGradientOperatorAtEdge_TangentialComponent(myMesh,myVector_CurlAtVertex))
    myVectorAtEdge_NormalComponent_Laplacian = np.zeros(myMesh.nEdges)
    for iEdge in range(0,myMesh.nEdges):
        if myMesh.boundaryEdge[iEdge] == 1.0:
            xEdge = myMesh.xEdge[iEdge]
            yEdge = myMesh.yEdge[iEdge]
            angleEdge = myMesh.angleEdge[iEdge]
            myVectorAtEdge_NormalComponent_Laplacian[iEdge] = (
            myVector_NormalComponent_LaplacianAtEdge_Function(lX,lY,xEdge,yEdge,angleEdge))
        else:
            myVectorAtEdge_NormalComponent_Laplacian[iEdge] = (
            (myVector_GradientOfDivergenceAtEdge_NormalComponent[iEdge] 
             - myVector_GradientOfCurlAtEdge_TangentialComponent[iEdge]))
    return myVectorAtEdge_NormalComponent_Laplacian