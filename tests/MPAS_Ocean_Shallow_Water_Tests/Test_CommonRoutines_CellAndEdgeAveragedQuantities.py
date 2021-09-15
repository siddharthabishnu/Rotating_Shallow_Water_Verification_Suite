"""
Name: Test_CommonRoutines_CellAndEdgeAveragedQuantities.py
Author: Siddhartha Bishnu
Details: As the name implies, this script tests various functions of 
../../src/MPAS_Ocean_Shallow_Water/CommonRoutines_CellAndEdgeAveragedQuantities.py.
"""


import numpy as np
import sympy as sp
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/MPAS_Ocean_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR
    import CommonRoutines_CellAndEdgeAveragedQuantities as CR_CEAQ


def TestLegendreGaussQuadrature1D_PlotWeightsVsNodes(nX):
    myLegendreGaussQuadrature1D = CR_CEAQ.LegendreGaussQuadrature1D(nX,True,True)
    CR.PythonPlot1DSaveAsPDF('../../output/MPAS_Ocean_Shallow_Water_Output/','regular',myLegendreGaussQuadrature1D.x,
                             myLegendreGaussQuadrature1D.w,2.0,'-','k',True,'s',7.5,
                             ['Legendre Gauss Nodes','Legendre Gauss Weights'],[17.5,17.5],[10.0,10.0],[15.0,15.0],
                             'Legendre Gauss Weights vs Nodes',20.0,True,'LegendreGaussWeightsVsNodes',False)
    

do_TestLegendreGaussQuadrature1D_PlotWeightsVsNodes = False
if do_TestLegendreGaussQuadrature1D_PlotWeightsVsNodes:
    TestLegendreGaussQuadrature1D_PlotWeightsVsNodes(nX=20)


def PolynomialTestFunction(x,polynomialOrder=21):
    function = 0.0
    for i in range(0,polynomialOrder+1):
        function += x**float(i)
    return function


def PolynomialTestFunctionIntegral(xLimits,polynomialOrder=21):
    functionIntegral = 0.0
    for i in range(0,polynomialOrder+1):
        functionIntegral += xLimits[1]**float(i+1)/float(i+1) - xLimits[0]**float(i+1)/float(i+1)
    return functionIntegral


def TrigonometricTestFunction(x):
    function = np.cos(np.pi*(x+1.0))
    return function


def TrigonometricTestFunctionIntegral(xLimits):
    functionIntegral = (np.sin(np.pi*(xLimits[1]+1.0)) - np.sin(np.pi*(xLimits[0]+1.0)))/np.pi
    return functionIntegral


def ExponentialTestFunction(x):
    function = np.exp(x)
    return function


def ExponentialTestFunctionIntegral(xLimits):
    functionIntegral = np.exp(xLimits[1]) - np.exp(xLimits[0])
    return functionIntegral


def LogarithmicTestFunction(x,a=2.0):
    function = np.log(x+a)
    return function


def LogarithmicTestFunctionIntegral(xLimits,a=2.0):
    functionIntegral = (
    ((xLimits[1]+a)*np.log(xLimits[1]+a) - xLimits[1]) - ((xLimits[0]+a)*np.log(xLimits[0]+a) - xLimits[0]))
    return functionIntegral


def TestLegendreGaussQuadrature1D_ConvergenceStudy(functionType,nXLimits):
    if functionType == 'Polynomial':
        TestFunction = PolynomialTestFunction
        TestFunctionIntegral = PolynomialTestFunctionIntegral
        Title = 'Gauss Quadrature Errror for Polynomial Function\nf(x) = x**21 + x**20 + ... + x**2 + x + 1'
    elif functionType == 'Trigonometric':
        TestFunction = TrigonometricTestFunction
        TestFunctionIntegral = TrigonometricTestFunctionIntegral
        Title = 'Gauss Quadrature Errror for Trigonometric Function\nf(x) = cos(pi*(x+1))'
    elif functionType == 'Exponential':
        TestFunction = ExponentialTestFunction
        TestFunctionIntegral = ExponentialTestFunctionIntegral    
        Title = 'Gauss Quadrature Errror for Exponential Function\nf(x) = exp(x)'
    elif functionType == 'Logarithmic':
        TestFunction = LogarithmicTestFunction
        TestFunctionIntegral = LogarithmicTestFunctionIntegral
        Title = 'Gauss Quadrature Errror for Logarithmic Function\nf(x) = log(x+2.0)'
    ExactIntegral = TestFunctionIntegral([-1.0,1.0])    
    nXMin = nXLimits[0]
    nXMax = nXLimits[1]
    n_nX = nXMax - nXMin + 1
    nX = np.linspace(nXMin,nXMax,n_nX,dtype=int)
    NumericalIntegral = np.zeros(n_nX)
    NumericalIntegralError = np.zeros(n_nX)
    epsilon = np.finfo(float).eps
    for i_nX in range(0,n_nX):
        myLegendreGaussQuadrature1D = CR_CEAQ.LegendreGaussQuadrature1D(nX[i_nX])
        functionValues = np.zeros(nX[i_nX]+1)
        for iX in range(0,nX[i_nX]+1):
            functionValues[iX] = TestFunction(myLegendreGaussQuadrature1D.x[iX])
        NumericalIntegral[i_nX] = myLegendreGaussQuadrature1D.EvaluateQuadrature(functionValues)
        NumericalIntegralError[i_nX] = abs(NumericalIntegral[i_nX] - ExactIntegral)
        if NumericalIntegralError[i_nX] < epsilon:
            NumericalIntegralError[i_nX] = epsilon
    xLabel = 'Number of Legendre Gauss Nodes'
    yLabel = 'Legendre Gauss Quadrature Error'
    FileName = 'ConvergencePlot_LegendreGaussQuadratureError_' + functionType + 'Function'
    CR.PythonPlot1DSaveAsPDF('../../output/MPAS_Ocean_Shallow_Water_Output/','semi-log_y',nX+1,NumericalIntegralError,
                             2.0,'-','k',True,'s',7.5,[xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,
                             True,FileName,False)


do_TestLegendreGaussQuadrature1D_ConvergenceStudy = False
if do_TestLegendreGaussQuadrature1D_ConvergenceStudy:
    nXLimits = [2,15]
    TestLegendreGaussQuadrature1D_ConvergenceStudy('Polynomial',nXLimits)
    TestLegendreGaussQuadrature1D_ConvergenceStudy('Trigonometric',nXLimits)
    TestLegendreGaussQuadrature1D_ConvergenceStudy('Exponential',nXLimits)
    TestLegendreGaussQuadrature1D_ConvergenceStudy('Logarithmic',nXLimits)
    

def TestQuadratureOnHexagon():
    x0 = 10.0
    y0 = 10.0
    l = 10.0
    print('x0 = %d' %x0)
    print('y0 = %d' %y0)
    print('l = %d' %l) 
    x, y = sp.symbols('x y')
    print('\nf(x,y) = 1')
    print('Computing quadrature of f(x,y) over a regular hexagon of unit length centered at the origin:')
    f = 1.0
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon()
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(np.ones(myQuadratureOnHexagon.n))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %myQuadratureOnHexagon.a)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = 1')
    print('Computing quadrature of f(x,y) over a regular hexagon of length 10.0 centered at [10.0,10.0]:')
    f = 1.0
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon(origin=[x0,y0],l=l)
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(np.ones(myQuadratureOnHexagon.n))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %myQuadratureOnHexagon.a)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = x**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of unit length centered at the origin:')
    f = x**2
    A1 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 - y),np.sqrt(3.0)*(1.0 - y)),(y,0.5,1.0))
    A2 = sp.integrate(f,(x,-np.sqrt(3.0)/2.0,np.sqrt(3.0)/2.0),(y,-0.5,0.5))
    A3 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 + y),np.sqrt(3.0)*(1.0 + y)),(y,-1.0,-0.5))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon()
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(myQuadratureOnHexagon.x**2.0)
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = ((x - x0)/l)**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of length 10.0 centered at [10.0,10.0]:')
    f = ((x - x0)/l)**2
    A1 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y0+l-y),x0+np.sqrt(3.0)*(y0+l-y)),(y,y0+0.5*l,y0+l))
    A2 = sp.integrate(f,(x,x0-np.sqrt(3.0)/2.0*l,x0+np.sqrt(3.0)/2.0*l),(y,y0-0.5*l,y0+0.5*l))
    A3 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y-y0+l),x0+np.sqrt(3.0)*(y-y0+l)),(y,y0-l,y0-0.5*l))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon(origin=[x0,y0],l=l)
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(((myQuadratureOnHexagon.x - x0)/l)**2.0)
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = y**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of unit length centered at the origin:')
    f = y**2
    A1 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 - y),np.sqrt(3.0)*(1.0 - y)),(y,0.5,1.0))
    A2 = sp.integrate(f,(x,-np.sqrt(3.0)/2.0,np.sqrt(3.0)/2.0),(y,-0.5,0.5))
    A3 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 + y),np.sqrt(3.0)*(1.0 + y)),(y,-1.0,-0.5))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon()
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(myQuadratureOnHexagon.y**2.0)
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = ((y - y0)/l)**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of length 10.0 centered at [10.0,10.0]:')
    f = ((y - y0)/l)**2
    A1 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y0+l-y),x0+np.sqrt(3.0)*(y0+l-y)),(y,y0+0.5*l,y0+l))
    A2 = sp.integrate(f,(x,x0-np.sqrt(3.0)/2.0*l,x0+np.sqrt(3.0)/2.0*l),(y,y0-0.5*l,y0+0.5*l))
    A3 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y-y0+l),x0+np.sqrt(3.0)*(y-y0+l)),(y,y0-l,y0-0.5*l))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon(origin=[x0,y0],l=l)
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(((myQuadratureOnHexagon.y - y0)/l)**2.0)
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = x**2*y**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of unit length centered at the origin:')
    f = x**2*y**2
    A1 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 - y),np.sqrt(3.0)*(1.0 - y)),(y,0.5,1.0))
    A2 = sp.integrate(f,(x,-np.sqrt(3.0)/2.0,np.sqrt(3.0)/2.0),(y,-0.5,0.5))
    A3 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 + y),np.sqrt(3.0)*(1.0 + y)),(y,-1.0,-0.5))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon()
    NumericalIntegral = (
    myQuadratureOnHexagon.DetermineQuadrature(myQuadratureOnHexagon.x**2.0*myQuadratureOnHexagon.y**2.0))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = ((x - x0)/l)**2*((y - y0)/l)**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of length 10.0 centered at [10.0,10.0]:')
    f = ((x - x0)/l)**2*((y - y0)/l)**2
    A1 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y0+l-y),x0+np.sqrt(3.0)*(y0+l-y)),(y,y0+0.5*l,y0+l))
    A2 = sp.integrate(f,(x,x0-np.sqrt(3.0)/2.0*l,x0+np.sqrt(3.0)/2.0*l),(y,y0-0.5*l,y0+0.5*l))
    A3 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y-y0+l),x0+np.sqrt(3.0)*(y-y0+l)),(y,y0-l,y0-0.5*l))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon(origin=[x0,y0],l=l)
    NumericalIntegral = (
    myQuadratureOnHexagon.DetermineQuadrature(((myQuadratureOnHexagon.x - x0)/l)**2.0
                                              *((myQuadratureOnHexagon.y - y0)/l)**2.0))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = (sin x)**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of unit length centered at the origin:')
    f = (sp.sin(x))**2
    A1 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 - y),np.sqrt(3.0)*(1.0 - y)),(y,0.5,1.0))
    A2 = sp.integrate(f,(x,-np.sqrt(3.0)/2.0,np.sqrt(3.0)/2.0),(y,-0.5,0.5))
    A3 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 + y),np.sqrt(3.0)*(1.0 + y)),(y,-1.0,-0.5))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon()
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature((np.sin(myQuadratureOnHexagon.x))**2.0)
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = (sin ((x - x0)/l))**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of length 10.0 centered at [10.0,10.0]:')
    f = (sp.sin((x - x0)/l))**2
    A1 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y0+l-y),x0+np.sqrt(3.0)*(y0+l-y)),(y,y0+0.5*l,y0+l))
    A2 = sp.integrate(f,(x,x0-np.sqrt(3.0)/2.0*l,x0+np.sqrt(3.0)/2.0*l),(y,y0-0.5*l,y0+0.5*l))
    A3 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y-y0+l),x0+np.sqrt(3.0)*(y-y0+l)),(y,y0-l,y0-0.5*l))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon(origin=[x0,y0],l=l)
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature((np.sin((myQuadratureOnHexagon.x - x0)/l))**2.0)
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = (sin y)**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of unit length centered at the origin:')
    f = (sp.sin(y))**2
    A1 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 - y),np.sqrt(3.0)*(1.0 - y)),(y,0.5,1.0))
    A2 = sp.integrate(f,(x,-np.sqrt(3.0)/2.0,np.sqrt(3.0)/2.0),(y,-0.5,0.5))
    A3 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 + y),np.sqrt(3.0)*(1.0 + y)),(y,-1.0,-0.5))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon()
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature((np.sin(myQuadratureOnHexagon.y))**2.0)
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = (sin ((y - y0)/l))**2')
    print('Computing quadrature of f(x,y) over a regular hexagon of length 10.0 centered at [10.0,10.0]:')
    f = sp.sin((y - y0)/l)**2
    A1 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y0+l-y),x0+np.sqrt(3.0)*(y0+l-y)),(y,y0+0.5*l,y0+l))
    A2 = sp.integrate(f,(x,x0-np.sqrt(3.0)/2.0*l,x0+np.sqrt(3.0)/2.0*l),(y,y0-0.5*l,y0+0.5*l))
    A3 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y-y0+l),x0+np.sqrt(3.0)*(y-y0+l)),(y,y0-l,y0-0.5*l))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon(origin=[x0,y0],l=l)
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature((np.sin((myQuadratureOnHexagon.y - y0)/l))**2.0)
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = exp(-x)')
    print('Computing quadrature of f(x,y) over a regular hexagon of unit length centered at the origin:')
    f = sp.exp(-x)
    A1 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 - y),np.sqrt(3.0)*(1.0 - y)),(y,0.5,1.0))
    A2 = sp.integrate(f,(x,-np.sqrt(3.0)/2.0,np.sqrt(3.0)/2.0),(y,-0.5,0.5))
    A3 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 + y),np.sqrt(3.0)*(1.0 + y)),(y,-1.0,-0.5))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon()
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(np.exp(-myQuadratureOnHexagon.x))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = exp(-(x - x0)/l)')
    print('Computing quadrature of f(x,y) over a regular hexagon of length 10.0 centered at [10.0,10.0]:')
    f = sp.exp(-(x - x0)/l)
    A1 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y0+l-y),x0+np.sqrt(3.0)*(y0+l-y)),(y,y0+0.5*l,y0+l))
    A2 = sp.integrate(f,(x,x0-np.sqrt(3.0)/2.0*l,x0+np.sqrt(3.0)/2.0*l),(y,y0-0.5*l,y0+0.5*l))
    A3 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y-y0+l),x0+np.sqrt(3.0)*(y-y0+l)),(y,y0-l,y0-0.5*l))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon(origin=[x0,y0],l=l)
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(np.exp(-(myQuadratureOnHexagon.x - x0)/l))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = exp(-y)')
    print('Computing quadrature of f(x,y) over a regular hexagon of unit length centered at the origin:')
    f = sp.exp(-y)
    A1 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 - y),np.sqrt(3.0)*(1.0 - y)),(y,0.5,1.0))
    A2 = sp.integrate(f,(x,-np.sqrt(3.0)/2.0,np.sqrt(3.0)/2.0),(y,-0.5,0.5))
    A3 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 + y),np.sqrt(3.0)*(1.0 + y)),(y,-1.0,-0.5))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon()
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(np.exp(-myQuadratureOnHexagon.y))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = exp(-(y - y0)/l)')
    print('Computing quadrature of f(x,y) over a regular hexagon of length 10.0 centered at [10.0,10.0]:')
    f = sp.exp(-(y - y0)/l)
    A1 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y0+l-y),x0+np.sqrt(3.0)*(y0+l-y)),(y,y0+0.5*l,y0+l))
    A2 = sp.integrate(f,(x,x0-np.sqrt(3.0)/2.0*l,x0+np.sqrt(3.0)/2.0*l),(y,y0-0.5*l,y0+0.5*l))
    A3 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y-y0+l),x0+np.sqrt(3.0)*(y-y0+l)),(y,y0-l,y0-0.5*l))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon(origin=[x0,y0],l=l)
    NumericalIntegral = myQuadratureOnHexagon.DetermineQuadrature(np.exp(-(myQuadratureOnHexagon.y - y0)/l))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = exp(-x)*exp(-y)')
    print('Computing quadrature of f(x,y) over a regular hexagon of unit length centered at the origin:')
    f = sp.exp(-x)*sp.exp(-y)
    A1 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 - y),np.sqrt(3.0)*(1.0 - y)),(y,0.5,1.0))
    A2 = sp.integrate(f,(x,-np.sqrt(3.0)/2.0,np.sqrt(3.0)/2.0),(y,-0.5,0.5))
    A3 = sp.integrate(f,(x,-np.sqrt(3.0)*(1.0 + y),np.sqrt(3.0)*(1.0 + y)),(y,-1.0,-0.5))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon()
    NumericalIntegral = (
    myQuadratureOnHexagon.DetermineQuadrature(np.exp(-myQuadratureOnHexagon.x)*np.exp(-myQuadratureOnHexagon.y)))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    print('\nf(x,y) = exp(-(x - x0)/l)*exp(-(y - y0)/l)')
    print('Computing quadrature of f(x,y) over a regular hexagon of length 10.0 centered at [10.0,10.0]:')
    f = sp.exp(-(x - x0)/l)*sp.exp(-(y - y0)/l) 
    A1 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y0+l-y),x0+np.sqrt(3.0)*(y0+l-y)),(y,y0+0.5*l,y0+l))
    A2 = sp.integrate(f,(x,x0-np.sqrt(3.0)/2.0*l,x0+np.sqrt(3.0)/2.0*l),(y,y0-0.5*l,y0+0.5*l))
    A3 = sp.integrate(f,(x,x0-np.sqrt(3.0)*(y-y0+l),x0+np.sqrt(3.0)*(y-y0+l)),(y,y0-l,y0-0.5*l))
    A = A1 + A2 + A3
    myQuadratureOnHexagon = CR_CEAQ.QuadratureOnHexagon(origin=[x0,y0],l=l)
    NumericalIntegral = (
    myQuadratureOnHexagon.DetermineQuadrature(np.exp(-(myQuadratureOnHexagon.x - x0)/l)
                                              *np.exp(-(myQuadratureOnHexagon.y - y0)/l)))
    print('The exact integral of f(x,y) over the hexagon is %.15f.' %A)
    print('The quadrature i.e. the numerical integral of f(x,y) over the hexagon is %.15f.' %NumericalIntegral)
    
    
do_TestQuadratureOnHexagon = False
if do_TestQuadratureOnHexagon:
    TestQuadratureOnHexagon()