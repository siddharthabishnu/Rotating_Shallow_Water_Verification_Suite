"""
Name: Test_LegendreGaussLobattoQuadrature1DClass.py
Author: Sid Bishnu
Details: As the name implies, this script tests the Legendre Gauss-Lobatto quadrature class defined in
../../src/DGSEM_Rotating_Shallow_Water/LegendreGaussLobattoQuadrature1DClass.py.
"""


import numpy as np
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/DGSEM_Rotating_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR
    import LegendreGaussLobattoQuadrature1DClass as LGLQ1D
    
    
def TestLegendreGaussLobattoQuadrature1D_PlotWeightsVsNodes(nX):
    myLegendreGaussLobattoQuadrature1D = LGLQ1D.LegendreGaussLobattoQuadrature1D(nX,True,True)
    OutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output/'
    CR.PythonPlot1DSaveAsPDF(OutputDirectory,'regular',myLegendreGaussLobattoQuadrature1D.x,
                             myLegendreGaussLobattoQuadrature1D.w,2.0,'-','k',True,'s',7.5,
                             ['Legendre Gauss-Lobatto Nodes','Legendre Gauss-Lobatto Weights'],[17.5,17.5],[10.0,10.0],
                             [15.0,15.0],'Legendre Gauss-Lobatto Weights vs Nodes',20.0,True,
                             'LegendreGaussLobattoWeightsVsNodes',False)


do_TestLegendreGaussLobattoQuadrature1D_PlotWeightsVsNodes = False
if do_TestLegendreGaussLobattoQuadrature1D_PlotWeightsVsNodes:
    TestLegendreGaussLobattoQuadrature1D_PlotWeightsVsNodes(nX=20)


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


def TestLegendreGaussLobattoQuadrature1D_ConvergenceStudy(functionType,nXLimits):
    if functionType == 'Polynomial':
        TestFunction = PolynomialTestFunction
        TestFunctionIntegral = PolynomialTestFunctionIntegral
        Title = (
        'Gauss-Lobatto Quadrature Errror for Polynomial Function\nf(x) = x**21 + x**20 + ... + x**2 + x + 1')
    elif functionType == 'Trigonometric':
        TestFunction = TrigonometricTestFunction
        TestFunctionIntegral = TrigonometricTestFunctionIntegral
        Title = 'Gauss-Lobatto Quadrature Errror for Trigonometric Function\nf(x) = cos(pi*(x+1))'
    elif functionType == 'Exponential':
        TestFunction = ExponentialTestFunction
        TestFunctionIntegral = ExponentialTestFunctionIntegral    
        Title = 'Gauss-Lobatto Quadrature Errror for Exponential Function\nf(x) = exp(x)'
    elif functionType == 'Logarithmic':
        TestFunction = LogarithmicTestFunction
        TestFunctionIntegral = LogarithmicTestFunctionIntegral
        Title = 'Gauss-Lobatto Quadrature Errror for Logarithmic Function\nf(x) = log(x+2.0)'
    ExactIntegral = TestFunctionIntegral([-1.0,1.0])    
    nXMin = nXLimits[0]
    nXMax = nXLimits[1]
    n_nX = nXMax - nXMin + 1
    nX = np.linspace(nXMin,nXMax,n_nX,dtype=int)
    NumericalIntegral = np.zeros(n_nX)
    NumericalIntegralError = np.zeros(n_nX)
    epsilon = np.finfo(float).eps
    for i_nX in range(0,n_nX):
        myLegendreGaussLobattoQuadrature1D = LGLQ1D.LegendreGaussLobattoQuadrature1D(nX[i_nX])
        functionValues = np.zeros(nX[i_nX]+1)
        for iX in range(0,nX[i_nX]+1):
            functionValues[iX] = TestFunction(myLegendreGaussLobattoQuadrature1D.x[iX])
        NumericalIntegral[i_nX] = myLegendreGaussLobattoQuadrature1D.EvaluateQuadrature(functionValues)
        NumericalIntegralError[i_nX] = abs(NumericalIntegral[i_nX] - ExactIntegral)
        if NumericalIntegralError[i_nX] < epsilon:
            NumericalIntegralError[i_nX] = epsilon
    xLabel = 'Number of Legendre Gauss Nodes'
    yLabel = 'Legendre Gauss-Lobatto Quadrature Error'
    FileName = 'ConvergencePlot_LegendreGaussLobattoQuadratureError_' + functionType + 'Function'
    OutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output/'
    CR.PythonPlot1DSaveAsPDF(OutputDirectory,'semi-log_y',nX+1,NumericalIntegralError,2.0,'-','k',True,'s',7.5,
                             [xLabel,yLabel],[17.5,17.5],[10.0,10.0],[15.0,15.0],Title,20.0,True,FileName,False)


do_TestLegendreGaussLobattoQuadrature1D_ConvergenceStudy = False
if do_TestLegendreGaussLobattoQuadrature1D_ConvergenceStudy:
    nXLimits = [2,15]
    TestLegendreGaussLobattoQuadrature1D_ConvergenceStudy('Polynomial',nXLimits)
    TestLegendreGaussLobattoQuadrature1D_ConvergenceStudy('Trigonometric',nXLimits)
    TestLegendreGaussLobattoQuadrature1D_ConvergenceStudy('Exponential',nXLimits)
    TestLegendreGaussLobattoQuadrature1D_ConvergenceStudy('Logarithmic',nXLimits)