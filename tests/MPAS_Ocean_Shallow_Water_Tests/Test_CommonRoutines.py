"""
Name: Test_CommonRoutines.py
Author: Siddhartha Bishnu
Details: As the name implies, this script tests various functions of 
../../src/MPAS_Ocean_Shallow_Water/CommonRoutines.py.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import sys
sys.path.append(os.path.realpath('../..') + '/src/MPAS_Ocean_Shallow_Water/')
from IPython.utils import io
with io.capture_output() as captured:
    import CommonRoutines as CR


def TestPrintDataType():
    x = 10.0
    CR.PrintDataType(x)
    

do_TestPrintDataType = False
if do_TestPrintDataType:
    TestPrintDataType()


def TestInitializeCharacterArrays():
    CR.InitializeCharacterArrays()
    
    
do_TestInitializeCharacterArrays = False
if do_TestInitializeCharacterArrays:
    TestInitializeCharacterArrays()


def TestRemoveElementFrom1DArray():
    x = np.linspace(1.0,5.0,5)
    index = 1
    xNew = CR.RemoveElementFrom1DArray(x,index)
    print('The array x is')
    print(x)
    print('After removing element with index %d, the array x becomes' %index)
    print(xNew)
    
    
do_TestRemoveElementFrom1DArray = False
if do_TestRemoveElementFrom1DArray:
    TestRemoveElementFrom1DArray()
    

def TestRoundArray():
    x = np.array([-1.75,1.75,-2.25,2.25])
    x_rounded = CR.RoundArray(x)
    print('The array x is')
    print(x)
    print('After rounding, the array x becomes')
    print(x_rounded)


do_TestRoundArray = False
if do_TestRoundArray:
    TestRoundArray()
    
    
def TestComputeRelativeError():
    ExactSolutions = np.array([100.0,0.0,0.0,5.0])
    NumericalSolutions = np.array([105.0,0.0,5.0,0.0])
    nCases = len(ExactSolutions)
    for iCase in range(0,nCases):
        ExactSolution = ExactSolutions[iCase]
        NumericalSolution = NumericalSolutions[iCase]
        RelativeError = CR.ComputeRelativeError(ExactSolution,NumericalSolution)
        if iCase != 0:
            print(' ')
        print('The exact solution is %.2f.' %ExactSolution)
        print('The numerical solution is %.2f.' %NumericalSolution)
        print('The relative error is %.2f.' %RelativeError)
    
    
do_TestComputeRelativeError = False
if do_TestComputeRelativeError:
    TestComputeRelativeError()


def TestPythonPlot1DSaveAsPDF():
    x = np.arange(0.0,10.0,1.0) # Syntax is x = np.arange(First Point, Last Point, Interval).
    y = np.arange(0.0,20.0,2.0)
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    CR.PythonPlot1DSaveAsPDF(output_directory,'regular',x,y,2.0,'-','k',True,'s',7.5,['x','y'],[17.5,17.5],[10.0,10.0],
                             [15.0,15.0],'Python Plot 1D',20.0,True,'PythonPlot1D',False)


do_TestPythonPlot1DSaveAsPDF = False
if do_TestPythonPlot1DSaveAsPDF:
    TestPythonPlot1DSaveAsPDF()
    
    
def TestPythonPlots1DSaveAsPDF():
    x = np.arange(0.0,10.0,1) # Syntax is x = np.arange(First Point, Last Point, Interval).
    y1 = np.arange(0.0,20.0,2)
    y2 = np.arange(0.0,40.0,4)
    yAll = np.zeros((2,len(x)))
    yAll[0,:] = y1
    yAll[1,:] = y2
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    CR.PythonPlots1DSaveAsPDF(output_directory,'regular',x,yAll,[2.0,2.0],['-','--'],['r','b'],[True,True],['s','D'],
                              [10.0,10.0],['x','y'],[17.5,17.5],[10.0,10.0],[15.0,15.0],['y1','y2'],17.5,'center left',
                              'Python Plots 1D',20.0,True,'PythonPlots1D',False)


do_TestPythonPlots1DSaveAsPDF = False
if do_TestPythonPlots1DSaveAsPDF:
    TestPythonPlots1DSaveAsPDF()
    
    
def TestPythonConvergencePlot1DSaveAsPDF():
    x = np.arange(0.0,10.0,1) # Syntax is x = np.arange(First Point, Last Point, Interval)
    y1 = np.arange(0.0,20.0,2)
    y2 = np.arange(0.0,20.0,2)
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    CR.PythonConvergencePlot1DSaveAsPDF(output_directory,'regular',x,y1,y2,[2.0,2.0],['-',' '],['k','k'],[False,True],
                                        ['s','s'],[10.0,10.0],['x','y'],[17.5,17.5],[10.0,10.0],[15.0,15.0],['y1','y2'],
                                        17.5,'upper left','Convergence Plot 1D',20.0,True,'ConvergencePlot1D',False,
                                        drawMajorGrid=True,legendWithinBox=True)


do_TestPythonConvergencePlot1DSaveAsPDF = False
if do_TestPythonConvergencePlot1DSaveAsPDF:
    TestPythonConvergencePlot1DSaveAsPDF()
    
    
def TestPythonConvergencePlots1DSaveAsPDF():
    x = np.arange(0.0,10.0,1) # Syntax is x = np.arange(First Point, Last Point, Interval)
    nPlots = 2
    nSubplots = 2
    y = np.zeros((nPlots,nSubplots,len(x)))
    y[0,0,:] = np.arange(0.0,20.0,2)
    y[0,1,:] = y[0,0,:]
    y[1,0,:] = np.arange(0.0,40.0,4)
    y[1,1,:] = y[1,0,:]
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    CR.PythonConvergencePlots1DSaveAsPDF(output_directory,'regular',x,y,[2.0,2.0],['-','--'],['r','b'],['s','s'],
                                         [10.0,10.0],['x','y'],[17.5,17.5],[10.0,10.0],[15.0,15.0],['y1','y2'],17.5,
                                         'upper left','Convergence Plots 1D',20.0,True,'ConvergencePlots1D',False,
                                         drawMajorGrid=True,legendWithinBox=True)


do_TestPythonConvergencePlots1DSaveAsPDF = False
if do_TestPythonConvergencePlots1DSaveAsPDF:
    TestPythonConvergencePlots1DSaveAsPDF()
    

def TestScatterPlot():
    xLeft = 0.0
    xRight = 6.0
    nX = 6
    x = np.linspace(xLeft,xRight,nX+1) 
    yBottom = 0.0
    yTop = 5.0
    nY = 5
    y = np.linspace(yBottom,yTop,nY+1)
    xUnstructured = np.zeros((nX+1)*(nY+1))
    yUnstructured = np.zeros((nX+1)*(nY+1))
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            i = iY*(nX+1) + iX
            xUnstructured[i] = x[iX]
            yUnstructured[i] = y[iY]
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    CR.ScatterPlot(output_directory,xUnstructured,yUnstructured,'k','s',7.5,['x','y'],[17.5,17.5],[10.0,10.0],
                   [15.0,15.0],'Scatter Plot',20.0,True,'ScatterPlot',False,fig_size=[9.25,9.25],
                   UseDefaultMethodToSpecifyTickFontSize=True,titlepad=1.035)
    
    
do_TestScatterPlot = False
if do_TestScatterPlot:
    TestScatterPlot()


def TestLagrangeInterpolation1D():
    xData = np.linspace(0.0,2.0*np.pi,15)
    fData = np.sin(xData)
    xInterpolatingPoint = np.pi/6.0
    fInterpolatingPoint = CR.LagrangeInterpolation1D(xData,fData,xInterpolatingPoint)
    print('The exact solution of sin(x) at x = pi/6 is %.10f.' %(np.sin(xInterpolatingPoint)))
    print('The solution of sin(x) at x = pi/6 obtained using Lagrange interpolation is %.10f.' %fInterpolatingPoint)
    

do_TestLagrangeInterpolation1D = False
if do_TestLagrangeInterpolation1D:
    TestLagrangeInterpolation1D()


def TestLinearInterpolation():
    xData = np.linspace(0.0,2.0,2)
    fData = 3.0*xData
    xInterpolatingPoint = 1.0
    fInterpolatingPoint = CR.LinearInterpolation(xData,fData,xInterpolatingPoint)
    print('The exact solution of f(x) = 3x at x = 1.0 is %.10f.' %(3.0*xInterpolatingPoint))
    print('The solution of f(x) = 3x at x = 1.0 obtained using linear interpolation is %.10f.' %fInterpolatingPoint)
    

do_TestLinearInterpolation = False
if do_TestLinearInterpolation:
    TestLinearInterpolation()


def TestBilinearInterpolation():
    xData = np.linspace(0.0,2.0,2)
    yData = np.linspace(0.0,3.0,2)
    fData = np.zeros((2,2))
    for iX in range(0,2):
        for iY in range(0,2):
            fData[iX,iY] = (xData[iX] + 3.0)*(yData[iY] + 2.0)
    xInterpolatingPoint = 1.0
    yInterpolatingPoint = 1.5
    fInterpolatingPoint = CR.BilinearInterpolation(xData,yData,fData,xInterpolatingPoint,yInterpolatingPoint)
    print('The exact solution of f(x,y) = (x+3)*(y+2) at (x,y) = (1.0,1.5) is %.10f.' 
          %((xInterpolatingPoint + 3.0)*(yInterpolatingPoint + 2.0)))
    print('The solution of f(x,y) = (x+3)*(y+2) at (x,y) = (1.0,1.5) obtained using bilinear interpolation is %.10f.' 
          %fInterpolatingPoint)
    

do_TestBilinearInterpolation = False
if do_TestBilinearInterpolation:
    TestBilinearInterpolation()


def TestWriteCurve1D():
    x = np.arange(0.0,10.0,1.0)
    y = np.arange(0.0,20.0,2.0)
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    CR.WriteCurve1D(output_directory,x,y,'TestWriteCurve1D')
    

do_TestWriteCurve1D = False
if do_TestWriteCurve1D:
    TestWriteCurve1D()


def TestReadCurve1D():
    x1 = np.arange(0.0,10.0,1.0)
    y1 = np.arange(0.0,20.0,2.0)
    print('Write to file:')
    for i in range(0,np.size(x1)):
        print('%4.2f %5.2f' %(x1[i],y1[i]))
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    CR.WriteCurve1D(output_directory,x1,y1,'TestWriteCurve1D')
    x2, y2 = CR.ReadCurve1D(output_directory,'TestWriteCurve1D.curve')
    print(' ')
    print('Read from file:')
    for i in range(0,np.size(x2)):
        print('%4.2f %5.2f' %(x2[i],y2[i]))
    
    
do_TestReadCurve1D = False
if do_TestReadCurve1D:
    TestReadCurve1D()


def TestWriteTecPlot2DStructured():
    xLeft = 0.0
    xRight = 60.0
    nX = 60
    x = np.linspace(xLeft,xRight,nX+1) 
    xCenter = x[int(nX/2)]
    yBottom = 0.0
    yTop = 50.0
    nY = 50
    y = np.linspace(yBottom,yTop,nY+1)
    yCenter = y[int(nY/2)]
    phi = np.zeros((nX+1,nY+1))
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            phi[iX,iY] = (x[iX]-xCenter)**2.0 + (y[iY]-yCenter)**2.0
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    filename = 'TestWriteTecPlot2DStructured'
    CR.WriteTecPlot2DStructured(output_directory,x,y,phi,filename)


do_TestWriteTecPlot2DStructured = False
if do_TestWriteTecPlot2DStructured:   
    TestWriteTecPlot2DStructured()


def TestWriteTecPlot2DUnstructured():
    xLeft = 0.0
    xRight = 60.0
    nX = 60
    x = np.linspace(xLeft,xRight,nX+1) 
    xCenter = x[int(nX/2)]
    yBottom = 0.0
    yTop = 50.0
    nY = 50
    y = np.linspace(yBottom,yTop,nY+1)
    yCenter = y[int(nY/2)]
    xUnstructured = np.zeros((nX+1)*(nY+1))
    yUnstructured = np.zeros((nX+1)*(nY+1))
    phiUnstructured = np.zeros((nX+1)*(nY+1))
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            i = iY*(nX+1) + iX
            xUnstructured[i] = x[iX]
            yUnstructured[i] = y[iY]
            phiUnstructured[i] = (xUnstructured[i]-xCenter)**2.0 + (yUnstructured[i]-yCenter)**2.0
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    filename = 'TestWriteTecPlot2DUnstructured'
    CR.WriteTecPlot2DUnstructured(output_directory,xUnstructured,yUnstructured,phiUnstructured,filename)


do_TestWriteTecPlot2DUnstructured = False
if do_TestWriteTecPlot2DUnstructured:   
    TestWriteTecPlot2DUnstructured()


def TestReadTecPlot2DStructured():
    x = np.array([1,2,3],dtype=float) 
    y = x
    nX = len(x) - 1
    nY = len(y) - 1
    phi = np.zeros((nX+1,nY+1))
    print('Writing structured array to file:\niX iY x y phi')
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            phi[iX,iY] = (x[iX])**2.0 + (y[iY])**2.0
            print('%1d %1d %1d %1d %2.2d' %(iX,iY,x[iX],y[iY],phi[iX,iY]))
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    filename = 'TestReadTecPlot2DStructured'
    CR.WriteTecPlot2DStructured(output_directory,x,y,phi,filename)
    filename += '.tec'
    x, y, phi = CR.ReadTecPlot2DStructured(output_directory,filename,ReturnIndependentVariables=True)
    print('\nReading structured array from file:\niX iY x y phi')
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            print('%1d %1d %1d %1d %2.2d' %(iX,iY,x[iX],y[iY],phi[iX,iY]))
            
            
do_TestReadTecPlot2DStructured = False
if do_TestReadTecPlot2DStructured:   
    TestReadTecPlot2DStructured()


def TestReadTecPlot2DUnstructured():
    x = np.array([1,2,3],dtype=float) 
    y = x
    nX = len(x) - 1
    nY = len(y) - 1
    xUnstructured = np.zeros((nX+1)*(nY+1))
    yUnstructured = np.zeros((nX+1)*(nY+1))
    phiUnstructured = np.zeros((nX+1)*(nY+1))
    print('Writing unstructured array to file:\ni x y phi')
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            i = iY*(nX+1) + iX
            xUnstructured[i] = x[iX]
            yUnstructured[i] = y[iY]
            phiUnstructured[i] = (xUnstructured[i])**2.0 + (yUnstructured[i])**2.0
            print('%1d %1d %1d %2.2d' %(i,xUnstructured[i],yUnstructured[i],phiUnstructured[i]))
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    filename = 'TestReadTecPlot2DUnstructured'
    CR.WriteTecPlot2DUnstructured(output_directory,xUnstructured,yUnstructured,phiUnstructured,filename)
    filename += '.tec'
    xUnstructured, yUnstructured, phiUnstructured = (
    CR.ReadTecPlot2DUnstructured(output_directory,filename,ReturnIndependentVariables=True))
    print('\nReading structured array from file:\ni x y phi')
    for i in range(0,(nX+1)*(nY+1)):
        print('%1d %1d %1d %2.2d' %(i,xUnstructured[i],yUnstructured[i],phiUnstructured[i]))
            
            
do_TestReadTecPlot2DUnstructured = False
if do_TestReadTecPlot2DUnstructured:   
    TestReadTecPlot2DUnstructured()


def TestPythonReadFileAndMakeFilledContourAndScatterPlot2D():
    output_directory = '../../output/MPAS_Ocean_Shallow_Water_Output/'
    filenameStructured = 'TestWriteTecPlot2DStructured.tec'
    filenameUnstructured = 'TestWriteTecPlot2DUnstructured.tec'
    nContours = 300
    marker = 's'
    markersize = 12.0
    labels = ['x','y']
    labelfontsizes = [17.5,17.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    useGivenColorBarLimits = False
    ColorBarLimits = [0.0,0.0]
    nColorBarTicks = 6
    title = 'Two-Dimensional Gaussian Function'
    titlefontsize = 22.5
    SaveAsPDF = True
    Show = False
    CR.PythonReadFileAndMakeFilledContourPlot2D(output_directory,filenameStructured,nContours,labels,labelfontsizes,
                                                labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
                                                nColorBarTicks,title,titlefontsize,SaveAsPDF,Show,DataType='Structured')
    CR.PythonReadFileAndMakeFilledContourPlot2D(output_directory,filenameUnstructured,nContours,labels,labelfontsizes,
                                                labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
                                                nColorBarTicks,title,titlefontsize,SaveAsPDF,Show,
                                                DataType='Unstructured')
    CR.PythonReadFileAndMakeScatterPlotWithColorBar(output_directory,filenameUnstructured,marker,markersize,labels,
                                                    labelfontsizes,labelpads,tickfontsizes,useGivenColorBarLimits,
                                                    ColorBarLimits,nColorBarTicks,title,titlefontsize,SaveAsPDF,Show)
    
    
do_TestPythonReadFileAndMakeFilledContourAndScatterPlot2D = False
if do_TestPythonReadFileAndMakeFilledContourAndScatterPlot2D:   
    TestPythonReadFileAndMakeFilledContourAndScatterPlot2D()
    
    
def Test_WriteStateVariableLimitsToFile_ReadStateVariableLimitsFromFile():
    StateVariableLimits = np.zeros(2)
    StateVariableLimits[0] = np.exp(1.0)
    StateVariableLimits[1] = np.pi
    output_directory = '../../output/DGSEM_Rotating_Shallow_Water_Output/'
    filename = 'StateVariableLimits'
    CR.WriteStateVariableLimitsToFile(output_directory,StateVariableLimits,filename)
    print('The state variable limits written to file are: [%.15f %.15f].' 
          %(StateVariableLimits[0],StateVariableLimits[1]))
    filename += '.curve'
    StateVariableLimits = CR.ReadStateVariableLimitsFromFile(output_directory,filename)
    print('The state variable limits read from file are:  [%.15f %.15f].' 
          %(StateVariableLimits[0],StateVariableLimits[1]))
    
    
do_Test_WriteStateVariableLimitsToFile_ReadStateVariableLimitsFromFile = False
if do_Test_WriteStateVariableLimitsToFile_ReadStateVariableLimitsFromFile:   
    Test_WriteStateVariableLimitsToFile_ReadStateVariableLimitsFromFile()