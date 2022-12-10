"""
Name: SpatialOperators.py
Author: Sid Bishnu
Details: This script contains functions for computing the spatial operators of DGSEM.
"""


import numpy as np
from sympy.utilities.lambdify import lambdify
from IPython.utils import io
import os
with io.capture_output() as captured: 
    import CommonRoutines as CR
    import Initialization
    import DGSEM2DClass

    
def SurfaceElevation(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    return eta


def SurfaceElevationGradient(lX,lY,x,y):
    eta0 = 0.1
    eta_x = eta0*(2.0*np.pi/lX)*np.cos(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_y = eta0*(2.0*np.pi/lY)*np.sin(2.0*np.pi*x/lX)*np.cos(2.0*np.pi*y/lY)
    return eta_x, eta_y


def SurfaceElevationLaplacian(lX,lY,x,y):
    eta0 = 0.1
    eta = eta0*np.sin(2.0*np.pi*x/lX)*np.sin(2.0*np.pi*y/lY)
    eta_xx = -(2.0*np.pi/lX)**2.0*eta
    eta_yy = -(2.0*np.pi/lY)**2.0*eta
    Laplacian = eta_xx + eta_yy
    return Laplacian


def WriteStateDGSEM2D(myDGSEM2D,OutputDirectory,filename):
    nElements = myDGSEM2D.myDGSEM2DParameters.nElements
    nXi = myDGSEM2D.myDGSEM2DParameters.nXi
    nEta = myDGSEM2D.myDGSEM2DParameters.nEta
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    os.chdir(path)
    filename += '.tec'
    outputfile = open(filename,'w')
    outputfile.write('VARIABLES = "X", "Y", "Jacobian", "ExactZonalGradient", "ExactMeridionalGradient", '
                     + '"ExactDivergence", "NumericalZonalGradient", "NumericalMeridionalGradient", '
                     + '"NumericalDivergence", "ZonalGradientError", "MeridionalGradientError", "DivergenceError"\n')
    for iElement in range(0,nElements):
        x = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[:,:]
        y = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[:,:]
        Jacobian = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.Jacobian[:,:]
        ExactZonalGradient = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[0,:,:]
        ExactMeridionalGradient = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[1,:,:]
        ExactDivergence = myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[2,:,:]
        NumericalZonalGradient = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,:,:]
        NumericalMeridionalGradient = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,:,:]
        NumericalDivergence = myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,:,:]
        ZonalGradientError = NumericalZonalGradient - ExactZonalGradient
        MeridionalGradientError = NumericalMeridionalGradient - ExactMeridionalGradient
        DivergenceError = NumericalDivergence - ExactDivergence
        ZoneID = myDGSEM2D.myQuadMesh.myQuadElements[iElement].ElementID
        ZoneIDString = 'Element' + '%7.7d' %ZoneID
        outputfile.write('ZONE T="%s", I=%d, J=%d, F=POINT\n' %(ZoneIDString,nXi+1,nEta+1))
        for iEta in range(0,nEta+1):
            for iXi in range(0,nXi+1):
                outputfile.write('%.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n' 
                                 %(x[iXi,iEta],y[iXi,iEta],Jacobian[iXi,iEta],ExactZonalGradient[iXi,iEta],
                                   ExactMeridionalGradient[iXi,iEta],ExactDivergence[iXi,iEta],
                                   NumericalZonalGradient[iXi,iEta],NumericalMeridionalGradient[iXi,iEta],
                                   NumericalDivergence[iXi,iEta],ZonalGradientError[iXi,iEta],
                                   MeridionalGradientError[iXi,iEta],DivergenceError[iXi,iEta]))
    outputfile.close()
    os.chdir(cwd)  


def DetermineNumericalSpatialOperatorsAndError(ProblemType,nElementsX,nXi,WriteState=False):
    PrintPhaseSpeedOfWaveModes = False
    PrintAmplitudesOfWaveModes = False
    TimeIntegrator = 'WilliamsonLowStorageThirdOrderRungeKuttaMethod'
    LF_TR_and_LF_AM3_with_FB_Feedback_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    Generalized_FB_with_AB2_AM3_Step_Type = 'ThirdOrderAccurate_WideStabilityRange'
    Generalized_FB_with_AB3_AM4_Step_Type = 'ThirdOrderAccurate_MaximumStabilityRange'
    nElementsY = nElementsX
    nEta = nXi
    nXiPlot = 10
    nEtaPlot = nXiPlot
    CourantNumber = 0.5
    UseCourantNumberToDetermineTimeStep = True
    SpecifyRiemannSolver = True
    RiemannSolver = 'BassiRebay'
    myDGSEM2D = DGSEM2DClass.DGSEM2D(ProblemType,PrintPhaseSpeedOfWaveModes,PrintAmplitudesOfWaveModes,TimeIntegrator,
                                     LF_TR_and_LF_AM3_with_FB_Feedback_Type,Generalized_FB_with_AB2_AM3_Step_Type,
                                     Generalized_FB_with_AB3_AM4_Step_Type,nElementsX,nElementsY,nXi,nEta,nXiPlot,
                                     nEtaPlot,CourantNumber,UseCourantNumberToDetermineTimeStep,
                                     SpecifyRiemannSolver=SpecifyRiemannSolver,RiemannSolver=RiemannSolver)
    lX = myDGSEM2D.myNameList.lX
    lY = myDGSEM2D.myNameList.lY
    nElements = myDGSEM2D.myQuadMesh.nElements
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                x = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.x[iXi,iEta]
                y = myDGSEM2D.myQuadMesh.myQuadElements[iElement].myMappedGeometry2D.y[iXi,iEta]
                ExactZonalGradient, ExactMeridionalGradient = SurfaceElevationGradient(lX,lY,x,y)
                ExactDivergence = SurfaceElevationLaplacian(lX,lY,x,y)
                myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[0,iXi,iEta] = ExactZonalGradient
                myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[1,iXi,iEta] = ExactMeridionalGradient
                myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[2,iXi,iEta] = ExactDivergence
                myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,iXi,iEta] = ExactZonalGradient
                myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,iXi,iEta] = ExactMeridionalGradient
                myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,iXi,iEta] = SurfaceElevation(lX,lY,x,y)
    DGSEM2DClass.GlobalTimeDerivative(myDGSEM2D,myDGSEM2D.time)
    for iElement in range(0,nElements):
        for iXi in range(0,nXi+1):
            for iEta in range(0,nEta+1):
                NumericalZonalGradient = -myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[0,iXi,iEta]
                NumericalMeridionalGradient = -myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[1,iXi,iEta]
                NumericalDivergence = -myDGSEM2D.myDGSolution2D[iElement].TendencyAtInteriorNodes[2,iXi,iEta]
                myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[0,iXi,iEta] = NumericalZonalGradient
                myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[1,iXi,iEta] = NumericalMeridionalGradient
                myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[2,iXi,iEta] = NumericalDivergence
                myDGSEM2D.myDGSolution2D[iElement].ErrorAtInteriorNodes[:,iXi,iEta] = (
                (myDGSEM2D.myDGSolution2D[iElement].SolutionAtInteriorNodes[:,iXi,iEta] 
                 - myDGSEM2D.myDGSolution2D[iElement].ExactSolutionAtInteriorNodes[:,iXi,iEta]))
    L2ErrorNorm = DGSEM2DClass.ComputeErrorNorm(myDGSEM2D)
    OutputDirectory = myDGSEM2D.OutputDirectory + '/../Convergence_of_Spatial_Operators'
    if WriteState:
        FileName = 'ConvergenceOfSpatialOperators_%2.2dx%2.2d_Mesh_PolynomialOrder_%d' %(nElementsX,nElementsY,nXi+1)
        WriteStateDGSEM2D(myDGSEM2D,OutputDirectory,FileName)
    return OutputDirectory, L2ErrorNorm

        
def WriteL2ErrorNorm(OutputDirectory,nIntervals,Intervals,L2ErrorNorm,FileName):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    nCases = len(nIntervals)
    FileName += '.curve'
    OutputFile = open(FileName,'w')
    OutputFile.write('#phi\n')
    for iCase in range(0,nCases):
        OutputFile.write('%.15g %.15g %.15g %.15g %.15g\n' 
                         %(nIntervals[iCase],Intervals[iCase],L2ErrorNorm[0,iCase],L2ErrorNorm[1,iCase],
                           L2ErrorNorm[2,iCase]))
    OutputFile.close()
    os.chdir(cwd)
    
    
def ReadL2ErrorNorm(OutputDirectory,FileName,ReadDivergenceErrorNorm=False):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    data = []
    count = 0
    with open(FileName,'r') as InputFile:
        for line in InputFile:
            if count != 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    nCases = data.shape[0]
    nIntervals = np.zeros(nCases)
    Intervals = np.zeros(nCases)
    if ReadDivergenceErrorNorm:
        L2ErrorNorm = np.zeros((3,nCases))
    else:
        L2ErrorNorm = np.zeros((2,nCases))
    for iCase in range(0,nCases):
        nIntervals[iCase] = data[iCase,0]
        Intervals[iCase] = data[iCase,1]
        if ReadDivergenceErrorNorm:
            L2ErrorNorm[:,iCase] = data[iCase,2:5]
        else:
            L2ErrorNorm[:,iCase] = data[iCase,2:4]
    os.chdir(cwd)
    return nIntervals, Intervals, L2ErrorNorm
        
             
def ConvergenceStudyOfSpatialOperators(nXi,WriteState=False):
    ProblemType = 'Convergence_of_Spatial_Operators'
    nCases = 5
    nElementsX = np.zeros(nCases,dtype=int)
    nElementsX_Minimum = 4
    for iCase in range(0,nCases):
        if iCase == 0:
            nElementsX[iCase] = nElementsX_Minimum
        else:
            nElementsX[iCase] = nElementsX[iCase-1]*2
    lX, lY = Initialization.SpecifyDomainExtents(ProblemType,ProblemType_NoExactSolution=False,
                                                 ProblemType_EquatorialWave=False)
    dx = lX/nElementsX
    L2ErrorNorm = np.zeros((3,nCases))
    for iCase in range(0,nCases):
        OutputDirectory, L2ErrorNorm[:,iCase] = (
        DetermineNumericalSpatialOperatorsAndError(ProblemType,nElementsX[iCase],nXi,WriteState))
    FileName = 'ConvergencePlot_SpatialOperators_PolynomialOrder_%d_L2ErrorNorm' %(nXi+1)
    nIntervals = nElementsX
    Intervals = dx
    WriteL2ErrorNorm(OutputDirectory,nIntervals,Intervals,L2ErrorNorm,FileName)
    
    
def PlotConvergenceDataOfSpatialOperators(nXi,PlotAgainstNumberOfCellsInZonalDirection=True,UseBestFitLine=False,
                                          set_xticks_manually=False,ReadFromSELFOutputData=False,
                                          ReadDivergenceErrorNorm=False):
    OutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output/Convergence_of_Spatial_Operators'
    if ReadFromSELFOutputData:
        OutputDirectory += '_SELFOutputData'
    FileName = 'ConvergencePlot_SpatialOperators_PolynomialOrder_%d_L2ErrorNorm' %(nXi+1)
    nIntervals, Intervals, L2ErrorNorm = ReadL2ErrorNorm(OutputDirectory,FileName+'.curve',ReadDivergenceErrorNorm)
    linewidth = 2.0
    linewidths = [2.0,2.0]
    linestyle = '-'
    linestyles  = [' ','-']
    color = 'k'
    colors = ['k','k']
    marker = True
    markers = [True,False]
    markertype = 's'
    markertypes = ['s','s']
    markersize = 10.0
    markersizes = [10.0,10.0]
    if PlotAgainstNumberOfCellsInZonalDirection:
        xLabel = 'Number of cells in zonal direction'
    else:
        xLabel = 'Cell width'    
    yLabels = ['L$^2$ error norm of\nzonal gradient operator',
               'L$^2$ error norm of\nmeridional gradient operator',
               'L$^2$ error norm of\ndivergence operator']
    labelfontsizes = [22.5,22.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    legendfontsize = 22.5
    if PlotAgainstNumberOfCellsInZonalDirection:
        legendposition = 'upper right'
    else:
        legendposition = 'upper left'
        set_xticks_manually = False
    Titles = ['Convergence of Zonal Gradient Operator\nfor Polynomial Order %d' %(nXi+1),
              'Convergence of Meridional Gradient Operator\nfor Polynomial Order %d' %(nXi+1),
              'Convergence of Divergence Operator\nfor Polynomial Order %d' %(nXi+1)]
    titlefontsize = 27.5 
    FileNames = ['ConvergencePlot_ZonalGradientOperator_PolynomialOrder_%d_L2ErrorNorm' %(nXi+1),
                 'ConvergencePlot_MeridionalGradientOperator_PolynomialOrder_%d_L2ErrorNorm' %(nXi+1),
                 'ConvergencePlot_DivergenceOperator_PolynomialOrder_%d_L2ErrorNorm' %(nXi+1)]
    if PlotAgainstNumberOfCellsInZonalDirection:
        dx = nIntervals
    else:
        dx = Intervals
    if ReadDivergenceErrorNorm:
        nSpatialOperators = 3
    else:
        nSpatialOperators = 2
    for iSpatialOperator in range(0,nSpatialOperators):
        L2ErrorNormPerOperator = L2ErrorNorm[iSpatialOperator,:]
        yLabel = yLabels[iSpatialOperator]
        labels = [xLabel,yLabel]
        Title = Titles[iSpatialOperator]
        FileName = FileNames[iSpatialOperator]
        if set_xticks_manually:
            xticks_set_manually = dx 
        else:
            xticks_set_manually = []
        A = np.vstack([np.log10(dx),np.ones(len(dx))]).T
        m, c = np.linalg.lstsq(A,np.log10(L2ErrorNormPerOperator),rcond=None)[0]
        if UseBestFitLine:
            L2ErrorNorm_BestFitLine = m*(np.log10(dx)) + c
            L2ErrorNorm_BestFitLine = 10.0**L2ErrorNorm_BestFitLine
            FileName += '_BestFitLine'
            legends = ['L$^2$ error norm','Best fit line:\nslope is %.2f' %m]
            CR.PythonConvergencePlot1DSaveAsPDF(OutputDirectory,'log-log',dx,L2ErrorNormPerOperator,
                                                L2ErrorNorm_BestFitLine,linewidths,linestyles,colors,markers,
                                                markertypes,markersizes,labels,labelfontsizes,labelpads,tickfontsizes,
                                                legends,legendfontsize,legendposition,Title,titlefontsize,True,FileName,
                                                False,drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=True,
                                                set_xticks_manually=set_xticks_manually,
                                                xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        else:
            Title += '\nSlope is %.2f' %m
            CR.PythonPlot1DSaveAsPDF(OutputDirectory,'log-log',dx,L2ErrorNormPerOperator,linewidth,linestyle,color,
                                     marker,markertype,markersize,labels,labelfontsizes,labelpads,tickfontsizes,Title,
                                     titlefontsize,True,FileName,False,fig_size=[9.25,9.25],
                                     useDefaultMethodToSpecifyTickFontSize=False,drawMajorGrid=True,drawMinorGrid=True,
                                     set_xticks_manually=set_xticks_manually,xticks_set_manually=xticks_set_manually,
                                     FileFormat='pdf')
            

def SpecifyAsymptoticPointsForSlopeComputation(PolynomialOrder,SpatialOperator,ReadFromSELFOutputData,nCases):
    iPointLowerLimit = 0
    iPointUpperLimit = nCases - 1
    return iPointLowerLimit, iPointUpperLimit


def SetOfLegends(nXiMinimum,slopes):
    legends = ['Polynomial Order %d' %(nXiMinimum+1),'Polynomial Order %d' %(nXiMinimum+2),
               'Polynomial Order %d' %(nXiMinimum+3)]            
    for i_legend in range(0,len(legends)):
        legends[i_legend] += ':\nSlope = %.2f' %slopes[i_legend]
    legendfontsize = 14.0
    legendpads = [0.0,0.0]
    return legends, legendfontsize, legendpads


def PlotConvergenceDataOfAllSpatialOperators(nXiMinimum,PlotAgainstNumberOfCellsInZonalDirection=True,
                                             UseBestFitLine=False,set_xticks_manually=False,
                                             ReadFromSELFOutputData=False,ReadDivergenceErrorNorm=False):
    OutputDirectory = '../../output/DGSEM_Rotating_Shallow_Water_Output/Convergence_of_Spatial_Operators'
    if ReadFromSELFOutputData:
        OutputDirectory += '_SELFOutputData'
    FileName = 'ConvergencePlot_SpatialOperators_PolynomialOrder_%d_L2ErrorNorm' %(nXiMinimum+1)
    nIntervals, Intervals, L2ErrorNorm = ReadL2ErrorNorm(OutputDirectory,FileName+'.curve',ReadDivergenceErrorNorm)
    nCases = len(nIntervals)
    if PlotAgainstNumberOfCellsInZonalDirection:
        dx = nIntervals
    else:
        dx = Intervals
    if ReadDivergenceErrorNorm:
        nSpatialOperators = 3
    else:
        nSpatialOperators = 2
    n_nXi = 3
    L2ErrorNormPerOperator = np.zeros((n_nXi,nSpatialOperators,nCases))
    L2ErrorNormPerOperatorWithBestFitLine = np.zeros((n_nXi,nSpatialOperators,2,nCases))
    mPerOperator = np.zeros((n_nXi,nSpatialOperators))
    SpatialOperators = ['ZonalGradientOperator','MeridionalGradientOperator','DivergenceOperator']
    for i_nXi in range(0,n_nXi):
        nXi = nXiMinimum + i_nXi
        FileName = 'ConvergencePlot_SpatialOperators_PolynomialOrder_%d_L2ErrorNorm' %(nXi+1)
        nIntervals, Intervals, L2ErrorNorm = ReadL2ErrorNorm(OutputDirectory,FileName+'.curve',ReadDivergenceErrorNorm)
        for iSpatialOperator in range(0,nSpatialOperators):
            SpatialOperator = SpatialOperators[iSpatialOperator]
            L2ErrorNormPerOperator[i_nXi,iSpatialOperator,:] = L2ErrorNorm[iSpatialOperator,:]
            iPointLowerLimit, iPointUpperLimit = (
            SpecifyAsymptoticPointsForSlopeComputation(nXi+1,SpatialOperator,ReadFromSELFOutputData,nCases))
            dx_SlopeComputation = dx[iPointLowerLimit:iPointUpperLimit+1]    
            if set_xticks_manually:
                xticks_set_manually = dx 
            else:
                xticks_set_manually = []
            A = np.vstack([np.log10(dx_SlopeComputation),np.ones(len(dx_SlopeComputation))]).T
            L2ErrorNormPerOperator_SlopeComputation = (
            L2ErrorNormPerOperator[i_nXi,iSpatialOperator,iPointLowerLimit:iPointUpperLimit+1])
            mPerOperator[i_nXi,iSpatialOperator], c = (
            np.linalg.lstsq(A,np.log10(L2ErrorNormPerOperator_SlopeComputation),rcond=None)[0])
            if UseBestFitLine:
                y = mPerOperator[i_nXi,iSpatialOperator]*(np.log10(dx)) + c
                y = 10.0**y
                L2ErrorNormPerOperatorWithBestFitLine[i_nXi,iSpatialOperator,0,:] = (
                L2ErrorNormPerOperator[i_nXi,iSpatialOperator,:])
                L2ErrorNormPerOperatorWithBestFitLine[i_nXi,iSpatialOperator,1,:] = y
    linewidths = 2.0*np.ones(n_nXi)
    linestyles  = ['-','-','-']
    colors = ['blue','green','red']
    markers = np.ones(n_nXi,dtype=bool)
    markertypes = ['o','H','s']
    markersizes = np.array([12.5,12.5,10.0])
    if PlotAgainstNumberOfCellsInZonalDirection:
        xLabel = 'Number of cells in zonal direction'
    else:
        xLabel = 'Cell width'    
    yLabels = ['L$^2$ error norm of\nzonal gradient operator','L$^2$ error norm of\nmeridional gradient operator',
               'L$^2$ error norm of\ndivergence operator']    
    labelfontsizes = [22.5,22.5]
    labelpads = [10.0,10.0]
    tickfontsizes = [15.0,15.0]
    legendfontsize = 22.5
    if PlotAgainstNumberOfCellsInZonalDirection:
        legendposition = 'upper right'
    else:
        legendposition = 'upper left'
        set_xticks_manually = False
    Titles = ['Convergence of Zonal Gradient Operator','Convergence of Meridional Gradient Operator',
              'Convergence of Divergence Operator']
    titlefontsize = 27.5 
    SaveAsPDF = True
    FileNames = ['ConvergencePlot_ZonalGradientOperator_L2ErrorNorm',
                 'ConvergencePlot_MeridionalGradientOperator_L2ErrorNorm',
                 'ConvergencePlot_DivergenceOperator_L2ErrorNorm']
    Show = False
    shadow = False
    framealpha = 0.75
    if PlotAgainstNumberOfCellsInZonalDirection:
        legendposition = 'lower left'
    else:
        legendposition = 'lower right'
        set_xticks_manually = False
    for iSpatialOperator in range(0,nSpatialOperators):
        yLabel = yLabels[iSpatialOperator]
        labels = [xLabel,yLabel]
        Title = Titles[iSpatialOperator]
        FileName = FileNames[iSpatialOperator]
        if set_xticks_manually:
            xticks_set_manually = dx 
        else:
            xticks_set_manually = []
        legends, legendfontsize, legendpads = SetOfLegends(nXiMinimum,mPerOperator[:,iSpatialOperator])
        if UseBestFitLine:
            FileName += '_BestFitLine'
            CR.PythonConvergencePlots1DSaveAsPDF(
            OutputDirectory,'log-log',dx,L2ErrorNormPerOperatorWithBestFitLine[:,iSpatialOperator,:,:],linewidths,
            linestyles,colors,markertypes,markersizes,labels,labelfontsizes,labelpads,tickfontsizes,legends,
            legendfontsize,legendposition,Title,titlefontsize,SaveAsPDF,FileName,Show,fig_size=[9.25,9.25],
            useDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=True,drawMinorGrid=True,legendWithinBox=False,
            legendpads=legendpads,shadow=shadow,framealpha=framealpha,titlepad=1.035,
            set_xticks_manually=set_xticks_manually,xticks_set_manually=xticks_set_manually,FileFormat='pdf')
        else:
            CR.PythonPlots1DSaveAsPDF(
            OutputDirectory,'log-log',dx,L2ErrorNormPerOperator[:,iSpatialOperator,:],linewidths,linestyles,colors,
            markers,markertypes,markersizes,labels,labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
            legendposition,Title,titlefontsize,SaveAsPDF,FileName,Show,fig_size=[9.25,9.25],
            useDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=True,drawMinorGrid=True,
            setXAxisLimits=[False,False],xAxisLimits=[0.0,0.0],setYAxisLimits=[False,False],yAxisLimits=[0.0,0.0],
            legendWithinBox=True,legendpads=legendpads,shadow=shadow,framealpha=framealpha,titlepad=1.035,
            set_xticks_manually=set_xticks_manually,xticks_set_manually=xticks_set_manually,FileFormat='pdf')