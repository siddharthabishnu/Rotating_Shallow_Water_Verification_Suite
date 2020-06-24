
# coding: utf-8

# Name: Common_Routines.ipynb <br/>
# Author: Siddhartha Bishnu <br/>
# Details: This code contains customized routines for writing output to text files, plotting figures etc.

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:

def CurrentWorkingDirectory():
    cwd = os.getcwd()
    PrintCWDToFile = False
    if PrintCWDToFile:
        outputfile = open('CurrentWorkingDirectory.txt','w')
        outputfile.write(cwd)
        outputfile.close()
    return cwd
        
cwd = CurrentWorkingDirectory()


# In[3]:

def InitializeCharacterArrays():
    nStrings = 5
    strings = ['John Doe' for x in range(nStrings)]
    for iString in range(0,nStrings):
        if np.mod(float(iString),2.0) == 0.0:
            strings[iString] = 'Jane Doe'
    print(strings)
        
InitializeCharacterArrays()


# In[4]:

def PythonPlot1DSaveAsPNG(output_directory,plot_type,x,y,linewidth,linestyle,color,marker,markersize,labels,
                          labelfontsizes,labelpads,tickfontsizes,title,titlefontsize,SaveAsPNG,FigureTitle,Show,
                          drawGrid=False):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)    
    fig = plt.figure(figsize=(9.25,9.25)) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    if marker:
        if plot_type == 'regular':
            plt.plot(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=marker,markersize=markersize)
        elif plot_type == 'semi-log_x':
            plt.semilogx(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=marker,
                         markersize=markersize)
        elif plot_type == 'semi-log_y':
            plt.semilogy(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=marker,
                         markersize=markersize)
        elif plot_type == 'log-log':
            plt.loglog(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=marker,
                       markersize=markersize)
    else:
        if plot_type == 'regular':
            plt.plot(x,y,linewidth=linewidth,linestyle=linestyle,color=color)
        elif plot_type == 'semi-log_x':
            plt.semilogx(x,y,linewidth=linewidth,linestyle=linestyle,color=color)
        elif plot_type == 'semi-log_y':
            plt.semilogy(x,y,linewidth=linewidth,linestyle=linestyle,color=color)
        elif plot_type == 'log-log':
            plt.loglog(x,y,linewidth=linewidth,linestyle=linestyle,color=color)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    plt.xticks(fontsize=tickfontsizes[0])
    plt.yticks(fontsize=tickfontsizes[1])
    ax.set_title(title,fontsize=titlefontsize,y=1.035)
    if drawGrid:
        plt.grid()
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[5]:

def TestPythonPlot1DSaveAsPNG():
    x = np.arange(0.0,10.0,1) # Syntax is x = np.arange(First Point, Last Point, Interval)
    y = np.arange(0.0,20.0,2)
    PythonPlot1DSaveAsPNG('MPAS_O_Shallow_Water_Output','regular',x,y,2.0,'-','k','s',7.5,['x','y'],[17.5,17.5],
                          [10.0,10.0],[15.0,15.0],'Python Plot 1D',20.0,True,'PythonPlot1D',True,drawGrid=True)

do_TestPythonPlot1DSaveAsPNG = False
if do_TestPythonPlot1DSaveAsPNG:
    TestPythonPlot1DSaveAsPNG()


# In[6]:

def PythonPlots1DSaveAsPNG(output_directory,x,y1,y2,line_width,y1stem,y2stem,xlabel,xlabelpad,ylabel,ylabelpad,
                           y1legend,y2legend,legend_position,title,marker,marker_size,SaveAsPNG,FigureTitle,Show):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)   
    fig = plt.figure(figsize=(9.25,9.25)) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    if marker:
        ax.plot(x,y1,linewidth=line_width,linestyle='-',color='r',marker='s',markersize=marker_size,label=y1legend)
        ax.plot(x,y2,linewidth=line_width,linestyle='-',color='b',marker='s',markersize=marker_size,label=y2legend)
    else:
        ax.plot(x,y1,linewidth=line_width,linestyle='-',color='r',label=y1legend)
        ax.plot(x,y2,linewidth=line_width,linestyle='-',color='b',label=y2legend)
    if y1stem:
        plt.stem(x,y1,'r',markerfmt='rs',linefmt='r-.')
    if y2stem:
        plt.stem(x,y2,'b',markerfmt='bs',linefmt='b--')
    plt.xlabel(xlabel,fontsize=17.5,labelpad=xlabelpad)
    plt.ylabel(ylabel,fontsize=17.5,labelpad=ylabelpad)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(fontsize=17.5,loc=legend_position,bbox_to_anchor=(1,0.5),shadow=True) 
    ax.set_title(title,fontsize=20,y=1.035)
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)
    
def TestPythonPlots1DSaveAsPNG():
    x = np.arange(0.0,10.0,1) # Syntax is x = np.arange(First Point, Last Point, Interval)
    y1 = np.arange(0.0,20.0,2)
    y2 = np.arange(0.0,40.0,4)
    PythonPlots1DSaveAsPNG('MPAS_O_Shallow_Water_Output',x,y1,y2,2.0,True,True,'x',10,'y',10,'y1','y2',
                           'center left','Python Plots 1D',True,7.5,True,'PythonPlots1D',True)

do_TestPythonPlots1DSaveAsPNG = False
if do_TestPythonPlots1DSaveAsPNG:
    TestPythonPlots1DSaveAsPNG()


# In[7]:

def PythonPlots1DWithLimitsSaveAsPNG(output_directory,x,y1,y2,xLimits,yLimits,line_width,y1stem,y2stem,xlabel,
                                     xlabelpad,ylabel,ylabelpad,y1legend,y2legend,legend_position,title,marker,
                                     marker_size,SaveAsPNG,FigureTitle,Show):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)   
    fig = plt.figure(figsize=(9.25,9.25)) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    if marker:
        ax.plot(x,y1,linewidth=line_width,linestyle='-',color='r',marker='s',markersize=marker_size,label=y1legend)
        ax.plot(x,y2,linewidth=line_width,linestyle='-',color='b',marker='s',markersize=marker_size,label=y2legend)
    else:
        ax.plot(x,y1,linewidth=line_width,linestyle='-.',color='r',label=y1legend)
        ax.plot(x,y2,linewidth=line_width,linestyle='--',color='b',label=y2legend)
    if y1stem:
        plt.stem(x,y1,'r',markerfmt='rs',linefmt='r-.')
    if y2stem:
        plt.stem(x,y2,'b',markerfmt='bs',linefmt='b--')
    SetXLimits = True
    if SetXLimits:
        plt.xlim(xLimits)
    SetYLimits = True
    if SetYLimits:
        plt.ylim(yLimits)
    plt.xlabel(xlabel,fontsize=17.5,labelpad=xlabelpad)
    plt.ylabel(ylabel,fontsize=17.5,labelpad=ylabelpad)
    plt.xticks(np.arange(xLimits[0], xLimits[1]+1.0, xLimits[1]/4.0),fontsize=15)
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x/1000.0), ',')))
    plt.yticks(fontsize=15)
    ax.legend(fontsize=17.5,loc=legend_position,bbox_to_anchor=(1,0.5),shadow=True) 
    ax.set_title(title,fontsize=20,y=1.035)
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[8]:

def PythonConvergencePlot1DSaveAsPNG(output_directory,plot_type,x,y1,y2,linewidths,linestyles,colors,useMarkers,
                                     markers,markersizes,labels,labelfontsizes,labelpads,tickfontsizes,legends,
                                     legendfontsize,legendposition,title,titlefontsize,SaveAsPNG,FigureTitle,Show,
                                     drawGrid=False,legendWithinBox=False):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)   
    fig = plt.figure(figsize=(9.25,9.25)) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    if useMarkers[0]:
        if plot_type == 'regular':
            ax.plot(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],marker=markers[0],
                    markersize=markersizes[0],label=legends[0])
        elif plot_type == 'semi-log_x':
            ax.semilogx(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],marker=markers[0],
                        markersize=markersizes[0],label=legends[0])
        elif plot_type == 'semi-log_y':
            ax.semilogy(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],marker=markers[0],
                        markersize=markersizes[0],label=legends[0])
        elif plot_type == 'log-log':
            ax.loglog(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],marker=markers[0],
                      markersize=markersizes[0],label=legends[0])      
    else:
        if plot_type == 'regular':
            ax.plot(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],label=legends[0])
        elif plot_type == 'semi-log_x':
            ax.semilogx(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],label=legends[0])
        elif plot_type == 'semi-log_y':
            ax.semilogy(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],label=legends[0])
        elif plot_type == 'log-log':
            ax.loglog(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],label=legends[0])
    if useMarkers[1]:
        if plot_type == 'regular':
            ax.plot(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],marker=markers[1],
                    markersize=markersizes[1],label=legends[1])
        elif plot_type == 'semi-log_x':
            ax.semilogx(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],marker=markers[1],
                        markersize=markersizes[1],label=legends[1])
        elif plot_type == 'semi-log_y':
            ax.semilogy(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],marker=markers[1],
                        markersize=markersizes[1],label=legends[1])
        elif plot_type == 'log-log':
            ax.loglog(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],marker=markers[1],
                      markersize=markersizes[1],label=legends[1])      
    else:
        if plot_type == 'regular':
            ax.plot(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],label=legends[1])
        elif plot_type == 'semi-log_x':
            ax.semilogx(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],label=legends[1])
        elif plot_type == 'semi-log_y':
            ax.semilogy(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],label=legends[1])
        elif plot_type == 'log-log':
            ax.loglog(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],label=legends[1])  
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    plt.xticks(fontsize=tickfontsizes[0])
    plt.yticks(fontsize=tickfontsizes[1])
    if legendWithinBox:
        ax.legend(fontsize=legendfontsize,loc=legendposition,shadow=True) 
    else:
        ax.legend(fontsize=legendfontsize,loc=legendposition,bbox_to_anchor=(1,0.5),shadow=True) 
    ax.set_title(title,fontsize=titlefontsize,y=1.035)
    if drawGrid:
        plt.grid()
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png',bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


# In[9]:

def TestPythonConvergencePlot1DSaveAsPNG():
    x = np.arange(0.0,10.0,1) # Syntax is x = np.arange(First Point, Last Point, Interval)
    y1 = np.arange(0.0,20.0,2)
    y2 = np.arange(0.0,20.0,2)
    PythonConvergencePlot1DSaveAsPNG('MPAS_O_Shallow_Water_Output','regular',x,y1,y2,[2.0,2.0],['-',' '],['k','k'],
                                     [False,True],['s','s'],[10.0,10.0],['x','y'],[17.5,17.5],[10.0,10.0],
                                     [15.0,15.0],['y1','y2'],17.5,'upper left','Convergence Plot 1D',20.0,True,
                                     'ConvergencePlot1D',True,drawGrid=True,legendWithinBox=True)

do_TestPythonConvergencePlot1DSaveAsPNG = False
if do_TestPythonConvergencePlot1DSaveAsPNG:
    TestPythonConvergencePlot1DSaveAsPNG()


# In[10]:

def LagrangeInterpolation1D(xData,fData,x):
    N = len(xData) - 1
    LagInterp1D = 0.0
    for i in range(0,N+1):
        LagrangeProduct = fData[i]
        for j in range(0,N+1):
            if i != j:
                LagrangeProduct = LagrangeProduct*(x - xData[j])/(xData[i] - xData[j])
        LagInterp1D = LagInterp1D + LagrangeProduct
    return LagInterp1D

def TestLagrangeInterpolation1D():
    xData = np.linspace(0.0,2.0*np.pi,15)
    fData = np.sin(xData)
    xInterp = np.pi/6.0
    fInterp = LagrangeInterpolation1D(xData,fData,xInterp)
    print('The exact solution of sin(x) at x = pi/6 is %.10f.' %(np.sin(xInterp)))
    print('The solution of sin(x) at x = pi/6 obtained using Lagrange interpolation is %.10f.' %(fInterp))
    
do_TestLagrangeInterpolation1D = False
if do_TestLagrangeInterpolation1D:
    TestLagrangeInterpolation1D()


# In[11]:

def WriteTecPlot2DStructured(output_directory,x,y,phi,filename):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    nX = len(x) - 1
    nY = len(y) - 1
    filename = filename + '.tec'
    outputfile = open(filename,'w')
    outputfile.write('VARIABLES = "X", "Y", "PHI"\n')
    outputfile.write('ZONE T="EL00001", I=%d, J=%d, F=POINT\n' %(nX+1,nY+1))
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            outputfile.write("%.15f %.15f %.15f\n" %(x[iX],y[iY],phi[iX,iY]))
    outputfile.close()
    os.chdir(cwd)
    
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
    WriteTecPlot2DStructured('MPAS_O_Shallow_Water_Output',x,y,phi,'TestWriteTecPlot2DStructured')
    
do_TestWriteTecPlot2DStructured = False
if do_TestWriteTecPlot2DStructured:   
    TestWriteTecPlot2DStructured()


# In[12]:

def WriteTecPlot2DUnstructured(output_directory,x,y,phi,filename):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    N = len(x)
    filename = filename + '.tec'
    outputfile = open(filename,'w')
    outputfile.write('VARIABLES = "X", "Y", "PHI"\n')
    for i in range(0,N):
        outputfile.write('ZONE T="EL%5.5d", I=1, J=1, F=BLOCK\n' %(i+1))
        outputfile.write("%.15f %.15f %.15f\n" %(x[i],y[i],phi[i]))
    outputfile.close()
    os.chdir(cwd)
    
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
    WriteTecPlot2DUnstructured('MPAS_O_Shallow_Water_Output',xUnstructured,yUnstructured,phiUnstructured,
                               'TestWriteTecPlot2DUnstructured')
    
do_TestWriteTecPlot2DUnstructured = False
if do_TestWriteTecPlot2DUnstructured:
    TestWriteTecPlot2DUnstructured()


# In[13]:

def PythonFilledStructuredContourPlot2DSaveAsPNG(output_directory,x,y,phi,nContours,useGivenColorBarLimits,
                                                 ColorBarLimits,xlabel,xlabelpad,ylabel,ylabelpad,title,SaveAsPNG,
                                                 FigureTitle,Show):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    fig = plt.figure(figsize=(10,10)) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    set_aspect_equal = False
    if set_aspect_equal:
        ax.set_aspect('equal')
    else:
        xMin = min(x[:])
        xMax = max(x[:])
        yMin = min(y[:])
        yMax = max(y[:])        
        aspect_ratio = (xMax - xMin)/(yMax - yMin)
        ax.set_aspect(aspect_ratio,adjustable='box')
    if useGivenColorBarLimits:
        cbar_min = ColorBarLimits[0]
        cbar_max = ColorBarLimits[1]
    else:
        cbar_min = np.min(phi)
        cbar_max = np.max(phi)
    n_cbar_ticks = 6
    cbarlabels = np.linspace(cbar_min,cbar_max,num=n_cbar_ticks,endpoint=True)
    FCP = plt.contourf(x,y,phi,nContours,vmin=cbar_min,vmax=cbar_max,cmap=plt.cm.jet) 
    # FCP stands for filled contour plot
    plt.title(title,fontsize=20,y=1.035)
    cbarShrinkRatio = 0.825
    cbar = plt.colorbar(shrink=cbarShrinkRatio) # draw colorbar
    cbar.set_ticks(cbarlabels)
    cbar.set_ticklabels(cbarlabels)
    cbar.ax.set_yticklabels(['{:.2g}'.format(x) for x in cbarlabels], fontsize=13.75)
    plt.xlabel(xlabel,fontsize=17.5,labelpad=xlabelpad)
    plt.ylabel(ylabel,fontsize=17.5,labelpad=ylabelpad)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)

def PythonReadFileAndFilledStructuredContourPlot2D(filename,nContours,xlabel,xlabelpad,ylabel,ylabelpad,title,
                                                   SaveAsPNG,FigureTitle,Show):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/MPAS_O_Shallow_Water_Output/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    nXPlus1TimesnYPlus1 = len(open(filename).readlines( )[2:])
    data = np.loadtxt(filename,skiprows=2)
    xAll = np.zeros(nXPlus1TimesnYPlus1)
    for iXAll in range(0,nXPlus1TimesnYPlus1):
        xAll[iXAll] = data[iXAll,0]    
        if iXAll > 0 and xAll[iXAll] < xAll[iXAll-1]:
            nXPlus1 = iXAll
            break
    nYPlus1 = int(nXPlus1TimesnYPlus1/nXPlus1)
    x = np.zeros(nXPlus1)
    y = np.zeros(nYPlus1)
    phi = np.zeros((nYPlus1,nXPlus1))
    for iX in range(0,nXPlus1):
        x[iX] = data[iX,0]
    for iY in range(0,nYPlus1):
        y[iY] = data[iY*nXPlus1,1]
    for iY in range(0,nYPlus1):
        for iX in range(0,nXPlus1):
            phi[iY,iX] = data[iY*nXPlus1+iX,2] # Study the shape of phi here.
    os.chdir(cwd)
    PythonFilledStructuredContourPlot2DSaveAsPNG('MPAS_O_Shallow_Water_Output',x,y,phi,nContours,False,[0.0,0.0],
                                                 xlabel,xlabelpad,ylabel,ylabelpad,title,SaveAsPNG,FigureTitle,
                                                 Show)

do_PythonReadFileAndFilledStructuredContourPlot2D = False
if do_PythonReadFileAndFilledStructuredContourPlot2D:
    PythonReadFileAndFilledStructuredContourPlot2D('TestWriteTecPlot2DStructured.tec',300,'x',10,'y',10,' ',True,
                                                   'TestWriteTecPlot2DStructured',True)


# In[14]:

def line_contains_text(line):
    return line[0] == 'V' or line[0] == 'Z'

def PythonFilledUnstructuredContourPlot2DSaveAsPNG(output_directory,x,y,phi,nContours,useGivenColorBarLimits,
                                                   ColorBarLimits,xlabel,xlabelpad,ylabel,ylabelpad,title,
                                                   SaveAsPNG,FigureTitle,Show,myXTicks=np.zeros(6),
                                                   myYTicks=np.zeros(6)):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    fig = plt.figure(figsize=(10,10)) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    set_aspect_equal = False
    if set_aspect_equal:
        ax.set_aspect('equal')
    else:
        xMin = min(x[:])
        xMax = max(x[:])
        yMin = min(y[:])
        yMax = max(y[:])        
        aspect_ratio = (xMax - xMin)/(yMax - yMin)
        ax.set_aspect(aspect_ratio,adjustable='box')
    if useGivenColorBarLimits:
        cbar_min = ColorBarLimits[0]
        cbar_max = ColorBarLimits[1]
    else:
        cbar_min = np.min(phi)
        cbar_max = np.max(phi)
    n_cbar_ticks = 6
    cbarlabels = np.linspace(cbar_min,cbar_max,num=n_cbar_ticks,endpoint=True)
    FCP = plt.tricontourf(x,y,phi,nContours,vmin=cbar_min,vmax=cbar_max,cmap=plt.cm.jet) 
    # FCP stands for filled contour plot
    plt.title(title,fontsize=20,y=1.035)
    cbarShrinkRatio = 0.825
    cbar = plt.colorbar(shrink=cbarShrinkRatio) # draw colorbar
    cbar.set_ticks(cbarlabels)
    cbar.set_ticklabels(cbarlabels)
    cbar.ax.set_yticklabels(['{:.2g}'.format(x) for x in cbarlabels], fontsize=13.75)
    plt.xlabel(xlabel,fontsize=17.5,labelpad=xlabelpad)
    plt.ylabel(ylabel,fontsize=17.5,labelpad=ylabelpad)
    if max(abs(myXTicks)) == 0.0:
        plt.xticks(fontsize=15)
    else:
        plt.xticks(myXTicks,fontsize=15)
    if max(abs(myYTicks)) == 0.0:
        plt.yticks(fontsize=15)
    else:
        plt.yticks(myYTicks,fontsize=15)
    if SaveAsPNG:
        plt.savefig(FigureTitle+'.png',format='png')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)
        
def PythonReadFileAndFilledUnstructuredContourPlot2D(filename,nContours,xlabel,xlabelpad,ylabel,ylabelpad,title,
                                                     SaveAsPNG,FigureTitle,Show):
    cwd = CurrentWorkingDirectory()
    path = cwd + '/MPAS_O_Shallow_Water_Output/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    data = [];
    cnt = 0;
    with open(filename, 'r') as infile:
        for line in infile:
            if cnt != 0 and cnt % 2 == 0:
                data.append(line)
            cnt += 1
    data = np.loadtxt(data)
    nX = data.shape[0]
    x = np.zeros(nX)
    y = np.zeros(nX)
    phi = np.zeros(nX)
    for iX in range(0,nX):
        x[iX] = data[iX,0]
    for iX in range(0,nX):
        y[iX] = data[iX,1]
    for iX in range(0,nX):
        phi[iX] = data[iX,2]
    os.chdir(cwd)
    PythonFilledUnstructuredContourPlot2DSaveAsPNG('MPAS_O_Shallow_Water_Output',x,y,phi,nContours,False,[0.0,0.0],
                                                   xlabel,xlabelpad,ylabel,ylabelpad,title,SaveAsPNG,FigureTitle,
                                                   Show)
    
do_PythonReadFileAndFilledUnstructuredContourPlot2D = False
if do_PythonReadFileAndFilledUnstructuredContourPlot2D:
    PythonReadFileAndFilledUnstructuredContourPlot2D('TestWriteTecPlot2DUnstructured.tec',300,'x',10,'y',10,' ',
                                                     True,'TestWriteTecPlot2DUnstructured',True)