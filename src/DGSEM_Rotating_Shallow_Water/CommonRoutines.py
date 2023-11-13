"""
Name: CommonRoutines.py
Author: Sid Bishnu
Details: This script contains customized functions for writing output to text files, plotting figures etc.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import os


def AlmostEqual(x,y):
    epsilon = np.finfo(float).eps
    if x == 0.0 or y == 0.0:
        if abs(x-y) <= 2.0*epsilon:
            almost_equal = True
        else:
            almost_equal = False
    else:
        if abs(x-y) <= epsilon*abs(x) and abs(x-y) <= epsilon*abs(y):
            almost_equal = True
        else:
            almost_equal = False
    return almost_equal


def RoundArray(x):
    nX = len(x)
    xRounded = np.zeros(nX)
    for iX in range(0,nX):
        xRounded[iX] = round(x[iX])
    return xRounded


def PythonPlot1DSaveAsPDF(output_directory,plot_type,x,y,linewidth,linestyle,color,marker,markertype,markersize,labels,
                          labelfontsizes,labelpads,tickfontsizes,title,titlefontsize,SaveAsPDF,FileName,Show,
                          fig_size=[9.25,9.25],useDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=True,
                          drawMinorGrid=True,setXAxisLimits=[False,False],xAxisLimits=[0.0,0.0],
                          setYAxisLimits=[False,False],yAxisLimits=[0.0,0.0],plot_label='Python Plot 1D',
                          titlepad=1.035,set_xticks_manually=False,xticks_set_manually=[],FileFormat='pdf'):
    cwd = os.getcwd()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)    
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object.
    ax = fig.add_subplot(111) # Create an axes object in the figure.
    if not(marker):
        markertype = None
    if plot_type == 'regular':
        plt.plot(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=markertype,markersize=markersize,
                 label=plot_label)
    elif plot_type == 'semi-log_x':
        plt.semilogx(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=markertype,markersize=markersize,
                     label=plot_label)
    elif plot_type == 'semi-log_y':
        plt.semilogy(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=markertype,markersize=markersize,
                     label=plot_label)
    elif plot_type == 'log-log':
        plt.loglog(x,y,linewidth=linewidth,linestyle=linestyle,color=color,marker=markertype,markersize=markersize,
                   label=plot_label)
    if plot_type == 'regular':
        if setXAxisLimits[0]:
            ax.set_xlim(bottom=xAxisLimits[0])
        if setXAxisLimits[1]:
            ax.set_xlim(top=xAxisLimits[1])
        if setYAxisLimits[0]:
            ax.set_ylim(bottom=yAxisLimits[0])
        if setYAxisLimits[1]:
            ax.set_ylim(top=yAxisLimits[1])            
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    if useDefaultMethodToSpecifyTickFontSize:
        plt.xticks(fontsize=tickfontsizes[0])
        plt.yticks(fontsize=tickfontsizes[1])
    else:
        ax.tick_params(axis='x',labelsize=tickfontsizes[0])
        ax.tick_params(axis='y',labelsize=tickfontsizes[1])
    if set_xticks_manually:
        ax.set_xticks(xticks_set_manually,minor=False)
    ax.set_title(title,fontsize=titlefontsize,fontweight='bold',y=titlepad)
    if drawMajorGrid and not(drawMinorGrid):
        plt.grid(which='major')
    elif not(drawMajorGrid) and drawMinorGrid:
        plt.grid(which='minor')       
    elif drawMajorGrid and drawMinorGrid:
        plt.grid(which='both')
    if SaveAsPDF:
        plt.savefig(FileName+'.'+FileFormat,format=FileFormat,bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


def PythonPlots1DSaveAsPDF(output_directory,plot_type,x,yAll,linewidths,linestyles,colors,markers,markertypes,
                           markersizes,labels,labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
                           legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,fig_size=[9.25,9.25],
                           useDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=True,drawMinorGrid=True,
                           setXAxisLimits=[False,False],xAxisLimits=[0.0,0.0],setYAxisLimits=[False,False],
                           yAxisLimits=[0.0,0.0],legendWithinBox=False,legendpads=[1.0,0.5],shadow=True,
                           framealpha=1.0,titlepad=1.035,set_xticks_manually=False,xticks_set_manually=[],
                           FileFormat='pdf'):
    cwd = os.getcwd()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)    
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object.
    ax = fig.add_subplot(111) # Create an axes object in the figure.
    nPlots = yAll.shape[0]
    for iPlot in range(0,nPlots):
        marker = markers[iPlot]
        if not(marker):
            markertypes[iPlot] = None
        if plot_type == 'regular':
            plt.plot(x,yAll[iPlot,:],linewidth=linewidths[iPlot],linestyle=linestyles[iPlot],color=colors[iPlot],
                     marker=markertypes[iPlot],markersize=markersizes[iPlot],label=legends[iPlot])
        elif plot_type == 'semi-log_x':
            plt.semilogx(x,yAll[iPlot,:],linewidth=linewidths[iPlot],linestyle=linestyles[iPlot],color=colors[iPlot],
                         marker=markertypes[iPlot],markersize=markersizes[iPlot],label=legends[iPlot])
        elif plot_type == 'semi-log_y':
            plt.semilogy(x,yAll[iPlot,:],linewidth=linewidths[iPlot],linestyle=linestyles[iPlot],color=colors[iPlot],
                         marker=markertypes[iPlot],markersize=markersizes[iPlot],label=legends[iPlot])
        elif plot_type == 'log-log':
            plt.loglog(x,yAll[iPlot,:],linewidth=linewidths[iPlot],linestyle=linestyles[iPlot],color=colors[iPlot],
                       marker=markertypes[iPlot],markersize=markersizes[iPlot],label=legends[iPlot])
    if plot_type == 'regular':
        if setXAxisLimits[0]:
            ax.set_xlim(bottom=xAxisLimits[0])
        if setXAxisLimits[1]:
            ax.set_xlim(top=xAxisLimits[1])
        if setYAxisLimits[0]:
            ax.set_ylim(bottom=yAxisLimits[0])
        if setYAxisLimits[1]:
            ax.set_ylim(top=yAxisLimits[1])            
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    if useDefaultMethodToSpecifyTickFontSize:
        plt.xticks(fontsize=tickfontsizes[0])
        plt.yticks(fontsize=tickfontsizes[1])
    else:
        ax.tick_params(axis='x',labelsize=tickfontsizes[0])
        ax.tick_params(axis='y',labelsize=tickfontsizes[1])
    if set_xticks_manually:
        ax.set_xticks(xticks_set_manually,minor=False)
    if legendWithinBox:
        ax.legend(fontsize=legendfontsize,loc=legendposition,shadow=shadow,framealpha=framealpha) 
    else:
        ax.legend(fontsize=legendfontsize,loc=legendposition,bbox_to_anchor=(legendpads[0],legendpads[1]),shadow=shadow,
                  framealpha=framealpha)
    ax.set_title(title,fontsize=titlefontsize,fontweight='bold',y=titlepad)
    if drawMajorGrid and not(drawMinorGrid):
        plt.grid(which='major')
    elif not(drawMajorGrid) and drawMinorGrid:
        plt.grid(which='minor')       
    elif drawMajorGrid and drawMinorGrid:
        plt.grid(which='both')
    if SaveAsPDF:
        plt.savefig(FileName+'.'+FileFormat,format=FileFormat,bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


def PythonConvergencePlot1DSaveAsPDF(output_directory,plot_type,x,y1,y2,linewidths,linestyles,colors,markers,
                                     markertypes,markersizes,labels,labelfontsizes,labelpads,tickfontsizes,legends,
                                     legendfontsize,legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,
                                     fig_size=[9.25,9.25],drawMajorGrid=False,drawMinorGrid=False,legendWithinBox=False,
                                     legendpads=[1.0,0.5],shadow=True,framealpha=1.0,titlepad=1.035,
                                     set_xticks_manually=False,xticks_set_manually=[],FileFormat='pdf'):
    cwd = os.getcwd()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)   
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    if markers[0]:
        if plot_type == 'regular':
            ax.plot(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],marker=markertypes[0],
                    markersize=markersizes[0],label=legends[0])
        elif plot_type == 'semi-log_x':
            ax.semilogx(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],marker=markertypes[0],
                        markersize=markersizes[0],label=legends[0])
        elif plot_type == 'semi-log_y':
            ax.semilogy(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],marker=markertypes[0],
                        markersize=markersizes[0],label=legends[0])
        elif plot_type == 'log-log':
            ax.loglog(x,y1,linewidth=linewidths[0],linestyle=linestyles[0],color=colors[0],marker=markertypes[0],
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
    if markers[1]:
        if plot_type == 'regular':
            ax.plot(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],marker=markertypes[1],
                    markersize=markersizes[1],label=legends[1])
        elif plot_type == 'semi-log_x':
            ax.semilogx(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],marker=markertypes[1],
                        markersize=markersizes[1],label=legends[1])
        elif plot_type == 'semi-log_y':
            ax.semilogy(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],marker=markertypes[1],
                        markersize=markersizes[1],label=legends[1])
        elif plot_type == 'log-log':
            ax.loglog(x,y2,linewidth=linewidths[1],linestyle=linestyles[1],color=colors[1],marker=markertypes[1],
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
    if set_xticks_manually:
        ax.set_xticks(xticks_set_manually,minor=False)
    if legendWithinBox:
        ax.legend(fontsize=legendfontsize,loc=legendposition,shadow=shadow) 
    else:
        ax.legend(fontsize=legendfontsize,loc=legendposition,bbox_to_anchor=(legendpads[0],legendpads[1]),shadow=shadow,
                  framealpha=framealpha) 
    ax.set_title(title,fontsize=titlefontsize,fontweight='bold',y=titlepad)
    if drawMajorGrid and not(drawMinorGrid):
        plt.grid(which='major')
    elif not(drawMajorGrid) and drawMinorGrid:
        plt.grid(which='minor')       
    elif drawMajorGrid and drawMinorGrid:
        plt.grid(which='both') 
    if SaveAsPDF:
        plt.savefig(FileName+'.'+FileFormat,format=FileFormat,bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)
    
    
def PythonConvergencePlots1DSaveAsPDF(output_directory,plot_type,x,y,linewidths,linestyles,colors,markertypes,
                                      markersizes,labels,labelfontsizes,labelpads,tickfontsizes,legends,legendfontsize,
                                      legendposition,title,titlefontsize,SaveAsPDF,FileName,Show,fig_size=[9.25,9.25],
                                      useDefaultMethodToSpecifyTickFontSize=True,drawMajorGrid=False,
                                      drawMinorGrid=False,legendWithinBox=False,legendpads=[1.0,0.5],shadow=True,
                                      framealpha=1.0,titlepad=1.035,set_xticks_manually=False,xticks_set_manually=[],
                                      FileFormat='pdf'):
    cwd = os.getcwd()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)   
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object
    ax = fig.add_subplot(111) # Create an axes object in the figure
    # Note that the shape of y is [nPlots,nSubplots,nPoints] where nPoints = len(x)
    nPlots = y.shape[0]
    nSubplots = 2
    for iPlot in range(0,nPlots):
        for iSubplot in range(0,nSubplots):
            if iSubplot == 0:
                mylinestyle = ""
                mylabel = legends[iPlot]
                mymarker = markertypes[iPlot]
            else:
                mylinestyle = linestyles[iPlot]
                mylabel = ""
                mymarker = ""
            if plot_type == 'regular':
                ax.plot(x,y[iPlot,iSubplot,:],linewidth=linewidths[iPlot],linestyle=mylinestyle,color=colors[iPlot],
                        marker=mymarker,markersize=markersizes[iPlot],label=mylabel)
            elif plot_type == 'semi-log_x':
                ax.semilogx(x,y[iPlot,iSubplot,:],linewidth=linewidths[iPlot],linestyle=mylinestyle,color=colors[iPlot],
                            marker=mymarker,markersize=markersizes[iPlot],label=mylabel)
            elif plot_type == 'semi-log_y':
                ax.semilogy(x,y[iPlot,iSubplot,:],linewidth=linewidths[iPlot],linestyle=mylinestyle,color=colors[iPlot],
                            marker=mymarker,markersize=markersizes[iPlot],label=mylabel)
            elif plot_type == 'log-log':
                ax.loglog(x,y[iPlot,iSubplot,:],linewidth=linewidths[iPlot],linestyle=mylinestyle,color=colors[iPlot],
                          marker=mymarker,markersize=markersizes[iPlot],label=mylabel)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    if useDefaultMethodToSpecifyTickFontSize:
        plt.xticks(fontsize=tickfontsizes[0])
        plt.yticks(fontsize=tickfontsizes[1])
    else:
        ax.tick_params(axis='x',labelsize=tickfontsizes[0])
        ax.tick_params(axis='y',labelsize=tickfontsizes[1])
    if set_xticks_manually:
        ax.set_xticks(xticks_set_manually,minor=False)
    if legendWithinBox:
        ax.legend(fontsize=legendfontsize,loc=legendposition,shadow=shadow,framealpha=framealpha) 
    else:
        ax.legend(fontsize=legendfontsize,loc=legendposition,bbox_to_anchor=(legendpads[0],legendpads[1]),shadow=shadow,
                  framealpha=framealpha) 
    ax.set_title(title,fontsize=titlefontsize,fontweight='bold',y=titlepad)
    if drawMajorGrid and not(drawMinorGrid):
        plt.grid(which='major')
    elif not(drawMajorGrid) and drawMinorGrid:
        plt.grid(which='minor')       
    elif drawMajorGrid and drawMinorGrid:
        plt.grid(which='both') 
    if SaveAsPDF:
        plt.savefig(FileName+'.'+FileFormat,format=FileFormat,bbox_inches='tight')
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)
    

def WriteTecPlot2DStructured(output_directory,x,y,phi,filename):
    cwd = os.getcwd()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    nX = len(x) - 1
    nY = len(y) - 1
    filename += '.tec'
    outputfile = open(filename,'w')
    outputfile.write('VARIABLES = "X", "Y", "PHI"\n')
    outputfile.write('ZONE T="EL00001", I=%d, J=%d, F=POINT\n' %(nX+1,nY+1))
    for iY in range(0,nY+1):
        for iX in range(0,nX+1):
            outputfile.write('%.15g %.15g %.15g\n' %(x[iX],y[iY],phi[iX,iY]))
    outputfile.close()
    os.chdir(cwd)
    
    
def WriteTecPlot2DUnstructured(output_directory,x,y,phi,filename):
    cwd = os.getcwd()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    nX = len(x) - 1
    filename += '.tec'
    outputfile = open(filename,'w')
    outputfile.write('VARIABLES = "X", "Y", "PHI"\n')
    for iX in range(0,nX+1):
        outputfile.write('ZONE T="EL%5.5d", I=1, J=1, F=BLOCK\n' %(iX+1))
        outputfile.write('%.15g %.15g %.15g\n' %(x[iX],y[iX],phi[iX]))
    outputfile.close()
    os.chdir(cwd)
    
    
def ReadTecPlot2DStructured(output_directory,filename,ReturnIndependentVariables=True):
    cwd = os.getcwd()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    nXPlusOneTimesnYPlusOne = len(open(filename).readlines( )[2:])
    data = np.loadtxt(filename,skiprows=2)
    xAll = np.zeros(nXPlusOneTimesnYPlusOne)
    for iXAll in range(0,nXPlusOneTimesnYPlusOne):
        xAll[iXAll] = data[iXAll,0]    
        if iXAll > 0 and xAll[iXAll] < xAll[iXAll-1]:
            nXPlusOne = iXAll
            break
    nYPlusOne = int(nXPlusOneTimesnYPlusOne/nXPlusOne)
    x = np.zeros(nXPlusOne)
    y = np.zeros(nYPlusOne)
    phi = np.zeros((nYPlusOne,nXPlusOne)) # Study the shape of phi here.
    for iX in range(0,nXPlusOne):
        x[iX] = data[iX,0]
    for iY in range(0,nYPlusOne):
        y[iY] = data[iY*nXPlusOne,1]
    for iY in range(0,nYPlusOne):
        for iX in range(0,nXPlusOne):
            phi[iY,iX] = data[iY*nXPlusOne+iX,2] # Study the shape of phi here.
    os.chdir(cwd)
    if ReturnIndependentVariables:
        return x, y, phi
    else:
        return phi
    
    
def ReadTecPlot2DUnstructured(output_directory,filename,ReturnIndependentVariables=True):
    cwd = os.getcwd()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    data = []
    count = 0
    with open(filename,'r') as infile:
        for line in infile:
            if count != 0 and count % 2 == 0:
                data.append(line)
            count += 1
    data = np.loadtxt(data)
    nX = data.shape[0]
    x = np.zeros(nX)
    y = np.zeros(nX)
    phi = np.zeros(nX)
    for iX in range(0,nX):
        x[iX] = data[iX,0]
        y[iX] = data[iX,1]
        phi[iX] = data[iX,2]
    os.chdir(cwd)
    if ReturnIndependentVariables:
        return x, y, phi
    else:
        return phi


def PythonFilledContourPlot2DSaveAsPDF(output_directory,x,y,phi,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
                                       useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,title,titlefontsize,
                                       SaveAsPDF,FileName,Show,fig_size=[10.0,10.0],set_aspect_equal=False,
                                       DataType='Structured',colormap=plt.cm.jet,cbarlabelformat='%.2g',
                                       cbarfontsize=13.75,set_xticks_manually=False,xticks_set_manually=[],
                                       set_yticks_manually=False,yticks_set_manually=[],FileFormat='pdf',
                                       bbox_inches='tight',specify_n_ticks=False,n_ticks=[0,0]):
    cwd = os.getcwd()
    path = cwd + '/' + output_directory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    fig = plt.figure(figsize=(fig_size[0],fig_size[1])) # Create a figure object.
    ax = fig.add_subplot(111) # Create an axes object in the figure.
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
    cbarlabels = np.linspace(cbar_min,cbar_max,num=nColorBarTicks,endpoint=True)
    if DataType == 'Structured':
        FCP = plt.contourf(x,y,phi,nContours,vmin=cbar_min,vmax=cbar_max,cmap=colormap)
    elif DataType == 'Unstructured':
        FCP = plt.tricontourf(x,y,phi,nContours,vmin=cbar_min,vmax=cbar_max,cmap=colormap)
    # FCP stands for filled contour plot.
    plt.title(title,fontsize=titlefontsize,fontweight='bold',y=1.035)
    cbarShrinkRatio = 0.8075
    cbar = plt.colorbar(ScalarMappable(norm=FCP.norm, cmap=FCP.cmap),shrink=cbarShrinkRatio) # Draw colorbar.
    cbar.set_ticks(cbarlabels)
    cbar.set_ticklabels(cbarlabels)
    cbarlabels_final = cbar.get_ticks()
    cbar.ax.set_yticklabels([cbarlabelformat %x for x in cbarlabels_final],fontsize=cbarfontsize)
    plt.xlabel(labels[0],fontsize=labelfontsizes[0],labelpad=labelpads[0])
    plt.ylabel(labels[1],fontsize=labelfontsizes[1],labelpad=labelpads[1])
    plt.xticks(fontsize=tickfontsizes[0])
    plt.yticks(fontsize=tickfontsizes[1])
    if set_xticks_manually:
        ax.set_xticks(xticks_set_manually,minor=False)
    if set_yticks_manually:
        ax.set_yticks(yticks_set_manually,minor=False)
    if specify_n_ticks:
        n_xticks = n_ticks[0]
        n_yticks = n_ticks[1]
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_xticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(n_yticks))
    if SaveAsPDF:
        plt.savefig(FileName+'.'+FileFormat,format=FileFormat,bbox_inches=bbox_inches)
    if Show:
        plt.show()
    plt.close()
    os.chdir(cwd)


def PythonReadFileAndMakeFilledContourPlot2D(output_directory,filename,nContours,labels,labelfontsizes,
                                             labelpads,tickfontsizes,useGivenColorBarLimits,ColorBarLimits,
                                             nColorBarTicks,title,titlefontsize,SaveAsPDF,Show,DataType):
    if DataType == 'Structured':
        x, y, phi = ReadTecPlot2DStructured(output_directory,filename,ReturnIndependentVariables=True)
    elif DataType == 'Unstructured':
        x, y, phi = ReadTecPlot2DUnstructured(output_directory,filename,ReturnIndependentVariables=True)
    filename = filename.replace('.tec','')
    PythonFilledContourPlot2DSaveAsPDF(output_directory,x,y,phi,nContours,labels,labelfontsizes,labelpads,tickfontsizes,
                                       useGivenColorBarLimits,ColorBarLimits,nColorBarTicks,title,titlefontsize,
                                       SaveAsPDF,filename,Show,DataType=DataType)
    
    
def WriteStateVariableLimitsToFile(OutputDirectory,StateVariableLimitsLimits,filename):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    filename += '.curve'
    outputfile = open(filename,'w')
    outputfile.write('%.15g %.15g\n' %(StateVariableLimitsLimits[0],StateVariableLimitsLimits[1]))
    outputfile.close()
    os.chdir(cwd)


def ReadStateVariableLimitsFromFile(OutputDirectory,filename):
    cwd = os.getcwd()
    path = cwd + '/' + OutputDirectory + '/'
    if not os.path.exists(path):
        os.mkdir(path) # os.makedir(path)
    os.chdir(path)
    data = []
    with open(filename,'r') as infile:
        for line in infile:
            data.append(line)
    data = np.loadtxt(data)
    StateVariableLimitsLimits = [data[0],data[1]]
    os.chdir(cwd)
    return StateVariableLimitsLimits