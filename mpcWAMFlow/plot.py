#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:28:55 2023

@author: conlinm
"""
import copy
import cv2
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from . import computation


def drawFlow_cv2Lines(img, flow, step=16):
    '''
    Function created by Dylan Anderson at the FRF, see https://github.com/anderdyl/waveAveragedMovies/blob/master/opticalFlowWAM.py#L30
    '''

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    vis = cv2.resize(vis,(960,540))
    return vis


def drawFlow_Streamplot(flowX,flowY,xg,yg,frm1):
    
    speed = np.sqrt(np.power(flowX,2)+np.power(flowY,2))
    extents = [np.min(xg),
               np.max(xg),
               np.min(yg),
               np.max(yg)]

    fig,ax = plt.subplots(1)
    ax.imshow(frm1,extent=extents,interpolation='bilinear',cmap='gray')
    strm = ax.streamplot(xg,
               yg,
               np.flipud(flowX)*0.1/5*255,-np.flipud(flowY)*0.1/5*255,
               linewidth = 2*np.flipud(speed)/np.percentile(speed,97),
               color = np.flipud(flowY)*0.1/5*255,
               broken_streamlines=False,
               minlength=3.2,
               density=2,cmap='seismic',arrowstyle ='simple',arrowsize = 0,
               norm=mc.Normalize(-0.2,0.2))
    fig.colorbar(cm.ScalarMappable(norm=mc.Normalize(-0.000001,0.000001),cmap='seismic_r'))
    
    return strm

   
def drawFlow_Streamplot_WithMeanFlows(flowX,flowY,xg,yg,frm1,yBound=0.035,minLength=3,winSize=5,density=1):
       
    
    speed = np.sqrt(np.power(flowX,2)+np.power(flowY,2))
    extents = [np.min(xg),
               np.max(xg),
               np.min(yg),
               np.max(yg)]
    
    fig = plt.figure(figsize=(4.5,8))
    ax = plt.axes([0.1,0.4,0.8,0.55])

    ax.imshow(frm1,extent=extents,interpolation='bilinear',cmap='gray')
    strm = ax.streamplot(xg,
               yg,
               flowX,flowY,
               linewidth = 2*speed/np.percentile(speed,97),
               color = flowY,
               broken_streamlines=False,
               minlength=minLength,
               density=density,cmap='seismic',arrowstyle ='simple',arrowsize = 0,
               norm=mc.Normalize(-0.5,0.5))
    ax.plot(16.4190,69.5857,'w^',markeredgecolor='k')
    ax.plot(8.6423,48.6137,'wo',markeredgecolor='k')
    ax.set_xticklabels([])
    ax.set_ylabel('Cross-shore (m)')
    axlim = ax.get_xlim()

    axpos = ax.get_position()
    axs1 = plt.axes([axpos.xmin,0.27,axpos.xmax-axpos.xmin,0.1])
    axs2 = plt.axes([axpos.xmin,0.14,axpos.xmax-axpos.xmin,0.1])
    cbax = plt.axes([axpos.xmax+0.02,0.4,0.02,axpos.ymax-axpos.ymin])


    xMean,yMean = computation.calcXShoreAverageFlow(flowX,flowY,windowSize=winSize)
    fillPos = copy.deepcopy(yMean); fillPos[yMean<0] = 0
    fillNeg = copy.deepcopy(yMean); fillNeg[yMean>=0] = 0
    axs1.fill_between(xg[0,:],np.zeros(np.size(yMean)),fillPos,color='b',edgecolor='k')
    axs1.fill_between(xg[0,:],np.zeros(np.size(yMean)),fillNeg,color='r',edgecolor='k')
    axs1.set_xlim(axlim)
    axs1.set_xticklabels([])
    axs1.set_ylabel(r'$\hat{\bar{U}}$ (m/s)')
    axs1.set_ylim([-yBound,yBound])

    fillPos = copy.deepcopy(xMean); fillPos[xMean<0] = 0
    fillNeg = copy.deepcopy(xMean); fillNeg[xMean>=0] = 0
    axs2.fill_between(xg[0,:],np.zeros(np.size(xMean)),fillPos,color='b',edgecolor='k')
    axs2.fill_between(xg[0,:],np.zeros(np.size(xMean)),fillNeg,color='r',edgecolor='k')
    axs2.set_xlim(axlim)
    axs2.set_xlabel('Alongshore (m)')
    axs2.set_ylabel(r'$\hat{\bar{V}}$ (m/s)')
    axs2.set_ylim([-yBound,yBound])

    fig.colorbar(cm.ScalarMappable(norm=mc.Normalize(-0.5,0.5),cmap='seismic_r'),cbax,label=r'$\bar{U}$ (m/s)')    
    
    axsinset = plt.axes([0.05,0.8,0.3,0.18])
    MergedRectFrameREF = loadmat('/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Results/MergedRectFrames/202207131030/f0001.mat')['MergedRectFrame']           
    extentsREF = [np.min(MergedRectFrameREF['frc1'][0][0][0]['xg'][0]),
               np.max(MergedRectFrameREF['frc1'][0][0][0]['xg'][0]),
               np.min(MergedRectFrameREF['frc1'][0][0][0]['yg'][0]),
               np.max(MergedRectFrameREF['frc1'][0][0][0]['yg'][0])]
    
    
    frmREF = np.flipud(MergedRectFrameREF['mergedRectFrame'][0][0])
    axsinset.imshow(frmREF,extent=extentsREF,interpolation='bilinear',cmap='gray')
    axsinset.plot((axlim[0],axlim[0]),(0,250),'y')
    axsinset.plot((axlim[1],axlim[1]),(0,250),'y')
    axsinset.plot((axlim[0],axlim[1]),(0.8,0.8),'y')
    axsinset.plot((axlim[0],axlim[1]),(248,248),'y')
    axsinset.set_xticklabels([])
    axsinset.set_yticklabels([])
    
    return strm