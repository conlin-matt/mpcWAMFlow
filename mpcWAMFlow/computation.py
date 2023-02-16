#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:27:03 2023

@author: conlinm
"""

import copy
import numpy as np



def calcTimeAverageFlow(flowX_all,flowY_all):
    
    # Just doing np.nanmean(flowX_all,axis=2) gave memory overload errors, so had to do it row by row. Also turn an zeros to NaNs
    # and (?)filter for existance at each point(?)
    flowX = np.empty(np.shape(flowX_all)[0:2])
    flowY = np.empty(np.shape(flowX_all)[0:2])
    for r in np.arange(0,len(flowX_all[:,0])):
        
        xrow = flowX_all[r,:,:]
        yrow = flowY_all[r,:,:]
        xrow[xrow==0] = np.nan
        yrow[yrow==0] = np.nan
        flowX[r,:] = np.nanmean(xrow,axis=1)
        flowY[r,:] = np.nanmean(yrow,axis=1)
        # nNaNx = [len(xrow[i,:][np.isnan(xrow[i,:])]) for i in range(0,len(xrow[:,0]))]
        # nNaNy = [len(yrow[i,:][np.isnan(yrow[i,:])]) for i in range(0,len(yrow[:,0]))]
        # obsThresh = 3/len(xrow[0,:])
        # flowX[r,np.where(np.array(nNaNx)/len(xrow[0,:])>(1-obsThresh))[0]] = np.nan
        # flowY[r,np.where(np.array(nNaNy)/len(yrow[0,:])>(1-obsThresh))[0]] = np.nan
        
    flowX[np.isnan(flowX)] = 0    
    flowY[np.isnan(flowY)] = 0    
    
    return flowX,flowY


def calcXShoreAverageFlow(flowX,flowY,windowSize=20):
    
    xMean = np.zeros([1,len(flowX[0,:])])[0]*np.nan
    yMean = np.zeros([1,len(flowX[0,:])])[0]*np.nan
    for c in np.arange(int(np.floor(windowSize/2)),len(flowX[0,:])-int(np.ceil(windowSize/2))): # Slide through the columns #
        subX = copy.deepcopy(flowX)[:,int(c-np.floor(windowSize/2)):int(c-np.floor(windowSize/2))+windowSize]
        subY = copy.deepcopy(flowY)[:,int(c-np.floor(windowSize/2)):int(c-np.floor(windowSize/2))+windowSize]
        subX[subX==0] = np.nan
        subY[subY==0] = np.nan
        meanX = np.nanmean(subX)
        meanY = np.nanmean(subY)
        xMean[int(c+np.floor(windowSize/2))] = meanX
        yMean[int(c+np.floor(windowSize/2))] = -meanY
    return xMean,yMean   
    
  
def calcRipAndFeederVelocity(flowX,flowY,xg,yg):
    
    feederLeft = flowX[np.logical_and(np.logical_and(yg>50,yg<150),xg<0)]
    feederRight = flowX[np.logical_and(np.logical_and(yg>50,yg<100),xg>0)]
    vel_feeder = np.mean([np.percentile(feederLeft,67),np.percentile(abs(feederRight),67)])
    
    rip1 = flowY[np.logical_and(yg>100,yg<200)]
    rip = rip1[rip1>0] # Offshore velocities are >0 #
    vel_rip = np.percentile(rip,67)
    
    return vel_feeder,vel_rip

   
def computeRipPositionAndVelocity(flowX,flowY,xMean,yMean,xg,windowSize=20):
    
    # Take the position as the position of min mean cross-shore vel #
    xg = xg[0,:]
    iRipPos = np.where(yMean==np.nanmin(yMean))[0][0]
    ripPos = xg[iRipPos]
    
    # Find the rip velocity #
    roi = copy.deepcopy(flowY)[:,iRipPos-int(windowSize/2):iRipPos+int(windowSize/2)]
    roi[roi==0] = np.nan
    ripVelDict = {'median':-np.nanmedian(roi),
                  'mean':-np.nanmean(roi),
                  '25th':-np.nanpercentile(roi,25),
                  '75th':-np.nanpercentile(roi,75),
                  '95th':-np.nanpercentile(roi,95),
                  'sig':-np.mean(roi[roi>np.nanpercentile(roi,67)]),
                  'max':-np.nanmax(roi)}
    
    return ripPos,ripVelDict