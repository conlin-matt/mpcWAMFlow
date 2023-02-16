#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:24:59 2023

@author: conlinm
"""

import copy
import cv2
from matplotlib.backend_bases import MouseButton
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy



def createLandMask(frm,xg,yg):
    extents = [np.min(xg),
               np.max(xg),
               np.min(yg),
               np.max(yg)]
    fig,ax = plt.subplots(1)
    ax.imshow(frm,extent=extents,interpolation='bilinear',cmap='gray')
    
    mask_south = plt.ginput(-1,timeout=300,show_clicks=True,mouse_stop=MouseButton.RIGHT)
    mask_north = plt.ginput(-1,timeout=300,show_clicks=True,mouse_stop=MouseButton.RIGHT)
    masks = [mask_south,mask_north]

    return masks


def applyLandMask(frm,masks,xg,yg):
    for mask in masks:
        xgf = np.array(xg.flatten()).reshape(np.size(xg),1)
        ygf = np.array(yg.flatten()).reshape(np.size(yg),1)
        xgygf = np.hstack([xgf,ygf])
        
        pth = path.Path(mask)
        inp = pth.contains_points(xgygf)
        inp = np.reshape(inp,np.shape(xg))
        
        frm[np.flipud(inp)] = 0
        
    return frm
        

def rescaleImages(frm1,frm2):
    # Re-scale the image to be 0-255, which seems to make the algorithm happy? #
    frm1 = ((frm1/np.max(frm1))*255).astype('uint8')
    frm2 = ((frm2/np.max(frm2))*255).astype('uint8')
    
    return frm1,frm2



def opticalFlow(frm1,frm2):
    
    flow = cv2.calcOpticalFlowFarneback(prev=frm1,next=frm2, 
                                       flow=None, pyr_scale=0.5, levels=3, winsize=13, iterations=10, 
                                       poly_n=7, poly_sigma=1.5, flags=0)
    # flow = cv2.calcOpticalFlowFarneback(prev=frm1,next=frm2, 
    #                                    flow=None, pyr_scale=0.5, levels=3, winsize=24, iterations=3, 
    #                                    poly_n=7, poly_sigma=1.5, flags=0)
    
    # flow = flow*10
    # flow[np.abs(flow)<0.005] = 0
    
    # Compute magnite and angle of 2D vector
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    return flow,mag,ang



def filterFlow_Intensity(frm1,frm2,flow,mag,ang,kernel_size=5,mean_thresh=5,std_thresh=2):
    
    def calcLocalstd(image, N):
        im = np.array(image, dtype=float)
        im2 = im**2
        ones = np.ones(im.shape)
        
        kernel = np.ones((2*N+1, 2*N+1))
        s = scipy.signal.convolve2d(im, kernel, mode="same")
        s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
        ns = scipy.signal.convolve2d(ones, kernel, mode="same")
        
        return np.sqrt((s2 - s**2 / ns) / ns)

    kernel_mean = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    frm1_means = cv2.filter2D(frm1, -1, kernel_mean)
    frm2_means = cv2.filter2D(frm2, -1, kernel_mean)
    
    frm1_std = calcLocalstd(frm1,N=int((kernel_size-1)/2)) # e.g. N=2 gives a kernel size of 5 %
    frm2_std = calcLocalstd(frm2,N=int((kernel_size-1)/2))
    iB = (abs(frm1_means.astype(int)-frm2_means.astype(int)).astype('uint8')>mean_thresh) | ((frm1_std+frm2_std)/2<std_thresh)# | (frm1_means<50)# | ((frm1_std+frm2_std)/2>10)
    
    filterflag = np.zeros_like(frm1)
    filterflag[iB] = 1
    
    flow_filt = copy.deepcopy(flow)
    mag_filt = copy.deepcopy(mag)
    ang_filt = copy.deepcopy(ang)
    
    flow_filt[:,:,0][iB] = 0
    flow_filt[:,:,1][iB] = 0
    mag_filt[iB] = 0
    ang_filt[iB] = 0
    
    return flow_filt,mag_filt,ang_filt,filterflag,frm1_means,frm2_means,frm1_std,frm2_std



def filterFlow_Velocity_Global(flowX_all,flowY_all,thresh):
    flowX_all_w = copy.deepcopy(flowX_all)
    flowY_all_w = copy.deepcopy(flowY_all)
    
    flowX_all_w[flowX_all_w==0] = np.nan
    flowY_all_w[flowY_all_w==0] = np.nan
       
    threshX = np.nanpercentile(abs(flowX_all_w),thresh,axis=2)
    threshY = np.nanpercentile(abs(flowY_all_w),thresh,axis=2)
    for time in range(len(flowX_all_w[0,0,:])):
        flowX_all_w[:,:,time][abs(flowX_all_w[:,:,time])<threshX] = np.nan
        flowY_all_w[:,:,time][abs(flowY_all_w[:,:,time])<threshY] = np.nan
    flowX_all_w[np.isnan(flowX_all_w)] = 0
    flowY_all_w[np.isnan(flowY_all_w)] = 0

    return flowX_all_w,flowY_all_w

    
def filterFlow_Velocity_Windowed(flowX_all,flowY_all,thresh,time_window):    
    flowX_all_w = copy.deepcopy(flowX_all)
    flowY_all_w = copy.deepcopy(flowY_all)
    
    flowX_all_w[flowX_all_w==0] = np.nan
    flowY_all_w[flowY_all_w==0] = np.nan
    
    windows = [np.arange(i-(time_window/2),i+(time_window/2)) for i in np.arange(int(time_window/2),int(len(flowX_all[0,0,:])-(time_window/2)),time_window)]
    
    for window in windows:

        threshX = np.nanpercentile(abs(flowX_all_w)[:,:,window.astype(int)],thresh,axis=2)
        threshY = np.nanpercentile(abs(flowY_all_w)[:,:,window.astype(int)],thresh,axis=2)
     
        for time in window.astype(int):
            flowX_all_w[:,:,time][abs(flowX_all_w[:,:,time])<threshX] = np.nan
            flowY_all_w[:,:,time][abs(flowY_all_w[:,:,time])<threshY] = np.nan

    flowX_all_w[np.isnan(flowX_all_w)] = 0
    flowY_all_w[np.isnan(flowY_all_w)] = 0
    flowX_all[np.isnan(flowX_all)] = 0
    flowY_all[np.isnan(flowY_all)] = 0
        
    return flowX_all_w,flowY_all_w
        
    


def pix2xy(flow,pixel_size,frame_dt):
    '''
    Convert velocities in pixels/frame to local m/s using frame timestep and pixel size.
    Also flip the velocity arrays up down so they can be plotted directly.
    
    *Note: the x and y directions are relative to the rectified image. Therefore,
    the velocity in the x-direction is local V, and velocity in the y-direction
    is local U (positive onshore).
    
    args:
        flow: The mxnx2 array of pixel flow velocities returned by opticalFlow
        pixel_size: size of each pixel in m/pix
        frame_dt: time between each frame in seconds/frame
    
    returns:
        flow_xy: m x n x 2 array of x (V) and y (U) flow velocities in m/s
    '''
    
    flow_xy = flow*pixel_size/frame_dt
    flow_x = np.flipud(flow_xy[:,:,0])
    flow_y = -np.flipud(flow_xy[:,:,1])
    flow_xy2 = np.stack([flow_x,flow_y],axis=2)

    return flow_xy2




    
    