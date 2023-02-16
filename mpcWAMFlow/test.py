#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:49:16 2023

@author: conlinm
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy

os.chdir('/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Tools')
from mpcWAMFlow import opticalFlow,io,validation


class filtering():
    def __init__(self,vidDate,vidTime,WAMPixSize):
        self.vidDate = vidDate
        self.vidTime = vidTime
        self.indir = '/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Results/MergedRectFrames/'
        self.WAMPixSize = WAMPixSize
    
        def batchProcessOpticalFlowRun():
            frames = sorted(os.listdir(self.indir+self.vidDate+self.vidTime+'_ForPub_'+str(self.WAMPixSize)+'mperpix'))
            for iframe in np.arange(0,len(frames)-1):
                print('frame pair #'+str(iframe)+' of '+str(len(frames)-2))
                
                # read first frame #
                frm1,xg,yg = io.readMergedRectFrame(self.indir+self.vidDate+self.vidTime+'_ForPub_'+str(self.WAMPixSize)+'mperpix/'+frames[iframe])
                frm1ref,_,_ = io.readMergedRectFrame(self.indir+self.vidDate+self.vidTime+'_ForPub_'+str(self.WAMPixSize)+'mperpix/'+frames[iframe])
    
                # Initiaize the output arrays if this is the first iteration #
                if iframe==0:
                    flowX_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                    flowY_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                    flowX_all_filt = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                    flowY_all_filt = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                    filterflag_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                    frm1_mean_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                    frm2_mean_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                    frm1_std_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                    frm2_std_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                    
                # Load the land mask, or if it hasn't been created yet then make and save it #
                if iframe==0:
                        with open('/home/server/pi/homes/conlinm/Documents/ROXSI/'+
                                  'Analysis/ArgusWAMFlow/Results/LandMasks_ROI.pkl','rb') as f:
                            masks = pickle.load(f)
                                
                
                # read second frame #
                frm2,_,_ = io.readMergedRectFrame(self.indir+self.vidDate+self.vidTime+'_ForPub_'+str(self.WAMPixSize)+'mperpix/'+frames[iframe+1])
    
                # Apply the land masks to each frame #
                frm1 = opticalFlow.applyLandMask(frm1,masks,xg,yg) 
                frm2 = opticalFlow.applyLandMask(frm2,masks,xg,yg) 
                
                # Re-scale the images to 0-255 %
                # frm1,frm2 = opticalFlow.rescaleImages(frm1, frm2)
    
                # run the optical flow #
                flow,mag,ang = opticalFlow.opticalFlow(frm1,frm2)
                
                # do pixel intensity filtering for flow results #
                flow_filt,mag_filt,ang_filt,filterflag,frm1_mean,frm2_mean,frm1_std,frm2_std = opticalFlow.filterFlow_Intensity(frm1,frm2,flow,mag,ang,kernel_size=5,mean_thresh=5,std_thresh=2)
                filterflag_all[:,:,iframe] = filterflag
                frm1_mean_all[:,:,iframe] = frm1_mean
                frm2_mean_all[:,:,iframe] = frm2_mean
                frm1_std_all[:,:,iframe] = frm1_std
                frm2_std_all[:,:,iframe] = frm2_std
                
                # Convert to m/s #
                flow = opticalFlow.pix2xy(flow,self.WAMPixSize,frame_dt=5)
                flow_filt = opticalFlow.pix2xy(flow_filt,pixel_size=self.WAMPixSize,frame_dt=5)
                
                # Store the result for this frame pair #
                flowX_all[:,:,iframe] = flow[:,:,0]
                flowY_all[:,:,iframe] = flow[:,:,1]
                flowX_all_filt[:,:,iframe] = flow_filt[:,:,0]
                flowY_all_filt[:,:,iframe] = flow_filt[:,:,1]
                
            return flowX_all,flowY_all,flowX_all_filt,flowY_all_filt,filterflag_all,frm1_mean_all,frm2_mean_all,frm1_std_all,frm2_std_all,frm1ref
        
        self.flowX_all,self.flowY_all,self.flowX_all_filt,self.flowY_all_filt,self.filterflag_all,self.frm1_mean_all,self.frm2_mean_all,self.frm1_std_all,self.frm2_std_all,self.frm1 = batchProcessOpticalFlowRun()


    def plot_timeseries(self,r,c):
        '''
        Plot timeseries of velocity at a single point, where filtered values are given with red points.

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        fig = plt.figure(figsize=(6.5,2.5))
        ax0 = plt.axes([0.05,0.05,0.2,0.9])
        ax1 = plt.axes([0.35,0.2,0.62,0.75])

        ax0.imshow(self.frm1,cmap='gray')
        ax0.plot(c,r,'r*')   
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        
        ax1.plot(np.arange(0,len(self.flowX_all[r,c,:])),-self.flowY_all[r,c,:],'k.-')
        ax1.plot(np.arange(0,len(self.flowX_all[r,c,:]))[np.flipud(self.filterflag_all)[r,c,:]==1],-self.flowY_all[r,c,:][np.flipud(self.filterflag_all)[r,c,:]==1],'r.')
        ax1.set_xlabel('Frame pair')
        ax1.set_ylabel('U (m/s)')

        return fig,ax0,ax1
        
           
    def plot_DLAScatter(self,r,c):
        '''
        Plot scatter plot of Anderson et al. (2021) showing pixel mean change and std for all observations
        at a single point as well as which values are filtered 

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
    
        fig = plt.figure()
        ax0 = plt.axes([0.1,0.1,0.4,0.8])
        ax1 = plt.axes([0.55,0.52,0.4,0.35])
        ax2 = plt.axes([0.55,0.1,0.4,0.35])
        cbax = plt.axes([0.55,0.97,0.4,0.02])
        
        ax0.imshow(self.frm1,cmap='gray')
        ax0.plot(c,r,'r*')    
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
               
        h = ax1.scatter((np.flipud(self.frm1_std_all)[r,c,:]+np.flipud(self.frm1_std_all)[r,c,:])/2,
                   -np.flipud(self.flowY_all)[r,c,:],5,
                   abs((np.flipud(self.frm1_mean_all)[r,c,:]).astype(int)-(np.flipud(self.frm2_mean_all)[r,c,:]).astype(int)).astype('uint8'),
                   vmin=0,vmax=5)
        ax1.set_ylabel('V (m/s)')
        
        ax2.plot((np.flipud(self.frm1_std_all)[r,c,:]+np.flipud(self.frm1_std_all)[r,c,:])/2,
                 -np.flipud(self.flowY_all)[r,c,:],'k.')
        ax2.plot((np.flipud(self.frm1_std_all)[r,c,:]+np.flipud(self.frm1_std_all)[r,c,:])[np.flipud(self.filterflag_all)[r,c,:]==1]/2,
                 -np.flipud(self.flowY_all)[r,c,:][np.flipud(self.filterflag_all)[r,c,:]==1],'r.')
        ax2.set_xlabel('std')
        ax2.set_ylabel('V (m/s)')
        plt.colorbar(h,cax=cbax,label='change in mean',orientation='horizontal')
    
    
    def plot_histograms(self,iframe):

        flow = np.stack([self.flowX_all[:,:,iframe],self.flowY_all[:,:,iframe]],axis=2)


        # Histogram of u and v velocities for one image #
        fig,ax = plt.subplots(2,1)
        ax[0].hist(abs(flow[:,:,0].flatten()),bins=np.arange(0,1.05,0.01),density=True)
        ax[0].set_xlabel('abs(U) (m/s)')
        ax[0].set_ylabel('pdf')
        ax[1].hist(abs(flow[:,:,1].flatten()),bins=np.arange(0,1.05,0.01),density=True)
        ax[1].set_xlabel('abs(V) (m/s)')
        ax[1].set_ylabel('pdf')

        # 2d histogram with intensity #
        fig = plt.figure()
        ax0 = plt.axes([0.1,0.58,0.75,0.4])
        cbax0 = plt.axes([0.87,0.58,0.03,0.4])
        ax1 = plt.axes([0.1,0.12,0.75,0.4])
        cbax1 = plt.axes([0.87,0.12,0.03,0.4])
        h1 = ax0.hist2d(abs(flow[:,:,0].flatten()),self.frm1.flatten(),bins=[np.arange(0,1.01,0.01),np.arange(0,255,5)],density=True,vmin=0,vmax=0.0001)
        ax0.set_xlabel('U (m/s)')
        ax0.set_ylabel('I')
        plt.colorbar(h1[3],cax=cbax0,label='pdf')
        h2 = ax1.hist2d(abs(flow[:,:,1].flatten()),self.frm1.flatten(),bins=[np.arange(0,1.01,0.01),np.arange(0,255,5)],density=True,vmin=0,vmax=0.0001)
        ax1.set_xlabel('V (m/s)')
        ax1.set_ylabel('I')
        plt.colorbar(h2[3],cax=cbax1,label='pdf')

        # 2d histogram with std(intensity) in neighborhood #
        frm1_std = self.frm1_std_all[:,:,iframe]
        frm2_std = self.frm2_std_all[:,:,iframe]
        frm_std = (frm1_std+frm2_std)/2       
        fig = plt.figure()
        ax0 = plt.axes([0.1,0.58,0.75,0.4])
        cbax0 = plt.axes([0.87,0.58,0.03,0.4])
        ax1 = plt.axes([0.1,0.12,0.75,0.4])
        cbax1 = plt.axes([0.87,0.12,0.03,0.4])
        h1 = ax0.hist2d(abs(flow[:,:,0].flatten()),frm_std.flatten(),bins=[np.arange(0,1.01,0.01),np.arange(0,80,5)],density=True,vmin=0,vmax=0.0001)
        ax0.set_xlabel('abs(U) (m/s)')
        ax0.set_ylabel('std(I)')
        plt.colorbar(h1[3],cax=cbax0,label='pdf')
        h2 = ax1.hist2d(abs(flow[:,:,1].flatten()),frm_std.flatten(),bins=[np.arange(0,1.01,0.01),np.arange(0,80,5)],density=True,vmin=0,vmax=0.0001)
        ax1.set_xlabel('abs(V) (m/s)')
        ax1.set_ylabel('std(I)')
        plt.colorbar(h2[3],cax=cbax1,label='pdf')

        # 2d histogram with I/std(I) in neighborhood #
        frm1_std = self.frm1_std_all[:,:,iframe]
        frm2_std = self.frm2_std_all[:,:,iframe]
        frm_std = (frm1_std+frm2_std)/2
        fig = plt.figure()
        ax0 = plt.axes([0.1,0.58,0.75,0.4])
        cbax0 = plt.axes([0.87,0.58,0.03,0.4])
        ax1 = plt.axes([0.1,0.12,0.75,0.4])
        cbax1 = plt.axes([0.87,0.12,0.03,0.4])
        yVal = self.frm1.flatten()/self.frm_std.flatten()
        yVal[np.isinf(yVal)] = 0
        h1 = ax0.hist2d(abs(flow[:,:,0].flatten()),yVal,bins=[np.arange(0,1.01,0.01),np.arange(0,100,1)],density=True,vmin=0,vmax=0.001)
        ax0.set_xlabel('abs(U) (m/s)')
        ax0.set_ylabel('I/std(I)')
        ax0.plot(ax0.get_xlim(),(0.1,0.1),'w--')
        ax0.plot(ax0.get_xlim(),(40,40),'w--')
        plt.colorbar(h1[3],cax=cbax0,label='pdf')
        h2 = ax1.hist2d(abs(flow[:,:,1].flatten()),yVal,bins=[np.arange(0,1.01,0.01),np.arange(0,100,1)],density=True,vmin=0,vmax=0.001)
        ax1.set_xlabel('abs(V) (m/s)')
        ax1.set_ylabel('I/std(I)')
        ax1.plot(ax0.get_xlim(),(0.1,0.1),'w--')
        ax1.plot(ax0.get_xlim(),(40,40),'w--')
        plt.colorbar(h2[3],cax=cbax1,label='pdf')

        # 2d histogram with change in mean in neighborhood #
        frm1_means = self.frm1_mean_all[:,:,iframe]
        frm2_means = self.frm2_mean_all[:,:,iframe]
        frm_mean = abs(frm1_means.astype(int)-frm2_means.astype(int)).astype('uint8')       
        fig = plt.figure()
        ax0 = plt.axes([0.1,0.58,0.75,0.4])
        cbax0 = plt.axes([0.87,0.58,0.03,0.4])
        ax1 = plt.axes([0.1,0.12,0.75,0.4])
        cbax1 = plt.axes([0.87,0.12,0.03,0.4])
        h1 = ax0.hist2d(abs(flow[:,:,0].flatten()),frm_mean.flatten(),bins=[np.arange(0,1.01,0.01),np.arange(0,255,1)],density=True,vmin=0,vmax=0.0001)
        ax0.set_xlabel('abs(U) (m/s)')
        ax0.set_ylabel(r"$\Delta\bar{I}$")
        plt.colorbar(h1[3],cax=cbax0,label='pdf')
        h2 = ax1.hist2d(abs(flow[:,:,1].flatten()),frm_mean.flatten(),bins=[np.arange(0,1.01,0.01),np.arange(0,255,1)],density=True,vmin=0,vmax=0.0001)
        ax1.set_xlabel('abs(V) (m/s)')
        ax1.set_ylabel(r"$\Delta\bar{I}$")
        plt.colorbar(h2[3],cax=cbax1,label='pdf')
        
    
    def secondaryVelocityFilter(self,r,c):
        ''' 
        Look at using different secondary velocity filtering thresholds and types 
        '''
        # Compare local and global filters at different thresholds #
        fig1,ax01,ax11 = self.plot_timeseries(r,c)
        ax11.set_xlim(ax11.get_xlim())
        ax11.set_ylim(-0.5,0.5)    
        ax11.plot((-10,-8),(-50,-50),'k.-',label='Unfiltered')
        ax11.plot(-10,-50,'r.',label='Intensity filtered')
        fig2,ax02,ax12 = self.plot_timeseries(r,c)
        h = []
        means = []
        for thresh,col in zip([50,67,90],['g','b','m']):
            flowX_all_VelFilt,flowY_all_VelFilt = opticalFlow.filterFlow_Velocity_Global(self.flowX_all_filt,self.flowY_all_filt,thresh)
            flowX_all_VelFilt_w,flowY_all_VelFilt_w = opticalFlow.filterFlow_Velocity_Windowed(self.flowX_all_filt,self.flowY_all_filt,thresh,time_window=8)
            
            flowX_all_VelFilt[flowX_all_VelFilt==0] = np.nan
            flowY_all_VelFilt[flowY_all_VelFilt==0] = np.nan
            flowX_all_VelFilt_w[flowX_all_VelFilt_w==0] = np.nan
            flowY_all_VelFilt_w[flowY_all_VelFilt_w==0] = np.nan
            
            # Plot the different windowed threhsolded timeseries for the local filters #
            yy = -flowY_all_VelFilt_w[r,c,:]
            xx = np.arange(0,len(yy))
            yy[np.isnan(yy)]= np.interp(xx[np.isnan(yy)], xx[~np.isnan(yy)], yy[~np.isnan(yy)])
            ax11.plot(yy,'-',color=col,label=str(thresh)+'th vel. filtered')
            ######################################################
            
            # Plot the means of the timeseries for local and global filters #
            yMean = np.nanmean(-np.flipud(flowY_all_VelFilt)[r,c,:])
            yMean_w = np.nanmean(-np.flipud(flowY_all_VelFilt_w)[r,c,:])
            
            means.append([yMean,yMean_w])
            
            hh = ax12.plot((0,300),(yMean,yMean),'-',color=col)
            hh_w = ax12.plot((0,300),(yMean_w,yMean_w),'--',color=col)
            
            h.append(hh[0])
            h.append(hh_w[0])
            ##############################################################           
        # fig1.legend(ncol=3,loc='upper center',framealpha=1)
        fig2.legend(h,['50 g','50 w','67 g','67 w','90 g','90 w'],loc='center right')
        return means
        



def pixelSize():
    
    #########################################################
    # Get flowX_all and flowY_all for each pixel size #
    #########################################################
    flowCompDict = dict()
    for pixSize in (0.1,0.5,1,2):
    
        indir = '/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Results/MergedRectFrames/'
        vidDate = '20220713'
        vidTime = '1030'
        
        if pixSize==0.1:
            frames = sorted(os.listdir(indir+vidDate+vidTime+'_ForPub'))
            ioPathStart = indir+vidDate+vidTime+'_ForPub/'
            kSize=25
            maxFrame = 25
        elif pixSize==0.5:
            frames = sorted(os.listdir(indir+vidDate+vidTime+'_ForPub_0.5mperpix'))
            ioPathStart = indir+vidDate+vidTime+'_ForPub_0.5mperpix/'
            kSize=11
            maxFrame = len(frames)-1
        elif pixSize==1:
            frames = sorted(os.listdir(indir+vidDate+vidTime+'_ForPub_1mperpix'))
            ioPathStart = indir+vidDate+vidTime+'_ForPub_1mperpix/'
            kSize=5
            maxFrame = len(frames)-1
        elif pixSize==2:
            frames = sorted(os.listdir(indir+vidDate+vidTime+'_ForPub_2mperpix'))
            ioPathStart = indir+vidDate+vidTime+'_ForPub_2mperpix/'
            kSize=5
            maxFrame = len(frames)-1            
   
        for iframe in np.arange(0,maxFrame):
            print('frame pair #'+str(iframe)+' of '+str(len(frames)-2))
            
            # read first frame #
            frm1,xg,yg = io.readMergedRectFrame(ioPathStart+frames[iframe])
            frm1ref,_,_ = io.readMergedRectFrame(ioPathStart+frames[iframe])
    
            # Initiaize the output arrays if this is the first iteration #
            if iframe==0:
                flowX_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                flowY_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
                
            # Load the land mask, or if it hasn't been created yet then make and save it #
            if iframe==0:
                with open('/home/server/pi/homes/conlinm/Documents/ROXSI/'+
                          'Analysis/ArgusWAMFlow/Results/LandMasks_ROI.pkl','rb') as f:
                    masks = pickle.load(f)
                            
            
            # read second frame #
            frm2,_,_ = io.readMergedRectFrame(ioPathStart+frames[iframe+1])
    
            # Apply the land masks to each frame #
            frm1 = opticalFlow.applyLandMask(frm1,masks,xg,yg) 
            frm2 = opticalFlow.applyLandMask(frm2,masks,xg,yg) 
            
            # Re-scale the images to 0-255 %
            # frm1,frm2 = opticalFlow.rescaleImages(frm1, frm2)
        
            # run the optical flow #
            flow,mag,ang = opticalFlow.opticalFlow(frm1,frm2)
            
            # do pixel intensity filtering for flow results #
            # _,_,_,filterflag,frm1_mean,frm2_mean,frm1_std,frm2_std = opticalFlow.filterFlow(frm1,frm2,flow,mag,ang,kernel_size=25,mean_thresh=5,std_thresh=2)
            flow,mag,ang,_,_,_,_,_ = opticalFlow.filterFlow_Intensity(frm1,frm2,flow,mag,ang,kernel_size=kSize,mean_thresh=5,std_thresh=2)
            
            # Convert to m/s #
            flow = opticalFlow.pix2xy(flow,pixel_size=pixSize,frame_dt=5)
            
            # Store the result for this frame pair #
            flowX_all[:,:,iframe] = flow[:,:,0]
            flowY_all[:,:,iframe] = flow[:,:,1]
            
        # Calculate values #
        minn = [np.min(flowX_all),np.min(flowY_all)]
        pctile25 = [np.percentile(flowX_all[flowX_all!=0],25),np.percentile(flowY_all[flowY_all!=0],25)]
        pctile50 = [np.percentile(flowX_all[flowX_all!=0],50),np.percentile(flowY_all[flowY_all!=0],50)]
        pctile75 = [np.percentile(flowX_all[flowX_all!=0],75),np.percentile(flowY_all[flowY_all!=0],75)]
        maxx = [np.max(flowX_all),np.max(flowY_all)]
        
        flowCompDict[str(pixSize)] = [minn,pctile25,pctile50,pctile75,maxx]

    #########################################################
    # Compare #
    #########################################################
    fig,ax = plt.subplots(1)
    c=0
    for pixSize in [0.1,0.5,1,2]:
        c+=1
        # IQRs #
        ax.plot((c-0.15,c-0.05),(flowCompDict[str(pixSize)][1][0],flowCompDict[str(pixSize)][1][0]),'b')
        ax.plot((c-0.15,c-0.05),(flowCompDict[str(pixSize)][3][0],flowCompDict[str(pixSize)][3][0]),'b')
        ax.plot((c-0.15,c-0.15),(flowCompDict[str(pixSize)][1][0],flowCompDict[str(pixSize)][3][0]),'b')
        ax.plot((c-0.05,c-0.05),(flowCompDict[str(pixSize)][1][0],flowCompDict[str(pixSize)][3][0]),'b')
 
        ax.plot((c+0.05,c+0.15),(flowCompDict[str(pixSize)][1][1],flowCompDict[str(pixSize)][1][1]),'r')
        ax.plot((c+0.05,c+0.15),(flowCompDict[str(pixSize)][3][1],flowCompDict[str(pixSize)][3][1]),'r')
        ax.plot((c+0.05,c+0.05),(flowCompDict[str(pixSize)][1][1],flowCompDict[str(pixSize)][3][1]),'r')
        ax.plot((c+0.15,c+0.15),(flowCompDict[str(pixSize)][1][1],flowCompDict[str(pixSize)][3][1]),'r')
        
        # Medians #
        ax.plot((c-0.15,c-0.05),(flowCompDict[str(pixSize)][2][0],flowCompDict[str(pixSize)][2][0]),'b')
        ax.plot((c+0.05,c+0.15),(flowCompDict[str(pixSize)][2][1],flowCompDict[str(pixSize)][2][1]),'r')

    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['0.1 m','0.5 m','1 m','2 m'])
    ax.set_ylabel('Pixel velocity (pix/frame)')
    
    return fig



def adcpVelocityType(numLayers=21,surfaceLayers=3,windowSize=5):
    
    validation_adcp = validation.ADCP()
    
    # Make dummy WAMFlow outputs #
    xx = np.arange(-60,50)
    yy = np.arange(0,250)
    xg,yg = np.meshgrid(xx,yy)
    flowX = np.zeros([len(yy),len(xx)])
    flowY = np.zeros([len(yy),len(xx)])
    
    vidDates = ['20220712','20220713']
    # vidTimes = [ '0800','0830','0900','0930','1000','1030','1100','1130']
    vidTimes = ['0030','0100','0130','0200','0230','0300','0330','0400','0430','0500','0530','0600','0630','0700','0730',
                    '0800','0830','0900','0930','1000','1030','1100','1130','1200','1230','1300','1330','1400','1430',
                    '1500','1530','1600','1630','1700','1730','1800','1830','1900','1930','2000','2030','2100','2130',
                    '2200','2230','2300','2330']
    x11 = {'Sigma':np.empty([len(vidTimes)*len(vidDates),2]),
           'DepthAvg':np.empty([len(vidTimes)*len(vidDates),2]),
           'TopBin':np.empty([len(vidTimes)*len(vidDates),2])}
    x13 = {'Sigma':np.empty([len(vidTimes)*len(vidDates),2]),
           'DepthAvg':np.empty([len(vidTimes)*len(vidDates),2]),
           'TopBin':np.empty([len(vidTimes)*len(vidDates),2])}
    
    c = -1     
    for vidDate in vidDates:
        for vidTime in vidTimes :
    
            c+=1
            [_,adcp_x11,_] = validation_adcp.quantitative(flowX,flowY,xg,yg,vidDate,vidTime,'x11',
                                                        numLayers=numLayers,
                                                        surfaceLayers=surfaceLayers,
                                                        windowSize=windowSize)
            [_,adcp_x13,_] = validation_adcp.quantitative(flowX,flowY,xg,yg,vidDate,vidTime,'x13',
                                                        numLayers=numLayers,
                                                        surfaceLayers=surfaceLayers,
                                                        windowSize=windowSize)
            
            x11['Sigma'][c,:] = np.array(adcp_x11['Sigma'])
            x11['DepthAvg'][c,:] = np.array(adcp_x11['DepthAvg'])
            x11['TopBin'][c,:] = np.array(adcp_x11['TopBin'])
            
            x13['Sigma'][c,:] = np.array(adcp_x13['Sigma'])
            x13['DepthAvg'][c,:] = np.array(adcp_x13['DepthAvg'])
            x13['TopBin'][c,:] = np.array(adcp_x13['TopBin'])
            
    t = np.arange(0,len(vidTimes)*len(vidDates)*30,30)
    fig,ax = plt.subplots(2,2,sharex=True)  
    ax[0][0].plot(t,x11['Sigma'][:,0],'.-',label='Sigma')
    ax[0][0].plot(t,x11['DepthAvg'][:,0],'.-',label='Depth Avg')
    ax[0][0].plot(t,x11['TopBin'][:,0],'.-',label='Top Bin')
    ax[0][0].set_title('V (m/s)')
    ax[0][0].set_ylabel('x11',rotation=90,fontweight='bold')
    fig.legend(loc='upper center',ncol=3)
    ax[0][1].plot(t,x11['Sigma'][:,1],'.-',label='Sigma')
    ax[0][1].plot(t,x11['DepthAvg'][:,1],'.-',label='Depth Avg')
    ax[0][1].plot(t,x11['TopBin'][:,1],'.-',label='Top Bin')
    ax[0][1].set_title('U (m/s)')
    ax[1][0].plot(t,x13['Sigma'][:,0],'.-',label='Sigma')
    ax[1][0].plot(t,x13['DepthAvg'][:,0],'.-',label='Depth Avg')
    ax[1][0].plot(t,x13['TopBin'][:,0],'.-',label='Top Bin')
    ax[1][0].set_ylabel('x13',rotation=90,fontweight='bold')  
    ax[1][0].set_xlabel('time (s)')          
    ax[1][1].plot(t,x13['Sigma'][:,1],'.-',label='Sigma')
    ax[1][1].plot(t,x13['DepthAvg'][:,1],'.-',label='Depth Avg')
    ax[1][1].plot(t,x13['TopBin'][:,1],'.-',label='Top Bin')
    ax[1][1].set_xlabel('time (s)')

