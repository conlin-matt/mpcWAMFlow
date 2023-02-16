#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:03:55 2022

@author: conlinm
"""

from matplotlib.pyplot import savefig,close
import numpy as np
import os
import pickle

os.chdir('/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Tools')
from mpcWAMFlow import opticalFlow,computation,plot,io,validation

indir = '/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Results/MergedRectFrames/'
savedir = '/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Results/OpticalFlowResults/'


###################################################################
# Input video date and time to use #
###################################################################
vidDate = '20220628'
for vidTime in ['0930','1000','1030','1100','1130',
                '1200','1230','1300','1330','1400',
                '1430','1500']:
             
  
    ###################################################################
    # Do WAMFlow for entire WAM and store results #
    ###################################################################
    frames = sorted(os.listdir(indir+vidDate+vidTime+'_ForPub_1mperpix'))
    for iframe in np.arange(0,len(frames)-1):
        print('frame pair #'+str(iframe)+' of '+str(len(frames)-2))
        
        # read first frame #
        frm1,xg,yg = io.readMergedRectFrame(indir+vidDate+vidTime+'_ForPub_1mperpix/'+frames[iframe])
        frm1ref,_,_ = io.readMergedRectFrame(indir+vidDate+vidTime+'_ForPub_1mperpix/'+frames[iframe])
    
        # Initiaize the output arrays if this is the first iteration #
        if iframe==0:
            flowX_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
            flowY_all = np.empty(list(np.shape(frm1)+(len(frames)-1,)))
            
        # Load the land mask, or if it hasn't been created yet then make and save it #
        if iframe==0:
            if not os.path.exists('/home/server/pi/homes/conlinm/Documents/ROXSI/'+
                          'Analysis/ArgusWAMFlow/Results/LandMasks_ROI.pkl'):
                # Manually select boundary areas for image masks. Do twice, fist on the southern point and then on the northern point #
                masks= opticalFlow.createLandMask(frm1,xg,yg) 
                with open('/home/server/pi/homes/conlinm/Documents/ROXSI'+
                          '/Analysis/ArgusWAMFlow/Results/LandMasks_ROI.pkl','wb') as f:
                    pickle.dump(masks,f)
            else:
                with open('/home/server/pi/homes/conlinm/Documents/ROXSI/'+
                          'Analysis/ArgusWAMFlow/Results/LandMasks_ROI.pkl','rb') as f:
                    masks = pickle.load(f)
                        
        
        # read second frame #
        frm2,_,_ = io.readMergedRectFrame(indir+vidDate+vidTime+'_ForPub_1mperpix/'+frames[iframe+1])
    
        # Apply the land masks to each frame #
        frm1 = opticalFlow.applyLandMask(frm1,masks,xg,yg) 
        frm2 = opticalFlow.applyLandMask(frm2,masks,xg,yg) 
        
        # Re-scale the images to 0-255 %
        # frm1,frm2 = opticalFlow.rescaleImages(frm1, frm2)
    
        # run the optical flow #
        flow,mag,ang = opticalFlow.opticalFlow(frm1,frm2)
        
        # do pixel intensity filtering for flow results #
        flow_filt,mag_filt,ang_filt,_,_,_,_,_ = opticalFlow.filterFlow_Intensity(frm1,frm2,flow,mag,ang,kernel_size=5,mean_thresh=5,std_thresh=2)
        
        # Convert to m/s #
        flow_filt = opticalFlow.pix2xy(flow_filt,pixel_size=1,frame_dt=5)
        
        # Store the result for this frame pair #
        flowX_all[:,:,iframe] = flow_filt[:,:,0]
        flowY_all[:,:,iframe] = flow_filt[:,:,1]
        
    
    ###################################################################
    # Do secondary velocity filtering #
    ###################################################################
    # flowX_all_filt_min,flowY_all_filt_min = opticalFlow.filterFlow_Velocity_Windowed(flowX_all,flowY_all,thresh=50,time_window=8)
    flowX_all_filt,flowY_all_filt = opticalFlow.filterFlow_Velocity_Windowed(flowX_all,flowY_all,thresh=67,time_window=8)
    # flowX_all_filt_max,flowY_all_filt_max = opticalFlow.filterFlow_Velocity_Windowed(flowX_all,flowY_all,thresh=90,time_window=8)
    
    ###################################################################
    # Calc time average flows #
    ###################################################################
    flowX,flowY = computation.calcTimeAverageFlow(flowX_all_filt,flowY_all_filt) # Temporally averaged x and y flow #
    
    
    ###################################################################
    # Calc space average of time average flows and extract rip #
    ###################################################################
    feederVel,ripVel = computation.calcRipAndFeederVelocity(flowX,flowY,xg,yg)
    # ymeanFlowX,ymeanFlowY = computation.calcXShoreAverageFlow(flowX,flowY,windowSize=5) # Cross-shore averages of temporal averaged flow #
    # ripPos,ripVelDict = computation.computeRipPositionAndVelocity(flowX,flowY,ymeanFlowX,ymeanFlowY,xg,windowSize=5)
    
    
    ###################################################################
    # Plot the flow #
    ###################################################################
    vis = plot.drawFlow_Streamplot_WithMeanFlows(flowX,flowY,
                                                 xg,yg,frm1,
                                                 yBound=0.5,minLength=2,winSize=2,density=1.5)
    
    
    ###################################################################
    # Validation #
    ###################################################################
    if int(vidDate+vidTime)>202206211800:
        validation_adcp = validation.ADCP()
        wam_x11,adcp_x11,error_x11 = validation_adcp.quantitative(flowX,flowY,xg,yg,vidDate,vidTime,'x11',
                                                          numLayers=11,
                                                          surfaceLayers=3,
                                                          windowSize=5)
        wam_x13,adcp_x13,error_x13 = validation_adcp.quantitative(flowX,flowY,xg,yg,vidDate,vidTime,'x13',
                                                          numLayers=11,
                                                          surfaceLayers=3,
                                                          windowSize=5)
    else:
        wam_x11 = np.nan
        adcp_x11 = np.nan
        error_x11 = [np.nan,np.nan,np.nan]
        wam_x13 = np.nan
        adcp_x13 = np.nan
        error_x13 = [np.nan,np.nan,np.nan]
            
    
    if vidDate == '20220713':
        if vidTime=='1030':
            validation_drifter = validation.drifter(deploymentNum=1)
            validation_drifter.qualitative(flowX,flowY,frm1,xg,yg,dx=1,dy=1)
            rmse_drifter,meanE_drifter = validation_drifter.quantitative(flowX,flowY,frm1,xg,yg,dx=1,dy=1,drifterObsThresh=5)
                    
        elif vidTime=='1230':
            validation_drifter = validation.drifter(deploymentNum=2)
            validation_drifter.qualitative(flowX,flowY,frm1,xg,yg,dx=1,dy=1)
            rmse_drifter,meanE_drifter = validation_drifter.quantitative(flowX,flowY,frm1,xg,yg,dx=1,dy=1,drifterObsThresh=5)
    
    
    ###################################################################
    # Save #
    ###################################################################
    savedirfull = savedir+vidDate+vidTime
    os.mkdir(savedirfull)
    np.save(savedirfull+'/flowX.npy',flowX)
    np.save(savedirfull+'/flowY.npy',flowY)
    np.save(savedirfull+'/feederVel.npy',feederVel)
    np.save(savedirfull+'/ripVel.npy',ripVel)
    np.save(savedirfull+'/adcp_x11.npy',adcp_x11)
    np.save(savedirfull+'/adcp_x13.npy',adcp_x13)
    np.save(savedirfull+'/error_x11.npy',error_x11)
    np.save(savedirfull+'/error_x13.npy',error_x13)
    if vidDate == '20220713':
        if vidTime=='1030' or vidTime=='1230':
            np.save(savedirfull+'/rmse_drifter.npy',rmse_drifter)
            np.save(savedirfull+'/meanE_drifter.npy',meanE_drifter)
            savefig(savedirfull+'/drifter_quantiative.png',dpi=450)
            close()
            savefig(savedirfull+'/drifter_qualitative.png',dpi=450)
        else:
            savefig(savedirfull+'/vis.png',dpi=350)
    else:
        savefig(savedirfull+'/vis.png',dpi=350)





