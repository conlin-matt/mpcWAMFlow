#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:00:45 2023

@author: conlinm
"""

import copy
import datetime
import mat73
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from pyproj import Proj
import scipy

from . import plot

os.chdir('pyROXSI')
from roxsi_pyfuns import transfer_functions as tf # pyROXSI package by Mika Malila, UNC #
os.chdir('../')


class drifter():
    
    def __init__(self,deploymentNum):
        self.deploymentNum = deploymentNum
        if self.deploymentNum==1:
            self.timeBounds = (datetime.datetime(2022,7,13,10,19),
                               datetime.datetime(2022,7,13,11,27)) 
        elif self.deploymentNum==2:
            self.timeBounds = (datetime.datetime(2022,7,13,12,16),
                               datetime.datetime(2022,7,13,12,54)) 
        self.timeBoundsMat = [366 + d.toordinal() + (d - datetime.datetime.fromordinal(d.toordinal())).total_seconds()/(24*60*60) for d in self.timeBounds]
            
        
    def ll2local(self,lon,lat):
        myProj = Proj(proj='utm',zone=10,ellps='WGS84',datum='WGS84') # Define projection for converting between ll and UTM #
        # From Olavo: Local coord sys transformation parameters #
        latref = 36 +  (37/60) + (26.5187/3600)
        lonref = -121 -(56/60) - (25.19048/3600)
        angleref = 293;             
        xref,yref = myProj(lonref,latref)
        
        x,y = myProj(lon,lat)           
        R = np.vstack([[np.cos(np.radians(angleref)),-np.sin(np.radians(angleref))],
                       [np.sin(np.radians(angleref)),np.cos(np.radians(angleref))]])           
        T = np.vstack([x-xref,y-yref])
        rr = R@T          
        xLocal = rr[0,:]
        yLocal = rr[1,:]
        
        return xLocal,yLocal,angleref
        
        
    def c2LocalAng(self,courseV):
        '''
        Function to transform course angles to the local coordinate system 
        '''
        corr = courseV-self.angleref
        corr[corr<0] = 360+corr[corr<0]
        return corr
        
               
    def SC2VxVy(self,speedV,courseV):
        '''
        Function to calculate x and y velcocity components from speed and
        (local) course angle info.
        '''

        Vxv = []
        Vyv = []
        for i in range(len(speedV)):
            speed = speedV[i]
            course = courseV[i]
            if np.isnan(speed) | np.isnan(course):
                Vx = np.nan
                Vy = np.nan
            else:
                if 0<course<90:
                    a = course
                    Vx = speed*np.sin(np.radians(a))
                    Vy = -speed*np.cos(np.radians(a))
                elif 90<course<180:
                    a = course-90
                    Vx = speed*np.cos(np.radians(a))
                    Vy = speed*np.sin(np.radians(a))
                elif 180<course<270:
                    a = course-180
                    Vx = -speed*np.sin(np.radians(a))
                    Vy = speed*np.cos(np.radians(a))    
                elif 270<course<360:
                    a = course-270
                    Vx = -speed*np.cos(np.radians(a))
                    Vy = -speed*np.sin(np.radians(a)) 
                elif course==0:
                    Vx = 0
                    Vy = -speed
                elif course==90:
                    Vx = speed
                    Vy = 0
                elif course==180:
                    Vx = 0
                    Vy = speed
                elif course==270:
                    Vx = -speed
                    Vy = 0
                elif course==360:
                    Vx = 0
                    Vy = -speed
            Vxv.append(Vx)
            Vyv.append(Vy)
        return np.array(Vxv),np.array(Vyv)
    
       
    def binAverageVelocities(self,x,y,Vx,Vy,xg,yg,obsThresh):     
        xx = xg[0,:]
        yy = yg[:,0]
        dx = abs(xx[1]-xx[0])
        dy = abs(yy[1]-yy[0])
        
        # Do the bin-averaging #
        Vx_mean = np.zeros_like(xg).astype(float)*np.nan
        Vy_mean = np.zeros_like(xg).astype(float)*np.nan
        for ix in range(len(xx)-1):
            for iy in range(len(yy)-1):
                inBool = np.logical_and(np.logical_and(x>xx[ix],x<=xx[ix]+dx),
                                        np.logical_and(y>yy[iy],y<=yy[iy]+dy))
                if len(~np.isnan(Vx[inBool]))>=obsThresh:
                    Vx_mean[iy,ix] = np.nanmean(Vx[inBool]) # Make sure to put the value in the correct position in the array - indexing goes top to bottom but coordinates go bottom to top #
                    Vy_mean[iy,ix] = np.nanmean(Vy[inBool])
                else:
                    Vx_mean[iy,ix] = np.nan 
                    Vy_mean[iy,ix] = np.nan
                    
                
        # Vx_mean[np.isnan(Vx_mean)] = 0
        # Vy_mean[np.isnan(Vy_mean)] = 0
        return Vx_mean,Vy_mean
        
       
    def loadAndFormatDrifterData(self,individual_or_combined):                  
        if individual_or_combined == 'combined':
            file = 'combined_drifter.mat'
        elif individual_or_combined == 'individual':
            file = 'drifter_asilomar_13Jul22_with_deployed_index_forPython.mat'
        else:
            raise('Argument should be either combined or individual as a string.')
            
        direc = os.path.realpath('../../../Data/DrifterData/'+file)
        
        # Load the data and get local coordinates #
        if individual_or_combined == 'combined':
            # Load the combined drifter positions #
            data = mat73.loadmat(direc)
            # Add new entries for local x and y coordinates #
            data['xLocal'],data['yLocal'],self.angleref = self.ll2local(data['lon'],data['lat'])
            # Remove anomolous data points that appear over land using the land mask #
            with open('/home/server/pi/homes/conlinm/Documents/ROXSI/'+
                      'Analysis/ArgusWAMFlow/Results/LandMasks_ROI.pkl','rb') as f:
                masks = pickle.load(f)           
            for mask in masks:
                xyl = np.array([data['xLocal'],data['yLocal']]).T
                pth = path.Path(mask)
                inp = pth.contains_points(xyl) # Get drifter points within the mask #
                for key in data:
                    data[key] = data[key][~inp] # Take only data points NOT within the mask #     
        else:
            # Load the individual drifter tracks #
            data = scipy.io.loadmat(direc)   
            data = data['drift2'][0][0]
            # Replace the lat/lon columns (1 and 2) with local x and y coordinates
            for drifter in range(len(data)):
                xx,yy,self.angleref = self.ll2local(data[drifter][:,2],data[drifter][:,1])
                data[drifter][:,1] = xx
                data[drifter][:,2] = yy
                # Filter to only include observations when deployed #
                data[drifter] = data[drifter][data[drifter][:,8]==1,:]
                # Filter to only include observations with GPS fix (quality indicator = 1 or 2)
                data[drifter] = data[drifter][(data[drifter][:,6]==1) | (data[drifter][:,6]==2) ,:]
                # Remove anomolous data points that appear over land using the land mask #
                with open('/home/server/pi/homes/conlinm/Documents/ROXSI/'+
                          'Analysis/ArgusWAMFlow/Results/LandMasks_ROI.pkl','rb') as f:
                    masks = pickle.load(f)           
                for mask in masks:
                    pth = path.Path(mask)
                    xyl = np.array([data[drifter][:,1],data[drifter][:,2]]).T
                    inp = pth.contains_points(xyl)
                    data[drifter] = data[drifter][~inp,:]
                # Crop to only within deployment time #
                inTime = np.logical_and(data[drifter][:,0]>=self.timeBoundsMat[0],
                                         data[drifter][:,0]<=self.timeBoundsMat[1])
                data[drifter] = data[drifter][inTime,:]
            # Combine indiviudal tracks into combined format #
            data_combined = {'xLocal':np.empty([0,1]),
                             'yLocal':np.empty([0,1]),
                             'speed':np.empty([0,1]),
                             'course':np.empty([0,1])}
            for drifter in range(len(data)):
                data_combined['xLocal'] = np.vstack([data_combined['xLocal'],
                                                  data[drifter][:,1].reshape(-1,1)])
                data_combined['yLocal'] = np.vstack([data_combined['yLocal'],
                                                  data[drifter][:,2].reshape(-1,1)])
                data_combined['speed'] = np.vstack([data_combined['speed'],
                                                  data[drifter][:,3].reshape(-1,1)])
                data_combined['course'] = np.vstack([data_combined['course'],
                                                  data[drifter][:,4].reshape(-1,1)])
            for key in data_combined:
                data_combined[key] = data_combined[key].reshape(-1)
            data = data_combined
              
        return data
            
        
    def gridDrifterData(self,data,dx,dy,obsThresh):
        '''
        Grid the individual drifter tracks. Break the domain into a grid. For each grid cell,
        average together all the individual drifter observations that are within it. If there are none,
        velocity=0.
        '''
        # Establish things #
        xx = np.arange(-60,50+dx,dx)
        yy = np.arange(0,250+dy,dy)
        xg,yg = np.meshgrid(xx,yy)
        
        # Calculate local Vx, Vy #
        clocal = self.c2LocalAng(data['course'])
        Vx,Vy = self.SC2VxVy(data['speed'], clocal)
        
        Vx_mean,Vy_mean = self.binAverageVelocities(data['xLocal'],data['yLocal'],Vx,Vy,xg,yg,obsThresh)
        
        return xg,yg,Vx_mean,Vy_mean
                

    
    def plot_rawTracks(self,frm,xg,yg):
        # Load and format the individual drifter data #
        data = self.loadAndFormatDrifterData('individual')
        
        # Calculate local Vx and Vy #
        s = data['speed']
        c = data['course']
        clocal = self.c2LocalAng(c)
        Vx,Vy = self.SC2VxVy(s, clocal)
        
        # Make the plot #
        fig = plt.figure(figsize=(4.5,8))
        ax = plt.axes([0.1,0.4,0.8,0.55])
        
        extents = [np.min(xg),
                   np.max(xg),
                   np.min(yg),
                   np.max(yg)]
        ax.imshow(frm,extent=extents,interpolation='bilinear',cmap='gray')
        
        h = ax.scatter(data['xLocal'],data['yLocal'],0.2,-Vy,vmin=-0.5,vmax=0.5,cmap='seismic',marker='s')
        ax.set_xlim(np.min(xg),np.max(xg))    
        ax.set_ylim(np.min(yg),np.max(yg))  
        axpos = ax.get_position()
        cbax = plt.axes([axpos.xmax+0.02,0.4,0.02,axpos.ymax-axpos.ymin])
        fig.colorbar(cm.ScalarMappable(norm=mc.Normalize(-0.5,0.5),cmap='seismic_r'),cbax,label=r'$\bar{V}$ (m/s)')    
        
        
    def plot_griddedTracks(self,frm,xg,yg,dx=1,dy=1):
        # Load and format the individual drifter data #
        data = self.loadAndFormatDrifterData('individual')
        
        # Bin average the data #
        dx = dx
        dy = dy
        gridx,gridy,Vx_mean,Vy_mean = self.gridDrifterData(data, dx=dx, dy=dy, obsThresh=0)
        
        x_plot = gridx[0,0:len(gridx[0,:])-1]+(np.diff(gridx[0,:])/2)
        y_plot = gridy[0:len(gridy[:,0])-1,0]+(np.diff(gridy[:,0])/2)
        x_plotg,y_plotg = np.meshgrid(x_plot,y_plot)
        Vx_mean = Vx_mean[0:-1,0:-1]
        Vy_mean = Vy_mean[0:-1,0:-1]
        Vx_mean[Vx_mean==0] = np.nan
        Vy_mean[Vy_mean==0] = np.nan
        
        
        # Make a streamplot #
        Vx_mean[np.isnan(Vx_mean)] = 0
        Vy_mean[np.isnan(Vy_mean)] = 0
        vis = plot.drawFlow_Streamplot_WithMeanFlows(Vx_mean,
                                                     -Vy_mean,
                                                     x_plotg,y_plotg,
                                                     frm,
                                                     yBound=0.5,minLength=0,
                                                     winSize=2, density=1.5)
    
    
    def qualitative(self,flowX,flowY,frm,xg,yg,dx=1,dy=1):
        # Plot streamplots from WAMFlow and drifters side by side. Had to get a
        # bit creative to do this because of how I coded the figure creation,
        # so to do this each stramplot is made, saved as an image, and then
        # both images are loaded and plotted side by side and then removed from
        # the file system. #
        wamflow = plot.drawFlow_Streamplot_WithMeanFlows(flowX,flowY,
                                                     xg,yg,frm,
                                                     yBound=0.5,minLength=2,winSize=2,density=2)
        plt.savefig('../Results/Figures/temp_drifterQualitativeComp_wamflow.png',dpi=450)
        plt.close('all')
        
        drifters = self.plot_griddedTracks(frm,xg,yg,dx=dx,dy=dy)
        plt.savefig('../Results/Figures/temp_drifterQualitativeComp_drifters.png',dpi=450)
        plt.close('all')   
        
        fig = plt.figure(figsize=(10,12))
        ax0 = plt.axes([0,0,0.5,1])
        ax1 = plt.axes([0.5,0,0.5,1])
        ax0.imshow(plt.imread('../Results/Figures/temp_drifterQualitativeComp_wamflow.png'))
        ax0.text(1000,150,'WAMFlow',ha='center',va='bottom',fontsize=10)
        # ax0.text(1325,275,'(a)',color='w',fontweight='bold',fontsize=10)
        ax0.axis('off')
        ax1.imshow(plt.imread('../Results/Figures/temp_drifterQualitativeComp_drifters.png'))
        ax1.text(1000,150,'Drifters',ha='center',va='bottom',fontsize=10)
        # ax1.text(1325,275,'(b)',color='w',fontweight='bold',fontsize=10)        
        ax1.axis('off')  
        
        os.remove('../Results/Figures/temp_drifterQualitativeComp_wamflow.png')
        os.remove('../Results/Figures/temp_drifterQualitativeComp_drifters.png')
        
    
    def quantitative(self,flowX,flowY,frm,xg,yg,dx=2,dy=2,drifterObsThresh=0):
        # Load and format the individual drifter data #
        data = self.loadAndFormatDrifterData('individual')
        
        # Bin average the data #
        dx = dx
        dy = dy
        gridx,gridy,Vx_mean_drifter,Vy_mean_drifter = self.gridDrifterData(data, dx=dx, dy=dy, obsThresh=drifterObsThresh)
        Vy_mean_drifter = -Vy_mean_drifter
        # Vx_mean_drifter = Vx_mean_drifter[0:-1,0:-1]
        # Vy_mean_drifter = Vy_mean_drifter[0:-1,0:-1]
        
        # Bin average the optical data to the same grid #
        if np.shape(gridx)==np.shape(xg): # If drifter grid is the same as the WAM grid #
            Vx_mean_wamflow = copy.deepcopy(flowX)
            Vy_mean_wamflow = copy.deepcopy(flowY)
        else:
            Vx_mean_wamflow,Vy_mean_wamflow = self.binAverageVelocities(xg.flatten(),yg.flatten(),flowX.flatten(),flowY.flatten(),gridx,gridy,obsThresh=0)
            x_plot = gridx[0,0:len(gridx[0,:])-1]+(np.diff(gridx[0,:])/2)
            y_plot = gridy[0:len(gridy[:,0])-1,0]+(np.diff(gridy[:,0])/2)
            x_plotg,y_plotg = np.meshgrid(x_plot,y_plot)
            Vx_mean_wamflow = Vx_mean_wamflow[0:-1,0:-1]
            Vy_mean_wamflow = Vy_mean_wamflow[0:-1,0:-1]

        Vx_mean_wamflow[abs(Vx_mean_wamflow)==0] = np.nan
        Vy_mean_wamflow[abs(Vy_mean_wamflow)==0] = np.nan
        
        # Make comparison plots #
        fig = plt.figure(figsize=(6.5,2))
        ax0 = plt.axes([0.12,0.25,0.2,0.7])
        ax1 = plt.axes([0.42,0.25,0.2,0.7])
        ax2 = plt.axes([0.72,0.25,0.2,0.7])
        ax = [ax0,ax1,ax2]
        ax[0].plot(Vx_mean_drifter,Vx_mean_wamflow,'k.',markersize=0.5)
        ax[0].plot(ax[0].get_xlim(),ax[0].get_xlim(),'r--',linewidth=1)
        # ax[0].axis('equal')
        ax[0].set_xlim(-1,1)
        ax[0].set_ylim(-0.4,0.4)
        ax[0].set_xlabel('Drifter V (m/s)')
        ax[0].set_ylabel('WAMFlow V (m/s)')
        ax[1].plot(-Vy_mean_drifter,-Vy_mean_wamflow,'k.',markersize=0.5)
        ax[1].plot(ax[1].get_xlim(),ax[1].get_xlim(),'r--',linewidth=1)
        # ax[1].axis('equal')
        ax[1].set_xlim(-1,1)
        ax[1].set_ylim(-0.4,0.4)
        ax[1].set_xlabel('Drifter U (m/s)')
        ax[1].set_ylabel('WAMFlow U (m/s)')
        ax[2].plot(np.sqrt(np.square(Vy_mean_drifter)+np.square(Vx_mean_drifter)),
                   np.sqrt(np.square(Vy_mean_wamflow)+np.square(Vx_mean_wamflow)),
                   'k.',markersize=0.5)
        ax[2].plot(ax[2].get_xlim(),ax[2].get_xlim(),'r--',linewidth=1)
        # ax[2].axis('equal')
        ax[2].set_xlim(0,1.5)
        ax[2].set_ylim(0,0.4)
        ax[2].set_xlabel('Drifter S (m/s)')
        ax[2].set_ylabel('WAMFlow S (m/s)')         
        
        # Calculate comparison statistics #     
        def rmse(y,x):
            rmse = np.sqrt(np.nansum(np.square((y-x)))/np.size(x[np.logical_and(~np.isnan(y),~np.isnan(x))]))
            return rmse
        def meanE(y,x):
            meanE = np.nanmean(y-x)
            return meanE
        
        rmse_u = rmse(Vx_mean_wamflow,Vx_mean_drifter)
        rmse_v = rmse(-Vy_mean_wamflow,-Vy_mean_drifter)
        rmse_s = rmse(np.sqrt(np.square(Vy_mean_wamflow)+np.square(Vx_mean_wamflow)),np.sqrt(np.square(Vy_mean_drifter)+np.square(Vx_mean_drifter)))

        meanE_u = meanE(Vx_mean_wamflow,Vx_mean_drifter)
        meanE_v = meanE(-Vy_mean_wamflow,-Vy_mean_drifter)
        meanE_s = meanE(np.sqrt(np.square(-Vy_mean_wamflow)+np.square(Vx_mean_wamflow)),np.sqrt(np.square(-Vy_mean_drifter)+np.square(Vx_mean_drifter)))

        return [rmse_u,rmse_v,rmse_s],[meanE_u,meanE_v,meanE_s]
    

class ADCP():

    def __init__(self):
        pass   
    
    def getDataTimeBounds(self,vidDate,vidTime):
        '''
        Get the matlab-type datetimes when the collect started and ended
        to pull from the data 
        '''
        viddt = vidDate+vidTime
        d_start = datetime.datetime(int(viddt[0:4]),int(viddt[4:6]),int(viddt[6:8]),int(viddt[8:10]),int(viddt[10:12]))
        dtime_start = 366 + d_start.toordinal() + (d_start - datetime.datetime.fromordinal(d_start.toordinal())).total_seconds()/(24*60*60)
        d_end = datetime.datetime(int(viddt[0:4]),int(viddt[4:6]),int(viddt[6:8]),int(viddt[8:10]),int(viddt[10:12])+25)
        dtime_end = 366 + d_end.toordinal() + (d_end - datetime.datetime.fromordinal(d_end.toordinal())).total_seconds()/(24*60*60)
        return dtime_start,dtime_end
      
    def getSignatureSegment(self,vidDate):
        '''
        Signature data is broken into different files covering time segments. Get the correct segment
        file to load based on video date
        '''
        segs = np.hstack([np.arange(620,632,2),np.arange(702,728,2)])
        date = int(vidDate[4:len(vidDate)])
        if date>=segs[0] and date<=segs[-1]:
            seg = max(np.where(segs<=date)[0])
            seg=int(seg+1) # To correct for python 0 indexing #
            if seg<10:
                s='0'
            else:
                s=''
            seg = s+str(seg)
        else:
            seg = np.nan
            
        return seg
    
    def loadAndFormatData(self,instrument,vidDate,vidTime):
        if instrument == 'x11':
            seg = self.getSignatureSegment(vidDate)
            file = 'roxsi_signature_L1_X11_101941_seg'+seg+'_forPython.mat'               
        elif instrument == 'x13':
            file = 'roxsi_aquadopp_L1_X13_9945_forPython.mat'
        
        direc = os.path.realpath('../../../Data/'+file)
    
        data = scipy.io.loadmat(direc)   
        if instrument == 'x11':
            data = data['x11_sig_L1']
        elif instrument == 'x13':
            data = data['x13_dop_L1']
               
        data2 = dict()
        data2['X'] = float(data['X'][0][0])
        data2['Y'] = float(data['Y'][0][0])
        data2['dtime'] = data['dtime'][0][0].reshape(-1)
        data2['U'] = data['u'][0][0]
        data2['V'] = data['v'][0][0]
        data2['W'] = data['w'][0][0]
        data2['pressure'] = data['pressure'][0][0].reshape(-1)
        data2['transducerHAB'] = float(data['transducerHAB'][0][0])
        if instrument == 'x11':
            data2['fs'] = int(data['samplingrateHz'][0][0])
            data2['zhab'] = data['zhab'][0][0].T
        elif instrument == 'x13':
            data2['fs'] = 1
            data2['zhab'] = data['zhab'][0][0]
                  
        data = data2
        
        # Crop to video time #
        dtime_start,dtime_end = self.getDataTimeBounds(vidDate,vidTime)
        inTimeBool = np.logical_and(data['dtime']>=dtime_start,data['dtime']<=dtime_end)
        for key in data:
            if np.size(data[key])<100:
                pass
            elif len(np.shape(data[key]))==2:
                data[key] = data[key][:,inTimeBool]
            else:
                data[key] = data[key][inTimeBool]
                        
        return data
           
    def reconstructSurface(self,data,recon_type):
        '''
        Surface reconstruction from pressure data. Use the package from Mika
        Malila to do the reconstructions, which implements both linear and
        non-linear reconstruction approaches.
        
        args:
            data: (dict) The adcp data dictionary
            recon_type: (str) One of either linear,linear_Krms, or nonlinear.
        
        returns:
            z: The reconstructed surface at each time 
        '''
        press = pd.Series(data['pressure'])
        z_hs = np.array(tf.z_hydrostatic(press)) # hydrostatic pressure head from pressure measurements #
        trf = tf.TRF(data['fs'],data['transducerHAB'])
        if recon_type=='linear':
            z = trf.p2z_lin(z_hs) # Linear transfer function reconstruction #
        elif recon_type=='linear_Krms' or recon_type=='nonlinear':
            z_lin_krms,z_nlin_krms = trf.p2eta_krms(z_hs,h0=np.mean(z_hs),return_nl=True) # Nonlinear transfer function reconstruction using Krms computed following Martins et al. (2021)
            if recon_type=='linear_Krms':
                z = z_lin_krms+np.mean(z_hs)
            elif recon_type=='nonlinear':
                z = z_nlin_krms+np.mean(z_hs)
        return z
    
    def plot_VerticalProfiles(self,vidDate,vidTime):                
        # Load the data #
        data = self.loadAndFormatData('x11',vidDate,vidTime)
        
        # Make a movie of u,v vs. z to examine depth dependence #
        os.mkdir('/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Results/Figures/ADCPCurrentProfiles_202207131030')
        z = data['zhab']
        for i in range(len(data['dtime'])):
            u = data['U'][:,i]
            v = data['V'][:,i]
            
            fig,ax = plt.subplots(1,2,sharey=True,visible=False)
            ax[0].plot(u,z,linewidth=2)
            ax[0].set_xlabel('U (m/s)')
            ax[0].set_ylabel('Z above bed (m)')
            ax[0].set_ylim(0,2.72)
            ax[0].set_xlim(-1,1)
            ax[0].plot((0,0),ax[0].get_ylim(),'k')
            ax[1].plot(v,z,linewidth=2)
            ax[1].set_xlabel('V (m/s)')
            ax[1].set_ylabel('Z above bed (m)')
            ax[1].set_xlim(-1,1)
            ax[1].plot((0,0),ax[1].get_ylim(),'k')
            if i<10:
                s = '000'
            elif i<100:
                s='00'
            elif i<1000:
                s = '0'
            else:
                s=''
            plt.savefig('/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Results/Figures/ADCPCurrentProfiles_202207131030/f'+s+str(i)+'.png')
            plt.close('all')
        
    def plot_SurfaceReconstructionComp(self,vidDate,vidTime):
        # Load the adcp data #
        data = self.loadAndFormatData('x11',vidDate,vidTime)
        
        # Surface reconstructions #
        z_lin = self.reconstructSurface(data,'linear')
        z_lin_krms = self.reconstructSurface(data,'linear_Krms')
        z_nlin_krms = self.reconstructSurface(data,'nonlinear')


        fig,ax = plt.subplots(2,1,sharex=True)
        ax[0].plot((data['dtime']-data['dtime'][0])*24*60,z_lin,'k',label='Linear')
        ax[0].plot((data['dtime']-data['dtime'][0])*24*60,z_lin_krms,'b',label='Linear (K_rms)')
        ax[0].plot((data['dtime']-data['dtime'][0])*24*60,z_nlin_krms,'c',label='Nonlinear (K_rms)')
        ax[0].legend(ncol=3,loc='upper center')
        ax[0].set_title('Reconstruction',fontweight='normal')
        ax[0].set_ylabel(r'$\eta$ (m)')
        ax[1].plot((data['dtime']-data['dtime'][0])*24*60,z_lin_krms-z_lin,'b')
        ax[1].plot((data['dtime']-data['dtime'][0])*24*60,z_nlin_krms-z_lin,'c')
        ax[1].plot(ax[1].get_xlim(),(0,0),'k--',linewidth=0.5)
        ax[1].set_title('Difference from Linear',fontweight='normal')
        ax[1].set_ylabel(r'$\Delta\eta$ (m)')
        ax[1].set_xlabel('Time (minutes)')
            
    def compute_MeanCurrents_Sigma(self,data,numLayers=21,surfaceLayers=3):
        
        if np.isnan(data['pressure']).any():
            U_SurfaceMean = np.nan
            V_SurfaceMean = np.nan        
        else:       
            # Reconstruct the surface #
            z = self.reconstructSurface(data,'nonlinear')
            
            # Transform to sigma coordinates #
            iTopBin = [max(np.where(data['zhab']<z[i])[0]) if z[i]>data['zhab'][0] else np.nan for i in range(len(z))]
            Deltaz = (z-(data['zhab'][0]).reshape(-1))/(numLayers-1)
            Deltaz[Deltaz<0] = np.nan
            
            U_sigma = np.empty([numLayers,len(data['dtime'])])
            V_sigma = np.empty([numLayers,len(data['dtime'])])
            for t in range(len(data['dtime'])):
                if ~np.isnan(iTopBin[t]):
                    zz = data['zhab'][0:iTopBin[t]+1]
                    zz_norm = (zz-min(zz))/(max(zz)-min(zz))
                    sigmas = [float(((j-1)*(Deltaz[t]))/(z[t]-zz[0])) for j in range(1,numLayers+1)]
                    
                    U_forInterp = data['U'][:,t][0:iTopBin[t]+1].reshape(-1)
                    V_forInterp = data['V'][:,t][0:iTopBin[t]+1].reshape(-1)
                    z_forInterp = zz_norm.reshape(-1)
                    
                    U_sigma1 = np.interp(sigmas,z_forInterp,U_forInterp)
                    V_sigma1 = np.interp(sigmas,z_forInterp,V_forInterp)
                else:
                    U_sigma1 = np.zeros([numLayers])*np.nan
                    V_sigma1 = np.zeros([numLayers])*np.nan
                
                U_sigma[:,t] = U_sigma1
                V_sigma[:,t] = V_sigma1
                
            U_SurfaceMean = np.nanmean(U_sigma[-surfaceLayers:numLayers,:])
            V_SurfaceMean = np.nanmean(V_sigma[-surfaceLayers:numLayers,:])
            
        return U_SurfaceMean,V_SurfaceMean
    
    
    def compute_MeanCurrents_DepthAverage(self,data):
        
        U_mean = []
        V_mean = []
        for t in range(len(data['dtime'])):
            U = data['U'][:,t]
            V = data['V'][:,t]
            
            U_mean1 = np.nanmean(U)
            V_mean1 = np.nanmean(V)
            
            U_mean.append(U_mean1)
            V_mean.append(V_mean1)
            
        U_SurfaceMean = np.nanmean(U_mean)
        V_SurfaceMean = np.nanmean(V_mean)
            
        return U_SurfaceMean,V_SurfaceMean
    
    
    def compute_MeanCurrents_TopBin(self,data):
        U_top = []
        V_top = []
        for t in range(len(data['dtime'])):
            U = data['U'][:,t]
            V = data['V'][:,t]
            
            if len(U[~np.isnan(U)])>0:
                U_top1 = U[~np.isnan(U)][-1]
                V_top1 = V[~np.isnan(V)][-1]
            else:
                U_top1 = np.nan
                V_top1 = np.nan
            
            U_top.append(U_top1)
            V_top.append(V_top1)
            
        U_SurfaceMean = np.nanmean(U_top)
        V_SurfaceMean = np.nanmean(V_top)
        
            
        return U_SurfaceMean,V_SurfaceMean  
    
    
    def quantitative(self,flowX,flowY,xg,yg,vidDate,vidTime,instrument,numLayers=21,surfaceLayers=3,windowSize=5):
        # Load the adcp data #
        data = self.loadAndFormatData(instrument,vidDate,vidTime)
        
        # Compute the mean currents #
        U_adcp_sigma,V_adcp_sigma = self.compute_MeanCurrents_Sigma(data,numLayers,surfaceLayers) 
        U_adcp_depthavg,V_adcp_depthavg = self.compute_MeanCurrents_DepthAverage(data) 
        U_adcp_topbin,V_adcp_topbin = self.compute_MeanCurrents_TopBin(data) 
        
        # Pull the WAMFlow estimates values from around the adcp position #
        x_adcp = data['Y']
        y_adcp = -data['X']
        
        dist = np.sqrt((xg-x_adcp)**2+(yg-y_adcp)**2)
        
        U_WAMFlow = -np.mean(flowY[dist<5])
        V_WAMFlow = np.mean(flowX[dist<5])
        
        adcp = {'Sigma':[V_adcp_sigma,U_adcp_sigma],
                'DepthAvg':[V_adcp_depthavg,U_adcp_depthavg],
                'TopBin':[V_adcp_topbin,U_adcp_topbin]} # Keep consistent with x and y directions rather than u and v directly #
        WAMFlow = [V_WAMFlow,U_WAMFlow]
        error = {'Sigma':[V_WAMFlow-V_adcp_sigma,U_WAMFlow-U_adcp_sigma],
                'DepthAvg':[V_WAMFlow-V_adcp_depthavg,U_WAMFlow-U_adcp_depthavg],
                'TopBin':[V_WAMFlow-V_adcp_topbin,U_WAMFlow-U_adcp_topbin]}
        
        return WAMFlow,adcp,error
        
        
        
        
        
        
    