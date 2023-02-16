#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:59:22 2023

@author: conlinm
"""

import cv2
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat


class analyze():
    
    def __init__(self,baseDir,dates):
        
        def getResultsFolders(baseDir,dates):   
            folders1 = sorted(os.listdir(baseDir))
            folders=[]
            for d in range(len(dates)):
                folders11 = [folders1[i] for i in range(len(folders1)) if dates[d] in folders1[i]]
                [folders.append(folders11[i]) for i in range(len(folders11))]
                
            return folders
        
        def getResultsDict(baseDir,resFolders):
            t = []
            ripVel = []
            feederVel = []
            adcp_x13 = []
            adcp_x11 = []
            e_x13 = []
            e_x11 = []
            for f in resFolders:
                t.append(datetime.datetime(int(f[0:4]),int(f[4:6]),int(f[6:8]),int(f[8:10]),int(f[10:12])))
                ripVel.append(np.load(baseDir+f+'/ripVel.npy'))
                feederVel.append(np.load(baseDir+f+'/feederVel.npy'))
                adcp_x13.append(np.load(baseDir+f+'/adcp_x13.npy',allow_pickle=True))
                adcp_x11.append(np.load(baseDir+f+'/adcp_x11.npy',allow_pickle=True))
                e_x13.append(np.load(baseDir+f+'/error_x13.npy',allow_pickle=True))
                e_x11.append(np.load(baseDir+f+'/error_x11.npy',allow_pickle=True))
            
            results = {'t':t,
                       'ripVel':ripVel,
                       'feederVel':feederVel,
                       'adcp_x13':adcp_x13,
                       'adcp_x11':adcp_x11,
                       'e_x13':e_x13,
                       'e_x11':e_x11}
            
            return results
        
        self.baseDir = baseDir
        self.dates = dates
        self.resFolders = getResultsFolders(self.baseDir,self.dates)
        self.results = getResultsDict(self.baseDir,self.resFolders)
        self.angleRef = 293
        
    
    def WAMFlowMovie(self,saveDir):
        
        for f in self.resFolders:
            im = cv2.imread(self.baseDir+f+'/vis.png')
            size = (np.shape(im)[1],np.shape(im)[0])
            
            if f==self.resFolders[0]:
                	out = cv2.VideoWriter(saveDir+'WAMFlowMovie.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)   
                    
            out.write(im)
            
        out.release()
        
    def ADCPValidation(self):
        breakpoint()
        
        v_x11_sigma_v = [self.results['adcp_x11'][i][()]['Sigma'][0] if type(self.results['adcp_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x11_sigma_u = [self.results['adcp_x11'][i][()]['Sigma'][1] if type(self.results['adcp_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x11_depthavg_v = [self.results['adcp_x11'][i][()]['DepthAvg'][0] if type(self.results['adcp_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x11_depthavg_u = [self.results['adcp_x11'][i][()]['DepthAvg'][1] if type(self.results['adcp_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x11_topbin_v = [self.results['adcp_x11'][i][()]['TopBin'][0] if type(self.results['adcp_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x11_topbin_u = [self.results['adcp_x11'][i][()]['TopBin'][1] if type(self.results['adcp_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x13_sigma_v = [self.results['adcp_x13'][i][()]['Sigma'][0] if type(self.results['adcp_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x13_sigma_u = [self.results['adcp_x13'][i][()]['Sigma'][1] if type(self.results['adcp_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x13_depthavg_v = [self.results['adcp_x13'][i][()]['DepthAvg'][0] if type(self.results['adcp_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x13_depthavg_u = [self.results['adcp_x13'][i][()]['DepthAvg'][1] if type(self.results['adcp_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x13_topbin_v = [self.results['adcp_x13'][i][()]['TopBin'][0] if type(self.results['adcp_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        v_x13_topbin_u = [self.results['adcp_x13'][i][()]['TopBin'][1] if type(self.results['adcp_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]

        
        e_x11_sigma_v = [self.results['e_x11'][i][()]['Sigma'][0] if type(self.results['e_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x11_sigma_u = [self.results['e_x11'][i][()]['Sigma'][1] if type(self.results['e_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x11_depthavg_v = [self.results['e_x11'][i][()]['DepthAvg'][0] if type(self.results['e_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x11_depthavg_u = [self.results['e_x11'][i][()]['DepthAvg'][1] if type(self.results['e_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x11_topbin_v = [self.results['e_x11'][i][()]['TopBin'][0] if type(self.results['e_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x11_topbin_u = [self.results['e_x11'][i][()]['TopBin'][1] if type(self.results['e_x11'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x13_sigma_v = [self.results['e_x13'][i][()]['Sigma'][0] if type(self.results['e_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x13_sigma_u = [self.results['e_x13'][i][()]['Sigma'][1] if type(self.results['e_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x13_depthavg_v = [self.results['e_x13'][i][()]['DepthAvg'][0] if type(self.results['e_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x13_depthavg_u = [self.results['e_x13'][i][()]['DepthAvg'][1] if type(self.results['e_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x13_topbin_v = [self.results['e_x13'][i][()]['TopBin'][0] if type(self.results['e_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        e_x13_topbin_u = [self.results['e_x13'][i][()]['TopBin'][1] if type(self.results['e_x13'][i][()]) is dict else np.nan for i in range(len(self.results['t']))]
        
        
        
        # Absolute difference #
        fig,ax = plt.subplots(2,2)
        # x11 #
        ax[0][0].plot(self.results['t'],e_x11_sigma_v,'b.-',zorder=5)
        ax[0][0].plot(self.results['t'],e_x11_depthavg_v,'r.-',zorder=5)
        ax[0][0].plot(self.results['t'],e_x11_topbin_v,'g.-',zorder=5)
        ax[0][0].plot(ax[0][0].get_xlim(),(0,0),'k--',zorder=1)
        ax[0][0].set_ylim(-0.3,0.3)
        ax[0][1].plot(self.results['t'],e_x11_sigma_u,'b.-',zorder=5)
        ax[0][1].plot(self.results['t'],e_x11_depthavg_u,'r.-',zorder=5)
        ax[0][1].plot(self.results['t'],e_x11_topbin_u,'g.-',zorder=5)
        ax[0][1].plot(ax[0][1].get_xlim(),(0,0),'k--',zorder=1)
        ax[0][1].set_ylim(-0.3,0.3)
        # x13 #
        ax[1][0].plot(self.results['t'],e_x13_sigma_v,'b.-',zorder=5)
        ax[1][0].plot(self.results['t'],e_x13_depthavg_v,'r.-',zorder=5)
        ax[1][0].plot(self.results['t'],e_x13_topbin_v,'g.-',zorder=5)
        ax[1][0].plot(ax[0][0].get_xlim(),(0,0),'k--',zorder=1)
        ax[1][0].set_ylim(-0.3,0.3)
        ax[1][1].plot(self.results['t'],e_x13_sigma_u,'b.-',zorder=5)
        ax[1][1].plot(self.results['t'],e_x13_depthavg_u,'r.-',zorder=5)
        ax[1][1].plot(self.results['t'],e_x13_topbin_u,'g.-',zorder=5)
        ax[1][1].plot(ax[0][1].get_xlim(),(0,0),'k--',zorder=1)
        ax[1][1].set_ylim(-0.3,0.3)

        # Difference normalized by ADCP observed velocity #
        fig,ax = plt.subplots(2,2)
        # x11 #
        ax[0][0].plot(self.results['t'],np.divide(e_x11_sigma_v,v_x11_sigma_v),'b.-',zorder=5)
        ax[0][0].plot(self.results['t'],np.divide(e_x11_depthavg_v,v_x11_depthavg_v),'r.-',zorder=5)
        ax[0][0].plot(self.results['t'],np.divide(e_x11_topbin_v,v_x11_topbin_v),'g.-',zorder=5)
        ax[0][0].plot(ax[0][0].get_xlim(),(0,0),'k--',zorder=1)
        ax[0][0].set_ylim(-2,2)
        ax[0][1].plot(self.results['t'],np.divide(e_x11_sigma_u,v_x11_sigma_u),'b.-',zorder=5)
        ax[0][1].plot(self.results['t'],np.divide(e_x11_depthavg_u,v_x11_depthavg_u),'r.-',zorder=5)
        ax[0][1].plot(self.results['t'],np.divide(e_x11_topbin_u,v_x11_topbin_u),'g.-',zorder=5)
        ax[0][1].plot(ax[0][1].get_xlim(),(0,0),'k--',zorder=1)
        ax[0][1].set_ylim(-2,2)
        # x13 #
        ax[1][0].plot(self.results['t'],np.divide(e_x13_sigma_v,v_x13_sigma_v),'b.-',zorder=5)
        ax[1][0].plot(self.results['t'],np.divide(e_x13_depthavg_v,v_x13_depthavg_v),'r.-',zorder=5)
        ax[1][0].plot(self.results['t'],np.divide(e_x13_topbin_v,v_x13_topbin_v),'g.-',zorder=5)
        ax[1][0].plot(ax[0][0].get_xlim(),(0,0),'k--',zorder=1)
        ax[1][0].set_ylim(-2,2)
        ax[1][1].plot(self.results['t'],np.divide(e_x13_sigma_u,v_x13_sigma_u),'b.-',zorder=5)
        ax[1][1].plot(self.results['t'],np.divide(e_x13_depthavg_u,v_x13_depthavg_u),'r.-',zorder=5)
        ax[1][1].plot(self.results['t'],np.divide(e_x13_topbin_u,v_x13_topbin_u),'g.-',zorder=5)
        ax[1][1].plot(ax[0][1].get_xlim(),(0,0),'k--',zorder=1)
        ax[1][1].set_ylim(-2,2)

        
    
    def getHydroData(self):
        # Waves #
        x01_mat = '/home/server/pi/homes/conlinm/Documents/ROXSI/Data/X01_spot_forPython.mat'
        x01 = loadmat(x01_mat)['x01']
        
        # wl #
        product = 'water_level&application=CEOAS'       
        api = 'https://tidesandcurrents.noaa.gov/api/datagetter?product='+product+'&begin_date='
        station = 9413450
        
        H = []
        T = []
        D = []
        wl = []
        for f in self.resFolders:            
            # Get the time the WAM started in YYYYMMDDhhmm format #
            dateStart = f
            dateEnd = str(int(f)+25)
            # Convert YYYYMMDDhhmm format to Matlab datetime. Second line below from SO: https://stackoverflow.com/questions/32991934/equivalent-function-of-datenumdatestring-of-matlab-in-python (answer of T.C. Helsloot) #
            dStart = datetime.datetime(int(dateStart[0:4]),int(dateStart[4:6]),int(dateStart[6:8]),int(dateStart[8:10]),int(dateStart[10:12]))
            dtime_start = 366 + dStart.toordinal() + (dStart - datetime.datetime.fromordinal(dStart.toordinal())).total_seconds()/(24*60*60)
            dEnd = datetime.datetime(int(dateEnd[0:4]),int(dateEnd[4:6]),int(dateEnd[6:8]),int(dateEnd[8:10]),int(dateEnd[10:12]))
            dtime_end = 366 + dEnd.toordinal() + (dEnd - datetime.datetime.fromordinal(dEnd.toordinal())).total_seconds()/(24*60*60)           
            
            # Buoy measures hourly. Interpolate values to the start and end of the collect, and take the averages as a representative value.
            H.append(np.mean([np.interp(dtime_start,x01[:,0],x01[:,1]),np.interp(dtime_end,x01[:,0],x01[:,1])]))  # Hs #
            T.append(np.mean([np.interp(dtime_start,x01[:,0],x01[:,3]),np.interp(dtime_end,x01[:,0],x01[:,3])])) # T mean #
            D.append(np.mean([np.interp(dtime_start,x01[:,0],x01[:,6]),np.interp(dtime_end,x01[:,0],x01[:,6])])-self.angleRef) # D mean #
            
            if f==self.resFolders[0]:
                # Get the water level record and take the average height over the 25 minutes %
                url = api+dateStart[0:-4]+'&range=750&datum=NAVD&station='+str(station)+'&time_zone=lst_ldt&units=metric&format=csv'
                wlts = pd.read_csv(url)
                wl_dates = np.array([int(np.array(wlts['Date Time'])[i].replace('-','').replace(' ','').replace(':','')) for i in range(len(wlts))])
            iStart = max(np.where(wl_dates<int(dateStart))[0])+1
            iEnd = min(np.where(wl_dates>int(dateEnd))[0])-1
            wl.append(np.mean(wlts[' Water Level'][iStart:iEnd]))
            

        
        hydro = {'Hs':H,
                 'Tm':T,
                 'Dm':D,
                 'n':wl}
        
        return hydro
    
    
       
    def forcingResponse(self):
        breakpoint()
        hydro = self.getHydroData()
                
        fig,ax = plt.subplots(2,sharex=True)
        axr = ax[1].twinx()
        ax[0].plot(self.results['t'],self.results['ripVel'],'k.-',label='Rip (xshore)')
        ax[0].plot(self.results['t'],self.results['feederVel'],'.-',color='gray',label='Feeder (longshore)')
        ax[0].set_ylabel('Velocity (m/s)')
        ax[0].legend(loc='lower right')
        ax[1].plot(self.results['t'],hydro['Hs'],'b.-',label=r'$H_s$')
        axr.plot(self.results['t'],hydro['Tm'],'r.-',label=r'$T_m$')
        axr.plot(self.results['t'],hydro['Dm'],'m.-',label=r'$D_m$')
        ax[1].plot(self.results['t'],hydro['n'],'c.-',label=r'$\eta$')
        ax[1].legend(loc='lower left')
        axr.legend(loc='lower right')
        ax[1].set_ylabel(r"$H_s$ , $\eta$ (m)")
        axr.set_ylabel(r"$T_m$ (s) , $D_m$ (deg)")
        myFmt = mdates.DateFormatter('%m-%d %H:00')
        ax[1].xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate()
        
        fig,ax = plt.subplots(1,3,sharey=True)
        ax[0].plot(hydro['n'],self.results['ripVel'],'k.')
        ax[0].set_xlabel(r"$\eta$ (m)")
        ax[0].set_ylabel('Rip velocity (m/s)')
        ax[1].plot(hydro['Dm'],self.results['ripVel'],'k.')
        ax[1].set_xlabel(r'$D_m$ (deg)')
        ax[2].plot((1/8)*1025*9.81*np.array(hydro['Hs'])**2*9.81*np.array(hydro['Tm'])/(2*np.pi)/1000,self.results['ripVel'],'k.')
        ax[2].set_xlabel(r'$EC_g$ (kW/m)')
        
        fig,ax = plt.subplots(1,3,sharey=True)
        ax[0].plot(hydro['n'],self.results['feederVel'],'k.')
        ax[0].set_xlabel(r"$\eta$ (m)")
        ax[0].set_ylabel('Feeder velocity (m/s)')
        ax[1].plot(hydro['Dm'],self.results['feederVel'],'k.')
        ax[1].set_xlabel(r'$D_m$ (deg)')
        ax[2].plot((1/8)*1025*9.81*np.array(hydro['Hs'])**2*9.81*np.array(hydro['Tm'])/(2*np.pi)/1000,self.results['feederVel'],'k.')
        ax[2].set_xlabel(r'$EC_g$ (kW/m)')
    

    



        
    
    
    