#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:13:33 2023

@author: conlinm
"""
import os

os.chdir('/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Tools')
from mpcWAMFlow import test



# Test how WAM pixel size influences results #
pixSizeTest = test.pixelSize()

# Test filter knobs #
filteringTestObj = test.filtering('20220713','1030',1)
r = [140,140,5]
c = [72,5,5]
for i in range(len(r)):
    # filteringTestObj.plot_timeseries(r[i],c[i])
    # filteringTestObj.plot_DLAScatter(r[i],c[i])
    means = filteringTestObj.secondaryVelocityFilter(r[i],c[i])
    print(means)


# Test different ADCP velocity estimates #
adcpVelTypeTest = test.adcpVelocityType(numLayers=11,surfaceLayers=3,windowSize=5)





