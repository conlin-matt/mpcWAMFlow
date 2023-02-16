#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:39:07 2023

@author: conlinm
"""

import os

os.chdir('/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Tools')
from mpcWAMFlow import analysis

dates = ['20220622','20220623','20220624','20220625','20220626','20220713']
baseDir = '/home/server/pi/homes/conlinm/Documents/ROXSI/Analysis/ArgusWAMFlow/Results/OpticalFlowResults/'

ao = analysis.analyze(baseDir,dates)
ao.forcingResponse()
