#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:21:18 2023

@author: conlinm
"""

import numpy as np
from scipy.io import loadmat

def readMergedRectFrame(path):
    MergedRectFrame = loadmat(path)['MergedRectFrame']           
    frm = np.flipud(MergedRectFrame['mergedRectFrame'][0][0]) # Not sure why it is saved as a multi-nested array, but whatever #
    frm = frm[:,:,0] # It is a greyscale image, though the CIRN code saved it as 3 channels. All 3 channels are the same, so just take the first #

    xg = MergedRectFrame['frc1'][0][0][0]['xg'][0]
    yg = MergedRectFrame['frc1'][0][0][0]['yg'][0]
    
    return frm,xg,yg