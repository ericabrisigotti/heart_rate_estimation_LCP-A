#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import fftpack, stats, signal
from scipy.optimize import curve_fit
import scipy.linalg as la
import pywt


# In[2]:


def band_pass_filter( vect_all_freq , vect_power, down_freq_lim , up_freq_lim ):
    vect_mask = np.where( ( down_freq_lim <= vect_all_freq ) & ( vect_all_freq <= up_freq_lim ) )
    vect_freq = vect_all_freq [ vect_mask ]
    vect_peak_freq = vect_freq [ vect_power [ vect_mask ].argmax() ]
    vect_err_peak_freq = ( vect_freq [ (vect_power [ vect_mask ]).argmax() + 1 ] - vect_freq [ (vect_power [ vect_mask ]).argmax() - 1 ] ) / 2
    return vect_mask , vect_freq , vect_peak_freq , vect_err_peak_freq


# In[3]:


def filtered_power( vect_fft , vect_all_freq , down_freq_lim , up_freq_lim ):
    vect_filtered_fft = vect_fft.copy()
    vect_filtered_fft[ np.abs(vect_all_freq) >= up_freq_lim/60 ] = 0
    vect_filtered_fft[ np.abs(vect_all_freq) <= down_freq_lim/60 ] = 0
    vect_filtered = ( fftpack.ifft( vect_filtered_fft ) ).real
    return vect_filtered


# In[4]:


def dwt(time, vect, level, coefa, coefb):
    i = level+1
    coef = pywt.wavedec(vect , 'rbio3.9', level=i )

    for k in range (i):
        if (k != coefa)&(k != coefb): 
            coef[k] = np.zeros(coef[k].shape)
    dwt_sig = pywt.waverec( coef, 'rbio3.9' )
    fig9 , ax9 = plt.subplots( nrows = 1 , ncols = 1 , figsize = ( 15 , 5 ) )
    ax9.plot(time, dwt_sig[:-1])
    ax9.set_xlabel ("Time [s]", fontsize=16)
    ax9.set_ylabel ("Magnitude freq [Hz]", fontsize=16)
    ax9.set_xlim(min(time),max(time))
    return dwt_sig


# In[5]:


def peaks(time, vect):
    peaks, _ = signal.find_peaks(vect, width = 20) 
    
    num_peak = len(peaks)
    t  = time.max() - time.min()
    bps = num_peak/t
    print('BPM is', bps*60)

    plt.figure( figsize = ( 15 , 4 ) )
    plt.plot(time,vect[:-1])
    plt.plot(time[peaks], vect[peaks], "x")
    plt.xlim(min(time),max(time))
    plt.xlabel ("Time [s]", fontsize=16)
    plt.ylabel ("Magnitude freq [Hz]", fontsize=16)
    return (peaks)


# In[6]:


def cwt(time, waves,size):
    peakind = signal.find_peaks_cwt(waves, size, window_size = 20)
    temp = np.array(time[peakind])
    fig,ax = plt.subplots( figsize=(15,7) )
    ax.plot(time,waves,'-')
    ax.plot(temp, waves[peakind],'*')
    plt.xlim(min(time),max(time))
    plt.xlabel ("Time [s]", fontsize=16)
    plt.ylabel ("Magnitude freq [Hz]", fontsize=16)
    print(len(peakind),' peaks were found')
    print(len(peakind)/(time.max() - time.min())*60, 'BPM')
    return temp

