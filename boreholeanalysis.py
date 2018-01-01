#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from obspy.core import Trace, Stream
from obspy.core import UTCDateTime
from obspy.core import read
from obspy.signal.invsim import cosTaper
from obspy.signal.cross_correlation import xcorr
import os
from obspy import signal
from numpy import float64
import math
from matplotlib.ticker import ScalarFormatter
from cmath import exp
from transformfunc import transformfunc

#Spectral ratio function (This is called later in the script)===================================
def specrat(comp,stat_top,stat_bot,eq_delay,winlen):
        filepath_top=os.path.join('/Users/freddiejackson/Python/catalog5/sacdata/' + stat_top + comp) #define the filepaths of the time series (north east and vertical)
        filepath_bot=os.path.join('/Users/freddiejackson/Python/catalog5/sacdata/' + stat_bot + comp)


        st=read(filepath_top) 						#read input files and saves them as stream files
        top=st[0]         						#extracts and saves the first trace from the stream
        st=read(filepath_bot)
        bot=st[0]

        #======Processing==========
        top.detrend('simple') 						#removes simple linear trend
        bot.detrend('simple')
        
        top.detrend('demean') 						#removes mean
        bot.detrend('demean')

        top.filter('bandpass', freqmin=0.1, freqmax=20, corners=2, zerophase=True) 	#bandpass filter
        bot.filter('bandpass', freqmin=0.1, freqmax=20, corners=2, zerophase=True)

        #=======Window parameters==========
        t_start=top.stats.starttime	#extract start time of the trace
        t_end=top.stats.endtime           #extract end time
        trlen=t_end-t_start             #calculates length
        time=np.arange(0,top.stats.npts*top.stats.delta,top.stats.delta) #creates an array for the time axis

        eq_t=t_start+eq_delay				#variable for when the S-wave energy arrives

        top_tw1 = top.slice(eq_t, eq_t+winlen)		#slice the traces into a short time window
        bot_tw1 = bot.slice(eq_t, eq_t+winlen)
       
        top_tw1.taper(type='cosine',max_percentage=0.1,side='both')   #taper the edges of the time window
        bot_tw1.taper(type='cosine',max_percentage=0.1,side='both')

        #======Spectral analysis=========
        top_fft=np.fft.fft(top_tw1.data) 			#Fast fourier transform of each trace
	bot_fft=np.fft.fft(bot_tw1.data)

	fftn=len(top_fft) 				#extract no of samples
        df=top.stats.sampling_rate                      #reads the sampling rate from the header info
	top_fft=top_fft[0:fftn/2] 		        #remove freqyencies above the nyquist and negatives
	bot_fft=bot_fft[0:fftn/2]

	freq_axis=np.linspace(0,df/2,fftn/2) 		#defines a frequency axis
	top_freq=abs(top_fft) 				#take amplitude of spectra
	bot_freq=abs(bot_fft)

	freq_axis = float64(freq_axis) 			#change precision
	top_freq = float64(top_freq)
	bot_freq = float64(bot_freq)

							#smooths the spectra
	smoothing_cons=40
	top_freq_smooth=signal.konnoohmachismoothing.konnoOhmachiSmoothing(top_freq, freq_axis, bandwidth=smoothing_cons, max_memory_usage=1024, normalize=False)
	bot_freq_smooth=signal.konnoohmachismoothing.konnoOhmachiSmoothing(bot_freq, freq_axis, bandwidth=smoothing_cons, max_memory_usage=1024, normalize=False)

        specrat=np.divide(top_freq_smooth,bot_freq_smooth)
        xmin=0.1
        xmax=15
        ymin=0
        ymax=120

	return time,top,freq_axis,specrat;	  #return the function outputs
#=============================================================================








#Cross Correlation ============================================================
def xcorrfunc(comp,stat_top,stat_bot,eq_delay,winlen,shift_len):
        filepath_top=os.path.join('/Users/freddiejackson/Python/catalog5/sacdata/' + stat_top + comp) #define the filepaths of the time series (north east and vertical)
        filepath_bot=os.path.join('/Users/freddiejackson/Python/catalog5/sacdata/' + stat_bot + comp)

        st=read(filepath_top) 		     	#read input files and saves them as stream files
        top=st[0]         		       #extracts and saves the first trace from the stream
        st=read(filepath_bot)
        bot=st[0]

        t_start=top.stats.starttime	#extract start time of the trace
        t_end=top.stats.endtime           #extract end time
        trlen=t_end-t_start             #calculates length
        time=np.arange(0,top.stats.npts*top.stats.delta,top.stats.delta) #creates an array for the time axis

        #======Processing==========
        top.detrend('simple') 						#removes simple linear trend
        bot.detrend('simple')
        
        top.detrend('demean') 						#removes mean
        bot.detrend('demean')

        top.filter('bandpass', freqmin=0.1, freqmax=20, corners=2, zerophase=True) 	#bandpass filter
        bot.filter('bandpass', freqmin=0.1, freqmax=20, corners=2, zerophase=True)

        bot_factor=abs(top.max()/bot.max())
        #bot=scale(bot,bot_factor)

        #====X corr=======
        xn=500
        xcorr_tr=xcorr(top,bot,xn,full_xcorr=True)
        xcorr_time=np.linspace(-xn,xn,xn*2+1)/top.stats.sampling_rate
        

        xmin=0.1
        xmax=15
        ymin=0
        ymax=60

        #====plot=====
        plt.figure(figsize=(6,10))
       
        ax=plt.subplot(311)
        xmin=292
        xmax=294
        plt.plot(time,top)
        plt.plot(time,bot,'r--')
        plt.grid(which='both')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (g)')
        major_ticks = np.arange(xmin, xmax+1, 1)
        minor_ticks = np.arange(xmin, xmax+1, 0.1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)          
        plt.xlim(xmin,xmax)
        plt.legend((stat_top,stat_bot), loc=3)

        ax.set_title('c)',x=0.05,y=0.85)
        
        ax=plt.subplot(312)
        xmin=-0.5
        xmax=0.5
        plt.plot(xcorr_time,xcorr_tr[2])
        plt.grid(which='both')
        major_ticks = np.arange(xmin, xmax+1, 0.1)
        minor_ticks = np.arange(xmin, xmax+1, 0.01)
        ax.set_xticks(major_ticks)
        #ax.set_xticks(minor_ticks, minor=True)
        plt.xlim(xmin,xmax)
        #ax.set_title('b) Correlation Function',x=0.2,y=0.85)
        plt.xlabel('Time (s)')
        plt.ylabel('Correlation Coefficient')

        ax.set_title('f)',x=0.05,y=0.85)
        
        time_shift=np.arange(0+shift_len,top.stats.npts*top.stats.delta+shift_len,top.stats.delta)
        caption=os.path.join('Shifted by ' + str(shift_len) +' seconds')

        ax=plt.subplot(313)
        xmin=292
        xmax=294
        plt.plot(time,top,'b-')
        plt.plot(time_shift,bot,'r--')
        plt.grid(which='both') 
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (g)')
        ax.text(0.5,0.1,caption,fontsize=12,horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
        major_ticks = np.arange(xmin, xmax+1, 1)
        minor_ticks = np.arange(xmin, xmax+1, 0.1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        plt.xlim(xmin,xmax)
        ax.set_title('i)',x=0.05,y=0.85)
        
        plt.savefig('foo.png', bbox_inches='tight')
        plt.close()

	return time,top,xcorr_tr;		#return the function outputs

#====================================================================================

#RUN XCOR (0) OR SPEC RAT? (1)
fish=1
comp='_HNE.SAC'
tw_len=40
eq_delay=291

if fish == 0: # RUN CROSS CORRELATION FUNCTION
        
        #Borehole 3
        stat_top='PMB04A4'
        stat_bot='PMB06B2'

        shift_len=0.19
        time1,top1,xcorr1=xcorrfunc(comp,stat_top,stat_bot,eq_delay,tw_len,shift_len)

if fish == 1: # RUN SPEC RAT FUNCTION        

        plt.figure(figsize=(10,2.3))
        ax=plt.subplot(111)

        #which borehole?
        Bh=1

        if Bh==1:
                #Borehole 1
                time1,top1,freq_axis1,specrat1=specrat(comp,'PMB10C2','PMB11C3',eq_delay,tw_len)
                time2,top2,freq_axis2,specrat2=specrat(comp,'PMB11C3','PMB12C4',eq_delay,tw_len)
                time3,top3,freq_axis3,specrat3=specrat(comp,'PMB10C2','PMB12C4',eq_delay,tw_len)
                leg1='-4m/9m'
                leg2='9m/41m'
                leg3='-4m/41m)'
                ax.set_title('a) Borehole 1',x=0.5,y=1.05)
        if Bh==2:
                #Borehole 2
                time1,top1,freq_axis1,specrat1=specrat(comp,'PMB02A2','PMB01A1',eq_delay,tw_len)
                time2,top2,freq_axis2,specrat2=specrat(comp,'PMB01A1','PMB03A3',eq_delay,tw_len)
                time3,top3,freq_axis3,specrat3=specrat(comp,'PMB02A2','PMB03A3',eq_delay,tw_len)
                leg1='2m/36m'
                leg2='36m/46m'
                leg3='2m/46m'
                ax.set_title('b) Borehole 2',x=0.5,y=1.05)
        if Bh==3:
                #Borehole 3
                time1,top1,freq_axis1,specrat1=specrat(comp,'PMB04A4','PMB05B1',eq_delay,tw_len)
                time2,top2,freq_axis2,specrat2=specrat(comp,'PMB05B1','PMB06B2',eq_delay,tw_len)
                time3,top3,freq_axis3,specrat3=specrat(comp,'PMB04A4','PMB06B2',eq_delay,tw_len)
                leg1='16m/33m'
                leg2='33m/57m'
                leg3='16m/57m'
                ax.set_title('c) Borehole 3',x=0.5,y=1.05)

        plt.semilogx(freq_axis1,specrat1,'r-.',linewidth=1.2)
        plt.semilogx(freq_axis1,specrat2,'g--',linewidth=1.2)
        plt.semilogx(freq_axis1,specrat3,'b-',linewidth=1.2)
        plt.xlim(0.5,15)
        plt.grid(which='both')
        plt.legend((leg1,leg2,leg3),loc='upper right',framealpha=0.5)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Spectral Ratio')
        plt.savefig('foo.png', bbox_inches='tight')
        plt.close()

