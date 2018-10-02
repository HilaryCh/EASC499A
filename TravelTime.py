import obspy
import trigger
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from obspy.signal.trigger import trigger_onset
from copy import copy


##############################################################################################################
##############################################################################################################
def v(z): 
	"""
	Given a depth in meter, return P wave velocity based on Tryggvason et al 2002 Model for SW Iceland.
	______
	:input: 
		- z: depth in meter (float)
	_______
	:output:
		- velocity in m/s (float) 
	"""
	if 0<= z <4400:                                 # 0 ~ 4400m: Zone 1 (exponential growth)
		return -3180.8316744597078 * np.exp(-0.0005059661974657385 * z) +6777.559256827945
	elif 4400<= z <6000:                            # 4400~6000m: Zone 2 (connecting zone 1&2: interpolation)       
		# Data for Zone 2 velocity extrapolation (Tryggvason et al 2002 Model)
		m_depth = np.array([0 , 2 ,4 ,6,9 ,12 ,16 ,21, 32])
		m_velocity = np.array([3.6,5.6,6.4,6.6,6.7,6.8,7.0,7.1,7.4])
		xp=m_depth[2:4] # Where interpolation is needed
		yp=m_velocity[2:4]
		#print(v(3000)) # Test	
		return 1000*np.interp(z/1000, xp,yp) 
	elif 6000<= z:                                  # 6000m up: Zone 3 (Deeper linear growth)
		return 0.030941704035874536*(z-5900)+ 6590

##############################################################################################################
##############################################################################################################

def Signal_Processing(st,hp=5.0,lp=49.0):
	"""
	Given a stream (obspy.core.stream), do signal processing procedure.
	______
	:input: 
		-  st: Stream (class:obspy.core.stream)
		- *hp: Corner frequency of high-pass filter
		- *lp: Corner frequency of low-pass filter
	_______
	:output:
		- st: Stream (class:obspy.core.stream)
	"""
	st.detrend(type='linear')
	st.taper(max_percentage=0.05, type='triang')
	st.filter('bandpass', freqmin=hp,freqmax=lp, corners=4, zerophase=True)
	return st
	
##############################################################################################################
##############################################################################################################


def tt_Correlate(stm, phase='P', trim_start=10, trim_end=100, 
							thr_on=.1, thr_off=.05, hp=5.0,lp=49.0, process=True,
							plot= True, plot_size=(40,100)):
	"""
	Given a stream (obspy.core.stream), calculate travel time based on correlation picker 
	in Massin & Malcolm(2016). 
	Use the trigger.py module in Massin & Malcolm(2016)
	Use the obspy.signal.trigger.trigger_onset module.
	! Assume the traces in stm have the same sampling rate.
	
	:input: 
		- stm			: Stream that contains waveform traces (class:obspy.core.stream)
		- *phase		: P arrival (pick the 1st onset) or S arrival (pick the 2nd onset) (options:'P','S')(string)
		- *trim_start	: For trimming the trace (seconds after start time) (float)
		- *trim_end		: For trimming the trace (seconds after start time) (float)
		- *thr_on		: Threshold of onset triggering (float)
		- *thr_off		: Threshold of onset closing (float)
		- *hp			: Corner frequency of high-pass filter (float)
		- *lp			: Corner frequency of low-pass filter (float)
		- *process		: Need signal processing or not (booline)
	_______
	:output:
		- A list of travel time time series for the traces in the stream (list) 
		
		
	"""
	# Signal processing before calculating Cf (default)
	if process==True:
		stm=Signal_Processing(stm,hp=hp, lp=lp)
	
	
	# Extract value from SAC attributes (extract form the very first trace in the stream)
	# Trace start time in UTM
	starttime= stm[0].stats.starttime
	# Sampling rate
	sampling_rate=stm[0].stats.sampling_rate
	# Event origin time
	to=stm[0].stats.sac['o']
	
	stm_trimed = stm.copy()
	
	# Trim the stream (mainly for getting rid of the false correlation peaks due to tapering)
	stm_trimed.trim(starttime=starttime+ trim_start, endtime=starttime+ trim_end)
	
	# Calculating characteristic function for all ENZ 3 channels
	data = stm_trimed
	data_preprocessed = (trigger.Components(data, preprocessor = 'rms')).output()
	cf_ENZ= trigger.Correlate(data_preprocessed, data, multiplexor = "components", preprocessor = 'rms').output() 

	
	# Plot the cf (if applicable)
	if plot == True:	
		cf_Correlate_c = trigger.Correlate(data_preprocessed, data, multiplexor = "components", preprocessor = 'rms')
		ax,shift =	trigger.stream_processor_plot( cf_Correlate_c.data, cf_Correlate_c.output(), 
														cfcolor='b', label=r'$^MC\star\rms$',size=plot_size)
		ax.legend()
	
	
	if phase == 'P' or phase == 'p':
		# Get Cf of Z channels only (most sensitive to P wave)
		cf_Z=[]
		for i in range(len(cf_ENZ)):
			if i%3 == 2:
				cf_Z.append(cf_ENZ[i])
		
		# Initialize travel time list tt
		Travel_Time_List= []
		
		# Calculation of travel time for traces in the stream 
		for i, tr in enumerate(cf_Z):
		
			try:
				# Get the very first triggered nt
				On1 = trigger_onset(cf_Z[i], thr_on, thr_off)[0][0]
				# Convert to travel time
				tt1  =    (On1/sampling_rate) +trim_start -to
			
			# Exception: If there's only noise, the picker won't be triggered:
			except:
				tt1 = None
			
			
			Travel_Time_List.append(tt1)
	
	if phase == 'S' or phase == 's':
		# Get Cf of E and N channels 
		cf_E	=[]
		cf_N	=[]
		for i in range(len(cf_ENZ)):
			if   i%3 == 0:
				cf_E.append(cf_ENZ[i])
			elif i%3 == 1:
				cf_N.append(cf_ENZ[i])	
		
		# Initialize travel time list tt
		Travel_Time_List	= []
		Travel_Time_List_E	= []
		Travel_Time_List_N	= []
		
		# Calculation of travel time for traces in the stream 
		for i, tr in enumerate(cf_E):
		
			try:
				# Get the second triggered nt
				On2 = trigger_onset(cf_E[i], thr_on, thr_off)[1][0]
				# Convert to travel time
				tt2  =    (On2/sampling_rate) +trim_start -to
			
			# Exception: If there's only noise, the picker won't be triggered:
			except:
				tt2 = None
			
			
			Travel_Time_List_E.append(tt2)
		
		print(Travel_Time_List_E)
		
		for i, tr in enumerate(cf_N):
		
			try:
				# Get the second triggered nt
				On2 = trigger_onset(cf_N[i], thr_on, thr_off)[1][0]
				# Convert to travel time
				tt2  =    (On2/sampling_rate) +trim_start -to
			
			# Exception: If there's only noise, the picker won't be triggered:
			except:
				tt2 = None
			
			
			Travel_Time_List_N.append(tt2)
		
		print(Travel_Time_List_N)
		
		# Choose the smaller one btw E and N channel
		for i, tt_EN in enumerate(Travel_Time_List_E):
		
			try:
				Travel_Time_List.append(min(Travel_Time_List_E[i],Travel_Time_List_N[i]))
			
			# Exception: No second triggered point in E or N channel
			except:
				if   Travel_Time_List_E:
					Travel_Time_List.append(Travel_Time_List_E[i])
				
				elif Travel_Time_List_N:
					Travel_Time_List.append(Travel_Time_List_N[i])
				
				else:
					Travel_Time_List.append(None)
	
	return Travel_Time_List
	
	'''	
	# Initialize travel time list tt
	Travel_Time_List= []
	# The time difference between the occurence of maxima and the onset threshold
	Time_Delay_List= []
	# New Cf that has been shifted from the occurence of maxima to triggered threshold.
	cf_shifted= []
	
	# Calculation of travel time for traces in the stream 
	for i, tr in enumerate(cf_Z):
		# Get the very first triggered nt
		On1 = trigger_onset(cf_Z[i], thr_on, thr_off)[0][0]
		# Get the occurence (nt) of maxima
		max_it = np.argmax(cf_Z[i])
		it_delay= max_it - On1
		
		# Convert to travel time
		tt1  =    (On1/sampling_rate) +trim_start -to
		
		# For to the time of the maxima: 
		max_t= (max_nt/sampling_rate) +trim_start -to
		
		# For calculating time delay between maxima and the onset threshold:
		t_delay = max_t - tt1
		
		Travel_Time_List.append(tt1)
		
		# For returning time delay list:
		Time_Delay_List.append(t_delay)
		
		# Shift the Cf from the occurence of maxima to triggered threshold
		if return_shifted_cf == True:
			cf_shifted = np.roll(cf_Z, -it_delay)
			
	if return_shifted_cf == True:
		return Travel_Time_List , cf_shifted , Time_Delay_List
	
	else:
		return Travel_Time_List  
	'''
	
##############################################################################################################
##############################################################################################################

def tt_SAC(stm, phase='P'):
	"""
	Given a stream (obspy.core.stream), calculate travel time based on SAC attributes. 
	______
	:input: 
		- stm_trimed: Stream (class:obspy.core.stream)
		- *phase: P or S wave
	_______
	:output:
		- travel time time series (list) (Z for P and E,N for S)
	"""
	# initialize
	tt=[]
	
	if phase =='P' or phase =='p':
		# Pull station info from all the stations in this event in the same stream
		for tr in stm:         
			if tr.stats.channel=='Z':   # Only extract location info from the Z channel so no repeated info
				
				try:
					to = tr.stats.sac['o'] 
					t2 = tr.stats.sac['t2'] 
					tt[len(tt):] = [t2-to]
				
				# Exception: no t0 information
				except:
					tt[len(tt):] = [None]
				
				
				
	elif phase =='S' or phase =='s':
		tt_E	=[]
		tt_N	=[]
		
		# Pull station info from all the stations in this event in the same stream
		for tr in stm:        
			
			if tr.stats.channel=='E':   # Only extract location info from the E channel so no repeated info
				try:	
					
					to = tr.stats.sac['o'] 
					t0 = tr.stats.sac['t0']
					tt_E[len(tt_E):] = [t0-to]
				
				# Exception: no t0 information
				except:
					tt_E[len(tt_E):] = [None]
				
			elif tr.stats.channel=='N':   # Only extract location info from the N channel so no repeated info
				try:
				
					to = tr.stats.sac['o'] 
					t0 = tr.stats.sac['t0']
					tt_N[len(tt_N):] = [t0-to]
					
				# Exception: no t0 information
				except:
					tt_N[len(tt_N):] = [None]
		
		# Choose the smaller one btw E and N channel		
		for i,tt_EN in enumerate(tt_E):
			try:
			
				tt.append(min(tt_E[i],tt_N[i]))
		
			# Exception: No second triggered point in E or N channel
			except:
				if   tt_E:
					tt.append(tt_E[i])
				
				elif tt_N:
					tt.append(tt_N[i])
				
				else:
					tt.append(None)	
				
			
	return tt	

##############################################################################################################
##############################################################################################################
def StTravelTimeModel(tr):
	"""
	Given a station, use a 2D finite difference scheme to calculate the travel time model of the intrested 
	region of that station. 
	______
	:input: 
		- tr: SAC trace that represent a station for extracting station location
	_______
	:output:
		- travel time model (3d array) 
	"""
	
	
	

##############################################################################################################	
##############################################################################################################	
	
	


##############################################################################################################
##############################################################################################################
