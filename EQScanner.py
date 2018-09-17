'''
'''
import obspy
import numpy as np
from copy import copy


# For importing event data
import sys 
sys.path.append("/Users/Hilary/Documents/Github/Py-NnK/scan")
import trigger

# Calculate the index of the occurence of maximum corherence 
from numpy import unravel_index

# For plotting 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.gridspec as gridspec

# For the progress counter in Source Scanning Procedure
import datetime

# For saving files
import os

# Convert degree to distance
from obspy.geodetics.base import calc_vincenty_inverse
import utm
from scipy.interpolate import interp2d

# For generating synthetic seismograms
from obspy import  Trace, Stream
from obspy.core.trace import Stats

# Sound alarm when finishing code
import winsound


# Converting factors
m_to_km = 0.001
x_to_easting = 100                   # From 'x' to 'Easting'
y_to_northing = 100                  # From 'y' to 'Northing'
km_to_100m = 10                      # From unit 'km'(ie.'z') to '100m'
                                     # (ie. the default z,x,y spacing in the map) 
x_to_km=0.1                          # From 'x' to km
y_to_km=0.1                          # From 'y' to km
easting_to_km=0.001                  # From 'Easting' to km
northing_to_km=0.001                 # From 'Northing' to km



##############################################################################################################





def spherical_to_cartesian(vector):
    """
    Convert the spherical coordinates [azimuth, polar angle
    radial distance] to Cartesian coordinates [x, y, z].
	This function is used in generating synthetic seismogram.

    ______________________________________________________________________
    :type vector : 3D array, list | np.array
    :param vector :  The spherical coordinate vector.

    :rtype : 3D array, np.array
    :return : The vector of cartesian coordinates.

    .. note::

        This file is extracted & modified from the program relax (Edward
            d'Auvergne).

    .. seealso::

        http://svn.gna.org/svn/relax/1.3/maths_fns/coord_transform.py
    ______________________________________________________________________
    """

    # Unit vector if r is missing
    if len(vector) == 2 :
        radius =1
    else:
        radius=vector[2]

    # Trig alias.
    sin_takeoff = np.sin(vector[1])

    # The vector.
    x = radius * sin_takeoff * np.cos(vector[0])
    y = radius * sin_takeoff * np.sin(vector[0])
    z = radius * np.cos(vector[1])

    return [x, y, z]
##############################################################################################################


def evenly_sampled_globe(r=1., n=1000):
	'''
	Generate a spherical grid that is evenly sampled.
	This function is not used in any of the function below right now.
	It's just a back-up in case I suddenly need it in sampling.
	'''
	'''
	# Plot the sampling grid
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	fig = plt.figure()
	ax = Axes3D(fig)
	ax.set_aspect('equal')
	points= EQScanner.evenly_sampled_globe(r=1., n=1000)
	pts_x=points[:,0]
	pts_y=points[:,1]
	pts_z=points[:,2]
	ax.scatter(pts_x,pts_y, pts_z,marker='.')
	plt.show()
	'''
	golden_angle = np.pi * (3 - np.sqrt(5))
	theta = golden_angle * np.arange(n)
	z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n) *r

	radius = np.sqrt(r**2 - z**2)

	grid_points = np.zeros((n, 3))
	grid_points [:,0] =  radius * np.cos(theta)
	grid_points [:,1] =  radius * np.sin(theta)
	grid_points [:,2] =  z

	return grid_points 


def st_locations(stm):
	'''
	Extract station lat/lon info to lists 
	______
	:input:
		-  stm: Stream (class:obspy.core.stream) 
	_______
	:output:
		- latitude (in degree) , longitude (in degree) , station name (string)
		
	'''
	# initialize

	stlo=[]                                 # Station location (longitude)
	stla=[]                                 # Station location (latitude)      
	st_name=[]                                # Station name

	# Only extract location info from the Z channel so no repeated info
	for tr in stm:            
		if tr.stats.channel=='Z':  
		
			stla[len(stla):] = [tr.stats.sac['stla']]  
			stlo[len(stlo):] = [tr.stats.sac['stlo']]
			st_name[len(stlo):] = [tr.stats['station'][:3]]


	return stla, stlo, st_name
	
	

##############################################################################################################
##############################################################################################################	

def single_ev_location(stm):
	'''
	Extract event lat/lon/depth info (assume traces in this folder are from the same 1 event) 
	______
	:input:
		-  stm: Stream (class:obspy.core.stream)
	_______
	:output:
		- (List): [latitude (in degree) , longitude (in degree) , depth]
		
	'''
	evla=stm[0].stats.sac['evla']
	evlo=stm[0].stats.sac['evlo']
	evdp=stm[0].stats.sac['evdp']

	ev_loc=[evla, evlo, evdp]

	return ev_loc
	

##############################################################################################################
##############################################################################################################	
	
def single_st_ev_map(stm, resolution='h', width=70000, height=50000, sca_r=0.1, sca_b=0.04):
	'''
	Input a stream of a single event. Show a map with the locations of stations and the event.
	______
	:input:
		-  stm: Stream (class:obspy.core.stream)
		-  resolution: Resolution of the map. 
			(options: 'c':crude, 'l':low, 'h': high, 'f':full)
		-  width: The width of the map
		-  height: The height of the map
		-  sca_r: Position of the scale bar from the right bdry
		-  sca_b: Position of the scale bar from the bottom
	_______
	:output:
		- (List): [latitude (in degree) , longitude (in degree) , depth]
		
	'''    
	
	from mpl_toolkits.basemap import Basemap

	stla, stlo, kstnm = st_locations(stm)
	evla, evlo, evdp = single_ev_location(stm)

	fig, ax = plt.subplots( figsize=(7,7))       # Adjust the size of the map
	latb=min(stla)#-0.05                         # Bottom of the map
	latt=max(stla)#+0.05                         # Top of the map
	latc=np.mean(stla)                           # Central lat of the map (for locating the centre of the map)
	lonl=min(stlo)#-0.05                         # Left bdry of the map
	lonr=max(stlo)#-0.05                         # Right bdry of the map
	lonc=np.mean(stlo)                           # Central lon of the map (for locating the centre of the map)
	res = resolution                             # Options: 'c','l','h','f'
	map_width=width                              # Adjust the area of the map
	map_height=height
	title=stm[0].stats.starttime

	# Set up basemap 
	m = Basemap(width=map_width, height=map_height,                      # adjust the map size
				resolution=res, projection='aea',                        # aea=Albers Equal Area Projection
				lat_1=latb, lat_2=latt, lon_0=lonc, lat_0=latc, ax=ax)   # adjust the span of lat/lon
				#lat_1=63.8, lat_2=64.1, lon_0=-22, lat_0=63.95, ax=ax)  # Or use this line to adjust 
																		 # the bdry manually

	m.drawcoastlines()
	m.drawcountries()
	m.fillcontinents(color='wheat', lake_color='skyblue')
	m.drawmapboundary(fill_color='skyblue')


	# Draw latitude and longititude lines

	m.drawparallels(np.linspace(latb, latt, 2), labels=[1, 0, 0, 1], fmt="%.2f", dashes=[2, 2])
	m.drawmeridians(np.linspace(lonl, lonr, 3), labels=[1, 0, 0, 1], fmt="%.2f", dashes=[2, 2])


	# Draw a map scale at lon,lat of length length representing distance in the map projection 
	# coordinates at lon0,lat0.

	m.drawmapscale(lonr-sca_r, latb-sca_b, lonc, latc, 10, barstyle='simple', units='km', fontsize=9, 
				   yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', fillcolor2='k',
				   ax=None, format='%d', zorder=None, linecolor=None, linewidth=None)

	ax.set_title(title)


	# attach the basemap object to the figure

	fig.bmap = m  


	# Plot station positions and names

	x, y = m(stlo, stla) 
	m.scatter(x, y, 100, color="r", marker="v", edgecolor="k", zorder=3)

	for i in range(len(kstnm)):
		plt.text(x[i], y[i], kstnm[i], va="top", family="monospace", weight="bold") # va:line-up at the top

		
	# Plot event location

	x, y = m(evlo, evla)  
	m.scatter(x, y, 100, color="w", marker="*", edgecolor="b", zorder=3)


	plt.show()


##############################################################################################################
##############################################################################################################

def signal_processing(stm, process=True, hp=5.0,lp=49.0, 
						trim= True, trim_start=20, trim_end=50, 
						plot=True, size=(800, 250)):
	"""
	Given a stream (obspy.core.stream) or trace, do signal processing procedure.
	_______
	:input: 
		-  stm: Stream (class:obspy.core.stream) or trace
		- *hp: Corner frequency of high-pass filter
		- *lp: Corner frequency of low-pass filter
	________
	:output:
		- stm_processed: Stream (class:obspy.core.stream) or trace
		
	"""
	# Keep the original stream intact
	stm_processed= stm.copy()
	
	# The 'process' parameter is just a convinience if the user simply want to trim the trace without thinking too much
	if process == True:
		
		stm_processed.detrend(type='linear')
		stm_processed.taper(max_percentage=0.05, type='triang')
		stm_processed.filter('bandpass', freqmin=hp,freqmax=lp, corners=4, zerophase=True)
	else:
		pass
	
	# Trim the trace 
	'''
	Purposes:	1. Getting rid of the parts being altered by tapering
				2. Showing nicer waveform for analysis and plotting
				3. Efficiency: Accelerate scanning, minimize stacking
	Note: Recommend leaving 'at least' 10~20s (difference between min arrival and max arrival for
	reginal scale earthquakes) before the event origin time and after the last arrival.	When doing 
	stacking in EDT, there will be unwanted part== "lenth of the arrival time differences btw 
	stations" at the start and the end of the trace because the code operates the shifting by np.roll.
	
	'''
	if trim == True:
	
		# If the input is a stream:
		try:
			starttime	=	stm[0].stats.starttime
			endtime		=	stm[0].stats.endtime
			
		# Exception: Input is a trace:
		except:
			starttime	=	stm.stats.starttime
			endtime		=	stm.stats.endtime
		
		stm_processed.trim(starttime= starttime+trim_start, endtime= endtime+trim_end)
		
	else:
		pass
		
	if plot == True:
		stm_processed.plot(size=size)
		
	return stm_processed
			
			
			
def chracteristic_function(stm_processed, cf_type_list, 
							plot= True, plot_size=(40,100), labelsize=18,linewidth=1.5, 
							save_cf=True, no_of_event='' ,
							save_fig=False, add_figname=''):
							
	'''
	Given a processed stream, return the characteristic function using the component 
	energy correlation method (Massin & Malcolm 2016) of E,N,Z channel, respectively.
	_______
	:input: 
		-  stm_processed: Stream (class:obspy.core.stream) 
		-  cf_type		: Type of characteristic function used in stacking (string) 
							(options: 'CECM', 'ST*LT', 'RP*LP', 'ST/LT', 'RP/LP') 
		-  plot			: Plot the cf along with waveforms or not (booline)
		-  plot_size	: The size of the plot; might vary due to the size of the stream. (2-tuple)
		-  save_cf		: Save the generated cf or not (booline)
		-  no_of_event	: Number of event (string or int)
		-  save_fig		: Save the generated cf plot or not (booline)
		-  filename		: Filename added at the end of the saved figure (string)
	_________________
	:save and output:
		- cf_E, cf_N, cf_Z
		
		(if channel=='Z') cf_Z: A list of cheracteristic function time series (A list of 1D list/array)
		  (if channel=='EN') cf_E, cf_N: 2 lists of cheracteristic function time series (2 lists of 1D list/array)
		
	'''

	cf_type, cf_filename	= proper_cf_name(cf_type_list)


	
	#Preprocessor: STA/LTA(averageabs), RP/LP(sumabs), components(rms)
	data 	= stm_processed
	
	cf_ENZ_list		= []
	
	for cf_ty in cf_type_list:
	
		if cf_ty =='CECM':
			# Compute the cf using all 3 channels
			#data_preprocessed = (trigger.Components(data, preprocessor = 'rms')).output() # This is done in trigger.Correlate() # data is a stream. data_preprocessed is a tuple
			charfct	= trigger.Correlate(data, multiplexor = "components", preprocessor = 'rms')
			cf_ENZ_list.append(	charfct.output() )
		
		
		if cf_ty =='ST/LT':
			charfct = trigger.Ratio(data, multiplexor = "shortlongterms", preprocessor = 'averageabs')  # charfct is a Ratio object
			cf_ENZ_list.append(	charfct.output() )
		
		if cf_ty =='RP/LP':
			charfct = trigger.Ratio(data, multiplexor = "leftrightterms", preprocessor = 'sumabs')  # charfct is a Ratio object
			cf_ENZ_list.append(	charfct.output() )

		if cf_ty =='ST*LT':
			charfct = trigger.Correlate(data, multiplexor = "shortlongterms", preprocessor = 'rms')  # charfct is a Correlate object
			cf_ENZ_list.append(	charfct.output() )

		if cf_ty =='RP*LP':
			charfct = trigger.Correlate(data, multiplexor = "leftrightterms", preprocessor = 'rms')  # charfct is a Correlate object
			cf_ENZ_list.append(	charfct.output() )
	
	
	cf_ENZ = np.ones_like(cf_ENZ_list[0])
	# Combining all the cf
	for cf_of_the_type in cf_ENZ_list:
		
		cf_ENZ	*=	cf_of_the_type
	
	#cf_ENZ	=	cf_ENZ **(1./len(cf_ENZ_list))
	

	
	# Plot the cf (if applicable)
	if plot == True:	
		ax,shift =	trigger.stream_processor_plot( charfct.data, cf_ENZ, 
														cfcolor='b', label=r'{}'.format(cf_type),
														size=plot_size, linewidth=linewidth)
		ax.legend(loc='upper right',fontsize=labelsize)
	
	
	
	# Get Cf of every channel (The traces have to be in order of E->N->Z in the folder)
	cf_E	=[]
	cf_N	=[]
	cf_Z	=[]
	
	for i in range(len(cf_ENZ)):
		if i%3 == 2:
			cf_Z.append(cf_ENZ[i])
		if i%3 == 0:
			cf_E.append(cf_ENZ[i])	
		elif i%3 == 1:
			cf_N.append(cf_ENZ[i])
	
	# Turn into np.array
	cf_E	=np.asarray(cf_E)
	cf_N	=np.asarray(cf_N)
	cf_Z	=np.asarray(cf_Z)
	
	# Save ############ Save ############# Save ############# Save ##########
	# Save to a saparate folder 											#
	if save_cf == True:														#
																			#
		# Create the directory if the folder does not exist					#				
		if not os.path.exists('save/'):										#				
			os.makedirs('save/')											#
																			#
		# Re-ordered cf_ENZ based on components (not station)				#
		cf_ENZ = {'E':cf_E, 'N':cf_N, 'Z':cf_Z}								#
																			#
		# Save data to file													#
		for ch_name, ch_cf in cf_ENZ.items():								#
			dir= 'save/{}_cf_{}_{}'.format(no_of_event,ch_name, cf_filename)	#
			np.save(dir, ch_cf)												#
			print('Saved as {}.npy'.format(dir))							#
																			#
	if save_fig == True:													#
		# Create the directory if the folder does not exist					#
		if not os.path.exists('save/figs'):									#
			os.makedirs('save/figs')										#
																			#

		#
		dir_s= 'save/figs/{}_cf_{}{}'.format(									#
				no_of_event, cf_filename, add_figname)								#
		plt.savefig(dir_s)													#
		print('Saved as {}'.format(dir_s))									#
																			#
	# Save ############# Save ############# Save ############# Save #########
			
	return cf_type, cf_E, cf_N, cf_Z
	
				
	
##############################################################################################################




def Travel_Time_Model(TTModel):
	'''
	Building travel time interpolation between model grid points.
	_______
	:input: 
	
		-  TTModel	: Travel Time model (2d array)
	________
	:Output: 
	
		-  tt(depth,distance)  (ie. Travel Time Model as a function of depth and distance betweem source and receiver)
	'''
	
	TTM_nz         = len(TTModel)
	TTM_nx         = len(TTModel[0])

	TTM_z          = np.arange(TTM_nz)
	TTM_x          = np.arange(TTM_nx)
	
	return  interp2d(TTM_z,	TTM_x,	TTModel.T, kind='cubic')


def proper_cf_name(cf_type_list):
	'''
	The purpose of this function is to produce a "combined CF name" in the case we 
	consider multiple CFs. 
	_______
	:input: 
		-  cf_type_list	: cf name list  (a list of strings)
	________
	:Output: 
		-  cf_type	: a combined cf name
		-  cf_filename: a string of cf names that can be used to save files
	'''
	# Combining CF names and generate filenames #####################
	# For saving file ('/' is not allowed in file name)				
	filename_list	= []											
	for cf_type in cf_type_list:									
		if cf_type =='CECM':										
			filename_list.append('CECM') # For saving file later		
		if cf_type =='ST/LT':										
			filename_list.append('ST_LT') # For saving file later	
		if cf_type =='RP/LP':								
			filename_list.append('RP_LP') # For saving file later
		if cf_type =='ST*LT':
			filename_list.append('STcLT') # For saving file later
		if cf_type =='RP*LP':
			filename_list.append('RPcLP') # For saving file later	
	
	# Combining all the cf_type name
	sj_filename	= '-' # String jointer
	cf_filename		= sj_filename.join(filename_list)
	sj_type		= '-' # String jointer
	cf_type		= sj_type.join(cf_type_list)
	
	return cf_type,cf_filename



##############################################################################################################




def SourceScan_mp(stm_processed, Cf_Z, Cf_E,Cf_N, 
				tmin, tmax, dt,
				zmin, zmax, dz,
				xmin, xmax, dx,
				ymin, ymax, dy,
				utm_zone, TTModel_1, TTModel_2,  tmin_checker = True, 
				add_filename='', method='L1',  save_C= True, cf_type_list=None, 
				no_of_event=None ,alarm=False ):
	'''
	This version of scanning function is for calculating marginal probability.
	Instead of generating temporal and spatial brightness matrices, it generate a 4D (z,x,y,t) matrix
	for stacking along each dimension maybe later.
	_______
	:input: 
	
		-  stm_processed: Stream (class:obspy.core.stream)
		-  Cf			: Characteristic function of the stream (list of 1D list/array)
		-  tmin			: Range of possible origin time (in sec after the start time)
		-  tmax
		-  dt			: Step for scanning specific time (in sec)
		-  zmin			: Range of possible depth (in km)
		-  zmax
		-  dz			: z interval (ex. dx = 0.5 then resolusion = 0.5		
		-  xmin			: E-W boundaries of the scanning area
		-  xmax			  (in Easting; omit the last 2 digits: ie. dx=100m)
		-  dx			: x interval (ex. dx = 5 then resolusion = 500m)
		-  ymin			: S-N boundaries of the scanning area
		-  ymax			  (in Northing; omit the last 2 digits: ie. dy=100m)
		-  dy			: y interval (ex. dy = 5 then resolusion = 500m)
		-  utm_zone		: The utm_zone of the scanning grid (ex. (27,'V))
		-  TTModel		: The 2D velocity model (2D array)
		-  tmin_checker : If ==True	: Raise warning when max(br) appears to be before tmin.
						  If ==False: Use alternate method when max(br) appears to be before tmin.
		-  no_of_event	: Number of event (for saving file) (int)
		-  method		: Method of Source Scanning (string) (options: 'L1', 'EDT')
		-  save			: Save the resulting raw brightness matrices to disk or not (booline)
		-  cf_type		: Type of characteristic function used in stacking 
							(options: 'CECM', 'ST*LT', 'RP*LP', 'ST/LT', 'RP/LP')
		
	____________________________
	:globalize, save and output: 
	
		-  C_4D	: 4D matrix: z,x,y,t (4D array) (need to be stacked to get maximum)
		
	
	'''
	
	# Information in SAC headers
	st_lalo         = st_locations(stm_processed)				# Stations latitude/longitude in SAC attribute
	nst             = len(stm_processed)//3				# Number of stations
	npts			= len(stm_processed[0])
	
	sampling_rate= []							# Sampling rate
	for tr in range(len(stm_processed)):
		if stm_processed[tr].stats.channel =='Z':
			sampling_rate.append(stm_processed[tr].stats.sampling_rate)


	# Name cf_type properly
	try:
		cf_type, cf_filename	= proper_cf_name(cf_type_list)
	except:
		cf_type, cf_filename	= '',''
		

	# Parameters
	number_of_time = len(	np.arange(tmin,tmax,dt)  )
	time_range     = np.linspace(tmin,tmax,number_of_time,endpoint=True)
	
	
	nz             = len(	np.arange(zmin,zmax,dz) )+1           # Number of survey grid points in z-direction
	nx             = len(	np.arange(xmin,xmax,dx) )+1           # Number of survey grid points in x-direction
	ny             = len(	np.arange(ymin,ymax,dy) )+1           # Number of survey grid points in y-direction
	
	gpts_in_z      = np.linspace(zmin,zmax,nz,endpoint=True)      # Lining up z axis of the grid 
	gpts_in_x      = np.linspace(xmin,xmax,nx,endpoint=True)      # Lining up x axis of the grid 
	gpts_in_y      = np.linspace(ymin,ymax,ny,endpoint=True)      # Lining up y axis of the grid
	
	
	
	# Building travel time interpolation between model grid points
	Travel_Time_Interp_P =     Travel_Time_Model(TTModel_1)
	Travel_Time_Interp_S =     Travel_Time_Model(TTModel_2)
	
	# Make sure the cfs are in the form of np.array
	Cf_E=np.asarray(Cf_E)
	Cf_N=np.asarray(Cf_N)
	Cf_Z=np.asarray(Cf_Z)
	

	# Scan the grid to get the brightness matrices
	# Initialize spatial brightness matrix and temporal brightness matrix
	global C_3D, C_4D
	
	C_3D      = np.zeros(shape=(nz,nx,ny))
	C_4D	= np.zeros(shape=(nz,nx,ny,number_of_time))
	

		
	# Combining the horizontal cfs
	Cf_EN	= np.sqrt(Cf_E*Cf_N)
	
	
	
	# Prepare for Scanning ##########################################################################
	# Starting the scan
	start_scan = datetime.datetime.now()

	print('Dimension of the survey grid (nst, nt+window_len-1, nz, nx, ny): ', nst,number_of_time,nz,nx,ny)
	print('Total points in 3D: ', nz*nx*ny)
	print('Scanning the grid...')
	print('0%   at {}'.format(start_scan.strftime("%H:%M:%S")))
	p_done = 0
	
	# Source Scanning procedure #####################################################################
	# START OF SCANNING ############## START OF SCANNING ############## START OF SCANNING ###########
	

	#
	# Scan all z,x,y in the grid:
	for iz, z in enumerate(gpts_in_z):
		
		# Progress counter ##############################################
		# It takes less than 1s to run this counter for a nz=1000 loop  #
		progress = iz/nz                                                #
																		#
		# Print when 25% is done										#
		if progress> 0.25 and p_done <0.25:								#
			print('{:.0f}%  at {}'.format(progress*100,  				#
					datetime.datetime.now().strftime("%H:%M:%S")))		#
			p_done = progress											#
																		#
		# Print when 50% is done										#
		elif progress> 0.5 and p_done <0.5:								#
			print('{:.0f}%  at {}'.format(progress*100,  				#
					datetime.datetime.now().strftime("%H:%M:%S")))		#
			p_done = progress											#
																		#
		# Print when 70% is done										#
		elif progress> 0.75 and p_done <0.75:							#
			print('{:.0f}%  at {}'.format(progress*100,  				#
					datetime.datetime.now().strftime("%H:%M:%S")))		#
			p_done = progress											#
																		#
		else:															#
			pass														#
																		#
		#################################################################
		
		for ix, x in enumerate(gpts_in_x):
			for iy, y in enumerate(gpts_in_y):
				# For each grid points:
				# Calculate location of this grid point
				easting = x * x_to_easting 
				northing = y * y_to_northing 
				gpt_lat, gpt_lon = utm.to_latlon(easting, northing, utm_zone[0], utm_zone[1])

				# Calculate travel time from this grid point to every station
				travel_time_P=np.zeros(shape=(nst))
				travel_time_S=np.zeros(shape=(nst))
				for st in range(nst):
					distance= calc_vincenty_inverse(st_lalo[0][st], st_lalo[1][st], gpt_lat, gpt_lon)[0]
					# Interpolate in the Travel Time Model for travel time
					travel_time_P[st] = Travel_Time_Interp_P(z,distance/1000)
					travel_time_S[st] = Travel_Time_Interp_S(z,distance/1000)
	
				#EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$
				if method == 'EDT':														#
																						#
					cf_zxy_P = np.zeros_like(Cf_Z)	# For P wave						#
					cf_zxy_S = np.zeros_like(Cf_EN)	# For S wave						#
																						#
					# Equal Differential Time Calculation								#
					# Compute new CF for each station at this point:					#
																						#
					cf_shifted_P = [0]*nst									# For P		#
					cf_shifted_S = [0]*nst									# For S		#
					nt_shift_P = [0]*nst									# For P		#
					nt_shift_S = [0]*nst									# For S		#
																						#
					# "Shift" the CFs based on travel time difference to Cf[0]'s tt		#
					for i in range(nst):											#
																						#
						t_shift_P = travel_time_P[0] - travel_time_P[i] 	# For P		#
						t_shift_S = travel_time_S[0] - travel_time_S[i] 	# For S		#
						nt_shift_P[i] = int(t_shift_P*sampling_rate[i])		# For P		#
						nt_shift_S[i] = int(t_shift_S*sampling_rate[i])		# For S		#
						# Although np.roll is slower than deque.rotate,					#
						# it saves time by not requiring conversion						#
						cf_shifted_P[i] = np.roll( Cf_Z[i], nt_shift_P[i])		# For P	#
						cf_shifted_S[i] = np.roll(Cf_EN[i], nt_shift_S[i])		# For S	#
																						#
					
					# "Stack" the shifted CFs to get new CF for Cf[0] 					#
					cf_zxy__P = 1											# For P		#
					cf_zxy__S = 1											# For S	
					for i in range(len(cf_shifted_P)):									#
						cf_zxy__P =np.multiply(cf_zxy__P, cf_shifted_P[i])				#
						cf_zxy__S =np.multiply(cf_zxy__S, cf_shifted_S[i])				#
					
					cf_zxy__P = (cf_zxy__P)** (1./float(nst))							#
					cf_zxy__S = (cf_zxy__S)** (1./float(nst))							#
					
					#
					# "Shift back" the stacked Cf[0] to the rest of the Cfs				#	
					for st in range(nst):											#
						cf_zxy_P[st] = np.roll(cf_zxy__P, -nt_shift_P[st])	# For P		#
						cf_zxy_S[st] = np.roll(cf_zxy__S, -nt_shift_S[st])				#
																						#
				# If the user not choosing EDT (method == 'L1'):					#
				else:																	#
					cf_zxy_P= Cf_Z											# For P		#
					cf_zxy_S= Cf_EN											# For S		#						
				#EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$ 
				'''
				if ix==xmin and iy==ymin and iz==zmin:
					# Plot the CFs ##########################################################
					fig, ax = plt.subplots()												#
																							#
					# Determine the verticle seperation in the graph						#
					vertical_sep_P = cf_zxy_P.ravel().max()									#
					vertical_sep_S = cf_zxy_S.ravel().max()									#
																							#
					itime = np.arange(len(cf_zxy_P[0]))										#
																							#
					for i,st in enumerate(cf_zxy_P):										#
																							#
						# Line seimograms up with verticle seperation= vertical_sep			#
						y= st  + vertical_sep_P *(len(cf_zxy_P)-i)							#
						seismogram, = plt.plot(itime/sampling_rate[i], y )					#
																							#
					fig, ax = plt.subplots()												#
					for i,st in enumerate(cf_zxy_S):										#
																							#
						# Line seimograms up with verticle seperation= vertical_sep			#
						y= st  + vertical_sep_S *(len(cf_zxy_S)-i)							#
						seismogram, = plt.plot(itime/sampling_rate[i], y )					#
					# Plot ######## Plot ######## Plot ######## Plot ######## Plot ##########
				'''
				
				# Calculate the maximum temperol brightness assuming this point is the source
				C_t = np.zeros(shape=(number_of_time)) 
				C_t_not = np.zeros(shape=(number_of_time)) 
				
				#####################################################################
				for nt, specific_time in enumerate(time_range):
				
					# Initialize 1-Cf probability list for this specific time
					#prob_not	=np.zeros(shape=(nst))
					prob_not_P=np.zeros(shape=(nst))
					prob_not_S=np.zeros(shape=(nst))
					# Get 1-Cf probability at every station (break when)
					#################################################
					for st in range(nst):
						# Convert tt from seconds to time units
						specific_nt= specific_time * sampling_rate[st]
						ntt_P = travel_time_P[st] * sampling_rate[st]			# For P
						ntt_S = travel_time_S[st] * sampling_rate[st]			# For S

						# Turn into int so it can be used as index
						assumed_arrival_nt_P = int(specific_nt + ntt_P)     	# For P
						assumed_arrival_nt_S = int(specific_nt + ntt_S) 		# For S

						# 1 - Cf probability at assumed arrival: ie. The probability 
						# of the wave "NOT arriving at this assumed arrival time"
						# (The usage of 1-Cf is preventing Cf=0 from killing the stack)
						if assumed_arrival_nt_P <= npts:						# For P
							prob_not_P[st]= 1. - cf_zxy_P[st][assumed_arrival_nt_P]
					
						# In case the trimed Cf is too short for the assumed arrival: 
						# Then this is definately not the arrival. ie: 1-Cf = 1
						else:
							prob_not_P[st]= 1.
							
						if assumed_arrival_nt_S <= npts:						# For S
							prob_not_S[st]= 1. - cf_zxy_S[st][assumed_arrival_nt_S]
						else:
							prob_not_S[st]= 1.
						
						# Combining the P and S probabilities (this way is faster than np.sqrt)
						#prob_not[st]	= (prob_not_P[st]*prob_not_S[st])**(1./2.)
					################################################

					# Compute the geometric average of the 1-Cf probability over all stations 
					# to find the 1-brightness of this point in this specific time
					# ie. The probablility of this point "NOT being the source" at this specific time.
					C_t_not[nt]= np.prod(prob_not_P*prob_not_S)**(0.5/float(nst))

					# Convert to "the probablility of this point being the source" at this specific time.
					C_t[nt] = 1.- C_t_not[nt]
					
				#####################################################################
				'''
				if mp==False:
					if teo_by == 'max C_t':
						C_t_conv =np.convolve(C_t,sum_window,'full')/window_len 
						imax_C_t= np.argmax(C_t_conv[window_len-1:])
						
						C_temporal[iz,ix,iy]= (imax_C_t ) *dt
						C_spatial[iz,ix,iy]	= C_t.ravel().max()			
						
					# For this point, the temporal brightness matrix==the onset of the maximum brightness 
					# is selected at the maximum time derivative in its time evolution.
					# Compute the derivative of the brightness time evolution before the maxima for this point					
					elif teo_by== 'max dC_t': # ie. teo_by == 'max dC_t'
						imax_C_t= np.argmax(C_t)
						
						# Compute average C_t in a running window before each time in the C_t time series 
						C_t_conv =np.convolve(C_t,sum_window,'full')/window_len  
						dC_t= np.diff(C_t_conv[:imax_C_t]) 

							
						# Get the occurence of the maximum derivative before the maxima
						try:
							# Note: value before C_t_conv[window_len] are not valid. Need to skip them. 
							# Note that imax_dC_t here "NOT includes window_len-1"!!!! (more efficient)
							imax_dC_t= np.argmax(dC_t[window_len-1:]) #+ window_len -1 
							
							# The temperol brightness matrix is obtained by the maximum derivative of the brightness
							# at this grid point
							C_temporal[iz,ix,iy]= (imax_dC_t ) *dt

							# For this point, the spacial brightness= max(temperol brightness)
							#C_spatial[iz,ix,iy]= C_t[imax_dC_t]
							C_spatial[iz,ix,iy]= C_t.ravel().max()								
							
							
						# If the maximum of the derivative happens in the beginning of the dC_t list:
						# Happens when the cf is messy
						except:
							# Raise exception and urge to try an earlier tmin:
							if tmin_checker == True:
								print('The calculated onset of the brightness maxima is ealier than tmin for some grid points.')
								print('This is a warning that you should using a smaller tmin.')
								print('Or use alternative method.')
								print('(ix,iy,iz)={},{},{}\nC_t={}\nimax_C_t={}\nC_t_conv={}\ndC_t={}\ndC_t[window_len-1:]={}'.format(
											ix,iy,iz,C_t,imax_C_t,C_t_conv,dC_t,dC_t[window_len-1:]))
								raise 
							

							# Find the max(dC_t) first instead and then find max C_t after that
							else:
								# Compute average C_t in a running window before each time in the C_t time series 
								C_t_conv =np.convolve(C_t,sum_window,'full')/window_len  
								dC_t= np.diff(C_t_conv)  
								# Note: value before C_t_conv[window_len-1] are not valid. Need to skip them. 
								# Note that imax_dC_t here "includes window_len-1"!!!!
								imax_dC_t= np.argmax(dC_t[window_len-1:]) + window_len-1
								imax_C_t= np.argmax(C_t[imax_dC_t:]) + imax_dC_t							
								

								# The temperol brightness matrix is abtained by the maximum derivative of the brightness
								# at this grid point
								C_temporal[iz,ix,iy]= (imax_dC_t - (window_len-1))* dt

								# For this point, the spacial brightness= max(temperol brightness)
								#C_spatial[iz,ix,iy]= C_t[imax_dC_t]
								C_spatial[iz,ix,iy]= C_t[imax_dC_t:].ravel().max()					
				'''
				
				# Store the temporal evolution sequence to the 4D matrix under this point
				C_4D[iz,ix,iy]	= C_t
				# Store the summed marginal probability over time at this point 
				# (still need to be divided by the total sum to get the true marginal prob)
				#C_3D[iz,ix,iy]	= np.sum(C_t)
				
				#C_t_conv =np.convolve(C_t,sum_window,'full')/window_len
				#dC_t= np.diff(C_t_conv[:imax_C_t])

					
	# Finish the marginal prob method
	# Normalized to get marginal probability	
	#C_3D	/= np.sum(C_3D)
	
		
	# END OF SCANNNING ############ END OF SCANNNING ############ END OF SCANNNING ##################
	#################################################################################################

		
		
	# Finish scan
	end_scan = datetime.datetime.now()
	print('100% at {}'.format(datetime.datetime.now().strftime("%H:%M:%S")))
	
	elapsed_time = end_scan - start_scan 
	print('Elapsed Time: ', elapsed_time)
	
	print(' ')
	print('Generated brightness 4D matrices: C_4D')

	# Save data to file #####################################################################
	# Save to a saparate folder 															#
	if save_C == True:																		#
																							#
		# Create the directory if the folder does not exist									#
		if not os.path.exists('save/'):														#
			os.makedirs('save/')															#
																							#
		#dir= 'save/{}_C_3D_{}km_{}s_{}_{}{}'.format(								#
		#		no_of_event,dz,dt,cf_filename,method,add_filename)									#
		#np.save(dir, C_3D)															#
		#print('Saved as {}.npy'.format(dir))												#
		
		dir= 'save/{}_C_4D_{}km_{}s_{}_{}{}'.format(								#
				no_of_event,dz,dt,cf_filename,method,add_filename)									#
		np.save(dir, C_4D)															#
		print('Saved as {}.npy'.format(dir))	
																							#
	#########################################################################################
	if alarm == True:
		# Use Do Re Do in case the alarm get blended in the jazz I'm listening to 
		winsound.Beep(523, 1000)
		winsound.Beep(587, 500)
		winsound.Beep(523, 500)
	
	return C_4D



	
def SourceScan(stm_processed, Cf_Z, Cf_E,Cf_N, 
				tmin, tmax, dt,
				zmin, zmax, dz,
				xmin, xmax, dx,
				ymin, ymax, dy,
				utm_zone, TTModel_1, TTModel_2,  tmin_checker = True, 
				add_filename='', method='L1',  save_C= True, cf_type_list=None, 
				no_of_event=None ,alarm=False ):
	'''
	Given the stream, charecteristic function at stations and configurations of the san, calculate the 
	spatial and temporal brightness matrices.
	Stack both P and S.
	_______
	:input: 
	
		-  stm_processed: Stream (class:obspy.core.stream)
		-  Cf			: Characteristic function of the stream (list of 1D list/array)
		-  tmin			: Range of possible origin time (in sec after the start time)
		-  tmax
		-  dt			: Step for scanning specific time (in sec)
		-  zmin			: Range of possible depth (in km)
		-  zmax
		-  dz			: z interval (ex. dx = 0.5 then resolusion = 0.5		
		-  xmin			: E-W boundaries of the scanning area
		-  xmax			  (in Easting; omit the last 2 digits: ie. dx=100m)
		-  dx			: x interval (ex. dx = 5 then resolusion = 500m)
		-  ymin			: S-N boundaries of the scanning area
		-  ymax			  (in Northing; omit the last 2 digits: ie. dy=100m)
		-  dy			: y interval (ex. dy = 5 then resolusion = 500m)
		-  utm_zone		: The utm_zone of the scanning grid (ex. (27,'V))
		-  TTModel		: The 2D velocity model (2D array)
		-  tmin_checker : If ==True	: Raise warning when max(br) appears to be before tmin.
						  If ==False: Use alternate method when max(br) appears to be before tmin.
		-  no_of_event	: Number of event (for saving file) (int)
		-  method		: Method of Source Scanning (string) (options: 'L1', 'EDT')
		-  save			: Save the resulting raw brightness matrices to disk or not (booline)
		-  cf_type		: Type of characteristic function used in stacking 
							(options: 'CECM', 'ST*LT', 'RP*LP', 'ST/LT', 'RP/LP')
		
	____________________________
	:globalize, save and output: 
	
		-  C_spatial	: Brightness spatial matrix (3D array)
		-  C_temporal	: Brightness temporal matrix (3D array)
		
		
	
	'''
	
	# Information in SAC headers
	st_lalo         = st_locations(stm_processed)				# Stations latitude/longitude in SAC attribute
	nst             = len(stm_processed)//3				# Number of stations
	npts			= len(stm_processed[0])
	
	sampling_rate= []							# Sampling rate
	for tr in range(len(stm_processed)):
		if stm_processed[tr].stats.channel =='Z':
			sampling_rate.append(stm_processed[tr].stats.sampling_rate)


	# Name cf_type properly
	try:
		cf_type, cf_filename	= proper_cf_name(cf_type_list)
	except:
		cf_type, cf_filename	= '',''
		
	# Choose stacking method for different cf (ratio or correlation methods)
	if cf_type == 'CECM':
		prob_or_like, teo_by	= 'prob', 'max dC_t'	# dC5:-0.1s, 0.1 real # 2 0.2
		window_len = 2
	elif cf_type =='ST*LT':
		prob_or_like, teo_by	= 'prob', 'max dC_t' # C2:+0.2s # dC5:-0.2s, 0.2 r # dc5 -0.2 #2r -0.1 
		window_len = 2
	elif cf_type =='ST*LT-CECM':
		prob_or_like, teo_by	= 'prob', 'max dC_t' # C2:+0.1s # dC5:-0.2s, 0.3 r #dc2 -0.2 #r 0.1
		window_len = 2	
	elif cf_type == 'ST/LT':
		prob_or_like, teo_by	= 'like', 'max dC_t' # dC 1:0s v # r 0.2
		window_len = 1   #3 cause smth dt=0.1
	elif cf_type =='RP/LP':
		prob_or_like, teo_by	= 'like', 'max C_t'# C 1:0s:-0.1s,-0.7 # dc r -1.3
		window_len = 1
	else:
		print('Input cf_type is not supported.')
		
			
			

	# The window to average over for determining the max change in brightness for event origin time
	sum_window	= np.ones(window_len,dtype=int)
	# Need to exam more earlier t because window_len>1 will create invalid value
	# for the first few time step (the artefact of np.convolution 'full' method)
	tmin -= (window_len-1)*dt #
			
	# Parameters
	number_of_time = len(	np.arange(tmin,tmax,dt)  )
	time_range     = np.linspace(tmin,tmax,number_of_time,endpoint=True)
	
	
	nz             = len(	np.arange(zmin,zmax,dz) )+1           # Number of survey grid points in z-direction
	nx             = len(	np.arange(xmin,xmax,dx) )+1           # Number of survey grid points in x-direction
	ny             = len(	np.arange(ymin,ymax,dy) )+1           # Number of survey grid points in y-direction
	
	gpts_in_z      = np.linspace(zmin,zmax,nz,endpoint=True)      # Lining up z axis of the grid 
	gpts_in_x      = np.linspace(xmin,xmax,nx,endpoint=True)      # Lining up x axis of the grid 
	gpts_in_y      = np.linspace(ymin,ymax,ny,endpoint=True)      # Lining up y axis of the grid
	
	
	
	# Building travel time interpolation between model grid points
	Travel_Time_Interp_P =     Travel_Time_Model(TTModel_1)
	Travel_Time_Interp_S =     Travel_Time_Model(TTModel_2)
	
	# Make sure the cfs are in the form of np.array
	Cf_E=np.asarray(Cf_E)
	Cf_N=np.asarray(Cf_N)
	Cf_Z=np.asarray(Cf_Z)
	
	# If the cf are correlation functions (scaled btw 0~1), use negation instead
	if prob_or_like == 'prob':
		Cf_E	= 1-Cf_E
		Cf_N	= 1-Cf_N
		Cf_Z	= 1-Cf_Z
	
	
	# Scan the grid to get the brightness matrices
	# Initialize spatial brightness matrix and temporal brightness matrix
	global C_spatial, C_temporal#, C_5D
	
	C_spatial      	= np.zeros(shape=(nz,nx,ny))
	C_temporal     	= np.zeros(shape=(nz,nx,ny))
	#C_6D			= np.zeros(shape=(nz,nx,ny,number_of_time, nst,2)) 

		
	# Combining the horizontal cfs
	Cf_EN	= np.sqrt(Cf_E*Cf_N)
	
	
	# Prepare for Scanning ##########################################################################
	# Starting the scan
	start_scan = datetime.datetime.now()

	print('Dimension of the survey grid (nst, nt+window_len-1, nz, nx, ny): ', nst,number_of_time,nz,nx,ny)
	print('Total points in 3D: ', nz*nx*ny)
	print('Scanning the grid...')
	print('0%   at {}'.format(start_scan.strftime("%H:%M:%S")))
	p_done = 0
	
	# Source Scanning procedure #####################################################################
	# START OF SCANNING ############## START OF SCANNING ############## START OF SCANNING ###########
	

	#
	# Scan all z,x,y in the grid:
	for iz, z in enumerate(gpts_in_z):
		
		# Progress counter ##############################################
		# It takes less than 1s to run this counter for a nz=1000 loop  #
		progress = iz/nz                                                #
																		#
		# Print when 25% is done										#
		if progress> 0.25 and p_done <0.25:								#
			print('{:.0f}%  at {}'.format(progress*100,  				#
					datetime.datetime.now().strftime("%H:%M:%S")))		#
			p_done = progress											#
																		#
		# Print when 50% is done										#
		elif progress> 0.5 and p_done <0.5:								#
			print('{:.0f}%  at {}'.format(progress*100,  				#
					datetime.datetime.now().strftime("%H:%M:%S")))		#
			p_done = progress											#
																		#
		# Print when 70% is done										#
		elif progress> 0.75 and p_done <0.75:							#
			print('{:.0f}%  at {}'.format(progress*100,  				#
					datetime.datetime.now().strftime("%H:%M:%S")))		#
			p_done = progress											#
																		#
		else:															#
			pass														#
																		#
		#################################################################
		
		for ix, x in enumerate(gpts_in_x):
			for iy, y in enumerate(gpts_in_y):
				# For each grid points:
				# Calculate location of this grid point
				easting = x * x_to_easting 
				northing = y * y_to_northing 
				gpt_lat, gpt_lon = utm.to_latlon(easting, northing, utm_zone[0], utm_zone[1])

				# Calculate travel time from this grid point to every station
				travel_time_P=np.zeros(shape=(nst))
				travel_time_S=np.zeros(shape=(nst))
				for st in range(nst):
					distance= calc_vincenty_inverse(st_lalo[0][st], st_lalo[1][st], gpt_lat, gpt_lon)[0]
					# Interpolate in the Travel Time Model for travel time
					travel_time_P[st] = Travel_Time_Interp_P(z,distance/1000)
					travel_time_S[st] = Travel_Time_Interp_S(z,distance/1000)
	
				#EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$
				if method == 'EDT':														#
																						#
					cf_zxy_P = np.zeros_like(Cf_Z)	# For P wave						#
					cf_zxy_S = np.zeros_like(Cf_EN)	# For S wave						#
																						#
					# Equal Differential Time Calculation								#
					# Compute new CF for each station at this point:					#
																						#
					cf_shifted_P = [0]*nst									# For P		#
					cf_shifted_S = [0]*nst									# For S		#
					nt_shift_P = [0]*nst									# For P		#
					nt_shift_S = [0]*nst									# For S		#
																						#
					# "Shift" the CFs based on travel time difference to Cf[0]'s tt		#
					for i in range(nst):											#
																						#
						t_shift_P = travel_time_P[0] - travel_time_P[i] 	# For P		#
						t_shift_S = travel_time_S[0] - travel_time_S[i] 	# For S		#
						nt_shift_P[i] = int(t_shift_P*sampling_rate[i])		# For P		#
						nt_shift_S[i] = int(t_shift_S*sampling_rate[i])		# For S		#
						# Although np.roll is slower than deque.rotate,					#
						# it saves time by not requiring conversion						#
						cf_shifted_P[i] = np.roll( Cf_Z[i], nt_shift_P[i])		# For P	#
						cf_shifted_S[i] = np.roll(Cf_EN[i], nt_shift_S[i])		# For S	#
																						#
					# "Stack" the shifted CFs to get new CF for Cf[0] 					#
					cf_zxy__P = 1											# For P		#
					cf_zxy__S = 1											# For S	
					for i in range(len(cf_shifted_P)):									#
						cf_zxy__P =np.multiply(cf_zxy__P, cf_shifted_P[i])				#
						cf_zxy__S =np.multiply(cf_zxy__S, cf_shifted_S[i])				#
					
					cf_zxy__P = (cf_zxy__P)** (1./float(nst))							#
					cf_zxy__S = (cf_zxy__S)** (1./float(nst))							#
					
					# "Shift back" the stacked Cf[0] to the rest of the Cfs				#	
					for st in range(nst):											#
						cf_zxy_P[st] = np.roll(cf_zxy__P, -nt_shift_P[st])	# For P		#
						cf_zxy_S[st] = np.roll(cf_zxy__S, -nt_shift_S[st])				#
																						#
				# If the user not choosing EDT (method == 'L1'):					#
				else:																	#
					cf_zxy_P= Cf_Z											# For P		#
					cf_zxy_S= Cf_EN											# For S		#						
				#EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$ 
				'''
				if ix==xmin and iy==ymin and iz==zmin:
					# Plot the CFs ##########################################################
					fig, ax = plt.subplots()												#
																							#
					# Determine the verticle seperation in the graph						#
					vertical_sep_P = cf_zxy_P.ravel().max()									#
					vertical_sep_S = cf_zxy_S.ravel().max()									#
																							#
					itime = np.arange(len(cf_zxy_P[0]))										#
																							#
					for i,st in enumerate(cf_zxy_P):										#
																							#
						# Line seimograms up with verticle seperation= vertical_sep			#
						y= st  + vertical_sep_P *(len(cf_zxy_P)-i)							#
						seismogram, = plt.plot(itime/sampling_rate[i], y )					#
																							#
					fig, ax = plt.subplots()												#
					for i,st in enumerate(cf_zxy_S):										#
																							#
						# Line seimograms up with verticle seperation= vertical_sep			#
						y= st  + vertical_sep_S *(len(cf_zxy_S)-i)							#
						seismogram, = plt.plot(itime/sampling_rate[i], y )					#
					# Plot ######## Plot ######## Plot ######## Plot ######## Plot ##########
				'''
				
				# Calculate the maximum temperol brightness assuming this point is the source
				C_t = np.zeros(shape=(number_of_time)) 
				
				if prob_or_like == 'prob':
					# Use negation method
					C_t_not = np.zeros(shape=(number_of_time)) 
					#####################################################################
					for nt, specific_time in enumerate(time_range):
						
						# Initialize 1-Cf probability list for this specific time
						#prob_not	=np.zeros(shape=(nst))
						prob_not_P=np.zeros(shape=(nst))
						prob_not_S=np.zeros(shape=(nst))
						# Get  probability at every station (break when)
						#################################################
						for st in range(nst):
							# Convert tt from seconds to time units
							specific_nt= specific_time * sampling_rate[st]
							ntt_P = travel_time_P[st] * sampling_rate[st]			# For P
							ntt_S = travel_time_S[st] * sampling_rate[st]			# For S

							# Turn into int so it can be used as index
							assumed_arrival_nt_P = int(specific_nt + ntt_P)     	# For P
							assumed_arrival_nt_S = int(specific_nt + ntt_S) 		# For S

							# Notice that cf here is actually the negation: 1 - Cf: 
							# probability at assumed arrival: ie. The probability 
							# of the wave "NOT arriving at this assumed arrival time"
							# (The usage of 1-Cf is preventing Cf=0 from killing the stack)
							if assumed_arrival_nt_P <= npts:						# For P
								prob_not_P[st]= cf_zxy_P[st][assumed_arrival_nt_P]
						
							# In case the trimed Cf is too short for the assumed arrival: 
							# Then this is definately not the arrival. ie: 1-Cf = 1
							else:
								prob_not_P[st]= 1.
								
							if assumed_arrival_nt_S <= npts:						# For S
								prob_not_S[st]= cf_zxy_S[st][assumed_arrival_nt_S]
							else:
								prob_not_S[st]= 1.
							
							# Store the probability at stations to the 4D matrix under this (z,x,y,t) point
							# This C_5D is for marginal probability analysis
							#C_6D[iz,ix,iy,nt,st,0]	= 1-prob_not_P[st]
							#C_6D[iz,ix,iy,nt,st,1]	= 1-prob_not_S[st]
							
							# Combining the P and S probabilities later together
						################################################

						# Compute the geometric average of the 1-Cf probability over all stations 
						# to find the 1-brightness of this point in this specific time
						# ie. The probablility of this point "NOT being the source" at this specific time.
						C_t_not[nt]= np.prod(prob_not_P*prob_not_S)**(0.5/float(nst))

						# Convert to "the probablility of this point being the source" at this specific time.
						C_t[nt] = 1.- C_t_not[nt]
				
				elif prob_or_like == 'like':
					
					#####################################################################
					for nt, specific_time in enumerate(time_range):
						
						# Initialize Cf likelihood list for this specific time
						#prob_not	=np.zeros(shape=(nst))
						like_P=np.zeros(shape=(nst))
						like_S=np.zeros(shape=(nst))
						# Get Cf likelihood at every station (break when)
						#################################################
						for st in range(nst):
							# Convert tt from seconds to time units
							specific_nt= specific_time * sampling_rate[st]
							ntt_P = travel_time_P[st] * sampling_rate[st]			# For P
							ntt_S = travel_time_S[st] * sampling_rate[st]			# For S

							# Turn into int so it can be used as index
							assumed_arrival_nt_P = int(specific_nt + ntt_P)     	# For P
							assumed_arrival_nt_S = int(specific_nt + ntt_S) 		# For S

							# Cf: likelihood at assumed arrival
							if assumed_arrival_nt_P <= npts:						# For P
								like_P[st]= cf_zxy_P[st][assumed_arrival_nt_P]
						
							# In case the trimed Cf is too short for the assumed arrival: 
							# Then this is definately not the arrival by definition.
							else:
								like_P[st]= 0.
								
							if assumed_arrival_nt_S <= npts:						# For S
								like_S[st]= cf_zxy_S[st][assumed_arrival_nt_S]
							else:
								like_S[st]= 0.
								
							# Store the probability at stations to the 4D matrix under this (z,x,y,t) point
							# This C_5D is for marginal probability analysis
							#C_6D[iz,ix,iy,nt,st,0]	= like_P[st]
							#C_6D[iz,ix,iy,nt,st,1]	= like_S[st]
							
							# Combining the P and S likelihood later together
						################################################

						# Compute the geometric average of the Cf likelihood over all stations 
						# to find the brightness of this point in this specific time
						C_t[nt]= np.prod(like_P*like_S)**(0.5/float(nst))
				
				#####################################################################
				
				if teo_by == 'max C_t':
					C_t_conv =np.convolve(C_t,sum_window,'full')/window_len 
					imax_C_t= np.argmax(C_t_conv[window_len-1:])
					
					C_temporal[iz,ix,iy]= (imax_C_t ) *dt
					C_spatial[iz,ix,iy]= C_t.ravel().max()			
					
				# For this point, the temporal brightness matrix==the onset of the maximum brightness 
				# is selected at the maximum time derivative in its time evolution.
				# Compute the derivative of the brightness time evolution before the maxima for this point					
				else: # ie. teo_by == 'max dC_t'
					imax_C_t= np.argmax(C_t)
					
					# Compute average C_t in a running window before each time in the C_t time series 
					C_t_conv =np.convolve(C_t,sum_window,'full')/window_len  
					dC_t= np.diff(C_t_conv[:imax_C_t]) 

						
					# Get the occurence of the maximum derivative before the maxima
					try:
						# Note: value before C_t_conv[window_len] are not valid. Need to skip them. 
						# Note that imax_dC_t here "NOT includes window_len-1"!!!! (more efficient)
						imax_dC_t= np.argmax(dC_t[window_len-1:]) #+ window_len -1 
						
						# The temperol brightness matrix is obtained by the maximum derivative of the brightness
						# at this grid point
						C_temporal[iz,ix,iy]= (imax_dC_t ) *dt

						# For this point, the spacial brightness= max(temperol brightness)
						#C_spatial[iz,ix,iy]= C_t[imax_dC_t]
						C_spatial[iz,ix,iy]= C_t.ravel().max()								
						
						
					# If the maximum of the derivative happens in the beginning of the dC_t list:
					# Happens when the cf is messy
					except:
						# Raise exception and urge to try an earlier tmin:
						if tmin_checker == True:
							print('The calculated onset of the brightness maxima is ealier than tmin for some grid points.')
							print('This is a warning that you should using a smaller tmin.')
							print('Or use alternative method.')
							print('(ix,iy,iz)={},{},{}\nC_t={}\nimax_C_t={}\nC_t_conv={}\ndC_t={}\ndC_t[window_len-1:]={}'.format(
										ix,iy,iz,C_t,imax_C_t,C_t_conv,dC_t,dC_t[window_len-1:]))
							raise 
						

						# Find the max(dC_t) first instead and then find max C_t after that
						else:
							# Compute average C_t in a running window before each time in the C_t time series 
							C_t_conv =np.convolve(C_t,sum_window,'full')/window_len  
							dC_t= np.diff(C_t_conv)  
							# Note: value before C_t_conv[window_len-1] are not valid. Need to skip them. 
							# Note that imax_dC_t here "includes window_len-1"!!!!
							imax_dC_t= np.argmax(dC_t[window_len-1:]) + window_len-1
							imax_C_t= np.argmax(C_t[imax_dC_t:]) + imax_dC_t							
							

							# The temperol brightness matrix is abtained by the maximum derivative of the brightness
							# at this grid point
							C_temporal[iz,ix,iy]= (imax_dC_t - (window_len-1))* dt

							# For this point, the spacial brightness= max(temperol brightness)
							#C_spatial[iz,ix,iy]= C_t[imax_dC_t]
							C_spatial[iz,ix,iy]= C_t[imax_dC_t:].ravel().max()					
			
		# END OF SCANNNING ############ END OF SCANNNING ############ END OF SCANNNING ##################
		#################################################################################################

		
		
	# Finish scan
	end_scan = datetime.datetime.now()
	print('100% at {}'.format(datetime.datetime.now().strftime("%H:%M:%S")))
	
	elapsed_time = end_scan - start_scan 
	print('Elapsed Time: ', elapsed_time)
	
	print(' ')
	print('Generated brightness 3D matrices: C_spatial, C_temporal')

	# Save data to file #####################################################################
	# Save to a saparate folder 															#
	if save_C == True:																		#
																							#
		# Create the directory if the folder does not exist									#
		if not os.path.exists('save/'):														#
			os.makedirs('save/')															#
																							#
		dir= 'save/{}_C_spatial_{}km_{}s_{}_{}_w{}{}'.format(								#
				no_of_event,dz,dt,cf_filename,method,window_len,add_filename)									#
		np.save(dir, C_spatial)															#
		print('Saved as {}.npy'.format(dir))												#
																							#
		dir= 'save/{}_C_temporal_{}km_{}s_{}_{}_w{}{}'.format(								#
				no_of_event,dz,dt,cf_filename,method,window_len,add_filename)									#
		np.save(dir, C_temporal)															#
		print('Saved as {}.npy'.format(dir))												#
		
		#dir= 'save/{}_C_6D_{}km_{}s_{}_{}{}'.format(								#
		#		no_of_event,dz,dt,cf_filename,method,add_filename)									#
		#np.save(dir, C_6D)															#
		#print('Saved as {}.npy'.format(dir))		
		
																							#
	#########################################################################################
	if alarm == True:
		winsound.Beep(523, 1000)
		winsound.Beep(587, 500)
		winsound.Beep(523, 500)
		
	return C_spatial, C_temporal#, C_6D



	
	

	
def SourceScan_Phase(stm_processed, Cf_Z, 
				tmin, tmax, dt,
				zmin, zmax, dz,
				xmin, xmax, dx,
				ymin, ymax, dy,
				utm_zone, TTModel_1, TTModel_2=None,  Cf_E=None,Cf_N=None, tmin_checker = True, 
				add_filename='', method='L1',  save_C= True, cf_type_list=None, 
				no_of_event=None ,alarm=False,phase='' ):
	'''
	Given the stream, charecteristic function at stations and configurations of the san,
	calculate the temperol and spatial brightness matrices.
	
	_______
	:input: 
	
		-  stm_processed: Stream (class:obspy.core.stream)
		-  Cf			: Characteristic function of the stream (list of 1D list/array)
		-  tmin			: Range of possible origin time (in sec after the start time)
		-  tmax
		-  dt			: Step for scanning specific time (in sec)
		-  zmin			: Range of possible depth (in km)
		-  zmax
		-  dz			: z interval (ex. dx = 0.5 then resolusion = 0.5		
		-  xmin			: E-W boundaries of the scanning area
		-  xmax			  (in Easting; omit the last 2 digits: ie. dx=100m)
		-  dx			: x interval (ex. dx = 5 then resolusion = 500m)
		-  ymin			: S-N boundaries of the scanning area
		-  ymax			  (in Northing; omit the last 2 digits: ie. dy=100m)
		-  dy			: y interval (ex. dy = 5 then resolusion = 500m)
		-  utm_zone		: The utm_zone of the scanning grid (ex. (27,'V))
		-  TTModel		: The 2D velocity model (2D array)
		-  tmin_checker : If ==True	: Raise warning when max(br) appears to be before tmin.
						  If ==False: Use alternate method when max(br) appears to be before tmin.
		-  no_of_event	: Number of event (for saving file) (int)
		-  method		: Method of Source Scanning (string) (options: 'L1', 'EDT')
		-  save			: Save the resulting raw brightness matrices to disk or not (booline)
		-  cf_type		: Type of characteristic function used in stacking 
							(options: 'CECM', 'ST*LT', 'RP*LP', 'ST/LT', 'RP/LP')
		
	____________________________
	:globalize, save and output: 
	
		-  C_spatial	: Brightness spatial matrix (3D array)
		-  C_temporal	: Brightness temporal matrix (3D array)
		
		
	
	'''
	
	# Information in SAC headers
	st_lalo         = st_locations(stm_processed)				# Stations latitude/longitude in SAC attribute
	nst             = len(stm_processed)//3				# Number of stations
	npts			= len(stm_processed[0])
	
	sampling_rate= []							# Sampling rate
	for tr in range(len(stm_processed)):
		if stm_processed[tr].stats.channel =='Z':
			sampling_rate.append(stm_processed[tr].stats.sampling_rate)


	# Name cf_type properly
	try:
		cf_type, cf_filename	= proper_cf_name(cf_type_list)
	except:
		cf_type, cf_filename	= '',''
		
	# Choose stacking method for different cf (ratio or correlation methods)
	if cf_type == 'CECM':
		prob_or_like, teo_by	= 'prob', 'max dC_t'	# dC5:-0.1s, 0.1 real # 2 0.2
		window_len = 2
	elif cf_type =='ST*LT':
		prob_or_like, teo_by	= 'prob', 'max dC_t' # C2:+0.2s # dC5:-0.2s, 0.2 r # dc5 -0.2 #2r -0.1 
		window_len = 2
	elif cf_type =='ST*LT-CECM':
		prob_or_like, teo_by	= 'prob', 'max dC_t' # C2:+0.1s # dC5:-0.2s, 0.3 r #dc2 -0.2 #r 0.1
		window_len = 2	
	elif cf_type == 'ST/LT':
		prob_or_like, teo_by	= 'like', 'max dC_t' # dC 1:0s v # r 0.2
		window_len = 1   #3 cause smth dt=0.1
	elif cf_type =='RP/LP':
		prob_or_like, teo_by	= 'like', 'max C_t'# C 1:0s:-0.1s,-0.7 # dc r -1.3
		window_len = 1
	else:
		print('Input cf_type is not supported.')
		
			
			

	# The window to average over for determining the max change in brightness for event origin time
	sum_window	= np.ones(window_len,dtype=int)
	# Need to exam more earlier t because window_len>1 will create invalid value
	# for the first few time step (the artefact of np.convolution 'full' method)
	tmin -= (window_len-1)*dt #
			
	# Parameters
	number_of_time = len(	np.arange(tmin,tmax,dt)  )
	time_range     = np.linspace(tmin,tmax,number_of_time,endpoint=True)
	
	
	nz             = len(	np.arange(zmin,zmax,dz) )+1           # Number of survey grid points in z-direction
	nx             = len(	np.arange(xmin,xmax,dx) )+1           # Number of survey grid points in x-direction
	ny             = len(	np.arange(ymin,ymax,dy) )+1           # Number of survey grid points in y-direction
	
	gpts_in_z      = np.linspace(zmin,zmax,nz,endpoint=True)      # Lining up z axis of the grid 
	gpts_in_x      = np.linspace(xmin,xmax,nx,endpoint=True)      # Lining up x axis of the grid 
	gpts_in_y      = np.linspace(ymin,ymax,ny,endpoint=True)      # Lining up y axis of the grid
	
	
	
	# Building travel time interpolation between model grid points
	Travel_Time_Interp_P =     Travel_Time_Model(TTModel_1)
	
	# Make sure the cfs are in the form of np.array
	Cf_Z=np.asarray(Cf_Z)
	
	# If the cf are correlation functions (scaled btw 0~1), use negation instead
	if prob_or_like == 'prob':
		Cf_Z	= 1-Cf_Z
	
	
	# Scan the grid to get the brightness matrices
	# Initialize spatial brightness matrix and temporal brightness matrix
	global C_spatial, C_temporal#, C_5D
	
	C_spatial      	= np.zeros(shape=(nz,nx,ny))
	C_temporal     	= np.zeros(shape=(nz,nx,ny))
	#C_6D			= np.zeros(shape=(nz,nx,ny,number_of_time, nst,1)) 

		
	
	# Prepare for Scanning ##########################################################################
	# Starting the scan
	start_scan = datetime.datetime.now()

	print('Dimension of the survey grid (nst, nt+window_len-1, nz, nx, ny): ', nst,number_of_time,nz,nx,ny)
	print('Total points in 3D: ', nz*nx*ny)
	print('Scanning the grid...')
	print('0%   at {}'.format(start_scan.strftime("%H:%M:%S")))
	p_done = 0
	
	# Source Scanning procedure #####################################################################
	# START OF SCANNING ############## START OF SCANNING ############## START OF SCANNING ###########
	

	#
	# Scan all z,x,y in the grid:
	for iz, z in enumerate(gpts_in_z):
		
		# Progress counter ##############################################
		# It takes less than 1s to run this counter for a nz=1000 loop  #
		progress = iz/nz                                                #
																		#
		# Print when 25% is done										#
		if progress> 0.25 and p_done <0.25:								#
			print('{:.0f}%  at {}'.format(progress*100,  				#
					datetime.datetime.now().strftime("%H:%M:%S")))		#
			p_done = progress											#
																		#
		# Print when 50% is done										#
		elif progress> 0.5 and p_done <0.5:								#
			print('{:.0f}%  at {}'.format(progress*100,  				#
					datetime.datetime.now().strftime("%H:%M:%S")))		#
			p_done = progress											#
																		#
		# Print when 70% is done										#
		elif progress> 0.75 and p_done <0.75:							#
			print('{:.0f}%  at {}'.format(progress*100,  				#
					datetime.datetime.now().strftime("%H:%M:%S")))		#
			p_done = progress											#
																		#
		else:															#
			pass														#
																		#
		#################################################################
		
		for ix, x in enumerate(gpts_in_x):
			for iy, y in enumerate(gpts_in_y):
				# For each grid points:
				# Calculate location of this grid point
				easting = x * x_to_easting 
				northing = y * y_to_northing 
				gpt_lat, gpt_lon = utm.to_latlon(easting, northing, utm_zone[0], utm_zone[1])

				# Calculate travel time from this grid point to every station
				travel_time_P=np.zeros(shape=(nst))
				for st in range(nst):
					distance= calc_vincenty_inverse(st_lalo[0][st], st_lalo[1][st], gpt_lat, gpt_lon)[0]
					# Interpolate in the Travel Time Model for travel time
					travel_time_P[st] = Travel_Time_Interp_P(z,distance/1000)
	
				#EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$
				if method == 'EDT':														#
																						#
					cf_zxy_P = np.zeros_like(Cf_Z)	# For P wave						#
																						#
					# Equal Differential Time Calculation								#
					# Compute new CF for each station at this point:					#
																						#
					cf_shifted_P = [0]*nst									# For P		#
					nt_shift_P = [0]*nst									# For P		#
																						#
					# "Shift" the CFs based on travel time difference to Cf[0]'s tt		#
					for i in range(nst):											#
																						#
						t_shift_P = travel_time_P[0] - travel_time_P[i] 	# For P		#
						nt_shift_P[i] = int(t_shift_P*sampling_rate[i])		# For P		#
						# Although np.roll is slower than deque.rotate,					#
						# it saves time by not requiring conversion						#
						cf_shifted_P[i] = np.roll( Cf_Z[i], nt_shift_P[i])		# For P	#
																						#
					# "Stack" the shifted CFs to get new CF for Cf[0] 					#
					cf_zxy__P = 1											# For P		#
					for i in range(len(cf_shifted_P)):									#
						cf_zxy__P =np.multiply(cf_zxy__P, cf_shifted_P[i])				#
					
					cf_zxy__P = (cf_zxy__P)** (1./float(nst))							#
					
					# "Shift back" the stacked Cf[0] to the rest of the Cfs				#	
					for st in range(nst):											#
						cf_zxy_P[st] = np.roll(cf_zxy__P, -nt_shift_P[st])	# For P		#
																						#
				# If the user not choosing EDT (method == 'L1'):					#
				else:																	#
					cf_zxy_P= Cf_Z											# For P		#
				#EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$#$#$#$#$EDT$#$#$ 
				'''
				if ix==xmin and iy==ymin and iz==zmin:
					# Plot the CFs ##########################################################
					fig, ax = plt.subplots()												#
																							#
					# Determine the verticle seperation in the graph						#
					vertical_sep_P = cf_zxy_P.ravel().max()									#
					vertical_sep_S = cf_zxy_S.ravel().max()									#
																							#
					itime = np.arange(len(cf_zxy_P[0]))										#
																							#
					for i,st in enumerate(cf_zxy_P):										#
																							#
						# Line seimograms up with verticle seperation= vertical_sep			#
						y= st  + vertical_sep_P *(len(cf_zxy_P)-i)							#
						seismogram, = plt.plot(itime/sampling_rate[i], y )					#
																							#
					fig, ax = plt.subplots()												#
					for i,st in enumerate(cf_zxy_S):										#
																							#
						# Line seimograms up with verticle seperation= vertical_sep			#
						y= st  + vertical_sep_S *(len(cf_zxy_S)-i)							#
						seismogram, = plt.plot(itime/sampling_rate[i], y )					#
					# Plot ######## Plot ######## Plot ######## Plot ######## Plot ##########
				'''
				
				# Calculate the maximum temperol brightness assuming this point is the source
				C_t = np.zeros(shape=(number_of_time)) 
				
				if prob_or_like == 'prob':
					# Use negation method
					C_t_not = np.zeros(shape=(number_of_time)) 
					#####################################################################
					for nt, specific_time in enumerate(time_range):
						
						# Initialize 1-Cf probability list for this specific time
						#prob_not	=np.zeros(shape=(nst))
						prob_not_P=np.zeros(shape=(nst))
						#################################################
						for st in range(nst):
							# Convert tt from seconds to time units
							specific_nt= specific_time * sampling_rate[st]
							ntt_P = travel_time_P[st] * sampling_rate[st]			# For P

							# Turn into int so it can be used as index
							assumed_arrival_nt_P = int(specific_nt + ntt_P)     	# For P

							# Notice that cf here is actually the negation: 1 - Cf: 
							# probability at assumed arrival: ie. The probability 
							# of the wave "NOT arriving at this assumed arrival time"
							# (The usage of 1-Cf is preventing Cf=0 from killing the stack)
							if assumed_arrival_nt_P <= npts:						# For P
								prob_not_P[st]= cf_zxy_P[st][assumed_arrival_nt_P]
						
							# In case the trimed Cf is too short for the assumed arrival: 
							# Then this is definately not the arrival. ie: 1-Cf = 1
							else:
								prob_not_P[st]= 1.
								
							
							# Store the probability at stations to the 4D matrix under this (z,x,y,t) point
							# This C_5D is for marginal probability analysis
							#C_6D[iz,ix,iy,nt,st,0]	= 1-prob_not_P[st]
							
							# Combining the P and S probabilities later together
						################################################

						# Compute the geometric average of the 1-Cf probability over all stations 
						# to find the 1-brightness of this point in this specific time
						# ie. The probablility of this point "NOT being the source" at this specific time.
						C_t_not[nt]= np.prod(prob_not_P)**(1/float(nst))

						# Convert to "the probablility of this point being the source" at this specific time.
						C_t[nt] = 1.- C_t_not[nt]
				
				elif prob_or_like == 'like':
					
					#####################################################################
					for nt, specific_time in enumerate(time_range):
						
						# Initialize Cf likelihood list for this specific time
						#prob_not	=np.zeros(shape=(nst))
						like_P=np.zeros(shape=(nst))
						# Get Cf likelihood at every station (break when)
						#################################################
						for st in range(nst):
							# Convert tt from seconds to time units
							specific_nt= specific_time * sampling_rate[st]
							ntt_P = travel_time_P[st] * sampling_rate[st]			# For P

							# Turn into int so it can be used as index
							assumed_arrival_nt_P = int(specific_nt + ntt_P)     	# For P

							# Cf: likelihood at assumed arrival
							if assumed_arrival_nt_P <= npts:						# For P
								like_P[st]= cf_zxy_P[st][assumed_arrival_nt_P]
						
							# In case the trimed Cf is too short for the assumed arrival: 
							# Then this is definately not the arrival by definition.
							else:
								like_P[st]= 0.
								
								
							# Store the probability at stations to the 4D matrix under this (z,x,y,t) point
							# This C_5D is for marginal probability analysis
							#C_6D[iz,ix,iy,nt,st,0]	= like_P[st]
							
							# Combining the P and S likelihood later together
						################################################

						# Compute the geometric average of the Cf likelihood over all stations 
						# to find the brightness of this point in this specific time
						C_t[nt]= np.prod(like_P)**(1/float(nst))
				
				#####################################################################
				
				if teo_by == 'max C_t':
					C_t_conv =np.convolve(C_t,sum_window,'full')/window_len 
					imax_C_t= np.argmax(C_t_conv[window_len-1:])
					
					C_temporal[iz,ix,iy]= (imax_C_t ) *dt
					C_spatial[iz,ix,iy]= C_t.ravel().max()			
					
				# For this point, the temporal brightness matrix==the onset of the maximum brightness 
				# is selected at the maximum time derivative in its time evolution.
				# Compute the derivative of the brightness time evolution before the maxima for this point					
				else: # ie. teo_by == 'max dC_t'
					imax_C_t= np.argmax(C_t)
					
					# Compute average C_t in a running window before each time in the C_t time series 
					C_t_conv =np.convolve(C_t,sum_window,'full')/window_len  
					dC_t= np.diff(C_t_conv[:imax_C_t]) 

						
					# Get the occurence of the maximum derivative before the maxima
					try:
						# Note: value before C_t_conv[window_len] are not valid. Need to skip them. 
						# Note that imax_dC_t here "NOT includes window_len-1"!!!! (more efficient)
						imax_dC_t= np.argmax(dC_t[window_len-1:]) #+ window_len -1 
						
						# The temperol brightness matrix is obtained by the maximum derivative of the brightness
						# at this grid point
						C_temporal[iz,ix,iy]= (imax_dC_t ) *dt

						# For this point, the spacial brightness= max(temperol brightness)
						#C_spatial[iz,ix,iy]= C_t[imax_dC_t]
						C_spatial[iz,ix,iy]= C_t.ravel().max()								
						
						
					# If the maximum of the derivative happens in the beginning of the dC_t list:
					# Happens when the cf is messy
					except:
						# Raise exception and urge to try an earlier tmin:
						if tmin_checker == True:
							print('The calculated onset of the brightness maxima is ealier than tmin for some grid points.')
							print('This is a warning that you should using a smaller tmin.')
							print('Or use alternative method.')
							print('(ix,iy,iz)={},{},{}\nC_t={}\nimax_C_t={}\nC_t_conv={}\ndC_t={}\ndC_t[window_len-1:]={}'.format(
										ix,iy,iz,C_t,imax_C_t,C_t_conv,dC_t,dC_t[window_len-1:]))
							raise 
						

						# Find the max(dC_t) first instead and then find max C_t after that
						else:
							# Compute average C_t in a running window before each time in the C_t time series 
							C_t_conv =np.convolve(C_t,sum_window,'full')/window_len  
							dC_t= np.diff(C_t_conv)  
							# Note: value before C_t_conv[window_len-1] are not valid. Need to skip them. 
							# Note that imax_dC_t here "includes window_len-1"!!!!
							imax_dC_t= np.argmax(dC_t[window_len-1:]) + window_len-1
							imax_C_t= np.argmax(C_t[imax_dC_t:]) + imax_dC_t							
							

							# The temperol brightness matrix is abtained by the maximum derivative of the brightness
							# at this grid point
							C_temporal[iz,ix,iy]= (imax_dC_t - (window_len-1))* dt

							# For this point, the spacial brightness= max(temperol brightness)
							#C_spatial[iz,ix,iy]= C_t[imax_dC_t]
							C_spatial[iz,ix,iy]= C_t[imax_dC_t:].ravel().max()					
			
		# END OF SCANNNING ############ END OF SCANNNING ############ END OF SCANNNING ##################
		#################################################################################################

		
		
	# Finish scan
	end_scan = datetime.datetime.now()
	print('100% at {}'.format(datetime.datetime.now().strftime("%H:%M:%S")))
	
	elapsed_time = end_scan - start_scan 
	print('Elapsed Time: ', elapsed_time)
	
	print(' ')
	print('Generated brightness 3D matrices: C_spatial, C_temporal')

	# Save data to file #####################################################################
	# Save to a saparate folder 															#
	if save_C == True:																		#
																							#
		# Create the directory if the folder does not exist									#
		if not os.path.exists('save/'):														#
			os.makedirs('save/')															#
																							#
		dir= 'save/{}_C_spatial_{}km_{}s_{}_{}_w{}_{}{}'.format(								#
				no_of_event,dz,dt,cf_filename,method,window_len,phase,add_filename)									#
		np.save(dir, C_spatial)															#
		print('Saved as {}.npy'.format(dir))												#
																							#
		dir= 'save/{}_C_temporal_{}km_{}s_{}_{}_w{}_{}{}'.format(								#
				no_of_event,dz,dt,cf_filename,method,window_len,phase,add_filename)									#
		np.save(dir, C_temporal)															#
		print('Saved as {}.npy'.format(dir))												#
		
		#dir= 'save/{}_C_6D_{}km_{}s_{}_{}_{}{}'.format(								#
		#		no_of_event,dz,dt,cf_filename,method,phase,add_filename)									#
		#np.save(dir, C_6D)															#
		#print('Saved as {}.npy'.format(dir))		
		
																							#
	#########################################################################################
	if alarm == True:
		winsound.Beep(523, 1000)
		winsound.Beep(587, 500)
		winsound.Beep(523, 500)
		
	return C_spatial, C_temporal#, C_6D

	



		
				
def smooth(C_spatial,				
			zmin, zmax, dz,
			xmin, xmax, dx,
			ymin, ymax, dy):
	'''
	Given 3D matrix, smooth the values by taking geometric average of every grid point 
	with the surrounding points.
	_______
	:input: 
		-  C_spatial: 3D matrix
		-  zmin			: Range of possible depth (in km)
		-  zmax
		-  dz			: z interval (ex. dx = 0.5 then resolusion = 0.5		
		-  xmin			: E-W boundaries of the scanning area
		-  xmax			  (in Easting; omit the last 2 digits: ie. dx=100m)
		-  dx			: x interval (ex. dx = 5 then resolusion = 500m)
		-  ymin			: S-N boundaries of the scanning area
		-  ymax			  (in Northing; omit the last 2 digits: ie. dy=100m)
		-  dy			: y interval (ex. dy = 5 then resolusion = 500m)
	________
	:output:
		-  C_smth: smoothed 3D matrix
		
	'''
	nz             = len(	np.arange(zmin,zmax,dz) )+1           # Number of survey grid points in z-direction
	nx             = len(	np.arange(xmin,xmax,dx) )+1           # Number of survey grid points in x-direction
	ny             = len(	np.arange(ymin,ymax,dy) )+1           # Number of survey grid points in y-direction
	
	gpts_in_z      = np.linspace(zmin,zmax,nz,endpoint=True)      # Lining up z axis of the grid 
	gpts_in_x      = np.linspace(xmin,xmax,nx,endpoint=True)      # Lining up x axis of the grid 
	gpts_in_y      = np.linspace(ymin,ymax,ny,endpoint=True)      # Lining up y axis of the grid
	
	
	
	C_smth= C_spatial.copy()

	# Loop over the entire grid
	for z,zz in enumerate(C_smth):
		for x,xx in enumerate(C_smth[z]):
			for y, yy in enumerate(C_smth[z,x]):
			
				# For points inside the matrix (not at the boundaries)
				z1,z2,x1,x2,y1,y2 =1,1,1,1,1,1
				# Dealing with boundaries:
				if z == 0 :
					z2=0
				elif z== nz-1:
					z1=0

				if x == 0 :
					x2=0
				elif x== nx-1:
					x1=0

				if y == 0 :
					y2=0
				elif y== ny-1:
					y1=0
					
				# Smoothing function(3 points at 3D geometric average) 
				C_smth[z,x,y]= (C_spatial[z,x,y]*C_spatial[z+z1,x,y]*C_spatial[z-z2,x,y]
							 *C_spatial[z,x+x1,y]*C_spatial[z,x-x2,y]
							 *C_spatial[z,x,y+y1]*C_spatial[z,x,y-y2])**(1./7.)
							 
							 
	return C_smth


				
def analyse_spatial(stm_processed,C_s,
			zmin, zmax, dz,
			xmin, xmax, dx,
			ymin, ymax, dy):
	'''
	Given spatial brightness matrices, calculate the occurence of maxima brightness on 
	respective xy, yz, xz 2D planes.
	______
	:input: 
		-  stm_processed: Stream (class:obspy.core.stream)
		-  C_s			: Brightness 3D spatial matrix (smooth or unsmoothed)
		-  zmin			: Range of possible depth (in km)
		-  zmax
		-  dz			: z interval (ex. dx = 0.5 then resolusion = 0.5		
		-  xmin			: E-W boundaries of the scanning area
		-  xmax			  (in Easting; omit the last 2 digits: ie. dx=100m)
		-  dx			: x interval (ex. dx = 5 then resolusion = 500m)
		-  ymin			: S-N boundaries of the scanning area
		-  ymax			  (in Northing; omit the last 2 digits: ie. dy=100m)
		-  dy			: y interval (ex. dy = 5 then resolusion = 500m)
		
	______________________
	:globalize and output:
		-  C_xy			: Slice of the occurece of maximum brightness (xy, zy, zx) (2D array)
		-  C_zy			 
		-  C_zx			
		-  coor_max_Cxy	: Coordinates of the occurece of maximum brightness (xy, zy, zx) (2-tuple)
		-  coor_max_Czy	
		-  coor_max_Czx
		-  imax_Cs		: Index of the occurece of maximum brightness (spatial)(3-tuple)
		__________
	:globalize:
		-  imax_Cxy		: Index of the occurece of maximum brightness (xy, zy, zx) (2-tuple)
		-  imax_Czy 
		-  imax_Czx

	
	'''
	nz             = len(	np.arange(zmin,zmax,dz) )+1           # Number of survey grid points in z-direction
	nx             = len(	np.arange(xmin,xmax,dx) )+1           # Number of survey grid points in x-direction
	ny             = len(	np.arange(ymin,ymax,dy) )+1           # Number of survey grid points in y-direction
	
	gpts_in_z      = np.linspace(zmin,zmax,nz,endpoint=True)      # Lining up z axis of the grid 
	gpts_in_x      = np.linspace(xmin,xmax,nx,endpoint=True)      # Lining up x axis of the grid 
	gpts_in_y      = np.linspace(ymin,ymax,ny,endpoint=True)      # Lining up y axis of the grid

	# Get slices on XY, ZX, ZY plane with the maximum brightness value
	global C_xy, C_zy, C_zx

	# Get slices on XY, ZX, ZY plane with the maximum brightness value
	C_xy = np.zeros(shape=(nx,ny))
	for ix in range(nx):
		for iy in range(ny):
		
			C_xy[ix,iy]=max(C_s[:,ix,iy])

	C_zx = np.zeros(shape=(nz,nx))
	for ix in range(nx):
		for iz in range(nz):
		
			C_zx[iz,ix]=max(C_s[iz,ix,:])

	C_zy = np.zeros(shape=(nz,ny))
	for iy in range(ny):
		for iz in range(nz):
		
			C_zy[iz,iy]=max(C_s[iz,:,iy])
	
	'''	
	print('Maximum values projected on xy, zy, zx plane: C_xy, C_zy, C_zx')
	print(' ')		
	'''
	
	# The indices of the occurence of maximum brightness
	global imax_Cs #,imax_Cxy, imax_Czy, imax_Czx

	#imax_Cxy     = unravel_index(C_xy.argmax(), C_xy.shape)
	#imax_Czy     = unravel_index(C_zy.argmax(), C_zy.shape)
	#imax_Czx     = unravel_index(C_zx.argmax(), C_zx.shape)
	imax_Cs      = unravel_index(C_s.argmax(), C_s.shape) 
	
	#print('Index of maximum Cxy[x,y]: imax_Cxy= ', imax_Cxy)
	#print('Index of maximum Czy[z,y]: imax_Czy= ', imax_Czy)
	#print('Index of maximum Czx[z,x]: imax_Czx= ', imax_Czx)
	#print('Index of maximum C[z,x,y]: imax_Cs = ', imax_Cs)
	#print(' ')
	
	
	# Convert to the real coordinates of the occurence of maximum brightness
	global coor_max_Cxy, coor_max_Czy, coor_max_Czx

	#coor_max_Cxy = [xmin + imax_Cxy[0] * dx  ,  ymin + imax_Cxy[1] * dy]
	#coor_max_Czy = [(zmin + imax_Czy[0] * dz) * km_to_100m  ,  ymin + imax_Czy[1] * dy]
	#coor_max_Czx = [(zmin + imax_Czx[0] * dz) * km_to_100m  ,  xmin + imax_Czx[1] * dx]
	coor_max_Cxy = [xmin + imax_Cs[1] * dx  ,  ymin + imax_Cs[2] * dy]
	coor_max_Czy = [(zmin + imax_Cs[0] * dz) * km_to_100m  ,  ymin + imax_Cs[2] * dy]
	coor_max_Czx = [(zmin + imax_Cs[0] * dz) * km_to_100m  ,  xmin + imax_Cs[1] * dx]

	'''
	print('Location of maximum Cxy[x,y]: coor_max_Cxy= ', coor_max_Cxy)
	print('Location of maximum Czy[z,y]: coor_max_Czy= ', coor_max_Czy)
	print('Location of maximum Czx[z,x]: coor_max_Czx= ', coor_max_Czx)
	print(' ')
	'''
	
	return C_xy, C_zy, C_zx, coor_max_Cxy, coor_max_Czy, coor_max_Czx, imax_Cs 

	
	
def analyse_temporal(stm_processed, C_temporal, imax_Cs, tmin):
	'''
	Given the temporal brightness matrices and hypoceter index, calculate event origin time.
	_______
	:input: 
		-  stm_processed: Stream (class:obspy.core.stream)
		-  C_temporal	: Brightness 3D temporal matrix (smooth or unsmoothed)
		-  imax_Cs		: Index of the occurece of maximum brightness (spatial)(3-tuple)
		-  tmin			: Range of possible origin time (in sec after the start time)
		
	______________________
	:globalize and output:
		-  t_eo			: Event origin time (sec after start time + tmin) (float)
	___________
	:globalize:
		-  origin_time_utm: Event origin time (in UTM time)

	
	'''			
	# This gives the event origin time in the form of 'second after tmin'
	global t_eo
	t_eo = C_temporal[imax_Cs]
	print('Event origin time (sec after start time + tmin):  t_eo= ', t_eo)

	# Convert to UTM time to find the event origin time
	global origin_time_utm

	origin_time_utm= stm_processed[0].stats.starttime + tmin + C_temporal[imax_Cs]
	print('Event origin time:  origin_time_utm= ', origin_time_utm)
	'''
	C_xy_t	= C_temporal[int(imax_Cs[0]),:,:]
	C_zy_t	= C_temporal[:,int(imax_Cs[1]),:]
	C_zx_t	= C_temporal[:,:,int(imax_Cs[2])]
	'''
	
	
	
	return t_eo #,C_xy_t,C_zy_t,C_zx_t
	
	

	
				
def analyse_mp(C_6D):
	'''
	Given the 6D matrix, stack from higher to lower dimension to get the 
	marginal probability on xy, yz, xz 2D planes.
	
	'''
	'''
	# Get slices on XY, ZX, ZY plane with the maximum brightness value
	global MP_5D,MP_4D, MP_3D, MP_xy, MP_zy, MP_zx

	# Marginal probability over travel time at stations
	MP_5D	=  np.sum(C_6D, axis=5)
	MP_5D	/= np.sum(MP_5D)
	
	# Marginal probability over travel time at stations
	MP_4D	=  np.sum(C_5D, axis=4)
	MP_4D	/= np.sum(MP_4D)
	
	# Marginal probability over time
	MP_3D	=  np.sum(C_4D, axis=3)
	MP_3D	/= np.sum(MP_3D)
	
	# Marginal probability over z
	MP_xy	=  np.sum(MP_3D, axis=0)
	MP_xy	/= np.sum(MP_xy)
	
	# Marginal probability over y
	MP_zx	=  np.sum(MP_3D, axis=2)
	MP_zx	/= np.sum(MP_zx)
	
	# Marginal probability over x
	MP_zy	=  np.sum(MP_3D, axis=1)
	MP_zy	/= np.sum(MP_zy)

	# Total stack of brightness on xy, yz, and zx planes
	# For xy plane
	MP_x_xy	=  np.sum(MP_xy, axis=1)	# Marginal probability along x axis (stacking over y)
	MP_x_xy	/= np.sum(MP_x_xy)
	MP_y_xy	=  np.sum(MP_xy, axis=0)	# Marginal probability along y axis (stacking over x)
	MP_y_xy	/= np.sum(MP_y_xy)
	# For yz plane
	MP_z_zy	=  np.sum(MP_zy, axis=1)	# Marginal probability along z axis (stacking over y)
	MP_z_zy	/= np.sum(MP_z_zy)
	MP_y_zy	=  np.sum(MP_zy, axis=0)	# Marginal probability along y axis (stacking over z)
	MP_y_zy	/= np.sum(MP_y_zy)
	# For zx plane
	MP_z_zx	=  np.sum(MP_zx, axis=1)	# Marginal probability along z axis (stacking over x)
	MP_z_zx	/= np.sum(MP_z_zx)
	MP_x_zx	=  np.sum(MP_zx, axis=0)	# Marginal probability along x axis (stacking over z)
	MP_x_zx	/= np.sum(MP_x_zx)		
	'''
	
	# Marginal probability over p and s arrivals
	MP_5D	=  np.sum(C_6D, axis=5)
	MP_5D	/= np.sum(MP_5D) 
	
	# Marginal probability over travel time at stations
	MP_4D	=  np.sum(C_5D, axis=4)
	MP_4D	/= np.sum(MP_4D)
	
	# Marginal probability over time
	MP_3D	=  np.sum(C_4D, axis=3)
	MP_3D	/= np.sum(MP_3D)
	
	# Marginal probability over z
	MP_xy	=  np.sum(MP_3D, axis=0)
	MP_xy	/= np.sum(MP_xy)
	
	# Marginal probability over y
	MP_zx	=  np.sum(MP_3D, axis=2)
	MP_zx	/= np.sum(MP_zx)
	
	# Marginal probability over x
	MP_zy	=  np.sum(MP_3D, axis=1)
	MP_zy	/= np.sum(MP_zy)
	
	# Total stack of brightness on xy, yz, and zx planes
	# For xy plane
	MP_x_xy	=  np.sum(MP_xy, axis=1)	# Marginal probability along x axis (stacking over y)
	MP_x_xy	/= np.sum(MP_x_xy)
	MP_y_xy	=  np.sum(MP_xy, axis=0)	# Marginal probability along y axis (stacking over x)
	MP_y_xy	/= np.sum(MP_y_xy)
	# For yz plane
	MP_z_zy	=  np.sum(MP_zy, axis=1)	# Marginal probability along z axis (stacking over y)
	MP_z_zy	/= np.sum(MP_z_zy)
	MP_y_zy	=  np.sum(MP_zy, axis=0)	# Marginal probability along y axis (stacking over z)
	MP_y_zy	/= np.sum(MP_y_zy)
	# For zx plane
	MP_z_zx	=  np.sum(MP_zx, axis=1)	# Marginal probability along z axis (stacking over x)
	MP_z_zx	/= np.sum(MP_z_zx)
	MP_x_zx	=  np.sum(MP_zx, axis=0)	# Marginal probability along x axis (stacking over z)
	MP_x_zx	/= np.sum(MP_x_zx)		
	
	
	# Return results in the order of 1. Axes number of source plot. 
	# 2. horizontal then vertical axis on the plot
	return MP_x_xy, MP_y_xy, MP_z_zy, MP_y_zy, MP_x_zx, MP_z_zx

	
	
	
	

def source_plot(	stm_processed,
					C_xy, C_zy, C_zx,
					coor_max_Cxy, coor_max_Czy, coor_max_Czx,
					zmin, zmax, dz,
					xmin, xmax, dx,
					ymin, ymax, dy,
					method, cf_type_list,
					dt= None, 	
					plot_mp = True, mp_scale=0.1,
					draw_rect=False, 
					plot_ref= '--', figsize=[8,8], vmin=None, #cbar_format='%.2f',
					compare_dtdr=False,
					save_fig=False, no_of_event=None, add_figname='',**kwargs):
	'''
	Plot the brightness function on xy, zy and zx plane with the estimated hypocenter.
	Red cross: Source scanning result
	Black cross: Data in sac attribute (for comparison)
	_______
	:input: 
		-  stm_processed: Stream (class:obspy.core.stream)
		-  C_xy			: Slice of the occurece of maximum brightness (xy, zy, zx) (2D array)
		-  C_zy			 
		-  C_zx			
		-  coor_max_Cxy	: Coordinates of the occurece of maximum brightness (xy, zy, zx) (2-tuple)
		-  coor_max_Czy	
		-  coor_max_Czx	
		-  dt			: Step for scanning specific time (in sec)
		-  zmin			: Range of possible depth (in km)
		-  zmax
		-  dz			: z interval (ex. dx = 0.5 then resolusion = 0.5		
		-  xmin			: E-W boundaries of the scanning area
		-  xmax			  (in Easting; omit the last 2 digits: ie. dx=100m)
		-  dx			: x interval (ex. dx = 5 then resolusion = 500m)
		-  ymin			: S-N boundaries of the scanning area
		-  ymax			  (in Northing; omit the last 2 digits: ie. dy=100m)
		-  dy			: y interval (ex. dy = 5 then resolusion = 500m)
		-  method		: Method of Source Scanning (string) (options: 'L1', 'EDT')
		-  cf_type		: Type of characteristic function used in stacking 
							(options: 'CECM', 'ST*LT', 'RP*LP', 'ST/LT', 'RP/LP')
		-  draw_rect	: Draw rectangles on the map showing the possible area or not (booline)
		-  rect_x		: E-W boundaries of the rectangle (2-list)
		-  rect_y		: S-N boundaries of the rectangle (2-list)
		-  rect_z		: Top-bottom boundaries of the rectangle (2-list)
		-  plot_ref		: Plot reference source location from SAC attributes or not (booline)
							(options:'sac': plot a small cross, 'synthetic': plot horizontal/vertical dashed lines)
		-  save_fig		: Save the generated cf plot or not (booline)
		-  no_of_event	: Number of event (string or int)
		-  filename		: Filename added at the end of the saved figure (string)
		
	______
	:Plot: 
			
		__________
	
	
	
	'''
	
	nz             = len(	np.arange(zmin,zmax,dz) )+1           # Number of survey grid points in z-direction
	nx             = len(	np.arange(xmin,xmax,dx) )+1           # Number of survey grid points in x-direction
	ny             = len(	np.arange(ymin,ymax,dy) )+1           # Number of survey grid points in y-direction
	
	gpts_in_z      = np.linspace(zmin,zmax,nz,endpoint=True)      # Lining up z axis of the grid 
	gpts_in_x      = np.linspace(xmin,xmax,nx,endpoint=True)      # Lining up x axis of the grid 
	gpts_in_y      = np.linspace(ymin,ymax,ny,endpoint=True)      # Lining up y axis of the grid
	
	cf_type, cf_filename	= proper_cf_name(cf_type_list)
	
	

									 

	
	#####################################################################################################
	# Get source location from SAC attribute (if applicable)						 					#
	if plot_ref == '+' or plot_ref == '--':																				#
		evla, evlo, evdp = single_ev_location(stm_processed)    # Event latitude/longitude in SAC attribute 		#
													# (extract from the first trace)					#
		ev_utm		= utm.from_latlon(evla, evlo)	# Convert event location to utm easting/northing	#
		
		sac_easting	= ev_utm[0]/x_to_easting
		sac_northing= ev_utm[1]/y_to_northing
		sac_depth_in_100m = evdp*km_to_100m
		
		
																										#
	#####################################################################################################
	
	
	# Plot the brightness map on XY, ZX and ZY planes

	fig = plt.figure(figsize=figsize)
	# nz*1.05 is for the color bar
	gs 	= gridspec.GridSpec(2, 2,  width_ratios=[nx,nz*1.06], height_ratios=[ny,nz])


	# XY Plane ######### XY Plane ######### XY Plane ######### XY Plane ######### XY Plane ########
	# XY Plane ####################################################################################
	ax1 = plt.subplot(gs[0] )
	
	# Visualize the matrix
	plt.imshow(C_xy.T, extent= [xmin , xmax , ymax , ymin], vmin=vmin)
		
	plt.suptitle('Brightness ({}, {})\nResolution: {}km, {}s'.format(cf_type, method,dz,dt), fontsize=20)
		
	

	# Axis Settings
	#plt.xlabel('Easting (2km/tick)', fontsize= 12)
	plt.ylabel('Northing (2km/tick)', fontsize= 16)
	ax1.tick_params(axis= 'both', labelsize= 10, right= True)
	plt.xticks(np.arange( 10*(xmin//10), xmax+1, 20 ))
	plt.yticks(np.arange( 10*(ymin//10), ymax+1, 20 ))
	ax1.invert_yaxis()
	ax1.set_xticklabels([])  # Disable y-axis label
	

	



	# ZY Plane ######### ZY Plane ######### ZY Plane ######### ZY Plane ######### ZY Plane ########
	# ZY Plane ####################################################################################
	ax2 = plt.subplot(gs[1])#, sharey= ax1)
	
	# Visualize the matrix
	plt.imshow(C_zy.T, extent= [zmin*km_to_100m , zmax*km_to_100m , ymax , ymin],vmin=vmin)


	
	# Axis settings
	plt.xlabel('Depth (2km/tick)', fontsize= 16)
	#plt.ylabel('Northing (2km/tick)', fontsize= 12)
	plt.xticks(np.arange( 10*(zmin*km_to_100m//10), zmax*km_to_100m+1, 20 ))
	plt.yticks(np.arange( 10*(ymin//10), ymax+1, 20 ))
	ax2.tick_params(axis= 'both', labelsize= 10)
	ax2.invert_yaxis()
	ax2.set_yticklabels([])  # Disable y-axis label
	
	# color bar ########## color bar ########## color bar ###############		
	# Set colorbar to have the same height of the subplot 				#		
	ax_divider 	= make_axes_locatable(ax2)								#
	cax			= ax_divider.append_axes("right", size= "5%", pad= 0.05)#
																		#
	cbar		= plt.colorbar(cax= cax)#, format=cbar_format)			#
	cbar.formatter.set_powerlimits((0, 0))								#
	cbar.update_ticks()													#
	cbar.ax.tick_params(labelsize= 12)									#
																		#
	# Set the font size of the power label (such as: 1e-3)				#
	cbar.ax.yaxis.offsetText.set(size=12)								#
	
	cbar.ax.set_title(r'           $\rm P_{max}$')

	#
	# color bar ########## color bar ########## color bar ###############


	# XZ Plane ######### XZ Plane ######### XZ Plane ######### XZ Plane ######### XZ Plane ########
	# XZ Plane ####################################################################################
	ax3 = plt.subplot(gs[2])#, sharex= ax1)
	
	# Visualize the matrix
	plt.imshow(C_zx, extent= [xmin , xmax , zmax*km_to_100m , zmin*km_to_100m], vmin=vmin)



		
	# Axis settings
	plt.xlabel('Easting (2km/tick)', fontsize= 16)
	plt.ylabel('Depth (2km/tick)', fontsize= 16)
	plt.xticks(np.arange( 10*(xmin//10), xmax+1, 20 ))
	plt.yticks(np.arange( 10*(zmin*km_to_100m//10), zmax*km_to_100m+1, 20 ))
	ax3.tick_params(axis= 'both', labelsize= 10, top=True, bottom=True)
	

	

		
	# Fix features overlapping
	# rect =[left, bottom, right, top]
	plt.tight_layout(h_pad=-11, w_pad=1,  rect= [0, -0.01, 1, 0.99])

	
	# For the lower-right plotting space:
	#ax4 = plt.subplot(gs[3])
	
	
	
	
	
	
	# Plot reference location ################################################################
	##########################################################################################
	
	# Plot SAC location (plot a small black cross for reference)
	if plot_ref == '+':
		ax1.plot(sac_easting, sac_northing, 'kP', markersize=8, label= 'Data in SAC Attribute')
		ax2.plot(sac_depth_in_100m, sac_northing, 'kP', markersize=8, label= 'Data in SAC Attribute')
		ax3.plot(sac_easting, sac_depth_in_100m, 'kP', markersize=8, label= 'Data in SAC Attribute')
		
	# Plot synthetic source location (plot dashed horizontal/vertical lines)
	if plot_ref == '--':
		ax1.vlines(x=sac_easting, ymin=ymin, ymax=ymax, linestyles= 'dashed')
		ax1.hlines(y=sac_northing, xmin=xmin, xmax=xmax, linestyles= 'dashed')
		ax2.vlines(x=sac_depth_in_100m, ymin=ymin, ymax=ymax ,linestyles= 'dashed')
		ax2.hlines(y=sac_northing, xmin=zmin*km_to_100m, xmax=zmax*km_to_100m, linestyles= 'dashed')
		ax3.vlines(x=sac_easting, ymin=zmin*km_to_100m, ymax=zmax*km_to_100m, linestyles= 'dashed')
		ax3.hlines(y=sac_depth_in_100m, xmin=xmin, xmax=xmax, linestyles= 'dashed')

	
	
	# marginal probability ########## marginal probability ########## marginal probability ###
	##########################################################################################
	if plot_mp == True:
		try:
			# Compute marginal probability
			C_6D	= kwargs['C_6D']
			MP_x_xy, MP_y_xy, MP_z_zy, MP_y_zy, MP_x_zx, MP_z_zx	= analyse_mp(C_6D=C_6D)
			
			# Scaling for properly fitting in the figure
			xy_m_plot	= MP_x_xy		- min(MP_x_xy)										# Remove min from marginal
			xy_m_plot	= xy_m_plot 	* (ymax-ymin)/max(xy_m_plot) *mp_scale	+	ymin	# Scaling for plotting
			
			yx_m_plot	= MP_y_xy		- min(MP_y_xy)										# Remove min from marginal
			yx_m_plot	= yx_m_plot 	* (xmax-xmin)/max(yx_m_plot) *mp_scale	+	xmin	# Scaling for plotting
			
			xz_m_plot	= MP_x_zx		- min(MP_x_zx)										# Remove min from marginal
			xz_m_plot	= zmax*km_to_100m -  (xz_m_plot 	* (zmax-zmin)/max(xz_m_plot)*km_to_100m *mp_scale)	+	zmin*km_to_100m	# Scaling for plotting
			
			yz_m_plot	= MP_y_zy		- min(MP_y_zy)										# Remove min from marginal
			yz_m_plot	= yz_m_plot 	* (zmax-zmin)/max(yz_m_plot)*km_to_100m *mp_scale	+	zmin*km_to_100m	# Scaling for plotting
			
			zx_m_plot	= MP_z_zx		- min(MP_z_zx)										# Remove min from marginal
			zx_m_plot	= zx_m_plot 	* (xmax-xmin)/max(zx_m_plot) *mp_scale	+	xmin	# Scaling for plotting
			
			zy_m_plot	= MP_z_zy		- min(MP_z_zy)										# Remove min from marginal
			zy_m_plot	= zy_m_plot 	* (ymax-ymin)/max(zy_m_plot) *mp_scale	+	ymin	# Scaling for plotting
			
			
			ax1.plot( gpts_in_x,	xy_m_plot,	color ='w')
			ax1.plot( yx_m_plot,	gpts_in_y, 	color ='w')
			ax2.plot( gpts_in_z*km_to_100m,	zy_m_plot,	color ='w')
			ax2.plot( yz_m_plot,	gpts_in_y, 	color ='w')
			ax3.plot( gpts_in_x,	xz_m_plot,	color ='w')
			ax3.plot( zx_m_plot,	gpts_in_z*km_to_100m, 	color ='w')	
			
			print(MP_x_xy)
			print(MP_y_xy) 
			print(MP_z_zy)
			print(MP_y_zy)
			print(MP_x_zx)
			print(MP_z_zx)
		except:
			print('Marginal plots are not generated. ')
			print('C_6D  =?		(6D np.array)')
			

		

	# Plot rectangles ########################################################################
	##########################################################################################
	# Draw rectengle on the graph
	if draw_rect == True:
		try:
			rect_x, rect_y, rect_z	= kwargs['rect_x'], kwargs['rect_y'], kwargs['rect_z']
			
			#####################################################################################
			# Calculate the height and width of the rectangles drew on the map (if applicable) 	#
			rect_dxy	= [rect_x[1] - rect_x[0],	rect_y[1] - rect_y[0]]
			rect_dzy	= [rect_z[1] - rect_z[0],	rect_y[1] - rect_y[0]]
			rect_dxz	= [rect_x[1] - rect_x[0],	rect_z[1] - rect_z[0]]

			#####################################################################################			
			
			rect= patches.Rectangle(( rect_x[0],  rect_y[0]), rect_dxy[0], rect_dxy[1], 
									linewidth= 1, edgecolor= 'm', facecolor= 'none')
			ax1.add_patch(rect)
		
			rect= patches.Rectangle(( rect_z[0],  rect_y[0]), rect_dzy[0], rect_dzy[1], 
									linewidth= 1, edgecolor= 'm', facecolor= 'none')
			ax2.add_patch(rect)
		
			rect= patches.Rectangle(( rect_x[0],  rect_z[0]), rect_dxz[0], rect_dxz[1], 
									linewidth= 1, edgecolor= 'm', facecolor= 'none')
			ax3.add_patch(rect)
			
		except:
			print('Rectangles are not plotted. ')
			print('Need rect_x(2-tuple), rect_y(2-tuple), rect_z2(2-tuple)')
			pass

	# Plot the hypocenter markers ############################################################
	##########################################################################################		
	ax1.plot(coor_max_Cxy[0], coor_max_Cxy[1], 'rP', markersize=8, label= 'Source Scanning Result')		
	ax2.plot(coor_max_Czy[0], coor_max_Czy[1], 'rP', markersize=8, label= 'Source Scanning Result')		
	ax3.plot(coor_max_Czx[1], coor_max_Czx[0] ,'rP', markersize=8, label= 'Source Scanning Result')
	

					
		
	
	# compare ######### compare ######### compare ######### compare ######### compare ########
	# compare ################################################################################
	if compare_dtdr == True:
		try:
			tmin		= kwargs['tmin']
			t_eo		= kwargs['t_eo']
			utm_zone	= kwargs['utm_zone']
			trim_start	= kwargs['trim_start']
			
			
			diff_in_t, diff_in_s	= compare(	
							stm_processed=stm_processed,
							C_xy=C_xy, C_zy=C_zy, C_zx=C_zx,
							coor_max_Cxy=coor_max_Cxy, 
							coor_max_Czy=coor_max_Czy, 
							coor_max_Czx=coor_max_Czx,
							tmin= tmin, t_eo=t_eo, utm_zone= utm_zone, trim_start= trim_start)		
			
			diff_z, diff_x, diff_y, dr = diff_in_s[0], diff_in_s[1], diff_in_s[2], diff_in_s[3]
			dt_eo					  = diff_in_t
			
			plt.figtext(0.5, 0.03, 
					r'$\rm dt_{eo}$'+'={:.1f}s, dr={:.1f}km       (dx, dy, dz) = ({:.1f}, {:.1f}, {:.1f})km'
					.format(dt_eo,dr, diff_x,diff_y,diff_z), 
					horizontalalignment='center', fontsize=14)
		except:
			print('Comparing results failed.')
			print('Need tmin(float), t_eo(float), utm_zone(ex. (27,"V")), trim_start(float).')
			

	# Save ############ Save ############# Save ############# Save ##########
	# Save to a saparate folder 											#
	if save_fig == True:													#
		# Create the directory if the folder does not exist					#
		if not os.path.exists('save/figs'):									#
			os.makedirs('save/figs')										#
																			#
																			#
		dir_s= 'save/figs/{}_src_plot_{}_{}__res_{}km_{}s{}.png'.format(	#
				no_of_event, cf_filename, method, dz, dt,add_figname )				#
		plt.savefig(dir_s)													#
		print('Saved as {}'.format(dir_s))									#
																			#
	# Save ############# Save ############# Save ############# Save #########
	
	#################################################################################################
	


	
'''	
def source_plot2(	stm_processed,
					channel,
					C_xy, C_zy, C_zx,
					coor_max_Cxy, coor_max_Czy, coor_max_Czx,
					zmin, zmax, dz,
					xmin, xmax, dx,
					ymin, ymax, dy,
					method,
					dt= None,
					draw_rect=False, rect_x=[4445,4475], rect_y=[70840,70875], rect_z= [25,70],
					plot_sac= True, figsize=[8,8], vmin=None		):
	
'''
'''
	Plot the brightness function on xy, zy and zx plane with the estimated hypocenter.
	Red cross: Source scanning result
	Black cross: Data in sac attribute (for comparison)
	_______
	:input: 
		-  stm_processed: Stream (class:obspy.core.stream)
		-  C_xy			: Slice of the occurece of maximum brightness (xy, zy, zx) (2D array)
		-  C_zy			 
		-  C_zx			
		-  coor_max_Cxy	: Coordinates of the occurece of maximum brightness (xy, zy, zx) (2-tuple)
		-  coor_max_Czy	
		-  coor_max_Czx	
		-  dt			: Step for scanning specific time (in sec)
		-  zmin			: Range of possible depth (in km)
		-  zmax
		-  dz			: z interval (ex. dx = 0.5 then resolusion = 0.5		
		-  xmin			: E-W boundaries of the scanning area
		-  xmax			  (in Easting; omit the last 2 digits: ie. dx=100m)
		-  dx			: x interval (ex. dx = 5 then resolusion = 500m)
		-  ymin			: S-N boundaries of the scanning area
		-  ymax			  (in Northing; omit the last 2 digits: ie. dy=100m)
		-  dy			: y interval (ex. dy = 5 then resolusion = 500m)
		-  draw_rect	: Draw rectangles on the map showing the possible area or not (booline)
		-  rect_x		: E-W boundaries of the rectangle (2-list)
		-  rect_y		: S-N boundaries of the rectangle (2-list)
		-  rect_z		: Top-bottom boundaries of the rectangle (2-list)
		-  plot_sac		: Plot reference source location from SAC attributes or not (booline)
		
	______
	:Plot: 
			
		__________
	
	
'''
'''
	
	nz             = len(	np.arange(zmin,zmax,dz) )+1           # Number of survey grid points in z-direction
	nx             = len(	np.arange(xmin,xmax,dx) )+1           # Number of survey grid points in x-direction
	ny             = len(	np.arange(ymin,ymax,dy) )+1           # Number of survey grid points in y-direction
	
	gpts_in_z      = np.linspace(zmin,zmax,nz,endpoint=True)      # Lining up z axis of the grid 
	gpts_in_x      = np.linspace(xmin,xmax,nx,endpoint=True)      # Lining up x axis of the grid 
	gpts_in_y      = np.linspace(ymin,ymax,ny,endpoint=True)      # Lining up y axis of the grid
									 
	#####################################################################################
	# Calculate the height and width of the rectangles drew on the map (if applicable) 	#
	if draw_rect==True:
		rect_dxy	= [rect_x[1] - rect_x[0],	rect_y[1] - rect_y[0]]
		rect_dzy	= [rect_z[1] - rect_z[0],	rect_y[1] - rect_y[0]]
		rect_dxz	= [rect_x[1] - rect_x[0],	rect_z[1] - rect_z[0]]

	#####################################################################################
	
	#####################################################################################################
	# Get source location from SAC attribute (if applicable)						 					#
	if plot_sac == True:																				#
		evla, evlo, evdp = single_ev_location(stm_processed)    # Event latitude/longitude in SAC attribute 		#
													# (extract from the first trace)					#
		ev_utm = utm.from_latlon(evla, evlo)		# Convert event location to utm easting/northing	#
																										#
	#####################################################################################################
	
	
	
	
	# Plot the brightness map on XY, ZX and ZY planes

	fig = plt.figure(figsize=figsize)
	# nz*1.05 is for the color bar
	gs 	= gridspec.GridSpec(3, 3,  width_ratios=[nx,nx/5,nz*1.05], height_ratios=[ny/5,ny,nz])


	# XY Plane ######### XY Plane ######### XY Plane ######### XY Plane ######### XY Plane ########
	# XY Plane ####################################################################################
	ax1 = plt.subplot(gs[3] )
	
	# Visualize the matrix
	plt.imshow(C_xy.T, extent= [xmin , xmax , ymax , ymin], vmin=vmin)
		
	# For P scanning
	if dt!= None:
		plt.suptitle('Brightness ({}) \n Resolution: {}km, {}s'.format(method,dz,dt), fontsize=20)
	# For S scanning
	else:
		plt.suptitle('Brightness ({}, S{}) \n Resolution: {}km'.format(method,channel,dz), fontsize=20)
		
	#plt.title('XY')

	# Plot the hypocenter makers
	plt.plot(coor_max_Cxy[0], coor_max_Cxy[1], 'r+',  markersize=8, label= 'Source Scanning Result')
	
	if plot_sac == True:
	
		plt.plot(ev_utm[0]/x_to_easting, ev_utm[1]/y_to_northing, 'k+', markersize=8, label= 'Data in SAC Attribute')

	# Axis Settings
	#plt.xlabel('Easting (2km/tick)', fontsize= 12)
	plt.ylabel('Northing (2km/tick)', fontsize= 16)
	ax1.tick_params(axis= 'both', labelsize= 10, right= True)
	plt.xticks(np.arange( xmin, xmax+1, 20 ))
	plt.yticks(np.arange( ymin, ymax+1, 20 ))
	ax1.invert_yaxis()
	ax1.set_xticklabels([])  # Disable x-axis label
	

	# Set colorbar to have the same height of the subplot
	#ax_divider 	= make_axes_locatable(ax1)
	#cax			= ax_divider.append_axes("right", size= "5%", pad= 0.05)
	#cbar		= plt.colorbar(cax= cax, format='%.2f')
	#cbar.ax.tick_params(labelsize= 10)
	
	# Draw rectengle on the graph
	if draw_rect == True:
	
		rect= patches.Rectangle(( rect_x[0],  rect_y[0]), rect_dxy[0], rect_dxy[1], 
								linewidth= 1, edgecolor= 'm', facecolor= 'none')
		ax1.add_patch(rect)
	
	
	
	# Marginal probability
	ax1_x = plt.subplot(gs[0] )
	
	integrate_x = np.zeros(shape=nx)
	x
	
	

	# ZY Plane ######### ZY Plane ######### ZY Plane ######### ZY Plane ######### ZY Plane ########
	# ZY Plane ####################################################################################
	ax2 = plt.subplot(gs[5])#, sharey= ax1)
	
	# Visualize the matrix
	plt.imshow(C_zy.T, extent= [zmin*km_to_100m , zmax*km_to_100m , ymax , ymin],vmin=vmin)
	#plt.title('YZ')

	# Plot the hypocenter makers
	plt.plot(coor_max_Czy[0], coor_max_Czy[1], 'r+', markersize=8, label= 'Source Scanning Result')
	
	if plot_sac == True:
		plt.plot(evdp*km_to_100m, ev_utm[1]/y_to_northing, 'k+', markersize=8, label= 'Data in SAC Attribute')

	# Axis settings
	plt.xlabel('Depth (2km/tick)', fontsize= 16)
	#plt.ylabel('Northing (2km/tick)', fontsize= 12)
	plt.xticks(np.arange( zmin*km_to_100m, zmax*km_to_100m+1, 20 ))
	ax2.tick_params(axis= 'both', labelsize= 10)
	ax2.invert_yaxis()
	ax2.set_yticklabels([])  # Disable y-axis label
	
	
	# Set colorbar to have the same height of the subplot
	ax_divider 	= make_axes_locatable(ax2)
	cax			= ax_divider.append_axes("right", size= "5%", pad= 0.05)
	cbar		= plt.colorbar(cax= cax, format='%.2f')
	cbar.ax.tick_params(labelsize= 12)
	
	# Draw rectengle on the graph
	if draw_rect == True:
	
		rect= patches.Rectangle(( rect_z[0],  rect_y[0]), rect_dzy[0], rect_dzy[1], 
								linewidth= 1, edgecolor= 'm', facecolor= 'none')
		ax2.add_patch(rect)

	
	
	# XZ Plane ######### XZ Plane ######### XZ Plane ######### XZ Plane ######### XZ Plane ########
	# XZ Plane ####################################################################################
	ax3 = plt.subplot(gs[6])#, sharex= ax1)
	
	# Visualize the matrix
	plt.imshow(C_zx, extent= [xmin , xmax , zmax*km_to_100m , zmin*km_to_100m], vmin=vmin)
	#plt.title('XZ')

	# Plot the hypocenter makers
	plt.plot(coor_max_Czx[1], coor_max_Czx[0] ,'r+', markersize=8, label= 'Source Scanning Result')
	
	if plot_sac == True:
	
		plt.plot(ev_utm[0]/x_to_easting, evdp*km_to_100m, 'k+', markersize=8, label= 'Data in SAC Attribute')

	# Axis settings
	plt.xlabel('Easting (2km/tick)', fontsize= 16)
	plt.ylabel('Depth (2km/tick)', fontsize= 16)
	plt.xticks(np.arange( xmin, xmax+1, 20 ))
	plt.yticks(np.arange( zmin*km_to_100m, zmax*km_to_100m+1, 20 ))
	ax3.tick_params(axis= 'both', labelsize= 10, top=True, bottom=True)
	
	# Set colorbar to have the same height of the subplot
	#ax_divider 	= make_axes_locatable(ax3)
	#cax			= ax_divider.append_axes("right", size= "5%", pad= 0.05)
	#cbar		= plt.colorbar(cax= cax, format='%.2f')
	#cbar.ax.tick_params(labelsize= 8)
	
	# Draw rectengle on the graph
	if draw_rect == True:
	
		rect= patches.Rectangle(( rect_x[0],  rect_z[0]), rect_dxz[0], rect_dxz[1], 
								linewidth= 1, edgecolor= 'm', facecolor= 'none')
		ax3.add_patch(rect)
		
	# Fix features overlapping
	# rect =[left, bottom, right, top]
	plt.tight_layout(h_pad=-13, w_pad=1,  rect= [0, 0, 1, 1])

	
	# For the lower-right plotting space:
	#ax4 = plt.subplot(gs[3])
	

	#################################################################################################
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
'''	
	
'''
def marginal_prob():
'''
'''
	import numpy as np
	import pandas as pd
	from scipy import stats, integrate
	import matplotlib.pyplot as plt
	import seaborn as sns
	
	mean, cov = [0, 1], [(1, .5), (.5, 1)]
	data = np.random.multivariate_normal(mean, cov, 200)
	df = pd.DataFrame(data, columns=["x", "y"])
	
	sns.jointplot(x="x", y="y", data=df, kind="kde")
	
	
	# Use the original matplotlib colors
	sns.reset_orig()





'''






def compare(	stm_processed,
					t_eo,
					C_xy, C_zy, C_zx,
					coor_max_Cxy, coor_max_Czy, coor_max_Czx,
					tmin,
					utm_zone,
					trim_start, save=False,**kwargs):
	'''
	______
	:input: 
		-  stm_processed: Stream (class:obspy.core.stream)
		-  t_eo			: Event origin time (sec after start time + tmin) (float)
		-  C_xy			: Slice of the occurece of maximum brightness (xy, zy, zx) (2D array)
		-  C_zy			 
		-  C_zx			
		-  coor_max_Cxy	: Coordinates of the occurece of maximum brightness (xy, zy, zx) (2-tuple)
		-  coor_max_Czy	
		-  coor_max_Czx
		-  tmin			: Range of possible origin time (in sec after the start time)
		-  utm_zone		: The utm_zone of the scanning grid (ex. (27,'V))
		-  trim_start	: The amount of time the stream has been trimed in the signal_processing step (s) (float)
		
	______________________
	:output:
		-  src_SSA : Source coordinates derived from SSA (3-list)
		-  src_SAC : Source coordinates derived from SAC (3-list)

	'''
	
	
	
	# Hypocenter location determined by Source Scanning 
	src_SSA		= [coor_max_Czy[0]/km_to_100m, coor_max_Cxy[0]*x_to_km, coor_max_Cxy[1]*y_to_km]
	src_lalo	= utm.to_latlon(src_SSA[1]/easting_to_km, src_SSA[2]/northing_to_km, utm_zone[0], utm_zone[1])
	
	# by SAC attribute
	evla, evlo, evdp = single_ev_location(stm_processed)    # Event latitude/longitude in SAC attribute 		
															# (extract from the first trace)					
	ev_utm = utm.from_latlon(evla, evlo)					# Convert event location to utm easting/northing	
	src_SAC		= [evdp, ev_utm[0]*easting_to_km, ev_utm[1]*northing_to_km]

	# The event origin time (from source scanning and SAC attributes)
	# 'C_temporal[imax_Cs]'==  the event origin time in the form of 'second after tmin' 
	ot_SSA	= t_eo + tmin + trim_start
	ot_SAC	= stm_processed[0].stats.sac['o']
	
	'''
	print('Source Scanning Result: {:.1f}s, {:.1f}km, {:.1f}km, {:.1f}km  {}'
		  .format(ot_SSA, src_SSA[0], src_SSA[1], src_SSA[2], src_lalo))
	print('Data in SAC headers:    {:.1f}s, {:.1f}km, {:.1f}km, {:.1f}km  ({}, {})'
		  .format(ot_SAC, src_SAC[0], src_SAC[1], src_SAC[2], evla, evlo))
	'''	  
	# Calculate the difference between SSA result and SAC 	  
	diff_in_s = np.subtract(src_SSA,src_SAC)  # diff_in_s has the form: z,x,y
	diff_in_t = ot_SSA-ot_SAC
	dr=np.sqrt(diff_in_s[0]**2+diff_in_s[1]**2+diff_in_s[2]**2)
	diff_in_s=np.append(diff_in_s,dr)
	'''
	print('Differences:')
	print('dt= {:.1f}s, dz= {:.1f}km, dx= {:.1f}km, dy= {:.1f}km'
		  .format(diff_in_t, diff_in_s[0], diff_in_s[1], diff_in_s[2]))

	'''
	
	if save == True:
		try:
			no_of_event,method,cf_type,filename, first_cf_type,add_title= kwargs['no_of_event'],kwargs['method'],kwargs['cf_type'],kwargs['filename'], kwargs['first_cf_type'],kwargs['add_title']
			result = open('{}.txt'.format(filename),'a')
			if cf_type=='{}'.format(first_cf_type):
				result.write('{}{}{}\n'.format(method,add_title,no_of_event))
				result.write('  dt  dr  dx  dy  dz\n')
			result.write('{0:>4.1f}{1:>4.1f}{2:>4.1f}{3:>4.1f}{4:>4.1f}{5:>12}\n'.format(
						diff_in_t, diff_in_s[3],diff_in_s[1],diff_in_s[2],diff_in_s[0],cf_type)) 
			result.close() 
		except:
			print('Results are not saved.')
			print('Need no_of_event(string), method(string), cf_type(string),first_cf_type(string), filename(string)')
			print('Please create a .txt file manually in the main directory.')
			pass

	return diff_in_t,  diff_in_s


	
	

def compare_detail(	stm_processed,
					t_eo,
					C_xy, C_zy, C_zx,
					coor_max_Cxy, coor_max_Czy, coor_max_Czx,
					tmin,
					utm_zone,
					trim_start):
	'''
	______
	:input: 
		-  stm_processed: Stream (class:obspy.core.stream)
		-  t_eo			: Event origin time (sec after start time + tmin) (float)
		-  C_xy			: Slice of the occurece of maximum brightness (xy, zy, zx) (2D array)
		-  C_zy			 
		-  C_zx			
		-  coor_max_Cxy	: Coordinates of the occurece of maximum brightness (xy, zy, zx) (2-tuple)
		-  coor_max_Czy	
		-  coor_max_Czx
		-  tmin			: Range of possible origin time (in sec after the start time)
		-  utm_zone		: The utm_zone of the scanning grid (ex. (27,'V))
		-  trim_start	: The amount of time the stream has been trimed in the signal_processing step (s) (float)
		
	______________________
	:output:
		-  src_SSA : Source coordinates derived from SSA (3-list)
		-  src_SAC : Source coordinates derived from SAC (3-list)

	'''
	
	
	
	# Hypocenter location determined by Source Scanning 
	src_SSA		= [coor_max_Czy[0]/km_to_100m, coor_max_Cxy[0]*x_to_km, coor_max_Cxy[1]*y_to_km]
	src_lalo	= utm.to_latlon(src_SSA[1]/easting_to_km, src_SSA[2]/northing_to_km, utm_zone[0], utm_zone[1])
	
	# by SAC attribute
	evla, evlo, evdp = single_ev_location(stm_processed)    # Event latitude/longitude in SAC attribute 		
															# (extract from the first trace)					
	ev_utm = utm.from_latlon(evla, evlo)					# Convert event location to utm easting/northing	
	src_SAC		= [evdp, ev_utm[0]*easting_to_km, ev_utm[1]*northing_to_km]

	# The event origin time (from source scanning and SAC attributes)
	# 'C_temporal[imax_Cs]'==  the event origin time in the form of 'second after tmin' 
	ot_SSA	= t_eo + tmin + trim_start
	ot_SAC	= stm_processed[0].stats.sac['o']
	
	
	# Convert to UTM time to find the  event origin time
	origin_time_utm= stm_processed[0].stats.starttime + tmin + t_eo
	print('Event origin time: ',origin_time_utm)
		
	print('Source Scanning Result: {:.1f}s, {:.1f}km, {:.1f}km, {:.1f}km  {}'
		  .format(ot_SSA, src_SSA[0], src_SSA[1], src_SSA[2], src_lalo))
	print('Data in SAC headers:    {:.1f}s, {:.1f}km, {:.1f}km, {:.1f}km  ({}, {})'
		  .format(ot_SAC, src_SAC[0], src_SAC[1], src_SAC[2], evla, evlo))
		  
	# Calculate the difference between SSA result and SAC 	  
	diff_in_s = np.subtract(src_SSA,src_SAC)
	diff_in_t = ot_SSA-ot_SAC
	
	print('Differences:')
	print('dt= {:.1f}s, dz= {:.1f}km, dx= {:.1f}km, dy= {:.1f}km'
		  .format(diff_in_t, diff_in_s[0], diff_in_s[1], diff_in_s[2]))
	
	return src_SSA, src_SAC

	
	
	
	
def st_basic_info(stm, channel, src_lalo, src_z ,TTModel):
	'''
	Generate a list of basic information of each stations given the event geometry of 
	station and source locations.
	Basic information includes: station name, distance from the epicenter, travel time,
	sampling rate.
	Used in waveform_plot and artificial_stream.
	______
	:input: 
		-  stm			: Stream (class:obspy.core.stream). It doesn't matter whether the stm has been processed or not.
		-  channel		: Channel wanted to extract from (option: 'E', 'N', 'Z')
		-  src_lalo		: Source latitude and longitude (2-list)
		-  src_z		: Source depth (unit in km)(float)
		-  TTModel		: The 2D velocity model (2D array)
	________
	:output:
		- A list of sublists sorted by station names. 
			The indivisual sublist has the form: [st_name, distance, travel_time, sampling_rate, npts]

		
	'''
	if channel in ['E','N','Z']:
		
		# Information in SAC headers
		st_lalo          = st_locations(stm)				# Stations latitude/longitude in SAC attribute
		nst              = len(stm)//3					# Number of stations
		
		sampling_rate	= []											# Sampling rate
		npts			= []	
		for tr in range(len(stm)):
			if stm[tr].stats.channel ==channel:
				sampling_rate.append(stm[tr].stats.sampling_rate)
				npts.append(stm[tr].stats.npts)
		
		# Building travel time interpolation between model grid points
		Travel_Time_Interp =     Travel_Time_Model(TTModel)
		
		info	=[]
		for i in range(nst):
			st_name		= st_lalo[2][i][:3]+ channel   # Get the first 3 charactors for st name, 1 charactor for channel name.
			distance	= calc_vincenty_inverse(st_lalo[0][i], st_lalo[1][i], src_lalo[0], src_lalo[1])[0]  # distance is in m
			travel_time = Travel_Time_Interp(src_z, distance*m_to_km)  # Based on TTModel # src_z is in km
			info.append([st_name, distance, travel_time, sampling_rate[i], npts[i]])
	
	# If want information for all channels
	else:
		# Information in SAC headers
		#st_lalo          = st_locations(stm)				# Stations latitude/longitude in SAC attribute
		stlo=[]                                 # Station location (longitude)
		stla=[]                                 # Station location (latitude)      
		st_name=[]                                # Station name
		for tr in stm:            
			stla[len(stla):] = [tr.stats.sac['stla']]  
			stlo[len(stlo):] = [tr.stats.sac['stlo']]
			st_name[len(stlo):] = [tr.stats['station']]
		st_lalo =[stla,stlo,st_name]
		
		nst              = len(stm)	#//3				# Number of stations
		
		sampling_rate	= []											# Sampling rate
		npts			= []	
		for tr in range(len(stm)):
			#if stm[tr].stats.channel ==channel:
			sampling_rate.append(stm[tr].stats.sampling_rate)
			npts.append(stm[tr].stats.npts)
		
		
		# Building travel time interpolation between model grid points
		Travel_Time_Interp =     Travel_Time_Model(TTModel)
		
		info	=[]
		for i in range(len(stm)):
			st_name		= st_lalo[2][i]  #[:3]+ channel  
			distance	= calc_vincenty_inverse(st_lalo[0][i], st_lalo[1][i], src_lalo[0], src_lalo[1])[0]  # distance is in m
			travel_time = Travel_Time_Interp(src_z, distance*m_to_km)  # Based on TTModel # src_z is in km
			
			info.append([st_name, distance, travel_time, sampling_rate[i], npts[i]])		
		
				
	return info
		
			



	

def waveform_plot(stm_processed, duration, 
						TTModel, 
						trim_start=None,
						tmin=None, t_eo=None,  utm_zone=None, 
						channel='Z',figsize=[7,11],amp=0.6,color_dict=None, TTModel_2=None, 
						save_fig=False, no_of_event= None,cf_type_list=None,method =None, add_figname='',
						title_size=24, xy_label_size=20,add_v_sep=0,
						tt_marker='k+',tt_mk_size=6,tt_marker_sep=0.2):
	'''
	Plot the waveforms as a function of increasing distance from the SSA hypocenter.
	Mark the theoretical travel time as black +.
	______
	:input: 
		-  stm_processed: Stream (class:obspy.core.stream)
		-  duration		: The x (time) duration on the plot. (Few sec after the last arrival) (sec) (float)
		-  TTModel		: The 2D velocity model (2D array)
		-  tmin
		-  utm_zone
		-  channel		: Channel wanted to extract the waveform from (option: 'E', 'N', 'Z')
		-  t_eo			: Event origin time (sec after start time + tmin) (float)
		
		-  figsize		: The output figure size. Might change due to the number of stations in the stream (2-list)
		-  amp			: Amplitude of the waveform (float)
		-  color_dict	: Color dictionary for stations (dict)
		-  save_fig		: Save the generated cf plot or not (booline)
		-  no_of_event	: Number of event (string or int)
		-  cf_type		: Type of characteristic function used in stacking 
							(options: 'CECM', 'ST*LT', 'RP*LP', 'ST/LT', 'RP/LP')
		-  method		: Method of Source Scanning (string) (options: 'L1', 'EDT')
	______________________
	:Plot:
		- Plot starts from event origin time 
		- Black +	: Theoretical travel time by the 2D velocity model 
		
	'''
	try:
		cf_type, cf_filename	= proper_cf_name(cf_type_list)
	except:
		cf_type, cf_filename	= '',''

	# Get the waveform from stm and trim it (need only event origin~ a few sec after the last arrival)
	starttime=stm_processed[0].stats.starttime
	stm= obspy.Stream()
	for tr in stm_processed:
		if tr.stats.channel== channel:
			stm.append(tr.copy())
		
	if method == 'SAC':
		ot_SAC= stm_processed[0].stats.sac['o']						# Event origin time
		evla, evlo, src_z = single_ev_location(stm_processed)   		# Event latitude/longitude in SAC attribute 		
		src_lalo=[evla, evlo]
		endtime		= starttime+ot_SAC-trim_start+duration
		starttime +=ot_SAC-trim_start

		
	else:
		# Hypocenter location determined by Source Scanning
		src_z		= coor_max_Czy[0]/km_to_100m
		#src_SSA		= [coor_max_Czy[0]/km_to_100m, coor_max_Cxy[0]*x_to_km, coor_max_Cxy[1]*y_to_km]
		src_lalo	= utm.to_latlon( coor_max_Cxy[0]*x_to_km/easting_to_km,  coor_max_Cxy[1]*y_to_km/northing_to_km, utm_zone[0], utm_zone[1])
		endtime		=  starttime+tmin+t_eo+duration
		starttime	+= tmin+t_eo

			
	stm= stm.trim(starttime=starttime, endtime=endtime)

	# Create a list to store all information categorized by stations
	st_info	= st_basic_info(stm_processed, channel, src_lalo, src_z,TTModel)
	
	# If given TTModel_2
	try:
		st_info_2=st_basic_info(stm_processed, channel, src_lalo, src_z, TTModel_2)
	except:
		pass
	# st_info has the form: [st_name, distance, travel_time, sampling_rate, npts]
	
	
	# Append waveform data to the list
	for i, st in enumerate(st_info):
		st.append(stm[i])

	# Sort the list by distance
	st_info_sorted=sorted(st_info, key=lambda x: x[1])  # st_info[:,1]: x distance between station and source
	# If given TTModel_2
	try:
		st_info_2_sorted=sorted(st_info_2, key=lambda x: x[1])
	except:
		pass
	# Put info into separate lists for plotting
	st_name		=[]     # Station name
	dis   		=[]     # x distance between station and source 
	tt	     	=[]		# Travel time by the velocity model
	spl_rate	=[] 	# Sampling rate
	waveform   	=[]     # Waveform
	tt_2	    =[]		# Travel time by the velocity model_2
	
	for i in range(len(st_info)):  
	
		st_name.append(st_info_sorted[i][0])
		dis.append(st_info_sorted[i][1])
		tt.append(st_info_sorted[i][2])
		spl_rate.append(st_info_sorted[i][3])
		waveform.append(st_info_sorted[i][5])
		# If given TTModel_2
		try:
			tt_2.append(st_info_2_sorted[i][2])
		except:
			pass
		
	# Normalize the waveform   
	waveform= np.asarray(waveform)
	wf_norm= waveform.copy() 
	wf_norm_original= waveform.copy() 
	
	for i in range(len(wf_norm)):
		# For calculating vertical_sep
		wf_norm_original[i] = wf_norm[i]/ (wf_norm[i].max())
		# For plotting
		wf_norm[i]			= wf_norm_original[i]*amp

		
	# Plot the waveforms by distances from the epicenter #########################################################
	fig, ax = plt.subplots(figsize=figsize)
	
	# Determine the verticle seperation in the graph
	vertical_sep = wf_norm_original.ravel().max() + add_v_sep

	itime = np.arange(len(wf_norm[0]))
	
	for i,st in enumerate(wf_norm):
	
		# Line seimograms up with verticle seperation= vertical_sep
		y= st  + vertical_sep *(len(wf_norm)-i)
		seismogram, = plt.plot(itime/spl_rate[i], y )
		
		# Set seismogram color according to stations
		try:	
			seismogram.set_color(color_dict[st_name[i][:3]])
		except:
			pass
			
		plt.plot(tt[i], vertical_sep *(len(wf_norm)-i)+tt_marker_sep, tt_marker, markersize=tt_mk_size)
		# If given TTModel_2
		try:
			plt.plot(tt_2[i], vertical_sep *(len(wf_norm)-i)+tt_marker_sep, tt_marker, markersize=tt_mk_size)
		except:
			pass
		ax.annotate(st_name[i],(-0.1,vertical_sep *(len(wf_norm)-i),),xytext=(0,10), textcoords='offset points', size=14)
		
	plt.xlabel('Time (s)', fontsize=xy_label_size)
	plt.ylabel('Distances increasing downward\n(not to scale)', fontsize=xy_label_size)
	if method == 'SAC':
		plt.title('Normalized Seismogram\n ({})\n'.format(method), fontsize=title_size)
	else:
		plt.title('Normalized Seismogram\n ({}, {})\n'.format(cf_type, method), fontsize=title_size)	
	ax.set_yticklabels([])  # Disable y-axis label
	ax.tick_params(axis= 'x', labelsize= 16)
	
	plt.tight_layout()
	
	# Save ############# Save ############# Save ############# Save #
	# Save to a saparate folder 									#
	if save_fig == True:											#
		# Create the directory if the folder does not exist			#
		if not os.path.exists('save/figs'):							#
			os.makedirs('save/figs')								#
																	#									#			
		if method == 'SAC':															#
			dir_s= 'save/figs/{}_wf{}{}.png'.format(				#
					no_of_event, method, add_figname )			#
		else:	
			dir_s= 'save/figs/{}_wfSSA_{}_{}{}.png'.format(				#
					no_of_event, cf_filename, method, add_figname )			#
		plt.savefig(dir_s)											#
		print('Saved as {}'.format(dir_s))							#
																	#
	# Save ############# Save ############# Save ############# Save #
	
	

def waveform_plot_all(stm_processed, duration, 
						TTModel, 
						trim_start=None,
						tmin=None, t_eo=None,  utm_zone=None, 
						channel='Z',figsize=[7,11],amp=0.6,color_dict=None, TTModel_2=None, 
						save_fig=False, no_of_event= None,cf_type_list=None,method =None, add_figname='',
						title_size=24, xy_label_size=20,add_v_sep=0,
						tt_marker='k+',tt_mk_size=6,tt_marker_sep=0.2,plot_start=-1.2):
	'''
	Plot the waveforms as a function of increasing distance from the SSA hypocenter.
	Mark the theoretical travel time as black +.
	______
	:input: 
		-  stm_processed: Stream (class:obspy.core.stream)
		-  duration		: The x (time) duration on the plot. (Few sec after the last arrival) (sec) (float)
		-  TTModel		: The 2D velocity model (2D array)
		-  tmin
		-  utm_zone
		-  channel		: Channel wanted to extract the waveform from (option: 'E', 'N', 'Z')
		-  t_eo			: Event origin time (sec after start time + tmin) (float)
		
		-  figsize		: The output figure size. Might change due to the number of stations in the stream (2-list)
		-  amp			: Amplitude of the waveform (float)
		-  color_dict	: Color dictionary for stations (dict)
		-  save_fig		: Save the generated cf plot or not (booline)
		-  no_of_event	: Number of event (string or int)
		-  cf_type		: Type of characteristic function used in stacking 
							(options: 'CECM', 'ST*LT', 'RP*LP', 'ST/LT', 'RP/LP')
		-  method		: Method of Source Scanning (string) (options: 'L1', 'EDT')
	______________________
	:Plot:
		- Plot starts from event origin time 
		- Black +	: Theoretical travel time by the 2D velocity model 
		
	'''
	try:
		cf_type, cf_filename	= proper_cf_name(cf_type_list)
	except:
		cf_type, cf_filename	= '',''

	
	starttime=stm_processed[0].stats.starttime
	
	if method == 'SAC':
		ot_SAC= stm_processed[0].stats.sac['o']						# Event origin time
		evla, evlo, src_z = single_ev_location(stm_processed)   		# Event latitude/longitude in SAC attribute 		
		src_lalo=[evla, evlo]
		endtime		= starttime+ot_SAC-trim_start+duration
		starttime +=ot_SAC-trim_start

	else:
		# Hypocenter location determined by Source Scanning
		src_z		= coor_max_Czy[0]/km_to_100m
		#src_SSA		= [coor_max_Czy[0]/km_to_100m, coor_max_Cxy[0]*x_to_km, coor_max_Cxy[1]*y_to_km]
		src_lalo	= utm.to_latlon( coor_max_Cxy[0]*x_to_km/easting_to_km,  coor_max_Cxy[1]*y_to_km/northing_to_km, utm_zone[0], utm_zone[1])
		endtime		=  starttime+tmin+t_eo+duration
		starttime	+= tmin+t_eo	
	
	
	
	stm= obspy.Stream()
	# Get the waveform from stm and trim it (need only event origin~ a few sec after the last arrival)
	for tr in stm_processed:
		# If want to plot a particular channel
		if channel in ['E','N','Z']:
			if tr.stats.channel== channel:
				stm.append(tr.copy())
		# If want to plot all ENZ channels
		else:
			stm.append(tr.copy())
			
	stm= stm.trim(starttime=starttime, endtime=endtime)
	# Create a list to store all information categorized by stations
	st_info	= st_basic_info(stm_processed, channel, src_lalo, src_z,TTModel)
	# If given TTModel_2
	try:
		st_info_2=st_basic_info(stm_processed, channel, src_lalo, src_z, TTModel_2)
	except:
		pass
	# st_info has the form: [st_name, distance, travel_time, sampling_rate, npts]
	
	
	# Append waveform data to the list
	for i, st in enumerate(st_info):
		st.append(stm[i])

	# Sort the list by distance
	st_info_sorted=sorted(st_info, key=lambda x: x[1])  # st_info[:,1]: x distance between station and source
	# If given TTModel_2
	try:
		st_info_2_sorted=sorted(st_info_2, key=lambda x: x[1])
	except:
		pass
	# Put info into separate lists for plotting
	st_name		=[]     # Station name
	dis   		=[]     # x distance between station and source 
	tt	     	=[]		# Travel time by the velocity model
	spl_rate	=[] 	# Sampling rate
	waveform   	=[]     # Waveform
	tt_2	    =[]		# Travel time by the velocity model_2
	
	for i in range(len(st_info)):  
	
		st_name.append(st_info_sorted[i][0])
		dis.append(st_info_sorted[i][1])
		tt.append(st_info_sorted[i][2])
		spl_rate.append(st_info_sorted[i][3])
		waveform.append(st_info_sorted[i][5])
		# If given TTModel_2
		try:
			tt_2.append(st_info_2_sorted[i][2])
		except:
			pass
			
	######################################################################################################
	# Normalize the waveform   
	waveform= np.asarray(waveform)
	wf_norm= waveform.copy() 
	wf_norm_original= waveform.copy() 
	
	for i in range(len(wf_norm)):
		# For calculating vertical_sep
		wf_norm_original[i] = wf_norm[i]/ (wf_norm[i].max())
		# For plotting
		wf_norm[i]			= wf_norm_original[i]*amp

		
	# Plot the waveforms by distances from the epicenter #########################################################
	fig, ax = plt.subplots(figsize=figsize)
	
	# Determine the verticle seperation in the graph
	vertical_sep = wf_norm_original.ravel().max() + add_v_sep

	itime = np.arange(len(wf_norm[0]))
	
	for i,st in enumerate(wf_norm):
	
		# Line seimograms up with verticle seperation= vertical_sep
		y= st  + vertical_sep *(len(wf_norm)-i)
		seismogram, = plt.plot(itime/spl_rate[i], y ,linewidth=1,label=st_name[i])
		
		# Set seismogram color according to stations
		try:	
			seismogram.set_color(color_dict[st_name[i][:3]])
		except:
			pass
			
		plt.plot(tt[i], vertical_sep *(len(wf_norm)-i)+tt_marker_sep, tt_marker, markersize=tt_mk_size)
		# If given TTModel_2
		try:
			plt.plot(tt_2[i], vertical_sep *(len(wf_norm)-i)+tt_marker_sep, tt_marker, markersize=tt_mk_size)
		except:
			pass
			
		# Label station names
		# Station names (avoid repetitively labeling names for 3 ENZ)
		if st_name[i][:3] !=st_name[i-1][:3]:
			ax.annotate(st_name[i][:3],(plot_start+0.1,vertical_sep *(len(wf_norm)-i),), size=14)
		else:
			pass
		# Channel names
		ax.annotate(st_name[i][3],(plot_start+0.8,vertical_sep *(len(wf_norm)-i),), size=14)
		
	plt.xlabel('Time (s)', fontsize=xy_label_size)
	plt.ylabel('Distances increasing downward\n(not to scale)', fontsize=xy_label_size)
	if method == 'SAC':
		plt.title('Normalized Seismogram\n ({})\n'.format(method), fontsize=title_size)
	else:
		plt.title('Normalized Seismogram\n ({}, {})\n'.format(cf_type, method), fontsize=title_size)	
	ax.set_yticklabels([])  # Disable y-axis label
	ax.tick_params(axis= 'x', labelsize= 16)
	ax.tick_params(axis= 'y', left=False)
	# Set the x start of the plot to fit the station names
	plt.xlim(plot_start,duration)
	
	plt.tight_layout()
	
	# Save ############# Save ############# Save ############# Save #
	# Save to a saparate folder 									#
	if save_fig == True:											#
		# Create the directory if the folder does not exist			#
		if not os.path.exists('save/figs'):							#
			os.makedirs('save/figs')								#
																	#									#			
		if method == 'SAC':															#
			dir_s= 'save/figs/{}_wf{}{}.png'.format(				#
					no_of_event, method, add_figname )			#
		else:	
			dir_s= 'save/figs/{}_wfSSA_{}_{}{}.png'.format(				#
					no_of_event, cf_filename, method, add_figname )			#
		plt.savefig(dir_s)											#
		print('Saved as {}'.format(dir_s))							#
																	#
	# Save ############# Save ############# Save ############# Save #
	
	
	
##############################################################################################################

def synthetic_seis(	stm,  TTModel_P, TTModel_S, 
						src_lalo, src_z , to,
						src_lalo_2=None, src_z_2=None , to_2=None,
						P_duration= 1, S_duration= 2, 
						P_duration_2= 1, S_duration_2= 2, 
						noise_a=10., noise_f= None, noise_fa=10,
						#amp=[10.,15.], freq=[15.,15.],
						P=[10., 15., 3.**.5], S=[17., 15., 3.**.5],
						res=1000., P_pola=30., S_pola=30., amp_decay=0.01,
						save=True, no_of_event='Test') :
	"""
	Generate artificial data for testing purposes.
	_______
	:input:
		- no_of_event	: Number of event (string or int)
		- src_lalo		: Source location in [latitude, longitude]
		- src_z			: Source depth in km
		- TTModel_P	: The 2D P velocity model (2D array)
		- TTModel_S	: The 2D S velocity model (2D array)
		- to
		- 
		- npts: int (optional).
		- noise: float (optional).
		- P: list (1x3 type float [optional]).
		- S: list (1x3 type float [optional]).
		- npts: data length in samples.
		- noise: amplitude of noise.
		- P: P-wave properties [amplitude, frequency, H/Z ratio]
		- S: S-wave properties [amplitude, frequency, Z/H ratio]
	_______
	:output: ObsPy :class:`~obspy.core.stream`
	:return: artificial noisy data with [noise only, frequency change,
		amplitude change, polarization change].
	_________
	.. note::

		The noise is a random 3d motion projection on 3 cartesian
		components.

		The amplitude decays are linear and the body-wave span a third
		of the signal duration (npts/6 samples each).
	___________
	.. rubric:: Example

		Generate data
		>>> import trigger
		>>> a = trigger.artificial_stream(npts=1000)
		>>> print(a)

		Plot	
		>>> import matplotlib.pyplot as plt
		>>> from mpl_toolkits.mplot3d import Axes3D
		>>> from obspy.signal.tf_misfit import plot_tfr

		>>> npts=1000
		>>> plot_tfr((a[1]).data, dt=.01, fmin=0.1, fmax=25)
		>>> plot_tfr((a[2]).data, dt=.01, fmin=0.1, fmax=25)
		>>> fig = plt.figure()
		>>> ax = fig.gca(projection='3d')
		>>> ax.plot(a[5].data, a[6].data, a[7].data, label='noise', alpha=.5, color='g')
		>>> ax.plot(a[5].data[npts//3:npts//2],a[6].data[npts//3:npts//2],a[7].data[npts//3:npts//2], label='P', color='b')
		>>> ax.plot(a[5].data[npts//2:npts*2//3],a[6].data[npts//2:npts*2//3],a[7].data[npts//2:npts*2//3], label='S', color='r')
		>>> ax.legend()
		>>> plt.show()

	"""
	

	'''
	# Convert the polarity range from degree to radian
	pola_p_range = [pola_P[0] *np.pi/180., pola_P[1] *np.pi/180.]
	pola_s_range = [pola_S[0] *np.pi/180., pola_S[1] *np.pi/180.]
	'''
	
	# Number of source in the seismogram
	if src_lalo_2 == None or src_z_2 ==None or to_2 ==None:
		num_of_src	= 1
	else:
		num_of_src	= 2
	
	
	stla, stlo, name= st_locations(stm)
	# Get travel time info given the synthetic source location to every station
	st_info_P	= st_basic_info(stm, 'Z', src_lalo, src_z ,TTModel_P)
	st_info_S	= st_basic_info(stm, 'Z', src_lalo, src_z ,TTModel_S)
	# st_info is a list of sublists: [st_name, distance, travel_time, sampling_rate, npts]
	
	# If we have The second source:
	if num_of_src	== 2:
		# Get travel time info given the synthetic source location to every station
		st_info_P_2	= st_basic_info(stm, 'Z', src_lalo_2, src_z_2 ,TTModel_P)
		st_info_S_2	= st_basic_info(stm, 'Z', src_lalo_2, src_z_2 ,TTModel_S)
		# st_info is a list of sublists: [st_name, distance, travel_time, sampling_rate, npts]		
	
	
	synthetic_stm = Stream()
	
	# Start to fill the data. Stations have different data due to travel time diferences
	for st, info in enumerate(st_info_P):
		
		starttime		= stm[0].stats.starttime  # Assume the starttime and endtime are the same for all traces
		endtime			= stm[0].stats.endtime
		st_name 		= st_info_P[st][0][:3]
		sampling_rate	= st_info_P[st][3]
		npts			= st_info_P[st][4]
		
		travel_time_P	= st_info_P[st][2]
		travel_time_S	= st_info_S[st][2]
		
		if num_of_src	== 2:
			travel_time_P_2	= st_info_P_2[st][2]
			travel_time_S_2	= st_info_S_2[st][2]

		sac		= obspy.core.AttribDict({'stla':stla[st], 'stlo':stlo[st], 
											'evla': src_lalo[0], 'evlo': src_lalo[1], 'evdp':src_z, 
											'o':to, 'kstnm':name[st] })
	
		stats_z = Stats({'network':"Test", 'station':st_name +'Z',  'channel':"Z", 
							'starttime':starttime, 'endtime':endtime, 'sampling_rate':sampling_rate,
							'npts':npts, 'sac':sac})
		stats_e = Stats({'network':"Test", 'station':st_name +'E',  'channel':"E", 
							'starttime':starttime, 'endtime':endtime, 'sampling_rate':sampling_rate,
							'npts':npts, 'sac':sac})
		stats_n = Stats({'network':"Test", 'station':st_name +'N',  'channel':"N", 
							'starttime':starttime, 'endtime':endtime, 'sampling_rate':sampling_rate,
							'npts':npts, 'sac':sac})
		
				
		# Convert second to number of points
		to_npt		=            to *sampling_rate
		tt_npt_P	= travel_time_P *sampling_rate
		tt_npt_S	= travel_time_S *sampling_rate
		P_duration_npt = P_duration *sampling_rate
		S_duration_npt = S_duration *sampling_rate
		
		Onset_npt_P	= int(to_npt + tt_npt_P)
		Onset_npt_S	= int(to_npt + tt_npt_S)
		
		# The range of P and S onsets
		Pspot = range(Onset_npt_P, int(Onset_npt_P + P_duration_npt))
		Sspot = range(Onset_npt_S, int(Onset_npt_S + S_duration_npt))
		
		if num_of_src	== 2:
			# Convert second to number of points
			to_npt_2		=         to_2 *sampling_rate
			tt_npt_P_2	= travel_time_P_2 *sampling_rate
			tt_npt_S_2	= travel_time_S_2 *sampling_rate
			P_duration_npt_2 = P_duration_2 *sampling_rate
			S_duration_npt_2 = S_duration_2 *sampling_rate
			
			Onset_npt_P_2	= int(to_npt_2 + tt_npt_P_2)
			Onset_npt_S_2	= int(to_npt_2 + tt_npt_S_2)
			
			# The range of P and S onsets
			Pspot_2 = range(Onset_npt_P_2, int(Onset_npt_P_2 + P_duration_npt_2))
			Sspot_2 = range(Onset_npt_S_2, int(Onset_npt_S_2 + S_duration_npt_2))
			
		
		
		
		''' Convert SNR= (Amp_signal/Amp_noise)**2 to S_N = (Amp_signal/Amp_noise)
		S_N	= np.sqrt(SNR)
		#S_N_S	= np.sqrt(SNR_S)
		#print('S_N_P = ', S_N_P)
		noise_amp	= P[0]/S_N
		
		'''
		# Generate Noise ############# Generate Noise ############# Generate Noise ############
		# Generate Noise ############# Generate Noise ############# Generate Noise ############
		
		# Nagano et al (1989): Noise is stoichastic
		noise_signal_ch	= np.random.randint(  -noise_a*res, noise_a*res, int(npts))/res	# for one channel
		
		# Add frequency to noise
		if not noise_f == None:	
			t	= np.arange(npts)
			#print(len(t))
			#noise_signal_ch	*= ((noise_fa)*np.sin(2*np.pi*noise_f/sampling_rate*t)+2) #not correct
			noise_signal_ch	+= (noise_fa) *np.sin(2*np.pi*noise_f/sampling_rate*t)
		# or: use random noise, no freqency.
		else:
			noise_signal	= np.asarray([   noise_signal_ch,			# Duplicate to get 3 channel
											 noise_signal_ch,
											 noise_signal_ch])

			# Put noise onto the waveforms
			waveform = np.copy(noise_signal)
			
		

		
		# Generate P and S Waves #################### Generate P and S Waves ###################
		# Generate P and S Waves #################### Generate P and S Waves ###################
		
		P_polar	= P_pola *np.pi /180.
		S_polar	= S_pola *np.pi /180.
		
		#t_seris	= np.linspace(0, npts, num = npt+1, endpont =True)
		#p_t		= np.linspace(0, npts, num = npt+1, endpont =True)
		
		p_signal = np.asarray([ np.random.randint(	-np.pi*res, np.pi*res, len(Pspot))/res ,
								np.random.randint(	(np.pi - P_polar) *res, (np.pi + P_polar)*res, len(Pspot))/res ,
								np.random.randint(	-P[0]*res, P[0]*res, len(Pspot))/res ] )
								
		p_signal = np.asarray(spherical_to_cartesian(p_signal))
		# p_signal has the form: [x,y,z]
		s_signal = np.asarray([ np.random.randint(-np.pi*res, np.pi*res, len(Sspot))/res ,
								np.random.randint((np.pi/2 - S_polar) *res, (np.pi/2 + S_polar)*res, len(Sspot))/res ,
								np.random.randint(-S[0]*res, S[0]*res, len(Sspot))/res ] )
		#print('max s = ', np.average(s_signal[0]), np.average(s_signal[1]),np.average(s_signal[2]))
		s_signal = np.asarray(spherical_to_cartesian(s_signal))
		#print('max s2 = ', np.average(s_signal[0]), np.average(s_signal[1]),np.average(s_signal[2]))
		# s_signal has the form: [x,y,z]
		
		
		if num_of_src	== 2:
			p_signal_2 = np.asarray([ np.random.randint(	-np.pi*res, np.pi*res, len(Pspot_2))/res ,
									np.random.randint(	(np.pi - P_polar) *res, (np.pi + P_polar)*res, len(Pspot_2))/res ,
									np.random.randint(	-P[0]*res, P[0]*res, len(Pspot_2))/res ] )
									
			p_signal_2 = np.asarray(spherical_to_cartesian(p_signal_2))
			# p_signal has the form: [x,y,z]
			s_signal_2 = np.asarray([ np.random.randint(-np.pi*res, np.pi*res, len(Sspot_2))/res ,
									np.random.randint((np.pi/2 - S_polar) *res, (np.pi/2 + S_polar)*res, len(Sspot_2))/res ,
									np.random.randint(-S[0]*res, S[0]*res, len(Sspot_2))/res ] )
			#print('max s = ', np.average(s_signal[0]), np.average(s_signal[1]),np.average(s_signal[2]))
			s_signal_2 = np.asarray(spherical_to_cartesian(s_signal_2))
			#print('max s2 = ', np.average(s_signal[0]), np.average(s_signal[1]),np.average(s_signal[2]))
			# s_signal has the form: [x,y,z]		
		
		
		#print('p_signal = ',p_signal)
		# waveform has the form: [Z,E,N]
		#ins_P	= np.random.randint( 0, 10, len(Pspot))/10
		#ins_S	= np.random.randint( 0, 10, len(Sspot))/10
		
		# Put P onto the trace
		waveform[0][Pspot] += p_signal[2] * (np.exp( -amp_decay*np.arange(len(Pspot)))) #Z
		waveform[1][Pspot] += p_signal[1] * (np.exp( -amp_decay*np.arange(len(Pspot)))) #E
		waveform[2][Pspot] += p_signal[0] * (np.exp( -amp_decay*np.arange(len(Pspot)))) #N
		# Put S onto the trace
		waveform[0][Sspot] += s_signal[2] * (np.exp( -amp_decay*np.arange(len(Sspot))))
		waveform[1][Sspot] += s_signal[1] * (np.exp( -amp_decay*np.arange(len(Sspot))))
		waveform[2][Sspot] += s_signal[0] * (np.exp( -amp_decay*np.arange(len(Sspot))))
		
		if num_of_src ==2:
			# Put P onto the trace
			waveform[0][Pspot_2] += p_signal_2[2] * (np.exp( -amp_decay*np.arange(len(Pspot_2)))) #Z
			waveform[1][Pspot_2] += p_signal_2[1] * (np.exp( -amp_decay*np.arange(len(Pspot_2)))) #E
			waveform[2][Pspot_2] += p_signal_2[0] * (np.exp( -amp_decay*np.arange(len(Pspot)))) #N
			# Put S onto the trace
			waveform[0][Sspot_2] += s_signal_2[2] * (np.exp( -amp_decay*np.arange(len(Sspot_2))))
			waveform[1][Sspot_2] += s_signal_2[1] * (np.exp( -amp_decay*np.arange(len(Sspot_2))))
			waveform[2][Sspot_2] += s_signal_2[0] * (np.exp( -amp_decay*np.arange(len(Sspot_2))))
		
		# Append Data Trace to the Stream ######### Append Data Trace to the Stream ############
		# Append Data Trace to the Stream ######### Append Data Trace to the Stream ############
		
		# P: P-wave properties [amplitude, frequency, H/Z ratio]		
		synthetic_stm.append(Trace(data=waveform[1], header=stats_e))
		synthetic_stm.append(Trace(data=waveform[2], header=stats_n))
		synthetic_stm.append(Trace(data=waveform[0], header=stats_z))
	
	
	if save == True:
		
		# Create the directory if the folder does not exist									
		if not os.path.exists('save/Test_Stream_{}'.format(no_of_event)):														
			os.makedirs('save/Test_Stream_{}'.format(no_of_event))
		else:
			raise Exception('Files not saved. Target folder already exists. \nPlease manually delete existing folder before proceed.')
		
		
		# Save data to file
		synthetic_stm.write('save/Test_Stream_{}/{}_Trace_'.format(no_of_event,no_of_event),  format='SAC')
		

	return synthetic_stm


##############################################################################################################	
	
	
	
	
	
		
def all_st_ev_map(evla_list,evlo_list, stm=None, resolution='h', width=70000, height=50000, sca_r=-0.15, sca_b=0.15):
	'''
    Input the location of the events. Show a map with the event locations.
    ______
	:input:
		-  evla_list: A list contains all the event latitudes (list)
		-  evlo_list: A list contains all the event longitude (list)
		-  stm: A Stream (class:obspy.core.stream) that contains all the stations
		-  resolution: Resolution of the map. 
			(options: 'c':crude, 'l':low, 'h': high, 'f':full)
		-  width: The width of the map
		-  height: The height of the map
		-  sca_r: Position of the scale bar from the right bdry
		-  sca_b: Position of the scale bar from the bottom
	_______
	:output:
		- (List): [latitude (in degree) , longitude (in degree) , depth]
		
	'''
        
	from mpl_toolkits.basemap import Basemap
	import numpy as np
	import matplotlib.pyplot as plt

	# Assign parameters

	evla, evlo= evla_list, evlo_list          # Just to make it shorter

	fig, ax = plt.subplots( figsize=(7,7))    # Adjust the size of the map
	latb=min(evla)#-0.05                      # Bottom of the map
	latt=max(evla)#+0.05                      # Top of the map
	latc=np.mean(evla)                        # Central lat of the map (for locating the centre of the map)
	lonl=min(evlo)#-0.05                      # Left bdry of the map
	lonr=max(evlo)#-0.05                      # Right bdry of the map
	lonc=np.mean(evlo)                        # Central lon of the map (for locating the centre of the map)
	res = resolution                          # Options: 'c','l','h','f'
	map_width=width                           # Adjust the area of the map
	map_height=height
	title="Event List"


	# Set up basemap

	m = Basemap(width=map_width, height=map_height,                      # Adjust the map size
				resolution=res, projection='aea',                        # aea=Albers Equal Area Projection
				lat_1=latb, lat_2=latt, lon_0=lonc, lat_0=latc, ax=ax)   # Adjust the span of lat/lon
				#lat_1=63.8, lat_2=64.1, lon_0=-22, lat_0=63.95, ax=ax)  # Or use this line to adjust 
																		 # the bdry manually
																		 
	m.drawcoastlines()
	m.drawcountries()
	m.fillcontinents(color='wheat', lake_color='skyblue')
	m.drawmapboundary(fill_color='skyblue')


	# Draw latitude and longititude lines

	m.drawparallels(np.linspace(latb, latt, 2), labels=[1, 0, 0, 1], fmt="%.2f", dashes=[2, 2])
	m.drawmeridians(np.linspace(lonl, lonr, 2), labels=[1, 0, 0, 1], fmt="%.2f", dashes=[2, 2])


	# Draw a map scale at lon,lat of length length representing distance in the map projection 
	# coordinates at lon0,lat0.

	m.drawmapscale(lonr-sca_r, latb-sca_b, lonc, latc, 10, barstyle='simple', units='km', fontsize=9, 
				   yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', fillcolor2='k',
				   ax=None, format='%d', zorder=None, linecolor=None, linewidth=None)

	ax.set_title(title)

	# attach the basemap object to the figure

	fig.bmap = m  


	# Plot event locations

	x, y = m(evlo, evla) 
	m.scatter(x, y, 100, color="w", marker="*", edgecolor="b", zorder=3)


	# Plot station locations(for this specific case)

	stla, stlo, kstnm = st_locations(stm)
	x, y = m(stlo, stla) 
	m.scatter(x, y, 100, color="r", marker="v", edgecolor="k", zorder=3)

	for i in range(len(kstnm)):
		plt.text(x[i], y[i], kstnm[i], va="top", family="monospace", weight="bold") # va:line-up at the top       

		
	plt.show()
	
		
