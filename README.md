# SpotDIPy
An easy way for stellar Doppler imaging of cool single stars.

A new version of SpotDIPy has been released that generates surface brightness maps for single cool stars by simultaneously modeling both spectral and light curve data using three-temperature approximation.


# Installation

SpotDIPy can be installed easily via pip. To install the latest version, run the following commands:
    
    pip install mayavi
    pip install spotdipy

The code has been tested on both Linux (Ubuntu 22.04) and Windows (Windows 10). However, please note that on Windows 10, there is a known issue with the ExoTiC-LD module related to downloading required files. To resolve this issue, it is recommended to manually download the necessary files in advance. These files are available at the following link:

https://zenodo.org/records/7874921

Once the files are downloaded, you can continue using SpotDIPy without any issues on Windows 10.


# Example Usage

    from SpotDIPy import SpotDIPy
    import numpy as np
    import multiprocessing
    from glob import glob

    # Initialize the SpotDIPy class. This class is used for Doppler Imaging of stellar surfaces.
    # cpu_num: Number of CPU cores to be used for parallelization tasks excluding the optimization
    # process itself, which is handled separately. Setting this to  multiprocessing.cpu_count() - 1
    # reserves one core for other system processes.
    DIP = SpotDIPy(cpu_num=multiprocessing.cpu_count() - 1, platform_name='cpu')
    
    # Set stellar parameters that define the star and its properties.
    # These parameters are crucial for accurate Doppler Imaging.

    # t0: Reference time for the observed data, typically a Julian Day (JD).  This represents
    # the zero point in time for phasing the rotational period.
    DIP.set_param('t0', value=2453200.0)  
    
    # period: Rotational period of the star at the equator, measured in days.
    DIP.set_param('period', value=1.756604)  
    
    # Tphot: Effective photospheric temperature of the star in Kelvin. This represents the average
    # temperature of the undisturbed stellar surface.
    DIP.set_param('Tphot', value=5080) 

    # Tcool: Minimum temperature for cool spots on the stellar surface, in Kelvin.  This defines the
    # coolest temperature allowed for spot features during the imaging process.
    DIP.set_param('Tcool', value=3800)  

    # Thot: Maximum temperature for hot spots on the stellar surface, in Kelvin.
    # This sets the upper limit for the temperature of bright features. Note that in this example,
    # Thot is set to the same value as Tphot, effectively disabling hot spots.
    DIP.set_param('Thot', value=5080)  
    
    # incl: Inclination angle of the stellar rotational axis relative to the line of sight,
    # in degrees. An inclination of 90 degrees means we view the star equator-on, while 0 degrees
    # means we view it pole-on.
    DIP.set_param('incl', value=46)  
    
    # vsini: Projected equatorial rotational velocity of the star, in km/s. This is the rotational velocity
    # at the equator multiplied by the sine of the inclination angle. Only one of 'vsini' or 'R'
    # (stellar radius) should be set, as they are related.
    DIP.set_param('vsini', value=21.26272)  

    # R: Stellar radius in solar radii. As mentioned above, only use 'R' if you haven't set 'vsini'.
    # DIP.set_param('R', value=0.770)
    
    # vrt: Radial-tangential macroturbulence velocity in km/s. This parameter accounts for the
    # broadening of spectral lines due to large-scale convective motions in the stellar atmosphere.
    DIP.set_param('vrt', value=7.44827)
    
    # mass: Stellar mass in solar masses. This parameter is important for accounting for gravity darkening.
    DIP.set_param('mass', value=0.85) 
    
    # dOmega: Differential rotation parameter. A value of 0.0 indicates solid body rotation (no
    # differential rotation). Non-zero values represent the difference in angular velocity
    # between the equator and the poles.
    DIP.set_param('dOmega', value=0.0)
    
    # resolution: Resolution of the observed spectrum. This value is for correctly simulating
    # the instrumental broadening of the spectral lines.
    DIP.set_param('resolution', value=85000)
    
    # Set limb-darkening parameters
    # mh: Metallicity of the star, measured in dex (decimal exponent) relative to solar metallicity.
    # law: The law used to fit I(mu) to determine limb darkening coefficients. Options include:
    #       - 'linear': Linear limb darkening law.
    #       - 'quadratic': Quadratic limb darkening law.
    #       - 'square-root': Square-root limb darkening law.
    # model: The model grid used to model the stellar intensity (I) as a function of radial
    #       position on the stellar disc (mu) based on pre-computed grids spanning a range
    #       of metallicity, effective temperature, and surface gravity.
    # mu_min: The minimum mu, determining the range over which limb darkening coefficient is calculated.
    # data_path: Path to the directory containing the required data files. If the files are not present,
    #            they will be downloaded from the internet.
    DIP.set_limb_darkening_params(mh=-0.14, law='linear', model='mps2', mu_min=0.1,
                                  data_path='exotic_ld_data')
    
    # Set configuration
    # Define settings for processing atomic line profiles and light curve data.
    DIP.set_conf({
        'line': { 
            'mode': 'on',  # Enable processing of atomic line profiles when set to 'on'
            'wave_range': [4400, 6800],  # Wavelength range of the observed spectrum, in angstroms
            'eqw': 0.08261,  # Equivalent width of the atomic lines
            'scaling': {'method': 'mean'},  # Method used to scale observed and synthetic line profiles
            'corr': {'rv': -0.1337, 'amp': None}  # Radial velocity correction (rv) and amplitude correction (amp)
        },
        'lc': { 
            'mode': 'on',  # Enable processing of light curve data when set to 'on'
            'passband': 'TESS',  # Passband in which the light curve was observed
            'scaling': {'method': 'mean'},  # Method used to scale observed and synthetic light curve profiles
            'corr': {'amp': None}  # Amplitude correction (amp) for the light curve
        }
    })
    
    # Construct the surface grid
    # Define the method and parameters for constructing the stellar surface grid.
    # Examples of available methods:
    # DIP.construct_surface_grid(method='phoebe2_marching', noes=5000)  # Use the PHOEBE2 marching algorithm with about 5000 surface elements
    # DIP.construct_surface_grid(method='healpy', nside=16)  # Use the HEALPix grid with NSIDE = 16
    DIP.construct_surface_grid(method='trapezoid', nlats=40)  # Use the trapezoid method with 40 latitude divisions

    
    # Import initial local line profiles (LLPs) for the photosphere, cool spots, and hot spots.
    # These profiles are used to model the local line intensities on the stellar surface.
    # Not: If only light curve modeling is desired, these profiles are not needed.

    # Load velocity data for the line profiles (e.g., photosphere, cool, and hot spots).
    llp_vels = np.loadtxt('../synth_lsds/synth_T5080.0_logg4.4_mh-0.14_mic1.93_with-err_lsd.out', skiprows=2)[:, 0]
    
    # Load intensity data for the photosphere's local line profile.
    llp_phot_int = np.loadtxt('../synth_lsds/synth_T5080.0_logg4.4_mh-0.14_mic1.93_with-err_lsd.out', skiprows=2)[:, 1]
    
    # Load intensity data for cool spots' local line profile.
    llp_cool_int = np.loadtxt('../synth_lsds/synth_T3800.0_logg4.4_mh-0.14_mic1.93_with-err_lsd.out', skiprows=2)[:, 1]
    
    # Load intensity data for hot spots' local line profile (reusing the photospheric profile here as an example).
    llp_hot_int = np.loadtxt('../synth_lsds/synth_T5080.0_logg4.4_mh-0.14_mic1.93_with-err_lsd.out', skiprows=2)[:, 1]
    
    # Set the loaded local line profiles in the DIP configuration for modeling line profiles.
    DIP.set_local_profiles({
        'line': {
            'lp_vels': llp_vels,          # Velocities for the local line profiles
            'phot_lp_data': llp_phot_int,  # Photosphere local line profile data
            'cool_lp_data': llp_cool_int,  # Cool spots local line profile data
            'hot_lp_data': llp_hot_int    # Hot spots local line profile data
        }
    })
    
    # Import observed data
    # Load observed atomic line data
    line_paths = glob('target_lsds/*.out')
    line_obs_data = []
    line_mid_times = []
    
    for i, line_path in enumerate(line_paths):
        line_data = np.loadtxt(line_path, skiprows=2)
        
        # Extract the mid-time of the observation from the filename
        line_mid_time = float(line_path.split('/')[-1].split('_')[1].split("=")[1])
        line_mid_times.append(float(line_mid_time))

        line_obs_data.append(line_data)
    
    # Load observed light curve data
    lc_obs_data = np.loadtxt("target_lc/target_lc.txt")
    
    # Prepare a dictionary to pass the observational data into the SpotDIPy class
    # 'line' is for atomic line profiles (LSD profiles).
    # - 'times' should be an array of mid-times for each observation, formatted consistently with "t0".
    # - 'data' is a list where each element is a 2D array containing radial velocity, normalized intensity, 
    #   and their corresponding errors for each observation.
    # 
    # 'lc' is for the light curve profile.
    # - 'times' should be an array of observation times, formatted consistently with "t0".
    # - 'data' should be a 2D array where the first column contains flux values and the second column contains 
    #   their corresponding errors.
    input_data_dict = {
        'line': {
            'times': line_mid_times,  # Mid-times for LSD profiles
            'data': line_obs_data     # Radial velocity, normalized intensity, and errors
        },
        'lc': {
            'times': lc_obs_data[:, 0],  # Times for light curve observations
            'data': lc_obs_data[:, 1:]   # Flux values and errors
        }
    }
    
    # Pass the input data dictionary to the SpotDIPy class
    DIP.set_input_data(input_data_dict)

    # Reconstruct the stellar surface brightness distribution map
    # `alpha`: Weight for chi-square of line profiles
    # `delta`: Weight for chi-square of light curve
    # `lmbd`: Weight for maximum entropy regularization
    # `maxiter`: Maximum number of iterations
    # `tol`: Convergence tolerance for stopping criteria
    recons_result = DIP.reconstructor(alpha=1.0, delta=1.0, lmbd=1, maxiter=5500, tol=1e-7, disp=True)
    
    # Plot the results with customized parameters
    DIP.plot(plot_params={
    'line_sep_prf': 0.06,  # Separation between line profiles in the plot
    'line_sep_res': 0.01,  # Separation between line profile residuals
    'mol_sep_prf': 0.3,    # Separation between molecular line profiles in the plot
    'mol_sep_res': 0.2,    # Separation between molecular line profile residuals
    'show_err_bars': True, # Whether to display error bars on the plot
    'fmt': '%0.3f',        # Format for numeric labels
    'markersize': 2,       # Size of the markers used in plots
    'linewidth': 1,        # Line width for plotted curves
    'fontsize': 15,        # Font size for labels
    'ticklabelsize': 12    # Font size for axis tick labels
    })


<p align="center">
<img src="https://github.com/EnginBahar/SpotDIPy/assets/122885382/b7701307-0c5b-4761-8b54-2a1b94c17228" width=60% height=60%>
<img src="https://github.com/EnginBahar/SpotDIPy/assets/122885382/2365a668-40bc-403d-aef6-f5eed17c8c4a" width=60% height=60%>
<img src="https://github.com/EnginBahar/SpotDIPy/assets/122885382/b47c05ae-1f48-4ebe-9cee-c78fc8509156" width=60% height=60%>
<img src="https://github.com/EnginBahar/SpotDIPy/assets/122885382/c5255fb6-126b-4b20-973b-9dcfdae615ab" width=60% height=60%>
<img src="https://github.com/EnginBahar/SpotDIPy/assets/122885382/bf52cef5-3ee0-4401-bc72-ac829b076195" width=60% height=60%>
<img src="https://github.com/EnginBahar/SpotDIPy/assets/122885382/87901edb-2892-4e78-b043-4d9d8d9d5145" width=60% height=60%>
<img src="https://github.com/EnginBahar/SpotDIPy/assets/122885382/cf50ca96-0a0e-46bc-82e7-6c084f114fc6" width=60% height=60%>
</p>



