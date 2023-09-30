from SpotDIPy import SpotDIPy
import numpy as np
import multiprocessing
from glob import glob


DIP = SpotDIPy(processes=multiprocessing.cpu_count() - 1)

DIP.set_param('t0', 2453200.00)
DIP.set_param('period', 1.756604)
DIP.set_param('Tphot', 5082)
DIP.set_param('Tspot', 3800)
DIP.set_param('incl', 46)
DIP.set_param('vsini', 21.5)
DIP.set_param('vrt', 3.249)
DIP.set_param('mass', 0.85)
DIP.set_param('gdc', 'auto')
DIP.set_param('rv_corr_line', 0.0)
DIP.set_param('resolution', 85000)

""" Set modes """
DIP.set_modes({
               'line': {'mode': 'on', 'wrange': [4400, 6800], 'eqw': 0.8, 'scale': {'method': 'mean'}},
               'mol1': {'mode': 'off', 'wrange': [7000, 7100], 'scale': {'method': 'mean'}},
               'mol2': {'mode': 'off', 'wrange': [8800, 8900], 'scale': {'method': 'mean'}}
               })
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Construction surface grid """
DIP.construct_surface_grid(method='trapezoid', nlats=40)  # , test=True)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Set limb darkening parameters """
DIP.set_limb_darkening_params(law='linear', mh=-0.138, ld_model='mps2',
                              ld_data_path='ExoTiC-LD_data_v3.1.2', mu_min=0.1)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


""" Import initial local line profiles (LLP) (for photosphere and spot) data"""
llp_vels = np.loadtxt('synths/lsds/spectrum_5082_4.4_-0.138_0.055_1.932_0.0_0.0_0.0_0-err_lsd.out',
                      skiprows=2)[:, 0]
llp_phot_int = np.loadtxt('synths/lsds/spectrum_5082_4.4_-0.138_0.055_1.932_0.0_0.0_0.0_0-err_lsd.out',
                          skiprows=2)[:, 1]
llp_spot_int = np.loadtxt('synths/lsds/spectrum_3800_4.4_-0.138_0.055_1.932_0.0_0.0_0.0_0-err_lsd.out',
                          skiprows=2)[:, 1]

DIP.set_local_profiles(llp_vels=llp_vels, llps={'phot': llp_phot_int, 'spot': llp_spot_int} )
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


""" Make a test """
lats_spots = [0., 30., 60., 90., -30.]  # spot latitudes (degrees)
longs_spots = [0., 90., 180., 270., 270.]  # spot longitudes (degrees)
rs_spots = [20., 15., 20., 10., 15.]  # spot radius (degrees)
cs_spots = [0.8, 0.9, 0.6, 0.7, 0.9]  # spot contrast between 0 and 1

spots_params = {'lats_spots': lats_spots, 'longs_spots': longs_spots, 'rs_spots': rs_spots, 'cs_spots': cs_spots}

line_vels = np.arange(-40, 40 + 1.75, 1.75)
phases = np.linspace(0, 0.95, 10)
mid_times = DIP.params['period'] * phases + DIP.params['t0']

line_snr = 500

opt_params = {'method': 'L-BFGS-B', 'tol': 1e-10, 'lmbd': 5.0, 'maxiter': 10000, 'disp': True,  'verbose': 2}

DIP.test(times=mid_times, line_vels=line_vels, line_snr=line_snr, opt_params=opt_params, spots_params=spots_params,
         plot_params={'line_sep_prf': 1.5, 'line_sep_res': 0.01, 'mol_sep_prf': 0.3, 'mol_sep_res': 0.3,
                      'show_err_bars': True, 'fmt': '%0.3f', 'markersize': 2, 'linewidth': 1, 'fontsize': 15,
                      'ticklabelsize': 12}
         )
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
