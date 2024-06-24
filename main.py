import numpy as np
from SpotDIPy import SpotDIPy
import multiprocessing


DIP = SpotDIPy(cpu_num=multiprocessing.cpu_count() - 1, platform_name='gpu')

""" Set required parameters """
DIP.set_param('t0', value=2449681.5)
DIP.set_param('period', value=1.3571)
DIP.set_param('Tphot', value=5500)
DIP.set_param('Tcool', value=4500)
DIP.set_param('Thot', value=6000)
DIP.set_param('incl', value=60)
DIP.set_param('vsini', value=42.32)
DIP.set_param('vrt', value=2.0)
DIP.set_param('mass', value=1.184)
DIP.set_param('mh', value=0.0)
DIP.set_param('dOmega', value=0.0)
DIP.set_param('resolution', value=0)

DIP.set_limb_darkening_params(law='linear', model='mps2', mu_min=0.1)

""" Set modes """
DIP.set_conf({
               'line': {'mode': 'on',
                        'wave_range': [4020, 6270],
                        'eqw': 0.093,
                        'scaling': {'method': 'mean'},
                        'corr': {'rv': None, 'amp': None}
                        },
               'mol1': {'mode': 'off',
                        'wave_range': [7000, 7100],
                        'scaling': {'method': 'mean'},
                        'corr': {'rv': None}
                        },
               'mol2': {'mode': 'off',
                        'wave_range': [8800, 8900],
                        'scaling': {'method': 'mean'},
                        'corr': {'rv': None}
                        },
               'lc': {'mode': 'on',
                      'passband': 'TESS',
                      'scaling': {'method': 'mean'},
                      'corr': {'amp': None}
                      }
               })
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Construction surface grid """
# DIP.construct_surface_grid(method='phoebe2_marching', noes=4000)  # , test=True)
# DIP.construct_surface_grid(method='healpy', nside=16)  # , test=True)
DIP.construct_surface_grid(method='trapezoid', nlats=88)  # , test=True)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Import initial local profiles (LLP) (for photosphere, cool and hot spots) data"""
llp_vels = np.loadtxt('synths/LSDs/teff5500_logg4.3_mh0.00_vmic1.7_lsd.out', skiprows=2)[:, 0]
llp_phot_int = np.loadtxt('synths/LSDs/teff5500_logg4.3_mh0.00_vmic1.7_lsd.out', skiprows=2)[:, 1]
llp_cool_int = np.loadtxt('synths/LSDs/teff4500_logg4.3_mh0.00_vmic1.7_lsd.out',  skiprows=2)[:, 1]
llp_hot_int = np.loadtxt('synths/LSDs/teff6000_logg4.3_mh0.00_vmic1.7_lsd.out', skiprows=2)[:, 1]

DIP.set_local_profiles({'line': {'lp_vels': llp_vels, 'phot_lp_data': llp_phot_int, 'cool_lp_data': llp_cool_int,
                                 'hot_lp_data': llp_hot_int}})
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Make a test """
lats_spots = [0., 30., 60., -30.]  # spot latitudes (degrees)
longs_spots = [0., 90., 180., 270.]  # spot longitudes (degrees)
rs_spots = [20., 15., 20., 15.]  # spot radius (degrees)
cs_cools = [0.8, 0.2, 0.5, 0.0]  # cool spot contrast between 0 and 1
cs_hots = [0.0, 0.8, 0.0, 1.0]  # hot spot contrast between 0 and 1

spots_params = {'lats_spots': lats_spots, 'longs_spots': longs_spots, 'rs_spots': rs_spots, 'cs_cools': cs_cools,
                'cs_hots': cs_hots}

line_phases = np.arange(0, 1.0, 0.1)
line_times = DIP.params['t0'] + DIP.params['period'] * line_phases
line_vels = np.arange(-60, 60 + 1.75, 1.75)
line_snr = 3000

lc_phases = np.arange(0, 2, 0.01)
lc_times = DIP.params['t0'] + DIP.params['period'] * lc_phases
lc_snr = 3000


opt_params = {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'lmbd': 1.0, 'maxiter': 2500,
              'tol': 1e-5, 'disp': True}

modes_inp = {'line': {'times': line_times, 'vels': line_vels, 'snr': line_snr},
             'lc': {'times': lc_times, 'snr': lc_snr}}


plot_params = {'line_sep_prf': 0.06, 'line_sep_res': 0.01, 'mol_sep_prf': 0.03, 'mol_sep_res': 0.01,
               'show_err_bars': True, 'fmt': '%0.3f', 'markersize': 2, 'linewidth': 1, 'fontsize': 15,
               'ticklabelsize': 12}

DIP.test(spots_params, modes_input=modes_inp, opt_params=opt_params, plot_params=plot_params)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
