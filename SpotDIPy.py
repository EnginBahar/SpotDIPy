import os
import sys

import matplotlib.pyplot as plt
from jax import vmap, config as jax_config, numpy as jnp
from jaxopt import ScipyBoundedMinimize
import pickle
import numpy as np
from astropy import units as au, constants as ac
import utils as dipu


class SpotDIPy:

    def __init__(self, cpu_num=1, platform_name='cpu', mp=True, info=True):

        if not isinstance(cpu_num, (int, np.int16, np.int32, np.int64)):
            raise ValueError("'cpu_num' keyword must be an integer number")

        if platform_name not in ['cpu', 'gpu']:
            raise KeyError("'platform_name' keyword must be one of two options: 'cpu' or 'gpu'")

        if mp not in [True, False]:
            raise KeyError("'mp' keyword must be one of two options: True or False")

        if info not in [True, False]:
            raise KeyError("'info' keyword must be one of two options: True or False")

        jax_config.update("jax_platform_name", platform_name)
        jax_config.update("jax_enable_x64", False)

        if info:
            print('\033[93m' + "Number of CPU cores used: " + str(cpu_num) + '\033[0m')

        self.cpu_num = cpu_num
        self.platform_name = platform_name

        self.params = {
            't0': 2450000.0,
            'period': 1.0,
            'Tphot': 5500,
            'Tcool': 3500,
            'Thot': 6500,
            'incl': 90,
            'vsini': 20.0,
            'vrt': 3.0,
            'mass': 1.0,
            'mh': 0.0,
            'dOmega': 0.0,
            'resolution': 0.0
        }

        self.conf = {
            'line': {'mode': 'off',
                     'wave_range': None,
                     'eqw': {},
                     'scaling': {'method': None, 'percent': None, 'side': None},
                     'corr': {'rv': 0, 'amp': 0},
                     'corr_result': {'rv': None, 'amp': None},
                     'mask': {'rv': None, 'amp': None},
                     'scale_factor': {}
                     },
            'mol1': {'mode': 'off',
                     'wave_range': None,
                     'scaling': {'method': None},
                     'corr': {'rv': 0, 'amp': None},
                     'corr_result': {'rv': None, 'amp': None},
                     'mask': {'rv': None, 'amp': None},
                     'scale_factor': {}
                     },
            'mol2': {'mode': 'off',
                     'wave_range': None,
                     'scaling': {'method': None},
                     'corr': {'rv': 0, 'amp': None},
                     'corr_result': {'rv': None, 'amp': None},
                     'mask': {'rv': None, 'amp': None},
                     'scale_factor': {}
                     },
            'lc': {'mode': 'off',
                   'passband': None,
                   'scaling': {'method': None},
                   'corr': {'rv': None, 'amp': 0},
                   'corr_result': {'rv': None, 'amp': None},
                   'mask': {'rv': None, 'amp': None},
                   'scale_factor': 1.0,
                   }
        }

        self.idc = {
            'line': {'times': [], 'data': [], 'data_cube': [], 'vels': None, 'noo': 0, 'nop': None,
                     'snr': None, 'lp_vels': None, 'phot_lp_data_raw': None, 'cool_lp_data_raw': None,
                     'hot_lp_data_raw': None},

            'mol1': {'times': [], 'data': [], 'data_cube': [], 'vels': None, 'noo': 0, 'nop': None,
                     'snr': None, 'lp_vels': None, 'phot_lp_data_raw': None, 'cool_lp_data_raw': None,
                     'hot_lp_data_raw': None},

            'mol2': {'times': [], 'data': [], 'data_cube': [], 'vels': None, 'noo': 0, 'nop': None,
                     'snr': None, 'lp_vels': None, 'phot_lp_data_raw': None, 'cool_lp_data_raw': None,
                     'hot_lp_data_raw': None},

            'lc': {'times': [], 'data': [], 'norsp': None, 'noo': 1, 'nop': None, 'snr': None}
        }

        self.ld_rgi = {}

        self.ld_params = {'law': 'linear', 'model': 'mps2', 'mu_min': 0.1, 'data_path': 'exotic_ld_data'}

        self.opt_stats = {
            'Chi-square for Line Profile(s)': '',
            'Chi-square for Molecular(1) Profile(s)': '',
            'Chi-square for Molecular(2) Profile(s)': '',
            'Chi-square for Light Curve Profile': '',
            'Alpha * Line Profile(s) Chi-square': '',
            'Beta * Molecular(1) Profile(s) Chi-square': '',
            'Gamma * Molecular(2) Profile(s) Chi-square': '',
            'Delta * Light Curve Profile Chi-square': '',
            'Total Weighted Chi-square': '',
            'Total Entropy': '',
            'Lambda * Total Entropy': '',
            'Loss Function Value': ''
        }

        self.opt_results = {
            'rfssc': [], 'rfssh': [], 'nit': 0, 'nfev': 0,
            'lmbd': 0, 'line': {}, 'mol1': {}, 'mol2': {}, 'lc': {},
            'total_chisqs': [], 'mems': [], 'lmbds': [], 'maxcurve': 0
        }
        self.com_times = []
        self.surface_grid = {'init_noes': 0, 'nlats': 0, 'nside': 0}

        self.fsl = 1e-7
        self.fsu = 1.0 - self.fsl

    def set_param(self, name, value):

        if name not in self.params.keys():
            raise KeyError("'" + name + "'" + " is not a valid parameter. Please choose one of " +
                           str(list(self.params.keys())))

        else:
            if dipu.isfloatable(value) is False:
                raise ValueError(" Invalid value (" + str(value) + ") for the '" + name +
                                 "' parameter. Please set it to an appropriate floating-point or integer value.")

            else:
                self.params[name] = float(value)

    def set_limb_darkening_params(self, law='linear', model='mps2', mu_min=0.1, data_path='exotic_ld_data'):

        if law not in ['linear', 'square-root', 'quadratic']:
            raise KeyError("'law' keyword must be one of three options: 'linear', 'square-root' or 'quadratic'")

        if model not in ['kurucz', 'mps1', 'mps2', 'stagger']:
            raise KeyError("'model' keyword must be one of four options: 'kurucz', 'mps1', 'mps2' or 'stagger'")

        if dipu.isfloatable(mu_min) is False:
            raise ValueError("'mu_min' keyword must be a float value")

        if data_path != 'exotic_ld_data':
            if not os.path.exists(data_path):
                raise FileNotFoundError(data_path + ": file not found!")

            if not os.path.isdir(data_path):
                raise NotADirectoryError(data_path + ": not a directory!")

        self.ld_params = {'law': law, 'model': model, 'mu_min': mu_min, 'data_path': data_path}

    def set_conf(self, conf):

        common_eqw_err = ("'eqw' keyword must be a float value greater than 0.0 or a dictionary in the"
                          " format {'phot': a float value greater than 0.0, 'cool': a float value greater than 0.0,"
                          " 'hot': a float value greater than 0.0}")

        common_corr_err = {'line': "'corr' keyword must be a dictionary in the format {'rv': a float value, list,"
                                   " 'free' or None, 'amp': a float value, list, 'free' or None}",
                           'mol1': "'corr' keyword must be a dictionary in the format {'rv': a float value, list, "
                                   "'free' or None}",
                           'mol2': "'corr' keyword must be a dictionary in the format {'rv': a float value, list, "
                                   "'free' or None}",
                           'lc': "'corr' keyword must be a dictionary in the format {'amp': a float value, list,"
                                 " 'free' or None}"}

        for mode in conf:
            if mode not in self.conf:
                raise KeyError("'" + mode + "'" + " is not a valid 'conf' keyword. Please choose one or more of the"
                                                  " following keywords: 'line', 'mol1', 'mol2' or 'lc'")

            else:
                if conf[mode].keys() != set(list(self.conf[mode].keys())[:-3]):
                    raise KeyError("All required keywords for the " + "'" + mode + "'" + " must be set, including "
                                   + str(list(self.conf[mode].keys())[:-3]))

                else:
                    if conf[mode]['mode'] not in ['on', 'off']:
                        raise KeyError("'mode' keyword must be one of two options: 'on' or 'off'")

                    if conf[mode]['mode'] == 'on':
                        for par in conf[mode]:
                            if par == 'wave_range':
                                if (not isinstance(conf[mode][par], (list, tuple, np.ndarray)) or
                                        np.array(conf[mode][par]).ndim != 1 or len(conf[mode][par]) < 2):
                                    raise ValueError("'wave_range' keyword must be a 1D list, tuple or numpy array "
                                                     "containing at least 2 float or integer elements greater than 0.0")
                                else:
                                    for wave in conf[mode][par]:
                                        if not dipu.isfloatable(wave) or float(wave) <= 0.0:
                                            raise ValueError("'wave_range' keyword must be a 1D list, tuple or numpy"
                                                             " array containing at least 2 float or integer elements"
                                                             " greater than 0.0")

                            if par == 'eqw' and isinstance(conf[mode][par], dict):
                                if sorted(list(conf[mode][par].keys())) != sorted(['phot', 'cool', 'hot']):
                                    raise KeyError(common_eqw_err)

                                for subpar in conf[mode][par]:
                                    if not dipu.isfloatable(conf[mode][par][subpar]):
                                        raise ValueError(common_eqw_err)

                                    elif float(conf[mode][par][subpar]) <= 0.0:
                                        raise ValueError(common_eqw_err)

                            if par == 'eqw' and not isinstance(conf[mode][par], dict):
                                if not dipu.isfloatable(conf[mode][par]):
                                    raise ValueError(common_eqw_err)

                                elif float(conf[mode][par]) <= 0.0:
                                    raise ValueError(common_eqw_err)

                            if par == 'corr' and isinstance(conf[mode][par], dict):
                                for subpar in conf[mode][par]:
                                    if subpar not in ['rv', 'amp']:
                                        raise KeyError(common_corr_err[mode])

                                for subpar in conf[mode][par]:
                                    if (not dipu.isfloatable(conf[mode][par][subpar]) and
                                            not conf[mode][par][subpar] in ['free', None]):
                                        raise ValueError(common_corr_err[mode])

                            if par == 'corr' and not isinstance(conf[mode][par], dict):
                                raise KeyError(common_corr_err[mode])

                            if par != 'eqw' and isinstance(self.conf[mode][par], dict):
                                for subpar in conf[mode][par]:
                                    self.conf[mode][par][subpar] = conf[mode][par][subpar]
                            else:
                                self.conf[mode][par] = conf[mode][par]

        keys = list(self.conf.keys())
        for mode in keys:
            if self.conf[mode]['mode'] != 'on':
                del self.conf[mode]

    def construct_surface_grid(self, method='trapezoid', noes=1500, nlats=20, nside=16, info=True, test=False):

        self.surface_grid['init_noes'], self.surface_grid['nlats'], self.surface_grid['nside'] = noes, nlats, nside

        radius = dipu.calc_radius(vsini=self.params['vsini'], incl=self.params['incl'], period=self.params['period'])

        methods = ['trapezoid', 'phoebe2_marching', 'healpy']
        if method not in methods:
            raise KeyError("'" + method + "'" + " keyword must be one of three options: 'trapezoid', 'phoebe2_marching'"
                                                " or 'healpy'")

        omega, requiv, rp = dipu.calc_omega_and_requiv(mass=self.params['mass'], period=self.params['period'],
                                                       re=radius)

        if omega > 1.0:
            raise ValueError('\033[91m' + "The rotation rate exceeds 1! Probably,"
                                          " one or more of the 'vsini', 'incl.',"
                                          " 'period', and 'mass' parameters are not"
                                          " in the appropriate values." + '\033[0m')

        if info:
            print('\033[92m' + 'Constructing stellar surface grid...' + '\033[0m')

        if method == 'phoebe2_marching':
            dipu.p2m_surface_grid(container=self.surface_grid, requiv=requiv, noes=noes, t0=0,
                                  period=self.params['period'], mass=self.params['mass'])

        elif method == 'trapezoid':
            dipu.td_surface_grid(container=self.surface_grid, omega=omega, nlats=nlats, radius=radius,
                                 mass=self.params['mass'], cpu_num=self.cpu_num)

        elif method == 'healpy':
            dipu.hp_surface_grid(container=self.surface_grid, omega=omega, nside=nside, radius=radius,
                                 mass=self.params['mass'], cpu_num=self.cpu_num)

        self.surface_grid['gds'] = dipu.calc_gds(omega=omega, thetas=self.surface_grid['grid_lats'] + np.pi / 2.0)

        self.surface_grid['noes'] = len(self.surface_grid['grid_lats'])
        self.surface_grid['coslats'] = np.cos(self.surface_grid['grid_lats'])
        self.surface_grid['sinlats'] = np.sin(self.surface_grid['grid_lats'])
        self.surface_grid['cosi'] = np.cos(np.deg2rad(self.params['incl']))
        self.surface_grid['sini'] = np.sin(np.deg2rad(self.params['incl']))

        veq = self.params['vsini'] / self.surface_grid['sini']

        if info:
            print('\033[96m' + 'Number of total surface element: ' + '\033[0m', self.surface_grid['noes'])
            print('\033[96m' + 'Equatorial radius: ' + '\033[0m', np.round(radius, 3), 'SolRad')
            print('\033[96m' + 'Equatorial rotational velocity: ' + '\033[0m', np.round(veq, 2), 'km/s')

        self.surface_grid['ctps'] = self.params['Tphot'] * self.surface_grid['gds']
        self.surface_grid['ctcs'] = self.params['Tcool'] * self.surface_grid['gds']
        self.surface_grid['cths'] = self.params['Thot'] * self.surface_grid['gds']

        if test:
            xyzs = self.surface_grid['grid_xyzs']
            if self.surface_grid['method'] == 'p2m':
                scalars1 = self.surface_grid['gds'].copy()
                scalars2 = self.surface_grid['grid_areas'].copy()
                scalars3 = self.surface_grid['grid_lats'].copy()
                scalars4 = self.surface_grid['grid_longs'].copy()

            else:
                scalars1 = np.repeat(self.surface_grid['gds'], 2)
                scalars2 = np.repeat(self.surface_grid['grid_areas'], 2)
                scalars3 = np.repeat(self.surface_grid['grid_lats'], 2)
                scalars4 = np.repeat(self.surface_grid['grid_longs'], 2)

            dipu.grid_test(xyzs=xyzs, scalars=scalars1, title='Gravity Darkening Distribution', show=False)
            dipu.grid_test(xyzs=xyzs, scalars=scalars2, title='Surface Element Area Distribution', show=False)
            dipu.grid_test(xyzs=xyzs, scalars=scalars3, title='Latitude Distribution', show=False)
            dipu.grid_test(xyzs=xyzs, scalars=scalars4, title='Longitude Distribution')

    def prep_ld_and_int_lpt(self, info=False, grid_search=False):

        if info:
            print('\033[92m' + 'Preparing limb-darkening and intensity lookup table...' + '\033[0m')

        min_temp = np.min(np.hstack((self.surface_grid['ctps'], self.surface_grid['ctcs'], self.surface_grid['cths'])))
        max_temp = np.max(np.hstack((self.surface_grid['ctps'], self.surface_grid['ctcs'], self.surface_grid['cths'])))
        temps = np.arange(min_temp - 50, max_temp + 100, 50)

        min_logg = np.min(self.surface_grid['grid_loggs'])
        max_logg = np.max(self.surface_grid['grid_loggs'])
        loggs = np.arange(min_logg - 0.01, max_logg + 0.02, 0.01)

        for mode in self.conf:
            if mode != 'lc':
                ld_rgi = dipu.ldcs_rgi_prep(temps=temps, loggs=loggs, law=self.ld_params['law'],
                                            mh=self.params['mh'], wrange=self.conf[mode]['wave_range'],
                                            model=self.ld_params['model'],
                                            data_path=self.ld_params['data_path'],
                                            mu_min=self.ld_params['mu_min'], cpu_num=self.cpu_num)

                self.ld_rgi[mode] = ld_rgi

            elif mode == 'lc':
                ld_rgi = dipu.ldcs_rgi_prep(temps=temps, loggs=loggs, law=self.ld_params['law'],
                                            mh=self.params['mh'], passband=self.conf[mode]['passband'],
                                            model=self.ld_params['model'],
                                            data_path=self.ld_params['data_path'],
                                            mu_min=self.ld_params['mu_min'], cpu_num=self.cpu_num)

                self.ld_rgi['lc'] = ld_rgi

        if not grid_search:
            if 'lc' not in self.conf and len(self.conf) == 1:
                self.ld_rgi['lc'] = self.ld_rgi[list(self.conf.keys())[0]]

            elif 'lc' not in self.conf and len(self.conf) > 1:
                wrange = []
                for mode in self.conf:
                    wrange.append(self.conf[mode]['wave_range'])

                ld_rgi = dipu.ldcs_rgi_prep(temps=temps, loggs=loggs, law=self.ld_params['law'],
                                            mh=self.params['mh'], wrange=[np.min(wrange), np.max(wrange)],
                                            model=self.ld_params['model'],
                                            data_path=self.ld_params['data_path'],
                                            mu_min=self.ld_params['mu_min'], cpu_num=self.cpu_num)

                self.ld_rgi['lc'] = ld_rgi

    def set_local_profiles(self, lp_dict):

        for mode in self.conf:
            if mode != 'lc':
                self.idc[mode]['lp_vels'] = lp_dict[mode]['lp_vels']
                self.idc[mode]['phot_lp_data_raw'] = lp_dict[mode]['phot_lp_data']
                self.idc[mode]['cool_lp_data_raw'] = lp_dict[mode]['cool_lp_data']
                self.idc[mode]['hot_lp_data_raw'] = lp_dict[mode]['hot_lp_data']

    def set_input_data(self, input_dict=None):

        if input_dict:
            for mode in input_dict:
                if mode in self.conf:
                    for par in input_dict[mode]:
                        self.idc[mode][par] = input_dict[mode][par]

        dipu.input_data_prep(container=self.idc, conf=self.conf)

        nofp = 0
        for mode in self.conf:
            for item in self.conf[mode]['corr']:
                if (type(self.conf[mode]['corr'][item]) not in [list, np.ndarray] and
                        self.conf[mode]['corr'][item] == 'free'):
                    nofp += self.idc[mode]['noo']
        self.idc['nofp'] = nofp

        dmask = np.zeros(self.surface_grid['noes'] * 2 + nofp, dtype=bool)
        cmask = dmask.copy()
        for mode in self.conf:
            noo = self.idc[mode]['noo']

            for parn in self.conf[mode]['corr']:
                if self.conf[mode]['corr'][parn] == 'free':
                    if not cmask.any():
                        cmask = dmask.copy()
                        cmask[2 * self.surface_grid['noes']: 2 * self.surface_grid['noes'] + noo] = True

                    else:
                        ind = max(np.where(cmask)[0]) + 1
                        cmask = dmask.copy()
                        cmask[ind: ind + noo] = True

                    self.conf[mode]['mask'][parn] = cmask.copy()

    def prepare_lps(self, info=False):

        if info:
            print('\033[92m' + 'Preparing the local spectral profiles...' + '\033[0m')

        for item in ['phot', 'cool', 'hot']:
            for mode in self.conf:
                if mode != 'lc':
                    wrange = self.conf[mode]['wave_range']
                    lp_vels = self.idc[mode]['lp_vels']
                    self.idc[mode][item + '_lp_data'] = self.idc[mode][item + '_lp_data_raw'].copy()

                    if self.params['resolution'] > 0.0:
                        resolution = self.params['resolution']
                        self.idc[mode][item + '_lp_data'] = dipu.set_instrbroad(wrange=wrange,
                                                                                vels=lp_vels,
                                                                                ints=self.idc[mode][item + '_lp_data'],
                                                                                resolution=resolution)

                    if mode == 'line' and type(self.conf[mode]['eqw']) is dict:
                        eqw = self.conf[mode]['eqw'][item]
                        self.idc[mode][item + '_lp_data'] = dipu.set_eqw(wrange=wrange,
                                                                         vels=lp_vels,
                                                                         ints=self.idc[mode][item + '_lp_data'],
                                                                         eqw=eqw)

                    elif mode == 'line' and dipu.isfloatable(self.conf[mode]['eqw']):
                        eqw = self.conf[mode]['eqw']
                        self.idc[mode][item + '_lp_data'] = dipu.set_eqw(wrange=wrange,
                                                                         vels=lp_vels,
                                                                         ints=self.idc[mode][item + '_lp_data'],
                                                                         eqw=eqw)

    def calc_pixel_coeffs(self, mode=None, line_times=None, mol1_times=None, mol2_times=None, lc_times=None, info=True,
                          plps=(True, True), plil=(True, True, False)):

        plats = (2.0 * np.pi * self.params['period'] / (2.0 * np.pi - self.params['period'] * self.params['dOmega'] *
                                                        self.surface_grid['sinlats'] ** 2))
        vlats = (2.0 * np.pi * self.surface_grid['grid_rs'] * au.solRad.to(au.km) * self.surface_grid['coslats'] /
                 (plats * au.day.to(au.second)))

        ctps = self.surface_grid['ctps']
        ctcs = self.surface_grid['ctcs']
        cths = self.surface_grid['cths']
        loggs = self.surface_grid['grid_loggs']
        areas = self.surface_grid['grid_areas']

        cpcs = {}

        all_times = {'line': line_times, 'mol1': mol1_times, 'mol2': mol2_times, 'lc': lc_times}

        if plil[0]:
            self.prep_ld_and_int_lpt(info=plil[1], grid_search=plil[2])

        if plps[0]:
            self.prepare_lps(info=plps[1])

        if mode is None:
            modes = list(self.conf.keys())
        else:
            modes = [mode]

        com_times = []
        for k in range(len(modes)):
            com_times += list(all_times[modes[k]])
        com_times = np.unique(com_times)
        if mode is None:
            self.com_times = com_times.copy()

        if info:
            print('\033[92m' + 'Calculating the coefficients related to the surface elements...' + '\033[0m')

        for mode in modes:
            if info:
                if mode == 'line':
                    print('\033[1m' + 'For atomic line profile(s)...' + '\033[0m')

                elif mode == 'mol1':
                    print('\033[1m' + 'For molecular band profile(s) (1)...' + '\033[0m')

                elif mode == 'mol2':
                    print('\033[1m' + 'For molecular band profile(s) (2)...' + '\033[0m')

                elif mode == 'lc':
                    print('\033[1m' + 'For light curve profile...' + '\033[0m')

            rgi = self.ld_rgi[mode]

            ldcs_phot = np.array([rgi[0](ctps, loggs, grid=False), rgi[1](ctps, loggs, grid=False)]).T
            ldcs_cool = np.array([rgi[0](ctcs, loggs, grid=False), rgi[1](ctcs, loggs, grid=False)]).T
            ldcs_hot = np.array([rgi[0](cths, loggs, grid=False), rgi[1](cths, loggs, grid=False)]).T

            lis_phot = rgi[2](ctps, loggs, grid=False)
            lis_cool = rgi[2](ctcs, loggs, grid=False)
            lis_hot = rgi[2](cths, loggs, grid=False)

            nop = self.idc[mode]['nop']

            times = all_times[mode]

            noes = self.surface_grid['noes']
            grid_lats = self.surface_grid['grid_lats']
            grid_longs = self.surface_grid['grid_longs']
            t0 = self.params['t0']
            period = self.params['period']
            incl = np.deg2rad(self.params['incl'])
            ld_law = self.ld_params['law']

            if mode != 'lc':
                lp_vels = self.idc[mode]['lp_vels']
                phot_lp_data = self.idc[mode]['phot_lp_data']
                cool_lp_data = self.idc[mode]['cool_lp_data']
                hot_lp_data = self.idc[mode]['hot_lp_data']
                vrt = self.params['vrt']
                vels = self.idc[mode]['vels']

                input_args_spec = [(itime, plats, vlats, ldcs_phot, ldcs_cool, ldcs_hot, lis_phot,
                                    lis_cool, lis_hot, areas, grid_lats, grid_longs, nop, t0, incl,
                                    ld_law, noes, lp_vels, phot_lp_data, cool_lp_data, hot_lp_data,
                                    vrt, vels, period, info) for itime in times]

                results = dipu.mp_calc_pixel_coeffs(cpu_num=self.cpu_num, input_args=input_args_spec, mode=mode)

            if mode == 'lc':
                input_args_lc = [(itime, plats, ldcs_phot, ldcs_cool, ldcs_hot, lis_phot,
                                  lis_cool, lis_hot, areas, grid_lats, grid_longs, t0, incl,
                                  ld_law, noes, period, info) for itime in times]

                results = dipu.mp_calc_pixel_coeffs(cpu_num=self.cpu_num, input_args=input_args_lc, mode=mode)

            cpcs[mode] = {'coeffs_cube': np.array(results), 'times': {}}
            for i, itime in enumerate(times):
                cpcs[mode]['times'][itime] = np.array(results[i])

        return cpcs

    def generate_synthetic_profile(self, fssc, fssh, rv, amp, coeffs_cube, mode, lib=np):

        nop = self.idc[mode]['nop']

        fssp = 1.0 - (fssc + fssh)

        wgt_phot = coeffs_cube[:, 0] * fssp
        wgtn_phot = lib.sum(wgt_phot)

        wgt_cool = coeffs_cube[:, 1] * fssc
        wgtn_cool = lib.sum(wgt_cool)

        wgt_hot = coeffs_cube[:, 2] * fssh
        wgtn_hot = lib.sum(wgt_hot)

        prf = lib.sum(wgt_phot[:, None] * coeffs_cube[:, 3: 3 + nop] +
                      wgt_cool[:, None] * coeffs_cube[:, 3 + nop: 3 + 2 * nop] +
                      wgt_hot[:, None] * coeffs_cube[:, 3 + 2 * nop:], axis=0)
        prf /= wgtn_phot + wgtn_cool + wgtn_hot

        vels = self.idc[mode]['vels'].copy()

        if rv is not None:
            prf = lib.interp(vels, vels + rv, prf)

        if amp is not None:
            # prf = prf + amp
            prf = (prf - lib.median(prf)) * amp + 1.0

        scaling = self.conf[mode]['scaling'].copy()
        wrange = self.conf[mode]['wave_range'].copy()

        max_reg = [1.0]
        scale_factor = 1.0
        if scaling['method'] == 'mean':
            scale_factor = lib.mean(prf)

        elif scaling['method'] == 'max':
            per = 100. / scaling['percent']
            if scaling['side'] == 'both':
                max_reg = lib.hstack((prf[:int(len(prf) / per)], prf[-int(len(prf) / per):]))
            elif scaling['side'] == 'left':
                max_reg = prf[:int(len(prf) / per)]
            elif scaling['side'] == 'right':
                max_reg = prf[-int(len(prf) / per):]
            scale_factor = lib.mean(max_reg)

        elif scaling['method'] == 'none':
            scale_factor = 1.0

        elif scaling['method'] == 'region':
            wavs = (vels * lib.mean(wrange)) / ac.c.to(au.kilometer / au.second).value + lib.mean(wrange)

            wavgw = lib.array(
                [lib.mean(wavs[lib.where((wavs >= region[0]) & (wavs <= region[1]))]) for region in
                 scaling['ranges']])
            wavgf = lib.array(
                [lib.mean(prf[lib.where((wavs >= region[0]) & (wavs <= region[1]))]) for region in
                 scaling['ranges']])

            x1, x2, x3 = wavgw
            y1, y2, y3 = wavgf
            denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
            a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
            b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
            c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
            scale_factor = a * wavs ** 2 + b * wavs + c

        prf = prf / scale_factor

        return prf, scale_factor

    def generate_synthetic_lightcurve(self, fssc, fssh, coeffs_cube, lib=np):

        fssp = 1.0 - (fssc + fssh)

        wgt_phot = coeffs_cube[:, 0] * fssp
        wgt_cool = coeffs_cube[:, 1] * fssc
        wgt_hot = coeffs_cube[:, 2] * fssh

        flux = lib.sum(wgt_phot + wgt_cool + wgt_hot)

        return flux

    def get_rv_and_amp(self, x0s, mode, pnoo=None):

        if mode != 'lc':
            vals = []
            for item in ['rv', 'amp']:
                if type(self.conf[mode]['corr'][item]) not in [list, np.ndarray]:
                    if self.conf[mode]['corr'][item] is None:
                        val = np.array([None] * pnoo)[:, None]

                    elif self.conf[mode]['corr'][item] == 'free':
                        val = x0s[self.conf[mode]['mask'][item]]

                    else:
                        val = np.array([self.conf[mode]['corr'][item]] * pnoo)[:, None]

                else:
                    val = np.array(self.conf[mode]['corr'][item])[:, None]

                vals.append(val)
                self.conf[mode]['corr_result'][item] = val

            return vals

        else:
            if self.conf['lc']['corr']['amp'] is None:
                val = 0

            elif self.conf['lc']['corr']['amp'] == 'free':
                val = x0s[self.conf['lc']['mask']['amp']]

            else:
                val = self.conf['lc']['corr']['amp']

            self.conf['lc']['corr_result']['amp'] = val[0] if hasattr(val, '__len__') else val

            return val

    # @profile
    def objective_func(self, x0s, alpha, beta, gamma, delta, lmbd, cpcs):

        noes = self.surface_grid['noes']

        fssc = x0s[:noes]
        fssh = x0s[noes:2 * noes]
        fssp = 1.0 - (fssc + fssh)

        chisqs = {'line': 0.0, 'mol1': 0.0, 'mol2': 0.0, 'lc': 0.0}

        for mode in self.conf:
            if mode != 'lc':
                pnoo = self.idc[mode]['noo']
                pnop = self.idc[mode]['nop']

                rv, amp = self.get_rv_and_amp(x0s=x0s, mode=mode, pnoo=pnoo)
                rv_axis, amp_axis = 0, 0
                if rv.all() is None:
                    rv_axis = None
                    rv = None
                if amp.all() is None:
                    amp_axis = None
                    amp = None

                pvmap = vmap(self.generate_synthetic_profile, in_axes=[None, None, rv_axis, amp_axis, 0, None, None])
                sprf, scale_factors = pvmap(fssc, fssh, rv, amp, cpcs[mode]['coeffs_cube'], mode, jnp)
                oprf = jnp.array(self.idc[mode]['data_cube'][0])
                oprf_errs = jnp.array(self.idc[mode]['data_cube'][1])

                chisqs[mode] = jnp.sum(((oprf - sprf) / oprf_errs) ** 2) / (pnoo * pnop)

                self.conf[mode]['scale_factor'] = scale_factors

            if mode == 'lc':

                lnop = self.idc['lc']['nop']

                amp = self.get_rv_and_amp(x0s=x0s, mode=mode)

                olc = jnp.array(self.idc[mode]['data_cube'][0])
                olc_errs = jnp.array(self.idc[mode]['data_cube'][1])

                lvmap = vmap(self.generate_synthetic_lightcurve, in_axes=[None, None, 0, None])
                slc = lvmap(fssc, fssh, cpcs['lc']['coeffs_cube'], jnp)

                slc += amp

                scaling = self.conf['lc']['scaling']

                scale_factor = 1.0
                if scaling['method'] == 'mean':
                    scale_factor = jnp.mean(slc)

                elif scaling['method'] == 'none':
                    scale_factor = 1.0
                self.conf['lc']['scale_factor'] = scale_factor

                slc /= scale_factor

                chisqs['lc'] = jnp.sum(((olc - slc) / olc_errs) ** 2) / lnop

        alpha_line = alpha * chisqs['line']
        beta_mol1 = beta * chisqs['mol1']
        gamma_mol2 = gamma * chisqs['mol2']
        delta_lc = delta * chisqs['lc']

        total_weighted_chisq = alpha_line + beta_mol1 + gamma_mol2 + delta_lc

        wp = self.surface_grid['grid_areas'] / np.max(self.surface_grid['grid_areas'])

        mem = jnp.sum(wp * (fssc * jnp.log(fssc / self.fsl) + fssh * jnp.log(fssh / self.fsl) +
                            (1.0 - fssp) * jnp.log((1.0 - fssp) / self.fsu))) / self.surface_grid['noes']

        lmbd_mem = lmbd * mem

        ftot = total_weighted_chisq + lmbd_mem

        return ftot, (chisqs['line'], chisqs['mol1'], chisqs['mol2'], chisqs['lc'], alpha_line, beta_mol1, gamma_mol2,
                      delta_lc, total_weighted_chisq, mem, lmbd_mem)

    def minimize(self, x0s, minx0s, maxx0s, maxiter, tol, alpha, beta, gamma, delta, lmbd, cpcs, disp):

        bounds = jnp.array((minx0s, maxx0s))

        optimizer = ScipyBoundedMinimize(fun=self.objective_func, has_aux=True, maxiter=maxiter, tol=tol,
                                         method='L-BFGS-B', options={'disp': disp})
        x0s, info = optimizer.run(x0s, bounds, alpha, beta, gamma, delta, lmbd, cpcs)

        ftot, others = self.objective_func(x0s, alpha, beta, gamma, delta, lmbd, cpcs)

        metrics = np.hstack((others, ftot))

        for i, item in enumerate(self.opt_stats):
            self.opt_stats[item] = metrics[i]

        return x0s, info.iter_num, info.num_fun_eval

    def reconstructor(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, lmbd=1.0, maxiter=100, tol=1e-10, disp=True,
                      cpcs=None):

        if not cpcs:
            cpcs = self.calc_pixel_coeffs(line_times=self.idc['line']['times'], mol1_times=self.idc['mol1']['times'],
                                          mol2_times=self.idc['mol2']['times'], lc_times=self.idc['lc']['times'],
                                          info=disp)

        if isinstance(lmbd, (list, np.ndarray)):
            lmbd = self.lambda_search(alpha=alpha, beta=beta, gamma=gamma, delta=delta, lmbds=lmbd, maxiter=maxiter,
                                      tol=tol, cpcs=cpcs)

        noes = self.surface_grid['noes']

        spotless_fss = jnp.ones(2 * noes) * self.fsl
        x0s = jnp.hstack((spotless_fss, jnp.zeros(self.idc['nofp'])))

        minx0s = jnp.array([self.fsl] * (2 * noes) + [-jnp.inf] * self.idc['nofp'])
        maxx0s = jnp.array([self.fsu] * (2 * noes) + [jnp.inf] * self.idc['nofp'])

        recons_params, nit, nfev = self.minimize(x0s=x0s, minx0s=minx0s, maxx0s=maxx0s, maxiter=maxiter, tol=tol,
                                                 alpha=alpha, beta=beta, gamma=gamma, delta=delta, lmbd=lmbd, cpcs=cpcs,
                                                 disp=disp)

        print()
        print('\033[96m' + '*** Optimization Results ***' + '\033[0m')
        for ositem in self.opt_stats:
            print('\033[93m' + '{:<45}'.format(ositem) + ':' + '\033[0m',
                  '\033[1m' + str(self.opt_stats[ositem]) + '\033[0m')

        recons_fssc = np.asarray(recons_params[:noes], dtype=np.float64)
        recons_fssh = np.asarray(recons_params[noes:2 * noes], dtype=np.float64)
        recons_fssp = 1.0 - (recons_fssc + recons_fssh)

        intensities = ((recons_fssc * self.params['Tcool'] ** 4 + recons_fssh * self.params['Thot'] ** 4 +
                        recons_fssp * self.params['Tphot'] ** 4) / self.params['Tphot'] ** 4)

        self.opt_results['line'] = {}
        self.opt_results['mol1'] = {}
        self.opt_results['mol2'] = {}
        self.opt_results['lc'] = {}

        recons_slc = []
        for mode in self.conf:
            coeffs_cube = cpcs[mode]['coeffs_cube']

            if mode != 'lc':
                rv = self.conf[mode]['corr_result']['rv'].flatten()
                amp = self.conf[mode]['corr_result']['amp'].flatten()

                spotless_sprfs = {}
                recons_sprfs = {}
                for i, itime in enumerate(self.idc[mode]['times']):
                    spotless_sprfs[itime] = {}
                    recons_sprfs[itime] = {}

                    spotless_sprf, _ = self.generate_synthetic_profile(fssc=0.0, fssh=0.0, rv=rv[i], amp=amp[i],
                                                                       coeffs_cube=coeffs_cube[i], mode=mode)
                    recons_sprf, _ = self.generate_synthetic_profile(fssc=recons_fssc, fssh=recons_fssh, rv=rv[i],
                                                                     amp=amp[i], coeffs_cube=coeffs_cube[i], mode=mode)

                    spotless_sprfs[itime]['prf'] = np.array(spotless_sprf)
                    recons_sprfs[itime]['prf'] = np.array(recons_sprf)

                self.opt_results[mode]['spotless_sprfs'] = spotless_sprfs
                self.opt_results[mode]['recons_sprfs'] = recons_sprfs

            if mode == 'lc':
                for i, itime in enumerate(self.idc[mode]['times']):
                    lc_amp = self.conf['lc']['corr_result']['amp']
                    flux = self.generate_synthetic_lightcurve(fssc=recons_fssc, fssh=recons_fssh,
                                                              coeffs_cube=coeffs_cube[i])

                    recons_slc.append(flux + lc_amp)

                recons_slc = np.array(recons_slc) / self.conf['lc']['scale_factor']

        if 'lc' not in self.conf:
            lc_ntimes = (self.params['t0'] + np.linspace(0, 2, 200) * self.params['period'])
            cpcs_lc = self.calc_pixel_coeffs(mode='lc', lc_times=lc_ntimes, info=False, plps=(False, False),
                                             plil=(False, False, False))
            self.opt_results['lc']['ntimes'] = lc_ntimes

            for i, itime in enumerate(lc_ntimes):
                flux = self.generate_synthetic_lightcurve(fssc=recons_fssc, fssh=recons_fssh,
                                                          coeffs_cube=cpcs_lc['lc']['coeffs_cube'][i])
                recons_slc.append(flux)
            recons_slc = np.array(recons_slc) / np.mean(recons_slc)

        self.opt_results['lc']['recons_slc'] = recons_slc

        spotted_area = dipu.get_total_fs(fssc=recons_fssc, fssh=recons_fssh, areas=self.surface_grid['grid_areas'],
                                         lats=self.surface_grid['grid_lats'], incl=self.params['incl'])

        self.opt_results['alpha'] = alpha
        self.opt_results['beta'] = beta
        self.opt_results['gamma'] = gamma
        self.opt_results['delta'] = delta
        self.opt_results['lmbd'] = lmbd
        self.opt_results['recons_fssc'] = recons_fssc
        self.opt_results['recons_fssh'] = recons_fssh
        self.opt_results['recons_fssp'] = recons_fssp
        self.opt_results['ints'] = intensities
        self.opt_results['nit'] = nit
        self.opt_results['nfev'] = nfev
        self.opt_results['total_cool_spotted_area'] = spotted_area[0]
        self.opt_results['partial_cool_spotted_area'] = spotted_area[3]
        self.opt_results['total_hot_spotted_area'] = spotted_area[1]
        self.opt_results['partial_hot_spotted_area'] = spotted_area[4]
        self.opt_results['total_unspotted_area'] = spotted_area[2]
        self.opt_results['partial_unspotted_area'] = spotted_area[5]

        self.mapprojs = dipu.grid_to_rect_map(surface_grid=self.surface_grid, ints=self.opt_results['ints'])

    def lambda_search(self, alpha, beta, gamma, delta, lmbds, maxiter, tol, cpcs):
        global lambda_search_run

        noes = self.surface_grid['noes']
        spotless_fss = np.ones(2 * noes) * self.fsl

        minx0s = np.array([self.fsl] * (2 * noes) + [-jnp.inf] * self.idc['nofp'])
        maxx0s = np.array([self.fsu] * (2 * noes) + [jnp.inf] * self.idc['nofp'])

        optimizer = ScipyBoundedMinimize(fun=self.objective_func, has_aux=True, maxiter=maxiter, tol=tol,
                                         method='L-BFGS-B', options={'disp': False})

        def lambda_search_run(lmbd):
            x0s = jnp.hstack((spotless_fss, jnp.zeros(self.idc['nofp'])))

            bounds = jnp.array((minx0s, maxx0s))

            x0s, info = optimizer.run(x0s, bounds, alpha, beta, gamma, delta, lmbd, cpcs)

            _, others = self.objective_func(x0s, alpha, beta, gamma, delta, lmbd, cpcs)

            return others[8], others[9], lmbd

        results = dipu.mp_search(cpu_num=self.cpu_num, input_args=lmbds, func=lambda_search_run)
        parts = np.array(results)

        sort = np.argsort(parts[:, 2])
        self.opt_results['total_chisqs'] = parts[:, 0][sort]
        self.opt_results['mems'] = parts[:, 1][sort]
        self.opt_results['lmbds'] = parts[:, 2][sort]

        from kneebow.rotor import Rotor

        rotor = Rotor()
        rotor.fit_rotate(np.vstack((self.opt_results['total_chisqs'], self.opt_results['mems'])).T)
        maxcurve = rotor.get_elbow_index()

        self.opt_results['maxcurve'] = maxcurve

        return self.opt_results['lmbds'][maxcurve]

    def grid_search(self, fit_params, opt_params, info=True, save=False):
        global grid_search_run

        grid_cpu_num = self.cpu_num
        self.cpu_num = 1

        optp = {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'lmbd': 1.0, 'maxiter': 100, 'tol': 1e-10,
                'disp': True}
        if opt_params is not None:
            for oitem in opt_params:
                if oitem in optp:
                    optp[oitem] = opt_params[oitem]
                else:
                    print('\033[93m' + 'Warning: ' + oitem + ' is not a valid parameter for opt_params!' + '\033[0m')

        if type(optp['lmbd']) in [list, np.ndarray]:
            optp['lmbd'] = np.min(optp['lmbd'])

        fp_keys = list(fit_params.keys())
        for citem in fp_keys:
            if len(fit_params[citem]) == 1:
                if citem == 'eqw' and 'line' in self.conf:
                    self.conf['line']['eqw'] = fit_params[citem][0]

                elif citem == 'eqw_phot':
                    self.conf['line']['eqw']['phot'] = fit_params[citem][0]

                elif citem == 'eqw_cool':
                    self.conf['line']['eqw']['cool'] = fit_params[citem][0]

                elif citem == 'eqw_hot':
                    self.conf['line']['eqw']['hot'] = fit_params[citem][0]

                else:
                    self.params[citem] = fit_params[citem][0]

                del fit_params[citem]

        fit_params_mg = np.array(np.meshgrid(*list(fit_params.values()))).T.reshape(-1, len(fit_params))

        chisq_grid = {}
        for i, citem in enumerate(fit_params):
            chisq_grid[citem] = fit_params_mg[:, i]

        def grid_search_run(fpmg):

            for j, item in enumerate(chisq_grid):
                if item == 'eqw' and 'line' in self.conf:
                    self.conf['line']['eqw'] = fpmg[j]

                elif item == 'eqw_phot' and 'line' in self.conf:
                    self.conf['line']['eqw']['phot'] = fpmg[j]

                elif item == 'eqw_cool' and 'line' in self.conf:
                    self.conf['line']['eqw']['cool'] = fpmg[j]

                elif item == 'eqw_hot' and 'line' in self.conf:
                    self.conf['line']['eqw']['hot'] = fpmg[j]

                else:
                    self.params[item] = fpmg[j]

            self.construct_surface_grid(method=self.surface_grid['method'], noes=self.surface_grid['init_noes'],
                                        nlats=self.surface_grid['nlats'], nside=self.surface_grid['nside'],
                                        info=False)

            cpcs = self.calc_pixel_coeffs(line_times=self.idc['line']['times'], mol1_times=self.idc['mol1']['times'],
                                          mol2_times=self.idc['mol2']['times'], lc_times=self.idc['lc']['times'],
                                          plps=(True, False), plil=(True, False, True), info=False)

            noes = self.surface_grid['noes']

            x0s = jnp.hstack((jnp.ones(2 * noes) * self.fsl, jnp.zeros(self.idc['nofp'])))

            minx0s = jnp.array([self.fsl] * (2 * noes) + [-jnp.inf] * self.idc['nofp'])
            maxx0s = jnp.array([self.fsu] * (2 * noes) + [jnp.inf] * self.idc['nofp'])

            self.minimize(x0s=x0s, minx0s=minx0s, maxx0s=maxx0s, maxiter=optp['maxiter'], tol=optp['tol'],
                          alpha=optp['alpha'], beta=optp['beta'], gamma=optp['gamma'], delta=optp['delta'],
                          lmbd=optp['lmbd'], cpcs=cpcs, disp=False)

            if info:
                output = self.params.copy()
                output['eqw'] = self.conf['line']['eqw']
                output['Loss Function Value'] = self.opt_stats['Loss Function Value']

                print()
                print('\033[96m' + '*** Optimization Results ***' + '\033[0m')
                for out in output:
                    print('\033[93m' + '{:<11}'.format(out) + ':' + '\033[0m', '\033[1m' + str(output[out]) + '\033[0m')

                # output['Set'] = str(say) + '/' + str(nog)

            return self.opt_stats['Loss Function Value']

        chisqs = dipu.mp_search(cpu_num=grid_cpu_num, func=grid_search_run, input_args=fit_params_mg)
        chisq_grid['chisqs'] = np.array(chisqs)

        self.cpu_num = grid_cpu_num

        if save:
            file = open(save, 'wb')
            pickle.dump(chisq_grid, file)
            file.close()

        # file = open('test_gs.pkl', 'rb')
        # chisq_grid = pickle.load(file)
        # file.close()

        dipu.make_grid_contours(chisq_grid)

    def plot(self, plot_params=None):

        plotp = {'line_sep_prf': 0.4, 'line_sep_res': 0.01, 'mol_sep_prf': 0.4, 'mol_sep_res': 0.01,
                 'show_err_bars': True, 'fmt': '%0.3f',
                 'markersize': 2, 'linewidth': 1, 'fontsize': 15, 'ticklabelsize': 12}
        if plot_params is not None:
            for pitem in plot_params:
                if pitem in plotp:
                    plotp[pitem] = plot_params[pitem]
                else:
                    print('\033[93m' + 'Warning: ' + pitem + ' is not a valid parameter for plot_params!' + '\033[0m')

        from PyQt5 import QtWidgets
        from plot_GUI import PlotGUI

        msg = '\n' + '\033[94m' + 'Preparing the GUI for display. Please wait...' + '\033[0m'
        sys.stdout.write('\r' + msg)

        app = QtWidgets.QApplication(sys.argv)
        pg = PlotGUI(self, plot_params=plotp)
        pg.show()

        msg = '\033[94m' + "The GUI is currently being displayed..." + '\033[0m' + '\n'
        sys.stdout.write('\r' + msg)

        sys.exit(app.exec_())

    def test(self, spots_params, modes_input, opt_params=None, plot_params=None):

        plotp = {'line_sep_prf': 0.03, 'line_sep_res': 0.01, 'mol_sep_prf': 0.03, 'mol_sep_res': 0.01,
                 'show_err_bars': True, 'fmt': '%0.3f', 'markersize': 2, 'linewidth': 1, 'fontsize': 15,
                 'ticklabelsize': 12}
        if plot_params is not None:
            for pitem in plot_params:
                if pitem in plotp:
                    plotp[pitem] = plot_params[pitem]
                else:
                    print('\033[93m' + 'Warning: ' + pitem + ' is not a valid parameter for plot_params!' + '\033[0m')

        optp = {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'lmbd': 1.0, 'maxiter': 100, 'tol': 1e-10,
                'disp': True}
        if opt_params is not None:
            for oitem in opt_params:
                if oitem in optp:
                    optp[oitem] = opt_params[oitem]
                else:
                    print('\033[93m' + 'Warning: ' + oitem + ' is not a valid parameter for opt_params!' + '\033[0m')

        for mode in self.conf:
            for item in modes_input[mode]:
                self.idc[mode][item] = modes_input[mode][item]
                if item == 'times':
                    if mode != 'lc':
                        self.idc[mode]['noo'] = len(modes_input[mode][item])
                    else:
                        self.idc[mode]['nop'] = len(modes_input[mode][item])
                if item == 'vels':
                    self.idc[mode]['nop'] = len(modes_input[mode][item])

        fake_fssc, fake_fssh = dipu.generate_spotted_surface(surface_grid=self.surface_grid, spots_params=spots_params)

        cpcs = self.calc_pixel_coeffs(line_times=self.idc['line']['times'], mol1_times=self.idc['mol1']['times'],
                                      mol2_times=self.idc['mol2']['times'], lc_times=self.idc['lc']['times'], info=True)

        np.random.seed(0)
        for mode, mitem in self.conf.items():
            if mitem['corr']['rv'] in [0, 'free', None]:
                rvs = [None] * self.idc[mode]['noo']

            elif type(mitem['corr']['rv']) in [list, np.ndarray]:
                rvs = mitem['corr']['rv']

            else:
                rvs = [mitem['corr']['rv']] * self.idc[mode]['noo']

            if mitem['corr']['amp'] in [0, 'free', None]:
                amps = [None] * self.idc[mode]['noo']

            elif type(mitem['corr']['amp']) in [list, np.ndarray]:
                amps = mitem['corr']['amp']

            else:
                amps = [mitem['corr']['amp']] * self.idc[mode]['noo']

            if mode != 'lc':
                for i, itime in enumerate(self.idc[mode]['times']):
                    sprf, _ = self.generate_synthetic_profile(fssc=fake_fssc, fssh=fake_fssh, rv=rvs[i], amp=amps[i],
                                                              coeffs_cube=cpcs[mode]['coeffs_cube'][i], mode=mode,
                                                              lib=np)

                    same_random_line_err = np.random.normal(0.0, np.mean(sprf) / self.idc[mode]['snr'],
                                                            self.idc[mode]['nop'])
                    fprf = sprf + same_random_line_err
                    fperr = 2.0 * np.ones(self.idc[mode]['nop']) * (np.mean(sprf) / self.idc[mode]['snr'])
                    self.idc[mode]['data'].append(np.vstack((self.idc[mode]['vels'], fprf, fperr)).T)

            else:
                slc = np.zeros(self.idc[mode]['nop'])
                for i, itime in enumerate(self.idc[mode]['times']):
                    flux = self.generate_synthetic_lightcurve(fssc=fake_fssc, fssh=fake_fssh,
                                                              coeffs_cube=cpcs[mode]['coeffs_cube'][i],
                                                              lib=np)

                    slc[i] = flux

                if amps[0] is not None:
                    slc += amps[0]

                slc /= self.conf['lc']['scale_factor']

                same_random_lc_err = np.random.normal(0.0, np.mean(slc) / self.idc[mode]['snr'], self.idc[mode]['nop'])
                flc = slc + same_random_lc_err

                flerr = 2.0 * np.ones(self.idc[mode]['nop']) * (np.mean(slc) / self.idc[mode]['snr'])

                self.idc[mode]['data'] = np.vstack((flc, flerr)).T

        # file = open('line_and_lc_test.txt', 'wb')
        # pickle.dump(self.idc, file)
        # file.close()

        self.set_input_data()

        self.reconstructor(alpha=optp['alpha'], beta=optp['beta'], gamma=optp['gamma'], delta=optp['delta'],
                           lmbd=optp['lmbd'], maxiter=optp['maxiter'], tol=optp['tol'], disp=optp['disp'],
                           cpcs=cpcs)

        fake_total_fs = dipu.get_total_fs(fssc=fake_fssc, fssh=fake_fssh, areas=self.surface_grid['grid_areas'],
                                          lats=self.surface_grid['grid_lats'], incl=self.params['incl'])
        recons_total_fs = dipu.get_total_fs(fssc=self.opt_results['recons_fssc'], fssh=self.opt_results['recons_fssh'],
                                            areas=self.surface_grid['grid_areas'], lats=self.surface_grid['grid_lats'],
                                            incl=self.params['incl'])

        for mode in self.conf:
            if mode != 'lc':
                dipu.test_prf_plot(DIP=self, mode=mode, plotp=plotp)

        dipu.test_lc_plot(DIP=self, plotp=plotp)

        dipu.test_map_plot(DIP=self, fake_fssc=fake_fssc, fake_fssh=fake_fssh, fake_total_fs=fake_total_fs,
                           recons_total_fs=recons_total_fs, plotp=plotp)

        if len(self.opt_results['lmbds']) > 0:
            dipu.lmbds_plot(DIP=self, plotp=plotp)

        plt.show()

    def save_obs_model_data(self, path):

        obs_model_data_dict = {}
        for mode in self.conf:
            odata = self.idc[mode]['data'].copy()
            otimes = self.idc[mode]['times'].copy()

            obs_model_data_dict[mode] = {}
            if mode != 'lc':
                spl_line_slps = self.opt_results[mode]['spotless_sprfs'].copy()
                rcs_line_slps = self.opt_results[mode]['recons_sprfs'].copy()

                for time in otimes:

                    obs_model_data_dict[mode][time] = {}

                    iodata = odata[time]
                    isdata = spl_line_slps[time]['prf']
                    irdata = rcs_line_slps[time]['prf']

                    obs_model_data_dict[mode][time]['velocities'] = iodata['vels']
                    obs_model_data_dict[mode][time]['intensities'] = iodata['prf']
                    obs_model_data_dict[mode][time]['errors'] = iodata['errs']

                    obs_model_data_dict[mode][time]['spotless-model'] = isdata
                    obs_model_data_dict[mode][time]['spotted-model'] = irdata

            else:
                obs_model_data_dict[mode]['times'] = otimes
                obs_model_data_dict[mode]['fluxs'] = odata['fluxs']
                obs_model_data_dict[mode]['errors'] = odata['errs']
                obs_model_data_dict[mode]['model'] = self.opt_results[mode]['recons_slc'].copy()

        file = open(path, 'wb')
        pickle.dump(obs_model_data_dict, file)
        file.close()

    def save_map_data(self, path):

        recons_fssc = self.opt_results['recons_fssc']
        recons_fssh = self.opt_results['recons_fssh']
        recons_fssp = 1.0 - (recons_fssc + recons_fssh)

        ints = self.opt_results['ints'].copy()

        map_proj_data_dict = {'grid latitudes': np.rad2deg(self.surface_grid['grid_lats']),
                              'grid longitudes': np.rad2deg(self.surface_grid['grid_longs']),
                              'photosphere filling factors': recons_fssp, 'cool spot filling factors': recons_fssc,
                              'hot spot filling factors': recons_fssh, 'normalized intensities': ints,
                              'map2D': self.mapprojs['crmap'].copy(),
                              'meshgrid latitudes': self.mapprojs['mlats'].copy(),
                              'meshgrid longitudes': self.mapprojs['mlongs'].copy()}

        file = open(path, 'wb')
        pickle.dump(map_proj_data_dict, file)
        file.close()
