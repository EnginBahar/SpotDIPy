import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import vmap, config as jax_config, numpy as jnp, debug as jax_debug
from jaxopt import ScipyBoundedMinimize
from astropy import units as au, constants as ac
import platform
import copy
from PyQt5 import QtWidgets
try:
    from . import utils as dipu
    from .plot_GUI import PlotGUI
except ImportError:
    import utils as dipu
    from plot_GUI import PlotGUI
from functools import partial


class SpotDIPy:

    def __init__(self, cpu_num=1, platform_name='cpu', info=True):
        """
        Initializes the SpotDIPy class.

        This class is used for modeling the surface brightness distributions of single stars using Doppler imaging technique.

        Parameters:
        -----------
        cpu_num : int, optional
            Number of CPU cores to be used for all other parallelization tasks except for the optimization process.
            The default is 1.
        platform_name : str, optional
            The computation platform used during the optimization process to find the best-fitting model to the
            observational data, either 'cpu' or 'cuda'. The default is 'cpu'. This setting does not affect other
            parallelization tasks.
            Note: If 'cuda' is selected and the code is executed multiple times simultaneously, there is a possibility
            of running out of GPU memory, which may cause the code to stop execution. This is particularly relevant when
            using the grid_search function.
        info : bool, optional
            Whether to display information messages during initialization. Can be True or False. The default is True.

        Raises:
        -------
        ValueError
            If cpu_num is not an integer.
        KeyError
            If platform_name is not 'cpu' or 'cuda'.
        KeyError
            If info is not True or False.
        """

        if not isinstance(cpu_num, (int, float, np.integer, np.floating)):
            raise ValueError("'cpu_num' keyword must be an integer number")
        self.cpu_num = int(cpu_num)

        if platform_name not in ['cpu', 'cuda']:
            raise KeyError("'platform_name' keyword must be one of two options: 'cpu' or 'cuda'")

        if info not in [True, False]:
            raise KeyError("'info' keyword must be one of two options: True or False")

        self.platform_name = platform_name
        if platform_name == 'cuda':
            platform_name += ",cpu"

        jax_config.update("jax_platforms", platform_name)
        jax_config.update("jax_enable_x64", True)

        if info:
            print(f"\033[93mNumber of CPU cores used: {self.cpu_num}\033[0m")

        self.params = {
            't0': None,
            'period': None,
            'Tphot': None,
            'Tcool': None,
            'Thot': None,
            'incl': None,
            'R': None,
            'vsini': None,
            'vrt': None,
            'mass': None,
            'dOmega': None,
            'resolution': None,
            'gd_beta': None
        }

        def_conf_temp = {
            'mode': 'off',
            'scaling': {'method': None, 'percent': None, 'side': None},
            'corr': {'rv': None, 'amp': None},
            'corr_result': {'rv': None, 'amp': None},
            'mask': {'rv': None, 'amp': None},
            'scale_factor': {}
        }

        self.conf = {
            'line': {'corr': {'rv': 0, 'amp': 0}, 'eqw': {}, 'wave_range': None, **copy.deepcopy(def_conf_temp)},
            'mol1': {'wave_range': None, **copy.deepcopy(def_conf_temp)},
            'mol2': {'wave_range': None, **copy.deepcopy(def_conf_temp)},
            'lc': {'passband': None, 'corr': {'rv': None, 'amp': None}, **copy.deepcopy(def_conf_temp), 'scale_factor': 1.0}
        }

        common_idc_template = {
            'times': [],
            'data': [],
            'data_cube': [],
            'vels': None,
            'noo': 0,
            'nop': None,
            'snr': None,
            'read_noise': 5.,
            'lp_vels': None,
            'phot_lp_data_raw': None,
            'cool_lp_data_raw': None,
            'hot_lp_data_raw': None
        }

        self.idc = {
            'line': copy.deepcopy(common_idc_template),
            'mol1': copy.deepcopy(common_idc_template),
            'mol2': copy.deepcopy(common_idc_template),
            'lc': {
                'times': [],
                'data': [],
                'norsp': None,
                'noo': 1,
                'nop': None,
                'snr': None,
                'read_noise': 5.
            }
        }

        self.opt_stats = {
            'Chi-square for Line Profile(s)': None,
            'Chi-square for Light Curve Profile': None,
            'Alpha * Line Profile(s) Chi-square': None,
            'Delta * Light Curve Profile Chi-square': None,
            'Total Weighted Chi-square': None,
            'Entropy': None,
            'Lambda * Entropy': None,
            'RMS of Line Residauls': None,
            'RMS of Light Curve Residauls': None,
            'Total RMS of Residauls': None,
            'Loss Function Value': None
        }

        self.opt_results = {'line': {}, 'mol1': {}, 'mol2': {}, 'lc': {}, 'lmbds': []}
        self.ldi = {'ld_params': {}, 'line': {}, 'mol1': {}, 'mol2': {}, 'lc': {}}
        self.surface_grid = {}

        self.fsl = 1e-7
        self.fsu = 1.0 - self.fsl

    def set_param(self, param_name, value):
        """
        Sets the specified parameter for the stellar model.

        Parameters:
        -----------
        param_name : str
            The name of the parameter to set. Possible values include:
            - 't0': Reference time (typically a Julian date).
            - 'period': Rotational period of the star (in days).
            - 'Tphot': Temperature of the photosphere (in Kelvin).
            - 'Tcool': Temperature of cooler regions (in Kelvin).
            - 'Thot': Temperature of hotter regions (in Kelvin).
            - 'incl': Inclination angle of the star's rotational axis (in degrees).
            - 'R': Stellar equator radius (in solar radius).
            - 'vsini': Projected rotational velocity (in km/s).
            - 'vrt': Radial-tangential macroturbulence velocity (in km/s).
            - 'mass': Stellar mass (in solar masses).
            - 'dOmega': Differential rotation parameter.
            - 'resolution': Resolution of the observed spectrum.

        value : float or int
            The value to assign to the specified parameter.

        Raises:
        -------
        ValueError
            If the provided param_name is not recognized or the value is of an incorrect type.
        """

        if param_name not in self.params.keys():
            raise KeyError(f"'{param_name}' is not a valid parameter name. Please choose one of"
                           f" {list(self.params.keys())}")

        else:
            if dipu.isfloatable(value) is False:
                raise ValueError(f"Invalid value ({value}) for the '{param_name}' parameter. Please set it to"
                                 f" an appropriate float or int value.")

            else:
                self.params[param_name] = float(value)

        if self.params['vsini'] is not None and self.params['R'] is not None:
            raise KeyError("Only one of the parameters, R or vsini, should be set.")

    def set_limb_darkening_params(self, mh, law='linear', model='mps2', mu_min=0.1, data_path='exotic_ld_data'):
        """
        Sets the limb darkening parameters for the stellar model for the calculation limb darkening coefficient using
        the exotic-ld package.

        For more information about the exotic-ld package: https://exotic-ld.readthedocs.io/en/latest/

        Parameters:
        -----------
        mh : float
            Metallicity of the star (typically [Fe/H]), used to select the appropriate model grid for limb darkening
            coefficient calculations.
        law : str, optional
            The law used to fit I(mu) to determine limb darkening coefficients. Options include:
            - 'linear': Linear limb darkening law.
            - 'quadratic': Quadratic limb darkening law.
            - 'square-root': Square-root limb darkening law.
            The default is 'linear'.
        model : str, optional
            The model grid used to model the stellar intensity (I) as a function of radial position on the stellar disc
            (mu) based on pre-computed grids spanning a range of metallicity, effective temperature, and surface
            gravity. Default is 'mps2'.
        mu_min : float, optional
            The minimum mu, determining the range over which limb darkening coefficient is calculated.
            The default is 0.1.
        data_path : str, optional
            Path to the directory containing the required data files. If the files are not present,
            they will be downloaded from the internet. The default is 'exotic_ld_data'.

        Raises:
        -------
        ValueError
            If the provided parameters are not recognized or their values are of incorrect types.
        """

        if dipu.isfloatable(mh) is False:
            raise ValueError("'mh' argument must be a float value")

        if law not in ['linear', 'square-root', 'quadratic']:
            raise KeyError("'law' argument must be one of three options: 'linear', 'square-root' or 'quadratic'")

        if model not in ['kurucz', 'mps1', 'mps2', 'stagger']:
            raise KeyError("'model' argument must be one of four options: 'kurucz', 'mps1', 'mps2' or 'stagger'")

        if dipu.isfloatable(mu_min) is False:
            raise ValueError("'mu_min' argument must be a float value")

        if data_path is None:
            data_path = 'exotic_ld_data'

        else:
            if not os.path.exists(data_path):
                raise FileNotFoundError(data_path + ": directory not found!")

            if not os.path.isdir(data_path):
                raise NotADirectoryError(data_path + ": not a directory!")

        self.ldi['ld_params']['mh'] = mh
        self.ldi['ld_params']['law'] = law
        self.ldi['ld_params']['model'] = model
        self.ldi['ld_params']['mu_min'] = mu_min
        self.ldi['ld_params']['data_path'] = data_path

    def set_conf(self, conf):

        common_eqw_err = (
            "'eqw' keyword must be a positive value or a dictionary in the format:\n"
            "  {'phot': a positive value, 'cool': a positive value, 'hot': a positive value}"
        )

        bcerr = "'{}': a float value, list of floats, 'free' or None"
        common_corr_err = ("For the '{}' dictionary in the 'set_conf', 'corr' keyword must be a dictionary with the"
                           " format: {{{}, {}}}")

        for mode in conf:
            corr_txt1 = bcerr.format('amp') if mode == 'lc' else bcerr.format('rv')
            corr_txt2 = bcerr.format('amp') if mode == 'line' else ''

            if mode not in self.conf:
                raise KeyError(f"'{mode}' is not a valid 'conf' keyword. Please choose one or more of the"
                                                  " following keywords: 'line', 'mol1', 'mol2' or 'lc'")

            else:
                if conf[mode].keys() != set(list(self.conf[mode].keys())[:-3]):
                    raise KeyError(f"All required keywords for the '{mode}' must be set, including "
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
                                                     "containing at least 2 positive wavelength values in Angstrom")
                                else:
                                    for wave in conf[mode][par]:
                                        if not dipu.isfloatable(wave) or float(wave) <= 0.0:
                                            raise ValueError("'wave_range' keyword must be a 1D list, tuple or numpy"
                                                             " array containing at least 2 positive wavelength values"
                                                             " in Angstrom")

                            if par == 'eqw':
                                if conf[mode][par] is not None:
                                    if isinstance(conf[mode][par], dict):
                                        if sorted(list(conf[mode][par].keys())) != sorted(['phot', 'cool', 'hot']):
                                            raise KeyError(common_eqw_err)

                                        for subpar in conf[mode][par]:
                                            if not dipu.isfloatable(conf[mode][par][subpar]):
                                                raise ValueError(common_eqw_err)

                                            elif float(conf[mode][par][subpar]) <= 0.0:
                                                raise ValueError(common_eqw_err)

                                    else:
                                        if not dipu.isfloatable(conf[mode][par]):
                                            raise ValueError(common_eqw_err)

                                        elif float(conf[mode][par]) <= 0.0:
                                            raise ValueError(common_eqw_err)

                            if par == 'corr':
                                if isinstance(conf[mode][par], dict) and len(conf[mode][par]) > 0:
                                    for subpar in conf[mode][par]:
                                        if subpar not in ['rv', 'amp']:
                                            raise KeyError(common_corr_err.format(mode, corr_txt1, corr_txt2))

                                        if isinstance(conf[mode][par][subpar], (list, tuple, np.ndarray)):
                                            items = conf[mode][par][subpar]
                                            for item in items:
                                                if not dipu.isfloatable(item):
                                                    raise KeyError(common_corr_err.format(mode, corr_txt1, corr_txt2))
                                        else:
                                            if (not dipu.isfloatable(conf[mode][par][subpar]) and
                                                    not conf[mode][par][subpar] in ['free', None]):
                                                raise KeyError(common_corr_err.format(mode, corr_txt1, corr_txt2))

                                else:
                                    raise KeyError(common_corr_err.format(mode, corr_txt1, corr_txt2))

                            if mode in ['mol1', 'mol2']:
                                if 'amp' in conf[mode]['corr'].keys():
                                    raise KeyError(common_corr_err.format(mode, corr_txt1, corr_txt2))

                            if mode == 'lc':
                                if 'rv' in conf[mode]['corr'].keys():
                                    raise KeyError(common_corr_err.format(mode, corr_txt1, corr_txt2))

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

        if self.params['R'] is None:
            radius_eq = dipu.calc_radius(vsini=self.params['vsini'], incl=self.params['incl'], period=self.params['period'])
        else:
            radius_eq = self.params['R']
        self.radius_eq = radius_eq

        methods = ['trapezoid', 'phoebe2_marching', 'healpy']
        if method not in methods:
            raise KeyError(f"'{method}' argument must be one of three options: 'trapezoid', 'phoebe2_marching'"
                                                " or 'healpy'")

        try:
            omega, requiv, rp = dipu.calc_omega_and_requiv(mass=self.params['mass'], period=self.params['period'],
                                                           re=radius_eq)
            self.omega = omega
        except:
            raise ValueError("The rotation rate cannot be calculated! Probably, one or more of the 'R', 'vsini',"
                             " 'incl', 'period', and 'mass' parameters are not in the appropriate values.")
        if omega > 1.0:
            raise ValueError("The rotation rate exceeds 1! Probably, one or more of the 'R', 'vsini', 'incl', "
                             "'period', and 'mass' parameters are not in the appropriate values.")

        if info:
            print('\033[92mConstructing stellar surface grid...\033[0m')

        if method == 'phoebe2_marching':
            dipu.p2m_surface_grid(container=self.surface_grid, requiv=requiv, noes=noes, t0=0,
                                  period=self.params['period'], mass=self.params['mass'])

        elif method == 'trapezoid':
            dipu.td_surface_grid(container=self.surface_grid, omega=omega, nlats=nlats, radius=radius_eq,
                                 mass=self.params['mass'], cpu_num=self.cpu_num)

        elif method == 'healpy':
            if platform.system() == "Windows":
                raise OSError("'healpy' is not supported on Windows. Please select 'trapezoid' or 'phoebe2_marching'")
            else:
                dipu.hp_surface_grid(container=self.surface_grid, omega=omega, nside=nside, radius=radius_eq,
                                     mass=self.params['mass'], cpu_num=self.cpu_num)

        if self.params['gd_beta'] is not None:
            gdcs = dipu.calc_gds_classic(self.surface_grid['grid_loggs'], beta=self.params['gd_beta'])
        else:
            gdcs = dipu.calc_gds_LR(omega=omega, thetas=self.surface_grid['grid_lats'] + np.pi / 2.0)

        self.surface_grid['Tgdc'], self.surface_grid['Fgdc'] = gdcs
        self.surface_grid['noes'] = len(self.surface_grid['grid_lats'])
        self.surface_grid['coslats'] = np.cos(self.surface_grid['grid_lats'])
        self.surface_grid['sinlats'] = np.sin(self.surface_grid['grid_lats'])
        self.surface_grid['cosi'] = np.cos(np.deg2rad(self.params['incl']))
        self.surface_grid['sini'] = np.sin(np.deg2rad(self.params['incl']))
        self.surface_grid['norm_grid_areas'] = self.surface_grid['grid_areas'] / np.sum(self.surface_grid['grid_areas'])

        if self.params['R'] is None:
            veq = self.params['vsini'] / self.surface_grid['sini']
        else:
            veq = (2.0 * np.pi * (radius_eq * au.solRad.to(au.km))) / (self.params['period'] * au.day.to(au.second))
            self.params['vsini'] = veq * self.surface_grid['sini']

        if info:
            print('\033[96mNumber of total surface element:\033[0m', self.surface_grid['noes'])
            if self.params['R'] is None:
                print('\033[96mEquatorial radius:\033[0m', np.round(radius_eq, 3), 'SolRad')
            else:
                print('\033[96mProjected equatorial rotational velocity:\033[0m',
                      np.round(self.params['vsini'], 3), 'km/s')
            print('\033[96mEquatorial rotational velocity:\033[0m', np.round(veq, 2), 'km/s')
            print('\033[96mMean surface gravity:\033[0m', np.round(np.mean(self.surface_grid['grid_loggs']), 2),
                  'dex')

        self.surface_grid['ctps'] = self.params['Tphot'] * self.surface_grid['Tgdc']
        self.surface_grid['ctcs'] = self.params['Tcool'] * self.surface_grid['Tgdc']
        self.surface_grid['cths'] = self.params['Thot'] * self.surface_grid['Tgdc']

        if test:
            xyzs = self.surface_grid['grid_xyzs']
            if self.surface_grid['method'] == 'p2m':
                scalars1 = self.surface_grid['Fgdc'].copy()
                scalars2 = self.surface_grid['grid_areas'].copy()
                scalars3 = self.surface_grid['grid_lats'].copy()
                scalars4 = self.surface_grid['grid_longs'].copy()

            else:
                scalars1 = np.repeat(self.surface_grid['Fgdc'], 2)
                scalars2 = np.repeat(self.surface_grid['grid_areas'], 2)
                scalars3 = np.repeat(self.surface_grid['grid_lats'], 2)
                scalars4 = np.repeat(self.surface_grid['grid_longs'], 2)

            dipu.grid_test(xyzs=xyzs, scalars=scalars1, bar_title='Gravity Darkening Coeffs.',
                           win_title='Gravity Darkening Coeffs. Distribution')
            dipu.grid_test(xyzs=xyzs, scalars=scalars2, bar_title='Area (SolRad^2)',
                           win_title='Surface Element Area Distribution')
            dipu.grid_test(xyzs=xyzs, scalars=scalars3, bar_title='Latitude (radians)',
                           win_title='Latitude Distribution')
            dipu.grid_test(xyzs=xyzs, scalars=scalars4, bar_title='Longitude (radians)',
                           win_title='Longitude Distribution')

    def prep_ld_and_int_lpt(self, info=False, grid_search=False):

        if info:
            print('\033[92m' + 'Preparing limb-darkening and intensity lookup table...' + '\033[0m')

        teffps = np.full(self.surface_grid['noes'], self.params['Tphot'])
        teffcs = np.full(self.surface_grid['noes'], self.params['Tcool'])
        teffhs = np.full(self.surface_grid['noes'], self.params['Thot'])

        loggs = self.surface_grid['grid_loggs'].copy()

        for mode in self.conf:
            if mode != 'lc':
                ld_rgi, ld_info = dipu.ldcs_rgi_prep(teffps=teffps, teffcs=teffcs, teffhs=teffhs, loggs=loggs,
                                            law=self.ldi['ld_params']['law'], mh=self.ldi['ld_params']['mh'],
                                            wrange=self.conf[mode]['wave_range'], model=self.ldi['ld_params']['model'],
                                            data_path=self.ldi['ld_params']['data_path'],
                                            mu_min=self.ldi['ld_params']['mu_min'], cpu_num=self.cpu_num)

                ld_rgi['phot'][-1] *= self.surface_grid['Fgdc']
                ld_rgi['cool'][-1] *= self.surface_grid['Fgdc']
                ld_rgi['hot'][-1] *= self.surface_grid['Fgdc']
                self.ldi[mode] = ld_rgi

            else:
                ld_rgi, ld_info = dipu.ldcs_rgi_prep(teffps=teffps, teffcs=teffcs, teffhs=teffhs, loggs=loggs,
                                            law=self.ldi['ld_params']['law'],  mh=self.ldi['ld_params']['mh'],
                                            passband=self.conf[mode]['passband'], model=self.ldi['ld_params']['model'],
                                            data_path=self.ldi['ld_params']['data_path'],
                                            mu_min=self.ldi['ld_params']['mu_min'],  cpu_num=self.cpu_num)

                ld_rgi['phot'][-1] *= self.surface_grid['Fgdc']
                ld_rgi['cool'][-1] *= self.surface_grid['Fgdc']
                ld_rgi['hot'][-1] *= self.surface_grid['Fgdc']
                self.ldi[mode] = ld_rgi

        if not grid_search:
            if 'lc' not in self.conf and len(self.conf) == 1:
                self.ldi['lc'] = self.ldi[list(self.conf.keys())[0]]

            elif 'lc' not in self.conf and len(self.conf) > 1:
                wrange = []
                for mode in self.conf:
                    wrange.append(self.conf[mode]['wave_range'])

                ld_rgi, ld_info = dipu.ldcs_rgi_prep(teffps=teffps, teffcs=teffcs, teffhs=teffhs, loggs=loggs,
                                            law=self.ldi['ld_params']['law'], mh=self.ldi['ld_params']['mh'],
                                            wrange=[np.min(wrange), np.max(wrange)], model=self.ldi['ld_params']['model'],
                                            data_path=self.ldi['ld_params']['data_path'],
                                            mu_min=self.ldi['ld_params']['mu_min'], cpu_num=self.cpu_num)

                ld_rgi['phot'][-1] *= self.surface_grid['Fgdc']
                ld_rgi['cool'][-1] *= self.surface_grid['Fgdc']
                ld_rgi['hot'][-1] *= self.surface_grid['Fgdc']
                self.ldi['lc'] = ld_rgi

        if info:
            print(f"\033[96mTemperature range of the {self.ldi['ld_params']['model']} stellar atmosphere models"
                  f" to be used:\033[0m", ld_info[0], 'K')
            print(f"\033[96mSurface gravity range of the {self.ldi['ld_params']['model']} stellar atmosphere models"
                  f" to be used:\033[0m", ld_info[1], 'dex')
            print(f"\033[96mMetallicty of the {self.ldi['ld_params']['model']} stellar atmosphere models"
                  f" to be used:\033[0m", ld_info[2], 'dex')

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
        tnoo = 0
        for mode in self.conf:
            tnoo += self.idc[mode]['noo']
            for item in self.conf[mode]['corr']:
                if isinstance(self.conf[mode]['corr'][item], str) and self.conf[mode]['corr'][item] == 'free':
                    nofp += self.idc[mode]['noo']
        self.idc['nofp'] = nofp
        self.idc['tnoo'] = tnoo

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

    def _ensure_f64(self, *arrays):
        """Convert all inputs to float64 ndarrays."""
        return tuple(np.asarray(a, dtype=np.float64) for a in arrays)

    def calc_pixel_coeffs(self, mode=None, line_times=None, lc_times=None,
                          info=True, plps=(True, True), plil=(True, True, False)):

        sg = self.surface_grid
        p = self.params

        # --- Differential rotation periods & velocities ---
        plats = (2.0 * np.pi * p['period'] /
                 (2.0 * np.pi - p['period'] * p['dOmega'] * sg['sinlats'] ** 2))
        vlats = (2.0 * np.pi * sg['grid_rs'] * au.solRad.to(au.km) * sg['coslats'] /
                 (plats * au.day.to(au.second)))

        # --- Prep steps (conditional) ---
        if plil[0]:
            self.prep_ld_and_int_lpt(info=plil[1], grid_search=plil[2])
        if plps[0]:
            self.prepare_lps(info=plps[1])

        # --- Resolve modes & times ---
        modes = list(self.conf.keys()) if mode is None else [mode]
        all_times = {'line': line_times, 'lc': lc_times}

        com_times = np.unique(np.concatenate([all_times[m] for m in modes]))
        if mode is None:
            self.com_times = com_times.copy()

        # --- Precompute shared arrays once ---
        incl = np.deg2rad(p['incl'])
        plats_f, areas_f, lats_f, longs_f = self._ensure_f64(
            plats, sg['grid_areas'], sg['grid_lats'], sg['grid_longs']
        )

        if info:
            print('\033[92mCalculating the coefficients related to the surface elements...\033[0m')

        cpcs = {}
        for m in modes:
            if info:
                label = 'Atomic line profile(s)' if m == 'line' else 'Light curve profile'
                print(f'\033[1m{label}...\033[0m')

            rgi = self.ldi[m]
            ldcs_phot_f, ldcs_cool_f, ldcs_hot_f = self._ensure_f64(
                np.array(rgi['phot'][:2]).T,
                np.array(rgi['cool'][:2]).T,
                np.array(rgi['hot'][:2]).T,
            )
            lis_phot_f, lis_cool_f, lis_hot_f = self._ensure_f64(
                rgi['phot'][-1], rgi['cool'][-1], rgi['hot'][-1]
            )

            times = all_times[m]

            # --- Shared arrays: passed once via Pool initializer ---
            shared = {
                'plats': plats_f, 'ldcs_phot': ldcs_phot_f,
                'ldcs_cool': ldcs_cool_f, 'ldcs_hot': ldcs_hot_f,
                'lis_phot': lis_phot_f, 'lis_cool': lis_cool_f,
                'lis_hot': lis_hot_f, 'areas': areas_f,
                'lats': lats_f, 'longs': longs_f,
                't0': p['t0'], 'incl': incl,
                'ld_law': self.ldi['ld_params']['law'],
                'noes': sg['noes'], 'period': p['period'],
            }

            if m != 'lc':
                vlats_f, lp_vels_f, phot_lp_f, cool_lp_f, hot_lp_f, vels_f = self._ensure_f64(
                    vlats,
                    self.idc[m]['lp_vels'], self.idc[m]['phot_lp_data'],
                    self.idc[m]['cool_lp_data'], self.idc[m]['hot_lp_data'],
                    self.idc[m]['vels'],
                )
                shared.update({
                    'vlats': vlats_f, 'nop': self.idc[m]['nop'],
                    'lp_vels': lp_vels_f, 'phot_lp': phot_lp_f,
                    'cool_lp': cool_lp_f, 'hot_lp': hot_lp_f,
                    'vrt': p['vrt'], 'vels': vels_f,
                })

            # --- Per-timestep args: only the varying parts ---
            per_ts = [(t, i, info) for i, t in enumerate(times)]

            results = dipu.mp_calc_pixel_coeffs(
                cpu_num=self.cpu_num, shared_arrays=shared,
                per_timestep_args=per_ts, mode=m,
            )

            results_arr = np.array(results)
            cpcs[m] = {
                'coeffs_cube': results_arr,
                'times': {t: results_arr[i] for i, t in enumerate(times)},
            }

        return cpcs

    def generate_synthetic_profile(self, fssc, fssh, coeffs_cube, rv=None, vels=None, amp=None):

        fssp = 1.0 - (fssc + fssh)

        profiles = coeffs_cube[:, 3:]
        nop = profiles.shape[1] // 3
        p_phot, p_cool, p_hot = profiles[:, :nop], profiles[:, nop:2 * nop], profiles[:, 2 * nop:]

        wgt_phot = coeffs_cube[:, 0] * fssp
        wgt_cool = coeffs_cube[:, 1] * fssc
        wgt_hot = coeffs_cube[:, 2] * fssh

        prf = jnp.sum(wgt_phot[:, None] * p_phot +
                      wgt_cool[:, None] * p_cool +
                      wgt_hot[:, None] * p_hot, axis=0)
        prf /= jnp.sum(wgt_phot + wgt_cool + wgt_hot)

        if rv is not None:
            prf = jnp.interp(vels, vels + rv, prf)

        if amp is not None:
            prf = prf + amp

        prf /= jnp.mean(prf)

        return prf

    def generate_synthetic_lightcurve(self, fssc, fssh, coeffs_cube, amp=None):

        fssp = 1.0 - (fssc + fssh)

        flux = jnp.sum(coeffs_cube[:, 0] * fssp +
                       coeffs_cube[:, 1] * fssc +
                       coeffs_cube[:, 2] * fssh)

        if amp is not None:
            flux += amp

        return flux

    def entropy(self, fssc, fssh, fssp, wp, fsl):

        return jnp.sum(
            wp * (
                    fssc * jnp.log(fssc / fsl) +
                    fssh * jnp.log(fssh / fsl) +
                    (1.0 - fssp) * jnp.log((1.0 - fssp) / fsl)
            )
        )

    def _unpack_surface(self, x0s, noes):
        fssc = x0s[:noes]
        fssh = x0s[noes:2 * noes]
        fssp = 1.0 - (fssc + fssh)
        return fssp, fssc, fssh

    def _chi2_rms(self, data, model):
        res = data[0] - model
        chi2 = jnp.mean((res / data[1]) ** 2)
        rms = jnp.sqrt(jnp.mean(res ** 2))
        return chi2, rms

    def _objective(self, x0s, *, data_line=None, coeffs_line=None, alpha=1.0, data_lc=None, coeffs_lc=None, delta=1.0,
                   rv_line=None, vels_line=None, amp_line=None, amp_lc=None, lmbd=None, rv_line_mask=None, amp_line_mask=None,
                   amp_lc_mask=None, noes=None, norm_grid_areas=None, fsl=None, disp=None):

        fssp, fssc, fssh = self._unpack_surface(x0s, noes)

        chi2_line = rms_line = chi2_lc = rms_lc= 0.0

        if data_line is not None:
            if rv_line is None and rv_line_mask is not None:
                rv_line = x0s[rv_line_mask]

            if amp_line is None and amp_line_mask is not None:
                amp_line = x0s[amp_line_mask]

            model_line = self.vmap_line(fssc, fssh, coeffs_line, rv_line, vels_line, amp_line)
            chi2_line, rms_line = self._chi2_rms(data_line, model_line)

        if data_lc is not None:
            if amp_lc is None and amp_lc_mask is not None:
                amp_lc = x0s[amp_lc_mask][0]

            model_lc = self.vmap_lc(fssc, fssh, coeffs_lc, amp_lc)
            model_lc = model_lc / jnp.mean(model_lc)
            chi2_lc, rms_lc = self._chi2_rms(data_lc, model_lc)

        weighted_chi2 = alpha * chi2_line + delta * chi2_lc
        entropy = self.entropy(fssc, fssh, fssp, norm_grid_areas, fsl)
        lmbd_entropy = lmbd * entropy
        cost = weighted_chi2 + lmbd_entropy

        jax_debug.callback(
            self._print_callback, disp, cost,
            chi2_line, chi2_lc,
            alpha * chi2_line, delta * chi2_lc,
            weighted_chi2, entropy, lmbd_entropy,
            rms_line, rms_lc, rms_line + rms_lc
        )
        return cost

    def _resolve_corr_param(self, mode, parn, noo, x0s_len, offset):
        corr = self.conf[mode]['corr'][parn]

        if corr == 'free':
            mask = np.zeros(x0s_len, dtype=bool)
            mask[offset:offset + noo] = True
            self.conf[mode]['mask'][parn] = mask
            return None, mask, 0, offset + noo

        if corr is None:
            self.conf[mode]['corr'][parn] = np.full((noo, 1), None, dtype=object)
            return None, None, None, offset

        corr = np.atleast_1d(np.asarray(corr))
        if corr.size == 1:
            corr = np.full((noo, 1), corr.item())
        elif corr.ndim == 1:
            corr = corr[:, None]

        self.conf[mode]['corr'][parn] = corr
        return corr, None, 0, offset

    def _obj(self, noes, nofp, lmbd, cpcs, alpha, delta, disp):

        norm_grid_areas = jnp.array(self.surface_grid['norm_grid_areas'])
        x0s_len = 2 * noes + nofp

        has_line = 'line' in self.conf
        has_lc = 'lc' in self.conf

        kwargs = {'lmbd': jnp.float64(lmbd)}
        offset = 2 * noes

        rv_line_mask = amp_line_mask = amp_lc_mask = None

        if has_line:
            noo = self.idc['line']['noo']
            kwargs['data_line'] = jnp.array(self.idc['line']['data_cube'])
            kwargs['coeffs_line'] = jnp.array(cpcs['line']['coeffs_cube'])
            kwargs['alpha'] = jnp.float64(alpha if has_lc else 1.0)

            rv_val, rv_line_mask, rv_axis, offset = self._resolve_corr_param('line', 'rv', noo,
                                                                             x0s_len, offset)
            amp_val, amp_line_mask, amp_axis, offset = self._resolve_corr_param('line', 'amp', noo,
                                                                                x0s_len, offset)

            if rv_val is not None:
                kwargs['rv_line'] = rv_val
            if amp_val is not None:
                kwargs['amp_line'] = amp_val
            if rv_val is not None or rv_line_mask is not None:
                kwargs['vels_line'] = self.idc['line']['vels']

            self.vmap_line = jax.vmap(self.generate_synthetic_profile, in_axes=[None, None, 0, rv_axis, None,
                                                                                amp_axis])

        if has_lc:
            noo = self.idc['lc']['noo']
            kwargs['data_lc'] = jnp.array(self.idc['lc']['data_cube'])
            kwargs['coeffs_lc'] = jnp.array(cpcs['lc']['coeffs_cube'])
            kwargs['delta'] = jnp.float64(delta if has_line else 1.0)

            amp_val, amp_lc_mask, _, offset = self._resolve_corr_param('lc', 'amp', noo, x0s_len, offset)

            if amp_val is not None:
                kwargs['amp_lc'] = amp_val[0][0]

            self.vmap_lc = jax.vmap(self.generate_synthetic_lightcurve, in_axes=[None, None, 0, None])

        obj = jax.jit(partial(
            self._objective, noes=noes,
            rv_line_mask=rv_line_mask, amp_line_mask=amp_line_mask, amp_lc_mask=amp_lc_mask,
            norm_grid_areas=norm_grid_areas, fsl=self.fsl, disp=disp
        ))

        return obj, kwargs

    def minimize(self, maxiter, tol, alpha, delta, lmbd, cpcs, initial_map=None, disp=True):
        noes = self.surface_grid['noes']
        nofp = self.idc['nofp']

        if initial_map is None:
            fss = np.full(2 * noes, self.fsl)
        else:
            fss = initial_map.copy()

        x0s = jnp.hstack((fss, jnp.zeros(nofp)))

        iter_num = num_fun_eval = 0

        if maxiter > 0:
            self._iter_count = 0

            import time as time_calc
            start_time = time_calc.time()

            minx0s = jnp.array([self.fsl] * (2 * noes) + [-jnp.inf] * nofp)
            maxx0s = jnp.array([self.fsu] * (2 * noes) + [jnp.inf] * nofp)
            bounds = jnp.array((minx0s, maxx0s))

            if disp:
                print(f"\n{'=' * 160}\n{'nFree'} ==> {len(x0s)}\n{'=' * 160}")

            obj, kwargs = self._obj(noes, nofp, lmbd, cpcs, alpha, delta, disp)

            optimizer = ScipyBoundedMinimize(fun=obj, has_aux=False, maxiter=maxiter, tol=tol, method='L-BFGS-B')
            x0s, res = optimizer.run(x0s, bounds, **kwargs)

            iter_num, num_fun_eval = res.iter_num, res.num_fun_eval

            print(f"\n{'=' * 170}")
            print(f"Iterations: {iter_num}, Function evaluations: {num_fun_eval}")
            print(f"Time taken for optimization: {time_calc.time() - start_time:.2f}s")
            print(f"{'=' * 170}")

        for mode in self.conf:
            for parn in ['rv', 'amp']:
                corr = self.conf[mode]['corr'][parn]
                if isinstance(corr, str) and corr == 'free':
                    self.opt_results[mode][parn] = np.asarray(x0s[self.conf[mode]['mask'][parn]]).flatten()
                elif corr is None:
                    self.opt_results[mode][parn] = np.array([None] * self.idc[mode]['noo'])
                else:
                    self.opt_results[mode][parn] = np.asarray(corr).flatten()

        fssp, fssc, fssh = self._unpack_surface(x0s, noes)

        return np.array(fssp), np.array(fssc), np.array(fssh), iter_num, num_fun_eval

    def _print_callback(self, disp, loss, chi_line, chi_lc, a_line, d_lc, tw_chi, entropy, lmbd_entropy, rms_line, rms_lc,
                        rms_tot):
        if disp:
            self._iter_count += 1
            print("-" * 170)
            print(f"{'Func. eval.':^11} | {'loss':^11} | "
                  f"{'χ²_line':^11} | {'χ²_lc':^11} | {'α·χ²_line':^11} | {'δ·χ²_lc':^11} | "
                  f"{'Σw·χ²':^11} | {'Entropy':^11} | {'λ·Entropy':^11} | "
                  f"{'rms_line_res':^11} | {'rms_lc_res':^11} | {'rms_tot_res':^11}")
            print("-" * 170)
            print(f"{self._iter_count:^11d} | {loss:^11.6f} | "
                  f"{chi_line:^11.6f} | {chi_lc:^11.6f} | {a_line:^11.6f} | {d_lc:^11.6f} | "
                  f"{tw_chi:^11.6f} | {entropy:^11.6f} | {lmbd_entropy:^11.6f} | "
                  f"{rms_line:^11.6f} | {rms_lc:^11.6f} | {rms_tot:^11.6f}")

        self.opt_stats['Chi-square for Line Profile(s)'] = chi_line
        self.opt_stats['Chi-square for Light Curve Profile'] = chi_lc
        self.opt_stats['Alpha * Line Profile(s) Chi-square'] = a_line
        self.opt_stats['Delta * Light Curve Profile Chi-square'] = d_lc
        self.opt_stats['Total Weighted Chi-square'] = tw_chi
        self.opt_stats['Entropy'] = entropy
        self.opt_stats['Lambda * Entropy'] = lmbd_entropy
        self.opt_stats['RMS of Line Residauls'] = rms_line
        self.opt_stats['RMS of Light Curve Residauls'] = rms_lc
        self.opt_stats['Total RMS of Residauls'] = rms_tot
        self.opt_stats['Loss Function Value'] = loss

    def reconstructor(self, alpha=1.0, delta=1.0, lmbd=1.0, maxiter=100, tol=1e-10, cpcs=None, initial_map=None,
                      disp=True):

        if not cpcs:
            plps = (True, True)
            if len(self.conf) == 1 and "lc" in self.conf:
                plps = (False, False)

            import time as time_calc
            itime_1 = time_calc.time()
            cpcs = self.calc_pixel_coeffs(line_times=self.idc['line']['times'], lc_times=self.idc['lc']['times'],
                                          info=True, plps=plps)
            itime_2 = time_calc.time()
            print(f"Time taken to calculate the surface coefficients: {itime_2 - itime_1} seconds")

        if 'lc' not in self.conf:
            tmin, tmax = [], []
            for mode in self.conf:
                tmin.append(min(self.idc[mode]['times']))
                tmax.append(max(self.idc[mode]['times']))

            self.opt_results['lc']['ntimes'] = self.params['t0'] + self.params['period'] * np.linspace(0, 2, 200)
            cpcs_lc = self.calc_pixel_coeffs(mode='lc', lc_times=self.opt_results['lc']['ntimes'], info=False,
                                             plps=(False, False), plil=(False, False, False))

        if isinstance(lmbd, (list, np.ndarray)):
            lmbd = self.lambda_search(alpha=alpha, delta=delta, lmbds=lmbd, maxiter=maxiter, tol=tol, cpcs=cpcs)

        rfssp, rfssc, rfssh, nit, nfev = self.minimize(maxiter=maxiter, tol=tol, alpha=alpha, delta=delta,
                                                       lmbd=lmbd, cpcs=cpcs, initial_map=initial_map, disp=disp)

        print()
        print('\033[96m' + '*** Optimization Results ***' + '\033[0m')
        for ositem in self.opt_stats:
            print('\033[93m' + '{:<45}'.format(ositem) + ':' + '\033[0m',
                  '\033[1m' + str(self.opt_stats[ositem]) + '\033[0m')

        intensities = ((rfssc * self.params['Tcool'] ** 4 + rfssh * self.params['Thot'] ** 4 +
                        rfssp * self.params['Tphot'] ** 4) / self.params['Tphot'] ** 4)

        recons_slc = []
        for mode in self.conf:
            coeffs_cube = cpcs[mode]['coeffs_cube']

            if mode == 'line':
                rv = self.opt_results[mode]['rv'].flatten()
                amp = self.opt_results[mode]['amp'].flatten()
                vels = self.idc['line']['vels']

                spotless_sprfs = {}
                recons_sprfs = {}
                for i, itime in enumerate(self.idc[mode]['times']):
                    spotless_sprfs[itime] = {}
                    recons_sprfs[itime] = {}

                    spotless_sprf = self.generate_synthetic_profile(fssc=0.0, fssh=0.0, rv=rv[i], vels=vels, amp=amp[i],
                                                                       coeffs_cube=coeffs_cube[i])

                    recons_sprf = self.generate_synthetic_profile(fssc=rfssc, fssh=rfssh, rv=rv[i], vels=vels, amp=amp[i],
                                                                     coeffs_cube=coeffs_cube[i])

                    spotless_sprfs[itime]['prf'] = np.array(spotless_sprf)
                    recons_sprfs[itime]['prf'] = np.array(recons_sprf)

                self.opt_results[mode]['spotless_sprfs'] = spotless_sprfs
                self.opt_results[mode]['recons_sprfs'] = recons_sprfs

            else:
                lc_amp = self.opt_results[mode]['amp'][0]
                for i, itime in enumerate(self.idc[mode]['times']):
                    flux = self.generate_synthetic_lightcurve(fssc=rfssc, fssh=rfssh, coeffs_cube=coeffs_cube[i],
                                                              amp=lc_amp)
                    recons_slc.append(flux)
                recons_slc = np.array(recons_slc) / np.mean(recons_slc)

        if 'lc' not in self.conf:
            for i, itime in enumerate(self.opt_results['lc']['ntimes']):
                flux = self.generate_synthetic_lightcurve(fssc=rfssc, fssh=rfssh,
                                                          coeffs_cube=cpcs_lc['lc']['coeffs_cube'][i])
                recons_slc.append(flux)
            recons_slc = np.array(recons_slc) / np.mean(recons_slc)

        self.opt_results['lc']['recons_slc'] = recons_slc

        spotted_area = dipu.get_total_fs(fssc=rfssc, fssh=rfssh, areas=self.surface_grid['grid_areas'],
                                         lats=self.surface_grid['grid_lats'], incl=self.params['incl'])

        self.opt_results['alpha'] = alpha
        self.opt_results['delta'] = delta
        self.opt_results['lmbd'] = lmbd
        self.opt_results['recons_fssc'] = rfssc
        self.opt_results['recons_fssh'] = rfssh
        self.opt_results['recons_fssp'] = rfssp
        self.opt_results['ints'] = intensities
        self.opt_results['nit'] = nit
        self.opt_results['nfev'] = nfev
        self.opt_results['total_cool_spotted_area'] = spotted_area[0]
        self.opt_results['partial_cool_spotted_area'] = spotted_area[3]
        self.opt_results['total_hot_spotted_area'] = spotted_area[1]
        self.opt_results['partial_hot_spotted_area'] = spotted_area[4]
        self.opt_results['total_unspotted_area'] = spotted_area[2]
        self.opt_results['partial_unspotted_area'] = spotted_area[5]

        self.mapprojs = dipu.grid_to_rect_map(surface_grid=self.surface_grid, ints=self.opt_results['ints'],
                                              fssc=rfssc, fssp=rfssp, fssh=rfssh)

    def lambda_search(self, alpha, delta, lmbds, maxiter, tol, cpcs):

        import tqdm

        noes = self.surface_grid['noes']
        nofp = self.idc['nofp']
        spotless_fss = np.full(2 * noes, self.fsl)

        minx0s = jnp.array([self.fsl] * (2 * noes) + [-jnp.inf] * nofp)
        maxx0s = jnp.array([self.fsu] * (2 * noes) + [jnp.inf] * nofp)

        obj, kwargs = self._obj(noes, nofp, 1.0, cpcs, alpha, delta, False)
        optimizer = ScipyBoundedMinimize(fun=obj, has_aux=False, maxiter=maxiter, tol=tol, method='L-BFGS-B')

        results = []
        for lmbd in tqdm.tqdm(lmbds):
            x0s = jnp.hstack((spotless_fss, jnp.zeros(nofp)))
            bounds = jnp.array((minx0s, maxx0s))
            kwargs['lmbd'] = jnp.float64(lmbd)
            _, _ = optimizer.run(x0s, bounds, **kwargs)
            results.append((self.opt_stats['Total Weighted Chi-square'],
                             self.opt_stats['Entropy'], lmbd))

        parts = np.array(results)

        sort = np.argsort(parts[:, 2])
        lmbds = parts[:, 2][sort]
        twchisqs = parts[:, 0][sort]
        entropies = parts[:, 1][sort]

        from kneebow.rotor import Rotor

        rotor = Rotor()
        rotor.fit_rotate(np.vstack((twchisqs, entropies)).T)
        maxcurve = rotor.get_elbow_index()

        self.opt_results['lmbds'] = lmbds
        self.opt_results['twchisqs'] = twchisqs
        self.opt_results['entropies'] = entropies
        self.opt_results['maxcurve'] = maxcurve

        return lmbds[maxcurve]

    # --- Helper: map param name to setter ---
    def _set_param(self, name, value):
        """Route a grid-search parameter to the correct config location."""
        dispatch = {
            'eqw': lambda v: self.conf['line'].__setitem__('eqw', v),
            'eqw_phot': lambda v: self.conf['line']['eqw'].__setitem__('phot', v),
            'eqw_cool': lambda v: self.conf['line']['eqw'].__setitem__('cool', v),
            'eqw_hot': lambda v: self.conf['line']['eqw'].__setitem__('hot', v),
            'rv_line': lambda v: self.conf['line']['corr'].__setitem__(
                'rv', np.full((self.idc['line']['noo'], 1), v)),
            'amp_line': lambda v: self.conf['line']['corr'].__setitem__(
                'amp', np.full((self.idc['line']['noo'], 1), v)),
            'amp_lc': lambda v: self.conf['lc']['corr'].__setitem__(
                'amp', np.full((self.idc['lc']['noo'], 1), v)),
        }
        if name in dispatch:
            dispatch[name](value)
        else:
            self.params[name] = value

    def grid_search_run(self, fpmg):

        for mode in self.conf:
            self.conf[mode]['corr'] = copy.deepcopy(self._orig_corr[mode])

        for name, value in zip(self.gs_keys, fpmg):
            self._set_param(name, value)

        self.construct_surface_grid(
            method=self.surface_grid['method'], noes=self.surface_grid['init_noes'],
            nlats=self.surface_grid['nlats'], nside=self.surface_grid['nside'], info=False,
        )

        cpcs = self.calc_pixel_coeffs(
            line_times=self.idc['line']['times'], lc_times=self.idc['lc']['times'],
            plps=(True, False), plil=(True, False, True), info=False,
        )

        self.minimize(
            maxiter=self.optp_gs['maxiter'], tol=self.optp_gs['tol'],
            alpha=self.optp_gs['alpha'], delta=self.optp_gs['delta'],
            lmbd=self.optp_gs['lmbd'], cpcs=cpcs, disp=False,
        )

        if self.info_gs:
            output = self.params.copy()
            info_map = {
                'eqw': ('line', 'eqw', None),
                'rv_line': ('line', 'corr', 'rv'),
                'amp_line': ('line', 'corr', 'amp'),
                'amp_lc': ('lc', 'corr', 'amp'),
            }
            for key, (mode, k1, k2) in info_map.items():
                if key in self.gs_all_keys:
                    output[key] = self.conf[mode][k1] if k2 is None else self.conf[mode][k1][k2][0]

            output['Total Weighted Chi-square'] = self.opt_stats['Total Weighted Chi-square']
            output['Loss Function Value'] = self.opt_stats['Loss Function Value']

            print('\n\033[96m*** Optimization Results ***\033[0m')
            for k, v in output.items():
                print(f'\033[93m{k:<11}:\033[0m \033[1m{v}\033[0m')

        return self.opt_stats['Total Weighted Chi-square'], self.opt_stats['Loss Function Value']

    def grid_search(self, fit_params, opt_params=None, info=True, minv="chi", save=False):
        grid_cpu_num = self.cpu_num
        self.cpu_num = 1
        self.minv_gs = minv

        # --- Optimization defaults ---
        optp = {'alpha': 1.0, 'delta': 1.0, 'lmbd': 1.0, 'maxiter': 100, 'tol': 1e-5, 'disp': True}
        if opt_params is not None:
            invalid = set(opt_params) - set(optp)
            if invalid:
                print(f'\033[93mWarning: invalid opt_params keys: {invalid}\033[0m')
            optp.update({k: v for k, v in opt_params.items() if k in optp})

        if isinstance(optp['lmbd'], (list, np.ndarray)):
            optp['lmbd'] = np.min(optp['lmbd'])

        self.optp_gs = optp
        self.gs_all_keys = list(fit_params.keys())

        # --- Separate single-value params (fixed) from multi-value (grid) ---
        grid_params = {}
        for name, values in fit_params.items():
            if len(values) == 1:
                self._set_param(name, values[0])
            else:
                grid_params[name] = values

        # --- Build meshgrid ---
        self.gs_keys = list(grid_params.keys())
        fit_params_mg = np.array(
            np.meshgrid(*grid_params.values())
        ).T.reshape(-1, len(grid_params))

        self.info_gs = info

        self._orig_corr = {mode: copy.deepcopy(self.conf[mode]['corr']) for mode in self.conf}

        stats = dipu.mp_search(cpu_num=grid_cpu_num, func=self.grid_search_run, input_args=fit_params_mg,
                               backend=self.platform_name)

        # --- Collect results ---
        stats_grid = {name: fit_params_mg[:, i] for i, name in enumerate(self.gs_keys)}
        stats_grid['stats'] = np.array(stats)
        stats_grid['fit_params'] = fit_params

        self.cpu_num = grid_cpu_num

        if save:
            with open(save, 'wb') as f:
                pickle.dump(stats_grid, f)

        dipu.make_grid_contours(stats_grid, minv)

    def plot(self, plot_params=None):

        plotp = {'line_sep_prf': 0.4, 'line_sep_res': 0.01, 'mol_sep_prf': 0.4, 'mol_sep_res': 0.01,
                 'show_err_bars': True, 'fmt': '%0.3f', 'markersize': 2, 'linewidth': 1, 'fontsize': 15,
                 'ticklabelsize': 12}
        if plot_params is not None:
            for pitem in plot_params:
                if pitem in plotp:
                    plotp[pitem] = plot_params[pitem]
                else:
                    print('\033[93m' + 'Warning: ' + pitem + ' is not a valid parameter for plot_params!' + '\033[0m')

        msg = '\n' + '\033[94m' + 'Preparing the GUI for display. Please wait...' + '\033[0m'
        sys.stdout.write('\r' + msg)

        app = QtWidgets.QApplication(sys.argv)
        pg = PlotGUI(self, plot_params=plotp)
        pg.show()

        msg = '\033[94m' + "The GUI is currently being displayed..." + '\033[0m' + '\n'
        sys.stdout.write('\r' + msg)

        sys.exit(app.exec_())

    def test(self, modes_input, artificial_map=None, spots_params=None, opt_params=None, plot_params=None,
             seed=None, initial_map=False, save_data_path=None, plot=True):


        plotp = {'map_projection': 'rectangular', 'line_sep_prf': 0.03, 'line_sep_res': 0.01, 'mol_sep_prf': 0.03,
                 'mol_sep_res': 0.01, 'show_err_bars': True, 'fmt': '%0.3f', 'markersize': 2, 'linewidth': 1,
                 'fontsize': 15, 'ticklabelsize': 12, 'fill_color': 'yellow', 'cmap': 'gray'}
        if plot_params is not None:
            for pitem in plot_params:
                if pitem in plotp:
                    plotp[pitem] = plot_params[pitem]
                else:
                    print('\033[93m' + 'Warning: ' + pitem + ' is not a valid parameter for plot_params!' + '\033[0m')

        optp = {'alpha': 1.0, 'delta': 1.0, 'lmbd': 1.0, 'maxiter': 100, 'tol': 1e-10, 'disp': True}
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

        if artificial_map is None and spots_params is not None:
            fake_fssc, fake_fssh = dipu.generate_spotted_surface(surface_grid=self.surface_grid,
                                                                 spots_params=spots_params)
        if artificial_map is not None and spots_params is None:
            noes = self.surface_grid['noes']
            fake_fssc, fake_fssh = artificial_map[:noes], artificial_map[noes:2 * noes]

        import time as time_calc
        itime_1 = time_calc.time()
        cpcs = self.calc_pixel_coeffs(line_times=self.idc['line']['times'], lc_times=self.idc['lc']['times'], info=True)
        itime_2 = time_calc.time()
        print(f"Time taken to calculate the surface coefficients: {itime_2 - itime_1} seconds")

        np.random.seed(seed)
        for mode, mitem in self.conf.items():
            if isinstance(mitem['corr']['rv'], (tuple, list, np.ndarray)):
                rvs = mitem['corr']['rv']

            elif mitem['corr']['rv'] in ['free', None]:
                rvs = [None] * self.idc[mode]['noo']

            else:
                rvs = [mitem['corr']['rv']] * self.idc[mode]['noo']

            if isinstance(mitem['corr']['amp'], (list, np.ndarray)):
                amps = mitem['corr']['amp']

            elif mitem['corr']['amp'] in ['free', None]:
                amps = [None] * self.idc[mode]['noo']

            else:
                amps = [mitem['corr']['amp']] * self.idc[mode]['noo']

            if mode == 'line':
                fake_sprfs = {}
                vels = self.idc['line']['vels']
                for i, itime in enumerate(self.idc[mode]['times']):
                    sprf = self.generate_synthetic_profile(fssc=fake_fssc, fssh=fake_fssh, rv=rvs[i], vels=vels,
                                                              amp=amps[i], coeffs_cube=cpcs[mode]['coeffs_cube'][i])

                    fprf, fperr = dipu.generate_noisy_normalized_profile(model=sprf, snr=self.idc[mode]['snr'],
                                                                         read_noise=self.idc[mode]['read_noise'])

                    self.idc[mode]['data'].append(np.vstack((self.idc[mode]['vels'], fprf, fperr)).T)

                    fake_sprfs[itime] = {}
                    fake_sprfs[itime]['prf'] = np.array(sprf)

                self.opt_results[mode]['fake_sprfs'] = fake_sprfs

            else:
                slc = np.zeros(self.idc[mode]['nop'])
                for i, itime in enumerate(self.idc[mode]['times']):
                    flux = self.generate_synthetic_lightcurve(fssc=fake_fssc, fssh=fake_fssh,
                                                              coeffs_cube=cpcs[mode]['coeffs_cube'][i], amp=amps[0])

                    slc[i] = flux

                slc /= np.mean(slc)
                flc, flerr = dipu.generate_noisy_normalized_profile(model=slc, snr=self.idc[mode]['snr'],
                                                                    read_noise=self.idc[mode]['read_noise'])

                self.idc[mode]['data'] = np.vstack((flc, flerr)).T

        if save_data_path is not None:
            file = open(save_data_path, 'wb')
            pickle.dump(self.idc, file)
            file.close()

        self.set_input_data()

        if initial_map:
            initial_map = np.hstack((fake_fssc, fake_fssh))
        else:
            initial_map = None

        self.reconstructor(alpha=optp['alpha'], delta=optp['delta'], lmbd=optp['lmbd'], maxiter=optp['maxiter'],
                           tol=optp['tol'], disp=optp['disp'], cpcs=cpcs, initial_map=initial_map)

        fake_total_fs = dipu.get_total_fs(fssc=fake_fssc, fssh=fake_fssh, areas=self.surface_grid['grid_areas'],
                                          lats=self.surface_grid['grid_lats'], incl=self.params['incl'])
        recons_total_fs = dipu.get_total_fs(fssc=self.opt_results['recons_fssc'], fssh=self.opt_results['recons_fssh'],
                                            areas=self.surface_grid['grid_areas'], lats=self.surface_grid['grid_lats'],
                                            incl=self.params['incl'])

        if plot:
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
        recons_fssp = self.opt_results['recons_fssp']

        ints = self.opt_results['ints'].copy()

        map_proj_data_dict = {'grid latitudes': self.surface_grid['grid_lats'],
                              'grid longitudes': self.surface_grid['grid_longs'],
                              'photosphere filling factors': recons_fssp, 'cool spot filling factors': recons_fssc,
                              'hot spot filling factors': recons_fssh, 'normalized intensities': ints,
                              '2D photosphere filling factor map': self.mapprojs['rmap_p'].copy(),
                              '2D cool spot filling factor map': self.mapprojs['rmap_c'].copy(),
                              '2D hot spot filling factor map': self.mapprojs['rmap_h'].copy(),
                              '2D normalized intensity map': self.mapprojs['rmap_int'].copy(),
                              'meshgrid latitudes': self.mapprojs['mlats'].copy(),
                              'meshgrid longitudes': self.mapprojs['mlongs'].copy()}

        file = open(path, 'wb')
        pickle.dump(map_proj_data_dict, file)
        file.close()
