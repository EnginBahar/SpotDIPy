import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from jax import vmap, config as jax_config, numpy as jnp
from jaxopt import ScipyBoundedMinimize
from astropy import units as au, constants as ac
import platform
import copy
from PyQt5 import QtWidgets
from . import utils as dipu
# import utils as dipu
from .plot_GUI import PlotGUI
# from plot_GUI import PlotGUI
from jax.scipy.stats import norm


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
            observational data, either 'cpu' or 'gpu'. The default is 'cpu'. This setting does not affect other
            parallelization tasks.
            Note: If 'gpu' is selected and the code is executed multiple times simultaneously, there is a possibility
            of running out of GPU memory, which may cause the code to stop execution. This is particularly relevant when
            using the grid_search function.
        info : bool, optional
            Whether to display information messages during initialization. Can be True or False. The default is True.

        Raises:
        -------
        ValueError
            If cpu_num is not an integer.
        KeyError
            If platform_name is not 'cpu' or 'gpu'.
        KeyError
            If info is not True or False.
        """

        if not isinstance(cpu_num, (int, float, np.integer, np.floating)):
            raise ValueError("'cpu_num' keyword must be an integer number")
        self.cpu_num = int(cpu_num)

        if platform_name not in ['cpu', 'gpu']:
            raise KeyError("'platform_name' keyword must be one of two options: 'cpu' or 'gpu'")

        if info not in [True, False]:
            raise KeyError("'info' keyword must be one of two options: True or False")

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
            'resolution': None
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
                'snr': None
            }
        }

        self.opt_stats = {
            'Chi-square for Line Profile(s)': None,
            'Chi-square for Molecular(1) Profile(s)': None,
            'Chi-square for Molecular(2) Profile(s)': None,
            'Chi-square for Light Curve Profile': None,
            'Alpha * Line Profile(s) Chi-square': None,
            'Beta * Molecular(1) Profile(s) Chi-square': None,
            'Gamma * Molecular(2) Profile(s) Chi-square': None,
            'Delta * Light Curve Profile Chi-square': None,
            'Total Weighted Chi-square': None,
            'Total Entropy': None,
            'Lambda * Total Entropy': None,
            'p-value': None,
            'Loss Function Value': None
        }

        self.opt_results = {'line': {}, 'mol1': {}, 'mol2': {}, 'lc': {}, 'lmbds': []}
        self.ldi = {'ld_params': {}, 'line': {}, 'mol1': {}, 'mol2': {}, 'lc': {}}
        self.surface_grid = {}
        self.com_times = []

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
            - 'R': Stellar radius (in solar radius).
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
            radius = dipu.calc_radius(vsini=self.params['vsini'], incl=self.params['incl'], period=self.params['period'])
        else:
            radius = self.params['R']

        methods = ['trapezoid', 'phoebe2_marching', 'healpy']
        if method not in methods:
            raise KeyError(f"'{method}' argument must be one of three options: 'trapezoid', 'phoebe2_marching'"
                                                " or 'healpy'")

        try:
            omega, requiv, rp = dipu.calc_omega_and_requiv(mass=self.params['mass'], period=self.params['period'],
                                                           re=radius)
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
            dipu.td_surface_grid(container=self.surface_grid, omega=omega, nlats=nlats, radius=radius,
                                 mass=self.params['mass'], cpu_num=self.cpu_num)

        elif method == 'healpy':
            if platform.system() == "Windows":
                raise OSError("'healpy' is not supported on Windows. Please select 'trapezoid' or 'phoebe2_marching'")
            else:
                dipu.hp_surface_grid(container=self.surface_grid, omega=omega, nside=nside, radius=radius,
                                     mass=self.params['mass'], cpu_num=self.cpu_num)

        self.surface_grid['gds'] = dipu.calc_gds(omega=omega, thetas=self.surface_grid['grid_lats'] + np.pi / 2.0)
        self.surface_grid['noes'] = len(self.surface_grid['grid_lats'])
        self.surface_grid['coslats'] = np.cos(self.surface_grid['grid_lats'])
        self.surface_grid['sinlats'] = np.sin(self.surface_grid['grid_lats'])
        self.surface_grid['cosi'] = np.cos(np.deg2rad(self.params['incl']))
        self.surface_grid['sini'] = np.sin(np.deg2rad(self.params['incl']))

        if self.params['R'] is None:
            veq = self.params['vsini'] / self.surface_grid['sini']
        else:
            veq = (2.0 * np.pi * (radius * au.solRad.to(au.km))) / (self.params['period'] * au.day.to(au.second))
            self.params['vsini'] = veq * self.surface_grid['sini']

        if info:
            print('\033[96mNumber of total surface element:\033[0m', self.surface_grid['noes'])
            if self.params['R'] is None:
                print('\033[96mEquatorial radius:\033[0m', np.round(radius, 3), 'SolRad')
            else:
                print('\033[96mProjected equatorial rotational velocity:\033[0m',
                      np.round(self.params['vsini'], 3), 'km/s')
            print('\033[96mEquatorial rotational velocity:\033[0m', np.round(veq, 2), 'km/s')

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

        teffcs = self.surface_grid['ctcs'].copy()
        teffhs = self.surface_grid['cths'].copy()
        teffps = self.surface_grid['ctps'].copy()
        loggs = self.surface_grid['grid_loggs'].copy()

        for mode in self.conf:
            if mode != 'lc':
                ld_rgi = dipu.ldcs_rgi_prep(teffps=teffps, teffcs=teffcs, teffhs=teffhs, loggs=loggs,
                                            law=self.ldi['ld_params']['law'], mh=self.ldi['ld_params']['mh'],
                                            wrange=self.conf[mode]['wave_range'], model=self.ldi['ld_params']['model'],
                                            data_path=self.ldi['ld_params']['data_path'],
                                            mu_min=self.ldi['ld_params']['mu_min'], cpu_num=self.cpu_num)

                self.ldi[mode] = ld_rgi

            else:
                ld_rgi = dipu.ldcs_rgi_prep(teffps=teffps, teffcs=teffcs, teffhs=teffhs, loggs=loggs,
                                            law=self.ldi['ld_params']['law'],  mh=self.ldi['ld_params']['mh'],
                                            passband=self.conf[mode]['passband'], model=self.ldi['ld_params']['model'],
                                            data_path=self.ldi['ld_params']['data_path'],
                                            mu_min=self.ldi['ld_params']['mu_min'],  cpu_num=self.cpu_num)

                self.ldi[mode] = ld_rgi

        if not grid_search:
            if 'lc' not in self.conf and len(self.conf) == 1:
                self.ldi['lc'] = self.ldi[list(self.conf.keys())[0]]

            elif 'lc' not in self.conf and len(self.conf) > 1:
                wrange = []
                for mode in self.conf:
                    wrange.append(self.conf[mode]['wave_range'])

                ld_rgi = dipu.ldcs_rgi_prep(teffps=teffps, teffcs=teffcs, teffhs=teffhs, loggs=loggs,
                                            law=self.ldi['ld_params']['law'], mh=self.ldi['ld_params']['mh'],
                                            wrange=[np.min(wrange), np.max(wrange)], model=self.ldi['ld_params']['model'],
                                            data_path=self.ldi['ld_params']['data_path'],
                                            mu_min=self.ldi['ld_params']['mu_min'], cpu_num=self.cpu_num)

                self.ldi['lc'] = ld_rgi

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
                if self.conf[mode]['corr'][item] == 'free':
                    nofp += self.idc[mode]['noo']
        self.idc['nofp'] = nofp
        self.idc['tnoo'] = tnoo

        # dmask = np.zeros(self.surface_grid['noes'] * 3 + nofp, dtype=bool)
        dmask = np.zeros(self.surface_grid['noes'] * 2 + nofp, dtype=bool)
        cmask = dmask.copy()
        for mode in self.conf:
            noo = self.idc[mode]['noo']

            for parn in self.conf[mode]['corr']:
                if self.conf[mode]['corr'][parn] == 'free':

                    if not cmask.any():
                        cmask = dmask.copy()
                        # cmask[3 * self.surface_grid['noes']: 3 * self.surface_grid['noes'] + noo] = True
                        cmask[2 * self.surface_grid['noes']: 2 * self.surface_grid['noes'] + noo] = True

                    else:
                        ind = max(np.where(cmask)[0]) + 1
                        cmask = dmask.copy()
                        cmask[ind: ind + noo] = True

                    self.conf[mode]['mask'][parn] = cmask.copy()

                elif isinstance(self.conf[mode]['corr'][parn], (tuple, list, np.ndarray)):
                    self.conf[mode]['corr'][parn] = np.array(self.conf[mode]['corr'][parn])[:, None]

                elif self.conf[mode]['corr'][parn] is None:
                    self.conf[mode]['corr'][parn] = np.array([None] * noo)[:, None]

                else:
                    self.conf[mode]['corr'][parn] = np.array([self.conf[mode]['corr'][parn]] * noo)[:, None]

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
                    print('\033[1m' + 'Atomic line profile(s)...' + '\033[0m')

                elif mode == 'mol1':
                    print('\033[1m' + 'Molecular band profile(s) (1)...' + '\033[0m')

                elif mode == 'mol2':
                    print('\033[1m' + 'Molecular band profile(s) (2)...' + '\033[0m')

                elif mode == 'lc':
                    print('\033[1m' + 'Light curve profile...' + '\033[0m')

            rgi = self.ldi[mode]

            ldcs_phot = np.array(rgi['phot'][:2]).T
            ldcs_cool = np.array(rgi['cool'][:2]).T
            ldcs_hot = np.array(rgi['hot'][:2]).T

            lis_phot = rgi['phot'][-1]
            lis_cool = rgi['cool'][-1]
            lis_hot = rgi['hot'][-1]

            nop = self.idc[mode]['nop']

            times = all_times[mode]

            noes = self.surface_grid['noes']
            grid_lats = self.surface_grid['grid_lats']
            grid_longs = self.surface_grid['grid_longs']
            t0 = self.params['t0']
            period = self.params['period']
            incl = np.deg2rad(self.params['incl'])
            ld_law = self.ldi['ld_params']['law']

            if mode != 'lc':
                lp_vels = self.idc[mode]['lp_vels']
                phot_lp_data = self.idc[mode]['phot_lp_data']
                cool_lp_data = self.idc[mode]['cool_lp_data']
                hot_lp_data = self.idc[mode]['hot_lp_data']
                vrt = self.params['vrt']
                vels = self.idc[mode]['vels']

                input_args_spec = [(itime, np.asarray(plats, dtype=np.float64), np.asarray(vlats, dtype=np.float64),
                                    np.asarray(ldcs_phot, dtype=np.float64), np.asarray(ldcs_cool, dtype=np.float64),
                                    np.asarray(ldcs_hot, dtype=np.float64), np.asarray(lis_phot, dtype=np.float64),
                                    np.asarray(lis_cool, dtype=np.float64), np.asarray(lis_hot, dtype=np.float64),
                                    np.asarray(areas, dtype=np.float64), np.asarray(grid_lats, dtype=np.float64),
                                    np.asarray(grid_longs, dtype=np.float64), nop, t0, incl, ld_law, noes,
                                    np.asarray(lp_vels, dtype=np.float64), np.asarray(phot_lp_data, dtype=np.float64),
                                    np.asarray(cool_lp_data, dtype=np.float64), np.asarray(hot_lp_data, dtype=np.float64),
                                    vrt, np.asarray(vels, dtype=np.float64), period, info) for itime in times]

                results = dipu.mp_calc_pixel_coeffs(cpu_num=self.cpu_num, input_args=input_args_spec, mode=mode)

            if mode == 'lc':
                input_args_lc = [(itime, np.asarray(plats, dtype=np.float64), np.asarray(ldcs_phot, dtype=np.float64),
                                  np.asarray(ldcs_cool, dtype=np.float64), np.asarray(ldcs_hot, dtype=np.float64),
                                  np.asarray(lis_phot, dtype=np.float64), np.asarray(lis_cool, dtype=np.float64),
                                  np.asarray(lis_hot, dtype=np.float64), np.asarray(areas, dtype=np.float64),
                                  np.asarray(grid_lats, dtype=np.float64), np.asarray(grid_longs, dtype=np.float64),
                                  t0, incl, ld_law, noes, period, info) for itime in times]

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
            prf = prf + amp
            # prf = (prf - lib.median(prf)) * amp + 1.0

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

    def generate_synthetic_lightcurve(self, fssc, fssh, coeffs_cube, amp=None, lib=np):

        fssp = 1.0 - (fssc + fssh)

        wgt_phot = coeffs_cube[:, 0] * fssp
        wgt_cool = coeffs_cube[:, 1] * fssc
        wgt_hot = coeffs_cube[:, 2] * fssh

        flux = lib.sum(wgt_phot + wgt_cool + wgt_hot)

        if amp is not None:
            flux += amp

        return flux

    # def norm_x0s(self, x0s):
    #
    #     noes = self.surface_grid['noes']
    #
    #     fssc = x0s[:noes]
    #     fssh = x0s[noes:2 * noes]
    #     fssp = x0s[2 * noes:3 * noes]
    #
    #     fsst = fssc + fssh + fssp
    #     fssc /= fsst
    #     fssh /= fsst
    #     fssp /= fsst
    #
    #     return fssc, fssh, fssp

    def custom_kstest(self, data, cdf='norm', loc=0, scale=1):
        """
        Custom implementation of the Kolmogorov-Smirnov test.

        Parameters:
            data (array-like): The observed data.
            cdf (str): The name of the theoretical distribution (default is 'norm' for normal distribution).
            loc (float): Mean of the theoretical distribution (default is 0 for standard normal).
            scale (float): Standard deviation of the theoretical distribution (default is 1 for standard normal).

        Returns:
            D (float): The KS statistic (maximum difference).
            p_value (float): The p-value for the test.
        """
        # Sort the data
        data = jnp.sort(data)

        # Compute the empirical CDF
        n = len(data)
        ecdf = jnp.arange(1, n + 1) / n

        # Compute the theoretical CDF for the given distribution
        if cdf == 'norm':
            theoretical_cdf = norm.cdf(data, loc=loc, scale=scale)
        else:
            raise ValueError(f"Unsupported CDF: {cdf}")

        # Compute the KS statistic (maximum absolute difference)
        D_plus = ecdf - theoretical_cdf
        D_minus = theoretical_cdf - jnp.arange(0, n) / n
        D = np.max(jnp.maximum(D_plus, D_minus))

        # Compute the p-value
        sqrt_n = jnp.sqrt(n)
        lambda_ = (sqrt_n + 0.12 + 0.11 / sqrt_n) * D
        p_value = 0
        for k in range(100):
            p_value += (-1) ** k * 2.0 * jnp.exp(-2.0 * lambda_ ** 2.0 * (k + 1.0) ** 2.0)

        return D, p_value

    def entropy(self, fssc, fssh, fssp, wp):

        memc = jnp.sum(wp * fssc * jnp.log(fssc / self.fsl))
        memh = jnp.sum(wp * fssh * jnp.log(fssh / self.fsl))
        memp = jnp.sum(wp * (1.0 - fssp) * jnp.log((1.0 - fssp) / self.fsl))
        mem = (memc + memh + memp)

        return mem

    def objective_func(self, x0s, alpha, beta, gamma, delta, lmbd, lmbd2, cpcs):

        # fssc, fssh, fssp = self.norm_x0s(x0s)

        noes = self.surface_grid['noes']
        fssc = x0s[:noes]
        fssh = x0s[noes:2 * noes]
        fssp = 1.0 - (fssc + fssh)

        chisqs = {'line': 0.0, 'mol1': 0.0, 'mol2': 0.0, 'lc': 0.0}
        p_value = 0
        ks_stat = 0

        for mode in self.conf:
            if mode != 'lc':
                pnoo = self.idc[mode]['noo']
                pnop = self.idc[mode]['nop']

                rv = self.conf[mode]['corr']['rv']
                if isinstance(rv, str) and rv == 'free':
                    rv = x0s[self.conf[mode]['mask']['rv']]
                rv_axis = None if None in rv else 0
                rv = None if None in rv else rv

                amp = self.conf[mode]['corr']['amp']
                if isinstance(amp, str) and amp == 'free':
                    amp = x0s[self.conf[mode]['mask']['amp']]
                amp_axis = None if None in amp else 0
                amp = None if None in amp else amp

                pvmap = vmap(self.generate_synthetic_profile, in_axes=[None, None, rv_axis, amp_axis, 0, None, None])
                sprf, scale_factors = pvmap(fssc, fssh, rv, amp, cpcs[mode]['coeffs_cube'], mode, jnp)
                oprf = jnp.array(self.idc[mode]['data_cube'][0])
                oprf_errs = jnp.array(self.idc[mode]['data_cube'][1])

                norm_presiduals = (oprf - sprf) / oprf_errs
                chisqs[mode] = jnp.sum(norm_presiduals ** 2) / (pnoo * pnop)

                for prow in norm_presiduals:
                    p_value += self.custom_kstest(prow)[1]
                    ks_stat += self.custom_kstest(prow)[0]

                self.conf[mode]['scale_factor'] = scale_factors

            if mode == 'lc':

                lnop = self.idc['lc']['nop']

                amp = self.conf[mode]['corr']['amp'][0]
                if isinstance(amp, str) and self.conf[mode]['corr']['amp'] == 'free':
                    amp = x0s[self.conf[mode]['mask']['amp']]

                olc = jnp.array(self.idc[mode]['data_cube'][0])
                olc_errs = jnp.array(self.idc[mode]['data_cube'][1])

                lvmap = vmap(self.generate_synthetic_lightcurve, in_axes=[None, None, 0, None, None])
                slc = lvmap(fssc, fssh, cpcs['lc']['coeffs_cube'], amp[0], jnp)

                scaling = self.conf['lc']['scaling']
                scale_factor = 1.0
                if scaling['method'] == 'mean':
                    scale_factor = jnp.mean(slc)
                elif scaling['method'] == 'none':
                    scale_factor = 1.0
                self.conf['lc']['scale_factor'] = scale_factor

                slc /= scale_factor

                norm_lresiduals = (olc - slc) / olc_errs
                chisqs['lc'] = jnp.sum(norm_lresiduals ** 2) / lnop

                p_value += self.custom_kstest(norm_lresiduals)[1]
                ks_stat += self.custom_kstest(norm_lresiduals)[0]


        alpha_line = alpha * chisqs['line']
        beta_mol1 = beta * chisqs['mol1']
        gamma_mol2 = gamma * chisqs['mol2']
        delta_lc = delta * chisqs['lc']

        total_weighted_chisq = alpha_line + beta_mol1 + gamma_mol2 + delta_lc

        wp = self.surface_grid['grid_areas'] / np.sum(self.surface_grid['grid_areas'])
        mem = self.entropy(fssc, fssh, fssp, wp)
        # memmax = self.entropy(jnp.ones(noes) * self.fsu, jnp.ones(noes) * self.fsu, jnp.ones(noes) * self.fsl, wp)
        lmbd_mem = lmbd * mem  # / memmax

        mem2 =  jnp.sum(fssc * wp) + jnp.sum(fssh * wp)
        lmbd2_mem = lmbd2 * mem2

        ftot = total_weighted_chisq + lmbd_mem + lmbd2_mem

        if self.force_lines_only_map:
            ftot += jnp.sum((jnp.hstack((fssc, fssh)) - self.lines_only_map) ** 2)

        # lines_only_map = np.loadtxt("EKDra_only_line_fs.txt")
        # ftot += jnp.sum((jnp.hstack((fssc, fssh)) - lines_only_map) ** 2)

        return ftot, (chisqs['line'], chisqs['mol1'], chisqs['mol2'], chisqs['lc'], alpha_line, beta_mol1, gamma_mol2,
                      delta_lc, total_weighted_chisq, mem, lmbd_mem, p_value)

    def minimize(self, x0s, minx0s, maxx0s, maxiter, tol, alpha, beta, gamma, delta, lmbd, lmbd2, cpcs, disp,
                 force_lines_only_map):

        bounds = jnp.array((minx0s, maxx0s))

        if force_lines_only_map:
            optimizer = ScipyBoundedMinimize(fun=self.objective_func, has_aux=True, maxiter=maxiter, tol=tol,
                                             method='L-BFGS-B', options={'disp': disp})

            x0s, _ = optimizer.run(x0s, bounds, alpha, 0.0, 0.0, 0.0, lmbd, lmbd2, cpcs)

            self.lines_only_map = x0s[:2 * self.surface_grid['noes']]
            self.force_lines_only_map = force_lines_only_map

        optimizer = ScipyBoundedMinimize(fun=self.objective_func, has_aux=True, maxiter=maxiter, tol=tol,
                                         method='L-BFGS-B', options={'disp': disp})

        x0s, info = optimizer.run(x0s, bounds, alpha, beta, gamma, delta, lmbd, lmbd2, cpcs)
        x0s = np.array(x0s.copy())

        fssc = x0s[:self.surface_grid['noes']]
        fssh = x0s[self.surface_grid['noes']:2 * self.surface_grid['noes']]
        fssp = 1.0 - (fssc + fssh)

        for mode in self.conf:
            for parn in ['rv', 'amp']:
                if isinstance(self.conf[mode]['corr'][parn], str) and self.conf[mode]['corr'][parn] == 'free':
                    self.opt_results[mode][parn] = x0s.copy()[self.conf[mode]['mask'][parn]].flatten()

                elif  self.conf[mode]['corr'][parn] is None:
                    self.opt_results[mode][parn] = np.array([None] * self.idc[mode]['noo'])

                else:
                    self.opt_results[mode][parn] = self.conf[mode]['corr'][parn].flatten()

        ftot, others = self.objective_func(x0s, alpha, beta, gamma, delta, lmbd, lmbd2, cpcs)

        metrics = np.hstack((others, ftot))
        for i, item in enumerate(self.opt_stats):
            self.opt_stats[item] = metrics[i]

        return np.array(fssc), np.array(fssh), np.array(fssp), info.iter_num, info.num_fun_eval

    def reconstructor(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, lmbd=1.0, lmbd2=0.0, maxiter=100, tol=1e-10,
                      cpcs=None, disp=True, force_lines_only_map=False):

        self.force_lines_only_map = False

        if not cpcs:
            cpcs = self.calc_pixel_coeffs(line_times=self.idc['line']['times'], mol1_times=self.idc['mol1']['times'],
                                          mol2_times=self.idc['mol2']['times'], lc_times=self.idc['lc']['times'],
                                          info=disp)
        if cpcs and 'lc' not in self.conf:

            tmin, tmax = [], []
            for mode in self.conf:
                tmin.append(min(self.idc[mode]['times']))
                tmax.append(max(self.idc[mode]['times']))

            self.opt_results['lc']['ntimes'] = self.params['t0'] + self.params['period'] * np.linspace(0, 2, 200)
            cpcs_lc = self.calc_pixel_coeffs(mode='lc', lc_times=self.opt_results['lc']['ntimes'], info=False,
                                             plps=(False, False), plil=(False, False, False))

        if isinstance(lmbd, (list, np.ndarray)):
            lmbd = self.lambda_search(alpha=alpha, beta=beta, gamma=gamma, delta=delta, lmbds=lmbd, lmbd2=lmbd2,
                                      maxiter=maxiter, tol=tol, cpcs=cpcs, force_lines_only_map=force_lines_only_map)

        noes = self.surface_grid['noes']
        fss = np.zeros(2 * noes) + self.fsl

        x0s = jnp.hstack((fss, jnp.zeros(self.idc['nofp'])))
        # x0s = jnp.hstack((np.loadtxt("pwand_only_line_fs.txt"), jnp.zeros(self.idc['nofp'])))

        minx0s = jnp.array([self.fsl] * (2 * noes) + [-jnp.inf] * self.idc['nofp'])
        maxx0s = jnp.array([self.fsu] * (2 * noes) + [jnp.inf] * self.idc['nofp'])

        if force_lines_only_map:
            print('\033[96m' + 'WARNING: force_lines_only_map activated. First, the surface map will be computed using '
                               'only line profiles. Then, this map will be used as a constraint.' + '\033[0m')

        rfssc, rfssh, rfssp, nit, nfev = self.minimize(x0s=x0s, minx0s=minx0s, maxx0s=maxx0s, maxiter=maxiter,
                                                       tol=tol,alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                                                       lmbd=lmbd, lmbd2=lmbd2, cpcs=cpcs, disp=disp,
                                                       force_lines_only_map=force_lines_only_map)

        # np.savetxt("EKDra_only_line_fs.txt", np.hstack((rfssc, rfssh)))

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

            if mode != 'lc':
                rv = self.opt_results[mode]['rv'].flatten()
                amp = self.opt_results[mode]['amp'].flatten()

                spotless_sprfs = {}
                recons_sprfs = {}
                for i, itime in enumerate(self.idc[mode]['times']):
                    spotless_sprfs[itime] = {}
                    recons_sprfs[itime] = {}

                    spotless_sprf, _ = self.generate_synthetic_profile(fssc=0.0, fssh=0.0, rv=rv[i], amp=amp[i],
                                                                       coeffs_cube=coeffs_cube[i], mode=mode)

                    recons_sprf, _ = self.generate_synthetic_profile(fssc=rfssc, fssh=rfssh, rv=rv[i],
                                                                     amp=amp[i], coeffs_cube=coeffs_cube[i], mode=mode)

                    spotless_sprfs[itime]['prf'] = np.array(spotless_sprf)
                    recons_sprfs[itime]['prf'] = np.array(recons_sprf)

                self.opt_results[mode]['spotless_sprfs'] = spotless_sprfs
                self.opt_results[mode]['recons_sprfs'] = recons_sprfs

            if mode == 'lc':
                lc_amp = self.opt_results[mode]['amp'][0]
                for i, itime in enumerate(self.idc[mode]['times']):
                    flux = self.generate_synthetic_lightcurve(fssc=rfssc, fssh=rfssh, coeffs_cube=coeffs_cube[i],
                                                              amp=lc_amp)
                    recons_slc.append(flux)
                recons_slc = np.array(recons_slc) / self.conf['lc']['scale_factor']

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
        self.opt_results['beta'] = beta
        self.opt_results['gamma'] = gamma
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

    def lambda_search(self, alpha, beta, gamma, delta, lmbds, lmbd2, maxiter, tol, cpcs):
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

            x0s, info = optimizer.run(x0s, bounds, alpha, beta, gamma, delta, lmbd, lmbd2, cpcs)

            _, others = self.objective_func(x0s, alpha, beta, gamma, delta, lmbd, lmbd2, cpcs)

            return others[8], others[9], lmbd

        results = dipu.mp_search(cpu_num=self.cpu_num, input_args=lmbds, func=lambda_search_run)
        parts = np.array(results)

        sort = np.argsort(parts[:, 2])
        self.opt_results['total_wchisqs'] = parts[:, 0][sort]
        self.opt_results['mems'] = parts[:, 1][sort]
        self.opt_results['lmbds'] = parts[:, 2][sort]

        from kneebow.rotor import Rotor

        rotor = Rotor()
        rotor.fit_rotate(np.vstack((self.opt_results['total_wchisqs'], self.opt_results['mems'])).T)
        maxcurve = rotor.get_elbow_index()

        self.opt_results['maxcurve'] = maxcurve

        return self.opt_results['lmbds'][maxcurve]

    def grid_search_run(self, fpmg):

        for j, item in enumerate(self.chisq_grid):
            if item == 'eqw' and 'line' in self.conf:
                self.conf['line']['eqw'] = fpmg[j]

            elif item == 'eqw_phot' and 'line' in self.conf:
                self.conf['line']['eqw']['phot'] = fpmg[j]

            elif item == 'eqw_cool' and 'line' in self.conf:
                self.conf['line']['eqw']['cool'] = fpmg[j]

            elif item == 'eqw_hot' and 'line' in self.conf:
                self.conf['line']['eqw']['hot'] = fpmg[j]

            elif item == 'rv_line' and 'line' in self.conf:
                self.conf['line']['corr']['rv'] = np.array([fpmg[j]] * self.idc['line']['noo'])[:, None]

            elif item == 'amp_line' and 'line' in self.conf:
                self.conf['line']['corr']['amp'] = np.array([fpmg[j]] * self.idc['line']['noo'])[:, None]

            elif item == 'rv_mol1' and 'mol1' in self.conf:
                self.conf['mol1']['corr']['rv'] = np.array([fpmg[j]] * self.idc['mol1']['noo'])[:, None]

            elif item == 'rv_mol2' and 'mol2' in self.conf:
                self.conf['mol2']['corr']['rv'] = np.array([fpmg[j]] * self.idc['mol2']['noo'])[:, None]

            elif item == 'amp_lc' and 'lc' in self.conf:
                self.conf['lc']['corr']['amp'] = np.array([fpmg[j]] * self.idc['lc']['noo'])[:, None]

            else:
                self.params[item] = fpmg[j]

        self.construct_surface_grid(method=self.surface_grid['method'], noes=self.surface_grid['init_noes'],
                                    nlats=self.surface_grid['nlats'], nside=self.surface_grid['nside'],
                                    info=False)

        # dmask = np.zeros(self.surface_grid['noes'] * 3 + nofp, dtype=bool)
        dmask = np.zeros(self.surface_grid['noes'] * 2 + self.idc['nofp'], dtype=bool)
        cmask = dmask.copy()
        for mode in self.conf:
            noo = self.idc[mode]['noo']

            for parn in self.conf[mode]['corr']:
                if isinstance(self.conf[mode]['corr'][parn], str) and self.conf[mode]['corr'][parn] == 'free':

                    if not cmask.any():
                        cmask = dmask.copy()
                        # cmask[3 * self.surface_grid['noes']: 3 * self.surface_grid['noes'] + noo] = True
                        cmask[2 * self.surface_grid['noes']: 2 * self.surface_grid['noes'] + noo] = True

                    else:
                        ind = max(np.where(cmask)[0]) + 1
                        cmask = dmask.copy()
                        cmask[ind: ind + noo] = True

                    self.conf[mode]['mask'][parn] = cmask.copy()

        cpcs = self.calc_pixel_coeffs(line_times=self.idc['line']['times'], mol1_times=self.idc['mol1']['times'],
                                      mol2_times=self.idc['mol2']['times'], lc_times=self.idc['lc']['times'],
                                      plps=(True, False), plil=(True, False, True), info=False)

        noes = self.surface_grid['noes']
        fss = np.zeros(2 * noes) + self.fsl

        x0s = jnp.hstack((fss, jnp.zeros(self.idc['nofp'])))

        minx0s = jnp.array([self.fsl] * (2 * noes) + [-jnp.inf] * self.idc['nofp'])
        maxx0s = jnp.array([self.fsu] * (2 * noes) + [jnp.inf] * self.idc['nofp'])

        self.minimize(x0s=x0s, minx0s=minx0s, maxx0s=maxx0s, maxiter=self.optp_gs['maxiter'], tol=self.optp_gs['tol'],
                      alpha=self.optp_gs['alpha'], beta=self.optp_gs['beta'], gamma=self.optp_gs['gamma'],
                      delta=self.optp_gs['delta'], lmbd=self.optp_gs['lmbd'], lmbd2=self.optp_gs['lmbd2'], cpcs=cpcs,
                      disp=False, force_lines_only_map=self.optp_gs['force_lines_only_map'])

        if self.info_gs:
            output = self.params.copy()
            if 'eqw' in self.fp_keys_gs:
                output['eqw'] = self.conf['line']['eqw']
            if 'rv_line' in self.fp_keys_gs:
                output['rv_line'] = self.conf['line']['corr']['rv'][0]
            if 'amp_line' in self.fp_keys_gs:
                output['amp_line'] = self.conf['line']['corr']['amp'][0]
            if 'rv_mol1' in self.fp_keys_gs:
                output['rv_mol1'] = self.conf['mol1']['corr']['rv'][0]
            if 'rv_mol2' in self.fp_keys_gs:
                output['rv_mol2'] = self.conf['mol2']['corr']['rv'][0]
            if 'amp_lc' in self.fp_keys_gs:
                output['amp_lc'] = self.conf['lc']['corr']['amp'][0]
            output['Total Weighted Chi-square'] = self.opt_stats['Total Weighted Chi-square']
            output['Loss Function Value'] = self.opt_stats['Loss Function Value']

            print()
            print('\033[96m' + '*** Optimization Results ***' + '\033[0m')
            for out in output:
                print('\033[93m' + '{:<11}'.format(out) + ':' + '\033[0m', '\033[1m' + str(output[out]) + '\033[0m')

        if self.minv_gs == "chi":
            return self.opt_stats['Total Weighted Chi-square']
        elif self.minv_gs == "loss":
            return self.opt_stats['Loss Function Value']

    def grid_search(self, fit_params, opt_params=None, info=True, minv="chi", save=False):
        # global grid_search_run

        grid_cpu_num = self.cpu_num
        self.cpu_num = 1
        self.minv_gs = minv

        optp = {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0, 'lmbd': 1.0, 'lmbd2': 0.0, 'maxiter': 100,
                'tol': 1e-5, 'disp': True, 'force_lines_only_map': False}
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

        self.optp_gs = optp
        self.fp_keys_gs = fp_keys
        self.info_gs = info
        self.chisq_grid = chisq_grid

        # def grid_search_run(fpmg):
        #
        #     for j, item in enumerate(chisq_grid):
        #         if item == 'eqw' and 'line' in self.conf:
        #             self.conf['line']['eqw'] = fpmg[j]
        #
        #         elif item == 'eqw_phot' and 'line' in self.conf:
        #             self.conf['line']['eqw']['phot'] = fpmg[j]
        #
        #         elif item == 'eqw_cool' and 'line' in self.conf:
        #             self.conf['line']['eqw']['cool'] = fpmg[j]
        #
        #         elif item == 'eqw_hot' and 'line' in self.conf:
        #             self.conf['line']['eqw']['hot'] = fpmg[j]
        #
        #         elif item == 'rv_line' and 'line' in self.conf:
        #             self.conf['line']['corr']['rv'] = np.array([fpmg[j]] * self.idc['line']['noo'])[:, None]
        #
        #         elif item == 'amp_line' and 'line' in self.conf:
        #             self.conf['line']['corr']['amp'] = np.array([fpmg[j]] * self.idc['line']['noo'])[:, None]
        #
        #         elif item == 'rv_mol1' and 'mol1' in self.conf:
        #             self.conf['mol1']['corr']['rv'] = np.array([fpmg[j]] * self.idc['mol1']['noo'])[:, None]
        #
        #         elif item == 'rv_mol2' and 'mol2' in self.conf:
        #             self.conf['mol2']['corr']['rv'] = np.array([fpmg[j]] * self.idc['mol2']['noo'])[:, None]
        #
        #         elif item == 'amp_lc' and 'lc' in self.conf:
        #             self.conf['lc']['corr']['amp'] = np.array([fpmg[j]] * self.idc['lc']['noo'])[:, None]
        #
        #         else:
        #             self.params[item] = fpmg[j]
        #
        #     self.construct_surface_grid(method=self.surface_grid['method'], noes=self.surface_grid['init_noes'],
        #                                 nlats=self.surface_grid['nlats'], nside=self.surface_grid['nside'],
        #                                 info=False)
        #
        #     # dmask = np.zeros(self.surface_grid['noes'] * 3 + nofp, dtype=bool)
        #     dmask = np.zeros(self.surface_grid['noes'] * 2 + self.idc['nofp'], dtype=bool)
        #     cmask = dmask.copy()
        #     for mode in self.conf:
        #         noo = self.idc[mode]['noo']
        #
        #         for parn in self.conf[mode]['corr']:
        #             if isinstance(self.conf[mode]['corr'][parn], str) and self.conf[mode]['corr'][parn] == 'free':
        #
        #                 if not cmask.any():
        #                     cmask = dmask.copy()
        #                     # cmask[3 * self.surface_grid['noes']: 3 * self.surface_grid['noes'] + noo] = True
        #                     cmask[2 * self.surface_grid['noes']: 2 * self.surface_grid['noes'] + noo] = True
        #
        #                 else:
        #                     ind = max(np.where(cmask)[0]) + 1
        #                     cmask = dmask.copy()
        #                     cmask[ind: ind + noo] = True
        #
        #                 self.conf[mode]['mask'][parn] = cmask.copy()
        #
        #     cpcs = self.calc_pixel_coeffs(line_times=self.idc['line']['times'], mol1_times=self.idc['mol1']['times'],
        #                                   mol2_times=self.idc['mol2']['times'], lc_times=self.idc['lc']['times'],
        #                                   plps=(True, False), plil=(True, False, True), info=False)
        #
        #     noes = self.surface_grid['noes']
        #     fss = np.zeros(2 * noes) + self.fsl
        #
        #     x0s = jnp.hstack((fss, jnp.zeros(self.idc['nofp'])))
        #
        #     minx0s = jnp.array([self.fsl] * (2 * noes) + [-jnp.inf] * self.idc['nofp'])
        #     maxx0s = jnp.array([self.fsu] * (2 * noes) + [jnp.inf] * self.idc['nofp'])
        #
        #     self.minimize(x0s=x0s, minx0s=minx0s, maxx0s=maxx0s, maxiter=optp['maxiter'], tol=optp['tol'],
        #                   alpha=optp['alpha'], beta=optp['beta'], gamma=optp['gamma'], delta=optp['delta'],
        #                   lmbd=optp['lmbd'], lmbd2=optp['lmbd2'], cpcs=cpcs, disp=False)
        #
        #     if info:
        #         output = self.params.copy()
        #         if 'eqw' in fp_keys:
        #             output['eqw'] = self.conf['line']['eqw']
        #         if 'rv_line' in fp_keys:
        #             output['rv_line'] = self.conf['line']['corr']['rv'][0]
        #         if 'amp_line' in fp_keys:
        #             output['amp_line'] = self.conf['line']['corr']['amp'][0]
        #         if 'rv_mol1' in fp_keys:
        #             output['rv_mol1'] = self.conf['mol1']['corr']['rv'][0]
        #         if 'rv_mol2' in fp_keys:
        #             output['rv_mol2'] = self.conf['mol2']['corr']['rv'][0]
        #         if 'amp_lc' in fp_keys:
        #             output['amp_lc'] = self.conf['lc']['corr']['amp'][0]
        #         output['Total Weighted Chi-square'] = self.opt_stats['Total Weighted Chi-square']
        #         output['Loss Function Value'] = self.opt_stats['Loss Function Value']
        #
        #         print()
        #         print('\033[96m' + '*** Optimization Results ***' + '\033[0m')
        #         for out in output:
        #             print('\033[93m' + '{:<11}'.format(out) + ':' + '\033[0m', '\033[1m' + str(output[out]) + '\033[0m')
        #
        #     # return self.opt_stats['Loss Function Value']
        #     return self.opt_stats['Total Weighted Chi-square']

        chisqs = dipu.mp_search(cpu_num=grid_cpu_num, func=self.grid_search_run, input_args=fit_params_mg)
        chisq_grid['chisqs'] = np.array(chisqs)

        self.cpu_num = grid_cpu_num

        if save:
            file = open(save, 'wb')
            pickle.dump(chisq_grid, file)
            file.close()

        # print(chisq_grid)

        dipu.make_grid_contours(chisq_grid, minv)

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

        msg = '\n' + '\033[94m' + 'Preparing the GUI for display. Please wait...' + '\033[0m'
        sys.stdout.write('\r' + msg)

        app = QtWidgets.QApplication(sys.argv)
        pg = PlotGUI(self, plot_params=plotp)
        pg.show()

        msg = '\033[94m' + "The GUI is currently being displayed..." + '\033[0m' + '\n'
        sys.stdout.write('\r' + msg)

        sys.exit(app.exec_())

    def test(self, modes_input, artificial_map=None, spots_params=None, opt_params=None, plot_params=None, save_data_path=None):

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

        if artificial_map is None and spots_params is not None:
            fake_fssc, fake_fssh = dipu.generate_spotted_surface(surface_grid=self.surface_grid,
                                                                 spots_params=spots_params)
        if artificial_map is not None and spots_params is None:
            noes = self.surface_grid['noes']
            fake_fssc, fake_fssh = artificial_map[:noes], artificial_map[noes:2 * noes]

        cpcs = self.calc_pixel_coeffs(line_times=self.idc['line']['times'], mol1_times=self.idc['mol1']['times'],
                                      mol2_times=self.idc['mol2']['times'], lc_times=self.idc['lc']['times'], info=True)

        np.random.seed(0)
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

            if mode != 'lc':
                for i, itime in enumerate(self.idc[mode]['times']):
                    sprf, _ = self.generate_synthetic_profile(fssc=fake_fssc, fssh=fake_fssh, rv=rvs[i], amp=amps[i],
                                                              coeffs_cube=cpcs[mode]['coeffs_cube'][i], mode=mode,
                                                              lib=np)

                    # same_random_line_err = np.random.normal(0.0, np.mean(sprf) / self.idc[mode]['snr'],
                    #                                         self.idc[mode]['nop'])
                    same_random_line_err = np.random.normal(0.0, 1.0 / self.idc[mode]['snr'],
                                                            self.idc[mode]['nop'])
                    fprf = sprf + same_random_line_err
                    # fperr = 2.0 * np.ones(self.idc[mode]['nop']) * (np.mean(sprf) / self.idc[mode]['snr'])
                    fperr = np.ones(self.idc[mode]['nop']) / self.idc[mode]['snr']
                    self.idc[mode]['data'].append(np.vstack((self.idc[mode]['vels'], fprf, fperr)).T)

            else:
                slc = np.zeros(self.idc[mode]['nop'])
                for i, itime in enumerate(self.idc[mode]['times']):
                    flux = self.generate_synthetic_lightcurve(fssc=fake_fssc, fssh=fake_fssh,
                                                              coeffs_cube=cpcs[mode]['coeffs_cube'][i],
                                                              amp=amps[0], lib=np)

                    slc[i] = flux

                slc /= self.conf['lc']['scale_factor']

                same_random_lc_err = np.random.normal(0.0, np.mean(slc) / self.idc[mode]['snr'], self.idc[mode]['nop'])
                # same_random_lc_err = np.random.normal(0.0, 1.0 / self.idc[mode]['snr'], self.idc[mode]['nop'])
                flc = slc + same_random_lc_err

                flerr = 1.0 * np.ones(self.idc[mode]['nop']) * (np.mean(slc) / self.idc[mode]['snr'])
                # flerr = np.mean(slc) * np.ones(self.idc[mode]['nop']) / self.idc[mode]['snr']

                self.idc[mode]['data'] = np.vstack((flc, flerr)).T

        if save_data_path is not None:
            file = open(save_data_path, 'wb')
            pickle.dump(self.idc, file)
            file.close()

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
        recons_fssp = self.opt_results['recons_fssp']

        ints = self.opt_results['ints'].copy()

        map_proj_data_dict = {'grid latitudes': np.rad2deg(self.surface_grid['grid_lats']),
                              'grid longitudes': np.rad2deg(self.surface_grid['grid_longs']),
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
