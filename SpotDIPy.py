import numpy as np
import sys
import utils as dipu
from astropy import units as au
import scipy.optimize as sc_optimize
import matplotlib.pyplot as plt
import autograd.numpy as anp
import autograd
from inspect import stack as inspect_stack
import multiprocessing
from kneebow.rotor import Rotor
import tqdm
from p_tqdm import p_map
import pickle


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKRED = '\033[31m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class SpotDIPy:

    def __init__(self, processes=1):

        """
        dict. key 't0': reference time in days
        dict. key 'period': rotational period in days
        dict. key 'Tphot': photosphere temperature of the star in Kelvin
        dict. key 'Tspot': spot temperature of the star in Kelvin
        dict. key 'incl': rotational axial inclination in degree
        dict. key 'period': days
        dict. key 'vsini': projected rotational velocity in km/s
        dict. key 'vrt': macroturbulence velocity in km/s
        dict. key 'gdc': gravity-darkening coefficient
        dict. key 'rv_corr': radial velocity correction value
        dict. key 'resolution': resolution of the observed spectrum
        dict. key 'wrange': wavelength range of the observed spectrum
        """

        self.processes = processes

        self.params = {
                        't0': 2450000.0, 'period': 1.0, 'Tphot': 5500, 'Tspot': 3500,
                        'incl': 90, 'vsini': 20.0, 'vrt': 3.0, 'mass': 1.0,
                        'gdc': 'auto', 'rv_corr_line': 0.0, 'rv_corr_mol1': 0.0,
                        'rv_corr_mol2': 0.0, 'resolution': 0.0
                       }

    def set_param(self, name, value):

        if name not in self.params.keys():
            print('ERROR: ' + name + ' is not a valid parameter. Please choose one of ' + str(self.params.keys()))
            sys.exit()

        self.params[name] = value

    def set_modes(self, modes):

        self.modes = modes.copy()

    def construct_surface_grid(self, method='trapezoid', noes=1500, nlats=20, nside=32, info=True, test=False):

        methods = ['trapezoid', 'phoebe2_marching', 'dots_grid', 'healpy']
        if method not in methods:
            print('\033[93m' + 'WARNING: ' + method + ' is not a valid grid method. Please choose one of ' +
                  str(methods) + '\033[0m')
            print('trapezoid method has been selected')
            method = 'trapezoid'

        radius = dipu.calc_radius(vsini=self.params['vsini'], incl=self.params['incl'], period=self.params['period'])
        omega, requiv, rp = dipu.calc_omega_and_requiv(mass=self.params['mass'], period=self.params['period'],
                                                       re=radius)

        if omega > 1.0:
            print('\033[31m' + 'ERROR: ' + "The rotation rate exceeds 1! Probably,"
                                              " one or more of the 'vsini', 'incl.',"
                                              " 'period', and 'mass' parameters are not"
                                              " in the appropriate values." + '\033[0m')
            sys.exit()

        if info:
            print('\033[93m' + 'Constructing stellar surface...' + '\033[0m')
        if method == 'phoebe2_marching':
            self.surface_grid = dipu.p2m_surface_grid(requiv=requiv, noes=noes, t0=self.params['t0'],
                                                      period=self.params['period'],
                                                      mass=self.params['mass'])
            self.surface_grid['init_noes'] = self.grid_size = noes

        elif method == 'dots_grid':
            self.surface_grid = dipu.dg_surface_grid(omega=omega, nlats=nlats, radius=radius,
                                                     mass=self.params['mass'])
            self.surface_grid['nlats'] = self.grid_size = nlats

        elif method == 'trapezoid':
            self.surface_grid = dipu.tg_surface_grid(omega=omega, nlats=nlats, radius=radius,
                                                     mass=self.params['mass'], processes=self.processes)
            self.surface_grid['nlats'] = self.grid_size = nlats

        elif method == 'healpy':
            self.surface_grid = dipu.hp_surface_grid(omega=omega, nside=nside, radius=radius,
                                                     mass=self.params['mass'], processes=self.processes)
            self.surface_grid['nside'] = self.grid_size = nside

        if self.params['gdc'] != 'auto':
            self.surface_grid['gds'] = np.power(10, self.surface_grid['grid_loggs'] * self.params['gdc'])
        else:
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

            dipu.grid_test(xyzs=xyzs, scalars1=scalars1, scalars2=scalars2,
                           scalars3=scalars3, scalars4=scalars4)

    def set_limb_darkening_params(self, law='linear', mh=0.0, ld_model='mps2',
                                  ld_data_path='/media/eng/Storage/ExoTiC-LD_data_v3.1.2', mu_min=0.1):

        self.ldcs_rgi_cube = {}

        ctps = self.params['Tphot'] * self.surface_grid['gds']
        ctss = self.params['Tspot'] * self.surface_grid['gds']
        min_temp = np.min(np.hstack((ctps, ctss)))
        max_temp = np.max(np.hstack((ctps, ctss)))
        temps = np.arange(min_temp - 50, max_temp + 100, 50)

        min_logg = np.min(self.surface_grid['grid_loggs'])
        max_logg = np.max(self.surface_grid['grid_loggs'])
        loggs = np.arange(min_logg - 0.01, max_logg + 0.02, 0.01)

        if self.modes['line']['mode'] == 'on':
            line_rgi = dipu.ldcs_rgi_prep(temps=temps, loggs=loggs, law=law, mh=mh,
                                          wrange=self.modes['line']['wrange'], ld_model=ld_model,
                                          ld_data_path=ld_data_path, mu_min=mu_min)

            self.ldcs_rgi_cube['line'] = line_rgi

        if self.modes['mol1']['mode'] == 'on':
            mol_rgi_1 = dipu.ldcs_rgi_prep(temps=temps, loggs=loggs, law=law, mh=mh,
                                           wrange=self.modes['mol1']['wrange'], ld_model=ld_model,
                                           ld_data_path=ld_data_path, mu_min=mu_min)

            self.ldcs_rgi_cube['mol1'] = mol_rgi_1

        if self.modes['mol2']['mode'] == 'on':
            mol_rgi_2 = dipu.ldcs_rgi_prep(temps=temps, loggs=loggs, law=law, mh=mh,
                                           wrange=self.modes['mol2']['wrange'], ld_model=ld_model,
                                           ld_data_path=ld_data_path, mu_min=mu_min)

            self.ldcs_rgi_cube['mol2'] = mol_rgi_2

        self.ld_law = law
        self.mh = mh
        self.ld_model = ld_model
        self.mu_min = mu_min
        self.ld_data_path = ld_data_path

    def set_observational_data(self, times, line_obs_data=None, mol1_obs_data=None, mol2_obs_data=None,
                               line_obs_vels=None, mol1_obs_vels=None, mol2_obs_vels=None):

        self.times = times.copy()
        self.phases = (((times - self.params['t0']) / self.params['period']) -
                       np.floor((times - self.params['t0']) / self.params['period']))

        srt = np.argsort(self.phases)
        self.phases = self.phases[srt]
        self.times = self.times[srt]


        self.line_vels = []
        self.mol1_vels = []
        self.mol2_vels = []

        if self.modes['line']['mode'] == 'on':
            self.line_obs_data, self.line_vels, self.nline_vels = dipu.obs_data_prep(data=line_obs_data,
                                                                                     phases=self.phases, srt=srt,
                                                                                     new_vels=line_obs_vels,
                                                                                     mode=self.modes['line']
                                                                                     )

        if self.modes['mol1']['mode'] == 'on':
            self.mol1_obs_data, self.mol1_vels, self.nmol1_vels = dipu.obs_data_prep(data=mol1_obs_data,
                                                                                     phases=self.phases, srt=srt,
                                                                                     new_vels=mol1_obs_vels,
                                                                                     mode=self.modes['mol1']
                                                                                     )

        if self.modes['mol2']['mode'] == 'on':
            self.mol2_obs_data, self.mol2_vels, self.nmol2_vels = dipu.obs_data_prep(data=mol2_obs_data,
                                                                                     phases=self.phases, srt=srt,
                                                                                     new_vels=mol2_obs_vels,
                                                                                     mode=self.modes['mol2']
                                                                                     )

    def set_local_profiles(self, llp_vels=None, llps=None, lmp1_vels=None, lmps1=None, lmp2_vels=None, lmps2=None):

        if self.modes['line']['mode'] == 'on':
            self.llp_data = {'vels': llp_vels, 'prfs': llps}

        if self.modes['mol1']['mode'] == 'on':
            self.lmp1_data = {'vels': lmp1_vels, 'prfs': lmps1}

        if self.modes['mol2']['mode'] == 'on':
            self.lmp2_data = {'vels': lmp2_vels, 'prfs': lmps2}

    def calc_pixel_coeffs(self, times=None, line_vels=None, mol1_vels=None, mol2_vels=None,
                          info=True, processes=1):

        if info:
            print('\033[93m' + 'Calculating coefficients related to the surface elements and preparing'
                  ' the local line profiles for the reconstruction...' + '\033[0m')

        if self.modes['line']['mode'] == 'on':
            self.llp_data['vels'] = self.llp_data['vels'] + self.params['rv_corr_line']

        if self.modes['mol1']['mode'] == 'on':
            self.lmp1_data['vels'] = self.lmp1_data['vels'] + self.params['rv_corr_mol1']

        if self.modes['mol2']['mode'] == 'on':
            self.lmp2_data['vels'] = self.lmp2_data['vels'] + self.params['rv_corr_mol2']

        if self.params['resolution'] > 0.0:
            if self.modes['line']['mode'] == 'on':
                self.llp_data['prfs']['phot'] = dipu.set_instrbroad(wrange=self.modes['line']['wrange'],
                                                                    vels=self.llp_data['vels'],
                                                                    ints=self.llp_data['prfs']['phot'],
                                                                    resolution=self.params['resolution'])
                self.llp_data['prfs']['spot'] = dipu.set_instrbroad(wrange=self.modes['line']['wrange'],
                                                                    vels=self.llp_data['vels'],
                                                                    ints=self.llp_data['prfs']['spot'],
                                                                    resolution=self.params['resolution'])

            if self.modes['mol1']['mode'] == 'on':
                self.lmp1_data['prfs']['phot'] = dipu.set_instrbroad(wrange=self.modes['mol1']['wrange'],
                                                                      vels=self.lmp1_data['vels'],
                                                                      ints=self.lmp1_data['prfs']['phot'],
                                                                      resolution=self.params['resolution'])
                self.lmp1_data['prfs']['spot'] = dipu.set_instrbroad(wrange=self.modes['mol1']['wrange'],
                                                                     vels=self.lmp1_data['vels'],
                                                                     ints=self.lmp1_data['prfs']['spot'],
                                                                     resolution=self.params['resolution'])

            if self.modes['mol2']['mode'] == 'on':
                self.lmp2_data['prfs']['phot'] = dipu.set_instrbroad(wrange=self.modes['mol2']['wrange'],
                                                                     vels=self.lmp2_data['vels'],
                                                                     ints=self.lmp2_data['prfs']['phot'],
                                                                     resolution=self.params['resolution'])
                self.lmp2_data['prfs']['spot'] = dipu.set_instrbroad(wrange=self.modes['mol2']['wrange'],
                                                                     vels=self.lmp2_data['vels'],
                                                                     ints=self.lmp2_data['prfs']['spot'],
                                                                     resolution=self.params['resolution'])

        if self.modes['line']['mode'] == 'on':
            if self.modes['line']['eqw'] is not None:
                if type(self.modes['line']['eqw']) is dict:
                    self.llp_data['prfs']['phot'] = dipu.set_eqw(vels=self.llp_data['vels'],
                                                                 ints=self.llp_data['prfs']['phot'],
                                                                 wrange=self.modes['line']['wrange'],
                                                                 eqw=self.modes['line']['eqw']['phot'])

                    self.llp_data['prfs']['spot'] = dipu.set_eqw(vels=self.llp_data['vels'],
                                                                 ints=self.llp_data['prfs']['spot'],
                                                                 wrange=self.modes['line']['wrange'],
                                                                 eqw=self.modes['line']['eqw']['spot'])
                else:
                    self.llp_data['prfs']['phot'] = dipu.set_eqw(vels=self.llp_data['vels'],
                                                                 ints=self.llp_data['prfs']['phot'],
                                                                 wrange=self.modes['line']['wrange'],
                                                                 eqw=self.modes['line']['eqw'])

                    self.llp_data['prfs']['spot'] = dipu.set_eqw(vels=self.llp_data['vels'],
                                                                 ints=self.llp_data['prfs']['spot'],
                                                                 wrange=self.modes['line']['wrange'],
                                                                 eqw=self.modes['line']['eqw'])

        if times is not None:
            self.times = times.copy()
            self.phases = (((times - self.params['t0']) / self.params['period']) -
                           np.floor((times - self.params['t0']) / self.params['period']))

            srt = np.argsort(self.phases)
            self.phases = self.phases[srt]
            self.times = self.times[srt]

        if line_vels is not None:
            self.line_vels = line_vels.copy()
            self.nline_vels = len(self.line_vels)

        if mol1_vels is not None:
            self.mol1_vels = mol1_vels.copy()
            self.nmol1_vels = len(self.mol1_vels)

        if mol2_vels is not None:
            self.mol2_vels = mol2_vels.copy()
            self.nmol2_vels = len(self.mol2_vels)

        vlats = 2.0 * np.pi * self.surface_grid['grid_rs'] * au.solRad.to(au.km) * self.surface_grid['coslats'] / (
                    self.params['period'] * 86400)

        cpcs = {}
        for i, phase in enumerate(self.phases):

            cpcs[phase] = {}

            nlongs = self.surface_grid['grid_longs'] + 2.0 * np.pi * phase
            coslong = np.cos(nlongs)
            sinlong = np.sin(nlongs)

            mus = (self.surface_grid['sinlats'] * self.surface_grid['cosi'] +
                   self.surface_grid['coslats'] * self.surface_grid['sini'] * coslong)
            ivis = np.where(mus > 0.0)[0]

            cpcs[phase]['ivis'] = ivis.copy()

            dvs = vlats[ivis] * sinlong[ivis] * self.surface_grid['sini']

            gds = self.surface_grid['gds'][ivis]

            loggs = self.surface_grid['grid_loggs'][ivis]

            cpcs[phase]['areas'] = self.surface_grid['grid_areas'][ivis]
            cpcs[phase]['mus'] = mus[ivis]

            pool = multiprocessing.Pool(processes=processes)

            input_args = [(gd, logg, mu, dv) for gd, logg, mu, dv in zip(gds, loggs, mus[ivis], dvs)]
            parts = pool.map(self.pixel_coeffs, input_args)

            pool.close()
            pool.join()

            parts = np.array(parts, dtype=object)

            cpcs[phase]['ctps'] = np.hstack(parts[:, 0])
            cpcs[phase]['ctss'] = np.hstack(parts[:, 1])
            cpcs[phase]['line_ldfs_phot'] = np.hstack(parts[:, 2])
            cpcs[phase]['line_ldfs_spot'] = np.hstack(parts[:, 3])
            cpcs[phase]['line_lps_phot'] = np.vstack(parts[:, 4])
            cpcs[phase]['line_lps_spot'] = np.vstack(parts[:, 5])

            cpcs[phase]['mol1_ldfs_phot'] = np.hstack(parts[:, 6])
            cpcs[phase]['mol1_ldfs_spot'] = np.hstack(parts[:, 7])
            cpcs[phase]['mol1_lps_phot'] = np.vstack(parts[:, 8])
            cpcs[phase]['mol1_lps_spot'] = np.vstack(parts[:, 9])

            cpcs[phase]['mol2_ldfs_phot'] = np.hstack(parts[:, 10])
            cpcs[phase]['mol2_ldfs_spot'] = np.hstack(parts[:, 11])
            cpcs[phase]['mol2_lps_phot'] = np.vstack(parts[:, 12])
            cpcs[phase]['mol2_lps_spot'] = np.vstack(parts[:, 13])

            if info:
                print('\033[92m' + 'Mid-time = ' + ('%0.7f' % self.times[i]) + '  phase = ' + ('%0.5f' % phase) + '\033[0m')

        return cpcs

    def pixel_coeffs(self, args):

        gd, logg, mu, dv = args

        ctp = self.params['Tphot'] * gd
        cts = self.params['Tspot'] * gd


        line_ldf_phot, mol1_ldf_phot, mol2_ldf_phot = 0, 0, 0
        line_ldf_spot, mol1_ldf_spot, mol2_ldf_spot = 0, 0, 0
        line_lp_phot, mol1_lp_phot, mol2_lp_phot = [], [], []
        line_lp_spot, mol1_lp_spot, mol2_lp_spot = [], [], []

        if self.modes['line']['mode'] == 'on':

            line_prop = dipu.pixel_prop_prep(vels=self.line_vels,  lvels=self.llp_data['vels'],
                                             ints_phot=self.llp_data['prfs']['phot'],
                                             ints_spot=self.llp_data['prfs']['spot'],
                                             rgi=self.ldcs_rgi_cube['line'], ctp=ctp, cts=cts, logg=logg,
                                             ld_law=self.ld_law, mu=mu, dv=dv, vrt=self.params['vrt'])

            line_ldf_phot = line_prop[0]
            line_ldf_spot = line_prop[1]
            line_lp_phot = line_prop[2]
            line_lp_spot = line_prop[3]

        if self.modes['mol1']['mode'] == 'on':

            mol1_prop = dipu.pixel_prop_prep(vels=self.mol1_vels, lvels=self.lmp1_data['vels'],
                                              ints_phot=self.lmp1_data['prfs']['phot'],
                                              ints_spot=self.lmp1_data['prfs']['spot'],
                                              rgi=self.ldcs_rgi_cube['mol1'], ctp=ctp, cts=cts, logg=logg,
                                              ld_law=self.ld_law, mu=mu, dv=dv, vrt=self.params['vrt'])

            mol1_ldf_phot = mol1_prop[0]
            mol1_ldf_spot = mol1_prop[1]
            mol1_lp_phot = mol1_prop[2]
            mol1_lp_spot = mol1_prop[3]

        if self.modes['mol2']['mode'] == 'on':

            mol2_prop = dipu.pixel_prop_prep(vels=self.mol2_vels, lvels=self.lmp2_data['vels'],
                                              ints_phot=self.lmp2_data['prfs']['phot'],
                                              ints_spot=self.lmp2_data['prfs']['spot'],
                                              rgi=self.ldcs_rgi_cube['mol2'], ctp=ctp, cts=cts, logg=logg,
                                              ld_law=self.ld_law, mu=mu, dv=dv, vrt=self.params['vrt'])

            mol2_ldf_phot = mol2_prop[0]
            mol2_ldf_spot = mol2_prop[1]
            mol2_lp_phot = mol2_prop[2]
            mol2_lp_spot = mol2_prop[3]

        return (ctp, cts, line_ldf_phot, line_ldf_spot, line_lp_phot, line_lp_spot,
                mol1_ldf_phot, mol1_ldf_spot, mol1_lp_phot, mol1_lp_spot,
                mol2_ldf_phot, mol2_ldf_spot, mol2_lp_phot, mol2_lp_spot)

    def generate_synthetic_profile(self, fss, cpcs, phase):

        areas = cpcs[phase]['areas']
        mus = cpcs[phase]['mus']

        ctps = cpcs[phase]['ctps']
        ctss = cpcs[phase]['ctss']
        ivis = cpcs[phase]['ivis']

        line_prf, mol1_prf, mol2_prf = [], [], []
        if self.modes['line']['mode'] == 'on':

            line_ldfs_phot = cpcs[phase]['line_ldfs_phot']
            line_ldfs_spot = cpcs[phase]['line_ldfs_spot']
            line_lps_phot = cpcs[phase]['line_lps_phot']
            line_lps_spot = cpcs[phase]['line_lps_spot']

            line_prf = dipu.calc_model_prf(fss=fss, ctps=ctps, ctss=ctss, ldfs_phot=line_ldfs_phot,
                                           ldfs_spot=line_ldfs_spot, lps_phot=line_lps_phot, lps_spot=line_lps_spot,
                                           mode=self.modes['line'], ivis=ivis, areas=areas, mus=mus)

        if self.modes['mol1']['mode'] == 'on':

            mol1_ldfs_phot = cpcs[phase]['mol1_ldfs_phot']
            mol1_ldfs_spot = cpcs[phase]['mol1_ldfs_spot']
            mol1_lps_phot = cpcs[phase]['mol1_lps_phot']
            mol1_lps_spot = cpcs[phase]['mol1_lps_spot']

            mol1_prf = dipu.calc_model_prf(fss=fss, ctps=ctps, ctss=ctss, ldfs_phot=mol1_ldfs_phot,
                                           ldfs_spot=mol1_ldfs_spot, lps_phot=mol1_lps_phot, lps_spot=mol1_lps_spot,
                                           ivis=ivis, areas=areas, mus=mus, mode=self.modes['mol1'],
                                           vels=self.mol1_vels)

        if self.modes['mol2']['mode'] == 'on':

            mol2_ldfs_phot = cpcs[phase]['mol2_ldfs_phot']
            mol2_ldfs_spot = cpcs[phase]['mol2_ldfs_spot']
            mol2_lps_phot = cpcs[phase]['mol2_lps_phot']
            mol2_lps_spot = cpcs[phase]['mol2_lps_spot']

            mol2_prf = dipu.calc_model_prf(fss=fss, ctps=ctps, ctss=ctss, ldfs_phot=mol2_ldfs_phot,
                                           ldfs_spot=mol2_ldfs_spot, lps_phot=mol2_lps_phot, lps_spot=mol2_lps_spot,
                                           ivis=ivis,  areas=areas, mus=mus, mode=self.modes['mol2'],
                                           vels=self.mol2_vels)

        return line_prf, mol1_prf, mol2_prf

    def generate_all_synthetic_profiles(self, fss, cpcs):

        line_slps = {}
        mol1_slps = {}
        mol2_slps = {}
        for i, phase in enumerate(cpcs):
            line_slps[phase] = {}
            mol1_slps[phase] = {}
            mol2_slps[phase] = {}

            line_prf, mol1_prf, mol2_prf = self.generate_synthetic_profile(fss, cpcs, phase)

            line_slps[phase]['prf'] = line_prf.copy()
            mol1_slps[phase]['prf'] = mol1_prf.copy()
            mol2_slps[phase]['prf'] = mol2_prf.copy()

        return line_slps, mol1_slps, mol2_slps

    def objective_func(self, fss, cpcs, lmbd, fsl, disp):

        line_chisq = 0.0
        rms_line = 0.0
        mol1_chisq = 0.0
        mol2_chisq = 0.0
        for i, phase in enumerate(cpcs):

            line_slps_prf, mol1_slps_prf, mol2_slps_prf = self.generate_synthetic_profile(fss, cpcs, phase)

            if self.modes['line']['mode'] == 'on':

                line_obs_prf = self.line_obs_data[phase]['prf'].copy()
                line_obs_errs = self.line_obs_data[phase]['errs'].copy()

                line_chisq += anp.sum(((line_obs_prf - line_slps_prf) / line_obs_errs) ** 2) / (
                            self.nline_vels * len(self.phases))

                rms_line += anp.sqrt(anp.sum((line_obs_prf - line_slps_prf) ** 2) / (
                            self.nline_vels * len(self.phases)))

            if self.modes['mol1']['mode'] == 'on':
                mol1_obs_prf = self.mol1_obs_data[phase]['prf'].copy()
                mol1_obs_errs = self.mol1_obs_data[phase]['errs'].copy()

                mol1_chisq += anp.sum(((mol1_obs_prf - mol1_slps_prf) / mol1_obs_errs) ** 2) / (
                            self.nmol1_vels * len(self.phases))

            if self.modes['mol2']['mode'] == 'on':
                mol2_obs_prf = self.mol2_obs_data[phase]['prf'].copy()
                mol2_obs_errs = self.mol2_obs_data[phase]['errs'].copy()

                mol2_chisq += anp.sum(((mol2_obs_prf - mol2_slps_prf) / mol2_obs_errs) ** 2) / (
                            self.nmol2_vels * len(self.phases))

        # TODO: Yüzey elemanlarının alanlarını nasıl ele alacağımızı tekrar düşünelim.
        wp = self.surface_grid['grid_areas'] / anp.max(self.surface_grid['grid_areas'])

        mem = (1.0 / self.surface_grid['noes']) * anp.sum(wp * (fss * anp.log(fss / fsl) +
                                                                (1.0 - fss) * anp.log((1.0 - fss) / (1.0 - fsl))))

        total_chisq = line_chisq + mol1_chisq + mol2_chisq
        ftot = total_chisq + lmbd * mem

        if inspect_stack()[1].function == 'fun_wrapped':
            self.line_chisq = line_chisq
            self.mol1_chisq = mol1_chisq
            self.mol2_chisq = mol2_chisq
            self.chisq = total_chisq
            self.mem = mem
            self.ftot = ftot
            self.lmbd = lmbd

            if disp:
                
                if mem != 0:
                    cdm = total_chisq/mem
                else:
                    cdm = np.inf

                print('total chisq = ', total_chisq, 'MEM =', mem, 'total_chisq/mem =', cdm, 'ftot =', ftot,
                      'line-chisq = ', line_chisq, 'mol1-chisq = ', mol1_chisq, 'mol2-chisq = ', mol2_chisq, 'rms_line =',
                      rms_line)

        return ftot

    def minimize(self, args):

        func, x0, args, bounds, tol, method, options = args

        sc_optimize.minimize(func, x0, args=args, jac=autograd.grad(func), bounds=bounds, tol=tol,
                             method=method, options=options)

        return self.chisq, self.mem, args[1]

    def reconstructor(self, lmbd, method, maxiter, tol, disp, verbose=2, cpcs=None, cpcs_info=True):

        fsl = np.float64(1e-5)
        fsu = np.float64(1.0) - fsl

        if cpcs is None:
            cpcs = self.calc_pixel_coeffs(processes=self.processes, info=cpcs_info)

        dfss = np.ones(self.surface_grid['noes']) * fsl
        bounds = np.array([(fsl, fsu)] * self.surface_grid['noes'])

        options = {'maxiter': maxiter, 'disp': disp}
        if method == 'trust-constr':
            options['verbose'] = verbose

        lmbds = []
        chisqs = []
        mems = []
        maxcurve = 0
        if type(lmbd) in [list, np.ndarray]:

            options['disp'] = disp = False

            pool = multiprocessing.Pool(processes=self.processes)

            input_args = [(self.objective_func, dfss, (cpcs, lmbd[i], fsl, disp), bounds, tol, method, options) for i in range(len(lmbd))]
            results = []
            for result in tqdm.tqdm(pool.imap_unordered(self.minimize, input_args), total=len(input_args)):
                results.append(result)

            pool.close()
            pool.join()

            parts = np.array(results, dtype=object)

            sort = np.argsort(parts[:, 2])
            chisqs = parts[:, 0][sort]
            mems = parts[:, 1][sort]
            lmbds = parts[:, 2][sort]

            np.savetxt('chisqs_ve_mem_test-5.txt', np.vstack((lmbds, chisqs, mems)).T)

            rotor = Rotor()
            rotor.fit_rotate(np.vstack((chisqs, mems)).T)
            maxcurve = rotor.get_elbow_index()

            lmbd = lmbds[maxcurve]

        opt_result = sc_optimize.minimize(self.objective_func, dfss, args=(cpcs, lmbd, fsl, disp),
                                          jac=autograd.grad(self.objective_func), bounds=bounds, tol=tol, method=method,
                                          options=options)

        # TODO: lekesiz profillerin üretilmesi için fss değerleri 0 mı olmalı yoksa 0.00001 mi?
        spotless_line_slps, spotless_mol1_slps, spotless_mol2_slps = self.generate_all_synthetic_profiles(dfss, cpcs)
        recons_line_slps, recons_mol1_slps, recons_mol2_slps = self.generate_all_synthetic_profiles(opt_result.x, cpcs)
        total_spotted_area, partial_spotted_area = dipu.get_total_fs(fss=opt_result.x, areas=self.surface_grid['grid_areas'],
                                                                     lats=self.surface_grid['grid_lats'],
                                                                     incl=self.params['incl'])

        self.recons_result = {'opt_result': opt_result,
                              'spotless_line_slps': spotless_line_slps,
                              'recons_line_slps': recons_line_slps,
                              'spotless_mol1_slps': spotless_mol1_slps,
                              'recons_mol1_slps': recons_mol1_slps,
                              'spotless_mol2_slps': spotless_mol2_slps,
                              'recons_mol2_slps': recons_mol2_slps,
                              'total_spotted_area': total_spotted_area,
                              'partial_spotted_area': partial_spotted_area,
                              'lmbds': lmbds,
                              'chisqs': chisqs,
                              'mems': mems,
                              'maxcurve': maxcurve,
                              }

        return self.recons_result

    def grid_search_main(self, args):

        fit_params, pars, opt_params, ignore_csg = args

        for i, item in enumerate(fit_params):
            if item == 'eqw' and self.modes['line']['mode'] == 'on':
                self.modes['line']['eqw'] = pars[i]

            elif item == 'eqw_phot':
                self.modes['line']['eqw']['phot'] = pars[i]

            elif item == 'eqw_spot':
                self.modes['line']['eqw']['spot'] = pars[i]

            else:
                self.params[item] = pars[i]

        if ignore_csg == 0:
            self.construct_surface_grid(method=self.surface_grid['method'], noes=self.grid_size,
                                        nlats=self.grid_size, nside=self.grid_size, info=False)

        self.set_limb_darkening_params(law=self.ld_law, mh=self.mh, ld_model=self.ld_model,
                                       ld_data_path=self.ld_data_path, mu_min=self.mu_min)

        self.reconstructor(lmbd=opt_params['lmbd'], method=opt_params['method'], maxiter=opt_params['maxiter'],
                           tol=opt_params['tol'], disp=False, cpcs_info=False)

        return self.chisq

    def grid_search(self, fit_params, opt_params, save=False):

        ignore_csg = 1
        fit_arr = []
        for item in fit_params:
            fit_arr.append(fit_params[item])
            if item in ['period', 'vsini', 'mass', 'incl']:
                ignore_csg *= 0
        fit_params_mg = np.array(np.meshgrid(*fit_arr)).T.reshape(-1, len(fit_params))

        chisq_grid = {}
        for i, item in enumerate(fit_params):
            chisq_grid[item] = fit_params_mg[:, i]

        pool = multiprocessing.Pool(processes=self.processes)
        input_args = [(fit_params, fit_params_mg[i], opt_params, ignore_csg) for i in range(len(fit_params_mg))]
        chisqs = np.array(p_map(self.grid_search_main, input_args))

        pool.close()
        pool.join()

        chisq_grid['chisqs'] = chisqs

        if save != False:

            file = open(save, 'wb')
            pickle.dump(chisq_grid, file)
            file.close()

        dipu.make_grid_contours(chisq_grid)

    def plot(self, plot_params=None, save_maps=None):

        from plot_GUI import PlotGUI
        from PyQt5 import QtWidgets

        plotp = {'line_sep_prf': 0.4, 'line_sep_res': 0.01, 'mol_sep_prf': 0.4, 'mol_sep_res': 0.01,
                 'show_err_bars': True, 'fmt': '%0.3f',
                 'markersize': 2, 'linewidth': 1, 'fontsize': 15, 'ticklabelsize': 12}
        if plot_params is not None:
            for pitem in plot_params:
                if pitem in plotp.keys():
                    plotp[pitem] = plot_params[pitem]
                else:
                    print('\033[93m' + 'Warning: ' + pitem + ' is not a valid parameter for plot_params!' + '\033[0m')

        app = QtWidgets.QApplication(sys.argv)
        pg = PlotGUI(self, plot_params=plotp, save_maps=save_maps)
        pg.show()
        app.exec()

    def test(self, times, line_vels, line_snr, spots_params, mol1_vels=None, mol2_vels=None, mol_snr=None,
             opt_params=None, plot_params=None):

        plotp = {'line_sep_prf': 0.03, 'line_sep_res': 0.01, 'mol_sep_prf': 0.03, 'mol_sep_res': 0.01,
                 'show_err_bars': True, 'fmt': '%0.3f', 'markersize': 2, 'linewidth': 1, 'fontsize': 15,
                 'ticklabelsize': 12}
        if plot_params is not None:
            for pitem in plot_params:
                if pitem in plotp.keys():
                    plotp[pitem] = plot_params[pitem]
                else:
                    print('\033[93m' + 'Warning: ' + pitem + ' is not a valid parameter for plot_params!' + '\033[0m')

        optp = {'method': 'L-BFGS-B', 'tol': 1e-6, 'lmbd': 1.0,
                'maxiter': 5000, 'disp': True, 'iprint': True, 'verbose': 2}
        if opt_params is not None:
            for oitem in opt_params:
                if oitem in optp.keys():
                    optp[oitem] = opt_params[oitem]
                else:
                    print('\033[93m' + 'Warning: ' + oitem + ' is not a valid parameter for opt_params!' + '\033[0m')

        cpcs = self.calc_pixel_coeffs(times=times, line_vels=line_vels, mol1_vels=mol1_vels,
                                      mol2_vels=mol2_vels, processes=self.processes)

        fake_fss = dipu.generate_spotted_surface(surface_grid=self.surface_grid, spots_params=spots_params)

        # TODO. Aşağıyı kaldır sonra gerekirse
        self.fake_fss = fake_fss.copy()

        print('\033[93m' + 'Generating the artificial line profiles...' + '\033[0m')
        line_slps, mol1_slps, mol2_slps = self.generate_all_synthetic_profiles(fss=fake_fss, cpcs=cpcs)

        np.random.seed(0)
        fake_line_data = []
        fake_mol1_data = []
        fake_mol2_data = []
        for j, phase in enumerate(line_slps):
            if self.modes['line']['mode'] == 'on':
                same_random_line_err = np.random.normal(0.0, 1.0 / line_snr, len(line_vels))
                fake_line_prf = line_slps[phase]['prf'] + same_random_line_err
                fake_line_err = np.ones(len(line_vels)) * (1.0 / line_snr)
                fake_line_data.append(np.vstack((line_vels, fake_line_prf, fake_line_err)).T)

            if self.modes['mol1']['mode'] == 'on':
                same_random_mol_err = np.random.normal(0.0, 1.0 / mol_snr, len(mol1_vels))
                fake_mol1_prf = mol1_slps[phase]['prf'] + same_random_mol_err
                fake_mol1_err = np.ones(len(mol1_vels)) * (1.0 / mol_snr)
                fake_mol1_data.append(np.vstack((mol1_vels, fake_mol1_prf, fake_mol1_err)).T)

            if self.modes['mol2']['mode'] == 'on':
                same_random_mol_err = np.random.normal(0.0, 1.0 / mol_snr, len(mol2_vels))
                fake_mol2_prf = mol2_slps[phase]['prf'] + same_random_mol_err
                fake_mol2_err = np.ones(len(mol2_vels)) * (1.0 / mol_snr)
                fake_mol2_data.append(np.vstack((mol2_vels, fake_mol2_prf, fake_mol2_err)).T)

        self.set_observational_data(times=times, line_obs_data=fake_line_data, mol1_obs_data=fake_mol1_data,
                                    mol2_obs_data=fake_mol2_data)

        recons_result = self.reconstructor(lmbd=optp['lmbd'], method=optp['method'],
                                           maxiter=optp['maxiter'], tol=optp['tol'],
                                           disp=optp['disp'], verbose=optp['verbose'],
                                           cpcs=cpcs)

        recons_fss = recons_result['opt_result'].x.copy()

        fake_total_fs, _ = dipu.get_total_fs(fss=fake_fss, areas=self.surface_grid['grid_areas'],
                                             lats=self.surface_grid['grid_lats'], incl=self.params['incl'])
        recons_total_fs, _ = dipu.get_total_fs(fss=recons_fss, areas=self.surface_grid['grid_areas'],
                                               lats=self.surface_grid['grid_lats'], incl=self.params['incl'])

        if self.modes['line']['mode'] == 'on':

            dipu.test_prf_plot(phases=self.phases, vels=self.line_vels, data=self.line_obs_data, plotp=plotp,
                               spotless_slps=recons_result['spotless_line_slps'],
                               recons_slps=recons_result['recons_line_slps'], mode='line')

        if self.modes['mol1']['mode'] == 'on':
            dipu.test_prf_plot(phases=self.phases, vels=self.mol1_vels, data=self.mol1_obs_data, plotp=plotp,
                               spotless_slps=recons_result['spotless_mol1_slps'],
                               recons_slps=recons_result['recons_mol1_slps'], mode='mol')

        if self.modes['mol2']['mode'] == 'on':
            dipu.test_prf_plot(phases=self.phases, vels=self.mol2_vels, data=self.mol2_obs_data, plotp=plotp,
                               spotless_slps=recons_result['spotless_mol2_slps'],
                               recons_slps=recons_result['recons_mol2_slps'], mode='mol')

        dipu.test_map_plot(phases=self.phases, surface_grid=self.surface_grid, fake_fss=fake_fss,
                           recons_fss=recons_fss, fake_total_fs=fake_total_fs,
                           recons_total_fs=recons_total_fs, plotp=plotp)

        plt.tight_layout()

        plt.show()
