from exotic_ld import StellarLimbDarkening
from functools import reduce
from itertools import combinations
import platform
import numpy as np
from astropy.coordinates import spherical_to_cartesian as ast_stc
from astropy.coordinates import cartesian_to_spherical as ast_cts
from astropy import units as au, constants as ac
from scipy.interpolate import interp1d, griddata
from scipy.optimize import curve_fit, newton as sc_newton
from scipy.special import roots_legendre
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyvista as pv
from uncertainties.unumpy import uarray as uc_uarray, nominal_values as uc_nominal, std_devs as uc_std
import multiprocessing
import tqdm
import jax


try:
    from . import cutils as cutils
except ImportError:
    import cutils as cutils

if not platform.system() == "Windows":
    import healpy as hp

from screeninfo import get_monitors
screen_width = get_monitors()[0].width
px = 1 / plt.rcParams['figure.dpi']
figsize = (screen_width * px / 3, screen_width * px / 3)


sol = ac.c.to(au.kilometer / au.second).value  # km/s


def p2m_surface_grid(container, requiv, noes, t0, period, mass):

    import os
    os.environ['PHOEBE_ENABLE_ONLINE_PASSBANDS'] = 'FALSE'
    import phoebe
    from phoebe import u as phoebe_units

    b = phoebe.default_star()

    b.set_value('incl', value=90)
    b.set_value('period', value=period * phoebe_units.day)
    b.set_value('t0', value=t0 * phoebe_units.day)
    b.set_value('requiv', value=requiv * phoebe_units.solRad)
    b.set_value('mass', value=mass * phoebe_units.solMass)

    b['ntriangles'] = noes

    b.add_dataset('mesh', compute_times=t0, coordinates=['xyz'],
                  columns=['xs', 'ys', 'zs', 'areas', 'loggs', 'rs', 'volume'])

    b.run_compute(distortion_method='rotstar', irrad_method='none')

    grid_xyzs = b['mesh@xyz_elements@latest@model'].value.copy()
    grid_xcs = b['mesh@xs@latest@model'].value.copy()
    grid_ycs = b['mesh@ys@latest@model'].value.copy()
    grid_zcs = b['mesh@zs@latest@model'].value.copy()
    grid_areas = b['mesh@areas@latest@model'].value.copy()
    grid_loggs = b['mesh@loggs@latest@model'].value.copy()

    rs, lats, longs = ast_cts(grid_xcs, grid_ycs, grid_zcs)

    grid_rs = rs.copy()
    grid_lats = lats.value.copy()
    grid_longs = longs.value.copy()

    container['method'] = 'phoebe2_marching'
    container['grid_xyzs'] = grid_xyzs
    container['grid_areas'] = grid_areas
    container['grid_loggs'] = grid_loggs
    container['grid_lats'] = grid_lats
    container['grid_longs'] = grid_longs
    container['grid_rs'] = grid_rs


def hp_surface_grid(container, omega, nside, radius, mass, cpu_num=1):

    npix = hp.nside2npix(nside)

    input_args = [(nside, pixind, omega, radius) for pixind in range(npix)]
    if cpu_num > 1:
        pool = multiprocessing.Pool(processes=cpu_num)
        results = pool.map(hp_surface_grid_main, input_args)
        pool.close()
        pool.join()
    else:
        results = []
        for item in input_args:
            results.append(hp_surface_grid_main(item))

    parts = np.array(results, dtype=object)

    grid_areas = np.hstack(parts[:, 0])
    grid_xcs = np.hstack(parts[:, 1])
    grid_ycs = np.hstack(parts[:, 2])
    grid_zcs = np.hstack(parts[:, 3])
    grid_xyzs = np.vstack(parts[:, 4])

    rs, lats, longs = ast_cts(grid_xcs, grid_ycs, grid_zcs)

    grid_rs = rs.value.copy()
    grid_lats = lats.value.copy()
    grid_longs = longs.value.copy()
    grid_loggs = calc_logg(mass=mass, re=radius, rs=grid_rs, omega=omega, theta=grid_lats + np.pi / 2.0)

    container['method'] = 'healpy'
    container['grid_xyzs'] = grid_xyzs
    container['grid_areas'] = grid_areas
    container['grid_loggs'] = grid_loggs
    container['grid_lats'] = grid_lats
    container['grid_longs'] = grid_longs
    container['grid_rs'] = grid_rs


def hp_surface_grid_main(args):

    nside, pixind, omega, radius = args

    center = hp.pix2vec(nside, pixind, nest=False)
    polygon = hp.boundaries(nside, pixind, step=1, nest=False)

    _, plats, plongs = ast_cts(polygon[0], polygon[1], polygon[2])

    xps = []
    yps = []
    zps = []
    for plat, plong in zip(plats.value, plongs.value):
        rplat = calc_r_to_Re_ratio(omega=omega, theta=plat + np.pi / 2.0)
        rplat = rplat * radius

        xp, yp, zp = ast_stc(rplat, plat, plong)

        xps.append(xp.value)
        yps.append(yp.value)
        zps.append(zp.value)

    polygon = [[[xps[0], yps[0], zps[0]],
                [xps[1], yps[1], zps[1]],
                [xps[2], yps[2], zps[2]]],

               [[xps[2], yps[2], zps[2]],
                [xps[3], yps[3], zps[3]],
                [xps[0], yps[0], zps[0]]]]

    cr, clat, clong = ast_cts(center[0], center[1], center[2])

    rclat = calc_r_to_Re_ratio(omega=omega, theta=clat.value + np.pi / 2.0)
    rclat = rclat * radius

    xpc, ypc, zpc = ast_stc(rclat, clat.value, clong.value)

    area = calc_triangle_area(polygon[0]) + calc_triangle_area(polygon[1])

    return area, xpc.value, ypc.value, zpc.value, polygon


def td_surface_grid(container, omega, nlats, radius, mass, cpu_num=1):

    blats = np.linspace(np.pi / 2., -np.pi / 2., nlats + 1)

    ipn = 4
    pzh = abs(np.sin(blats[0]) - np.sin(blats[1])) * radius
    uarea = 2. * np.pi * radius * pzh / ipn

    input_args = [([blats[i], blats[i + 1]], uarea, omega, radius, mass) for i in range(len(blats) - 1)]

    if cpu_num > 1:
        pool = multiprocessing.Pool(processes=cpu_num)
        # results = pool.map(td_surface_grid_main, input_args)
        results = pool.map(cutils.td_grid_sect, input_args)
        pool.close()
        pool.join()
    else:
        results = []
        for item in input_args:
            results.append(cutils.td_grid_sect(item))

    parts = np.array(list(results), dtype=object)

    container['method'] = 'trapezoid'
    container['blats'] = blats
    container['nlons'] = parts[:, 0]
    container['blongs'] = parts[:, 1]
    container['grid_areas'] = np.hstack(parts[:, 2]).reshape(-1, 2).sum(axis=1)
    container['grid_rs'] = np.hstack(parts[:, 3])
    container['grid_lats'] = np.hstack(parts[:, 4])
    container['grid_longs'] = np.hstack(parts[:, 5])
    container['grid_xyzs'] = np.vstack(parts[:, 6])
    container['grid_loggs'] = np.hstack(parts[:, 7])


def calc_triangle_area(polygon):

    v1 = np.array(polygon[0])
    v2 = np.array(polygon[1])
    v3 = np.array(polygon[2])

    vector1 = v2 - v1
    vector2 = v3 - v1

    cross_product = np.cross(vector1, vector2)

    area = np.linalg.norm(cross_product) / 2.0

    return area


def input_data_prep(container, conf):

    scale_factor = 1.0
    max_reg = [1.0]

    for mode in conf:
        if mode != 'lc':
            times = np.array(container[mode]['times'])
            data = np.array(container[mode]['data'])

            srt = np.argsort(times)
            times = times[srt]
            data = data[srt]

            data_dict = {}
            data_cube = [[], []]
            for i, itime in enumerate(times):

                data_dict[itime] = {}

                ovels, oprf, oerrs = data[i][:, 0], data[i][:, 1], data[i][:, 2]

                if conf[mode]['scaling']['method'] == 'mean':
                    scale_factor = np.average(oprf)

                elif conf[mode]['scaling']['method'] == 'max':
                    per = 100. / conf[mode]['scaling']['percent']
                    if conf[mode]['scaling']['side'] == 'both':
                        max_reg = np.hstack((oprf[:int(len(oprf) / per)], oprf[-int(len(oprf) / per):]))
                    elif conf[mode]['scaling']['side'] == 'left':
                        max_reg = oprf[:int(len(oprf) / per)]
                    elif conf[mode]['scaling']['side'] == 'right':
                        max_reg = oprf[-int(len(oprf) / per):]
                    scale_factor = np.average(max_reg)

                elif conf[mode]['scaling']['method'] == 'region':
                    owavs = ((ovels * np.average(conf[mode]['wrange'])) / sol + np.average(conf[mode]['wrange']))

                    wavgw = []
                    wavgf = []
                    for region in conf[mode]['scaling']['ranges']:
                        wr = np.argwhere((owavs >= region[0]) & (owavs <= region[1])).T[0]

                        wavgw.append(np.average(owavs[wr]))
                        wavgf.append(np.average(oprf[wr]))

                    x1, x2, x3 = wavgw
                    y1, y2, y3 = wavgf
                    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
                    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
                    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
                    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
                    scale_factor = A * owavs ** 2 + B * owavs + C

                oprf /= scale_factor
                oerrs /= scale_factor

                if container[mode]['vels'] is not None:
                    irese = lin_interp_with_unc(ovels, uc_uarray(oprf, oerrs) , container[mode]['vels'])
                    ovels, oprf, oerrs = container[mode]['vels'], uc_nominal(irese), uc_std(irese)

                data_dict[itime]['vels'] = ovels.copy()
                data_dict[itime]['prf'] = oprf.copy()
                data_dict[itime]['errs'] = oerrs.copy()
                data_cube[0].append(oprf.copy())
                data_cube[1].append(oerrs.copy())

            container[mode]['times'] = times.copy()
            container[mode]['data'] = data_dict.copy()
            container[mode]['vels'] = data_dict[times[0]]['vels'].copy()
            container[mode]['noo'] = len(container[mode]['times'])
            container[mode]['nop'] = len(container[mode]['vels'])
            container[mode]['data_cube'] = np.array(data_cube)

        else:
            times = np.array(container[mode]['times'])
            data = np.array(container[mode]['data'])
            fluxs = data[:, 0]
            errs = data[:, 1]

            srt = np.argsort(times)
            times = times[srt]
            fluxs = fluxs[srt]
            errs = errs[srt]

            if conf[mode]['scaling']['method'] == 'mean':
                scale_factor = np.average(fluxs)

            fluxs /= scale_factor
            errs /= scale_factor

            if container[mode]['norsp'] is not None:
                norsp = container[mode]['norsp']
                ntimes = np.linspace(min(times), max(times), norsp)
                irese = lin_interp_with_unc(times, uc_uarray(fluxs, errs), ntimes)
                times, fluxs, errs = ntimes, uc_nominal(irese), uc_std(irese)

            data_dict = {'fluxs': fluxs, 'errs': errs}

            container[mode]['times'] = times.copy()
            container[mode]['data'] = data_dict.copy()
            container[mode]['noo'] = 1
            container[mode]['nop'] = len(container[mode]['times'])
            container[mode]['data_cube'] = np.array([fluxs, errs])


def get_total_fs(fssc, fssh, areas, lats, incl):

    total_fsc = np.sum(fssc * areas) / np.sum(areas)
    total_fsh = np.sum(fssh * areas) / np.sum(areas)
    total_fsp = np.sum((1. - fssc - fssh) * areas) / np.sum(areas)

    w = np.argwhere(lats >= -np.deg2rad(incl)).T[0]
    partial_fsc = np.sum(fssc[w] * areas[w]) / np.sum(areas[w])
    partial_fsh = np.sum(fssh[w] * areas[w]) / np.sum(areas[w])
    partial_fsp = np.sum((1. - fssc[w] - fssh[w]) * areas[w]) / np.sum(areas[w])

    return total_fsc, total_fsh, total_fsp, partial_fsc, partial_fsh, partial_fsp


def get_hemisphere_fs(fssc, fssh, epoch, areas, lats, longs, incl):

    nlong = longs + 2.0 * np.pi * epoch
    mu = np.sin(lats) * np.cos(np.deg2rad(incl)) + np.cos(lats) * np.sin(np.deg2rad(incl)) * np.cos(nlong)
    w = np.argwhere(mu > 0.0).T[0]

    hemi_fsc = np.sum(fssc[w] * areas[w]) / np.sum(areas[w])
    hemi_fsh = np.sum(fssh[w] * areas[w]) / np.sum(areas[w])
    hemi_fsp = np.sum((1. - fssc[w] - fssh[w]) * areas[w]) / np.sum(areas[w])

    return hemi_fsc, hemi_fsh, hemi_fsp


def calc_logg(mass, re, rs, omega, theta):

    mass_kg = mass * au.solMass.to(au.kg)
    g_cm = mass_kg * ac.G.to(au.cm**3 / (au.kg * au.s**2)).value
    r_cm = re * au.solRad.to(au.cm)
    r_ = rs / re

    if type(theta) in [list, np.ndarray]:
        theta[theta > np.pi / 2.] = np.pi - theta[theta > np.pi / 2.]

    else:
        theta = np.pi - theta if theta > np.pi / 2. else theta

    geff = (g_cm / r_cm**2) * (1.0 / r_**4 + omega**4 * r_**2 * np.sin(theta)**2
                               - 2.0 * omega**2 * np.sin(theta)**2 / r_)**0.5

    return np.log10(geff)


def calc_gds_classic(loggs, beta=0.08):

    gs = 10 ** loggs
    g_ratio = gs / np.max(gs)

    Tratio = g_ratio ** beta
    Fratio = Tratio ** 4

    return Tratio, Fratio


def calc_gds_LR(omega, thetas):

    Tratio = np.zeros(len(thetas))
    Fratio = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):

        theta = np.pi - theta if theta > np.pi / 2. else theta

        r = calc_r_to_Re_ratio(omega=omega, theta=theta)

        if theta == 0:
            fluxw = np.exp(2.0 * omega**2 * r**3 / 3.0)

        elif theta == np.pi / 2.:
            fluxw = (1.0 - omega**2 * r**3)**-(2.0 / 3.0)

        else:
            nu = calc_nu(omega=omega, r=r, theta=theta)
            fluxw = np.tan(nu)**2 / np.tan(theta)**2

        Tratio[i] = ((r**-4 + omega**4 * r**2 * np.sin(theta)**2 - (2.0 * omega**2 * np.sin(theta)**2 / r))**(1.0/8.0)
                  * fluxw**0.25)

        Fratio[i] = Tratio[i]**4

    return Tratio, Fratio


def r_func(r, omega, theta):

    return omega ** -2 + 0.5 - (omega ** 2 * r) ** -1 - 0.5 * r ** 2 * np.sin(theta)**2


def calc_r_to_Re_ratio(omega, theta):

    theta = np.pi - theta if theta > np.pi / 2. else theta
    r = sc_newton(r_func, 1.0, args=(omega, theta))

    return r


def nu_func(nu, omega, r, theta):

    return (np.cos(nu) + np.log(np.tan(nu / 2.)) - omega**2 * r**3 *
            np.cos(theta)**3 / 3.0 - np.cos(theta) - np.log(np.tan(theta / 2.)))


def calc_nu(omega, r, theta):

    theta = np.pi - theta if theta > np.pi / 2. else theta

    nu = sc_newton(nu_func, theta, args=(omega, r, theta))

    return nu


def calc_radius(vsini, incl, period):

    veq = vsini / np.sin(np.deg2rad(incl))
    radius = (veq * period * 86400. / (2. * np.pi)) * au.km.to(au.solRad)

    return radius


def calc_omega_and_requiv(mass, period, re):

    mass_kg = mass * au.solMass.to(au.kg)
    gm = mass_kg * ac.G.value
    period_s = period * au.day.to(au.second)
    re_m = re * au.solRad.to(au.m)
    omega = (2.0 * np.pi / period_s) * np.sqrt(re_m ** 3 / gm)
    rp = calc_r_to_Re_ratio(omega=omega, theta=0) * re

    requiv = (re ** 2 * rp) ** (1. / 3.)

    return omega, requiv, rp


def gaussian(x, amp, cen, sig, lvl):
    return amp * np.exp(-(x - cen) ** 2 / (2 * sig ** 2)) + lvl


def fit_gaussian(x, y):
    cen = np.sum(x * y) / np.sum(y)
    sig = np.sqrt(np.sum(y * (x - cen) ** 2) / np.sum(y))
    amp, lvl = max(y), min(y)

    bounds = [(-np.inf, -np.inf, 0.0, -np.inf), (np.inf, np.inf, np.inf, np.inf)]
    popt, _ = curve_fit(gaussian, x, y, p0=[amp, cen, sig, lvl], bounds=bounds)

    return popt


def set_eqw(vels, ints, wrange, eqw):

    amp, cen, sig, lvl = fit_gaussian(vels, ints)
    cent_wave = (wrange[0] + wrange[1]) / 2.
    ceqw = (amp * sig * np.sqrt(2 * np.pi)) / sol * cent_wave * -1.0
    median = np.median(ints)
    neqw = eqw / (ceqw * median)

    return (ints - median) * neqw + 1.0


def set_instrbroad(wrange, vels, ints, resolution):

    wb = np.average(wrange)
    waves = wb + vels * wb / sol

    convolved_ints = pyasl.instrBroadGaussFast(waves, ints, resolution, edgeHandling="firstlast", fullout=False,
                                               maxsig=None, equid=False)

    return convolved_ints


def _integrate_I_mu(sc, wavelength_range, mode, custom_wavelengths=None, custom_throughput=None):

    """ taken from the exotic_ld Python package for the calculation of Imu_0 """

    if mode == "custom":
        s_wavelengths = custom_wavelengths
        s_throughputs = custom_throughput
    else:
        s_wavelengths, s_throughputs = sc._read_sensitivity_data(mode)

    wavelength_range = np.sort(np.array(wavelength_range))
    s_mask = np.logical_and(wavelength_range[0] < s_wavelengths,
                            s_wavelengths < wavelength_range[1])

    i_mask = np.logical_and(wavelength_range[0] < sc.stellar_wavelengths,
                            sc.stellar_wavelengths < wavelength_range[1])

    s_wvs = s_wavelengths[s_mask]
    s_thp = s_throughputs[s_mask]
    i_wvs = sc.stellar_wavelengths[i_mask]
    i_int = sc.stellar_intensities[i_mask]

    # Ready sensitivity interpolator.
    if s_wvs.shape[0] >= 2:
        s_interp_func = interp1d(s_wvs, s_thp, kind="linear", bounds_error=False, fill_value=0.)
    else:
        mean_wv = np.mean(wavelength_range)
        match_wv_idx = np.argmin(np.abs(s_wavelengths - mean_wv))
        match_s = s_throughputs[match_wv_idx]
        s_interp_func = lambda _sw: np.ones(_sw.shape) * match_s

    # Pre-compute Gauss-legendre roots and rescale to lims.
    roots, weights = roots_legendre(500)
    a = wavelength_range[0]
    b = wavelength_range[1]
    t = (b - a) / 2 * roots + (a + b) / 2

    # Iterate mu values computing intensity.
    I_mu = np.zeros(i_int.shape[1])
    for mu_idx in range(sc.mus.shape[0]):

        # Ready intensity interpolator.
        if i_wvs.shape[0] >= 2:
            i_interp_func = interp1d(
                i_wvs, i_int[:, mu_idx], kind="linear",
                bounds_error=False, fill_value=0.)
        else:
            mean_wv = np.mean(wavelength_range)
            match_wv_idx = np.argmin(np.abs(sc.stellar_wavelengths - mean_wv))
            match_i = sc.stellar_intensities[match_wv_idx, mu_idx]
            i_interp_func = lambda _iw: np.ones(_iw.shape) * match_i

        def integrand(_lambda):
            return s_interp_func(_lambda) * i_interp_func(_lambda)

        # Approximate integral.
        I_mu[mu_idx] = (b - a) / 2. * integrand(t).dot(weights)

    return I_mu[0]


def _get_stellar_model_kd_tree(ld_model, ld_data_version=""):
    """ taken from the exotic_ld Python package for the download required model atmospehere files before
    multiprocessing """

    import pkg_resources
    import pickle

    tree_path = pkg_resources.resource_filename(
        "grid_build.kd_trees", "{}_tree{}.pickle".format(
            ld_model, ld_data_version))

    try:
        with open(tree_path, "rb") as f:
            stellar_kd_tree = pickle.load(f)
    except FileNotFoundError as err:
        raise ValueError("ld_model not recognised.")
    except pickle.UnpicklingError as err:
        raise pickle.UnpicklingError("Failed loading stellar model pickle, "
                                     "check python version is supported.")

    return stellar_kd_tree


def ldcs_rgi_prep(teffps, teffcs, teffhs, loggs, law, mh, model, data_path, mu_min, wrange=None, passband=None,
                  cpu_num=1):
    nop = len(teffps)
    teffs = np.hstack((teffps, teffcs, teffhs))
    loggs_ = np.array(list(loggs) * 3)

    r_M_H = 1.00
    r_Teff = 607.
    r_logg = 1.54
    par_grid = np.vstack(([mh / r_M_H] * len(teffs), teffs / r_Teff, loggs_ / r_logg)).T

    stellar_kd_tree = _get_stellar_model_kd_tree(ld_model=model)
    distance, nearest_idx = stellar_kd_tree.query(par_grid, k=1)
    matches = stellar_kd_tree.data[nearest_idx] * np.array([r_M_H, r_Teff, r_logg])
    umatches = np.unique(matches, axis=0)

    teffr = (int(min(umatches[:, 1])), int(max(umatches[:, 1])))
    loggr = (float(min(umatches[:, 2])), float(max(umatches[:, 2])))
    mhr = np.unique(umatches[:, 0])[0]

    input_args = []
    for order, item in enumerate(umatches):
        input_args += [(item[1], item[2], law, mh, model, data_path, mu_min, wrange, passband, order)]

    if cpu_num > 1:
        pool = multiprocessing.Pool(processes=cpu_num)
        results = pool.map(ldcs_rgi_prep_main, input_args)
        pool.close()
        pool.join()
    else:
        results = []
        for input_arg in input_args:
            results.append(ldcs_rgi_prep_main(input_arg))

    parts = np.array(results, dtype=object)
    srt = np.argsort(parts[:, -1])
    parts = parts[srt]

    ldc1 = np.zeros(len(teffs))
    ldc2 = np.zeros(len(teffs))
    Imu0 = np.zeros(len(teffs))
    for i, x_row in enumerate(matches):
        for j, y_row in enumerate(umatches):
            if np.array_equal(x_row, y_row):
                ldc1[i] = parts[j, 0]
                ldc2[i] = parts[j, 1]
                Imu0[i] = parts[j, 2]
    Imu0 /= np.mean(Imu0)

    rgi = {'phot': [ldc1[:nop], ldc2[:nop], Imu0[:nop]],
           'cool': [ldc1[nop:nop * 2], ldc2[nop:nop * 2], Imu0[nop:nop * 2]],
           'hot': [ldc1[nop * 2:nop * 3], ldc2[nop * 2:nop * 3], Imu0[nop * 2:nop * 3]]}

    return rgi, (teffr, loggr, mhr)


def ldcs_rgi_prep_main(args):
    temp, logg, law, mh, model, data_path, mu_min, wrange, passband, order = args

    if wrange is not None:
        ldc, I0 = ld_coeff_calc(ld_law=law, teff=temp, logg=logg, mh=mh, model=model,
                                data_path=data_path, mu_min=mu_min, wrange=wrange)

    elif passband is not None:
        ldc, I0 = ld_coeff_calc(ld_law=law, teff=temp, logg=logg, mh=mh, model=model,
                                data_path=data_path, mu_min=mu_min, passband=passband)

    if law == 'linear':
        return ldc[0], 0.0, I0, order

    else:
        return ldc[0], ldc[1], I0, order


def ld_coeff_calc(ld_law, teff, logg, mh, model, data_path, mu_min=0.1, wrange=None, passband=None):

    sc = StellarLimbDarkening(Teff=teff, logg=logg, M_H=mh, ld_model=model,  # kurucz mps1 mps2 stagger
                              ld_data_path=data_path)

    if ld_law == 'linear':
        clc = sc.compute_linear_ld_coeffs

    if ld_law == 'square-root':
        clc = sc.compute_squareroot_ld_coeffs

    if ld_law == 'quadratic':
        clc = sc.compute_quadratic_ld_coeffs

    if wrange is not None:
        custom_wavelengths, custom_throughput = np.array([np.min(wrange), np.max(wrange)]), np.array([1, 1])
        wavelength_range = custom_wavelengths.copy()

    elif passband is not None:
        custom_wavelengths, custom_throughput = sc._read_sensitivity_data(passband)
        wavelength_range = [np.min(custom_wavelengths), np.max(custom_wavelengths)]
        custom_wavelengths = np.hstack(([custom_wavelengths[0] - 1], custom_wavelengths, [custom_wavelengths[-1] + 1]))
        custom_throughput = np.hstack(([custom_throughput[0]], custom_throughput, [custom_throughput[-1]]))

    ldc = clc(wavelength_range, mode='custom', custom_wavelengths=custom_wavelengths,
              custom_throughput=custom_throughput, mu_min=mu_min, return_sigmas=False)

    I_mu0 = _integrate_I_mu(sc=sc, wavelength_range=wavelength_range, mode='custom',
                            custom_wavelengths=custom_wavelengths, custom_throughput=custom_throughput)

    return ldc, I_mu0


def generate_spotted_surface(surface_grid, spots_params, default=1e-5):

    lats_spots = spots_params['lats_spots']
    longs_spots = spots_params['longs_spots']
    rs_spots = spots_params['rs_spots']
    cs_cools = spots_params['cs_cools']
    cs_hots = spots_params['cs_hots']

    cmap = np.ones(len(surface_grid['grid_lats'])) * default
    hmap = np.ones(len(surface_grid['grid_lats'])) * default
    for (lat_spot, long_spot, r_spot, c_cool, c_hot) in zip(lats_spots, longs_spots, rs_spots, cs_cools, cs_hots):

        dlon = surface_grid['grid_longs'] - np.deg2rad(long_spot)
        angles = np.arccos(np.sin(surface_grid['grid_lats']) * np.sin(np.deg2rad(lat_spot)) +
                           np.cos(surface_grid['grid_lats']) * np.cos(np.deg2rad(lat_spot)) * np.cos(dlon))

        ii = np.argwhere(angles <= np.deg2rad(r_spot)).T[0]
        ni = len(ii)
        if ni > 0:
            cmap[ii] = c_cool
            hmap[ii] = c_hot

    return cmap, hmap


def td_to_rect_map(lats, longs, ints, fssc, fssp, fssh, xsize, ysize):

    xlats = np.linspace(np.pi / 2., -np.pi / 2., ysize)
    xlongs = np.linspace(0, 2.0 * np.pi, xsize)

    mlongs, mlats = np.meshgrid(xlongs, xlats)

    points = np.array([lats, longs]).T
    grid_points = (mlats, mlongs)

    rmap_int = griddata(points, ints, grid_points, method='nearest')
    rmap_c = griddata(points, fssc, grid_points, method='nearest')
    rmap_p = griddata(points, fssp, grid_points, method='nearest')
    rmap_h = griddata(points, fssh, grid_points, method='nearest')

    xs = (max(xlongs) - min(xlongs)) / (len(xlongs) - 1) / 2.
    ys = (max(xlats) - min(xlats)) / (len(xlats) - 1) / 2.

    extent = [min(xlongs) - xs, max(xlongs) + xs, min(xlats) - ys, max(xlats) + ys]

    return xlongs, xlats, rmap_int, rmap_c, rmap_p, rmap_h, extent, mlats, mlongs


def hp_to_rect_map(ints, fssc, fssp, fssh, nside, xsize=180, ysize=90):

    xlats = np.linspace(0.0, np.pi, ysize)
    xlongs = np.linspace(0.0, 2 * np.pi, xsize)

    mlongs, mlats = np.meshgrid(xlongs, xlats)
    grid_pix = hp.ang2pix(nside, mlats, mlongs, nest=False)

    rmap_int = ints[grid_pix]
    rmap_c = fssc[grid_pix]
    rmap_p = fssp[grid_pix]
    rmap_h = fssh[grid_pix]

    xlats = xlats[::-1] - np.pi / 2.
    xs = (max(xlongs) - min(xlongs)) / (len(xlongs) - 1) / 2.
    ys = (max(xlats) - min(xlats)) / (len(xlats) - 1) / 2.

    extent = [min(xlongs) - xs, max(xlongs) + xs, min(xlats) - ys, max(xlats) + ys]

    return xlongs, xlats, rmap_int, rmap_c, rmap_p, rmap_h, extent, np.flipud(mlats - np.pi / 2.), mlongs


def p2m_to_rect_map(lats, longs, ints, fssc, fssp, fssh, xsize, ysize):

    xlats = np.linspace(np.pi / 2., -np.pi / 2., ysize)
    xlongs = np.linspace(0, 2 * np.pi, xsize)

    mlongs, mlats = np.meshgrid(xlongs, xlats)

    points = np.array([lats, longs]).T
    grid_points = (mlats, mlongs)

    rmap_int = griddata(points, ints, grid_points, method='nearest')
    rmap_c = griddata(points, fssc, grid_points, method='nearest')
    rmap_p = griddata(points, fssp, grid_points, method='nearest')
    rmap_h = griddata(points, fssh, grid_points, method='nearest')

    xs = (max(xlongs) - min(xlongs)) / (len(xlongs) - 1) / 2.
    ys = (max(xlats) - min(xlats)) / (len(xlats) - 1) / 2.

    extent = [min(xlongs) - xs, max(xlongs) + xs, min(xlats) - ys, max(xlats) + ys]

    return xlongs, xlats, rmap_int, rmap_c, rmap_p, rmap_h, extent, mlats, mlongs


def grid_to_rect_map(surface_grid, ints, fssc, fssp, fssh):

    if surface_grid['method'] == 'trapezoid':
        nlons = surface_grid['nlons'].copy()
        outputs = td_to_rect_map(lats=surface_grid['grid_lats'], longs=surface_grid['grid_longs'], ints=ints,
                                 fssc=fssc, fssp=fssp, fssh=fssh, xsize=max(nlons), ysize=surface_grid['nlats'])

    elif surface_grid['method'] == 'healpy':
        outputs = hp_to_rect_map(ints=ints, fssc=fssc, fssp=fssp, fssh=fssh, nside=surface_grid['nside'],
                                 xsize=180, ysize=90)

    elif surface_grid['method'] == 'phoebe2_marching':
        outputs = p2m_to_rect_map(lats=surface_grid['grid_lats'], longs=surface_grid['grid_longs'], ints=ints,
                                  fssc=fssc, fssp=fssp, fssh=fssh, xsize=180, ysize=90)

    xlongs, xlats, rmap_int, rmap_c, rmap_p, rmap_h, extent, mlats, mlongs = outputs
    mapprojs = {'xlongs': xlongs, 'xlats': xlats, 'rmap_int': rmap_int, 'rmap_c': rmap_c, 'rmap_p': rmap_p,
                'rmap_h': rmap_h, 'extent': extent, 'mlats': mlats, 'mlongs': mlongs}

    return mapprojs


def test_prf_plot(DIP, mode, plotp):

    times = DIP.idc[mode]['times']
    vels = DIP.idc[mode]['vels']
    data = DIP.idc[mode]['data']
    fake_slps = DIP.opt_results[mode]['fake_sprfs']
    spotless_slps = DIP.opt_results[mode]['spotless_sprfs']
    recons_slps = DIP.opt_results[mode]['recons_sprfs']
    epochs = (times - DIP.params['t0']) / DIP.params['period']

    plt.figure(figsize=figsize)
    ax3 = plt.subplot2grid((20, 2), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((20, 2), (1, 0), rowspan=19)
    ax2 = plt.subplot2grid((20, 2), (1, 1), rowspan=19)

    ax3.axis("off")

    if mode == 'line':
        plotp['sep_prf'] = plotp['line_sep_prf']
        plotp['sep_res'] = plotp['line_sep_res']

    if mode == 'mol':
        plotp['sep_prf'] = plotp['mol_sep_prf']
        plotp['sep_res'] = plotp['mol_sep_res']

    for i, itime in enumerate(sorted(times)):

        maxv = max(vels)
        maxi = max(data[itime]['prf']) + i * plotp['sep_prf']
        residual = data[itime]['prf'] - recons_slps[itime]['prf']
        maxir = np.average(residual + i * plotp['sep_res'])

        if plotp['show_err_bars']:
            ax1.errorbar(vels, data[itime]['prf'] + i * plotp['sep_prf'],
                         yerr=data[itime]['errs'], fmt='o', color='k', ms=plotp['markersize'])
            ax2.errorbar(vels, residual + i * plotp['sep_res'],
                         yerr=data[itime]['errs'], fmt='o', color='k', ms=plotp['markersize'])
        else:
            ax1.plot(vels, data[itime]['prf'] + i * plotp['sep_prf'], 'ko', ms=plotp['markersize'])
            ax2.plot(vels, residual + i * plotp['sep_res'], 'ko', ms=plotp['markersize'])

        ax1.plot(vels, fake_slps[itime]['prf'] + i * plotp['sep_prf'], 'g', linewidth=plotp['linewidth'], zorder=2)
        ax1.plot(vels, spotless_slps[itime]['prf'] + i * plotp['sep_prf'], 'b', linewidth=plotp['linewidth'], zorder=2)
        ax1.plot(vels, recons_slps[itime]['prf'] + i * plotp['sep_prf'], 'r', linewidth=plotp['linewidth'], zorder=3)
        ax1.annotate(str('%0.3f' % round(epochs[i], 3)), xy=(maxv - maxv / 3.1, maxi + plotp['sep_prf'] / 10.),
                     color='g')
        ax2.annotate(str('%0.3f' % round(epochs[i], 3)), xy=(maxv - maxv / 10., maxir + plotp['sep_res'] / 10.),
                     color='g')
        ax2.axhline(i * plotp['sep_res'], color='r', zorder=3)

    ax1.set_xlabel('Radial Velocity (km / s)', fontsize=plotp['fontsize'])
    ax1.set_ylabel('I / Ic', fontsize=plotp['fontsize'])

    ax3.plot([], [], 'ko', label='Obs. Data', ms=plotp['markersize'])
    ax3.plot([], [], 'b', label='Spotless Model', linewidth=plotp['linewidth'])
    ax3.plot([], [], 'r', label='Spotted Model', linewidth=plotp['linewidth'])
    ax3.legend(loc='center', ncol=3, frameon=False)

    ax2.set_xlabel('Radial Velocity (km / s)', fontsize=plotp['fontsize'])
    ax2.set_ylabel('Residuals', fontsize=plotp['fontsize'])

    ax1.tick_params(axis='both', labelsize=plotp['ticklabelsize'])
    ax2.tick_params(axis='both', labelsize=plotp['ticklabelsize'])

    plt.tight_layout()


def test_lc_plot(DIP, plotp):

    recons_slc = DIP.opt_results['lc']['recons_slc']

    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    if 'lc' in DIP.conf:

        times = DIP.idc['lc']['times']
        fluxs = DIP.idc['lc']['data']['fluxs']
        errs = DIP.idc['lc']['data']['errs']

        epochs = (times - DIP.params['t0']) / DIP.params['period']

        ax1.errorbar(epochs, fluxs, yerr=errs, fmt='o', color='k', label='Obs. Light Curve', ms=plotp['markersize'],
                     zorder=1)
        ax1.plot(epochs, recons_slc, 'r', label='Spotted Model', linewidth=plotp['linewidth'], zorder=2)
        ax2.errorbar(epochs, fluxs - recons_slc, yerr=errs, fmt='o', color='k', ms=plotp['markersize'], zorder=1)
    else:

        ntimes = DIP.opt_results['lc']['ntimes']
        nepochs = (ntimes - DIP.params['t0']) / DIP.params['period']

        ax1.plot(nepochs, recons_slc, 'ko', label='Synthetic Light Curve', ms=plotp['markersize'])
        ax2.set_xlim([0, 2])
        ax2.set_ylim([-1, 1])

    ax1.set_xlabel('Epoch', fontsize=plotp['fontsize'])
    ax1.set_ylabel('Normalized Flux', fontsize=plotp['fontsize'])
    ax1.tick_params(axis='both', labelsize=plotp['ticklabelsize'])

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)

    ax2.axhline(0.0, color='r', linewidth=plotp['linewidth'])

    ax2.set_xlabel('Epoch', fontsize=plotp['fontsize'])
    ax2.set_ylabel('Residuals', fontsize=plotp['fontsize'])
    ax2.tick_params(axis='both', labelsize=plotp['ticklabelsize'])

    plt.tight_layout()


def test_map_plot(DIP, fake_fssc, fake_fssh, fake_total_fs, recons_total_fs, plotp):

    Tcool = DIP.params['Tcool']
    Tphot = DIP.params['Tphot']
    Thot = DIP.params['Thot']
    surface_grid = DIP.surface_grid
    recons_fssc = DIP.opt_results['recons_fssc']
    recons_fssh = DIP.opt_results['recons_fssh']

    fake_ints = (fake_fssc * DIP.params['Tcool'] ** 4 + fake_fssh * DIP.params['Thot'] ** 4 +
                (1.0 - (fake_fssc + fake_fssh)) * DIP.params['Tphot'] ** 4) / DIP.params['Tphot'] ** 4

    recons_ints = (recons_fssc * DIP.params['Tcool'] ** 4 + recons_fssh * DIP.params['Thot'] ** 4 +
                  (1.0 - (recons_fssc + recons_fssh)) * DIP.params['Tphot'] ** 4) / DIP.params['Tphot'] ** 4

    fmapprojs = grid_to_rect_map(surface_grid=DIP.surface_grid, ints=fake_ints, fssc=fake_fssc,
                                 fssp=1.0 - (fake_fssc + fake_fssh), fssh=fake_fssh)
    cmapprojs = grid_to_rect_map(surface_grid=DIP.surface_grid, ints=recons_ints, fssc=recons_fssc,
                                 fssp=1.0 - (recons_fssc + recons_fssh), fssh=recons_fssh)

    frmap, fextent = fmapprojs['rmap_int'], fmapprojs['extent']
    fextent = np.rad2deg(fextent)
    crmap, cextent = cmapprojs['rmap_int'], cmapprojs['extent']
    cextent = np.rad2deg(cextent)

    vmin = (Tcool / Tphot) ** 4
    vmax = (Thot / Tphot) ** 4

    tsa_title1 = 'Total Spotted Area (%) = ' + str('%0.3f' % round((fake_total_fs[0] + fake_total_fs[1]) * 100, 3))
    tsa_title2 = 'Total Spotted Area (%) = ' + str('%0.3f' % round((recons_total_fs[0] + recons_total_fs[1]) * 100, 3))

    if plotp['map_projection'] == 'rectangular':
        fig = plt.figure(figsize=figsize)
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0))

        img2 = ax1.imshow(frmap, cmap=plotp['cmap'], aspect='auto', extent=fextent, interpolation='bicubic',
                          vmin=vmin, vmax=vmax)
        img3 = ax2.imshow(crmap, cmap=plotp['cmap'], aspect='auto', extent=cextent, interpolation='bicubic',
                          vmin=vmin, vmax=vmax)

        ax1.text(0.5, 1.10, "Artificial Map", transform=ax1.transAxes, ha='center', va='bottom',
                 fontsize=plotp['fontsize'], fontweight='bold')
        ax1.text(0.5, 1.00, tsa_title1, transform=ax1.transAxes, ha='center', va='bottom',
                 fontsize=plotp['fontsize'] - 5)

        ax2.text(0.5, 1.10, "Reconstructed Map", transform=ax2.transAxes, ha='center', va='bottom',
                 fontsize=plotp['fontsize'], fontweight='bold')
        ax2.text(0.5, 1.00, tsa_title2, transform=ax2.transAxes, ha='center', va='bottom',
                 fontsize=plotp['fontsize'] - 5)

        divider2 = make_axes_locatable(ax1)
        cax2 = divider2.append_axes('right', size='5%', pad='5%')
        clb2 = fig.colorbar(img2, cax=cax2)
        clb2.set_label(r'$\frac{I}{I_{phot}}$', fontsize=plotp['fontsize'])

        divider3 = make_axes_locatable(ax2)
        cax3 = divider3.append_axes('right', size='5%', pad='5%')
        clb3 = fig.colorbar(img3, cax=cax3)
        clb3.set_label(r'$\frac{I}{I_{phot}}$', fontsize=plotp['fontsize'])

        ax1.set_xticks(np.arange(0, 420, 60))
        ax1.set_yticks(np.arange(-90, 120, 30))
        ax1.set_xlabel(r'Longitude ($^\circ$)', fontsize=plotp['fontsize'])
        ax1.set_ylabel(r'Latitude ($^\circ$)', fontsize=plotp['fontsize'])
        ax1.tick_params(axis='both', labelsize=plotp['ticklabelsize'])

        ax2.set_xticks(np.arange(0, 420, 60))
        ax2.set_yticks(np.arange(-90, 120, 30))
        ax2.set_xlabel(r'Longitude ($^\circ$)', fontsize=plotp['fontsize'])
        ax2.set_ylabel(r'Latitude ($^\circ$)', fontsize=plotp['fontsize'])
        ax2.tick_params(axis='both', labelsize=plotp['ticklabelsize'])

        colors = {'line': 'r', 'mol1': 'g', 'mol2': 'purple', 'lc': 'b'}
        zorders = {'line': 4, 'mol1': 3, 'mol2': 2, 'lc': 1}
        for i, mode in enumerate(reversed(DIP.conf)):
            epochs = (DIP.idc[mode]['times'] - DIP.params['t0']) / DIP.params['period']
            phases = epochs - np.floor(epochs)

            ax1.plot([360 * (1.0 - phases), 360 * (1.0 - phases)], [-85, -75], '-', color=colors[mode],
                     linewidth=2, zorder=zorders[mode])
            ax2.plot([360 * (1.0 - phases), 360 * (1.0 - phases)], [-85, -75], '-', color=colors[mode],
                     linewidth=2, zorder=zorders[mode])

            if 0.0 in phases:
                ax1.plot([0, 0], [-85, -75], '-', color=colors[mode], linewidth=2, zorder=zorders[mode])
                ax2.plot([0, 0], [-85, -75], '-', color=colors[mode], linewidth=2, zorder=zorders[mode])

        ax1.fill_between(x=[fextent[0], fextent[1]], y1=-DIP.params['incl'], y2=fextent[2], color=plotp['fill_color'],
                         alpha=0.15)
        ax2.fill_between(x=[cextent[0], cextent[1]], y1=-DIP.params['incl'], y2=cextent[2], color=plotp['fill_color'],
                         alpha=0.15)

        ax1.grid()
        ax2.grid()

        plt.tight_layout()

    elif plotp['map_projection'] == 'mollweide':

        xlats = DIP.mapprojs['xlats']
        xlongs = DIP.mapprojs['xlongs']

        fig = plt.figure(figsize=figsize)
        ax1 = plt.subplot2grid((2, 1), (0, 0), projection="mollweide")
        ax2 = plt.subplot2grid((2, 1), (1, 0), projection="mollweide")

        img1 = ax1.pcolormesh(xlongs - np.pi, xlats, frmap, cmap=plotp['cmap'], shading='gouraud', vmin=vmin,
                              vmax=vmax)
        img2 = ax2.pcolormesh(xlongs - np.pi, xlats, crmap, cmap=plotp['cmap'], shading='gouraud', vmin=vmin,
                              vmax=vmax)

        ax1.text(0.5, 1.20, "Artificial Map", transform=ax1.transAxes, ha='center', va='bottom',
                 fontsize=plotp['fontsize'], fontweight='bold')
        ax1.text(0.5, 1.10, tsa_title1, transform=ax1.transAxes, ha='center', va='bottom',
                 fontsize=plotp['fontsize'] - 5)

        ax2.text(0.5, 1.20, "Reconstructed Map", transform=ax2.transAxes, ha='center', va='bottom',
                 fontsize=plotp['fontsize'], fontweight='bold')
        ax2.text(0.5, 1.10, tsa_title2, transform=ax2.transAxes, ha='center', va='bottom',
                 fontsize=plotp['fontsize'] - 5)

        cb1 = fig.colorbar(img1, ax=ax1, location='right')
        cb1.set_label(r'$\frac{I}{I_{phot}}$', fontsize=plotp['fontsize'] + 5)
        cb1.ax.tick_params(labelsize=plotp['ticklabelsize'])
        cb1.set_ticks((0.4, 0.6, 0.8, 1.))

        cb2 = fig.colorbar(img2, ax=ax2, location='right')
        cb2.set_label(r'$\frac{I}{I_{phot}}$', fontsize=plotp['fontsize'] + 5)
        cb2.ax.tick_params(labelsize=plotp['ticklabelsize'])
        cb2.set_ticks((0.4, 0.6, 0.8, 1.))

        ax1.set_xticks(np.deg2rad(np.arange(-120, 180, 60)))
        ax1.set_yticks(np.deg2rad(np.arange(-90, 120, 30)))
        ax1.set_xlabel(r'Longitude ($^\circ$)', fontsize=plotp['fontsize'], labelpad=15)
        ax1.set_ylabel(r'Latitude ($^\circ$)', fontsize=plotp['fontsize'])
        xtick_labels = np.arange(60, 360, 60)
        ax1.set_xticklabels(xtick_labels, zorder=15)
        ax1.xaxis.set_label_coords(0.5, -0.100)
        ax1.tick_params(axis='both', labelsize=plotp['ticklabelsize'])

        ax2.set_xticks(np.deg2rad(np.arange(-120, 180, 60)))
        ax2.set_yticks(np.deg2rad(np.arange(-90, 120, 30)))
        ax2.set_xlabel(r'Longitude ($^\circ$)', fontsize=plotp['fontsize'], labelpad=15)
        ax2.set_ylabel(r'Latitude ($^\circ$)', fontsize=plotp['fontsize'])
        xtick_labels = np.arange(60, 360, 60)
        ax2.set_xticklabels(xtick_labels, zorder=15)
        ax2.xaxis.set_label_coords(0.5, -0.100)
        ax2.tick_params(axis='both', labelsize=plotp['ticklabelsize'])

        colors = ['b', 'g']
        ranges = [[-35, -30], [-40, -35]]
        for i, mode in enumerate(DIP.conf):
            if mode in ['line', 'lc']:
                epochs = (DIP.idc[mode]['times'] - DIP.params['t0']) / DIP.params['period']
                phases = epochs - np.floor(epochs)
                obslong = 360 * (1.0 - phases) - 180

                ax1.plot(np.deg2rad([obslong, obslong]), np.deg2rad(ranges[i]), '-', color=colors[i], linewidth=1)
                ax2.plot(np.deg2rad([obslong, obslong]), np.deg2rad(ranges[i]), '-', color=colors[i], linewidth=1)
                if 0.0 in phases:
                    ax1.plot(np.deg2rad([0.0, 0.0]), np.deg2rad(ranges[i]), '-', color=colors[i], linewidth=1)
                    ax2.plot(np.deg2rad([0.0, 0.0]), np.deg2rad(ranges[i]), '-', color=colors[i], linewidth=1)

        ax1.fill_between(x=[-np.pi, np.pi], y1=np.deg2rad(-DIP.params['incl']), y2=-np.pi / 2., color=plotp['fill_color'],
                         alpha=0.3)
        ax2.fill_between(x=[-np.pi, np.pi], y1=np.deg2rad(-DIP.params['incl']), y2=-np.pi / 2., color=plotp['fill_color'],
                         alpha=0.3)

        ax1.grid(True)
        ax2.grid(True)

        plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05)


def lmbds_plot(DIP, plotp):

    lmbds = DIP.opt_results['lmbds']
    twchisqs = DIP.opt_results['twchisqs']
    entropies = DIP.opt_results['entropies']
    maxcurve = DIP.opt_results['maxcurve']

    fig, ax1 = plt.subplots()

    lmbd = lmbds[maxcurve]

    ax1.plot(twchisqs, entropies, 'ko', ms=2)
    ax1.plot(twchisqs[maxcurve], entropies[maxcurve], 'ro', ms=3, label='Best lmbd = ' + str(lmbd))
    ax1.set_xlabel(r'$\chi^2$', fontsize=plotp['fontsize'])
    ax1.set_ylabel('Entropy', fontsize=plotp['fontsize'])
    ax1.tick_params(axis='both', labelsize=plotp['ticklabelsize'])
    ax1.legend()

    plt.tight_layout()


def make_grid_contours(stats_grid, minv):

    keys = list(stats_grid.keys())
    for key in keys:
        if key not in ['stats', 'fit_params']:
            if len(np.unique(stats_grid[key])) == 1:
                del stats_grid[key]
                del stats_grid['fit_params'][key]

    # Deduplicate after removing constant parameters
    remaining_keys = [k for k in stats_grid if k not in ['stats', 'fit_params']]
    if remaining_keys:
        stacked = np.column_stack([stats_grid[k] for k in remaining_keys])
        _, unique_idx = np.unique(stacked, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        for key in remaining_keys:
            stats_grid[key] = stats_grid[key][unique_idx]
        stats_grid['stats'] = stats_grid['stats'][unique_idx]

    names = list(stats_grid.keys())
    names.remove('stats')
    names.remove('fit_params')

    if minv == "chi":
        indicator = stats_grid['stats'][:, 0]
        label = r'$\chi^2$'
    elif minv == "loss":
        indicator = stats_grid['stats'][:, 1]
        label = r'$\chi^2$ + $\lambda$ * $MEM$'

    mn = np.argmin(indicator)

    print('\n' + '\033[96m' + '*** Grid Search Results ***' + '\033[0m')
    if len(names) == 1:

        mpar = stats_grid[names[0]][mn]

        print(names[0], ':', mpar)

        plt.plot(stats_grid[names[0]], indicator, 'ko')
        plt.xlabel(r"$\nu sini$ (km/s)" if names[0] == 'vsini' else ('EW' if names[0] == 'eqw' else names[0]),
                   fontsize=15)
        plt.ylabel(label, fontsize=15)
        plt.show()

        return

    combin = np.array(list(combinations(names, 2)))
    combin[len(names) - 2] = combin[len(names) - 2][::-1]

    fit_params = stats_grid['fit_params']
    for pair in combin:

        par1_name = pair[0]
        par2_name = pair[1]

        index = []
        for rpar in stats_grid:
            if rpar not in [par1_name, par2_name, 'stats', 'fit_params']:
                index.append(np.argwhere(stats_grid[rpar] == stats_grid[rpar][mn]).flatten())

        if len(index) > 0:
            inters = reduce(np.intersect1d, index)
            chi = np.array(indicator)[inters]
            par1 = stats_grid[par1_name][inters]
            par2 = stats_grid[par2_name][inters]
        else:
            chi = np.array(indicator)
            par1 = stats_grid[par1_name]
            par2 = stats_grid[par2_name]

        contour = np.zeros((len(fit_params[par2_name]), len(fit_params[par1_name])))
        for i, ei in enumerate(fit_params[par2_name]):
            vss = np.argsort(par1[par2 == ei])
            contour[i] = chi[par2 == ei][vss]

        mn = np.argmin(indicator)
        mpar1 = stats_grid[par1_name][mn]
        mpar2 = stats_grid[par2_name][mn]

        fig, ax = plt.subplots()
        cont = ax.contourf(fit_params[par1_name], fit_params[par2_name], contour, 50, cmap='RdYlBu')
        levels = [1 + indicator[mn]]
        uc = plt.contour(cont, levels=levels)
        ax.clabel(uc, inline=False, use_clabeltext=True, fmt=lambda x: "min + 1 = " + str('%0.3f' % x), colors='k',
                  fontsize=10)

        chipone = np.vstack(uc.allsegs[0])

        if len(chipone) != 0:

            if (
                    max(chipone[:, 0]) == max(stats_grid[par1_name]) or
                    min(chipone[:, 0]) == min(stats_grid[par1_name]) or
                    max(chipone[:, 1]) == max(stats_grid[par2_name]) or
                    min(chipone[:, 1]) == min(stats_grid[par2_name])
            ):
                print(
                    '\033[91m' + "Warning: Unable to fully estimate parameter uncertainties because the region defined "
                                 "by (minimum Chi-square + 1) is only partially covered by the computed Chi-square"
                                 " distribution. As a result, for some parameters, the uncertainty range could not be"
                                 " properly determined and is limited by the edges of the explored parameter space. "
                                 "Consider expanding the parameter ranges to properly estimate parameter uncertainties."
                    + '\033[0m')

            print('\033[93m' + '{:<25}'.format(par1_name) + ':' + '\033[0m',
                  '\033[1m' + str(mpar1) + '\033[0m', '\033[1m' + '-' + str(mpar1 - min(chipone[:, 0])) + '\033[0m',
                  '\033[1m' + '+' + str(max(chipone[:, 0]) - mpar1) + '\033[0m')
            print('\033[93m' + '{:<25}'.format(par2_name) + ':' + '\033[0m',
                  '\033[1m' + str(mpar2) + '\033[0m', '\033[1m' + '-' + str(mpar2 - min(chipone[:, 1])) + '\033[0m',
                  '\033[1m' + '+' + str(max(chipone[:, 1]) - mpar2) + '\033[0m')
        else:
            print('\033[91m' + "Warning: Unable to estimate parameter uncertainties because the value of"
                               " (minimum Chi-square + 1) lies outside the computed Chi-square distribution. This"
                               " suggests that the explored parameter ranges may be too narrow. Consider expanding"
                               " the parameter space to include a wider region around the minimum." + '\033[0m')
            print('\033[93m' + '{:<25}'.format(par1_name) + ':' + '\033[0m', '\033[1m' + str(mpar1) + '\033[0m')
            print('\033[93m' + '{:<25}'.format(par2_name) + ':' + '\033[0m', '\033[1m' + str(mpar2) + '\033[0m')

        ax.set_xlabel(r"$\nu sini$ (km/s)" if par1_name == 'vsini' else ('EW' if par1_name == 'eqw' else par1_name),
                   fontsize=15)
        ax.set_ylabel(r"$\nu sini$ (km/s)" if par2_name == 'vsini' else ('EW' if par2_name == 'eqw' else par2_name),
                   fontsize=15)
        ax.tick_params(labelsize=15)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(cont, cax=cax, orientation='horizontal', format="%0.3f",
                            ticks=np.linspace(np.min(contour), np.max(contour), 5))
        cbar.ax.set_xlim((np.min(contour), np.max(contour)))
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(label, fontsize=15)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.show()


def grid_test(xyzs, scalars, win_title, bar_title, show=True):

    def plot(dx, dy, dz, dtriangles, dtdmap, plotter):

        faces = np.hstack(
            [np.column_stack((np.full((len(dtriangles), 1), 3, dtype=np.int64), dtriangles)).ravel()]
        )
        points = np.column_stack((dx, dy, dz))
        mesh = pv.PolyData(points, faces)
        mesh["scalars"] = dtdmap  # vertex scalars

        plotter.add_mesh(
            mesh,
            scalars="scalars",
            cmap="gist_heat",
            # clim=[self.vmin, self.vmax],
            show_edges=False,  # kenar görmek istersen True + edge_color="black"
            scalar_bar_args=dict(
                title=bar_title,
                n_labels=6,
                fmt="%.4f",
                vertical=False,
                title_font_size=15,
                label_font_size=15,
                width=0.60,  # bar genişliği (ekranın %60’ı)
                height=0.08,  # bar yüksekliği
                position_x=0.20,  # (1 - width)/2 = (1 - 0.60)/2 = 0.20
                position_y=0.02,  # alta yakın
            )
        )

    x = []
    y = []
    z = []
    triangles = []
    nscalars = []
    for i, xyz in enumerate(xyzs):
        nscalars.append([scalars[i]] * 3)
        for row in xyz:
            x.append(row[0])
            y.append(row[1])
            z.append(row[2])
        triangles.append((0 + i * 3, 1 + i * 3, 2 + i * 3))
    nscalars = np.hstack(nscalars)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    tdmap = np.hstack(nscalars).astype(float)
    triangles = np.asarray(triangles, dtype=np.int64)  # 0-based

    plotter = pv.Plotter()
    plot(x, y, z, triangles, tdmap, plotter)

    if show:
        plotter.add_title(win_title, font_size=20)
        plotter.show()


def draw_3D_surf(params, surface_grid, spots_params):

    fssc, fssh = generate_spotted_surface(surface_grid, spots_params)

    Thot = params['Thot']
    Tphot = params['Tphot']
    Tcool = params['Tcool']
    xyzs = surface_grid['grid_xyzs']

    brightness = (fssc * Tcool ** 4 + fssh * Thot ** 4 + (1.0 - (fssc + fssh)) * Tphot ** 4) / Tphot ** 4

    if surface_grid['method'] != 'phoebe2_marching':
        scalars = np.repeat(brightness, 2)
    else:
        scalars = brightness

    grid_test(xyzs=xyzs, scalars=scalars, title='Spotted Star')


_shared = {}

def _init_worker(shared_arrays, mode):
    _shared['arrays'] = shared_arrays
    _shared['mode'] = mode

def _worker_spec(per_timestep_args):
    s = _shared['arrays']
    itime, icount, info = per_timestep_args
    return cutils.calc_pixel_coeffs_spec((
        itime, s['plats'], s['vlats'],
        s['ldcs_phot'], s['ldcs_cool'], s['ldcs_hot'],
        s['lis_phot'], s['lis_cool'], s['lis_hot'],
        s['areas'], s['lats'], s['longs'],
        s['nop'], s['t0'], s['incl'], s['ld_law'], s['noes'],
        s['lp_vels'], s['phot_lp'], s['cool_lp'], s['hot_lp'],
        s['vrt'], s['vels'], s['period'], icount, info
    ))

def _worker_lc(per_timestep_args):
    s = _shared['arrays']
    itime, icount, info = per_timestep_args
    return cutils.calc_pixel_coeffs_lc((
        itime, s['plats'], s['ldcs_phot'], s['ldcs_cool'], s['ldcs_hot'],
        s['lis_phot'], s['lis_cool'], s['lis_hot'],
        s['areas'], s['lats'], s['longs'],
        s['t0'], s['incl'], s['ld_law'], s['noes'],
        s['period'], icount, info
    ))

def mp_calc_pixel_coeffs(cpu_num, shared_arrays, per_timestep_args, mode):
    func = _worker_spec if mode != 'lc' else _worker_lc

    if cpu_num > 1:
        with multiprocessing.Pool(
            processes=cpu_num,
            initializer=_init_worker,
            initargs=(shared_arrays, mode),
        ) as pool:
            return pool.map(func, per_timestep_args)
    else:
        _init_worker(shared_arrays, mode)
        return [func(item) for item in per_timestep_args]


def mp_search(cpu_num, func, input_args, backend):

    results = []
    if backend == "cpu":
        pool = multiprocessing.Pool(processes=cpu_num)
        for result in tqdm.tqdm(pool.imap(func=func, iterable=input_args), total=len(input_args)):
            results.append(result)

    elif backend == "cuda":
        for fpmg in tqdm.tqdm(input_args, total=len(input_args)):
            results.append(func(fpmg))
            jax.clear_caches()

    return results

def generate_noisy_normalized_profile(model, snr=100, read_noise=5.0):
    """
    model: normalize tayf (continuum ~ 1)
    snr: continuum'daki hedef SNR
    read_noise: elektron cinsinden readout noise
    """

    continuum_level = snr**2

    counts_model = model * continuum_level

    noisy_counts = np.random.poisson(counts_model).astype(np.float64)

    noisy_counts += np.random.normal(0, read_noise, size=len(model))
    noisy_flux = noisy_counts / continuum_level

    noisy_counts_clipped = np.clip(noisy_counts, 0, None)
    sigma_counts = np.sqrt(noisy_counts_clipped + read_noise ** 2)
    flux_error = sigma_counts / continuum_level

    return noisy_flux, flux_error


def lin_interp_with_unc(x_data, y_data, xq):
    """
    Parça-parça doğrusal interpolasyon.
    # - Eğer xq == x_data: doğrudan aynı y_data (ve hatası).
    - Eğer xq < min(x_data) veya xq > max(x_data): en yakın uçtaki y_data.
    - Aksi halde doğrusal interpolasyon (belirsizlikle).
    """
    x_data = np.asarray(x_data)
    xq = np.atleast_1d(xq)  # vektörel çalışsın

    results = []
    for x in xq:
        # # Eğer tam çakışma varsa
        # if np.any(np.isclose(x, x_data)):
        #     i = np.where(np.isclose(x, x_data))[0][0]
        #     results.append(y_data[i])
        #     continue

        # Eğer dışa taştıysa
        if x <= x_data[0]:
            results.append(y_data[0])
            continue
        if x >= x_data[-1]:
            results.append(y_data[-1])
            continue

        # Normal interpolasyon
        idx = np.searchsorted(x_data, x, side="right") - 1
        x0, x1 = x_data[idx], x_data[idx+1]
        y0, y1 = y_data[idx], y_data[idx+1]

        t = (x - x0) / (x1 - x0)
        results.append(y0 + t * (y1 - y0))

    return np.array(results, dtype=object)


def isfloatable(x):

    if x is None or isinstance(x, (bool, np.bool_)):
        return False

    try:
        float(x)
        return True

    except ValueError:
        return False
