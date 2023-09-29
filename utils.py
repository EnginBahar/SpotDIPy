import pickle

import numpy as np
from astropy.coordinates import spherical_to_cartesian as ast_stc
from astropy.coordinates import cartesian_to_spherical as ast_cts
from astropy import units as au
from astropy import constants as ac
from scipy.optimize import curve_fit
from PyAstronomy import pyasl
from exotic_ld import StellarLimbDarkening
import multiprocessing
import healpy as hp
from scipy.optimize import newton as sc_newton
from functools import reduce
import matplotlib.pyplot as plt
import PyDynamic as PyD
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve as scipy_convolve
import autograd.numpy as anp
from itertools import combinations


sol = ac.c.to(au.kilometer / au.second).value  # km/s


def p2m_surface_grid(requiv, noes, t0, period, mass):

    import phoebe
    from phoebe import u as phoebe_units

    """ Source: Phoebe2 """

    """ Basıklaşmaktan kaynaklı yüzey elemanlarının hız değeri değişiyor olabilir.
    Buna bakılmalı!!!"""

    b = phoebe.default_star()

    b.set_value('incl', value=90)
    b.set_value('period', value=period * phoebe_units.day)
    b.set_value('t0', value=t0 * phoebe_units.day)
    b.set_value('requiv', value=requiv * phoebe_units.solRad)
    b.set_value('mass', value=mass * phoebe_units.solMass)

    b['ntriangles'] = noes

    b.add_dataset('mesh', compute_times=t0, coordinates=['uvw'],
                  columns=['us', 'vs', 'ws', 'areas', 'loggs', 'rs', 'volume'])

    b.run_compute(distortion_method='rotstar', irrad_method='none')

    grid_xyzs = b['mesh@uvw_elements@latest@model'].value.copy()
    grid_xcs = b['mesh@us@latest@model'].value.copy()
    grid_ycs = b['mesh@vs@latest@model'].value.copy()
    grid_zcs = b['mesh@ws@latest@model'].value.copy()
    grid_areas = b['mesh@areas@latest@model'].value.copy()
    grid_loggs = b['mesh@loggs@latest@model'].value.copy()

    rs, lats, longs = ast_cts(grid_xcs, grid_ycs, grid_zcs)

    grid_rs = rs.copy()
    grid_lats = lats.value.copy()
    grid_longs = longs.value.copy()

    surface_grid = {'method': 'phoebe2_marching', 'grid_xyzs': grid_xyzs, 'grid_areas': grid_areas, 'grid_loggs': grid_loggs,
                    'grid_lats': grid_lats, 'grid_longs': grid_longs, 'grid_rs': grid_rs}

    return surface_grid


def dg_surface_grid(omega, nlats, radius, mass):

    nlons = np.zeros(nlats, dtype=int)
    dlat = np.pi / float(nlats)
    nmax = nlats * 2

    npix = np.zeros(nmax)
    isbm = 0
    for iblk in range(1, nmax + 1):
        nblk = int(nmax / iblk)
        if nmax == nblk * iblk:
            npix[isbm] = nblk
            isbm = isbm + 1

    nsbm = isbm

    nadp = 0
    coshi = 1.0
    for ilat in range(nlats):
        coslo = np.cos(dlat * (ilat + 1))
        band = (coshi - coslo) * 2. * np.pi
        target = band / dlat / dlat

        for isbm in range(nsbm):
            if npix[isbm] > target:
                nlons[ilat] = npix[isbm]

        nadp = nadp + nlons[ilat]
        coshi = coslo

    nlonrct = nlats * 2
    nrct = nlats * nlonrct

    iadp = 0
    irct = 0
    pwgt = np.zeros(nrct)
    iptr = np.zeros(nrct, dtype=int)
    for ilat in range(nlats):
        nblk = nlonrct / nlons[ilat]

        fs = dlat * (float(ilat + 1) - (float(nlats) / 2.) - 0.5)
        wgt = (abs(fs)) ** 0.25

        for ilonadp in range(nlons[ilat]):
            for iblk in range(int(nblk)):
                iptr[irct] = iadp
                irct = irct + 1

            pwgt[iadp] = wgt
            iadp = iadp + 1

    blats = np.linspace(np.pi / 2, -np.pi / 2, nlats + 1)

    blongs = []
    grid_xcs = []
    grid_ycs = []
    grid_zcs = []
    grid_areas = []
    uvws = []
    for i in range(len(blats) - 1):
        blong = np.linspace(0.0, 2.0 * np.pi, nlons[i] + 1)
        blongs.append(blong)

        inds1 = [(0, 0), (1, 0), (0, 1)]
        inds2 = [(1, 0), (1, 1), (0, 1)]
        for j in range(nlons[i]):
            for inds in [inds1, inds2]:
                polygon = []
                for ind in inds:
                    rlat = calc_r_to_Re_ratio(omega=omega, theta=blats[i + ind[0]] + np.pi / 2.0)
                    rlat = rlat * radius

                    xp, yp, zp = ast_stc(rlat, blats[i + ind[0]], blong[j + ind[1]])
                    polygon.append([xp.value, yp.value, zp.value])
                uvws.append(np.array(polygon.copy()))

                area = calc_area(*np.hstack(polygon))
                grid_areas.append(area)

        rzlat = calc_r_to_Re_ratio(omega=omega, theta=(blats[i] + blats[i + 1] + np.pi) / 2.0)
        rzlat = rzlat * radius

        xpc, ypc, zpc = ast_stc(rzlat, [(blats[i] + blats[i + 1]) / 2.0] * nlons[i],
                                (blong[1:] + blong[:-1]) / 2.)

        grid_xcs.append(xpc)
        grid_ycs.append(ypc)
        grid_zcs.append(zpc)

    grid_xyzs = np.array(uvws).copy()
    grid_xcs = np.hstack(grid_xcs).copy()
    grid_ycs = np.hstack(grid_ycs).copy()
    grid_zcs = np.hstack(grid_zcs).copy()

    rs, lats, longs = ast_cts(grid_xcs, grid_ycs, grid_zcs)

    grid_rs = rs.value.copy()
    grid_areas = np.array(grid_areas).reshape(-1, 2).sum(axis=1)

    grid_lats = lats.value.copy()
    grid_longs = longs.value.copy()
    grid_loggs = calc_loggs(mass=mass, re=radius, rs=grid_rs, omega=omega, thetas=grid_lats + np.pi / 2.0)

    blats = blats.copy()
    blongs = blongs.copy()
    nlons = nlons.copy()
    nlonrct = nlonrct
    iptr = iptr

    surface_grid = {'method': 'dots_grid', 'grid_xyzs': grid_xyzs, 'grid_areas': grid_areas,
                    'grid_loggs': grid_loggs, 'grid_lats': grid_lats, 'grid_longs': grid_longs,
                    'grid_rs': grid_rs, 'blats': blats, 'blongs': blongs, 'nlons': nlons,
                    'nlats': nlats, 'nlonrct': nlonrct, 'iptr': iptr}

    return surface_grid


def hp_surface_grid(omega, nside, radius, mass, processes=1):

    # TODO: Paralel hale getirilmeli.

    npix = hp.nside2npix(nside)

    pool = multiprocessing.Pool(processes=processes)

    input_args = [(nside, pixind, omega, radius) for pixind in np.arange(npix)]
    results = pool.map(hp_surface_grid_main, input_args)

    pool.close()
    pool.join()

    parts = np.array(results, dtype=object)

    grid_areas = np.hstack(parts[:, 0]).copy()
    grid_xcs = np.hstack(parts[:, 1])
    grid_ycs = np.hstack(parts[:, 2])
    grid_zcs = np.hstack(parts[:, 3])
    grid_xyzs = np.vstack(parts[:, 4]).copy()

    rs, lats, longs = ast_cts(grid_xcs, grid_ycs, grid_zcs)

    grid_rs = rs.value.copy()
    grid_lats = lats.value.copy()
    grid_longs = longs.value.copy()
    grid_loggs = calc_loggs(mass=mass, re=radius, rs=grid_rs, omega=omega, thetas=grid_lats + np.pi / 2.0)

    surface_grid = {'method': 'healpy', 'grid_xyzs': grid_xyzs, 'grid_areas': grid_areas,
                    'grid_loggs': grid_loggs, 'grid_lats': grid_lats, 'grid_longs': grid_longs,
                    'grid_rs': grid_rs}

    return surface_grid

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

    area = calc_area(*np.hstack(polygon[0])) + calc_area(*np.hstack(polygon[1]))

    return area, xpc.value, ypc.value, zpc.value, polygon

def tg_surface_grid(omega, nlats, radius, mass, processes=1):

    blats = np.linspace(np.pi / 2., -np.pi / 2., nlats + 1)

    ipn = 4
    pzh = abs(np.sin(blats[0]) - np.sin(blats[1])) * radius
    uarea = 2. * np.pi * radius * pzh / ipn

    pool = multiprocessing.Pool(processes=processes)

    input_args = [([blats[i], blats[i + 1]], uarea, omega, radius) for i in range(len(blats) - 1)]
    results = pool.map(tg_surface_grid_main, input_args)

    pool.close()
    pool.join()

    parts = np.array(results, dtype=object)

    nlons = parts[:, 0]
    blongs = parts[:, 1]
    grid_areas = np.hstack(parts[:, 2])
    grid_xcs = np.hstack(parts[:, 3])
    grid_ycs = np.hstack(parts[:, 4])
    grid_zcs = np.hstack(parts[:, 5])
    grid_xyzs = np.vstack(parts[:, 6])

    rs, lats, longs = ast_cts(grid_xcs, grid_ycs, grid_zcs)

    grid_rs = rs.value.copy()
    grid_areas = np.array(grid_areas).reshape(-1, 2).sum(axis=1)
    grid_lats = lats.value.copy()
    grid_longs = longs.value.copy()
    grid_loggs = calc_loggs(mass=mass, re=radius, rs=grid_rs, omega=omega, thetas=grid_lats + np.pi / 2.0)

    blats = blats.copy()
    blongs = blongs.copy()
    nlons = nlons.copy()

    surface_grid = {'method': 'trapezoid', 'grid_xyzs': grid_xyzs, 'grid_areas': grid_areas,
                    'grid_loggs': grid_loggs, 'grid_lats': grid_lats, 'grid_longs': grid_longs,
                    'grid_rs': grid_rs, 'blats': blats, 'blongs': blongs, 'nlons': nlons}

    return surface_grid


def tg_surface_grid_main(args):

    blatpn, uarea, omega, radius = args

    zh = abs(np.sin(blatpn[0]) - np.sin(blatpn[1])) * radius
    zarea = 2. * np.pi * radius * zh

    nlon = int(round(zarea / uarea))
    blong = np.linspace(0.0, 2.0 * np.pi, nlon + 1)

    areas = []
    uvws = []

    inds1 = [(0, 0), (1, 0), (0, 1)]
    inds2 = [(1, 0), (1, 1), (0, 1)]
    for j in range(nlon):
        for inds in [inds1, inds2]:
            polygon = []

            for ind in inds:
                rlat = calc_r_to_Re_ratio(omega=omega, theta=blatpn[ind[0]] + np.pi / 2.0)
                rlat = rlat * radius

                xp, yp, zp = ast_stc(rlat, blatpn[ind[0]], blong[j + ind[1]])

                polygon.append([xp.value, yp.value, zp.value])
            uvws.append(np.array(polygon.copy()))

            area = calc_area(*np.hstack(polygon))
            areas.append(area)

    rzlat = calc_r_to_Re_ratio(omega=omega, theta=(blatpn[0] + blatpn[1] + np.pi) / 2.0)
    rzlat = rzlat * radius

    xpc, ypc, zpc = ast_stc(rzlat, [(blatpn[0] + blatpn[1]) / 2.0] * nlon,
                            (blong[1:] + blong[:-1]) / 2.)

    return nlon, blong, areas, xpc.value, ypc.value, zpc.value, uvws


def calc_area(px1, py1, pz1, px2, py2, pz2, px3, py3, pz3):

    dx1 = px2 - px1
    dy1 = py2 - py1
    dz1 = pz2 - pz1

    dx2 = px3 - px1
    dy2 = py3 - py1
    dz2 = pz3 - pz1

    nx = dy1 * dz2 - dy2 * dz1
    ny = dz1 * dx2 - dz2 * dx1
    nz = dx1 * dy2 - dx2 * dy1

    sarea = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= sarea
    ny /= sarea
    nz /= sarea

    sarea = sarea / 2.

    return sarea


def rectmap_tg(nlons, blats, blongs, fss):

    nln = max(nlons)
    nlt = len(nlons)
    xlats = (blats[1:] + blats[:-1]) / 2.
    blongs = blongs[np.argmax(nlons)]
    xlongs = (blongs[1:] + blongs[:-1]) / 2.

    mlats, mlongs = np.meshgrid(xlongs, xlats)

    ll = np.zeros(nlt + 1, dtype=int)
    for i in range(nlt):
        ll[i + 1] = ll[i] + nlons[i]

    rmap = np.zeros((nlt, nln))
    x = np.arange(nln) + 0.5
    for i in range(nlt):
        lll = ((np.arange(nlons[i] + 2) - 0.5) * nln) / nlons[i]
        y = np.hstack([fss[ll[i + 1] - 1], fss[ll[i]: ll[i + 1] - 1], fss[ll[i]]])
        for j in range(nln):
            # dmin, imin = min(abs(x[j] - lll)), np.argmin(abs(x[j] - lll))
            imin = np.argmin(abs(x[j] - lll))
            rmap[i, j] = y[imin]

    xs = (max(xlongs) - min(xlongs)) / (len(xlongs) - 1) / 2.
    ys = (max(xlats) - min(xlats)) / (len(xlats) - 1) / 2.

    extent = [min(xlongs) - xs, max(xlongs) + xs, min(xlats) - ys, max(xlats) + ys]

    return xlongs, xlats, rmap, extent, mlats, mlongs


def rectmap_tg2(nlons, fss):

    nln = max(nlons)
    nlt = len(nlons)

    ll = np.zeros(nlt + 1, dtype=int)
    for i in range(nlt):
        ll[i + 1] = ll[i] + nlons[i]

    rmap = np.zeros((nlt, nln))
    x = np.arange(nln) + 0.5
    for i in range(nlt):
        lll = ((np.arange(nlons[i] + 2) - 0.5) * nln) / nlons[i]
        y = np.hstack([fss[ll[i + 1] - 1], fss[ll[i]: ll[i + 1] - 1], fss[ll[i]]])
        for j in range(nln):
            # dmin, imin = min(abs(x[j] - lll)), np.argmin(abs(x[j] - lll))
            imin = np.argmin(abs(x[j] - lll))
            rmap[i, j] = y[imin]

    xlats = np.linspace(np.pi / 2., -np.pi / 2., rmap.shape[0])
    xlongs = np.linspace(0, 2.0 * np.pi, rmap.shape[1])

    mlongs, mlats = np.meshgrid(xlongs, xlats)

    xs = (max(xlongs) - min(xlongs)) / (len(xlongs) - 1) / 2.
    ys = (max(xlats) - min(xlats)) / (len(xlats) - 1) / 2.

    extent = [min(xlongs) - xs, max(xlongs) + xs, min(xlats) - ys, max(xlats) + ys]

    return xlongs, xlats, rmap, extent, mlats, mlongs


def rectmap_hg(fss, nside, xsize=180, ysize=90):

    xlats = np.linspace(0.0, np.pi, ysize)
    xlongs = np.linspace(0.0, 2 * np.pi, xsize)

    mlongs, mlats = np.meshgrid(xlongs, xlats)
    grid_pix = hp.ang2pix(nside, mlats, mlongs, nest=False)

    rmap = fss[grid_pix]

    xlats = xlats[::-1] - np.pi / 2.
    xs = (max(xlongs) - min(xlongs)) / (len(xlongs) - 1) / 2.
    ys = (max(xlats) - min(xlats)) / (len(xlats) - 1) / 2.

    extent = [min(xlongs) - xs, max(xlongs) + xs, min(xlats) - ys, max(xlats) + ys]

    return xlongs, xlats, rmap, extent, np.flipud(mlats - np.pi / 2.), mlongs


def calc_fs_variation(phases, fss, areas, lats, longs, incl):

    sinlats = np.sin(lats)
    coslats = np.cos(lats)
    cosi = np.cos(np.deg2rad(incl))
    sini = np.sin(np.deg2rad(incl))

    pvf = []
    for i, phase in enumerate(phases):
        nlongs = longs + 2.0 * np.pi * phase
        coslong = np.cos(nlongs)

        mus = (sinlats * cosi + coslats * sini * coslong)
        ivis = np.where(mus > 0.0)[0]

        pvf.append(sum(fss[ivis] * areas[ivis] * mus[ivis]) / sum(areas[ivis] * mus[ivis]))

    return pvf

def get_total_fs(fss, areas, lats, incl):

    total_fs = sum(fss * areas) / sum(areas)

    w = np.argwhere(lats >= -np.deg2rad(incl)).T[0]
    partial_fs  = sum(fss[w] * areas[w]) / sum(areas[w])

    return total_fs, partial_fs


def calc_loggs(mass, re, rs, omega, thetas):

    mass_kg = mass * au.solMass.to(au.kg)
    g_cm = mass_kg * ac.G.to(au.cm**3 / (au.kg * au.s**2)).value
    r_cm = re * au.solRad.to(au.cm)
    r_ = rs / re

    geffs = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):

        theta = np.pi - theta if theta > np.pi / 2. else theta

        geffs[i] = (g_cm / r_cm**2) * (1.0 / r_[i]**4 + omega**4 * r_[i]**2 * np.sin(theta)**2
                                       - 2.0 * omega**2 * np.sin(theta)**2 / r_[i])**0.5

    return np.log10(geffs)


def calc_gds(omega, thetas):

    """
    :param omega
    :param thetas: radians

    :return: gds
    """

    gds = np.zeros(len(thetas))
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

        gds[i] = ((r**-4 + omega**4 * r**2 * np.sin(theta)**2 - (2.0 * omega**2 * np.sin(theta)**2 / r))**(1.0/8.0)
                  * fluxw**0.25)

    return gds


def r_func(r, omega, theta):
    return (1.0 / omega ** 2) + 0.5 - (1.0 / (omega ** 2 * r)) - 0.5 * r ** 2 * np.sin(theta)**2


def calc_r_to_Re_ratio(omega, theta):

    theta = np.pi - theta if theta > np.pi / 2. else theta

    r = sc_newton(r_func, 1.0, args=(omega, theta))  # , fprime=autograd.grad(r_func))

    return r


def nu_func(nu, omega, r, theta):

    return (np.cos(nu) + np.log(np.tan(nu / 2.)) - omega**2 * r**3 *
            np.cos(theta)**3 / 3.0 - np.cos(theta) - np.log(np.tan(theta / 2.)))


def calc_nu(omega, r, theta):

    theta = np.pi - theta if theta > np.pi / 2. else theta

    nu = sc_newton(nu_func, theta, args=(omega, r, theta))  # , fprime=autograd.grad(nu_func))

    return nu


def calc_radius(vsini, incl, period):

    """
    :param vsini: km/s
    :param incl: degree
    :param period: days

    :return: radius: solRad
    """

    veq = vsini / np.sin(np.deg2rad(incl))
    radius = (veq * period * 86400. / (2. * np.pi)) * au.km.to(au.solRad)

    return radius


def obs_data_prep(data, phases, srt, new_vels, mode):

    data = np.array(data)[srt]

    data_dict = {}
    for i, phase in enumerate(phases):

        data_dict[phase] = {}

        ovels, oprf, oerrs = data[i][:, 0], data[i][:, 1], data[i][:, 2]


        oprf_avg = 1.0
        if mode['scale']['method'] == 'mean':
            oprf_avg = np.average(oprf)

        elif mode['scale']['method'] == 'max':
            per = 100. / mode['scale']['percent']
            if mode['scale']['side'] == 'both':
                max_reg = np.hstack((oprf[:int(len(oprf) / per)], oprf[-int(len(oprf) / per):]))
            elif mode['scale']['side'] == 'left':
                max_reg = oprf[:int(len(oprf) / per)]
            elif mode['scale']['side'] == 'right':
                max_reg = oprf[-int(len(oprf) / per):]

            oprf_avg = np.average(max_reg)

        elif mode['scale']['method'] == 'region':
            owavs = (ovels * np.average(mode['wrange'])) / ac.c.to(au.kilometer / au.second).value + np.average(mode['wrange'])
            wavgw = []
            wavgf = []

            for region in mode['scale']['ranges']:
                wr = np.argwhere((owavs >= region[0]) & (owavs <= region[1])).T[0]

                wavgw.append(np.average(owavs[wr]))
                wavgf.append(np.average(oprf[wr]))

            x1, x2, x3 = wavgw
            y1, y2, y3 = wavgf
            denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
            A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
            B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
            C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
            oprf_avg = A * owavs ** 2 + B * owavs + C

        oprf = oprf / oprf_avg
        oerrs = oerrs / oprf_avg

        if new_vels is not None:
            vels, oprf, oerrs = PyD.interp1d_unc(new_vels, ovels, oprf, oerrs, kind='linear')
        else:
            vels = data[0][:, 0].copy()

        nvels = len(vels)

        data_dict[phase]['vels'] = vels.copy()
        data_dict[phase]['prf'] = oprf.copy()
        data_dict[phase]['errs'] = oerrs.copy()

    return data_dict, vels, nvels


def pixel_prop_prep(vels, lvels, ints_phot, ints_spot, rgi, ctp, cts,
                    logg, ld_law, mu, dv, vrt):

    ldcs_phot = [rgi[0]([ctp, logg]), rgi[1]([ctp, logg])]
    ldcs_spot = [rgi[0]([cts, logg]), rgi[1]([cts, logg])]

    ldf_phot = ld_factors_calc(ld_law=ld_law, ldc=ldcs_phot, mu=mu)
    ldf_spot = ld_factors_calc(ld_law=ld_law, ldc=ldcs_spot, mu=mu)

    pcm_phot, pnmk = apply_macro_conv(vels=lvels, ints=ints_phot, mu=mu, vrt=vrt)
    pcm_spot, snmk = apply_macro_conv(vels=lvels, ints=ints_spot, mu=mu, vrt=vrt)

    lp_phot = np.interp(vels, lvels + dv, pcm_phot[pnmk: -pnmk])
    lp_spot = np.interp(vels, lvels + dv, pcm_spot[snmk: -snmk])

    return ldf_phot, ldf_spot, lp_phot, lp_spot


def calc_model_prf(fss, ctps, ctss, ldfs_phot, ldfs_spot, lps_phot, lps_spot, mode, ivis, areas, mus, vels=None):

    coeffs_phot = ctps ** 4 * ldfs_phot * areas * mus
    wgt_phot = coeffs_phot * (1.0 - fss[ivis])
    wgtn_phot = anp.sum(wgt_phot)

    coeffs_spot = ctss ** 4 * ldfs_spot * areas * mus
    wgt_spot = coeffs_spot * fss[ivis]
    wgtn_spot = anp.sum(wgt_spot)

    prf = anp.sum(wgt_phot[:, None] * lps_phot + wgt_spot[:, None] * lps_spot, axis=0)
    prf /= wgtn_phot + wgtn_spot

    scale_factor = 1.0
    if mode['scale']['method'] == 'mean':
        scale_factor = anp.sum(prf) / len(prf)

    elif mode['scale']['method'] == 'max':
        per = 100. / mode['scale']['percent']
        if mode['scale']['side'] == 'both':
            max_reg = np.hstack((prf[:int(len(prf) / per)], prf[-int(len(prf) / per):]))
        elif mode['scale']['side'] == 'left':
            max_reg = prf[:int(len(prf) / per)]
        elif mode['scale']['side'] == 'right':
            max_reg = prf[-int(len(prf) / per):]
        scale_factor = anp.sum(max_reg) / len(max_reg)

    elif mode['scale']['method'] == 'region' and vels is not None:
        wavs = (vels * np.average(mode['wrange'])) / ac.c.to(au.kilometer / au.second).value + np.average(mode['wrange'])
        wavgw = []
        wavgf = []

        for region in mode['scale']['ranges']:
            wr = np.argwhere((wavs >= region[0]) & (wavs <= region[1])).T[0]

            wavgw.append(anp.sum(wavs[wr]) / len(wr))
            wavgf.append(anp.sum(prf[wr]) / len(wr))

        x1, x2, x3 = wavgw
        y1, y2, y3 = wavgf
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
        scale_factor = A * wavs ** 2 + B * wavs + C

    prf = prf / scale_factor

    return prf


def ldcs_rgi_prep(temps, loggs, law, mh, wrange, ld_model, ld_data_path, mu_min):

    dist = np.zeros((2, len(temps), len(loggs)))
    for ta, temp in enumerate(temps):
        for la, logg in enumerate(loggs):

                ldc = ld_coeff_calc(ld_law=law, teff=temp, logg=logg, mh=mh, ld_model=ld_model,
                                              wrange=wrange, ld_data_path=ld_data_path,
                                              mu_min=mu_min)

                dist[0][ta, la] = ldc[0]
                if len(ldc) == 2:
                    dist[1][ta, la] = ldc[1]

    points = (temps, loggs)

    rgi = [RegularGridInterpolator(points, dist[0]),
           RegularGridInterpolator(points, dist[1])]

    return rgi

def calc_omega_and_requiv(mass, period, re):

    """
    :param mass: solMass
    :param period: days
    :param re: solRad

    :return omega
    :return requiv: solRad
    :return Rp: solRad
    """

    mass_kg = mass * au.solMass.to(au.kg)
    gm = mass_kg * ac.G.value
    period_s = period * 86400
    re_m = re * au.solRad.to(au.m)
    omega = (2.0 * np.pi / period_s) * np.sqrt(re_m ** 3 / gm)
    rp = calc_r_to_Re_ratio(omega=omega, theta=0) * re

    requiv = (re ** 2 * rp) ** (1. / 3.)

    return omega, requiv, rp


def gaussian(x, amp, cen, sig, lvl):
    return amp * np.exp(-(x - cen) ** 2 / (2 * sig ** 2)) + lvl


def fit_gaussian(x, y):
    cen = sum(x * y) / sum(y)
    sig = np.sqrt(sum(y * (x - cen) ** 2) / sum(y))
    amp, lvl = max(y), min(y)

    bounds = [(-np.inf, -np.inf, 0.0, -np.inf), (np.inf, np.inf, np.inf, np.inf)]
    popt, pcov = curve_fit(gaussian, x, y, p0=[amp, cen, sig, lvl], bounds=bounds)

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

    convolved_ints = pyasl.instrBroadGaussFast(waves, ints, resolution,
                                               edgeHandling="firstlast",
                                               fullout=False, maxsig=None, equid=False)

    return convolved_ints


def ld_coeff_calc(ld_law, teff, logg, mh, ld_model, wrange, ld_data_path, mu_min=0.1):

    sc = StellarLimbDarkening(Teff=teff, logg=logg, M_H=mh, ld_model=ld_model,  # kurucz mps1 mps2 stagger
                              ld_data_path=ld_data_path)

    if ld_law == 'linear':
        ldc = sc.compute_linear_ld_coeffs(wrange, mode='custom',
                                          custom_wavelengths=np.array(wrange),
                                          custom_throughput=np.array([1, 1]),
                                          mu_min=mu_min, return_sigmas=False)

    if ld_law == 'square-root':
        ldc = sc.compute_squareroot_ld_coeffs(wrange, mode='custom',
                                              custom_wavelengths=np.array(wrange),
                                              custom_throughput=np.array([1, 1]),
                                              mu_min=mu_min, return_sigmas=False)

    if ld_law == 'quadratic':
        ldc = sc.compute_quadratic_ld_coeffs(wrange, mode='custom',
                                             custom_wavelengths=np.array(wrange),
                                             custom_throughput=np.array([1, 1]),
                                             mu_min=mu_min, return_sigmas=False)

    return ldc


def ld_factors_calc(ld_law, ldc, mu):

    if ld_law == 'linear':
        return 1. - ldc[0] * (1. - mu)

    elif ld_law == 'square-root':
        return 1. - ldc[0] * (1. - mu) - ldc[1] * (1. - np.sqrt(mu))

    elif ld_law == 'quadratic':
        return 1. - ldc[0] * (1. - mu) - ldc[1] * (1. - mu) ** 2


def macro_kernel(nop, muval, vrt, deltav):

    # Calculate projected sigma for radial and tangential velocity distributions.
    # muval		                                                        # current value of mu
    os = 1  # internal oversampling factor for convolutions
    sigma = os * vrt / np.sqrt(2) / deltav  # standard deviation in points
    sigr = sigma * muval  # reduce by current mu value
    sigt = sigma * np.sqrt(1.0 - muval ** 2)  # reduce by sqrt(1-mu^2)
    nfine = os * nop  # # of oversampled points

    # Figure out how many points to use in macroturbulence kernel.
    nmk = max(int(10 * sigma), ((nfine - 3) / 2))  # extend kernel to 10 sigma
    nmk = max(nmk, 3)  # pad with at least 3 pixels

    nmk = int(nmk)  # tek sayı gerekiyor convol için

    # Construct radial macroturbulence kernel with a sigma of mu*VRT/sqrt(2).
    if sigr > 0:
        xarg = (np.arange(2 * nmk + 1, dtype=int) - nmk) / sigr  # exponential argument
        exparg = (-0.5 * (xarg ** 2))
        exparg[exparg < -20] = -20
        mrkern = np.exp(exparg)  # compute the gaussian
        mrkern = mrkern / sum(mrkern)  # normalize the profile
    else:
        mrkern = np.zeros(2 * nmk + 1)  # init with 0d0
        mrkern[nmk] = 1.0  # delta function

    # Construct tangential kernel with a sigma of sqrt(1-mu^2)*VRT/sqrt(2).
    if sigt > 0:
        xarg = (np.arange(2 * nmk + 1, dtype=int) - nmk) / sigt  # exponential argument
        exparg = (-0.5 * (xarg ** 2))
        exparg[exparg < -20] = -20
        mtkern = np.exp(exparg)  # compute the gaussian
        mtkern = mtkern / sum(mtkern)  # normalize the profile
    else:
        mtkern = np.zeros(2 * nmk + 1)  # init with 0d0
        mtkern[nmk] = 1.0  # delta function

    # Sum the radial and tangential components, weighted by surface area.
    area_r = 0.5  # assume equal areas
    area_t = 0.5  # ar+at must equal 1
    mkern = area_r * mrkern + area_t * mtkern  # add both components

    return mkern, nmk


def apply_macro_conv(vels, ints, mu, vrt):
    deltav = vels[1] - vels[0]
    mkernel, nmk = macro_kernel(nop=len(vels), muval=mu, vrt=vrt, deltav=deltav)

    conv_mac = scipy_convolve(np.pad(ints, (nmk, nmk), 'edge'),
                              mkernel, mode='same', method='auto')

    return conv_mac, nmk


def generate_spotted_surface(surface_grid, spots_params, default=1e-5):

    """
    :param surface_grid: A dictionary containing the parameters of the surface grid.
    :param spots_params: A dictionary containing the parameters of the spots.
    :param default     : Minimum spot contrast value.
    :return: smap      : Surface brightness contrast corresponding to each latitude - longitude pair
    """

    lats_spots = spots_params['lats_spots']
    longs_spots = spots_params['longs_spots']
    rs_spots = spots_params['rs_spots']
    cs_spots = spots_params['cs_spots']

    smap = np.ones(len(surface_grid['grid_lats'])) * default
    for (lat_spot, long_spot, r_spot, c_spot) in zip(lats_spots, longs_spots, rs_spots, cs_spots):

        dlon = surface_grid['grid_longs'] - np.deg2rad(long_spot)
        angles = np.arccos(np.sin(surface_grid['grid_lats']) * np.sin(np.deg2rad(lat_spot)) +
                           np.cos(surface_grid['grid_lats']) * np.cos(np.deg2rad(lat_spot)) * np.cos(dlon))

        ii = np.argwhere(angles <= np.deg2rad(r_spot)).T[0]
        ni = len(ii)
        if ni > 0:
            smap[ii] = c_spot

    return smap


def test_prf_plot(phases, vels, data, plotp, spotless_slps, recons_slps, mode):

    plt.figure()
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    if mode == 'line':
        plotp['sep_prf'] = plotp['line_sep_prf']
        plotp['sep_res'] = plotp['line_sep_res']

    if mode == 'mol':
        plotp['sep_prf'] = plotp['mol_sep_prf']
        plotp['sep_res'] = plotp['mol_sep_res']

    for i, phase in enumerate(sorted(phases)):

        maxv = max(vels)
        maxi = max(data[phase]['prf']) + i * plotp['sep_prf']
        residual = data[phase]['prf'] - recons_slps[phase]['prf']
        maxir = np.average(residual + i * plotp['sep_res'])

        if plotp['show_err_bars']:
            ax1.errorbar(vels, data[phase]['prf'] + i * plotp['sep_prf'],
                         yerr=data[phase]['errs'], fmt='o', color='k', ms=plotp['markersize'])
            ax2.errorbar(vels, residual + i * plotp['sep_res'],
                         yerr=data[phase]['errs'], fmt='o', color='k', ms=plotp['markersize'])
        else:
            ax1.plot(vels, data[phase]['prf'] + i * plotp['sep_prf'], 'ko', ms=plotp['markersize'])
            ax2.plot(vels, residual + i * plotp['sep_res'], 'ko', ms=plotp['markersize'])

        ax1.plot(vels, spotless_slps[phase]['prf'] + i * plotp['sep_prf'], 'b', linewidth=plotp['linewidth'], zorder=2)
        ax1.plot(vels, recons_slps[phase]['prf'] + i * plotp['sep_prf'], 'r', linewidth=plotp['linewidth'], zorder=3)
        ax1.annotate(str('%0.3f' % round(phase, 3)), xy=(maxv - maxv / 3.1, maxi + plotp['sep_prf'] / 10.),
                     color='g')
        ax2.annotate(str('%0.3f' % round(phase, 3)), xy=(maxv - maxv / 10., maxir + plotp['sep_res'] / 10.),
                     color='g')
        ax2.axhline(i * plotp['sep_res'], color='r', zorder=3)

    ax1.plot([], [], 'ko', label='Obs. Data', ms=plotp['markersize'])
    ax1.plot([], [], 'b', label='Spotless Model', linewidth=plotp['linewidth'])
    ax1.plot([], [], 'r', label='Spotted Model', linewidth=plotp['linewidth'])
    ax1.set_xlabel('Radial Velocity (km / s)', fontsize=plotp['fontsize'])
    ax1.set_ylabel('I / Ic', fontsize=plotp['fontsize'])
    ax1.legend()

    ax2.set_xlabel('Radial Velocity (km / s)', fontsize=plotp['fontsize'])
    ax2.set_ylabel('Residuals', fontsize=plotp['fontsize'])

    ax1.tick_params(axis='both', labelsize=plotp['ticklabelsize'])
    ax2.tick_params(axis='both', labelsize=plotp['ticklabelsize'])

    plt.tight_layout()


def test_map_plot(phases, surface_grid, fake_fss, recons_fss, fake_total_fs, recons_total_fs, plotp):

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))

    if surface_grid['method'] == 'trapezoid':
        nlons = surface_grid['nlons'].copy()

        _, _, frmap, fextent, _, _ = rectmap_tg2(nlons=nlons, fss=fake_fss)
        _, _, crmap, cextent, _, _ = rectmap_tg2(nlons=nlons, fss=recons_fss)

        img2 = ax1.imshow(frmap, cmap='gray_r', aspect='equal', extent=np.rad2deg(fextent))
        img3 = ax2.imshow(crmap, cmap='gray_r', aspect='equal', extent=np.rad2deg(cextent))

    if surface_grid['method'] == 'healpy':
        _, _, frmap, fextent, _, _ = rectmap_hg(fss=fake_fss, nside=surface_grid['nside'], xsize=1000,
                                                     ysize=500)
        _, _, crmap, cextent, _, _ = rectmap_hg(fss=recons_fss, nside=surface_grid['nside'], xsize=1000,
                                                     ysize=500)

        img2 = ax1.imshow(frmap, cmap='gray_r', aspect='equal', extent=np.rad2deg(fextent))
        img3 = ax2.imshow(crmap, cmap='gray_r', aspect='equal', extent=np.rad2deg(cextent))

    ax1.text(30, -70, 'Total Spotted Area (%) = ' + str(round(fake_total_fs * 100, 1)))
    ax2.text(30, -70, 'Total Spotted Area (%) = ' + str(round(recons_total_fs * 100, 1)))

    clb2 = fig.colorbar(img2, ax=ax1, location='right', shrink=1.0)
    clb2.set_label('$f_s$', fontsize=12)
    clb2.ax.tick_params(labelsize=12)

    clb3 = fig.colorbar(img3, ax=ax2, location='right', shrink=1.0)
    clb3.set_label('$f_s$', fontsize=12)
    clb3.ax.tick_params(labelsize=12)

    ax1.set_xticks(np.arange(0, 420, 60))
    ax1.set_yticks(np.arange(-90, 120, 30))
    ax1.set_title('Fake Map', fontsize=plotp['fontsize'])
    ax1.set_xlabel('Longitude ($^\circ$)', fontsize=plotp['fontsize'])
    ax1.set_ylabel('Latitude ($^\circ$)', fontsize=plotp['fontsize'])

    ax2.set_xticks(np.arange(0, 420, 60))
    ax2.set_yticks(np.arange(-90, 120, 30))
    ax2.set_title('Reconstructed Map', fontsize=plotp['fontsize'])
    ax2.set_xlabel('Longitude ($^\circ$)', fontsize=plotp['fontsize'])
    ax2.set_ylabel('Latitude ($^\circ$)', fontsize=plotp['fontsize'])

    ax1.plot([360 * (1.0 - phases), 360 * (1.0 - phases)], [-85, -75], 'k', linewidth=2)
    ax2.plot([360 * (1.0 - phases), 360 * (1.0 - phases)], [-85, -75], 'k', linewidth=2)
    if 0.0 in phases:
        ax1.plot([0, 0], [-85, -75], 'k', linewidth=2)
        ax2.plot([0, 0], [-85, -75], 'k', linewidth=2)

    ax1.tick_params(axis='both', labelsize=plotp['ticklabelsize'])
    ax2.tick_params(axis='both', labelsize=plotp['ticklabelsize'])

    plt.tight_layout()


def make_grid_contours(chisq_grid):

    names = list(chisq_grid.keys())
    names.remove('chisqs')

    if len(names) == 1:

        mn = np.argmin(chisq_grid['chisqs'])
        mpar = chisq_grid[names[0]][mn]

        print(names[0], ':', mpar)

        plt.plot(chisq_grid[names[0]], chisq_grid['chisqs'], 'ko')
        if names[0] == 'vsini':
            plt.xlabel('vsini (km/s)', fontsize=15)

        if names[0] == 'eqw':
            plt.xlabel('EW', fontsize=15)
        plt.ylabel('$\chi^2$', fontsize=15)
        plt.show()

        return

    combin = np.array(list(combinations(names, 2)))
    combin[len(names) - 2] = combin[len(names) - 2][::-1]

    for pair in combin:

        par1_name = pair[0]
        par2_name = pair[1]

        wf = np.argmin(chisq_grid['chisqs'])
        index = []
        for rpar in chisq_grid:
            if rpar not in [par1_name, par2_name, 'chisqs']:
                index.append(np.argwhere(chisq_grid[rpar] == chisq_grid[rpar][wf]).T[0])
        if len(index) != 0:
            inters = reduce(np.intersect1d, index)
            chi = chisq_grid['chisqs'][inters]
            par1 = chisq_grid[par1_name][inters]
            par2 = chisq_grid[par2_name][inters]
        else:
            chi = chisq_grid['chisqs']
            par1 = chisq_grid[par1_name]
            par2 = chisq_grid[par2_name]

        contour = np.zeros((len(np.unique(par2)), len(np.unique(par1))))
        for i, ei in enumerate(sorted(np.unique(par2))):
            vss = np.argsort(par1[par2 == ei])
            contour[i] = chi[par2 == ei][vss]

        mn = np.argmin(chisq_grid['chisqs'])
        mpar1 = chisq_grid[par1_name][mn]
        mpar2 = chisq_grid[par2_name][mn]

        fig = plt.figure()
        levels = [1 + chisq_grid['chisqs'][mn]]
        uc = plt.contour(sorted(np.unique(par1)), sorted(np.unique(par2)), contour, levels=levels,
                         cmap='RdYlBu')  # alpha=0.7,
        plt.show(block=False)
        plt.close(fig=fig)

        chipone = uc.allsegs[0][0]

        print(par1_name, ':', mpar1, '-' + str(mpar1 - min(chipone[:, 0])), '+' + str(max(chipone[:, 0]) - mpar1))
        print(par2_name, ':', mpar2, '-' + str(mpar2 - min(chipone[:, 1])), '+' + str(max(chipone[:, 1]) - mpar2))

        fig = plt.figure()
        cs = plt.contourf(sorted(np.unique(par1)), sorted(np.unique(par2)), contour, 50, cmap='RdYlBu')  # alpha=0.7,
        if par1_name == 'vsini' and par2_name == 'eqw':
            plt.xlabel('vsini (km/s)', fontsize=15)
            plt.xlabel('EW', fontsize=15)

        elif par2_name == 'vsini' and par1_name == 'eqw':
            plt.xlabel('EW', fontsize=15)
            plt.xlabel('vsini (km/s)', fontsize=15)

        else:
            plt.xlabel(par1_name, fontsize=15)
            plt.ylabel(par2_name, fontsize=15)

        plt.tick_params(labelsize=15)

        cbaxes = fig.add_axes([0.150, 0.85, 0.82, 0.03])
        cbar = plt.colorbar(cax=cbaxes, orientation='horizontal',
                            ticks=np.linspace(np.min(contour), np.max(contour), 5), format="%0.3f")
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label('$\chi^2$', fontsize=15)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")

        plt.subplots_adjust(left=0.15, bottom=0.14, right=0.97, top=0.84)

    plt.show()


def grid_test(xyzs, scalars1, scalars2, scalars3, scalars4):

    from mayavi import mlab

    def plot(title, dx, dy, dz, dtriangles, dnscalars1):
        mlab.figure(title)
        smesh = mlab.triangular_mesh(dx, dy, dz, dtriangles, scalars=dnscalars1,
                                     colormap='YlOrBr', line_width=3.0)

        carr = smesh.module_manager.scalar_lut_manager.lut.table.to_array()
        ncarr = carr[::-1]
        smesh.module_manager.scalar_lut_manager.lut.table = ncarr

        cb = mlab.colorbar(smesh, title=title, nb_labels=5, label_fmt='%0.4f', orientation='horizontal')
        cb.label_text_property.font_family = 'times'
        cb.label_text_property.bold = 0
        # cb.label_text_property.font_size = 50

    x = []
    y = []
    z = []
    triangles = []
    nscalars1 = []
    nscalars2 = []
    nscalars3 = []
    nscalars4 = []
    for i, xyz in enumerate(xyzs):
        nscalars1.append([scalars1[i]] * 3)
        nscalars2.append([scalars2[i]] * 3)
        nscalars3.append([scalars3[i]] * 3)
        nscalars4.append([scalars4[i]] * 3)
        for row in xyz:
            x.append(row[0])
            y.append(row[1])
            z.append(row[2])
        triangles.append((0 + i * 3, 1 + i * 3, 2 + i * 3))
    nscalars1 = np.hstack(nscalars1)
    nscalars2 = np.hstack(nscalars2)
    nscalars3 = np.hstack(nscalars3)
    nscalars4 = np.hstack(nscalars4)

    plot('Gravity Darkening Distribution', x, y, z, triangles, nscalars1)
    plot('Surface Element Area Distribution', x, y, z, triangles, nscalars2)
    plot('Latitude Distribution', x, y, z, triangles, nscalars3)
    plot('Longitude Distribution', x, y, z, triangles, nscalars4)

    mlab.show()
