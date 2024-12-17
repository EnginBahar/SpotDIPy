# cython: language_level=3

cimport cython

import numpy as np
cimport numpy as np
# from cython cimport Py_ssize_t
from libc.math cimport sin, cos, atan2, log10, sqrt, pi, exp
from libc.stdlib cimport malloc, free
# from cython.parallel import prange

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t solMass = 1.988409870698051e+30  # kg
cdef DTYPE_t solRad = 695700000.0             # m
cdef DTYPE_t G = 6.6743e-11                   # m3 / (kg s2)

cpdef tuple spherical_to_cartesian(DTYPE_t r, DTYPE_t lat, DTYPE_t lon):
    cdef DTYPE_t x, y, z
    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)
    return x, y, z

cpdef tuple cartesian_to_spherical(DTYPE_t x, DTYPE_t y, DTYPE_t z):
    cdef DTYPE_t r, lat, lon
    r = sqrt(x**2 + y**2 + z**2)
    lat = atan2(z, sqrt(x**2 + y**2))
    lon = atan2(y, x)
    lon = lon + 2.0 * pi if lon < 0 else lon

    return r, lat, lon

cpdef tuple cartesian_to_spherical_vectorized(np.ndarray[DTYPE_t, ndim=1] x,
                                              np.ndarray[DTYPE_t, ndim=1] y,
                                              np.ndarray[DTYPE_t, ndim=1] z):
    cdef np.ndarray[DTYPE_t, ndim=1] r = np.sqrt(x**2 + y**2 + z**2)
    cdef np.ndarray[DTYPE_t, ndim=1] lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    cdef np.ndarray[DTYPE_t, ndim=1] lon = np.arctan2(y, x)
    lon = np.where(lon < 0, lon + 2.0 * pi, lon)  # Negatif uzunlukları 0-2π aralığına getir

    return r, lat, lon

cpdef DTYPE_t newtonRaphson(DTYPE_t x0, DTYPE_t omega, DTYPE_t theta, DTYPE_t tol=1.0e-8):
    cdef DTYPE_t h

    h = r_func(x0, omega, theta) / deriv_r_func(x0, omega, theta)
    while abs(h) >= tol:
        h = r_func(x0, omega, theta) / deriv_r_func(x0, omega, theta)
        x0 = x0 - h

    return x0

cpdef DTYPE_t r_func(DTYPE_t r, DTYPE_t omega, DTYPE_t theta):

    return (1.0 / omega ** 2) + 0.5 - (1.0 / (omega ** 2 * r)) - 0.5 * r ** 2 * sin(theta)**2

cpdef DTYPE_t deriv_r_func(DTYPE_t r, DTYPE_t omega, DTYPE_t theta):

    return 1.0 / omega**2 / r**2 - sin(theta)**2 * r

cpdef DTYPE_t calc_r_to_Re_ratio(DTYPE_t omega, DTYPE_t theta):
    cdef DTYPE_t x0=1.0

    return newtonRaphson(x0=x0, omega=omega, theta=theta)

cpdef DTYPE_t calc_triangle_area(np.ndarray[DTYPE_t, ndim=2] triangle):

    cdef DTYPE_t area
    cdef np.ndarray[DTYPE_t, ndim=1] cp = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])

    area = np.linalg.norm(cp) / 2.0

    return area

cpdef DTYPE_t calc_logg(DTYPE_t mass, DTYPE_t re, DTYPE_t rs, DTYPE_t omega, DTYPE_t theta):
    cdef DTYPE_t mass_kg = mass * solMass
    cdef DTYPE_t g_cm = mass_kg * G * 1.0e6
    cdef DTYPE_t r_cm = re * solRad * 100
    cdef DTYPE_t r_ = rs / re
    cdef DTYPE_t geff

    theta = pi - theta if theta > pi / 2. else theta

    geff = (g_cm / r_cm**2) * sqrt(1.0 / r_**4 + omega**4 * r_**2 * sin(theta)**2 - 2.0 * omega**2 * sin(theta)**2 / r_)
    return log10(geff)

cpdef DTYPE_t ld_factors_calc(str ld_law, DTYPE_t mu, DTYPE_t ldc1, DTYPE_t ldc2):

    if ld_law == 'linear':
        return 1. - ldc1 * (1. - mu)

    elif ld_law == 'square-root':
        return 1. - ldc1 * (1. - mu) - ldc2 * (1. - sqrt(mu))

    elif ld_law == 'quadratic':
        return 1. - ldc1 * (1. - mu) - ldc2 * (1. - mu) ** 2

cpdef tuple td_grid_sect(tuple args):

    cdef DTYPE_t[2] blatpn = args[0]
    cdef DTYPE_t uarea = args[1]
    cdef DTYPE_t omega = args[2]
    cdef DTYPE_t radius = args[3]
    cdef DTYPE_t mass = args[4]
    cdef int nlon = int(round(2 * pi * radius ** 2 * abs(sin(blatpn[0]) - sin(blatpn[1])) / uarea))
    cdef np.ndarray[DTYPE_t, ndim=1] blong = np.linspace(0.0, 2.0 * pi, nlon + 1, dtype=DTYPE)
    cdef list areas = []
    cdef list xyzs = []
    cdef list xpc = []
    cdef list ypc = []
    cdef list zpc = []
    cdef list rs = []
    cdef list lats = []
    cdef list longs = []
    cdef list loggs = []
    cdef int i, j, k
    cdef list inds = [[(0, 0), (1, 0), (0, 1)], [(1, 0), (1, 1), (0, 1)]]
    cdef DTYPE_t rzlat, rlat, xp, yp, zp, x, y, z

    rzlat = calc_r_to_Re_ratio(omega=omega, theta=(blatpn[0] + blatpn[1] + pi) / 2.0) * radius

    for j in range(nlon):
        for i in range(2):
            polygon = []
            for k in range(3):
                rlat = calc_r_to_Re_ratio(omega=omega, theta=blatpn[inds[i][k][0]] + pi / 2.0) * radius
                xp, yp, zp = spherical_to_cartesian(rlat, blatpn[inds[i][k][0]], blong[j + inds[i][k][1]])
                polygon.append([xp, yp, zp])
            xyzs.append(np.array(polygon))
            areas.append(calc_triangle_area(np.array(polygon)))

        rs.append(rzlat)
        lats.append((blatpn[0] + blatpn[1]) / 2.0)
        longs.append((blong[j] + blong[j + 1]) / 2.0)

        x, y, z = spherical_to_cartesian(rzlat, (blatpn[0] + blatpn[1]) / 2.0, (blong[j] + blong[j + 1]) / 2.0)

        xpc.append(x)
        ypc.append(y)
        zpc.append(z)

        loggs.append(calc_logg(mass=mass, re=radius, rs=rzlat, omega=omega, theta=((blatpn[0] + blatpn[1])
                                                                                   / 2.0) + pi / 2.0))

    return nlon, blong, areas, rs, lats, longs, xyzs, loggs

cpdef list td_surface_grid(np.ndarray[DTYPE_t, ndim=1] blats, DTYPE_t omega, DTYPE_t radius, DTYPE_t mass):

    cdef int j
    cdef int ipn = 4
    cdef DTYPE_t uarea, pzh
    cdef list results = []

    pzh = abs(sin(blats[0]) - sin(blats[1])) * radius
    uarea = 2. * pi * radius * pzh / ipn

    for j in range(len(blats) - 1):
        results.append(td_grid_sect(([blats[j], blats[j + 1]], uarea, omega, radius, mass)))

    # parts = np.array(results, dtype=object)
    #
    # container['method'] = 'trapezoid'
    # container['nlons'] = parts[:, 0]
    # container['blongs'] = parts[:, 1]
    # container['grid_areas'] = np.hstack(parts[:, 2]).reshape(-1, 2).sum(axis=1)
    # container['grid_rs'] = np.hstack(parts[:, 3])
    # container['grid_lats'] = np.hstack(parts[:, 4])
    # container['grid_longs'] = np.hstack(parts[:, 5])
    # container['grid_xyzs'] = np.vstack(parts[:, 6])
    # container['grid_loggs'] = np.hstack(parts[:, 7])
    # container['blats'] = blats

    return results

cpdef tuple macro_kernel(int nop, DTYPE_t muval, DTYPE_t vrt, DTYPE_t deltav):

    """ Adapted from SME (Valenti, J. A., & Piskunov, N. 1996, A&AS, 118, 595) """

    cdef DTYPE_t sigma, sigr, sigt, nfine
    cdef int nmk = 3

    sigma = vrt / sqrt(2.0) / deltav
    sigr = sigma * muval
    sigt = sigma * sqrt(1.0 - muval ** 2)
    nfine = nop

    cdef DTYPE_t temp1, temp2, mrsum = 0.0, mtsum = 0.0

    temp1 = 10.0 * sigma
    if temp1 > nmk:
        nmk = <int> temp1

    temp2 = (nfine - 3) / 2
    if temp2 > nmk:
        nmk = <int> temp2

    if nmk < 3:
        nmk = 3

    cdef np.ndarray[DTYPE_t, ndim=1] mrkern = np.zeros(2 * nmk + 1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] mtkern = np.zeros(2 * nmk + 1, dtype=DTYPE)
    cdef int i, j

    if sigr > 0.0:
        for i in range(2 * nmk + 1):
            if -0.5 * ((i - nmk) / sigr)**2 < -20.0:
                mrkern[i] = exp(-20)
            else:
                mrkern[i] = exp(-0.5 * ((i - nmk) / sigr)**2)
            mrsum = mrsum + mrkern[i]
        mrkern /= mrsum
        # mrkern /= np.sum(mrkern)

    else:
        mrkern[nmk] = 1.0

    if sigt > 0.0:
        for j in range(2 * nmk + 1):
            if -0.5 * ((j - nmk) / sigt)**2 < -20.0:
                mtkern[j] = exp(-20)
            else:
                mtkern[j] = exp(-0.5 * ((j - nmk) / sigt)**2)
            mtsum += mtkern[j]
        mtkern /= mtsum
        # mtkern /= np.sum(mtkern)

    else:
        mtkern[nmk] = 1.0

    return (mrkern + mtkern) / 2.0, nmk

cpdef apply_macro_conv(np.ndarray[DTYPE_t, ndim=1] vels, np.ndarray[DTYPE_t, ndim=1] ints, DTYPE_t mu, DTYPE_t vrt):

    cdef DTYPE_t deltav = vels[1] - vels[0]

    mkernel, nmk = macro_kernel(len(vels), mu, vrt, deltav)

    conv_mac = np.convolve(np.pad(ints, (nmk, nmk), 'edge'), mkernel, mode='same')

    return conv_mac, nmk

# @cython.boundscheck(False)
# @cython.wraparound(False)
cpdef linear_interpolation(DTYPE_t[:] x, DTYPE_t[:] y, DTYPE_t[:] x_interp):
    cdef int n = len(x)
    cdef int m = len(x_interp)
    cdef DTYPE_t[:] y_interp = np.zeros(m, dtype=DTYPE)

    cdef int i, j
    cdef DTYPE_t slope, intercept

    for j in range(m):
        i = 0
        while i < n and x_interp[j] > x[i]:
            i += 1

        if i == 0:
            slope = (y[1] - y[0]) / (x[1] - x[0])
            intercept = y[0] - slope * x[0]
        elif i == n:
            slope = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
            intercept = y[n - 1] - slope * x[n - 1]
        else:
            slope = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            intercept = y[i] - slope * x[i]

        y_interp[j] = slope * x_interp[j] + intercept

    return y_interp

# @cython.boundscheck(False)
# @cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] calc_pixel_coeffs_spec(tuple args):

    cdef DTYPE_t itime = args[0]
    cdef np.ndarray[DTYPE_t, ndim=1] plats = args[1]
    cdef np.ndarray[DTYPE_t, ndim=1] vlats = args[2]
    cdef np.ndarray[DTYPE_t, ndim=2] ldcs_phot = args[3]
    cdef np.ndarray[DTYPE_t, ndim=2] ldcs_cool = args[4]
    cdef np.ndarray[DTYPE_t, ndim=2] ldcs_hot = args[5]
    cdef np.ndarray[DTYPE_t, ndim=1] lis_phot = args[6]
    cdef np.ndarray[DTYPE_t, ndim=1] lis_cool = args[7]
    cdef np.ndarray[DTYPE_t, ndim=1] lis_hot = args[8]
    cdef np.ndarray[DTYPE_t, ndim=1] areas = args[9]
    cdef np.ndarray[DTYPE_t, ndim=1] grid_lats = args[10]
    cdef np.ndarray[DTYPE_t, ndim=1] grid_longs = args[11]
    cdef int nop = args[12]
    cdef DTYPE_t t0 = args[13]
    cdef DTYPE_t incl = args[14]
    cdef str ld_law = args[15]
    cdef int noes = args[16]
    cdef np.ndarray[DTYPE_t, ndim=1] lp_vels = args[17]
    cdef np.ndarray[DTYPE_t, ndim=1] phot_lp_data = args[18]
    cdef np.ndarray[DTYPE_t, ndim=1] cool_lp_data = args[19]
    cdef np.ndarray[DTYPE_t, ndim=1] hot_lp_data = args[20]
    cdef DTYPE_t vrt = args[21]
    cdef np.ndarray[DTYPE_t, ndim=1] vels = args[22]
    cdef DTYPE_t period = args[23]
    cdef int info = args[24]

    cdef np.ndarray[DTYPE_t, ndim=2] coeffs_cube = np.zeros((noes, 3 + nop * 3), dtype=np.float64)
    cdef int i
    cdef DTYPE_t nlong, iepoch
    cdef DTYPE_t ldf_phot, ldf_cool, ldf_hot, mu, cosincl = cos(incl), sinincl = sin(incl)

    if info:
        iepoch = (itime - t0) / period
        print('\033[94m  mid-time = {0:.5f} - mid-epoch = {1:.5f} \033[0m'.format(itime, iepoch))

    for i in range(noes):

        nlong = grid_longs[i] + 2.0 * pi * (itime - t0) / plats[i]
        mu = sin(grid_lats[i]) * cosincl + cos(grid_lats[i]) * sinincl * cos(nlong)

        if mu > 0.0:

            ldf_phot = ld_factors_calc(ld_law, mu, ldcs_phot[i, 0], ldcs_phot[i, 1])
            ldf_cool = ld_factors_calc(ld_law, mu, ldcs_cool[i, 0], ldcs_cool[i, 1])
            ldf_hot = ld_factors_calc(ld_law, mu, ldcs_hot[i, 0], ldcs_hot[i, 1])

            coeffs_cube[i, 0] = lis_phot[i] * ldf_phot * areas[i] * mu
            coeffs_cube[i, 1] = lis_cool[i] * ldf_cool * areas[i] * mu
            coeffs_cube[i, 2] = lis_hot[i] * ldf_hot * areas[i] * mu


            dv = vlats[i] * sin(nlong) * sinincl

            pcm_phot, pnmk = apply_macro_conv(lp_vels, phot_lp_data, mu, vrt)
            pcm_cool, cnmk = apply_macro_conv(lp_vels, cool_lp_data, mu, vrt)
            pcm_hot, hnmk = apply_macro_conv(lp_vels, hot_lp_data, mu, vrt)

            lp_phot = np.interp(vels, lp_vels + dv, pcm_phot[pnmk: -pnmk])
            lp_cool = np.interp(vels, lp_vels + dv, pcm_cool[cnmk: -cnmk])
            lp_hot = np.interp(vels, lp_vels + dv, pcm_hot[hnmk: -hnmk])

            # lp_phot = linear_interpolation(lp_vels + dv, pcm_phot[pnmk: -pnmk], vels)
            # lp_cool = linear_interpolation(lp_vels + dv, pcm_cool[cnmk: -cnmk], vels)
            # lp_hot = linear_interpolation(lp_vels + dv, pcm_hot[hnmk: -hnmk], vels)

            coeffs_cube[i, 3: 3 + nop] = lp_phot
            coeffs_cube[i, 3 + nop: 3 + 2 * nop] = lp_cool
            coeffs_cube[i, 3 + 2 * nop:] = lp_hot

    return coeffs_cube

cpdef np.ndarray[DTYPE_t, ndim=2] calc_pixel_coeffs_lc(tuple args):

    cdef DTYPE_t itime = args[0]
    cdef np.ndarray[DTYPE_t, ndim=1] plats = args[1]
    cdef np.ndarray[DTYPE_t, ndim=2] ldcs_phot = args[2]
    cdef np.ndarray[DTYPE_t, ndim=2] ldcs_cool = args[3]
    cdef np.ndarray[DTYPE_t, ndim=2] ldcs_hot = args[4]
    cdef np.ndarray[DTYPE_t, ndim=1] lis_phot = args[5]
    cdef np.ndarray[DTYPE_t, ndim=1] lis_cool = args[6]
    cdef np.ndarray[DTYPE_t, ndim=1] lis_hot = args[7]
    cdef np.ndarray[DTYPE_t, ndim=1] areas = args[8]
    cdef np.ndarray[DTYPE_t, ndim=1] grid_lats = args[9]
    cdef np.ndarray[DTYPE_t, ndim=1] grid_longs = args[10]
    cdef DTYPE_t t0 = args[11]
    cdef DTYPE_t incl = args[12]
    cdef str ld_law = args[13]
    cdef int noes = args[14]
    cdef DTYPE_t period = args[15]
    cdef int info = args[16]

    cdef np.ndarray[DTYPE_t, ndim=2] coeffs_cube = np.zeros((noes, 3), dtype=np.float64)
    cdef int i
    cdef DTYPE_t nlong
    cdef DTYPE_t ldf_phot, ldf_cool, ldf_hot, mu, cosincl = cos(incl), sinincl = sin(incl)

    if info:
        iepoch = (itime - t0) / period
        print('\033[94m  mid-time = {0:.5f} - mid-epoch = {1:.5f} \033[0m'.format(itime, iepoch))

    for i in range(noes):

        nlong = grid_longs[i] + 2.0 * pi * (itime - t0) / plats[i]
        mu = sin(grid_lats[i]) * cosincl + cos(grid_lats[i]) * sinincl * cos(nlong)

        if mu > 0.0:

            ldf_phot = ld_factors_calc(ld_law, mu, ldcs_phot[i, 0], ldcs_phot[i, 1])
            ldf_cool = ld_factors_calc(ld_law, mu, ldcs_cool[i, 0], ldcs_cool[i, 1])
            ldf_hot = ld_factors_calc(ld_law, mu, ldcs_hot[i, 0], ldcs_hot[i, 1])

            coeffs_cube[i, 0] = lis_phot[i] * ldf_phot * areas[i] * mu
            coeffs_cube[i, 1] = lis_cool[i] * ldf_cool * areas[i] * mu
            coeffs_cube[i, 2] = lis_hot[i] * ldf_hot * areas[i] * mu

    return coeffs_cube

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def subdivision(np.ndarray[DTYPE_t, ndim=3] xyzs, np.ndarray[DTYPE_t, ndim=1] fss, int subno=1):
#     cdef np.ndarray[DTYPE_t, ndim=3] xyzsb = xyzs.copy()
#     cdef np.ndarray[DTYPE_t, ndim=1] fssb = fss.copy()
#     cdef int i, j, k, n, m
#     cdef double x1, y1, z1, x2, y2, z2, x3, y3, z3
#     cdef double xa, ya, za, xb, yb, zb, xc, yc, zc
#     cdef np.ndarray[DTYPE_t, ndim=2] tri
#     cdef np.ndarray[DTYPE_t, ndim=2] new_tri
#     cdef np.ndarray[DTYPE_t, ndim=1] xcs, ycs, zcs
#     cdef list tri_all, rd_fss
#
#     for i in range(subno):
#         rd_fss = []
#         tri_all = []
#         for j in range(len(xyzsb)):
#             x1, y1, z1 = xyzsb[j, 0, 0], xyzsb[j, 0, 1], xyzsb[j, 0, 2]
#             x2, y2, z2 = xyzsb[j, 1, 0], xyzsb[j, 1, 1], xyzsb[j, 1, 2]
#             x3, y3, z3 = xyzsb[j, 2, 0], xyzsb[j, 2, 1], xyzsb[j, 2, 2]
#
#             xa, ya, za = (x1 + x2) / 2., (y1 + y2) / 2., (z1 + z2) / 2.
#             xb, yb, zb = (x1 + x3) / 2., (y1 + y3) / 2., (z1 + z3) / 2.
#             xc, yc, zc = (x2 + x3) / 2., (y2 + y3) / 2., (z2 + z3) / 2.
#
#             tri_all.append(np.array([[x1, y1, z1], [xa, ya, za], [xb, yb, zb]], dtype=np.float64))
#             tri_all.append(np.array([[xa, ya, za], [x2, y2, z2], [xc, yc, zc]], dtype=np.float64))
#             tri_all.append(np.array([[xc, yc, zc], [x3, y3, z3], [xb, yb, zb]], dtype=np.float64))
#             tri_all.append(np.array([[xa, ya, za], [xb, yb, zb], [xc, yc, zc]], dtype=np.float64))
#
#             rd_fss.append([fssb[j]] * 4)
#
#         fssb = np.hstack(rd_fss)
#         xyzsb = np.array(tri_all, dtype=np.float64)
#
#     n = len(xyzsb)
#     rs = np.empty(n, dtype=np.float64)
#     lats = np.empty(n, dtype=np.float64)
#     longs = np.empty(n, dtype=np.float64)
#
#     for j in range(n):
#         tri = xyzsb[j]
#         xc = (tri[0, 0] + tri[1, 0] + tri[2, 0]) / 3.
#         yc = (tri[0, 1] + tri[1, 1] + tri[2, 1]) / 3.
#         zc = (tri[0, 2] + tri[1, 2] + tri[2, 2]) / 3.
#
#         r, lat, long = cartesian_to_spherical(xc, yc, zc)
#
#         rs[j] = r
#         lats[j] = lat
#         longs[j] = long
#
#     return xyzsb, fssb, r, lats, longs

def subdivision(np.ndarray[DTYPE_t, ndim=3] xyzs, np.ndarray[DTYPE_t, ndim=1] fss, int subno=1):
    cdef np.ndarray[DTYPE_t, ndim=3] xyzsb = xyzs.copy()
    cdef np.ndarray[DTYPE_t, ndim=1] fssb = fss.copy()
    cdef int i, j, n
    cdef double[:, ::1] midpoints
    cdef np.ndarray[DTYPE_t, ndim=2] tri
    cdef np.ndarray[DTYPE_t, ndim=3] new_xyzsb
    cdef np.ndarray[DTYPE_t, ndim=1] new_fssb

    for i in range(subno):
        n = len(xyzsb) * 4
        new_xyzsb = np.empty((n, 3, 3), dtype=np.float64)
        new_fssb = np.repeat(fssb, 4)

        for j in range(len(xyzsb)):
            tri = xyzsb[j]
            x1, y1, z1 = tri[0]
            x2, y2, z2 = tri[1]
            x3, y3, z3 = tri[2]

            xa, ya, za = (x1 + x2) / 2., (y1 + y2) / 2., (z1 + z2) / 2.
            xb, yb, zb = (x1 + x3) / 2., (y1 + y3) / 2., (z1 + z3) / 2.
            xc, yc, zc = (x2 + x3) / 2., (y2 + y3) / 2., (z2 + z3) / 2.

            new_xyzsb[j * 4 + 0] = [[x1, y1, z1], [xa, ya, za], [xb, yb, zb]]
            new_xyzsb[j * 4 + 1] = [[xa, ya, za], [x2, y2, z2], [xc, yc, zc]]
            new_xyzsb[j * 4 + 2] = [[xc, yc, zc], [x3, y3, z3], [xb, yb, zb]]
            new_xyzsb[j * 4 + 3] = [[xa, ya, za], [xb, yb, zb], [xc, yc, zc]]

        xyzsb = new_xyzsb
        fssb = new_fssb

    n = len(xyzsb)
    rs = np.empty(n, dtype=np.float64)
    lats = np.empty(n, dtype=np.float64)
    longs = np.empty(n, dtype=np.float64)

    # Tüm üçgenler için ortalama koordinatları hesapla
    xcs = (xyzsb[:, 0, 0] + xyzsb[:, 1, 0] + xyzsb[:, 2, 0]) / 3
    ycs = (xyzsb[:, 0, 1] + xyzsb[:, 1, 1] + xyzsb[:, 2, 1]) / 3
    zcs = (xyzsb[:, 0, 2] + xyzsb[:, 1, 2] + xyzsb[:, 2, 2]) / 3

    # Kartesyen'den küresel koordinatlara dönüştürme
    rs, lats, longs = cartesian_to_spherical_vectorized(xcs, ycs, zcs)

    return xyzsb, fssb, rs, lats, longs


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def subdivision(np.ndarray[DTYPE_t, ndim=3] xyzs, np.ndarray[DTYPE_t, ndim=1] fss, int subno=1):
#     cdef np.ndarray[DTYPE_t, ndim=3] xyzsb = xyzs.copy()
#     cdef np.ndarray[DTYPE_t, ndim=1] fssb = fss.copy()
#     cdef int i, j
#     cdef double x1, y1, z1, x2, y2, z2, x3, y3, z3
#     cdef double xa, ya, za, xb, yb, zb, xc, yc, zc
#     cdef list tri_all, rd_fss
#     cdef np.ndarray[DTYPE_t, ndim=2] tri
#     cdef np.ndarray[DTYPE_t, ndim=2] new_tri
#     cdef np.ndarray[DTYPE_t, ndim=1] xcs, ycs, zcs
#     cdef double xc, yc, zc
#
#     for i in range(subno):
#         rd_fss = []
#         tri_all = []
#         for j in range(len(xyzsb)):
#             x1, y1, z1 = xyzsb[j, 0, 0], xyzsb[j, 0, 1], xyzsb[j, 0, 2]
#             x2, y2, z2 = xyzsb[j, 1, 0], xyzsb[j, 1, 1], xyzsb[j, 1, 2]
#             x3, y3, z3 = xyzsb[j, 2, 0], xyzsb[j, 2, 1], xyzsb[j, 2, 2]
#
#             xa, ya, za = (x1 + x2) / 2., (y1 + y2) / 2., (z1 + z2) / 2.
#             xb, yb, zb = (x1 + x3) / 2., (y1 + y3) / 2., (z1 + z3) / 2.
#             xc, yc, zc = (x2 + x3) / 2., (y2 + y3) / 2., (z2 + z3) / 2.
#
#             tri_all.append(np.array([[x1, y1, z1], [xa, ya, za], [xb, yb, zb]], dtype=DTYPE_t))
#             tri_all.append(np.array([[xa, ya, za], [x2, y2, z2], [xc, yc, zc]], dtype=DTYPE_t))
#             tri_all.append(np.array([[xc, yc, zc], [x3, y3, z3], [xb, yb, zb]], dtype=DTYPE_t))
#             tri_all.append(np.array([[xa, ya, za], [xb, yb, zb], [xc, yc, zc]], dtype=DTYPE_t))
#
#             rd_fss.append([fssb[j]] * 4)
#
#         fssb = np.hstack(rd_fss)
#         xyzsb = np.array(tri_all, dtype=DTYPE_t)
#
#     xcs = np.empty(len(xyzsb), dtype=DTYPE_t)
#     ycs = np.empty(len(xyzsb), dtype=DTYPE_t)
#     zcs = np.empty(len(xyzsb), dtype=DTYPE_t)
#
#     for j in range(len(xyzsb)):
#         tri = xyzsb[j]
#         xc = (tri[0, 0] + tri[1, 0] + tri[2, 0]) / 3.
#         yc = (tri[0, 1] + tri[1, 1] + tri[2, 1]) / 3.
#         zc = (tri[0, 2] + tri[1, 2] + tri[2, 2]) / 3.
#
#         xcs[j] = xc
#         ycs[j] = yc
#         zcs[j] = zc
#
#
#     return , fssb

# cpdef tuple macro_kernel(int nop, double muval, double vrt, double deltav):
#
#     cdef int os
#     cdef double area_r = 0.5
#     cdef double area_t = 0.5
#     cdef double sigma, sigr, sigt, nfine
#     cdef int nmk
#
#     cdef np.ndarray[np.double_t, ndim=1] xarg_r, exparg_r, xarg_t, exparg_t, mrkern, mtkern, mkern
#
#     # cdef double[:] xarg_r, exparg_r, xarg_t, exparg_t, mrkern, mtkern, mkern
#
#     os = 1
#     sigma = os * vrt / np.sqrt(2.0) / deltav
#     sigr = sigma * muval
#     sigt = sigma * np.sqrt(1.0 - muval ** 2)
#     nfine = os * nop
#
#     nmk = max(int(10.0 * sigma), int((nfine - 3) / 2), 3)
#
#     if sigr > 0.0:
#         xarg_r = (np.arange(2 * nmk + 1, dtype=np.int) - nmk) / sigr
#         exparg_r = (-0.5 * np.power(xarg_r, 2))
#         exparg_r = np.where(exparg_r < -20.0, -20.0, exparg_r)
#         mrkern = np.exp(exparg_r)
#         mrkern = mrkern / np.sum(mrkern)
#     else:
#         mrkern = np.zeros(2 * nmk + 1)
#         mrkern[nmk] = 1.0
#
#     if sigt > 0.0:
#         xarg_t = (np.arange(2 * nmk + 1, dtype=np.int) - nmk) / sigt
#         exparg_t = (-0.5 * np.power(xarg_t, 2))
#         exparg_t = np.where(exparg_t < -20.0, -20.0, exparg_t)
#         mtkern = np.exp(exparg_t)
#         mtkern = mtkern / np.sum(mtkern)
#     else:
#         mtkern = np.zeros(2 * nmk + 1)
#         mtkern[nmk] = 1.0
#
#     mkern = area_r * np.array(mrkern) + area_t * np.array(mtkern)
#
#     return mkern, nmk
