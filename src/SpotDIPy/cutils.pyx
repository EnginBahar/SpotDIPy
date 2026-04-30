# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sin, asin, cos, atan2, log10, sqrt, fabs, pi, exp
from libc.stdlib cimport malloc, free

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# ── Constants ──
cdef DTYPE_t solMass = 1.988409870698051e+30  # kg
cdef DTYPE_t solRad = 695700000.0             # m
cdef DTYPE_t G = 6.6743e-11                   # m3 / (kg s2)

# ── LD law enum (avoid string comparison in hot loops) ──
DEF LD_LINEAR = 0
DEF LD_SQRT = 1
DEF LD_QUAD = 2


cpdef int ld_law_to_int(str ld_law):
    if ld_law == 'linear':
        return LD_LINEAR
    elif ld_law == 'square-root':
        return LD_SQRT
    elif ld_law == 'quadratic':
        return LD_QUAD


# ── Core math (all inlined via cdef) ──

cdef inline void spherical_to_cartesian_c(DTYPE_t r, DTYPE_t lat, DTYPE_t lon,
                                           DTYPE_t *x, DTYPE_t *y, DTYPE_t *z) noexcept nogil:
    cdef DTYPE_t cos_lat = cos(lat)
    x[0] = r * cos_lat * cos(lon)
    y[0] = r * cos_lat * sin(lon)
    z[0] = r * sin(lat)


cpdef tuple spherical_to_cartesian(DTYPE_t r, DTYPE_t lat, DTYPE_t lon):
    cdef DTYPE_t x, y, z
    spherical_to_cartesian_c(r, lat, lon, &x, &y, &z)
    return x, y, z


cdef inline DTYPE_t r_func_c(DTYPE_t r, DTYPE_t inv_omega2, DTYPE_t sin2theta) noexcept nogil:
    return inv_omega2 + 0.5 - inv_omega2 / r - 0.5 * r * r * sin2theta


cdef inline DTYPE_t deriv_r_func_c(DTYPE_t r, DTYPE_t inv_omega2, DTYPE_t sin2theta) noexcept nogil:
    return inv_omega2 / (r * r) - sin2theta * r


cdef inline DTYPE_t newtonRaphson_c(DTYPE_t x0, DTYPE_t inv_omega2, DTYPE_t sin2theta,
                                     DTYPE_t tol, int *converged) noexcept nogil:
    cdef DTYPE_t h, f, df
    cdef int maxiter = 100
    cdef int it

    converged[0] = 0
    for it in range(maxiter):
        f = r_func_c(x0, inv_omega2, sin2theta)
        df = deriv_r_func_c(x0, inv_omega2, sin2theta)
        h = f / df
        x0 = x0 - h
        if fabs(h) < tol:
            converged[0] = 1
            break
    return x0


cdef inline DTYPE_t calc_r_to_Re_ratio_c(DTYPE_t inv_omega2, DTYPE_t sin2theta) noexcept nogil:
    cdef int converged
    cdef DTYPE_t result

    result = newtonRaphson_c(1.0, inv_omega2, sin2theta, 1.0e-8, &converged)
    if not converged:
        raise RuntimeError(
            f"Newton-Raphson did not converge in 100 iterations: x0={1.0}, "
            f"omega={sqrt(inv_omega2):.6e}, theta={asin(sqrt(sin2theta)):.6e}, tol={1.0e-8}"
        )
    return result


# ── Public wrappers (backward compat) ──

cpdef DTYPE_t r_func(DTYPE_t r, DTYPE_t omega, DTYPE_t theta):
    cdef DTYPE_t s = sin(theta)
    return r_func_c(r, 1.0 / (omega * omega), s * s)


cpdef DTYPE_t deriv_r_func(DTYPE_t r, DTYPE_t omega, DTYPE_t theta):
    cdef DTYPE_t s = sin(theta)
    return deriv_r_func_c(r, 1.0 / (omega * omega), s * s)


cpdef DTYPE_t newtonRaphson(DTYPE_t x0, DTYPE_t omega, DTYPE_t theta, DTYPE_t tol=1.0e-8):
    cdef DTYPE_t s = sin(theta)
    cdef int converged
    cdef DTYPE_t result

    result = newtonRaphson_c(x0, 1.0 / (omega * omega), s * s, tol, &converged)
    if not converged:
        raise RuntimeError(
            f"Newton-Raphson did not converge in 100 iterations: x0={x0:.6e}, "
            f"omega={omega:.6e}, theta={theta:.6e}, tol={tol:.2e}"
        )
    return result

cpdef DTYPE_t calc_r_to_Re_ratio(DTYPE_t omega, DTYPE_t theta):
    cdef DTYPE_t s = sin(theta)
    return calc_r_to_Re_ratio_c(1.0 / (omega * omega), s * s)


# ── Triangle area with pure C math (no numpy) ──

cdef inline DTYPE_t calc_triangle_area_c(DTYPE_t x0, DTYPE_t y0, DTYPE_t z0,
                                          DTYPE_t x1, DTYPE_t y1, DTYPE_t z1,
                                          DTYPE_t x2, DTYPE_t y2, DTYPE_t z2) noexcept nogil:
    cdef DTYPE_t ux = x1 - x0, uy = y1 - y0, uz = z1 - z0
    cdef DTYPE_t vx = x2 - x0, vy = y2 - y0, vz = z2 - z0
    cdef DTYPE_t cx = uy * vz - uz * vy
    cdef DTYPE_t cy = uz * vx - ux * vz
    cdef DTYPE_t cz = ux * vy - uy * vx
    return 0.5 * sqrt(cx * cx + cy * cy + cz * cz)


cpdef DTYPE_t calc_triangle_area(DTYPE_t[:, :] triangle):
    return calc_triangle_area_c(
        triangle[0, 0], triangle[0, 1], triangle[0, 2],
        triangle[1, 0], triangle[1, 1], triangle[1, 2],
        triangle[2, 0], triangle[2, 1], triangle[2, 2])


# ── logg ──

cdef inline DTYPE_t calc_logg_c(DTYPE_t mass, DTYPE_t re, DTYPE_t rs,
                                 DTYPE_t omega, DTYPE_t theta) noexcept nogil:
    cdef DTYPE_t mass_kg = mass * solMass
    cdef DTYPE_t g_cm = mass_kg * G * 1.0e6
    cdef DTYPE_t r_cm = re * solRad * 100.0
    cdef DTYPE_t r_ = rs / re
    cdef DTYPE_t omega2 = omega * omega
    cdef DTYPE_t sintheta

    if theta > pi / 2.0:
        theta = pi - theta
    sintheta = sin(theta)

    cdef DTYPE_t geff = (g_cm / (r_cm * r_cm)) * sqrt(
        1.0 / (r_ * r_ * r_ * r_) +
        omega2 * omega2 * r_ * r_ * sintheta * sintheta -
        2.0 * omega2 * sintheta * sintheta / r_)
    return log10(geff)


cpdef DTYPE_t calc_logg(DTYPE_t mass, DTYPE_t re, DTYPE_t rs, DTYPE_t omega, DTYPE_t theta):
    return calc_logg_c(mass, re, rs, omega, theta)


# ── Limb darkening (integer dispatch, no string compare) ──

cdef inline DTYPE_t ld_factors_c(int ld_law, DTYPE_t mu,
                                  DTYPE_t ldc1, DTYPE_t ldc2) noexcept nogil:
    cdef DTYPE_t omu = 1.0 - mu
    if ld_law == LD_LINEAR:
        return 1.0 - ldc1 * omu
    elif ld_law == LD_SQRT:
        return 1.0 - ldc1 * omu - ldc2 * (1.0 - sqrt(mu))
    elif ld_law == LD_QUAD:
        return 1.0 - ldc1 * omu - ldc2 * omu * omu


cpdef DTYPE_t ld_factors_calc(str ld_law, DTYPE_t mu, DTYPE_t ldc1, DTYPE_t ldc2):
    return ld_factors_c(ld_law_to_int(ld_law), mu, ldc1, ldc2)


# ── Surface grid (minimized Python object creation) ──

cpdef tuple td_grid_sect(tuple args):
    cdef DTYPE_t blat0 = args[0][0]
    cdef DTYPE_t blat1 = args[0][1]
    cdef DTYPE_t uarea = args[1]
    cdef DTYPE_t omega = args[2]
    cdef DTYPE_t radius = args[3]
    cdef DTYPE_t mass = args[4]

    cdef DTYPE_t inv_omega2 = 1.0 / (omega * omega)
    cdef DTYPE_t mid_lat = (blat0 + blat1) / 2.0
    cdef DTYPE_t mid_theta = mid_lat + pi / 2.0
    cdef DTYPE_t sin_mt = sin(mid_theta)

    cdef int nlon = <int>(round(2.0 * pi * radius * radius * fabs(sin(blat0) - sin(blat1)) / uarea))
    if nlon < 1:
        nlon = 1

    cdef DTYPE_t dlon = 2.0 * pi / nlon
    cdef DTYPE_t rzlat = calc_r_to_Re_ratio_c(inv_omega2, sin_mt * sin_mt) * radius

    # Pre-compute r for the two latitude boundaries
    cdef DTYPE_t sin_t0 = sin(blat0 + pi / 2.0)
    cdef DTYPE_t sin_t1 = sin(blat1 + pi / 2.0)
    cdef DTYPE_t r_lat0 = calc_r_to_Re_ratio_c(inv_omega2, sin_t0 * sin_t0) * radius
    cdef DTYPE_t r_lat1 = calc_r_to_Re_ratio_c(inv_omega2, sin_t1 * sin_t1) * radius

    # Output arrays (pre-allocated numpy)
    cdef np.ndarray[DTYPE_t, ndim=1] blong = np.empty(nlon + 1, dtype=DTYPE)
    cdef int k
    for k in range(nlon + 1):
        blong[k] = k * dlon

    cdef np.ndarray[DTYPE_t, ndim=1] rs_arr = np.empty(nlon, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] lats_arr = np.empty(nlon, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] longs_arr = np.empty(nlon, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] loggs_arr = np.empty(nlon, dtype=DTYPE)

    cdef list areas = []
    cdef list xyzs = []

    # Triangle vertex coords
    cdef DTYPE_t px[3]
    cdef DTYPE_t py[3]
    cdef DTYPE_t pz[3]
    cdef DTYPE_t r_vals[2]
    r_vals[0] = r_lat0
    r_vals[1] = r_lat1
    cdef DTYPE_t lat_vals[2]
    lat_vals[0] = blat0
    lat_vals[1] = blat1

    # Triangle index patterns: tri0 = [(0,0),(1,0),(0,1)], tri1 = [(1,0),(1,1),(0,1)]
    cdef int tri_lat[2][3]
    cdef int tri_lon[2][3]
    tri_lat[0][0] = 0; tri_lat[0][1] = 1; tri_lat[0][2] = 0
    tri_lat[1][0] = 1; tri_lat[1][1] = 1; tri_lat[1][2] = 0
    tri_lon[0][0] = 0; tri_lon[0][1] = 0; tri_lon[0][2] = 1
    tri_lon[1][0] = 0; tri_lon[1][1] = 1; tri_lon[1][2] = 1

    cdef int j, i, vi
    cdef DTYPE_t lon0, lon1, rlat_v, lat_v, lon_v
    cdef DTYPE_t area_val, x, y, z
    cdef DTYPE_t mid_lon

    for j in range(nlon):
        lon0 = blong[j]
        lon1 = blong[j + 1]
        mid_lon = (lon0 + lon1) / 2.0

        for i in range(2):
            for vi in range(3):
                rlat_v = r_vals[tri_lat[i][vi]]
                lat_v = lat_vals[tri_lat[i][vi]]
                if tri_lon[i][vi] == 0:
                    lon_v = lon0
                else:
                    lon_v = lon1
                spherical_to_cartesian_c(rlat_v, lat_v, lon_v, &px[vi], &py[vi], &pz[vi])

            area_val = calc_triangle_area_c(px[0], py[0], pz[0],
                                            px[1], py[1], pz[1],
                                            px[2], py[2], pz[2])
            areas.append(area_val)
            xyzs.append(np.array([[px[0], py[0], pz[0]],
                                  [px[1], py[1], pz[1]],
                                  [px[2], py[2], pz[2]]]))

        rs_arr[j] = rzlat
        lats_arr[j] = mid_lat
        longs_arr[j] = mid_lon
        loggs_arr[j] = calc_logg_c(mass, radius, rzlat, omega, mid_theta)

    return nlon, blong, areas, rs_arr, lats_arr, longs_arr, xyzs, loggs_arr


cpdef list td_surface_grid(DTYPE_t[:] blats, DTYPE_t omega, DTYPE_t radius, DTYPE_t mass):
    cdef int j
    cdef int ipn = 4
    cdef int nlats = blats.shape[0]
    cdef DTYPE_t pzh = fabs(sin(blats[0]) - sin(blats[1])) * radius
    cdef DTYPE_t uarea = 2.0 * pi * radius * pzh / ipn
    cdef list results = []

    for j in range(nlats - 1):
        results.append(td_grid_sect(([blats[j], blats[j + 1]], uarea, omega, radius, mass)))

    return results


# ── Macro kernel (pure C arrays, no numpy in inner loops) ──

cpdef tuple macro_kernel(int nop, DTYPE_t muval, DTYPE_t vrt, DTYPE_t deltav):
    cdef DTYPE_t sigma = vrt / sqrt(2.0) / deltav
    cdef DTYPE_t sigr = sigma * muval
    cdef DTYPE_t sigt = sigma * sqrt(1.0 - muval * muval)
    cdef int nmk = 3
    cdef DTYPE_t temp1, temp2
    cdef int size, i
    cdef DTYPE_t val, mrsum, mtsum

    temp1 = 10.0 * sigma
    if temp1 > nmk:
        nmk = <int>temp1

    temp2 = (<DTYPE_t>nop - 3.0) / 2.0
    if temp2 > nmk:
        nmk = <int>temp2

    if nmk < 3:
        nmk = 3

    size = 2 * nmk + 1
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(size, dtype=DTYPE)
    cdef DTYPE_t *mrkern = <DTYPE_t *>malloc(size * sizeof(DTYPE_t))
    cdef DTYPE_t *mtkern = <DTYPE_t *>malloc(size * sizeof(DTYPE_t))

    if mrkern == NULL or mtkern == NULL:
        if mrkern != NULL:
            free(mrkern)
        if mtkern != NULL:
            free(mtkern)
        raise MemoryError()

    # Radial kernel
    mrsum = 0.0
    if sigr > 0.0:
        for i in range(size):
            val = -0.5 * ((<DTYPE_t>(i - nmk)) / sigr) ** 2
            if val < -20.0:
                val = -20.0
            mrkern[i] = exp(val)
            mrsum += mrkern[i]
        for i in range(size):
            mrkern[i] /= mrsum
    else:
        for i in range(size):
            mrkern[i] = 0.0
        mrkern[nmk] = 1.0

    # Tangential kernel
    mtsum = 0.0
    if sigt > 0.0:
        for i in range(size):
            val = -0.5 * ((<DTYPE_t>(i - nmk)) / sigt) ** 2
            if val < -20.0:
                val = -20.0
            mtkern[i] = exp(val)
            mtsum += mtkern[i]
        for i in range(size):
            mtkern[i] /= mtsum
    else:
        for i in range(size):
            mtkern[i] = 0.0
        mtkern[nmk] = 1.0

    # Combine
    for i in range(size):
        result[i] = (mrkern[i] + mtkern[i]) / 2.0

    free(mrkern)
    free(mtkern)

    return result, nmk


cpdef apply_macro_conv(DTYPE_t[:] vels, DTYPE_t[:] ints, DTYPE_t mu, DTYPE_t vrt):
    cdef DTYPE_t deltav = vels[1] - vels[0]
    mkernel, nmk = macro_kernel(vels.shape[0], mu, vrt, deltav)
    conv_mac = np.convolve(np.pad(np.asarray(ints), (nmk, nmk), 'edge'), mkernel, mode='same')
    return conv_mac, nmk


# ── Pixel coefficients (spectroscopic) ──

cpdef np.ndarray[DTYPE_t, ndim=2] calc_pixel_coeffs_spec(tuple args):
    cdef DTYPE_t itime = args[0]
    cdef DTYPE_t[:] plats = args[1]
    cdef DTYPE_t[:] vlats = args[2]
    cdef DTYPE_t[:, :] ldcs_phot = args[3]
    cdef DTYPE_t[:, :] ldcs_cool = args[4]
    cdef DTYPE_t[:, :] ldcs_hot = args[5]
    cdef DTYPE_t[:] lis_phot = args[6]
    cdef DTYPE_t[:] lis_cool = args[7]
    cdef DTYPE_t[:] lis_hot = args[8]
    cdef DTYPE_t[:] areas = args[9]
    cdef DTYPE_t[:] grid_lats = args[10]
    cdef DTYPE_t[:] grid_longs = args[11]
    cdef int nop = args[12]
    cdef DTYPE_t t0 = args[13]
    cdef DTYPE_t incl = args[14]
    cdef str ld_law = args[15]
    cdef int noes = args[16]
    cdef DTYPE_t[:] lp_vels = args[17]
    cdef DTYPE_t[:] phot_lp_data = args[18]
    cdef DTYPE_t[:] cool_lp_data = args[19]
    cdef DTYPE_t[:] hot_lp_data = args[20]
    cdef DTYPE_t vrt = args[21]
    cdef DTYPE_t[:] vels = args[22]
    cdef DTYPE_t period = args[23]
    cdef int count = args[24]
    cdef int info = args[25]

    cdef int ld_int = ld_law_to_int(ld_law)
    cdef np.ndarray[DTYPE_t, ndim=2] coeffs_cube = np.zeros((noes, 3 + nop * 3), dtype=np.float64)
    cdef DTYPE_t[:, :] cc = coeffs_cube  # memoryview for fast access

    cdef int i
    cdef DTYPE_t nlong, dv, mu
    cdef DTYPE_t ldf_phot, ldf_cool, ldf_hot
    cdef DTYPE_t cosincl = cos(incl), sinincl = sin(incl)
    cdef DTYPE_t two_pi_dt_inv
    cdef DTYPE_t area_mu

    two_pi_dt_inv = 2.0 * pi * (itime - t0)

    if info:
        iepoch = (itime - t0) / period
        print(f'\033[94m {count + 1:d}- mid-time = {itime:.8f} - mid-epoch = {iepoch:.8f} \033[0m')

    # Pre-compute macro convolution once (same for all pixels)
    cdef DTYPE_t deltav = lp_vels[1] - lp_vels[0]
    pcm_phot_arr, pnmk = macro_kernel(lp_vels.shape[0], 1.0, vrt, deltav)
    # We still need per-pixel mu-dependent convolution, but we can cache kernel for mu~1

    cdef np.ndarray[DTYPE_t, ndim=1] vels_np = np.asarray(vels)
    cdef np.ndarray[DTYPE_t, ndim=1] lp_vels_np = np.asarray(lp_vels)
    cdef np.ndarray[DTYPE_t, ndim=1] phot_lp_np = np.asarray(phot_lp_data)
    cdef np.ndarray[DTYPE_t, ndim=1] cool_lp_np = np.asarray(cool_lp_data)
    cdef np.ndarray[DTYPE_t, ndim=1] hot_lp_np = np.asarray(hot_lp_data)

    for i in range(noes):
        nlong = grid_longs[i] + two_pi_dt_inv / plats[i]
        mu = sin(grid_lats[i]) * cosincl + cos(grid_lats[i]) * sinincl * cos(nlong)

        if mu > 0.0:
            ldf_phot = ld_factors_c(ld_int, mu, ldcs_phot[i, 0], ldcs_phot[i, 1])
            ldf_cool = ld_factors_c(ld_int, mu, ldcs_cool[i, 0], ldcs_cool[i, 1])
            ldf_hot = ld_factors_c(ld_int, mu, ldcs_hot[i, 0], ldcs_hot[i, 1])

            area_mu = areas[i] * mu
            cc[i, 0] = lis_phot[i] * ldf_phot * area_mu
            cc[i, 1] = lis_cool[i] * ldf_cool * area_mu
            cc[i, 2] = lis_hot[i] * ldf_hot * area_mu

            dv = vlats[i] * sin(nlong) * sinincl

            pcm_phot, pnmk = apply_macro_conv(lp_vels, phot_lp_data, mu, vrt)
            pcm_cool, cnmk = apply_macro_conv(lp_vels, cool_lp_data, mu, vrt)
            pcm_hot, hnmk = apply_macro_conv(lp_vels, hot_lp_data, mu, vrt)

            lp_phot = np.interp(vels_np, lp_vels_np + dv, pcm_phot[pnmk: -pnmk])
            lp_cool = np.interp(vels_np, lp_vels_np + dv, pcm_cool[cnmk: -cnmk])
            lp_hot = np.interp(vels_np, lp_vels_np + dv, pcm_hot[hnmk: -hnmk])

            coeffs_cube[i, 3: 3 + nop] = lp_phot
            coeffs_cube[i, 3 + nop: 3 + 2 * nop] = lp_cool
            coeffs_cube[i, 3 + 2 * nop:] = lp_hot

    return coeffs_cube


# ── Pixel coefficients (light curve) ──

cpdef np.ndarray[DTYPE_t, ndim=2] calc_pixel_coeffs_lc(tuple args):
    cdef DTYPE_t itime = args[0]
    cdef DTYPE_t[:] plats = args[1]
    cdef DTYPE_t[:, :] ldcs_phot = args[2]
    cdef DTYPE_t[:, :] ldcs_cool = args[3]
    cdef DTYPE_t[:, :] ldcs_hot = args[4]
    cdef DTYPE_t[:] lis_phot = args[5]
    cdef DTYPE_t[:] lis_cool = args[6]
    cdef DTYPE_t[:] lis_hot = args[7]
    cdef DTYPE_t[:] areas = args[8]
    cdef DTYPE_t[:] grid_lats = args[9]
    cdef DTYPE_t[:] grid_longs = args[10]
    cdef DTYPE_t t0 = args[11]
    cdef DTYPE_t incl = args[12]
    cdef str ld_law = args[13]
    cdef int noes = args[14]
    cdef DTYPE_t period = args[15]
    cdef int count = args[16]
    cdef int info = args[17]

    cdef int ld_int = ld_law_to_int(ld_law)
    cdef np.ndarray[DTYPE_t, ndim=2] coeffs_cube = np.zeros((noes, 3), dtype=np.float64)
    cdef DTYPE_t[:, :] cc = coeffs_cube

    cdef int i
    cdef DTYPE_t nlong, mu, area_mu
    cdef DTYPE_t ldf_phot, ldf_cool, ldf_hot
    cdef DTYPE_t cosincl = cos(incl), sinincl = sin(incl)
    cdef DTYPE_t two_pi_dt_inv = 2.0 * pi * (itime - t0)

    if info:
        iepoch = (itime - t0) / period
        print(f'\033[94m {count + 1:d}- mid-time = {itime:.8f} - mid-epoch = {iepoch:.8f} \033[0m')

    for i in range(noes):
        nlong = grid_longs[i] + two_pi_dt_inv / plats[i]
        mu = sin(grid_lats[i]) * cosincl + cos(grid_lats[i]) * sinincl * cos(nlong)

        if mu > 0.0:
            ldf_phot = ld_factors_c(ld_int, mu, ldcs_phot[i, 0], ldcs_phot[i, 1])
            ldf_cool = ld_factors_c(ld_int, mu, ldcs_cool[i, 0], ldcs_cool[i, 1])
            ldf_hot = ld_factors_c(ld_int, mu, ldcs_hot[i, 0], ldcs_hot[i, 1])

            area_mu = areas[i] * mu
            cc[i, 0] = lis_phot[i] * ldf_phot * area_mu
            cc[i, 1] = lis_cool[i] * ldf_cool * area_mu
            cc[i, 2] = lis_hot[i] * ldf_hot * area_mu

    return coeffs_cube
