'''Standard Incomplete Market model with Endogenous Labor Supply'''

import numpy as np
from numba import vectorize, njit

from sequence_jacobian.blocks.het_block import het
from sequence_jacobian import interpolate, grids


def hh_init(a_grid, we, r, eis, T):
    fininc = (1 + r) * a_grid + T[:, np.newaxis] - a_grid[0]
    coh = (1 + r) * a_grid[np.newaxis, :] + we[:, np.newaxis] + T[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return fininc, Va

def finacial_cost(a, rborr):
    a_neg = a.copy()
    a_neg[a>0] = 0
    fin_cost = np.abs(a_neg*(rborr * a_neg**2))
    return fin_cost

@het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(Va_p, a_grid, we, T, r, beta, eis, frisch, vphi, rborr):
    '''Single backward step via EGM.'''

    # a_grid_neg = a_grid.copy()
    # a_grid_neg[a_grid > 0] = 0
    r_grid = a_grid.copy()
    r_grid[:] = r
    r_grid[a_grid<0] = r + rborr * a_grid[a_grid<0]**2

    uc_nextgrid = beta * Va_p
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, we[:, np.newaxis], eis, frisch, vphi)

    lhs = c_nextgrid - we[:, np.newaxis] * n_nextgrid + a_grid[np.newaxis, :] - T[:, np.newaxis]
    rhs = (1 + r_grid) * a_grid
    c = interpolate.interpolate_y(lhs, rhs, c_nextgrid)
    n = interpolate.interpolate_y(lhs, rhs, n_nextgrid)

    a = rhs + we[:, np.newaxis] * n + T[:, np.newaxis] - c
    iconst = np.nonzero(a < a_grid[0])
    a[iconst] = a_grid[0]

    if iconst[0].size != 0 and iconst[1].size != 0:
        c[iconst], n[iconst] = solve_cn(we[iconst[0]],
                                        rhs[iconst[1]] + T[iconst[0]] - a_grid[0],
                                        eis, frisch, vphi, Va_p[iconst])

    Va = (1 + r_grid) * c ** (-1 / eis)
    
    return Va, a, c, n


'''Supporting functions for HA block'''

@njit
def cn(uc, w, eis, frisch, vphi):
    """Return optimal c, n as function of u'(c) given parameters"""
    return uc ** (-eis), (w * uc / vphi) ** frisch


def solve_cn(w, T, eis, frisch, vphi, uc_seed):
    uc = solve_uc(w, T, eis, frisch, vphi, uc_seed)
    return cn(uc, w, eis, frisch, vphi)


@vectorize
def solve_uc(w, T, eis, frisch, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.

    max_{c, n} c**(1-1/eis) + vphi*n**(1+1/frisch) s.t. c = w*n + T
    """
    log_uc = np.log(uc_seed)
    for i in range(30):
        ne, ne_p = netexp(log_uc, w, T, eis, frisch, vphi)
        if abs(ne) < 1E-11:
            break
        else:
            log_uc -= ne / ne_p
    else:
        raise ValueError("Cannot solve constrained household's problem: No convergence after 30 iterations!")

    return np.exp(log_uc)


@njit
def netexp(log_uc, w, T, eis, frisch, vphi):
    """Return net expenditure as a function of log uc and its derivative."""
    c, n = cn(np.exp(log_uc), w, eis, frisch, vphi)
    ne = c - w * n - T

    # c and n have elasticities of -eis and frisch wrt log u'(c)
    c_loguc = -eis * c
    n_loguc = frisch * n
    netexp_loguc = c_loguc - w * n_loguc

    return ne, netexp_loguc
