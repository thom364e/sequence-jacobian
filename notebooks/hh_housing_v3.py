import numpy as np
from numba import guvectorize
from scipy.interpolate import interp1d
from numba import njit

# from ..src.sequence_jacobian.blocks.het_block import het
# from ..src.sequence_jacobian import interpolate

from sequence_jacobian.blocks.het_block import het
from sequence_jacobian import interpolate, grids

def hh_init(b_bhat_grid, h_bhat_grid, z_grid, sigma, theta):
    Vh_bhat = (0.6 + 1.1 * b_bhat_grid[:, np.newaxis] + h_bhat_grid) ** (-1/sigma) * np.ones((z_grid.shape[0], 1, 1))
    Vb_bhat = (0.5 + b_bhat_grid[:, np.newaxis] + 1.2 * h_bhat_grid) ** (-1/sigma) * np.ones((z_grid.shape[0], 1, 1))

    # Vh_bhat = (0.1 + 1.5 * b_bhat_grid[:, np.newaxis] + h_bhat_grid) ** (-1 / sigma) * np.ones((z_grid.shape[0], 1, 1))
    # Vb_bhat = (0.05 + b_bhat_grid[:, np.newaxis] + 1.6 * h_bhat_grid) ** (-1 / sigma) * np.ones((z_grid.shape[0], 1, 1))

    # Vh_bhat = (b_bhat_grid[:, np.newaxis] + h_bhat_grid) ** (-1 / sigma) * np.ones((z_grid.shape[0], 1, 1))
    # Vb_bhat = (b_bhat_grid[:, np.newaxis] + h_bhat_grid) ** (-1 / sigma) * np.ones((z_grid.shape[0], 1, 1))

    return Vh_bhat, Vb_bhat


def adjustment_costs_housing(h_bhat, h_bhat_grid, alpha):
    chi = get_PsiHousing_and_deriv(h_bhat, h_bhat_grid, alpha)[0]
    return chi


def marginal_cost_grid_housing(h_bhat_grid, alpha):
    # precompute Psi1(a', a) on grid of (a', a) for steps 3 and 5
    Psi1 = get_PsiHousing_and_deriv(h_bhat_grid[:, np.newaxis],
                                    h_bhat_grid[np.newaxis, :], alpha)[1]
    return Psi1

def finacial_cost(b_bhat, xi):
    # b_neg = b_bhat.copy()
    # b_neg[b_bhat >= 0] = 1
    # b_neg[b_bhat != 0] = 0
    fin_cost = np.abs(b_bhat*xi)
    return fin_cost

# # policy and bacward order as in grid!
@het(exogenous='Pi', policy=['b_bhat', 'h_bhat'], backward=['Vb_bhat', 'Vh_bhat'],
     hetinputs=[marginal_cost_grid_housing], hetoutputs=[adjustment_costs_housing], backward_init=hh_init)  
def hh_housecons(Vh_bhat_p, Vb_bhat_p, h_bhat_grid, b_bhat_grid, z_grid, e_grid, k_grid, beta, gamma, theta, sigma, qh, qh_lag, r, alpha, Psi1, gamma_p):
    
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    Wb = beta * Vb_bhat_p
    Wh = beta * Vh_bhat_p
    W_ratio = Wh / Wb

    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===

    # for each (z, b', a), linearly interpolate to find h' between gridpoints
    # satisfying optimality condition W_ratio == (1 - gamma)*qh + Psi1
    i, pi = lhs_equals_rhs_interpolate(W_ratio, (1 - gamma)*qh + Psi1)

    # use same interpolation to get Wb and then c
    h_endo_unc = interpolate.apply_coord(i, pi, h_bhat_grid)
    # c_endo_unc = (1/theta * interpolate.apply_coord(i, pi, Wb) * h_bhat_grid[np.newaxis, np.newaxis, :] 
    #               ** (-(1 - sigma)*(1 - theta)))**(1/((1 - sigma)*theta - 1))

    c_endo_unc = (1/theta * interpolate.apply_coord(i, pi, Wb) * (h_bhat_grid[np.newaxis, np.newaxis, :])**(-(1 - sigma)*(1 - theta))
                )**(1/((1 - sigma)*theta - 1))

    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===

    # solve out budget constraint to get b(z, b', h)
    b_endo = (c_endo_unc + qh*(1-gamma)*h_endo_unc + addouter(-z_grid, b_bhat_grid, -(qh - (1 + r)*gamma_p*qh_lag) * h_bhat_grid)
              + get_PsiHousing_and_deriv(h_endo_unc, h_bhat_grid, alpha)[0]) / (1 + r)

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (note utils.interpolate.interpolate_coord and utils.interpolate.apply_coord work on last axis,
    #  so we need to swap 'b' to the last axis, then back when done)
    i, pi = interpolate.interpolate_coord(b_endo.swapaxes(1, 2), b_bhat_grid)
    h_unc = interpolate.apply_coord(i, pi, h_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = interpolate.apply_coord(i, pi, b_bhat_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===

    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == (1-gamma)*qh + Psi1, assuming bhat'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, (1 - gamma)*qh + Psi1)

    # use same interpolation to get Wb and then c
    h_endo_con = interpolate.apply_coord(i, pi, h_bhat_grid)
    c_endo_con = (1/theta * (1 + k_grid[np.newaxis, :, np.newaxis]) * interpolate.apply_coord(i, pi, Wb[:, 0:1, :])
                  * h_bhat_grid[np.newaxis, np.newaxis, :] ** (-(1 - sigma) * (1 - theta))) ** (1/((1 - sigma) * theta - 1))

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===

    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + qh*(1-gamma)*h_endo_con
              + addouter(-z_grid, np.full(len(k_grid), b_bhat_grid[0]), -(qh - (1 + r)*gamma_p*qh_lag) * h_bhat_grid)
              + get_PsiHousing_and_deriv(h_endo_con, h_bhat_grid, alpha)[0]) / (1 + r)

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (utils.interpolate.interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    h_con = interpolate.interpolate_y(b_endo.swapaxes(1, 2), b_bhat_grid,
                                      h_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===

    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    h_bhat, b_bhat = h_unc.copy(), b_unc.copy()
    b_bhat[b_bhat <= b_bhat_grid[0]] = b_bhat_grid[0]
    h_bhat[b_bhat <= b_bhat_grid[0]] = h_con[b_bhat <= b_bhat_grid[0]]

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_PsiHousing_and_deriv(h_bhat, h_bhat_grid, alpha)

    # solve out budget constraint to get consumption and marginal utility
    c_bhat = addouter(z_grid, (1 + r) * b_bhat_grid, (qh - (1 + r)*gamma_p*qh_lag) * h_bhat_grid) - Psi - qh*(1-gamma)*h_bhat - b_bhat

    c_bhat[c_bhat<0] = 1e-8 # for numerical stability while converging

    uc = theta * c_bhat ** ((1-sigma)*theta - 1) * h_bhat_grid[np.newaxis, np.newaxis, :] ** ((1 - theta)*(1 - sigma))
    
    uh = (1 - theta) * (c_bhat**theta * h_bhat_grid[np.newaxis, np.newaxis, :]**(1-theta))**(-sigma) * (c_bhat**theta * h_bhat_grid[np.newaxis, np.newaxis, :]**(-theta))
    
    uce_bhat = e_grid[:, np.newaxis, np.newaxis] * uc

    # update derivatives of value function using envelope conditions
    Vh_bhat = (qh - (1 + r)*gamma_p*qh_lag - Psi2) * uc + uh
    Vb_bhat = (1 + r) * uc

    return Vh_bhat, Vb_bhat, h_bhat, b_bhat, c_bhat, uce_bhat

# # policy and bacward order as in grid!
@het(exogenous='Pi', policy=['b_bhat', 'h_bhat'], backward=['Vb_bhat', 'Vh_bhat'],
     hetinputs=[marginal_cost_grid_housing], hetoutputs=[adjustment_costs_housing, finacial_cost], backward_init=hh_init)  
def hh_spread(Vh_bhat_p, Vb_bhat_p, h_bhat_grid, b_bhat_grid, z_grid, e_grid, k_grid, beta, gamma, theta, sigma, qh, qh_lag, r, alpha, Psi1, gamma_p, xi):
    
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    Wb = beta * Vb_bhat_p
    Wh = beta * Vh_bhat_p
    W_ratio = Wh / Wb

    r_grid = r*np.ones_like(b_bhat_grid) #+ xi / (1 + b_bhat_grid)
    r_grid = r + xi

    # print(r_grid.shape)
    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===

    # for each (z, b', a), linearly interpolate to find h' between gridpoints
    # satisfying optimality condition W_ratio == (1 - gamma)*qh + Psi1
    i, pi = lhs_equals_rhs_interpolate(W_ratio, (1 - gamma)*qh + Psi1)

    # use same interpolation to get Wb and then c
    h_endo_unc = interpolate.apply_coord(i, pi, h_bhat_grid)
    # c_endo_unc = (1/theta * interpolate.apply_coord(i, pi, Wb) * h_bhat_grid[np.newaxis, np.newaxis, :] 
    #               ** (-(1 - sigma)*(1 - theta)))**(1/((1 - sigma)*theta - 1))

    c_endo_unc = (1/theta * interpolate.apply_coord(i, pi, Wb) * (h_bhat_grid[np.newaxis, np.newaxis, :])**(-(1 - sigma)*(1 - theta))
                )**(1/((1 - sigma)*theta - 1))

    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===

    # solve out budget constraint to get b(z, b', h)
    b_endo = (c_endo_unc + qh*(1-gamma)*h_endo_unc + addouter(-z_grid, b_bhat_grid, -(qh - (1 + r_grid[np.newaxis, :, np.newaxis])*gamma_p*qh_lag) * h_bhat_grid)
              + get_PsiHousing_and_deriv(h_endo_unc, h_bhat_grid, alpha)[0]) / (1 + r_grid[np.newaxis, :, np.newaxis])

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (note utils.interpolate.interpolate_coord and utils.interpolate.apply_coord work on last axis,
    #  so we need to swap 'b' to the last axis, then back when done)
    i, pi = interpolate.interpolate_coord(b_endo.swapaxes(1, 2), b_bhat_grid)
    h_unc = interpolate.apply_coord(i, pi, h_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = interpolate.apply_coord(i, pi, b_bhat_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===

    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == (1-gamma)*qh + Psi1, assuming bhat'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, (1 - gamma)*qh + Psi1)

    # use same interpolation to get Wb and then c
    h_endo_con = interpolate.apply_coord(i, pi, h_bhat_grid)
    c_endo_con = (1/theta * (1 + k_grid[np.newaxis, :, np.newaxis]) * interpolate.apply_coord(i, pi, Wb[:, 0:1, :])
                  * h_bhat_grid[np.newaxis, np.newaxis, :] ** (-(1 - sigma) * (1 - theta))) ** (1/((1 - sigma) * theta - 1))

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===

    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + qh*(1-gamma)*h_endo_con
              + addouter(-z_grid, np.full(len(k_grid), b_bhat_grid[0]), -(qh - (1 + r_grid[0])*gamma_p*qh_lag) * h_bhat_grid)
              + get_PsiHousing_and_deriv(h_endo_con, h_bhat_grid, alpha)[0]) / (1 + r_grid[0])

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (utils.interpolate.interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    h_con = interpolate.interpolate_y(b_endo.swapaxes(1, 2), b_bhat_grid,
                                      h_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===

    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    h_bhat, b_bhat = h_unc.copy(), b_unc.copy()
    b_bhat[b_bhat <= b_bhat_grid[0]] = b_bhat_grid[0]
    h_bhat[b_bhat <= b_bhat_grid[0]] = h_con[b_bhat <= b_bhat_grid[0]]

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_PsiHousing_and_deriv(h_bhat, h_bhat_grid, alpha)

    # solve out budget constraint to get consumption and marginal utility
    c_bhat = addouter(z_grid, (1 + r_grid) * b_bhat_grid, 
                      (qh - (1 + r_grid[np.newaxis, :, np.newaxis])*gamma_p*qh_lag) * h_bhat_grid) - Psi - qh*(1-gamma)*h_bhat - b_bhat

    c_bhat[c_bhat<0] = 1e-8 # for numerical stability while converging

    uc = theta * c_bhat ** ((1-sigma)*theta - 1) * h_bhat_grid[np.newaxis, np.newaxis, :] ** ((1 - theta)*(1 - sigma))
    
    uh = (1 - theta) * (c_bhat**theta * h_bhat_grid[np.newaxis, np.newaxis, :]**(1-theta))**(-sigma) * (c_bhat**theta * h_bhat_grid[np.newaxis, np.newaxis, :]**(-theta))
    
    uce_bhat = e_grid[:, np.newaxis, np.newaxis] * uc

    # update derivatives of value function using envelope conditions
    Vh_bhat = (qh - (1 + r_grid[np.newaxis, :, np.newaxis])*gamma_p*qh_lag - Psi2) * uc + uh
    Vb_bhat = (1 + r_grid[np.newaxis, :, np.newaxis]) * uc

    return Vh_bhat, Vb_bhat, h_bhat, b_bhat, c_bhat, uce_bhat

# # policy and bacward order as in grid!
@het(exogenous='Pi', policy=['b_bhat', 'h_bhat'], backward=['Vb_bhat', 'Vh_bhat'],
     hetinputs=[marginal_cost_grid_housing], hetoutputs=[adjustment_costs_housing], backward_init=hh_init)  
def hh_housecons_sep(Vh_bhat_p, Vb_bhat_p, h_bhat_grid, b_bhat_grid, z_grid, e_grid, 
                     k_grid, beta, gamma, theta, sigma, qh, qh_lag, r, alpha, Psi1, gamma_p):
    
    # gamma_p = gamma(+1)
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    Wb = beta * Vb_bhat_p
    Wh = beta * Vh_bhat_p
    W_ratio = Wh / Wb

    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===

    # for each (z, b', a), linearly interpolate to find h' between gridpoints
    # satisfying optimality condition W_ratio == (1 - gamma)*qh + Psi1
    i, pi = lhs_equals_rhs_interpolate(W_ratio, (1 - gamma)*qh + Psi1)

    # use same interpolation to get Wb and then c
    h_endo_unc = interpolate.apply_coord(i, pi, h_bhat_grid)

    c_endo_unc = interpolate.apply_coord(i, pi, Wb) ** (-1/sigma)

    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===

    # solve out budget constraint to get b(z, b', h)
    b_endo = (c_endo_unc + qh*(1-gamma)*h_endo_unc + addouter(-z_grid, b_bhat_grid, -(qh - (1 + r)*gamma_p*qh_lag) * h_bhat_grid)
              + get_PsiHousing_and_deriv(h_endo_unc, h_bhat_grid, alpha)[0]) / (1 + r)

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (note utils.interpolate.interpolate_coord and utils.interpolate.apply_coord work on last axis,
    #  so we need to swap 'b' to the last axis, then back when done)
    i, pi = interpolate.interpolate_coord(b_endo.swapaxes(1, 2), b_bhat_grid)
    h_unc = interpolate.apply_coord(i, pi, h_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = interpolate.apply_coord(i, pi, b_bhat_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===

    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == (1-gamma)*qh + Psi1, assuming bhat'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, (1 - gamma)*qh + Psi1)

    # use same interpolation to get Wb and then c
    h_endo_con = interpolate.apply_coord(i, pi, h_bhat_grid)
    c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis]) * interpolate.apply_coord(i, pi, Wb[:, 0:1, :])) ** (-1/sigma)

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===

    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + qh*(1-gamma)*h_endo_con
              + addouter(-z_grid, np.full(len(k_grid), b_bhat_grid[0]), -(qh - (1 + r)*gamma_p*qh_lag) * h_bhat_grid)
              + get_PsiHousing_and_deriv(h_endo_con, h_bhat_grid, alpha)[0]) / (1 + r)

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (utils.interpolate.interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    h_con = interpolate.interpolate_y(b_endo.swapaxes(1, 2), b_bhat_grid,
                                      h_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===

    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    h_bhat, b_bhat = h_unc.copy(), b_unc.copy()
    b_bhat[b_bhat <= b_bhat_grid[0]] = b_bhat_grid[0]
    h_bhat[b_bhat <= b_bhat_grid[0]] = h_con[b_bhat <= b_bhat_grid[0]]

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_PsiHousing_and_deriv(h_bhat, h_bhat_grid, alpha)

    # solve out budget constraint to get consumption and marginal utility
    c_bhat = addouter(z_grid, (1 + r) * b_bhat_grid, (qh - (1 + r)*gamma_p*qh_lag) * h_bhat_grid) - Psi - qh*(1-gamma)*h_bhat - b_bhat

    # c_bhat[c_bhat<0] = 1e-8 # for numerical stability while converging

    uc = c_bhat ** (-sigma)
    
    uh = theta*(h_bhat_grid[np.newaxis, np.newaxis, :]) ** (-sigma)
    
    uce_bhat = e_grid[:, np.newaxis, np.newaxis] * uc

    # update derivatives of value function using envelope conditions
    Vh_bhat = (qh - (1 + r)*gamma_p*qh_lag - Psi2) * uc + uh
    Vb_bhat = (1 + r) * uc

    return Vh_bhat, Vb_bhat, h_bhat, b_bhat, c_bhat, uce_bhat



'''Supporting functions for HA block'''
def get_PsiHousing_and_deriv(hp, h, alpha):
    """Adjustment cost Psi(hp, h) for housing"""

    # h = h + 1e-8  # avoid division by zero

    Psi = alpha / 2 * ((hp - h)/h)**2*h
    Psi1 = alpha * (hp - h)/h
    # Psi2 = alpha / 2 * (-2*(hp - h)/h**2 + (hp - h)**2/h**2)
    Psi2 = alpha / (2*h**2) * (-2*hp*(hp - h) + (hp - h)**2)

    return Psi, Psi1, Psi2


def matrix_times_first_dim(A, X):
    """Take matrix A times vector X[:, i1, i2, i3, ... , in] separately
    for each i1, i2, i3, ..., in. Same output as A @ X if X is 1D or 2D"""
    # flatten all dimensions of X except first, then multiply, then restore shape
    return (A @ X.reshape(X.shape[0], -1)).reshape(X.shape)


def addouter(z, b, a):
    """Take outer sum of three arguments: result[i, j, k] = z[i] + b[j] + a[k]"""
    return z[:, np.newaxis, np.newaxis] + b[:, np.newaxis] + a


@guvectorize(['void(float64[:], float64[:,:], uint32[:], float64[:])'], '(ni),(ni,nj)->(nj),(nj)')
def lhs_equals_rhs_interpolate(lhs, rhs, iout, piout):
    """
    Given lhs (i) and rhs (i,j), for each j, find the i such that

    lhs[i] > rhs[i,j] and lhs[i+1] < rhs[i+1,j]

    i.e. where given j, lhs == rhs in between i and i+1.

    Also return the pi such that

    pi*(lhs[i] - rhs[i,j]) + (1-pi)*(lhs[i+1] - rhs[i+1,j]) == 0

    i.e. such that the point at pi*i + (1-pi)*(i+1) satisfies lhs == rhs by linear interpolation.

    If lhs[0] < rhs[0,j] already, just return u=0 and pi=1.

    ***IMPORTANT: Assumes that solution i is monotonically increasing in j
    and that lhs - rhs is monotonically decreasing in i.***
    """

    ni, nj = rhs.shape
    assert len(lhs) == ni

    i = 0
    for j in range(nj):
        while True:
            if lhs[i] < rhs[i, j]:
                break
            elif i < nj - 1:
                i += 1
            else:
                break

        if i == 0:
            iout[j] = 0
            piout[j] = 1
        else:
            iout[j] = i - 1
            err_upper = rhs[i, j] - lhs[i]
            err_lower = rhs[i - 1, j] - lhs[i - 1]
            piout[j] = err_upper / (err_upper - err_lower)


# *************************************************
# Sandbox function to test the above functions
# *************************************************
def hh_housecons_sandbox(Vh_bhat_p, Vb_bhat_p, h_bhat_grid, b_bhat_grid, z_grid, e_grid, k_grid, beta, gamma, theta, sigma, qh, qh_lag, r, alpha, Psi1, Pi, debug = False):
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    Wb = beta * Vb_bhat_p
    # Wb = (Vb_bhat_p.T @ (beta * Pi.T)).T
    Wh = beta * Vh_bhat_p
    # Wh = (Vh_bhat_p.T @ (beta * Pi.T)).T
    W_ratio = Wh / Wb

    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===

    # for each (z, b', a), linearly interpolate to find h' between gridpoints
    # satisfying optimality condition W_ratio == (1 - gamma)*qh + Psi1
    i, pi = lhs_equals_rhs_interpolate(W_ratio, (1 - gamma)*qh + Psi1)

    # use same interpolation to get Wb and then c
    h_endo_unc = interpolate.apply_coord(i, pi, h_bhat_grid)

    # c_endo_con_temp = interpolate.apply_coord(i, pi, Wb)
    # c_endo_con_temp[c_endo_con_temp<0] = 1e-8 # for numerical stability while converging

    # c_endo_unc = (1/theta * c_endo_con_temp * (h_bhat_grid[np.newaxis, np.newaxis, :]**(-(1 - sigma)*(1 - theta)))
    #               )**(1/((1 - sigma)*theta - 1))
    
    c_endo_unc = (1/theta * interpolate.apply_coord(i, pi, Wb) * (h_bhat_grid[np.newaxis, np.newaxis, :])**(-(1 - sigma)*(1 - theta))
                  )**(1/((1 - sigma)*theta - 1))
    
    if debug:
        print(f"Negative values in h_bhat_grid: {len(h_bhat_grid[np.newaxis, np.newaxis, :][h_bhat_grid[np.newaxis, np.newaxis, :] < 0])}")
        print(f"Negative values in c_endo_unc: {len(interpolate.apply_coord(i, pi, Wb)[interpolate.apply_coord(i, pi, Wb) < 0])}")
    
    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===

    # solve out budget constraint to get b(z, b', h)
    b_endo = (c_endo_unc + qh*(1-gamma)*h_endo_unc + addouter(-z_grid, b_bhat_grid, -(qh - (1 + r)*gamma*qh_lag) * h_bhat_grid)
              + get_PsiHousing_and_deriv(h_endo_unc, h_bhat_grid, alpha)[0]) / (1 + r)

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (note utils.interpolate.interpolate_coord and utils.interpolate.apply_coord work on last axis,
    #  so we need to swap 'b' to the last axis, then back when done)
    i, pi = interpolate.interpolate_coord(b_endo.swapaxes(1, 2), b_bhat_grid)
    h_unc = interpolate.apply_coord(i, pi, h_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = interpolate.apply_coord(i, pi, b_bhat_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===

    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == (1-gamma)*qh + Psi1, assuming bhat'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, (1 - gamma)*qh + Psi1)

    # use same interpolation to get Wb and then c
    h_endo_con = interpolate.apply_coord(i, pi, h_bhat_grid)
    c_endo_con = (1/theta * (1 + k_grid[np.newaxis, :, np.newaxis]) * interpolate.apply_coord(i, pi, Wb[:, 0:1, :])
                  * h_bhat_grid[np.newaxis, np.newaxis, :] ** (-(1 - sigma) * (1 - theta))) ** (1/((1 - sigma) * theta - 1))

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===

    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + qh*(1-gamma)*h_endo_con
              + addouter(-z_grid, np.full(len(k_grid), b_bhat_grid[0]), -(qh - (1 + r)*gamma*qh_lag) * h_bhat_grid)
              + get_PsiHousing_and_deriv(h_endo_con, h_bhat_grid, alpha)[0]) / (1 + r)

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (utils.interpolate.interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    h_con = interpolate.interpolate_y(b_endo.swapaxes(1, 2), b_bhat_grid,
                                      h_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===

    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    h_bhat, b_bhat = h_unc.copy(), b_unc.copy()
    b_bhat[b_bhat <= b_bhat_grid[0]] = b_bhat_grid[0]
    h_bhat[b_bhat <= b_bhat_grid[0]] = h_con[b_bhat <= b_bhat_grid[0]]

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_PsiHousing_and_deriv(h_bhat, h_bhat_grid, alpha)

    # solve out budget constraint to get consumption and marginal utility
    c_bhat = addouter(z_grid, (1 + r) * b_bhat_grid, (qh - (1 + r)*gamma*qh_lag) * h_bhat_grid) - Psi - qh*(1-gamma)*h_bhat - b_bhat
    
    c_bhat[c_bhat<0] = 1e-8 # for numerical stability while converging

    uc = theta * c_bhat ** ((1-sigma)*theta - 1) * h_bhat_grid[np.newaxis, np.newaxis, :] ** ((1 - theta)*(1 - sigma))

    if debug: 
        print(f"Negative values in c_bhat: {len(c_bhat[c_bhat < 0])}")

    # uh = (1 - theta) * c_bhat ** (theta*(1 - sigma)) * h_bhat_grid[np.newaxis, np.newaxis, :] ** (-sigma*(1 - theta) - theta)
    uh = (1 - theta) * (c_bhat**theta * h_bhat_grid[np.newaxis, np.newaxis, :]**(1-theta))**(-sigma) * (c_bhat**theta * h_bhat_grid[np.newaxis, np.newaxis, :]**(-theta))

    uce_bhat = e_grid[:, np.newaxis, np.newaxis] * uc

    # update derivatives of value function using envelope conditions
    Vh_bhat = (qh - (1 + r)*gamma*qh_lag - Psi2) * uc + uh
    Vb_bhat = (1 + r) * uc

    return Vh_bhat, Vb_bhat, h_bhat, b_bhat, c_bhat, uce_bhat
