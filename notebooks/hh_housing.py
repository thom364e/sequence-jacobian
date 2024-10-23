import numpy as np
from numba import guvectorize
from scipy.interpolate import interp1d
from numba import njit

# from ..src.sequence_jacobian.blocks.het_block import het
# from ..src.sequence_jacobian import interpolate

from sequence_jacobian.blocks.het_block import het
from sequence_jacobian import interpolate, grids

def hh_init(b_bhat_grid, a_bhat_grid, z_grid, eis):
    Va_bhat = (0.6 + 1.1 * b_bhat_grid[:, np.newaxis] + a_bhat_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
    Vb_bhat = (0.5 + b_bhat_grid[:, np.newaxis] + 1.2 * a_bhat_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
    return Va_bhat, Vb_bhat


def adjustment_costs(a_bhat, a_bhat_grid, ra, chi0, chi1, chi2):
    chi = get_Psi_and_deriv(a_bhat, a_bhat_grid, ra, chi0, chi1, chi2)[0]
    return chi


def marginal_cost_grid(a_bhat_grid, ra, chi0, chi1, chi2):
    # precompute Psi1(a', a) on grid of (a', a) for steps 3 and 5
    Psi1 = get_Psi_and_deriv(a_bhat_grid[:, np.newaxis],
                             a_bhat_grid[np.newaxis, :], ra, chi0, chi1, chi2)[1]
    return Psi1


# policy and bacward order as in grid!
@het(exogenous='Pi', policy=['b_bhat', 'a_bhat'], backward=['Vb_bhat', 'Va_bhat'],
     hetinputs=[marginal_cost_grid], hetoutputs=[adjustment_costs], backward_init=hh_init)  
def hh(Va_bhat_p, Vb_bhat_p, a_bhat_grid, b_bhat_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2, Psi1, gamma):
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    Wb = beta * Vb_bhat_p
    Wa = beta * Va_bhat_p
    W_ratio = Wa / Wb

    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===

    # for each (z, b', a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio == 1 - gamma + Psi1
    i, pi = lhs_equals_rhs_interpolate(W_ratio, (1 - gamma) + Psi1)

    # use same interpolation to get Wb and then c
    a_endo_unc = interpolate.apply_coord(i, pi, a_bhat_grid)
    c_endo_unc = interpolate.apply_coord(i, pi, Wb) ** (-eis)
    
    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===

    # solve out budget constraint to get b(z, b', a)
    b_endo = (c_endo_unc + (1-gamma)*a_endo_unc + addouter(-z_grid, b_bhat_grid, ((1 + rb)*gamma - (1 + ra)) * a_bhat_grid)
              + get_Psi_and_deriv(a_endo_unc, a_bhat_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (note utils.interpolate.interpolate_coord and utils.interpolate.apply_coord work on last axis,
    #  so we need to swap 'b' to the last axis, then back when done)
    i, pi = interpolate.interpolate_coord(b_endo.swapaxes(1, 2), b_bhat_grid)
    a_unc = interpolate.apply_coord(i, pi, a_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = interpolate.apply_coord(i, pi, b_bhat_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===

    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == 1+Psi1, assuming b'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, (1 - gamma) + Psi1)

    # use same interpolation to get Wb and then c
    a_endo_con = interpolate.apply_coord(i, pi, a_bhat_grid)
    c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis]) ** (-eis)
                  * interpolate.apply_coord(i, pi, Wb[:, 0:1, :]) ** (-eis))

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===

    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + (1-gamma)*a_endo_con
              + addouter(-z_grid, np.full(len(k_grid), b_bhat_grid[0]), ((1 + rb)*gamma - (1 + ra)) * a_bhat_grid)
              + get_Psi_and_deriv(a_endo_con, a_bhat_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (utils.interpolate.interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    a_con = interpolate.interpolate_y(b_endo.swapaxes(1, 2), b_bhat_grid,
                                      a_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===

    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    a_bhat, b_bhat = a_unc.copy(), b_unc.copy()
    b_bhat[b_bhat <= b_bhat_grid[0]] = b_bhat_grid[0]
    a_bhat[b_bhat <= b_bhat_grid[0]] = a_con[b_bhat <= b_bhat_grid[0]]

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_Psi_and_deriv(a_bhat, a_bhat_grid, ra, chi0, chi1, chi2)

    # solve out budget constraint to get consumption and marginal utility
    c_bhat = addouter(z_grid, (1 + rb) * b_bhat_grid, -((1 + rb)*gamma - (1 + ra)) * a_bhat_grid) - Psi - (1-gamma)*a_bhat - b_bhat
    uc = c_bhat ** (-1 / eis)
    uce_bhat = e_grid[:, np.newaxis, np.newaxis] * uc

    # update derivatives of value function using envelope conditions
    Va_bhat = (1 + ra - (1 + rb)*gamma - Psi2) * uc
    Vb_bhat = (1 + rb) * uc

    return Va_bhat, Vb_bhat, a_bhat, b_bhat, c_bhat, uce_bhat



'''Supporting functions for HA block'''

def get_Psi_and_deriv(ap, a, ra, chi0, chi1, chi2):
    """Adjustment cost Psi(ap, a) and its derivatives with respect to
    first argument (ap) and second argument (a)"""
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)

    adj_denominator = a_with_return + chi0
    core_factor = (abs_a_change / adj_denominator) ** (chi2 - 1)

    Psi = chi1 / chi2 * abs_a_change * core_factor
    Psi1 = chi1 * sign_change * core_factor
    Psi2 = -(1 + ra) * (Psi1 + (chi2 - 1) * Psi / adj_denominator)
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



'''Own code not used in the final version'''
def hh_sandbox(Va_p, Vb_p, a_grid, b_grid, bhat_grid, b_endo_2d_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2, gamma, Psi1, debug=False):
    
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    Wb = beta * Vb_p
    Wa = beta * Va_p
    W_ratio = Wa / Wb

    if debug:
        print(W_ratio.shape)

    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===

    # for each (z, b', a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio == (1 - gamma)+Psi1
    i, pi = lhs_equals_rhs_interpolate(W_ratio, (1 - gamma) + Psi1)
    if debug:
        # print(f"Shape of i {pi}")
        # print(f"Nan count i: {np.sum(np.isnan(i))}")
        # print(f"Nan count pi: {np.sum(np.isnan(pi))}")
        pass

    # use same interpolation to get Wb and then c
    a_endo_unc = interpolate.apply_coord(i, pi, a_grid)
    c_endo_unc = interpolate.apply_coord(i, pi, Wb) ** (-eis)
    
    if debug: 
        # print(f" a_endo_unc {a_endo_unc}")
        print(c_endo_unc.shape)
        # print(f"Nan count for c {np.sum(np.isnan(c_endo_unc))}")
        # print(f"Nan count for a {np.sum(np.isnan(a_endo_unc))}")
        # if np.sum(np.isnan(c_endo_unc)) > 0:
        #     print(c_endo_unc)

    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===

    # solve out budget constraint to get b(z, b', a)
    b_endo = (c_endo_unc + (1-gamma)*a_endo_unc + addouter(-z_grid, bhat_grid, ((1 + rb)*gamma - (1 + ra)) * a_grid)
              + get_Psi_and_deriv(a_endo_unc, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (note utils.interpolate.interpolate_coord and utils.interpolate.apply_coord work on last axis,
    #  so we need to swap 'b' to the last axis, then back when done)
    i, pi = interpolate.interpolate_coord(b_endo.swapaxes(1, 2), bhat_grid)
    a_unc = interpolate.apply_coord(i, pi, a_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = interpolate.apply_coord(i, pi, bhat_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===

    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == 1+Psi1, assuming b'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, (1 - gamma) + Psi1)

    if debug:
        # print(f"Shape of i for constrained {i.shape}")
        # print(f"W_ratio {W_ratio[:, 0:1, :]}")
        # print(f"W_ratio {W_ratio[:, 0:1, :]}")
        pass

    # use same interpolation to get Wb and then c
    a_endo_con = interpolate.apply_coord(i, pi, a_grid)
    c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis]) ** (-eis)
                  * interpolate.apply_coord(i, pi, Wb[:, 0:1, :]) ** (-eis))
    
    if debug:
        print(f"k_grid shape: {k_grid[np.newaxis, :, np.newaxis].shape}")
        print(f"Apply_coord shape: {interpolate.apply_coord(i, pi, Wb[:, 0:1, :]).shape}")
        print(f"Wb[:, 0:1, :] shape: {Wb[:, 0:1, :].shape}")

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===

    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + (1-gamma)*a_endo_con
              + addouter(-z_grid, np.full(len(k_grid), bhat_grid[0]), ((1 + rb)*gamma - (1 + ra)) * a_grid)
              + get_Psi_and_deriv(a_endo_con, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (utils.interpolate.interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    a_con = interpolate.interpolate_y(b_endo.swapaxes(1, 2), bhat_grid,
                                      a_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===

    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= bhat_grid[0]] = bhat_grid[0]
    a[b <= bhat_grid[0]] = a_con[b <= bhat_grid[0]]

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)

    # solve out budget constraint to get consumption and marginal utility
    c = addouter(z_grid, (1 + rb) * bhat_grid, -((1 + rb)*gamma - (1 + ra)) * a_grid) - Psi - (1-gamma)*a - b
    uc = c ** (-1 / eis)
    uce = e_grid[:, np.newaxis, np.newaxis] * uc

    # update derivatives of value function using envelope conditions
    Va = (1 + ra - (1 + rb)*gamma - Psi2) * uc
    Vb = (1 + rb) * uc

    # _, c, a, b, Va, Vb, uce = bhat_to_b(a_grid, b_grid, bhat_grid, b_endo_2d_grid, e_grid, z_grid, a, c, Va, Vb, ra, rb, chi0, chi1, chi2, eis, gamma)
    
    return Va, Vb, a, b, c, uce


# @njit           
def bhat_to_b(a_grid, b_grid, bhat_grid, b_endo_2d_grid, e_grid, z_grid, a_bhat, c_bhat, Va_bhat, Vb_bhat, ra, rb, chi0, chi1, chi2, eis, gamma):
    # bhat_to_b(gamma, b_grid, a_grid, c_bhat, a_bhat, bhat_grid, b_endo_2d_grid, z_grid, ra, rb, chi0, chi1, chi2, Va_bhat, Vb_bhat, e_grid, eis):
    '''
    Function to transform the a' and c policy functions from a'(z,bhat,a) and c(z,bhat,a) to a'(z,b,a) and c(z,b,a)
    '''

    nZ, nB, nA = c_bhat.shape
    # print(nB)

    # b_grid = np.zeros([nA,nB])
    # # print(a_grid.shape)
    # for j_a in range(nA):
    #     # grid for b, starting from the borrowing constraint
    #     print(grids.agrid(amax=bmax, n=nB, amin = -gamma_diff*a_grid[j_a]).shape)
    #     b_grid[j_a,:] = grids.agrid(amax=bmax, n=nB, amin = -gamma_diff*a_grid[j_a])

    # b_endo_temp = bhat_grid[None,:] - gamma*a_grid[:,None]
    c = np.zeros_like(c_bhat)
    a = np.zeros_like(a_bhat)
    b = np.zeros_like(c_bhat)
    Va = np.zeros_like(Va_bhat)
    Vb = np.zeros_like(Vb_bhat)

    for j_z in range(nZ):
        for j_a in range(nA):
            # Create the interpolation function for consumption 
            interp_func_c = interp1d(b_endo_2d_grid[j_a, :], c_bhat[j_z, :, j_a], kind='linear', fill_value="extrapolate")
            interp_func_a = interp1d(b_endo_2d_grid[j_a, :], a_bhat[j_z, :, j_a], kind='linear', fill_value="extrapolate")
            interp_func_Va = interp1d(b_endo_2d_grid[j_a, :], Va_bhat[j_z, :, j_a], kind='linear', fill_value="extrapolate")
            interp_func_Vb = interp1d(b_endo_2d_grid[j_a, :], Vb_bhat[j_z, :, j_a], kind='linear', fill_value="extrapolate")
            # Use the interpolation function to get the interpolated values for the entire b_grid[j_a, :]
            c[j_z, :, j_a] = interp_func_c(b_grid[j_a, :])
            a[j_z, :, j_a] = interp_func_a(b_grid[j_a, :])
            Va[j_z, :, j_a] = interp_func_Va(b_grid[j_a, :])
            Vb[j_z, :, j_a] = interp_func_Vb(b_grid[j_a, :])

    for j_z in range(nZ):
        for j_a in range(nA):
            for j_b in range(nB):
                # Use the original budget constraint to back out b'(z,b,a)
                b[j_z, j_b, j_a] = (z_grid[j_z] + (1 + ra) * a_grid[j_a] + (1 + rb) * b_grid[j_a, j_b]
                                - a[j_z, j_b, j_a] - c[j_z, j_b, j_a]
                                - get_Psi_and_deriv(a[j_z, j_b, j_a], a_grid[j_a], ra, chi0, chi1, chi2)[0])

    uc = c ** (-1 / eis)
    uce = e_grid[:, np.newaxis, np.newaxis] * uc

    return b_grid, c, a, b, Va, Vb, uce