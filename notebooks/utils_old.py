from sequence_jacobian import simple, solved, combine, create_model
from sequence_jacobian import grids
from scipy.interpolate import interp1d
import hh_housing_v2
import numpy as np
import matplotlib.pyplot as plt

'''Part 1: Blocks'''
@simple
def mkt_clearing(C_BHAT, Y, HBAR, H_BHAT, BBAR_BHAT, B_BHAT):
    goods_mkt = Y - C_BHAT
    house_mkt = HBAR - H_BHAT
    asset_mkt = BBAR_BHAT - B_BHAT
    return goods_mkt, house_mkt, asset_mkt

@simple
def house_mkt_clearing(HBAR, H):
    house_mkt = HBAR - H
    return house_mkt


def make_grids(bmax, hmax, kmax, nB, nH, nK, nZ, rho_z, sigma_z, gamma, qh_lag):
    b_bhat_grid = grids.agrid(amax=bmax, n=nB, amin = 0.01)
    h_bhat_grid = grids.agrid(amax=hmax, n=nH, amin = 0.01)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

    b_grid = np.zeros([nH,nB])
    for j_a in range(nH):
        # grid for b, starting from the borrowing constraint
        b_grid[j_a,:] = grids.agrid(amax=bmax, n=nB, amin = -qh_lag*gamma*h_bhat_grid[j_a])

    return b_bhat_grid, h_bhat_grid, k_grid, e_grid, Pi, b_grid


def income(e_grid, tax, w, N):
    z_grid = (1 - tax) * w * N * e_grid
    return z_grid


def policy_ss(Pi, h_bhat_grid, b_bhat_grid, z_grid, e_grid, k_grid, beta, gamma, theta, sigma, qh, qh_lag, r, alpha, tol=1E-12, max_iter=10_000, debug = False):
    '''
    Iterates on the policy functions to find the steady state policy functions
    '''

    # initialize value function and policy function
    Vh_p, Vb_p = hh_housing_v2.hh_init(b_bhat_grid, h_bhat_grid, z_grid, sigma, theta)
    
    if debug:
        plt.plot(b_bhat_grid, Vh_p[0,:,0])
        # pass

    Psi1 = hh_housing_v2.marginal_cost_grid_housing(h_bhat_grid, alpha)

    if debug:
        pass
        # print(Psi1)

    # iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    for it in range(max_iter):
        Vh_p, Vb_p, h, b, c, _ = hh_housing_v2.hh_housecons_sandbox(Vh_p, Vb_p, h_bhat_grid, b_bhat_grid, 
                                                                    z_grid, e_grid, k_grid, beta, gamma, theta, 
                                                                    sigma, qh, qh_lag, r, alpha, Psi1, Pi, debug)

        if debug:
            print(f"Iteration #{it}")
            if it > 0: #and it % 1000 == 0:
                print("Iteration:", it)
                print("Max difference in h:", np.max(np.abs(h - h_old)))
                print("Max difference in b:", np.max(np.abs(b - b_old)))

        # after iteration 0, can compare new policy function to old one
        if it > 0:
            h_max_diff = np.max(np.abs(h - h_old))
            b_max_diff = np.max(np.abs(b - b_old))
            
            if h_max_diff < tol and b_max_diff < tol:
                print(f"Converged after {it} iterations")
                return Vh_p, Vb_p, h, b, c, Psi1
        
        h_old = h.copy()
        b_old = b.copy()

    print(f"Failed to converge after {max_iter} iterations")
    return Vh_p, Vb_p, h, b, c, Psi1


def bhat_to_b(h_bhat_grid, b_bhat_grid, b_grid, z_grid, c_bhat, h_bhat, gamma, alpha, r, qh, qh_lag):
    '''
    Function to transform the a', bhat' and c policy functions from a'(z,bhat,a), bhat'(z,bhat,a) and c(z,bhat,a) to a'(z,b,a), b'(z,b,a) and c(z,b,a)
    '''

    nZ, nB, nH = c_bhat.shape
    
    b_endo = b_bhat_grid[None,:] - qh_lag*gamma*h_bhat_grid[:,None]
    c = np.zeros_like(c_bhat)
    h = np.zeros_like(h_bhat)
    b = np.zeros_like(c_bhat)

    for j_z in range(nZ):
        for j_h in range(nH):
            # Create the interpolation function for consumption 
            interp_func_c = interp1d(b_endo[j_h, :], c_bhat[j_z, :, j_h], kind='linear', bounds_error = False, fill_value=10)
            interp_func_a = interp1d(b_endo[j_h, :], h_bhat[j_z, :, j_h], kind='linear', bounds_error = False, fill_value=10)

            # Use the interpolation function to get the interpolated values for the entire b_grid[j_a, :]
            c[j_z, :, j_h] = interp_func_c(b_grid[j_h, :])
            h[j_z, :, j_h] = interp_func_a(b_grid[j_h, :])

    for j_z in range(nZ):
        for j_h in range(nH):
            for j_b in range(nB):
                # Use the original budget constraint to back out b'(z,b,a)
                b[j_z, j_b, j_h] = (z_grid[j_z] + qh * h_bhat_grid[j_h] + (1 + r) * b_grid[j_h, j_b]
                                - qh*h[j_z, j_b, j_h] - c[j_z, j_b, j_h]
                                - hh_housing_v2.get_PsiHousing_and_deriv(h[j_z, j_b, j_h], h_bhat_grid[j_h], alpha)[0])
            
    return c, h, b, b_endo