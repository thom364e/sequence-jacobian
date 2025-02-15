from sequence_jacobian import simple, solved, combine, create_model
from sequence_jacobian import grids
from scipy.interpolate import interp1d
import hh_housing_v2
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit
import hh_housing_v3

'''Part 1: Blocks'''
@simple
def firm(N, w, Z, pi, mu, kappa):
    # N = Y / Z
    Y = Z * N
    # Div = Y - w * N - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    Div = Y - w * N
    return Y, Div

@simple
def monetary(pi, i):
    r = i(-1) - pi
    # r = (1 + i(-1)) / (1 + pi) - 1
    return r

@simple
def monetary_old(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r

@simple
def taylor_simple(rstar, pi, phi):
    i = rstar + phi * pi
    return i

@simple
def fiscal(r, BBAR, G):
    Tax = r * BBAR + G
    return Tax

@simple
def wage_res(C_BHAT, H_BHAT, N, varphi, nu, theta, sigma, w):
    wage_res = varphi * N ** nu * 1 / theta * (C_BHAT**theta*H_BHAT**(1-theta))**sigma * (C_BHAT/H_BHAT)**(1-theta) - w
    return wage_res

@simple
def wage_res_sep(C_BHAT, H_BHAT, N, varphi, nu, theta, sigma, w):
    wage_res = varphi * N ** nu * C_BHAT**sigma - w
    return wage_res

@simple
def mkt_clearing(B_BHAT, C_BHAT, Y, BBAR, pi, mu, kappa, HBAR, H_BHAT, CHI, qh, gamma, G):
    asset_mkt = BBAR + gamma*qh*HBAR - B_BHAT
    # goods_mkt = Y - C_BHAT - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y - CHI - G
    goods_mkt = Y - C_BHAT - CHI - G
    house_mkt = HBAR - H_BHAT
    return asset_mkt, goods_mkt, house_mkt

@simple
def mkt_clearing_spread(B_BHAT, C_BHAT, Y, BBAR, pi, mu, kappa, HBAR, H_BHAT, CHI, qh, gamma, G, FIN_COST):
    asset_mkt = BBAR + gamma*qh*HBAR - B_BHAT
    goods_mkt = Y - C_BHAT - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y - CHI - G - FIN_COST
    house_mkt = HBAR - H_BHAT
    return asset_mkt, goods_mkt, house_mkt

@simple
def rotemberg_costs(Y, mu, kappa, pi):
    rotemberg_cost = mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    # rotemberg_cost = mu/(mu-1)/(2*kappa) * np.log(1+pi)**2 * Y
    return rotemberg_cost

@solved(unknowns={'i': (-0.1, 0.1)}, targets=['i_res'], solver="brentq")
def taylor(i, rstar, rhom, pi, phi, epsm):
    i_res = (1 + rstar)**(1 - rhom)*(1 + i(-1))**rhom*(1 + pi)**((1 - rhom)*phi)*(1+epsm) - 1 - i
    return i_res

@simple
def real_rate(pi, i):
    # r = i(-1) - pi
    r = (1 + i(-1)) / (1 + pi) - 1
    # r = (1 + i) / (1 + pi) - 1
    r_opp = r
    return r, r_opp

@simple
def real_rate_nom(pi, i):
    r = i(-1) - pi
    return r

@simple
def dummy_block(kappa, phi):
    kappa_res = kappa - phi
    return kappa_res

# @simple 
# def qhouse_lag(qh_lag):
#     qh = qh_lag(+1)
#     return qh

@simple 
def qhouse_lag(qh):
    qh_lag = qh(-1)
    qh_col = qh
    return qh_lag, qh_col

@simple 
def qhouse_lag_decomp(qh):
    qh_lag = qh.ss
    qh_col = qh.ss
    # print(qh_lag)
    return qh_lag, qh_col

@simple
def mkt_clearing_decomp(B_BHAT, C_BHAT, Y, BBAR, pi, mu, kappa, HBAR, H_BHAT, CHI, qh, gamma, G):
    asset_mkt = BBAR + gamma*qh.ss*HBAR - B_BHAT
    goods_mkt = Y - C_BHAT - CHI - G
    house_mkt = HBAR - H_BHAT
    return asset_mkt, goods_mkt, house_mkt

@simple 
def gamma_prime(gamma):
    gamma_p = gamma(-1)
    return gamma_p

@simple
def nkpc_ss(Z, mu):
    w = Z / mu
    return w

@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res


'''Part 1.2: Blocks for the sticky wages model'''
# @simple
# def nkpc_wage(pi, w, kappaw, varphi, N, muw, UCE_BHAT, beta_hi, omega, dbeta, nu):
    
#     beta = omega * beta_hi + (1 - omega) * (beta_hi - dbeta)
#     nkpc_res = kappaw * (varphi*N**(1+nu) - w*N/muw*UCE_BHAT) + beta * (1 + pi(+1)).apply(np.log)\
#                 - (1 + pi).apply(np.log)
#     return nkpc_res

@simple
def nkpc_wage(pi, kappaw, varphi, N, muw, beta_hi, omega, dbeta, nu, C_BHAT, sigma):
    
    beta = omega * beta_hi + (1 - omega) * (beta_hi - dbeta)

    # Auclert Rognlie Straub (2024): Annual Review sticky wages Phillips curve 
    nkpc_res = kappaw * (varphi*N**(nu) - C_BHAT**(-sigma)/muw) + beta * pi(+1) - pi
    return nkpc_res

@simple
def real_wage(muw, Z):
    w = Z/muw
    return w 

@simple
def mkt_clearing_wage(B_BHAT, C_BHAT, Y, BBAR, HBAR, H_BHAT, CHI, qh, gamma, G):
    asset_mkt = BBAR + gamma*qh*HBAR - B_BHAT
    goods_mkt = Y - C_BHAT - CHI - G
    house_mkt = HBAR - H_BHAT
    return asset_mkt, goods_mkt, house_mkt

@simple
def firm_wage(N, w, Z):
    # N = Y / Z
    Y = Z * N
    Div = Y - w * N
    return Y, Div

# @solved(unknowns={'C_s': 1, 'C_b':1, 'lambda_b': 1, 'A': 1}, targets=["euler_s", "euler_b"])
# def hh_ta(C_s, C_b, lambda_b, A, Z, eis, beta, r, lam):
#     euler_s = (beta * (1 + r(+1))) ** (-eis) * C_s(+1) - C_s                         # euler for saver agent
#     euler_b = (beta * (1 + r(+1))) ** (-eis) * C_b(+1) + (1 + r(+1))*lambda_b - C_b  # euler for borrower agent
    
#     housing_euler_s = (1 + r) * A(-1) + Z - C_s - A
#     C_H2M = Z   # computes consumption of an hand to mouth agent
#     C = (1 - lam) * C_RA + lam * C_H2M
    
#     budget_constraint = (1 + r) * A(-1) + Z - C - A
    
#     return euler_s, euler_b, budget_constraint, C_H2M, C


'''Part 2: Hetinput functions'''

def make_betas(beta_hi, dbeta, omega, q):
    """Return beta grid [beta_hi-dbeta, beta_high] and transition matrix,
    where q is probability of getting new random draw from [1-omega, omega]"""
    beta_lo = beta_hi - dbeta
    b_grid = np.array([beta_lo, beta_hi])
    pi_b = np.array([1 - omega, omega])
    Pi_b = (1-q)*np.eye(2) + q*np.outer(np.ones(2), pi_b)
    return b_grid, Pi_b, pi_b

def make_grids(bmin, bmax, hmax, kmax, nB, nH, nK, nZ, rho_z, sigma_z, beta_hi, dbeta, omega, q):
    b_bhat_grid = grids.agrid(amax=bmax, n=nB, amin = bmin)

    beta_grid_short, Pi_b, pi_b = make_betas(beta_hi, dbeta, omega, q)
    
    h_bhat_grid = grids.agrid(amax=hmax, n=nH, amin = 0.001)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid_short, pi_e, Pi_e = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

    e_grid = np.kron(np.ones_like(beta_grid_short), e_grid_short)
    beta = np.kron(beta_grid_short, np.ones_like(e_grid_short))

    Pi = np.kron(Pi_b, Pi_e)
    pi_pdf = np.kron(pi_b, pi_e)

    return b_bhat_grid, h_bhat_grid, k_grid, e_grid, Pi, pi_pdf, beta, pi_e

# def make_grids(bmin, bmax, hmax, kmax, nB, nH, nK, nZ, rho_z, sigma_z, gamma, qh_lag):
#     b_bhat_grid = grids.agrid(amax=bmax, n=nB, amin = bmin)
#     # h_bhat_grid = grids.agrid(amax=hmax, n=nH, amin = 0.01)
#     h_bhat_grid = grids.agrid(amax=hmax, n=nH, amin = 0.001)
#     k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
#     e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

#     # b_grid = np.zeros([nH,nB])
#     # for j_a in range(nH):
#     #     # grid for b, starting from the borrowing constraint
#     #     b_grid[j_a,:] = grids.agrid(amax=bmax, n=nB, amin = -qh_lag*gamma*h_bhat_grid[j_a])

#     return b_bhat_grid, h_bhat_grid, k_grid, e_grid, Pi, pi_e

def income(e_grid, w, N, Div, Tax, pi_e, pi_pdf):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid
    # Pi_e_temp = np.tile(pi_e, 2)
    div = Div / np.sum(pi_pdf * div_rule) * div_rule
    tax = Tax / np.sum(pi_pdf * tax_rule) * tax_rule
    T = div - tax

    # wage per effective unit of labor
    labor_income = w * e_grid * N
    z_grid = labor_income + T

    # # hardwired incidence rules are proportional to skill; scale does not matter 
    # tax_rule, div_rule = e_grid, e_grid
    # div = Div / np.sum(pi_e * div_rule) * div_rule
    # tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    # T = div - tax

    # # wage per effective unit of labor
    # labor_income = w * e_grid * N
    # z_grid = labor_income + T

    return z_grid


def make_grids_renter(rho_e, sd_e, n_e, min_a, max_a, n_a):
    er_grid, _, Pi_r = grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    br_grid = grids.asset_grid(min_a, max_a, n_a)
    return er_grid, Pi_r, br_grid

def income_renter(w, N, e_grid, Div, Tax, pir, lambdaa):
    tax_rule, div_rule = lambdaa*e_grid, lambdaa*e_grid
    div_renter = Div / np.sum(pir * div_rule) * div_rule
    tax_renter = Tax / np.sum(pir * tax_rule) * tax_rule

    T = div_renter - tax_renter

    labor_income_renter = w * e_grid * N
    y = labor_income_renter + T
    return y

def make_grids_old(bmin, bmax, hmax, kmax, nB, nH, nK, nZ, rho_z, sigma_z, gamma, qh_lag):
    b_bhat_grid = grids.agrid(amax=bmax, n=nB, amin = bmin)
    # h_bhat_grid = grids.agrid(amax=hmax, n=nH, amin = 0.01)
    h_bhat_grid = grids.agrid(amax=hmax, n=nH, amin = 0.001)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)

    return b_bhat_grid, h_bhat_grid, k_grid, e_grid, Pi, pi_e

def income_old(e_grid, w, N, Div, Tax, pi_e):

    # # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax

    # wage per effective unit of labor
    labor_income = w * e_grid * N
    z_grid = labor_income + T

    return z_grid

def transfers(pi_e, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T

def wages(w, e_grid):
    we = w * e_grid
    return we


'''Part 3: Hetoutput functions'''
def compute_weighted_mpc(c, b, b_grid, r, e_grid):
    """Approximate mpc out of wealth, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    mpc = np.empty_like(c)
    post_return = (1 + r) * b_grid
    mpc[:, 1:-1, :] = (c[:, 2:,:] - c[:, 0:-2, :]) / (post_return[2:] - post_return[:-2])
    mpc[:, 0, :] = (c[:, 1, :] - c[:, 0, :]) / (post_return[1] - post_return[0])
    mpc[:, -1, :] = (c[:, -1, :] - c[:, -2, :]) / (post_return[-1] - post_return[-2])
    mpc[b == b_grid[0]] = 1
    mpc = mpc * e_grid[:, np.newaxis]
    return mpc

'''Part 4: Old blocks:'''

'''Part 4: Miscellaneous functions'''
def show_irfs(irfs_list, variables, labels=[" "], ylabel=r"Percentage points (dev. from ss)", T_plot=50, figsize=(18, 6)):
    if len(irfs_list) != len(labels):
        labels = [" "] * len(irfs_list)
    n_var = len(variables)
    n_cols = min(n_var, 3)
    n_rows = math.ceil(n_var / 3)
    print(n_rows, n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)

    if n_var > 1:
        ax = ax.flatten()
    
    for i in range(n_var):
        # plot all irfs
        for j, irf in enumerate(irfs_list):
            if j % 2 == 0:
                ax[i].plot(100 * irf[variables[i]][:T_plot], linewidth = 2.5, label=labels[j])
            else :
                ax[i].plot(100 * irf[variables[i]][:T_plot], linewidth = 2.5, ls = '--', label=labels[j])
        ax[i].set_title(variables[i])
        ax[i].set_xlabel(r"$t$")
        if i==0:
            ax[i].set_ylabel(ylabel)
        ax[i].legend()
    plt.show()


def standard_setup():
        hh = hh_housing_v3.hh_housecons_sep
        hh1 = hh.add_hetinputs([make_grids, income])

        blocks_ss = [hh1, firm, monetary, fiscal, wage_res_sep, taylor_simple,
                     mkt_clearing, nkpc_ss, qhouse_lag, gamma_prime, rotemberg_costs]
        
        hank_ss = create_model(blocks_ss, name="Housing HANK SS")

        T = 300
        unknowns = ['pi', 'w', 'N', 'qh']
        targets = ['nkpc_res', 'asset_mkt', 'wage_res', 'house_mkt']
        exogenous = ['rstar', 'Z', 'gamma']

        blocks = [hh1, firm, monetary, fiscal, wage_res_sep, taylor_simple,
                     mkt_clearing, nkpc, qhouse_lag, gamma_prime, rotemberg_costs]
        
        hank = create_model(blocks, name="Housing HANK")
            
        return hank_ss, hank, T, unknowns, targets, exogenous

def model_setup(Calibration = None, unknowns_ss = None, targets_ss = None):
    """ This function takes as input a calibration dictionary and returns the model object and the steady state object"""
    hank_ss, hank, T, unknowns, targets, exogenous = standard_setup()

    if unknowns_ss is None:    
        unknowns_ss = {'beta': 0.983, 'varphi': 0.833, 'theta': 0.05}
    else:
        unknowns_ss = unknowns_ss
    
    if targets_ss is None:
        targets_ss = {'goods_mkt': 0, 'wage_res': 0, 'house_mkt': 0}
    else:
        targets_ss = targets_ss

    if Calibration is not None:
        ss0 = hank_ss.solve_steady_state(Calibration, unknowns_ss, targets_ss, solver="hybr")
        ss = hank.steady_state(ss0)

        for k in ss0.keys():
            assert np.all(np.isclose(ss[k], ss0[k])) # check that the steady state is the same 
    else:
        Calibration = {'gamma': 0.8, 'qh': 8.0, 'sigma': 1.0, 'alpha': 0.05, 'bmax': 45, 'rotemberg_cost': 0.0,
            'hmax': 5.0, 'kmax': 2.0, 'nB': 50, 'nH': 70, 'nK': 50, 'nZ': 3, 'G': 0.0,
            'rho_z': 0.966, 'sigma_z': 0.92, 'N': 1.0, 'Z': 1.0, 'pi': 0.0, 'mu': 1.2, 'bmin': 0.0,
            'kappa': 0.1, 'rstar': 0.005, 'phi': 1.5, 'nu': 1.0, 'BBAR': 0.26, 'HBAR': 1.0}
        Calibration['rstar'] = 0.03/4
        Calibration['gamma'] = 0.8
        Calibration['alpha'] = 0.0

        ss0 = hank_ss.solve_steady_state(Calibration, unknowns_ss, targets_ss, solver="hybr")
        ss = hank.steady_state(ss0)

        for k in ss0.keys():
            assert np.all(np.isclose(ss[k], ss0[k])) # check that the steady state is the same 

    return hank, ss, T, unknowns, targets, Calibration

def calc_mpc(ss, ha_block):
    MPC = np.zeros(ss.internals[ha_block]['D'].shape)
    dc = (ss.internals[ha_block]['c_bhat'][:,1:,:]-ss.internals[ha_block]['c_bhat'][:,:-1,:])
    dm = (1+ss['r'])*ss.internals[ha_block]['b_bhat_grid'][np.newaxis,1:,np.newaxis]-(1+ss['r'])*ss.internals[ha_block]['b_bhat_grid'][np.newaxis,:-1,np.newaxis]
    MPC[:,:-1,:] = dc/dm
    MPC[:,-1,:] = MPC[:,-1,:] # assuming constant MPC at end
    mean_MPC = np.sum(MPC*ss.internals[ha_block]['D'])

    return MPC, mean_MPC

def calc_mpch(ss, ha_block):
    """Calculates the MPC out of housing wealth"""
    MPC = np.zeros(ss.internals[ha_block]['D'].shape)
    dc = (ss.internals[ha_block]['c_bhat'][:,:,1:]-ss.internals[ha_block]['c_bhat'][:,:,:-1])
    dh = ss['qh']*ss.internals[ha_block]['h_bhat_grid'][np.newaxis,np.newaxis,1:]-ss['qh']*ss.internals[ha_block]['h_bhat_grid'][np.newaxis,np.newaxis,:-1]
    # print(dh.shape, dc.shape)
    # print(dh)
    MPC[:,:,:-1] = dc/dh
    MPC[:,:,-1] = MPC[:,:,-1] # assuming constant MPC at end
    mean_MPC = np.sum(MPC*ss.internals[ha_block]['D'])

    return MPC, mean_MPC

def compute_mpc(D, c_bhat, rstar, b_bhat_grid):
    mpc = np.zeros(D.shape)
    dc = (c_bhat[:,1:,:]-c_bhat[:,:-1,:])
    dm = (1+rstar)*b_bhat_grid[np.newaxis,1:,np.newaxis]-(1+rstar)*b_bhat_grid[np.newaxis,:-1,np.newaxis]
    mpc[:,:-1,:] = dc/dm
    mpc[:,-1,:] = mpc[:,-1,:] # assuming constant MPC at end
    mean_MPC = np.sum(mpc*D)

    return mpc

def get_distribution(model_ss, irf, hh_name, T):
    c_dist = np.zeros((irf.internals[hh_name]['c_bhat'].shape))
    h_dist = np.zeros((irf.internals[hh_name]['c_bhat'].shape))
    b_dist = np.zeros((irf.internals[hh_name]['c_bhat'].shape))
    dist = np.zeros((irf.internals[hh_name]['c_bhat'].shape))
    dist_beg = np.zeros((irf.internals[hh_name]['c_bhat'].shape))
    for t in range(T):
        dist[t] = (model_ss.internals[hh_name]['D'] + irf.internals[hh_name]['D'][t])
        dist_beg[t] = (model_ss.internals[hh_name]['Dbeg'] + irf.internals[hh_name]['Dbeg'][t])
        c_dist[t] = (model_ss.internals[hh_name]['D'] + irf.internals[hh_name]['D'][t]) \
                  * (model_ss.internals[hh_name]['c_bhat'] + irf.internals[hh_name]['c_bhat'][t])
        h_dist[t] = (model_ss.internals[hh_name]['D'] + irf.internals[hh_name]['D'][t]) \
            * (model_ss.internals[hh_name]['h_bhat'] + irf.internals[hh_name]['h_bhat'][t])
        b_dist[t] = (model_ss.internals[hh_name]['D'] + irf.internals[hh_name]['D'][t]) \
            * (model_ss.internals[hh_name]['h_bhat'] + irf.internals[hh_name]['h_bhat'][t])
    return c_dist, h_dist, b_dist, dist, dist_beg

def get_policies(model_ss, irf, hh_name, t, devss = True):
    c_pol_ss = model_ss.internals[hh_name]['c_bhat']
    h_pol_ss = model_ss.internals[hh_name]['h_bhat']
    b_pol_ss = model_ss.internals[hh_name]['b_bhat']

    if devss:
        c_pol_path = c_pol_ss + irf.internals[hh_name]['c_bhat']
        h_pol_path = h_pol_ss + irf.internals[hh_name]['h_bhat']
        b_pol_path = b_pol_ss + irf.internals[hh_name]['b_bhat']
    else:
        c_pol_path = irf.internals[hh_name]['c_bhat']
        h_pol_path = irf.internals[hh_name]['h_bhat']
        b_pol_path = irf.internals[hh_name]['b_bhat']

    return c_pol_ss, h_pol_ss, b_pol_ss, c_pol_path[t], h_pol_path[t], b_pol_path[t]

def get_b_grid(model_ss, irf, hh_name):
    # returns the endogenous grid for b (T, nH x nB)
    b_grid_ss = model_ss.internals[hh_name]['b_bhat_grid'][None,:] - model_ss['gamma'] * model_ss['qh'] \
           * model_ss.internals[hh_name]['h_bhat_grid'][:,None]

    gamma_level = model_ss['gamma'] + irf['gamma']
    qh_level = model_ss['qh'] + irf['qh']
    h_bhat_grid = model_ss.internals[hh_name]['h_bhat_grid']

    # Initialize the time-varying grid
    T = len(gamma_level)
    nH = h_bhat_grid.shape[0]
    nB = model_ss.internals[hh_name]['b_bhat_grid'].shape[0]
    b_grid_path = np.zeros((T, nH, nB))

    # Calculate the time-varying grid for each time step
    for t in range(T):
        b_grid_path[t] = model_ss.internals[hh_name]['b_bhat_grid'][None, :] - gamma_level[t] * qh_level[t] \
                                 * h_bhat_grid[:, None]
        # print(np.sum(gamma_level[t] * qh_level[t] * h_bhat_grid[:, None]))
    return b_grid_ss, b_grid_path

def manipulate_separable(M, E):
    """ Here, E is the expectation matrix, M is the FIRE Jacobian """
    T, m = M.shape
    assert T == m
    assert E.shape == (T, T)
    
    M_beh = np.empty_like(M)
    for t in range(T):
        for s in range(T):
            summand = 0
            for tau in range(min(s,t)+1):
                if tau > 0:
                    summand += (E[tau, s] - E[tau-1, s]) * M[t - tau, s - tau]
                else:
                    summand += E[tau, s] * M[t - tau, s - tau]
            M_beh[t, s] = summand
    return M_beh

def E_sticky_exp(theta, T=300, sticky_info=False):
    col = 1 - theta**(1 + np.arange(T))
    E = np.tile(col[:, np.newaxis], (1, T))
    if sticky_info:
        return E
    else:
        E = np.triu(E, +1) + np.tril(np.ones((T, T)))
        return E

'''Old functions'''
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