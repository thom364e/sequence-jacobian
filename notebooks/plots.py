import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils
from tabulate import tabulate
import os 
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

plt.rcParams.update({'font.size': 15})
plt.rcParams['lines.linewidth'] = 2.5
font = {'family': 'serif', 'serif': ['Palatino'], 'size': 15}
plt.rc('font', **font)
plt.rc('text', usetex=True)

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#0072BD', '#D95319', '#FAAD26', '#7E2F8E', '#77AC30', '#4DBEEE',
                                                     '#A2142F', '#000000'])  

def linear_irf_baseline(irf, model_ss, H = 17):
    fig, ax = plt.subplots(2, 3, figsize=(16,12*2/3), sharex = True)
    ax = ax.flatten()

    var_name = {'Y': r'Output ($Y$)', 'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 
                'pi': r'Inflation ($\pi$)', 'i': r'Policy rate ($i$)'}

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    
    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    for i, (key, value) in enumerate(var_name.items()):   
        if key in devss:
            ax[i].plot(100*(irf[key][:H]/model_ss[key]))   
            ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Percent from steady state')
        else:
            ax[i].plot(100*((1 + model_ss[key] +irf[key][:H])**4-1))
            ax[i].axhline(y=100*((1 + model_ss[key])**4 - 1), color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Annualised level, percent')
        
        ax[i].set_title(var_name[key])

        ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax[i].yaxis.set_major_formatter(formatter)
        ax[i].xaxis.set_major_locator(MultipleLocator(4))
        ax[i].set_xlim(0, H -1)
        ax[i].grid(True, alpha=0.3)

        if i == 3 or i == 4:
            ax[i].set_xlabel('Quarters')

    trans_inc_var = {'Div': r'Dividends ($d$)', 'Tax': r'Tax ($\tau$)', 'N': r'Hours ($N$)', 'r': r'Real rate ($r$)', 'w': r'Wage ($w$)'}

    for i, (key, value) in enumerate(trans_inc_var.items()):   
        ax[5].plot(100*(irf[key][:H]), label=f"{value}")

    ax[5].axhline(y=0, color='grey', linestyle='--', linewidth=1.0)
    ax[5].set_ylabel('Deviation from steady state')

    ax[5].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax[5].yaxis.set_major_formatter(formatter)
    ax[5].xaxis.set_major_locator(MultipleLocator(4))
    ax[5].set_xlim(0, H -1)
    ax[5].set_title('Transitory income variables')
    ax[5].legend(fontsize=12, frameon=False, ncols = 1, loc = 'upper right')
    ax[5].grid(True, alpha=0.3)
    ax[5].set_xlabel('Quarters')

    plt.show()

    return fig

def compare_decomp(c_decomp, model_ss, H = 21):
    fig, ax = plt.subplots()
    colors=['#0072BD', '#D95319', '#FAAD26', '#7E2F8E', '#77AC30', '#4DBEEE','#A2142F', '#000000']
    # counterfactual_decom_baseline =  consumption_decomp(Js_dict['inatt'], irfs['inatt'], hh_name, do_assert=False)
    # counterfactual_decom_qh10 =  consumption_decomp(Js_dict['inatt_hw'], irfs['inatt'], hh_name, do_assert=False)
    # counterfactual_decom_diffmpc =  consumption_decomp(Js_dict['diffmpc'], irfs['inatt'], hh_name, do_assert=False)

    H = 21
    p1 = ax.plot(100*c_decomp['baseline']['housing'][:H]/model_ss['baseline']['C_BHAT'], label='Housing price')
    p2 = ax.plot(100*c_decomp['baseline']['indirect'][:H]/model_ss['baseline']['C_BHAT'], label='Indirect')
    p3 = ax.plot(100*c_decomp['baseline']['direct'][:H]/model_ss['baseline']['C_BHAT'], label='Direct')
    p4 = ax.plot(100*c_decomp['baseline']['collateral'][:H]/model_ss['baseline']['C_BHAT'], label='Collateral')
    # ax.plot(100*c_decomp['baseline']['total'][:H]/model_ss['baseline']['C_BHAT'], label='Total response')

    p5 = ax.plot(100*c_decomp['highwealth']['housing'][:H]/model_ss['highwealth']['C_BHAT'], color = colors[0], linestyle='dashed')
    p6 = ax.plot(100*c_decomp['highwealth']['indirect'][:H]/model_ss['highwealth']['C_BHAT'], color = colors[1], linestyle='dashed')
    p7 = ax.plot(100*c_decomp['highwealth']['direct'][:H]/model_ss['highwealth']['C_BHAT'], color = colors[2], linestyle='dashed')
    p8 = ax.plot(100*c_decomp['highwealth']['collateral'][:H]/model_ss['highwealth']['C_BHAT'], color = colors[3], linestyle='dashed')
    # ax.plot(100*c_decomp['highwealth']['total'][:H]/model_ss['highwealth']['C_BHAT'], color = colors[4], linestyle='dashed', label='Total response')
    # p10 = ax.plot(100*counterfactual_decom_qh10['total'][:H]/model['qh10']['C_BHAT'], label='Total response')

#     p9 = ax.plot(100*counterfactual_decom_diffmpc['housing'][:H]/model['diffmpc']['C_BHAT'], color = colors[0], linestyle='dotted')
#     p10 = ax.plot(100*counterfactual_decom_diffmpc['indirect'][:H]/model['diffmpc']['C_BHAT'], color = colors[1], linestyle='dotted')
#     p11 = ax.plot(100*counterfactual_decom_diffmpc['direct'][:H]/model['diffmpc']['C_BHAT'], color = colors[2], linestyle='dotted')
#     p12 = ax.plot(100*counterfactual_decom_diffmpc['collateral'][:H]/model['diffmpc']['C_BHAT'], color = colors[3], linestyle='dotted')
    # ax.plot(100*counterfactual_decom_diffmpc['total'][:H]/model['diffmpc']['C_BHAT'], label='Total response')
    plt.legend([(p1[0], p5[0]), (p2[0], p6[0]), (p3[0], p7[0]), (p4[0], p8[0])], 
            ['Housing price', 'Indirect', 'Direct', 'Collateral'], numpoints=1, 
            handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=5, frameon=False, fontsize = 12)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Percent from steady state')
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlim(0, H-1)
    ax.axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
    ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax.grid(True, alpha = 0.3)
    plt.show()

    return fig

def compare_decomp_counterfactual(c_decomp, model_ss, counterfactual_decom_highwealth, H = 21):
    fig, ax = plt.subplots()
    colors=['#0072BD', '#D95319', '#FAAD26', '#7E2F8E', '#77AC30', '#4DBEEE','#A2142F', '#000000']

    # counterfactual_decom_highwealth =  consumption_decomp(CurlyJs['highwealth'], irf_lin['baseline'], hh_name, do_assert=False)

    H = 21
    p1 = ax.plot(100*c_decomp['baseline']['housing'][:H]/model_ss['baseline']['C_BHAT'], label='Housing price')
    p2 = ax.plot(100*c_decomp['baseline']['indirect'][:H]/model_ss['baseline']['C_BHAT'], label='Indirect')
    p3 = ax.plot(100*c_decomp['baseline']['direct'][:H]/model_ss['baseline']['C_BHAT'], label='Direct')
    p4 = ax.plot(100*c_decomp['baseline']['collateral'][:H]/model_ss['baseline']['C_BHAT'], label='Collateral')
    # ax.plot(100*c_decomp['baseline']['total'][:H]/model_ss['baseline']['C_BHAT'], label='Total response')

    p9 = ax.plot(100*counterfactual_decom_highwealth['housing'][:H]/model_ss['baseline']['C_BHAT'], color = colors[0], linestyle='dotted')
    p10 = ax.plot(100*counterfactual_decom_highwealth['indirect'][:H]/model_ss['baseline']['C_BHAT'], color = colors[1], linestyle='dotted')
    p11 = ax.plot(100*counterfactual_decom_highwealth['direct'][:H]/model_ss['baseline']['C_BHAT'], color = colors[2], linestyle='dotted')
    p12 = ax.plot(100*counterfactual_decom_highwealth['collateral'][:H]/model_ss['baseline']['C_BHAT'], color = colors[3], linestyle='dotted')
    # ax.plot(100*counterfactual_decom_highwealth['total'][:H]/model_ss['baseline']['C_BHAT'], label='Total response')
    # plt.legend([(p1[0], p9[0]), (p2[0], p10[0]), (p3[0], p11[0]), (p4[0], p12[0])], 
    #         ['Housing price', 'Indirect', 'Direct', 'Collateral'], numpoints=1, 
    #         handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=5, frameon=False, fontsize = 12)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Percent from steady state')
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlim(0, H-1)
    ax.axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
    ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax.grid(True, alpha = 0.3)
    plt.show()

    return fig


def linear_irf_nocol(irf_baseline, irf_nocol, model_ss, alt_plot = None, H = 21):
    fig, ax = plt.subplots(1, 3, figsize=(16,12*1/3), sharex = True)
    ax = ax.flatten()

    if alt_plot is not None:
        var_name = alt_plot
    else:
        var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    
    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    for i, (key, value) in enumerate(var_name.items()):   
        if key in devss:
            ax[i].plot(100*(irf_baseline[key][:H]/model_ss[key]))
            ax[i].plot(100*(irf_nocol[key][:H]/model_ss[key]), ls = '--')     
            ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Percent from steady state')
        else:
            ax[i].plot(100*((1 + model_ss[key] +irf_baseline[key][:H])**4-1))
            ax[i].plot(100*((1 + model_ss[key] +irf_nocol[key][:H])**4-1), ls = '--')
            ax[i].axhline(y=100*((1 + model_ss[key])**4 - 1), color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Annualised level, percent')
        
        ax[i].set_title(var_name[key])

        ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax[i].yaxis.set_major_formatter(formatter)
        ax[i].xaxis.set_major_locator(MultipleLocator(4))
        ax[i].set_xlim(0, H -1)
        ax[i].grid(True, alpha=0.3)
        ax[i].set_xlabel('Quarters')

        if i == 0:
            ax[i].legend(['Baseline', 'No collateral constraint'], fontsize=14, frameon=False, loc = 'lower right')

    plt.show()

    return fig

def linear_irf_statedependence(irf_baseline, irf_sw, model_ss, model_ss_sw, alt_plot = None, H = 21):
    fig, ax = plt.subplots(1, 3, figsize=(16,12*1/3), sharex = True)
    ax = ax.flatten()

    if alt_plot is not None:
        var_name = alt_plot
    else:
        # var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}
        var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    
    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    for i, (key, value) in enumerate(var_name.items()):   
        if key in devss:
            ax[i].plot(100*(irf_baseline[key][:H]/model_ss[key]))
            ax[i].plot(100*(irf_sw[key][:H]/model_ss_sw[key]), ls = '--')     
            ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Percent from steady state')
        else:
            ax[i].plot(100*((1 + model_ss[key] + irf_baseline[key][:H])**4-1))
            ax[i].plot(100*((1 + model_ss_sw[key] + irf_sw[key][:H])**4-1), ls = '--')
            ax[i].axhline(y=100*((1 + model_ss[key])**4 - 1), color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Annualised level, percent')
        
        ax[i].set_title(var_name[key])

        ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax[i].yaxis.set_major_formatter(formatter)
        ax[i].xaxis.set_major_locator(MultipleLocator(4))
        ax[i].set_xlim(0, H -1)
        ax[i].grid(True, alpha=0.3)
        ax[i].set_xlabel('Quarters')

        if i == 0:
            ax[i].legend([r'$q^hH/(4\times Y) = 1.6$', r'$q^hH/(4\times Y) = 2.4$'], fontsize=14, frameon=False)

    plt.show()

    return fig

def ltvshock(irf_baseline, irf_sw, model_ss, model_ss_sw, alt_plot = None, H = 21):
    fig, ax = plt.subplots(1, 3, figsize=(16,12*1/3), sharex = True)
    ax = ax.flatten()

    if alt_plot is not None:
        var_name = alt_plot
    else:
        # var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}
        var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    
    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    for i, (key, value) in enumerate(var_name.items()):   
        if key in devss:
            ax[i].plot(100*(irf_baseline[key][:H]/model_ss[key]))
            ax[i].plot(-100*(irf_sw[key][:H]/model_ss_sw[key]), ls = '--')     
            ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Percent from steady state')
        else:
            ax[i].plot(100*((1 + model_ss[key] + irf_baseline[key][:H])**4-1))
            ax[i].plot(100*((1 + model_ss_sw[key] - irf_sw[key][:H])**4-1), ls = '--')
            ax[i].axhline(y=100*((1 + model_ss[key])**4 - 1), color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Annualised level, percent')
        
        ax[i].set_title(var_name[key])

        ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax[i].yaxis.set_major_formatter(formatter)
        ax[i].xaxis.set_major_locator(MultipleLocator(4))
        ax[i].set_xlim(0, H -1)
        ax[i].grid(True, alpha=0.3)
        ax[i].set_xlabel('Quarters')

        if i == 0:
            ax[i].legend(['LTV tightening', 'LTV loosening (mirrored)'], fontsize=14, frameon=False, loc = 'lower right')

    plt.show()

    return fig

def linear_irf_sw(irf_baseline, irf_sw, model_ss, model_ss_sw, alt_plot = None, H = 21):
    fig, ax = plt.subplots(1, 3, figsize=(16,12*1/3), sharex = True)
    ax = ax.flatten()

    if alt_plot is not None:
        var_name = alt_plot
    else:
        # var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}
        var_name = {'Y': r'Output ($Y$)', 'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 
            'pi': r'Inflation ($\pi$)', 'i': r'Policy rate ($i$)'}

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    
    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    for i, (key, value) in enumerate(var_name.items()):   
        if key in devss:
            ax[i].plot(100*(irf_baseline[key][:H]/model_ss[key]))
            ax[i].plot(100*(irf_sw[key][:H]/model_ss_sw[key]), ls = '--')     
            ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Percent from steady state')
        else:
            ax[i].plot(100*((1 + model_ss[key] +irf_baseline[key][:H])**4-1))
            ax[i].plot(100*((1 + model_ss[key] +irf_sw[key][:H])**4-1), ls = '--')
            ax[i].axhline(y=100*((1 + model_ss[key])**4 - 1), color='grey', linestyle='--', linewidth=1.0)
            ax[i].set_ylabel('Annualised level, percent')
        
        ax[i].set_title(var_name[key])

        ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax[i].yaxis.set_major_formatter(formatter)
        ax[i].xaxis.set_major_locator(MultipleLocator(4))
        ax[i].set_xlim(0, H -1)
        ax[i].grid(True, alpha=0.3)
        ax[i].set_xlabel('Quarters')
        if i == 0:
            ax[i].legend(['Sticky prices', 'Sticky wages'], fontsize=14, frameon=False, loc = 'lower right')

    # trans_inc_var = {'Div': r'Dividends ($d$)', 'Tax': r'Tax ($\tau$)', 'N': r'Hours ($N$)', 'r': r'Real rate ($r$)'}

    # for i, (key, value) in enumerate(trans_inc_var.items()):   
    #     ax[5].plot(100*(irf_sw[key][:H]), label=f"{value}")

    # ax[5].axhline(y=0, color='grey', linestyle='--', linewidth=1.0)
    # ax[5].set_ylabel('Deviation from steady state')

    # ax[5].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    # ax[5].yaxis.set_major_formatter(formatter)
    # ax[5].xaxis.set_major_locator(MultipleLocator(4))
    # ax[5].set_xlim(0, H -1)
    # ax[5].set_title('Transitory income variables')
    # ax[5].legend(fontsize=12, frameon=False, ncols = 1, loc = 'upper right')
    # ax[5].grid(True, alpha=0.3)
    plt.show()

    return fig

def compare_with_empirical(irfs, model_ss, H = 16):
    # Define the path to the Excel file
    subfolder = 'Misc'
    filename = 'rrshockirfs_v5.xlsx'  # Replace with your actual file name
    file_path = os.path.join(subfolder, filename)

    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path, sheet_name='gamma', header = None)

    col_name = ['i', 'C_BHAT', 'qh', 'Y', 'pi', 'Investment', 'w', 'Mortgage', 'Rent']
    df.columns = col_name

    bounds_name = ['lb', 'ub']
    col_name_bands = []
    for i in bounds_name:
        for j in col_name:
            col_name_bands.append(f'{i}_{j}')
    df_bands = pd.read_excel(file_path, sheet_name='gammabands', header = None)
    df_bands.columns = col_name_bands

    filename = 'rrshockirfs_v5_68bands.xlsx'  # Replace with your actual file name
    file_path = os.path.join(subfolder, filename)
    df_bands68 = pd.read_excel(file_path, sheet_name='gammabands', header = None)
    df_bands68.columns = col_name_bands

    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    fig, ax = plt.subplots(2, 3, figsize=(16*3/3,8), sharex=True)
    ax = ax.flatten()

    var_name = {'C_BHAT': r'Consumption ($C$)', 'Y': 'Output ($Y$)', 'qh': 'House Price ($q^h$)', 
                'w': r'Wage ($w$)', 'i': r'Policy Rate ($i$)'}
    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    for i, (key, value) in enumerate(var_name.items()):   

        ax[i].plot(100*df[key][:H], ls = '--', color = '#D95319', label = 'Empirical')
        ax[i].fill_between(range(H), 100*df_bands[f"lb_{key}"][:H], 100*df_bands[f"ub_{key}"][:H], 
                        alpha = 0.1, color = '#D95319', edgecolor = None)
        
        if key in devss:   
            ax[i].plot(100*(irfs[key][:H]/model_ss[key]), color = '#0072BD', label = 'Model', zorder = 10)
        else:
            ax[i].plot(100*((1 + irfs[key][:H])**4-1), color = '#0072BD', label = 'Model')
        
        ax[i].set_title(var_name[key])
        ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax[i].set_xlim(0, 16)
        ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
        ax[i].yaxis.set_major_formatter(formatter)
        ax[i].xaxis.set_major_locator(MultipleLocator(4))

        if key != 'i':
            ax[i].set_ylabel('Percent from s.s.')
        else:
            ax[i].set_ylabel('Percentage points from s.s.')

        if i == 3 or i == 4:
            ax[i].set_xlabel('Quarters')

        if i == 2:
            ax[i].legend(fontsize=12, frameon=False)

    ax[5].axis('off')
    fig.subplots_adjust(hspace=0.2, wspace=0.25)
    for i in range(3, 5):
        pos = ax[i].get_position()
        ax[i].set_position([pos.x0 + 0.17, pos.y0, pos.width, pos.height])

    plt.show()

def steady_state_dist(model_ss, hh_name):
    fig, ax = plt.subplots(2, 3, figsize=(17.5*3/3,13*2/3+0.4))
    ax = ax.flatten()

    # Extract the grid for b, h, and z and calculate the marginal distributions
    b_grid = model_ss.internals[hh_name]['b_bhat_grid']
    z_grid = model_ss.internals[hh_name]['z_grid']
    h_grid = model_ss.internals[hh_name]['h_bhat_grid']

    bdmargdist = np.sum(model_ss.internals[hh_name]['D'], axis=0)
    b_margdist = np.sum(bdmargdist,axis=1) # sum out housing
    h_margdist = np.sum(bdmargdist,axis=0) # sum out bonds
    bmargcum = np.cumsum(b_margdist)
    hmargcum = np.cumsum(h_margdist)
    index_h = np.argmin(np.abs(hmargcum - 0.5)) # index where cumulative distribution is closest to 0.5 (median)

    MPCH, mean_MPCH = utils.calc_mpch(model_ss, hh_name)

    MPC, mean_MPC = utils.calc_mpc(model_ss, hh_name)

    #########################################
    ## Fig. 1. Voluntary equity distribution
    #########################################
    i = 0
    ax[i].grid(True, alpha=0.3, zorder = 0)
    ax[i].hist(b_grid, weights=b_margdist, bins=30, edgecolor='k', alpha=1.0, zorder = 2)

    ax[i].set_xlabel(r'Voluntary equity $\hat{b}$')
    ax[i].set_ylabel('Mass of agents')
    ax[i].set_xlim(left=0, right=50)
    ax[i].hist(b_grid[0:1], weights=b_margdist[0:1], bins=1, edgecolor='k', alpha=0.9, zorder = 3)
    ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax[i].set_title('Distribution of voluntary equity')

    ax[i].annotate(f'Mass at constraint = {b_margdist[0]:.2f}',
                xy=(b_grid[0], b_margdist[0]), xycoords='data',
                xytext=(b_grid[0] + 2, b_margdist[0] + 0.1), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left', verticalalignment='top')

    # Calculate the mass of agents with b_margdist corresponding to b_grid > 20
    mass_greater_than_20 = np.sum(b_margdist[b_grid > 50])

    # Add an arrow pointing to the end of the x-axis and show the mass of agents with b_margdist > 20
    ax[i].annotate(f'Mass above = {mass_greater_than_20:.2f}',
                xy=(50, 0), xycoords='data',
                xytext=(40, 0.3), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top')
    plt.tick_params(direction='in', top=True, bottom=True, left=True, right=True)

    #########################################
    ## Fig. 2. Housing distribution
    #########################################
    i = 1
    ax[i].grid(True, alpha=0.3, zorder = 0)
    ax[i].hist(h_grid, weights=h_margdist, bins=30, edgecolor='k', alpha=1.0, zorder = 2)
    ax[i].set_xlim(left=0, right=4)
    ax[i].set_xlabel(r'Housing $h$')
    ax[i].set_ylabel('Mass of agents')
    ax[i].set_title('Distribution of housing')
    ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    # ax[i].set_xlim(left=0, right=4)

    #########################################
    ## Fig. 3. Cum vol. eq. dist.
    #########################################
    i = 2
    ax[i].plot(b_grid, bmargcum, label='b')
    ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax[i].set_xlabel(r'Voluntary equity ($\hat{b}$)')
    ax[i].set_ylabel('Cumulative distribution')
    ax[i].set_title('CDF of voluntary equity')
    ax[i].grid(True, alpha=0.3, zorder = 0)

    #########################################
    ## Fig. 4. Cum housing distribution
    #########################################
    i = 3
    ax[i].plot(h_grid, hmargcum, label='b')
    ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax[i].grid(True, alpha=0.3, zorder = 0)
    ax[i].set_xlabel(r'Housing ($h$)')
    ax[i].set_ylabel('Cumulative distribution')
    ax[i].set_title('CDF of housing')

    #########################################
    ## Fig. 5. MPC, housing wealth
    #########################################
    i = 4

    z_idx = 1
    h_idx = 10
    b_idx = 10

    def format_func(value, tick_number):
        return f'{value:.3f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    ax[i].plot(b_grid, MPCH[z_idx,:,index_h], label=f'z = {z_grid[z_idx]:.2f}, h = {h_grid[index_h]:.2f}', linewidth=2)
    ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax[i].set_xlabel(r'Voluntary equity ($\hat{b}$)')
    ax[i].set_ylabel('MPC out of housing wealth')
    ax[i].set_xlim(-0.35, 40)
    ax[i].yaxis.set_major_formatter(formatter)
    ax[i].grid(True, alpha=0.3, zorder = 0)
    ax[i].set_title('MPC out of housing wealth')
   
    ax[i].text(0.5, 0.95, f'Mean MPCH = {mean_MPCH:.3f}', 
             horizontalalignment='left', verticalalignment='top', transform=ax[i].transAxes)

    #########################################
    ## Fig. 6. MPC, liquid wealth
    #########################################
    i = 5
    ax[i].plot(b_grid[:-1], MPC[z_idx,:-1,index_h], label=f'z = {z_grid[z_idx]:.2f}, h = {h_grid[index_h]:.2f}', linewidth=2)
    ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    # plt.legend(frameon=False)
    ax[i].set_xlabel(r'Voluntary equity ($\hat{b}$)')
    ax[i].set_ylabel('MPC out of liquid wealth')
    ax[i].set_xlim(-0.35, 40)
    ax[i].yaxis.set_major_formatter(formatter)
    ax[i].grid(True, alpha=0.3, zorder = 0)
    ax[i].set_title('MPC out of liquid wealth')
    ax[i].text(0.5, 0.95, f'Mean MPC = {mean_MPC:.3f}', 
            horizontalalignment='left', verticalalignment='top', transform=ax[i].transAxes)

    fig.subplots_adjust(hspace=0.4, wspace=0.25)
    plt.show()

    return fig

def bh_distribution_3d(model_ss, hh_name):
    XX, YY = np.meshgrid(model_ss.internals[hh_name]['h_bhat_grid'],model_ss.internals[hh_name]['b_bhat_grid'])
    bdmargdist = np.sum(model_ss.internals[hh_name]['D'], axis=0)

    b_lim = 55
    h_lim = 4
    bdmargdist[model_ss.internals[hh_name]['b_bhat_grid'] > b_lim] = np.nan
    bdmargdist[:,model_ss.internals[hh_name]['h_bhat_grid'] > h_lim] = np.nan

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(111, projection='3d')
    surf2 = ax.plot_surface(XX, YY, bdmargdist, cmap='coolwarm', alpha=1.0, label = ' z = 0',
                            edgecolor='k', rcount = 20, ccount = 20)
    ax.set_xlabel('Housing')
    ax.set_ylabel('Voluntary equity')
    ax.set_zlabel('Mass of agents')
    ax.set_xlim(0, h_lim)
    ax.set_ylim(0, b_lim)
    ax.view_init(elev=30, azim=30)

    fig.tight_layout()
    plt.show()

    return fig

def mpch_dist(model_ss, hh_name, b_plot_start = 0):
    MPCH, mean_MPC = utils.calc_mpch(model_ss, hh_name)
    MPCH_average = np.mean(MPCH, axis=0)
    XX, YY = np.meshgrid(model_ss.internals[hh_name]['h_bhat_grid'],model_ss.internals[hh_name]['b_bhat_grid'])

    b_lim = 55
    h_lim = 4
    MPCH_average[model_ss.internals[hh_name]['b_bhat_grid'] > b_lim] = np.nan
    MPCH_average[:,model_ss.internals[hh_name]['h_bhat_grid'] > h_lim] = np.nan
    
    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(111, projection='3d')
    b_plot_start = b_plot_start
    surf2 = ax.plot_surface(XX[:-1,b_plot_start:-1], YY[:-1,b_plot_start:-1], MPCH_average[:-1,b_plot_start:-1], cmap='coolwarm', alpha=1.0, label = ' z = 0',
                            edgecolor='k', rcount = 20, ccount = 20)
    ax.set_xlabel('Housing')
    ax.set_ylabel('Voluntary equity')
    ax.set_zlabel('MPC out of housing wealth')
    ax.set_xlim(0, h_lim)
    ax.set_ylim(0, b_lim)
    ax.view_init(elev=30, azim=30)
    fig.tight_layout()
    plt.show()

    return fig

def cumulative_effects(decomp, model_ss):
    # print(f"Total cumulative effect: {np.round(100*np.cumsum(decomp['total']/model_ss['C_BHAT'])[[0, 3, 7, 11, 15]],2)}")
    # print(f"qh cumulative effect: {np.round(100*np.cumsum(decomp['housing']/model_ss['C_BHAT'])[[0, 3, 7, 11, 15]],2)}")
    # print(f"indirect cumulative effect: {np.round(100*np.cumsum(decomp['indirect']/model_ss['C_BHAT'])[[0, 3, 7, 11, 15]],2)}")
    # print(f"Direct cumulative effect: {np.round(100*np.cumsum(decomp['direct']/model_ss['C_BHAT'])[[0, 3, 7, 11, 15]],2)}")
    # print(f"Collateral cumulative effect: {np.round(100*np.cumsum(decomp['collateral']/model_ss['C_BHAT'])[[0, 3, 7, 11, 15]],2)}")
    # Calculate cumulative effects
    total_cumulative = np.round(100 * np.cumsum(decomp['total'] / model_ss['C_BHAT'])[[0, 3, 7, 11, 15]], 2)
    qh_cumulative = np.round(100 * np.cumsum(decomp['housing'] / model_ss['C_BHAT'])[[0, 3, 7, 11, 15]], 2)
    indirect_cumulative = np.round(100 * np.cumsum(decomp['indirect'] / model_ss['C_BHAT'])[[0, 3, 7, 11, 15]], 2)
    direct_cumulative = np.round(100 * np.cumsum(decomp['direct'] / model_ss['C_BHAT'])[[0, 3, 7, 11, 15]], 2)
    collateral_cumulative = np.round(100 * np.cumsum(decomp['collateral'] / model_ss['C_BHAT'])[[0, 3, 7, 11, 15]], 2)

    # Create a table
    table_data = [
        ["Total"] + total_cumulative.tolist(),
        ["House price"] + qh_cumulative.tolist(),
        ["Collateral"] + collateral_cumulative.tolist(),
        ["Indirect"] + indirect_cumulative.tolist(),
        ["Direct"] + direct_cumulative.tolist()
    ]

    # Generate LaTeX table
    latex_table = tabulate(table_data, headers=["Effect", "0", "3", "7", "11", "15"], tablefmt="latex")

    # Print the LaTeX table
    print(latex_table)

    housing_cum_share = np.round(np.cumsum(decomp['housing']) / np.cumsum(decomp['total']), 2)
    indirect_cum_share = np.round(np.cumsum(decomp['indirect']) / np.cumsum(decomp['total']),2)
    direct_cum_share = np.round(np.cumsum(decomp['direct']) / np.cumsum(decomp['total']),2)
    collateral_cum_share = np.round(np.cumsum(decomp['collateral']) / np.cumsum(decomp['total']),2)

    housing_cum_share = 100*housing_cum_share[[0, 3, 7, 11, 15]]
    indirect_cum_share = 100*indirect_cum_share[[0, 3, 7, 11, 15]]
    direct_cum_share = 100*direct_cum_share[[0, 3, 7, 11, 15]]
    collateral_cum_share = 100*collateral_cum_share[[0,3, 7, 11, 15]]

    # Create a table
    table_data = [
        ["House price"] + housing_cum_share.tolist(),
        ["Collateral"] + collateral_cum_share.tolist(),
        ["Indirect"] + indirect_cum_share.tolist(),
        ["Direct"] + direct_cum_share.tolist()
    ]

    # Generate LaTeX table
    latex_table = tabulate(table_data, headers=["Effect", "0", "3", "7", "11", "15"], tablefmt="latex")

    # Print the LaTeX table
    print(latex_table)

def linear_irf_diffmpc(irf_dict, model_ss_dict, alt_plot = None, H = 21):
    fig, ax = plt.subplots(1, 3, figsize=(16,12*1/3), sharex = True)
    ax = ax.flatten()

    if alt_plot is not None:
        var_name = alt_plot
    else:
        var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    
    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    linestyles = ['-', '--', '-.', ':']
    labels = ['MPC = 0.15', 'MPC = 0.20', 'MPC = 0.25']
    for i, (key, value) in enumerate(var_name.items()):
        for j, (key_model, value) in enumerate(irf_dict.items()):
            if key_model in ['baseline', 'mpc20', 'mpc25']:   
                if key in devss:
                    ax[i].plot(100*(irf_dict[key_model][key][:H]/model_ss_dict[key_model][key]), ls = linestyles[j], label = labels[j])
                    ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
                    ax[i].set_ylabel('Percent from steady state')
                else:
                    ax[i].plot(100*((1 + model_ss_dict[key_model][key] +irf_dict[key_model][key][:H])**4-1), ls = linestyles[j])
                    ax[i].axhline(y=100*((1 + model_ss_dict[key_model][key])**4 - 1), color='grey', linestyle='--', linewidth=1.0)
                    ax[i].set_ylabel('Annualised level, percent')
                
                ax[i].set_title(var_name[key])

                ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
                ax[i].yaxis.set_major_formatter(formatter)
                ax[i].xaxis.set_major_locator(MultipleLocator(4))
                ax[i].set_xlim(0, H -1)
                ax[i].grid(True, alpha=0.3)
                ax[i].set_xlabel('Quarters')
                if i == 0:
                    ax[i].legend(fontsize=14, frameon=False, loc = 'lower right')

    plt.show()

    return fig

def linear_irf_diffgamma(irf_dict, model_ss_dict, gamma_names, gamma_list, alt_plot = None, H = 21):
    fig, ax = plt.subplots(1, 3, figsize=(16,12*1/3), sharex = True)
    ax = ax.flatten()

    if alt_plot is not None:
        var_name = alt_plot
    else:
        var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    
    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    linestyles = ['-', '--', '-.', ':', '-']
    labels = ['MPC = 0.15', 'MPC = 0.20', 'MPC = 0.25']
    
    for i, (key, value) in enumerate(var_name.items()):
        ls_count = 0
        for j, (key_model, value) in enumerate(irf_dict.items()):
            if key_model in gamma_names:   
                if key in devss:
                    # print(ls_count)
                    ax[i].plot(100*(irf_dict[key_model][key][:H]/model_ss_dict[key_model][key]),
                               ls = linestyles[ls_count],
                               label = fr"$\gamma =$ {gamma_list[ls_count]}, MPC = {model_ss_dict[key_model]['MPC']:.3f}")
                    ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
                    ax[i].set_ylabel('Percent from steady state')
                    ls_count += 1
                    # print(ls_count)
                else:
                    ax[i].plot(100*((1 + model_ss_dict[key_model][key] +irf_dict[key_model][key][:H])**4-1),
                               ls = linestyles[ls_count],
                               label = fr"$\gamma =$ {gamma_list[ls_count]}, MPC = {model_ss_dict[key_model]['MPC']:.3f}")
                    ax[i].axhline(y=100*((1 + model_ss_dict[key_model][key])**4 - 1), color='grey', linestyle='--', linewidth=1.0)
                    ax[i].set_ylabel('Annualised level, percent')
                    ls_count += 1
                ax[i].set_title(var_name[key])

                ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
                ax[i].yaxis.set_major_formatter(formatter)
                ax[i].xaxis.set_major_locator(MultipleLocator(4))
                ax[i].set_xlim(0, H -1)
                ax[i].grid(True, alpha=0.3)
                ax[i].set_xlabel('Quarters')
                if i == 2:
                    ax[i].legend(fontsize=14, frameon=False, loc = 'lower right')

    plt.show()

    return fig

def linear_irf_diffkappa(irf_dict, model_ss_dict, gamma_names, gamma_list, alt_plot = None, H = 21):
    fig, ax = plt.subplots(1, 3, figsize=(16,12*1/3), sharex = True)
    ax = ax.flatten()

    if alt_plot is not None:
        var_name = alt_plot
    else:
        var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    
    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    linestyles = ['-', '--', '-.', ':', '-']
    labels = ['MPC = 0.15', 'MPC = 0.20', 'MPC = 0.25']
    
    for i, (key, value) in enumerate(var_name.items()):
        ls_count = 0
        for j, (key_model, value) in enumerate(irf_dict.items()):
            if key_model in gamma_names:   
                if key in devss:
                    # print(ls_count)
                    ax[i].plot(100*(irf_dict[key_model][key][:H]/model_ss_dict[key]),
                               ls = linestyles[ls_count],
                               label = fr"$\kappa =$ {gamma_list[ls_count]:.3f}")
                    ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
                    ax[i].set_ylabel('Percent from steady state')
                    ls_count += 1
                    # print(ls_count)
                else:
                    ax[i].plot(100*((1 + model_ss_dict[key] +irf_dict[key_model][key][:H])**4-1),
                               ls = linestyles[ls_count],
                               label = fr"$\kappa =$ {gamma_list[ls_count]:.3f}")
                    ax[i].axhline(y=100*((1 + model_ss_dict[key])**4 - 1), color='grey', linestyle='--', linewidth=1.0)
                    ax[i].set_ylabel('Annualised level, percent')
                    ls_count += 1
                ax[i].set_title(var_name[key])

                ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
                ax[i].yaxis.set_major_formatter(formatter)
                ax[i].xaxis.set_major_locator(MultipleLocator(4))
                ax[i].set_xlim(0, H -1)
                ax[i].grid(True, alpha=0.3)
                ax[i].set_xlabel('Quarters')
                if i == 2:
                    ax[i].legend(fontsize=14, frameon=False, loc = 'lower right')

    plt.show()

    return fig

def linear_irf_diffphi(irf_dict, model_ss_dict, gamma_names, gamma_list, alt_plot = None, H = 21):
    fig, ax = plt.subplots(1, 3, figsize=(16,12*1/3), sharex = True)
    ax = ax.flatten()

    if alt_plot is not None:
        var_name = alt_plot
    else:
        var_name = {'C_BHAT': r'Consumption ($C$)', 'qh': r'House price ($q^h$)', 'pi': r'Inflation ($\pi$)'}

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']
    
    def format_func(value, tick_number):
        return f'{value:.1f}'  # Format with 2 decimal places

    formatter = FuncFormatter(format_func)

    linestyles = ['-', '--', '-.', ':', '-']
    labels = ['MPC = 0.15', 'MPC = 0.20', 'MPC = 0.25']
    
    for i, (key, value) in enumerate(var_name.items()):
        ls_count = 0
        for j, (key_model, value) in enumerate(irf_dict.items()):
            if key_model in gamma_names:   
                if key in devss:
                    # print(ls_count)
                    ax[i].plot(100*(irf_dict[key_model][key][:H]/model_ss_dict[key]),
                               ls = linestyles[ls_count],
                               label = fr"$\phi =$ {gamma_list[ls_count]:.2f}")
                    ax[i].axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
                    ax[i].set_ylabel('Percent from steady state')
                    ls_count += 1
                    # print(ls_count)
                else:
                    ax[i].plot(100*((1 + model_ss_dict[key] +irf_dict[key_model][key][:H])**4-1),
                               ls = linestyles[ls_count],
                               label = fr"$\phi =$ {gamma_list[ls_count]:.2f}")
                    ax[i].axhline(y=100*((1 + model_ss_dict[key])**4 - 1), color='grey', linestyle='--', linewidth=1.0)
                    ax[i].set_ylabel('Annualised level, percent')
                    ls_count += 1
                ax[i].set_title(var_name[key])

                ax[i].tick_params(direction='in', top=True, bottom=True, left=True, right=True)
                ax[i].yaxis.set_major_formatter(formatter)
                ax[i].xaxis.set_major_locator(MultipleLocator(4))
                ax[i].set_xlim(0, H -1)
                ax[i].grid(True, alpha=0.3)
                ax[i].set_xlabel('Quarters')
                if i == 2:
                    ax[i].legend(fontsize=14, frameon=False, loc = 'lower right')

    plt.show()

    return fig

def htm_share(model_ss, irf_nonlin, hh_name, T):    
    constrained_share_pos = []
    constrained_share_neg = []
    fig, ax = plt.subplots()
    H = 9
    for t in range(T):
        bmarg_pos = np.sum(model_ss.internals[hh_name]['D'] + irf_nonlin['pos'].internals[hh_name]['D'][t], axis = (0,2))
        cons_share = bmarg_pos[0]
        constrained_share_pos.append(cons_share)

        bmarg_neg = np.sum(model_ss.internals[hh_name]['D'] + irf_nonlin['neg'].internals[hh_name]['D'][t], axis = (0,2))
        cons_share = bmarg_neg[0]
        constrained_share_neg.append(cons_share)
    ax.plot(constrained_share_neg[:H], label = 'Expansionary shock, 29 bps')
    ax.plot(constrained_share_pos[:H], label = 'Contractionary shock, 29 bps')

    constrained_share_pos = []
    constrained_share_neg = []
    for t in range(T):
        bmarg_pos = np.sum(model_ss.internals[hh_name]['D'] + irf_nonlin['pos_100bps'].internals[hh_name]['D'][t], axis = (0,2))
        cons_share = bmarg_pos[0]
        constrained_share_pos.append(cons_share)

        bmarg_neg = np.sum(model_ss.internals[hh_name]['D'] + irf_nonlin['neg_100bps'].internals[hh_name]['D'][t], axis = (0,2))
        cons_share = bmarg_neg[0]
        constrained_share_neg.append(cons_share)
    # print(constrained_share_pos[:H])
    # print(constrained_share_neg[:H])

    ax.plot(constrained_share_neg[:H], label = 'Expansionary shock, 100 bps', linestyle = '--', color = '#0072BD')
    ax.plot(constrained_share_pos[:H], label = 'Contractionary shock, 100 bps', linestyle = '--', color = '#D95319')
    ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim(0, H - 1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Quarters')
    ax.set_ylabel(r'Share of HtM agents ($\hat{b} = 0$)')
    ax.legend(frameon=False, loc='lower right', fontsize=12)
    plt.show()

    return fig

def nonlin_consumption(model_ss, irf_nonlin, key = 'C_BHAT', H = 9, big_shock = False):    
    fig, ax = plt.subplots()

    devss = ['Y', 'qh', 'C_BHAT', 'w', 'N', 'Div', 'Tax', 'CHI']

    if big_shock:
        if key in devss:
            ax.plot(-100*(irf_nonlin['neg_100bps'][key][:H]/model_ss[key]), label = 'Expansionary (mirrored)')
            ax.plot(100*(irf_nonlin['pos_100bps'][key][:H]/model_ss[key]), label = 'Contractionary', ls = 'dashed')
        else:
            ax.plot(-100*((1 + model_ss[key][key] +irf_nonlin['neg_100bps'][key][:H])**4-1))
            ax.plot(100*((1 + model_ss[key][key] +irf_nonlin['pos_100bps'][key][:H])**4-1), ls = '--')
    else:
        if key in devss:
            ax.plot(-100*(irf_nonlin['neg'][key][:H]/model_ss[key]), label = 'Expansionary (mirrored)')
            ax.plot(100*(irf_nonlin['pos'][key][:H]/model_ss[key]), label = 'Contractionary', ls = 'dashed')
            # ax.set_ylim(-0.85, 0.)
        else:
            ax.plot(-100*((1 + model_ss[key] +irf_nonlin['neg'][key][:H])**4-1), label = 'Expansionary (mirrored)')
            ax.plot(100*((1 + model_ss[key] +irf_nonlin['pos'][key][:H])**4-1), label = 'Contractionary', ls = 'dashed')
        ax.legend(frameon=False)  
    
    ax.axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
    ax.set_ylabel('Percent from steady state')
    ax.set_xlabel('Quarters')
    ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim(0, H - 1)
    ax.grid(True, alpha=0.3)

    return fig

def changeLTV_creditcapacity(irf_lin_dict, model_ss_dict, H = 21):

    fig, ax = plt.subplots()
    ax.plot(100*irf_lin_dict['baseline']['C_BHAT'][:H]/model_ss_dict['baseline']['C_BHAT'], label = r'$q^hH/(4\times Y) = 1.6, \gamma = 0.85$')
    ax.plot(100*irf_lin_dict['highwealth']['C_BHAT'][:H]/model_ss_dict['highwealth']['C_BHAT'], ls = 'dashed', label = r'$q^hH/(4\times Y) = 2.4, \gamma = 0.85$')
    ax.plot(100*irf_lin_dict['highwealth_lowgam']['C_BHAT'][:H]/model_ss_dict['highwealth_lowgam']['C_BHAT'], 
            ls = 'dashdot', label = r'$q^hH/(4\times Y) = 2.4, \gamma = 0.57$', )

    ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlim(0, H - 1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Percent from steady state')
    ax.legend(frameon=False, loc = 'lower right')

    return fig

def persistentMPshock(irf_nonlin, model_ss):
    fig, ax = plt.subplots()
    H = 17
    plt.plot(-100*irf_nonlin['neg_noborr']['C_BHAT'][:H]/model_ss['C_BHAT'], label = 'Expansionary (mirrored)')
    plt.plot(100*irf_nonlin['pos_noborr']['C_BHAT'][:H]/model_ss['C_BHAT'], ls = 'dashed', label = 'Contractionary')

    ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlim(0, H - 1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Percent from steady state')
    ax.legend(frameon=False)

    plt.show()

    return fig

def generic_twoirfs(irf1, irf2, model_ss1, model_ss2, key, legend_text, H = 21):
    fig, ax = plt.subplots()

    ax.plot(100*irf1['C_BHAT'][:H]/model_ss1[key])
    ax.plot(100*irf2['C_BHAT'][:H]/model_ss2[key], ls = 'dashed')
    ax.axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)

    ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlim(0, H - 1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Percent from steady state')
    ax.legend(legend_text, frameon=False)

    plt.show()

    return fig

def plot_c_dist(model_ss, c_dist, t_plot, hh_name, var, z_plot = None, dev_ss = False):
    XX, YY = np.meshgrid(model_ss.internals[hh_name]['h_bhat_grid'],model_ss.internals[hh_name]['b_bhat_grid'])

    if dev_ss:
        if z_plot is None:
            c_dist_dev = (c_dist - model_ss.internals[hh_name]['D']*model_ss.internals[hh_name][var])
            c_dist_sum = np.sum(c_dist_dev, axis = 1)
            c_dist_plot = c_dist_sum[t_plot]
            print(np.sum(c_dist_plot))
        else: 
            c_dist_dev = c_dist - model_ss.internals[hh_name]['D']*model_ss.internals[hh_name][var]
            c_dist_plot = c_dist_dev[t_plot, z_plot]
    else:
        if z_plot is None:
            c_dist_sum = np.sum(c_dist, axis = 1)
            c_dist_plot = c_dist_sum[t_plot]
        else:
            c_dist_plot = c_dist[t_plot, z_plot]

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(111, projection='3d')

    b_lim = 49 #model_ss.internals[hh_name]['b_bhat_grid'][5] #55
    h_lim = 4
    c_dist_plot[model_ss.internals[hh_name]['b_bhat_grid'] > b_lim] = np.nan
    c_dist_plot[:,model_ss.internals[hh_name]['h_bhat_grid'] > h_lim] = np.nan

    surf2 = ax.plot_surface(XX, YY, c_dist_plot, cmap='coolwarm', alpha=1.0, label = ' z = 0',
                            edgecolor='k', rcount = 20, ccount = 20)
    ax.set_xlabel('Housing')
    ax.set_ylabel('Voluntary equity')
    ax.set_zlabel('Mass of agents')
    ax.view_init(elev=30)
    ax.set_xlim(0, h_lim)
    ax.set_ylim(0, b_lim + 1)
    fig.tight_layout()
    plt.show()

def linear_nonlinear_ltv(model_lin, irf_lin, irf_ltv_shock, H = 17):
        
        fig, ax = plt.subplots()
        
        ax.plot(100*irf_lin['neg_rho60']['C_BHAT'][:21]/model_lin['ltv_baseline']['C_BHAT'], 
                label=r'Linear, $\rho_\gamma = 0.60$')
        ax.plot(100*irf_ltv_shock['neg_rho60']['C_BHAT'][:21]/model_lin['ltv_baseline']['C_BHAT'], 
                label=r'Non-linear, $\rho_\gamma = 0.60$', linestyle='--')
        ax.plot(100*irf_lin['neg_rho95']['C_BHAT'][:21]/model_lin['ltv_baseline']['C_BHAT'],
                label=r'Linear, $\rho_\gamma = 0.95$')
        ax.plot(100*irf_ltv_shock['neg_rho95']['C_BHAT'][:21]/model_lin['ltv_baseline']['C_BHAT'], 
                label=r'Non-linear, $\rho_\gamma = 0.95$', linestyle='--')

        ax.axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
        ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.set_xlim(0, H - 1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('Percent from steady state')
        ax.legend(frameon=False)

        return fig

def nonlinear_MPC(model_ss, irf_nonlin, H = 9):
        
        fig, ax = plt.subplots()
        
        ax.plot(100*irf_nonlin['pos']['MPC'][:H]/model_ss['MPC'], 
                label=r'Expansionary')
        ax.plot(-100*irf_nonlin['neg']['MPC'][:H]/model_ss['MPC'], 
                label=r'Contractionary (mirrored)', linestyle='--')
        ax.axhline(y=0.0, color='grey', linestyle='--', linewidth=1.0)
        
        ax.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_xlim(0, H - 1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('Percent from steady state')
        # ax.legend(frameon=False)

        return fig