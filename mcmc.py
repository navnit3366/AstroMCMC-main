import os
import emcee
import corner
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count

from norm_algo import norm_df
from predict import param_interpol
from log_prob import ln_posterior, ln_likelihood

def get_inputs():

    parser = argparse.ArgumentParser(description='Script to get input parameters.')

    parser.add_argument('--obs_file_path', type=str, help='Path to the observed spectrum')
    parser.add_argument('--params', type=str, nargs='+', help='Parameter names')
    parser.add_argument('--param_range', type=str, nargs='+', help='Parameter ranges')
    parser.add_argument('--truth_val', type=float, nargs='+', help='True parameter values')
    parser.add_argument('--nwalkers', type=int, help='Number of walkers')
    parser.add_argument('--nsteps', type=int, help='Number of steps')
    parser.add_argument('--wave_min', type=int, help='Minimum wavelength (open end)')
    parser.add_argument('--wave_max', type=int, help='Maximum wavelength (open end)')
    parser.add_argument('--is_grid', type=int, help='Use a model grid(0) or parameter interpolation(1)')
    parser.add_argument('--choice', type=int, help='Initialization choice (1, 2, or 3)')
    parser.add_argument('--spread', type=float, nargs='+', help='Spread around the initial parameters')
    parser.add_argument('--use_telluric', type=int, help='Consider telluric regions (True or False)')
    parser.add_argument('--initial_params', type=float, nargs='+', help='Initial parameters (only needed if choice 1 is selected)')

    args = parser.parse_args()
    param_range = [list(map(float, i.split())) for i in args.param_range[0].split(',')]

    choice = args.choice
    spread = args.spread
    nwalkers = args.nwalkers
    ndim = len(args.params)

    if choice == 1:
        pos = args.initial_params + spread * np.random.randn(nwalkers, ndim)
    elif choice == 2:
        param_range = np.array(param_range)
        pos = np.random.uniform(low=param_range[:,0], high=param_range[:,1], size=(nwalkers, ndim))
    elif choice == 3:
        pos = []

    return args.params, param_range, ndim, nwalkers, args.nsteps, args.wave_min, args.wave_max, pos, args.truth_val, args.is_grid, args.obs_file_path, args.use_telluric


if __name__ == '__main__':
    params, param_range, ndim, nwalkers, nsteps, wave_min, wave_max, pos, truth_val, is_grid, obs_file_path, use_telluric = get_inputs()

    newpath = os.path.join(os.getcwd(), 'mcmc_data')
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # read observed spectrum file
    df_obs = pd.read_table(obs_file_path, delim_whitespace=True, header=None)
    if df_obs.shape[1] == 2:
        df_obs.columns = ['wave', 'flux']
    if df_obs.shape[1] == 3:
        df_obs.columns = ['wave', 'flux', 'error']

    # sort the observed spectrum
    arg_idx = np.argsort(df_obs['wave'])
    df_obs = df_obs.iloc[arg_idx]

    # drop duplicate wavelengths
    df_obs = df_obs.drop_duplicates(subset='wave', keep='first')

    # trim the observed spectrum in range of interest
    df_obs = df_obs[(df_obs['wave'] > wave_min) & (df_obs['wave'] < wave_max)]

    # define telluric regions
    if use_telluric:
        print('\nTaking Telluric regions pre-mentioned in code.')
        gaprange = [[8200,8390]]
        telluric_ranges = [[6860, 6960],[7550, 7750],[8200, 8430],[8930,9000]]
        telluric_ranges += gaprange

        # trim the observed spectrum
        for i in telluric_ranges:
            df_obs = df_obs[(df_obs['wave'] < i[0]) | (df_obs['wave'] > i[1])]
    else:
        print('\nNot considering Telluric regions.')

    # reset the index and normalize the flux
    df_obs = df_obs.reset_index(drop=True)
    # df_obs = norm_df(df_obs)

    if df_obs.shape[1] == 2:
        SNR = 32
        error_eso = df_obs['flux'] / SNR
        print('\nNo error column found, taking default error SNR = 32')
    if df_obs.shape[1] == 3:
        error_eso = df_obs['error']
        print('\nIndepedent error column found')

    if len(pos) == 0:
        np.random.seed(42)
        nll = lambda *args: -ln_likelihood(*args)
        initial = np.array(truth_val) + 1e-1 * np.random.randn(3)
        soln = minimize(nll, initial, args=(df_obs['wave'], df_obs['flux'], error_eso, is_grid, use_telluric))
        teff_init, logg_init, m_init = soln.x

        pos = soln.x + [50, 0.2, 0.2] * np.random.randn(nwalkers, ndim)

        print("\nMaximum likelihood estimates:")
        print("Teff = {0:.3f}".format(teff_init))
        print("logg = {0:.3f}".format(logg_init))
        print("m = {0:.3f}".format(m_init))

    # cpu count
    ncpu = cpu_count()
    print('\nNumber of CPUs availble for multiprocessing:', ncpu)

    # warning message
    print("\nWarning -> If you have a data/result image files in the mcmc_data directory from previous MCMC run, it will be overwritten.\nPlease move or delete the old files.")
    if input("type 'run' when ready: ").strip() == 'run':
        pass
    else:
        exit()

    # Set up the backend
    filename = "./mcmc_data/data.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    print('\nRunning MCMC...')
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, args=(df_obs['wave'], df_obs['flux'], error_eso, param_range, is_grid, use_telluric), pool=pool, backend=backend, moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)],)
        sampler.run_mcmc(pos, nsteps, progress=True, store=True)

    print('MCMC finished.')

    # acceptance fraction
    print('\nAcceptance fraction:', np.mean(sampler.acceptance_fraction),'\n')

    # autocorrelation time
    print('\nAutocorrelation analysics:', sampler.get_autocorr_time(quiet=True))

    # convergence plot
    samples = sampler.get_chain()
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    fig.suptitle('Trace Plot')

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], 'k', alpha=0.3)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel('Step')
    for ax, label in zip(axes, params):
        ax.set_ylabel(label)

    plt.savefig('./mcmc_data/trace_plot.png')
    plt.show()
    
    # print final parameter values
    print('\nlast parameter values:')
    print(sampler.flatchain[-1],'\n')

    # make corner plot
    fig = corner.corner(sampler.flatchain,
                        labels=params,
                        show_titles=True,
                        range=param_range,
                        quantiles=[0.16, 0.5, 0.84],
                        title_kwargs={"fontsize": 12})
    
    plt.savefig('./mcmc_data/corner.png')
    plt.show()

    # print the parameter values and errors
    final_theta = []
    error = []
    for i in range(ndim):
        mcmc = np.percentile(sampler.get_chain(flat = True)[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(params[i], '--->',mcmc[1], '  err  ', -q[0],', ', q[1])
        final_theta.append(mcmc[1])
        error.append([q[0],q[1]])

    # make best fit graph
    fig = plt.figure(figsize=(15,5))
    best_fit = param_interpol(final_theta[0], final_theta[1], final_theta[2], df_obs['wave'], use_telluric= use_telluric)
    plt.plot(df_obs['wave'], df_obs['flux'], label="Observation")
    plt.plot(df_obs['wave'], best_fit, label= "Best Fit")
    plt.legend()
    plt.title('Best Fit')
    plt.savefig('./mcmc_data/best_fit.png')
    plt.show()

    # save best fit data to txt file
    best_fit_df = pd.DataFrame({'wave':df_obs['wave'], 'flux':best_fit})
    best_fit_df.to_csv('./mcmc_data/best_fit.txt', sep='\t', index=False)

    # best fit with the error band
    fig = plt.figure(figsize=(15,5))
    lower_error = param_interpol(final_theta[0] - error[0][0], final_theta[1] - error[1][0], final_theta[2] - error[2][0], df_obs['wave'])
    upper_error = param_interpol(final_theta[0] + error[0][1], final_theta[1] + error[1][1], final_theta[2] + error[2][1], df_obs['wave'])
    plt.fill_between(df_obs['wave'], lower_error, upper_error, alpha=0.5, label='Error Band')
    plt.plot(df_obs['wave'], df_obs['flux'], label="Observation")
    plt.plot(df_obs['wave'], best_fit, label= "Best Fit")
    plt.title('Best Fit with Error Band')
    plt.legend()
    plt.savefig('./mcmc_data/best_fit_error.png')
    plt.show()

    # SNR error band and best fit
    if df_obs.shape[1] == 2:
        fig = plt.figure(figsize=(15,5))
        error_best_fit = best_fit / SNR
        plt.fill_between(df_obs['wave'], best_fit - error_best_fit, best_fit + error_best_fit, alpha=0.5, label='Error Band (SNR = 32)')
        plt.plot(df_obs['wave'], best_fit, label= "Best Fit")
        plt.title('Best Fit with SNR Error Band')
        plt.legend()
        plt.savefig('./mcmc_data/best_fit_snr.png')
        plt.show()