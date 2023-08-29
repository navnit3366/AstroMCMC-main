from scipy.interpolate import RegularGridInterpolator, interp1d
import pandas as pd
import numpy as np
import os
import re

def param_interpol(teff_value, logg_value, m_value, x, use_telluric=True):

    # get file names in directory
    dir  =  os.getcwd() + '/norm_models/'
    files = os.listdir(dir)

    # get param from the files
    pattern = r'(\d{3})-([\d.]+)-([\d.-]+)'

    teff = []
    logg = []
    m = []

    for i in files:
        matches = re.search(pattern, i)
        teff.append(float(matches.group(1)))
        logg.append(float(matches.group(2)))
        m.append(float(matches.group(3)))

    teff = np.array(teff) * 100
    m = np.array(m) * -1

    # calculate distance from other points 
    dist = []
    for i in range(len(files)):
        dist.append(np.sqrt((teff_value - teff[i])**2 + (logg_value - logg[i])**2 + (m_value - m[i])**2))

    # get the 'k' closest points
    k = 20
    dist = np.array(dist)
    idx = np.argsort(dist)[:k]

    # removing not needed points
    teff = teff[idx]
    logg = np.array(logg)[idx]
    m = m[idx]

    # removing not needed file paths
    files = np.array(files)[idx]

    # define telluric regions
    if use_telluric:
        gaprange = [[8200,8390]]
        telluric_ranges = [[6860, 6960],[7550, 7750],[8200, 8430],[8930,9000]]
        telluric_ranges += gaprange

    wave = x
    min_wave = x.min()
    max_wave = x.max()

    flux_arrays = []

    # iterate through all models
    for i in range(len(files)):

        # read in model
        df_model = pd.read_pickle(dir + files[i])
        df_model = df_model.iloc[:,[0,1]]
        df_model.columns = ['wave', 'flux']

        # find the value of just smaller than min_wave and just larger than max_wave
        min_wave_model = df_model['wave'][df_model['wave'] < min_wave].max()
        max_wave_model = df_model['wave'][df_model['wave'] > max_wave].min()

        # trim the model to the observed wavelength range
        df_model = df_model[(df_model['wave'] >= min_wave_model) & (df_model['wave'] <= max_wave_model)]

        # remove telluric regions
        if use_telluric:
            for j in telluric_ranges:
                df_model = df_model[(df_model['wave'] < j[0]) | (df_model['wave'] > j[1])]

        # remove duplicate wavelengths
        df_model = df_model.drop_duplicates(subset='wave', keep='first')

        df_model = df_model.reset_index(drop=True)
        
        # interpolate model to match observed wavelength grid
        f = interp1d(df_model['wave'], df_model['flux'], kind='cubic')
        
        flux_arrays.append(f(wave))


    # build grid axes
    teff_grid_val = np.arange(teff.min(), teff.max() + 100, 100)
    logg_grid_val = np.arange(logg.min(), logg.max() + 0.5, 0.5)
    m_grid_val = np.arange(m.min(), m.max() + 0.5, 0.5)

    # zip all the parameters together
    all_param = list(zip(teff, logg, m))

    # make a new 4D array of fluxes corresponding to teff, logg, m and wave
    flux4D = np.zeros((len(teff_grid_val), len(logg_grid_val), len(m_grid_val), len(wave)))

    for i in range(len(teff_grid_val)):
        for j in range(len(logg_grid_val)):
            for k in range(len(m_grid_val)):
                try: 
                    flux4D[i,j,k,:] = flux_arrays[all_param.index((teff_grid_val[i], logg_grid_val[j], m_grid_val[k]))]
                except:
                    pass

    # interpolating through grid 
    rgi = RegularGridInterpolator([teff_grid_val, logg_grid_val, m_grid_val, wave], flux4D, bounds_error=False, fill_value=None)

    flux = []
    for i in range(len(x)):
        point = np.array([teff_value, logg_value, m_value, x[i]])
        flux.append(rgi(point)[0])


    return np.array(flux)