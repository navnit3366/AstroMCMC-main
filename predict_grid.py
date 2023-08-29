from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import os
import re

def param_grid(teff_value, logg_value, m_value, x, use_telluric=True):

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


    dist = np.array(dist)
    idx = np.argsort(dist)[0]

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
    min_wave = np.min(wave)
    max_wave = np.max(wave)

    # read in model
    df_model = pd.read_pickle(dir + files)
    df_model = df_model.iloc[:,[0,1]]
    df_model.columns = ['wave', 'flux']

    # trim the excess part
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

    return f(wave)