import numpy as np
import pandas as pd
import os
from norm_algo import norm_df
from scipy.ndimage import gaussian_filter1d

# ****************** PARAMETERS
sigma = 1.34
min_range = 3000
max_range = 7000

# ****************** DIRECTORIES
dir = '/Volumes/RUSHIRAJG/BT-Settl/' # directory containing the models
files = os.listdir(dir)

dir2 = os.getcwd() + '/norm_models/' # directory to save the normalized models (default in the current directory)

# *****************************

for file in files:

    if file.startswith('.'):
        continue

    flux = []
    wave = []

    f=open(dir + file,"r")
    lines=f.readlines()

    for line in lines:
        wave.append(line.split()[0])
        flux.append(line.split()[1])
    f.close()

    df_model = pd.DataFrame()
    df_model['wave'] = wave
    df_model['flux'] = flux

    # df_model = pd.read_table(dir + file, delim_whitespace=True, header=None)
    # df_model = df_model.iloc[:,[0,1]]
    # df_model.columns = ['wave', 'flux']

    # replace D in flux column with E
    df_model['flux'] = df_model['flux'].astype(str).str.replace('D', 'E').astype(float)
    df_model['wave'] = df_model['wave'].astype(str).str.replace('D', 'E').astype(float)

    # remove duplicate wavelengths
    df_model = df_model.drop_duplicates(subset='wave', keep='first')

    # sort the model spectrum
    arg = np.argsort(df_model['wave'])
    df_model = df_model.iloc[arg]

    # normalization
    df_model = norm_df(df_model, min_range, max_range) 

    # guassian smoothing
    df_model['flux'] = gaussian_filter1d(df_model['flux'], sigma)

    # save the pickle file
    df_model.to_pickle( dir2 + file.split('+')[0] + '.pkl')

    print(file.split('+')[0] + '.pkl saved')