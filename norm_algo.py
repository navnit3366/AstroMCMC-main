import numpy as np
from scipy.interpolate import UnivariateSpline

def norm_df(df, min_range, max_range):
    df = df.iloc[:,[0,1]]
    df.columns = ['wave', 'flux']

    df = df[(df['wave'] > min_range) & (df['wave'] < max_range)]
    df.reset_index(inplace=True, drop=True)

    # Define the step size and threshold
    step_size = 500 # ~~ 50 A
    threshold1 = 0.001
    z = 100

    # Iterate through the spectrum with the given step size
    for i in range(0, len(df), step_size):
        # Get the local minimum within the step
        local_min = np.min(df['flux'][i:i+step_size])
        local_max = np.max(df['flux'][i:i+step_size])

        # Find the indices of points above the threshold
        above_threshold = np.where(df['flux'][i:i+step_size] > local_max - threshold1)[0] # & (df['flux'][i:i+step_size] > flux_mean)

        # Mark the z number of points above the threshold
        for idx in above_threshold[:z]:
            df.loc[i+idx, 'marked'] = True


    marked_points = df[df['marked'] == True]
    
    sm = 0.001
    spl = UnivariateSpline(marked_points['wave'], marked_points['flux'])
    spl.set_smoothing_factor(sm)

    norm_f = spl(df['wave'])
    norm_model = df['flux'] - norm_f + 1
    df['flux'] = norm_model

    return df