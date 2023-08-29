from predict import param_interpol
from predict_grid import param_grid
import numpy as np

def ln_prior(parameters, param_range):
    
    for i, param in enumerate(parameters):
        if param_range[i][0] <= param <= param_range[i][1]:
            continue
        else:
            return -np.inf
        
    return 0.0


def ln_likelihood(parameters, x, y, yerr, is_grid, use_telluric):

    # Compute the model predictions
    if is_grid == 1:
        model_flux = param_interpol(*parameters, x, use_telluric= use_telluric)
    elif is_grid == 0:
        model_flux = param_grid(*parameters, x, use_telluric= use_telluric)

    sigma2 = yerr**2 + model_flux**2
    ln_likelihood = -0.5 * np.sum((y - model_flux) ** 2 / sigma2)
    
    return ln_likelihood


def ln_posterior(parameters, x, y, yerr, param_range, is_grid, use_telluric):
    # Calculate the log prior
    ln_prior_val = ln_prior(parameters, param_range)
    
    # If the parameters are outside the prior range, return -inf
    if not np.isfinite(ln_prior_val):
        return -np.inf
    
    # Calculate the log likelihood
    ln_likelihood_val = ln_likelihood(parameters, x, y, yerr, is_grid, use_telluric)
    
    # Calculate the log posterior
    ln_posterior = ln_prior_val + ln_likelihood_val
    
    return ln_posterior