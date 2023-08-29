#!/bin/bash

# Set the path to your Python script
python_script="mcmc.py"
# Path to the observed spectrum
obs_file_path="/Users/rushirajgadhvi/Desktop/mcmc/hd55575_SME_SPECTRUM.dat"   

# Parameter Names
params=("Teff" "Logg" "[M/H]") 
# Parameter Range
param_range="5100 6200, 4.5 5.5, -1.0 0.0"
# True Parameter Values
truth_val=(5650 5.0 -0.5)  

# Number of walkers
nwalkers=30
# Number of steps
nsteps=200

# Minimum wavelength (open end)
wave_min=3900
# Maximum wavelength (open end) 
wave_max=6700

# Use a model grid (0) or parameter interpolation (1)
is_grid=1

# 1. Self define initial parameters
# 2. Uniformly distributed in the parameter space
# 3. Use the MLE (takes time to find)
choice=2

# Spread around the initial parameters
spread=(50 0.2 0.2)

# Initial parameters (only needed if choice 1 is selected)  
initial_params=(param1_initial param2_initial param3_initial)  

# consider pre-mentioned telluric regions
use_telluric=0

# Run the Python script with the specified arguments
python3 "$python_script" \
  --obs_file_path "$obs_file_path" \
  --params "${params[@]}" \
  --param_range "${param_range[@]}" \
  --truth_val "${truth_val[@]}" \
  --nwalkers "$nwalkers" \
  --nsteps "$nsteps" \
  --wave_min "$wave_min" \
  --wave_max "$wave_max" \
  --is_grid "$is_grid" \
  --choice "$choice" \
  --spread "${spread[@]}" \
  --use_telluric "$use_telluric" \
  # --initial_params "param1_initial" "param2_initial" "param3_initial"