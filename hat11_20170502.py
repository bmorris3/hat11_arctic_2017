import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from toolkit import (generate_master_flat_and_dark, photometry,
                     PhotometryResults, PCA_light_curve, params_b,
                     transit_model_b)

# Image paths
image_paths = sorted(glob('/Users/bmmorris/data/Q2UW01/UT170502/cleaned/hat11*.fits'))
dark_paths = glob('/Users/bmmorris/data/Q2UW01/UT170502/dark_10s_2x2.????.fits')
flat_paths = glob('/Users/bmmorris/data/Q2UW01/UT170502/domeflat_r.????.fits')
master_flat_path = 'outputs/masterflat_20170502.fits'
master_dark_path = 'outputs/masterdark_20170502.fits'

# Photometry settings
target_centroid = np.array([[613], [750]])
comparison_flux_threshold = 0.001
aperture_radii = np.arange(50, 90, 2)
centroid_stamp_half_width = 30
psf_stddev_init = 30
aperture_annulus_radius = 10
transit_parameters = params_b


output_path = 'outputs/hat11_20170502.npz'
force_recompute_photometry = False #True

# Calculate master dark/flat:
if not os.path.exists(master_dark_path) or not os.path.exists(master_flat_path):
    print('Calculating master flat:')
    generate_master_flat_and_dark(flat_paths, dark_paths,
                                  master_flat_path, master_dark_path)

# Do photometry:

if not os.path.exists(output_path) or force_recompute_photometry:
    print('Calculating photometry:')
    phot_results = photometry(image_paths, master_dark_path, master_flat_path,
                              target_centroid, comparison_flux_threshold,
                              aperture_radii, centroid_stamp_half_width,
                              psf_stddev_init, aperture_annulus_radius,
                              output_path)

else:
    phot_results = PhotometryResults.load(output_path)

print('Calculating PCA...')
light_curve = PCA_light_curve(phot_results, transit_parameters, plots=True,
                              validation_duration_fraction=0.2,
                              buffer_time=5*u.min, flux_threshold=0.5,
                              validation_time=-0.8, plot_validation=False)

plt.figure()
plt.plot(phot_results.times, light_curve, 'k.')
plt.plot(phot_results.times, transit_model_b(phot_results.times), 'r')
#egress = 2457777.01
#post_egress_std = np.std(light_curve[phot_results.times > egress])
#plt.axvline(egress)
plt.xlabel('Time [JD]')
plt.ylabel('Flux')
plt.title('rms = {0}'.format(np.std(light_curve - transit_model_b(phot_results.times))))
plt.show()