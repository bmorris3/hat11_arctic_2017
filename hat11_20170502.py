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
comparison_flux_threshold = 0.01
aperture_radii = np.arange(45, 70, 1)
centroid_stamp_half_width = 30
psf_stddev_init = 30
aperture_annulus_radius = 10
transit_parameters = params_b


output_path = 'outputs/hat11_20170502.npz'
force_recompute_photometry = False#True

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
light_curve = PCA_light_curve(phot_results, transit_parameters, plots=False,
                              validation_duration_fraction=0.05,
                              buffer_time=0*u.min, flux_threshold=0.5,
                              validation_time=-0.55, plot_validation=False)

target_flux = phot_results.fluxes[:, 0, 0]
not_cloudy = target_flux > 0.1*np.median(target_flux)

# further de-trend with airmass:

X = np.array([light_curve[not_cloudy],
              phot_results.airmass[not_cloudy],
              phot_results.ycentroids[:, 0]]).T

c = np.linalg.lstsq(X, transit_model_b(phot_results.times[not_cloudy]))[0]

detrended_light_curve = np.dot(X, c)

plt.plot(phot_results.times[not_cloudy], detrended_light_curve, '.', color='gray')

from scipy.stats import binned_statistic

bs = binned_statistic(phot_results.times[not_cloudy], detrended_light_curve, bins=50)
bin_centers = 0.5*(bs.bin_edges[1:] + bs.bin_edges[:-1])
plt.plot(bin_centers, bs.statistic, 'rs')

plt.plot(phot_results.times, transit_model_b(phot_results.times), 'r')
#egress = 2457777.01
#post_egress_std = np.std(light_curve[phot_results.times > egress])
#plt.axvline(egress)
plt.xlabel('Time [JD]')
plt.ylabel('Flux')
plt.show()

output_lc = 'outputs/hat11_20170502.txt'
np.savetxt(output_lc, np.vstack([phot_results.times,
                                 detrended_light_curve,
                                 phot_results.fluxes[:, 0, 0]]).T)
#
# plt.figure()
# plt.plot(phot_results.times, light_curve, '.', color='gray')
# plt.plot(phot_results.times, transit_model_b(phot_results.times), 'r')
#
# from scipy.stats import binned_statistic
#
# bs = binned_statistic(phot_results.times, light_curve, bins=50)
# bin_centers = 0.5*(bs.bin_edges[1:] + bs.bin_edges[:-1])
# plt.plot(bin_centers, bs.statistic, 'rs')
#
# plt.plot(phot_results.times, transit_model_b(phot_results.times), 'r')
#
#
# #egress = 2457777.01
# #post_egress_std = np.std(light_curve[phot_results.times > egress])
# #plt.axvline(egress)
# plt.xlabel('Time [JD]')
# plt.ylabel('Flux')
# plt.title('rms = {0}'.format(np.std(light_curve - transit_model_b(phot_results.times))))
# plt.show()