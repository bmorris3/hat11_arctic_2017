from glob import glob
from astroscrappy import detect_cosmics
from astropy.io import fits
import os
from astropy.utils.console import ProgressBar
import numpy as np

# image_paths = sorted(glob('/Users/bmmorris/data/Q1UW01/UT170123/wasp85.????.fits'))[360:]
# outpath = '/Users/bmmorris/data/Q1UW01/UT170123/cleaned/'

#image_paths = sorted(glob('/Users/bmmorris/data/Q2UW01/UT170427/HATP11.????.fits'))
#outpath = '/Users/bmmorris/data/Q2UW01/UT170427/cleaned/'
#  image_paths = sorted(glob('/Users/bmmorris/data/Q2UW01/UT170615/HAT-P-11.????.fits'))
# outpath = '/Users/bmmorris/data/Q2UW01/UT170615/cleaned/'

image_paths = sorted(glob('/Users/bmmorris/data/Q3UW01/UT170730/HAT-P-11.????.fits'))
outpath = '/Users/bmmorris/data/Q3UW01/UT170730/cleaned/'

# Mask out hot pixel column from cosmic ray detection for 4x4 binning
first_img = fits.getdata(image_paths[0])
mask = np.zeros_like(first_img)
mask[0:515, 597:599] += 1
bool_mask = mask.astype(bool)

with ProgressBar(len(image_paths)) as bar:
    for path in image_paths:
        bar.update()
        f = fits.open(path)
        file_name = path.split(os.sep)[-1]
        mask, cleaned_image = detect_cosmics(f[0].data, sigfrac=2,
                                             inmask=bool_mask)
        f[0].data = cleaned_image
        fits.writeto(outpath + file_name, cleaned_image, header=f[0].header,
                     clobber=True)
