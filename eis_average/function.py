import numpy as np
from astropy import units as u
from sunpy.coordinates import frames
from ndcube import NDCube
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SpectralCoord
from ndcube import NDCube
from eispac.core.eiscube import EISCube


def average_spectral_data(data_cube, tmplt, lower_left, upper_right, shift2fexii = True, ref_wave=195.12):

    from scipy.ndimage import shift as shift_img
    from eispac.instr.ccd_offset import ccd_offset
    from astropy.wcs.utils import wcs_to_celestial_frame

    """Average spectral data within a specified rectangular region.

    Parameters:
    - data_cube (ndcube.NDCube): The input data cube containing spectral data.
    - template (dict): A dictionary representing the template with 'data_x' key containing wavelength values.
    - lower_left (list): The lower-left corner coordinates of the rectangular region [x, y].
    - upper_right (list): The upper-right corner coordinates of the rectangular region [x, y].

    Returns:
    - data_cutout_sum (ndcube.NDCube): The averaged spectral data within the specified region,
      rebinned into a single macropixel.
    """

    eis_frame = wcs_to_celestial_frame(data_cube.wcs)
    
    wavelength_average = np.nanmean(data_cube.wavelength, axis=(0, 1))

    if shift2fexii:
        this_wave = np.mean(wavelength_average)
        disp = ccd_offset(ref_wave) - ccd_offset(this_wave)
        print(f'SHIFT2WAVE: shifted to {ref_wave} FOV according to CCD offset - OFFSET: {disp[0]:.1f}')
    else:
        disp = 0


    lower_left = [SpectralCoord(min(wavelength_average), unit=u.angstrom),
                  SkyCoord(Tx=lower_left[0], Ty=int(lower_left[1]-disp[0]), unit=u.arcsec, frame=eis_frame)]
    upper_right = [SpectralCoord(max(wavelength_average), unit=u.angstrom),
                   SkyCoord(Tx=upper_right[0], Ty=int(upper_right[1]-disp[0]), unit=u.arcsec, frame=eis_frame)]
    

    # Crop ndcube data to the desired subpixels
    data_cutout = data_cube.crop(lower_left, upper_right)
    
    # Calculate the mean wavelength
    wavelength_average = np.nanmean(data_cutout.wavelength, axis=(0, 1))

    # Calculate the averaged error
    sum_squared_errors = np.nansum(data_cutout.uncertainty.array**2, axis=(0, 1))
    
    # Calculate the averaged error by taking the square root of the sum of squared errors divided by the number of elements
    num_elements = data_cutout.data.shape[0] * data_cutout.data.shape[1]
    averaged_error = np.sqrt(sum_squared_errors / num_elements)

    # Rebin data into 1 single pixel - default is np.mean
    hack_rebinned = NDCube(data_cutout.data, wcs=data_cutout.wcs).rebin(np.array([data_cutout.data[:,0,0].shape[0],data_cutout.data[0,:,0].shape[0],1]))
    data_cutout_sum = EISCube(hack_rebinned, wcs = hack_rebinned.wcs, 
                              uncertainty = [[averaged_error]],
                              wavelength = np.array([[wavelength_average]]),
                              unit = 'erg / (cm2 s sr)',
                              meta = data_cutout.meta)
    return data_cutout_sum, data_cutout


    # data_cutout_sum = data_cutout.rebin(np.array([data_cutout.data[:,0,0].shape[0],data_cutout.data[0,:,0].shape[0],1]))
    # # data_cutout_sum = EISCube(hack_cube.rebin(np.array([data_cutout.data[:,0,0].shape[0],data_cutout.data[0,:,0].shape[0],1])), unit = data_cutout.unit, meta = data_cutout.meta)

    # # # fit everything into the final 1 macropixel datacube
    # data_cutout_sum.uncertainty = [[averaged_error]]
    # data_cutout_sum.wavelength = np.array([[wavelength_average]])

    # return data_cutout_sum, data_cutout