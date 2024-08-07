import numpy as np
import astropy.units as u
import eispac
from eis_average.function import average_spectral_data
from eis_calibration.eis_calib_2014 import calib_2014
from ashmcmc import ashmcmc, interp_emis_temp
from demcmc import EmissionLine, TempBins, ContFuncDiscrete
from mcmc.mcmc_utils import calc_chi2, mcmc_process
from mcmc_para import pred_intensity_compact
import glob
from asheis import asheis
import pdb  # Add this import at the top of your file

"""
This script processes EIS data to perform spectral analysis and 
differential emission measure (DEM) calculations. It includes the following main steps:

1. Process individual emission lines:
    - Reads EIS data and fit templates
    - Averages spectral data within a specified region
    - Fits the spectra and applies calibration

2. Calculate electron density using Fe XIII line pair

3. Prepare data for MCMC (Markov Chain Monte Carlo) analysis:
    - Reads emissivity data
    - Creates emission line objects for MCMC

4. Perform MCMC process to derive DEM

5. Calculate FIP (First Ionization Potential) ratios

The script utilizes various modules including eispac for EIS data handling, 
ashmcmc for MCMC calculations, and custom modules for specific functionalities.

Usage:
This script is designed to be run as part of a larger analysis pipeline. 
It requires pre-defined input parameters such as EIS event file path and 
coordinates for the region of interest.

Note: Some parts of the code (e.g., FIP ratio calculation) are prepared but 
not fully implemented in the main execution flow.
"""

def process_line(line, dict, eis_evt, r1_bl_crd, r1_tr_crd):
    template_name=dict[f'{line}'][0]
    if template_name != 'fe_13_203_826.2c.template.h5':
        template = eispac.read_template(eispac.data.get_fit_template_filepath(template_name))
    else:
        template = eispac.read_template('eis_density/fe_13_203_830.3c.template.h5')
        template_name = 'fe_13_203_830.3c.template.h5'


    data_cube = eispac.read_cube(eis_evt, template.central_wave)
    data_averaged, data_cropped = average_spectral_data(
        data_cube, template, 
        [r1_bl_crd[0], r1_bl_crd[1]], 
        [r1_tr_crd[0], r1_tr_crd[1]]
    )
    # pdb.set_trace()

    fit_res = eispac.fit_spectra(data_averaged, template, ncpu='max')
    fitted_result = fit_res.get_map(component=dict[f'{line}'][1], measurement='intensity')
    fitted_result, ratio = calib_2014(fitted_result, ratio=True)
    # Print the calibration ratio prettily
    print(f'---------------------Calibrated using Warren et al. 2014; Ratio: {ratio:.2f}---------------------')
    return fitted_result.data[0][0]


def eis_averaged_fip(eis_evt, r1_bl_crd, r1_tr_crd):
    # Initiate asheis object
    eis_cube = asheis(eis_evt)
    ash = ashmcmc(eis_evt)
    Lines, dim, dem_num = ash.check_existing_lines()
    dict = eis_cube.dict 

    # Process emission lines
    Intensity = np.array([process_line(line, dict, eis_evt, r1_bl_crd, r1_tr_crd) for line in Lines])
    # pdb.set_trace()

    # Process density lines
    density_lines = ['fe_13_203.83', 'fe_13_202.04']
    density_intensity = np.array([process_line(line, dict, eis_evt, r1_bl_crd, r1_tr_crd) for line in density_lines])
    dens_ratio = density_intensity[0] / density_intensity[1]
    density_ratios, density_values = eis_cube.read_density_file('fe_13_203.83')
    ldens = eis_cube.find_closest_density(dens_ratio, density_ratios, density_values)

    # Prepare MCMC
    logt, emis, linenames = ash.read_emissivity(ldens)
    logt_interp = interp_emis_temp(logt.value)
    
    # Limit the logt range to 5.2 < logt < 7.2
    mask = (np.log10(logt_interp) > 5.2) & (np.log10(logt_interp) < 7.2)
    logt_interp = logt_interp[mask]
    temp_bins = TempBins(logt_interp * u.K)
    
    emis_sorted = ash.emis_filter(emis[:, mask], linenames, Lines)
    # Create MCMC lines
    mcmc_lines = []
    for ind, line in enumerate(Lines):
        if line.startswith('fe') and Intensity[ind] > 10:
            mcmc_emis = ContFuncDiscrete(
                logt_interp * u.K,
                interp_emis_temp(emis_sorted[ind, :]) * u.cm ** 5 / u.K,
                name=line
            )
            mcmc_intensity = Intensity[ind]
            mcmc_int_error = Intensity[ind] * 0.3
            emissionLine = EmissionLine(
                mcmc_emis,
                intensity_obs=mcmc_intensity,
                sigma_intensity_obs=mcmc_int_error,
                name=line
            )
            mcmc_lines.append(emissionLine)

    # Process MCMC
    # Print a pretty message for DEM calculation using MCMC
    print("Calculating DEM using MCMC".center(50))
    dem_median = mcmc_process(mcmc_lines, temp_bins)
    chi2 = calc_chi2(mcmc_lines, dem_median, temp_bins)

    print(mcmc_intensity)
    # Process FIP ratios
    line_databases = {
        "sis": ['si_10_258.37', 's_10_264.23', 'SiX_SX'],
        # "CaAr": ['ca_14_193.87', 'ar_14_194.40', 'CaXIV_ArXIV'],
    }

    for comp_ratio, lines in line_databases.items():
        intensities = np.array([process_line(line, dict, eis_evt, r1_bl_crd, r1_tr_crd) for line in lines[:2]])
        
        logt, emis, linenames = ash.read_emissivity(ldens)
        logt_interp = interp_emis_temp(logt.value)
        emis_sorted = ash.emis_filter(emis, linenames, lines[:2])
        
        int_lf = pred_intensity_compact(emis_sorted[0], logt_interp, lines[0], dem_median)
        dem_scaled = dem_median * (intensities[0] / int_lf)
        int_hf = pred_intensity_compact(emis_sorted[1], logt_interp, lines[1], dem_scaled)
        fip_ratio = int_hf / intensities[1]

        print(f"FIP ratio for {comp_ratio}: {fip_ratio}")
        print(f"Number of lines used for DEM: {len(mcmc_lines)}")
        print(f"Chi-squared value: {chi2}")

    return fip_ratio, mcmc_lines, chi2

if __name__ == "__main__":
    from useful_packages.align_aia_EIS import alignment

    eis_evts = sorted(glob.glob('/Users/andysh.to/Script/Python_Script/demcmc_FIP_averaged/data_eis/*.data.h5'))
    eis_evt = eis_evts[0]
    r1_bl_crd = [216.88862131045445, -210.08356892192884]
    r1_tr_crd = [220.8914582334983, -206.08513765573548]
    eis_map, Txshift, Tyshift = alignment(eis_evt, return_shift=True)

    # Subtract the shift to get coordinates in the original, unshifted map
    new_r1_bl_crd = [r1_bl_crd[0] - Txshift, r1_bl_crd[1] - Tyshift]
    new_r1_tr_crd = [r1_tr_crd[0] - Txshift, r1_tr_crd[1] - Tyshift]

    fip_ratio, mcmc_lines, chi2 = eis_averaged_fip(eis_evt, new_r1_bl_crd, new_r1_tr_crd)
