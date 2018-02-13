"""
Create 4 panel plot with f(RH) differences for each of the aerosol species and the average f(RH)
Designed to deal with multiple bands

! WARNING - currently fixed to do 910, including save file names. Needs updating

Created by Elliott 06/04/17
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from f_RH_difference import create_f_RH, calc_f_RH, calc_r_RH_difference, calc_r_RH_ratio, \
                            plot_difference, plot_ratio, plot_absolute
from f_RH_creation import read_spec_bands, read_aer_data


def create_f_RH_multi_band(file_path, bands, aer_index, aer_order, Q_type):

    """
    Read in data and create the f_RH data for each aerosol, as well as an average.

    :param f_RH_file:
    :param band:
    :param aer_index:
    :param aer_order:
    :return:

    Designed to only do this for one file at a time, in order to split the main file from the others.
    """

    f_RH = {}
    band_order = []

    for band_i in bands:

        # read in the spectral band information
        spec_bands = read_spec_bands(file_path)

        # wavelength range in current band
        band_idx = np.where(spec_bands['band'] == band_i)[0][0]
        band_lam_range = '%.0f' % (spec_bands['lower_limit'][band_idx] * 1.0e9) + '-' + \
                         '%.0f' % (spec_bands['upper_limit'][band_idx] * 1.0e9) + 'nm'

        # This will be used for plotting in order, later on
        band_order += [band_lam_range]

        # read the aerosol data
        data = read_aer_data(file_path, aer_index, aer_order, band=band_i)

        # Extract RH for plotting (RH is the same across all aerosol types)
        # convert from [frac] to [%]
        RH = np.array(data[aer_order[0]][:, 0]) * 100.0

        # calculate f(RH)
        # define wavelength key using band_lam_range
        Q, f_RH[band_lam_range] = calc_f_RH(data, aer_order, Q_type=Q_type)

        # create an average f(RH)
        f_RH[band_lam_range]['average'] = np.mean(f_RH[band_lam_range].values(), axis=0)
        # f_RH['average'] = np.mean((f_RH['Accum. Sulphate'], f_RH['Aged fossil-fuel OC'], f_RH['Ammonium nitrate']), axis=0)

    return f_RH, band_order, RH


def create_colours(intervals):

    r = 0.2
    g = 0.5

    step = 1.0 / intervals

    b_colours = np.arange(0.0, 1.0 + step, step)
    g_colours = np.arange(1.0, 0.0 - step, -step)

    colours = [[r, g_colours[i], b_colours[i]] for i in range(len(b_colours))]

    return colours


def main():

    # User set args
    # band that read_spec_bands() uses to find the correct band
    #! Manually set
    bands = np.arange(1, 21)

    # -------------------------

    # directories
    savedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/figures/Mie/f(RH)/'
    specdir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/Mie/spectral/'
    f_RHdir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/Mie/'

    # file_name = 'spec3a_sw_hadgem1_7lean_so' # original file given to me by Claire Ryder 25/01/17
    # file_name = 'sp_sw_ga7' # current UM file
    # file_name = 'sp_ew_910' # my own made file with 1 band at 910 nm

    main_file = 'sp_ew_910'
    main_path = specdir + main_file
    main_file_band = 1

    band_file = 'sp_ew_ceil_multi_895-915'
    band_path = specdir + band_file

    # variables to take from file (as listed within the file) with index from BLOCK = 0
    # NOTE: data MUST be in ascending index order
    aer_index = {'Accum. Sulphate': 1, 'Aged fossil-fuel OC': 2, 'Ammonium nitrate': 3}
    aer_order = ['Accum. Sulphate', 'Aged fossil-fuel OC', 'Ammonium nitrate']

    # super rigid, needs replacing
    # lam_colours = {'1062-1066nm': 'r', '908-912nm': 'b', '903-907nm': 'g'}


    # Q type to use in calculating f(RH)
    Q_type = 'extinction'
    print 'Q_type = ' + Q_type

    # ---------------------------------------------------
    # Read and Process
    # ---------------------------------------------------

    # create f(RH) for the main file
    f_RH_main, main_band_range, RH = create_f_RH(main_path, main_file_band, aer_index, aer_order, Q_type)

    # create f(RH) for all bands
    f_RH, band_order, RH = create_f_RH_multi_band(band_path, bands, aer_index, aer_order, Q_type)

    # take differences [reference - main] -> +ve = reference is higher; -ve = reference is lower
    # keep separate to plotting, in order to improve code modularity
    f_RH_diff = calc_r_RH_difference(f_RH, f_RH_main, main_band_range)

    # calculate the ratio [reference / main] -> shows impact on extinction coefficient.
    # if ref = 2x high, then ext coeff will be 2x higher.
    f_RH_ratio = calc_r_RH_ratio(f_RH, f_RH_main, main_band_range)

    # ---------------------------------------------------
    # Plotting
    # ---------------------------------------------------

    # get colour gradation based on number of lamda ranges in band_order
    lam_colours = create_colours(len(band_order) - 1)

    # plot the absolute value of f(RH): lam_i
    fig = plot_absolute(f_RH_main, f_RH, RH, savedir, aer_order, band_order, lam_colours, Q_type)

    # plot the difference in f(RH): lam_i - lam_reference
    fig = plot_difference(f_RH_diff, RH, savedir, aer_order, band_order, lam_colours, Q_type)

    # plot the ratio in f(RH): lam_i / lam_reference
    fig = plot_ratio(f_RH_ratio, RH, savedir, aer_order, band_order, lam_colours, Q_type)

    print 'END PROGRAM'

if __name__ == '__main__':
    main()
