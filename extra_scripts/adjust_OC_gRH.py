"""
Code to adjust the g(RH) of OC down so that its scattering enhancement factor parameters match that from
Kotchenruther and Hobbs (1998) in the model 2 equation (The paper Claire sent).

Created by Elliott Tues 16 Oct 2018

Iterative method to adjust g(RH) of OC
"""


import matplotlib.pyplot as plt

import numpy as np
import datetime as dt
from dateutil import tz

import ellUtils as eu
from mie_sens_mult_aerosol import linear_interpolate_n
from pymiecoated import Mie
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def read_organic_carbon_growth_factors(ffoc_gfdir, OCtype='freshOCGF'):

    """
    Read in the organic carbon growth factors
    Make sure OCtype has GF at the end so the numpy save at the end will make it clear what makes this .npy unique
    :param ffoc_gfdir:
    :return: gf_ffoc [dictionary: GF = growth factor and RH_frac = RH fraction]:
    """
    if OCtype == 'agedOCGF':
        gf_ffoc_raw = eu.csv_read(ffoc_gfdir + 'GF_fossilFuelOC_calcS.csv')
    elif OCtype == 'freshOCGF':
        gf_ffoc_raw = eu.csv_read(ffoc_gfdir + 'GF_freshFossilFuelOC.csv')
    else:
        raise ValueError('Organic carbon type not defined as aged or fresh. No other options present')

    gf_ffoc_raw = np.array(gf_ffoc_raw)[1:, :] # skip header
    gf_ffoc = {'RH_frac': np.array(gf_ffoc_raw[:, 0], dtype=float),
                    'GF': np.array(gf_ffoc_raw[:, 1], dtype=float)}

    return gf_ffoc

def read_n_data(aer_particles, aer_names, ceil_lambda, getH2O=True):


    n_species = {}
    # Read in complex index of refraction data
    for aer_i in aer_particles:

        # get the name of the aerosol as is appears in the function
        aer_i_name = aer_names[aer_i]

        # get the complex index of refraction for the n (linearly interpolated from a lookup table)
        n_species[aer_i], _ = linear_interpolate_n(aer_i_name, ceil_lambda)

    # get water too?
    if getH2O == True:
        n_species['H2O'], _ = linear_interpolate_n('water', ceil_lambda)

    return n_species

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # # site information
    # site_meta = {'site_short':'Ch', 'site_long': 'Chilbolton', 'period': '2016',
    #         'instruments': ['SMPS', 'GRIMM'], 'ceil_lambda': 0.905e-06}

    # NK: 2014 - 2016 inclusively
    site_meta = {'site_short':'NK', 'site_long': 'North_Kensington', 'period': 'long_term',
    'instruments': ['SMPS', 'APS']}

    site_meta['ceil_lambda'] = 0.905e-06 # 0.355e-06 # 0.905e-06 # 0.355e-06  # 1.064e-06, 0.532e-06

    ceil_lambda = [site_meta['ceil_lambda']]
    period = site_meta['period']


    # wavelength to aim for (in a list! e.g. [905e-06])
    ceil_lambda_str = str(int(site_meta['ceil_lambda'] * 1e9)) + 'nm'

    # string for saving figures and choosing subdirectories
    savesub = 'PM10_withSoot'

    # directories
    maindir = '/home/nerc/Documents/MieScatt/'
    datadir = '/home/nerc/Documents/MieScatt/data/' + site_meta['site_long'] + '/'
    pickledir = '/home/nerc/Documents/MieScatt/data/pickle/' + site_meta['site_long'] + '/'

    # save dir
    savesubdir = savesub

    # full save directory (including sub directories)
    savedir = maindir + 'figures/LidarRatio/' + savesubdir + '/'

    # data
    massdatadir = datadir
    ffoc_gfdir = '/home/nerc/Documents/MieScatt/data/'


    # resolution to average data to (in minutes! e.g. 60)
    timeRes = 60

    # aerosol particles to calculate (OC = Organic carbon, CBLK = black carbon, both already measured)
    # match dictionary keys further down
    aer_particles = ['CORG']

    # aer names in the complex index of refraction files
    aer_names = {'CORG': 'Organic carbon'}

    aer_colours = {'(NH4)2SO4': 'red', 'NH4NO3': 'orange',
                   'CORG': [0.05, 0.9, 0.4], 'NaCl': 'magenta', 'CBLK':'brown'}

    # raw data used to make aerosols
    orig_particles = ['CORG', 'CL', 'CBLK', 'NH4', 'SO4', 'NO3']

    # density of molecules [kg m-3]
    # CBLK: # Zhang et al., (2016) Measuring the morphology and density of internally mixed black carbon
    #           with SP2 and VTDMA: New insight into the absorption enhancement of black carbon in the atmosphere
    # CORG: Range of densities for organic carbon is mass (0.625 - 2 g cm-3)
    #  Haywood et al 2003 used 1.35 g cm-3 but Schkolink et al., 2007 claim the average is 1.1 g cm-3 after a lit review
    aer_density = {'(NH4)2SO4': 1770.0,
                   'NH4NO3': 1720.0,
                   'NaCl': 2160.0,
                   'CORG': 1100.0,
                   'CBLK': 1200.0}


    # pure water density
    water_density = 1000.0 # kg m-3

    # ==============================================================================
    # Process
    # ==============================================================================

    n_species = read_n_data(aer_particles, aer_names, ceil_lambda, getH2O=True)

    # temporarily set OC absorption to 0
    # n_species['CBLK'] = complex(n_species['CBLK'].real, 0.44)
    # n_species['CORG'] = complex(n_species['CORG'].real, 0.01)

    # Read in physical growth factors (GF) for organic carbon (assumed to be the same as aged fossil fuel OC)
    OC_meta = {'type': 'agedOCGF', 'extra': ''}
    gf_ffoc = read_organic_carbon_growth_factors(ffoc_gfdir, OCtype=OC_meta['type'])

    lambda_m = 905.0e-9
    r_d_m = 0.18e-6 # half way into dry accumulation range (radii 40 - 400 nm)
    rh = np.arange(0.0,101.0, 1.0)






























