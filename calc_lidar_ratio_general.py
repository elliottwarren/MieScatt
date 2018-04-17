"""
Read in mass, number distribution, RH, T and pressure data from a site (e.g. Chilbolton) to calculate the Lidar Ratio (S) [sr]. Uses the interpolation
method from Geisinger et al., 2018 to increase the number of diameter bins and reduce the issue of high sensitivity of S to
diameter (the discussion version of Geisinger et al., 2018 is clearer and more elaborate than the final
published version!). Can swell and dry particles from the number distribution (follows the CLASSIC aerosol scheme)!


Created by Elliott Tues 23 Jan
Taken from calc_lidar_ratio_numdist.py (designed for a constant number dist, from clearFlo, for NK

Variables and their units are paird together in comments with the units in square brackets i.e.
variable [units]
"""

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pickle
from copy import deepcopy
import os

import numpy as np
import datetime as dt
from dateutil import tz
from scipy.stats import spearmanr
from scipy.stats import pearsonr

import ellUtils as eu
from mie_sens_mult_aerosol import linear_interpolate_n
from pymiecoated import Mie


# Set up

def fixed_radii_for_Nweights():

    """
    Fixed radii used for each aerosol species in calculating the number weights (Nweights).

    :return: rn_pmlt1p0_microns, rn_pmlt1p0_m, \
        rn_pm10_microns, rn_pm10_m, \
        rn_pmlt2p5_microns, rn_pmlt2p5_m, \
        rn_2p5_10_microns, rn_2p5_10_m

    Aerosol are those in aer_particles and include: '(NH4)2SO4', 'NH4NO3', 'NaCl', 'CORG', 'CBLK'
    """

    # 1. D < 1.0 micron
    # CLASSIC dry radii [microns] - Bellouin et al 2011
    rn_pmlt1p0_microns = {'(NH4)2SO4': 9.5e-02, # accumulation mode
                  'NH4NO3': 9.5e-02, # accumulation mode
                  'NaCl': 1.0e-01, # generic sea salt (fine mode)
                  'CORG': 1.2e-01, # aged fosil fuel organic carbon
                  'CBLK': 3.0e-02} # soot

    rn_pmlt1p0_m={}
    for key, r in rn_pmlt1p0_microns.iteritems():
        rn_pmlt1p0_m[key] = r * 1e-06

    # 2. D < 10 micron

    # pm1 to pm10 median volume mean radius calculated from clearflo winter data (calculated volume mean diameter / 2.0)
    rn_pm10_microns = 0.07478 / 2.0
    # turn units to meters and place an entry for each aerosol
    rn_pm10_m = {}
    for key in rn_pmlt1p0_m.iterkeys():
        rn_pm10_m[key] = rn_pm10_microns * 1.0e-6

    # # old 2. D < 10 micron
    # # pm1 to pm10 median volume mean radius calculated from clearflo winter data (calculated volume mean diameter / 2.0)
    # pm1t10_rv_microns = 1.9848902137534531 / 2.0
    # # turn units to meters and place an entry for each aerosol
    # pm1t10_rv_m = {}
    # for key in rn_pmlt1p0_m.iterkeys():
    #     pm1t10_rv_m[key] = pm1t10_rv_microns * 1.0e-6


    # 3. D < 2.5 microns
    # calculated from Chilbolton data (SMPS + GRIMM 2016)
    rn_pmlt2p5_microns = 0.06752 / 2.0

    rn_pmlt2p5_m = {}
    for key in rn_pmlt1p0_m.iterkeys():
        rn_pmlt2p5_m[key] = rn_pmlt2p5_microns * 1.0e-6

    # 4. 2.5 < D < 10 microns
    # calculated from Chilbolton data (SMPS + GRIMM 2016)
    rn_2p5_10_microns = 2.820 / 2.0

    rn_2p5_10_m = {}
    for key in rn_pmlt1p0_m.iterkeys():
        rn_2p5_10_m[key] = rn_2p5_10_microns * 1.0e-6


    return \
        rn_pmlt1p0_microns, rn_pmlt1p0_m, \
        rn_pm10_microns, rn_pm10_m, \
        rn_pmlt2p5_microns, rn_pmlt2p5_m, \
        rn_2p5_10_microns, rn_2p5_10_m

# Read

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

def read_organic_carbon_growth_factors(ffoc_gfdir):

    """
    Read in the organic carbon growth factors
    :param ffoc_gfdir:
    :return: gf_ffoc [dictionary: GF = growth factor and RH_frac = RH fraction]:
    """

    gf_ffoc_raw = eu.csv_read(ffoc_gfdir + 'GF_fossilFuelOC_calcS.csv')
    gf_ffoc_raw = np.array(gf_ffoc_raw)[1:, :] # skip header
    gf_ffoc = {'RH_frac': np.array(gf_ffoc_raw[:, 0], dtype=float),
                'GF': np.array(gf_ffoc_raw[:, 1], dtype=float)}

    return gf_ffoc

def read_PM1_mass_data(massdatadir, year):

    """
    Read in the mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :return: mass
    :return qaqc_idx_unique: unique index list where any of the main species observations are missing
    """

    massfname = 'PM_North_Kensington_'+year+'.csv'
    massfilepath = massdatadir + massfname
    massrawData = np.genfromtxt(massfilepath, delimiter=',', dtype="|S20") # includes the header

    mass = {'time': np.array([dt.datetime.strptime(i[0], '%d/%m/%Y %H:%M') for i in massrawData[1:]])}

    # get headers without the site part of it (S04@NK to S04)
    headers = [i.split('@')[0] for i in massrawData[0][1:]]

    # ignore first entry, as that is the date&time
    for h, header_site in enumerate(massrawData[0][1:]):

        # get the main part of the header from the
        split = header_site.split('@')
        header = split[0]

        if header == 'CL': # (what will be salt)
            # turn '' into 0.0, as missing values can be when there simply wasn't any salt recorded,
            # convert from micrograms to grams
            mass[header] = np.array([0.0 if i[h+1] == '' else i[h+1] for i in massrawData[1:]], dtype=float) * 1e-06

        else: # if not CL
            # turn '' into nans
            # convert from micrograms to grams
            mass[header] = np.array([np.nan if i[h+1] == '' else i[h+1] for i in massrawData[1:]], dtype=float) * 1e-06


    # QAQC - turn all negative values in each column into nans if one of them is negative
    qaqc_idx = {}
    for header_i in headers:
        bools = np.logical_or(mass[header_i] < 0.0, np.isnan(mass[header_i]))

        # store bool if it is one of the major pm consituents, so OM10 and OC/BC pm10 data can be removed too
        if header_i in ['NH4', 'NO3', 'SO4', 'CORG', 'CL', 'CBLK']:
            qaqc_idx[header_i] = np.where(bools == True)[0]


        # turn all values in the row negative
        for header_j in headers:
            mass[header_j][bools] = np.nan

    # find unique instances of missing data
    qaqc_idx_unique = np.unique(np.hstack(qaqc_idx.values()))


    return mass, qaqc_idx_unique

def read_PM_mass_data(massdatadir, site_meta, pmtype, year):

    """
    Read in PM2.5 mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :param pmtype: what type of pm to read in, that is in the filename (e.g. pm10, pm2p5)
    :return: mass
    :return qaqc_idx_unique: unique index list where any of the main species observations are missing
    """

    massfname = pmtype+'species_Hr_'+site_meta['site_long']+'_DEFRA_'+year+'.csv'
    massfilepath = massdatadir + massfname
    massrawData = np.genfromtxt(massfilepath, delimiter=',', dtype="|S20") # includes the header

    # extract and process time, converting from time ending GMT to time ending UTC

    from_zone = tz.gettz('GMT')
    to_zone = tz.gettz('UTC')

    # replace 24:00:00 with 00:00:00, then add 1 onto the day to compensate
    #   datetime can't handle the hours = 24 (doesn't round the day up).
    rawtime = [i[0] + ' ' + i[1].replace('24:00:00', '00:00:00') for i in massrawData[5:]]
    pro_time = np.array([dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') for i in rawtime])
    idx = [True if i.hour == 0 else False for i in pro_time]
    pro_time[idx] = pro_time[idx] + dt.timedelta(days=1)

    # convert from 'tme end GMT' to 'time end UTC'
    pro_time = [i.replace(tzinfo=from_zone) for i in pro_time] # set datetime's original timezone as GMT
    pro_time = np.array([i.astimezone(to_zone) for i in pro_time]) # convert from GMT to UTC

    mass = {'time': pro_time}

    # get headers without the site part of it (S04-PM2.5 to S04) and remove any trailing spaces
    headers = [i.split('-')[0] for i in massrawData[4]]
    headers = [i.replace(' ', '') for i in headers]

    # ignore first entry, as that is the date&time
    for h, header_site in enumerate(headers):

        # # get the main part of the header from the
        # split = header_site.split('-')
        # header = split[0]

        if header_site == 'CL': # (what will be salt)
            # turn '' into 0.0, as missing values can be when there simply wasn't any salt recorded,
            # convert from micrograms to grams
            mass[header_site] = np.array([0.0 if i[h] == 'No data' else i[h] for i in massrawData[5:]], dtype=float) * 1e-06

        elif header_site in ['NH4', 'SO4', 'NO3', 'Na']: # if not CL but one of the main gases needed for processing
            # turn '' into nans
            # convert from micrograms to grams
            mass[header_site] = np.array([np.nan if i[h] == 'No data' else i[h] for i in massrawData[5:]], dtype=float) * 1e-06


    # QAQC - turn all negative values in each column into nans if one of them is negative
    qaqc_idx = {}
    for header_i in headers:

        # store bool if it is one of the major pm consituents, so OM10 and OC/BC pm10 data can be removed too
        if header_i in ['NH4', 'NO3', 'SO4', 'CORG', 'Na', 'CL', 'CBLK']:

            bools = np.logical_or(mass[header_i] < 0.0, np.isnan(mass[header_i]))

            qaqc_idx[header_i] = np.where(bools == True)[0]


            # turn all values in the row negative
            for header_j in headers:
                if header_j not in ['Date', 'Time', 'Status']:
                    mass[header_j][bools] = np.nan

    # find unique instances of missing data
    qaqc_idx_unique = np.unique(np.hstack(qaqc_idx.values()))


    return mass, qaqc_idx_unique

def read_PM_mass_data_multiple_years(massdatadir, site_meta, pmtype, years):

    """
    Read in PM2.5 mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :param pmtype: what type of pm to read in, that is in the filename (e.g. pm10, pm2p5)
    :return: mass
    :return qaqc_idx_unique: unique index list where any of the main species observations are missing
    """

    massfname = pmtype+'species_Hr_'+site_meta['site_long']+'_DEFRA_'+year+'.csv'
    massfilepath = massdatadir + massfname
    massrawData = np.genfromtxt(massfilepath, delimiter=',', dtype="|S20") # includes the header

    # extract and process time, converting from time ending GMT to time ending UTC

    from_zone = tz.gettz('GMT')
    to_zone = tz.gettz('UTC')

    # replace 24:00:00 with 00:00:00, then add 1 onto the day to compensate
    #   datetime can't handle the hours = 24 (doesn't round the day up).
    rawtime = [i[0] + ' ' + i[1].replace('24:00:00', '00:00:00') for i in massrawData[5:]]
    pro_time = np.array([dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') for i in rawtime])
    idx = [True if i.hour == 0 else False for i in pro_time]
    pro_time[idx] = pro_time[idx] + dt.timedelta(days=1)

    # convert from 'tme end GMT' to 'time end UTC'
    pro_time = [i.replace(tzinfo=from_zone) for i in pro_time] # set datetime's original timezone as GMT
    pro_time = np.array([i.astimezone(to_zone) for i in pro_time]) # convert from GMT to UTC

    mass = {'time': pro_time}

    # get headers without the site part of it (S04-PM2.5 to S04) and remove any trailing spaces
    headers = [i.split('-')[0] for i in massrawData[4]]
    headers = [i.replace(' ', '') for i in headers]

    # ignore first entry, as that is the date&time
    for h, header_site in enumerate(headers):

        # # get the main part of the header from the
        # split = header_site.split('-')
        # header = split[0]

        if header_site == 'CL': # (what will be salt)
            # turn '' into 0.0, as missing values can be when there simply wasn't any salt recorded,
            # convert from micrograms to grams
            mass[header_site] = np.array([0.0 if i[h] == 'No data' else i[h] for i in massrawData[5:]], dtype=float) * 1e-06

        elif header_site in ['NH4', 'SO4', 'NO3', 'Na']: # if not CL but one of the main gases needed for processing
            # turn '' into nans
            # convert from micrograms to grams
            mass[header_site] = np.array([np.nan if i[h] == 'No data' else i[h] for i in massrawData[5:]], dtype=float) * 1e-06


    # QAQC - turn all negative values in each column into nans if one of them is negative
    qaqc_idx = {}
    for header_i in headers:

        # store bool if it is one of the major pm consituents, so OM10 and OC/BC pm10 data can be removed too
        if header_i in ['NH4', 'NO3', 'SO4', 'CORG', 'Na', 'CL', 'CBLK']:

            bools = np.logical_or(mass[header_i] < 0.0, np.isnan(mass[header_i]))

            qaqc_idx[header_i] = np.where(bools == True)[0]


            # turn all values in the row negative
            for header_j in headers:
                if header_j not in ['Date', 'Time', 'Status']:
                    mass[header_j][bools] = np.nan

    # find unique instances of missing data
    qaqc_idx_unique = np.unique(np.hstack(qaqc_idx.values()))


    return mass, qaqc_idx_unique

def read_EC_BC_mass_data(massdatadir, site_meta, pmtype, year):

    """
    Read in the elemental carbon (EC) and organic carbon (OC) mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :param pmtype: which PM to get the data for (must match that used in the filename) e.g. PM10, PM2p5
    :return: mass

    EC and BC (soot) are treated the same in CLASSIC, therefore EC will be taken as BC here.
    """

    # make sure year is string
    year = str(year)

    massfname = pmtype+'_OC_EC_Daily_'+site_meta['site_long']+'_DEFRA_'+year+'.csv'
    massfilepath = massdatadir + massfname
    massrawData = np.genfromtxt(massfilepath, delimiter=',', skip_header=4, dtype="|S20") # includes the header

    mass = {'time': np.array([dt.datetime.strptime(i[0], '%d/%m/%Y') for i in massrawData[1:]]),
            'CBLK': np.array([np.nan if i[2] == 'No data' else i[2] for i in massrawData[1:]], dtype=float),
            'CORG': np.array([np.nan if i[4] == 'No data' else i[4] for i in massrawData[1:]], dtype=float)}

    # times were valid from 12:00 so add the extra 12 hours on
    mass['time'] += dt.timedelta(hours=12)

    # convert timezone from GMT to UTC
    from_zone = tz.gettz('GMT')
    to_zone = tz.gettz('UTC')

    mass['time'] = [i.replace(tzinfo=from_zone) for i in mass['time']] # set datetime's original timezone as GMT
    mass['time'] = np.array([i.astimezone(to_zone) for i in mass['time']]) # convert from GMT to UTC

    # convert units from micrograms to grams
    mass['CBLK'] *= 1e-06
    mass['CORG'] *= 1e-06

    # QAQC - turn all negative values in each column into nans if one of them is negative
    for aer_i in ['CBLK', 'CORG']:
        idx = np.where(mass[aer_i] < 0.0)
        mass[aer_i][idx] = np.nan


    return mass

def read_pm10_mass_data(massdatadir, site_meta, year):

    """
    Read in the other pm10 mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :return: mass

    """

    # make sure year is string
    year = str(year)

    massfname = 'pm10species_Hr_'+site_meta['site_long']+'_DEFRA_'+year+'.csv'
    massfilepath = massdatadir + massfname
    massrawData = np.genfromtxt(massfilepath, delimiter=',', skip_header=4, dtype="|S20") # includes the header

    # sort out the raw time. Data only has the DD/MM/YYYY and no hour... and is somehow in GMT too...
    # therefore need to add the hour part of the date manually
    raw_time = np.array([dt.datetime.strptime(i[0], '%d/%m/%Y') for i in massrawData[1:]])
    # as data is internally complete (no time gaps) can create a datelist with the hours
    time_endHr = np.array(eu.date_range(raw_time[0],
                                        raw_time[-1] + dt.timedelta(days=1) - dt.timedelta(hours=1), 60, 'minutes'))
    time_strtHr = time_endHr - dt.timedelta(hours=1)

    mass = {'time': time_strtHr,
            'CL': np.array([np.nan if i[1] == 'No data' else i[1] for i in massrawData[1:]], dtype=float),
            'Na': np.array([np.nan if i[3] == 'No data' else i[3] for i in massrawData[1:]], dtype=float),
            'NH4': np.array([np.nan if i[5] == 'No data' else i[5] for i in massrawData[1:]], dtype=float),
            'NO3': np.array([np.nan if i[7] == 'No data' else i[7] for i in massrawData[1:]], dtype=float),
            'SO4': np.array([np.nan if i[9] == 'No data' else i[9] for i in massrawData[1:]], dtype=float)}

    # convert units from micrograms to grams
    # QAQC - turn all negative values in each column into nans if one of them is negative
    for key in mass.iterkeys():
        if key != 'time':
            mass[key] *= 1e-06

            # QAQC (values < 0 are np.nan)
            idx = np.where(mass[key] < 0.0)
            mass[key][idx] = np.nan


    return mass

def trim_mass_wxt_times(mass, WXT):

    """
    Trim the mass and WXT data based on their start and end times
    DOES NOT CHECK INTERNAL TIME MATCHING (BULK TRIM APPROACH)
    :return: mass
    :return: wxt
    """

    # Find start and end idx for mass and WXT
    if WXT['time'][0] < mass['time'][0]:
        wxt_start = np.where(WXT['time'] == mass['time'][0])[0][0]
        mass_start = 0
    else:
        wxt_start = 0
        mass_start = np.where(mass['time'] == WXT['time'][0])[0][0]
    # END
    if WXT['time'][-1] > mass['time'][-1]:
        wxt_end = np.where(WXT['time'] == mass['time'][-1])[0][0]
        mass_end = len(mass['time']) - 1
    else:
        wxt_end = len(WXT['time']) - 1
        mass_end = np.where(mass['time'] == WXT['time'][-1])[0][0]

    # create idx ranges where data exists for both mass and WXT
    wxt_range = np.arange(wxt_start, wxt_end, 1)
    mass_range = np.arange(mass_start, mass_end, 1)

    # trim data by only selecting the right time ranges
    for key, data in WXT.iteritems():
        WXT[key] = data[wxt_range]

    for key, data in mass.iteritems():
        mass[key] = data[mass_range]

    # returned data should have the same start and end times

    return mass, WXT


# Process

## data in processing

def Geisinger_increase_r_bins(dN, r_d_orig_bins_microns, n_samples=4.0):

    """
    Increase the number of sampling bins from the original data using a sampling method
    from Geisinger et al., 2017. Full equations are in the discussion manuscript, NOT the final draft.

    :param dN:
    :param r_d_orig_bins_microns:
    :param n_samples:
    :return: R_dg_microns
    :return: dN (with geisinger_idx) - which interpolated bin came from what instruments original diameters.
                        Need to be able to split which diameters need swelling and which need shrinking.
    """

    # get bin edges based on the current bins (R_da is lower, R_db is upper edge)
    R_db = (dN['D'] + (0.5 * dN['dD'])) / 2.0 # upper
    R_da = (dN['D'] - (0.5 * dN['dD'])) / 2.0 # lower

    # create the radii values in between the edges (evenly spaced within each bin)
    R_dg = np.array([(g * ((R_db[i] - R_da[i])/n_samples)) + R_da[i]
                     for i in range(len(r_d_orig_bins_microns))
                     for g in range(1,int(n_samples)+1)])

    # add the idx positions for the geisinger bins that came from each instrument to dN
    #   checked using smps and grimm from Chilbolton
    #   smps_orginal = 0 to 50, grimm_original = 51 to 74
    #   therefore, smps_geisinger = 0 to 203, grimm_geisinger = 204 to 299 (with smps + grimm total being 300 positions)
    #   all idx positions above are inclusive ranges
    dN['smps_geisinger_idx'] = np.arange(dN['smps_idx'][0] * int(n_samples), (dN['smps_idx'][-1] + 1) * int(n_samples))
    dN['grimm_geisinger_idx'] = np.arange(dN['grimm_idx'][0] * int(n_samples), (dN['grimm_idx'][-1] + 1) * int(n_samples))

    # convert to microns from nanometers
    R_dg_microns = R_dg * 1e-3

    return R_dg_microns, dN

### not used now
def WXT_hourly_average(WXT_in):


    """
    Average up the WXT data to hourly values
    :param WXT_in:
    :return: WXT_hourly
    """

    # WXT average up to hourly
    date_range = np.array(eu.date_range(WXT_in['time'][0], WXT_in['time'][-1] + dt.timedelta(days=1), 60, 'minutes'))

    # set up hourly array
    WXT_hourly = {'time': date_range}
    for var in WXT_in.iterkeys():
        if ((var == 'rawtime') | (var == 'time')) == False:

            WXT_hourly[var] = np.empty(len(date_range))
            WXT_hourly[var][:] = np.nan

    # take hourly averages of the 15 min data
    for t, time_t in enumerate(date_range):

        # find data within the hour
        bool = np.logical_and(WXT_in['time'] > time_t, WXT_in['time'] < (time_t + dt.timedelta(hours=1)))

        for var in WXT_in.iterkeys():
            if ((var == 'rawtime') | (var == 'time')) == False:

                # take mean of the data
                WXT_hourly[var][t] = np.nanmean(WXT_in[var][bool])


    return WXT_hourly

### not used now
def internal_time_completion(data, date_range):

    """
    Set up new dictionary for data with a complete time series (no gaps in the middle).
    :param: data (must be a dictionary with a list of datetimes with the keyname 'time', i.e. data['time'])
    :return: data_full

    Done by checking if time_i from the complete date range (with no time gaps) exists in the data, and if so, extract
    out the values and put it into the new dictionary.
    """


    # set up temporary dictionaries for data (e.g. data_full) with empty arrays for each key, ready to be filled
    data_full = {}
    for h in data.iterkeys():
        data_full[h] = np.empty(len(date_range))
        data_full[h][:] = np.nan

    # replace time with date range
    data_full['time'] = date_range

    # step through time and time match data for extraction
    for t, time_t in enumerate(date_range):
        idx = np.where(data['time'] == time_t)[0]

        # if not empty, put in the data to new array
        if idx.size != 0:
            for h in data.iterkeys():
                data_full[h][t] = data[h][idx]

    return data_full

def merge_pm_mass(pm_mass_in, pm_oc_bc):

    """
    Merge (including time matching) the pm10 masses together (OC and BC in with the others).

    :param pm_mass_in:
    :param pm_oc_bc:
    :return: pm_mass_all
    """

    # double check that OC_EC['time'] is an array, not a list, as lists do not work with the np.where() function below.
    if type(pm_oc_bc['time']) == list:
        pm_oc_bc['time'] = np.array(pm_oc_bc['time'])

    # set up pm_mass_all dictionary
    # doesn't matter which mass['time'] is used, as if data is missing from either dataset for a time t, then that
    #   time is useless anyway.


    # fill pm10_mass_all with the OC and BC
    for key in ['CORG', 'CBLK']:
       pm_mass_in[key] = np.empty(len(pm_mass_in['time']))
       pm_mass_in[key][:] = np.nan

    # fill the pm10_merge arrays
    for t, time_t in enumerate(pm_mass_in['time']):

        # find corresponding time
        idx_oc_bc = np.where(np.array(pm_oc_bc['time']) == time_t)

        # fill array
        if idx_oc_bc[0].size != 0:
            for key in ['CORG', 'CBLK']:
                pm_mass_in[key][t] = pm_oc_bc[key][idx_oc_bc]

    return pm_mass_in

### not used now
def coarsen_PM1_mass_hourly(WXT_hourly, PM1_mass):

    """
    Coarsen the PM1 data from 15 mins to hourly data to match pm10 and WXT hourly data
    :param WXT_hourly:
    :param PM1_mass:
    :return: PM1_mass_hourly
    """

    # variables to process from PM1_mass (ignore time, pm1, pm10 etc)
    PM1_process_vars = ['CORG', 'CL', 'CBLK', 'NH4', 'SO4', 'NO3']

    # time match pm1 to WXT_hourly and pm10 data
    PM1_mass_hourly = {'time': WXT_hourly['time']}

    for key in PM1_process_vars:
        PM1_mass_hourly[key] = np.empty(len(PM1_mass_hourly['time']))
        PM1_mass_hourly[key][:] = np.nan

    # define an idx range to time match in, that will 'move' up with each iteration of t, to minimise computation time as
    #   the 15 min raw array can be very large!
    start_idx = 0
    for t, start, end in zip(np.arange(len(PM1_mass_hourly['time'][:-1])), PM1_mass_hourly['time'][:-1], PM1_mass_hourly['time'][1:]):

        # search for matching times only this many idx positions ahead of start_idx
        end_idx = start_idx + 50

        # print t

        # find where data is within the hour
        bool = np.logical_and(PM1_mass['time'][start_idx:end_idx] > start, PM1_mass['time'][start_idx:end_idx] <= end)
        idx_trim = np.where(bool == True)[0] # idx in the [start_idx:end_idx] bit
        idx = idx_trim + start_idx # idx in the entire array [:]

        if idx.size != 0:
            for key in PM1_process_vars:
                PM1_mass_hourly[key][t] = np.nanmean(PM1_mass[key][idx])

            # move the start position by the extra few positions passed during this loop.
            start_idx += idx_trim[-1]



    return PM1_mass_hourly

def two_pm_dataset_difference(pm_small_mass, pm_big_mass):

    """
    Take the difference of the two pm datasets (smaller one first in the arguments list!)
    :param pm_small_mass: the smaller mass dataset (e.g. pm2p5 mass)
    :param pm_big_mass: the larger mass dataset (e.g. pm10 mass)
    :return: pm_diff_mass: differences of the two datasets
    """

    # create pm10m2p5 mass (pm10 minus pm2.5)
    pm_diff_mass = {'time': pm_big_mass['time']}
    for key in pm_small_mass.iterkeys():
        if key != 'time':
            pm_diff_mass[key] = pm_big_mass[key] - pm_small_mass[key]

            # QAQC
            # PM_big - PM_small is not always >0
            #   Therefore np.nan all negative masses!
            idx = np.where(pm_diff_mass[key] < 0)
            pm_diff_mass[key][idx] = np.nan

    return pm_diff_mass

def time_match_pm_RH_dN(pm2p5_mass_in, pm10_mass_in, met_in, dN_in, timeRes):

    """
    time match all the main data dictionaries together (pm, RH and dN data), according to the time resolution given
    (timeRes). Makes all values nan for a time, t, if any one of the variables has missing data (very conservative
    appoach).

    :param pm2p5_mass_in:
    :param pm10_mass_in:
    :param met_in: contains RH, Tair, air pressure
    :param dN_in:
    :param timeRes:
    :return: pm2p5_mass, pm10_mass, met, dN
    """

    ## 1. set up dictionaries with times
    # Match data to the dN data.
    # time range - APS time res: 5 min, DMPS time res: ~12 min
    start_time = dN_in['time'][0]
    end_time = dN_in['time'][-1]
    time_range = eu.date_range(start_time, end_time, timeRes, 'minutes')

    # make sure datetimes are in UTC
    from_zone = tz.gettz('GMT')
    to_zone = tz.gettz('UTC')
    time_range = np.array([i.replace(tzinfo=from_zone) for i in time_range])


    # set up dictionaries (just with time and any non-time related values at the moment)
    pm2p5_mass = {'time': time_range}
    pm10_mass = {'time': time_range}
    dN = {'time': time_range, 'D': dN_in['D'], 'dD': dN_in['dD'],
          'grimm_idx': dN_in['grimm_idx'], 'smps_idx': dN_in['smps_idx'],
          'grimm_geisinger_idx': dN_in['grimm_geisinger_idx'], 'smps_geisinger_idx': dN_in['smps_geisinger_idx']}
    met = {'time': time_range}

    ## 2. set up empty arrays within dictionaries
    # prepare empty arrays within the outputted dictionaries for the other variables, ready to be filled.
    for var, var_in in zip([pm2p5_mass, pm10_mass, met, dN], [pm2p5_mass_in, pm10_mass_in, met_in, dN_in]):

        for key in var_in.iterkeys():
            # only fill up the variables
            if key not in ['time', 'D', 'dD', 'grimm_idx', 'smps_idx', 'grimm_geisinger_idx', 'smps_geisinger_idx']:

                # make sure the dimensions of the arrays are ok. Will either be 1D (e.g RH) or 2D (e.g. dN)
                dims = var_in[key].ndim
                if dims == 1:
                    var[key] = np.empty(len(time_range))
                    var[key][:] = np.nan
                else:
                    var[key] = np.empty((len(time_range), var_in[key].shape[1]))
                    var[key][:] = np.nan


    ## 3. fill the variables with time averages
    # use a moving subsample assuming the data is in ascending order
    for var, var_in in zip([pm2p5_mass, pm10_mass, met, dN], [pm2p5_mass_in, pm10_mass_in, met_in, dN_in]):

        # set skip idx to 0 to begin with
        #   it will increase after each t loop
        skip_idx = 0

        for t in range(len(time_range)):


            # find data for this time
            binary = np.logical_and(var_in['time'][skip_idx:skip_idx+100] > time_range[t],
                                    var_in['time'][skip_idx:skip_idx+100] <= time_range[t] + dt.timedelta(minutes=timeRes))

            # actual idx of the data within the entire array
            skip_idx_set_i = np.where(binary == True)[0] + skip_idx

            # create means of the data for this time period
            for key in var.iterkeys():
                if key not in ['time', 'D', 'dD', 'grimm_idx', 'grimm_geisinger_idx', 'smps_idx', 'smps_geisinger_idx']:

                    dims = var_in[key].ndim
                    if dims == 1:
                        var[key][t] = np.nanmean(var_in[key][skip_idx_set_i])
                    else:
                        var[key][t, :] = np.nanmean(var_in[key][skip_idx_set_i, :], axis=0)

            # change the skip_idx for the next loop to start just after where last idx finished
            if skip_idx_set_i.size != 0:
                skip_idx = skip_idx_set_i[-1] + 1


    ## 4. nan across variables for missing data
    # make data for any instance of time, t, to be nan if any data is missing from dN, met or pm mass data

    ## 4.1 find bad items
    # make and append to a list, rows where bad data is present, across all the variables
    bad = []

    for var in [pm2p5_mass, pm10_mass, met, dN]:

        for key, data in var.iteritems():

            if key not in ['time', 'D', 'dD', 'grimm_idx', 'grimm_geisinger_idx', 'smps_idx', 'smps_geisinger_idx']:

                # number of dimensions for data
                dims = data.ndim
                if dims == 1:
                    for t in range(len(time_range)):
                        if np.isnan(data[t]):
                            bad += [t]

                else:
                    for t in range(len(time_range)):
                        if any(np.isnan(data[t, :]) == True): # any nans in the row
                            bad += [t] # store the time idx as being bad

    ## 4.2 find unique bad idxs and make all values at that time nan, across all the variables
    bad_uni = np.unique(np.array(bad))

    for var in [pm2p5_mass, pm10_mass, met, dN]:

        for key, data in var.iteritems():

            if key not in ['time', 'D', 'dD', 'grimm_idx', 'grimm_geisinger_idx', 'smps_idx', 'smps_geisinger_idx']:

                # number of dimensions for data
                dims = data.ndim
                if dims == 1:
                    var[key][bad_uni] = np.nan

                else:
                    var[key][bad_uni, :] = np.nan



    return pm2p5_mass, pm10_mass, met, dN

# create a combined num_concentration with the right num_conc for the right D bins. e.g. pm2p5 for D < 2.5
def merge_two_pm_dataset_num_conc(num_conc_pm2p5, num_conc_pm10m2p5, dN, limit):

    """
    Merge the number concentration dictionaries from the two pm datasets by taking the right part of each
    e.g. num_conc for D <2.5 = num_conc from 2.5
    :param num_conc_pm2p5:
    :param num_conc_pm10m2p5:
    :param limit:
    :return:
    """

    # pm10 - PM1 (use the right parts of the num concentration for the rbins e.g. pm1 mass for r<1, pm10-1 for r>1)
    idx_pm2p5 = np.where(dN['D'] <= limit)[0] # 'D' is in nm not microns!
    idx_pm10m2p5 = np.where(dN['D'] > limit)[0]

    # concatonate num_conc
    # r<=1 micron are weighted by PM1, r>1 are weighted by pm10-1
    num_conc = {}
    for aer_i in num_conc_pm2p5.iterkeys():
        num_conc[aer_i] = np.hstack((num_conc_pm2p5[aer_i][:, idx_pm2p5], num_conc_pm10m2p5[aer_i][:, idx_pm10m2p5]))

    return num_conc, idx_pm2p5, idx_pm10m2p5

## masses and moles

### main masses and moles script
def calculate_moles_masses(mass, met, aer_particles, inc_soot=False):

    """
    Calculate the moles and mass [kg kg-1] of the aerosol. Can set soot to on or off (turn all soot to np.nan)
    :param mass: [g cm-3]
    :param met:
    :param aer_particles:
    :param inc_soot: [bool]
    :return: moles, mass_kg_kg
    """

    # molecular mass of each molecule
    mol_mass_amm_sulp = 132
    mol_mass_amm_nit = 80
    mol_mass_nh4 = 18
    mol_mass_n03 = 62
    mol_mass_s04 = 96
    mol_mass_Cl = 35.45

    # Convert into moles
    # calculate number of moles (mass [g] / molar mass)
    # 1e-06 converts from micrograms to grams.
    moles = {'SO4': mass['SO4'] / mol_mass_s04,
             'NO3': mass['NO3'] / mol_mass_n03,
             'NH4': mass['NH4'] / mol_mass_nh4,
             'CL':  mass['CL'] / mol_mass_Cl}


    # calculate ammonium sulphate and ammonium nitrate from gases
    # adds entries to the existing dictionary
    moles, mass = calc_amm_sulph_and_amm_nit_from_gases(moles, mass)

    # convert chlorine into sea salt assuming all chlorine is sea salt, and enough sodium is present.
    #      potentially weak assumption for the chlorine bit due to chlorine depletion!
    mass['NaCl'] = mass['CL'] * 1.65
    moles['NaCl'] = moles['CL']

    # convert masses from g m-3 to kg kg-1_air for swelling.
    # Also creates the air density and is stored in WXT
    mass_kg_kg, WXT = convert_mass_to_kg_kg(mass, met)


    # temporarily make black carbon mass nan
    if inc_soot == False:
        print ' SETTING BLACK CARBON MASS TO NAN'
        mass_kg_kg['CBLK'][:] = np.nan

    return moles, mass_kg_kg

def calc_amm_sulph_and_amm_nit_from_gases(moles, mass):

    """
    Calculate the ammount of ammonium nitrate and sulphate from NH4, SO4 and NO3.
    Follows the CLASSIC aerosol scheme approach where all the NH4 goes to SO4 first, then to NO3.

    :param moles:
    :param mass:
    :return: mass [with the extra entries for the particles]
    """

    # define aerosols to make
    mass['(NH4)2SO4'] = np.empty(len(moles['SO4']))
    mass['(NH4)2SO4'][:] = np.nan
    mass['NH4NO3'] = np.empty(len(moles['SO4']))
    mass['NH4NO3'][:] = np.nan

    moles['(NH4)2SO4'] = np.empty(len(moles['SO4']))
    moles['(NH4)2SO4'][:] = np.nan
    moles['NH4NO3'] = np.empty(len(moles['SO4']))
    moles['NH4NO3'][:] = np.nan

    # calculate moles of the aerosols
    # help on GCSE bitesize:
    #       http://www.bbc.co.uk/schools/gcsebitesize/science/add_gateway_pre_2011/chemical/reactingmassesrev4.shtml
    for i in range(len(moles['SO4'])):
        if moles['SO4'][i] > (moles['NH4'][i] / 2):  # more SO4 than NH4 (2 moles NH4 to 1 mole SO4) # needs to be divide here not times

            # all of the NH4 gets used up making amm sulph.
            mass['(NH4)2SO4'][i] = mass['NH4'][i] * 7.3  # ratio of molecular weights between amm sulp and nh4
            moles['(NH4)2SO4'][i] = moles['NH4'][i]
            # rem_nh4 = 0

            # no NH4 left to make amm nitrate
            mass['NH4NO3'][i] = 0
            moles['NH4NO3'][i] = 0
            # some s04 gets wasted
            # rem_SO4 = +ve

        # else... more NH4 to SO4
        elif moles['SO4'][i] < (moles['NH4'][i] / 2):  # more NH4 than SO4 for reactions

            # all of the SO4 gets used in reaction
            mass['(NH4)2SO4'][i] = mass['SO4'][i] * 1.375  # ratio of SO4 to (NH4)2SO4
            moles['(NH4)2SO4'][i] = moles['SO4'][i]
            # rem_so4 = 0

            # some NH4 remains this time!
            # remove 2 * no of SO4 moles used from NH4 -> SO4: 2, NH4: 5; therefore rem_nh4 = 5 - (2*2)
            rem_nh4 = moles['NH4'][i] - (moles['SO4'][i] * 2)

            if moles['NO3'][i] > rem_nh4:  # if more NO3 to NH4 (1 mol NO3 to 1 mol NH4)

                # all the NH4 gets used up
                mass['NH4NO3'][i] = rem_nh4 * 4.4  # ratio of amm nitrate to remaining nh4
                moles['NH4NO3'][i]  = rem_nh4
                # rem_nh4 = 0

                # left over NO3
                # rem_no3 = +ve

            elif moles['NO3'][i] < rem_nh4:  # more remaining NH4 than NO3

                # all the NO3 gets used up
                mass['NH4NO3'][i] = mass['NO3'][i] * 1.29
                moles['NH4NO3'][i] = moles['NO3'][i]
                # rem_no3 = 0

                # some left over nh4 still
                # rem_nh4_2ndtime = +ve

    return moles, mass

def convert_mass_to_kg_kg(mass, met):

    """
    Convert mass molecules from g m-3 to kg kg-1

    :param mass
    :param met (for meteorological data - needs Tair and pressure)
    :param aer_particles (not all the keys in mass are the species, therefore only convert the species defined above)
    :return: mass_kg_kg: mass in kg kg-1 air
    """

    #
    T_K = met['Tair'] # [K]
    p_Pa = met['pressure'] # [Pa]

    # density of air [kg m-3] # assumes dry air atm
    # p = rho * R * T [K]
    met['dryair_rho'] = p_Pa / (286.9 * T_K)

    # convert g m-3 air to kg kg-1 of air
    mass_kg_kg = {'time': mass['time']}
    for key in mass.iterkeys():
        if key is not 'time':
            mass_kg_kg[key] = mass[key] * 1e3 / met['dryair_rho']

    return mass_kg_kg, met

def oc_bc_interp_hourly(oc_bc_in):

    """
    Increase EC and OC data resolution from daily to hourly with a simple linear interpolation

    :param oc_bc_in:
    :return:oc_bc
    """

    # Increase to hourly resolution by linearly interpolate between the measurement times
    date_range = eu.date_range(oc_bc_in['time'][0], oc_bc_in['time'][-1] + dt.timedelta(days=1), 60, 'minutes')
    oc_bc = {'time': date_range,
                    'CORG': np.empty(len(date_range)),
                    'CBLK': np.empty(len(date_range))}

    oc_bc['CORG'][:] = np.nan
    oc_bc['CBLK'][:] = np.nan

    # fill hourly data
    for aer_i in ['CORG', 'CBLK']:

        # for each day, spready data out into the hourly slots
        # do not include the last time, as it cannot be used as a start time (fullday)
        for t, fullday in enumerate(oc_bc_in['time'][:-1]):

            # start = fullday
            # end = fullday + dt.timedelta(days=1)

            # linearly interpolate between the two main dates
            interp = np.linspace(oc_bc_in[aer_i][t], oc_bc_in[aer_i][t+1],24)

            # idx range as the datetimes are internally complete, therefore a number sequence can be used without np.where()
            idx = np.arange(((t+1)*24)-24, (t+1)*24)

            # put the values in
            oc_bc[aer_i][idx] = interp


    return oc_bc

## aerosol physical properties besides mass

def est_num_conc_by_species_for_Ndist(aer_particles, mass_kg_kg, aer_density, met, radius_k, dN):

    """

    :param aer_particles:
    :param mass_kg_kg: [kg kg-1]
    :param aer_density: [kg m-3]
    :param radius_k: dictionary with a float value for each aerosol (aer_i) [m]
    :return:
    """

    # work out Number concentration (relative weight) for each species [m-3]
    # calculate the number of particles for each species using radius_m and the mass
    num_part = {}
    for aer_i in aer_particles:
        num_part[aer_i] = mass_kg_kg[aer_i] / ((4.0/3.0) * np.pi * (aer_density[aer_i]/met['dryair_rho']) * (radius_k[aer_i] ** 3.0))

    # find relative N from N(mass, r_md)
    N_weight = {}
    num_conc = {}
    for aer_i in aer_particles:

        # relative weighting of N for each species (aerosol_i / sum of all aerosol for each time)
        # .shape(time,) - N_weight['CORG'] has many over 1
        # was nansum but changed to just sum, so times with missing data are nan for all aerosols
        #   only nans this data set thougha nd not the other data (e.g. pm10 and therefore misses pm1)
        N_weight[aer_i] = num_part[aer_i] / np.sum(np.array(num_part.values()), axis=0)

        # estimated number for the species, from the main distribution data, using the weighting,
        #    for each time step
        num_conc[aer_i] = np.tile(N_weight[aer_i], (len(dN['med']),1)).transpose() * \
                          np.tile(dN['med'], (len(N_weight[aer_i]),1))


    return N_weight, num_conc

def N_weights_from_pm_mass(aer_particles, mass_kg_kg, aer_density, met, radius_m):

    """
    N_weight calcuated from pm mass, to be used to weight the number distribution, dN.
    :param aer_particles:
    :param mass_kg_kg: [kg kg-1]
    :param aer_density: [kg m-3]
    :param radius_k: dictionary with a float value for each aerosol (aer_i) [m]
    :return:
    """

    # work out Number concentration (relative weight) for each MAIN species as defined by aer_particles[m-3]
    # calculate the number of particles for each species using radius_m and the mass
    num_part = {}
    for aer_i in aer_particles:
        num_part[aer_i] = mass_kg_kg[aer_i] / ((4.0/3.0) * np.pi * (aer_density[aer_i]/met['dryair_rho']) * (radius_m[aer_i] ** 3.0))

    # find relative N from N(mass, r_md)
    N_weight = {}
    #num_conc = {}
    for aer_i in aer_particles:

        # relative weighting of N for each species (aerosol_i / sum of all aerosol for each time)
        # .shape(time,) - N_weight['CORG'] has many over 1
        # was nansum but changed to just sum, so times with missing data are nan for all aerosols
        #   only nans this data set thougha nd not the other data (e.g. pm10 and therefore misses pm1)
        N_weight[aer_i] = num_part[aer_i] / np.sum(np.array(num_part.values()), axis=0)

        # # estimated number for the species, from the main distribution data, using the weighting,
        # #    for each time step
        # num_conc[aer_i] = np.tile(N_weight[aer_i], (len(dN['med']),1)).transpose() * \
        #                   np.tile(dN['med'], (len(N_weight[aer_i]),1))


    return N_weight

def calc_dry_volume_from_mass(aer_particles, mass_kg_kg, aer_density):


    """
    Calculate the dry volume from the mass of all the species
    :param aer_particles:
    :param mass_kg_kg:
    :param aer_density:
    :return:
    """

    # calculate dry volume
    V_dry_from_mass = {}
    for aer_i in aer_particles:
        # V_dry[aer_i] = (4.0/3.0) * np.pi * (r_d_m ** 3.0)
        V_dry_from_mass[aer_i] = mass_kg_kg[aer_i] / aer_density[aer_i]  # [m3]

        # if np.nan (i.e. there was no mass therefore no volume) make it 0.0
        bin = np.isnan(V_dry_from_mass[aer_i])
        V_dry_from_mass[aer_i][bin] = 0.0

    return V_dry_from_mass

## swelling / drying

def calc_r_md_all(r_microns, met, pm_mass, gf_ffoc):

    """
    Swell the diameter bins for a set list of aerosol species below:
    ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CBLK', 'CORG']

    :param r_microns:
    :param met:
    :param pm_mass:
    :param gf_ffoc:
    :return: r_md [microns]
    :return r_md_m [meters]
    """

    # set up dictionary
    r_md = {}

    ## 1. ['(NH4)2SO4', 'NH4NO3', 'NaCl']

    # calculate the swollen particle size for these three aerosol types
    # Follows CLASSIC guidence, based off of Fitzgerald (1975)
    # guidance requires radii units to be microns
    for aer_i in ['(NH4)2SO4', 'NH4NO3', 'NaCl']:
        r_md[aer_i] = calc_r_md_species(r_microns, met, aer_i)

    ## 2. Black carbon ('CBLK')

    # set r_md for black carbon as r_d, assuming black carbon is completely hydrophobic
    # create a r_d_microns_dry_dup (rbins copied for each time, t) to help with calculations
    r_md['CBLK'] = np.tile(r_microns, (len(met['time']), 1))

    # make r_md['CBLK'] nan for all sizes, for times t, if mass data is not present for time t
    # doesn't matter which mass is used, as all mass data have been corrected for if nans were present in other datasets
    r_md['CBLK'][np.isnan(pm_mass['CBLK']), :] = np.nan

    ## 3. Organic carbon ('CORG')

    # calculate r_md for organic carbon using the MO empirically fitted g(RH) curves
    r_md['CORG'] = np.empty((len(met['time']), len(r_microns)))
    r_md['CORG'][:] = np.nan
    for t, time_t in enumerate(met['time']):

        _, idx, _ = eu.nearest(gf_ffoc['RH_frac'], met['RH_frac'][t])
        r_md['CORG'][t, :] = r_microns * gf_ffoc['GF'][idx]


    # convert r_md units from microns to meters
    r_md_m = {}
    for aer_i in r_md.iterkeys():
        r_md_m[aer_i] = r_md[aer_i] * 1e-06


    return r_md, r_md_m

def calc_r_md_species(r_d_microns, met, aer_i):

    """
    Calculate the r_md [microns] for all particles, given the RH [fraction] and what species
    Swells particles from a dry radius

    :param r_d_microns:
    :param met: meteorological variables (needed for RH and time)
    :param aer_i:
    :return: r_md_t: swollen radii at time, t

    Currently just works for ammonium sulphate, ammonium nitrate and NaCl
    04/12/17 - works for range of r values, not just a scaler.
    """


    # calulate r_md based on Fitzgerald (1975) eqn 8 - 10
    def calc_r_md_t(r_d_microns, rh_i, alpha_factor):

        """
        Calculate r_md for a single value of rh (rh_i) at a time t (alpha and beta will be applied to all rbins)
        :param rh_i:
        :param r_d_microns: NOt the duplicated array!
        :return: r_md_i


        The r_md calculated here will be for a fixed RH, therefore the single row of r_d_microns will be fine, as it
        will compute a single set of r_md as a result.
        """

        beta = np.exp((0.00077 * rh_i) / (1.009 - rh_i))
        if rh_i < 0.97:
            phi = 1.058 - ((0.0155 * (rh_i - 0.97))
                           / (1.02 - (rh_i ** 1.4)))
        else:
            phi = 1.058
        alpha = 1.2 * np.exp((0.066 * rh_i) / (phi - rh_i))

        # alpha factor comes from the Table 1 in Fitzgerald (1975) to be used with some other aerosol types
        r_md_t = alpha_factor * alpha * (r_d_microns ** beta)

        return r_md_t



    # duplicate the range of radii to multiple rows, one for each RH - shape(time, rbin).
    # Remember: the number in each diameter bin might change, but the bin diameters themselves will not.
    #   Therefore this approach works for constant and time varying number distirbutions.
    r_d_microns_dup = np.tile(r_d_microns, (len(met['time']), 1))

    # Set up array for aerosol
    r_md =  np.empty(len(met['time']))
    r_md[:] = np.nan

    phi = np.empty(len(met['time']))
    phi[:] = np.nan

    # limits for what approach to use, depending on the RH
    # from the CLASSIC guidence, follows Fitzgerald (1975)
    if aer_i == '(NH4)2SO4':
        rh_cap = 0.995 # calculate r_md specifically for the upper limit (considered max rh)
        rh_del = 0.81 # calculate r_md specifically for the upper limit (start of empirical formula)
                     # CLASSIC does linear interpolation bettween rh_del and rh_eff.
        rh_eff = 0.3 # efflorescence (below is dry)
        alpha_factor = 1.0 # a coefficient for alpha, which is specific for different aerosol types
    elif aer_i == 'NH4NO3':
        rh_cap = 0.995
        rh_del = 0.61
        rh_eff = 0.3
        alpha_factor = 1.06

    elif aer_i == 'NaCl':
        rh_cap = 0.995
        rh_del = 0.75
        rh_eff = 0.42
        alpha_factor = 1.35

    # --------------------------------------------
    # Calculate r_md for the species, given RH
    # -----------------------------------------------

    # empirical relationships fitted for radius in micrometers, not meters (according to CLASSIC guidance).

    # --- delequescence - rh cap (defined as 0.995. Above this empirical relationship breaks down) --- #

    # Currently just calculates it for all, then gets overwritten lower down, depending on their RH (e.g. below eff)
    # ToDo use the rh_bet_del_cap to only calc for those within the del - cap range.

    # # between deliquescence and rh_cap (set at 0.995 for all)
    # bool = np.logical_and(WXT['RH_frac'] >= rh_del, WXT['RH_frac'] <= rh_cap)
    # rh_bet_del_cap = np.where(bool == True)[0]

    beta = np.exp((0.00077 * met['RH_frac'])/(1.009 - met['RH_frac']))
    rh_lt_97 = met['RH_frac'] < 0.97
    phi[rh_lt_97] = 1.058
    phi[~rh_lt_97] = 1.058 - ((0.0155 * (met['RH_frac'][~rh_lt_97] - 0.97))
                              /(1.02 - (met['RH_frac'][~rh_lt_97] ** 1.4)))
    alpha = 1.2 * np.exp((0.066 * met['RH_frac'])/ (phi - met['RH_frac']))

    # duplicate values across to all radii bins to help r_md = .. calculation: alpha_dup.shape = (time, rbin)
    alpha_dup = np.tile(alpha, (len(r_d_microns), 1)).transpose()
    beta_dup = np.tile(beta, (len(r_d_microns), 1)).transpose()

    r_md = alpha_factor * alpha_dup * (r_d_microns_dup ** beta_dup)

    # --- above rh_cap ------#

    # set all r_md(RH>99.5%) to r_md(RH=99.5%) to prevent growth rates inconsistent with impirical equation.
    # replace all r_md values above 0.995 with 0.995
    rh_gt_cap = met['RH_frac'] > rh_cap
    r_md[rh_gt_cap, :] = calc_r_md_t(r_d_microns, rh_cap, alpha_factor)

    # --- 0 to efflorescence --- #

    # below efflorescence point (0.3 for sulhate, r_md = r_d)
    rh_lt_eff = met['RH_frac'] <= rh_eff
    r_md[rh_lt_eff, :] = r_d_microns

    # ------ efflorescence to deliquescence ----------#

    # calculate r_md for the deliquescence rh - used in linear interpolation
    r_md_del = calc_r_md_t(r_d_microns, rh_del, alpha_factor)

    # all values that need to have some linear interpolation
    bool = np.logical_and(met['RH_frac'] >= rh_eff, met['RH_frac'] <= rh_del)
    rh_bet_eff_del = np.where(bool == True)[0]

    # between efflorescence point and deliquescence point, r_md is expected to value linearly between the two
    low_rh = rh_eff
    up_rh = rh_del
    low_r_md = r_d_microns
    up_r_md = r_md_del

    diff_rh = up_rh - low_rh
    diff_r_md = r_md_del - r_d_microns
    abs_diff_r_md = abs(diff_r_md)

    # find distance rh is along linear interpolation [fraction] from lower limit
    # frac = np.empty(len(r_md))
    # frac[:] = np.nan
    frac = ((met['RH_frac'][rh_bet_eff_del] - low_rh) / diff_rh)

    # duplicate abs_diff_r_md by the number of instances needing to be interpolated - helps the calculation below
    #   of r_md = ...low + (frac * abs diff)
    abs_diff_r_md_dup = np.tile(abs_diff_r_md, (len(rh_bet_eff_del), 1))
    frac_dup = np.tile(frac, (len(r_d_microns), 1)).transpose()

    # calculate interpolated values for r_md
    r_md[rh_bet_eff_del, :] = low_r_md + (frac_dup * abs_diff_r_md_dup)

    return r_md

def calc_r_d_all(r_microns, met, pm_mass, gf_ffoc):

    """
    Calculate r_d [microns] for all particles, given the RH [fraction] and what species
    Dries particles from a wet radius

    :param r_d_microns:
    :param met: meteorological variables (needed for RH and time)
    :param aer_i:
    :return: r_d: dry radii [mircons]
    :return r_d_m dry radii [meters]

    Currently just works for ammonium sulphate, ammonium nitrate and NaCl
    04/12/17 - works for range of r values, not just a scaler.
    """

    # set up dictionary
    r_d = {}


    # calculate the dry particle size for these three aerosol types
    # Follows CLASSIC guidence, based off of Fitzgerald (1975)
    # guidance requires radii units to be microns
    # had to be reverse calculated from CLASSIC guidence.
    for aer_i in ['(NH4)2SO4', 'NH4NO3', 'NaCl']:
        r_d[aer_i] = calc_r_d_species(r_microns, met, aer_i)

    # set r_d for black carbon as r_d, assuming black carbon is completely hydrophobic
    # create a r_d_microns_dry_dup (rbins copied for each time, t) to help with calculations
    r_d['CBLK'] = np.tile(r_microns, (len(met['time']), 1))

    # make r_d['CBLK'] nan for all sizes, for times t, if mass data is not present for time t
    # doesn't matter which mass is used, as all mass data have been corrected for if nans were present in other datasets
    r_d['CBLK'][np.isnan(pm_mass['CBLK']), :] = np.nan

    # calculate r_d for organic carbon using the MO empirically fitted g(RH) curves
    r_d['CORG'] = np.empty((len(met['time']), len(r_microns)))
    r_d['CORG'][:] = np.nan
    for t, time_t in enumerate(met['time']):

        _, idx, _ = eu.nearest(gf_ffoc['RH_frac'], met['RH_frac'][t])
        r_d['CORG'][t, :] = r_microns / gf_ffoc['GF'][idx]

    # convert r_md units from microns to meters
    r_d_m = {}
    for aer_i in r_d.iterkeys():
        r_d_m[aer_i] = r_d[aer_i] * 1e-06


    return r_d, r_d_m

def calc_r_d_species(r_microns, met, aer_i):

    """
    Calculate the r_md [microns] for all particles, given the RH [fraction] and what species
    dries particles

    :param r_microns:
    :param met: meteorological variables (needed for RH and time)
    :param aer_i:
    :return: r_md_t: swollen radii at time, t

    Currently just works for ammonium sulphate, ammonium nitrate and NaCl
    04/12/17 - works for range of r values, not just a scaler.
    """


    # calulate r_md based on Fitzgerald (1975) eqn 8 - 10
    def calc_r_d_t(r_md_microns, rh_i, alpha_factor):

        """
        Calculate r_md for a single value of rh (rh_i) at a time t (alpha and beta will be applied to all rbins)
        :param rh_i:
        :param r_d_microns: NOt the duplicated array!
        :return: r_md_i


        The r_md calculated here will be for a fixed RH, therefore the single row of r_d_microns will be fine, as it
        will compute a single set of r_md as a result.
        """

        # beta
        beta = np.exp((0.00077 * rh_i) / (1.009 - rh_i))

        # alpha
        if rh_i < 0.97:
            phi = 1.058 - ((0.0155 * (rh_i - 0.97))
                           / (1.02 - (rh_i ** 1.4)))
        else:
            phi = 1.058

        alpha = 1.2 * np.exp((0.066 * rh_i) / (phi - rh_i))

        # alpha factor comes from the Table 1 in Fitzgerald (1975) to be used with some other aerosol types
        # original -> r_md_t = alpha_factor * alpha * (r_d_microns ** beta)

        # dry particles
        r_d_t = (r_md_microns/(alpha * alpha_factor)) ** (1.0/beta)

        return r_d_t



    # duplicate the range of radii to multiple rows, one for each RH - shape(time, rbin).
    # Remember: the number in each diameter bin might change, but the bin diameters themselves will not.
    #   Therefore this approach works for constant and time varying number distirbutions.
    r_md_microns_dup = np.tile(r_microns, (len(met['time']), 1))

    # Set up array for aerosol
    r_d =  np.empty(len(met['time']))
    r_d[:] = np.nan

    phi = np.empty(len(met['time']))
    phi[:] = np.nan

    # limits for what approach to use, depending on the RH
    # from the CLASSIC guidence, follows Fitzgerald (1975)
    if aer_i == '(NH4)2SO4':
        rh_cap = 0.995 # calculate r_md specifically for the upper limit (considered max rh)
        rh_del = 0.81 # calculate r_md specifically for the upper limit (start of empirical formula)
                     # CLASSIC does linear interpolation bettween rh_del and rh_eff.
        rh_eff = 0.3 # efflorescence (below is dry)
        alpha_factor = 1.0 # a coefficient for alpha, which is specific for different aerosol types
    elif aer_i == 'NH4NO3':
        rh_cap = 0.995
        rh_del = 0.61
        rh_eff = 0.3
        alpha_factor = 1.06

    elif aer_i == 'NaCl':
        rh_cap = 0.995
        rh_del = 0.75
        rh_eff = 0.42
        alpha_factor = 1.35

    # --------------------------------------------
    # Calculate r_md for the species, given RH
    # -----------------------------------------------

    # empirical relationships fitted for radius in micrometers, not meters (according to CLASSIC guidance).

    # --- delequescence - rh cap (defined as 0.995. Above this empirical relationship breaks down) --- #

    # Currently just calculates it for all, then gets overwritten lower down, depending on their RH (e.g. below eff)
    # ToDo use the rh_bet_del_cap to only calc for those within the del - cap range.

    # # between deliquescence and rh_cap (set at 0.995 for all)
    # bool = np.logical_and(WXT['RH_frac'] >= rh_del, WXT['RH_frac'] <= rh_cap)
    # rh_bet_del_cap = np.where(bool == True)[0]

    beta = np.exp((0.00077 * met['RH_frac'])/(1.009 - met['RH_frac']))
    rh_lt_97 = met['RH_frac'] < 0.97
    phi[rh_lt_97] = 1.058
    phi[~rh_lt_97] = 1.058 - ((0.0155 * (met['RH_frac'][~rh_lt_97] - 0.97))
                              /(1.02 - (met['RH_frac'][~rh_lt_97] ** 1.4)))
    alpha = 1.2 * np.exp((0.066 * met['RH_frac'])/ (phi - met['RH_frac']))

    # duplicate values across to all radii bins to help r_md = .. calculation: alpha_dup.shape = (time, rbin)
    alpha_dup = np.tile(alpha, (len(r_microns), 1)).transpose()
    beta_dup = np.tile(beta, (len(r_microns), 1)).transpose()


    # original -> r_md = alpha_factor * alpha_dup * (r_d_microns_dup ** beta_dup)
    r_d = (r_md_microns_dup/(alpha_dup * alpha_factor)) ** (1.0/beta_dup)

    # --- above rh_cap ------#

    # set all r_md(RH>99.5%) to r_md(RH=99.5%) to prevent growth rates inconsistent with impirical equation.
    # replace all r_md values above 0.995 with 0.995
    rh_gt_cap = met['RH_frac'] > rh_cap
    r_d[rh_gt_cap, :] = calc_r_d_t(r_microns, rh_cap, alpha_factor)

    # --- 0 to efflorescence --- #

    # below efflorescence point (0.3 for sulhate, r_md = r_d)
    rh_lt_eff = met['RH_frac'] <= rh_eff
    r_d[rh_lt_eff, :] = r_microns

    # ------ efflorescence to deliquescence ----------#

    # calculate r_d for the deliquescence rh - used in linear interpolation
    r_d_del = calc_r_d_t(r_microns, rh_del, alpha_factor)

    # all values that need to have some linear interpolation
    bool = np.logical_and(met['RH_frac'] >= rh_eff, met['RH_frac'] <= rh_del)
    rh_bet_eff_del = np.where(bool == True)[0]

    # between efflorescence point and deliquescence point, r_md is expected to value linearly between the two
    low_rh = rh_eff
    up_rh = rh_del
    up_r_md = r_microns
    low_r_d = r_d_del

    diff_rh = up_rh - low_rh
    diff_r_md = r_microns - r_d_del
    abs_diff_r_md = abs(diff_r_md)

    # find distance rh is along linear interpolation [fraction] from lower limit
    # frac = np.empty(len(r_md))
    # frac[:] = np.nan
    frac = ((met['RH_frac'][rh_bet_eff_del] - low_rh) / diff_rh)

    # duplicate abs_diff_r_md by the number of instances needing to be interpolated - helps the calculation below
    #   of r_md = ...low + (frac * abs diff)
    abs_diff_r_md_dup = np.tile(abs_diff_r_md, (len(rh_bet_eff_del), 1))
    frac_dup = np.tile(frac, (len(r_microns), 1)).transpose()

    # calculate interpolated values for r_md
    r_d[rh_bet_eff_del, :] = low_r_d + (frac_dup * abs_diff_r_md_dup)

    return r_d


# Optical properties

### not used now
def calculate_lidar_ratio(aer_particles, date_range, ceil_lambda, r_md_m,  n_wet, num_conc):

    """
    Calculate the lidar ratio and store all optic calculations in a single dictionary for export and pickle saving
    :param aer_particles:
    :param date_range:
    :param ceil_lambda:
    :param r_md_m:
    :param n_wet:
    :param num_conc:
    :return: optics [dict]
    """

    # Calculate Q_dry for each bin and species.
    #   The whole averaging thing up to cross sections comes later (Geisinger et al., 2016)

    # Create size parameters
    X = {}
    for aer_i in aer_particles:
        X[aer_i] = (2.0 * np.pi * r_md_m[aer_i])/ceil_lambda[0]

    # create Q_ext and Q_back arrays ready
    Q_ext = {}
    Q_back = {}

    C_ext = {}
    C_back = {}

    sigma_ext = {}
    sigma_back = {}

    print ''
    print 'Calculating extinction and backscatter efficiencies...'

    # 1) for aer_i in aerosols
    for aer_i in aer_particles:

        # if the aerosol has a number concentration above 0 (therefore sigma_aer_i > 0)
        if np.nansum(num_conc[aer_i]) != 0.0:

            # status tracking
            print '  ' + aer_i

            Q_ext[aer_i] = np.empty(r_md_m[aer_i].shape)
            Q_ext[aer_i][:] = np.nan

            Q_back[aer_i] = np.empty(r_md_m[aer_i].shape)
            Q_back[aer_i][:] = np.nan

            C_ext[aer_i] = np.empty(r_md_m[aer_i].shape)
            C_ext[aer_i][:] = np.nan

            C_back[aer_i] = np.empty(r_md_m[aer_i].shape)
            C_back[aer_i][:] = np.nan

            sigma_ext[aer_i] = np.empty(len(date_range))
            sigma_ext[aer_i][:] = np.nan

            sigma_back[aer_i] = np.empty(len(date_range))
            sigma_back[aer_i][:] = np.nan

            # 2) for time, t
            for t, time_t in enumerate(date_range):

                # status tracking
                if t in np.arange(0, 35000, 1000):
                    print '     ' + str(t)

                # 3) for radii bin, r
                for r, r_md_t_r in enumerate(r_md_m[aer_i][t, :]):

                    X_t_r = X[aer_i][t, r]  # size parameter_t (for all sizes at time t)
                    n_wet_t_r = n_wet[aer_i][t, r]  # complex index of refraction t (for all sizes at time t)


                    if np.logical_and(~np.isnan(X_t_r), ~np.isnan(n_wet_t_r)):

                        # Q_back / 4.0pi as normal .qb() is a hemispherical backscatter, and we want specifically 180 deg.
                        particle = Mie(x=X_t_r, m=n_wet_t_r)
                        Q_ext[aer_i][t, r] = particle.qext()
                        Q_back[aer_i][t, r] = particle.qb() / (4.0 * np.pi)


                        # calculate extinction cross section
                        C_ext[aer_i][t, r] = Q_ext[aer_i][t, r] * np.pi * (r_md_t_r ** 2.0)
                        C_back[aer_i][t, r] = Q_back[aer_i][t, r] * np.pi * (r_md_t_r ** 2.0)


                sigma_ext[aer_i][t] = np.nansum(num_conc[aer_i][t, :] * C_ext[aer_i][t, :])
                sigma_back[aer_i][t] = np.nansum(num_conc[aer_i][t, :] * C_back[aer_i][t, :])

    sigma_ext_tot = np.nansum(sigma_ext.values(), axis=0)
    sigma_back_tot = np.nansum(sigma_back.values(), axis=0)
    S = sigma_ext_tot / sigma_back_tot

    # store all variables in a dictionary
    optics = {'S': S, 'Q_ext':Q_ext, 'Q_back':Q_back, 'C_ext': C_ext, 'C_back': C_back,
              'sigma_ext': sigma_ext, 'sigma_back': sigma_back}

    return optics

def calculate_lidar_ratio_geisinger(aer_particles, date_range, ceil_lambda, r_md_m,  n_wet, num_conc,
                                    n_samples, r_orig_bins_m):

    """
    Calculate the lidar ratio and store all optic calculations in a single dictionary for export and pickle saving
    :param aer_particles:
    :param date_range:
    :param ceil_lambda:
    :param r_md_m:
    :param n_wet:
    :param num_conc:
    :return: optics [dict]
    """

    # Calculate Q_dry for each bin and species.
    #   The whole averaging thing up to cross sections comes later (Geisinger et al., 2016)

    # probably overkill...
    # Use Geisinger et al., (2016) (section 2.2.4) approach to calculate cross section
    #   because the the ext and backscatter components are really sensitive to variation in r (and as rbins are defined
    #   somewhat arbitrarily...


    # create Q_ext and Q_back arrays ready
    Q_ext = {}
    Q_back = {}

    C_ext = {}
    C_back = {}

    sigma_ext = {}
    sigma_back = {}

    print ''
    print 'Calculating extinction and backscatter efficiencies...'

    # Create size parameters
    X = {}
    for aer_i in aer_particles:
        X[aer_i] = (2.0 * np.pi * r_md_m[aer_i])/ceil_lambda[0]

    for aer_i in aer_particles:

         # if the aerosol has a number concentration above 0 (therefore sigma_aer_i > 0)
        if np.nansum(num_conc[aer_i]) != 0.0:

            # status tracking
            print '  ' + aer_i

            Q_ext[aer_i] = np.empty((len(date_range), len(r_orig_bins_m)))
            Q_ext[aer_i][:] = np.nan

            Q_back[aer_i] = np.empty((len(date_range), len(r_orig_bins_m)))
            Q_back[aer_i][:] = np.nan

            C_ext[aer_i] = np.empty((len(date_range), len(r_orig_bins_m)))
            C_ext[aer_i][:] = np.nan

            C_back[aer_i] = np.empty((len(date_range), len(r_orig_bins_m)))
            C_back[aer_i][:] = np.nan

            sigma_ext[aer_i] = np.empty(len(date_range))
            sigma_ext[aer_i][:] = np.nan

            sigma_back[aer_i] = np.empty(len(date_range))
            sigma_back[aer_i][:] = np.nan

            # 2) for time, t
            for t, time_t in enumerate(date_range):
            # for t, time_t in zip([135], [WXT_hourly['time'][135]]):

                # status tracking
                if t in np.arange(0, 35000, 100):
                    print '     ' + str(t)

                # for each r bin
                for r_bin_idx, r_i in enumerate(r_orig_bins_m):

                    # set up the extinction and backscatter efficiencies for this bin range
                    Q_ext_sample = np.empty(int(n_samples))
                    Q_back_sample = np.empty(int(n_samples))

                    # set up the extinction and backscatter cross sections for this bin range
                    C_ext_sample = np.empty(int(n_samples))
                    C_back_sample = np.empty(int(n_samples))

                    # get the R_dg for this range (pre-calculated as each of these needed to be swollen ahead of time)
                    # should increase in groups e.g. if n_samples = 3, [0-2 then 3-5, 6-8 etc...]
                    idx_s = r_bin_idx*int(n_samples)
                    idx_e = int((r_bin_idx*int(n_samples)) + (n_samples - 1))

                    # get the idx range for R_dg to match its location in the large R_dg array
                    # +1 to be inclusive of the last entry e.g. if n_samples = 2, idx_range = 0-2 inclusively
                    idx_range = range(idx_s, idx_e + 1)

                    # get relative idx range for C_ext_sample to be filled (should always be 0 to length of sample)
                    g_idx_range = range(int(n_samples))

                    # get swollen R_dg for this set
                    # R_dg_i_set = R_dg_m[idx_s:idx_e]
                    R_dg_wet_i_set = r_md_m[aer_i][t, idx_range]

                    # iterate over each subsample (g) to get R_dg for the bin, and calc the cross section
                    # g_idx will place it it the right spot in C_back
                    for g_idx_i, R_dg_wet_idx_i, R_dg_wet_i in zip(g_idx_range, idx_range, R_dg_wet_i_set):

                        # size parameter
                        X_t_r = X[aer_i][t, R_dg_wet_idx_i]

                        # complex index of refraction
                        n_wet_t_r = n_wet[aer_i][t, R_dg_wet_idx_i]

                        # skip instances of nan
                        if ~np.isnan(X_t_r):

                            # would need to swell wider range of particles (93 bins * subsamples)
                            particle = Mie(x=X_t_r, m=n_wet_t_r)
                            Q_ext_sample[g_idx_i]= particle.qext()
                            Q_back_sample[g_idx_i] = particle.qb() / (4.0 * np.pi)

                            # calculate the extinction and backscatter cross section for the subsample
                            #   part of Eqn 16 and 17
                            C_ext_sample[g_idx_i] = Q_ext_sample[g_idx_i] * np.pi * (R_dg_wet_i ** 2.0)
                            C_back_sample[g_idx_i] = Q_back_sample[g_idx_i] * np.pi * (R_dg_wet_i ** 2.0)

                    # once Q_back/ext for all subsamples g, have been calculated, Take the average for this main r bin
                    #   Eqn 17
                    Q_ext[aer_i][t, r_bin_idx] = (1.0 / n_samples) * np.nansum(Q_ext_sample)
                    Q_back[aer_i][t, r_bin_idx] = (1.0 / n_samples) * np.nansum(Q_back_sample)

                    # once C_back/ext for all subsamples g, have been calculated, Take the average for this main r bin
                    #   Eqn 17
                    C_ext[aer_i][t, r_bin_idx] = (1.0 / n_samples) * np.nansum(C_ext_sample)
                    C_back[aer_i][t, r_bin_idx] = (1.0 / n_samples) * np.nansum(C_back_sample)

                # calculate sigma_ext/back for this aerosol
                sigma_ext[aer_i][t] = np.nansum(num_conc[aer_i][t, :] * C_ext[aer_i][t, :])
                sigma_back[aer_i][t] = np.nansum(num_conc[aer_i][t, :] * C_back[aer_i][t, :])

    # calculate total sigma_ext/back for all aerosol, and the lidar ratio
    sigma_ext_tot = np.nansum(sigma_ext.values(), axis=0)
    sigma_back_tot = np.nansum(sigma_back.values(), axis=0)
    S = sigma_ext_tot / sigma_back_tot

    # store all variables in a dictionary
    optics = {'S': S, 'sigma_ext': sigma_ext, 'sigma_back': sigma_back}

    return optics

# saving
def pickle_optics_save(site_meta, optics, pickledir, savestr, savesub, ceil_lambda_str_nm, outputSave=False, **kwargs):

    """
    Save the calculated optical properties, given that they can easily take 3+ hours to compute
    :param site_meta:
    :param optics:
    :param datadir:
    :param savestr:
    :param savesub:
    :param ceil_lambda_str_nm:l
    :param kwargs:
    :return:
    """

    pickle_save = {'site_meta':site_meta, 'optics': optics}
    if kwargs is not None:
        pickle_save.update(kwargs)
    with open(pickledir +savestr+'_'+savesub+'_'+site_meta['period']+'_'+ceil_lambda_str_nm+'.pickle', 'wb') as handle:
        pickle.dump(pickle_save, handle)

    if outputSave == True:
        return pickle_save
    else:
        return

# def main():
if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # site information
    # site_meta = {'site_short':'Ch', 'site_long': 'Chilbolton', 'period': '2016',
    #         'instruments': ['SMPS', 'GRIMM'], 'ceil_lambda': 0.905e-06}
    
    site_meta = {'site_short':'NK', 'site_long': 'North_Kensington', 'period': 'long_term',
        'instruments': ['SMPS', 'GRIMM'], 'ceil_lambda': 0.905e-06}

    period = site_meta['period']

    # use PM1 or pm10 data?
    # process_type = 'pm10-2p5' # Ch
    process_type = 'pm10'

    # which PM vars to read in (linked to the process type)
    # pm_vars = ['pm2p5', 'pm10'] #Ch
    pm_vars = ['pm10']

    # soot or no soot? (1 = 'withSoot' or 0 = 'noSoot')
    soot_flag = 1

    if soot_flag == 1:
        soot_str = 'withSoot'
    else:
        soot_str = 'noSoot'

    # Geisinger et al., 2017 subsampling?
    Geisinger_subsample_flag = 1

    if Geisinger_subsample_flag == 1:
        Geisinger_str = 'geisingerSample'
    else:
        Geisinger_str = ''

    # number of samples to use in geisinger sampling
    n_samples = 4.0

    # wavelength to aim for (in a list! e.g. [905e-06])
    ceil_lambda = [site_meta['ceil_lambda']]
    ceil_lambda_str_nm = str(ceil_lambda[0] * 1.0e09) + 'nm'

    # string for saving figures and choosing subdirectories
    savesub = process_type+'_'+soot_str

    # directories
    maindir = '/home/nerc/Documents/MieScatt/'
    datadir = '/home/nerc/Documents/MieScatt/data/' + site_meta['site_long'] + '/'
    pickledir = '/home/nerc/Documents/MieScatt/data/pickle/' + site_meta['site_long'] + '/'

    # save dir
    if Geisinger_subsample_flag == 1:
        savesubdir = savesub
    else:
        savesubdir = savesub

    # full save directory (including sub directories)
    savedir = maindir + 'figures/LidarRatio/' + savesubdir + '/'

    # data
    #wxtdatadir = datadir
    massdatadir = datadir
    ffoc_gfdir = datadir

    # save all output data as a pickle?
    picklesave = True

    # site and instruments used to help with file saves
    savestr = site_meta['site_short'] + '_' + '_'.join(site_meta['instruments'])

    # RH data
    #wxt_inst_site = 'WXT_KSSW'

    # data years
    years = [2014, 2015, 2016]

    # resolution to average data to (in minutes! e.g. 60)
    timeRes = 60

    # aerosol particles to calculate (OC = Organic carbon, CBLK = black carbon, both already measured)
    # match dictionary keys further down
    aer_particles = ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CORG', 'CBLK']

    all_species = ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CORG', 'CBLK', 'H2O']
    # aer names in the complex index of refraction files
    aer_names = {'(NH4)2SO4': 'Ammonium sulphate', 'NH4NO3': 'Ammonium nitrate',
                'CORG': 'Organic carbon', 'NaCl': 'Generic NaCl', 'CBLK':'Soot', 'MURK': 'MURK'}

    # raw data used to make aerosols
    orig_particles = ['CORG', 'CL', 'CBLK', 'NH4', 'SO4', 'NO3']

    # density of molecules [kg m-3]
    # CBLK: # Zhang et al., (2016) Measuring the morphology and density of internally mixed black carbon
    #           with SP2 and VTDMA: New insight into the absorption enhancement of black carbon in the atmosphere
    # ORG: Range of densities for organic carbon is mass (0.625 - 2 g cm-3)
    #  Haywood et al 2003 used 1.35 g cm-3 but Schkolink et al., 2006 claim the average is 1.1 g cm-3 after a lit review
    aer_density = {'(NH4)2SO4': 1770.0,
                   'NH4NO3': 1720.0,
                   'NaCl': 2160.0,
                   'CORG': 1100.0,
                   'CBLK': 1200.0}

    # pure water density
    water_density = 1000.0 # kg m-3

    # # radii used for each aerosol species in calculating the number weights
    rn_pmlt1p0_microns, rn_pmlt1p0_m, \
    rn_pm10_microns, rn_pm10_m, \
    rn_pmlt2p5_microns, rn_pmlt2p5_m, \
    rn_2p5_10_microns, rn_2p5_10_m = fixed_radii_for_Nweights()

    # ============================================
    # Pickle read in
    # ============================================

    print 'Reading in data...'

    # # # read in any pickled S data from before
    # filename = pickledir+ 'Ch_SMPS_GRIMM_pm10-2p5_withSoot_2016_905.0nm.pickle'
    # with open(filename, 'rb') as handle:
    #     pickle_load_in = pickle.load(handle)
    #
    # optics = pickle_load_in['optics']
    # S = optics['S']
    # met = pickle_load_in['met']
    # N_weight_pm10 = pickle_load_in['N_weight']
    # pm10_mass = pickle_load_in['pm10_mass']

    for year in years:

    # ============================================
    # Read in number distribution and RH data
    # ============================================

    # Data needs to be prepared in calc.plot_N_r_obs.py on windows machine, and saved in pickle form to read it in here.
    #  D, dD, logD... dNdlogD, dN ... etc.

    if site_meta['period'] == 'ClearfLo':

        # read in clearflo winter number distribution
        # created on main PC space with calc_plot_N_r_obs.py
        # !Note: will not work if pickle was saved using protocol=Highest... (for some unknown reason)
        filename = datadir + 'dN_dmps_aps_clearfloWinter_lt60_cut.pickle'
        with open(filename, 'rb') as handle:
            dN_in = pickle.load(handle)

        # Test with the DMPS and APS data: effect in the end was very small.
        # # increase N in the first 2 bins of APS data as these are small due to the the discrepency between DMPS and APS
        # # measurements, as the first 3 APS bins are usually low and need to be corrected (Beddows et al ., 2010)
        # for b in [520, 560]:
        #     dN_in['binned'][:, dN_in['D'] == b] *= 2.0

        # make a median distribution of dN
        dN_in['med'] = np.nanmedian(dN_in['binned'], axis=0)

        # convert D and dD from nm to microns and meters separately, and keep the variables for clarity further down
        # these are the original bins from the dN data
        r_d_orig_bins_microns = dN_in['D'] * 1e-03 / 2.0
        r_d_orig_bins_m = dN_in['D'] * 1e-09 / 2.0

    if (site_meta['site_long'] == 'North_Kensington') & (site_meta['period'] == 'long_term'):

        # Read in NK long term data 2014 - 2016 inclusively



    if (site_meta['site_long'] == 'Chilbolton') & (site_meta['period'] == '2016'):

        # read in number distribution and RH from pickled data
        filename = datadir + 'N_hourly_Ch_SMPS_GRIMM.pickle'
        with open(filename, 'rb') as handle:
            dN_in = pickle.load(handle)

        # make sure datetimes are in UTC
        zone = tz.gettz('UTC')
        dN_in['time'] = np.array([i.replace(tzinfo=zone) for i in dN_in['time']])

        # convert D and dD from nm to microns and meters separately, and keep the variables for clarity further down
        # these are the original bins from the dN data
        r_orig_bins_microns = dN_in['D'] * 1e-03 / 2.0
        r_orig_bins_m = dN_in['D'] * 1e-09 / 2.0



    # interpolated r values to from the Geisinger et al 2017 approach
    # will increase the number of r bins to (number of r bins * n_samples)
    R_dg_microns, dN_in = Geisinger_increase_r_bins(dN_in, r_orig_bins_microns, n_samples=n_samples)
    R_dg_m = R_dg_microns * 1.0e-06

    # which set of radius is going to be swollen/dried?
    #   flag set True = interpolated bins
    #   flag set False = original diameter bins
    if Geisinger_subsample_flag == 1:
        r_microns = R_dg_microns
        r_m = R_dg_m
    else:
        r_microns = r_d_orig_bins_microns
        r_m = r_d_orig_bins_m

    # ==============================================================================
    # Read GF and meteorological data
    # ==============================================================================


    # read in the complex index of refraction data for the aerosol species (can include water)
    n_species = read_n_data(aer_particles, aer_names, ceil_lambda, getH2O=True)

    # Read in physical growth factors (GF) for organic carbon (assumed to be the same as aged fossil fuel OC)
    gf_ffoc = read_organic_carbon_growth_factors(ffoc_gfdir)


    # # Read WXT data
    # wxtfilepath = wxtdatadir + wxt_inst_site + '_' + year + '_15min.nc'
    # RH_in = eu.netCDF_read(wxtfilepath, vars=['RH', 'Tair','press', 'time'])
    # RH_in['RH_frac'] = WXT_in['RH'] * 0.01
    # RH_in['time'] -= dt.timedelta(minutes=15) # change time from 'obs end' to 'start of obs', same as the other datasets

    ## Read in species by mass data
    # -----------------------------------------

    if 'pm2p5' in pm_vars:

        # Read in the PM2.5 data [grams m-3]
        pm2p5_mass_in, _ = read_PM_mass_data(massdatadir, site_meta, 'PM2p5', year)


    if 'pm10' in pm_vars:

        # Read in the hourly other pm10 data [grams m-3]
        pm10_mass_in, _ = read_PM_mass_data(massdatadir, site_meta, 'PM10', year)

        # Read in the daily EC and OC data [grams m-3]
        pm10_oc_bc_in = read_EC_BC_mass_data(massdatadir, site_meta, 'PM10', year)

        # linearly interpolate daily data to hourly
        pm10_oc_bc_in = oc_bc_interp_hourly(pm10_oc_bc_in)

        # merge the pm10 data together and used RH to do it. RH and pm10 merged datasets will be in sync.
        pm10_mass_in = merge_pm_mass(pm10_mass_in, pm10_oc_bc_in)


    ## Read in meteorological data
    if (site_meta['site_long'] == 'Chilbolton') & (year == '2016'):

        # RH, Tair and pressure data was bundled with the dN data, so extract out here to make it clearly separate.
        met_in = {'time': dN_in['time'], 'RH': dN_in['RH'], 'Tair': dN_in['Tair'], 'pressure': dN_in['pressure']}

        if 'RH_frac' not in met_in:
            met_in['RH_frac'] = met_in['RH']/100.0


    # ==============================================================================
    # Time match processing
    # ==============================================================================


    # Time match and allign pm2.5, pm10, RH and dN data
    # Average up data to the same time resolution, according to timeRes
    # All values of an instance are set to nan if there is ANY missing data (e.g. if just the GRIMM was missing
    #   the PM, dN, and RH data for that time will be set to nan) -> avoid unrealistic weighting
    print 'time matching data...'
    pm2p5_mass, pm10_mass, met, dN = time_match_pm_RH_dN(pm2p5_mass_in, pm10_mass_in, met_in, dN_in, timeRes)


    # ==============================================================================
    # Main processing and calculations
    # ==============================================================================

    print 'processing data...'
    if process_type == 'pm10-2p5':

        # create pm10-2p5
        pm10m2p5_mass = two_pm_dataset_difference(pm2p5_mass, pm10_mass)


        # calculate the moles and the mass [kg kg-1] from mass [g cm-3] and met data for pm2p5, pm10-2.5 and pm10
        pm2p5_moles, pm2p5_mass_kg_kg = calculate_moles_masses(pm2p5_mass, met, aer_particles, inc_soot=soot_flag)
        pm10m2p5_moles,   pm10m2p5_mass_kg_kg = calculate_moles_masses(pm10m2p5_mass, met, aer_particles, inc_soot=soot_flag)
        pm10_moles,   pm10_mass_kg_kg = calculate_moles_masses(pm10_mass, met, aer_particles, inc_soot=soot_flag)

        # calculate N_weights for each species so the weights can be applied afterwards
        #
        #N_weight_pm2p5 = N_weights_from_pm_mass(aer_particles, pm2p5_mass_kg_kg, aer_density, met, rn_pmlt2p5_m)
        #N_weight_pm10m2p5 = N_weights_from_pm_mass(aer_particles, pm10m2p5_mass_kg_kg, aer_density, met, rn_2p5_10_m)
        N_weight_pm10 = N_weights_from_pm_mass(aer_particles, pm10_mass_kg_kg, aer_density, met, rn_pm10_m)


        # old way doing N_weight and applying the weight to get the num_conc at the same time
        # N_weight_pm2p5, num_conc_pm2p5 = est_num_conc_by_species_for_Ndist(aer_particles, pm2p5_mass_kg_kg, aer_density, met, rn_pmlt2p5_m, dN)
        # N_weight_pm10m2p5, num_conc_pm10m2p5 = est_num_conc_by_species_for_Ndist(aer_particles, pm10m2p5_mass_kg_kg, aer_density, met, rn_2p5_10_m, dN)
        # N_weight_pm10, num_conc_pm10 = est_num_conc_by_species_for_Ndist(aer_particles, pm10_mass_kg_kg, aer_density, met, rn_10_m, dN)

        # # merge the two num_conc datasets together
        # limit = 2500.0 # [nm]
        # num_conc, idx_pm2p5, idx_pm10m2p5 = merge_two_pm_dataset_num_conc(num_conc_pm2p5, num_conc_pm10m2p5, dN, limit)



    # calculate dry volume from the mass of each species
    # V_dry_from_mass = calc_dry_volume_from_mass(aer_particles, mass_kg_kg, aer_density)

    # ==============================================================================
    # Swelling / drying particles
    # ==============================================================================

    print 'swelling/drying particles...'

    # extract particle radii from each instrument, as some need swelling, others drying
    # microns
    r_md_smps_microns = r_microns[dN['smps_geisinger_idx']] # originally wet from measurements
    r_d_grimm_microns = r_microns[dN['grimm_geisinger_idx']] # originally dry from measurements

    # meters
    r_md_smps_m = r_m[dN['smps_geisinger_idx']] # originally wet from measurements
    r_d_grimm_m = r_m[dN['grimm_geisinger_idx']] # originally dry from measurements

    # duplicate 1D arrays so they can be appended onto varying radii from dry SMPS and wet GRIMM data
    r_md_smps_microns_dup = np.tile(r_md_smps_microns, (len(met['time']), 1))
    r_md_smps_m_dup = np.tile(r_md_smps_m, (len(met['time']), 1))

    r_d_grimm_microns_dup = np.tile(r_d_grimm_microns, (len(met['time']), 1))
    r_d_grimm_m_dup = np.tile(r_d_grimm_m, (len(met['time']), 1))

    # ---------------------------------------------------------
    # Swell the particle radii bins
    # r_md [microns]
    # r_md_m [meters]

    r_md_grimm_microns, r_md_grimm_m = calc_r_md_all(r_d_grimm_microns, met, pm10_mass, gf_ffoc)

    # -----------------------------------------------------------

    # Dry particles
    r_d_smps_microns, r_d_smps_m = calc_r_d_all(r_md_smps_microns, met, pm10_mass, gf_ffoc)

    # -----------------------------------------------------------

    # combine the dried SMPS to the constant GRIMM data together, then the constant wet SMPS to the wet GRIMM data.
    #   dry SMPS and wet GRIMM radii will vary by species, but the original wet SMPS and dry GRIMM wont as they are
    #   the original bins.
    #   Hence for example: r_d_microns[aer_i] = np.append(r_d_smps_microns[aer_i], r_d_grimm_microns)!
    r_d_microns = {}
    r_d_m = {}
    r_md_microns = {}
    r_md_m = {}

    for aer_i in aer_particles:
        r_d_microns[aer_i] = np.hstack((r_d_smps_microns[aer_i], r_d_grimm_microns_dup))
        r_d_m[aer_i] = np.hstack((r_d_smps_m[aer_i], r_d_grimm_m_dup))

        r_md_microns[aer_i] = np.hstack((r_md_smps_microns_dup, r_md_grimm_microns[aer_i]))
        r_md_m[aer_i] = np.hstack((r_md_smps_m_dup, r_md_grimm_m[aer_i]))

    # the swelling/drying wont work when RH was nan
    #   therefore make all bins nan when RH was nan.
    nan_idx = np.isnan(dN['RH_frac'])

    for aer_i in aer_particles:

        r_d_microns[aer_i][nan_idx] = np.nan
        r_d_m[aer_i][nan_idx] = np.nan

        r_md_microns[aer_i][nan_idx] = np.nan
        r_md_m[aer_i][nan_idx] = np.nan


    # -----------------------------------------------------------

    print 'calculating num_conc, GF and n_wet...'

    # Calculate the number concentration now that we know the dry radii
        # find relative N from N(mass, r_md)


    # Estimated number for the species, from the main distribution data, using the weighting,
    #    for each time step.
    # num_conc[aer_i].shape = (time, number of ORIGINAL bins) -
    #    not then number of bins from geisinger interpolation

    # multiply a 2D array (dN) by a 1D array N_weight_pm10[aer_i]


    num_conc = {aer_i: dN['dN'] * N_weight_pm10[aer_i][:, None] for aer_i in aer_particles}

    # -----------------------------------------------------------

    # Caulate the physical growth factor (GF) for the particles (swollen radii / dry radii)
    GF = {aer_i: r_md_microns[aer_i] / r_d_microns[aer_i] for aer_i in aer_particles}

    # --------------------------------------------------------------

    # calculate n_wet for each rbin (complex refractive index of dry aerosol and water based on physical growth)
    #   follows CLASSIC scheme parameterisation
    n_wet = {aer_i: (n_species[aer_i] / (GF[aer_i] ** 3.0)) + (n_species['H2O'] * (1 - (1/(GF[aer_i] ** 3.0))))
             for aer_i in aer_particles}

    # --------------------------

    print 'calculating optical properties...'
    # The main beast. Calculate all the optical properties, and outputs the lidar ratio
    if Geisinger_subsample_flag == 0:
        optics = calculate_lidar_ratio(aer_particles, met['time'], ceil_lambda, r_md_m,  n_wet, num_conc)
    else:
        optics = calculate_lidar_ratio_geisinger(aer_particles, met['time'], ceil_lambda, r_md_m,  n_wet, num_conc,
                                    n_samples, r_orig_bins_m)

    # extract out the lidar ratio
    S = optics['S']


    # save the output data encase it needs to be used again (pickle!)
    #   calculating the lidar ratio for 1 year can take 3-6 hours (depending on len(D))
    if picklesave == True:
        pickle_save = pickle_optics_save(site_meta, optics, pickledir, savestr, savesub, ceil_lambda_str_nm, outputSave=True, met=met, N_weight=N_weight_pm10, num_conc=num_conc, dN=dN, pm10_mass=pm10_mass,
                    ceil_lambda=ceil_lambda)



    # get mean and nanstd from data
    # set up the date range to fill (e.g. want daily statistics then stats_date_range = daily resolution)
    # stats = eu.simple_statistics(S, date_range, stats_date_range, np.nanmean, np.nanstd, np.nanmedian)
    stats_date_range = np.array(eu.date_range(met['time'][0], met['time'][-1] + dt.timedelta(days=1), 1, 'days'))

    stats ={}

    for stat_i in ['mean', 'median', 'stdev', '25pct', '75pct']:
        stats[stat_i] = np.empty(len(stats_date_range))
        stats[stat_i][:] = np.nan

    # create statistics
    for t, start, end in zip(np.arange(len(stats_date_range[:-1])), stats_date_range[:-1], stats_date_range[1:]):

        # get location of time period's data
        bool = np.logical_and(met['time'] >=start, met['time']<=end)

        # extract data
        subsample = S[bool]

        stats['mean'][t] = np.nanmean(subsample)
        stats['stdev'][t] = np.nanstd(subsample)
        stats['median'][t] = np.nanmedian(subsample)
        stats['25pct'][t] = np.percentile(subsample, 25)
        stats['75pct'][t] = np.percentile(subsample, 75)

    # TIMESERIES - S - stats
    # plot daily statistics of S
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    ax.plot_date(stats_date_range, stats['mean'], fmt='-')
    ax.fill_between(stats_date_range, stats['mean'] - stats['stdev'], stats['mean'] + stats['stdev'], alpha=0.3, facecolor='blue')
    plt.suptitle('Lidar Ratio:\n'+savesub+'masses; equal Number weighting per rbin; ClearfLo winter N(r)')
    plt.xlabel('Date [dd/mm]')
    # plt.ylim([20.0, 60.0])
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
    plt.ylabel('Lidar Ratio')
    plt.savefig(savedir + 'S_'+year+'_'+site_meta['site_short']+'_'+process_type+'_'+Geisinger_str+'_dailybinned_'+ceil_lambda_str_nm+'.png')
    # plt.savefig(savedir + 'S_'+year+'_'+process_type+'_'+Geisinger_str+'_dailybinned_lt60_'+ceil_lambda_str_nm+'.png')
    plt.close(fig)

    # HISTOGRAM - S

    idx = np.logical_or(np.isnan(S), np.isnan(met['RH']))

    # plot all the S in raw form (hist)
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    # ax.hist(S)
    ax.hist(S[~idx])
    plt.suptitle('Lidar Ratio:\n'+savesub+' masses; equal Number weighting per rbin; ClearfLo winter N(r)')
    plt.xlabel('Lidar Ratio')
    plt.ylabel('Frequency')
    plt.savefig(savedir + 'S_'+year+'_'+site_meta['site_short']+'_'+process_type+'_'+Geisinger_str+'_histogram_'+ceil_lambda_str_nm+'.png')
    plt.close(fig)

    # TIMESERIES - S - not binned
    # plot all the S in raw form (plot_date)
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    ax.plot_date(met['time'], S, fmt='-')
    plt.suptitle('Lidar Ratio:'+savesub+'\n masses; equal Number weighting per rbin; '+savestr + ' ' + site_meta['period'])
    plt.xlabel('Date [dd/mm]')
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
    plt.ylabel('Lidar Ratio [sr]')
    plt.savefig(savedir + 'S_'+year+'_'+site_meta['site_short']+'_'+process_type+'_'+Geisinger_str+'_timeseries_'+ceil_lambda_str_nm+'.png')
    plt.close(fig)

    # Pearson and Spearman correlation
    # WARNING spearman and pearson correlations give erronous values if nan values are present!!!!!
    idx1 = np.where(~np.isnan(met['RH']))
    idx2 = np.where(~np.isnan(S))
    idx = np.unique(np.append(idx1, idx2))
    corr_pearson = pearsonr(met['RH'][idx], S[idx])
    corr_spearman = spearmanr(met['RH'][idx], S[idx])

    # SCATTER - S vs RH (PM1)
    # quick plot 15 min S and RH for 2016.
    # corr = spearmanr(met['RH'], S) <- erronous value!!!! (https://github.com/scipy/scipy/issues/6530)
    # r_str = '%.2f' % corr[0]
    fig, ax = plt.subplots(1,1,figsize=(8, 4))
    scat = ax.scatter(met['RH'], S)
    scat = ax.scatter(met['RH'], S, c=N_weight_pm10['CBLK']*100.0, vmin= 0.0, vmax = 15.0)
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label('Soot [%]', labelpad=-20, y=1.1, rotation=0)
    #   cbar.set_label('Soot [%]', labelpad=-20, y=1.075, rotation=0)
    # plt.suptitle('Lidar Ratio with soot fraction - r='+r_str+':\n'+savesub+' masses; equal Number weighting per rbin; ClearfLo winter N(r)')
    plt.xlabel(r'$RH \/[\%]$')
    plt.ylabel(r'$Lidar Ratio \/[sr]$')
    plt.ylim([10.0, 80.0])
    plt.xlim([20.0, 110.0])
    plt.tight_layout()
    plt.savefig(savedir + 'S_vs_RH_'+year+'_'+site_meta['site_short']+'_'+process_type+'_'+Geisinger_str+'_scatter_'+ceil_lambda_str_nm+'.png')
    plt.close(fig)

    # ------------------------------------------------

    # LINE - wet and dry diameters
    colours = ['red', 'blue', 'green', 'black', 'purple']
    for i, aer_i in enumerate(aer_particles):
        plt.semilogy(np.nanmean(r_d_m[aer_i], axis=0), color=colours[i], linestyle = '-', label=aer_i + ' dry')
        plt.semilogy(np.nanmean(r_md_m[aer_i], axis=0), color=colours[i], linestyle='--', label=aer_i + ' wet')
    plt.legend(loc='best')


    # ------------------------------------------------

    # BOX PLOT - S binned by RH, then by soot

    ## 1. set up bins to divide the data [%]
    rh_bin_starts = np.array([0.0, 60.0, 70.0, 80.0, 90.0])
    rh_bin_ends = np.append(rh_bin_starts[1:], 100.0)

    # set up limit for soot last bin to be inf [fraction]
    soot_starts = np.array([0.0, 0.04, 0.08])
    soot_ends = np.append(soot_starts[1:], np.inf)
    soot_bins_num = len(soot_starts)

    # variables to help plot the legend
    soot_starts_str = [str(int(i*100.0)) for i in soot_starts]
    soot_ends_str = [str(int(i*100.0)) for i in soot_ends[:-1]] + ['100']
    soot_legend_str = [i+'-'+j+' %' for i, j in zip(soot_starts_str, soot_ends_str)]
    soot_colours = ['blue', 'orange', 'red']

    # positions for each boxplot (1/6, 3/6, 5/6 into each bin, given 3 soot groups)
    #   and widths for each boxplot
    pos = []
    widths = []
    mid = []

    for i, (rh_s, rh_e) in enumerate(zip(rh_bin_starts, rh_bin_ends)):

        bin_6th = (rh_e-rh_s) * 1.0/6.0 # 1/6th of the current bin width
        pos += [[rh_s + bin_6th, rh_s +(3*bin_6th), rh_s+(5*bin_6th)]] #1/6, 3/6, 5/6 into each bin for the soot boxplots
        widths += [bin_6th]
        mid += [rh_s +(3*bin_6th)]


    # Split the data - keep them in lists to preserve the order when plotting
    # bin_range_str will match each set of lists in rh_binned
    rh_split = {'binned': [], 'mean': [], 'n': [], 'bin_range_str': [], 'pos': []}

    for i, (rh_s, rh_e) in enumerate(zip(rh_bin_starts, rh_bin_ends)):

        # bin range
        rh_split['bin_range_str'] += [str(int(rh_s)) + '-' + str(int(rh_e))]

        # the list of lists for this RH bin (the actual data, the mean and sample number)
        rh_bin_i = []
        rh_bin_mean_i = []
        rh_bin_n_i = []

        # extract out all S values that occured for this RH range and their corresponding CBLK weights
        rh_bool = np.logical_and(met['RH'] >= rh_s, met['RH'] < rh_e)
        S_rh_i = S[rh_bool]
        N_weight_cblk_rh_i = N_weight_pm10['CBLK'][rh_bool]

        # idx of binned data
        for soot_s, soot_e in zip(soot_starts, soot_ends):

            # booleon for the soot data, for this rh subsample
            soot_bool = np.logical_and(N_weight_cblk_rh_i >= soot_s, N_weight_cblk_rh_i < soot_e)
            S_rh_i_soot_j = S_rh_i[soot_bool] # soot subsample from the rh subsample


            # store the values for this bin
            rh_bin_i += [S_rh_i_soot_j] # the of subsample
            rh_bin_mean_i += [np.mean(S_rh_i_soot_j)] # mean of of subsample
            rh_bin_n_i += [len(S_rh_i_soot_j)] # number of subsample

        # add each set of rh_bins onto the full set of rh_bins
        rh_split['binned'] += [rh_bin_i]
        rh_split['mean'] += [rh_bin_mean_i]
        rh_split['n'] += [rh_bin_n_i]


    ## 2. Start the boxplots
    # whis=[10, 90] wont work if the q1 or q3 extend beyond the whiskers... (the one bin with n=3...)
    fig = plt.figure(figsize=(7, 3.5))
    # fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    # plt.hold(True)
    for j, (rh_bin_j, bin_range_str_j) in enumerate(zip(rh_split['binned'], rh_split['bin_range_str'])):

        bp = plt.boxplot(list(rh_bin_j), widths=widths[j], positions=pos[j], sym='x')

        # colour the boxplots
        for c, colour_c in enumerate(soot_colours):

            # some parts of the boxplots are in two parts (e.g. 2 caps for each boxplot) therefore make an x_idx
            #   for each pair
            c_pair_idx = range(2*c, (2*c)+(len(soot_colours)-1))

            plt.setp(bp['boxes'][c], color=colour_c)
            plt.setp(bp['medians'][c], color=colour_c)
            [plt.setp(bp['caps'][i], color=colour_c) for i in c_pair_idx]
            [plt.setp(bp['whiskers'][i], color=colour_c) for i in c_pair_idx]
            #[plt.setp(bp['fliers'][i], color=colour_c) for i in c_pair_idx]

    print 'test'
    # add sample number at the top of each box
    (y_min, y_max) = ax.get_ylim()
    upperLabels = [str(np.round(n, 2)) for n in np.hstack(rh_split['n'])]
    for tick in range(len(np.hstack(pos))):
        k = tick % 3
        ax.text(np.hstack(pos)[tick], y_max - (y_max * (0.05)*(k+1)), upperLabels[tick],
                 horizontalalignment='center', size='x-small')

    ## 3. Prettify boxplot (legend, vertical lines, sample size at top)
    # prettify
    ax.set_xlim([0.0, 100.0])
    ax.set_xticks(mid)
    ax.set_xticklabels(rh_split['bin_range_str'])
    ax.set_ylabel(r'$S \/[sr]$')
    ax.set_xlabel(r'$RH \/[\%]$')


    # add vertical dashed lines to split the groups up
    (y_min, y_max) = ax.get_ylim()
    for rh_e in rh_bin_ends:
        plt.vlines(rh_e, y_min, y_max, alpha=0.3, color='grey', linestyle='--')

    # draw temporary lines to create a legend
    lin=[]
    for c, colour_c in enumerate(soot_colours):
        lin_i, = plt.plot([np.nanmean(S),np.nanmean(S)],color=colour_c) # plot line with matching colour
        lin += [lin_i] # keep the line handle in a list for the legend plotting
    plt.legend(lin, soot_legend_str, fontsize=10, loc=(0.02,0.68))
    [i.set_visible(False) for i in lin] # set the line to be invisible

    plt.tight_layout()

    ## 4. Save fig as unique image
    i = 1
    savepath = savedir + 'S_vs_RH_binnedSoot_'+period+'_'+savestr+'_boxplot_'+ceil_lambda_str_nm+'_'+str(i)+'.png'
    while os.path.exists(savepath) == True:
        i += 1
        savepath = savedir + 'S_vs_RH_binnedSoot_'+period+'_'+savestr+'_boxplot_'+ceil_lambda_str_nm+'_'+str(i)+'.png'

    plt.savefig(savepath)



    print 'END PROGRAM'