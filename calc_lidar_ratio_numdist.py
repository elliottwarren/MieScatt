"""
Read in mass data from London and calculate the Lidar ratio using the CLASSIC scheme approach, from Ben's help.

Created by Elliott Fri 17 Nov 2017
Taken from calc_lidar_ratio.py on 29 Nov 2017
"""

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pickle
from copy import deepcopy

import numpy as np
import datetime as dt
from scipy.stats import spearmanr

import ellUtils as eu
from mie_sens_mult_aerosol import linear_interpolate_n
from pymiecoated import Mie


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

        # store bool if it is one of the major pm consituents, so OM10 and OC/BC PM10 data can be removed too
        if header_i in ['NH4', 'NO3', 'SO4', 'CORG', 'CL', 'CBLK']:
            qaqc_idx[header_i] = np.where(bools == True)[0]


        # turn all values in the row negative
        for header_j in headers:
            mass[header_j][bools] = np.nan

    # find unique instances of missing data
    qaqc_idx_unique = np.unique(np.hstack(qaqc_idx.values()))


    return mass, qaqc_idx_unique

def read_EC_BC_mass_data(massdatadir, year):

    """
    Read in the elemental carbon (EC) and organic carbon (OC) mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :return: mass

    EC and BC (soot) are treated the same in CLASSIC, therefore EC will be taken as BC here.
    """

    # make sure year is string
    year = str(year)

    massfname = 'OC_EC_Daily_North_Kensington_DEFRA_'+year+'.csv'
    massfilepath = massdatadir + massfname
    massrawData = np.genfromtxt(massfilepath, delimiter=',', skip_header=4, dtype="|S20") # includes the header

    mass = {'time': np.array([dt.datetime.strptime(i[0], '%d/%m/%Y') for i in massrawData[1:]]),
            'CBLK': np.array([np.nan if i[2] == 'No data' else i[2] for i in massrawData[1:]], dtype=float),
            'CORG': np.array([np.nan if i[4] == 'No data' else i[4] for i in massrawData[1:]], dtype=float)}

    # times were valid from 12:00 so add the extra 12 hours on
    # but they were also hour ending, and as all other data is hour beginning, then take 1 off (so +11 in total)
    mass['time'] += dt.timedelta(hours=11)

    # convert units from micrograms to grams
    mass['CBLK'] *= 1e-06
    mass['CORG'] *= 1e-06

    # QAQC - turn all negative values in each column into nans if one of them is negative
    for aer_i in ['CBLK', 'CORG']:
        idx = np.where(mass[aer_i] < 0.0)
        mass[aer_i][idx] = np.nan


    return mass

def read_PM10_mass_data(massdatadir, year):

    """
    Read in the other PM10 mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :return: mass

    """

    # make sure year is string
    year = str(year)

    massfname = 'PM10species_Hr_North_Kensington_DEFRA_'+year+'.csv'
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

# fake number distribution

def normpdf(x, mean, sd):

    """
    Gives the probability of y, given a value of x, assuming a gaussian distribution
    :param x:
    :param mean:
    :param sd:
    :return:

    units of p(y) is a fraction
    """

    from math import exp

    var = float(sd) ** 2.0
    pi = 3.1415926
    denom = (2.0 * pi * var) ** 0.5
    num = exp(-(float(x) - float(mean)) ** 2.0 / (2.0 * var))

    return num / denom


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
    """

    # get bin edges based on the current bins (R_da is lower, R_db is upper edge)
    R_db = (dN['D'] + (0.5 * dN['dD'])) / 2.0 # upper
    R_da = (dN['D'] - (0.5 * dN['dD'])) / 2.0 # lower

    # create the radii values in between the edges (evenly spaced within each bin)
    R_dg = np.array([(g * ((R_db[i] - R_da[i])/n_samples)) + R_da[i]
                     for i in range(len(r_d_orig_bins_microns))
                     for g in range(1,int(n_samples)+1)])

    # append the last upper edge, as the calculation misses that one off
    # R_dg = np.append(R_dg, R_db[-1])

    # convert to microns from nanometers
    R_dg_microns = R_dg * 1e-3

    return R_dg_microns

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

def merge_timematch_PM10_mass_RH(WXT_hourly, PM10_mass_in, OC_BC_hourly):

    """
    Merge (including time matching) the PM10 masses together (OC and BC in with the others).
    RH used in time matching but is not merged with the PM10 data

    :param WXT_hourly:
    :param PM10_mass_in:
    :param OC_BC_hourly:
    :return:
    """

    # double check that OC_EC['time'] is an array, not a list, as lists do not work with the np.where() function below.
    if type(OC_BC_hourly['time']) == list:
        OC_BC_hourly['time'] = np.array(OC_BC_hourly['time'])

    # merge PM10 data (time will match WXT_hourly)
    PM10_mass_all = {'time': WXT_hourly['time']}
    for key, data in PM10_mass_in.iteritems():
        if key != 'time':
           PM10_mass_all[key] = np.empty(len(PM10_mass_all['time']))
           PM10_mass_all[key][:] = np.nan

    for key in OC_BC_hourly.iterkeys():
        if key != 'time':
           PM10_mass_all[key] = np.empty(len(PM10_mass_all['time']))
           PM10_mass_all[key][:] = np.nan

    # fill the PM10_merge arrays
    for t, time_t in enumerate(PM10_mass_all['time']):
        # _, idx_pm10, _ = eu.nearest(PM10_mass_in['time'], time_t)
        idx_pm10 = np.where(PM10_mass_in['time'] == time_t)

        if idx_pm10[0].size != 0:
            for key, data in PM10_mass_in.iteritems():
                if key != 'time':
                    PM10_mass_all[key][t] = data[idx_pm10]

        idx_oc_bc = np.where(np.array(OC_BC_hourly['time']) == time_t)
        if idx_oc_bc[0].size != 0:
            for key, data in OC_BC_hourly.iteritems():
                if key != 'time':
                    PM10_mass_all[key][t] = data[idx_oc_bc]


    # set mass and WXT data as the hourly data values
    mass = PM10_mass_all

    return mass

def coarsen_PM1_mass_hourly(WXT_hourly, PM1_mass):

    """
    Coarsen the PM1 data from 15 mins to hourly data to match PM10 and WXT hourly data
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

## masses and moles

### main masses and moles script
def calculate_moles_masses(mass, WXT, aer_particles, inc_soot=False):

    """
    Calculate the moles and mass [kg kg-1] of the aerosol. Can set soot to on or off (turn all soot to np.nan)
    :param mass: [g cm-3]
    :param WXT:
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
    mass_kg_kg, WXT = convert_mass_to_kg_kg(mass, WXT, aer_particles)


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

def convert_mass_to_kg_kg(mass, WXT, aer_particles):

    """
    Convert mass molecules from g m-3 to kg kg-1

    :param mass
    :param WXT (for meteorological data)
    :param aer_particles (not all the keys in mass are the species, therefore only convert the species defined above)
    :return: mass_kg_kg: mass in kg kg-1 air
    """

    # convert temperature to Kelvin
    T_K = WXT['Tair'] + 273.15
    p_Pa = WXT['press'] * 100.0

    # density of air [kg m-3] # assumes dry air atm
    # p = rho * R * T [K]
    WXT['dryair_rho'] = p_Pa / (286.9 * T_K)

    # convert g m-3 air to kg kg-1 of air
    mass_kg_kg = {'time': mass['time']}
    for aer_i in aer_particles:
        mass_kg_kg[aer_i] = mass[aer_i] * 1e3 / WXT['dryair_rho']

    return mass_kg_kg, WXT

def OC_BC_interp_hourly(OC_BC_in):

    """
    Increase EC and OC data resolution from daily to hourly with a simple linear interpolation

    :param OC_BC_in:
    :return:OC_BC_hourly
    """

    # Increase to hourly resolution by linearly interpolate between the measurement times
    date_range = eu.date_range(OC_BC_in['time'][0], OC_BC_in['time'][-1] + dt.timedelta(days=1), 60, 'minutes')
    OC_BC_hourly = {'time': date_range,
                    'CORG': np.empty(len(date_range)),
                    'CBLK': np.empty(len(date_range))}

    OC_BC_hourly['CORG'][:] = np.nan
    OC_BC_hourly['CBLK'][:] = np.nan

    # fill hourly data
    for aer_i in ['CORG', 'CBLK']:

        # for each day, spready data out into the hourly slots
        # do not include the last time, as it cannot be used as a start time (fullday)
        for t, fullday in enumerate(OC_BC_in['time'][:-1]):

            # start = fullday
            # end = fullday + dt.timedelta(days=1)

            # linearly interpolate between the two main dates
            interp = np.linspace(OC_BC_in[aer_i][t], OC_BC_in[aer_i][t+1],24)

            # idx range as the datetimes are internally complete, therefore a number sequence can be used without np.where()
            idx = np.arange(((t+1)*24)-24, (t+1)*24)

            # put the values in
            OC_BC_hourly[aer_i][idx] = interp


    return OC_BC_hourly

## aerosol physical properties besides mass

def est_num_conc_by_species_for_Ndist(aer_particles, mass_kg_kg, aer_density, WXT, radius_k, dN):

    """

    :param aer_particles:
    :param mass_kg_kg:
    :param aer_density:
    :param radius_k: dictionary with a float value for each aerosol (aer_i)
    :return:
    """

    # work out Number concentration (relative weight) for each species [m-3]
    # calculate the number of particles for each species using radius_m and the mass
    num_part = {}
    for aer_i in aer_particles:
        num_part[aer_i] = mass_kg_kg[aer_i] / ((4.0/3.0) * np.pi * (aer_density[aer_i]/WXT['dryair_rho']) * (radius_k[aer_i] ** 3.0))

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

## swelling

def calc_r_md_species(r_d_microns, WXT, aer_i):

    """
    Calculate the r_md [microns] for all particles, given the RH [fraction] and what species

    :param r_d_microns:
    :param WXT: (needed for RH)
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
    r_d_microns_dup = np.tile(r_d_microns, (len(WXT['time']), 1))

    # Set up array for aerosol
    r_md =  np.empty(len(WXT['time']))
    r_md[:] = np.nan

    phi = np.empty(len(WXT['time']))
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

    beta = np.exp((0.00077 * WXT['RH_frac'])/(1.009 - WXT['RH_frac']))
    rh_lt_97 = WXT['RH_frac'] < 0.97
    phi[rh_lt_97] = 1.058
    phi[~rh_lt_97] = 1.058 - ((0.0155 * (WXT['RH_frac'][~rh_lt_97] - 0.97))
                              /(1.02 - (WXT['RH_frac'][~rh_lt_97] ** 1.4)))
    alpha = 1.2 * np.exp((0.066 * WXT['RH_frac'])/ (phi - WXT['RH_frac']))

    # duplicate values across to all radii bins to help r_md = .. calculation: alpha_dup.shape = (time, rbin)
    alpha_dup = np.tile(alpha, (len(r_d_microns), 1)).transpose()
    beta_dup = np.tile(beta, (len(r_d_microns), 1)).transpose()

    r_md = alpha_factor * alpha_dup * (r_d_microns_dup ** beta_dup)

    # --- above rh_cap ------#

    # set all r_md(RH>99.5%) to r_md(RH=99.5%) to prevent growth rates inconsistent with impirical equation.
    # replace all r_md values above 0.995 with 0.995
    rh_gt_cap = WXT['RH_frac'] > rh_cap
    r_md[rh_gt_cap, :] = calc_r_md_t(r_d_microns, rh_cap, alpha_factor)

    # --- 0 to efflorescence --- #

    # below efflorescence point (0.3 for sulhate, r_md = r_d)
    rh_lt_eff = WXT['RH_frac'] <= rh_eff
    r_md[rh_lt_eff, :] = r_d_microns

    # ------ efflorescence to deliquescence ----------#

    # calculate r_md for the deliquescence rh - used in linear interpolation
    r_md_del = calc_r_md_t(r_d_microns, rh_del, alpha_factor)

    # all values that need to have some linear interpolation
    bool = np.logical_and(WXT['RH_frac'] >= rh_eff, WXT['RH_frac'] <= rh_del)
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
    frac = ((WXT['RH_frac'][rh_bet_eff_del] - low_rh) / diff_rh)

    # duplicate abs_diff_r_md by the number of instances needing to be interpolated - helps the calculation below
    #   of r_md = ...low + (frac * abs diff)
    abs_diff_r_md_dup = np.tile(abs_diff_r_md, (len(rh_bet_eff_del), 1))
    frac_dup = np.tile(frac, (len(r_d_microns), 1)).transpose()

    # calculate interpolated values for r_md
    r_md[rh_bet_eff_del, :] = low_r_md + (frac_dup * abs_diff_r_md_dup)

    return r_md

# Optical properties

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
                                    n_samples, r_d_orig_bins_m):

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

            Q_ext[aer_i] = np.empty((len(date_range), len(r_d_orig_bins_m)))
            Q_ext[aer_i][:] = np.nan

            Q_back[aer_i] = np.empty((len(date_range), len(r_d_orig_bins_m)))
            Q_back[aer_i][:] = np.nan

            C_ext[aer_i] = np.empty((len(date_range), len(r_d_orig_bins_m)))
            C_ext[aer_i][:] = np.nan

            C_back[aer_i] = np.empty((len(date_range), len(r_d_orig_bins_m)))
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
                for r_bin_idx, r_i in enumerate(r_d_orig_bins_m):

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
    optics = {'S': S, 'C_ext': C_ext, 'C_back': C_back,
              'sigma_ext': sigma_ext, 'sigma_back': sigma_back}

    return optics

def main():


    # Read in the mass data for 2016
    # Read in RH data for 2016
    # convert gases and such into the aerosol particles
    # swell the particles based on the CLASSIC scheme stuff
    # use Mie code to calculate the backscatter and extinction
    # calculate lidar ratio
    # plot lidar ratio

    # ==============================================================================
    # Setup
    # ==============================================================================

    # use PM1 or PM10 data?
    process_type = 'PM10-1'

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

    # wavelength to aim for
    ceil_lambda = [0.905e-06]
    # ceil_lambda = [0.532e-06]
    ceil_lambda_str_nm = str(ceil_lambda[0] * 1.0e09) + 'nm'

    # string for saving figures and choosing subdirectories
    savesub = process_type+'_'+soot_str

    # directories
    maindir = '/home/nerc/Documents/MieScatt/'
    datadir = '/home/nerc/Documents/MieScatt/data/'

    # save dir
    if Geisinger_subsample_flag == 1:
        savesubdir = savesub + '/geisingersample_2xdN_for_2lowestAPSbins/'
    else:
        savesubdir = savesub


    savedir = maindir + 'figures/LidarRatio/' + savesubdir

    # data
    wxtdatadir = datadir
    massdatadir = datadir
    ffoc_gfdir = datadir

    # save all output data as a pickle?
    picklesave = True

    # RH data
    wxt_inst_site = 'WXT_KSSW'

    # data year
    year = '2016'

    # site information
    site_ins = {'site_short':'Ch', 'site_long': 'Chilbolton', 'period': 'routine',
            'DMPS': False, 'APS': False, 'SMPS': True, 'GRIMM': True}


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

    # radii used for each aerosol species in calculating the number weights

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
    pm1t10_rv_microns = 1.9848902137534531 / 2.0

    # turn units to meters and place an entry for each aerosol
    pm1t10_rv_m = {}
    for key in rn_pmlt1p0_m.iterkeys():
        pm1t10_rv_m[key] = pm1t10_rv_microns * 1.0e-6

    # 3. D < 2.5 microns
    rn_pmlt2p5_microns = 0.06752 / 2.0

    rn_pmlt2p5_m = {}
    for key in rn_pmlt1p0_m.iterkeys():
        rn_pmlt2p5_m[key] = rn_pmlt2p5_microns * 1.0e-6

    # 4. 2.5 < D < 10 microns
    rn_2p5_10_microns = 2.820 / 2.0

    rn_2p5_10_m = {}
    for key in rn_pmlt1p0_m.iterkeys():
        rn_2p5_10_m[key] = rn_2p5_10_microns * 1.0e-6

    # ==============================================================================
    # Read data
    # ==============================================================================

    # read in the complex index of refraction data for the aerosol species (can include water)
    n_species = read_n_data(aer_particles, aer_names, ceil_lambda, getH2O=True)

    # Read in physical growth factors (GF) for organic carbon (assumed to be the same as aged fossil fuel OC)
    gf_ffoc_raw = eu.csv_read(ffoc_gfdir + 'GF_fossilFuelOC_calcS.csv')
    gf_ffoc_raw = np.array(gf_ffoc_raw)[1:, :] # skip header
    gf_ffoc = {'RH_frac': np.array(gf_ffoc_raw[:,0], dtype=float),
                'GF': np.array(gf_ffoc_raw[:, 1], dtype=float)}

    # Read WXT data
    wxtfilepath = wxtdatadir + wxt_inst_site + '_' + year + '_15min.nc'
    WXT_in = eu.netCDF_read(wxtfilepath, vars=['RH', 'Tair','press', 'time'])
    WXT_in['RH_frac'] = WXT_in['RH'] * 0.01
    WXT_in['time'] -= dt.timedelta(minutes=15) # change time from 'obs end' to 'start of obs', same as the other datasets

    # # load in S data
    # filename = datadir + 'pickle/' + 'S_woSoot_2016_equalWeight_lt60_PM10.pickle'
    # with open(filename, 'rb') as handle:
    #     S_loaded = pickle.load(handle)

    if site_ins['period'] == 'ClearfLo':

        # read in clearflo winter number distribution
        # created on main PC space with calc_plot_N_r_obs.py
        # !Note: will not work if pickle was saved using protocol=Highest... (for some unknown reason)
        filename = datadir + 'dN_dmps_aps_clearfloWinter_lt60_cut.pickle'
        with open(filename, 'rb') as handle:
            dN = pickle.load(handle)

        # increase N in the first 2 bins of APS data as these are small due to the the discrepency between DMPS and APS
        # measurements, as the first 3 APS bins are usually low and need to be corrected (Beddows et al ., 2010)
        for b in [520, 560]:
            dN['binned'][:, dN['D'] == b] *= 2.0

        # make a median distribution of dN
        dN['med'] = np.nanmedian(dN['binned'], axis=0)

        # convert D and dD from nm to microns and meters separately, and keep the variables for clarity further down
        # these are the original bins from the dN data
        r_d_orig_bins_microns = dN['D'] * 1e-03 / 2.0
        r_d_orig_bins_m = dN['D'] * 1e-09 / 2.0


    # interpolated r values to from the Geisinger et al 2017 approach
    # will increase the number of r bins to (number of r bins * n_samples)
    R_dg_microns = Geisinger_increase_r_bins(dN, r_d_orig_bins_microns, n_samples=n_samples)
    R_dg_m = R_dg_microns * 1.0e-06

    # which set of radius is going to be swollen?
    if Geisinger_subsample_flag == 1:
        r_d_microns = R_dg_microns
        r_d_m = R_dg_m
    else:
        r_d_microns = r_d_orig_bins_microns
        r_d_m = r_d_orig_bins_m

    # ---- pasted up to here ---- #

    # Read in species by mass data
    if (process_type == 'PM1') | (process_type == 'PM10-1'):

        # Read in mass data [grams m-3]
        # keep idx for rows that had missing data of the major species
        PM1_mass_in, pm1_qaqc_idx = read_PM1_mass_data(massdatadir, year)

        # Trim times
        # as WXT and mass data are 15 mins and both line up exactly already
        #   therefore trim WXT to match mass time
        PM1_mass_in, WXT_in = trim_mass_wxt_times(PM1_mass_in, WXT_in)

        # Time match so mass and WXT times line up INTERNALLY as well
        date_range = np.array(eu.date_range(WXT_in['time'][0], WXT_in['time'][-1], 15, 'minutes'))

        # ToDo - these two are very time consuming, make it more efficient
        # make sure there are no time stamp gaps in the data so mass and WXT will match up perfectly, timewise.
        print 'beginning time matching for WXT...'
        WXT_15min = internal_time_completion(WXT_in, date_range)
        print 'end time matching for WXT...'

        # same but for mass data
        print 'beginning time matching for mass...'
        PM1_mass = internal_time_completion(PM1_mass_in, date_range)
        print 'end time matching for mass...'

        WXT = WXT_15min

    if (process_type == 'PM10') | (process_type == 'PM10-1'):

        # Read in the daily EC and OC data [grams m-3]
        OC_BC_in = read_EC_BC_mass_data(massdatadir, year)

        # linearly interpolate daily data to hourly
        OC_BC_hourly = OC_BC_interp_hourly(OC_BC_in)

        # Read in the hourly other PM10 data [grams m-3]
        PM10_mass_in = read_PM10_mass_data(massdatadir, year)

        # average WXT data up to hourly
        WXT_hourly = WXT_hourly_average(WXT_in)

        # merge the PM10 data together and used RH to do it. RH and PM10 merged datasets will be in sync.
        PM10_mass = merge_timematch_PM10_mass_RH(WXT_hourly, PM10_mass_in, OC_BC_hourly)

        WXT = WXT_hourly

    if process_type == 'PM10-1':

        # coarsen PM1 data from 15 min to hourly
        PM1_mass_hourly = coarsen_PM1_mass_hourly(WXT_hourly, PM1_mass)

        # create pm10-1 mass
        PM10m1_mass_hourly = {'time': PM1_mass_hourly['time']}
        for key in PM1_mass_hourly.iterkeys():
            if key != 'time':
                PM10m1_mass_hourly[key] = PM10_mass[key] - PM1_mass_hourly[key]

                # PM10 - PM1 is not always >0, as the two instruments are different.
                #   Therefore np.nan all negative masses!
                idx = np.where(PM10m1_mass_hourly[key] < 0)
                PM10m1_mass_hourly[key][idx] = np.nan

        # ToDo make this more efficient...
        # data: PM10m1_mass_hourly, PM1_mass_hourly (will be raw input, no aerosols just yet!)
        # remove all aerosol values for rows that have any nans
        for pm10_set in [PM10m1_mass_hourly, PM1_mass_hourly, PM10_mass]:
            for part_i in orig_particles:
                bools = np.isnan(pm10_set[part_i])

                # turn all the part_i variables at the same time into nans
                for key in orig_particles:
                    PM10m1_mass_hourly[key][bools] = np.nan
                    PM1_mass_hourly[key][bools] = np.nan
                    PM10_mass[key][bools] = np.nan



    # ==============================================================================
    # Process data
    # ==============================================================================

    if process_type == 'PM1':
        # calculate the moles and the mass [kg kg-1] from mass [g cm-3] and WXT data
        moles, mass_kg_kg = calculate_moles_masses(PM1_mass, WXT_15min, aer_particles, inc_soot=soot_flag)

        # work out Number concentration (relative weight) for each species
        # calculate the number of particles for each species using radius_m and the mass
        N_weight, num_conc = est_num_conc_by_species_for_Ndist(aer_particles, mass_kg_kg, aer_density, WXT_15min, rn_pmlt1p0_m, dN)

    elif process_type == 'PM10':

        moles, mass_kg_kg = calculate_moles_masses(PM10_mass, WXT_hourly, aer_particles, inc_soot=soot_flag)
        N_weight, num_conc = est_num_conc_by_species_for_Ndist(aer_particles, mass_kg_kg, aer_density, WXT_hourly, rn_pmlt1p0_m, dN)

    elif process_type == 'PM10-1':


        # PM1 first
        moles, pm1_mass_kg_kg = calculate_moles_masses(PM1_mass_hourly, WXT_hourly, aer_particles, inc_soot=soot_flag)
        N_weight_pm1, num_conc_pm1 = est_num_conc_by_species_for_Ndist(aer_particles, pm1_mass_kg_kg, aer_density, WXT_hourly, rn_pmlt1p0_m, dN)


        # PM10-1 second
        moles, pm10m1_mass_kg_kg = calculate_moles_masses(PM10m1_mass_hourly, WXT_hourly, aer_particles, inc_soot=soot_flag)
        N_weight_pm10m1, num_conc_pm10m1 = est_num_conc_by_species_for_Ndist(aer_particles, pm10m1_mass_kg_kg, aer_density, WXT_hourly, pm1t10_rv_m, dN)

        # PM10 - PM1 (use the right parts of the num concentration for the rbins e.g. pm1 mass for r<1, pm10-1 for r>1)
        idx_pm1 = np.where(dN['D'] <= 1000.0)[0] # 'D' is in nm not microns!
        idx_pm10m1 = np.where(dN['D'] > 1000.0)[0]

        # concatonate num_conc
        # r<=1 micron are weighted by PM1, r>1 are weighted by PM10-1
        num_conc = {}
        for aer_i in num_conc_pm1.iterkeys():
            num_conc[aer_i] = np.hstack((num_conc_pm1[aer_i][:, idx_pm1], num_conc_pm10m1[aer_i][:, idx_pm10m1]))


    # calculate dry volume from the mass of each species
    # V_dry_from_mass = calc_dry_volume_from_mass(aer_particles, mass_kg_kg, aer_density)

    # ---------------------------------------------------------
    # Swell the particles (r_md,aer_i) [microns]

    # set up dictionary
    r_md = {}

    # create a r_d_microns_dry_dup (rbins copied for each time, t) to help with calculations
    r_d_microns_dup = np.tile(r_d_microns, (len(WXT['time']), 1))

    # calculate the swollen particle size for these three aerosol types
    # Follows CLASSIC guidence, based off of Fitzgerald (1975)
    # guidance requires radii units to be microns
    for aer_i in ['(NH4)2SO4', 'NH4NO3', 'NaCl']:
        r_md[aer_i] = calc_r_md_species(r_d_microns, WXT, aer_i)

    # set r_md for black carbon as r_d, assuming black carbon is completely hydrophobic
    # r_md['CBLK'] = np.empty((len(date_range), len(r_d_microns)))
    r_md['CBLK'] = r_d_microns_dup

    # make r_md['CBLK'] nan for all sizes, for times t, if mass data is not present for time t
    # doesn't matter which mass is used, as all mass data have been corrected for if nans were present in other datasets
    r_md['CBLK'][np.isnan(PM10m1_mass_hourly['CBLK']), :] = np.nan

    # calculate r_md for organic carbon using the MO empirically fitted g(RH) curves
    r_md['CORG'] = np.empty((len(WXT['time']), len(r_d_microns)))
    r_md['CORG'][:] = np.nan
    for t, time_t in enumerate(WXT['time']):

        _, idx, _ = eu.nearest(gf_ffoc['RH_frac'], WXT['RH_frac'][t])
        r_md['CORG'][t, :] = r_d_microns * gf_ffoc['GF'][idx]

    # convert r_md units from microns to meters
    r_md_m = {}
    for aer_i in aer_particles:
        r_md_m[aer_i] = r_md[aer_i] * 1e-06

    # -----------------------------------------------------------

    # caulate the physical growth factor for the particles (swollen radii / dry radii)
    GF = {}
    for aer_i in aer_particles: # aer_particles:

        # physical growth factor
        GF[aer_i] = r_md[aer_i] / r_d_microns_dup


    # --------------------------------------------------------------

    # calculate n_wet for each rbin (complex refractive index of dry aerosol and water based on physical growth)
    #   follows CLASSIC scheme
    n_wet = {}

    for aer_i in aer_particles: # aer_particles:

        # physical growth factor
        n_wet[aer_i] = (n_species[aer_i] / (GF[aer_i] ** 3.0)) + (n_species['H2O'] * (1 - (1/(GF[aer_i] ** 3.0))))

    # --------------------------

    if Geisinger_subsample_flag == 0:
        # The main beast. Calculate all the optical properties, and outputs the lidar ratio
        optics = calculate_lidar_ratio(aer_particles, WXT['time'], ceil_lambda, r_md_m,  n_wet, num_conc)
    else:
        optics = calculate_lidar_ratio_geisinger(aer_particles, WXT['time'], ceil_lambda, r_md_m,  n_wet, num_conc,
                                    n_samples, r_d_orig_bins_m)

    # extract out the lidar ratio
    S = optics['S']

    # pickle save
    if picklesave == True:

        if process_type == 'PM1':
                WXT = WXT_15min
        elif process_type == 'PM10':
                WXT = WXT_hourly
        elif process_type == 'PM10-1':
                mass_kg_kg = {'PM1': pm10m1_mass_kg_kg, 'PM10-1': pm10m1_mass_kg_kg}
                WXT = WXT_hourly
                N_weight = {'PM1': N_weight_pm1,'PM10-1': N_weight_pm10m1}


        pickle_save = {'optics': optics, 'WXT':WXT, 'N_weight': N_weight,
                       'num_conc':num_conc, 'mass_kg_kg': mass_kg_kg, 'dN':dN, 'r_md':r_md}
        with open(datadir + 'pickle/allvars_'+savesub+'_'+year+'_geisingerSample_'+ceil_lambda_str_nm+'.pickle', 'wb') as handle:
            pickle.dump(pickle_save, handle)

    # get mean and nanstd from data
    # set up the date range to fill (e.g. want daily statistics then stats_date_range = daily resolution)
    # stats = eu.simple_statistics(S, date_range, stats_date_range, np.nanmean, np.nanstd, np.nanmedian)
    stats_date_range = np.array(eu.date_range(WXT['time'][0], WXT['time'][-1] + dt.timedelta(days=1), 1, 'days'))

    stats ={}
    # S_keep = deepcopy(S)
    # S[S > 70] = np.nan

    for stat_i in ['mean', 'median', 'stdev', '25pct', '75pct']:
        stats[stat_i] = np.empty(len(stats_date_range))
        stats[stat_i][:] = np.nan

    # create statistics
    for t, start, end in zip(np.arange(len(stats_date_range[:-1])), stats_date_range[:-1], stats_date_range[1:]):

        # get location of time period's data
        bool = np.logical_and(WXT['time'] >=start, WXT['time']<=end)

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
    plt.savefig(savedir + 'S_'+year+'_'+process_type+'_'+Geisinger_str+'_dailybinned_lt60_'+ceil_lambda_str_nm+'.png')
    plt.close(fig)

    # HISTOGRAM - S

    idx = np.logical_or(np.isnan(S), np.isnan(WXT_hourly['RH']))

    # plot all the S in raw form (hist)
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    # ax.hist(S)
    ax.hist(S[~idx])
    plt.suptitle('Lidar Ratio:\n'+savesub+' masses; equal Number weighting per rbin; ClearfLo winter N(r)')
    plt.xlabel('Lidar Ratio')
    plt.ylabel('Frequency')
    plt.savefig(savedir + 'S_'+year+'_'+process_type+'_'+Geisinger_str+'_histogram_lt60_'+ceil_lambda_str_nm+'.png')
    plt.close(fig)

    # TIMESERIES - S - not binned
    # plot all the S in raw form (plot_date)
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    ax.plot_date(WXT['time'], S, fmt='-')
    plt.suptitle('Lidar Ratio:\n'+savesub+' masses; equal Number weighting per rbin; ClearfLo winter N(r)')
    plt.xlabel('Date [dd/mm]')
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
    plt.ylabel('Lidar Ratio [sr]')
    plt.savefig(savedir + 'S_'+year+'_'+process_type+'_'+Geisinger_str+'_timeseries_lt60_'+ceil_lambda_str_nm+'.png')
    plt.close(fig)

    # SCATTER - S vs RH (PM1)
    # quick plot 15 min S and RH for 2016.
    corr = spearmanr(WXT['RH'], S)
    r_str = '%.2f' % corr[0]
    fig, ax = plt.subplots(1,1,figsize=(8, 4))
    scat = ax.scatter(WXT['RH'], S)
    scat = ax.scatter(WXT['RH'], S, c=N_weight_pm10m1['CBLK']*100.0, vmin= 0.0, vmax = 30.0)
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label('Soot [%]', labelpad=-20, y=1.1, rotation=0)
    #   cbar.set_label('Soot [%]', labelpad=-20, y=1.075, rotation=0)
    # plt.suptitle('Lidar Ratio with soot fraction - r='+r_str+':\n'+savesub+' masses; equal Number weighting per rbin; ClearfLo winter N(r)')
    plt.xlabel(r'$RH \/[\%]$')
    plt.ylabel(r'$Lidar Ratio \/[sr]$')
    plt.tight_layout()
    plt.savefig(savedir + 'S_vs_RH_'+year+'_'+process_type+'_'+Geisinger_str+'_scatter_lt60_'+ceil_lambda_str_nm+'.png')
    plt.close(fig)

    # ------------------------------------------------

    return

if __name__ == '__main__':
    main()

print 'END PROGRAM'