"""
Create Qext,dry for MURK, based on the masses of aerosol species observed

Created by Elliott Warren: Mon 12 Feb 2018

"""

__author__ = 'elliott_warren'

import numpy as np
from pymiecoated import Mie
import matplotlib.pyplot as plt
import datetime as dt
import ellUtils as eu

from dateutil import tz

# Read

def read_PM_mass_long_term_data(massdatadir, filename):


    """
    Read in PM mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :param pmtype: what type of pm to read in, that is in the filename (e.g. pm10, pm2p5)
    :return: mass
    :return qaqc_idx_unique: unique index list where any of the main species observations are missing
    """

    massfilepath = massdatadir + filename
    massrawData = np.genfromtxt(massfilepath, delimiter=',', dtype="|S20") # includes the header
    # massrawData = np.loadtxt(massfilepath, delimiter=',', dtype="|S20") # includes the header

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

def read_EC_BC_mass_long_term_data(massdatadir, filename):

    """
    Read in the elemental carbon (EC) and organic carbon (OC) mass data from NK
    Raw data is micrograms m-3 but converted to and outputed as grams m-3
    :param year:
    :param pmtype: which PM to get the data for (must match that used in the filename) e.g. PM10, PM2p5
    :return: mass

    EC and BC (soot) are treated the same in CLASSIC, therefore EC will be taken as BC here.
    """

    massfilepath = massdatadir + filename
    massrawData = np.genfromtxt(massfilepath, delimiter=',', skip_header=4, dtype="|S20") # includes the header

    mass = {'time': np.array([dt.datetime.strptime(i[0], '%d/%m/%Y') for i in massrawData[1:]]),
            'CBLK': np.array([np.nan if i[1] == 'No data' else i[1] for i in massrawData[1:]], dtype=float),
            'CORG': np.array([np.nan if i[3] == 'No data' else i[3] for i in massrawData[1:]], dtype=float)}


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

def part_file_read(particle):

    """
    Locate and read the particle file. STore wavelength, n and k parts in dictionary
    """

    # print 'Reading particle ...' + particle

    from numpy import array

    # particle data dir
    # part_datadir = '/media/sf_HostGuestShared/MieScatt/complex index of refraction/'

    part_datadir = '/home/nerc/Documents/MieScatt/aerosol_files/'

    # find particle filename
    if particle == 'Ammonium nitrate':
        part_file = 'refract_ammoniumnitrate'
    elif particle == 'Ammonium sulphate':
        part_file = 'refract_ammoniumsulphate'
    elif particle == 'Organic carbon':
        part_file = 'refract_ocff'
    elif particle == 'Oceanic':
        part_file = 'refract_oceanic'
    elif particle == 'Soot':
        part_file = 'refract_soot_bond'
    elif particle == 'Biogenic':
        part_file = 'refract_biogenic'
    elif particle == 'Generic NaCl':
        part_file = 'refract_nacl'
    elif particle == 'water':
        part_file = 'refract_water.txt'
    else:
        raise ValueError("incorrect species or not yet included in particle list")

    # full path
    file_path = part_datadir + part_file

    # empty dictionary to hold data
    data = {'lambda': [],
            'real': [],
            'imaginary': []}

    # open file and read down to when data starts
    file = open(file_path, "r")
    s = file.readline()
    s = s.rstrip('\n\r')

    while s != '*BEGIN_DATA':
        s = file.readline()
        s = s.rstrip('\n\r')

    line = file.readline() # read line
    line = line.rstrip('\n\r')

    while (line != '*END') & (line != '*END_DATA'):
        line = ' '.join(line.split()) # remove leading and trailing spaces. Replace multiple spaces in the middle with one.
        line = line.decode('utf8').encode('ascii', errors='ignore') # remove all non-ASCII characters

        # if line isn't last line in file

        line_split = line.split(' ')
        data['lambda'] += [float(line_split[0])]
        data['real'] += [float(line_split[1])]
        data['imaginary'] += [float(line_split[2])]

        # read next line
        line = file.readline()
        line = line.rstrip('\n\r')

    # convert to numpy array
    for key, value in data.iteritems():
        data[key] = array(value)

    return data

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

def time_match_pm_masses(pm_mass_in, oc_bc_in, timeRes, nanMissingRow=False):

    # pm_mass_in = pm10_mass_in
    # oc_bc_in = pm10_oc_bc_in

    """
    time match all the main data dictionaries together (pm, RH and dN data), according to the time resolution given
    (timeRes). Makes all values nan for a time, t, if any one of the variables has missing data (very conservative
    appoach).

    :param pm2p5_mass_in:
    :param pm10_mass_in:
    :param met_in: contains RH, Tair, air pressure
    :param dN_in:
    :param timeRes:
    :param nanMissingRow: [bool] nan all variables in a row, if one of them is missing.
    :return: pm2p5_mass, pm10_mass, met, dN
    """

    # time buffer - time allowed the t_range_idx to be 'out' but still acceptable - set at 1/4 the timeRes [minutes]
    t_buffer = (0.25*timeRes)

    ## 1. set up dictionaries with times
    # as we want the daily time resolution, remove the hours and make a time range from that
    s = np.min([pm_mass_in['time'][0], oc_bc_in['time'][0]])
    start_time = dt.datetime(s.year, s.month, s.day)

    e = np.max([pm_mass_in['time'][-1], oc_bc_in['time'][-1]])
    end_time = dt.datetime(e.year, e.month, e.day)

    time_range = eu.date_range(start_time, end_time, timeRes, 'minutes')

    # make sure datetimes are in UTC
    to_zone = tz.gettz('UTC')
    time_range = np.array([i.replace(tzinfo=to_zone) for i in time_range])


    # set up dictionaries (just with time and any non-time related values at the moment)
    pm_mass = {'time': time_range}
    ## 2. set up empty arrays within dictionaries
    # prepare empty arrays within the outputted dictionaries for the other variables, ready to be filled.
    for var in [pm_mass_in, oc_bc_in]:

        for key in var.iterkeys():
            # only fill up the variables
            if key not in ['time']:

                pm_mass[key] = np.empty(len(time_range))
                pm_mass[key][:] = np.nan


    ## 3. fill the variables with time averages
    # use a moving subsample assuming the data is in ascending order
    for var in [pm_mass_in, oc_bc_in]:


        for t, time_t in enumerate(time_range):


            # time previous to time_t
            time_tm1 = time_t - dt.timedelta(minutes=timeRes)

            s_idx = int(eu.binary_search(var['time'], time_tm1))
            # end of time period
            e_idx = int(eu.binary_search(var['time'], time_t))

            # if the time_range time and data['time'] found in this iteration are within an acceptable range (15 mins)
            tm1_diff = time_tm1 - var['time'][s_idx]
            t_diff = time_t - var['time'][e_idx]

            t_range_idx = range(s_idx, e_idx+1)

            # allow a time buffer of 1/4 the timeRes (e.g. timeREs = 60, then allow the tm1_diff and t_diff be 15 minutes out)
            # t_buffer converted to seconds here for comparison
            if (tm1_diff.total_seconds() <= t_buffer * 60) & (t_diff.total_seconds() <= t_buffer * 60):

                # create means of the data for this time period
                for key in var.iterkeys():
                    if key not in ['time']:

                        # take mean of the time period
                        pm_mass[key][t] = np.nanmean(var[key][t_range_idx])

    ## 4. nan across variables for missing data
    # make data for any instance of time, t, to be nan if any data is missing from dN, met or pm mass data

    if nanMissingRow == True:

        print 'setting all observations to NaN if one of the variables is missing...'

        ## 4.1 find bad items
        # make and append to a list, rows where bad data is present, across all the variables
        bad = []

        for key, data in pm_mass.iteritems():
            if key not in ['time']:

                # if data was nan, store its idx
                for t in range(len(time_range)):
                    if np.isnan(data[t]):
                        bad += [t]


        ## 4.2 find unique bad idxs and make all values at that time nan, across all the variables
        bad_uni = np.unique(np.array(bad))

        for key, data in pm_mass.iteritems():
            if key not in ['time']:
                pm_mass[key][bad_uni] = np.nan

    else:
        print 'NOT setting all observations to NaN if one of the variables is missing...'


    return pm_mass

# Process

def monthly_averaging_mass(mass, aer_particles):

    """
    MOnthly average the mass, before calculating the relative volumes in a different function.
    :param mass:
    :param aer_particles:
    :return: mass_avg: monthly avergaes of the mass
    :return mass_avg_n: sample size of each month
    """

    # setup lists for averages for each month
    mass_avg = {aer_i: np.empty(12) for aer_i in aer_particles}
    mass_avg_n = {aer_i: np.empty(12) for aer_i in aer_particles}

    # extract months for monthly average processingl
    months_from_date = np.array([i.month for i in mass['time']])

    # create monthly averages
    for month_i, month in zip(range(12), range(1, 13)):

        # find all times with current month
        month_time_idx = np.where(months_from_date == month)

        for aer_i in aer_particles:

            # make the average for each aerosol type
            mass_avg[aer_i][month_i] = np.nanmean(mass[aer_i][month_time_idx])
            # sample size
            mass_avg_n[aer_i][month_i] = np.sum(~np.isnan(mass[aer_i][month_time_idx]))

    # turn all lists into np.arrays()
    for aer_i in aer_particles:

        mass_avg[aer_i] = np.array(mass_avg[aer_i])

    return mass_avg, mass_avg_n

def total_average_mass(mass, aer_particles):

    """
    average the mass, before calculating the relative volumes in a different function.
    :param mass:
    :param aer_particles:
    :return: mass_avg: monthly avergaes of the mass
    :return mass_avg_n: sample size of each month

    keep the values as a single element within a dictionary, so it is consistent with the other
    average dictionary's formats
    """

    # setup lists for averages for each month
    mass_avg = {aer_i: np.empty(1) for aer_i in aer_particles}
    mass_avg_n = {aer_i: np.empty(1) for aer_i in aer_particles}

    for aer_i in aer_particles:

        # make the average for each aerosol type
        mass_avg[aer_i][0] = np.nanmean(mass[aer_i])
        # sample size
        mass_avg_n[aer_i][0] = np.sum(~np.isnan(mass[aer_i]))

    # turn all lists into np.arrays()
    for aer_i in aer_particles:

        mass_avg[aer_i] = np.array(mass_avg[aer_i])

    return mass_avg, mass_avg_n

def individual_monthly_stats(pm10_mass_merged, input_species):

    """
    Create monthly statistics for the pm10_merged_mass_data
    :param pm10_mass_merged:
    :param input_species:
    :return: stats
    """

    # set up stats dictionary
    stats = {}
    for species in input_species:
        stats[species] = {'binned': [], 'mean': [], 'median': [], 'stdev': [], '25pct': [], '75pct': []}

    start = pm10_mass_merged['time'][0]
    end = pm10_mass_merged['time'][-1]

    for year_i in range(start.year, end.year + 1):

        # define month range for this year
        if year_i == start.year:
            month_range = range(start.month, 13)
        elif year_i == end.year:
            month_range = range(1, end.month + 1)
        else:
            month_range = range(1, 13)


        for month_i in month_range:

            # get location of time period's data
            bool = np.array([True if (i.year == year_i) & (i.month == month_i)
                             else False for i in pm10_mass_merged['time']])

            for species in input_species:

                # extract data
                subsample = pm10_mass_merged[species][bool]

                stats[species]['binned'] += [subsample]
                stats[species]['mean'] += [np.nanmean(subsample)]
                stats[species]['stdev'] += [np.nanstd(subsample)]
                stats[species]['median'] += [np.nanmedian(subsample)]
                stats[species]['25pct'] += [np.nanpercentile(subsample, 25)]
                stats[species]['75pct'] += [np.nanpercentile(subsample, 75)]


    # turn all lists into np.arrays
    for species in input_species:
        for key in stats[species].iterkeys():
            stats[species][key] = np.array(stats[species][key])

    return stats

def monthly_stats(pm10_mass_merged, input_species):

    """
    Create monthly statistics for the pm10_merged_mass_data
    :param pm10_mass_merged:
    :param input_species:
    :return: stats
    """

    # set up stats dictionary
    mean_stats = {}
    for species in input_species:
        mean_stats[species] = {}
        for month_i in range(1, 13):

            # get month name and create empty list ready for appending
            month_name = dt.date(1900, month_i, 1).strftime('%b')
            mean_stats[species][month_name] = []


    start = pm10_mass_merged['time'][0]
    end = pm10_mass_merged['time'][-1]

    for year_i in range(start.year, end.year + 1):

        # define month range for this year
        if year_i == start.year:
            month_range = range(start.month, 13)
        elif year_i == end.year:
            month_range = range(1, end.month + 1)
        else:
            month_range = range(1, 13)


        for month_i in (month_range):

            # get location of time period's data
            bool = np.array([True if (i.year == year_i) & (i.month == month_i)
                             else False for i in pm10_mass_merged['time']])

            # get month name
            month_name = dt.date(1900, month_i, 1).strftime('%b')

            for species in input_species:

                # extract data
                subsample = pm10_mass_merged[species][bool]

                mean_stats[species][month_name] += [np.nanmean(subsample)]


    # turn all lists into np.arrays
    for species in input_species:
        for key in mean_stats[species].iterkeys():
            mean_stats[species][key] = np.array(mean_stats[species][key])

    return mean_stats

def calculate_aerosol_moles_masses(mass, outputGases=False, **kwargs):

    """
    Calculate the moles and mass of aerosol from the input aerosol and gas data
    :param mass: [g cm-3]
    :param met:
    :param aer_particles:
    :keyword outputGases: output the gas mass and moles as well as just the aerosols [bool]
                aer_particles needs to be defined if outputGases=False
    :return: moles [moles], mass [g cm-3]
    """

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

    # extract just the aerosol masses and leave out the gases?
    if outputGases == False:
        if 'aer_particles' in kwargs:
            # create the output mass directory
            mass_out = {'time': mass['time']}
            for aer_i in kwargs['aer_particles']:
                mass_out[aer_i] = mass[aer_i]
    else:
        mass_out = mass

    return moles, mass_out

def calc_rel_mass(mass, species):

    """
    Calculate the relative mass concentration [kg m-3] of each aerosol species
    :param mass: [g m-3]
    :param species: aerosol/gas species to keep and make relative masses from
    :return: particle_mass: actual mass of each particle species [kg -3]
    :return: rel_mass: relative mass of species to the (species) list inputted [kg m-3]
    """

    from copy import deepcopy

    # keep this code encase the monthly averaging is to be applied to non-averaged dictionaries with 'time' as a key
    # # extract just the aerosol parts of the vol_mix dictionary
    # vol_mix_aer = deepcopy(vol_mix)
    # del vol_mix_aer['time']

    # only keep the gases/aerosol defined in species
    # convert units from g m-3 to [kg m-3]
    particle_mass = {}
    for species_i in species:
        particle_mass[species_i] = deepcopy(mass[species_i]) * 1.0e-03

    # calculate the relative volume for each species [fraction]
    rel_mass = {}
    for species_i in species:
        rel_mass[species_i] = particle_mass[species_i] / np.sum(np.array(particle_mass.values()), axis=0)

    return particle_mass, rel_mass

def calc_vol_and_rel_vol(mass, aer_particles, aer_density):

    # mass = pm10_mass

    """
    Calculate the volume mixing ratio with air and relative volume of each aerosol species
    :param mass: [g m-3]
    :param aer_particles:
    :param aer_density:
    :return: vol_mix [m3_aerosol m-3_air]
    :return: rel_vol [fraction]
    """

    from copy import deepcopy

    # calculate volume mixing ratios for each of the species [m3_aerosol m-3_air]
    # mass units come in as g m-3_air, so it needs to be converted to kg m-3_air to use with aer_density [kg m-3]
    vol_mix = {}
    for aer_i in aer_particles:
        vol_mix[aer_i] = (mass[aer_i] * 1.0e3) / aer_density[aer_i]

    # keep this code encase the monthly averaging is to be applied to non-averaged dictionaries with 'time' as a key
    # # extract just the aerosol parts of the vol_mix dictionary
    # vol_mix_aer = deepcopy(vol_mix)
    # del vol_mix_aer['time']

    # calculate the relative volume of aerosol species [fraction]
    rel_vol = {}
    for aer_i in aer_particles:
        rel_vol[aer_i] = vol_mix[aer_i] / np.sum(np.array(vol_mix.values()), axis=0)

    return vol_mix, rel_vol

def linear_interpolate_n(particle, aim_lambda):

    """
    linearly interpolate the complex index of refraction for a wavelength of the aerosol and water

    :input: dict: contains lower_lambda, upper_lambda, lower_n, upper_n, lower_k, upper_k for particle type
    :input: aim_lambda: what wavelength the values are being interpolated to

    :return:n: interpolated complex index of refraction
    :return:dict_parts: dictionary with the index refraction parts and how far it was interpolated
    """

    import numpy as np


    # read in the particle file data
    data = part_file_read(particle)

    # find locaiton of lambda within the spectral file
    idx = np.searchsorted(data['lambda'], aim_lambda)

    # find adjacent wavelengths
    # if lambda is same as one in spectral file, extract
    if data['lambda'][idx] == aim_lambda:

        lambda_n = data['real'][idx]
        lambda_k = data['imaginary'][idx]
        frac = np.nan

    # else interpolate to it
    else:
        upper_lambda = data['lambda'][idx]
        lower_lambda = data['lambda'][idx-1]
        upper_n = data['real'][idx]
        lower_n = data['real'][idx-1]
        upper_k = data['imaginary'][idx]
        lower_k = data['imaginary'][idx-1]

        # differences
        diff_lambda = upper_lambda - lower_lambda
        diff_n = upper_n - lower_n
        diff_k = upper_k - lower_k

        # distance aim_lambda is along linear interpolation [fraction] from the lower limit
        frac = ((aim_lambda - lower_lambda) / diff_lambda)

        # calc interpolated values for n and k
        lambda_n = lower_n + (frac * abs(diff_n))
        lambda_k = lower_k + (frac * abs(diff_k))


    # Complex index of refraction using lambda_n and k
    n = complex(lambda_n, lambda_k)

    dict_parts = {'lambda_n': lambda_n,
                'lambda_k': lambda_k,
                'frac': frac}

    return n, dict_parts

def calc_Q_ext_dry(pm_rel_vol, ceil_lambda, aer_particles_long, r_d, averageType='monthly'):

    """
    Calculate the dry extinction efficiency for each aerosol (Q_dry_aer) and for MURK (Q_dry_murk)

    :param pm_rel_vol:
    :param ceil_lambda:
    :param aer_particles_long:
    :param r_d: dry particle radius [meters]
    :param averageType: ['monthly' or 'yearly'] Type of average data being used to calculate Q_ext_dry
    :return: Q_dry_aer: dry extinction efficiencyfor each aerosol separately
    :return: Q_dry_murk: dry extinction efficiency for aerosols combined into the new MURK
    """

    def calc_n_aerosol(aer_particles_long, ceil_lambda):

        """
        Calculate n for each of the aerosol
        :param aer_particles_long: dictionary with the key as short name, and value as long name
        :return:
        """

        n_aerosol = {}

        if type(aer_particles_long) == dict:
            for aer_i, long_name in aer_particles_long.iteritems():
                n_aerosol[aer_i], _ = linear_interpolate_n(long_name, ceil_lambda)


        # print 'Read and linearly interpolated aerosols!'

        return n_aerosol

    def calc_Q_ext(x, m, type, y=[], m2=[],):

        """
        Calculate Q_ext. Can be dry, coated in water, or deliquescent with water

        :param x: dry particle size parameter
        :param m: complex index of refraction for particle
        :param y: wet particle size parameter
        :param m2: complex index of refraction for water
        :return: Q_ext
        :return: calc_type: how was particle treated? (e.g. dry, coated)
        """

        from pymiecoated import Mie
        import numpy as np

        # Coated aerosol (insoluble aerosol that gets coated as it takes on water)
        if type == 'coated':

            if (y != []) & (m2 != []):

                all_particles_coat = [Mie(x=x[i], m=m, y=y[i], m2=m2) for i in np.arange(len(x))]
                Q = np.array([particle.qext() for particle in all_particles_coat])

            else:
                raise ValueError("type = coated, but y or m2 is empty []")

        # Calc extinction efficiency for dry aerosol (using r_d!!!! NOT r_m)
        # if singular, then type == complex, else type == array
        elif type == 'dry':

            all_particles_dry = [Mie(x=x, m=m) for x_i in x]
            Q = np.array([particle.qext() for particle in all_particles_dry])

        # deliquescent aerosol (solute disolves as it takes on water)
        elif type == 'deliquescent':

            all_particles_del = [Mie(x=x[i], m=m[i]) for i in np.arange(len(x))]
            Q = np.array([particle.qext() for particle in all_particles_del])


        return Q

    # get the complex index of refraction (n) for each aerosol, for the wavelength
    # output n is complex index of refraction (n + ik)
    n_aerosol = calc_n_aerosol(aer_particles_long, ceil_lambda)

    # calculate n for MURK, for each month using volume mixing method
    n_murk = np.nansum([pm_rel_vol[aer_i] * n_aerosol[aer_i] for aer_i in pm_rel_vol.keys()], axis=0)

    # calculate size parameter for dry and wet
    x_dry = (2.0 * np.pi * r_d)/ceil_lambda

    # Calc extinction efficiency for each dry aerosol
    # Commented out to save computational resources
    Q_dry_aer = {}
    for key, n_i in n_aerosol.iteritems():
        all_particles_dry = [Mie(x=x_i, m=n_i) for x_i in x_dry]
        Q_dry_aer[key] = np.array([particle.qext() for particle in all_particles_dry])

    # Calc extinction efficiency for the monthly MURK values
    if averageType == 'hourly':
        Q_dry_murk = np.empty((len(r_d), 1))
    elif averageType == 'monthly':
        Q_dry_murk = np.empty((len(r_d), 12))
    elif averageType == 'total':
        Q_dry_murk = np.empty((len(r_d), 1))
    else:
        raise ValueError('averageType needs to be either "monthly" or "total"!')


    for month_idx, n_i in enumerate(n_murk):
        all_particles_dry = [Mie(x=x_i, m=n_i) for x_i in x_dry]
        Q_dry_murk[:, month_idx] = np.array([particle.qext() for particle in all_particles_dry])

    return Q_dry_murk

def calc_Q_ext_dry_aer(aer_i, ceil_lambda, aer_particles_long, r_d, averageType='monthly'):

    """
    Calculate the dry extinction efficiency for each aerosol (Q_dry_aer) and for MURK (Q_dry_murk)

    :param pm_rel_vol:
    :param ceil_lambda:
    :param aer_particles_long:
    :param r_d: dry particle radius [meters]
    :param averageType: ['monthly' or 'yearly'] Type of average data being used to calculate Q_ext_dry
    :return: Q_dry_aer: dry extinction efficiencyfor each aerosol separately
    :return: Q_dry_murk: dry extinction efficiency for aerosols combined into the new MURK
    """

    def calc_n_aerosol(aer_particles_long, ceil_lambda):

        """
        Calculate n for each of the aerosol
        :param aer_particles_long: dictionary with the key as short name, and value as long name
        :return:
        """

        n_aerosol = {}

        if type(aer_particles_long) == dict:
            for aer_i, long_name in aer_particles_long.iteritems():
                n_aerosol[aer_i], _ = linear_interpolate_n(long_name, ceil_lambda)


        # print 'Read and linearly interpolated aerosols!'

        return n_aerosol

    # get the complex index of refraction (n) for the aerosol, for the wavelength
    # output n is complex index of refraction (n + ik)
    n_aerosol = calc_n_aerosol(aer_particles_long, ceil_lambda)
    n_aerosol = n_aerosol[aer_i]

    # calculate size parameter for dry
    x_dry = (2.0 * np.pi * r_d)/ceil_lambda

    # Calc extinction efficiency for each dry aerosol
    # Commented out to save computational resources
    all_particles_dry = [Mie(x=x_i, m=n_aerosol) for x_i in x_dry]
    Q_dry_aer = np.array([particle.qext() for particle in all_particles_dry])

    return Q_dry_aer

def gaussian_weighted_Q_dry_aer_lambda(Q_dry, ceil_lambda_range, ceil_lambda, aer_type='murk'):

    """
    Calculate the gaussian weighted value for Q_dry_murk, with respect to wavelength.

    :param Q_dry_murk: Q_ext_dry for murk that needs the weighting applied to it
    :param ceil_lambda_range: range of lambda values to calculate P(lambda) for, and then used for the weighting
    :param ceil_lambda: mean/central wavelength for the ceilometer
    :return:Q_dry_murk_weighted: guassian weighted Q_ext_dry for MURK
    """

    # sigma backcalculated from FWHM: https://en.wikipedia.org/wiki/Full_width_at_half_maximum - based on wolfram alpha
    # calculate standard deviation (sigma) from full width half maximum for CL31
    FWHM = 4.0e-09
    sigma = FWHM /(2.0 * np.sqrt(2.0 * np.log(2.0)))

    # calculate P(lambda) across the whole set of blks for the main (mean) ceilometer wavelength (ceil_lambda)
    weights = np.array([eu.normpdf(i, ceil_lambda, sigma) for i in ceil_lambda_range])

    # modify weights so they add up to 1
    # convex combination - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Convex_combination_example
    convex_weights = weights/np.sum(weights)

    # calculate Q_dry_murk for each of the wavelengths, weights have been calculated for


    # 1) apply the weights for each wavelength to Q_dry_murk to find the contribution from each wavelength separately
    # 2) sum up the weighted Q_dry_murk, across all the wavelengths, to get the final guassian weighted Q_dry_murk
    if aer_type=='murk':
        Q_dry_weighted = np.sum(convex_weights[:, None, None] * Q_dry, axis=0)
    else:
        Q_dry_weighted = np.sum(convex_weights[:, None] * Q_dry, axis=0)

    return Q_dry_weighted

def guassian_weighted_Q_dry_aer_radii(Q_dry_lambda_weighted, r_d, geo_std_dev, aer_type='murk'):

    """
    Weight Q_dry_murk by radii after having been weighted by lambda.

    :param Q_dry_murk_lambda_weighted: Q_dry_murk already weighted by lambda
    :param r_d:
    :param geo_std_dev:
    :return:Q_dry_murk_lambda_radii_weighted: Q_dry_murk weighted by radii
    """

    # calculate log of the radius
    log10_r_d = np.log10(r_d)

    # store water vapour mass absorption for the different wavelengths [np.array]
    # store gaussian weights for plot [list]
    convex_weights = []
    Q_dry_lambda_radii_weighted = np.empty(Q_dry_lambda_weighted.shape)
    Q_dry_lambda_radii_weighted[:] = np.nan

    # convert geometric standard deviation (1.6: Harrison et al 2012 taken from a log10 transformed distribution,
    #   used in Warren et al. paper 1) to the standard deviation
    stdev_r_d = np.log10(geo_std_dev)

    for log10_r_d_idx, log10_r_d_i in enumerate(log10_r_d):

        # # make a small range of log_r_d for the weights to be calulate from, for this instance of log10_r_d_i
        # #   make sure the range is wide enough for the tails of the gaussian are not missed!
        # extend = np.log10(2.0e-06)
        # min_r = log10_r_d_i - extend
        # max_r = log10_r_d_i + extend
        # step = (max_r - min_r)/100.0
        # log10_r_d_range = np.arange(min_r, max_r, step)

        # calculate P(radii) for the main (mean) radii (log10_r_d_i) across a radii range
        weights = np.array([eu.normpdf(i, log10_r_d_i, stdev_r_d) for i in log10_r_d])

        # modify weights so they add up to 1
        # convex combination - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Convex_combination_example
        convex_weights_i = weights/np.sum(weights)
        convex_weights += [convex_weights_i]

        # 1) apply the weights across the different radii of Q_dry_murk_lambda_weighted to find Q_ext_dry for this radii
        # 2) sum up the weighted Q_dry_murk, across all the wavelengths, to get the final guassian weighted Q_dry_murk
        if aer_type =='murk':
            Q_dry_lambda_radii_weighted[log10_r_d_idx, :] = np.sum(convex_weights_i[None, :, None] * Q_dry_lambda_weighted, axis=1)
        else:
            Q_dry_lambda_radii_weighted[log10_r_d_idx] = np.sum(convex_weights_i[None, :] * Q_dry_lambda_weighted, axis=1)

    return Q_dry_lambda_radii_weighted

# plotting

## bar charts
def stacked_monthly_bar_rel_aerosol_vol(pm_rel_vol, pm_mass_merged, savedir, site_ins):

    """
    Plot the relative amount of each aerosol, across the months in a stacked bar chart with a legend.

    :param pm_rel_vol:
    :param pm_mass_merged:
    :param savedir:
    :param site_ins:
    :return: fig
    """

    # pm_rel_vol = pm10_rel_vol
    # pm_mass_merged = pm10_mass_merged

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # where to put the bottom of the bar chart, start at 0 and then move it up with each aer_i iteration
    bottom = np.zeros(12)

    index = np.arange(12)
    width = 1.0

    for aer_i, rel_vol_aer_i in pm_rel_vol.iteritems():

        plt.bar(index, rel_vol_aer_i, bottom=bottom, width=width, color = aer_colours[aer_i], label=aer_i)

        # move the bottom of the bar location up, for the next iteration
        bottom = bottom + rel_vol_aer_i

    plt.xlabel('month')
    plt.xticks(index+width/2.0, [str(i) for i in np.arange(1, 13)])
    plt.ylabel('fraction')
    plt.ylim([0.0, 1.0])
    plt.legend(loc='best', fontsize = 8, bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.9, right=0.8)

    # # add sample number at the top of each box
    # (y_min, y_max) = ax.get_ylim()
    # upperLabels = [str(np.round(n, 2)) for n in np.hstack(rh_split['n'])]
    # for tick in range(len(np.hstack(pos))):
    #     k = tick % 3
    #     ax.text(np.hstack(pos)[tick], y_max - (y_max * (0.05)*(k+1)), upperLabels[tick],
    #              horizontalalignment='center', size='x-small')

    # date for plotting
    title_date_range = pm_mass_merged['time'][0].strftime('%Y/%m/%d') + ' - ' + pm_mass_merged['time'][-1].strftime('%Y/%m/%d')
    plt.suptitle(site_ins['site_long'] + ': ' + title_date_range + '; Relative volume')

    save_date_range = pm_mass_merged['time'][0].strftime('%Y%m%d') + '-' + pm_mass_merged['time'][-1].strftime('%Y%m%d')
    plt.savefig(savedir + 'Q_ext_dry_monthly_nanmissing_' + site_ins['site_short'] + '_' + save_date_range)

    return fig

def stacked_monthly_bar_rel_aerosol_vol_sorted(pm_rel_vol, pm_mass_merged, savedir, site_ins, aer_particles):

    """
    Plot the SORTED relative amount of each aerosol, across the months in a stacked bar chart with a legend.

    :param pm_rel_vol:
    :param pm_mass_merged:
    :param savedir:
    :param site_ins:
    :param aer_particles: aerosol particles in a list to help the plotting order
    :return: fig
    """



    # 1 figure per species and site
    for aer_main, rel_vol_aer_main in pm_rel_vol.iteritems():

        # pm_rel_vol = pm10_rel_vol
        # pm_mass_merged = pm10_mass_merged

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        # where to put the bottom of the bar chart, start at 0 and then move it up with each aer_i iteration
        bottom = np.zeros(12)

        index = np.arange(12)
        width = 1.0

        # idx position - sorts smallest to largest by default so [::-1] reverses it to be largest to smallest
        idx_sort = np.argsort(rel_vol_aer_main)[::-1]

        # plot the main aerosol first, then loop through all the others
        plt.bar(index, rel_vol_aer_main[idx_sort], bottom=bottom, width=width, color=aer_colours[aer_main], label=aer_main)

        # move the bottom of the bar location up, for the next iteration
        bottom = bottom + rel_vol_aer_main[idx_sort]

        # loop through the ordered list so the plotting order, after aer_main, is the same each time
        for aer_i in aer_particles:

            # extract data for this aerosol
            rel_vol_aer_i = pm_rel_vol[aer_i]

            # if it's a new aerosol...
            if aer_i != aer_main:

                plt.bar(index, rel_vol_aer_i[idx_sort], bottom=bottom, width=width, color=aer_colours[aer_i], label=aer_i)

                # move the bottom of the bar location up, for the next iteration
                bottom = bottom + rel_vol_aer_i[idx_sort]


        plt.xlabel('month')
        plt.xticks(index+width/2.0, [dt.datetime(1900, i+1, 1).strftime('%b') for i in idx_sort])
        plt.ylabel('fraction')
        plt.ylim([0.0, 1.0])
        plt.legend(loc='best', fontsize = 8, bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

        plt.tight_layout(h_pad=0.1)
        plt.subplots_adjust(top=0.9, right=0.8)


        # date for plotting
        title_date_range = pm_mass_merged['time'][0].strftime('%Y/%m/%d') + ' - ' + pm_mass_merged['time'][-1].strftime('%Y/%m/%d')
        plt.suptitle(site_ins['site_long'] + ': ' + title_date_range + '; sorted rel vol: ' + aer_main)

        save_date_range = pm_mass_merged['time'][0].strftime('%Y%m%d') + '-' + pm_mass_merged['time'][-1].strftime('%Y%m%d')
        plt.savefig(savedir + 'ranked_species/rel_vol_' + aer_main + '_' +site_ins['site_short'] + '_' + save_date_range)

        plt.close(fig)

    return fig

def stacked_monthly_bar_rel_aerosol_vol_sorted_nit_sulph_stack(pm_rel_vol, pm_mass_merged, savedir, site_ins, aer_particles):

    """
    Plot the SORTED relative amount of each aerosol, across the months in a stacked bar chart with a legend.
    Stack ammonium nitrate and ammonium sulphate together

    :param pm_rel_vol:
    :param pm_mass_merged:
    :param savedir:
    :param site_ins:
    :param aer_particles: aerosol particles in a list to help the plotting order
    :return: fig
    """


    # 1 figure per species and site
    # for aer_main, rel_vol_aer_main in pm_rel_vol.iteritems():

    # add together amm. nitrate and sulphate
    nit_sulph_combined = pm_rel_vol['NH4NO3'] + pm_rel_vol['(NH4)2SO4']

    # idx position - sorts smallest to largest by default so [::-1] reverses it to be largest to smallest
    idx_sort = np.argsort(nit_sulph_combined)[::-1]

    # start plotting
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # where to put the bottom of the bar chart, start at 0 and then move it up with each aer_i iteration
    bottom = np.zeros(12)

    index = np.arange(12)
    width = 1.0

    # 1. plot the main 2 aerosol first, then loop through all the others
    for aer_main in ['NH4NO3', '(NH4)2SO4']:

        plt.bar(index, pm_rel_vol[aer_main][idx_sort], bottom=bottom, width=width, color=aer_colours[aer_main], label=aer_main)

        # move the bottom of the bar location up, for the next iteration
        bottom = bottom + pm_rel_vol[aer_main][idx_sort]


    # 2. loop through the ordered list so the plotting order, after aer_main, is the same each time
    for aer_i in aer_particles:

        # extract data for this aerosol
        rel_vol_aer_i = pm_rel_vol[aer_i]

        # if it's a new aerosol...
        if aer_i not in ['NH4NO3', '(NH4)2SO4']:

            plt.bar(index, rel_vol_aer_i[idx_sort], bottom=bottom, width=width, color=aer_colours[aer_i], label=aer_i)

            # move the bottom of the bar location up, for the next iteration
            bottom = bottom + rel_vol_aer_i[idx_sort]


    plt.xlabel('month')
    plt.xticks(index+width/2.0, [dt.datetime(1900, i+1, 1).strftime('%b') for i in idx_sort])
    plt.ylabel('fraction')
    plt.ylim([0.0, 1.0])
    plt.legend(loc='best', fontsize = 8, bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.9, right=0.8)


    # date for plotting
    title_date_range = pm_mass_merged['time'][0].strftime('%Y/%m/%d') + ' - ' + pm_mass_merged['time'][-1].strftime('%Y/%m/%d')
    plt.suptitle(site_ins['site_long'] + ': ' + title_date_range + '; sorted rel vol: NH4NO3 + (NH4)2SO4')

    save_date_range = pm_mass_merged['time'][0].strftime('%Y%m%d') + '-' + pm_mass_merged['time'][-1].strftime('%Y%m%d')
    plt.savefig(savedir + 'ranked_species/rel_vol_nitAndSulph_' +site_ins['site_short'] + '_' + save_date_range)

    plt.close(fig)

    return fig

def stacked_monthly_bar_rel_species(pm_rel, pm_mass_merged, dataType, savedir, site_ins, colours):

    """
    Plot the relative amount of each aerosol, across the months in a stacked bar chart with a legend.

    :param pm_rel:
    :param pm_mass_merged:
    :param savedir:
    :param site_ins:
    :param colours: colours for the plotting
    :return: fig
    """

    # pm_rel_vol = pm10_rel_vol
    # pm_mass_merged = pm10_mass_merged

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # where to put the bottom of the bar chart, start at 0 and then move it up with each aer_i iteration
    bottom = np.zeros(12)

    index = np.arange(12)
    width = 1.0

    for species_i, rel_species_i in pm_rel.iteritems():

        plt.bar(index, rel_species_i, bottom=bottom, width=width, color = colours[species_i], label=species_i)

        # move the bottom of the bar location up, for the next iteration
        bottom = bottom + rel_species_i

    plt.xlabel('month')
    plt.xticks(index+width/2.0, [str(i) for i in np.arange(1, 13)])
    plt.ylabel('fraction')
    plt.ylim([0.0, 1.0])
    plt.legend(loc='best', fontsize = 8, bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.9, right=0.8)

    # # add sample number at the top of each box
    # (y_min, y_max) = ax.get_ylim()
    # upperLabels = [str(np.round(n, 2)) for n in np.hstack(rh_split['n'])]
    # for tick in range(len(np.hstack(pos))):
    #     k = tick % 3
    #     ax.text(np.hstack(pos)[tick], y_max - (y_max * (0.05)*(k+1)), upperLabels[tick],
    #              horizontalalignment='center', size='x-small')

    # date for plotting
    title_date_range = pm_mass_merged['time'][0].strftime('%Y/%m/%d') + ' - ' + pm_mass_merged['time'][-1].strftime('%Y/%m/%d')
    plt.suptitle(site_ins['site_long'] + ': ' + title_date_range + '; Relative '+dataType)

    save_date_range = pm_mass_merged['time'][0].strftime('%Y%m%d') + '-' + pm_mass_merged['time'][-1].strftime('%Y%m%d')
    plt.savefig(savedir + 'Q_ext_dry_monthly_' + dataType + '_' + site_ins['site_short'] + '_' + save_date_range)

    return fig

def stacked_total_bar_rel_aerosol_vol(pm_rel_vol, pm_mass_merged, savedir, site_ins):

    """
    Plot the relative amount of each aerosol, in a stacked bar chart with a legend.

    :param pm_rel_vol:
    :param pm_mass_merged:
    :param savedir:
    :param site_ins:
    :return: fig
    """

    # pm_rel_vol = pm10_rel_vol
    # pm_mass_merged = pm10_mass_merged

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # allows the bar chart to stack, starting at 0
    bottom = 0.0

    for aer_i, rel_vol_aer_i in pm_rel_vol.iteritems():

        print aer_i

        plt.bar(np.array([1.0]), rel_vol_aer_i[0], bottom=bottom, width=1.0, color=aer_colours[aer_i], label=aer_i)

        # raise bottom for the next bar to be the top of the old one
        bottom = bottom + rel_vol_aer_i[0]

    plt.ylabel('fraction')
    plt.ylim([0.0, 1.0])
    plt.xlim([1.0, 2.0])
    plt.legend(loc='best', fontsize=8, bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.9, right=0.8)

    # date for plotting
    title_date_range = pm_mass_merged['time'][0].strftime('%Y/%m/%d') + ' - ' + pm_mass_merged['time'][-1].strftime('%Y/%m/%d')
    plt.suptitle(site_ins['site_long'] + ': ' + title_date_range + '; Relative volume')

    save_date_range = pm_mass_merged['time'][0].strftime('%Y%m%d') + '-' + pm_mass_merged['time'][-1].strftime('%Y%m%d')
    plt.savefig(savedir + 'Q_ext_dry_total_' + site_ins['site_short'] + '_' + save_date_range)

    return fig

## other

def lineplot_monthly_MURK(Q_dry_murk, pm10_mass_merged, site_ins, savedir, r_d_micron, extra=''):

    """
    Plot the monthly Q_ext,dry values for MURK
    :param Q_dry_murk:
    :param pm10_mass_merged:
    :param site_ins:
    :param savedir:
    :param extra: string to change the save filename (bit of a rough approach)
    :return:fig
    """

    # make a range of colours for the different months
    colour_range = eu.create_colours(12)[:-1]

    # plot the different MURK Q_ext,dry curves, for each month
    fig = plt.figure(figsize=(6, 4))

    for month_i, month in zip(range(12), range(1, 13)):

        # plot it
        plt.semilogx(r_d_micron, Q_dry_murk[:, month_i], label=str(month), color=colour_range[month_i])


    # prettify
    plt.xlabel(r'$r_{md} \/\mathrm{[\mu m]}$', labelpad=-10, fontsize=13)
    plt.xlim([0.05, 5.0])
    plt.ylim([0.0, 5.0])
    plt.ylabel(r'$Q_{ext,dry}$', fontsize=13)
    plt.legend(fontsize=8, loc='best')
    plt.tick_params(axis='both',labelsize=10)
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.grid(b=True, which='minor', color=[0.85, 0.85, 0.85], linestyle='--')

    title_date_range = pm10_mass_merged['time'][0].strftime('%Y/%m/%d') + ' - ' + pm10_mass_merged['time'][-1].strftime('%Y/%m/%d')
    plt.suptitle(site_ins['site_long'] + ': ' + title_date_range)

    save_date_range = pm10_mass_merged['time'][0].strftime('%Y%m%d') + '-' + pm10_mass_merged['time'][-1].strftime('%Y%m%d')

    plt.savefig(savedir + 'Q_ext_dry_murk_monthly_'+site_ins['site_short']+'_'+save_date_range+'_'+extra+'.png')
    plt.tight_layout(h_pad=10.0)
    plt.close()



    return fig

def lineplot_ratio_monthly_MURK(Q_dry_murk, pm10_mass_merged, site_ins, savedir, r_d_micron, extra=''):

    # make a range of colours for the different months
    colour_range = eu.create_colours(12)[:-1]

    # plot the different MURK Q_ext,dry curves, for each month
    fig = plt.figure(figsize=(6, 4))


    # create average and ratio for each month
    average = np.nanmean(Q_dry_murk, axis=1)
    # create ratio for each month
    ratio = Q_dry_murk / average[:, None]

    for month_i, month in zip(range(12), range(1, 13)):

        # plot it
        plt.semilogx(r_d_micron, ratio[:, month_i], label=str(month), color=colour_range[month_i])


    # prettify
    plt.xlabel(r'$r_{md} \/\mathrm{[\mu m]}$', labelpad=-10, fontsize=13)
    plt.xlim([0.05, 5.0])
    # plt.ylim([0.0, 5.0])
    plt.ylabel(r'$\frac{Q_{ext,dry,month}}{Q_{ext,dry,average}}$', labelpad=10, fontsize=13)
    plt.legend(fontsize=8, loc='best', ncol=2)
    plt.tick_params(axis='both',labelsize=10)
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.grid(b=True, which='minor', color=[0.85, 0.85, 0.85], linestyle='--')

    title_date_range = pm10_mass_merged['time'][0].strftime('%Y/%m/%d') + ' - ' + pm10_mass_merged['time'][-1].strftime('%Y/%m/%d')
    plt.suptitle(site_ins['site_long'] + ': ' + title_date_range)

    save_date_range = pm10_mass_merged['time'][0].strftime('%Y%m%d') + '-' + pm10_mass_merged['time'][-1].strftime('%Y%m%d')

    plt.savefig(savedir + 'Q_ext_dry_murk_monthly_ratio_'+site_ins['site_short']+'_'+save_date_range+'_'+extra+'.png')
    plt.tight_layout(h_pad=10.0)
    plt.close()


    return fig



if __name__ == '__main__':

    # -------------------------------------------------------------------
    # Setup

    # site information
    site_ins = {'site_short':'NK', 'site_long': 'North_Kensington',
                'ceil_lambda': 0.905e-06, 'land-type': 'urban'}
    # site_ins = {'site_short':'Ch', 'site_long': 'Chilbolton',
    #             'ceil_lambda': 0.905e-06, 'land-type': 'rural'}
    # site_ins = {'site_short':'Ha', 'site_long': 'Harwell',
    #             'ceil_lambda': 0.905e-06, 'land-type': 'rural'}

    # directories
    savedir = '/home/nerc/Documents/MieScatt/figures/Q_ext_monthly/'
    datadir = '/home/nerc/Documents/MieScatt/data/'+site_ins['site_long']+'/'
    csvsavedir = '/home/nerc/Documents/MieScatt/data/Q_ext/'
    barchartsavedir = '/home/nerc/Documents/MieScatt/figures/Q_ext_monthly/barcharts/'
    pickledir = '/home/nerc/Documents/MieScatt/data/pickle/' + site_ins['site_long'] + '/'

    # wavelength
    ceil_lambda = site_ins['ceil_lambda'] # [m]
    ceil_lambda_str_nm = str(site_ins['ceil_lambda']*1e9)+'nm'

    # aerosol particles to calculate (OC = Organic carbon, CBLK = black carbon, both already measured)
    # match dictionary keys further down
    aer_particles = ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CORG', 'CBLK']

    # original species used in estimating the total aerosols
    input_species = ['Na', 'NH4', 'NO3', 'SO4', 'CL', 'CORG', 'CBLK']

    # aer names in the complex index of refraction files
    aer_particles_long = {'(NH4)2SO4': 'Ammonium sulphate', 'NH4NO3': 'Ammonium nitrate',
                          'CORG': 'Organic carbon', 'NaCl': 'Generic NaCl', 'CBLK':'Soot'}

    # plotting colours - ERG c
    aer_colours = {'(NH4)2SO4': 'red', 'NH4NO3': 'orange',
                   'CORG': [0.05, 0.9, 0.4], 'NaCl': 'magenta', 'CBLK':'brown'}
    ERG_colours = {'NH4': 'orange', 'SO4': 'red', 'NO3': 'blue',
                   'CORG': '#00ff00', 'Na': '#33ccff', 'CL': '#cc0099', 'CBLK':'black'}


    # # aerosol with relative volume - average from the 4 Haywood et al 2008 flights
    # rel_vol = {'Ammonium sulphate': 0.295,
    #            'Ammonium nitrate': 0.325,
    #             'Organic carbon': 0.38}

    # all the aerosol types and colouring
    # all_aer_order = ['Ammonium sulphate', 'Ammonium nitrate', 'Organic carbon', 'Biogenic', 'Generic NaCl', 'Soot', 'MURK']

    all_aer = {'Ammonium sulphate': 'red', 'Ammonium nitrate':'orange', 'Organic carbon': [0.05, 0.9, 0.4],
               'Biogenic': [0.05,0.56,0.85], 'Generic NaCl': 'magenta', 'Soot': 'brown'}


    # density for each particle type [kg m-3]
    aer_density = {'(NH4)2SO4': 1770.0,
                   'NH4NO3': 1720.0,
                   'NaCl': 2160.0,
                   'CORG': 1100.0,
                   'CBLK': 1200.0}

    # geometric standard deviation for radii weighting
    geo_std_dev = 1.6

    # common time resolution for data, before processing [minutes]
    timeRes = 60 * 24 # daily

    # averaging up to
    average_up = 'monthly' # 'hourly'

    # save the Q(dry) curve for MURK?
    savedata = True

    # weight the calculations of Q_ext_dry for wavelength and radii?
    weight_Q_ext_dry = True

    if weight_Q_ext_dry == True:
        weight_str = 'guassian weighting with respect to wavelength and radii with geometric std dev of '+str(geo_std_dev)
    else:
        weight_str = 'no gaussian weighting applied'

    # -------------------------------------
    # Read in data
    # -------------------------------------


    # Read in the hourly other pm10 data [grams m-3]
    if site_ins['site_short'] == 'NK':
        filename_pm10species = 'PM10species_Hr_NK_DEFRA_02022011-08022018.csv'
        filename_oc_ec = 'PM10_OC_EC_Daily_NK_DEFRA_01012010-31122016.csv'

    elif site_ins['site_short'] == 'Ch':
        filename_pm10species = 'PM10species_Hr_Chilbolton_DEFRA_11012016-30092017.csv'
        filename_oc_ec = 'PM10_OC_EC_Daily_Chilbolton_DEFRA_11012016-31122016.csv'

    elif site_ins['site_short'] == 'Ha':
        filename_pm10species = 'PM10species_Hr_Harwell_DEFRA_01112011-31122015.csv'
        filename_oc_ec = 'PM10_OC_EC_Daily_Harwell_DEFRA_01012010-31122015.csv'

    pm10_mass_in, _ = read_PM_mass_long_term_data(datadir, filename_pm10species)

    # Read in the daily EC and OC data [grams m-3]
    pm10_oc_bc_in = read_EC_BC_mass_long_term_data(datadir, filename_oc_ec) # 27/01/16 - CBLK is present

    # linearly interpolate OC and BC from daily to hourly resolution if average_up == 'hourly'
    if average_up == 'hourly':
        pm10_oc_bc_in = oc_bc_interp_hourly(pm10_oc_bc_in)

    # merge pm10 mass data together and average up to a common time resolution defined by timeRes [grams m-3]
    #   which will be the new 'raw data' time
    pm10_mass_merged = time_match_pm_masses(pm10_mass_in, pm10_oc_bc_in, timeRes, nanMissingRow=True)

    # calculate aerosol moles and masses from the gas and aerosol input data [moles], [g m-3]
    # pm10_moles, pm10_mass = calculate_aerosol_moles_masses(pm10_mass_merged)
    pm10_moles, pm10_mass = calculate_aerosol_moles_masses(pm10_mass_merged, outputGases=False,
                                                           aer_particles=aer_particles)

    # to be used with the gases
    # use this for the inter-annual variability
    # pm10_rel_mass = calc_rel_mass(pm10_mass_merged, input_species)


    # average up data once more for the final MURK calculations
    if average_up == 'daily':

        # if data is already in daily, then no need for extra processing
        if timeRes == 1440: # if already in daily resolution (1440 mins = 1 day)

            pm10_mass_avg = pm10_mass

    if average_up == 'hourly':

        # if data is already in hourly, then no need for extra processing
        if timeRes == 60: # if already in hourly resolution

            pm10_mass_avg = pm10_mass

    elif average_up == 'monthly':
        # monthly average the mass
        pm10_mass_avg, pm10_mass_avg_n = monthly_averaging_mass(pm10_mass, aer_particles)
        # pm10_mass_avg, pm10_mass_avg_n = monthly_averaging_mass(pm10_mass_merged, input_species)

        # create stats for each month separately from the pm10_mass_merged data
        stats = individual_monthly_stats(pm10_mass_merged, input_species)

        # means for each month
        mean_stats = monthly_stats(pm10_mass_merged, input_species)


    elif average_up == 'total':
        # 2. Complete average
        pm10_mass_avg, pm10_mass_avg_n = total_average_mass(pm10_mass, aer_particles)


    # rel mass of averaged data
    # pm10_avg_mass, pm10_rel_avg_mass = calc_rel_mass(pm10_mass_avg, input_species) # gases and inputs
    # pm10_avg_mass, pm10_rel_avg_mass = calc_rel_mass(pm10_mass_avg, aer_particles) # final aerosol

    # calculate the volume mixing ratio [m3_aerosol m-3_air]
    #     and relative volume [fraction] for each of the aerosol species
    pm10_vol_mix, pm10_rel_vol = calc_vol_and_rel_vol(pm10_mass_avg, aer_particles, aer_density)

    # Rank and reorder the relative volume of species
    # pm10_ranked_rel_vol = rank_rel_vol



    # -----------------------------------------------
    # Calculate Q_ext,dry for each lambda
    # -----------------------------------------------

    # run Q_ext code for range of wavelengths

    # range of lambda around the main wavelength, to use in gaussian weighting
    #   make sure that the wavelength range is large enough so the tails of the distribution are not cut off!
    ceil_lambda_range = np.arange(ceil_lambda - 40e-09, ceil_lambda + 40e-09, 1e-09)

    # create dry size distribution [microns], [m]
    # 1. simple linear range
    # step_micron = 0.005
    # r_d_micron = np.arange(0.000 + step_micron, 5.000 + step_micron, step_micron)
    # r_d = r_d_micron * 1.0e-06

    # 2. varying range
    # step r_d by 1*10^i where i varies across the different size magnitudes e.g. -6 for microns, -9 for nm
    step = 0.025
    num_range = np.arange(1.0, 10.0, step)
    r_d = np.hstack([num_range*(10**i) for i in range(-11, -5)])
    r_d_micron = r_d * 1e6


    # create an array to store the _Q_dry_murk values
    Q_dry_murk = np.empty((len(ceil_lambda_range), len(r_d), 12))
    Q_dry_murk[:] = np.nan

    Q_dry_aer = {}
    for aer_i in aer_particles:
        Q_dry_aer[aer_i] = np.empty((len(ceil_lambda_range), len(r_d)))
        Q_dry_aer[aer_i][:] = np.nan

    for lambda_idx, lambda_i in enumerate(ceil_lambda_range):

        # current progress
        print 'Q_dry_murk: '+ str(lambda_idx+1) +' of ' + str(len(ceil_lambda_range))

        # calculate Q_dry for each aerosol, and for murk given the relative volume of each aerosol species
        Q_dry_murk[lambda_idx, :, :] = calc_Q_ext_dry(pm10_rel_vol, lambda_i, aer_particles_long, r_d,
                                                         averageType=average_up)

        # # Q_ext,dry for each aerosol - no time component, just as a function of radius
        # for aer_i in aer_particles:
        #     Q_dry_aer[aer_i][lambda_idx, :]  = calc_Q_ext_dry_aer(aer_i, lambda_i, aer_particles_long, r_d,
        #                                                  averageType='hourly')

    # -----------------------------------------------
    # Weighting
    # -----------------------------------------------

    # create guassian weighted Q_dry_murk across the wavelengths to make one for the main ceilometer wavelength
    Q_dry_murk_lambda_weighted = gaussian_weighted_Q_dry_aer_lambda(Q_dry_murk, ceil_lambda_range, ceil_lambda,
                                                                    aer_type='murk')

    # create a guassian weighted Q_dry_murk across a range of radii for each value in the radii range.
    #   need to do this as Q_dry_murk is extremely sensitive to radii, and small changes in radii = large changes in
    #   optical properties!
    Q_dry_murk_lambda_radii_weighted = guassian_weighted_Q_dry_aer_radii(Q_dry_murk_lambda_weighted, r_d, geo_std_dev,
                                                                          aer_type='murk')

    # # do the same for the individal aerosol species
    # Q_dry_aer_lambda_weighted = {}
    # Q_dry_aer_lambda_radii_weighted = {}
    # for aer_i in aer_particles:
    #     Q_dry_aer_lambda_weighted[aer_i] = gaussian_weighted_Q_dry_aer_lambda(Q_dry_aer[aer_i], ceil_lambda_range, ceil_lambda,
    #                                                                 aer_type=aer_i)
    #
    #     Q_dry_aer_lambda_radii_weighted[aer_i] = guassian_weighted_Q_dry_aer_radii(Q_dry_aer_lambda_weighted[aer_i], r_d,
    #                                                                         geo_std_dev, aer_type=aer_i)

    # -----------------------------------------------
    # Saving
    # -----------------------------------------------

    # if running for single 905 nm wavelength, save the calculated Q
    if savedata == True:
        if weight_Q_ext_dry == True:
            if average_up == 'monthly':

                # create headers and 2D array for saving, with column 1 = radius, columns 2 to 13 = months
                headers = '# Data created from '+site_ins['site_long']+' between ' \
                           + pm10_mass_merged['time'][0].strftime('%Y/%m/%d') + '-' + pm10_mass_merged['time'][-1].strftime('%Y/%m/%d') + \
                           ' ' + weight_str + ' ' + \
                           '\n# radius [m],'+','.join([str(i) for i in range(1, 13)]) # need to be comma delimited
                save_array = np.hstack((r_d[:, None], Q_dry_murk_lambda_radii_weighted))

                # save Q curve and radius [m]
                save_name = csvsavedir + site_ins['land-type'] +'_monthly_Q_ext_dry_'+ceil_lambda_str_nm+'.csv'
                np.savetxt(save_name, save_array, delimiter=',', header=headers)
                print save_name + ' is saved!'
            elif average_up == 'total':

                headers = '# Data created from '+site_ins['site_long']+' between ' \
                           + pm10_mass_merged['time'][0].strftime('%Y/%m/%d') + '-' + pm10_mass_merged['time'][-1].strftime('%Y/%m/%d') + \
                           ' ' + weight_str + ' ' + \
                          '\n# radius [m], Q_ext_dry'
                save_array = np.hstack((r_d, Q_dry_murk_lambda_radii_weighted))

                # save Q curve and radius [m]
                save_name = csvsavedir + site_ins['land-type'] +'_total_Q_ext_dry_'+ceil_lambda_str_nm+'.csv'
                np.savetxt(save_name, save_array, delimiter=',', header=headers)
                print save_name + ' is saved!'

    else:
        raise ValueError('savedata == True but average_up is not "monthly" or "total"!')

    # # save the relative volume of aerosol for making f(RH) curves in pickle form!
    # #   "I'm a pickle!"
    # # set up save_time as the original has a timezone and for some reason it messes up the pickle load later on
    # save_time = np.array([dt.datetime(i.year, i.month, i.day) for i in pm10_mass_avg['time']])
    # # pickle_save = {'pm10_rel_vol': pm10_rel_vol, 'time': save_time}
    # # with open(pickledir +site_ins['site_short']+'_'+average_up+'_aerosol_relative_volume.pickle', 'wb') as handle:
    # #     pickle.dump(pickle_save, handle)
    # npy_save = {'pm10_rel_vol': pm10_rel_vol, 'time': save_time}
    # np.save(pickledir +site_ins['site_short']+'_'+average_up+'_aerosol_relative_volume.npy', npy_save)

    #save the aerosol Q_ext,dry values for use in aerFO run experiment
    # save_time = np.array([dt.datetime(i.year, i.month, i.day) for i in pm10_mass_avg['time']])
    save_time = np.array([i.replace(tzinfo=None) for i in pm10_mass_avg['time']])
    npy_save = {'Q_dry_aer': Q_dry_aer_lambda_radii_weighted,
                'r_d': r_d,
                'time': save_time}
    save_name = pickledir +site_ins['site_short']+'_all_aerosol_Q_ext_dry_'+ceil_lambda_str_nm+'.npy'
    npy_save = np.save(save_name, npy_save)
    print save_name + ' was saved!'

    # save the hourly relative volume
    hourly_rel_vol_save = pm10_rel_vol
    hourly_rel_vol_save.update({'time': np.array([i.replace(tzinfo=None) for i in pm10_mass_avg['time']])})
    rel_vol_name = pickledir + 'NK_hourly_aerosol_relative_volume.npy'
    np.save(rel_vol_name, hourly_rel_vol_save)
    print rel_vol_name + ' was saved!'

    # -----------------------------------------------
    # Plotting
    # -----------------------------------------------

    if average_up == 'monthly':

        # BARCHART - plot the relative volume of each aerosol, across all months
        # https://matplotlib.org/1.3.1/examples/pylab_examples/bar_stacked.html use to improve x axis
        stacked_monthly_bar_rel_aerosol_vol(pm10_rel_vol, pm10_mass_merged, barchartsavedir, site_ins)

        # barchart - for relative volume of each species (sorted) and a separate
        #   function for amm. nitrate and sulphate together.
        stacked_monthly_bar_rel_aerosol_vol_sorted(pm10_rel_vol, pm10_mass_merged, barchartsavedir, site_ins, aer_particles)
        stacked_monthly_bar_rel_aerosol_vol_sorted_nit_sulph_stack(pm10_rel_vol, pm10_mass_merged, barchartsavedir, site_ins, aer_particles)

        # barchart for mass of input species
        # stacked_monthly_bar_rel_species(pm10_rel_avg_mass, pm10_mass_merged, 'Relative mass concentration', barchartsavedir, site_ins, ERG_colours)

        # line plot of Q_ext,dry
        lineplot_monthly_MURK(Q_dry_murk, pm10_mass_merged, site_ins, savedir, r_d_micron, extra='')

        # line plot of Q_ext,dry as a ratio of the average
        lineplot_ratio_monthly_MURK(Q_dry_murk, pm10_mass_merged, site_ins, r_d_micron, savedir, extra='tester')

        # # TIMESERIES - S - stats (the one with many months)
        # # plot daily statistics of S
        # fig, ax = plt.subplots(1,1,figsize=(8, 5))
        # for species in input_species:
        #     ax.plot(stats[species]['median'], color=ERG_colours[species], label=species)
        #     ax.plot(stats[species]['25pct'], linestyle='--', color=ERG_colours[species])
        #     ax.plot(stats[species]['75pct'], linestyle='--', color=ERG_colours[species])
        #     # ax.fill_between(range(len(stats[species]['mean'])), stats[species]['mean'] - stats[species]['stdev'], stats[species]['mean'] + stats[species]['stdev'], alpha=0.3) #  facecolor='blue'
        # plt.legend()
        # plt.suptitle('relative mass, binned by month')
        # # plt.xlabel('Date [dd/mm]')
        # # plt.ylim([20.0, 60.0])
        # # ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
        # plt.ylabel('Lidar Ratio')
        # plt.savefig(savedir + 'relative_species/' + 'S_'+year+'_'+site_ins['site_short']+'_'+process_type+'_'+Geisinger_str+'_dailybinned_'+ceil_lambda_str_nm+'.png')
        # # plt.savefig(savedir + 'S_'+year+'_'+process_type+'_'+Geisinger_str+'_dailybinned_lt60_'+ceil_lambda_str_nm+'.png')
        # plt.close(fig)


        # quick plot the guassian weights to check they are ok
        # if weight_Q_ext_dry == True:
        #
        #     # MURK line plot with the weighted wavelength
        #     lineplot_monthly_MURK(Q_dry_murk_lambda_weighted, pm10_mass_merged, site_ins, savedir, r_d_micron, extra='wavelength_radii_weighted')
        #
        #     # line plot of Q_ext,dry as a ratio of the average
        #     lineplot_ratio_monthly_MURK(Q_dry_murk_lambda_radii_weighted, pm10_mass_merged, site_ins, savedir, r_d_micron, extra='wavelength_radii_weighted')
        #
        #     # plot the guassians
        #     fig=plt.figure(1,figsize=(6,4))
        #     ax=plt.subplot(1,1,1)
        #
        #     for gaus_i, r_d_micron_i in zip(convex_weights, r_d_micron):
        #
        #         plt.semilogx(r_d_micron*2.0, gaus_i, label=str(r_d_micron_i))
        #         #ax.plot(log10_r_d, gaus_i, label=str(r_d_micron_i))
        #
        #     ax.set_ylabel('weighting')
        #     ax.set_xlabel('D [micron]')
        #     ax.set_xlim(10e-2, 2.0e1)
        #     # ax.legend()
        #     plt.tight_layout()
        #     plt.savefig(savedir + '/diags/' + 'D_gaussians_wrt_centralRadii.png')
        #
        #     plt.close(fig)

    elif average_up == 'total':

        stacked_total_bar_rel_aerosol_vol(pm10_rel_vol, pm10_mass_merged, savedir, site_ins)

    print 'END PROGRAM'




