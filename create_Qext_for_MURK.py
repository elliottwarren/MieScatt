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

    print 'Reading particle ...' + particle

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

        # set skip idx to 0 to begin with
        #   it will increase after each t loop
        skip_idx = 0

        # set the intial end of skip range to be the end of the array, in order to find the first proper skip idx
        end_skip_idx = len(var['time'])

        for t in range(len(time_range)):


            # find data for this time
            binary = np.logical_and(var['time'][skip_idx:skip_idx+end_skip_idx] >= time_range[t],
                                    var['time'][skip_idx:skip_idx+end_skip_idx] < time_range[t] + dt.timedelta(minutes=timeRes))

            # actual idx of the data within the entire array
            skip_idx_set_i = np.where(binary == True)[0] + skip_idx

            if t == 170:
                store = skip_idx_set_i

            # set the end_skip_idx to a smaller size once the start skip idx has been found so a smaller idx range
            #   is searched. Set it to timeRes, assuming the smallest the raw resolution data can be is 1 min.
            end_skip_idx = timeRes

            # create means of the data for this time period
            for key in var.iterkeys():
                if key not in ['time']:

                    # take mean of the time period
                    pm_mass[key][t] = np.nanmean(var[key][skip_idx_set_i])

            # change the skip_idx for the next loop to start just after where last idx finished
            if skip_idx_set_i.size != 0:
                skip_idx = skip_idx_set_i[-1] + 1


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

def calc_Q_ext_dry(pm_rel_vol, ceil_lambda, aer_particles_long, r_md, averageType='monthly'):

    """
    Calculate the dry extinction efficiency for each aerosol (Q_dry_aer) and for MURK (Q_dry_murk)

    :param pm_rel_vol:
    :param ceil_lambda:
    :param aer_particles_long:
    :param r_md: dry particle radius [meters]
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


        print 'Read and linearly interpolated aerosols!'

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

        # Calc extinction efficiency for dry aerosol (using r_md!!!! NOT r_m)
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
    x_dry = (2.0 * np.pi * r_md)/ceil_lambda


    # Calc extinction efficiency for each dry aerosol
    Q_dry_aer = {}
    for key, n_i in n_aerosol.iteritems():
        all_particles_dry = [Mie(x=x_i, m=n_i) for x_i in x_dry]
        Q_dry_aer[key] = np.array([particle.qext() for particle in all_particles_dry])

    # Calc extinction efficiency for the monthly MURK values

    if averageType == 'monthly':
        Q_dry_murk = np.empty((len(r_md), 12))
    elif averageType == 'total':
        Q_dry_murk = np.empty((len(r_md), 1))
    else:
        raise ValueError('averageType needs to be either "monthly" or "total"!')


    for month_idx, n_i in enumerate(n_murk):
        all_particles_dry = [Mie(x=x_i, m=n_i) for x_i in x_dry]
        Q_dry_murk[:, month_idx] = np.array([particle.qext() for particle in all_particles_dry])

    return Q_dry_aer, Q_dry_murk

def gaussian_weighted_Q_dry_murk(Q_dry_murk, ceil_lambda_range, ceil_lambda):

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

    # 1) apply the weights for each wavelength to Q_dry_murk to find the contribution from each wavelength separately
    # 2) sum up the weighted Q_dry_murk, across all the wavelengths, to get the final guassian weighted Q_dry_murk
    Q_dry_murk_weighted = np.sum(np.array([w * Q_dry_murk for w in convex_weights]), axis=0)

    return Q_dry_murk_weighted

# plotting

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

def lineplot_monthly_MURK(Q_dry_murk, pm10_mass_merged, site_ins, savedir):

    # make a range of colours for the different months
    colour_range = eu.create_colours(12)[:-1]

    # plot the different MURK Q_ext,dry curves, for each month
    fig = plt.figure(figsize=(6, 4))

    for month_i, month in zip(range(12), range(1, 13)):

        # plot it
        plt.semilogx(r_md_micron, Q_dry_murk[:, month_i], label=str(month), color=colour_range[month_i])


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

    plt.savefig(savedir + 'Q_ext_dry_murk_monthly_'+site_ins['site_short']+'_'+save_date_range+'.png')
    plt.tight_layout(h_pad=10.0)
    plt.close()



    return fig

def lineplot_ratio_monthly_MURK(Q_dry_murk, pm10_mass_merged, site_ins, savedir):

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
        plt.semilogx(r_md_micron, ratio[:, month_i], label=str(month), color=colour_range[month_i])


    # prettify
    plt.xlabel(r'$r_{md} \/\mathrm{[\mu m]}$', labelpad=-10, fontsize=13)
    plt.xlim([0.05, 5.0])
    # plt.ylim([0.0, 5.0])
    plt.ylabel(r'$Q_{ext,dry}$', fontsize=13)
    plt.legend(fontsize=8, loc='best', ncol=2)
    plt.tick_params(axis='both',labelsize=10)
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.grid(b=True, which='minor', color=[0.85, 0.85, 0.85], linestyle='--')

    title_date_range = pm10_mass_merged['time'][0].strftime('%Y/%m/%d') + ' - ' + pm10_mass_merged['time'][-1].strftime('%Y/%m/%d')
    plt.suptitle(site_ins['site_long'] + ': ' + title_date_range)

    save_date_range = pm10_mass_merged['time'][0].strftime('%Y%m%d') + '-' + pm10_mass_merged['time'][-1].strftime('%Y%m%d')

    plt.savefig(savedir + 'Q_ext_dry_murk_monthly_ratio_'+site_ins['site_short']+'_'+save_date_range+'.png')
    plt.tight_layout(h_pad=10.0)
    plt.close()


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

if __name__ == '__main__':

    # -------------------------------------------------------------------
    # Setup

    # directories
    savedir = '/home/nerc/Documents/MieScatt/figures/Q_ext_monthly/'
    datadir = '/home/nerc/Documents/MieScatt/data/monthly_masses/'
    csvsavedir = '/home/nerc/Documents/MieScatt/data/Q_ext/'

    # site information
    # site_ins = {'site_short':'NK', 'site_long': 'North Kensington',
    #             'ceil_lambda': 0.905e-06, 'land-type': 'urban'}
    # site_ins = {'site_short':'Ch', 'site_long': 'Chilbolton',
    #             'ceil_lambda': 0.905e-06, 'land-type': 'rural'}
    site_ins = {'site_short':'Ha', 'site_long': 'Harwell',
                'ceil_lambda': 0.905e-06, 'land-type': 'rural'}

    # wavelength
    ceil_lambda = site_ins['ceil_lambda'] # [m]
    ceil_lambda_str_nm = str(site_ins['ceil_lambda']*1e9)+'nm'

    # aerosol particles to calculate (OC = Organic carbon, CBLK = black carbon, both already measured)
    # match dictionary keys further down
    aer_particles = ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CORG', 'CBLK']

    # aer names in the complex index of refraction files
    aer_particles_long = {'(NH4)2SO4': 'Ammonium sulphate', 'NH4NO3': 'Ammonium nitrate',
                          'CORG': 'Organic carbon', 'NaCl': 'Generic NaCl', 'CBLK':'Soot'}

    aer_colours = {'(NH4)2SO4': 'red', 'NH4NO3': 'orange',
                   'CORG': [0.05, 0.9, 0.4], 'NaCl': 'magenta', 'CBLK':'brown'}

    # # aerosol with relative volume - average from the 4 Haywood et al 2008 flights
    # rel_vol = {'Ammonium sulphate': 0.295,
    #            'Ammonium nitrate': 0.325,
    #             'Organic carbon': 0.38}

    # all the aerosol types and colouring
    # all_aer_order = ['Ammonium sulphate', 'Ammonium nitrate', 'Organic carbon', 'Biogenic', 'Generic NaCl', 'Soot', 'MURK']

    all_aer = {'Ammonium sulphate': 'red', 'Ammonium nitrate':'orange', 'Organic carbon': [0.05, 0.9, 0.4],
               'Biogenic': [0.05,0.56,0.85], 'Generic NaCl': 'magenta', 'Soot': 'brown'}



    # air density for each particle type [kg m-3]
    aer_density = {'(NH4)2SO4': 1770.0,
                   'NH4NO3': 1720.0,
                   'NaCl': 2160.0,
                   'CORG': 1100.0,
                   'CBLK': 1200.0}

    # common time resolution for data, before processing [minutes]
    timeRes = 60 * 24 # daily

    # averaging up to
    average_up = 'monthly'

    # save the Q(dry) curve for MURK?
    savedata = True

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

    # merge pm10 mass data together and average up to a common time resolution defined by timeRes [grams m-3]
    #   which will be the new 'raw data' time
    pm10_mass_merged = time_match_pm_masses(pm10_mass_in, pm10_oc_bc_in, timeRes, nanMissingRow=True) # 27/01/16 - CBLK is NOT present

    # calculate aerosol moles and masses from the gas and aerosol input data [moles], [g m-3]
    # pm10_moles, pm10_mass = calculate_aerosol_moles_masses(pm10_mass_merged)
    pm10_moles, pm10_mass = calculate_aerosol_moles_masses(pm10_mass_merged, outputGases=False,
                                                           aer_particles=aer_particles)

    # average up data once more for the final MURK calculations
    if average_up == 'monthly':
        # monthly average the mass
        pm10_mass_avg, pm10_mass_avg_n = monthly_averaging_mass(pm10_mass, aer_particles)


    elif average_up == 'total':

        # 2. Complete average
        pm10_mass_avg, pm10_mass_avg_n = total_average_mass(pm10_mass, aer_particles)


    # calculate the volume mixing ratio [m3_aerosol m-3_air]
    #     and relative volume [fraction] for each of the aerosol species
    pm10_vol_mix, pm10_rel_vol = calc_vol_and_rel_vol(pm10_mass_avg, aer_particles, aer_density)

    # -----------------------------------------------
    # Calculate Q_ext,dry for each lambda
    # -----------------------------------------------


    # run Q_ext code for range of wavelengths

    # range of lambda around the main wavelength, to use in gaussian weighting
    #   make sure that the wavelength range is large enough so the tails of the distribution are not cut off!
    ceil_lambda_range = np.arange(ceil_lambda - 40e-09, ceil_lambda + 40e-09, 1e-09)

    # create dry size distribution [m]
    step = 0.005
    r_md_micron = np.arange(0.000 + step, 5.000 + step, step)
    r_md = r_md_micron * 1.0e-06

    for lambda_idx, lambda_i in enumerate(ceil_lambda_range):

        # calculate Q_dry for each aerosol, and for murk given the relative volume of each aerosol species
        Q_dry_aer, Q_dry_murk = calc_Q_ext_dry(pm10_rel_vol, lambda_i, aer_particles_long, r_md,
                                                                  averageType=average_up)

    # create guassian weighted Q_dry_murk across the wavelengths to make one for the main ceilometer wavelength
    Q_dry_murk_lambda_weighted = gaussian_weighted_Q_dry_murk(Q_dry_murk, ceil_lambda_range, ceil_lambda)

    # create a guassian weighted Q_dry_murk across a range of radii for each value in the radii range.
    #   need to do this as Q_dry_murk is extremely sensitive to radii, and small changes in radii = large changes in
    #   optical properties!



    # -----------------------------------------------
    # Saving
    # -----------------------------------------------

    # if running for single 905 nm wavelength, save the calculated Q
    if savedata == True:
        if average_up == 'monthly':

            # create headers and 2D array for saving, with column 1 = radius, columns 2 to 13 = months
            headers = '# radius [m],'+','.join([str(i) for i in range(1, 13)]) # need to be comma delimited
            save_array = np.hstack((r_md[:, None], Q_dry_murk))

            # save Q curve and radius [m]
            np.savetxt(csvsavedir + site_ins['land-type'] +'_monthly_Q_ext_dry_'+ceil_lambda_str_nm+'.csv',
                       save_array, delimiter=',', header=headers)
        elif average_up == 'total':

            headers = 'Data created from '+site_ins['site_long']+' between ' \
                       + pm10_mass_merged['time'][0].strftime('%Y/%m/%d') + '-' + pm10_mass_merged['time'][-1].strftime('%Y/%m/%d') + \
                      '\n# radius [m], Q_ext_dry'
            save_array = np.hstack((r_md, Q_dry_murk))

            # save Q curve and radius [m]
            np.savetxt(csvsavedir + site_ins['land-type'] +'_total_Q_ext_dry_'+ceil_lambda_str_nm+'.csv',
                       save_array, delimiter=',', header=headers)

        else:
            raise ValueError('savedata == True but average_up is not "monthly" or "total"!')

    # -----------------------------------------------
    # Plotting
    # -----------------------------------------------

    if average_up == 'monthly':

        # BARCHART - plot the relative volume of each aerosol, across all months
        # https://matplotlib.org/1.3.1/examples/pylab_examples/bar_stacked.html use to improve x axis
        stacked_monthly_bar_rel_aerosol_vol(pm10_rel_vol, pm10_mass_merged, savedir, site_ins)

        # line plot of Q_ext,dry
        lineplot_monthly_MURK(Q_dry_murk, pm10_mass_merged, site_ins, savedir)

        # line plot of Q_ext,dry as a ratio of the average
        lineplot_ratio_monthly_MURK(Q_dry_murk, pm10_mass_merged, site_ins, savedir)

    elif average_up == 'total':

        stacked_total_bar_rel_aerosol_vol(pm10_rel_vol, pm10_mass_merged, savedir, site_ins)

    print 'END PROGRAM'




