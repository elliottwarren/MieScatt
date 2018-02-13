"""
Carry out a simple Mie scattering calculation using the pymiecoated code used in Lally (2014).
Complex index of refraction = CIR

Revisions
July 16 - old version of calc_n_murk () incorrect proportions as it assumed to use the MURK equation, which was for equating NO2, SO2 and VOC
          to MURK and WAS NOT meant to be used for equating (NH4)2SO4 (NH4)NO3 and OC to MURK.This has been fixed.

"""

__author__ = 'elliott_warren'


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

def CIR_Hanel(n_w, n_0, r_0, r):

    """
    Calculate the complex index of refraction (CIR) for the swollen aerosol. Based off of Hanel 1976.

    :param n_w: CIR for pure water
    :param n_0: CIR for pure aerosol (before growth)
    :param r_0: radius before growth
    :param r: radius after growth
    :return: n_swoll: CIR of the swollen aerosol
    """

    n_swoll = n_w + (n_0 - n_w) * (r_0/r)**(3.0)

    return n_swoll

def calc_n_aerosol(aer, ceil_lambda):

    """
    Calculate n for each of the aerosol in rel_vol
    :param rel_vol:
    :return:
    """

    n_aerosol = {}

    if type(aer) == dict:
        for key in aer.iterkeys():
            n_aerosol[key], _ = linear_interpolate_n(key, ceil_lambda)

    elif type(aer) == list:
        for key in aer:
            n_aerosol[key], _ = linear_interpolate_n(key, ceil_lambda)

    print 'Read and linearly interpolated aerosols!'

    return n_aerosol


def calc_n_murk(rel_vol, n):

    """
    Calculate n_murk when inputed relative volumes

    :param rel_vol:
    :return: n_murk
    """

    # volume mixing for CIR (eq. 12, Liu and Daum 2008) -> seem pretty good to quote for this as well as alt. methods
    n_murk = (rel_vol['Ammonium sulphate'] * n['Ammonium sulphate']) + \
             (rel_vol['Ammonium nitrate'] * n['Ammonium nitrate']) + \
             (rel_vol['Organic carbon'] * n['Organic carbon'])

    return n_murk

def plot_radius(savedir, r_md, r_m):

    import matplotlib.pyplot as plt

    # plot radius for wet and dry
    plt.plot(r_m*1.0e6, label='r_m (wet)')
    plt.plot(r_md*1.0e6, label='r_md (dry)')
    plt.legend(loc=0)
    plt.savefig(savedir + 'radii.png')
    plt.close()

    return

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


def main():


    import numpy as np
    from pymiecoated import Mie
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------
    # Setup

    # setup
    ceil_lambda = [0.905e-06] # [m]
    # ceil_lambda = np.arange(0.69e-06, 1.19e-06, 0.05e-06) # [m]
    # ceil_lambda = np.arange(0.90e-06, 0.91e-06, 0.05e-08) # [m]
    # ceil_lambda = np.array(([0.90e-06, 0.91e-06, 0.92e-06])) # [m]
    B = 0.14
    RH_crit = 0.38

    # directories
    savedir = '/home/nerc/Documents/MieScatt/figures/'
    datadir = '/home/nerc/Documents/MieScatt/data/'

    # aerosol with relative volume - average from the 4 Haywood et al 2008 flights
    rel_vol = {'Ammonium sulphate': 0.295,
               'Ammonium nitrate': 0.325,
                'Organic carbon': 0.38}

    # all the aerosol types
    all_aer_order = ['Ammonium sulphate', 'Ammonium nitrate', 'Organic carbon', 'Biogenic', 'Generic NaCl', 'Soot', 'MURK']
    # all_aer = ['ammonium_sulphate', 'ammonium_nitrate', 'organic_carbon', 'biogenic', 'NaCl', 'soot']
    all_aer = {'Ammonium sulphate': 'red', 'Ammonium nitrate':'orange', 'Organic carbon': [0.05, 0.9, 0.4],
               'Biogenic': [0.05,0.56,0.85], 'Generic NaCl': 'magenta', 'Soot': 'brown'}
    # all_aer = ['soot']

    # create dry size distribution [m]
    # r_md_microm = np.arange(0.03, 5.001, 0.001) # .shape() = 4971
    # r_md_microm = np.arange(0.000 + step, 1.000 + step, step), when step = 0.005, .shape() = 200
    step = 0.005
    r_md_microm = np.arange(0.000 + step, 5.000 + step, step)
    r_md = r_md_microm * 1.0e-06

    # RH array [fraction]
    # This gets fixed for each Q iteration (Q needs to be recalculated for each RH used)
    RH = 0.8

    # densities of MURK constituents # [kg m-3]
    # range of densities of org carb is massive (0.625 - 2 g cm-3)
    # Haywood et al 2003 use 1.35 g cm-3 but Schkolnik et al., 2006 lit review this and claim 1.1 g cm-3
    dens_amm_sulph = 1770
    dens_amm_nit = 1720
    dens_org_carb = 1100 # NOTE ABOVE

    # define array to store Q for each wavelength
    Q_dry = []

    x_store =[]
    n_store=[]

    # save the Q(dry) curve for MURK?
    savedata = True

    # -----------------------------------------------
    # Calculate Q for each lambda
    # -----------------------------------------------

    # -------------------------------------------------------------------
    # Process

    # calculate complex index of refraction for MURK species
    # output n is complex index of refraction (n + ik)
    n_aerosol = calc_n_aerosol(all_aer, ceil_lambda)

    # NOTE: Able to use volume in MURK equation instead of mass because, if mass is evenly distributed within a volume
    # then taking x of the mass = taking x of the volume.
    # after calculating volumes used in MURK, can find relative % and do volume mixing.

    # bulk complex index of refraction (CIR) for the MURK species using volume mixing method
    n_murk = calc_n_murk(rel_vol, n_aerosol)
    n_aerosol['MURK'] = n_murk
    all_aer['MURK'] = 'black' # add color for plotting
    # n_murk = complex(1.53, 0.007) - this is for 550 nm
    n_store += [n_murk]

    # complex indices of refraction (n = n(bar) - ik) at ceilometer wavelength (905 nm) Hesse et al 1998
    # n_water, _ = linear_interpolate_n('water', lam)

    # swell particles using FO method
    # rm = np.ma.ones(RH.shape) - (B / np.ma.log(RH_ge_RHcrit))
    r_m = 1 - (B / np.log(RH))
    r_m2 = np.ma.power(r_m, 1. / 3.)
    r_m = np.ma.array(r_md) * r_m2
    r_m_microm = r_m * 1.0e06

    # calculate size parameter for dry and wet
    x_dry = (2.0 * np.pi * r_md)/ceil_lambda
    x_store += [x_dry]
    x_wet = (2.0 * np.pi * r_m)/ceil_lambda


    # calculate swollen index of refraction using MURK
    # n_swoll = CIR_Hanel(n_water, n_murk, r_md, r_m)


    # Calc extinction efficiency for dry aerosol (using r_md!!!! NOT r_m)
    # all_particles_dry = [Mie(x=x_i, m=n_murk) for x_i in x_dry]
    # Q_dry += [np.array([particle.qext() for particle in all_particles_dry])]

    Q_dry = {}
    for key, n_i in n_aerosol.iteritems():
        all_particles_dry = [Mie(x=x_i, m=n_i) for x_i in x_dry]
        Q_dry[key] = np.array([particle.qext() for particle in all_particles_dry])
    # qsca, qabs
    # -----------------------------------------------
    # Post processing, saving and plotting
    # -----------------------------------------------


    # if running for single 905 nm wavelength, save the calculated Q
    if savedata == True:
        if type(ceil_lambda) == list:
            if ceil_lambda[0] == 0.905e-06:
                # save Q curve and radius [m]
                np.savetxt(datadir + 'calculated_Q_ext_905nm.csv', np.transpose(np.vstack((r_md, Q_dry['MURK']))), delimiter=',', header='radius,Q_ext')


    # plot
    fig = plt.figure(figsize=(6, 4))

    for aer_i in all_aer_order:

        # plot it
        plt.semilogx(r_md_microm, Q_dry[aer_i], label=aer_i, color=all_aer[aer_i])
        # plt.semilogx(r_md_microm, Q_dry, label='dry murk', color=[0,0,0])
        # plt.semilogx(r_m_microm, Q_del, label='deliquescent murk (RH = ' + str(RH) + ')')
        # plt.semilogx(r_m_microm, Q_coat, label='coated murk (RH = ' + str(RH) + ')')

    # for aer_i, Q_dry_i in Q_dry.iteritems():
    #
    #     # plot it
    #     plt.semilogx(r_md_microm, Q_dry_i, label=aer_i, color=all_aer[aer_i])
    #     # plt.semilogx(r_md_microm, Q_dry, label='dry murk', color=[0,0,0])
    #     # plt.semilogx(r_m_microm, Q_del, label='deliquescent murk (RH = ' + str(RH) + ')')
    #     # plt.semilogx(r_m_microm, Q_coat, label='coated murk (RH = ' + str(RH) + ')')


    # plt.title('lambda = ' + str(ceil_lambda[0]) + 'nm')
    plt.xlabel(r'$r_{md} \/\mathrm{[\mu m]}$', labelpad=-10, fontsize=13)
    plt.xlim([0.05, 5.0])
    plt.ylim([0.0, 5.0])
    #plt.xlim([0.01, 0.2])
    #plt.ylim([0.0, 0.1])
    plt.ylabel(r'$Q_{ext,dry}$', fontsize=13)
    plt.legend(fontsize=8, loc='best')
    plt.tick_params(axis='both',labelsize=10)
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.grid(b=True, which='minor', color=[0.85, 0.85, 0.85], linestyle='--')
    plt.savefig(savedir + 'Q_ext_manyAer2_' + str(ceil_lambda[0]) + 'nm.png')
    print 'data dir is... ' + savedir + 'Q_ext_manyAer_' + str(ceil_lambda[0]) + 'nm.png'
    plt.tight_layout(h_pad=10.0)
    plt.close()

    # plot the radius
    # plot_radius(savedir, r_md, r_m)

    print 'END PROGRAM'

if __name__ == '__main__':
    main()


