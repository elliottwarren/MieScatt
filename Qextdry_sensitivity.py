"""
Sensitivity of Q_ext,dry to wavelength for different aerosol types.
910 nm is the 'standard' the others are compared to

"""

__author__ = 'nerc'



import numpy as np
from pymiecoated import Mie
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import ellUtils as eu


def create_colours(intervals):

    r = 0.2
    g = 0.5

    step = 1.0 / intervals

    b_colours = np.arange(0.0, 1.0 + step, step)
    g_colours = np.arange(1.0, 0.0 - step, -step)

    g_colours[-1] = 0.0

    colours = [[r, g_colours[i], b_colours[i]] for i in range(len(b_colours))]

    return colours

def linear_interpolate_n(particle, aim_lambda):

    """
    linearly interpolate the complex index of refraction for a wavelength of the aerosol and water

    :input: dict: contains lower_lambda, upper_lambda, lower_n, upper_n, lower_k, upper_k for particle type
    :input: aim_lambda: what wavelength the values are being interpolated to

    :return:n: interpolated complex index of refraction
    :return:dict_parts: dictionary with the index refraction parts and how far it was interpolated
    """

    import numpy as np

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
        if particle == 'ammonium_nitrate':
            part_file = 'refract_ammoniumnitrate'
        elif particle == 'ammonium_sulphate':
            part_file = 'refract_ammoniumsulphate'
        elif particle == 'organic_carbon':
            part_file = 'refract_ocff'
        elif particle == 'oceanic':
            part_file = 'refract_oceanic'
        elif particle == 'soot':
            part_file = 'refract_soot_bond'
        elif particle == 'biogenic':
            part_file = 'refract_biogenic'
        elif particle == 'NaCl':
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
    n_murk = (rel_vol['ammonium_sulphate'] * n['ammonium_sulphate']) + \
             (rel_vol['ammonium_nitrate'] * n['ammonium_nitrate']) + \
             (rel_vol['organic_carbon'] * n['organic_carbon'])

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


# plotting

def plot_one_aer_i(r_md_microm, Q_dry, ceil_lambda, ceil_lambda_str, all_aer_constits, aer_names, savedir, extra=''):

    """Plot the absolute Q_dry_ext values"""

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    lam_colours = create_colours(len(ceil_lambda) -1)

    for ax_i, aer_i in zip(ax.flatten(), all_aer_constits + ['MURK']):

        for lam_i, lam_i_colour in zip(ceil_lambda_str, lam_colours):

            ax_i.semilogx(r_md_microm, Q_dry[aer_i][lam_i], label=lam_i + ' nm', color=lam_i_colour)
            ax_i.set_xlim([0.05, 4.0])
            ax_i.set_ylim([0.0, 5.0])

        # subplot prettify
        ax_i.set_title(aer_i)



    # figure prettify
    ax_main = eu.fig_majorAxis(fig)

    ax_main.set_xlabel(r'$r_{md} \/\mathrm{[\mu m]}$', labelpad=5)
    ax_main.set_ylabel(r'$Q_{ext}(dry)$')
    ax.flatten()[1].legend(fontsize=8, bbox_to_anchor=(1.07, 1), loc=2, borderaxespad=0.0)
    fig.suptitle('Absolute (lam_i)')

    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.9, right=0.8)

    plt.savefig(savedir + 'Q_extdry_multiLam_abs_sensitivity'+extra+'.png')
    plt.close()



    return

def plot_absolute(r_md_microm, Q_dry, ceil_lambda, ceil_lambda_str, all_aer_constits, aer_names, savedir, extra=''):

    """Plot the absolute Q_dry_ext values"""

    fig, ax = plt.subplots(3, 2, figsize=(8, 7))

    lam_colours = create_colours(len(ceil_lambda) -1)

    for ax_i, aer_i in zip(ax.flatten(), all_aer_constits + ['MURK']):

        for lam_i, lam_i_colour in zip(ceil_lambda_str, lam_colours):

            ax_i.semilogx(r_md_microm, Q_dry[aer_i][lam_i], label=lam_i + ' nm', color=lam_i_colour)
            ax_i.set_xlim([0.05, 4.0])
            ax_i.set_ylim([0.0, 5.0])

        # subplot prettify
        ax_i.set_title(aer_i)



    # figure prettify
    ax_main = eu.fig_majorAxis(fig)

    ax_main.set_xlabel(r'$r_{md} \/\mathrm{[\mu m]}$', labelpad=5)
    ax_main.set_ylabel(r'$Q_{ext}(dry)$')
    ax.flatten()[1].legend(fontsize=8, bbox_to_anchor=(1.07, 1), loc=2, borderaxespad=0.0)
    fig.suptitle('Absolute (lam_i)')

    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.9, right=0.8)

    plt.savefig(savedir + 'Q_extdry_multiLam_abs_sensitivity'+extra+'.png')
    plt.close()



    return

def plot_diff(r_md_microm, Q_diff, ceil_lambda, ceil_lambda_str, all_aer_constits, aer_names, savedir, extra=''):

    """Plot the absolute Q_dry_ext values"""

    fig, ax = plt.subplots(3, 2, figsize=(8, 7))

    lam_colours = create_colours(len(ceil_lambda) -1)

    for ax_i, aer_i in zip(ax.flatten(), all_aer_constits + ['MURK']):

        for lam_i, lam_i_colour in zip(ceil_lambda_str, lam_colours):

            ax_i.semilogx(r_md_microm, Q_diff[aer_i][lam_i], label=lam_i + ' nm', color=lam_i_colour)
            ax_i.set_xlim([0.05, 4.0])
            # ax_i.set_ylim([0.0, 5.0])

        # subplot prettify
        ax_i.set_title(aer_i)



    # figure prettify
    ax_main = eu.fig_majorAxis(fig)

    ax_main.set_xlabel(r'$r_{md} \/\mathrm{[\mu m]}$', labelpad=5)
    ax_main.set_ylabel(r'$Q_{ext}(dry)$')
    ax.flatten()[1].legend(fontsize=8, bbox_to_anchor=(1.07, 1), loc=2, borderaxespad=0.0)
    fig.suptitle('Difference (lam_i - lam_910)')

    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.9, right=0.8)

    plt.savefig(savedir + 'Q_extdry_multiLam_diff_sensitivity'+extra+'.png')
    plt.close()



    return

def plot_ratio(r_md_microm, Q_ratio, ceil_lambda, ceil_lambda_str, aer_plot_order , aer_names, savedir, extra=''):

    """Plot the absolute Q_dry_ext values"""

    fig, ax = plt.subplots(3, 2, figsize=(8, 7))

    # For 895 - 915 nm
    # # lam_colours = create_colours(len(ceil_lambda) -1)
    # lam_colours = create_colours(len(ceil_lambda))
    #
    # for ax_i, aer_i in zip(ax.flatten(), aer_plot_order):
    #
    #     for lam_i, lam_i_colour in zip(ceil_lambda_str, lam_colours):
    #
    #         ax_i.semilogx(r_md_microm, Q_ratio[aer_i][lam_i], label=lam_i + ' nm', color=lam_i_colour)
    #         ax_i.set_xlim([0.05, 10.0])
    #         ax_i.set_ylim([0.7, 1.4])


    for ax_i, aer_i in zip(ax.flatten(), aer_plot_order):

        ax_i.axhline(y=1, linestyle='--', color='green', alpha=0.5)
        ax_i.semilogx(r_md_microm, Q_ratio[aer_i]['1064'])
        ax_i.set_xlim([0.05, 10.0])
        ax_i.set_ylim([0.4, 2.0])

        # subplot prettify
        ax_i.set_title(aer_names[aer_i], fontsize=12)
        ax_i.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))



    # figure prettify
    ax_main = eu.fig_majorAxis(fig)

    ax_main.set_xlabel(r'$r_{md} \/\mathrm{[\mu m]}$', fontsize=16, labelpad=5)
    ax_main.set_ylabel(r'$\frac{Q_{ext,dry}(\lambda=1064\/nm)}{Q_{ext,dry}(\lambda=905\/nm)}$', labelpad=15, fontsize=19)
    #    ax_main.set_ylabel(r'$\frac{Q_{ext,dry}(\lambda=i)}{Q_{ext,dry}(\lambda=905\/nm)}$', labelpad=15, fontsize=19)
    # ax.flatten()[1].legend(fontsize=9, bbox_to_anchor=(1.07, 1), loc=2, borderaxespad=0.0)
    # fig.suptitle('Ratio (lam_i / lam_905)')

    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.9, right=0.8)

    plt.savefig(savedir + 'Q_extdry_multiLam_ratio_sensitivity_10micronscale_1064-905'+extra+'.png')
    plt.close()

def main():


    # -------------------------------------------------------------------
    # Setup

    # setup
    # ceil_lambda = [0.91e-06] # [m]
    # ceil_lambda = np.arange(0.69e-06, 1.19e-06, 0.05e-06) # [m]
    # ceil_lambda = np.arange(8.95e-07, 9.16e-07, 1.0e-09) # [m]
    ceil_lambda = np.array([905e-09, 1064e-09])

    # work around, Issues with rounding errors when conv from int or float to str.
    #ceil_lambda_str = ['895','896', '897','898','899','900','901','902','903','904','905','906','907',
    #                   '908','909','910','911','912','913','914','915']
    # ceil_lambda_str = ['%d' % i for i in ceil_lambda]
    ceil_lambda_str = ['905', '1064']


    # extra =''
    extra = '_largelam'

    # directories
    savedir = '/home/nerc/Documents/MieScatt/figures/'
    datadir = '/home/nerc/Documents/MieScatt/data/'

    # aerosol with relative volume - average from the 4 Haywood et al 2008 flights
    rel_vol = {'ammonium_sulphate': 0.295,
               'ammonium_nitrate': 0.325,
                'organic_carbon': 0.38}

    # all the aerosol types
    # all_aer = ['ammonium_sulphate', 'ammonium_nitrate', 'organic_carbon', 'oceanic', 'biogenic', 'NaCl', 'soot']
    aer_names = {'ammonium_sulphate': 'Ammonium sulphate', 'ammonium_nitrate': 'Ammonium nitrate',
                'organic_carbon': 'Organic carbon', 'NaCl': 'Generic NaCl', 'soot':'Soot', 'MURK': 'MURK'}
    # all_aer = {'ammonium_sulphate': 'red', 'ammonium_nitrate':'orange', 'organic_carbon': 'green',
    #            'biogenic': 'cyan', 'NaCl': 'magenta', 'soot': 'brown'}
    all_aer = {'ammonium_sulphate': 'red', 'ammonium_nitrate':'orange', 'organic_carbon': 'green',
               'NaCl': 'magenta', 'soot': 'brown', 'MURK': 'black'}
    all_aer_constits = ['ammonium_sulphate', 'ammonium_nitrate', 'organic_carbon', 'NaCl', 'soot']
    aer_plot_order = ['ammonium_sulphate', 'ammonium_nitrate', 'organic_carbon', 'MURK', 'NaCl', 'soot']

    # create dry size distribution [m]
    # r_md_microm = np.arange(0.03, 5.001, 0.001) # .shape() = 4971
    # r_md_microm = np.arange(0.000 + step, 1.000 + step, step), when step = 0.005, .shape() = 200
    step = 0.005
    r_md_microm = np.arange(0.000 + step, 10.000 + step, step)
    r_md = r_md_microm * 1.0e-06


    # define array to store Q for each wavelength
    Q_dry = {}
    Q_diff = {}
    Q_ratio = {}


    # make each, first level key (i.e. Q_dry[firstkey]) the aerosol type, followed later by the wavelength [nm].
    for aer_i in all_aer.iterkeys():

        Q_dry[aer_i] = {}
        Q_diff[aer_i] = {}
        Q_ratio[aer_i] = {}

        for lam_i in ceil_lambda_str:

            Q_dry[aer_i][lam_i] = []
            Q_diff[aer_i][lam_i] = []
            Q_ratio[aer_i][lam_i] = []



    # save the Q(dry) curve for MURK?
    savedata = False

    # -----------------------------------------------
    # Calculate Q for each lambda
    # -----------------------------------------------

    for lam, lam_str in zip(ceil_lambda, ceil_lambda_str):

        print '\n lam_i = ' + lam_str + ' nm'
        # -------------------------------------------------------------------
        # Process

        # calculate complex index of refraction for MURK species
        # output n is complex index of refraction (n + ik)
        n_aerosol = calc_n_aerosol(all_aer_constits, lam)

        # NOTE: Able to use volume in MURK equation instead of mass because, if mass is evenly distributed within a volume
        # then taking x of the mass = taking x of the volume.
        # after calculating volumes used in MURK, can find relative % and do volume mixing.

        # bulk complex index of refraction (CIR) for the MURK species using volume mixing method
        n_murk = calc_n_murk(rel_vol, n_aerosol)
        n_aerosol['MURK'] = n_murk
        # n_murk = complex(1.53, 0.007) - this is for 550 nm
        # n_store += [n_murk]

        # complex indices of refraction (n = n(bar) - ik) at ceilometer wavelength (910 nm) Hesse et al 1998
        # n_water, _ = linear_interpolate_n('water', lam)

        # calculate size parameter for dry and wet
        x_dry = (2.0 * np.pi * r_md)/lam
        # x_store += [x_dry]


        # calculate swollen index of refraction using MURK
        # n_swoll = CIR_Hanel(n_water, n_murk, r_md, r_m)


        # Calc extinction efficiency for dry aerosol (using r_md!!!! NOT r_m)
        for aer_i, n_i in n_aerosol.iteritems():
            all_particles_dry = [Mie(x=x_i, m=n_i) for x_i in x_dry]
            Q_dry[aer_i][lam_str] = np.array([particle.qext() for particle in all_particles_dry])

    # once 905 has been calcualted
    for aer_i, value in Q_dry.iteritems():
        for lam_str_i in ceil_lambda_str:
            Q_diff[aer_i][lam_str_i]  = Q_dry[aer_i][lam_str_i] - Q_dry[aer_i]['905']
            Q_ratio[aer_i][lam_str_i] = Q_dry[aer_i][lam_str_i] / Q_dry[aer_i]['905']

    # qsca, qabs are alternatives to qext

    # -----------------------------------------------
    # Post processing, saving and plotting
    # -----------------------------------------------


    # plot
    # plot_one_aer_i

    extra = ''#'895-915'

    plot_absolute(r_md_microm, Q_dry, ceil_lambda, ceil_lambda_str, all_aer_constits, aer_names, savedir, extra)

    plot_diff(r_md_microm, Q_diff, ceil_lambda, ceil_lambda_str, all_aer_constits, aer_names, savedir, extra)

    plot_ratio(r_md_microm, Q_ratio, ceil_lambda, ceil_lambda_str, aer_plot_order , aer_names, savedir, extra)


    # plot the radius
    # plot_radius(savedir, r_md, r_m)

    print 'END PROGRAM'

if __name__ == '__main__':
    main()


