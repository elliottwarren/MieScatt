"""
Carry out a simple Mie scattering calculation using the pymiecoated code used in Lally (2014).
Complex index of refraction = CIR

"""

__author__ = 'nerc'


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

        from numpy import array

        # particle data dir
        part_datadir = '/media/sf_HostGuestShared/MieScatt/complex index of refraction/'

        # find particle filename
        if particle == 'ammonium_nitrate':
            part_file = 'refract_ammoniumnitrate'
        elif particle == 'ammonium_sulphate':
            part_file = 'refract_ammoniumsulphate'
        elif particle == 'organic_carbon':
            part_file = 'refract_ocff'
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

        while line != '*END':
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

def calc_n_aerosol(rel_vol, ceil_lambda):

    """
    Calculate n for each of the aerosol in rel_vol
    :param rel_vol:
    :return:
    """

    n_aerosol = {}
    for key in rel_vol.iterkeys():
        n_aerosol[key], _ = linear_interpolate_n(key, ceil_lambda)

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
    ceil_lambda = [0.91e-06] # [m]
    # ceil_lambda = np.arange(0.69e-06, 1.19e-06, 0.05e-06) # [m]

    # directories
    savedir = '/home/nerc/Documents/MieScatt/figures/'
    datadir = '/home/nerc/Documents/MieScatt/data/'

    # aerosol with relative volume - average from the 4 Haywood et al 2008 flights
    rel_vol = {'ammonium_sulphate': 0.295,
               'ammonium_nitrate': 0.325,
                'organic_carbon': 0.38}


    # create dry size distribution [m]
    # r_md_microm = np.arange(0.03, 5.001, 0.001) # .shape() = 4971
    # r_md_microm = np.arange(0.000 + step, 1.000 + step, step), when step = 0.005, .shape() = 200
    step = 0.0005
    r_md_microm = np.arange(0.000 + step, 2.000 + step, step)
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

    # -----------------------------------------------
    # Calculate Q for each lambda
    # -----------------------------------------------

    for lam in ceil_lambda:

        # -------------------------------------------------------------------
        # Process

        # calculate complex index of refraction for MURK species
        # output n is complex index of refraction (n + ik)
        n_aerosol = calc_n_aerosol(rel_vol, lam)

        # NOTE: Able to use volume in MURK equation instead of mass because, if mass is evenly distributed within a volume
        # then taking x of the mass = taking x of the volume.
        # after calculating volumes used in MURK, can find relative % and do volume mixing.

        # bulk complex index of refraction (CIR) for the MURK species using volume mixing method
        n_murk = calc_n_murk(rel_vol, n_aerosol)
        # n_murk = complex(1.53, 0.007) - makes no sense as this is for 550 nm
        # n_store += [n_murk]

        # complex indices of refraction (n = n(bar) - ik) at ceilometer wavelength (910 nm) Hesse et al 1998
        # n_water, _ = linear_interpolate_n('water', lam)

        # swell particles using FO method
        # rm = np.ma.ones(RH.shape) - (B / np.ma.log(RH_ge_RHcrit))
        # r_m = 1 - (B / np.log(RH))
        # r_m2 = np.ma.power(r_m, 1. / 3.)
        # r_m = np.ma.array(r_md) * r_m2
        # r_m_microm = r_m * 1.0e06

        # calculate size parameter for dry and wet
        x_dry = (2.0 * np.pi * r_md)/lam
        x_store += [x_dry]
        # x_wet = (2.0 * np.pi * r_m)/lam


        # calculate swollen index of refraction using MURK
        # n_swoll = CIR_Hanel(n_water, n_murk, r_md, r_m)


        # Calc extinction efficiency for dry aerosol (using r_md!!!! NOT r_m)
        all_particles_dry = [Mie(x=x_i, m=n_murk) for x_i in x_dry]
        Q_dry += [np.array([particle.qext() for particle in all_particles_dry])]

    # -----------------------------------------------
    # Post processing, saving and plotting
    # -----------------------------------------------


    # if running for single 910 nm wavelength, save the calculated Q
    if type(ceil_lambda) == list:
        if ceil_lambda[0] == 9.1e-07:
            # save Q curve and radius [m]
            np.savetxt(datadir + 'calculated_Q_ext_910nm.csv', np.transpose(np.vstack((r_md, Q_dry))), delimiter=',', header='radius,Q_ext')


    # plot
    fig = plt.figure(figsize=(7, 4.5))

    for Q_i, lam in zip(Q_dry, ceil_lambda):

        # plot it
        plt.semilogx(r_md_microm, Q_i, label=str(lam) + 'm')
        # plt.semilogx(r_md_microm, Q_dry, label='dry murk', color=[0,0,0])
        # plt.semilogx(r_m_microm, Q_del, label='deliquescent murk (RH = ' + str(RH) + ')')
        # plt.semilogx(r_m_microm, Q_coat, label='coated murk (RH = ' + str(RH) + ')')

    # # average Q_dry if multiple Q_drys were calculated
    # if Q_dry.__len__() != 1:
    #     q = np.array(Q_dry)
    #     Q_dry_avg = np.mean(q, axis=0)
    #     ax = plt.semilogx(r_md_microm, Q_dry_avg, label='average', color='black', linewidth=2)

    plt.title('lambda = ' + str(ceil_lambda[0]) + '-' + str(ceil_lambda[-1]) + 'm, n = murk')
    plt.xlabel('radius [micrometer]', labelpad=-5)
    plt.xlim([0.05, 5.0])
    plt.ylim([0.0, 5.0])
    #plt.xlim([0.01, 0.2])
    #plt.ylim([0.0, 0.1])
    plt.ylabel('Q_ext')
    plt.legend(fontsize=8, loc='best')
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.grid(b=True, which='minor', color=[0.85, 0.85, 0.85], linestyle='--')
    plt.savefig(savedir + 'Q_ext_murk_' + str(ceil_lambda[0]) + '-' + str(ceil_lambda[-1]) + 'lam.png')
    plt.tight_layout()
    plt.close()

    # plot the radius
    # plot_radius(savedir, r_md, r_m)

    print 'END PROGRAM'

if __name__ == '__main__':
    main()


