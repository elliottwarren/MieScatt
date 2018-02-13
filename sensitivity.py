__author__ = 'nerc'


from mie_scattering import *
import matplotlib.pyplot as plt
import numpy as np

def main():


    """
    Carry out a sensitivity analysis on the Mie scattering code

    :return: plots Mie scattering curve of murk, deliquecent and coated spheres at 80 % RH.
    """


    import numpy as np
    from pymiecoated import Mie
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------
    # Setup

    # setup
    ceil_lambda = 0.91e-06 # [m]
    B = 0.14
    RH_crit = 0.38

    # directories
    savedir = '/home/nerc/Documents/MieScatt/figures/sensitivity/'

    # aerosol with relative volume - varying manually - first one is average from haywood et al., 2008
    rel_vol = {'ammonium_sulphate': [0.295, 0.80, 0.10, 0.10],
               'ammonium_nitrate': [0.325, 0.10, 0.80, 0.10],
                'organic_carbon': [0.38, 0.10, 0.10, 0.80]}


    # create dry size distribution [m]
    r_md_microm = np.arange(0.03, 5.001, 0.001)
    r_md = r_md_microm*1.0e-06

    # RH array [fraction]
    # This gets fixed for each Q iteration (Q needs to be recalculated for each RH used)
    RH = 0.8

    # densities of MURK constituents # [kg m-3]
    # range of densities of org carb is massive (0.625 - 2 g cm-3)
    # Haywood et al 2003 use 1.35 g cm-3 but Schkolnik et al., 2006 lit review this and claim 1.1 g cm-3
    dens_amm_sulph = 1770
    dens_amm_nit = 1720
    dens_org_carb = 1100 # NOTE ABOVE

    # -------------------------------------------------------------------
    # Process

    # calculate complex index of refraction for each particle
    # output n is complex index of refraction (n + ik)
    n_aerosol = calc_n_aerosol(rel_vol, ceil_lambda)

    # NOTE: Able to use volume in MURK equation instead of mass because, if mass is evenly distributed within a volume
    # then taking x of the mass = taking x of the volume.
    # after calculating volumes used in MURK, can find relative % and do volume mixing.

    n_mixed = []
    for i in range(len(rel_vol['ammonium_sulphate'])):

        # extract out just this one set of relative amounts
        rel_vol_i = {}
        for key in rel_vol.keys():
            rel_vol_i[key] = rel_vol[key][i]

        # uses relative amounts in the MURK equation!
        n_mixed += [calc_n_murk(rel_vol_i, n_aerosol)]


    # complex indices of refraction (n = n(bar) - ik) at ceilometer wavelength (910 nm) Hesse et al 1998
    n_water, _ = linear_interpolate_n('water', ceil_lambda)

    # swell particles using FO method
    # rm = np.ma.ones(RH.shape) - (B / np.ma.log(RH_ge_RHcrit))
    r_m = 1 - (B / np.log(RH))
    r_m2 = np.ma.power(r_m, 1. / 3.)
    r_m = np.ma.array(r_md) * r_m2
    r_m_microm = r_m * 1.0e06

    # calculate size parameter for dry and wet
    x_dry = (2.0 * np.pi * r_md)/ceil_lambda
    x_wet = (2.0 * np.pi * r_m)/ceil_lambda


    # # calculate swollen index of refraction using MURK
    # n_swoll = CIR_Hanel(n_water, n_murk, r_md, r_m)

    for j in range(len(n_mixed)):


        # Calc extinction efficiency for dry aerosol (using r_md!!!! NOT r_m)
        all_particles_dry = [Mie(x=x_i, m=n_mixed[j]) for x_i in x_dry]
        Q_dry = np.array([particle.qext() for particle in all_particles_dry])

        # use proportions of each to scale the colours on the plot
        colours = [rel_vol['ammonium_sulphate'][j],
                   rel_vol['ammonium_nitrate'][j],
                   rel_vol['organic_carbon'][j]]

        lab = 'AS=' + str(rel_vol['ammonium_sulphate'][j]) + '; ' +\
            'AN=' + str(rel_vol['ammonium_nitrate'][j]) + '; ' +\
            'OC=' + str(rel_vol['organic_carbon'][j])

        # plot it
        plt.semilogx(r_md_microm, Q_dry, color=colours, label=lab)


    plt.title('lambda = interpolated to ' + str(ceil_lambda) + 'm')
    plt.xlabel('radius [micrometer]')
    plt.xlim([0.03, 5.0])
    plt.ylim([0.0, 5.0])
    plt.ylabel('Q')
    plt.legend(fontsize=9)
    plt.savefig(savedir + 'Q_ext_murk_sensitivity_' + str(ceil_lambda) + 'lam.png')
    # plt.close()


    print 'END PROGRAM'

if __name__ == '__main__':
    main()


