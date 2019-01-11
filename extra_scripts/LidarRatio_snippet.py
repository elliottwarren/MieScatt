__author__ = 'Elliott_Warren'


import numpy as np
import matplotlib.pyplot as plt
from pymiecoated import Mie
import ellUtils as eu

from calc_lidar_ratio_numdist import read_n_data

def test_lidar_computation(ceil_lambda, r_md_m):

    """
    Test my computation of the lidar ratio against Franco's. Done for a monodisperse, soot(like?) aerosol
    :param ceil_lambda:
    :param r_md_m:
    :return:
    """

    import ellUtils as eu

    # Testing lidar ratio computation

    # read in Franco's computation of the lidar ratio CIR=1.47 + 0.099i, lambda=905nm
    lr_f = eu.netCDF_read('/home/nerc/Documents/MieScatt/testing/lr_1.47_0.099_0.905.nc', ['DIAMETER','LIDAR_RATIO'])

    step = 0.005
    r_range_um = np.arange(0.000 + step, 10.000 + step, step)
    r_range_m = r_range_um * 1.0e-06
    x_range = (2.0 * np.pi * r_range_m)/ceil_lambda[0]


    # calculate Q_back and Q_ext from the avergae r_md and n_mixed
    #S_r = lidar ratio
    S_r = np.empty(len(r_range_m))
    S_r[:] = np.nan
    for r_idx, r_i in enumerate(r_range_m):

        x_i = x_range[r_idx]  # size parameter_i
        n_i = complex(1.47 + 0.0j)  # fixed complex index of refraction i
        # n_i = complex(1.47 + 0.099j)  # fixed complex index of refraction i for soot

        # print loop progress
        if r_idx in np.arange(0, 2100, 100):
            print r_idx


        particle = Mie(x=x_i, m=n_i)
        Q_ext = particle.qext()
        Q_back = particle.qb()
        Q_back_alt = Q_back / (4.0 * np.pi)

        # #Q_back = particle.qb()
        # S12 = particle.S12(-1)
        # S11 = S12[0].imag
        # S22 = S12[1].imag
        # Q_back_fancy = ((np.abs(S11)**2) + (np.abs(S22)**2))/(2 * np.pi * (x_i**2))


        # calculate the lidar ratio
        # S_t = Q_ext / Q_back
        S_r[r_idx] = Q_ext / Q_back_alt


    # simple plot of S
    fig, ax = plt.subplots(1,1, figsize=(8,7))
    plt.loglog(r_range_um * 2, S_r, label='mine') # diameter [microns]
    plt.loglog(lr_f['DIAMETER'], lr_f['LIDAR_RATIO'], label='Franco''s')

    for aer_i, r_md_m_aer_i in r_md_m.iteritems():
        for r_i in r_md_m_aer_i:
            plt.vlines(r_i, 1 ,1e6, linestyle='--', alpha=0.5)

    plt.xlim([0.01, 100.0])
    plt.ylim([1.0, 10.0e7])
    plt.ylabel('Lidar Ratio')
    plt.xlabel('Diameter [microns]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(maindir + 'figures/LidarRatio/' + 'quickplot_S_vs_r_with_rbin_lines.png')
    plt.close(fig)

    return

# directories
maindir = '/home/nerc/Documents/MieScatt/'
datadir = '/home/nerc/Documents/MieScatt/data/'

# save dir
savedir = maindir + 'figures/LidarRatio/'

ceil_lambda = 355.0e-09
# ceil_lambda = 355.0e-09
# ceil_lambda = 532.0e-09
# ceil_lambda = 1064e-09

# lambda in nanometers and as a string
ceil_lambda_str_nm = str(ceil_lambda * 1.0e09) + 'nm'

step = 0.005
r_range_um = np.arange(0.000 + step, 10.000 + step, step)
r_range_m = r_range_um * 1.0e-06
x_range = (2.0 * np.pi * r_range_m)/ceil_lambda

# step = 0.05
# num_range = np.arange(1.0, 10.0, step)
# r_range_m = np.hstack([num_range*(10**i) for i in range(-11, -5)])
# r_range_um = r_range_m * 1e6
# x_range = (2.0 * np.pi * r_range_m)/ceil_lambda

# aerosol species
aer_particles = ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CORG', 'CBLK']

# aer names in the complex index of refraction files
aer_names = {'(NH4)2SO4': 'Ammonium sulphate', 'NH4NO3': 'Ammonium nitrate',
            'CORG': 'Organic carbon', 'NaCl': 'Generic NaCl', 'CBLK':'Soot', 'MURK': 'MURK'}

# read in the complex index of refraction for the different aerosols and water
n_species = read_n_data(aer_particles, aer_names, [ceil_lambda], getH2O=True)

# # lambda = 905 nm!
# n_species_905 = {'(NH4)2SO4': (1.5328749999999998+2.3125000000000006e-07j),
#                  'CBLK': (1.85+0.6976j),
#                  'CORG': (1.5404635+0.006j),
#                  'H2O': (1.328+6.00800000000001e-07j),
#                  'NH4NO3': (1.611+1.7041000000000008e-06j),
#                  'NaCl': (1.5662+1e-07j)}

aer_particles = ['(NH4)2SO4', 'NH4NO3', 'NaCl', 'CORG', 'CBLK']

aer_labels = {'(NH4)2SO4': 'Ammonium sulphate',
             'NH4NO3': 'Ammonium nitrate',
             'NaCl': 'NaCl',
             'CORG': 'Organic carbon',
             'CBLK': 'Elemental carbon',
             'H2O': 'Pure water'}


## LINEPLOT - S for each species at 905 nm for monodisperse aerosol, with legend


#S_r = lidar ratio
S_r = {}

for aer_i in n_species:

    print ''
    print aer_i

    S_r[aer_i] = np.empty(len(r_range_m))
    S_r[aer_i][:] = np.nan

    # n_i = complex(1.47 + 0.099j)  # fixed complex index of refraction i
    n_i = n_species[aer_i]

    for r_idx, r_i in enumerate(r_range_m):

        x_i = x_range[r_idx]  # size parameter_i

        # print loop progress
        if r_idx in np.arange(0, 2100, 100):
            print r_idx

        # calculate Q_back and Q_ext efficiency for current size parameter and complex index of refraction
        particle = Mie(x=x_i, m=n_i)
        Q_ext = particle.qext()
        Q_back = particle.qb()
        Q_back_alt = Q_back / (4.0 * np.pi)

        # calculate the lidar ratio
        S_r[aer_i][r_idx] = Q_ext / Q_back_alt


# simple plot of S
fig, ax = plt.subplots(1,1, figsize=(7, 4))
for key, data in S_r.iteritems():
    plt.loglog(r_range_um * 2.0, data, label=aer_labels[key]) # diameter [microns]
# plt.xlim([0.01, 100.0])
plt.ylim([1.0, 10.0e5])
plt.xlim([0.01, 20])
plt.ylabel(r'$Lidar \/Ratio \/[sr]$')
plt.xlabel(r'$Diameter \/[\mu m]$')
plt.axvline(x=0.08, color='black', linestyle='--', linewidth=2)
plt.axvline(x=0.8, color='black', linestyle='--', linewidth=2)
plt.legend(loc='upper center', fontsize=10)
eu.add_at(ax, '(a) 355 nm')
# plt.text('(a) 355 nm', bbox={'edgecolor':'none'})
#plt.suptitle('lambda=' + ceil_lambda_str_nm)
plt.tight_layout()
# plt.show()
plt.savefig(savedir + 'quickplot_S_vs_r_bySpecies_'+ ceil_lambda_str_nm +'.png')
# plt.close(fig)



# -----------------------------------------------------

#S_r = lidar ratio
# S_r = np.empty(len(r_range_m))
# S_r[:] = np.nan
# for r_idx, r_i in enumerate(r_range_m):
#
#     x_i = x_range[r_idx]  # size parameter_i
#     n_i = complex(1.47 + 0.099j)  # fixed complex index of refraction i
#
#     # print loop progress
#     if r_idx in np.arange(0, 2100, 100):
#         print r_idx
#
#     # calculate Q_back and Q_ext efficiency for current size parameter and complex index of refraction
#     particle = Mie(x=x_i, m=n_i)
#     Q_ext = particle.qext()
#     Q_back = particle.qb()
#     Q_back_alt = Q_back / (4.0 * np.pi)
#
#     # calculate the lidar ratio
#     S_r[r_idx] = Q_ext / Q_back
#
#
# # simple plot of S
# fig, ax = plt.subplots(1,1, figsize=(6,5))
# plt.loglog(r_range_um * 2.0, S_r) # diameter [microns]
# plt.xlim([0.01, 100.0])
# plt.ylim([1.0, 10.0e7])
# plt.ylabel('Lidar Ratio')
# plt.xlabel('Diameter [microns]')
# plt.tight_layout()
# plt.show()
# #plt.savefig(savedir + 'quickplot_S_vs_r.png')
# #plt.close(fig)

# ---------------------------------------------------------

# # simple plot of S
# fig, ax = plt.subplots(1,1, figsize=(6,6))
# plt.plot_date(date_range, S)
# plt.savefig(savedir + 'quickplot.png')
# plt.close(fig)
#
# --------------------------

# Test my computation of the lidar ratio against Franco's - soot(like?) monodisperse aerosol
# test_lidar_computation(ceil_lambda, r_md_m)


