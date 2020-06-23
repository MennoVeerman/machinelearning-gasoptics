import numpy as np
import netCDF4 as nc
import shutil
import subprocess
import os

rrtmgp_build_dir = "../../rte-rrtmgp-cpp/build/"
rrtmgp_coeff_dir = "../../rte-rrtmgp-cpp/rte-rrtmgp/rrtmgp/data/"

if not os.path.exists('test_rte_rrtmgp'): os.symlink(rrtmgp_build_dir+'test_rte_rrtmgp', 'test_rte_rrtmgp')
if not os.path.exists('coefficients_lw.nc'): os.symlink(rrtmgp_coeff_dir+'rrtmgp-data-lw-g256-2018-12-04.nc', 'coefficients_lw.nc')
if not os.path.exists('coefficients_sw.nc'): os.symlink(rrtmgp_coeff_dir+'rrtmgp-data-sw-g224-2018-12-04.nc', 'coefficients_sw.nc')
nfile = 5

#rrtmgp-data-sw-g224-2018-12-04.nc
if not os.path.exists('data'):
    os.mkdir('data')

#create input data
subprocess.call(['python', 'rfmip_createinput.py', '--frand', '--finvt', '--nfile=%s'%nfile])

#run rrtmgp
for ifile in range(nfile):
    shutil.copyfile('data/rte_rrtmgp_input_{:03d}.nc'.format(ifile), 'rte_rrtmgp_input.nc')

#    subprocess.run(['./test_rte_rrtmgp'])
#    shutil.move('rte_rrtmgp_output.nc', 'rte_rrtmgp_output_expt_{:02d}.nc'.format(expt))
#    print(' ')
#
#
## Prepare the output file.
#for expt in range(expts):
#    # Save all the input data to NetCDF
#    nc_file = nc.Dataset('rte_rrtmgp_output_expt_{:02d}.nc'.format(expt), mode='r')
#    
#    nc_file_rld = nc.Dataset('rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
#    nc_file_rlu = nc.Dataset('rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
#    nc_file_rsd = nc.Dataset('rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
#    nc_file_rsu = nc.Dataset('rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
#    
#    nc_file_rld.variables['rld'][expt,:,:] = nc_file.variables['lw_flux_dn'][:,:].transpose()
#    nc_file_rlu.variables['rlu'][expt,:,:] = nc_file.variables['lw_flux_up'][:,:].transpose()
#    nc_file_rsd.variables['rsd'][expt,:,:] = nc_file.variables['sw_flux_dn'][:,:].transpose()
#    nc_file_rsu.variables['rsu'][expt,:,:] = nc_file.variables['sw_flux_up'][:,:].transpose()
#    
#    nc_file.close()
#    nc_file_rld.close()
#    nc_file_rlu.close()
#    nc_file_rsd.close()
#    nc_file_rsu.close()

