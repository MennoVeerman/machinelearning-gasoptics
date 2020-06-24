import numpy as np
import netCDF4 as nc
import shutil
import subprocess
import os

nfile = 5
rrtmgp_path =  "/path/to/rrtmgp-rte-cpp/folder"
rrtmgp_build_dir = "rrtmgp/path/build/"
rrtmgp_coeff_dir = "rrtmgp/path/rte-rrtmgp/rrtmgp/data/"
rrtmgp_cloud_dir = "rrtmgp/path/rte-rrtmgp/extensions/cloud_optics/"
if not os.path.exists('test_rte_rrtmgp'): os.symlink(rrtmgp_build_dir+'test_rte_rrtmgp', 'test_rte_rrtmgp')
if not os.path.exists('coefficients_lw.nc'): os.symlink(rrtmgp_coeff_dir+'rrtmgp-data-lw-g256-2018-12-04.nc', 'coefficients_lw.nc')
if not os.path.exists('coefficients_sw.nc'): os.symlink(rrtmgp_coeff_dir+'rrtmgp-data-sw-g224-2018-12-04.nc', 'coefficients_sw.nc')
if not os.path.exists('cloud_coefficients_lw.nc'): os.symlink(rrtmgp_cloud_dir+'rrtmgp-cloud-optics-coeffs-lw.nc', 'cloud_coefficients_lw.nc')
if not os.path.exists('cloud_coefficients_sw.nc'): os.symlink(rrtmgp_cloud_dir+'rrtmgp-cloud-optics-coeffs-sw.nc', 'cloud_coefficients_sw.nc')
if not os.path.exists('cloud_coefficients_lw.nc'): os.symlink(rrtmgp_cloud_dir+'rrtmgp-cloud-optics-coeffs-lw.nc', 'cloud_coefficients_lw.nc')
subprocess.call(['python', 'rfmip_download.py'])

#rrtmgp-data-sw-g224-2018-12-04.nc
if not os.path.exists('data'):
    os.mkdir('data')

#create input data
subprocess.call(['python', 'rfmip_createinput.py', '--frand', '--finvt', '--nfile=%s'%nfile])

#run rrtmgp
for ifile in range(nfile):
    shutil.copyfile('data/rte_rrtmgp_input_{:03d}.nc'.format(ifile), 'rte_rrtmgp_input.nc')
    subprocess.call(['./test_rte_rrtmgp', '--no-cloud-optics'])
    shutil.move('rte_rrtmgp_output.nc', 'data/rte_rrtmgp_output_{:03d}.nc'.format(ifile))

os.system('ncecat data/*input*  -O inputs.nc')
os.system('ncecat data/*output* -O outputs.nc')

