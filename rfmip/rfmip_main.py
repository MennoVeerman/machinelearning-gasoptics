import numpy as np
import netCDF4 as nc
import shutil
import subprocess
import os

nfile = 25
rrtmgp_path =  "/path/to/rrtmgp-rte-cpp/folder/"
rrtmgp_build_dir = rrtmgp_path+"build/"
rrtmgp_coeff_dir = rrtmgp_path+"rte-rrtmgp/rrtmgp/data/"
rrtmgp_cloud_dir = rrtmgp_path+"rte-rrtmgp/extensions/cloud_optics/"
if not os.path.exists('test_rte_rrtmgp'): os.symlink(rrtmgp_build_dir+'test_rte_rrtmgp', 'test_rte_rrtmgp')
if not os.path.exists('coefficients_lw.nc'): os.symlink(rrtmgp_coeff_dir+'rrtmgp-data-lw-g256-2018-12-04.nc', 'coefficients_lw.nc')
if not os.path.exists('coefficients_sw.nc'): os.symlink(rrtmgp_coeff_dir+'rrtmgp-data-sw-g224-2018-12-04.nc', 'coefficients_sw.nc')
if not os.path.exists('cloud_coefficients_sw.nc'): os.symlink(rrtmgp_cloud_dir+'rrtmgp-cloud-optics-coeffs-sw.nc', 'cloud_coefficients_sw.nc')
if not os.path.exists('cloud_coefficients_lw.nc'): os.symlink(rrtmgp_cloud_dir+'rrtmgp-cloud-optics-coeffs-lw.nc', 'cloud_coefficients_lw.nc')
subprocess.call(['python', 'rfmip_download.py'])

if not os.path.exists('data2'):
    os.mkdir('data2')

#create input data
#subprocess.call(['python', 'rfmip_createinput.py', '--finvt', '--nfile=%s'%1])
subprocess.call(['python', 'rfmip_createinput.py', '--frand', '--finvt', '--nfile=%s'%nfile])

#run rrtmgp
for ifile in range(nfile):
    shutil.copyfile('data2/rte_rrtmgp_input_{:03d}.nc'.format(ifile), 'rte_rrtmgp_input.nc')
    subprocess.call(['./test_rte_rrtmgp', '--no-cloud-optics', '--no-fluxes', '--output-optical'])
    shutil.move('rte_rrtmgp_output1.nc', 'data2/rte_rrtmgp_output_{:03d}.nc'.format(ifile))

#concetenate output and input file
os.system('ncecat data2/*input*  -u iter -O data2/inputs.nc')
os.system('ncecat data2/*output* -u iter -O data2/outputs.nc')

os.system('ncrename -v lay_source,Planck_lay data2/outputs.nc')
os.system('ncrename -v lev_source_inc,Planck_levinc data2/outputs.nc')
os.system('ncrename -v lev_source_dec,Planck_levdec data2/outputs.nc')
os.system('ncrename -v sw_tau,tauSW data2/outputs.nc')
os.system('ncrename -v lw_tau,tauLW data2/outputs.nc')
os.system('ncrename -v ssa,SSA data2/outputs.nc')
#delete individual input/output files
os.system('rm -f data2/rte_rrtmgp_input*')
os.system('rm -f data2/rte_rrtmgp_output*')

