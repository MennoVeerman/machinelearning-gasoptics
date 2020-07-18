import numpy as np
import netCDF4 as nc
import shutil
import subprocess
import os
import sys
import glob
import urllib.request
 
nfile=30
dpath = "data"
rrtmgp_path =  "/path/to/rrtmgp-rte-cpp/folder/"

def download_rfmip():
    conds_file      = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
    conds_url       = "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/RFMIP/UColorado/UColorado-RFMIP-1-2/" + \
                  "atmos/fx/multiple/none/v20190401/" + conds_file
    for f in glob.glob("multiple_input4MIPs_radiation_RFMIP*.nc"): os.remove(f)
    print("Dowloading RFMIP input files")
    urllib.request.urlretrieve(conds_url,     conds_file)

rrtmgp_build_dir = rrtmgp_path+"build/"
rrtmgp_coeff_dir = rrtmgp_path+"rte-rrtmgp/rrtmgp/data/"
rrtmgp_cloud_dir = rrtmgp_path+"rte-rrtmgp/extensions/cloud_optics/"
if not os.path.exists('test_rte_rrtmgp'): os.symlink(rrtmgp_build_dir+'test_rte_rrtmgp', 'test_rte_rrtmgp')
if not os.path.exists('coefficients_lw.nc'): os.symlink(rrtmgp_coeff_dir+'rrtmgp-data-lw-g256-2018-12-04.nc', 'coefficients_lw.nc')
if not os.path.exists('coefficients_sw.nc'): os.symlink(rrtmgp_coeff_dir+'rrtmgp-data-sw-g224-2018-12-04.nc', 'coefficients_sw.nc')
if not os.path.exists('cloud_coefficients_sw.nc'): os.symlink(rrtmgp_cloud_dir+'rrtmgp-cloud-optics-coeffs-sw.nc', 'cloud_coefficients_sw.nc')
if not os.path.exists('cloud_coefficients_lw.nc'): os.symlink(rrtmgp_cloud_dir+'rrtmgp-cloud-optics-coeffs-lw.nc', 'cloud_coefficients_lw.nc')

if len(glob.glob("multiple_inputs4MIPs_radiation_RFMIP*")) == 0:
    download_rfmip()

#create input data
if not os.path.exists(dpath):
    os.mkdir(dpath)
subprocess.call(['python', 'generate_samples.py', '--nfile=%s'%(nfile), '--dpath=%s'%dpath])

for ifile in range(nfile):
    shutil.copyfile(dpath+'/rte_rrtmgp_input_{:03d}.nc'.format(ifile), 'rte_rrtmgp_input.nc')
    subprocess.call(['./test_rte_rrtmgp', '--no-cloud-optics', '--no-fluxes', '--output-optical'])
    shutil.move('rte_rrtmgp_output.nc', dpath+'/rte_rrtmgp_output_{:03d}.nc'.format(ifile))

#concetenate output and input file
os.system('ncecat %s/*input*  -u iter -O %s/inputs.nc'%(dpath,dpath))
os.system('ncecat %s/*output* -u iter -O %s/outputs.nc'%(dpath,dpath))

os.system('ncrename -v lay_source,Planck_lay %s/outputs.nc'%dpath)
os.system('ncrename -v lev_source_inc,Planck_levinc %s/outputs.nc'%dpath)
os.system('ncrename -v lev_source_dec,Planck_levdec %s/outputs.nc'%dpath)
os.system('ncrename -v sw_tau,tauSW %s/outputs.nc'%dpath)
os.system('ncrename -v lw_tau,tauLW %s/outputs.nc'%dpath)
os.system('ncrename -v ssa,SSA %s/outputs.nc'%dpath)
#delete individual input/output files
os.system('rm -f %s/rte_rrtmgp_input*'%dpath)
os.system('rm -f %s/rte_rrtmgp_output*'%dpath)

