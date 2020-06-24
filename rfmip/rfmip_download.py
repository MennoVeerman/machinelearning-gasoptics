#! /usr/bin/env python
#
# This script downloads and creates files needed for the RFMIP off-line test cases
#
import sys
import os
import subprocess
import glob
import urllib.request

conds_file      = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
conds_url       = "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/RFMIP/UColorado/UColorado-RFMIP-1-2/" + \
                  "atmos/fx/multiple/none/v20190401/" + conds_file

for f in glob.glob("multiple_input4MIPs_radiation_RFMIP*.nc"): os.remove(f)

print("Dowloading RFMIP input files")
urllib.request.urlretrieve(conds_url,     conds_file)
