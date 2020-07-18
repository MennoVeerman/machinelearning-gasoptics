import netCDF4 as nc
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--simpath', default="", type=str)
args = parser.parse_args()
path = args.simpath

float_type = "f8"
rlv        = 2.53e6
cp         = 1004.
rd         = 287.04
pres0      = 1.e5
es0        = 610.78
rv         = 461.5
at         = 17.27
bt         = 35.86
tmelt      = 273.16
ep         = 0.622
wh2o_min   = 5e-6

qt = nc.Dataset(path+"qt.nc").variables['qt'][:]
ql = nc.Dataset(path+"ql.nc").variables['ql'][:]
qi = nc.Dataset(path+"qi.nc").variables['qi'][:]
qv = qt-(ql+qi)
nt,nz,ny,nx = qv.shape

qv_max = ep*wh2o_min/(1+ep*wh2o_min)
q = np.maximum(qv_max,qv)

T  = nc.Dataset(path+"T.nc").variables['T'][:]
p1 = nc.Dataset(path+"rcemip.default.0000000.nc").groups['thermo'].variables['phydro'][::24]
p2 = nc.Dataset(path+"rcemip.default.0008640.nc").groups['thermo'].variables['phydro'][::24]
p  = np.append(p1,p2,axis=0)

data = np.zeros((nz,5))
data[:,0] = np.mean(p,axis=0)
data[:,1] = np.min(T,axis=(0,2,3))
data[:,2] = np.max(T,axis=(0,2,3))
data[:,3] = np.min(q,axis=(0,2,3))
data[:,4] = np.max(q,axis=(0,2,3))
np.savetxt("profiles.txt",data,header="pres  Tmin  Tmax  qmin  qmax")

