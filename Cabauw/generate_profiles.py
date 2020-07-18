import ctypes as c
import numpy as np
import netCDF4 as nc
import multiprocessing as mp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--simpath', default="", type=str)
args = parser.parse_args()
path = args.simpath
path = "/archive/mveerman/machinelearning/cabauw_sim/"
ncname = 'threedheating.%s.%s.700.nc' 
nx = 12
ny = 12

cp  = 1004.
rd    = 287.04
pres0 = 1.e5
es0   = 610.78
ep    = 0.622
wh2o_min =1.6e-5
qv_max   = ep*wh2o_min/(1+ep*wh2o_min)

def do_stats(ncfile,exnf,mp_array,i):
    rlv = 2.53e6
    cp  = 1004.
    thl = ncfile.variables['thl'][:,:,:,:]
    ql  = ncfile.variables['ql'][:,:,:,:] * 1e-5
    qt  = np.maximum(0.,ncfile.variables['qt'][:,:,:,:]) * 1e-5
    t = exnf[:,:,np.newaxis,np.newaxis] * (thl + (rlv/cp) * ql / exnf[:,:,np.newaxis,np.newaxis])
    q = np.maximum(qv_max,qt-ql)

    tmp_arr = np.frombuffer(mp_array.get_obj())
    out_arr = tmp_arr.reshape(nx*ny,4,228)
    out_arr[i,0,:] = np.min(t,axis=(0,2,3))
    out_arr[i,1,:] = np.max(t,axis=(0,2,3))
    out_arr[i,2,:] = np.min(q,axis=(0,2,3))
    out_arr[i,3,:] = np.max(q,axis=(0,2,3))

presf = np.loadtxt(path+"pressures.dat")[:,1].reshape((600,228)) * 100.
exnf  = (presf/pres0)**(rd/cp)

tq_minmax = mp.Array(c.c_double,nx*ny*4*228)
processes = []
for ix in range(nx):
    for iy in range(ny):
        processes += [mp.Process(target=do_stats,args=(nc.Dataset(path+ncname%('%03d'%ix,'%03d'%iy)),exnf,tq_minmax,int(iy+ix*ny)))]

pi = 0
while pi < nx*ny:
    if np.sum([p.is_alive() for p in processes]) < 24:
        processes[pi].start()
        pi += 1
for p in processes:
    p.join()

tmp_arr = np.frombuffer(tq_minmax.get_obj())
out_arr = tmp_arr.reshape(nx*ny,4,228)

data = np.zeros((228,5))
data[:,0] = np.mean(presf,axis=0)
data[:,1] = np.min(out_arr[:,0,:],axis=0)
data[:,2] = np.max(out_arr[:,1,:],axis=0)
data[:,3] = np.min(out_arr[:,2,:],axis=0)
data[:,4] = np.max(out_arr[:,3,:],axis=0)
np.savetxt("profiles.txt",data,header="pres  Tmin  Tmax  qmin  qmax")




