import netCDF4 as nc
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datapath',   default='./', type=str)
parser.add_argument('--atmfile',    default='rte_rrtmgp_input.nc', type=str)
parser.add_argument('--optfile',    default='rte_rrtmgp_output.nc', type=str)
parser.add_argument('--filecount',  default=1 , type=int)
parser.add_argument('--log_input',  default=False, action='store_true')
parser.add_argument('--log_output', default=False, action='store_true')
parser.add_argument('--do_o3',      default=False, action='store_true')
args = parser.parse_args()

def fast_log(x):
    x = np.sqrt(x)
    x = np.sqrt(x)
    x = np.sqrt(x)
    x = np.sqrt(x)
    return (x-1)*16

def read_normalize_keys(name):
    ninp = 3 + args.do_o3 + (name=="Planck") * 2
    nc_atm   = nc.Dataset(args.atmfile)
    nc_opt   = nc.Dataset(args.optfile)
    optprop  = nc_opt.variables[name + (name=="Plank")*"_lay"][:]
    ngpt,nlay,ncol = optprop.shape
    optprop  = optprop.reshape(ngpt, nlay*ncol)
    optprop  = np.transpose(optprop)
    
    if name == "Planck":
        data  = np.zeros((optprop.shape[0], ninp+1+ngpt*3))
        optprop_inc  = np.transpose(nc_opt.variables[name+"_levinc"][:].reshape(ngpt,nlay*ncol))
        optprop_dec  = np.transpose(nc_opt.variables[name+"_levdec"][:].reshape(ngpt,nlay*ncol))
        data[:,ninp+1+ngpt*0:ninp+1+ngpt*1] =  optprop[:]
        data[:,ninp+1+ngpt*1:ninp+1+ngpt*2] =  optprop_inc[:]
        data[:,ninp+1+ngpt*2:ninp+1+ngpt*3] =  optprop_dec[:]
    else:
        data  = np.zeros((optprop.shape[0], ninp+1+ngpt))
        data[:,ninp+1:] =  optprop[:]

    data[:,0] = nc_atm.variables['vmr_h2o'][:].reshape(nlay*ncol)
    if args.do_o3: 
        data[:,1] = nc_atm.variables['vmr_o3'][:].reshape(nlay*ncol)
    data[:,1+args.do_o3] = nc_atm.variables['p_lay'][:].reshape(nlay*ncol)
    data[:,2+args.do_o3] = nc_atm.variables['t_lay'][:].reshape(nlay*ncol)

    if name == "Planck":
        keys = ['h2o']+['o3']*args.do_o3+['p_lay','t_lay','t_lev_bottom','t_lev_top','tropo']
        data[:,3+args.do_o3] = nc_atm.variables['t_lev'][:-1].reshape(nlay*ncol)
        data[:,4+args.do_o3] = nc_atm.variables['t_lev'][1:].reshape(nlay*ncol)
        keys += ["lbl"+"%03d"%i for i in range(1,ngpt*3+1)]
    else:
        keys = ['h2o']+['o3']*args.do_o3+['p_lay','t_lay','tropo']
        keys += ["lbl"+"%03d"%i for i in range(1,ngpt+1)]
    mask  = np.where(data[:,1] < 9948.431564193395,-1,1)
     
    data[:,ninp+1:] = np.where(data[:,ninp+1:]==0, 1e-12, data[:,ninp+1:])
    
    if np.min(mask) == -1:
        keepzero = (np.max(data[mask==-1],axis=0) == 1e-12)
        keepzero[:ninp+1] = False
        
    if args.log_output and name != "SSA":
        data[:,ninp+1:] = fast_log(data[:,ninp+1:])
        
    if args.log_input:
        data[:,0]  = fast_log(data[:,0]) 
        if args.do_o3:
            data[:,1]  = fast_log(data[:,1])
        data[:,1+args.do_o3]  = fast_log(data[:,1+args.do_o3])

    #above tropopause (0)
    if np.min(mask) == -1:    
        above = (mask == -1)
        mean_upr = np.mean(data[above],axis=0)
        stdv_upr = np.std(data[above],axis=0)
        stdv_upr[mean_upr==1] = 1.
        mean_upr[mean_upr==1] = 0.
        np.savetxt(args.datapath+"means_upr_%s.txt"%name,mean_upr)
        np.savetxt(args.datapath+"stdev_upr_%s.txt"%name,stdv_upr)
        data[above,ninp+1:] = (data[above,ninp+1:] -\
            mean_upr[np.newaxis,ninp+1:])/stdv_upr[np.newaxis,ninp+1:]
 
    #below tropopause (1)
    if np.max(mask) == 1:
        below = (mask == 1)
        mean_lwr = np.mean(data[below],axis=0)
        stdv_lwr = np.std(data[below],axis=0)
        stdv_lwr[mean_lwr==1] = 1.
        mean_lwr[mean_lwr==1] = 0.
        np.savetxt(args.datapath+"means_lwr_%s.txt"%name,mean_lwr)
        np.savetxt(args.datapath+"stdev_lwr_%s.txt"%name,stdv_lwr)
        data[below,ninp+1:] = (data[below,ninp+1:] -\
            mean_lwr[np.newaxis,ninp+1:])/stdv_lwr[np.newaxis,ninp+1:]
        
    data[:,ninp] = mask 
    
    if np.min(mask) == -1:    
        keepzeromask = ((mask == -1)[:,np.newaxis]*1 + keepzero[np.newaxis,:]*1) == 2  
        data[keepzeromask] = 0.

    np.random.shuffle(data)
    return np.array(data),keys


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value])) 

def serialize_example(featurelist): 
    feature = {}
    for i in range(len(keylist)):
        feature[keylist[i]] = _float_feature(featurelist[i])
    ExampleProto = tf.train.Example(features=tf.train.Features(feature=feature))
    return ExampleProto.SerializeToString()

def write_keys(keys,name):
    keyfile = open(args.datapath+'keylist_%s.txt'%name,'w')
    for key in keys:
        keyfile.write(key+",")  
    keyfile.close()

def write_tfrecords(inpdata,fname):
    with tf.python_io.TFRecordWriter(fname) as writer:
        for i in inpdata:
            writer.write(serialize_example(i))
    return

def write_main(files, name, ntrain=0.9):
    global keylist
    data,keylist = read_normalize_keys(name)
    write_keys(keylist,name)
    np.random.shuffle(data)
    ltrain = int(len(data) * ntrain)    
    subsubprocesses = [mp.Process(target=write_tfrecords,args=(data[ltrain:],args.datapath+'testing_%s.tfrecords'%name))]

    lsub = ltrain//args.filecount
    for i in range(args.filecount):
        subsubprocesses += [mp.Process(target=write_tfrecords, args=(data[lsub*i:lsub*(i+1)], args.datapath+'training%s_%s.tfrecords'%(i,name)))]

    for sp in subsubprocesses:
        sp.start()
    for sp in subsubprocesses:
        sp.join()

if __name__ == '__main__':  
    subprocesses = []
    for name in ["Planck","tauLW","tauSW","SSA"]:
        subprocesses += [mp.Process(target=write_main, args=(args.datapath,name))]
    for sp in subprocesses:
        sp.start()
    for sp in subprocesses:
        sp.join()


