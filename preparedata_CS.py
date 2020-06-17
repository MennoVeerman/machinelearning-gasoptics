import netCDF4 as nc
import os
import numpy as np
import tensorflow as tf
import multiprocessing as mp
def mylog(x):
    x = np.sqrt(x)
    x = np.sqrt(x)
    x = np.sqrt(x)
    x = np.sqrt(x)
    return (x-1)*16

datafile = "RCEMIP/samples_opticalprops.nc"
ninp = 3
path = "/scratch-shared/mveerman/PrepareData/RCEMIP/"
datapath = path
log_output = True
log_input  = True
def read_normalize_keys(name):
    ninp = 3 + (name=="Planck_lay") * 2
    ncfile   = nc.Dataset(datafile)
    optprop  = ncfile.variables[name][:]
    nfeat,nlay,ncol = optprop.shape
    print("X",nfeat,nlay,ncol)
    optprop  = optprop.reshape(nfeat,nlay*ncol)
    optprop  = np.transpose(optprop)
    
    if name == "Planck_lay":
        data  = np.zeros((optprop.shape[0], ninp+1+nfeat*3))
        optpropA  = np.transpose(ncfile.variables["Planck_levinc"][:].reshape(nfeat,nlay*ncol))
        optpropB  = np.transpose(ncfile.variables["Planck_levdec"][:].reshape(nfeat,nlay*ncol))
        print(np.min(optprop),np.min(optpropA),np.min(optpropB))
        data[:,ninp+1        :ninp+1+nfeat]   =  optprop[:]
        data[:,ninp+1+nfeat  :ninp+1+nfeat*2] =  optpropA[:]
        data[:,ninp+1+nfeat*2:ninp+1+nfeat*3] =  optpropB[:]
        print(np.min(data[:,ninp+1:]))
    else:
        data  = np.zeros((optprop.shape[0], ninp+1+nfeat))
        data[:,ninp+1:] =  optprop[:]

    data[:,0] = ncfile.variables['qlay'][:].reshape(nlay*ncol)
    data[:,1] = ncfile.variables['Play'][:].reshape(nlay*ncol)
    data[:,2] = ncfile.variables['Tlay'][:].reshape(nlay*ncol)

    if name == "Planck_lay":
        keys = ['h2o','p_lay','t_lay','t_levB','t_levT','tropo']
        data[:,3] = ncfile.variables['Tlev'][:-1].reshape(nlay*ncol)
        data[:,4] = ncfile.variables['Tlev'][1:].reshape(nlay*ncol)
        keys += ["lbl"+"0"*(3-len(str(i)))+str(i) for i in range(1,nfeat*3+1)]
    else:
        keys = ['h2o','p_lay','t_lay','tropo']
        keys += ["lbl"+"0"*(3-len(str(i)))+str(i) for i in range(1,nfeat+1)]
    mask  = np.where(data[:,1] < 9948.431564193395,-1,1)
     
    data[:,ninp+1:] = np.where(data[:,ninp+1:]==0,1e-12,data[:,ninp+1:])
    
    if np.min(mask) == -1:    
        keepzero = (np.max(data[mask==-1],axis=0) == 1e-12)
        keepzero[:ninp+1] = False
    if log_output and name != "SSA":
        data[:,ninp+1:] = mylog(data[:,ninp+1:])
    if name == "tauLW":
        print(data[:15,183],np.std(data[:15,183]))
        above = (mask == -1)
        abovedata = data[above,185]
        for i in abovedata:
            print("T",i)
        #print("XYZ",data[above,185],np.std(data[above,185]))
    if log_input:
        data[:,0]  = mylog(data[:,0]) 
        data[:,1]  = mylog(data[:,1])

    #above tropopause (0)
    if np.min(mask) == -1:    
        above = (mask == -1)
        Mean0 = np.mean(data[above],axis=0)
        Std0  = np.std(data[above],axis=0)
        Std0[Mean0==1] = 1.
        Mean0[Mean0==1] = 0.
        if name =="Planck_lay":
            name = "Planck"
        np.savetxt(datapath+"means0_%s.txt"%name,Mean0)
        np.savetxt(datapath+"stdev0_%s.txt"%name,Std0)
        data[above,ninp+1:] = (data[above,ninp+1:] -\
            Mean0[np.newaxis,ninp+1:])/Std0[np.newaxis,ninp+1:]
 
    #below tropopause (1)
    if np.max(mask) == 1:
        below = (mask == 1)
        Mean1 = np.mean(data[below],axis=0)
        Std1  = np.std(data[below],axis=0)
        Std1[Mean1==1] = 1.
        Mean1[Mean1==1] = 0.
        if name =="Planck_lay":
            name = "Planck"
        np.savetxt(datapath+"means1_%s.txt"%name,Mean1)
        np.savetxt(datapath+"stdev1_%s.txt"%name,Std1)
        data[below,ninp+1:] = (data[below,ninp+1:] -\
            Mean1[np.newaxis,ninp+1:])/Std1[np.newaxis,ninp+1:]
        
    data[:,ninp] = mask 
    
    if np.min(mask) == -1:    
        keepzeromask = ((mask == -1)[:,np.newaxis] * 1 + keepzero[np.newaxis,:] * 1) == 2  
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
    if name =="Planck_lay":
        name = "Planck"
    keyfile = open(datapath+'keylist_%s.txt'%name,'w')
    for key in keys:
        keyfile.write(key+",")  
    keyfile.close()
    return
def write_tfrecords(inpdata,fname):
    with tf.python_io.TFRecordWriter(fname) as writer:
        for i in inpdata:
            writer.write(serialize_example(i))
    return

def write_main(files,name,ntrain=0.9):
    global keylist
    data,keylist = read_normalize_keys(name)
    if 'keylist_%s.txt'%name not in os.listdir(datapath) :
        write_keys(keylist,name)
    np.random.shuffle(data)
    ltrain = int(len(data) * ntrain)    
    if name =="Planck_lay":
        name = "Planck"

    subsubprocesses = [mp.Process(target=write_tfrecords,args=(data[ltrain:],datapath+'testing_%s.tfrecords'%name))]

    lthird = ltrain//3
    print(lthird,ltrain)
    subsubprocesses += [mp.Process(target=write_tfrecords,args=(data[:lthird],datapath+'training0_%s.tfrecords'%name))]
    subsubprocesses += [mp.Process(target=write_tfrecords,args=(data[lthird:2*lthird],datapath+'training1_%s.tfrecords'%name))]
    subsubprocesses += [mp.Process(target=write_tfrecords,args=(data[2*lthird:ltrain],datapath+'training2_%s.tfrecords'%name))]

    for sp in subsubprocesses:
        sp.start()
    for sp in subsubprocesses:
        sp.join()

### run:
if __name__ == '__main__':  
#names: tauLW, tauSW, SSA, Planck_lay
    subprocesses = []
    for nm in ["Planck_lay","tauLW","tauSW","SSA"]:
        subprocesses += [mp.Process(target=write_main,args=(datapath,nm))]
    #write_main(datafile,'tauLW')
    for sp in subprocesses:
        sp.start()
    for sp in subprocesses:
        sp.join()


