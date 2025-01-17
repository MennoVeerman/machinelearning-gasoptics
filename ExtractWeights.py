import numpy as np
import netCDF4 as nc
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
from TrainNetwork import DNN_Regression
import argparse
from RunArgs import *

### dictionary containing all means and st_devs
def get_znorm(name):
    znorm={}
    if args.do_upper:
        znorm["means_upper"] = np.loadtxt(args.datapath+"means_upr_%s.txt"%name,dtype=np.float32)
        znorm["stdev_upper"] = np.loadtxt(args.datapath+"stdev_upr_%s.txt"%name,dtype=np.float32)
    if args.do_lower:
        znorm["means_lower"] = np.loadtxt(args.datapath+"means_lwr_%s.txt"%name,dtype=np.float32)
        znorm["stdev_lower"] = np.loadtxt(args.datapath+"stdev_lwr_%s.txt"%name,dtype=np.float32)
    return znorm

def get_model(modelpath, name, ngpt, nodes, keys):
    znorm = get_znorm(name)
    
    hyperparameters = {
            'train'              : False,
            'n_nodes'            : nodes,
            'n_labels'           : ngpt,
            'kernel_initializer' : tf.glorot_uniform_initializer(),
            'learning_rate'      : 0.01,
            'znorm'              : znorm,
            'keys'               : keys,
            'argparser'          : args}

    model = tf.estimator.Estimator(
    model_fn = DNN_Regression,
    params = hyperparameters,
    model_dir = modelpath)
    return model

### create placeholders for input
def get_placeholderdict(name, keys):
    placeholderdict = {}
    for key in keys:
        placeholderdict[key] = tf.placeholder(tf.float32, shape=[None], name=key)
    print(placeholderdict.keys())
    return placeholderdict

### function to make frozen graph
def frozen_graph_maker(export_dir, output_graph, output_node):
    with tf.Session(graph=tf.Graph()) as session:
        tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], export_dir)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                session,
                session.graph_def,
                [output_node]) 

    with tf.gfile.GFile(output_graph, "wb") as gf:
            gf.write(output_graph_def.SerializeToString())

def find_idx(namelist, name):
    return [namelist.index(s) for s in namelist if name in s]
    
def extract_weights(graphpath_lwr, graphpath_upr, ncgrp, nodes):
    graphs = []
    if args.do_lower: 
        graphs += [(graphpath_lwr,'_lower')]
    if args.do_upper: 
        graphs += [(graphpath_upr,'_upper')]

    for graph,uplow in graphs:
        with tf.Session() as session:
            with gfile.FastGFile(graph,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                session.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                const_nodes=[n for n in graph_def.node if n.op=='Const']
        nodenames = [const_nodes[n].name for n in range(len(const_nodes))]
        print(nodenames) 
        ft_mean = tensor_util.MakeNdarray(const_nodes[find_idx(nodenames, 'ft_mean')[0]].attr['value'].tensor)
        ft_stdv = tensor_util.MakeNdarray(const_nodes[find_idx(nodenames, 'ft_stdv')[0]].attr['value'].tensor)
        lb_mean = tensor_util.MakeNdarray(const_nodes[find_idx(nodenames, 'lb_mean')[0]].attr['value'].tensor)
        lb_stdv = tensor_util.MakeNdarray(const_nodes[find_idx(nodenames, 'lb_stdv')[0]].attr['value'].tensor)
        n_in  = len(ft_mean)
        n_out = len(lb_mean)        
        if uplow == graphs[0][1]:
            ncgrp.createDimension("Nin",n_in)
            ncgrp.createDimension("Nout",n_out)    
            
        n1=ncgrp.createVariable("Fmean"+uplow,np.float32,("Nin",))
        n2=ncgrp.createVariable("Fstdv"+uplow,np.float32,("Nin",))
        n3=ncgrp.createVariable("Lmean"+uplow,np.float32,("Nout",))
        n4=ncgrp.createVariable("Lstdv"+uplow,np.float32,("Nout",))
        n1[:] = ft_mean 
        n2[:] = ft_stdv 
        n3[:] = lb_mean 
        n4[:] = lb_stdv 
        
        bias_idx = find_idx(nodenames, 'bias')
        for i in range(len(bias_idx)):
            idx  = bias_idx[i]
            bias = tensor_util.MakeNdarray(const_nodes[idx].attr['value'].tensor)
            if uplow == "_lower" and idx != bias_idx[-1]:
                ncgrp.createDimension("Nl"+str(i+1),len(bias))
            if idx != bias_idx[-1]:
                b = ncgrp.createVariable("bias"+str(i+1)+uplow,np.float32,("Nl"+str(i+1),))
            else:
                b = ncgrp.createVariable("bias"+str(i+1)+uplow,np.float32,("Nout",))
            b[:] = bias

        wgth_idx = find_idx(nodenames, 'kernel')
        for i in range(len(wgth_idx)):
            idx  = wgth_idx[i]
            wgth = tensor_util.MakeNdarray(const_nodes[idx].attr['value'].tensor)
            if len(wgth_idx) ==1:
                w = ncgrp.createVariable("wgth1"+uplow,np.float32,("Nout","Nin"))
            else: 
                if idx  == wgth_idx[0]:
                    w = ncgrp.createVariable("wgth1"+uplow,np.float32,("Nl1","Nin"))
                elif idx == wgth_idx[-1]:
                    w = ncgrp.createVariable("wgth"+str(len(wgth_idx))+uplow,np.float32,("Nout","Nl"+str(len(wgth_idx)-1)))
                else:
                    w = ncgrp.createVariable("wgth"+str(i+1)+uplow,np.float32,("Nl"+str(i+1),"Nl"+str(i)))
            w[:] = np.transpose(wgth)
      
        #currently, we need to supply weights for both upper and lower atmosphere to the solver
        if args.do_lower + args.do_upper != 2:
            lowup_ref  = args.do_lower*'lower'+args.do_upper*'upper'
            lowup_sub  = args.do_upper*'lower'+args.do_lower*'upper'
            varnames = [name for name in ncgrp.variables if lowup_ref in name]
            for varname in varnames:
                oldvar = ncgrp.variables[varname]
                newvar = ncgrp.createVariable(varname[:-5]+lowup_sub, np.float32, oldvar.dimensions)
                newvar[:] = 0.

       
def main_extractweights(dirname, nodes):
    if dirname == "Linear/":
        wgth_name = "weights_%s.nc"%dirname[:-1]
    else:
        wgth_name = "weights_%s.nc"%dirname[3:-1]
    ncfile = nc.Dataset(args.trainpath+wgth_name, "w")    
    ncfile.createDimension('nlayers', len(nodes))
    ncfile.createDimension('nlayer1', (nodes+[0,0,0])[0])
    ncfile.createDimension('nlayer2', (nodes+[0,0,0])[1])
    ncfile.createDimension('nlayer3', (nodes+[0,0,0])[2])
    ncfile.createDimension('nout_sw', 224)
    ncfile.createDimension('nout_lw', 256)

    for groupname, name, graphname, ngpt in [("SSA",    "SSA",    "ssa", 224),
                                             ("TSW",    "tauSW",  "tsw", 224),
                                             ("TLW",    "tauLW",  "tlw", 256),
                                             ("Planck", "Planck", "plk", 768)]:
        keys  = open(args.datapath+'keylist_%s.txt'%name,'r').readline().split(',')[:-1]
        if name == 'SSA':
            ngases = len(keys)-ngpt-3
        keys  = keys[:-(ngpt)] 
        grp = ncfile.createGroup(groupname)
        modelpath = args.trainpath+"%s/%s"%(dirname,name)  
        serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(get_placeholderdict(name, keys))    
        n_do = args.do_upper + args.do_lower
        if args.do_lower:
            if n_do == 2: args.do_upper=False
            model_lwr = get_model(modelpath+"/lower_atm", name, ngpt, nodes, keys)
            frozengraph_path_lwr = modelpath+"/lower_atm/"+"frozen_graph_%s.pb"%graphname
            savedmodel_path = model_lwr.export_savedmodel(modelpath+"/lower_atm", serving_fn,
                                                          strip_default_attrs=True)
            frozen_graph_maker(savedmodel_path, frozengraph_path_lwr, 'output_lower')
            if n_do ==2: args.do_upper=True

        if args.do_upper: 
            if n_do ==2: args.do_lower=False
            model_upr = get_model(modelpath+"/upper_atm", name, ngpt, nodes, keys)
            frozengraph_path_upr = modelpath+"/upper_atm/"+"frozen_graph_%s.pb"%graphname
            savedmodel_path = model_upr.export_savedmodel(modelpath+"/upper_atm", serving_fn,
                                                          strip_default_attrs=True)
            frozen_graph_maker(savedmodel_path, frozengraph_path_upr, 'output_upper')
            
            if n_do ==2: args.do_lower=True
        if "frozengraph_path_lwr" not in locals(): frozengraph_path_lwr = ""
        if "frozengraph_path_upr" not in locals(): frozengraph_path_upr = ""
        extract_weights(frozengraph_path_lwr, frozengraph_path_upr, grp, nodes)
        tf.keras.backend.clear_session()    
    ncfile.createDimension('ngases', ngases)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_from_file', default=False, action='store_true')
    parser.add_argument('--args_inp_file',  default='./arguments.txt', type=str)
    parser.add_argument('--do_upper',  default=False, action='store_true')
    parser.add_argument('--do_lower',  default=False, action='store_true')
    parser.add_argument('--datapath',  default='./', type=str)
    parser.add_argument('--trainpath', default='./', type=str)
    args = parser.parse_args()

    if args.args_from_file:
        read_run_arguments(args, args.args_inp_file)
    for nodes,dirname in [([], "Linear/"), ([32],"1L-32/"), ([32,32],"2L-32_32/"), ([64],"1L-64/"), ([64,64],"2L-64_64/"), ([32,64,128],"3L-32_64_128/")]:
        main_extractweights(dirname, nodes)
 
