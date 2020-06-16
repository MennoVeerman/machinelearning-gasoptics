import numpy as np
import netCDF4 as nc
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
from TrainNetwork2 import DNN_Regression
basepath = "/scratch-shared/mveerman/Training/CABAUW/"
datapath = "/scratch-shared/mveerman/PrepareData/CABAUW/"
do_upper = False
do_lower = True
#NOTE: ONLY WORKS IF NETWORK IS TRAINED FOR BOTH ABOVE AND BELOW TROPOPAUSE

### dictionary containing all means and st_devs
def get_znorm(name):
    Znorm={}
    if do_upper:
        Znorm["means_upper"] = np.loadtxt(datapath+"means0_%s.txt"%name,dtype=np.float32)
        Znorm["stdev_upper"] = np.loadtxt(datapath+"stdev0_%s.txt"%name,dtype=np.float32)
    if do_lower:
        Znorm["means_lower"] = np.loadtxt(datapath+"means1_%s.txt"%name,dtype=np.float32)
        Znorm["stdev_lower"] = np.loadtxt(datapath+"stdev1_%s.txt"%name,dtype=np.float32)
    return Znorm

def get_model(modelpath,name,nlabel,nodes):
    Znorm = get_znorm(name)
    hyperparameters = {
            'train'             : False,
            'Nnodes'            : nodes,
            'n_labels'          : nlabel,
            'kernel_initializer': tf.glorot_uniform_initializer(),
            'learning_rate'     : 0.01,
            'Znorm'             : Znorm,
            'keys'              : ["h2o","p_lay","t_lay"]+(name=="Planck")*["t_levB","t_levT"],
            }
    model = tf.estimator.Estimator(
    model_fn = DNN_Regression,
    params = hyperparameters,
    model_dir = modelpath)
    return model

### create placeholders for input
def get_placeholderdict(name):
    if name == "Planck":
        placeholderdict = {
             "h2o": tf.placeholder(tf.float32, shape=[None], name="h2o"),
             "p_lay": tf.placeholder(tf.float32, shape=[None], name="p_lay"),
             "t_lay": tf.placeholder(tf.float32, shape=[None], name="t_lay"),
             "tropo": tf.placeholder(tf.float32, shape=[None], name="tropo"),
             "t_levB":tf.placeholder(tf.float32, shape=[None], name="t_levB"),
             "t_levT":tf.placeholder(tf.float32, shape=[None], name="t_levT"),
             }
    else:
        placeholderdict = {
             "h2o": tf.placeholder(tf.float32, shape=[None], name="h2o"),
             "p_lay": tf.placeholder(tf.float32, shape=[None], name="p_lay"),
             "t_lay": tf.placeholder(tf.float32, shape=[None], name="t_lay"),
             "tropo": tf.placeholder(tf.float32, shape=[None], name="tropo")}
    return placeholderdict
### function to make frozen graph
def frozen_graph_maker(export_dir,output_graph,outname):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        #name of output node(s):
        output_nodes = [outname]  
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                sess.graph_def,
                output_nodes # The output node names are used to select the usefull nodes
        ) 
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    with tf.Session() as sess:
        with gfile.FastGFile(output_graph,"rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            const_nodes=[n for n in graph_def.node]# if n.op=='Const']
        for n in range(len(const_nodes)):
            print(n, const_nodes[n].name)    

def extract_weights(graphpathL,graphpathU,ncgrp):
    graphs = []
    if do_lower: 
        graphs += [(graphpathL,'_lower')]
    if do_upper: 
        graphs += [(graphpathU,'_upper')]

    for graph,uplow in graphs:
        with tf.Session() as sess:
            with gfile.FastGFile(graph,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                const_nodes=[n for n in graph_def.node if n.op=='Const']
        nodenames = [const_nodes[n].name for n in range(len(const_nodes))]
        print("Name of all constant nodes:") #print all nodes that contain constant values (e.g. weights, biases)
        for n in range(len(const_nodes)):
            print(n, const_nodes[n].name)

        Fmean = tensor_util.MakeNdarray(const_nodes[14].attr['value'].tensor)
        Fstdv = tensor_util.MakeNdarray(const_nodes[15].attr['value'].tensor)
        Lmean = tensor_util.MakeNdarray(const_nodes[16].attr['value'].tensor)
        Lstdv = tensor_util.MakeNdarray(const_nodes[17].attr['value'].tensor)
        Nin  = len(Fmean)
        Nout = len(Lmean)        
        if uplow == "_lower":
            ncgrp.createDimension("Nin",Nin)
            ncgrp.createDimension("Nout",Nout)    
            
        n1=ncgrp.createVariable("Fmean"+uplow,np.float32,("Nin",))
        n2=ncgrp.createVariable("Fstdv"+uplow,np.float32,("Nin",))
        n3=ncgrp.createVariable("Lmean"+uplow,np.float32,("Nout",))
        n4=ncgrp.createVariable("Lstdv"+uplow,np.float32,("Nout",))
        n1[:] = Fmean 
        n2[:] = Fstdv 
        n3[:] = Lmean 
        n4[:] = Lstdv 
        
        bias_idx = [nodenames.index(s) for s in nodenames if "bias" in s]
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

        wgth_idx = [nodenames.index(s) for s in nodenames if "kernel" in s]
        for i in range(len(wgth_idx)):
            idx  = wgth_idx[i]
            wgth = tensor_util.MakeNdarray(const_nodes[idx].attr['value'].tensor)
            if idx   == wgth_idx[0]:
                w = ncgrp.createVariable("wgth1"+uplow,np.float32,("Nl1","Nin"))
            elif idx == wgth_idx[-1]:
                w = ncgrp.createVariable("wgth"+str(len(wgth_idx))+uplow,np.float32,("Nout","Nl"+str(len(wgth_idx)-1)))
            else:
                w = ncgrp.createVariable("wgth"+str(i+1)+uplow,np.float32,("Nl"+str(i+1),"Nl"+str(i)))
            w[:] = np.transpose(wgth)
      
        for n in nodenames:
            print(n,"kernel" in n)
        print([nodenames.index(s) for s in nodenames if "bias" in s])
        print([nodenames.index(s) for s in nodenames if "kernel" in s])
        if do_lower and not do_upper:
            lowup1  = do_lower*'lower'+do_upper*'upper'
            lowup2  = do_upper*'lower'+do_lower*'upper'
            varnames = [name for name in ncgrp.variables if lowup1 in name]
            for varname in varnames:
                oldvar = ncgrp.variables[varname]
                newvar = ncgrp.createVariable(varname[:-5]+lowup2,np.float32,oldvar.dimensions)
                newvar[:] = 0.

def main_frozengraph(dirname,nodes,name,outname,nlabel):
    serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(get_placeholderdict(name))    
    modelpath = basepath+"%s/%s/"%(dirname,name) 
#    if name =="Planck":
#        modelpath = basepath+"Cabauw/%s/%s/"%(dirname,name) 
    model = get_model(modelpath,name,nlabel,nodes)
    savedmodel_path = model.export_savedmodel(modelpath, serving_fn,
                        strip_default_attrs=True)
    frozengraph_path = modelpath+"/"+"frozen_graph_%s.pb"%outname
    
    frozen_graph_maker(savedmodel_path,frozengraph_path,'output'+'_noexp'*(name=='SSA'))
    tf.keras.backend.clear_session()
def main_loop():
    ###input function
    for nodes,dirname in [([64],"1L-64/"),([64,64],"2L-64_64/"),([32],"1L-32/"),([32,32],"2L-32_32/"),([32,64,128],"3L-32_64_128/")]:
        main_extractweights(dirname,nodes)
        #for name,outname,nlabel in [("tauLW","tlw",256)]:#[("Planck","plk",768)]:#,("SSA","ssa",224),("tauLW","tlw",256),("tauSW","tsw",224)]:
        #    main_frozengraph(dirname,nodes,name,outname,nlabel)

def main_extractweights(dirname,nodes):
    print(datapath+"weights_%s.nc"%dirname[3:])
    ncfile = nc.Dataset(basepath+"weights_%s.nc"%dirname[3:-1],"w")    

    grp = ncfile.createGroup("SSA"); name="SSA"
    modelpath = basepath+"%s/%s/"%(dirname,name)  
    serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(get_placeholderdict(name))    
    model = get_model(modelpath,name,224,nodes)
    savedmodel_path = model.export_savedmodel(modelpath, serving_fn,
                            strip_default_attrs=True)
    frozengraph_pathL = modelpath+"/"+"frozen_graph_ssa_lower.pb"
    frozengraph_pathU = modelpath+"/"+"frozen_graph_ssa_upper.pb"
    if do_lower: frozen_graph_maker(savedmodel_path,frozengraph_pathL,'output_lower')
    if do_upper: frozen_graph_maker(savedmodel_path,frozengraph_pathU,'output_upper')
    extract_weights(frozengraph_pathL,frozengraph_pathU,grp)
    tf.keras.backend.clear_session()
    
    grp = ncfile.createGroup("TSW"); name="tauSW"
    modelpath = basepath+"%s/%s/"%(dirname,name)  
    serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(get_placeholderdict(name))    
    model = get_model(modelpath,name,224,nodes)
    savedmodel_path = model.export_savedmodel(modelpath, serving_fn,
                            strip_default_attrs=True)
    frozengraph_pathL = modelpath+"/"+"frozen_graph_tsw_lower.pb"
    frozengraph_pathU = modelpath+"/"+"frozen_graph_tsw_upper.pb"
    if do_lower: frozen_graph_maker(savedmodel_path,frozengraph_pathL,'output_lower')
    if do_upper: frozen_graph_maker(savedmodel_path,frozengraph_pathU,'output_upper')
    extract_weights(frozengraph_pathL,frozengraph_pathU,grp)
    tf.keras.backend.clear_session()
    
    grp = ncfile.createGroup("TLW"); name="tauLW"
    modelpath = basepath+"%s/%s/"%(dirname,name)  
    serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(get_placeholderdict(name))    
    model = get_model(modelpath,name,256,nodes)
    savedmodel_path = model.export_savedmodel(modelpath, serving_fn,
                            strip_default_attrs=True)
    frozengraph_pathL = modelpath+"/"+"frozen_graph_tlw_lower.pb"
    frozengraph_pathU = modelpath+"/"+"frozen_graph_tlw_upper.pb"
    if do_lower: frozen_graph_maker(savedmodel_path,frozengraph_pathL,'output_lower')
    if do_upper: frozen_graph_maker(savedmodel_path,frozengraph_pathU,'output_upper')
    extract_weights(frozengraph_pathL,frozengraph_pathU,grp)
    tf.keras.backend.clear_session()
    
    grp = ncfile.createGroup("Planck"); name="Planck"
    modelpath = basepath+"%s/%s/"%(dirname,name)  
    serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(get_placeholderdict(name))    
    model = get_model(modelpath,name,768,nodes)
    savedmodel_path = model.export_savedmodel(modelpath, serving_fn,
                            strip_default_attrs=True)
    frozengraph_pathL = modelpath+"/"+"frozen_graph_plk_lower.pb"
    frozengraph_pathU = modelpath+"/"+"frozen_graph_plk_upper.pb"
    if do_lower: frozen_graph_maker(savedmodel_path,frozengraph_pathL,'output_lower')
    if do_upper: frozen_graph_maker(savedmodel_path,frozengraph_pathU,'output_upper')
    extract_weights(frozengraph_pathL,frozengraph_pathU,grp)
    ncfile.close()

if __name__ == "__main__":
#    main_extractweights("2L-64_64",[64,64])
    main_loop()
