import os
import tensorflow as tf
import numpy as np
import sys
from multiprocessing import Process
batchsize = 128
do_upper  = True
do_lower  = True
decaystep = 5062.5*(do_upper+do_lower)
#Print logging info during training and evaluation
tf.logging.set_verbosity(tf.logging.INFO)
#This should make it run faster(only on linux?)
os.environ['KMP_BLOCKTIME'] = str(0)# str(1)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = str(12)
#Define input files

def myexp(x):
    x = tf.add(1.,tf.divide(x,16.))   
    x = tf.multiply(x,x)
    x = tf.multiply(x,x)
    x = tf.multiply(x,x)
    x = tf.multiply(x,x)
    return x

#Setup parse function, which returns each sample in the format dict(features),labels
def _parse_function(protoexample):
    feature = {}
    for i in range(Nlabel+Nfeat+1): #len(keys) #no_dp  #add tropo
        feature[keys[i]] = tf.FixedLenFeature([],tf.float32)
    parsefeat = tf.parse_single_example(protoexample,feature)
    ##Seperate labels
    labels = []    
    for i in range(Nlabel):
        labels += [parsefeat.pop(keys[Nfeat+1+i])] 

    return parsefeat,labels

#Read data from TFRecord files
def recordinput(filenames,train=True):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(map_func=_parse_function,num_parallel_calls=12)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size= 1000)
    if train:
        dataset = dataset.repeat()
    dataset = dataset.batch(batchsize)
    dataset.prefetch(1) 
    return dataset


def DNN_Regression(features,labels,mode,params):
    ninp = len(params["keys"])
    do_train = params['train']
    #Optionally, we can train either only the troposhere (<9948 Pa) or above
    if do_upper:
        input_stack_upper = tf.boolean_mask(tf.stack([features[i] for i in params['keys']],axis=1),\
                                            features['tropo'] < 0.,axis=0)
        
        Fmeans_upper = tf.constant(params["Znorm"]["means_upper"][:ninp],name='FmeanU') #5
        Fstdev_upper = tf.constant(params["Znorm"]["stdev_upper"][:ninp],name='FStdvU') #5
        Lmeans_upper = tf.constant(params["Znorm"]["means_upper"][ninp+1:]) #6
        Lstdev_upper = tf.constant(params["Znorm"]["stdev_upper"][ninp+1:]) #6
        
        input_start_upper = tf.subtract(input_stack_upper,tf.constant(np.zeros(ninp).astype(np.float32)),name='zerosU')
        input_mean_upper  = tf.subtract(input_start_upper,Fmeans_upper,name='meansU')
        ################
        Layer_upper = tf.divide(input_mean_upper,Fstdev_upper,name='inputU')
        for Nnode in params["Nnodes"]:
            Layer_upper = tf.layers.dense(Layer_upper,
                                        units               = Nnode,
                                        activation          = tf.nn.leaky_relu,
                                        kernel_initializer  = params["kernel_initializer"])
        output_layer_upper_raw = tf.layers.dense(Layer_upper,
                                       units              = params["n_labels"],
                                       activation         = None,
                                       kernel_initializer = params["kernel_initializer"],name='outputA')
        output_layer_upper_noexp = tf.add(tf.multiply(output_layer_upper_raw,Lstdev_upper),Lmeans_upper,name='output_upper_noexp')
        output_layer_upper = myexp(tf.add(tf.multiply(output_layer_upper_raw,Lstdev_upper),Lmeans_upper,name='output_upper'))

    if do_lower:
        input_stack_lower = tf.boolean_mask(tf.stack([features[i] for i in params['keys']],axis=1),\
                                            features['tropo'] > 0.,axis=0)

        Fmeans_lower = tf.constant(params["Znorm"]["means_lower"][:ninp],name='FmeanL') #5
        Fstdev_lower = tf.constant(params["Znorm"]["stdev_lower"][:ninp],name='FstdvL') #5
        Lmeans_lower = tf.constant(params["Znorm"]["means_lower"][ninp+1:]) #6
        Lstdev_lower = tf.constant(params["Znorm"]["stdev_lower"][ninp+1:]) #6
        input_start_lower = tf.subtract(input_stack_lower,tf.constant(np.zeros(ninp).astype(np.float32)),name='zerosL')
        input_mean_lower  = tf.subtract(input_start_lower,Fmeans_lower,name='meansL')
        ################
        Layer_lower = tf.divide(input_mean_lower,Fstdev_lower,name='inputU')
        for Nnode in params["Nnodes"]:
            Layer_lower = tf.layers.dense(Layer_lower,
                                        units               = Nnode,
                                        activation          = tf.nn.leaky_relu,
                                        kernel_initializer  = params["kernel_initializer"])
        output_layer_lower_raw = tf.layers.dense(Layer_lower,
                                       units              = params["n_labels"],
                                       activation         = None,
                                       kernel_initializer = params["kernel_initializer"],name='outputB')
        output_layer_lower_noexp = tf.add(tf.multiply(output_layer_lower_raw,Lstdev_lower),Lmeans_lower,name='output_lower_noexp')
        output_layer_lower = myexp(tf.add(tf.multiply(output_layer_lower_raw,Lstdev_lower),Lmeans_lower,name='output_lower'))

    #setup decaying learning rate, every 10 epochs
    rate = tf.train.exponential_decay(params['learning_rate'],tf.train.get_global_step(), decaystep ,0.8,staircase=True)
    
    if do_upper and do_lower:
        output_layer_raw = tf.concat([output_layer_upper_raw,output_layer_lower_raw],axis=0)
        output_layer     = tf.concat([output_layer_upper,output_layer_lower],axis=0,name='output')
        output_layer_noexp = tf.concat([output_layer_upper_noexp,output_layer_lower_noexp],axis=0,name='output_noexp')
        if do_train:   
            labels1 = tf.boolean_mask(labels[:],features['tropo'] < 0.,axis=0)
            labels2 = tf.boolean_mask(labels[:],features['tropo'] > 0.,axis=0)
            labels= tf.concat([labels1,labels2],axis=0)
    elif do_upper:
        output_layer_raw = tf.identity(output_layer_upper_raw)
        output_layer     = tf.identity(output_layer_upper,name='output')
        if do_train:
            labels1 = tf.boolean_mask(labels[:],features['tropo'] < 0.,axis=0)
            labels  = tf.identity(labels1)
    elif do_lower:
        output_layer_raw = tf.identity(output_layer_lower_raw)
        output_layer_noexp  = tf.identity(output_layer_lower_noexp,name='output_noexp')
        output_layer     = tf.identity(output_layer_lower,name='output')
        if do_train:
            labels2 = tf.boolean_mask(labels[:],features['tropo'] > 0.,axis=0)
            labels  = tf.identity(labels2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'predictions' : output_layer}
        return tf.estimator.EstimatorSpec(
                mode,predictions=predictions,export_outputs={'predict':\
                                 tf.estimator.export.PredictOutput(predictions)})
            
    #define loss and metrics
    loss  = tf.losses.mean_squared_error(labels,output_layer_raw,reduction=tf.losses.Reduction.MEAN)
    if do_upper: loss1  = tf.losses.mean_squared_error(labels1,output_layer_upper_raw,reduction=tf.losses.Reduction.MEAN)
    if do_lower: loss2  = tf.losses.mean_squared_error(labels2,output_layer_lower_raw,reduction=tf.losses.Reduction.MEAN)
    metrics = {'MSE':tf.metrics.mean_squared_error(labels,output_layer_raw)}

    #If evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops = metrics)
    
    #If training  
    assert mode == tf.estimator.ModeKeys.TRAIN
    
    #Define metrics for training
    tf.summary.scalar("MSE",loss)

    #define optimizer and training operation
    if do_upper and do_lower: 
        optimizer1 = tf.train.AdamOptimizer(rate)
        optimizer2 = tf.train.AdamOptimizer(rate)
        train_op1  = optimizer1.minimize(loss1, global_step = tf.train.get_global_step())
        train_op2  = optimizer2.minimize(loss2, global_step = tf.train.get_global_step())
        train_op   = tf.group(train_op1,train_op2)
    elif do_upper: 
        optimizer1 = tf.train.AdamOptimizer(rate)
        train_op   = optimizer1.minimize(loss1, global_step = tf.train.get_global_step())
    elif do_lower:
        optimizer2 = tf.train.AdamOptimizer(rate)
        train_op   = optimizer2.minimize(loss2, global_step = tf.train.get_global_step())

#    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
#
    return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)
        


def run_model(name,nlabel,nodes,datapath,steps,dirname, myiter):
    trainnames = [datapath+"training%s_%s.tfrecords"%(i,name) for i in range(3)]
    testnames  = [datapath+"testing_%s.tfrecords"%name]

    global keys
    keys = open(datapath+'keylist_%s.txt'%name,'r').readline().split(',')[:-1]

    Znorm = {}
    if do_upper:
        Znorm["means_upper"] = np.loadtxt(datapath+"means0_%s.txt"%name,dtype=np.float32)
        Znorm["stdev_upper"] = np.loadtxt(datapath+"stdev0_%s.txt"%name,dtype=np.float32)    
    if do_lower:
        Znorm["means_lower"] = np.loadtxt(datapath+"means1_%s.txt"%name,dtype=np.float32)
        Znorm["stdev_lower"] = np.loadtxt(datapath+"stdev1_%s.txt"%name,dtype=np.float32)

    hyperparams = {
        'train'             : True,
        'n_labels'          : nlabel,
        'kernel_initializer': tf.glorot_uniform_initializer(),
        'Znorm'             : Znorm,
        'learning_rate'     : .01,
        "Nnodes"            : nodes,
        'keys'              : ["h2o","p_lay","t_lay"]+(name=="Planck")*["t_levB","t_levT"],
    }

    config=tf.ConfigProto(log_device_placement=False)
    config.intra_op_parallelism_threads = 12
    config.inter_op_parallelism_threads = 2
    myconfig = tf.estimator.RunConfig(session_config=config,save_summary_steps=10000,\
                save_checkpoints_steps=10000,log_step_count_steps=1000)     

    output_dir = '/scratch-shared/mveerman/Training/RCEMIP/'+dirname+name#+myiter

    DNNR = tf.estimator.Estimator(
      model_fn = DNN_Regression,
      params = hyperparams,
      config = myconfig,
      model_dir = output_dir
      )
    global Nfeat,Nlabel
    Nfeat = len(hyperparams["keys"])
    Nlabel = nlabel
    profiler_hook = tf.train.ProfilerHook(save_steps = 10000, output_dir = output_dir) 
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:recordinput(trainnames,True),max_steps=steps,hooks=[profiler_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:recordinput(testnames,False),steps=10000, start_delay_secs=0, throttle_secs=0)
    tf.estimator.train_and_evaluate(DNNR,train_spec,eval_spec)
  
def main(datapath,steps,nodes,dirname,myiter): 
    run_model("Planck", 768, nodes, datapath, steps, dirname, myiter)
    run_model("tauLW", 256, nodes, datapath, steps, dirname, myiter)
    run_model("tauSW", 224, nodes, datapath, steps, dirname, myiter)
    run_model("SSA",   224, nodes, datapath, steps, dirname, myiter)

if __name__ == '__main__':
    datapath = "/scratch-shared/mveerman/PrepareData/RCEMIP/"
    steps = 666630
    for myiter in ["1"]:
        for nodes,dirname in [([64],"1L-64/"),([64,64],"2L-64_64/"),([32],"1L-32/"),([32,32],"2L-32_32/"),([32,64,128],"3L-32_64_128/")]:
            main(datapath,steps,nodes,dirname,myiter)


