import os
import tensorflow as tf
import numpy as np
#import pandas as pd
import argparse
import sys
#from newparse import _parse_function2
print(sys.version)
#datapath = "C:/Users/veerm010/NeuralNetwork/AllTau/LogZscoreAll/data_split/TauLW_log_nCO2/"#data_split_log2/"TauSW_ray_log_nCO2/"#
datapath = "/scratch-shared/mveerman/PrepareData/Planck/"
#define parse
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize',default=128,type=int) #batchsize
parser.add_argument('--Layers',default=2,type=int) #layers
parser.add_argument('--NS',default=0,type=int) #repitiot nof steps
parser.add_argument('--nlabel',default=3*256,type=int) #number of labels to predict(out of 256)
parser.add_argument('--Learn',default=.01,type=float) #number of labels to predict(out of 256)
parser.add_argument('--outdir',default='run8-Linear',type=str) #output directory  
####used TSW4 for good tauSW, ssa2 for tauSW, TSW_log as log(tauSW)
####Without CO2(SW): TSW_log_nCO2, SSA_nCO2
####Without CO2(LW): TLW_nCO2, Planck_log_nCO2
###testing MSE weights: TLW_nCO2_reduced_MSEweight vs TLW_nCO2_reduced_MSEnormal
###OR TLW_nCO2_96x96_MSEweight_v2 vs TLW_nCO2_96x96_MSEnormal_v2
parser.add_argument('--Loss',default='MSE',type=str) #output directory
parser.add_argument('--Ntrain',default=40,type=int)#Number of training tfrecord files
parser.add_argument('--Ntest',default=10,type=int)  #Number of testing tfrecord files
parser.add_argument('--Nfeat',default=6,type=int)  #number of features in input data
args = parser.parse_args()

#Print logging info during training and evaluation
tf.logging.set_verbosity(tf.logging.INFO)

#This should make it run faster(only on linux?)
os.environ['KMP_BLOCKTIME'] = str(1)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = str(12)

#Define input files
trainnames =[datapath+"training%s_planck.tfrecords"%i for i in range(args.Ntrain)]
testnames = [datapath+'testing%s_planck.tfrecords'%i for i in range(args.Ntest)]
np.random.shuffle(trainnames)
np.random.shuffle(testnames)
#Setup parse function, which returns each sample in the format dict(features),labels
def _parse_function(protoexample):
    args = parser.parse_args()
    feature = {}
    for i in range(args.nlabel+args.Nfeat+1): #len(keys) #no_dp  #add tropo
        feature[keys[i]] = tf.FixedLenFeature([],tf.float32)
    parsefeat = tf.parse_single_example(protoexample,feature)
    ##Seperate labels
    labels = []    
    for i in range(args.nlabel):
        labels += [parsefeat.pop(keys[args.Nfeat+1+i])] 

    return parsefeat,labels

#Read data from TFRecord files
def recordinput(filenames,train=True):
    args = parser.parse_args()
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(map_func=_parse_function,num_parallel_calls=12)

    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size= 1000)

    if train:
        dataset = dataset.repeat()

    dataset = dataset.batch(args.batchsize)

    dataset.prefetch(1) 
    return dataset

#define feature columns
def define_feature_columns(keys):
    args=parser.parse_args()
    return [tf.feature_column.numeric_column(key=key) for key in keys[:args.Nfeat]]

def customloss_mse_gpt(x,y):  #x: labels, y: predictions
    mse = tf.square(tf.subtract(x,y))
    wgth = tf.constant(np.loadtxt(datapath+"gpt_weigtsk.txt",dtype=np.float32))
    meanwgth = tf.constant(np.mean(np.loadtxt(datapath+"gpt_weigtsk.txt",dtype=np.float32)))
    return tf.reduce_mean(tf.divide(mse*wgth,meanwgth))

def custommetric_mse_gpt(x,y):  #x: labels, y: predictions
    mse = tf.square(tf.subtract(x,y))
    wgth = tf.constant(np.loadtxt(datapath+"gpt_weigtsk.txt",dtype=np.float32))
    meanwgth = tf.constant(np.mean(np.loadtxt(datapath+"gpt_weigtsk.txt",dtype=np.float32)))
    return tf.metrics.mean(tf.divide(mse*wgth,meanwgth))

def customloss_mape(x,y): #x: labels, y: predictions
    return tf.reduce_mean(tf.abs(tf.div(tf.subtract(x,y),x)))

def customloss_smape(x,y): #x: labels, y: predictions
    tot = tf.math.add(tf.abs(x),tf.abs(y))
    return tf.reduce_mean(tf.div(tf.abs(tf.subtract(x,y)),tot))

def custommetric_mape(x,y): #x: labels, y: predictions
    mape = tf.abs(tf.div(tf.subtract(x,y),x))
    return tf.metrics.mean(mape)

def custommetric_smape(x,y): #x: labels, y: predictions
    tot = tf.math.add(tf.abs(x),tf.abs(y))
    mape = tf.div(tf.abs(tf.subtract(x,y)),tot)

def myexp(x):
    x = tf.add(1.,tf.divide(x,16.))   
    x = tf.multiply(x,x)
    x = tf.multiply(x,x)
    x = tf.multiply(x,x)
    x = tf.multiply(x,x)
    return x

Znorm = {
        "means_upper" : np.loadtxt(datapath+"means0_planck.txt",dtype=np.float32),
        "means_lower" : np.loadtxt(datapath+"means1_planck.txt",dtype=np.float32),
        "stdev_upper" : np.loadtxt(datapath+"stdev0_planck.txt",dtype=np.float32),
        "stdev_lower" : np.loadtxt(datapath+"stdev1_planck.txt",dtype=np.float32)
        }

def DNN_RegressionP(features,labels,mode,params):
    
    input_stack_upper = tf.boolean_mask(tf.stack([features[i] for i in ['h2o','o3','p_lay','t_lay','t_levB','t_levT']],axis=1),\
                                        features['tropo'] < 0.,axis=0)#,name='inp_upper_%s'%str(band)) #above tropopause
    input_stack_lower = tf.boolean_mask(tf.stack([features[i] for i in ['h2o','o3','p_lay','t_lay','t_levB','t_levT']],axis=1),\
                                        features['tropo'] > 0.,axis=0)#,name='inp_lower_%s'%str(band)) #below tropopause
    #means and standard deviations of features
    Fmeans_upper = tf.constant(params["Znorm"]["means_upper"][:6],name='FmeanU') #5
    Fmeans_lower = tf.constant(params["Znorm"]["means_lower"][:6],name='FmeanL') #5
    Fstdev_upper = tf.constant(params["Znorm"]["stdev_upper"][:6],name='FstdvU') #5
    Fstdev_lower = tf.constant(params["Znorm"]["stdev_lower"][:6],name='FstdvL') #5
   
    #means and standard deviations of labels    
    Lmeans_upper = tf.constant(params["Znorm"]["means_upper"][7:],name='LmeanU') #6
    Lmeans_lower = tf.constant(params["Znorm"]["means_lower"][7:],name='LmeanL') #6
    Lstdev_upper = tf.constant(params["Znorm"]["stdev_upper"][7:],name='LstdvU') #6
    Lstdev_lower = tf.constant(params["Znorm"]["stdev_lower"][7:],name='LstdvL') #6
    
    input_start_upper = tf.subtract(input_stack_upper,tf.constant(np.zeros(6).astype(np.float32)),name='zerosU')
    input_start_lower = tf.subtract(input_stack_lower,tf.constant(np.zeros(6).astype(np.float32)),name='zerosL')
    
    input_mean_upper = tf.subtract(input_start_upper,Fmeans_upper,name='meansU')
    input_mean_lower = tf.subtract(input_start_lower,Fmeans_lower,name='meansL')
    
    input_layer_upper = tf.divide(input_mean_upper,Fstdev_upper,name='inputU')
    input_layer_lower = tf.divide(input_mean_lower,Fstdev_lower,name='inputL')

#    #Optionally: define hidden layers 
#    HL1_upper = tf.layers.dense(input_layer_upper,
#                                   units               = params['hidden_layers'][0], 
#                                   activation          = tf.nn.leaky_relu,
#                                   kernel_initializer  = params["kernel_initializer"])
#    HL1_lower = tf.layers.dense(input_layer_lower,
#                                   units               = params['hidden_layers'][0], 
#                                   activation          = tf.nn.leaky_relu,
#                                   kernel_initializer  = params["kernel_initializer"])
#   HL2_upper = tf.layers.dense(HL1_upper,
#                                  units               = params['hidden_layers'][1], 
#                                  activation          = tf.nn.leaky_relu,
#                                  kernel_initializer  = params["kernel_initializer"])
#   HL2_lower = tf.layers.dense(HL1_lower,
#                                  units               = params['hidden_layers'][1], 
#                                  activation          = tf.nn.leaky_relu,
#                                  kernel_initializer  = params["kernel_initializer"])
#    HL3_upper = tf.layers.dense(HL2_upper,
#                                   units               = params['hidden_layers'][2], 
#                                   activation          = tf.nn.leaky_relu,
#                                   kernel_initializer  = params["kernel_initializer"])
#    HL3_lower = tf.layers.dense(HL2_lower,
#                                   units               = params['hidden_layers'][2], 
#                                   activation          = tf.nn.leaky_relu,
#                                   kernel_initializer  = params["kernel_initializer"])

    #define output layers
    output_layer_upper_raw = tf.layers.dense(input_layer_upper,
                                   units              = params["n_labels"],
                                   activation         = None,
                                   kernel_initializer = params["kernel_initializer"],name='outputA')
    output_layer_lower_raw = tf.layers.dense(input_layer_lower,
                                   units              = params["n_labels"],
                                   activation         = None,
                                   kernel_initializer = params["kernel_initializer"],name='outputB')
    #setup decaying learning rate, every 10 epochs
    rate = tf.train.exponential_decay(params['learning_rate'],tf.train.get_global_step(), 151875 ,0.8,staircase=True)
    output_layer_raw = tf.concat([output_layer_upper_raw,output_layer_lower_raw],axis=0)
    output_layer_upper = myexp(tf.add(tf.multiply(output_layer_upper_raw,Lstdev_upper),Lmeans_upper,name='output_upper'))
    output_layer_lower = myexp(tf.add(tf.multiply(output_layer_lower_raw,Lstdev_lower),Lmeans_lower,name='output_lower'))
    output_layer       = tf.concat([output_layer_upper,output_layer_lower],axis=0,name='output')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'predictions' : output_layer}
        return tf.estimator.EstimatorSpec(
                mode,predictions=predictions,export_outputs={'predict':\
                                 tf.estimator.export.PredictOutput(predictions)})
        
    labels1 = tf.boolean_mask(labels[:],features['tropo'] < 0.,axis=0)
    labels2 = tf.boolean_mask(labels[:],features['tropo'] > 0.,axis=0)
    labels= tf.concat([labels1,labels2],axis=0)
    
    #define loss and metrics
    if params['loss'] == 'MSE':
        loss  = tf.losses.mean_squared_error(labels,output_layer_raw,reduction=tf.losses.Reduction.MEAN)
        loss1  = tf.losses.mean_squared_error(labels1,output_layer_upper_raw,reduction=tf.losses.Reduction.MEAN)
        loss2  = tf.losses.mean_squared_error(labels2,output_layer_lower_raw,reduction=tf.losses.Reduction.MEAN)
        metrics = {'MSE':tf.metrics.mean_squared_error(labels,output_layer_raw)}
    if params['loss'] == 'MSE_weight':
        loss  = customloss_mse_gpt(labels,output_layer_raw)
        loss1  = customloss_mse_gpt(labels1,output_layer_upper_raw)
        loss2  = customloss_mse_gpt(labels2,output_layer_lower_raw)
        metrics = {'MSE':custommetric_mse_gpt(labels,output_layer_raw)}
    elif params['loss'] == 'MAPE':
        loss  = customloss_mape(labels,output_layer)
        metrics = {'MAPE':custommetric_mape(labels,output_layer)}
    elif params['loss'] == 'SMAPE':
        loss  = customloss_smape(labels,output_layer)
        metrics = {'MAPE':custommetric_smape(labels,output_layer)}
  
    #If evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops = metrics)
    
    #If training  
    assert mode == tf.estimator.ModeKeys.TRAIN
    
    #Define metrics for training
    tf.summary.scalar(params['loss'],loss)
    
    #define optimizer and training operation
    optimizer1 = tf.train.AdamOptimizer(rate)
    optimizer2 = tf.train.AdamOptimizer(rate)
#    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
#
    #Write trainiable variables(weigths, biases) to tensorboard as histograms
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name,var)
    
    train_op1  = optimizer1.minimize(loss1, global_step = tf.train.get_global_step())
    train_op2  = optimizer2.minimize(loss2, global_step = tf.train.get_global_step())
    train_op = tf.group(train_op1,train_op2)
    return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)
        

    
def main(hyperparameters,configuration=None,output_directory=None,steps=1001): #input: directory with hyperparameters
    #read keys from comma seperatedkeys.txt (created with prepare datascript) 
    global keys
    keys = open(datapath+'keylist.txt','r').readline().split(',')[:-1]
    args=parser.parse_args()
    
    #define feature columns
    DNNR = tf.estimator.Estimator(
      model_fn = DNN_RegressionP,
      params = hyperparameters,
      config = configuration,
      model_dir = output_directory
      )
  
    profiler_hook = tf.train.ProfilerHook(save_steps = 10000, output_dir = output_directory) 
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:recordinput(trainnames,True),max_steps=steps,hooks=[profiler_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:recordinput(testnames,False),steps=10000, start_delay_secs=5, throttle_secs=0)
    tf.estimator.train_and_evaluate(DNNR,train_spec,eval_spec)
    print("Training and evaluation done")       

#define default hyperparamet:qers
hyperparams = {
        'n_labels'          : args.nlabel,
        'kernel_initializer': tf.glorot_uniform_initializer(),
        'learning_rate'     : 0.01,
        'momentum'          : 0.99,   
        'Znorm'             : Znorm                      
        }

#define default configuration object
config=tf.ConfigProto(log_device_placement=False)
config.intra_op_parallelism_threads = 12
config.inter_op_parallelism_threads = 2
myconfig = tf.estimator.RunConfig(session_config=config,save_summary_steps=10000,\
            save_checkpoints_steps=100000,log_step_count_steps=1000)     

if __name__ == '__main__':
    outpath = "/scratch-shared/mveerman/Training/Planck/"
    outdir = outpath+args.outdir
    hyperparams['hidden_layers'] = [64,64,128] #[args.Nodes] * args.Layer
    hyperparams['learning_rate'] = args.Learn
    hyperparams['loss'] = args.Loss

    main(hyperparams,myconfig,output_directory=outdir,steps=10000000) #number of nodes per layer


