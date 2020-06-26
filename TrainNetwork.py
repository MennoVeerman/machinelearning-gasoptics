import os
import tensorflow as tf
import numpy as np
import argparse
from main import *
    
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['KMP_BLOCKTIME'] = str(1)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = str(5)


#Setup parse function, which returns each sample in the format dict(features),labels
def _parse_function(protoexample):
    feature = {}
    for i in range(nlabel+nfeat): 
        feature[keys[i]] = tf.FixedLenFeature([],tf.float32)
    parsefeat = tf.parse_single_example(protoexample,feature)
    labels = []    
    for i in range(nlabel):
        labels += [parsefeat.pop(keys[nfeat+i])] 
    return parsefeat,labels

#Read data from TFRecord files
def inputfunc(filenames,train=True):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(map_func=_parse_function,num_parallel_calls=np.int32(os.environ['OMP_NUM_THREADS']))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size = 1000)
    if train:
        dataset = dataset.repeat()
    dataset = dataset.batch(args.batchsize)
    dataset.prefetch(1) 
    return dataset


def DNN_Regression(features,labels,mode,params):
    nfeat = len(params["keys"])
    do_train = params['train']
    args =params['argparser']#Optionally, we can train either only the troposhere (<9948 Pa) or above
    if args.do_upper:
        input_stack_upper = tf.stack([features[i] for i in params['keys']],axis=1)
        
        ft_means_upper = tf.constant(params["znorm"]["means_upper"][:nfeat], name='ft_mean_upr')
        ft_stdev_upper = tf.constant(params["znorm"]["stdev_upper"][:nfeat], name='ft_stdv_upr') 
        lb_means_upper = tf.constant(params["znorm"]["means_upper"][nfeat:], name='lb_mean_upr') 
        lb_stdev_upper = tf.constant(params["znorm"]["stdev_upper"][nfeat:], name='lb_stdv_upr') 
        
        init_upper = tf.subtract(input_stack_upper, tf.constant(np.zeros(nfeat).astype(np.float32)))
        
        # create layers
        layer_upper = tf.subtract(init_upper, ft_means_upper)
        layer_upper = tf.divide(layer_upper,ft_stdev_upper)
        for n_node in params["n_nodes"]:
            layer_upper = tf.layers.dense(layer_upper,
                                        units               = n_node,
                                        activation          = tf.nn.leaky_relu,
                                        kernel_initializer  = params["kernel_initializer"])
        output_layer_upper_raw = tf.layers.dense(layer_upper,
                                       units              = params["n_labels"],
                                       activation         = None,
                                       kernel_initializer = params["kernel_initializer"])
        
        output_layer_upper = tf.add(tf.multiply(output_layer_upper_raw,lb_stdev_upper),lb_means_upper,name='output_upper')

    if args.do_lower:
        input_stack_lower = tf.stack([features[i] for i in params['keys']],axis=1)
        
        ft_means_lower = tf.constant(params["znorm"]["means_lower"][:nfeat], name='ft_mean_lwr')
        ft_stdev_lower = tf.constant(params["znorm"]["stdev_lower"][:nfeat], name='ft_stdv_lwr')
        lb_means_lower = tf.constant(params["znorm"]["means_lower"][nfeat:], name='lb_mean_lwr')
        lb_stdev_lower = tf.constant(params["znorm"]["stdev_lower"][nfeat:], name='lb_stdv_lwr')
   
        init_lower = tf.subtract(input_stack_lower, tf.constant(np.zeros(nfeat).astype(np.float32)))
        
        # create layers
        layer_lower = tf.subtract(init_lower, ft_means_lower)
        layer_lower = tf.divide(layer_lower, ft_stdev_lower)
        for n_node in params["n_nodes"]:
            layer_lower = tf.layers.dense(layer_lower,
                                          units               = n_node,
                                          activation          = tf.nn.leaky_relu,
                                          kernel_initializer  = params["kernel_initializer"])
        output_layer_lower_raw = tf.layers.dense(layer_lower,
                                       units              = params["n_labels"],
                                       activation         = None,
                                       kernel_initializer = params["kernel_initializer"])
        
        output_layer_lower = tf.add(tf.multiply(output_layer_lower_raw, lb_stdev_lower), lb_means_lower,name='output_lower')

    #setup decaying learning rate, every 10 epochs
    if mode == tf.estimator.ModeKeys.TRAIN:
        rate = tf.train.exponential_decay(params['learning_rate'], tf.train.get_global_step(), decaystep, 0.8, staircase=True)
    
    if args.do_upper:
        output_layer_raw   = tf.identity(output_layer_upper_raw)
        output_layer       = tf.identity(output_layer_upper, name='output')
        if do_train:
            labels_upr = labels[:]
            labels = tf.identity(labels_upr)
    if args.do_lower:
        output_layer_raw   = tf.identity(output_layer_lower_raw)
        output_layer       = tf.identity(output_layer_lower, name='output')
        if do_train:
            labels_lwr = labels[:]
            labels  = tf.identity(labels_lwr)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'predictions' : output_layer}
        return tf.estimator.EstimatorSpec(
                mode,predictions=predictions,export_outputs={'predict':\
                                 tf.estimator.export.PredictOutput(predictions)})
            
    loss  = tf.losses.mean_squared_error(labels, output_layer_raw, reduction=tf.losses.Reduction.MEAN)
    if args.do_upper: loss_upr = tf.losses.mean_squared_error(labels_upr, output_layer_upper_raw, reduction=tf.losses.Reduction.MEAN)
    if args.do_lower: loss_lwr = tf.losses.mean_squared_error(labels_lwr, output_layer_lower_raw, reduction=tf.losses.Reduction.MEAN)
    metrics = {'eval_MSE':tf.metrics.mean_squared_error(labels, output_layer_raw)}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops = metrics)
    
    assert mode == tf.estimator.ModeKeys.TRAIN

    tf.summary.scalar("MSE", loss)

    if args.do_upper: 
        optimizer_upr = tf.train.AdamOptimizer(rate)
        train_op      = optimizer_upr.minimize(loss_upr, global_step = tf.train.get_global_step())
    if args.do_lower:
        optimizer_lwr = tf.train.AdamOptimizer(rate)
        train_op      = optimizer_lwr.minimize(loss_lwr, global_step = tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)
        
def run_model(name, n_label, nodes, dirname, args):
    if args.do_lower: lwrupr = 'lower'
    if args.do_upper: lwrupr = 'upper'
    trainnames = [args.datapath+"training%s_%s_%s.tfrecords"%(i,name,lwrupr) for i in range(args.filecount)]
    testnames  = [args.datapath+"testing_%s_%s.tfrecords"%(name,lwrupr)]
    np.random.shuffle(trainnames)
    
    global keys, nfeat, nlabel
    keys   = open(args.datapath+'keylist_%s.txt'%name,'r').readline().split(',')[:-1]
    nfeat  = len(keys[:-(n_label)])
    nlabel = n_label
    
    znorm = {}
    if args.do_upper:
        znorm["means_upper"] = np.loadtxt(args.datapath+"means_upr_%s.txt"%name,dtype=np.float32)
        znorm["stdev_upper"] = np.loadtxt(args.datapath+"stdev_upr_%s.txt"%name,dtype=np.float32)    
    if args.do_lower:
        znorm["means_lower"] = np.loadtxt(args.datapath+"means_lwr_%s.txt"%name,dtype=np.float32)
        znorm["stdev_lower"] = np.loadtxt(args.datapath+"stdev_lwr_%s.txt"%name,dtype=np.float32)

    hyperparams = {
        'train'             : True,
        'n_labels'          : n_label,
        'kernel_initializer': tf.glorot_uniform_initializer(),
        'znorm'             : znorm,
        'learning_rate'     : 0.01,
        "n_nodes"           : nodes,
        'keys'              : keys[:-(n_label)],
        'argparser'         : args
    }

    config=tf.ConfigProto(log_device_placement=False)
    config.intra_op_parallelism_threads = np.int32(os.environ['OMP_NUM_THREADS'])
    config.inter_op_parallelism_threads = 1
    
    myconfig = tf.estimator.RunConfig(session_config=config, save_summary_steps=10*steps/args.nepochs,\
                save_checkpoints_steps=10*steps/args.nepochs, log_step_count_steps=1000)     

    output_dir = args.trainpath+dirname+name+"/"
    if args.do_lower: output_dir += "lower_atm"
    if args.do_upper: output_dir += "upper_atm"
    DNNR = tf.estimator.Estimator(
      model_fn  = DNN_Regression,
      params    = hyperparams,
      config    = myconfig,
      model_dir = output_dir)
    
    profiler_hook = tf.train.ProfilerHook(save_steps = 10000, output_dir = output_dir) 
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:inputfunc(trainnames, True), max_steps=steps,hooks=[profiler_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:inputfunc(testnames, False), steps=None, start_delay_secs=0, throttle_secs=0)
    tf.estimator.train_and_evaluate(DNNR, train_spec, eval_spec)
  
def main(nodes, dirname, args): 
    global decaystep, steps
    if args.do_lower: size = args.trainsize_lwr
    if args.do_upper: size = args.trainsize_upr
    decaystep = size/args.batchsize * 10 
    steps = int(args.nepochs * size/args.batchsize)    
    
    if args.nn==0 or args.nn==1: run_model("Planck", 768, nodes, dirname, args)
    if args.nn==0 or args.nn==2: run_model("tauLW",  256, nodes, dirname, args)
    if args.nn==0 or args.nn==3: run_model("tauSW",  224, nodes, dirname, args)
    if args.nn==0 or args.nn==4: run_model("SSA",    224, nodes, dirname, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--args_from_file', default=False, action='store_true')
    parser.add_argument('--args_inp_file',  default='./arguments.txt', type=str)
    parser.add_argument('--batchsize',      default=128, type=int)
    parser.add_argument('--do_upper',       default=False, action='store_true')
    parser.add_argument('--do_lower',       default=False, action='store_true')
    parser.add_argument('--datapath',       default='./', type=str)
    parser.add_argument('--trainpath',      default='./', type=str)
    parser.add_argument('--filecount',      default=1 , type=int)
    parser.add_argument('--trainsize_lwr',  default=1000*72*0.9, type=int)
    parser.add_argument('--trainsize_upr',  default=1000*72*0.9, type=int)
    parser.add_argument('--nepochs',        default=500, type=int)
    parser.add_argument('--nn',        default=0, type=int)
    args = parser.parse_args()
    if args.args_from_file:
        read_run_arguments(args, args.args_inp_file)
        
    if args.do_lower and args.do_upper:
        args.do_lower = False 
        for nodes, dirname in [([32],"1L-32/")]:#, ([32,32],"2L-32_32/"), ([64],"1L-64/"), ([64,64],"2L-64_64/"), ([32,64,128],"3L-32_64_128/")]:
            main(nodes, dirname, args)

        args.do_lower = True; args.do_upper = False
        for nodes, dirname in [([32],"1L-32/")]:#, ([32,32],"2L-32_32/"), ([64],"1L-64/"), ([64,64],"2L-64_64/"), ([32,64,128],"3L-32_64_128/")]:
            main(nodes, dirname, args)

    else:
        for nodes, dirname in [([32],"1L-32/"), ([32,32],"2L-32_32/"), ([64],"1L-64/"), ([64,64],"2L-64_64/"), ([32,64,128],"3L-32_64_128/")]:
            main(nodes, dirname, args)
