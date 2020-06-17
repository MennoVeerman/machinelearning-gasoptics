import os
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--do_upper', default=False, action='store_true')
parser.add_argument('--do_lower', default=False, action='store_true')
parser.add_argument('--datapath', default='./', type=str)
parser.add_argument('--trainpath', default='./', type=str)
parser.add_argument('--filecount', default=1 , type=int)
parser.add_argument('--trainsize', default=1000*72*0.9, type=int)
args = parser.parse_args()

decaystep = args.trainsize/args.batchsize * 10 * (args.do_upper+args.do_lower) 
steps = int(658 * (args.do_upper+args.do_lower) * args.trainsize/args.batchsize)
    
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['KMP_BLOCKTIME'] = str(1)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = str(15)

#Setup parse function, which returns each sample in the format dict(features),labels
def _parse_function(protoexample):
    feature = {}
    for i in range(nlabel+nfeat+1): 
        feature[keys[i]] = tf.FixedLenFeature([],tf.float32)
    parsefeat = tf.parse_single_example(protoexample,feature)
    labels = []    
    for i in range(nlabel):
        labels += [parsefeat.pop(keys[nfeat+1+i])] 
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
    n_inp = len(params["keys"])
    do_train = params['train']
    
    #Optionally, we can train either only the troposhere (<9948 Pa) or above
    if args.do_upper:
        input_stack_upper = tf.boolean_mask(tf.stack([features[i] for i in params['keys']],axis=1),\
                                            features['tropo'] < 0.,axis=0)
        
        ft_means_upper = tf.constant(params["znorm"]["means_upper"][:n_inp], name='ft_mean_upr')
        ft_stdev_upper = tf.constant(params["znorm"]["stdev_upper"][:n_inp], name='ft_stdv_upr') 
        lb_means_upper = tf.constant(params["znorm"]["means_upper"][n_inp+1:], name='lb_mean_upr') 
        lb_stdev_upper = tf.constant(params["znorm"]["stdev_upper"][n_inp+1:], name='lb_stdv_upr') 
        
        init_upper = tf.subtract(input_stack_upper, tf.constant(np.zeros(n_inp).astype(np.float32)))
        
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
        input_stack_lower = tf.boolean_mask(tf.stack([features[i] for i in params['keys']],axis=1),\
                                            features['tropo'] > 0.,axis=0)
        
        ft_means_lower = tf.constant(params["znorm"]["means_lower"][:n_inp], name='ft_mean_lwr')
        ft_stdev_lower = tf.constant(params["znorm"]["stdev_lower"][:n_inp], name='ft_stdv_lwr')
        lb_means_lower = tf.constant(params["znorm"]["means_lower"][n_inp+1:], name='lb_mean_lwr')
        lb_stdev_lower = tf.constant(params["znorm"]["stdev_lower"][n_inp+1:], name='lb_stdv_lwr')
   
        init_lower = tf.subtract(input_stack_lower, tf.constant(np.zeros(n_inp).astype(np.float32)))
        
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
    rate = tf.train.exponential_decay(params['learning_rate'], tf.train.get_global_step(), decaystep, 0.8, staircase=True)
    
    if args.do_upper and args.do_lower:
        output_layer_raw   = tf.concat([output_layer_upper_raw, output_layer_lower_raw], axis=0)
        output_layer       = tf.concat([output_layer_upper, output_layer_lower], axis=0, name='output')
        if do_train:   
            labels_upr = tf.boolean_mask(labels[:], features['tropo']<0., axis=0)
            labels_lwr = tf.boolean_mask(labels[:], features['tropo']>0., axis=0)
            labels = tf.concat([labels_upr, labels_lwr], axis=0)
    elif args.do_upper:
        output_layer_raw   = tf.identity(output_layer_upper_raw)
        output_layer       = tf.identity(output_layer_upper, name='output')
        if do_train:
            labels_upr = tf.boolean_mask(labels[:], features['tropo']<0., axis=0)
            labels = tf.identity(labels_upr)
    elif args.do_lower:
        output_layer_raw   = tf.identity(output_layer_lower_raw)
        output_layer       = tf.identity(output_layer_lower, name='output')
        if do_train:
            labels_lwr = tf.boolean_mask(labels[:], features['tropo']>0., axis=0)
            labels  = tf.identity(labels_lwr)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'predictions' : output_layer_raw}
        return tf.estimator.EstimatorSpec(
                mode,predictions=predictions,export_outputs={'predict':\
                                 tf.estimator.export.PredictOutput(predictions)})
            
    loss  = tf.losses.mean_squared_error(labels, output_layer_raw, reduction=tf.losses.Reduction.MEAN)
    if args.do_upper: loss_upr = tf.losses.mean_squared_error(labels_upr, output_layer_upper_raw, reduction=tf.losses.Reduction.MEAN)
    if args.do_lower: loss_lwr = tf.losses.mean_squared_error(labels_lwr, output_layer_lower_raw, reduction=tf.losses.Reduction.MEAN)
    metrics = {'MSE':tf.metrics.mean_squared_error(labels, output_layer_raw)}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops = metrics)
    
    assert mode == tf.estimator.ModeKeys.TRAIN

    tf.summary.scalar("MSE", loss)

    if args.do_upper and args.do_lower: 
        optimizer_upr = tf.train.AdamOptimizer(rate)
        optimizer_lwr = tf.train.AdamOptimizer(rate)
        train_op_upr  = optimizer_upr.minimize(loss_upr, global_step = tf.train.get_global_step())
        train_op_lwr  = optimizer_lwr.minimize(loss_lwr, global_step = tf.train.get_global_step())
        train_op      = tf.group(train_op_upr, train_op_lwr)
    elif args.do_upper: 
        optimizer_upr = tf.train.AdamOptimizer(rate)
        train_op      = optimizer_upr.minimize(loss_upr, global_step = tf.train.get_global_step())
    elif args.do_lower:
        optimizer_lwr = tf.train.AdamOptimizer(rate)
        train_op      = optimizer_lwr.minimize(loss_lwr, global_step = tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)
        
def run_model(name, n_label, nodes, dirname):
    trainnames = [args.datapath+"training%s_%s.tfrecords"%(i,name) for i in range(args.filecount)]
    testnames  = [args.datapath+"testing_%s.tfrecords"%name]
    np.random.shuffle(trainnames)
    
    global keys
    keys = open(args.datapath+'keylist_%s.txt'%name,'r').readline().split(',')[:-1]

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
        'keys'              : ["h2o","p_lay","t_lay"]+(name=="Planck")*["t_levB","t_levT"]
    }

    config=tf.ConfigProto(log_device_placement=False)
    config.intra_op_parallelism_threads = np.int32(os.environ['OMP_NUM_THREADS'])
    config.inter_op_parallelism_threads = 1
    
    myconfig = tf.estimator.RunConfig(session_config=config, save_summary_steps=10000,\
                save_checkpoints_steps=10000, log_step_count_steps=1000)     

    output_dir = args.trainpath+dirname+name

    DNNR = tf.estimator.Estimator(
      model_fn  = DNN_Regression,
      params    = hyperparams,
      config    = myconfig,
      model_dir = output_dir)

    global nfeat,nlabel
    nfeat = len(hyperparams["keys"])
    nlabel = n_label
    
    profiler_hook = tf.train.ProfilerHook(save_steps = 10000, output_dir = output_dir) 
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:inputfunc(trainnames, True), max_steps=steps,hooks=[profiler_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:inputfunc(testnames, False), steps=10000, start_delay_secs=0, throttle_secs=0)
    tf.estimator.train_and_evaluate(DNNR, train_spec, eval_spec)
  
def main(nodes, dirname): 
    run_model("Planck", 768, nodes, dirname)
    run_model("tauLW",  256, nodes, dirname)
    run_model("tauSW",  224, nodes, dirname)
    run_model("SSA",    224, nodes, dirname)

if __name__ == '__main__':
    for nodes, dirname in [([32],"1L-32/"), ([32,32],"2L-32_32/"), ([64],"1L-64/"), ([64,64],"2L-64_64/"), ([32,64,128],"3L-32_64_128/")]:
        main(nodes, dirname)


