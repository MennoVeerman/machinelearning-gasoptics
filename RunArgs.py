import subprocess
import os 
def create_run_arguments(filename, argname, argval):
    fl = open(filename,'w')
    for i in range(len(argname)):
        fl.write("%s %s\n"%(argname[i], argval[i]))
    fl.close()
          

def read_run_arguments(argparser, filename):
    with open(filename,'r') as fl:
        for line in fl:
            arg, val = line.split()
            if arg in argparser:
                exec("argparser.%s = %s" % (arg, val))
                
def write_run_arguments(filename, argname, argval):
    fl = open(filename,'a')
    L = len(argname)
    for i in range(L):
        fl.write("%s %s\n"%(argname[i], argval[i]))
    fl.close()
          
    
if __name__ == "__main__":
    datapath = "rfmip/data"
    trainpath = "rfmip/train"
    filename = 'arguments.txt'
    if not os.path.exists(datapath):
        os.mkdir(datapath)
    if not os.path.exists(trainpath):
        os.mkdir(trainpath)
    
    create_run_arguments(filename,
                         ['filecount','datapath','trainpath',
                          'log_input','log_output','do_o3'],
                         [5, '\"%s/\"'%datapath, '\"%s/\"'%trainpath, True, True, True])
