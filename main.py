import subprocess
def create_run_arguments(filename, argname, argval):
    fl = open(filename,'w')
    for i in range(len(argname)):
        fl.write("%s %s\n"%(argname[i], argval[i]))
    fl.close()
          

def read_run_arguments(argparser, filename):
    with open(filename,'r+') as fl:
        for line in fl:
            arg, val = line.split()
            if arg in argparser:
                exec("argparser.%s = %s" % (arg, val))
                
def write_run_arguments(filename, argname, argval):
    fl = open(filename,'r+')
    for i in range(len(argname)):
        fl.write("%s %s\n"%(argname[i], argval[i]))
    fl.close()
          
    
if __name__ == "__main__":
    filename = 'arguments.txt'
    create_run_arguments(filename,
                         ['filecount','datapath','trainpath',
                          'log_input','log_output','do_o3'],
                         [5, "data/", "train/", True, False, True])
    
    subprocess.call(['python', 'PrepateData.py', '--args_from_file', 'True', '--args_inp_file', 'arguments.txt'])
            
