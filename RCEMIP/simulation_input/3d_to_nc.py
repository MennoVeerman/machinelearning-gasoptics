import microhh_tools as mht     # available in microhh/python directory
import argparse
import os
import glob
import time as tm
import numpy as np
from multiprocessing import Pool


def convert_to_nc(variables):
    for variable in variables:
        filename = "{0}.nc".format(variable)
        dim = {
            'time': range(niter),
            'z': range(ktot),
            'y': range(jtot),
            'x': range(itot)}
        if variable is 'u':
            dim['xh'] = dim.pop('x')
        if variable is 'v':
            dim['yh'] = dim.pop('y')
        if variable is 'w':
            dim['zh'] = dim.pop('z')
        try:
            ncfile = mht.Create_ncfile(
                grid, filename, variable, dim, precision, compression)
            # Loop through the files and read 3d field
            for t in range(niter):
                otime = round((starttime + t * sampletime) / 10**iotimeprec)
                f_in = "{0:}.{1:07d}".format(variable, otime)

                try:
                    fin = mht.Read_binary(grid, f_in)
                except Exception as ex:
                    print (ex)
                    raise Exception(
                        'Stopping: cannot find file {}'.format(f_in))

                print("Processing %8s, time=%7i" % (variable, otime))
                ncfile.dimvar['time'][t] = otime * 10**iotimeprec
                if (perslice):
                    for k in range(ktot):
                        ncfile.var[t, k, :, :] = fin.read(itot * jtot)
                else:
                    ncfile.var[t, :, :, :] = fin.read(itot * jtot * ktot)

                fin.close()
            ncfile.close()
        except Exception as ex:
            print(ex)
            print("Failed to create %s" % filename)


# Parse command line and namelist options
parser = argparse.ArgumentParser(
    description='Convert MicroHH 3D binary to netCDF4 files.')
parser.add_argument('-d', '--directory', help='directory')
parser.add_argument('-f', '--filename', help='ini file name')
parser.add_argument('-v', '--vars', nargs='*', help='variable names')
parser.add_argument(
    '-p',
    '--precision',
    help='precision',
    choices=[
        'single',
         'double'])
parser.add_argument(
    '-t0',
    '--starttime',
    help='first time step to be parsed',
    type=float)
parser.add_argument(
    '-t1',
    '--endtime',
    help='last time step to be parsed',
    type=float)
parser.add_argument(
    '-tstep',
    '--sampletime',
    help='time interval to be parsed',
    type=float)
parser.add_argument(
    '-s',
    '--perslice',
    help='read/write per horizontal slice',
    action='store_true')
parser.add_argument(
    '-c',
    '--nocompression',
    help='do not compress the netcdf file',
    action='store_true')
parser.add_argument('-n', '--nprocs', help='Number of processes', type=int)
args = parser.parse_args()

if args.directory is not None:
    os.chdir(args.directory)

nl = mht.Read_namelist(args.filename)
itot = nl['grid']['itot']
jtot = nl['grid']['jtot']
ktot = nl['grid']['ktot']
starttime = args.starttime if args.starttime is not None else nl['time']['starttime']
endtime = args.endtime if args.endtime is not None else nl['time']['endtime']
sampletime = args.sampletime if args.sampletime is not None else nl['dump']['sampletime']
try:
    iotimeprec = nl['time']['iotimeprec']
except KeyError:
    iotimeprec = 0.

print("y",args.vars)
variables = args.vars if args.vars is not None else nl['dump']['dumplist']
print("x",variables)
precision = args.precision
perslice = args.perslice
compression = not(args.nocompression)
nprocs = args.nprocs if args.nprocs is not None else len(variables)

# End option parsing

# calculate the number of iterations
for time in np.arange(starttime, endtime, sampletime):
    otime = int(round(time / 10**iotimeprec))
    if not glob.glob('*.{0:07d}'.format(otime)):
        endtime = time - sampletime
        break

niter = int((endtime - starttime) / sampletime + 1)

grid = mht.Read_grid(itot, jtot, ktot)


chunks = [variables[i::nprocs] for i in range(nprocs)]

pool = Pool(processes=nprocs)

pool.imap_unordered(convert_to_nc, chunks)

pool.close()
pool.join()
