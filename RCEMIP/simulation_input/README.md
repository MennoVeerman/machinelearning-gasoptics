# RCEMIP simulation
Input for the RCEMIP LES-simulation performed with microHH (*Heerwaarden et al. (2017). MicroHH 1.0: a computational fluid dynamics code for direct numerical simulation and large-eddy simulation of atmospheric boundary layer flows, GMD.*
. For a detailed description of the case, see *Wing et al. (2018). Radiative Convective Equilibrium Intercomparison Project (RCEMIP), GMD.*

Intruction to run the case:
1. run the rcemip.py script to generate the input netCDF file
2. obtain and build the computational fluid dynamics code microHH (see https://github.com/microhh/microhh), copy the microhh executable to this directory
3. follow instruction at https://github.com/microhh/microhh/tree/master/cases/rcemip to obtain input for the radiative transfer computations
4. run the case: 

    ./microhh init rcemip
 
    ./microhh run rcemip

5. run 3d\_to\_nc.py to obtain netCDF output files of qt, ql and T









