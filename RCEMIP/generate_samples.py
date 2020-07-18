import numpy as np
import netCDF4 as nc
import argparse
from multiprocessing import Process
parser = argparse.ArgumentParser()
parser.add_argument('--iexpt', default=0, type=int)
parser.add_argument('--nfile', default=1, type=int)
parser.add_argument('--dpath', default="data", type=str)
args = parser.parse_args()
iexpt = args.iexpt
nfile = args.nfile
dpath = args.dpath

rfmippath = "../rfmip/"
float_type = "f8"
ep         = 0.622


#Saturation specific humidity and interpolation functions
def fill_ncstring(nc_var,str_var):
    for i in range(len(str_var)):
        for j in range(len(str_var[i])):
            nc_var[i,j] = str_var[i][j]
        for j in range(len(str_var[i]),10):
            nc_var[i,j] = " "

def sample_v(vmin,vmax):
    return vmin + np.random.random(1) * (vmax-vmin) 

def sample_v_prof(p,p_prof,vmin_prof,vmax_prof,skewed_rand): 
    vmin = np.interp(p,p_prof,vmin_prof) 
    vmax = np.interp(p,p_prof,vmax_prof) 
    if vmax/vmin >= 10. and np.random.random(1) > .5:
        return vmin + skewed_rand * (vmax-vmin)
    else:
        return vmin + np.random.random(1) * (vmax-vmin)

def get_o3(p):    
    p_hpa = p / 100.
    return np.maximum(5e-9,3.6478 * p_hpa**0.83209 * np.exp(-p_hpa/11.3515) * 1e-6)

def create_file(ifile): 
    nc_file_rfmip = nc.Dataset(rfmippath+'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', mode='r', datamodel='NETCDF4', clobber=False)
    np.random.seed(ifile)
        
    #skewed random numbers 
    randskew = -np.log(1-np.random.random(10000000))
    randskew /= np.max(randskew)
    np.random.shuffle(randskew)

    ############
    # User defined values or profiles of minimum and maximum allowed temperature, h20 and possible other gas concentrations
    ############
    data  = np.loadtxt("profiles.txt",skiprows=1)
    pres  = data[::-1,0]
    p_min = np.min(pres) - 100.
    p_max = np.max(pres) + 100.
    T_min = data[::-1,1] * 0.99
    T_max = data[::-1,2] * 1.01
    q_min = data[::-1,3] * 0.95
    q_max = data[::-1,4] * 1.05
    
    nlay    = len(pres)
    temp_pres = np.zeros(nlay+2)
    temp_pres[1:-1] = pres
    temp_pres[0]  = temp_pres[1]  + (temp_pres[1] - temp_pres[2])
    temp_pres[-1] = temp_pres[-2] + (temp_pres[-2]- temp_pres[-3])
    
    ncol = 100
    p_lay = np.zeros((nlay,ncol))
    T_lay = np.zeros((nlay,ncol))
    h2o_lay = np.zeros((nlay,ncol))
    o3_lay = np.zeros((nlay,ncol))

    for i in range(nlay):
        for j in range(ncol):
            p_lay[i,j]   = sample_v(temp_pres[i],temp_pres[i+2])
    p_lay = np.sort(p_lay,axis=0)

    irand = 0
    for i in range(nlay):
        for j in range(ncol):
            T_lay[i,j]   = sample_v_prof(p_lay[i,j],pres,T_min,T_max,randskew[irand])
            h2o_lay[i,j]   = sample_v_prof(p_lay[i,j],pres,q_min,q_max,randskew[irand])
            irand += 1
    h2o_lay = h2o_lay/(ep-ep*h2o_lay)

    for i in range(nlay):
        for j in range(ncol):
            o3_lay[i,j]  = get_o3(p_lay[i,j])


    #### get t_lev:
    T_lev = np.zeros((nlay+1,ncol))
    T_lev[1:-1,:] = T_lay[1:] + (T_lay[:-1]-T_lay[1:]) * np.random.random(T_lay[1:].shape)
    T_lev[0,:]  = T_lay[0]  + (T_lay[0]-T_lev[1]) * (np.random.random(ncol)*2)
    T_lev[-1,:] = T_lay[-1] + (T_lay[-1]-T_lev[-2]) * (np.random.random(ncol)*2)

    #### get p_lev:
    p_lev = np.zeros((nlay+1,ncol))
    p_lev[1:-1,:] = (p_lay[1:,:] + p_lay[:-1,:]) / 2.
    p_lev[0,:]  = p_lay[0]  + (p_lay[0]-p_lev[1])
    p_lev[-1,:] = p_lay[-1] + (p_lay[-1]-p_lev[-2])


    #### create netcdf file
    nc_file = nc.Dataset(dpath+"/rte_rrtmgp_input_{:03d}.nc".format(ifile), mode="w", datamodel="NETCDF4", clobber=True)
    
    nc_file.createDimension("lay", p_lay.shape[0])
    nc_file.createDimension("lev", p_lev.shape[0])
    nc_file.createDimension("col", p_lay.shape[1])
    nc_file.createDimension("band_lw", 16)
    nc_file.createDimension("band_sw", 14)
    
    nc_p_lay = nc_file.createVariable("p_lay", float_type, ("lay","col"))
    nc_p_lev = nc_file.createVariable("p_lev", float_type, ("lev","col"))
    nc_T_lay = nc_file.createVariable("t_lay", float_type, ("lay","col"))
    nc_T_lev = nc_file.createVariable("t_lev", float_type, ("lev","col"))

    nc_p_lay[:] = p_lay[::-1]
    nc_p_lev[:] = p_lev[::-1]
    nc_T_lay[:] = T_lay[::-1]
    nc_T_lev[:] = T_lev[::-1]

    nc_surface_emissivity = nc_file.createVariable('emis_sfc', float_type, ('col', 'band_lw'))
    nc_surface_temperature = nc_file.createVariable('t_sfc', float_type, 'col')
    
    nc_surface_emissivity[:] = nc_file_rfmip.variables['surface_emissivity'][0]
    nc_surface_temperature[:] = 300. #we don't need this for data generation
    
    nc_surface_albedo_dir = nc_file.createVariable('sfc_alb_dir', float_type, ('col', 'band_sw'))
    nc_surface_albedo_dif = nc_file.createVariable('sfc_alb_dif', float_type, ('col', 'band_sw'))
    
    nc_surface_albedo_dir[:,:] = .07
    nc_surface_albedo_dif[:,:] = .07
    
    nc_mu0 = nc_file.createVariable('mu0', float_type, ('col'))
    nc_mu0[:] = np.cos(np.deg2rad(42.))
    
    nc_h2o     = nc_file.createVariable('vmr_h2o'    , float_type, ('lay', 'col'))
    nc_o3      = nc_file.createVariable('vmr_o3'     , float_type, ('lay', 'col'))
    nc_h2o[:]  = h2o_lay[::-1]
    nc_o3[:]   = o3_lay[::-1]
    
    nc_co2     = nc_file.createVariable('vmr_co2'    , float_type)
    nc_n2o     = nc_file.createVariable('vmr_n2o'    , float_type)
    nc_co      = nc_file.createVariable('vmr_co'     , float_type)
    nc_ch4     = nc_file.createVariable('vmr_ch4'    , float_type)
    nc_o2      = nc_file.createVariable('vmr_o2'     , float_type)
    nc_n2      = nc_file.createVariable('vmr_n2'     , float_type)
    nc_ccl4    = nc_file.createVariable('vmr_ccl4'   , float_type)
    nc_cfc11   = nc_file.createVariable('vmr_cfc11'  , float_type)
    nc_cfc12   = nc_file.createVariable('vmr_cfc12'  , float_type)
    nc_cfc22   = nc_file.createVariable('vmr_cfc22'  , float_type)
    nc_hfc143a = nc_file.createVariable('vmr_hfc143a', float_type)
    nc_hfc125  = nc_file.createVariable('vmr_hfc125' , float_type)
    nc_hfc23   = nc_file.createVariable('vmr_hfc23'  , float_type)
    nc_hfc32   = nc_file.createVariable('vmr_hfc32'  , float_type)
    nc_hfc134a = nc_file.createVariable('vmr_hfc134a', float_type)
    nc_cf4     = nc_file.createVariable('vmr_cf4'    , float_type)
    
    # Constants
    nc_co2    [:] = nc_file_rfmip.variables['carbon_dioxide_GM'][iexpt] * float(nc_file_rfmip.variables['carbon_dioxide_GM'].units)
    nc_n2o    [:] = nc_file_rfmip.variables['nitrous_oxide_GM'][iexpt] * float(nc_file_rfmip.variables['nitrous_oxide_GM'].units)
    nc_co     [:] = nc_file_rfmip.variables['carbon_monoxide_GM'][iexpt] * float(nc_file_rfmip.variables['carbon_monoxide_GM'].units)
    nc_ch4    [:] = nc_file_rfmip.variables['methane_GM'][iexpt] * float(nc_file_rfmip.variables['methane_GM'].units)
    nc_o2     [:] = nc_file_rfmip.variables['oxygen_GM'][iexpt] * float(nc_file_rfmip.variables['oxygen_GM'].units)
    nc_n2     [:] = nc_file_rfmip.variables['nitrogen_GM'][iexpt] * float(nc_file_rfmip.variables['nitrogen_GM'].units)
    nc_ccl4   [:] = nc_file_rfmip.variables['carbon_tetrachloride_GM'][iexpt] * float(nc_file_rfmip.variables['carbon_tetrachloride_GM'].units)
    nc_cfc11  [:] = nc_file_rfmip.variables['cfc11_GM'][iexpt] * float(nc_file_rfmip.variables['cfc11_GM'].units)
    nc_cfc12  [:] = nc_file_rfmip.variables['cfc12_GM'][iexpt] * float(nc_file_rfmip.variables['cfc12_GM'].units)
    nc_cfc22  [:] = nc_file_rfmip.variables['hcfc22_GM'][iexpt] * float(nc_file_rfmip.variables['hcfc22_GM'].units)
    nc_hfc143a[:] = nc_file_rfmip.variables['hfc143a_GM'][iexpt] * float(nc_file_rfmip.variables['hfc143a_GM'].units)
    nc_hfc125 [:] = nc_file_rfmip.variables['hfc125_GM'][iexpt] * float(nc_file_rfmip.variables['hfc125_GM'].units)
    nc_hfc23  [:] = nc_file_rfmip.variables['hfc23_GM'][iexpt] * float(nc_file_rfmip.variables['hfc23_GM'].units)
    nc_hfc32  [:] = nc_file_rfmip.variables['hfc32_GM'][iexpt] * float(nc_file_rfmip.variables['hfc32_GM'].units)
    nc_hfc134a[:] = nc_file_rfmip.variables['hfc134a_GM'][iexpt] * float(nc_file_rfmip.variables['hfc134a_GM'].units)
    nc_cf4    [:] = nc_file_rfmip.variables['cf4_GM'][iexpt] * float(nc_file_rfmip.variables['cf4_GM'].units)
    nc_file.close()

if __name__=='__main__':
    procs = []
    for ifile in range(nfile):
        procs += [Process(target=create_file, args=(ifile,))]
    for i in range(len(procs)):
        procs[i].start()
    for i in range(len(procs)):
        procs[i].join()


