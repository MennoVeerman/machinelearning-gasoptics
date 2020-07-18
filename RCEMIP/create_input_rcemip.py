#import matplotlib.pyplot as pl
import numpy as np
from shutil import copyfile
import netCDF4 as nc
path = "/archive/mveerman/machinelearning/rcemip_sim/"
rfmippath = "../rfmip/"

float_type = "f8"
Tsfc = 300
ep = 0.622
wh2o_min = 5e-6

def interp_to_lev(tlay):
    nt,nz,ny,nx = tlay.shape
    tlev = np.zeros((nt,nz+1,ny,nx))
    tlev[:,1:-1] = (tlay[:,1:]+tlay[:,:-1]) / 2.
    tlev[:,0] = 2 * tlay[:,0] - tlev[:,1]
    tlev[:,-1] = 2 * tlay[:,-1] - tlev[:,-2]
    return tlev

def get_o3(p):
    p_hpa = p / 100.
    return np.maximum(5e-9,3.6478 * p_hpa**0.83209 * np.exp(-p_hpa/11.3515) * 1e-6)

qt   = nc.Dataset(path+"qt.nc").variables['qt'][:]
ql   = nc.Dataset(path+"ql.nc").variables['ql'][:]
qi   = nc.Dataset(path+"qi.nc").variables['qi'][:]

T_lay = nc.Dataset(path+"T.nc").variables['T'][:]
T_lev = interp_to_lev(T_lay)

pfile1 = nc.Dataset(path+"rcemip.default.0000000.nc").groups['thermo']
pfile2 = nc.Dataset(path+"rcemip.default.0008640.nc").groups['thermo']
p_lay = np.append(pfile1.variables['phydro'][::24],pfile2.variables['phydro'][::24],axis=0)
p_lev = np.append(pfile1.variables['phydroh'][::24],pfile2.variables['phydroh'][::24],axis=0)
nt,nz,ny,nx = qt.shape

qv  = qt-(ql+qi)
h2o = qv/(ep-ep*qv)
h2o = np.maximum(h2o,wh2o_min) 

##########################
nc_file_rfmip = nc.Dataset(rfmippath+'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', mode='r', datamodel='NETCDF4', clobber=False)
nc_file = nc.Dataset("rte_rrtmgp_input.nc", mode="w", datamodel="NETCDF4", clobber=True)

iexpt = 0
Nprofs = 100 
Lprofs = []
iprof  = 0

nc_file.createDimension("lay", p_lay.shape[1])
nc_file.createDimension("lev", p_lev.shape[1])
nc_file.createDimension("col", Nprofs)
nc_file.createDimension("band_lw", 16)
nc_file.createDimension("band_sw", 14)

nc_surface_emissivity = nc_file.createVariable('emis_sfc', float_type, ('col', 'band_lw'))
nc_surface_emissivity[:] = nc_file_rfmip.variables['surface_emissivity'][0]

nc_surface_temperature = nc_file.createVariable('t_sfc', float_type, 'col')
nc_surface_temperature[:] = Tsfc

nc_surface_albedo_dir = nc_file.createVariable('sfc_alb_dir', float_type, ('col', 'band_sw'))
nc_surface_albedo_dif = nc_file.createVariable('sfc_alb_dif', float_type, ('col', 'band_sw'))

nc_surface_albedo_dir[:,:] = .07
nc_surface_albedo_dif[:,:] = .07

nc_mu0 = nc_file.createVariable('mu0', float_type, ('col'))
nc_mu0[:] = np.cos(np.deg2rad(42.))

nc_p_lay     = nc_file.createVariable('p_lay'    , float_type, ('lay', 'col'))
nc_p_lev     = nc_file.createVariable('p_lev'    , float_type, ('lev', 'col'))
nc_T_lay     = nc_file.createVariable('t_lay'    , float_type, ('lay', 'col'))
nc_T_lev     = nc_file.createVariable('t_lev'    , float_type, ('lev', 'col'))

nc_h2o     = nc_file.createVariable('vmr_h2o'    , float_type, ('lay', 'col'))
nc_o3      = nc_file.createVariable('vmr_o3'     , float_type, ('lay', 'col'))
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

while iprof < Nprofs:
    it = np.random.randint(0,nt,size=(1))
    ix =  np.random.randint(0,nx,size=(1))
    iy =  np.random.randint(0,ny,size=(1))
    if (it,ix,iy) not in Lprofs:
        Lprofs += [(it,ix,iy)]
        nc_p_lay[:,iprof] = p_lay[it,:].flatten()
        nc_p_lev[:,iprof] = p_lev[it,:].flatten()
        nc_T_lay[:,iprof] = T_lay[it,:,iy,ix].flatten()
        nc_T_lev[:,iprof] = T_lev[it,:,iy,ix].flatten()
        nc_o3[:,iprof]    = get_o3(p_lay[it,:]).flatten()
        nc_h2o[:,iprof]   = h2o[it,:,iy,ix].flatten()
        iprof += 1

nc_file.close()




