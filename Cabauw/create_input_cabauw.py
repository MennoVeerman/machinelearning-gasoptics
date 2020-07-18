#import matplotlib.pyplot as pl
import numpy as np
from shutil import copyfile
import netCDF4 as nc
#location of Cabauw output 
path = "path/simulation/output/"

float_type = "f8"
cp  = 1004.
rd    = 287.04
pres0 = 1.e5

def interp_to_lev(tlay):
    tlev = np.zeros(tlay.shape[0]+1)
    tlev[1:-1] = (tlay[1:]+tlay[:-1]) / 2.
    tlev[0] = 2 * tlay[0] - tlev[1]
    tlev[-1] = 2 * tlay[-1] - tlev[-2]
    return tlev

def get_o3(p):
    p_hpa = p / 100.
    return np.maximum(5e-9,3.6478 * p_hpa**0.83209 * np.exp(-p_hpa/11.3515) * 1e-6)

nc_idx  = [(np.random.randint(0,12),np.random.randint(0,12)) for i in range(10)]
atmfiles = [nc.Dataset(path+"threedheating.%03d.%03d.700.nc"%(idx[0],idx[1])) for idx in nc_idx]
sfcfiles = [nc.Dataset(path+"crossAGS.x%03dy%03d.700.nc"%(idx[0],idx[1])) for idx in nc_idx]
nt,nz,ny,nx = atmfiles[0].variables['ql'][:].shape
inp_psfile = np.loadtxt(path+"pressures.dat")

pres_lay = np.reshape(inp_psfile[:,1] * 100., (600,228))
pres_lev = np.reshape(inp_psfile[:,2] * 100. ,(600,228))
pres_lev = np.append(pres_lev,np.reshape(pres_lay[:,-1]*2 - pres_lev[:,-1],(600,1)),axis=1)

##########################
nc_file = nc.Dataset("rte_rrtmgp_input.nc", mode="w", datamodel="NETCDF4", clobber=True)
nc_file_rfmip = nc.Dataset('multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', mode='r', datamodel='NETCDF4', clobber=False)

iexpt=0
# load a couple of ncfiles (one file = 1 core - 16*16 columns)
Nprofs = 100
Lprofs = []
iprof  = 0

nc_file.createDimension("lay", pres_lay.shape[1])
nc_file.createDimension("lev", pres_lev.shape[1])
nc_file.createDimension("col", Nprofs)
nc_file.createDimension("band_lw", 16)
nc_file.createDimension("band_sw", 14)

nc_surface_emissivity = nc_file.createVariable('emis_sfc', float_type, ('col', 'band_lw'))
nc_surface_emissivity[:,:] = nc_file_rfmip.variables['surface_emissivity'][0]

nc_surface_albedo_dir = nc_file.createVariable('sfc_alb_dir', float_type, ('col', 'band_sw'))
nc_surface_albedo_dif = nc_file.createVariable('sfc_alb_dif', float_type, ('col', 'band_sw'))

nc_surface_albedo_dir[:,:] = .07
nc_surface_albedo_dif[:,:] = .07

nc_mu0 = nc_file.createVariable('mu0', float_type, ('col'))
nc_mu0[:] = np.cos(np.deg2rad(42.))

nc_p_lay = nc_file.createVariable("p_lay", float_type, ("lay","col"))
nc_p_lev = nc_file.createVariable("p_lev", float_type, ("lev","col"))

nc_T_lay = nc_file.createVariable("t_lay", float_type, ("lay","col"))
nc_T_lev = nc_file.createVariable("t_lev", float_type, ("lev","col"))
nc_T_sfc = nc_file.createVariable("t_sfc", float_type, ("col"))

nc_h2o     = nc_file.createVariable('vmr_h2o'    , float_type, ("lay","col"))
nc_o3      = nc_file.createVariable('vmr_o3'     , float_type, ("lay","col"))
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

#random location
while iprof < Nprofs:
    iatm = np.random.randint(0,len(atmfiles),size=(1))[0]
    it   = np.random.randint(0,nt,size=(1))[0]
    iy   = np.random.randint(0,ny,size=(1))[0]
    ix   = np.random.randint(0,nx,size=(1))[0]
    if (iatm,it,ix,iy) not in Lprofs:
        Lprofs += [(iatm,it,ix,iy)]
        qt = atmfiles[iatm].variables['qt'][it,:,iy,ix] * 1e-5
        ql = atmfiles[iatm].variables['ql'][it,:,iy,ix] * 1e-5
        qv = np.maximum(0,qt-ql)
        h2o = np.maximum(1.6e-5,qv/(.622-.622*qv))
        th = atmfiles[iatm].variables['thl'][it,:,iy,ix]
        o3 = get_o3(pres_lay[it,:])
        exnf= (pres_lay[it,:]/pres0)**(rd/cp) #6*i
        tlay = exnf * (th + (2.53e6/1004.) * ql / exnf)
        tlev = interp_to_lev(tlay)
        tsfc = sfcfiles[iatm].variables['tskin'][it,iy,ix]

        nc_p_lay[:,iprof] = np.transpose(pres_lay[it])
        nc_p_lev[:,iprof] = np.transpose(pres_lev[it])
        nc_T_lay[:,iprof] = np.transpose(tlay[:])
        nc_T_lev[:,iprof] = np.transpose(tlev[:])
        nc_T_sfc[iprof] = np.transpose(tsfc)
        nc_o3   [:,iprof] = np.transpose(o3[:])
        nc_h2o  [:,iprof] = np.transpose(h2o[:])
        iprof += 1

nc_file.close()

