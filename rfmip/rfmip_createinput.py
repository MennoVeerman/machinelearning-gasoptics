import numpy as np
import netCDF4 as nc
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--iexpt', default=0, type=int)
parser.add_argument('--nfile', default=1, type=int)
parser.add_argument('--frand', default=False, action='store_true')
parser.add_argument('--finvt', default=False, action='store_true')
args = parser.parse_args()

# Settings
iexpt = args.iexpt
nfile = args.nfile
frand = args.frand
finvt = args.finvt
float_type = 'f8'
band_lw = 16
band_sw = 14

def new_preslev(plev, play):
    pmin = 1.005183574463
    nlev, ncol = plev.shape
    pnew = np.zeros(plev.shape)
    pnew[0,:] = pmin
    pnew[-1,:] = np.random.random(ncol)*2*(plev[-1,:] - play[-1,:])  + play[-1,:]
    pnew[1:-1,:] = np.random.random((nlev-2, ncol))*(plev[2:,:] - plev[:-2,:])  + plev[:-2,:]
    return np.sort(pnew,axis=0)

def get_qsat(T,P):
    Tc = T - 273.15
    es = 611. * np.exp(17.27*Tc/(Tc+237.3))
    r = es/(P-es)
    return r,es

for ifile in range(nfile):
    # Save all the input data to NetCDF
    nc_file = nc.Dataset('data/rte_rrtmgp_input_{:03d}.nc'.format(ifile), mode='w', datamodel='NETCDF4', clobber=True)
    nc_file_rfmip = nc.Dataset('multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', mode='r', datamodel='NETCDF4', clobber=False)
    
    # Create a group for the radiation and set up the values.
    nc_file.createDimension('lay', nc_file_rfmip.dimensions['layer'].size)
    nc_file.createDimension('lev', nc_file_rfmip.dimensions['level'].size)
    nc_file.createDimension('col', nc_file_rfmip.dimensions['site'].size)
    nc_file.createDimension('band_lw', band_lw)
    nc_file.createDimension('band_sw', band_sw)
    
    nc_pres_layer = nc_file.createVariable('p_lay', float_type, ('lay', 'col'))
    nc_pres_level = nc_file.createVariable('p_lev', float_type, ('lev', 'col'))
    nc_temp_layer = nc_file.createVariable('t_lay', float_type, ('lay', 'col'))
    nc_temp_level = nc_file.createVariable('t_lev', float_type, ('lev', 'col'))
    
    nc_pres_layer[:,:] = nc_file_rfmip.variables['pres_layer'][:,:].transpose()
    nc_pres_level[:,:] = nc_file_rfmip.variables['pres_level'][:,:].transpose()

    # Make sure the top edge does not exceed the minimum tolerable pressure
    # of the coefficient files.
    nc_pres_level[:,:] = np.maximum(nc_pres_level[:,:], np.nextafter(1.005183574463, 1e8))

    nc_temp_layer[:,:] = (nc_file_rfmip.variables['temp_layer'][iexpt,:,:]).transpose()
    nc_temp_level[:,:] = (nc_file_rfmip.variables['temp_level'][iexpt,:,:]).transpose()
    
    nc_surface_emissivity = nc_file.createVariable('emis_sfc', float_type, ('col', 'band_lw'))
    nc_surface_temperature = nc_file.createVariable('t_sfc', float_type, 'col')
    
    nc_surface_emissivity[:,:] = np.tile( (nc_file_rfmip.variables['surface_emissivity'][:]) [:,None], (1, band_lw) )
    nc_surface_temperature[:] = nc_file_rfmip.variables['surface_temperature'][iexpt,:]

    nc_surface_albedo_dir = nc_file.createVariable('sfc_alb_dir', float_type, ('col', 'band_sw'))
    nc_surface_albedo_dif = nc_file.createVariable('sfc_alb_dif', float_type, ('col', 'band_sw'))
    
    nc_surface_albedo_dir[:,:] = np.tile( (nc_file_rfmip.variables['surface_albedo'][:]) [:,None], (1, band_sw) )
    nc_surface_albedo_dif[:,:] = np.tile( (nc_file_rfmip.variables['surface_albedo'][:]) [:,None], (1, band_sw) )
    
    nc_mu0 = nc_file.createVariable('mu0', float_type, ('col'))
    nc_mu0[:] = np.maximum(0., np.cos( np.deg2rad(nc_file_rfmip.variables['solar_zenith_angle'][:]) ) )
    
    nc_tsi = nc_file.createVariable('tsi', float_type, ('col'))
    nc_tsi[:] = nc_file_rfmip.variables['total_solar_irradiance'][:]
   
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
    # nc_no2     = nc_file.createVariable('vmr_no2'    , float_type)

    # Profiles

    nc_h2o[:,:] = nc_file_rfmip.variables['water_vapor'][iexpt,:,:].transpose() * float(nc_file_rfmip.variables['water_vapor'].units)
    nc_o3 [:,:] = nc_file_rfmip.variables['ozone'][iexpt,:,:].transpose() * float(nc_file_rfmip.variables['ozone'].units)
    
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
    # nc_no2    [:] = nc_file_rfmip.variables['no2_GM'][iexpt] * float(nc_file_rfmip.variables['no2a_GM'].units)

    # CvH: To be removed if settings can be set.
    nc_lwp = nc_file.createVariable('lwp', float_type, ('lay', 'col'))
    nc_iwp = nc_file.createVariable('iwp', float_type, ('lay', 'col'))
    nc_rel = nc_file.createVariable('rei', float_type, ('lay', 'col'))
    nc_rei = nc_file.createVariable('rel', float_type, ('lay', 'col'))

    nc_lwp[:,:] = 0.
    nc_iwp[:,:] = 0.
    nc_rel[:,:] = 0.
    nc_rei[:,:] = 0.

    if frand:
        nc_pres_level[:] = new_preslev(nc_pres_level[:], nc_pres_layer[:])
        nc_pres_layer[:] = (nc_pres_level[1:] + nc_pres_level[:-1])/2.
        nc_temp_level[:] = nc_temp_level[:] + 5 * (np.random.random(nc_temp_level.shape)*2-1)
        nc_temp_layer[:] = (nc_temp_level[:-1,:] + nc_temp_level[1:,:]) / 2.
        nc_surface_temperature[:] = (nc_temp_level[-1,:]-10) + np.random.random(nc_surface_temperature[:].shape) * 20. 
        nc_o3[:] += nc_o3[:] *  0.75 * (np.random.random(nc_o3.shape )*2-1) 
        h2o_ref = nc_h2o[:]
        nc_h2o[:] += nc_h2o[:] * 0.75 * (np.random.random(nc_h2o.shape)*2-1) 
        qsat, es = get_qsat(nc_temp_layer[:], nc_pres_layer[:])
        for ilay in range(nc_h2o.shape[0]):
            for icol in range(nc_h2o.shape[1]):
                while nc_h2o[ilay,icol] > 1.01*qsat[ilay,icol] and es[ilay,icol]<nc_pres_layer[ilay,icol] and es[ilay,icol]>0. and h2o_ref[ilay,icol]<qsat[ilay,icol]:
                    nc_h2o[ilay,icol] = h2o_ref[ilay,icol] + h2o_ref[ilay,icol] * 0.75 * (np.random.random(1)*2-1) 

    if finvt:
        nc_pres_level[:] = nc_pres_level[::-1]
        nc_pres_layer[:] = nc_pres_layer[::-1]
        nc_temp_level[:] = nc_temp_level[::-1]
        nc_temp_layer[:] = nc_temp_layer[::-1]
        nc_h2o[:] = nc_h2o[::-1]
        nc_o3[:]  = nc_o3[::-1]

    nc_file_rfmip.close()
    nc_file.close()