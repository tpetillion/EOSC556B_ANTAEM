import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SimPEG import (
    maps,
    data,
    inverse_problem,
    optimization,
    regularization,
    inversion,
    directives,
)
import discretize
from SimPEG.data_misfit import L2DataMisfit
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.utils import plot_1d_layer_model

def load_data(data_directory='../usapdc_601373/SkyTEM2018_dat.xyz'):
    """
    Load in ANTAEM data.
    Certain parameters are hardcoded to match the data delivered in the ANTAEM study, modification might need to happen within this function.
    """
    # load in data
    data_file = pd.read_csv(data_directory, skiprows=11, sep='\s+', escapechar='/')

    # trim white space from first column
    data_file.columns = data_file.columns.str.strip()
    # convert nan values
    data_file[data_file==99999] = np.NaN

    # /DEFINED NUMBER OF GATES metadata
    n_gates = 40

    # get column position of data/std
    col_offset = 8 #number of columns to ignore before data entries
    data_keys = data_file.keys()[col_offset:n_gates+col_offset]
    std_keys = data_file.keys()[n_gates+col_offset:2*n_gates+col_offset]
    
    return data_file, data_keys, std_keys

def hm_wave_info():
    """
    High moment wave form retrieved from SkyTEM Hawkes Bay report and converted to ANTAEM specifications.
    """

    # high moment time array in seconds
    hm_wave_time_Hawkes = np.array(   
        [  
        -5.0000E-03,
        -4.9786E-03,
        -4.8349E-03,
        -4.6999E-03,
        -4.2992E-03,
        -3.4979E-03,
        -2.4918E-03,
        -1.4901E-03,
        -4.8842E-04,
        2.6518E-06 ,
        5.0035E-05 ,
        1.0049E-04 ,
        1.5007E-04 ,
        2.0008E-04 ,
        2.5054E-04 ,
        2.9660E-04 ,
        3.0108E-04 ,
        3.0581E-04 ,
        1.5000E-002,
        ]
    )
    # normalized high moment amplitude array 
    hm_wave_form = np.array(    
        [  
            0.0000E+00,
            3.5156E-02,
            6.0234E-01,
            9.5703E-01,
            9.5859E-01,
            9.6641E-01,
            9.8047E-01,
            9.8828E-01,
            1.0000E+00,
            9.9670E-01,
            8.4862E-01,
            6.8337E-01,
            5.1661E-01,
            3.4524E-01,
            1.7057E-01,
            1.2613E-02,
            3.8707E-03,
            0.0000E+00,
            0.0000E+000
        ]
    )

    # Change wave to antaem report specifications
    tx_on_time_hm = 4.0E-3
    tx_on_norm = hm_wave_time_Hawkes[0:9]/hm_wave_time_Hawkes[0]

    tx_off_time_hm = 12.667E-3
    tx_off_norm = hm_wave_time_Hawkes[9:]/hm_wave_time_Hawkes[-1] 

    hm_wave_time = np.concatenate((-tx_on_norm*tx_on_time_hm,tx_off_norm*tx_off_time_hm))
    
    return hm_wave_form, hm_wave_time

def get_gate_times(slice):
    gate_times = np.array(
        [
            3.3230E-5,
            4.2220E-5,
            5.3720E-5,
            6.8220E-5,
            8.6220E-5,
            1.0870E-4,
            1.3670E-4,
            1.7220E-4,
            2.1770E-4,
            2.7470E-4,
            3.4670E-4,
            4.1840E-4,
            4.3640E-4,
            4.3770E-4,
            4.5890E-4,
            4.8690E-4,
            5.2240E-4,
            5.5170E-4,
            5.6790E-4,
            6.2490E-4,
            6.9570E-4,
            6.9690E-4,
            7.8790E-4,
            8.7720E-4,
            9.0190E-4,
            1.0460E-3,
            1.1060E-3,
            1.2270E-3,
            1.3940E-3,
            1.4560E-3,
            1.7440E-3,
            2.1080E-3, 
            2.5670E-3,
            3.1450E-3,
            3.8640E-3,
            4.7440E-3,
            5.8210E-3,
            7.1390E-3,
            8.7530E-3,
            1.0730E-2,
        ]
    )
    return gate_times[slice]


def survey_setup(station, hm_times):
    
    hm_wave_form, hm_wave_time = hm_wave_info()
    # geometry of device from antaem readme [x, y, z] (m)
    TxLoopPoints = np.array(
        [
            [-12.64, -2.10, 0],
            [-6.14,  -8.58, 0],
            [ 6.14,  -8.58, 0],
            [ 11.41, -3.31, 0],
            [ 11.41,  3.31, 0],
            [ 6.14,   8.58, 0],
            [-6.14,   8.58, 0],
            [-12.64,  2.10, 0]
        ]
    )

    RxZ = np.array([-13.25, 0.00, 2.0])  # [x, y, -z] (m) flip signs in z for simpeg/positive upwards
    
    # add another row because we will need to close the transmitter loop.
    tx_shape = np.vstack([TxLoopPoints,TxLoopPoints[0]])
    # convert to UTM  
    tx_loc = tx_shape + [station.UTMX.iloc[0], station.UTMY.iloc[0], station.ALT.iloc[0]]
    rx_loc = RxZ + [station.UTMX.iloc[0], station.UTMY.iloc[0], station.ALT.iloc[0]]

    
    # High moment survey input:
    rx_hm = tdem.receivers.PointMagneticFluxTimeDerivative(
        rx_loc, hm_times, orientation='z'
    )

    hm_wave = tdem.sources.PiecewiseLinearWaveform(hm_wave_time, hm_wave_form)

    src_hm = tdem.sources.LineCurrent(rx_hm, tx_loc, waveform=hm_wave)
    
    # survey with only high moment as source
    srv = tdem.Survey([src_hm])
    
    return srv



def station_info(data_file, data_keys, std_keys, line_no, station_num):
    """
    """
    tx_area = 342 # m^2

    # get line data and group high moment (2)
    line_grouping = data_file.groupby('LINE_NO')
    line = line_grouping.get_group(line_no)
    hm_line = line.groupby('SEGMENT').get_group(2)
    hm_data = hm_line[data_keys]
    
    # select a single sounding along the line, and grab all of the data associated with that station    
    station_hm = hm_line[hm_line.RECORD==station_num]
    station_hm_data = station_hm[data_keys].to_numpy().flatten()
    station_hm_std = station_hm[std_keys].to_numpy().flatten()
    
    # process out nan values
    hm_good_data = ~np.isnan(station_hm_data)
    station_hm_data = station_hm_data[hm_good_data]
    station_hm_std = station_hm_std[hm_good_data]
    hm_times = get_gate_times(hm_good_data)
    n_hm_gates = len(hm_times)
    
    # survey set-up
    srv = survey_setup(station_hm, hm_times)
    dobs = -tx_area * np.r_[station_hm_data]
    rel_err = np.r_[station_hm_std]
    data_container = data.Data(srv, dobs=dobs, relative_error=rel_err)
    
    return srv, data_container



def halfspace_inversion(data_file, data_keys, std_keys, line_no, station_num):
    """
    Returns best fitting halfspace for station data. ie, background model
    """
    srv, data_container = station_info(data_file, data_keys, std_keys, line_no, station_num)

    # set-up inversion for best fitting half-space
    exp_map = maps.ExpMap()
    sim_inv = tdem.Simulation1DLayered(srv, sigmaMap=exp_map)
    sim_inv.model = np.log(np.r_[1E-3])
    
    # define objective function to be minimized
    phi_d = L2DataMisfit(data=data_container, simulation=sim_inv)
    m_0 = np.log(1E-2)  # sigma = 1E-2 S/m
    minimizer = opt = optimization.InexactGaussNewton(
        maxIter=10, maxIterLS=20, maxIterCG=10, tolCG=1e-3
    )
    empty_reg = regularization.Smallness(discretize.TensorMesh([1]))
    inv_prob = inverse_problem.BaseInvProblem(
        phi_d, reg=empty_reg, opt=minimizer, beta=0.0
    )
    inv = inversion.BaseInversion(inv_prob, [])
    
    # Run inversion
    recovered_model = inv.run(m_0)
    
    # Best fitting half-space
    background_model = recovered_model[0] 
    return background_model


def get_mesh(inv_directory = '../usapdc_601373/SkyTEM2018_inv.xyz'):
    
    # load in inversion file
    inv_file = pd.read_csv(inv_directory, skiprows=7, sep='\s+', escapechar='/')

    # trim white space from first column
    inv_file.columns = inv_file.columns.str.strip()
    
    # Extract inverted layer thickness
    inv_thicknesses = inv_file[inv_file.keys()[35:-1]].to_numpy()[0]
    mesh = discretize.TensorMesh([(np.r_[inv_thicknesses])], "0")
    
    return mesh
    
    
def multilayer_inversion(background_model,srv,data_container,n_layers=30):
    mesh = get_mesh()
    
    # Define number of layers 
    wire_map = maps.Wires(('sigma',n_layers), ('thicknesses', n_layers-1))
    exp_map = maps.ExpMap()
    sigma_map = exp_map * wire_map.sigma
    thick_map = exp_map * wire_map.thicknesses
    
    # define smoothest model regularization
    alpha_s = 1e-10
    alpha_z = 1

    reg_s = regularization.Smallness(mesh, reference_model=background_model)
    reg_z = regularization.SmoothnessFirstOrder(mesh, orientation="x") # x is first dimension in Simpeg

    reg = alpha_s * reg_s + alpha_z * reg_z 
    
    # inversion set-up
    sim_reg = tdem.Simulation1DLayered(srv, sigmaMap=exp_map, thicknesses=mesh.h[0][:-1])
    phi_d_reg = L2DataMisfit(data=data_container, simulation=sim_reg)
    m_0_reg = np.full(mesh.n_cells, background_model)
    beta0_ratio = 1e2
    beta_estimate = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
    beta_cooler = directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    save_dict = directives.SaveOutputDictEveryIteration()
    
    minimizer_reg = opt = optimization.InexactGaussNewton(
        maxIter=20, maxIterLS=20, maxIterCG=100, tolCG=1e-5
    )

    inv_prob = inverse_problem.BaseInvProblem(
        phi_d_reg, reg=reg, opt=minimizer_reg
    )
    
    directives_list = [
        beta_estimate,
        beta_cooler,
        save_dict
    ]
    
    inv = inversion.BaseInversion(inv_prob, directives_list)

    # Run inversion
    recovered_model = inv.run(m_0_reg)
    
    # get best fit model 
    phi_ds = [save_dict.outDict[i]['phi_d'] for i in save_dict.outDict.keys()]
    phi_ms = [save_dict.outDict[i]['phi_m'] for i in save_dict.outDict.keys()]
    betas =  [save_dict.outDict[i]['beta'] for i in save_dict.outDict.keys()]
    
    L_iter = (np.abs(np.asarray(phi_ds) < len(data_container.dobs))).argmax() + 1 # notebook says nD/2? (+1 since indexing of model out starts at 1)
        
    m_final = save_dict.outDict[L_iter]['m']
    d_final = sim_reg.make_synthetic_data(m_final)
    
    # percent error
    rel_diff = np.abs((data_container.dobs - d_final.dclean)/data_container.dobs)
    percent_err = rel_diff.mean()*100
    
    # estimated depth of investigation
    DOI = -sim_reg.depth[(exp_map * m_final).argmax()]
    
    return m_final, d_final, percent_err, DOI