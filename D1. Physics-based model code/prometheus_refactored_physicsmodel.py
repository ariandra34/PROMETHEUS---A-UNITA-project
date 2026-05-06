

!pip install pvlib

import pvlib
import pandas as pd
import glob
import numpy as np
from functools import reduce
import os
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""#Prepare data"""

meteo_data=pd.read_csv("meteo_fully_preprocessed.csv")

dew_pres=pd.read_csv("dewpoint_pressure_2024.csv", sep=';', encoding='latin1')

dew_pres['time'] = pd.to_datetime(dew_pres['ï»¿date'], format='%d.%m.%Y %H:%M')
dew_pres['pressure']=dew_pres['Atmospheric Pressure [hPa]']*100
dew_pres['dew point']=dew_pres['Dew point 2m above ground [Â°C]']

pres_dew_subset = dew_pres[['time','pressure', 'dew point']]

pres_dew_subset = pres_dew_subset.copy()
pres_dew_subset['time'] = pd.to_datetime(pres_dew_subset['time'], utc=True).dt.tz_convert("Europe/Zurich")

"""#Decomposition Model - Reindl"""

#find the day of the year
steps = meteo_data.copy()
steps['time'] = pd.to_datetime(steps['time'])
steps['day_of_year'] = steps['time'].dt.dayofyear

#compute the solar declination in degrees
steps['solar_declination'] = 23.45 * np.sin(np.deg2rad((360/365) * (284 + steps['day_of_year'])))

"""NOTE: The solar declination is maximum around June 21st and becomes negative around September 20th."""

steps['time'] = pd.to_datetime(steps['time'])
steps['time'] = steps['time'].dt.tz_localize("UTC")
steps['time'] = steps['time'].dt.tz_convert("Europe/Zurich")

#compute the Solar Local Time
longitude_deg = 7.364206

def compute_solar_local_time(row):
    day_of_year = row['day_of_year']
    B = 2.0 * np.pi * (day_of_year - 81) / 365.0
    EoT_min = 229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B) - 0.014615 * np.cos(2*B) - 0.040849 * np.sin(2*B))
    utc_offset_hours = row['time'].utcoffset().total_seconds()/3600
    LSTM_deg = 15.0 * utc_offset_hours
    longitude_correction_min = 4.0 * (longitude_deg - LSTM_deg)
    total_correction = EoT_min + longitude_correction_min
    return row['time'] + pd.to_timedelta(total_correction, unit='m')

steps['solar_local_time'] = steps.apply(compute_solar_local_time, axis=1)

"""NOTE: LSTM stands for Local Standard Time Meridian."""

#compute the hour angle in degrees
steps['solar_decimal_hour'] = (steps['solar_local_time'].dt.hour + steps['solar_local_time'].dt.minute / 60.0 + steps['solar_local_time'].dt.second / 3600.0)
steps['hour_angle'] = 15 * (steps['solar_decimal_hour'] - 12)

#compute the solar zenith angle in degrees
latitude_deg = 46.227379
phi = np.radians(latitude_deg)

delta_rad = np.radians(steps['solar_declination'])
H_rad = np.radians(steps['hour_angle'])

cos_theta_z = np.sin(phi) * np.sin(delta_rad) + np.cos(phi) * np.cos(delta_rad) * np.cos(H_rad)
steps['solar_zenith'] = np.degrees(np.arccos(cos_theta_z))
zenith=np.degrees(np.arccos(cos_theta_z))

"""NOTE: The solar zenith angle needs to be about 90 degrees at sunrise and sundown. For solar noon, it is the lowest (22 degrees). Nighttime values are between 90 and 180."""

#calculate the Sun-Earth distance in astronomical units
n = steps['day_of_year']
B = 2 * np.pi * (n - 1) / 365

steps['sun_earth_distance'] = 1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2*B) + 0.000077 * np.sin(2*B)

#calculate the extraterrestrial irradiance
G_sc = 1361
theta_z_rad = np.radians(steps['solar_zenith'])
n = steps['day_of_year']
B = 2 * np.pi * (n - 1) / 365
d_au = 1 + 0.033 * np.cos(B)

G0 = G_sc * (1 / d_au)**2 * np.cos(theta_z_rad)

G0[G0 < 0] = 0

steps['extraterrestrial_irradiance'] = G0

#calculate Sun elevation in degrees
steps['solar_elevation'] = 90 - steps['solar_zenith']
steps['solar_elevation'] = steps['solar_elevation'].clip(lower=0)

steps['relative_humidity']= meteo_data['humidity']
steps['dew_point']= dew_pres['dew point']

#Reindl decomposition model
GHI = steps['irradiance'].values
G0 = steps['extraterrestrial_irradiance'].values
elev = steps['solar_elevation'].values
Tdew = steps['dew_point'].values
RH = steps['relative_humidity'].values

theta_z_deg = 90 - elev
theta_z_rad = np.radians(theta_z_deg)
cos_z = np.cos(theta_z_rad)
sin_alpha = cos_z

Kt = np.zeros_like(GHI)
valid = G0 > 0
Kt[valid] = np.clip(GHI[valid] / G0[valid], 0, 1)

Kt_series = pd.Series(Kt, index=steps.index)
delta_Kt = Kt_series.diff().values

coef1 = {'a': 1.000, 'b': -0.232, 'c': 0.048, 'd': 0.051, 'e': -0.004, 'f': 0.0002}
coef2 = {'a': 1.456, 'b': -1.793, 'c': 0.177, 'd': 0.033, 'e': -0.005, 'f': 0.0003}
coef3 = {'a': 0.426, 'b': -0.256, 'c': -0.007, 'd': -0.003, 'e': 0.0001, 'f': -0.0001}

kd = np.full_like(Kt, np.nan, dtype=float)

mask1 = (Kt <= 0.3)
mask2 = (Kt > 0.3) & (Kt < 0.83)
mask3 = (Kt >= 0.83)

valid = ~np.isnan(delta_Kt)

idx = mask1 & valid
kd[idx] = (coef1['a'] + coef1['b'] * Kt[idx] + coef1['c'] * sin_alpha[idx] + coef1['d'] * delta_Kt[idx] + coef1['e'] * Tdew[idx] + coef1['f'] * RH[idx])

idx = mask2 & valid
kd[idx] = (coef2['a'] + coef2['b'] * Kt[idx] + coef2['c'] * sin_alpha[idx] + coef2['d'] * delta_Kt[idx] + coef2['e'] * Tdew[idx] + coef2['f'] * RH[idx])

idx = mask3 & valid
kd[idx] = (coef3['a'] + coef3['b'] * Kt[idx] + coef3['c'] * sin_alpha[idx] + coef3['d'] * delta_Kt[idx] + coef3['e'] * Tdew[idx] + coef3['f'] * RH[idx])

missing = np.isnan(kd)
if missing.any():
    kd1 = np.zeros_like(Kt)
    kd1[mask1] = 1.020 - 0.254 * Kt[mask1] + 0.0123 * sin_alpha[mask1]
    kd1[mask2] = 1.400 - 1.749 * Kt[mask2] + 0.177 * sin_alpha[mask2]
    kd1[mask3] = 0.486 * Kt[mask3] - 0.182 * sin_alpha[mask3]
    kd[missing] = kd1[missing]

kd = np.clip(kd, 0.0, 1.0)

DHI = kd * GHI

solar_elevation = 90 - theta_z_deg

min_elev = 5.0
max_dni = 1640
eps = 1e-6

valid = (solar_elevation > min_elev) & (cos_z > 0)

DNI = np.zeros_like(GHI)
DNI[valid] = (GHI[valid] - DHI[valid]) / np.maximum(cos_z[valid], eps)

DNI = np.minimum(DNI, max_dni)

steps['DHI'] = DHI
steps['DNI'] = DNI

meteo_data['DNI']= steps['DNI']
meteo_data['DHI']= steps['DHI']

"""#Hay-Davies Transposition model"""

#find solar azimuth
phi = np.radians(46.227379)
delta = np.radians(steps['solar_declination'])
H = np.radians(steps['hour_angle'])
theta_z = np.radians(steps['solar_zenith'])

azimuth_rad = np.arctan2(np.sin(H), np.cos(H)*np.sin(phi) - np.tan(delta)*np.cos(phi))

azimuth_deg = np.degrees(azimuth_rad)
azimuth_deg = (azimuth_deg + 360) % 360

steps['solar_azimuth'] = azimuth_deg

module_tilt = 24
module_orientation_east = 77
module_orientation_west = 257

#compute the incidence angle
theta_z_rad = np.radians(steps['solar_zenith'])
gamma_s_rad = np.radians(steps['solar_azimuth'])

beta_east = np.radians(module_tilt)
gamma_m_east = np.radians(module_orientation_east)
cos_theta_i_east = np.cos(theta_z_rad)*np.cos(beta_east) + np.sin(theta_z_rad)*np.sin(beta_east)*np.cos(gamma_s_rad - gamma_m_east)
steps['incidence_angle_east'] = np.degrees(np.arccos(np.clip(cos_theta_i_east, -1, 1)))

beta_west = np.radians(module_tilt)
gamma_m_west = np.radians(module_orientation_west)
cos_theta_i_west = np.cos(theta_z_rad)*np.cos(beta_west) + np.sin(theta_z_rad)*np.sin(beta_west)*np.cos(gamma_s_rad - gamma_m_west)
steps['incidence_angle_west'] = np.degrees(np.arccos(np.clip(cos_theta_i_west, -1, 1)))

#compute the beam component
steps['G_beam_east'] = np.maximum(steps['DNI'] * np.cos(np.radians(steps['incidence_angle_east'])), 0.0)
steps['G_beam_west'] = np.maximum(steps['DNI'] * np.cos(np.radians(steps['incidence_angle_west'])), 0.0)

#calculate the diffuse component
theta_z_rad = np.radians(steps['solar_zenith'])
theta_i_east_rad = np.radians(steps['incidence_angle_east'])
theta_i_west_rad = np.radians(steps['incidence_angle_west'])
beta_rad = np.radians(module_tilt)

G_b = steps['DNI']
G_d= steps['DHI']
G0 = steps['extraterrestrial_irradiance']

F = np.clip(G_b / G0, 0, 1)

G_d_east = G_d * (F * np.cos(theta_i_east_rad) / np.cos(theta_z_rad) + (1 - F) * (1 + np.cos(beta_rad)) / 2)
G_d_west = G_d * (F * np.cos(theta_i_west_rad) / np.cos(theta_z_rad) + (1 - F) * (1 + np.cos(beta_rad)) / 2)

steps['G_diffuse_east'] = np.maximum(G_d_east, 0.0)
steps['G_diffuse_west'] = np.maximum(G_d_west, 0.0)

steps['G_diffuse_east'] = steps['G_diffuse_east'].fillna(0)
steps['G_diffuse_west'] = steps['G_diffuse_west'].fillna(0)

#compute the albedo
G_h = steps['irradiance']
rho = 0.1
beta_rad = np.radians(module_tilt)

G_r = G_h * rho * (1 - np.cos(beta_rad)) / 2

steps['G_albedo_east'] = G_r
steps['G_albedo_west'] = G_r

#compute POA for both orientations
steps['G_POA_east'] = steps['G_beam_east'] + steps['G_diffuse_east'] + steps['G_albedo_east']
steps['G_POA_west'] = steps['G_beam_west'] + steps['G_diffuse_west'] + steps['G_albedo_west']

steps['G_POA_east'] = steps['G_POA_east'].fillna(0.0)
steps['G_POA_west'] = steps['G_POA_west'].fillna(0.0)

"""#Consider system characteristics"""

#compute module temperature
T_NOCT = 45
T_ambient = steps['temperature']
G_POA_east = steps['G_POA_east']
G_POA_west = steps['G_POA_west']

steps['T_module_east'] = T_ambient + ((T_NOCT - 20)/800) * G_POA_east
steps['T_module_west'] = T_ambient + ((T_NOCT - 20)/800) * G_POA_west

P_nominal_total = 22200
P_module = 370

N_modules = P_nominal_total / P_module
N_modules = int(N_modules)
print("Number of modules:", N_modules)

gamma = -0.003
G_POA_east = steps['G_POA_east']
G_POA_west = steps['G_POA_west']
T_module_east = steps['T_module_east']
T_module_west = steps['T_module_west']

P_DC_east_module = P_module * (G_POA_east / 1000) * (1 + gamma * (T_module_east - 25))
P_DC_west_module = P_module * (G_POA_west / 1000) * (1 + gamma * (T_module_west - 25))

P_DC_east_module = np.maximum(P_DC_east_module, 0)
P_DC_west_module = np.maximum(P_DC_west_module, 0)

N_east = 39
N_west = N_modules - N_east

steps['Ideal_P_DC_total'] = P_DC_east_module * N_east + P_DC_west_module * N_west

site_derate = 0.92
steps['estimated_P_DC'] = steps['Ideal_P_DC_total'] * site_derate

meteo_data["estimated_P_DC"] = steps["estimated_P_DC"]

meteo_data['estimated_P_DC'] = meteo_data['estimated_P_DC'].fillna(0)

steps

steps.to_csv("physical_features.csv", index=True)

"""#Compare with the actual PV output"""

actual_P_DC=pd.read_csv("production_fully_preprocessed.csv")

actual_P_DC['time'] = pd.to_datetime(actual_P_DC['time'])
actual_P_DC['time'] = actual_P_DC['time'].dt.tz_localize("UTC").dt.tz_convert("Europe/Zurich")

merged = pd.DataFrame({
    'actual': actual_P_DC['pv_production'],
    'estimated': steps['estimated_P_DC']
}).dropna()

residuals = merged['estimated'] - merged['actual']
mbe = residuals.mean()
mae = np.abs(residuals).mean()
rmse = np.sqrt((residuals ** 2).mean())
nrmse_cap = (rmse / P_nominal_total) * 100

print(f"RMSE: {rmse:.2f} W")
print(f"MBE:  {mbe:.2f} W")
print(f"MAE:  {mae:.2f} W")
print(f"nRMSE (by capacity): {nrmse_cap:.2f} %")

r2 = r2_score(merged['actual'], merged['estimated'])
print(f"R squared: {r2:.4f}")

merged['time'] = steps['time'].values
merged['time'] = pd.to_datetime(merged['time'])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=merged['time'],
    y=merged['actual'],
    mode='lines',
    name='Actual PV',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=merged['time'],
    y=merged['estimated'],
    mode='lines',
    name='Estimated PV',
    line=dict(color='orange')
))

fig.update_layout(
    title=f'Actual vs Estimated PV Output (RMSE = {rmse:.1f} W)',
    xaxis_title='Time',
    yaxis_title='Power [W]',
    legend=dict(x=0, y=1),
    template='plotly_white',
    hovermode='x unified',
    xaxis=dict(
        showspikes=True,
        spikemode='across',
        spikecolor='grey',
        spikethickness=1,
        tickformat='%Y-%m-%d\n%H:%M'
    )
)

fig.show()