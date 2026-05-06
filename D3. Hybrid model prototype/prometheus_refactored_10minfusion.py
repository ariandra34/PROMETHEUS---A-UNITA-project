

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

physics=pd.read_csv('complete_predictions_10min.csv')

tcn=pd.read_csv('tcn_1step_pred-unscaled.csv')

tcn['time'] = pd.to_datetime(tcn['time']).dt.tz_localize(None)
physics['timestamp'] = pd.to_datetime(physics['timestamp']).dt.tz_localize(None)

tcn['time'] = pd.to_datetime(tcn['time'], utc=True)
physics['timestamp'] = pd.to_datetime(physics['timestamp'], utc=True)

cols_tcn = ['time', 'prediction']
cols_phys = ['timestamp', 'prediction_w', 'actual_w', 'solar_elevation_deg', 'kd_value']

merged = (
    tcn[cols_tcn]
    .merge(
        physics[cols_phys],
        left_on='time',
        right_on='timestamp',
        how='inner')
    .drop(columns=['time'])
    .rename(columns={'timestamp': 'time'}))

n = len(merged)
validation = merged.iloc[:n//2]
test = merged.iloc[n//2:]

clear_mask = (validation['kd_value'] >= 0) & (validation['kd_value'] < 0.3)
mixed_mask = (validation['kd_value'] >= 0.3) & (validation['kd_value'] < 0.7)
cloudy_mask = (validation['kd_value'] >= 0.7) & (validation['kd_value'] < 1)
night_mask = (validation['kd_value'] == 1) & (validation['solar_elevation_deg'] <= 0)

night_mask = (validation['kd_value'] == 1) & (validation['solar_elevation_deg'] <= 0)
clear_mask = (validation['kd_value'] >= 0) & (validation['kd_value'] < 0.3) & (~night_mask)
mixed_mask = (validation['kd_value'] >= 0.3) & (validation['kd_value'] < 0.7) & (~night_mask)
cloudy_mask = (validation['kd_value'] >= 0.7) & (validation['kd_value'] < 1) & (~night_mask)

clear = validation[clear_mask]
mixed = validation[mixed_mask]
cloudy = validation[cloudy_mask]
night = validation[night_mask]

night['fusionpv'] = night['prediction_w']

model_clear = LinearRegression()

X_clear = validation.loc[clear_mask, ['prediction', 'prediction_w']]
y_clear = validation.loc[clear_mask, 'actual_w']

model_clear.fit(X_clear, y_clear)

a = model_clear.coef_[0]
b = model_clear.coef_[1]
c = model_clear.intercept_

print(f"fusionpv = {a:.3f} * prediction + {b:.3f} * prediction_w + {c:.3f}")

model_mixed = LinearRegression()

X_mixed = validation.loc[mixed_mask, ['prediction', 'prediction_w']]
y_mixed = validation.loc[mixed_mask, 'actual_w']

model_mixed.fit(X_mixed, y_mixed)

a = model_mixed.coef_[0]
b = model_mixed.coef_[1]
c = model_mixed.intercept_

print(f"fusionpv = {a:.3f} * prediction + {b:.3f} * prediction_w + {c:.3f}")

model_cloudy = LinearRegression()

X_cloudy = validation.loc[cloudy_mask, ['prediction', 'prediction_w']]
y_cloudy = validation.loc[cloudy_mask, 'actual_w']

model_cloudy.fit(X_cloudy, y_cloudy)

a = model_cloudy.coef_[0]
b = model_cloudy.coef_[1]
c = model_cloudy.intercept_

print(f"fusionpv = {a:.3f} * prediction + {b:.3f} * prediction_w + {c:.3f}")

"""#Prediction"""

night_mask  = (test['kd_value'] == 1) & (test['solar_elevation_deg'] <= 0)

clear_mask  = (test['kd_value'] < 0.3) & (~night_mask)
mixed_mask  = (test['kd_value'] >= 0.3) & (test['kd_value'] < 0.7) & (~night_mask)
cloudy_mask = (test['kd_value'] >= 0.7) & (test['kd_value'] < 1) & (~night_mask)

test['fusionpv'] = np.nan

test.loc[night_mask, 'fusionpv'] = test.loc[night_mask, 'prediction_w']

X_clear = test.loc[clear_mask, ['prediction', 'prediction_w']]
test.loc[clear_mask, 'fusionpv'] = model_clear.predict(X_clear)

X_mixed = test.loc[mixed_mask, ['prediction', 'prediction_w']]
test.loc[mixed_mask, 'fusionpv'] = model_mixed.predict(X_mixed)

X_cloudy = test.loc[cloudy_mask, ['prediction', 'prediction_w']]
test.loc[cloudy_mask, 'fusionpv'] = model_cloudy.predict(X_cloudy)

test['fusionpv'] = test['fusionpv'].fillna(test['prediction'])

rmse = np.sqrt(mean_squared_error(test['actual_w'], test['fusionpv']))
mae = mean_absolute_error(test['actual_w'], test['fusionpv'])
r2_fusion = r2_score(test['actual_w'], test['fusionpv'])

print("Fusion RMSE:", rmse)
print("Fusion MAE:", mae)
print("Fusion R squared:", r2_fusion)

rmse_tcn = np.sqrt(mean_squared_error(test['actual_w'], test['prediction']))
rmse_phys = np.sqrt(mean_squared_error(test['actual_w'], test['prediction_w']))

mae_tcn = mean_absolute_error(test['actual_w'], test['prediction'])
mae_phys = mean_absolute_error(test['actual_w'], test['prediction_w'])

r2_tcn = r2_score(test['actual_w'], test['prediction'])
r2_phys = r2_score(test['actual_w'], test['prediction_w'])

print("TCN RMSE:", rmse_tcn)
print("Physics RMSE:", rmse_phys)

print("TCN MAE:", mae_tcn)
print("Physics MAE:", mae_phys)

print("TCN R squared:", r2_tcn)
print("Physics R squared:", r2_phys)

test = merged.iloc[n//2:].copy()

def get_regime(kd, elev):
    if kd == 1 and elev <= 0:
        return 0
    elif kd < 0.3:
        return 1
    elif kd < 0.7:
        return 2
    else:
        return 3

test['true_regime'] = test.apply(lambda r: get_regime(r['kd_value'], r['solar_elevation_deg']), axis=1)
test['persisted_regime'] = test['true_regime'].shift(1)
test['fusionpv_persist_regime'] = np.nan

for idx, row in test.iterrows():
    regime = row['persisted_regime']
    if pd.isna(regime):
        continue
    if regime == 0:
        test.at[idx, 'fusionpv_persist_regime'] = row['prediction_w']
    else:
        X = pd.DataFrame([[row['prediction'], row['prediction_w']]],
                         columns=['prediction', 'prediction_w'])
        if regime == 1:
            pred = model_clear.predict(X)[0]
        elif regime == 2:
            pred = model_mixed.predict(X)[0]
        elif regime == 3:
            pred = model_cloudy.predict(X)[0]
        test.at[idx, 'fusionpv_persist_regime'] = pred

valid = test['fusionpv_persist_regime'].notna()
rmse_persist_regime = np.sqrt(mean_squared_error(
    test.loc[valid, 'actual_w'],
    test.loc[valid, 'fusionpv_persist_regime']
))
print("RMSE using persisted regime (96% accuracy):", rmse_persist_regime)

mae_persist_regime = mean_absolute_error(
    test.loc[valid, 'actual_w'],
    test.loc[valid, 'fusionpv_persist_regime']
)

r2_persist_regime = r2_score(
    test.loc[valid, 'actual_w'],
    test.loc[valid, 'fusionpv_persist_regime']
)

print("RMSE using persisted regime :", rmse_persist_regime)
print("MAE  using persisted regime :", mae_persist_regime)
print("R squared   using persisted regime :", r2_persist_regime)

model_global = LinearRegression()
X_all = test[['prediction', 'prediction_w']]
y_all = test['actual_w']
model_global.fit(X_all, y_all)
test['fusion_global'] = model_global.predict(X_all)

rmse_global = np.sqrt(mean_squared_error(test['actual_w'], test['fusion_global']))
mae_global = mean_absolute_error(test['actual_w'], test['fusion_global'])
r2_global = r2_score(test['actual_w'], test['fusion_global'])

print("=== Global Linear Fusion (no regimes) ===")
print(f"RMSE = {rmse_global:.2f} W")
print(f"MAE  = {mae_global:.2f} W")
print(f"R squared   = {r2_global:.4f}\n")