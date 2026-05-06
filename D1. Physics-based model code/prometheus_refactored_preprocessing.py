
!pip install astral

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from astral import LocationInfo
from astral.sun import sun
import datetime

"""#Fix meteo data errors"""

meteo=pd.read_csv('meteo_with_processed_wind_direction.csv')

meteo['time'] = pd.to_datetime(meteo['time'])

start_time = meteo['time'].min()
end_time = meteo['time'].max()
expected_times = pd.date_range(start=start_time.floor('min'), end=end_time.ceil('min'), freq='10min')
expected_times = expected_times + pd.Timedelta(minutes=0)

actual_times = set(meteo['time'])
missing_times = [t for t in expected_times if t not in actual_times]

print(f'Total missing timestamps: {len(missing_times)}')
print(missing_times)

df_expected = pd.DataFrame({'time': expected_times})
meteo_filled = df_expected.merge(meteo, on='time', how='left')
meteo_filled = meteo_filled.sort_values('time').reset_index(drop=True)

meteo_filled['wind direction'] = meteo_filled['wind direction'] % 360

print(meteo_filled.isna().sum())

print((meteo_filled == 0).sum())

def plot_meteo_with_nans(df, time_col='time'):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[time_col] = pd.to_datetime(df[time_col])

    for col in numeric_cols:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[col],
            mode='lines',
            name=col,
            line=dict(color='blue')
        ))

        nan_mask = df[col].isna()
        fig.add_trace(go.Scatter(
            x=df.loc[nan_mask, time_col],
            y=[df[col].min() - (0.05 * (df[col].max() - df[col].min()))]*nan_mask.sum(),
            mode='markers',
            marker=dict(color='red', size=6, symbol='x'),
            name='NaN'
        ))

        fig.update_layout(
            title=f'{col} over Time (NaNs in red)',
            xaxis_title='Time',
            yaxis_title=col,
            hovermode='x unified'
        )

        fig.show()

plot_meteo_with_nans(meteo_filled, time_col='time')

def consecutive_nans(df):
    results = {}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        mask = df[col].isna()
        group = (mask != mask.shift()).cumsum()
        counts = mask.groupby(group).sum()
        consecutive = counts[counts >= 2].tolist()
        results[col] = consecutive

    return results

nan_sequences = consecutive_nans(meteo_filled)

for col, seq in nan_sequences.items():
    if seq:
        print(f"Column '{col}' has consecutive NaNs with lengths: {seq}")
    else:
        print(f"Column '{col}' has no consecutive NaNs of length >=2")

def interpolate_small_gaps(df, time_col='time', max_gap=2, exclude_cols=('wind direction',)):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in exclude_cols]

    for col in numeric_cols:
        s = df[col]

        mask = s.isna()
        group = (mask != mask.shift()).cumsum()
        gap_sizes = mask.groupby(group).transform('sum')

        interp = s.interpolate(method='linear')
        df[col] = s.where(~mask | (gap_sizes > max_gap), interp)

    return df

meteo_interp = interpolate_small_gaps(meteo_filled, time_col='time', max_gap=2, exclude_cols=['wind direction'])

def interpolate_wind_direction(df, time_col='time', dir_col='wind direction', speed_col='wind speed', max_gap=2, speed_threshold=2):

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    s = df[dir_col]

    mask = s.isna()
    group = (mask != mask.shift()).cumsum()
    gap_sizes = mask.groupby(group).transform('sum')
    interp = s.interpolate(method='linear')

    cond = (mask & (gap_sizes <= max_gap) & (df[speed_col] > speed_threshold))
    df.loc[cond, dir_col] = interp[cond]

    return df

meteo_interp = interpolate_wind_direction(meteo_interp)

print(meteo_interp.isna().sum())

meteo_interp['time'] = pd.to_datetime(meteo_interp['time'])
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=meteo_interp['time'],
    y=meteo_interp['wind direction'],
    mode='lines',
    name='Raw Wind Direction',
    line=dict(color='blue', width=1)
))

fig.add_trace(go.Scatter(
    x=meteo_interp['time'],
    y=meteo_interp['wind_direction_processed'],
    mode='lines',
    name='Processed Wind Direction',
    line=dict(color='red', width=1)
))

fig.update_layout(
    title='Wind Direction: Raw vs Processed',
    xaxis_title='Time',
    yaxis_title='Wind Direction (degrees)',
    legend=dict(x=0.01, y=0.99),
    hovermode='x unified'
)

fig.show()

meteo_interp = meteo_interp.drop('wind direction', axis=1)

meteo_interp.rename(columns={'wind_direction_processed': 'wind direction'}, inplace=True)

meteo_interp.to_csv('meteo_fully_preprocessed.csv', index=False)

"""#Fix PV errors"""

PV=pd.read_csv('production_10min.csv')

city = LocationInfo("Sion", "Switzerland", "Sion/Switzerland", 46.227379, 7.364206)
start_date = datetime.date(2024, 7, 5)
end_date = datetime.date(2024, 10, 1)
delta = datetime.timedelta(days=1)

data = []
current_date = start_date
while current_date <= end_date:
    s = sun(city.observer, date=current_date)
    data.append({
        "Date": current_date,
        "Sunrise": s['sunrise'].strftime("%H:%M"),
        "Sunset": s['sunset'].strftime("%H:%M")
    })
    current_date += delta
rise_set= pd.DataFrame(data)

rise_set.to_csv("sunrise_sunset.csv", index=False)

rise_set['Sunrise'] = pd.to_datetime(rise_set['Date'].astype(str) + ' ' + rise_set['Sunrise'] + ':00')
rise_set['Sunset']  = pd.to_datetime(rise_set['Date'].astype(str) + ' ' + rise_set['Sunset'] + ':00')

rise_set['Next_Sunrise'] = rise_set['Sunrise'].shift(-1)

PV['Time'] = pd.to_datetime(PV['Time'])

PV['Date'] = PV['Time'].dt.date
PV_corrected = pd.merge(PV, rise_set[['Date','Sunrise','Sunset','Next_Sunrise']], how='left', on='Date')

night_mask = (PV_corrected['Time'] < PV_corrected['Sunrise']) | \
             ((PV_corrected['Time'] >= PV_corrected['Sunset']) & (PV_corrected['Time'] < PV_corrected['Next_Sunrise']))

PV_corrected['pv_production'] = PV_corrected['Value'].where(~night_mask, 0)

PV_corrected = PV_corrected[['Time', 'pv_production']]

PV_corrected = PV_corrected.iloc[:-1].reset_index(drop=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=PV_corrected['Time'],
    y=PV_corrected['pv_production'],
    mode='lines',
    name='PV Output',
    line=dict(color='orange')
))
fig.update_layout(
    title='PV Output',
    xaxis_title='Time',
    yaxis_title='PV Power Output',
    xaxis=dict(rangeslider=dict(visible=True)),
    legend=dict(x=0.01, y=0.99)
)

fig.show()

PV_corrected.rename(columns={'Time': 'time'}, inplace=True)

def get_weighted_avg_of_top_k(target_row, candidates, features, feature_weights, k=3, eps=1e-10):
    if candidates.empty:
        return 0.0

    if isinstance(feature_weights, dict):
        fw_flat = [feature_weights[feat] for feat in features]
    elif isinstance(feature_weights, (list, tuple, np.ndarray)):
        if feature_weights and isinstance(feature_weights[0], dict):
            weight_dict = {}
            for d in feature_weights:
                weight_dict.update(d)
            fw_flat = [weight_dict[feat] for feat in features]
        else:
            fw_flat = np.array(feature_weights, dtype=float).flatten().tolist()
    else:
        raise TypeError("feature_weights must be a dict, list, tuple, or numpy array")

    fw = np.array(fw_flat, dtype=float)
    if len(fw) != len(features):
        raise ValueError("Length of feature_weights must match length of features after extraction")

    distances = []
    for _, cand in candidates.iterrows():
        diff = 0.0
        for i, feat in enumerate(features):
            diff += fw[i] * (target_row[feat] - cand[feat]) ** 2
        distances.append(np.sqrt(diff))

    sorted_indices = np.argsort(distances)
    n_neighbours = min(k, len(candidates))
    top_indices = sorted_indices[:n_neighbours]

    top_distances = [distances[i] for i in top_indices]
    top_pv = candidates.iloc[top_indices]['pv_production'].values

    weights = 1.0 / (np.array(top_distances) + eps)
    weights /= weights.sum()
    return np.dot(weights, top_pv)


PV_corrected['time'] = pd.to_datetime(PV_corrected['time'])
meteo_interp['time'] = pd.to_datetime(meteo_interp['time'])

meteo_interp.columns = [col.replace(' ', '_') for col in meteo_interp.columns]

df = pd.merge(PV_corrected, meteo_interp, on='time', how='left')

features = ['irradiance', 'temperature', 'humidity', 'precipitation',
            'wind_speed', 'wind_direction', 'sunshine_duration']

df[features] = df[features].ffill().bfill()

df.dropna(subset=features, inplace=True)

target_dates = ['2024-08-30', '2024-08-31', '2024-09-01', '2024-09-02']

df_missing_days = df[df['time'].dt.strftime('%Y-%m-%d').isin(target_dates)].copy()
df_reference = df[~df['time'].dt.strftime('%Y-%m-%d').isin(target_dates)].copy()

df_reference['minutes'] = df_reference['time'].dt.hour * 60 + df_reference['time'].dt.minute

X_train = df_reference[features]
y_train = df_reference['pv_production']

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_weights = dict(zip(features, importances))
print("Feature weights from RF:", feature_weights)

df_missing_days['PV_imputed_weighted'] = np.nan

for idx, row in df_missing_days.iterrows():
    target_minutes = row['time'].hour * 60 + row['time'].minute

    lower = target_minutes - 30
    upper = target_minutes + 30
    if lower >= 0 and upper < 1440:
        mask = (df_reference['minutes'] >= lower) & (df_reference['minutes'] <= upper)
    elif lower < 0:
        mask = (df_reference['minutes'] >= lower + 1440) | (df_reference['minutes'] <= upper)
    else:
        mask = (df_reference['minutes'] >= lower) | (df_reference['minutes'] <= upper - 1440)

    same_time_rows = df_reference[mask]

    if same_time_rows.empty:
        new_val = 0
    elif (same_time_rows['pv_production'] == 0).all():
        new_val = 0
    else:
        new_val = get_weighted_avg_of_top_k(
            row, same_time_rows, features, feature_weights, k=3
        )

    df_missing_days.at[idx, 'PV_imputed_weighted'] = new_val

if df_missing_days['PV_imputed_weighted'].isna().any():
    print("Warning: Some imputed values are NaN. Filling with 0.")
    df_missing_days['PV_imputed_weighted'].fillna(0, inplace=True)

rmse_weighted = np.sqrt(mean_squared_error(
    df_missing_days['pv_production'],
    df_missing_days['PV_imputed_weighted']
))
print(f"Weighted Analogue RMSE: {rmse_weighted:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_missing_days['time'],
    y=df_missing_days['pv_production'],
    mode='lines', name='Actual PV'
))
fig.add_trace(go.Scatter(
    x=df_missing_days['time'],
    y=df_missing_days['PV_imputed_weighted'],
    mode='lines', name='Weighted Analogue PV'
))
fig.update_layout(
    title='Actual vs Weighted Analogue PV: Aug 30 to Sep 3',
    xaxis_title='Time',
    yaxis_title='PV Output (kW)'
)
fig.show()

PV_corrected['time'] = pd.to_datetime(PV_corrected['time'])
df_missing_days['time'] = pd.to_datetime(df_missing_days['time'])

start_time = pd.Timestamp('2024-08-30 14:01:00')
end_time   = pd.Timestamp('2024-09-02 16:01:00')

mask_replace = (PV_corrected['time'] > start_time) & (PV_corrected['time'] < end_time)

PV_corrected = PV_corrected.merge(
    df_missing_days[['time', 'PV_imputed_weighted']],
    on='time',
    how='left',
    suffixes=('', '_imputed')
)

PV_corrected.loc[mask_replace, 'pv_production'] = PV_corrected.loc[mask_replace, 'PV_imputed_weighted']

PV_corrected.drop(columns=['PV_imputed_weighted'], inplace=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=PV_corrected['time'],
    y=PV_corrected['pv_production'],
    mode='lines',
    name='PV Output',
    line=dict(color='orange')
))
fig.update_layout(
    title='PV Output',
    xaxis_title='Time',
    yaxis_title='PV Power Output',
    xaxis=dict(rangeslider=dict(visible=True)),
    legend=dict(x=0.01, y=0.99)
)

fig.show()

PV_corrected.to_csv('production_fully_preprocessed.csv', index=False)

"""#Missing days reconstruction experiments"""

#non-weighted similarity using only data from july
PV_corrected['time'] = pd.to_datetime(PV_corrected['time'])
meteo_interp['time'] = pd.to_datetime(meteo_interp['time'])

meteo_interp.columns = [col.replace(' ', '_') for col in meteo_interp.columns]
df = pd.merge(PV_corrected, meteo_interp, left_on='time', right_on='time', how='left')

meteo_features = [col for col in meteo_interp.columns if col not in ['time', 'pv_production']]

for col in meteo_features:
     df[col] = df[col].fillna(df[col].median())

scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(df[meteo_features])
for i, col in enumerate(meteo_features):
    df[col + '_scaled'] = X_scaled_all[:, i]
scaled_features = [col + '_scaled' for col in meteo_features]

df['date'] = df['time'].dt.date
july_mask = (df['time'].dt.month == 7)
df_july = df[july_mask].copy()

last_three_days_cutoff = df_july['time'].max() - pd.Timedelta(days=3)
train_df = df_july[df_july['time'] < last_three_days_cutoff]
validation_df = df_july[df_july['time'] >= last_three_days_cutoff]

def get_most_similar_value(current_row, other_rows, feature_cols, target_col='pv_production'):
    distances = euclidean_distances(
        current_row[feature_cols].values.reshape(1, -1),
        other_rows[feature_cols].values
    ).flatten()
    return other_rows.iloc[np.argmin(distances)][target_col]

validation_df = validation_df.copy()
validation_df['Value_imputed'] = np.nan

for idx, row in validation_df.iterrows():
    same_time_rows = train_df[train_df['time'].dt.time == row['time'].time()]

    if same_time_rows.empty:
        new_val = 0
    elif (same_time_rows['pv_production'] == 0).all():
        new_val = 0
    else:
        new_val = get_most_similar_value(row, same_time_rows, scaled_features)

    validation_df.at[idx, 'Value_imputed'] = new_val

rmse = np.sqrt(mean_squared_error(validation_df['pv_production'], validation_df['Value_imputed']))
print(f"RMSE for last 3 days of July: {rmse:.2f}")

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=validation_df['time'], y=validation_df['pv_production'],
                         mode='lines', name='Actual PV'))
fig.add_trace(go.Scatter(x=validation_df['time'], y=validation_df['Value_imputed'],
                         mode='lines', name='Imputed PV'))

fig.update_layout(title='Actual vs Imputed PV: Last 3 Days of July',
                  xaxis_title='Time', yaxis_title='PV Output')
fig.show()

#non-weighted similarity using all data
PV_corrected['time'] = pd.to_datetime(PV_corrected['time'])
meteo_interp['time'] = pd.to_datetime(meteo_interp['time'])
meteo_interp.columns = [col.replace(' ', '_') for col in meteo_interp.columns]

df = pd.merge(PV_corrected, meteo_interp, left_on='time', right_on='time', how='left')

meteo_features = [col for col in meteo_interp.columns if col not in ['time', 'pv_production']]

for col in meteo_features:
  df[col] = df[col].fillna(df[col].median())
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(df[meteo_features])
for i, col in enumerate(meteo_features):
    df[col + '_scaled'] = X_scaled_all[:, i]
scaled_features = [col + '_scaled' for col in meteo_features]
df['date'] = df['time'].dt.date
july_mask = (df['time'].dt.month == 7)
df_july = df[july_mask].copy()

target_dates = ['2024-07-29', '2024-07-30', '2024-07-31']
missing_days_mask = df_july['time'].dt.strftime('%Y-%m-%d').isin(target_dates)
df_missing_days = df_july[missing_days_mask].copy()

invalid_dates = ['2024-08-31', '2024-09-01', '2024-09-02', '2024-09-03']

df_reference = df[
    ~df['time'].dt.strftime('%Y-%m-%d').isin(target_dates + invalid_dates)
].copy()

def get_most_similar_value(current_row, other_rows, feature_cols, target_col='pv_production'):
    distances = euclidean_distances(
        current_row[feature_cols].values.reshape(1, -1),
        other_rows[feature_cols].values
    ).flatten()
    return other_rows.iloc[np.argmin(distances)][target_col]

df_missing_days['Value_imputed'] = np.nan

for idx, row in df_missing_days.iterrows():
    same_time_rows = df_reference[df_reference['time'].dt.time == row['time'].time()]

    if same_time_rows.empty:
        new_val = 0
    elif (same_time_rows['pv_production'] == 0).all():
        new_val = 0
    else:
        new_val = get_most_similar_value(row, same_time_rows, scaled_features)

    df_missing_days.at[idx, 'Value_imputed'] = new_val

rmse_all_data = np.sqrt(mean_squared_error(df_missing_days['pv_production'], df_missing_days['Value_imputed']))
print(f"RMSE for last 3 days of July using all available data: {rmse_all_data:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_missing_days['time'], y=df_missing_days['pv_production'],
                         mode='lines', name='Actual PV'))
fig.add_trace(go.Scatter(x=df_missing_days['time'], y=df_missing_days['Value_imputed'],
                         mode='lines', name='Imputed PV (all data)'))

fig.update_layout(title='Actual vs Imputed PV: Last 3 Days of July (all aata)',
                  xaxis_title='Time', yaxis_title='PV Output')
fig.show()

#random forest prediction
features = ['irradiance', 'temperature', 'humidity', 'precipitation', 'wind_speed', 'wind_direction', 'sunshine_duration']
df_reference = df[~df['time'].dt.strftime('%Y-%m-%d').isin(target_dates + invalid_dates)].copy()

X_train = df_reference[features]
y_train = df_reference['pv_production']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

X_val = df_missing_days[features]
X_val_scaled = scaler.transform(X_val)

df_missing_days['Value_imputed_RF'] = rf_model.predict(X_val_scaled)

rmse_rf = np.sqrt(mean_squared_error(df_missing_days['pv_production'], df_missing_days['Value_imputed_RF']))
print(f"Random Forest RMSE: {rmse_rf:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_missing_days['time'], y=df_missing_days['pv_production'],
                         mode='lines', name='Actual PV'))
fig.add_trace(go.Scatter(x=df_missing_days['time'], y=df_missing_days['Value_imputed_RF'],
                         mode='lines', name='Random Forest Predicted PV'))

fig.update_layout(title='Actual vs Random Forest Imputed PV: Last 3 Days of July',
                  xaxis_title='Time', yaxis_title='PV Output (kW)',
                  legend=dict(x=0.02, y=0.98))

fig.show()

#weighted similarity for last values of july
PV_corrected['time'] = pd.to_datetime(PV_corrected['time'])
meteo_interp['time'] = pd.to_datetime(meteo_interp['time'])

meteo_interp.columns = [col.replace(' ', '_') for col in meteo_interp.columns]
df = pd.merge(PV_corrected, meteo_interp, left_on='time', right_on='time', how='left')

meteo_features = [col for col in meteo_interp.columns if col not in ['time', 'pv_production']]
for col in meteo_features:
    df[col] = df[col].fillna(df[col].median())
features = ['irradiance', 'temperature', 'humidity', 'precipitation', 'wind_speed', 'wind_direction', 'sunshine_duration']
target_dates = ['2024-07-29', '2024-07-30', '2024-07-31']
invalid_dates = ['2024-08-31', '2024-09-01', '2024-09-02', '2024-09-03']

df['date'] = df['time'].dt.date
july_mask = (df['time'].dt.month == 7)
df_july = df[july_mask].copy()
df_missing_days = df_july[df_july['time'].dt.strftime('%Y-%m-%d').isin(target_dates)].copy()

df_reference = df[~df['time'].dt.strftime('%Y-%m-%d').isin(target_dates + invalid_dates)].copy()

X_train = df_reference[features]
y_train = df_reference['pv_production']

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_weights = dict(zip(features, importances))
print("Feature weights from RF:", feature_weights)

def get_most_similar_weighted(current_row, other_rows, feature_cols, feature_weights, target_col='pv_production'):
    X1 = current_row[feature_cols].values.astype(float).reshape(1, -1)
    X2 = other_rows[feature_cols].values.astype(float)
    W = np.array([float(feature_weights[f]) for f in feature_cols])
    W_sqrt = np.sqrt(W)
    X1_weighted = X1 * W_sqrt
    X2_weighted = X2 * W_sqrt
    diff = X2_weighted - X1_weighted
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return other_rows.iloc[np.argmin(distances)][target_col]

df_missing_days['PV_imputed_weighted'] = np.nan

for idx, row in df_missing_days.iterrows():

    same_time_rows = df_reference[df_reference['time'].dt.time == row['time'].time()]

    if same_time_rows.empty:
        new_val = 0
    elif (same_time_rows['pv_production'] == 0).all():
        new_val = 0
    else:
        new_val = get_most_similar_weighted(row, same_time_rows, features, feature_weights)

    df_missing_days.at[idx, 'PV_imputed_weighted'] = new_val

rmse_weighted = np.sqrt(mean_squared_error(df_missing_days['pv_production'], df_missing_days['PV_imputed_weighted']))
print(f"Weighted Similarity RMSE: {rmse_weighted:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_missing_days['time'], y=df_missing_days['pv_production'],
                         mode='lines', name='Actual PV'))
fig.add_trace(go.Scatter(x=df_missing_days['time'], y=df_missing_days['PV_imputed_weighted'],
                         mode='lines', name='Estimated PV'))

fig.update_layout(title='Actual vs Weighted Similarity PV: Last 3 Days of July',
                  xaxis_title='Time', yaxis_title='PV Output (kW)')
fig.show()

#weighted similarity for August
target_dates = ['2024-08-12', '2024-08-13', '2024-08-14']
invalid_dates = ['2024-08-31', '2024-09-01', '2024-09-02', '2024-09-03']

df['date'] = df['time'].dt.date
september_mask = (df['time'].dt.month == 8)
df_sep = df[september_mask].copy()

df_missing_days = df_sep[
    df_sep['time'].dt.strftime('%Y-%m-%d').isin(target_dates)
].copy()

df_reference = df[
    ~df['time'].dt.strftime('%Y-%m-%d').isin(target_dates + invalid_dates)
].copy()

df_missing_days['PV_imputed_weighted'] = np.nan

for idx, row in df_missing_days.iterrows():
    same_time_rows = df_reference[
        df_reference['time'].dt.time == row['time'].time()
    ]

    if same_time_rows.empty:
        new_val = 0
    elif (same_time_rows['pv_production'] == 0).all():
        new_val = 0
    else:
        new_val = get_most_similar_weighted(
            row, same_time_rows, features, feature_weights
        )

    df_missing_days.at[idx, 'PV_imputed_weighted'] = new_val

rmse_weighted = np.sqrt(mean_squared_error(
    df_missing_days['pv_production'],
    df_missing_days['PV_imputed_weighted']
))
print(f"Weighted Similarity RMSE (Aug): {rmse_weighted:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_missing_days['time'],
    y=df_missing_days['pv_production'],
    mode='lines', name='Actual PV'
))
fig.add_trace(go.Scatter(
    x=df_missing_days['time'],
    y=df_missing_days['PV_imputed_weighted'],
    mode='lines', name='Estimated PV'
))

fig.update_layout(
    title='Actual vs Weighted Similarity PV',
    xaxis_title='Time',
    yaxis_title='PV Output (kW)'
)

fig.show()

#time-window weighted similarity for August
target_dates = ['2024-08-12', '2024-08-13', '2024-08-14']
invalid_dates = ['2024-08-31', '2024-09-01', '2024-09-02', '2024-09-03']

df['date'] = df['time'].dt.date
september_mask = (df['time'].dt.month == 8)
df_sep = df[september_mask].copy()

df_missing_days = df_sep[
    df_sep['time'].dt.strftime('%Y-%m-%d').isin(target_dates)
].copy()

df_reference = df[
    ~df['time'].dt.strftime('%Y-%m-%d').isin(target_dates + invalid_dates)
].copy()

df_reference['minutes'] = df_reference['time'].dt.hour * 60 + df_reference['time'].dt.minute

df_missing_days['PV_imputed_weighted'] = np.nan

for idx, row in df_missing_days.iterrows():
    target_minutes = row['time'].hour * 60 + row['time'].minute

    lower = target_minutes - 30
    upper = target_minutes + 30
    if lower >= 0 and upper < 1440:
        mask = (df_reference['minutes'] >= lower) & (df_reference['minutes'] <= upper)
    elif lower < 0:
        mask = (df_reference['minutes'] >= lower + 1440) | (df_reference['minutes'] <= upper)
    else:
        mask = (df_reference['minutes'] >= lower) | (df_reference['minutes'] <= upper - 1440)

    same_time_rows = df_reference[mask]

    if same_time_rows.empty:
        new_val = 0
    elif (same_time_rows['pv_production'] == 0).all():
        new_val = 0
    else:
        new_val = get_most_similar_weighted(
            row, same_time_rows, features, feature_weights
        )

    df_missing_days.at[idx, 'PV_imputed_weighted'] = new_val

rmse_weighted = np.sqrt(mean_squared_error(
    df_missing_days['pv_production'],
    df_missing_days['PV_imputed_weighted']
))
print(f"Time Window Weighted Similarity RMSE (Aug): {rmse_weighted:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_missing_days['time'],
    y=df_missing_days['pv_production'],
    mode='lines', name='Actual PV'
))
fig.add_trace(go.Scatter(
    x=df_missing_days['time'],
    y=df_missing_days['PV_imputed_weighted'],
    mode='lines', name='Estimated PV'
))

fig.update_layout(
    title='Actual vs Time Window Weighted Similarity PV',
    xaxis_title='Time',
    yaxis_title='PV Output (kW)'
)

fig.show()

def get_weighted_avg_of_top_k(target_row, candidates, features, feature_weights, k=3, eps=1e-10):
    if candidates.empty:
        return 0.0
    if isinstance(feature_weights, dict):

        fw_flat = [feature_weights[feat] for feat in features]
    elif isinstance(feature_weights, (list, tuple, np.ndarray)):

        if feature_weights and isinstance(feature_weights[0], dict):

            weight_dict = {}
            for d in feature_weights:
                weight_dict.update(d)
            fw_flat = [weight_dict[feat] for feat in features]
        else:

            fw_flat = np.array(feature_weights, dtype=float).flatten().tolist()
    else:
        raise TypeError("feature_weights must be a dict, list, tuple, or numpy array")

    fw = np.array(fw_flat, dtype=float)
    if len(fw) != len(features):
        raise ValueError("Length of feature_weights must match length of features after extraction")

    distances = []
    for _, cand in candidates.iterrows():
        diff = 0.0
        for i, feat in enumerate(features):
            diff += fw[i] * (target_row[feat] - cand[feat]) ** 2
        distances.append(np.sqrt(diff))


    sorted_indices = np.argsort(distances)
    n_neighbours = min(k, len(candidates))
    top_indices = sorted_indices[:n_neighbours]

    top_distances = [distances[i] for i in top_indices]
    top_pv = candidates.iloc[top_indices]['pv_production'].values

    weights = 1.0 / (np.array(top_distances) + eps)
    weights /= weights.sum()

    return np.dot(weights, top_pv)

#neighbour-based time window weighted similarity
target_dates = ['2024-08-12', '2024-08-13', '2024-08-14']
invalid_dates = ['2024-08-31', '2024-09-01', '2024-09-02', '2024-09-03']

df['date'] = df['time'].dt.date
september_mask = (df['time'].dt.month == 8)
df_sep = df[september_mask].copy()

df_missing_days = df_sep[
    df_sep['time'].dt.strftime('%Y-%m-%d').isin(target_dates)
].copy()

df_reference = df[
    ~df['time'].dt.strftime('%Y-%m-%d').isin(target_dates + invalid_dates)
].copy()

df_reference['minutes'] = df_reference['time'].dt.hour * 60 + df_reference['time'].dt.minute

df_missing_days['PV_imputed_weighted'] = np.nan

for idx, row in df_missing_days.iterrows():
    target_minutes = row['time'].hour * 60 + row['time'].minute

    lower = target_minutes - 30
    upper = target_minutes + 30
    if lower >= 0 and upper < 1440:
        mask = (df_reference['minutes'] >= lower) & (df_reference['minutes'] <= upper)
    elif lower < 0:
        mask = (df_reference['minutes'] >= lower + 1440) | (df_reference['minutes'] <= upper)
    else:
        mask = (df_reference['minutes'] >= lower) | (df_reference['minutes'] <= upper - 1440)

    same_time_rows = df_reference[mask]

    if same_time_rows.empty:
        new_val = 0
    elif (same_time_rows['pv_production'] == 0).all():
        new_val = 0
    else:
        new_val = get_weighted_avg_of_top_k(
            row, same_time_rows, features, feature_weights, k=3
        )

    df_missing_days.at[idx, 'PV_imputed_weighted'] = new_val


rmse_weighted = np.sqrt(mean_squared_error(
    df_missing_days['pv_production'],
    df_missing_days['PV_imputed_weighted']
))
print(f"Neighbour-based Time Window Weighted Similarity RMSE (Aug): {rmse_weighted:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_missing_days['time'],
    y=df_missing_days['pv_production'],
    mode='lines', name='Actual PV'
))
fig.add_trace(go.Scatter(
    x=df_missing_days['time'],
    y=df_missing_days['PV_imputed_weighted'],
    mode='lines', name='Estimated PV'
))

fig.update_layout(
    title='Actual vs Neighbour-based Time Window Weighted Similarity PV',
    xaxis_title='Time',
    yaxis_title='PV Output (kW)'
)

fig.show()