import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import normalized_mutual_info_score



meteo = pd.read_csv('meteo_with_estimation.csv')
meteo['time'] = pd.to_datetime(meteo['time'])
meteo = meteo.sort_values('time')



def dir_to_sin_cos(angle_degree):
    #this decomposition allows to avoid problems when averaging angles (ie mean of 359° and 1° is 0° and not 180°)
    rad = np.radians(angle_degree)
    return np.sin(rad), np.cos(rad)


def sin_cos_to_dir(sin_val, cos_val):

    return np.degrees(np.arctan2(sin_val, cos_val)) % 360


def ang_diff(angle1, angle2):

    diff = np.abs(angle1 - angle2) % 360
    return np.minimum(diff, 360 - diff)


def circular_std(angles):
    angles = np.array(angles)[~np.isnan(angles)]
    if len(angles) < 2:
        return np.nan
    rad = np.radians(angles)
    R = np.sqrt(np.sum(np.sin(rad))**2 + np.sum(np.cos(rad))** 2) /len(rad)
    if R > 0:
        return np.sqrt(-2*np.log(R))*(180 / np.pi)
    return 180


def calculate_information_loss(original, smoothed, n_bins=36):


    

    eps = 1e-12


    valid = ~np.isnan(original) & ~np.isnan(smoothed)
    orig_valid = np.asarray(original[valid], dtype=float)
    smooth_valid = np.asarray(smoothed[valid], dtype=float)

    if len(orig_valid) == 0:
        return {
            'MSE_sin_cos': np.nan,
            'Angular_RMSE': np.nan,
            'Entropy_ratio': np.nan,
            'KL_divergence': np.nan,
            'Normalized_MI': np.nan,
            'AC_preservation': np.nan,
            'n_valid': 0
        }


    orig_valid = np.mod(orig_valid, 360.0)
    smooth_valid = np.mod(smooth_valid, 360.0)


    orig_sin, orig_cos = dir_to_sin_cos(orig_valid)
    smooth_sin, smooth_cos = dir_to_sin_cos(smooth_valid)

    mse_sin = np.mean((orig_sin - smooth_sin) ** 2)
    mse_cos = np.mean((orig_cos - smooth_cos) ** 2)
    mse_total = (mse_sin + mse_cos) / 2

    angular_errors = ang_diff(orig_valid, smooth_valid)
    angular_rmse = np.sqrt(np.mean(angular_errors ** 2))


    bins = np.linspace(0.0, 360.0, n_bins + 1)

    hist_orig, _ = np.histogram(orig_valid, bins=bins, density=False)
    hist_smooth, _ = np.histogram(smooth_valid, bins=bins, density=False)


    hist_orig = hist_orig.astype(float) + eps
    hist_smooth = hist_smooth.astype(float) + eps

    hist_orig /= hist_orig.sum()
    hist_smooth /= hist_smooth.sum()


    entropy_orig = entropy(hist_orig)
    entropy_smooth = entropy(hist_smooth)
    entropy_ratio = entropy_smooth / entropy_orig if entropy_orig > 0 else np.nan

    # KL divergence (= how the distribution is affected)
    kl_div = np.sum(hist_orig * (np.log(hist_orig) - np.log(hist_smooth)))


    orig_disc = np.digitize(orig_valid, bins) - 1
    smooth_disc = np.digitize(smooth_valid, bins) - 1

    orig_disc = np.clip(orig_disc, 0, n_bins - 1)
    smooth_disc = np.clip(smooth_disc, 0, n_bins - 1)


    if np.allclose(orig_valid, smooth_valid, atol=1e-10, rtol=1e-10):
        mi_normalized = 1.0
    else:
        mi_normalized = normalized_mutual_info_score(orig_disc, smooth_disc)


    if len(orig_valid) > 5:
        sample_size = min(1000, len(orig_valid))

        orig_sample = orig_valid[:sample_size]
        smooth_sample = smooth_valid[:sample_size]


        orig_complex = np.exp(1j * np.deg2rad(orig_sample))
        smooth_complex = np.exp(1j * np.deg2rad(smooth_sample))

        def lag1_autocorr(x):
            x0 = x[:-1]
            x1 = x[1:]
            if len(x0) < 2:
                return np.nan
            num = np.mean(x0 * np.conj(x1))
            den = np.mean(np.abs(x0) ** 2)
            return np.real(num / den) if den > eps else np.nan

        ac_orig = lag1_autocorr(orig_complex)
        ac_smooth = lag1_autocorr(smooth_complex)

        if not np.isnan(ac_orig) and not np.isnan(ac_smooth) and abs(ac_orig) > eps:
            ac_preservation = 1 - abs(ac_orig - ac_smooth) / abs(ac_orig)
        else:
            ac_preservation = np.nan
    else:
        ac_preservation = np.nan


    return {
        'MSE_sin_cos': mse_total,
        'Angular_RMSE': angular_rmse,
        'Entropy_ratio': entropy_ratio,
        'KL_divergence': kl_div,
        'Normalized_MI': mi_normalized,
        'AC_preservation': ac_preservation,
        'n_valid': len(orig_valid)
    }



print("\n" + "=" * 80)
print("PART 1: ANALYZING DATA CHARACTERISTICS")
print("=" * 80 + "\n")


mask = meteo['wind speed'].notna() & meteo['wind direction'].notna()
speeds = meteo.loc[mask, 'wind speed'].values
directions = meteo.loc[mask, 'wind direction'].values


variability = np.full_like(speeds, np.nan, dtype=float)
for i in range(len(directions)):
    start = max(0, i - 3)
    end = min(len(directions), i + 4)
    window = directions[start:end]
    window = window[~np.isnan(window)]
    if len(window) >= 3:
        variability[i] = circular_std(window)


valid = ~np.isnan(variability)
speeds = speeds[valid]
directions = directions[valid]
variability = variability[valid]

print(f"Total valid data points: {len(speeds)}")
print(f"Mean wind speed: {speeds.mean():.2f} m/s")
print(f"Mean variability: {variability.mean():.1f}°")


fig, axes = plt.subplots(2, 3, figsize=(15, 10))


ax1 = axes[0, 0]
scatter = ax1.scatter(speeds, variability, c=directions, cmap='viridis',
                      alpha=0.5, s=5, vmin=0, vmax=360)
ax1.set_xlabel('Wind Speed (m/s)')
ax1.set_ylabel('Circular Std (degrees)')
ax1.set_title('Original: Variability vs Speed', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(2, color='red', linestyle='--', alpha=0.5, label='Threshold')
ax1.legend()
plt.colorbar(scatter, ax=ax1, label='Direction (°)')


ax2 = axes[0, 1]
ax2.hist(variability, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
ax2.set_xlabel('Circular Std (degrees)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Local Variability', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(30, color='red', linestyle='--', alpha=0.5, label='Low variability')
ax2.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Medium')
ax2.axvline(90, color='green', linestyle='--', alpha=0.5, label='High')
ax2.legend()

ax3 = axes[0, 2]
speed_bins = np.linspace(0, 15, 16)
bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2
mean_var_by_speed = []
for i in range(len(speed_bins) - 1):
    mask_bin = (speeds >= speed_bins[i]) & (speeds < speed_bins[i + 1])
    if mask_bin.sum() > 0:
        mean_var_by_speed.append(np.mean(variability[mask_bin]))
    else:
        mean_var_by_speed.append(0)

ax3.bar(bin_centers, mean_var_by_speed, width=0.9, color='steelblue', alpha=0.7)
ax3.set_xlabel('Wind Speed (m/s)')
ax3.set_ylabel('Mean Circular Std')
ax3.set_title('Mean Variability by Speed Bin', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axvline(2, color='red', linestyle='--', alpha=0.5)


ax4 = axes[1, 0]
low_mask = speeds < 2
low_dirs = directions[low_mask]
ax4.hist(low_dirs, bins=36, range=(0, 360), color='crimson', alpha=0.7, edgecolor='white')
ax4.set_xlabel('Wind Direction (°)')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Low Speed (<2 m/s) Direction Distribution\nn={len(low_dirs)}', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 360)


ax5 = axes[1, 1]
high_mask = speeds >= 2
high_dirs = directions[high_mask]
ax5.hist(high_dirs, bins=36, range=(0, 360), color='navy', alpha=0.7, edgecolor='white')
ax5.set_xlabel('Wind Direction (°)')
ax5.set_ylabel('Frequency')
ax5.set_title(f'High Speed (≥2 m/s) Direction Distribution\nn={len(high_dirs)}', fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 360)


ax6 = axes[1, 2]
low_var_mask = variability < 30
med_var_mask = (variability >= 30) & (variability < 60)
high_var_mask = variability >= 60

sizes = [low_var_mask.sum(), med_var_mask.sum(), high_var_mask.sum()]
labels = [f'Low Var (<30°)\n{sizes[0]} pts',
          f'Medium Var (30-60°)\n{sizes[1]} pts',
          f'High Var (>60°)\n{sizes[2]} pts']
colors = ['forestgreen', 'orange', 'crimson']
ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax6.set_title('Data Points by Variability Level', fontweight='bold')

plt.tight_layout()
plt.savefig('data_characteristics.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("PART 2: TESTING ADAPTIVE SMOOTHING STRATEGIES")
print("=" * 80 + "\n")



def apply_no_smoothing(df):

    return df['wind direction'].copy()


def apply_uniform_smoothing(df, sigma=2):

    sin, cos = dir_to_sin_cos(df['wind direction'].values)

    sin_series = pd.Series(sin)
    cos_series = pd.Series(cos)

    sin_smooth = sin_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')

    cos_smooth = cos_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')

    sin_smooth = gaussian_filter1d(sin_smooth, sigma=sigma)
    cos_smooth = gaussian_filter1d(cos_smooth, sigma=sigma)

    norm = np.sqrt(sin_smooth ** 2 + cos_smooth ** 2)
    norm[norm == 0] = 1
    sin_smooth = sin_smooth / norm
    cos_smooth = cos_smooth / norm

    return sin_cos_to_dir(sin_smooth, cos_smooth)


def apply_savgol_filter(df, window_length=11, polyorder=3):

    sin, cos = dir_to_sin_cos(df['wind direction'].values)

    sin_series = pd.Series(sin)
    cos_series = pd.Series(cos)


    sin_filled = sin_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')
    cos_filled = cos_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')


    window_length = min(window_length, len(sin_filled) - 1)
    if window_length % 2 == 0:
        window_length += 1


    try:
        sin_smooth = savgol_filter(sin_filled.values, window_length, polyorder, mode='interp')
        cos_smooth = savgol_filter(cos_filled.values, window_length, polyorder, mode='interp')
    except ValueError:

        sin_smooth = gaussian_filter1d(sin_filled.values, sigma=2)
        cos_smooth = gaussian_filter1d(cos_filled.values, sigma=2)


    norm = np.sqrt(sin_smooth ** 2 + cos_smooth ** 2)
    norm[norm == 0] = 1
    sin_smooth = sin_smooth / norm
    cos_smooth = cos_smooth / norm

    return sin_cos_to_dir(sin_smooth, cos_smooth)


def apply_adaptive_savgol(df, speed_col='wind speed'):

    directions = df['wind direction'].values
    speeds = df[speed_col].values

    sin, cos = dir_to_sin_cos(directions)
    sin_series = pd.Series(sin)
    cos_series = pd.Series(cos)

    sin_filled = sin_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')
    cos_filled = cos_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')

    sin_smooth = np.zeros_like(sin_filled)
    cos_smooth = np.zeros_like(cos_filled)

    for i in range(len(sin_filled)):
        if pd.isna(speeds[i]):

            window = 11
            poly = 3
        elif speeds[i] < 2:
            window = 15
            poly = 2
        elif speeds[i] < 5:
            window = 11
            poly = 3
        else:
            window = 7
            poly = 3


        window = min(window, len(sin_filled))
        if window % 2 == 0:
            window += 1 # the window has to be odd



        start = max(0, i - window//2)
        end = min(len(sin_filled), i + window//2 + 1)

        
        sin_window = sin_filled.iloc[start:end].values
        cos_window = cos_filled.iloc[start:end].values
        actual_window = len(sin_window)

        if actual_window >= window // 2 + 1:
            # Apply Savgol to the window and take the central point
            try:
                if actual_window >= poly + 1:
                    sin_smooth_win = savgol_filter(sin_window, actual_window, min(poly, actual_window - 1),
                                                   mode='interp')
                    cos_smooth_win = savgol_filter(cos_window, actual_window, min(poly, actual_window - 1),
                                                   mode='interp')

                    # Find index of current point in the window
                    idx_in_window = i - start
                    sin_smooth[i] = sin_smooth_win[idx_in_window]
                    cos_smooth[i] = cos_smooth_win[idx_in_window]
                else:
                    # Fallback to original value if window too small
                    sin_smooth[i] = sin_filled.iloc[i]
                    cos_smooth[i] = cos_filled.iloc[i]
            except:
                # Fallback to original value if Savgol fails
                sin_smooth[i] = sin_filled.iloc[i]
                cos_smooth[i] = cos_filled.iloc[i]
        else:
            # Fallback to original value
            sin_smooth[i] = sin_filled.iloc[i]
            cos_smooth[i] = cos_filled.iloc[i]

    # Normalize
    norm = np.sqrt(sin_smooth ** 2 + cos_smooth ** 2)
    norm[norm == 0] = 1
    sin_smooth = sin_smooth / norm
    cos_smooth = cos_smooth / norm

    return sin_cos_to_dir(sin_smooth, cos_smooth)


def apply_adaptive_smoothing(df, speed_col='wind speed'):
    
    directions = df['wind direction'].values
    speeds = df[speed_col].values

    
    rolling_var = np.full_like(directions, np.nan, dtype=float)

    for i in range(len(directions)):
        start = max(0, i - 6)
        end = min(len(directions), i + 7)
        window = directions[start:end]
        window = window[~np.isnan(window)]
        if len(window) >= 3:
            rolling_var[i] = circular_std(window)

    
    sin, cos = dir_to_sin_cos(directions)
    sin_series = pd.Series(sin)
    cos_series = pd.Series(cos)

    sin_filled = sin_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')
    cos_filled = cos_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')

    sin_smooth = np.zeros_like(sin_filled)
    cos_smooth = np.zeros_like(cos_filled)

    for i in range(len(sin_filled)):
        if pd.isna(rolling_var[i]) or pd.isna(speeds[i]):
            sigma = 2
        elif speeds[i] < 2:
            #Low speed -> heavy smoothing
            sigma = 4
        elif speeds[i] < 5:
            sigma = 2
        else:
            sigma = 1

        window_size = int(sigma * 3)
        start = max(0, i - window_size)
        end = min(len(sin_filled), i + window_size + 1)

        positions = np.arange(start, end)
        weights = np.exp(-0.5 * ((positions - i) / sigma) ** 2)
        if weights.sum() > 0:
            weights = weights / weights.sum()

            sin_smooth[i] = np.sum(sin_filled.iloc[start:end].values * weights)
            cos_smooth[i] = np.sum(cos_filled.iloc[start:end].values * weights)
        else:
            sin_smooth[i] = sin_filled.iloc[i]
            cos_smooth[i] = cos_filled.iloc[i]

    norm = np.sqrt(sin_smooth ** 2 + cos_smooth ** 2)
    norm[norm == 0] = 1
    sin_smooth = sin_smooth / norm
    cos_smooth = cos_smooth / norm

    return sin_cos_to_dir(sin_smooth, cos_smooth)


def apply_speed_based_smoothing(df, speed_col='wind speed'):
    
    sin, cos = dir_to_sin_cos(df['wind direction'].values)
    sin_series = pd.Series(sin)
    cos_series = pd.Series(cos)

    sin_filled = sin_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')
    cos_filled = cos_series.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')

    speeds = df[speed_col].values
    sin_smooth = np.zeros_like(sin_filled)
    cos_smooth = np.zeros_like(cos_filled)

    for i in range(len(sin_filled)):
        if pd.isna(speeds[i]):
            sigma = 2
        elif speeds[i] < 2:
            sigma = 4
        elif speeds[i] < 5:
            sigma = 2
        else:
            sigma = 1

        window_size = int(sigma * 3)
        start = max(0, i - window_size)
        end = min(len(sin_filled), i + window_size + 1)

        positions = np.arange(start, end)
        weights = np.exp(-0.5 * ((positions - i) / sigma) ** 2)
        if weights.sum() > 0:
            weights = weights / weights.sum()

            sin_smooth[i] = np.sum(sin_filled.iloc[start:end].values * weights)
            cos_smooth[i] = np.sum(cos_filled.iloc[start:end].values * weights)
        else:
            sin_smooth[i] = sin_filled.iloc[i]
            cos_smooth[i] = cos_filled.iloc[i]

    norm = np.sqrt(sin_smooth ** 2 + cos_smooth ** 2)
    norm[norm == 0] = 1
    sin_smooth = sin_smooth / norm
    cos_smooth = cos_smooth / norm

    return sin_cos_to_dir(sin_smooth, cos_smooth)


def apply_rolling_median(df, window=5):
    
    sin, cos = dir_to_sin_cos(df['wind direction'].values)
    sin_series = pd.Series(sin)
    cos_series = pd.Series(cos)

    sin_med = sin_series.rolling(window=window, center=True, min_periods=1).median()
    cos_med = cos_series.rolling(window=window, center=True, min_periods=1).median()

    sin_med = sin_med.fillna(method='bfill').fillna(method='ffill')
    cos_med = cos_med.fillna(method='bfill').fillna(method='ffill')

    norm = np.sqrt(sin_med ** 2 + cos_med ** 2)
    norm[norm == 0] = 1
    sin_med = sin_med / norm
    cos_med = cos_med / norm

    return sin_cos_to_dir(sin_med.values, cos_med.values)






smoothing_methods = {
    'No Smoothing': apply_no_smoothing,
    'Uniform Smoothing (σ=1)': lambda df: apply_uniform_smoothing(df, sigma=1),
    'Uniform Smoothing (σ=3)': lambda df: apply_uniform_smoothing(df, sigma=3),
    'Savitzky-Golay (w=6,p=0)': lambda df: apply_savgol_filter(df, window_length=6, polyorder=0),
    'Savitzky-Golay (w=6,p=1)': lambda df: apply_savgol_filter(df, window_length=6, polyorder=1),
    'Savitzky-Golay (w=6,p=3)': lambda df: apply_savgol_filter(df, window_length=6, polyorder=3),
    'Savitzky-Golay (w=15,p=2)': lambda df: apply_savgol_filter(df, window_length=6, polyorder=2),
    'Savitzky-Golay (w=6,p=4)': lambda df: apply_savgol_filter(df, window_length=6, polyorder=4),
    'Savitzky-Golay (w=6,p=5)': lambda df: apply_savgol_filter(df, window_length=6, polyorder=5),
    'Adaptive Savitzky-Golay': lambda df: apply_adaptive_savgol(df),
    'Adaptive (variability-based)': lambda df: apply_adaptive_smoothing(df),
    'Speed-based': lambda df: apply_speed_based_smoothing(df),
    'Rolling Median (window=5)': lambda df: apply_rolling_median(df, window=5),
    'Rolling Median (window=11)': lambda df: apply_rolling_median(df, window=11),
}


info_loss_results_overall = []
info_loss_results_lowspeed = []

print("Calculating information loss metrics...")
print("----------------------------------------------------------------")


valid_mask = meteo['wind direction'].notna() & meteo['wind speed'].notna()
low_speed_mask = valid_mask & (meteo['wind speed'] < 2)

print(f"Total valid points: {valid_mask.sum()}")
print(f"Low speed points (<2 m/s): {low_speed_mask.sum()}")
print("----------------------------------------------------------------")

for name, method in smoothing_methods.items():
    print(f"Testing: {name}...")

    
    smoothed_dirs = method(meteo)

    
    loss_metrics_all = calculate_information_loss(
        meteo['wind direction'].values,
        smoothed_dirs
    )

    # Calculate low-speed only information loss metrics
    if low_speed_mask.sum() > 0:
        loss_metrics_lowspeed = calculate_information_loss(
            meteo.loc[low_speed_mask, 'wind direction'].values,
            smoothed_dirs[low_speed_mask]
        )
    else:
        loss_metrics_lowspeed = {
            'Angular_RMSE': np.nan,
            'Entropy_ratio': np.nan,
            'Normalized_MI': np.nan,
            'KL_divergence': np.nan,
            'n_valid': 0
        }

    info_loss_results_overall.append({
        'Method': name,
        'Angular_RMSE': loss_metrics_all['Angular_RMSE'],
        'Entropy_Ratio': loss_metrics_all['Entropy_ratio'],
        'Normalized_MI': loss_metrics_all['Normalized_MI'],
        'KL_Divergence': loss_metrics_all['KL_divergence'],
        'n_valid': loss_metrics_all['n_valid']
    })

    info_loss_results_lowspeed.append({
        'Method': name,
        'Angular_RMSE': loss_metrics_lowspeed['Angular_RMSE'],
        'Entropy_Ratio': loss_metrics_lowspeed['Entropy_ratio'],
        'Normalized_MI': loss_metrics_lowspeed['Normalized_MI'],
        'KL_Divergence': loss_metrics_lowspeed['KL_divergence'],
        'n_valid': loss_metrics_lowspeed['n_valid']
    })


loss_df_overall = pd.DataFrame(info_loss_results_overall)
loss_df_lowspeed = pd.DataFrame(info_loss_results_lowspeed)

loss_df_overall = loss_df_overall.sort_values('Normalized_MI', ascending=False)
loss_df_lowspeed = loss_df_lowspeed.sort_values('Normalized_MI', ascending=False)

print("\n================================================================")
print("INFORMATION LOSS ASSESSMENT - OVERALL")
print("================================================================")
print("\nMetrics interpretation:")
print("- Normalized MI: 1.0 = perfect preservation, 0.0 = complete loss")
print("- Entropy Ratio: 1.0 = same distribution, <1.0 = lost variability")
print("- Angular RMSE: lower is better (degrees)")
print("- KL Divergence: lower is better (0 = identical distributions)\n")

print(loss_df_overall.to_string(index=False))

print("\n" + "================================================================")
print("INFORMATION LOSS ASSESSMENT - LOW SPEED ONLY (<2 m/s)")
print("================================================================")
print(f"\nBased on {low_speed_mask.sum()} low speed data points\n")

print(loss_df_lowspeed.to_string(index=False))

