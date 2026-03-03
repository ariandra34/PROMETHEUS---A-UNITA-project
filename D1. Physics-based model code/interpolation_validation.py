import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde


meteo = pd.read_csv('meteo_with_estimation.csv')
meteo['time'] = pd.to_datetime(meteo['time'])


meteo = meteo.sort_values('time')


meteo_names = ['temperature', 'humidity', 'irradiance', 'precipitation',
              'wind speed', 'wind direction', 'sunshine duration']


def circular_mean(angles_degrees):
    """
    This function computes the mean of an array of angles, while taking into account the circularity.
    For example, the mean of 359° and 1° is 0° and not 180°.
    """
    if len(angles_degrees) == 0:
        return np.nan


    angles_rad = np.radians(angles_degrees)


    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))


    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.degrees(mean_angle_rad)


    if mean_angle_deg < 0:
        mean_angle_deg += 360

    return mean_angle_deg


def interpolate_wind_direction_weighted(val1, val2, weight1, weight2):
    """
    Attempt to interpolate wind direction using weights. Lower weight are given for data points when the wind
    speed is low because it leads to higher variability
    """

    rad1 = np.radians(val1)
    rad2 = np.radians(val2)


    sin_sum = weight1 * np.sin(rad1) + weight2 * np.sin(rad2)
    cos_sum = weight1 * np.cos(rad1) + weight2 * np.cos(rad2)


    mean_angle_rad = np.arctan2(sin_sum, cos_sum)
    mean_angle_deg = np.degrees(mean_angle_rad)


    if mean_angle_deg < 0:
        mean_angle_deg += 360

    return mean_angle_deg


def interpolate_wind_direction(prev_val, next_val):

    return circular_mean([prev_val, next_val])


def interpolate_value(prev_val, next_val, variable):

    if variable == 'wind direction':
        return interpolate_wind_direction(prev_val, next_val)
    else:
        # normal interpolation for other variables (equidistant datapoints)
        return (prev_val + next_val) / 2


def compute_interpolation_errors(df, variable, delta_t_minutes=10):



    mask = df[variable].notna()
    valid_df = df[mask].copy().reset_index(drop=True)

    if len(valid_df) < 3:
        return {'single': {'errors': np.array([]), 'timestamps': np.array([]), 'values': np.array([])},
                'double': {'errors': np.array([]), 'timestamps': np.array([]), 'values': np.array([])}}

    results = {
        'single': {'errors': [], 'timestamps': [], 'values': []},
        'double': {'errors': [], 'timestamps': [], 'values': []}
    }


    for i in range(1, len(valid_df) - 1):
        prev_time = valid_df.iloc[i - 1]['time']
        curr_time = valid_df.iloc[i]['time']
        next_time = valid_df.iloc[i + 1]['time']

        prev_val = valid_df.iloc[i - 1][variable]
        curr_val = valid_df.iloc[i][variable]
        next_val = valid_df.iloc[i + 1][variable]

        if pd.notna(prev_val) and pd.notna(curr_val) and pd.notna(next_val):
            time_diff_prev = (curr_time - prev_time).total_seconds() / 60
            time_diff_next = (next_time - curr_time).total_seconds() / 60

            if abs(time_diff_prev - delta_t_minutes) < 1 and abs(time_diff_next - delta_t_minutes) < 1:
                estimate = interpolate_value(prev_val, next_val, variable)
                error = abs(estimate - curr_val)

                results['single']['errors'].append(error)
                results['single']['timestamps'].append(curr_time)
                results['single']['values'].append(curr_val)



    for i in range(1, len(valid_df) - 2):

        first_valid_time = valid_df.iloc[i - 1]['time']
        last_valid_time = valid_df.iloc[i + 2]['time']


        first_valid_val = valid_df.iloc[i - 1][variable]
        last_valid_val = valid_df.iloc[i + 2][variable]


        mid1_time = valid_df.iloc[i]['time']
        mid2_time = valid_df.iloc[i + 1]['time']
        mid1_val = valid_df.iloc[i][variable]
        mid2_val = valid_df.iloc[i + 1][variable]


        if (pd.notna(first_valid_val) and pd.notna(mid1_val) and
                pd.notna(mid2_val) and pd.notna(last_valid_val)):

            time_diff_first = (mid1_time - first_valid_time).total_seconds() / 60
            time_diff_last = (last_valid_time - mid2_time).total_seconds() / 60
            time_diff_mid = (mid2_time - mid1_time).total_seconds() / 60

            if (abs(time_diff_first - delta_t_minutes) < 1 and
                    abs(time_diff_mid - delta_t_minutes) < 1 and
                    abs(time_diff_last - delta_t_minutes) < 1):

                total_time = (last_valid_time - first_valid_time).total_seconds() / 60
                time_to_mid1 = (mid1_time - first_valid_time).total_seconds() / 60
                time_to_mid2 = (mid2_time - first_valid_time).total_seconds() / 60

                if variable == 'wind direction':

                    weight_first_mid1 = 1 - (time_to_mid1 / total_time)
                    weight_last_mid1 = time_to_mid1 / total_time

                    weight_first_mid2 = 1 - (time_to_mid2 / total_time)
                    weight_last_mid2 = time_to_mid2 / total_time

                    estimate_mid1 = interpolate_wind_direction_weighted(
                        first_valid_val, last_valid_val, weight_first_mid1, weight_last_mid1
                    )
                    estimate_mid2 = interpolate_wind_direction_weighted(
                        first_valid_val, last_valid_val, weight_first_mid2, weight_last_mid2
                    )
                else:

                    estimate_mid1 = first_valid_val + (last_valid_val - first_valid_val) * (time_to_mid1 / total_time)
                    estimate_mid2 = first_valid_val + (last_valid_val - first_valid_val) * (time_to_mid2 / total_time)

                error_mid1 = abs(estimate_mid1 - mid1_val)
                error_mid2 = abs(estimate_mid2 - mid2_val)

                results['double']['errors'].extend([error_mid1, error_mid2])
                results['double']['timestamps'].extend([mid1_time, mid2_time])
                results['double']['values'].extend([mid1_val, mid2_val])


    for pattern in ['single', 'double']:
        results[pattern]['errors'] = np.array(results[pattern]['errors'])
        results[pattern]['timestamps'] = np.array(results[pattern]['timestamps'])

        results[pattern]['values'] = np.array(results[pattern]['values'])

    return results


print("================================================================")
print("Interpolation error analysis")
print("================================================================")
print("Analyzing both single missing data points and two consecutive missing data points")
print("================================================================")


results = {}

for var in meteo_names:
    if var in meteo.columns:
        print(f"\n================================================================")
        print(f"Processing {var}...")
        print("================================================================")


        valid_count = meteo[var].notna().sum()
        print(f"  Total valid data points: {valid_count}")


        error_results = compute_interpolation_errors(meteo, var)

        results[var] = {}

        for pattern in ['single', 'double']:
            errors = error_results[pattern]['errors']
            timestamps = error_results[pattern]['timestamps']
            values = error_results[pattern]['values']

            if len(errors) > 0:

                mean_error = np.mean(errors)
                std_error = np.std(errors)
                median_error = np.median(errors)
                p95_error = np.percentile(errors, 95)
                p99_error = np.percentile(errors, 99)
                max_error = np.max(errors)


                data_min = meteo[var].min()
                data_max = meteo[var].max()
                data_range = data_max - data_min


                results[var][pattern] = {
                    'errors': errors,
                    'timestamps': timestamps,
                    'values': values,
                    'n_points': len(errors),
                    'mean_error': mean_error,
                    'std_error': std_error,
                    'median_error': median_error,
                    'p95_error': p95_error,
                    'p99_error': p99_error,
                    'max_error': max_error,
                    'data_range': data_range,
                    'data_min': data_min,
                    'data_max': data_max,
                    'relative_mean_error': (mean_error / data_range * 100) if data_range > 0 else 0,
                    'relative_p95_error': (p95_error / data_range * 100) if data_range > 0 else 0
                }


                pattern_name = "Single missing" if pattern == 'single' else "Two consecutive missing"
                print(f"\n  {pattern_name} data points:")
                print(f"    Number of test points: {len(errors)}")
                print(f"    Mean absolute error: {mean_error:.4f}")
                print(f"    Std absolute error: {std_error:.4f}")
                print(f"    Median absolute error: {median_error:.4f}")
                print(f"    95th percentile error: {p95_error:.4f}")
                print(f"    99th percentile error: {p99_error:.4f}")
                print(f"    Max absolute error: {max_error:.4f}")
                print(f"    Relative mean error: {results[var][pattern]['relative_mean_error']:.2f}% of data range")
            else:
                pattern_name = "Single missing" if pattern == 'single' else "Two consecutive missing"
                print(f"\n  {pattern_name} data points:")
                print(f"    No valid points with 10-minute neighbors found")


