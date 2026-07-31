import numpy as np
import pandas as pd
from numpy import random

random.seed(42)


def check_data(data_by_id):

    for file_id, data in data_by_id.items():

        # 1. Missing values
        if data.isna().any().any():
            print(f"  WARNING: Missing values in {file_id}")

        # 2. Infinite values
        numeric_data = data.select_dtypes(include=np.number)

        if np.isinf(numeric_data.to_numpy()).any():
            print(f"  WARNING: Infinite values in {file_id}")

        # 3. Invalid characters
        # Non-numeric values in numeric columns become NaN
        if len(numeric_data.columns) != len(data.columns):
            print(f"  WARNING: Non-numeric/invalid characters in {file_id}")

        # 4. Signal length
        expected_length = 163840

        if len(data) != expected_length:
            print(f"  WARNING: Expected {expected_length} samples, "
                  f"found {len(data)} in {file_id}")

        # 5. All-zero signals
        signal_columns = ["X", "Y", "Z", "Sound"]

        for column in signal_columns:
            if (data[column] == 0).all():
                print(f"  WARNING: {column} is all zeros in {file_id}")

        expected_dt = 1/25600
        dt = data["Time"].diff().dropna()
        valid_dt = dt[dt > 0]

        if not np.isclose(valid_dt.mean(), expected_dt, rtol=0.01):
            print(f"WARNING: Unexpected sampling frequency in {file_id}")

        # Check for unusually large positive gaps
        if (valid_dt > expected_dt * 1.1).any():
            print(f"WARNING: Possible gap in recording in {file_id}")
        
    print("All checks complete")#


def segment_data(data_by_id, window_duration, window_overlap):

    windows_by_id = {}

    for file_id, data in data_by_id.items():

        window_size = int(window_duration * 25600)
        step = int(window_size * (1 - window_overlap))

        windows = []

        for start in range(0, len(data)-window_size + 1,step):

            end = start + window_size

            window = data.iloc[start:end].copy()

            windows.append(window)

        windows_by_id[file_id] = windows

    return windows_by_id



def remove_dc(windows_by_id):

    signal_columns = ["X", "Y", "Z", "Sound"]
    dc_means_by_id = {}

    for file_id, windows in windows_by_id.items():

        #list to store window means
        dc_means = []

        for window in windows:
            window_means = {}

            for column in signal_columns:

                mean = window[column].mean()

                #store column mean for later
                window_means[column] = mean

                # Remove DC component
                window[column] = window[column] - mean

            dc_means.append(window_means)

        dc_means_by_id[file_id] = dc_means

    return windows_by_id, dc_means_by_id













def extract_time_features(windows_by_id, dc_means_by_id):

    epsilon = 1e-12

    signal_columns = ["X", "Y", "Z", "Sound"]

    features_by_id = {}

    for file_id, windows in windows_by_id.items():

        feature_list = []

        for window_num, window in enumerate(windows):

            window_features = {}

            # Save the original means
            for column in signal_columns:
                window_features[f"{column}_mean"] = dc_means_by_id[file_id][window_num][column]

            # Calculate remaining features
            for column in signal_columns:

                signal = window[column]

                std = signal.std()
                rms = np.sqrt(np.mean(signal**2))
                maximum = signal.max()
                minimum = signal.min()
                pk_pk = maximum - minimum

                skewness = (((signal - signal.mean()) / (signal.std() + epsilon)**3).mean())
                kurtosis = (((signal - signal.mean()) / (signal.std() + epsilon)**4).mean())

                shape_factor = rms / (np.mean(np.abs(signal)) + epsilon)
                crest_factor = np.max(np.abs(signal)) / (rms + epsilon)
                impulse_factor = (
                    np.max(np.abs(signal))
                    / (np.mean(np.abs(signal)) + epsilon)
                )

                window_features[f"{column}_std"] = std
                window_features[f"{column}_rms"] = rms
                window_features[f"{column}_max"] = maximum
                window_features[f"{column}_min"] = minimum
                window_features[f"{column}_pk_pk"] = pk_pk
                window_features[f"{column}_skewness"] = skewness
                window_features[f"{column}_kurtosis"] = kurtosis
                window_features[f"{column}_shape_factor"] = shape_factor
                window_features[f"{column}_crest_factor"] = crest_factor
                window_features[f"{column}_impulse_factor"] = impulse_factor

            feature_list.append(window_features)

        features_by_id[file_id] = feature_list

    rows = []

    for file_id, windows in features_by_id.items():
        for window in windows:
            row = window.copy()
            row["file_id"] = file_id
            rows.append(row)

    features_df = pd.DataFrame(rows)

    column_order = [
    "file_id",

    "X_mean",
    "X_std",
    "X_rms",
    "X_max",
    "X_min",
    "X_pk_pk",
    "X_skewness",
    "X_kurtosis",
    "X_shape_factor",
    "X_crest_factor",
    "X_impulse_factor",

    "Y_mean",
    "Y_std",
    "Y_rms",
    "Y_max",
    "Y_min",
    "Y_pk_pk",
    "Y_skewness",
    "Y_kurtosis",   
    "Y_shape_factor",
    "Y_crest_factor",
    "Y_impulse_factor",

    "Z_mean",
    "Z_std",
    "Z_rms",
    "Z_max",
    "Z_min",
    "Z_pk_pk",
    "Z_skewness",
    "Z_kurtosis",
    "Z_shape_factor",
    "Z_crest_factor",
    "Z_impulse_factor",

    "Sound_mean",
    "Sound_std",
    "Sound_rms",
    "Sound_max",
    "Sound_min",
    "Sound_pk_pk",
    "Sound_skewness",
    "Sound_kurtosis",
    "Sound_shape_factor",
    "Sound_crest_factor",
    "Sound_impulse_factor",
    ]

    features_df = features_df[column_order]

    numeric_df = features_df.select_dtypes(include=np.number)

    if np.isinf(numeric_df).any().any():
        print("WARNING: Infinite values found")

    if numeric_df.isna().any().any():
        print("WARNING: NaN values found")

    return features_df





def extract_freq_features(windows_by_id,sample_freq):

    epsilon = 1e-12

    signal_columns = ["X", "Y", "Z", "Sound"]

    features_by_id = {}

    for file_id, windows in windows_by_id.items():

        feature_list = []

        for window_num, window in enumerate(windows):

            window_features = {}

            for column in signal_columns:

                signal = window[column]

                #Apply fft
                fft_values = np.fft.rfft(signal)

                magnitude = np.abs(fft_values)

                power = magnitude**2

                frequencies = np.fft.rfftfreq(
                    len(signal),
                    d=1/sample_freq
                )

                dominant_frequency = frequencies[np.argmax(magnitude)]

                spectral_centroid = np.sum(frequencies*power)/(np.sum(power) + epsilon)

                spectral_bandwidth = np.sqrt(
                    np.sum(power*(frequencies-spectral_centroid)**2)
                    /(np.sum(power) + epsilon)
                )

                probability = power / (np.sum(power) + epsilon)

                spectral_entropy = -np.sum(
                    probability * np.log2(probability + 1e-12)
                )

                spectral_energy = np.sum(power)


                window_features[f"{column}_dominant_frequency"] = dominant_frequency
                window_features[f"{column}_spectral_centroid"] = spectral_centroid
                window_features[f"{column}_spectral_bandwidth"] = spectral_bandwidth
                window_features[f"{column}_spectral_entropy"] = spectral_entropy
                window_features[f"{column}_spectral_energy"] = spectral_energy

                #create frequency bands
                band_edges = np.linspace(
                                0,
                                sample_freq/2,
                                9
                                )


                for i in range(8):
                    band_power = 0 

                    for k in range(len(power)):

                        if band_edges[i] <= frequencies[k] < band_edges[i+1]:

                            band_power += power[k]

                    band_ratio = band_power/(spectral_energy + epsilon)

                    window_features[f"{column}_band_{i+1}_energy_ratio"] = band_ratio


            feature_list.append(window_features)

        features_by_id[file_id] = feature_list

    rows = []

    for file_id, windows in features_by_id.items():
        for window in windows:
            row = window.copy()
            row["file_id"] = file_id
            rows.append(row)

    features_df = pd.DataFrame(rows)

    numeric_df = features_df.select_dtypes(include=np.number)

    if np.isinf(numeric_df).any().any():
        print("WARNING: Infinite values found")

    if numeric_df.isna().any().any():
        print("WARNING: NaN values found")

    return(features_df)






def get_vib_sound(time_features_df,freq_features_df):

    time_vibration = time_features_df[
    ["file_id"] +
    [col for col in time_features_df.columns
     if col.startswith(("X_", "Y_", "Z_"))]
    ]
    

    freq_vibration = freq_features_df[
    [col for col in freq_features_df.columns
     if col.startswith(("X_", "Y_", "Z_"))]
    ]

    vibration_df = pd.concat(
    [time_vibration, freq_vibration],
    axis=1
    )



    time_sound = time_features_df[
    ["file_id"] +
    [col for col in time_features_df.columns
     if col.startswith("Sound_")
    ]
    ]
    
    freq_sound = freq_features_df[
    [col for col in freq_features_df.columns
     if col.startswith("Sound_")
    ]
    ]
    
    sound_df = pd.concat(
    [time_sound, freq_sound],
    axis=1
    )

    return(vibration_df, sound_df)




def split_label(df, column):

    df[["Health label", "Operating frequency"]] = (
        df[column].str.split("_", expand=True)
    )


    return df




def preprocess(data_by_id):

    check_data(data_by_id=data_by_id)


    windows_by_id = segment_data(
                    data_by_id=data_by_id,
                    window_overlap=0.5,
                    window_duration=0.5
                    )

    windows_by_id, dc_means_by_id = remove_dc(windows_by_id=windows_by_id)


    time_df = extract_time_features(windows_by_id=windows_by_id, dc_means_by_id=dc_means_by_id)

    freq_df = extract_freq_features(windows_by_id=windows_by_id, sample_freq=25600)

    vibration_df, sound_df = get_vib_sound(time_features_df=time_df, freq_features_df=freq_df)

    vibration_df = split_label(vibration_df, "file_id")

    sound_df = split_label(sound_df, "file_id")

    vibration_df.to_csv("data/vibration_features.csv", index=False)
    sound_df.to_csv("data/sound_features.csv", index=False)

    print("Data saved to csv")







    
    










