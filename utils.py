import os
import pymongo
from datetime import datetime
from os import listdir
from os.path import isfile, join
import scipy
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


def sliding_window_pd(
        df,
        ws=500,
        overlap=250,
        w_type="hann",
        w_center=True,
        print_stats=False
) -> list:
    """Applies the sliding window algorithm to the DataFrame rows.

    Args:
        df: The DataFrame with all the values that will be inserted to the
            sliding window algorithm.
        ws: The window size in number of samples.
        overlap: The hop length in number of samples.
        w_type: The windowing function.
        w_center: If False, set the window labels as the right edge of the
            window index. If True, set the window labels as the center of the
            window index.
        print_stats: Print statistical inferences from the process. Defaults
            to False.

    Returns:
        A list of DataFrames each one corresponding to a produced window.
    """
    counter = 0
    windows_list = list()
    # min_periods: Minimum number of observations in window required to have
    # a value;
    # For a window that is specified by an integer, min_periods will default
    # to the size of the window.
    for window in df.rolling(window=ws, step=overlap, min_periods=ws,
                             win_type=w_type, center=w_center):
        sensor_columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        if window[sensor_columns].count().min() >= ws:
            if print_stats:
                print("Print Window:", counter)
                print("Number of samples:", window[sensor_columns].count().min())
            windows_list.append(window)
        counter += 1
    if print_stats:
        print("List number of window instances:", len(windows_list))

    return windows_list


def apply_filter(
        arr,
        order=5,
        wn=0.1,
        filter_type="lowpass"
) -> np.ndarray:
    """Applies filter to the multi-axis signal.

    Args:
        arr: The initial NumPy signal array values.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’,
            ‘bandstop’}

    Returns:
        NumPy Array with the filtered signal.
    """
    fbd_filter = scipy.signal.butter(N=order, Wn=wn, btype=filter_type,
                                     output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos=fbd_filter, x=arr, padlen=0)

    return filtered_signal


def filter_instances(
        instances_list,
        order,
        wn,
        filter_type
) -> list:
    """Applies filter to a list of windows (each window is a DataFrame).

    Args:
        instances_list: List of DataFrames.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’,
            ‘bandstop’}

    Returns:

    """
    filtered_instances_list = list()
    for item in instances_list:
        filtered_instance = item.copy()
        numeric_columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        filtered_instance[numeric_columns] = filtered_instance[numeric_columns].apply(
            apply_filter,
            args=(order, wn, filter_type)
        )
        filtered_instances_list.append(filtered_instance)
    print("Number of filtered instances in the list:",
          len(filtered_instances_list)
          )

    return filtered_instances_list


def flatten_instances_df(instances_list: list) -> pd.DataFrame:
    """Flattens each instance and create a DataFrame with the whole flattened
        instances.

    Args:
        instances_list: The list of DataFrames to be flattened

    Returns:
        A DataFrame that includes the whole flattened DataFrames
    """
    flattened_instances_list = list()
    for item in instances_list:
        instance = item.to_numpy().flatten()
        flattened_instances_list.append(instance)
    df = pd.DataFrame(flattened_instances_list)

    return df


def df_rebase(
        df: pd.DataFrame,
        target_list: list,
        ref_list: list
) -> pd.DataFrame:
    """Changes the order and name of DataFrame columns to the project's needs
        for readability.

    Args:
        df: The pandas DataFrame.
        order_list: List object that contains the proper order of the default
             column names.
        ref_list: List object that contains the renaming list based
            on the project needs.

    Returns:
        A DataFrame with the new columns order and names.
    """
    print("Initial columns:", list(df.columns))

    if are_lists_equal(list(df.columns), ref_list):
        pass

    else:
        if len(target_list) == len(ref_list): 
            # keep and re-order only the necessary columns of the initial DataFrame
            df = df[target_list]
            rename_dict = dict(zip(target_list, ref_list))
            df = df.rename(columns=rename_dict)  # rename the columns
        else:
            print("The length of the target list and the reference list is not equal.")

    print("Processed columns:", list(df.columns))

    return df

def encode_labels(instances_list, le=None) -> np.ndarray:
    """Encodes target labels.

    Args:
        instances_list: List of instances to be encoded.
        le: Optional fitted LabelEncoder to reuse. If None, a new encoder is
            created and fitted to instances_list.

    Returns:
        A tuple of (encoded_array, label_encoder).
    """
    if le is None:
        le = preprocessing.LabelEncoder()
        le.fit(instances_list)

    instances_arr = le.transform(instances_list)
    return instances_arr, le

def rename_df_column_values(
    np_array: np.ndarray, 
    y: list, 
    columns_names: tuple = ("acc_x", "acc_y", "acc_z")
):
    """Creates a DataFrame with a "y" label column and replaces the values of the y with the index
    of the unique values of y.

    Args:
        np_array: 2D NumPy array.
        y: List with the y labels
        columns_names: List with the DF columns names.

    Returns:
        DataFrame with the multi-axes values and the target labels column.
    """
    arr_y = np.array(y)  # list to numpy array
    unique_values_list = np.unique(arr_y)  # unique list of values

    df = pd.DataFrame(np_array, columns=columns_names)
    df["y"] = y

    # replace the row item value in the y column of the df, with its index in the unique list
    for idx, x in enumerate(unique_values_list):
        df["y"] = np.where(df["y"] == x, idx, df["y"])

    return df


def are_lists_equal(
    list1: list, 
    list2: list
) -> bool:
    return set(list1) == set(list2)


def list_files_in_folder(folder_path) -> list:
    """Returns a list of all CSV files within the specified folder.

    Args:
        folder_path (str): The directory path to search for files.

    Returns:
        list: A list containing the filenames (strings) of all files
              in the directory that end with the '.csv' extension.
    """
    files_list = list()
    for f in listdir(folder_path):
        if isfile(join(folder_path, f)):
            if f.endswith(".csv"):
                files_list.append(f)

    return files_list

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Checks for missing values in each column of the DataFrame and prints a summary.
    If a 'user' column is present, it also provides a breakdown per user.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A pandas Series containing the counts of missing values per column.
    """
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    print("=" * 45)
    print("OVERALL MISSING VALUES SUMMARY")
    print("=" * 45)
    
    if total_missing == 0:
        print("No missing values found in the entire dataset.\n")
    else:
        for col, count in missing_counts.items():
            if count > 0:
                percentage = (count / len(df)) * 100
                print(f"{col:<15}: {count:>6} missing ({percentage:>6.2f}%)")
        print(f"\nTotal missing values: {total_missing}\n")

    if "user" in df.columns:
        print("=" * 45)
        print("MISSING VALUES BREAKDOWN PER USER")
        print("=" * 45)
        
        missing_by_user = df.isnull().groupby(df['user']).sum()
        cols_with_missing = missing_counts[missing_counts > 0].index
        
        if len(cols_with_missing) == 0:
            print("All users have complete data (no missing values).")
        else:
            missing_by_user = missing_by_user[cols_with_missing]
            for user_id, row in missing_by_user.iterrows():
                user_total = row.sum()
                user_len = (df['user'] == user_id).sum()
                print(f"\nUser {user_id} (Total instances: {user_len})")
                print("-" * 35)
                if user_total == 0:
                    print("  No missing values.")
                else:
                    for col in cols_with_missing:
                        count = row[col]
                        if count > 0:
                            percentage = (count / user_len) * 100
                            print(f"  {col:<13}: {count:>6} missing ({percentage:>6.2f}%)")
                    print(f"  User Total   : {user_total} missing values")
        print("=" * 45 + "\n")

    return missing_counts


def parse_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) < 8:
        raise ValueError(f"Unexpected filename format: {filename}")
    return {
        "gesture_id": "_".join(parts[:-7]),
        "hand": int(parts[-7]),
        "sr": int(parts[-6]),
        "sensor": parts[-5],
        "primary": int(parts[-4]),
        "spontaneous": int(parts[-3]),
        "user": parts[-2],
        "unique_id_hash": parts[-1],
    }


COLUMN_ALIASES = {
    "acc_x": ["acc_x", "x-axis (g)"],
    "acc_y": ["acc_y", "y-axis (g)"],
    "acc_z": ["acc_z", "z-axis (g)"],
    "gyr_x": ["gyr_x", "x-axis (deg/s)"],
    "gyr_y": ["gyr_y", "y-axis (deg/s)"],
    "gyr_z": ["gyr_z", "z-axis (deg/s)"],
}


def resolve_columns(columns):
    cols_lower = {c.lower(): c for c in columns}
    resolved = {}
    for key, aliases in COLUMN_ALIASES.items():
        match = None
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in cols_lower:
                match = cols_lower[alias_lower]
                break
        if match is None:
            return None
        resolved[key] = match
    return resolved


def import_sensor_csvs_to_mongodb(data_path, collection):
    """Parses and ingests session CSV files from data_path into the MongoDB collection.
    
    Args:
        data_path: Path to the root directory containing gesture folders.
        collection: The pymongo Collection object to insert documents into.
        
    Returns:
        tuple: (inserted_count, skipped_files_list)
    """
    classes_folders_list = [
        f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))
    ]
    
    documents = []
    skipped = []

    for gesture in classes_folders_list:
        gesture_dir = os.path.join(data_path, gesture)
        for filename in os.listdir(gesture_dir):
            if not filename.lower().endswith(".csv"):
                continue
            file_path = os.path.join(gesture_dir, filename)
            try:
                meta = parse_filename(filename)
            except ValueError as exc:
                skipped.append((filename, str(exc)))
                continue

            df = pd.read_csv(file_path)
            resolved = resolve_columns(df.columns)
            if resolved is None:
                skipped.append((filename, "Missing expected columns"))
                continue

            data = {
                "acc_x": df[resolved["acc_x"]].tolist(),
                "acc_y": df[resolved["acc_y"]].tolist(),
                "acc_z": df[resolved["acc_z"]].tolist(),
                "gyr_x": df[resolved["gyr_x"]].tolist(),
                "gyr_y": df[resolved["gyr_y"]].tolist(),
                "gyr_z": df[resolved["gyr_z"]].tolist(),
            }

            doc = {
                "_id": meta["unique_id_hash"],
                "data": data,
                "gesture_id": meta["gesture_id"],
                "hand": meta["hand"],
                "sr": meta["sr"],
                "sensor": meta["sensor"],
                "primary": meta["primary"],
                "spontaneous": meta["spontaneous"],
                "user": meta["user"],
                "datetime": datetime.now(),
            }
            documents.append(doc)

    inserted_count = 0
    if documents:
        try:
            result = collection.insert_many(documents)
            if result is not None:
                inserted_count = len(result.inserted_ids)
        except pymongo.errors.BulkWriteError as exc:
            write_errors = exc.details.get("writeErrors", [])[0].get("errmsg", "Unknown error")
            print(f"Error inserting documents: {write_errors}")
            
    return inserted_count, skipped
