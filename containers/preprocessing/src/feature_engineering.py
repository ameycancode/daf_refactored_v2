import os
import pandas as pd
from datetime import datetime

import boto3
from io import StringIO
import logging
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv_from_s3(s3_bucket, s3_key):
    """Loads a CSV file from S3 into a Pandas DataFrame."""
    s3_client = boto3.client("s3")
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        df = pd.read_csv(response["Body"])
        logger.info(f"Successfully loaded data from s3://{s3_bucket}/{s3_key}")
        return df
    except Exception as e:
        logger.error(f"Error loading {s3_key} from S3: {str(e)}")
        raise

def save_csv_to_s3(df, s3_bucket, s3_key):
    """Saves a Pandas DataFrame as a CSV file to an S3 bucket."""
    s3_client = boto3.client("s3")
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())
        logger.info(f"Successfully saved data to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Error saving {s3_key} to S3: {str(e)}")
        raise
       

def generate_data_profile(df):
    """
    Generates and assigns DataFrames for each profile based on specific starting dates.

    Args:
        df (DataFrame): The merged DataFrame containing profile and time data.

    Returns:
        dict: A dictionary containing individual DataFrames for each profile.
    """
    #replace ESQMD for ASDMD
    # Replace null values in Count and Load_Per_Meter with values from Count_I and Load_Per_Meter_I
    df['Count'] = df['Count'].fillna(df['Count_I'])
    df['Load'] = df['Load'].fillna(df['Load_I'])


    # Define individual DataFrames based on profiles and their starting dates
    profile_dfs = {
        'df_RNN': df[(df['Profile'] == 'RES_Non_NEM') ],
        'df_RN': df[(df['Profile'] == 'RES_NEM')],
        'df_M': df[(df['Profile'] == 'MEDCI') ],
        'df_S': df[(df['Profile'] == 'SMLCOM') ],
        'df_AGR': df[(df['Profile'] == 'AGR') ],
        'df_L': df[(df['Profile'] == 'LIGHT') ],
        'df_A6': df[(df['Profile'] == 'A6') ]
    }
   
    # # Access individual DataFrames by their names
    # df_RNN = profile_dfs['df_RNN']
    # df_RN = profile_dfs['df_RN']
    # df_M = profile_dfs['df_M']
    # df_S = profile_dfs['df_S']
    # df_AGR = profile_dfs['df_AGR']
    # df_L = profile_dfs['df_L']
    # df_A6 = profile_dfs['df_A6']
   
    return profile_dfs


def create_specific_day_lag_features(df, days_ago_columns={14: 'Load_I', 70: 'Load'}):
    """
    Creates lag features for specific days in the past at the same hour.

    Args:
        df (DataFrame): Input DataFrame with hourly data.
        days_ago_columns (dict): Dictionary specifying columns and days ago for lag features.

    Returns:
        DataFrame: DataFrame with specific day lag features.
    """
    df = df.copy()
    for days_ago, column in days_ago_columns.items():
        shift_hours = days_ago * 24
        # Create a single column with the value from the specified day ago at the same hour
        df[f'{column}_lag_{days_ago}_days'] = df[column].shift(shift_hours)

    return df


def save_lagged_profiles(profile_dfs, s3_bucket, s3_output_prefix="xgboost/processed/lagged_profiles"):
    """
    Applies lag features to each profile-specific DataFrame, filters by a specific date,
    and saves each profile to S3.

    Args:
        profile_dfs (dict): Dictionary of DataFrames per profile.
        s3_bucket (str): Name of the S3 bucket.
        s3_output_prefix (str): S3 prefix where lagged profiles will be saved.

    Returns:
        dict: Dictionary of filtered and lagged DataFrames.
    """
   
    # Define the filtering start dates for each profile
    date_filters = {
        'df_RNN': '2022-03-01',
        'df_RN': '2022-03-01',
        'df_M': '2021-07-01',
        'df_S': '2021-07-01',
        'df_AGR': '2023-05-01',
        'df_L': '2021-07-01',
        'df_A6': '2021-07-10'
    }


    # Set local timezone (US Pacific Time)
    local_tz = pytz.timezone("America/Los_Angeles")
    current_date = datetime.now(local_tz).strftime("%Y%m%d")
   
    lagged_dfs = {}
    for profile, df in profile_dfs.items():
        # Apply lag features
        lagged_df = create_specific_day_lag_features(df)

        # Filter by the specified start date
        start_date = pd.to_datetime(date_filters[profile])
        filtered_df = lagged_df[lagged_df['Time'] >= start_date]

        # Save to S3
        s3_key = f"{s3_output_prefix}/{profile}_lagged_{current_date}.csv"
        save_csv_to_s3(filtered_df, s3_bucket, s3_key)

        lagged_dfs[profile] = filtered_df

    return lagged_dfs



def replace_count_i(lagged_dfs, s3_bucket, s3_output_prefix="xgboost/processed/count_replaced"):
    """
    Replaces `Count_I` with the mean of the last available day and saves results to S3.

    Args:
        lagged_dfs (dict): Dictionary of lagged DataFrames.
        s3_bucket (str): Name of the S3 bucket.
        s3_output_prefix (str): S3 prefix for saving results.

    Returns:
        dict: Updated dictionary with `Count_I` replaced.
    """

    # Get current date
    # Set local timezone (US Pacific Time)
    local_tz = pytz.timezone("America/Los_Angeles")
    current_date = datetime.now(local_tz).strftime("%Y%m%d")
   
    s3_output_prefix = s3_output_prefix.rstrip("/")  # Remove trailing slash
    updated_dfs = {}

    for profile, df in lagged_dfs.items():
        # Find the last day with non-null `Count_I`
        last_non_null_date = df[df['Count_I'].notnull()]['Time'].dt.date.max()

        if pd.isna(last_non_null_date):
            raise ValueError(f"No non-null `Count_I` values found in {profile}. Cannot perform replacement.")

        # Filter data for the last day with non-null `Count_I`
        last_day_data = df[df['Time'].dt.date == last_non_null_date]

        # Calculate the mean of `Count_I` for the last day
        count_mean = last_day_data['Count_I'].mean()

        # Replace `Count_I` in the entire DataFrame with the calculated mean
        df['Count_I'] = df['Count_I'].fillna(count_mean)

        # Save to S3
        s3_key = f"{s3_output_prefix}/{profile}_count_replaced_{current_date}.csv"
        save_csv_to_s3(df, s3_bucket, s3_key)

        updated_dfs[profile] = df

    return updated_dfs



def add_radiation_to_df_RN(updated_dfs, s3_bucket, s3_radiation_key, s3_output_prefix="xgboost/processed/radiation"):
    """
    Adds radiation data from S3 to the df_RN profile and saves the updated DataFrame.

    Args:
        updated_dfs (dict): Dictionary containing profile DataFrames.
        s3_bucket (str): Name of the S3 bucket.
        s3_radiation_key (str): S3 key for the radiation data CSV.
        s3_output_prefix (str): S3 prefix where the updated df_RN will be saved.

    Returns:
        dict: Updated dictionary with modified df_RN.
    """


    # Get current date
    # Set local timezone (US Pacific Time)
    local_tz = pytz.timezone("America/Los_Angeles")
    current_date = datetime.now(local_tz).strftime("%Y%m%d")

   
    # Extract the df_RN profile
    df_RN = updated_dfs['df_RN']

    # Load and preprocess the radiation data
    df_meteo = load_csv_from_s3(s3_bucket, s3_radiation_key)

    df_meteo.columns = df_meteo.columns.str.replace(r"\s*\(.*?\)", "", regex=True)
    df_meteo['time'] = pd.to_datetime(df_meteo['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_meteo['time'] = df_meteo['time'].str.replace("T", " ", regex=False)
    df_meteo['time'] = pd.to_datetime(df_meteo['time'])
    df_meteo = df_meteo[['time', 'shortwave_radiation']]

    # Merge the radiation data with df_RN
    updated_df_RN = df_RN.merge(df_meteo, left_on='Time', right_on='time', how='left')

    # Drop the duplicate 'time' column after the merge
    updated_df_RN.drop(columns=['time'], inplace=True)

    # Hardcoded output columns
    output_columns = [
        'Time', 'Profile', 'Load', 'Count', 'Load_I', 'Count_I', 'Year',
        'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday',
        'TradeDate', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days',
        'shortwave_radiation'
    ]

    # Filter the DataFrame to keep only the predefined columns
    updated_df_RN = updated_df_RN[output_columns]

    # Save to S3
    s3_key = f"{s3_output_prefix}/df_RN_with_radiation_{current_date}.csv"
    save_csv_to_s3(updated_df_RN, s3_bucket, s3_key)

    updated_dfs['df_RN'] = updated_df_RN
    return updated_dfs
