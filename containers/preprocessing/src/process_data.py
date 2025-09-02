import boto3
from io import StringIO
import logging


import os
import pandas as pd
import numpy as np
import time
import plotly.express as px
from datetime import datetime

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


def process_load_data(s3_bucket, s3_key, save_csv=True, s3_output_prefix="xgboost/processed/hourly_data"):
    """
    Reads a CSV file from S3, processes the data, and saves the output back to S3.
    
    Args:
        s3_bucket (str): Name of the S3 bucket.
        s3_key (str): S3 key of the input CSV file.
        save_csv (bool): If True, saves the processed data to S3.
        s3_output_prefix (str): S3 prefix where output files will be saved.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    
    # Record start time
    start_time = time.time()

    # Load data from S3
    df = load_csv_from_s3(s3_bucket, s3_key)

    # Step 1: Read the CSV file and select relevant columns
    df['TradeDateTime'] = pd.to_datetime(df['TradeDate'] + ' ' + df['TradeTime'], format='%Y-%m-%d %H:%M:%S')
    df['RateGroup'] = df['RateGroup'].astype(str)
    df['NEM'] = df['RateGroup'].apply(lambda x: 'NEM' if x.startswith(('NEM', 'SBP')) else 'Non_NEM')
    df['Profile'] = df.apply(lambda row: row['LoadProfile'] + '_' + row['NEM'] if row['LoadProfile'] == 'RES' else row['LoadProfile'], axis=1)
    df = df[['TradeDateTime', 'TradeDate', 'TradeTime', 'Profile', 'LossAdjustedLoad', 'MeterCount', 'Submission']].copy()
    
    # Step 2: Filter out 'Final' and 'Initial' submissions
    df_final = df[df['Submission'] == 'Final']
    df_initial = df[df['Submission'] == 'Initial']
    
    # Step 3: Group by hourly (sum of load, sum of meter count)
    df_hour_final = df_final.groupby(['TradeDateTime', 'Profile']).agg(
        LoadHour=('LossAdjustedLoad', 'sum'),
        Count=('MeterCount', 'sum')
    ).reset_index()
    
    df_hour_initial = df_initial.groupby(['TradeDateTime', 'Profile']).agg(
        LoadHour=('LossAdjustedLoad', 'sum'),
        Count=('MeterCount', 'sum')
    ).reset_index()
    
    # Step 4: Calculate Load_Per_Meter for final and initial
    df_hour_final['Load_Per_Meter'] = df_hour_final['LoadHour'] / df_hour_final['Count']
    df_hour_initial['Load_Per_Meter'] = df_hour_initial['LoadHour'] / df_hour_initial['Count']
    
    # Step 5: Rename initial aggregated results
    df_hour_initial.rename(columns={
        'LoadHour': 'LoadHour_I',
        'Count': 'Count_I',
        'Load_Per_Meter': 'Load_Per_Meter_I'
    }, inplace=True)
    
    # Step 6: Merge final and initial data
    df_merged = pd.merge(df_hour_final, df_hour_initial, on=['TradeDateTime', 'Profile'], how='right')
    df = df_merged[['TradeDateTime', 'Profile', 'Count', 'Load_Per_Meter', 'Count_I', 'Load_Per_Meter_I']].copy()
    
    # Step 7: Extend the dataset by adding 31 (can be adjusted) days of hourly data
    max_date = df['TradeDateTime'].max()
    extended_dates = pd.date_range(start=max_date + pd.Timedelta(hours=1), periods=40 * 24, freq='h')
    profiles = df['Profile'].unique()
    extended_df = pd.DataFrame({'TradeDateTime': extended_dates}).merge(
        pd.DataFrame(profiles, columns=['Profile']), how='cross'
    )

    # Concatenate the original and extended datasets
    df = pd.concat([df, extended_df], ignore_index=True)
    
    # Step 8: Add date-related features for the entire dataset
    df['Year'] = df['TradeDateTime'].dt.year
    df['Month'] = df['TradeDateTime'].dt.month
    df['Day'] = df['TradeDateTime'].dt.day
    df['Hour'] = df['TradeDateTime'].dt.hour
    df['Weekday'] = df['TradeDateTime'].dt.day_name()
    df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [1, 2, 3, 4, 5, 11, 12] else 'Summer')
    
    holidays = [
        '2021-01-01', '2021-02-15', '2021-05-31', '2021-07-05', '2021-09-06', '2021-11-11', '2021-11-25', '2021-12-25',
        '2022-01-01', '2022-02-21', '2022-05-30', '2022-07-04', '2022-09-05', '2022-11-11', '2022-11-24', '2022-12-26',
        '2023-01-02', '2023-02-20', '2023-05-29', '2023-07-04', '2023-09-04', '2023-11-11', '2023-11-23', '2023-12-25',
        '2024-01-01', '2024-02-19', '2024-05-27', '2024-07-04', '2024-09-02', '2024-11-11', '2024-11-28', '2024-12-25',
        '2025-01-01', '2025-02-17', '2025-05-26', '2025-07-04', '2025-09-01', '2025-11-11', '2025-11-27', '2025-12-25',
    ]
    
    df['TradeDate'] = df['TradeDateTime'].dt.date.astype(str)
    df['Holiday'] = df['TradeDate'].isin(holidays).astype(int)
    df['Workday'] = df.apply(lambda x: 0 if (x['Holiday'] == 1 or x['Weekday'] in ['Saturday', 'Sunday']) else 1, axis=1)

    #Save processed data to S3
    if save_csv:
        # Set local timezone (US Pacific Time)
        local_tz = pytz.timezone("America/Los_Angeles") 
        current_date = datetime.now(local_tz).strftime("%Y%m%d")
        
        s3_output_key = f"{s3_output_prefix}/Hourly_Load_Data_{current_date}.csv"
        #Save to S3
        save_csv_to_s3(df, s3_bucket, s3_output_key)

    # log processing time
    end_time = time.time()
    logger.info(f"Processing took {(end_time - start_time) / 60:.2f} minutes")

    return df


def process_temperature_data(s3_bucket, s3_key, save_csv=True, s3_output_prefix="xgboost/processed/hourly_data", plot_data=False):
    """
    Reads a CSV file from S3, processes the temperature data, and saves the output back to S3.

    Args:
        s3_bucket (str): Name of the S3 bucket.
        s3_key (str): S3 key of the input temperature CSV file.
        save_csv (bool): If True, saves the processed data to S3.
        s3_output_prefix (str): S3 prefix where output files will be saved.
        plot_data (bool): If True, displays plots for original and interpolated temperature data.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    
    # Start timing the process
    start_time = time.time()
    
    # Step 1: Read the CSV file and select relevant columns
    # Load data from S3
    df_t = load_csv_from_s3(s3_bucket, s3_key)
    df_t = df_t[['DATE', 'HourlyDryBulbTemperature']]
    
    # Step 2: Rename 'HourlyDryBulbTemperature' to 'Temperature'
    df_t = df_t.rename(columns={'HourlyDryBulbTemperature': 'Temperature'})
    
    # Step 3: Convert 'DATE' to datetime format and extract 'TradeDate' and 'Hour'
    df_t['DATE'] = pd.to_datetime(df_t['DATE'])
    df_t['TradeDate'] = df_t['DATE'].dt.date.astype(str)
    df_t['Hour'] = df_t['DATE'].dt.hour
    
    # Step 4: Clean the 'Temperature' column by removing any 's' characters and converting to numeric
    df_t['Temperature'] = df_t['Temperature'].astype(str).str.replace('s', '', regex=False)
    df_t['Temperature'] = pd.to_numeric(df_t['Temperature'], errors='coerce')
    
    # Step 5: Aggregate by TradeDate and Hour to ensure unique combinations
    df_t = df_t.groupby(['TradeDate', 'Hour'])['Temperature'].mean().reset_index()
    
    # Step 6: Create a full range of hours for each TradeDate
    full_range = pd.MultiIndex.from_product([df_t['TradeDate'].unique(), range(24)], names=['TradeDate', 'Hour'])
    df_t = df_t.set_index(['TradeDate', 'Hour']).reindex(full_range).reset_index()
    
    # Step 7: Interpolate missing values for the Temperature column
    df_t['Temperature_Fill'] = df_t['Temperature'].interpolate(method='linear')
    
    # Step 8: Create TradeDateTime for each hour
    df_t['TradeDateTime'] = pd.to_datetime(df_t['TradeDate'] + ' ' + df_t['Hour'].astype(str).str.zfill(2) + ':00:00')
    
    # Save processed data to S3
    if save_csv:
        # Set local timezone (US Pacific Time)
        local_tz = pytz.timezone("America/Los_Angeles") 
        current_date = datetime.now(local_tz).strftime("%Y%m%d")
        s3_output_key = f"{s3_output_prefix}/Hourly_Temperature_Data_{current_date}.csv"
        save_csv_to_s3(df_t, s3_bucket, s3_output_key)

    # Log processing time
    end_time = time.time()
    logger.info(f"Temperature processing took {(end_time - start_time) / 60:.2f} minutes")

    return df_t



def merge_load_temperature(df_load, df_t, save_csv=True, s3_bucket=None, s3_output_prefix="xgboost/processed/hourly_data"):
    """
    Merges processed load and temperature data, renames columns, and saves to S3.

    Args:
        df_load (pd.DataFrame): Processed load data.
        df_t (pd.DataFrame): Processed temperature data.
        save_csv (bool): If True, saves the merged data to S3.
        s3_bucket (str): Name of the S3 bucket.
        s3_output_prefix (str): S3 prefix for saving the merged CSV file.

    Returns:
        pd.DataFrame: Merged and renamed DataFrame.
    """
    
    # Start timing the process
    start_time = time.time()


    # Merge and select specific columns
    df = pd.merge(df_load, df_t, on=['TradeDate', 'Hour'], how='left')

    # Remove unnecessary 'TradeDateTime_x' column and keep relevant columns
    df = df[['TradeDateTime_y', 'Profile', 'Load_Per_Meter', 'Count', 'Load_Per_Meter_I', 
             'Count_I', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 
             'Workday', 'TradeDate', 'Temperature_Fill']]

    # Rename columns for consistency
    df = df.rename(columns={
        'TradeDateTime_y': 'Time',
        'Load_Per_Meter': 'Load',
        'Load_Per_Meter_I': 'Load_I',
        'Temperature_Fill': 'Temperature'
    })

    # Fill null 'Time' column with the corresponding 'TradeDate' and 'Hour'
    df['Time'] = pd.to_datetime(df['TradeDate'] + ' ' + df['Hour'].astype(str).str.zfill(2) + ':00:00')

    # Save to S3 if requested
    if save_csv and s3_bucket:
        # Set local timezone (US Pacific Time)
        local_tz = pytz.timezone("America/Los_Angeles") 
        current_date = datetime.now(local_tz).strftime("%Y%m%d")
        s3_output_key = f"{s3_output_prefix}/Merged_Hourly_Data_{current_date}.csv"
        
        save_csv_to_s3(df, s3_bucket, s3_output_key)

    # Log processing time
    end_time = time.time()
    logger.info(f"Load & temperature merge took {(end_time - start_time) / 60:.2f} minutes")

    return df
