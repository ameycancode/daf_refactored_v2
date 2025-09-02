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

       

def train_test_split(updated_dfs, split_date, s3_bucket, s3_output_prefix="xgboost/processed/train_test_split", model_input_prefix="xgboost/input"):
    """
    Splits the updated DataFrames into train and test sets based on a given date,
    and saves the resulting sets to S3.

    Args:
        updated_dfs (dict): Dictionary of updated DataFrames with count replaced.
        split_date (str): Date to split the data ('YYYY-MM-DD').
        s3_bucket (str): Name of the S3 bucket.
        s3_output_prefix (str): S3 prefix where train/test sets will be saved.
        model_input_prefix (str): S3 prefix where model input files will be saved.

    Returns:
        dict: Dictionary containing train and test DataFrames for each profile.
    """

    # Set local timezone (US Pacific Time)
    local_tz = pytz.timezone("America/Los_Angeles")  
    current_date = datetime.now(local_tz).strftime("%Y%m%d")


    #s3_output_prefix = s3_output_prefix.rstrip("/")  # Ensure no trailing slash
    model_input_prefix = model_input_prefix.rstrip("/")  # Ensure no trailing slash


    split_date = pd.to_datetime(split_date)
    split_results = {}


    for profile, df in updated_dfs.items():
        # Split into train and test sets
        train_set = df[df['Time'] < split_date]
        test_set = df[df['Time'] >= split_date]

        # Add "_r" suffix for RN profile (but not RNN)
        suffix = "_r" if profile == "df_RN" else ""
       
        # S3 keys for train and test sets
        train_key = f"{s3_output_prefix}/train/{profile}_train_{current_date}.csv"
        test_key = f"{s3_output_prefix}/test/{profile}_test_{current_date}{suffix}.csv"
        model_test_key = f"{model_input_prefix}/{profile}_test_{current_date}{suffix}.csv"

        # Save to S3
        save_csv_to_s3(train_set, s3_bucket, train_key)
        save_csv_to_s3(test_set, s3_bucket, test_key)
        save_csv_to_s3(test_set, s3_bucket, model_test_key)

        logger.info(f"Saved train set for {profile} to s3://{s3_bucket}/{train_key}")
        logger.info(f"Saved test set for {profile} to s3://{s3_bucket}/{test_key}")
        logger.info(f"Copied test set for {profile} to model input at s3://{s3_bucket}/{model_test_key}")

        # Add to results dictionary
        split_results[profile] = {
            "train": train_set,
            "test": test_set
        }

    return split_results
