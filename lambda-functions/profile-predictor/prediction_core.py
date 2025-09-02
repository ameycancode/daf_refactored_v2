"""
Prediction Core Module
lambda-functions/profile-predictor/prediction_core.py

Core prediction logic converted from container main.py
Handles endpoint invocation and result processing
"""

import json
import boto3
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import pytz
from io import StringIO

logger = logging.getLogger(__name__)

class PredictionCore:
    """
    Core prediction functionality converted from container logic
    """
   
    def __init__(self, data_bucket: str, model_bucket: str, s3_utils):
        """Initialize prediction core"""
        self.data_bucket = data_bucket
        self.model_bucket = model_bucket
        self.s3_utils = s3_utils
       
        # Initialize SageMaker runtime client
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
       
        logger.info("Prediction core initialized")

    def invoke_endpoint_for_prediction(self, endpoint_name: str, test_data: pd.DataFrame, profile: str) -> List[float]:
        """
        Invoke SageMaker endpoint to get predictions
        Converted from container predictions.py logic
        """
       
        try:
            logger.info(f"Invoking endpoint {endpoint_name} for profile {profile}")
            logger.info(f"Input data shape: {test_data.shape}")
           
            # Prepare data payload (same logic as container)
            data_payload = test_data.values.tolist()
           
            payload = {
                'instances': data_payload
            }
           
            logger.info(f"Prepared payload with {len(data_payload)} instances")
           
            # Invoke endpoint (same logic as container)
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
           
            # Parse response (same logic as container)
            result = json.loads(response['Body'].read().decode())
           
            # Extract predictions (same logic as container)
            if 'predictions' in result:
                predictions = result['predictions']
            elif isinstance(result, list):
                predictions = result
            else:
                predictions = result
           
            # Validate predictions
            if not isinstance(predictions, list):
                raise ValueError(f"Invalid prediction format: expected list, got {type(predictions)}")
           
            logger.info(f"✓ Received {len(predictions)} predictions from endpoint {endpoint_name}")
           
            # Log prediction statistics
            if predictions:
                pred_stats = {
                    'min': float(min(predictions)),
                    'max': float(max(predictions)),
                    'mean': float(np.mean(predictions)),
                    'count': len(predictions)
                }
                logger.info(f"Prediction statistics: {pred_stats}")
           
            return predictions
           
        except Exception as e:
            logger.error(f"Failed to invoke endpoint {endpoint_name}: {str(e)}")
            raise Exception(f"Endpoint invocation failed for {profile}: {str(e)}")

    def post_process_and_save_predictions(self, profile: str, test_data: pd.DataFrame,
                                        predictions: List[float], execution_id: str,
                                        lambda_execution_id: str) -> Dict[str, Any]:
        """
        Post-process predictions and save to S3
        Converted from container predictions.py logic
        """
       
        try:
            logger.info(f"Post-processing predictions for profile {profile}")
           
            # Combine test data with predictions (same logic as container)
            test_df = test_data.copy()
            test_df['Predicted_Load'] = predictions
           
            # Add computed fields (same logic as container)
            test_df['TradeDateTime'] = pd.to_datetime(test_df[['Year', 'Month', 'Day', 'Hour']])
            test_df['Load_All'] = test_df['Predicted_Load'] * test_df['Count']
           
            # Calculate statistics (same logic as container)
            statistics = {
                'min_prediction': float(min(predictions)) if predictions else 0,
                'max_prediction': float(max(predictions)) if predictions else 0,
                'mean_prediction': float(np.mean(predictions)) if predictions else 0,
                'total_load': float(test_df['Load_All'].sum()) if 'Load_All' in test_df.columns else 0,
                'prediction_count': len(predictions),
                'total_meter_count': float(test_df['Count'].sum()) if 'Count' in test_df.columns else 0
            }
           
            logger.info(f"Calculated prediction statistics: {statistics}")
           
            # Generate output filename (same logic as container)
            pacific_tz = pytz.timezone("America/Los_Angeles")
            today_str = datetime.now(pacific_tz).strftime("%Y%m%d")
            filename = f"{profile}_predictions_{today_str}_{execution_id[:8]}_{lambda_execution_id[:8]}.csv"
           
            # Save to S3 (same logic as container)
            s3_key = f"archived_folders/forecasting/predictions/enhanced/{filename}"
           
            # Convert to CSV and upload
            csv_buffer = StringIO()
            test_df.to_csv(csv_buffer, index=False)
           
            self.s3_utils.put_object(
                bucket=self.data_bucket,
                key=s3_key,
                body=csv_buffer.getvalue(),
                content_type='text/csv'
            )
           
            s3_output_path = f"s3://{self.data_bucket}/{s3_key}"
           
            logger.info(f"✓ Saved predictions to {s3_output_path}")
           
            # Generate result summary
            result = {
                's3_output_path': s3_output_path,
                'statistics': statistics,
                'filename': filename,
                'row_count': len(test_df),
                'columns': list(test_df.columns),
                'data_summary': {
                    'prediction_period': {
                        'start_time': test_df['TradeDateTime'].min().isoformat() if 'TradeDateTime' in test_df.columns else None,
                        'end_time': test_df['TradeDateTime'].max().isoformat() if 'TradeDateTime' in test_df.columns else None,
                        'hours_predicted': len(test_df)
                    },
                    'load_analysis': {
                        'peak_hour': test_df.loc[test_df['Predicted_Load'].idxmax(), 'Hour'] if len(test_df) > 0 else None,
                        'peak_load': float(test_df['Predicted_Load'].max()) if len(test_df) > 0 else 0,
                        'minimum_load': float(test_df['Predicted_Load'].min()) if len(test_df) > 0 else 0,
                        'average_hourly_load': float(test_df['Predicted_Load'].mean()) if len(test_df) > 0 else 0
                    }
                }
            }
           
            logger.info(f"✓ Generated comprehensive result summary for {profile}")
           
            return result
           
        except Exception as e:
            logger.error(f"Failed to post-process predictions for {profile}: {str(e)}")
            raise Exception(f"Post-processing failed for {profile}: {str(e)}")

    def validate_prediction_data(self, test_data: pd.DataFrame, profile: str) -> bool:
        """
        Validate prediction data before endpoint invocation
        """
       
        try:
            # Check if DataFrame is not empty
            if test_data.empty:
                raise ValueError(f"Test data is empty for profile {profile}")
           
            # Check for required columns based on profile
            profile_features = {
                'RNN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                       'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'RN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                      'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days', 'shortwave_radiation'],
                'M': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                     'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'S': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                     'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'AGR': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                       'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'L': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                     'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'A6': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                      'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days']
            }
           
            required_features = profile_features.get(profile, profile_features['RNN'])
            missing_features = [feature for feature in required_features if feature not in test_data.columns]
           
            if missing_features:
                raise ValueError(f"Missing required features for {profile}: {missing_features}")
           
            # Check for null values in critical columns
            critical_columns = ['Count', 'Year', 'Month', 'Day', 'Hour']
            null_counts = test_data[critical_columns].isnull().sum()
           
            if null_counts.any():
                logger.warning(f"Found null values in critical columns for {profile}: {null_counts.to_dict()}")
           
            # Check data types
            numeric_columns = ['Count', 'Year', 'Month', 'Day', 'Hour', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days']
            if profile == 'RN':
                numeric_columns.append('shortwave_radiation')
           
            for col in numeric_columns:
                if col in test_data.columns and not pd.api.types.is_numeric_dtype(test_data[col]):
                    logger.warning(f"Column {col} is not numeric for profile {profile}")
           
            logger.info(f"✓ Data validation passed for profile {profile}")
            return True
           
        except Exception as e:
            logger.error(f"Data validation failed for profile {profile}: {str(e)}")
            raise
