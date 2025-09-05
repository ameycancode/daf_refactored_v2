"""
AWS Lambda Function - Main Handler
lambda-functions/profile-predictor/lambda_function.py

Main entry point for profile-specific predictions
Converts container logic to Lambda execution model
SECURITY FIX: Log injection vulnerability (CWE-117, CWE-93) patched
"""

import json
import os
import logging
import re
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import prediction modules
from prediction_core import PredictionCore
from data_loader import DataLoader
from weather_forecast import WeatherForecast
from radiation_forecast import RadiationForecast
from s3_utils import S3Utils

# Security: Input sanitization patterns
LOG_SANITIZER = {
    'newlines': re.compile(r'[\r\n]+'),
    'tabs': re.compile(r'[\t]+'),
    'control_chars': re.compile(r'[\x00-\x1f\x7f-\x9f]'),
    'excessive_whitespace': re.compile(r'\s{3,}')
}

def sanitize_for_logging(value, max_length=500):
    """
    Sanitize input for safe logging to prevent log injection
   
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length for logged values
   
    Returns:
        Sanitized string safe for logging
    """
    if value is None:
        return "None"
   
    str_value = str(value)
   
    # Truncate if too long
    if len(str_value) > max_length:
        str_value = str_value[:max_length] + "...[truncated]"
   
    # Remove dangerous characters
    str_value = LOG_SANITIZER['newlines'].sub(' ', str_value)
    str_value = LOG_SANITIZER['tabs'].sub(' ', str_value)
    str_value = LOG_SANITIZER['control_chars'].sub('', str_value)
    str_value = LOG_SANITIZER['excessive_whitespace'].sub(' ', str_value)
   
    return str_value.strip()

def sanitize_event_for_logging(event):
    """Create a sanitized version of event for safe logging"""
    sanitized_event = {}
   
    safe_fields = ['operation', 'profile', 'data_bucket', 'model_bucket']
   
    for field in safe_fields:
        if field in event:
            sanitized_event[field] = sanitize_for_logging(event[field])
   
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_profile': 'profile' in event,
        'has_endpoint_name': 'endpoint_name' in event,
        'has_execution_id': 'execution_id' in event
    }
   
    return sanitized_event

def lambda_handler(event, context):
    """
    Main Lambda handler for profile-specific predictions
   
    Expected event structure:
    {
        "operation": "run_profile_prediction",
        "profile": "RNN",
        "endpoint_name": "energy-forecasting-rnn-endpoint-20250820-133837",
        "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
        "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
        "execution_id": "d15723ad-434a-4614-828f-d252b3d61041"
    }
    """
   
    execution_id = context.aws_request_id
    start_time = datetime.now()
   
    try:
        logger.info(f"Starting profile prediction Lambda [execution_id={sanitize_for_logging(execution_id)}]")
       
        # SECURITY FIX: Sanitize event data before logging (Line 27 original)
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
       
        # Extract required parameters
        operation = event.get('operation', 'run_profile_prediction')
        profile = event.get('profile')
        endpoint_name = event.get('endpoint_name')
        data_bucket = event.get('data_bucket', os.environ.get('DATA_BUCKET'))
        model_bucket = event.get('model_bucket', os.environ.get('MODEL_BUCKET'))
        step_functions_execution_id = event.get('execution_id', execution_id)
       
        # Validate required parameters
        if not profile:
            raise ValueError("Profile is required for prediction operations")
        if not endpoint_name:
            raise ValueError("Endpoint name is required for prediction operations")
        if not data_bucket:
            raise ValueError("Data bucket is required for prediction operations")
       
        # SECURITY FIX: Sanitize values before logging (Lines 42, 47, 48 original)
        safe_operation = sanitize_for_logging(operation)
        safe_profile = sanitize_for_logging(profile, 50)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        safe_data_bucket = sanitize_for_logging(data_bucket, 100)
       
        logger.info(f"Processing operation={safe_operation} profile={safe_profile}")
        logger.info(f"Using endpoint={safe_endpoint_name}")
        logger.info(f"Data bucket={safe_data_bucket}")
       
        # Handle different operations
        if operation == 'run_profile_prediction':
            result = run_profile_prediction(
                profile=profile,
                endpoint_name=endpoint_name,
                data_bucket=data_bucket,
                model_bucket=model_bucket,
                execution_id=step_functions_execution_id,
                lambda_execution_id=execution_id
            )
        else:
            raise ValueError(f"Unknown operation: {safe_operation}")
       
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        result['lambda_execution_time_seconds'] = execution_time
        result['lambda_execution_id'] = execution_id
       
        logger.info(f"Successfully completed prediction profile={safe_profile} execution_time={execution_time:.2f}s")
       
        return result
       
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(event.get('profile', 'unknown'), 50)
       
        logger.error(f"Profile prediction failed profile={safe_profile} execution_time={execution_time:.2f}s error={error_msg}")
       
        # Raise exception to trigger Step Functions error handling
        raise Exception(error_msg)

def run_profile_prediction(profile: str, endpoint_name: str, data_bucket: str,
                         model_bucket: str, execution_id: str, lambda_execution_id: str) -> Dict[str, Any]:
    """
    Main prediction workflow for a single profile
    Preserves exact logic from container implementation
    """
   
    try:
        # SECURITY FIX: Sanitize profile name before logging (Line 65 original)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Starting prediction workflow profile={safe_profile}")
       
        # Initialize components
        s3_utils = S3Utils()
        data_loader = DataLoader(data_bucket, s3_utils)
        weather_forecast = WeatherForecast(data_bucket, s3_utils)
        radiation_forecast = RadiationForecast(data_bucket, s3_utils)
        prediction_core = PredictionCore(data_bucket, model_bucket, s3_utils)
       
        # Step 1: Fetch weather and radiation data (same as container logic)
        logger.info("Step 1: Fetching weather and radiation forecasts...")
        weather_df = weather_forecast.fetch_weather_forecast()
        radiation_df = None
       
        # Only fetch radiation for RN profile (same logic as container)
        if profile == 'RN':
            radiation_df = radiation_forecast.fetch_shortwave_radiation()
       
        # SECURITY FIX: Safe data count logging (Line 86 original)
        if weather_df is not None:
            weather_data_count = len(weather_df)
            logger.info(f"Weather forecast loaded data_points={weather_data_count}")
        else:
            logger.warning("Weather forecast not available")
       
        if radiation_df is not None:
            radiation_data_count = len(radiation_df)
            logger.info(f"Radiation forecast loaded data_points={radiation_data_count}")
        elif profile == 'RN':
            logger.warning("Radiation forecast not available for RN profile")
       
        # Step 2: Load profile-specific test data (same as container logic)
        logger.info(f"Step 2: Loading test data profile={safe_profile}")
        test_data = data_loader.load_profile_test_data(profile)
       
        if test_data is None or len(test_data) == 0:
            raise Exception(f"No test data found for profile {profile}")
       
        # SECURITY FIX: Safe data count logging (Line 94 original)
        test_data_rows = len(test_data)
        logger.info(f"Loaded test data rows={test_data_rows}")
       
        # Step 3: Prepare prediction data (same as container logic)
        logger.info(f"Step 3: Preparing prediction data profile={safe_profile}")
        prepared_data = data_loader.prepare_prediction_data(
            profile=profile,
            profile_data=test_data,
            weather_df=weather_df,
            radiation_df=radiation_df
        )
       
        logger.info(f"Prepared prediction data shape={prepared_data.shape}")
       
        # Step 4: Run predictions via endpoint (same as container logic)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.info(f"Step 4: Running predictions endpoint={safe_endpoint_name}")
        predictions = prediction_core.invoke_endpoint_for_prediction(
            endpoint_name=endpoint_name,
            test_data=prepared_data,
            profile=profile
        )
       
        if not predictions:
            raise Exception(f"No predictions received from endpoint {endpoint_name}")
       
        # SECURITY FIX: Safe prediction count logging (Line 108 original)
        predictions_count = len(predictions)
        logger.info(f"Received predictions from endpoint count={predictions_count}")
       
        # Step 5: Post-process and save results (same as container logic)
        logger.info(f"Step 5: Post-processing and saving results")
        results = prediction_core.post_process_and_save_predictions(
            profile=profile,
            test_data=prepared_data,
            predictions=predictions,
            execution_id=execution_id,
            lambda_execution_id=lambda_execution_id
        )
       
        # Step 6: Generate summary (same as container logic)
        summary = {
            'profile': profile,
            'endpoint_name': endpoint_name,
            'status': 'success',
            'prediction_count': len(predictions),
            'execution_id': execution_id,
            'completion_time': datetime.now().isoformat(),
            'output_location': results.get('s3_output_path'),
            'statistics': results.get('statistics', {}),
            'data_summary': {
                'input_features': prepared_data.shape[1] if hasattr(prepared_data, 'shape') else len(prepared_data.columns),
                'input_rows': prepared_data.shape[0] if hasattr(prepared_data, 'shape') else len(prepared_data),
                'weather_data_points': len(weather_df) if weather_df is not None else 0,
                'radiation_data_points': len(radiation_df) if radiation_df is not None else 0
            },
            'workflow_steps_completed': [
                'weather_forecast_fetched',
                'radiation_forecast_fetched' if profile == 'RN' else 'radiation_forecast_skipped',
                'test_data_loaded',
                'prediction_data_prepared',
                'endpoint_predictions_completed',
                'results_post_processed',
                'results_saved_to_s3'
            ]
        }
       
        # SECURITY FIX: Safe completion logging (Line 157 original)
        logger.info(f"Successfully completed prediction workflow profile={safe_profile}")
       
        return summary
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Prediction workflow failed profile={safe_profile} error={error_msg}")
        raise
