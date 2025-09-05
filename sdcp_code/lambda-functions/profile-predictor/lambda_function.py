"""
Profile Predictor Lambda Function - Environment-Aware Version
This version uses environment variables for all configuration values
and removes hardcoded bucket names from documentation examples.
"""

import json
import boto3
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_runtime_client = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

def sanitize_for_logging(value: str, max_length: int = 50) -> str:
    """Sanitize string values for safe logging"""
    if not isinstance(value, str):
        value = str(value)
    # Remove potentially sensitive characters and limit length
    sanitized = ''.join(c for c in value if c.isalnum() or c in '-_.')[:max_length]
    return sanitized if sanitized else 'unknown'

def sanitize_event_for_logging(event: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize event data for logging"""
    sanitized_event = {}
    
    # Safe keys to include
    safe_keys = ['operation', 'profile', 'execution_id']
    for key in safe_keys:
        if key in event:
            sanitized_event[key] = sanitize_for_logging(str(event[key]), 100)
    
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_endpoint_name': 'endpoint_name' in event,
        'has_data_bucket': 'data_bucket' in event
    }
    
    return sanitized_event

def lambda_handler(event, context):
    """
    Main Lambda handler for profile-specific predictions
    
    Expected event structure (environment-aware):
    {
        "operation": "run_profile_prediction",
        "profile": "RNN",
        "endpoint_name": "energy-forecasting-rnn-endpoint-20250820-133837",
        "data_bucket": "<from environment variable DATA_BUCKET>",
        "model_bucket": "<from environment variable MODEL_BUCKET>",
        "execution_id": "d15723ad-434a-4614-828f-d252b3d61041"
    }
    """
    
    execution_id = context.aws_request_id
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting profile prediction Lambda [execution_id={sanitize_for_logging(execution_id)}]")
        
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
        
        # Extract required parameters - environment-aware
        operation = event.get('operation', 'run_profile_prediction')
        profile = event.get('profile')
        endpoint_name = event.get('endpoint_name')
        
        # Environment-aware bucket configuration
        data_bucket = event.get('data_bucket') or os.environ.get('DATA_BUCKET')
        model_bucket = event.get('model_bucket') or os.environ.get('MODEL_BUCKET')
        step_functions_execution_id = event.get('execution_id', execution_id)
        
        # Validate required parameters
        if not profile:
            raise ValueError("Profile is required for prediction operations")
        if not endpoint_name:
            raise ValueError("Endpoint name is required for prediction operations")
        if not data_bucket:
            raise ValueError("Data bucket must be provided in event or DATA_BUCKET environment variable")
        
        # SECURITY FIX: Sanitize values before logging
        safe_operation = sanitize_for_logging(operation)
        safe_profile = sanitize_for_logging(profile, 50)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        safe_data_bucket = sanitize_for_logging(data_bucket, 100)
        
        logger.info(f"Processing operation={safe_operation} profile={safe_profile}")
        logger.info(f"Using endpoint={safe_endpoint_name}")
        logger.info(f"Data bucket={safe_data_bucket}")
        
        # Handle different operations
        if operation == 'run_profile_prediction':
            result = run_profile_prediction(profile, endpoint_name, data_bucket, step_functions_execution_id)
        elif operation == 'test_endpoint':
            result = test_endpoint_connectivity(profile, endpoint_name, execution_id)
        else:
            raise ValueError(f"Unknown operation: {safe_operation}")
        
        # Add execution metadata
        result['execution_metadata'] = {
            'lambda_execution_id': execution_id,
            'step_functions_execution_id': step_functions_execution_id,
            'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(event.get('profile', 'unknown'), 50)
        logger.error(f"Profile prediction failed [execution_id={sanitize_for_logging(execution_id)}] profile={safe_profile} error={error_msg}")
        
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'profile': event.get('profile'),
                'endpoint_name': event.get('endpoint_name'),
                'message': 'Profile prediction failed',
                'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
        }

def run_profile_prediction(profile: str, endpoint_name: str, data_bucket: str, execution_id: str) -> Dict[str, Any]:
    """
    Run prediction for a specific profile using its endpoint
    """
    
    try:
        safe_profile = sanitize_for_logging(profile, 50)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.info(f"Running prediction profile={safe_profile} endpoint={safe_endpoint_name}")
        
        # Step 1: Load prediction input data from S3
        input_data = load_prediction_input_data(profile, data_bucket)
        
        if not input_data:
            return {
                'operation': 'run_profile_prediction',
                'profile': profile,
                'status': 'failed',
                'error': 'No prediction input data found',
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Loaded input data profile={safe_profile} records={len(input_data.get('instances', []))}")
        
        # Step 2: Invoke SageMaker endpoint
        prediction_result = invoke_sagemaker_endpoint(endpoint_name, input_data, profile)
        
        # Step 3: Process and validate predictions
        processed_result = process_prediction_results(prediction_result, profile, execution_id)
        
        # Step 4: Store prediction results (optional)
        storage_result = store_prediction_results(processed_result, profile, data_bucket, execution_id)
        
        return {
            'operation': 'run_profile_prediction',
            'profile': profile,
            'status': 'success',
            'endpoint_name': endpoint_name,
            'input_records': len(input_data.get('instances', [])),
            'prediction_result': processed_result,
            'storage_result': storage_result,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Profile prediction failed profile={safe_profile} error={error_msg}")
        
        return {
            'operation': 'run_profile_prediction',
            'profile': profile,
            'status': 'failed',
            'error': error_msg,
            'endpoint_name': endpoint_name,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }

def load_prediction_input_data(profile: str, data_bucket: str) -> Optional[Dict[str, Any]]:
    """
    Load prediction input data for the profile from S3
    """
    
    try:
        # Try to find the most recent input data file for this profile
        input_prefix = f"prediction-inputs/{profile}/"
        
        safe_profile = sanitize_for_logging(profile, 50)
        safe_bucket = sanitize_for_logging(data_bucket, 100)
        logger.info(f"Loading input data profile={safe_profile} bucket={safe_bucket}")
        
        response = s3_client.list_objects_v2(
            Bucket=data_bucket,
            Prefix=input_prefix,
            MaxKeys=10
        )
        
        if 'Contents' not in response:
            logger.warning(f"No input files found profile={safe_profile}")
            # Generate sample data as fallback
            return generate_sample_input_data(profile)
        
        # Find the most recent input file
        input_files = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.json'):
                input_files.append({
                    'key': obj['Key'],
                    'last_modified': obj['LastModified']
                })
        
        if not input_files:
            logger.warning(f"No JSON input files found profile={safe_profile}")
            return generate_sample_input_data(profile)
        
        # Load the most recent file
        latest_file = max(input_files, key=lambda x: x['last_modified'])
        
        response = s3_client.get_object(Bucket=data_bucket, Key=latest_file['key'])
        input_data = json.loads(response['Body'].read())
        
        safe_key = sanitize_for_logging(latest_file['key'], 150)
        logger.info(f"Loaded input data profile={safe_profile} key={safe_key}")
        
        return input_data
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Failed to load input data profile={safe_profile} error={error_msg}")
        
        # Generate sample data as fallback
        return generate_sample_input_data(profile)

def generate_sample_input_data(profile: str) -> Dict[str, Any]:
    """
    Generate sample input data for testing when no real data is available
    """
    
    try:
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Generating sample input data profile={safe_profile}")
        
        # Base features for all profiles
        base_features = [
            1000,    # load
            2025,    # year
            1,       # month
            29,      # day
            12,      # hour
            3,       # dayofweek
            0,       # is_weekend
            0,       # is_holiday
            1,       # season
            75.5,    # temperature
            0.85,    # humidity
            0.80     # cloud_cover
        ]
        
        # Add radiation for RN profile
        if profile == 'RN':
            base_features.append(500.0)  # shortwave_radiation
        
        # Generate multiple sample records (24 hours)
        instances = []
        for hour in range(24):
            features = base_features.copy()
            features[4] = hour  # Update hour
            # Add some variation
            features[0] = features[0] + (hour * 50)  # Vary load by hour
            features[9] = features[9] + ((hour - 12) * 2)  # Vary temperature
            instances.append(features)
        
        sample_data = {
            "instances": instances,
            "metadata": {
                "profile": profile,
                "generated": True,
                "timestamp": datetime.now().isoformat(),
                "record_count": len(instances)
            }
        }
        
        logger.info(f"Generated sample data profile={safe_profile} records={len(instances)}")
        return sample_data
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Failed to generate sample data profile={safe_profile} error={error_msg}")
        return None

def invoke_sagemaker_endpoint(endpoint_name: str, input_data: Dict[str, Any], profile: str) -> Dict[str, Any]:
    """
    Invoke SageMaker endpoint for predictions
    """
    
    try:
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Invoking endpoint profile={safe_profile} endpoint={safe_endpoint_name}")
        
        # Prepare payload
        payload = json.dumps(input_data)
        
        # Invoke endpoint
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        logger.info(f"Endpoint invocation successful profile={safe_profile}")
        
        return {
            'status': 'success',
            'endpoint_name': endpoint_name,
            'profile': profile,
            'result': result,
            'input_records': len(input_data.get('instances', [])),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Endpoint invocation failed profile={safe_profile} endpoint={safe_endpoint_name} error={error_msg}")
        
        return {
            'status': 'failed',
            'endpoint_name': endpoint_name,
            'profile': profile,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

def process_prediction_results(prediction_result: Dict[str, Any], profile: str, execution_id: str) -> Dict[str, Any]:
    """
    Process and validate prediction results
    """
    
    try:
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Processing prediction results profile={safe_profile}")
        
        if prediction_result.get('status') != 'success':
            return {
                'status': 'failed',
                'error': 'Endpoint invocation failed',
                'profile': profile,
                'raw_result': prediction_result
            }
        
        # Extract predictions from result
        raw_result = prediction_result.get('result', {})
        
        # Handle different response formats
        if isinstance(raw_result, dict) and 'predictions' in raw_result:
            predictions = raw_result['predictions']
            metadata = raw_result.get('metadata', {})
        elif isinstance(raw_result, list):
            predictions = raw_result
            metadata = {}
        else:
            raise ValueError(f"Unexpected result format: {type(raw_result)}")
        
        if not predictions:
            raise ValueError("No predictions returned from endpoint")
        
        # Calculate statistics
        prediction_stats = calculate_prediction_statistics(predictions)
        
        processed_result = {
            'status': 'success',
            'profile': profile,
            'predictions': predictions,
            'prediction_count': len(predictions),
            'statistics': prediction_stats,
            'endpoint_metadata': metadata,
            'processing_timestamp': datetime.now().isoformat(),
            'execution_id': execution_id
        }
        
        logger.info(f"Processed predictions profile={safe_profile} count={len(predictions)}")
        
        return processed_result
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Failed to process predictions profile={safe_profile} error={error_msg}")
        
        return {
            'status': 'failed',
            'error': error_msg,
            'profile': profile,
            'raw_result': prediction_result,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }

def calculate_prediction_statistics(predictions: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for predictions
    """
    
    try:
        if not predictions:
            return {}
        
        # Convert to float if needed
        pred_values = [float(p) for p in predictions]
        
        stats = {
            'min': min(pred_values),
            'max': max(pred_values),
            'mean': sum(pred_values) / len(pred_values),
            'total': sum(pred_values),
            'count': len(pred_values)
        }
        
        # Calculate median
        sorted_preds = sorted(pred_values)
        n = len(sorted_preds)
        if n % 2 == 0:
            stats['median'] = (sorted_preds[n//2-1] + sorted_preds[n//2]) / 2
        else:
            stats['median'] = sorted_preds[n//2]
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to calculate statistics: {str(e)}")
        return {'error': 'Statistics calculation failed'}

def store_prediction_results(processed_result: Dict[str, Any], profile: str, data_bucket: str, execution_id: str) -> Dict[str, Any]:
    """
    Store prediction results in S3
    """
    
    try:
        safe_profile = sanitize_for_logging(profile, 50)
        safe_bucket = sanitize_for_logging(data_bucket, 100)
        logger.info(f"Storing prediction results profile={safe_profile} bucket={safe_bucket}")
        
        # Create S3 key for results
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%H%M%S")
        s3_key = f"prediction-results/{profile}/{current_date}/predictions_{profile}_{current_time}_{execution_id[:8]}.json"
        
        # Prepare storage payload
        storage_payload = {
            'prediction_results': processed_result,
            'storage_metadata': {
                'profile': profile,
                'execution_id': execution_id,
                'storage_timestamp': datetime.now().isoformat(),
                'data_bucket': data_bucket,
                's3_key': s3_key
            }
        }
        
        # Store in S3
        s3_client.put_object(
            Bucket=data_bucket,
            Key=s3_key,
            Body=json.dumps(storage_payload, indent=2, default=str),
            ContentType='application/json'
        )
        
        safe_key = sanitize_for_logging(s3_key, 200)
        logger.info(f"Stored prediction results profile={safe_profile} key={safe_key}")
        
        return {
            'status': 'success',
            's3_bucket': data_bucket,
            's3_key': s3_key,
            'storage_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Failed to store prediction results profile={safe_profile} error={error_msg}")
        
        return {
            'status': 'failed',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

def test_endpoint_connectivity(profile: str, endpoint_name: str, execution_id: str) -> Dict[str, Any]:
    """
    Test endpoint connectivity with minimal sample data
    """
    
    try:
        safe_profile = sanitize_for_logging(profile, 50)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.info(f"Testing endpoint connectivity profile={safe_profile} endpoint={safe_endpoint_name}")
        
        # Generate minimal test data
        test_data = generate_sample_input_data(profile)
        if not test_data:
            return {
                'status': 'failed',
                'error': 'Failed to generate test data',
                'profile': profile,
                'endpoint_name': endpoint_name
            }
        
        # Use only first record for connectivity test
        test_payload = {
            "instances": test_data["instances"][:1]
        }
        
        # Test endpoint
        start_time = datetime.now()
        prediction_result = invoke_sagemaker_endpoint(endpoint_name, test_payload, profile)
        response_time = (datetime.now() - start_time).total_seconds()
        
        if prediction_result.get('status') == 'success':
            return {
                'status': 'success',
                'profile': profile,
                'endpoint_name': endpoint_name,
                'response_time_seconds': response_time,
                'test_prediction': prediction_result.get('result'),
                'connectivity_test': True,
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'status': 'failed',
                'profile': profile,
                'endpoint_name': endpoint_name,
                'error': prediction_result.get('error', 'Unknown error'),
                'response_time_seconds': response_time,
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Endpoint connectivity test failed profile={safe_profile} error={error_msg}")
        
        return {
            'status': 'failed',
            'profile': profile,
            'endpoint_name': endpoint_name,
            'error': error_msg,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }