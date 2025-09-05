"""
Enhanced Model Registry Lambda Function - Environment-Aware Version
This version uses environment variables for all configuration values
instead of hardcoded dev environment fallback values.
"""

import json
import boto3
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import os

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
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
    safe_keys = ['operation', 'training_date', 'region', 'account_id']
    for key in safe_keys:
        if key in event:
            sanitized_event[key] = sanitize_for_logging(str(event[key]), 100)
    
    # Include metadata about training_metadata without exposing content
    if 'training_metadata' in event and isinstance(event['training_metadata'], dict):
        sanitized_event['training_metadata_keys'] = list(event['training_metadata'].keys())[:10]  # Max 10 keys
    
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_training_metadata': 'training_metadata' in event,
        'has_buckets': bool(event.get('model_bucket') or event.get('data_bucket'))
    }
    
    return sanitized_event

def lambda_handler(event, context):
    """
    Enhanced Lambda handler for model registry with Step Functions integration
    Handles both direct invocation and Step Functions pipeline integration
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting enhanced model registry process [execution_id={sanitize_for_logging(execution_id)}]")
        
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
        
        # Extract information from event (supports both formats)
        training_metadata = event.get('training_metadata', {})
        training_date = event.get('training_date', datetime.now().strftime('%Y%m%d'))
        
        # Environment-aware bucket configuration - no hardcoded fallbacks
        model_bucket = event.get('model_bucket') or os.environ.get('MODEL_BUCKET')
        data_bucket = event.get('data_bucket') or os.environ.get('DATA_BUCKET')
        
        # Validate required environment variables
        if not model_bucket:
            raise ValueError("MODEL_BUCKET must be provided in event or environment variables")
        if not data_bucket:
            raise ValueError("DATA_BUCKET must be provided in event or environment variables")
        
        # Enhanced configuration
        config = {
            "model_bucket": model_bucket,
            "data_bucket": data_bucket,
            "model_prefix": "xgboost/",
            "registry_prefix": "registry/",
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
            "customer_profile": "SDCP",
            "region": os.environ.get('REGION', 'us-west-2'),
            "account_id": os.environ.get('ACCOUNT_ID', context.invoked_function_arn.split(':')[4])
        }
        
        # SECURITY FIX: Sanitize config values for logging
        safe_training_date = sanitize_for_logging(training_date, 20)
        safe_model_bucket = sanitize_for_logging(config["model_bucket"], 100)
        safe_data_bucket = sanitize_for_logging(config["data_bucket"], 100)
        
        logger.info(f"Configuration: training_date={safe_training_date} model_bucket={safe_model_bucket} data_bucket={safe_data_bucket}")
        
        # Execute model registry workflow
        registry_result = execute_enhanced_model_registry_workflow(config, training_metadata, execution_id)
        
        # Return Step Functions compatible response
        return {
            'statusCode': 200,
            'body': registry_result
        }
        
    except Exception as e:
        error_message = sanitize_for_logging(str(e), 200)
        logger.error(f"Enhanced model registry process failed [execution_id={sanitize_for_logging(execution_id)}]: {error_message}")
        
        return {
            'statusCode': 500,
            'body': {
                'error': error_message,
                'execution_id': execution_id,
                'message': 'Enhanced model registry process failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def execute_enhanced_model_registry_workflow(config: Dict[str, Any], training_metadata: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Execute the complete enhanced model registry workflow
    """
    
    try:
        logger.info("Starting enhanced model registry workflow")
        
        # Step 1: Scan for available models
        logger.info("Step 1: Scanning for available models")
        available_models = scan_for_profile_models(config)
        
        if not available_models:
            logger.warning("No models found in S3")
            return {
                'stage': 'model_scanning',
                'status': 'no_models_found',
                'available_models': {},
                'message': 'No models found in S3 model bucket',
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Found {len(available_models)} models")
        
        # Step 2: Register models with SageMaker
        logger.info("Step 2: Registering models with SageMaker")
        registration_results = register_models_with_sagemaker(available_models, config, training_metadata, execution_id)
        
        # Step 3: Create model packages
        logger.info("Step 3: Creating model packages")
        model_package_results = create_model_packages(registration_results, config, execution_id)
        
        # Step 4: Store registry metadata
        logger.info("Step 4: Storing registry metadata")
        metadata_result = store_registry_metadata(model_package_results, config, training_metadata, execution_id)
        
        # Determine approved models for next stage
        approved_models = {}
        for profile, package_info in model_package_results.items():
            if package_info.get('status') == 'success':
                approved_models[profile] = {
                    'model_package_arn': package_info['model_package_arn'],
                    'model_name': package_info['model_name'],
                    'profile': profile,
                    'registration_date': package_info['timestamp']
                }
        
        # Return comprehensive results
        result = {
            'stage': 'complete',
            'status': 'success',
            'execution_id': execution_id,
            'available_models': available_models,
            'registration_results': registration_results,
            'model_package_results': model_package_results,
            'approved_models': approved_models,
            'metadata_storage': metadata_result,
            'summary': {
                'total_models_found': len(available_models),
                'total_models_registered': len([r for r in registration_results.values() if r.get('status') == 'success']),
                'total_packages_created': len([p for p in model_package_results.values() if p.get('status') == 'success']),
                'approved_models_count': len(approved_models)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Enhanced model registry workflow completed successfully: {len(approved_models)} approved models")
        return result
        
    except Exception as e:
        error_message = sanitize_for_logging(str(e), 200)
        logger.error(f"Enhanced model registry workflow failed: {error_message}")
        
        return {
            'stage': 'workflow_error',
            'status': 'failed',
            'error': error_message,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }

def scan_for_profile_models(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Scan S3 for available trained models for each profile
    """
    
    available_models = {}
    model_bucket = config['model_bucket']
    model_prefix = config['model_prefix']
    profiles = config['profiles']
    
    try:
        for profile in profiles:
            try:
                # Look for model artifacts for this profile
                profile_prefix = f"{model_prefix}{profile}/"
                
                safe_profile = sanitize_for_logging(profile, 20)
                safe_bucket = sanitize_for_logging(model_bucket, 100)
                logger.info(f"Scanning for models profile={safe_profile} bucket={safe_bucket}")
                
                response = s3_client.list_objects_v2(
                    Bucket=model_bucket,
                    Prefix=profile_prefix,
                    MaxKeys=100
                )
                
                if 'Contents' in response:
                    # Find model.tar.gz files
                    model_files = []
                    for obj in response['Contents']:
                        if obj['Key'].endswith('model.tar.gz'):
                            model_files.append({
                                'key': obj['Key'],
                                'last_modified': obj['LastModified'],
                                'size': obj['Size']
                            })
                    
                    if model_files:
                        # Get the most recent model
                        latest_model = max(model_files, key=lambda x: x['last_modified'])
                        
                        available_models[profile] = {
                            'profile': profile,
                            's3_path': f"s3://{model_bucket}/{latest_model['key']}",
                            'last_modified': latest_model['last_modified'].isoformat(),
                            'size_bytes': latest_model['size'],
                            'model_count': len(model_files)
                        }
                        
                        logger.info(f"Found model for profile={safe_profile} path={sanitize_for_logging(latest_model['key'], 100)}")
                    else:
                        logger.info(f"No model files found for profile={safe_profile}")
                else:
                    logger.info(f"No objects found for profile={safe_profile}")
                    
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                logger.error(f"Error scanning models for profile={safe_profile}: {error_msg}")
        
        return available_models
        
    except Exception as e:
        logger.error(f"Error scanning for profile models: {str(e)}")
        return {}

def register_models_with_sagemaker(available_models: Dict[str, Dict[str, Any]], config: Dict[str, Any], 
                                 training_metadata: Dict[str, Any], execution_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Register models with SageMaker Model Registry
    """
    
    registration_results = {}
    
    # Get execution role from environment
    execution_role = os.environ.get('SAGEMAKER_EXECUTION_ROLE')
    if not execution_role:
        raise ValueError("SAGEMAKER_EXECUTION_ROLE environment variable is required")
    
    for profile, model_info in available_models.items():
        try:
            safe_profile = sanitize_for_logging(profile, 20)
            logger.info(f"Registering model for profile={safe_profile}")
            
            # Create unique model name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"energy-forecasting-{profile.lower()}-{timestamp}-{execution_id[:8]}"
            
            # Create inference image URI (this would be your custom inference container)
            account_id = config['account_id']
            region = config['region']
            inference_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-prediction:latest"
            
            # Create the model
            sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': inference_image,
                    'ModelDataUrl': model_info['s3_path'],
                    'Environment': {
                        'PROFILE': profile,
                        'MODEL_VERSION': timestamp,
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                    }
                },
                ExecutionRoleArn=execution_role
            )
            
            registration_results[profile] = {
                'status': 'success',
                'model_name': model_name,
                'profile': profile,
                's3_path': model_info['s3_path'],
                'inference_image': inference_image,
                'timestamp': datetime.now().isoformat()
            }
            
            safe_model_name = sanitize_for_logging(model_name, 100)
            logger.info(f"Successfully registered model profile={safe_profile} model_name={safe_model_name}")
            
        except Exception as e:
            error_msg = sanitize_for_logging(str(e), 100)
            safe_profile = sanitize_for_logging(profile, 20)
            logger.error(f"Failed to register model profile={safe_profile}: {error_msg}")
            
            registration_results[profile] = {
                'status': 'failed',
                'error': error_msg,
                'profile': profile,
                'timestamp': datetime.now().isoformat()
            }
    
    return registration_results

def create_model_packages(registration_results: Dict[str, Dict[str, Any]], config: Dict[str, Any], 
                        execution_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Create SageMaker Model Packages for approved models
    """
    
    model_package_results = {}
    model_package_group_name = "energy-forecasting-models"
    
    # Ensure model package group exists
    try:
        sagemaker_client.describe_model_package_group(ModelPackageGroupName=model_package_group_name)
    except sagemaker_client.exceptions.ClientError:
        # Create model package group if it doesn't exist
        sagemaker_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription="Energy Forecasting Model Package Group"
        )
        logger.info(f"Created model package group: {model_package_group_name}")
    
    for profile, reg_result in registration_results.items():
        if reg_result.get('status') != 'success':
            model_package_results[profile] = {
                'status': 'skipped',
                'reason': 'registration_failed',
                'profile': profile
            }
            continue
        
        try:
            safe_profile = sanitize_for_logging(profile, 20)
            logger.info(f"Creating model package for profile={safe_profile}")
            
            model_name = reg_result['model_name']
            inference_image = reg_result['inference_image']
            s3_path = reg_result['s3_path']
            
            # Create model package
            response = sagemaker_client.create_model_package(
                ModelPackageGroupName=model_package_group_name,
                ModelPackageDescription=f"Energy Forecasting Model for {profile} Profile",
                InferenceSpecification={
                    'Containers': [
                        {
                            'Image': inference_image,
                            'ModelDataUrl': s3_path,
                            'Environment': {
                                'PROFILE': profile,
                                'MODEL_VERSION': reg_result['timestamp'],
                                'SAGEMAKER_PROGRAM': 'inference.py',
                                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                            }
                        }
                    ],
                    'SupportedContentTypes': ['application/json'],
                    'SupportedResponseMIMETypes': ['application/json'],
                    'SupportedRealtimeInferenceInstanceTypes': ['ml.t2.medium', 'ml.m5.large'],
                    'SupportedTransformInstanceTypes': ['ml.m5.large']
                },
                ModelApprovalStatus='Approved'
            )
            
            model_package_arn = response['ModelPackageArn']
            
            model_package_results[profile] = {
                'status': 'success',
                'model_package_arn': model_package_arn,
                'model_name': model_name,
                'profile': profile,
                'inference_image': inference_image,
                's3_path': s3_path,
                'timestamp': datetime.now().isoformat()
            }
            
            safe_package_arn = sanitize_for_logging(model_package_arn, 100)
            logger.info(f"Successfully created model package profile={safe_profile} arn={safe_package_arn}")
            
        except Exception as e:
            error_msg = sanitize_for_logging(str(e), 100)
            safe_profile = sanitize_for_logging(profile, 20)
            logger.error(f"Failed to create model package profile={safe_profile}: {error_msg}")
            
            model_package_results[profile] = {
                'status': 'failed',
                'error': error_msg,
                'profile': profile,
                'timestamp': datetime.now().isoformat()
            }
    
    return model_package_results

def store_registry_metadata(model_package_results: Dict[str, Dict[str, Any]], config: Dict[str, Any], 
                          training_metadata: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Store model registry metadata in S3
    """
    
    try:
        # Create comprehensive metadata
        registry_metadata = {
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat(),
            'model_package_results': model_package_results,
            'training_metadata': training_metadata,
            'config_summary': {
                'model_bucket': config['model_bucket'],
                'data_bucket': config['data_bucket'],
                'region': config['region'],
                'profiles_processed': config['profiles']
            },
            'summary': {
                'total_profiles': len(model_package_results),
                'successful_packages': len([r for r in model_package_results.values() if r.get('status') == 'success']),
                'failed_packages': len([r for r in model_package_results.values() if r.get('status') == 'failed'])
            }
        }
        
        # Store in S3
        data_bucket = config['data_bucket']
        registry_prefix = config['registry_prefix']
        current_date = datetime.now().strftime("%Y%m%d")
        
        s3_key = f"{registry_prefix}model_registry_{current_date}_{execution_id[:8]}.json"
        
        s3_client.put_object(
            Bucket=data_bucket,
            Key=s3_key,
            Body=json.dumps(registry_metadata, indent=2, default=str),
            ContentType='application/json'
        )
        
        safe_bucket = sanitize_for_logging(data_bucket, 100)
        safe_key = sanitize_for_logging(s3_key, 200)
        logger.info(f"Stored registry metadata bucket={safe_bucket} key={safe_key}")
        
        return {
            'status': 'success',
            's3_bucket': data_bucket,
            's3_key': s3_key,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        logger.error(f"Failed to store registry metadata: {error_msg}")
        
        return {
            'status': 'failed',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

def create_inference_script(profile: str) -> str:
    """
    Create a sample inference script for the model
    This would be embedded in your container or stored separately
    """
    
    inference_script = f'''
"""
Energy Forecasting Inference Script for {profile} Profile
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Model configuration
PROFILE = "{profile}"
MODEL_VERSION = "{{model_version}}"
EXPECTED_FEATURES = [
    "load", "year", "month", "day", "hour", "dayofweek", 
    "is_weekend", "is_holiday", "season", "temperature", 
    "humidity", "cloud_cover"
]

# Add radiation for RN profile
if PROFILE == "RN":
    EXPECTED_FEATURES.append("shortwave_radiation")

def model_fn(model_dir):
    """Load the model from the model directory"""
    try:
        logger.info(f"Loading {{PROFILE}} model from {{model_dir}}")
        model = joblib.load(f"{{model_dir}}/model.pkl")
        logger.info(f"Successfully loaded {{PROFILE}} model")
        return model
    except Exception as e:
        logger.error(f"Failed to load {{PROFILE}} model: {{str(e)}}")
        raise

def input_fn(request_body, content_type):
    """Parse input data for inference"""
    try:
        logger.info(f"Processing input for {{PROFILE}} model, content_type={{content_type}}")
        
        if content_type == 'application/json':
            input_data = json.loads(request_body)
            
            # Handle both formats: {{"instances": [[...]]}} and [[...]]
            if isinstance(input_data, dict) and "instances" in input_data:
                instances = input_data["instances"]
            elif isinstance(input_data, list):
                instances = input_data
            else:
                raise ValueError(f"Invalid input format for {{PROFILE}}")
            
            # Convert to numpy array
            input_array = np.array(instances)
            
            # Validate feature count
            expected_features = len(EXPECTED_FEATURES)
            if input_array.shape[1] != expected_features:
                raise ValueError(f"{{PROFILE}} model expects {{expected_features}} features, got {{input_array.shape[1]}}")
            
            logger.info(f"Successfully processed input for {{PROFILE}}: {{input_array.shape}}")
            return input_array
            
        else:
            raise ValueError(f"Unsupported content type: {{content_type}}")
            
    except Exception as e:
        logger.error(f"Input processing failed for {{PROFILE}}: {{str(e)}}")
        raise

def predict_fn(input_data, model):
    """Run inference on the input data"""
    try:
        logger.info(f"Running prediction for {{PROFILE}} model")
        logger.info(f"Input shape: {{input_data.shape}}")
        
        # Run prediction
        predictions = model.predict(input_data)
        
        # Validate predictions
        if len(predictions) != len(input_data):
            raise ValueError(f"Prediction count mismatch for {{PROFILE}}: expected {{len(input_data)}}, got {{len(predictions)}}")
        
        # Check for invalid predictions
        if np.isnan(predictions).any():
            logger.warning("Some predictions are NaN, replacing with median")
            median_pred = np.nanmedian(predictions)
            predictions = np.where(np.isnan(predictions), median_pred, predictions)
        
        if (predictions < 0).any():
            logger.warning("Some predictions are negative, clipping to zero")
            predictions = np.maximum(predictions, 0)
        
        logger.info(f"Enhanced prediction completed: {{len(predictions)}} predictions generated")
        logger.info(f"Prediction range: {{predictions.min():.4f}} to {{predictions.max():.4f}}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Enhanced prediction failed: {{str(e)}}")
        raise

def output_fn(prediction, accept):
    """Enhanced output formatting with comprehensive metadata"""
    try:
        logger.info(f"Formatting enhanced output for {{len(prediction)}} predictions")
        
        if accept == 'application/json':
            # Convert numpy arrays to lists for JSON serialization
            if hasattr(prediction, 'tolist'):
                prediction_list = prediction.tolist()
            else:
                prediction_list = list(prediction)
            
            # Enhanced response with comprehensive metadata
            response = {{
                "predictions": prediction_list,
                "metadata": {{
                    "profile": PROFILE,
                    "model_version": MODEL_VERSION,
                    "prediction_count": len(prediction_list),
                    "timestamp": datetime.now().isoformat(),
                    "statistics": {{
                        "min": float(min(prediction_list)),
                        "max": float(max(prediction_list)),
                        "mean": float(sum(prediction_list) / len(prediction_list)),
                        "total": float(sum(prediction_list))
                    }},
                    "expected_features": EXPECTED_FEATURES,
                    "feature_count": len(EXPECTED_FEATURES)
                }}
            }}
            
            logger.info("Enhanced output formatting completed")
            return json.dumps(response)
            
        else:
            raise ValueError(f"Unsupported accept type: {{accept}}")
            
    except Exception as e:
        logger.error(f"Enhanced output formatting failed: {{str(e)}}")
        raise
'''
    
    return inference_script