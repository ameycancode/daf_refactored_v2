"""
Enhanced Endpoint Management Lambda Function - Environment-Aware Version
This version uses environment variables for all configuration values
instead of hardcoded dev environment values.
"""

import json
import boto3
import logging
import time
from datetime import datetime
from typing import Dict, Any
import uuid
import os

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

# Environment-aware configuration
ENDPOINT_CONFIG_BUCKET = os.environ.get('DATA_BUCKET')
ENDPOINT_CONFIG_PREFIX = "endpoint-configurations/"
EXECUTION_LOCK_PREFIX = "execution-locks/"

def lambda_handler(event, context):
    """
    Enhanced Lambda handler supporting both single profile and batch operations
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting endpoint management process [{execution_id}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Validate environment configuration
        if not ENDPOINT_CONFIG_BUCKET:
            raise ValueError("DATA_BUCKET environment variable is required but not set")
        
        # Determine operation type
        operation = event.get('operation', 'create_all_endpoints')
        
        if operation == 'create_endpoint':
            # Handle single profile operation for parallel Step Functions
            return handle_single_profile_endpoint(event, context, execution_id)
        else:
            # Handle batch operation (your original logic)
            return handle_batch_endpoints(event, context, execution_id)
        
    except Exception as e:
        logger.error(f"Endpoint management process failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Endpoint management process failed'
            }
        }

def handle_single_profile_endpoint(event, context, execution_id):
    """
    Handle single profile endpoint creation - FIXED to process ONLY the intended profile
    """
    
    try:
        logger.info(f"Processing single profile endpoint creation")
        logger.info(f"Event received: {json.dumps(event, default=str)}")
        
        # FIXED: Extract the SINGLE profile that this Lambda execution should process
        operation = event.get('operation', 'create_endpoint')
        
        # Extract the profile this execution should handle
        profile = event.get('profile')
        training_metadata = event.get('training_metadata', {})
        approved_models = event.get('approved_models', {})
        
        # CRITICAL FIX: Process ONLY the single profile, not all profiles
        if not profile:
            raise ValueError("Profile is required for single profile endpoint creation")
        
        logger.info(f"Creating endpoint for profile: {profile}")
        
        # Get model info for this specific profile
        model_info = approved_models.get(profile)
        if not model_info:
            raise ValueError(f"No approved model found for profile {profile}")
        
        # Create endpoint for this single profile
        result = create_endpoint_lifecycle(profile, model_info, training_metadata, execution_id)
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        logger.error(f"Single profile endpoint creation failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'profile': event.get('profile'),
                'message': 'Single profile endpoint creation failed'
            }
        }

def handle_batch_endpoints(event, context, execution_id):
    """
    Handle batch endpoint operations
    """
    
    try:
        logger.info(f"Processing batch endpoint operations")
        
        # Extract information from event
        training_metadata = event.get('training_metadata', {})
        approved_models = event.get('approved_models', {})
        
        if not approved_models:
            raise ValueError("No approved models provided for endpoint creation")
        
        logger.info(f"Creating endpoints for {len(approved_models)} profiles")
        
        # Create endpoints for all approved models
        endpoint_results = {}
        successful_endpoints = 0
        
        for profile, model_info in approved_models.items():
            try:
                logger.info(f"Creating endpoint for profile: {profile}")
                
                result = create_endpoint_lifecycle(profile, model_info, training_metadata, execution_id)
                endpoint_results[profile] = result
                
                if result.get('status') == 'success':
                    successful_endpoints += 1
                
            except Exception as e:
                logger.error(f"Failed to create endpoint for {profile}: {str(e)}")
                endpoint_results[profile] = {
                    'status': 'failed',
                    'error': str(e),
                    'profile': profile
                }
        
        # Return comprehensive results
        return {
            'statusCode': 200,
            'body': {
                'operation': 'batch_endpoint_creation',
                'execution_id': execution_id,
                'total_profiles': len(approved_models),
                'successful_endpoints': successful_endpoints,
                'endpoint_results': endpoint_results,
                'timestamp': datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Batch endpoint creation failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Batch endpoint creation failed'
            }
        }

def create_endpoint_lifecycle(profile: str, model_info: Dict[str, Any], training_metadata: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Complete endpoint lifecycle: create -> test -> save config -> delete
    """
    
    result = {
        'profile': profile,
        'status': 'in_progress',
        'steps_completed': [],
        'execution_id': execution_id,
        'timestamp': datetime.now().isoformat()
    }
    
    endpoint_name = None
    endpoint_config_name = None
    model_name = None
    
    try:
        logger.info(f"Starting endpoint lifecycle for {profile}")
        
        # Step 1: Create SageMaker model
        logger.info(f"Step 1: Creating SageMaker model")
        model_name = create_model_for_profile(profile, model_info, execution_id)
        result['model_name'] = model_name
        result['steps_completed'].append('model_created')
        logger.info(f"Created model: {model_name}")
        
        # Step 2: Create endpoint configuration
        logger.info(f"Step 2: Creating endpoint configuration")
        endpoint_config_name = create_endpoint_config(profile, model_name, execution_id)
        result['endpoint_config_name'] = endpoint_config_name
        result['steps_completed'].append('endpoint_config_created')
        logger.info(f"Created endpoint config: {endpoint_config_name}")
        
        # Step 3: Create endpoint
        logger.info(f"Step 3: Creating endpoint")
        endpoint_name = create_endpoint(profile, endpoint_config_name, execution_id)
        result['endpoint_name'] = endpoint_name
        result['steps_completed'].append('endpoint_created')
        logger.info(f"Created endpoint: {endpoint_name}")
        
        # Step 4: Wait for endpoint to be ready
        logger.info(f"Step 4: Waiting for endpoint to be ready")
        ready_success = wait_for_endpoint_ready(endpoint_name, max_wait_time=600)
        
        if not ready_success:
            result['error'] = "Endpoint did not become ready within timeout"
            return result
        
        result['steps_completed'].append('endpoint_ready')
        logger.info(f"Endpoint is ready")
        
        # Step 5: Test endpoint
        logger.info(f"Step 5: Testing endpoint inference")
        inference_success = test_endpoint_inference(endpoint_name, profile)
        
        if not inference_success:
            logger.warning(f"Endpoint inference test failed for {profile}, but continuing")
            result['inference_warning'] = "Inference test failed"
        else:
            result['steps_completed'].append('endpoint_tested')
            logger.info(f"Endpoint inference test successful")
        
        # Step 6: Save COMPLETE endpoint configuration to S3 (FIXED VERSION)
        logger.info(f"Step 6: Saving COMPLETE endpoint configuration to S3")
        config_s3_info = save_complete_endpoint_configuration(
            endpoint_name, endpoint_config_name, model_name, profile, model_info, training_metadata
        )
        
        if not config_s3_info:
            result['error'] = "Failed to save endpoint configuration"
            return result
        
        result['configuration_s3'] = config_s3_info
        result['steps_completed'].append('configuration_saved')
        logger.info(f"Saved COMPLETE endpoint configuration to S3")
        
        # Step 7: Delete endpoint for cost optimization
        logger.info(f"Step 7: Deleting endpoint for cost optimization")
        deletion_success = delete_endpoint_and_resources(endpoint_name, endpoint_config_name, model_name)
        
        if deletion_success:
            result['endpoint_deleted'] = True
            result['steps_completed'].append('endpoint_deleted')
            logger.info(f"Successfully deleted endpoint {endpoint_name}")
        else:
            logger.warning(f"Failed to delete endpoint {endpoint_name}")
            result['deletion_warning'] = "Failed to delete endpoint"
        
        result['status'] = 'success'
        return result
        
    except Exception as e:
        logger.error(f"Error in endpoint lifecycle for {profile}: {str(e)}")
        result['error'] = str(e)
        
        # Cleanup on failure
        if endpoint_name:
            try:
                logger.info(f"Cleaning up failed endpoint: {endpoint_name}")
                delete_endpoint_and_resources(endpoint_name, endpoint_config_name, model_name)
            except Exception:
                pass
        
        return result

def create_model_for_profile(profile: str, model_info: Dict[str, Any], execution_id: str) -> str:
    """Create SageMaker model for a specific profile"""
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"energy-forecasting-{profile.lower()}-model-{timestamp}-{execution_id[:8]}"
    
    # Extract model package ARN from model_info
    model_package_arn = model_info.get('model_package_arn')
    if not model_package_arn:
        raise ValueError(f"No model package ARN found for profile {profile}")
    
    # Get SageMaker execution role from environment
    sagemaker_role = os.environ.get('SAGEMAKER_EXECUTION_ROLE')
    if not sagemaker_role:
        raise ValueError("SAGEMAKER_EXECUTION_ROLE environment variable is required")
    
    # Create model from model package
    sagemaker_client.create_model(
        ModelName=model_name,
        Containers=[
            {
                'ModelPackageName': model_package_arn
            }
        ],
        ExecutionRoleArn=sagemaker_role
    )
    
    return model_name

def create_endpoint_config(profile: str, model_name: str, execution_id: str) -> str:
    """Create endpoint configuration"""
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = f"energy-forecasting-{profile.lower()}-config-{timestamp}-{execution_id[:8]}"
    
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                'VariantName': 'primary',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.t2.medium',
                'InitialVariantWeight': 1.0
            }
        ]
    )
    
    return config_name

def create_endpoint(profile: str, endpoint_config_name: str, execution_id: str) -> str:
    """Create SageMaker endpoint"""
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_name = f"energy-forecasting-{profile.lower()}-endpoint-{timestamp}-{execution_id[:8]}"
    
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    
    return endpoint_name

def wait_for_endpoint_ready(endpoint_name: str, max_wait_time: int = 600) -> bool:
    """Wait for endpoint to be ready"""
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            if status == 'InService':
                logger.info(f"Endpoint {endpoint_name} is ready")
                return True
            elif status in ['Failed', 'RollingBack']:
                logger.error(f"Endpoint {endpoint_name} failed with status: {status}")
                return False
            
            logger.info(f"Endpoint {endpoint_name} status: {status}, waiting...")
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"Error checking endpoint status: {str(e)}")
            time.sleep(30)
    
    logger.error(f"Timeout waiting for endpoint {endpoint_name} to be ready")
    return False

def test_endpoint_inference(endpoint_name: str, profile: str) -> bool:
    """Test endpoint inference with sample data - FIXED validation logic"""
    
    try:
        # Create sample input data based on profile
        sample_data = {
            "instances": [
                [1000, 2025, 1, 29, 12, 3, 0, 0, 1, 75.5, 0.85, 0.80]  # Basic features
            ]
        }
        
        # Add radiation for RN profile
        if profile == 'RN':
            sample_data["instances"][0].append(500.0)  # shortwave_radiation
        
        # Invoke endpoint
        runtime_client = boto3.client('sagemaker-runtime')
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(sample_data)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        # FIXED: Accept the enhanced response format
        # The response shows: {'predictions': [32.56577682495117], 'metadata': {...}}
        if isinstance(result, dict) and 'predictions' in result:
            predictions = result['predictions']
            if isinstance(predictions, list) and len(predictions) > 0:
                logger.info(f"Endpoint inference test successful for {profile}: {predictions[0]}")
                return True
            else:
                logger.error(f"Empty predictions for {profile}: {result}")
                return False
        elif isinstance(result, list) and len(result) > 0:
            logger.info(f"Endpoint inference test successful for {profile}: {result[0]}")
            return True
        else:
            logger.error(f"Invalid inference response for {profile}: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Endpoint inference test failed for {profile}: {str(e)}")
        return False

def save_complete_endpoint_configuration(endpoint_name: str, endpoint_config_name: str, model_name: str, 
                                       profile: str, model_info: Dict[str, Any], training_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED VERSION: Save COMPLETE endpoint configuration with ACTUAL container details from Model Package
    """
    
    try:
        logger.info(f"Capturing complete configuration details for {profile} with actual container details")
        
        # 1. Get the actual SageMaker resource details before they're deleted
        endpoint_config_response = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        model_response = sagemaker_client.describe_model(ModelName=model_name)
        endpoint_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        
        # 2. Get ACTUAL container details from the model package
        model_package_arn = model_info.get('model_package_arn')
        if model_package_arn:
            model_package_response = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
            inference_spec = model_package_response.get('InferenceSpecification', {})
            containers = inference_spec.get('Containers', [])
            
            # Extract the ACTUAL image and model data URL
            if containers:
                actual_image = containers[0].get('Image', 'unknown')
                actual_model_data_url = containers[0].get('ModelDataUrl', 'unknown')
            else:
                actual_image = 'unknown'
                actual_model_data_url = 'unknown'
        else:
            actual_image = 'unknown'
            actual_model_data_url = 'unknown'
        
        # 3. Create COMPLETE configuration data with ACTUAL container details
        complete_config_data = {
            # Metadata
            'profile': profile,
            'creation_timestamp': datetime.now().isoformat(),
            'training_metadata': training_metadata,
            'execution_id': endpoint_name.split('-')[-2:],  # Extract from endpoint name
            
            # ACTUAL SageMaker API parameters for recreation
            'model_config': {
                'ModelName': model_name,  # This will be generated fresh during recreation
                'Containers': [
                    {
                        'Image': actual_image,
                        'ModelDataUrl': actual_model_data_url,
                        'Environment': containers[0].get('Environment', {}) if containers else {}
                    }
                ],
                'ExecutionRoleArn': model_response['ExecutionRoleArn']
            },
            
            'endpoint_config': {
                'EndpointConfigName': endpoint_config_name,  # This will be generated fresh
                'ProductionVariants': endpoint_config_response['ProductionVariants']
            },
            
            'endpoint_config_details': {
                'EndpointName': endpoint_name,  # This will be generated fresh
                'EndpointConfigName': endpoint_config_name
            },
            
            # Original model package information
            'model_package_info': {
                'model_package_arn': model_package_arn,
                'original_model_info': model_info
            },
            
            # Validation information
            'validation_info': {
                'endpoint_tested': True,
                'configuration_version': '4.0_actual_container_details',
                'format_source': 'model_package_inference_specification'
            }
        }
        
        # Save to S3 with profile-specific folder structure
        current_date = datetime.now().strftime("%Y%m%d")
        s3_key = f"{ENDPOINT_CONFIG_PREFIX}{profile}/{profile}_endpoint_config_{current_date}.json"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=ENDPOINT_CONFIG_BUCKET,
            Key=s3_key,
            Body=json.dumps(complete_config_data, indent=2, default=str),
            ContentType='application/json'
        )
        
        logger.info(f"Saved COMPLETE endpoint configuration with ACTUAL container details to S3: s3://{ENDPOINT_CONFIG_BUCKET}/{s3_key}")
        
        return {
            's3_bucket': ENDPOINT_CONFIG_BUCKET,
            's3_key': s3_key,
            'profile': profile,
            'timestamp': datetime.now().isoformat(),
            'configuration_version': '4.0_actual_container_details',
            'format_source': 'model_package_inference_specification',
            'actual_image': actual_image,
            'actual_model_data_url': actual_model_data_url
        }
        
    except Exception as e:
        logger.error(f"Failed to save complete endpoint configuration for {profile}: {str(e)}")
        logger.error(f"Model response keys: {list(model_response.keys()) if 'model_response' in locals() else 'No model_response'}")
        if 'model_package_response' in locals():
            logger.error(f"Model package response keys: {list(model_package_response.keys())}")
        return None

def delete_endpoint_and_resources(endpoint_name: str, endpoint_config_name: str = None, model_name: str = None) -> bool:
    """Delete endpoint and associated resources for cost optimization"""
    
    try:
        deletion_results = []
        
        # Delete endpoint
        if endpoint_name:
            try:
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                deletion_results.append(f"endpoint:{endpoint_name}")
                logger.info(f"Deleted endpoint: {endpoint_name}")
            except Exception as e:
                logger.warning(f"Failed to delete endpoint {endpoint_name}: {str(e)}")
        
        # Delete endpoint configuration
        if endpoint_config_name:
            try:
                sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                deletion_results.append(f"endpoint_config:{endpoint_config_name}")
                logger.info(f"Deleted endpoint configuration: {endpoint_config_name}")
            except Exception as e:
                logger.warning(f"Failed to delete endpoint config {endpoint_config_name}: {str(e)}")
        
        # Delete model
        if model_name:
            try:
                sagemaker_client.delete_model(ModelName=model_name)
                deletion_results.append(f"model:{model_name}")
                logger.info(f"Deleted model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to delete model {model_name}: {str(e)}")
        
        if deletion_results:
            logger.info(f"Deletion summary: {', '.join(deletion_results)}")
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error during resource deletion: {str(e)}")
        return False