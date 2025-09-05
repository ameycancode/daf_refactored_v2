"""
Profile Endpoint Creator Lambda Function - Environment-Aware Version
This version uses environment variables for all configuration values.
"""

import json
import boto3
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
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
    sanitized = ''.join(c for c in value if c.isalnum() or c in '-_.')[:max_length]
    return sanitized if sanitized else 'unknown'

def sanitize_event_for_logging(event: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize event data for logging"""
    sanitized_event = {}
    
    safe_keys = ['operation', 'profile']
    for key in safe_keys:
        if key in event:
            sanitized_event[key] = sanitize_for_logging(str(event[key]), 100)
    
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_profile': 'profile' in event,
        'has_s3_config_path': 's3_config_path' in event
    }
    
    return sanitized_event

def lambda_handler(event, context):
    """
    Main Lambda handler for profile endpoint creation
    
    Expected event structure:
    {
        "operation": "create_endpoint",
        "profile": "RNN",
        "s3_config_path": "s3://<data_bucket>/endpoint-configurations/RNN/config.json",
        "data_bucket": "<from environment variable DATA_BUCKET>"
    }
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting profile endpoint management [execution_id={sanitize_for_logging(execution_id)}]")
        
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
        
        # Extract operation and profile details
        operation = event.get('operation', 'create_endpoint')
        profile = event.get('profile')
        
        if not profile:
            raise ValueError("Profile is required for endpoint operations")
        
        # SECURITY FIX: Sanitize profile name before logging
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Processing operation={sanitize_for_logging(operation)} profile={safe_profile}")
        
        # Handle different operations
        if operation == 'create_endpoint':
            result = create_endpoint_from_s3_config(event, execution_id)
        elif operation == 'check_endpoint_status':
            result = check_endpoint_status(event, execution_id)
        else:
            raise ValueError(f"Unknown operation: {sanitize_for_logging(operation)}")
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Profile endpoint management failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'profile': event.get('profile'),
                'message': 'Profile endpoint management failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def create_endpoint_from_s3_config(event: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Create SageMaker endpoint from S3 configuration for a specific profile
    Creates an exact replica of the previously deleted endpoint
    """
    
    try:
        profile = event['profile']
        s3_config_path = event.get('s3_config_path')
        
        # Environment-aware bucket configuration
        data_bucket = event.get('data_bucket') or os.environ.get('DATA_BUCKET')
        if not data_bucket:
            raise ValueError("DATA_BUCKET must be provided in event or environment variables")
        
        # SECURITY FIX: Sanitize values for logging
        safe_profile = sanitize_for_logging(profile, 50)
        safe_s3_path = sanitize_for_logging(s3_config_path, 200) if s3_config_path else "default"
        
        logger.info(f"Creating endpoint profile={safe_profile} s3_path={safe_s3_path}")
        
        # Load configuration from S3
        if s3_config_path:
            # Extract bucket and key from S3 path
            s3_parts = s3_config_path.replace('s3://', '').split('/', 1)
            bucket = s3_parts[0]
            key = s3_parts[1]
        else:
            # Default path for profile-specific configurations
            bucket = data_bucket
            key = f"endpoint-configurations/{profile}/endpoint_config.json"
        
        safe_bucket = sanitize_for_logging(bucket, 100)
        safe_key = sanitize_for_logging(key, 200)
        logger.info(f"Loading config from bucket={safe_bucket} key={safe_key}")
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        config_data = json.loads(response['Body'].read())
        
        logger.info("Successfully loaded endpoint configuration from S3")
        
        # Generate unique names for new resources
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_suffix = f"{timestamp}-{execution_id[:8]}"
        
        # Generate new resource names
        new_model_name = f"energy-forecasting-{profile.lower()}-model-{unique_suffix}"
        new_endpoint_config_name = f"energy-forecasting-{profile.lower()}-config-{unique_suffix}"
        new_endpoint_name = f"energy-forecasting-{profile.lower()}-endpoint-{unique_suffix}"
        
        # SECURITY FIX: Sanitize resource names for logging
        safe_model_name = sanitize_for_logging(new_model_name, 100)
        safe_endpoint_config_name = sanitize_for_logging(new_endpoint_config_name, 100)
        safe_endpoint_name = sanitize_for_logging(new_endpoint_name, 100)
        
        logger.info(f"Generated resource names model={safe_model_name} config={safe_endpoint_config_name} endpoint={safe_endpoint_name}")
        
        # Step 1: Create Model using the correct approach based on available configuration
        logger.info("Step 1: Creating SageMaker Model...")
        model_config = config_data.get('model_config', {})
        
        # Get SageMaker execution role from environment
        sagemaker_role = os.environ.get('SAGEMAKER_EXECUTION_ROLE')
        if not sagemaker_role:
            raise ValueError("SAGEMAKER_EXECUTION_ROLE environment variable is required")
        
        if not model_config:
            raise ValueError("No model configuration found in S3 config")
        
        # Extract container information
        containers = model_config.get('Containers', [])
        if not containers:
            raise ValueError("No container configuration found")
        
        primary_container = containers[0]
        
        # Create the model
        sagemaker_client.create_model(
            ModelName=new_model_name,
            PrimaryContainer={
                'Image': primary_container.get('Image'),
                'ModelDataUrl': primary_container.get('ModelDataUrl'),
                'Environment': primary_container.get('Environment', {})
            },
            ExecutionRoleArn=sagemaker_role
        )
        
        logger.info(f"Created SageMaker model: {safe_model_name}")
        
        # Step 2: Create Endpoint Configuration
        logger.info("Step 2: Creating Endpoint Configuration...")
        endpoint_config = config_data.get('endpoint_config', {})
        
        if not endpoint_config:
            raise ValueError("No endpoint configuration found in S3 config")
        
        # Extract production variants and update with new model name
        production_variants = endpoint_config.get('ProductionVariants', [])
        if not production_variants:
            raise ValueError("No production variants found in endpoint config")
        
        # Update the variant to use our new model
        updated_variants = []
        for variant in production_variants:
            updated_variant = variant.copy()
            updated_variant['ModelName'] = new_model_name
            updated_variants.append(updated_variant)
        
        # Create endpoint configuration
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=new_endpoint_config_name,
            ProductionVariants=updated_variants
        )
        
        logger.info(f"Created endpoint configuration: {safe_endpoint_config_name}")
        
        # Step 3: Create Endpoint
        logger.info("Step 3: Creating Endpoint...")
        sagemaker_client.create_endpoint(
            EndpointName=new_endpoint_name,
            EndpointConfigName=new_endpoint_config_name
        )
        
        logger.info(f"Created endpoint: {safe_endpoint_name}")
        
        # Step 4: Wait for endpoint to be ready (optional immediate check)
        logger.info("Step 4: Checking initial endpoint status...")
        initial_status = check_endpoint_immediate_status(new_endpoint_name)
        
        # Return success result
        result = {
            'operation': 'create_endpoint',
            'status': 'success',
            'profile': profile,
            'endpoint_name': new_endpoint_name,
            'endpoint_config_name': new_endpoint_config_name,
            'model_name': new_model_name,
            'initial_status': initial_status,
            'source_config': config_data.get('creation_timestamp', 'unknown'),
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Endpoint creation initiated successfully profile={safe_profile}")
        return result
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(event.get('profile', 'unknown'), 50)
        logger.error(f"Endpoint creation failed profile={safe_profile} error={error_msg}")
        
        return {
            'operation': 'create_endpoint',
            'status': 'failed',
            'profile': event.get('profile'),
            'error': error_msg,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }

def check_endpoint_immediate_status(endpoint_name: str) -> Dict[str, Any]:
    """
    Check immediate status of endpoint (non-blocking)
    """
    
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        
        status_info = {
            'endpoint_status': response['EndpointStatus'],
            'creation_time': response['CreationTime'].isoformat(),
            'last_modified_time': response['LastModifiedTime'].isoformat(),
            'check_timestamp': datetime.now().isoformat()
        }
        
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.info(f"Endpoint status check endpoint={safe_endpoint_name} status={response['EndpointStatus']}")
        
        return status_info
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 100)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.warning(f"Could not check endpoint status endpoint={safe_endpoint_name} error={error_msg}")
        
        return {
            'endpoint_status': 'unknown',
            'error': error_msg,
            'check_timestamp': datetime.now().isoformat()
        }

def check_endpoint_status(event: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Check the status of an endpoint
    """
    
    try:
        profile = event.get('profile')
        endpoint_name = event.get('endpoint_name')
        
        if not endpoint_name:
            # Try to find endpoint by profile
            endpoint_name = find_endpoint_by_profile(profile)
            
        if not endpoint_name:
            return {
                'operation': 'check_endpoint_status',
                'status': 'not_found',
                'profile': profile,
                'message': 'No endpoint found for profile',
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat()
            }
        
        safe_profile = sanitize_for_logging(profile, 50)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.info(f"Checking endpoint status profile={safe_profile} endpoint={safe_endpoint_name}")
        
        # Get detailed endpoint status
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        
        # Get endpoint configuration details
        config_name = response['EndpointConfigName']
        config_response = sagemaker_client.describe_endpoint_config(EndpointConfigName=config_name)
        
        status_result = {
            'operation': 'check_endpoint_status',
            'status': 'success',
            'profile': profile,
            'endpoint_name': endpoint_name,
            'endpoint_status': response['EndpointStatus'],
            'endpoint_arn': response['EndpointArn'],
            'creation_time': response['CreationTime'].isoformat(),
            'last_modified_time': response['LastModifiedTime'].isoformat(),
            'endpoint_config': {
                'name': config_name,
                'production_variants': config_response.get('ProductionVariants', [])
            },
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add failure reason if endpoint failed
        if response['EndpointStatus'] == 'Failed' and 'FailureReason' in response:
            status_result['failure_reason'] = response['FailureReason']
        
        logger.info(f"Endpoint status retrieved profile={safe_profile} status={response['EndpointStatus']}")
        return status_result
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(event.get('profile', 'unknown'), 50)
        logger.error(f"Endpoint status check failed profile={safe_profile} error={error_msg}")
        
        return {
            'operation': 'check_endpoint_status',
            'status': 'error',
            'profile': event.get('profile'),
            'endpoint_name': event.get('endpoint_name'),
            'error': error_msg,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }

def find_endpoint_by_profile(profile: str) -> Optional[str]:
    """
    Find the most recent endpoint for a profile
    """
    
    try:
        if not profile:
            return None
        
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Searching for endpoint profile={safe_profile}")
        
        # List endpoints that match the profile naming pattern
        endpoints = sagemaker_client.list_endpoints(
            StatusEquals='InService',
            NameContains=f"energy-forecasting-{profile.lower()}"
        )
        
        if not endpoints['Endpoints']:
            # Also try with different status
            endpoints = sagemaker_client.list_endpoints(
                NameContains=f"energy-forecasting-{profile.lower()}"
            )
        
        if endpoints['Endpoints']:
            # Get the most recent endpoint
            latest_endpoint = max(endpoints['Endpoints'], key=lambda x: x['CreationTime'])
            endpoint_name = latest_endpoint['EndpointName']
            
            safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
            logger.info(f"Found endpoint profile={safe_profile} endpoint={safe_endpoint_name}")
            
            return endpoint_name
        else:
            logger.info(f"No endpoints found profile={safe_profile}")
            return None
            
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 100)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Error finding endpoint profile={safe_profile} error={error_msg}")
        return None

def wait_for_endpoint_ready(endpoint_name: str, max_wait_time: int = 600) -> Dict[str, Any]:
    """
    Wait for endpoint to be ready (blocking operation)
    """
    
    start_time = time.time()
    
    try:
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.info(f"Waiting for endpoint to be ready endpoint={safe_endpoint_name}")
        
        while time.time() - start_time < max_wait_time:
            try:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                if status == 'InService':
                    wait_time = time.time() - start_time
                    logger.info(f"Endpoint ready endpoint={safe_endpoint_name} wait_time={wait_time:.1f}s")
                    
                    return {
                        'status': 'ready',
                        'endpoint_status': status,
                        'wait_time_seconds': int(wait_time),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                elif status in ['Failed', 'RollingBack']:
                    failure_reason = response.get('FailureReason', 'Unknown failure')
                    logger.error(f"Endpoint failed endpoint={safe_endpoint_name} status={status} reason={failure_reason}")
                    
                    return {
                        'status': 'failed',
                        'endpoint_status': status,
                        'failure_reason': failure_reason,
                        'wait_time_seconds': int(time.time() - start_time),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                else:
                    # Still in progress
                    logger.info(f"Endpoint still creating endpoint={safe_endpoint_name} status={status}")
                    time.sleep(30)
                    
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                logger.error(f"Error checking endpoint status endpoint={safe_endpoint_name} error={error_msg}")
                time.sleep(30)
        
        # Timeout
        wait_time = time.time() - start_time
        logger.warning(f"Endpoint creation timeout endpoint={safe_endpoint_name} wait_time={wait_time:.1f}s")
        
        return {
            'status': 'timeout',
            'wait_time_seconds': int(wait_time),
            'message': f'Endpoint did not become ready within {max_wait_time} seconds',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.error(f"Wait for endpoint failed endpoint={safe_endpoint_name} error={error_msg}")
        
        return {
            'status': 'error',
            'error': error_msg,
            'wait_time_seconds': int(time.time() - start_time),
            'timestamp': datetime.now().isoformat()
        }