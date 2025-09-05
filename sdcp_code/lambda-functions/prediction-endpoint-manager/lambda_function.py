"""
Prediction Endpoint Manager Lambda Function
Recreates endpoints from S3 configurations for daily predictions
SECURITY FIX: Log injection vulnerability (CWE-117, CWE-93) patched
"""

import json
import boto3
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

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
    
    safe_fields = ['operation']
    
    for field in safe_fields:
        if field in event:
            sanitized_event[field] = sanitize_for_logging(event[field])
    
    # Handle profiles list safely
    if 'profiles' in event and isinstance(event['profiles'], list):
        sanitized_event['profiles_count'] = len(event['profiles'])
        sanitized_event['profiles_sample'] = [sanitize_for_logging(p, 20) for p in event['profiles'][:5]]
    
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_profiles': 'profiles' in event
    }
    
    return sanitized_event

def lambda_handler(event, context):
    """
    Main handler for prediction endpoint management
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting prediction endpoint management [execution_id={sanitize_for_logging(execution_id)}]")
        
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
        
        # Configuration
        config = {
            "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
            "max_wait_time": 900,  # 15 minutes
            "config_prefix": "endpoint-configurations/"
        }
        
        # Extract operation and parameters
        operation = event.get('operation', 'recreate_all_endpoints')
        profiles_to_process = event.get('profiles', config['profiles'])
        
        # SECURITY FIX: Sanitize operation and profile data for logging
        safe_operation = sanitize_for_logging(operation)
        safe_profiles_sample = [sanitize_for_logging(p, 20) for p in profiles_to_process[:5]] if isinstance(profiles_to_process, list) else []
        
        logger.info(f"Operation: {safe_operation}, Profiles count: {len(profiles_to_process) if isinstance(profiles_to_process, list) else 0}, Sample: {safe_profiles_sample}")
        
        if operation == 'recreate_all_endpoints':
            result = recreate_all_endpoints_from_configs(profiles_to_process, config, execution_id)
        elif operation == 'check_endpoints_status':
            result = check_endpoints_status(profiles_to_process, execution_id)
        elif operation == 'cleanup_endpoints':
            result = cleanup_endpoints(profiles_to_process, execution_id)
        else:
            raise ValueError(f"Unknown operation: {safe_operation}")
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Prediction endpoint management failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'message': 'Prediction endpoint management failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def recreate_all_endpoints_from_configs(profiles: List[str], config: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Recreate endpoints from saved S3 configurations
    """
    
    try:
        profiles_count = len(profiles) if isinstance(profiles, list) else 0
        logger.info(f"Recreating endpoints from S3 configurations profiles_count={profiles_count}")
        
        endpoint_details = {}
        successful_creations = 0
        
        # Create endpoints for each profile
        for profile in profiles:
            try:
                # SECURITY FIX: Sanitize profile name for logging
                safe_profile = sanitize_for_logging(profile, 50)
                logger.info(f"Recreating endpoint profile={safe_profile}")
                
                endpoint_result = recreate_endpoint_from_config(profile, config, execution_id)
                endpoint_details[profile] = endpoint_result
                
                if endpoint_result['status'] == 'success':
                    successful_creations += 1
                    
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                safe_profile = sanitize_for_logging(profile, 50)
                logger.error(f"Failed to recreate endpoint profile={safe_profile} error={error_msg}")
                endpoint_details[profile] = {
                    'status': 'failed',
                    'error': error_msg,
                    'profile': profile
                }
        
        # Wait for all endpoints to be ready
        if successful_creations > 0:
            logger.info(f"Waiting for endpoints to be ready successful_count={successful_creations}")
            wait_results = wait_for_endpoints_ready(endpoint_details, config['max_wait_time'])
        else:
            wait_results = {}
        
        return {
            'operation': 'recreate_all_endpoints',
            'execution_id': execution_id,
            'profiles_requested': profiles,
            'successful_creations': successful_creations,
            'endpoint_details': endpoint_details,
            'wait_results': wait_results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Failed to recreate endpoints error={error_msg}")
        return {
            'operation': 'recreate_all_endpoints',
            'execution_id': execution_id,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

def recreate_endpoint_from_config(profile: str, config: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Recreate endpoint from saved S3 configuration
    """
    
    try:
        # Find latest configuration file for this profile
        config_data = find_latest_endpoint_config(profile, config['data_bucket'], config['config_prefix'])
        
        if not config_data:
            return {
                'status': 'failed',
                'error': f'No configuration found for profile {profile}',
                'profile': profile
            }
        
        # SECURITY FIX: Sanitize config timestamp for logging
        config_timestamp = sanitize_for_logging(config_data.get('creation_timestamp', 'unknown'))
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Found configuration profile={safe_profile} timestamp={config_timestamp}")
        
        # Generate new unique names for recreation
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"energy-forecasting-{profile.lower()}-model-pred-{current_time}"
        endpoint_config_name = f"energy-forecasting-{profile.lower()}-config-pred-{current_time}"
        endpoint_name = f"energy-forecasting-{profile.lower()}-endpoint-pred-{current_time}"
        
        # Step 1: Create SageMaker Model from saved configuration
        model_info = config_data.get('model_info', {})
        model_package_arn = model_info.get('ModelPackageArn')
        
        if not model_package_arn:
            return {
                'status': 'failed',
                'error': f'No ModelPackageArn found in configuration for profile {profile}',
                'profile': profile
            }
        
        # Create model
        sagemaker_client.create_model(
            ModelName=model_name,
            Containers=[
                {
                    'ModelPackageName': model_package_arn
                }
            ],
            ExecutionRoleArn=f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role"
        )
        
        # SECURITY FIX: Sanitize model name for logging
        safe_model_name = sanitize_for_logging(model_name, 100)
        logger.info(f"Created model for prediction name={safe_model_name}")
        
        # Step 2: Create Endpoint Configuration
        endpoint_config_details = config_data.get('endpoint_configuration', {})
        instance_type = endpoint_config_details.get('instance_type', 'ml.m5.large')
        instance_count = endpoint_config_details.get('instance_count', 1)
        
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': instance_count,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        # SECURITY FIX: Sanitize config name for logging
        safe_config_name = sanitize_for_logging(endpoint_config_name, 100)
        logger.info(f"Created endpoint configuration for prediction name={safe_config_name}")
        
        # Step 3: Create Endpoint
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        # SECURITY FIX: Sanitize endpoint name for logging
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.info(f"Created endpoint for prediction name={safe_endpoint_name}")
        
        return {
            'status': 'success',
            'profile': profile,
            'endpoint_name': endpoint_name,
            'endpoint_config_name': endpoint_config_name,
            'model_name': model_name,
            'instance_type': instance_type,
            'source_config': config_data.get('creation_timestamp', 'unknown'),
            'created_for': 'prediction'
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Failed to recreate endpoint profile={safe_profile} error={error_msg}")
        return {
            'status': 'failed',
            'error': error_msg,
            'profile': profile
        }

def find_latest_endpoint_config(profile: str, bucket: str, config_prefix: str) -> Optional[Dict[str, Any]]:
    """
    Find the latest endpoint configuration for a profile
    """
    
    try:
        # List all configuration files for this profile
        profile_prefix = f"{config_prefix}{profile}/"
        
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=profile_prefix
        )
        
        if 'Contents' not in response:
            safe_profile = sanitize_for_logging(profile, 50)
            logger.warning(f"No configuration files found profile={safe_profile}")
            return None
        
        # Find the most recent configuration file
        config_files = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.json'):
                config_files.append({
                    'key': obj['Key'],
                    'last_modified': obj['LastModified']
                })
        
        if not config_files:
            safe_profile = sanitize_for_logging(profile, 50)
            logger.warning(f"No JSON configuration files found profile={safe_profile}")
            return None
        
        # Sort by last modified date and get the most recent
        config_files.sort(key=lambda x: x['last_modified'], reverse=True)
        latest_config_file = config_files[0]
        
        # SECURITY FIX: Sanitize config key for logging
        safe_profile = sanitize_for_logging(profile, 50)
        safe_config_key = sanitize_for_logging(latest_config_file['key'], 200)
        logger.info(f"Using latest config profile={safe_profile} key={safe_config_key}")
        
        # Load the configuration
        response = s3_client.get_object(
            Bucket=bucket,
            Key=latest_config_file['key']
        )
        
        config_data = json.loads(response['Body'].read().decode())
        
        return config_data
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Failed to find configuration profile={safe_profile} error={error_msg}")
        return None

def wait_for_endpoints_ready(endpoint_details: Dict[str, Dict[str, Any]], max_wait_time: int) -> Dict[str, Any]:
    """
    Wait for all endpoints to reach InService status
    """
    
    start_time = time.time()
    ready_endpoints = {}
    failed_endpoints = {}
    
    wait_minutes = max_wait_time / 60
    logger.info(f"Waiting for endpoints to be ready max_wait_minutes={wait_minutes:.1f}")
    
    while time.time() - start_time < max_wait_time:
        all_ready = True
        
        for profile, details in endpoint_details.items():
            if details.get('status') != 'success':
                continue
                
            endpoint_name = details.get('endpoint_name')
            if not endpoint_name:
                continue
                
            if profile in ready_endpoints or profile in failed_endpoints:
                continue
            
            try:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                # SECURITY FIX: Sanitize values for logging
                safe_profile = sanitize_for_logging(profile, 50)
                safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
                
                if status == 'InService':
                    ready_endpoints[profile] = {
                        'endpoint_name': endpoint_name,
                        'status': 'ready',
                        'ready_time': datetime.now().isoformat()
                    }
                    logger.info(f"Endpoint ready profile={safe_profile} endpoint={safe_endpoint_name}")
                elif status == 'Failed':
                    failure_reason = sanitize_for_logging(response.get('FailureReason', 'Unknown failure'))
                    failed_endpoints[profile] = {
                        'endpoint_name': endpoint_name,
                        'status': 'failed',
                        'error': failure_reason
                    }
                    logger.error(f"Endpoint failed profile={safe_profile} endpoint={safe_endpoint_name} reason={failure_reason}")
                else:
                    all_ready = False
                    
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                safe_profile = sanitize_for_logging(profile, 50)
                logger.warning(f"Could not check status profile={safe_profile} error={error_msg}")
                all_ready = False
        
        if all_ready:
            break
            
        time.sleep(30)  # Check every 30 seconds
    
    # Update endpoint_details with ready status
    for profile in ready_endpoints:
        if profile in endpoint_details:
            endpoint_details[profile]['ready'] = True
            endpoint_details[profile]['ready_time'] = ready_endpoints[profile]['ready_time']
    
    for profile in failed_endpoints:
        if profile in endpoint_details:
            endpoint_details[profile]['status'] = 'failed'
            endpoint_details[profile]['error'] = failed_endpoints[profile]['error']
    
    return {
        'ready_endpoints': ready_endpoints,
        'failed_endpoints': failed_endpoints,
        'total_wait_time': time.time() - start_time
    }

def check_endpoints_status(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Check the status of existing endpoints
    """
    
    try:
        profiles_count = len(profiles) if isinstance(profiles, list) else 0
        logger.info(f"Checking status for profiles_count={profiles_count}")
        
        endpoint_status = {}
        
        for profile in profiles:
            try:
                # SECURITY FIX: Sanitize profile for logging
                safe_profile = sanitize_for_logging(profile, 50)
                
                # Try to find existing endpoints for this profile
                endpoints = sagemaker_client.list_endpoints(
                    NameContains=f"energy-forecasting-{profile.lower()}",
                    StatusEquals='InService'
                )
                
                if endpoints['Endpoints']:
                    # Get the most recent endpoint
                    latest_endpoint = sorted(endpoints['Endpoints'], 
                                           key=lambda x: x['CreationTime'], reverse=True)[0]
                    
                    safe_endpoint_name = sanitize_for_logging(latest_endpoint['EndpointName'], 100)
                    endpoint_status[profile] = {
                        'status': 'active',
                        'endpoint_name': latest_endpoint['EndpointName'],
                        'endpoint_status': latest_endpoint['EndpointStatus'],
                        'creation_time': latest_endpoint['CreationTime'].isoformat(),
                        'instance_type': 'unknown'  # Would need to describe endpoint config to get this
                    }
                    logger.info(f"Found active endpoint profile={safe_profile} endpoint={safe_endpoint_name}")
                else:
                    endpoint_status[profile] = {
                        'status': 'not_found',
                        'message': f'No active endpoints found for {profile}'
                    }
                    logger.info(f"No active endpoints found profile={safe_profile}")
                    
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                safe_profile = sanitize_for_logging(profile, 50)
                endpoint_status[profile] = {
                    'status': 'error',
                    'error': error_msg
                }
                logger.error(f"Error checking endpoint status profile={safe_profile} error={error_msg}")
        
        return {
            'operation': 'check_endpoints_status',
            'execution_id': execution_id,
            'endpoint_status': endpoint_status,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Failed to check endpoint status error={error_msg}")
        return {
            'operation': 'check_endpoints_status',
            'execution_id': execution_id,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

def cleanup_endpoints(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Cleanup endpoints after predictions are complete
    """
    
    try:
        profiles_count = len(profiles) if isinstance(profiles, list) else 0
        logger.info(f"Cleaning up endpoints profiles_count={profiles_count}")
        
        cleanup_results = {}
        total_cost_saved = 0.0
        
        for profile in profiles:
            try:
                # SECURITY FIX: Sanitize profile for logging
                safe_profile = sanitize_for_logging(profile, 50)
                
                # Find endpoints for this profile created for predictions
                endpoints = sagemaker_client.list_endpoints(
                    NameContains=f"energy-forecasting-{profile.lower()}-endpoint-pred"
                )
                
                profile_cleanup = {
                    'endpoints_found': len(endpoints['Endpoints']),
                    'endpoints_deleted': 0,
                    'configs_deleted': 0,
                    'models_deleted': 0,
                    'errors': []
                }
                
                for endpoint in endpoints['Endpoints']:
                    endpoint_name = endpoint['EndpointName']
                    safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
                    
                    try:
                        # Get endpoint configuration name
                        endpoint_details = sagemaker_client.describe_endpoint(
                            EndpointName=endpoint_name
                        )
                        endpoint_config_name = endpoint_details['EndpointConfigName']
                        
                        # Get model names from endpoint configuration
                        config_details = sagemaker_client.describe_endpoint_config(
                            EndpointConfigName=endpoint_config_name
                        )
                        
                        model_names = []
                        for variant in config_details['ProductionVariants']:
                            model_names.append(variant['ModelName'])
                        
                        # Delete endpoint
                        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                        profile_cleanup['endpoints_deleted'] += 1
                        logger.info(f"Deleted endpoint name={safe_endpoint_name}")
                        
                        # Delete endpoint configuration
                        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                        profile_cleanup['configs_deleted'] += 1
                        safe_config_name = sanitize_for_logging(endpoint_config_name, 100)
                        logger.info(f"Deleted endpoint config name={safe_config_name}")
                        
                        # Delete models
                        for model_name in model_names:
                            try:
                                sagemaker_client.delete_model(ModelName=model_name)
                                profile_cleanup['models_deleted'] += 1
                                safe_model_name = sanitize_for_logging(model_name, 100)
                                logger.info(f"Deleted model name={safe_model_name}")
                            except Exception as e:
                                error_msg = sanitize_for_logging(str(e))
                                safe_model_name = sanitize_for_logging(model_name, 100)
                                profile_cleanup['errors'].append(f"Failed to delete model {safe_model_name}: {error_msg}")
                        
                        # Estimate cost savings (approximate)
                        # ml.m5.large costs ~$0.115/hour
                        cost_saved_per_hour = 0.115
                        total_cost_saved += cost_saved_per_hour
                        
                    except Exception as e:
                        error_msg = sanitize_for_logging(str(e))
                        error_entry = f"Failed to cleanup {safe_endpoint_name}: {error_msg}"
                        profile_cleanup['errors'].append(error_entry)
                        logger.error(f"Cleanup error endpoint={safe_endpoint_name} error={error_msg}")
                
                cleanup_results[profile] = profile_cleanup
                logger.info(f"Cleanup completed profile={safe_profile} endpoints_deleted={profile_cleanup['endpoints_deleted']}")
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                safe_profile = sanitize_for_logging(profile, 50)
                cleanup_results[profile] = {
                    'error': error_msg,
                    'status': 'failed'
                }
                logger.error(f"Profile cleanup failed profile={safe_profile} error={error_msg}")
        
        return {
            'operation': 'cleanup_endpoints',
            'execution_id': execution_id,
            'cleanup_results': cleanup_results,
            'estimated_cost_savings_per_hour': round(total_cost_saved, 3),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Failed to cleanup endpoints error={error_msg}")
        return {
            'operation': 'cleanup_endpoints',
            'execution_id': execution_id,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
