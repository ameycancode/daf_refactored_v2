"""
Prediction Cleanup Lambda Function - Environment-Aware Version
This version uses environment variables for all configuration values.
"""

import json
import boto3
import logging
from datetime import datetime
from typing import Dict, List, Any
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

def lambda_handler(event, context):
    """
    Main Lambda handler for prediction cleanup
    
    Expected event structure:
    {
        "operation": "cleanup_prediction_resources",
        "profiles": ["RNN", "RN", "M"],
        "cleanup_type": "endpoints_only" | "complete",
        "execution_id": "12345-abcde"
    }
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting prediction cleanup [execution_id={sanitize_for_logging(execution_id)}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Extract operation details
        operation = event.get('operation', 'cleanup_prediction_resources')
        profiles = event.get('profiles', [])
        cleanup_type = event.get('cleanup_type', 'endpoints_only')
        step_functions_execution_id = event.get('execution_id', execution_id)
        
        if not profiles:
            raise ValueError("Profiles list is required for cleanup operations")
        
        safe_operation = sanitize_for_logging(operation)
        safe_cleanup_type = sanitize_for_logging(cleanup_type)
        logger.info(f"Processing operation={safe_operation} cleanup_type={safe_cleanup_type}")
        logger.info(f"Profiles to cleanup: {len(profiles)}")
        
        # Handle different operations
        if operation == 'cleanup_prediction_resources':
            result = cleanup_prediction_resources(profiles, cleanup_type, step_functions_execution_id)
        elif operation == 'cleanup_endpoints_only':
            result = cleanup_endpoints_only(profiles, step_functions_execution_id)
        elif operation == 'cleanup_temporary_artifacts':
            result = cleanup_temporary_artifacts(profiles, step_functions_execution_id)
        else:
            raise ValueError(f"Unknown operation: {safe_operation}")
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        logger.error(f"Prediction cleanup failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'profiles': event.get('profiles', []),
                'message': 'Prediction cleanup failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def cleanup_prediction_resources(profiles: List[str], cleanup_type: str, execution_id: str) -> Dict[str, Any]:
    """
    Cleanup prediction resources for specified profiles
    """
    
    try:
        logger.info(f"Cleaning up prediction resources: {len(profiles)} profiles, type: {cleanup_type}")
        
        cleanup_results = {}
        total_resources_deleted = 0
        total_cost_saved = 0.0
        
        for profile in profiles:
            try:
                safe_profile = sanitize_for_logging(profile, 50)
                logger.info(f"Cleaning up profile={safe_profile}")
                
                if cleanup_type == 'endpoints_only':
                    profile_result = cleanup_profile_endpoints(profile)
                elif cleanup_type == 'complete':
                    profile_result = cleanup_profile_complete(profile)
                else:
                    profile_result = cleanup_profile_endpoints(profile)  # Default to endpoints only
                
                cleanup_results[profile] = profile_result
                
                # Aggregate metrics
                if profile_result.get('status') == 'success':
                    total_resources_deleted += profile_result.get('resources_deleted_count', 0)
                    total_cost_saved += profile_result.get('estimated_cost_saved_usd', 0.0)
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                safe_profile = sanitize_for_logging(profile, 50)
                logger.error(f"Profile cleanup failed profile={safe_profile} error={error_msg}")
                
                cleanup_results[profile] = {
                    'status': 'failed',
                    'error': error_msg,
                    'profile': profile,
                    'timestamp': datetime.now().isoformat()
                }
        
        # Calculate summary
        successful_cleanups = len([r for r in cleanup_results.values() if r.get('status') == 'success'])
        
        result = {
            'operation': 'cleanup_prediction_resources',
            'execution_id': execution_id,
            'cleanup_type': cleanup_type,
            'profiles_processed': profiles,
            'cleanup_results': cleanup_results,
            'summary': {
                'total_profiles': len(profiles),
                'successful_cleanups': successful_cleanups,
                'failed_cleanups': len(profiles) - successful_cleanups,
                'total_resources_deleted': total_resources_deleted,
                'total_cost_saved_usd': round(total_cost_saved, 2),
                'cleanup_rate': round((successful_cleanups / len(profiles)) * 100, 1) if profiles else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction cleanup completed: {successful_cleanups}/{len(profiles)} profiles cleaned")
        return result
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        logger.error(f"Prediction cleanup failed: {error_msg}")
        
        return {
            'operation': 'cleanup_prediction_resources',
            'execution_id': execution_id,
            'error': error_msg,
            'profiles_processed': profiles,
            'timestamp': datetime.now().isoformat()
        }

def cleanup_profile_endpoints(profile: str) -> Dict[str, Any]:
    """
    Cleanup endpoints only for a specific profile
    """
    
    try:
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Cleaning up endpoints profile={safe_profile}")
        
        deleted_resources = []
        cost_saved = 0.0
        
        # Find all endpoints for this profile
        endpoints = sagemaker_client.list_endpoints(
            NameContains=f"energy-forecasting-{profile.lower()}"
        )
        
        for endpoint in endpoints['Endpoints']:
            endpoint_name = endpoint['EndpointName']
            endpoint_status = endpoint['EndpointStatus']
            
            try:
                # Delete endpoint if it's running
                if endpoint_status in ['InService', 'Creating', 'Updating']:
                    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                    deleted_resources.append(f"endpoint:{endpoint_name}")
                    
                    # Estimate cost saved (rough calculation)
                    cost_saved += 1.20  # Rough estimate per endpoint per day
                    
                    safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
                    logger.info(f"Deleted endpoint profile={safe_profile} endpoint={safe_endpoint_name}")
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
                logger.warning(f"Failed to delete endpoint profile={safe_profile} endpoint={safe_endpoint_name} error={error_msg}")
        
        return {
            'status': 'success',
            'profile': profile,
            'cleanup_type': 'endpoints_only',
            'deleted_resources': deleted_resources,
            'resources_deleted_count': len(deleted_resources),
            'estimated_cost_saved_usd': round(cost_saved, 2),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Endpoint cleanup failed profile={safe_profile} error={error_msg}")
        
        return {
            'status': 'failed',
            'profile': profile,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

def cleanup_profile_complete(profile: str) -> Dict[str, Any]:
    """
    Complete cleanup for a specific profile (endpoints, configs, models)
    """
    
    try:
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Complete cleanup profile={safe_profile}")
        
        deleted_resources = []
        cost_saved = 0.0
        
        # Step 1: Delete endpoints
        endpoints = sagemaker_client.list_endpoints(
            NameContains=f"energy-forecasting-{profile.lower()}"
        )
        
        endpoint_configs_to_delete = []
        models_to_delete = []
        
        for endpoint in endpoints['Endpoints']:
            endpoint_name = endpoint['EndpointName']
            
            try:
                # Get endpoint details before deletion
                endpoint_details = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                endpoint_config_name = endpoint_details['EndpointConfigName']
                endpoint_configs_to_delete.append(endpoint_config_name)
                
                # Delete endpoint
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                deleted_resources.append(f"endpoint:{endpoint_name}")
                cost_saved += 1.20
                
                safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
                logger.info(f"Deleted endpoint profile={safe_profile} endpoint={safe_endpoint_name}")
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
                logger.warning(f"Failed to delete endpoint profile={safe_profile} endpoint={safe_endpoint_name} error={error_msg}")
        
        # Step 2: Delete endpoint configurations
        for config_name in set(endpoint_configs_to_delete):  # Remove duplicates
            try:
                # Get config details before deletion
                config_details = sagemaker_client.describe_endpoint_config(EndpointConfigName=config_name)
                
                # Extract model names
                for variant in config_details.get('ProductionVariants', []):
                    model_name = variant.get('ModelName')
                    if model_name:
                        models_to_delete.append(model_name)
                
                # Delete endpoint configuration
                sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
                deleted_resources.append(f"endpoint_config:{config_name}")
                
                safe_config_name = sanitize_for_logging(config_name, 100)
                logger.info(f"Deleted endpoint config profile={safe_profile} config={safe_config_name}")
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                safe_config_name = sanitize_for_logging(config_name, 100)
                logger.warning(f"Failed to delete endpoint config profile={safe_profile} config={safe_config_name} error={error_msg}")
        
        # Step 3: Delete models
        for model_name in set(models_to_delete):  # Remove duplicates
            try:
                sagemaker_client.delete_model(ModelName=model_name)
                deleted_resources.append(f"model:{model_name}")
                
                safe_model_name = sanitize_for_logging(model_name, 100)
                logger.info(f"Deleted model profile={safe_profile} model={safe_model_name}")
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                safe_model_name = sanitize_for_logging(model_name, 100)
                logger.warning(f"Failed to delete model profile={safe_profile} model={safe_model_name} error={error_msg}")
        
        return {
            'status': 'success',
            'profile': profile,
            'cleanup_type': 'complete',
            'deleted_resources': deleted_resources,
            'resources_deleted_count': len(deleted_resources),
            'estimated_cost_saved_usd': round(cost_saved, 2),
            'breakdown': {
                'endpoints_deleted': len([r for r in deleted_resources if r.startswith('endpoint:')]),
                'configs_deleted': len([r for r in deleted_resources if r.startswith('endpoint_config:')]),
                'models_deleted': len([r for r in deleted_resources if r.startswith('model:')])
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Complete cleanup failed profile={safe_profile} error={error_msg}")
        
        return {
            'status': 'failed',
            'profile': profile,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

def cleanup_endpoints_only(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Cleanup endpoints only for multiple profiles
    """
    
    try:
        logger.info(f"Cleaning up endpoints only for {len(profiles)} profiles")
        
        cleanup_results = {}
        total_endpoints_deleted = 0
        total_cost_saved = 0.0
        
        for profile in profiles:
            try:
                result = cleanup_profile_endpoints(profile)
                cleanup_results[profile] = result
                
                if result.get('status') == 'success':
                    total_endpoints_deleted += len([r for r in result.get('deleted_resources', []) if r.startswith('endpoint:')])
                    total_cost_saved += result.get('estimated_cost_saved_usd', 0.0)
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                safe_profile = sanitize_for_logging(profile, 50)
                logger.error(f"Endpoint cleanup failed profile={safe_profile} error={error_msg}")
                
                cleanup_results[profile] = {
                    'status': 'failed',
                    'error': error_msg,
                    'profile': profile
                }
        
        successful_cleanups = len([r for r in cleanup_results.values() if r.get('status') == 'success'])
        
        return {
            'operation': 'cleanup_endpoints_only',
            'execution_id': execution_id,
            'profiles_processed': profiles,
            'cleanup_results': cleanup_results,
            'summary': {
                'total_profiles': len(profiles),
                'successful_cleanups': successful_cleanups,
                'total_endpoints_deleted': total_endpoints_deleted,
                'total_cost_saved_usd': round(total_cost_saved, 2)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        logger.error(f"Endpoints cleanup failed: {error_msg}")
        
        return {
            'operation': 'cleanup_endpoints_only',
            'execution_id': execution_id,
            'error': error_msg,
            'profiles_processed': profiles,
            'timestamp': datetime.now().isoformat()
        }

def cleanup_temporary_artifacts(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Cleanup temporary artifacts in S3 for profiles
    """
    
    try:
        logger.info(f"Cleaning up temporary artifacts for {len(profiles)} profiles")
        
        # Environment-aware bucket configuration
        data_bucket = os.environ.get('DATA_BUCKET')
        if not data_bucket:
            raise ValueError("DATA_BUCKET environment variable is required")
        
        cleanup_results = {}
        total_objects_deleted = 0
        
        for profile in profiles:
            try:
                safe_profile = sanitize_for_logging(profile, 50)
                logger.info(f"Cleaning up artifacts profile={safe_profile}")
                
                deleted_objects = []
                
                # Cleanup temporary prediction results
                temp_prefix = f"prediction-results/{profile}/temp/"
                deleted_temp = cleanup_s3_prefix(data_bucket, temp_prefix)
                deleted_objects.extend(deleted_temp)
                
                # Cleanup old prediction inputs (keep recent ones)
                old_inputs_prefix = f"prediction-inputs/{profile}/old/"
                deleted_old = cleanup_s3_prefix(data_bucket, old_inputs_prefix)
                deleted_objects.extend(deleted_old)
                
                cleanup_results[profile] = {
                    'status': 'success',
                    'profile': profile,
                    'deleted_objects': deleted_objects,
                    'objects_deleted_count': len(deleted_objects),
                    'timestamp': datetime.now().isoformat()
                }
                
                total_objects_deleted += len(deleted_objects)
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                safe_profile = sanitize_for_logging(profile, 50)
                logger.error(f"Artifact cleanup failed profile={safe_profile} error={error_msg}")
                
                cleanup_results[profile] = {
                    'status': 'failed',
                    'error': error_msg,
                    'profile': profile
                }
        
        successful_cleanups = len([r for r in cleanup_results.values() if r.get('status') == 'success'])
        
        return {
            'operation': 'cleanup_temporary_artifacts',
            'execution_id': execution_id,
            'data_bucket': data_bucket,
            'profiles_processed': profiles,
            'cleanup_results': cleanup_results,
            'summary': {
                'total_profiles': len(profiles),
                'successful_cleanups': successful_cleanups,
                'total_objects_deleted': total_objects_deleted
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        logger.error(f"Temporary artifacts cleanup failed: {error_msg}")
        
        return {
            'operation': 'cleanup_temporary_artifacts',
            'execution_id': execution_id,
            'error': error_msg,
            'profiles_processed': profiles,
            'timestamp': datetime.now().isoformat()
        }

def cleanup_s3_prefix(bucket: str, prefix: str) -> List[str]:
    """
    Cleanup S3 objects with a specific prefix
    """
    
    deleted_objects = []
    
    try:
        safe_bucket = sanitize_for_logging(bucket, 100)
        safe_prefix = sanitize_for_logging(prefix, 200)
        logger.info(f"Cleaning up S3 objects bucket={safe_bucket} prefix={safe_prefix}")
        
        # List objects with prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=100  # Limit to avoid too many deletions at once
        )
        
        if 'Contents' not in response:
            logger.info(f"No objects found to cleanup bucket={safe_bucket} prefix={safe_prefix}")
            return deleted_objects
        
        # Delete objects
        objects_to_delete = []
        for obj in response['Contents']:
            objects_to_delete.append({'Key': obj['Key']})
        
        if objects_to_delete:
            delete_response = s3_client.delete_objects(
                Bucket=bucket,
                Delete={
                    'Objects': objects_to_delete,
                    'Quiet': True
                }
            )
            
            # Track successfully deleted objects
            for obj in objects_to_delete:
                deleted_objects.append(obj['Key'])
            
            logger.info(f"Deleted {len(deleted_objects)} objects bucket={safe_bucket} prefix={safe_prefix}")
        
        return deleted_objects
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        safe_bucket = sanitize_for_logging(bucket, 100)
        safe_prefix = sanitize_for_logging(prefix, 200)
        logger.error(f"S3 cleanup failed bucket={safe_bucket} prefix={safe_prefix} error={error_msg}")
        return deleted_objects