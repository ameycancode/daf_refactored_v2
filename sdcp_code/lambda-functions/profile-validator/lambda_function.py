"""
Profile Validator Lambda Function - Environment-Aware Version
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
s3_client = boto3.client('s3')

def sanitize_for_logging(value: str, max_length: int = 50) -> str:
    """Sanitize string values for safe logging"""
    if not isinstance(value, str):
        value = str(value)
    sanitized = ''.join(c for c in value if c.isalnum() or c in '-_.')[:max_length]
    return sanitized if sanitized else 'unknown'

def lambda_handler(event, context):
    """
    Main Lambda handler for profile validation
    
    Expected event structure:
    {
        "operation": "validate_and_filter_profiles",
        "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
        "data_bucket": "<from environment variable DATA_BUCKET>"
    }
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting profile validation [execution_id={sanitize_for_logging(execution_id)}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Environment-aware configuration
        data_bucket = event.get('data_bucket') or os.environ.get('DATA_BUCKET')
        if not data_bucket:
            raise ValueError("DATA_BUCKET must be provided in event or environment variables")
        
        # Extract operation details
        operation = event.get('operation', 'validate_and_filter_profiles')
        profiles = event.get('profiles', [])
        
        if not profiles:
            raise ValueError("Profiles list is required for validation")
        
        safe_operation = sanitize_for_logging(operation)
        safe_data_bucket = sanitize_for_logging(data_bucket, 100)
        logger.info(f"Processing operation={safe_operation} data_bucket={safe_data_bucket}")
        logger.info(f"Profiles to validate: {len(profiles)}")
        
        # Handle different operations
        if operation == 'validate_and_filter_profiles':
            result = validate_and_filter_profiles(profiles, data_bucket, execution_id)
        elif operation == 'check_profile_data_availability':
            result = check_profile_data_availability(profiles, data_bucket, execution_id)
        else:
            raise ValueError(f"Unknown operation: {safe_operation}")
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        logger.error(f"Profile validation failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'profiles': event.get('profiles', []),
                'message': 'Profile validation failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def validate_and_filter_profiles(profiles: List[str], data_bucket: str, execution_id: str) -> Dict[str, Any]:
    """
    Validate profiles and filter out those that don't have required data
    """
    
    try:
        logger.info(f"Validating {len(profiles)} profiles")
        
        valid_profiles = []
        invalid_profiles = []
        validation_details = {}
        
        for profile in profiles:
            try:
                safe_profile = sanitize_for_logging(profile, 20)
                logger.info(f"Validating profile={safe_profile}")
                
                # Check if profile has required data and configurations
                validation_result = validate_single_profile(profile, data_bucket)
                validation_details[profile] = validation_result
                
                if validation_result['is_valid']:
                    valid_profiles.append(profile)
                    logger.info(f"Profile validated successfully profile={safe_profile}")
                else:
                    invalid_profiles.append(profile)
                    logger.warning(f"Profile validation failed profile={safe_profile} reason={validation_result.get('reason', 'unknown')}")
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                safe_profile = sanitize_for_logging(profile, 20)
                logger.error(f"Error validating profile profile={safe_profile} error={error_msg}")
                
                invalid_profiles.append(profile)
                validation_details[profile] = {
                    'is_valid': False,
                    'reason': f'Validation error: {error_msg}',
                    'checks': {}
                }
        
        # Prepare result
        result = {
            'operation': 'validate_and_filter_profiles',
            'execution_id': execution_id,
            'input_profiles': profiles,
            'valid_profiles': valid_profiles,
            'invalid_profiles': invalid_profiles,
            'validation_details': validation_details,
            'summary': {
                'total_profiles': len(profiles),
                'valid_count': len(valid_profiles),
                'invalid_count': len(invalid_profiles),
                'validation_rate': round((len(valid_profiles) / len(profiles)) * 100, 1) if profiles else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Profile validation completed: {len(valid_profiles)}/{len(profiles)} profiles valid")
        return result
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        logger.error(f"Profile validation failed: {error_msg}")
        
        return {
            'operation': 'validate_and_filter_profiles',
            'execution_id': execution_id,
            'error': error_msg,
            'input_profiles': profiles,
            'timestamp': datetime.now().isoformat()
        }

def validate_single_profile(profile: str, data_bucket: str) -> Dict[str, Any]:
    """
    Validate a single profile for data availability and requirements
    """
    
    validation_result = {
        'profile': profile,
        'is_valid': False,
        'checks': {},
        'reason': '',
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Check 1: Profile name validation
        if not profile or not isinstance(profile, str):
            validation_result['reason'] = 'Invalid profile name'
            return validation_result
        
        if profile not in ['RNN', 'RN', 'M', 'S', 'AGR', 'L', 'A6']:
            validation_result['reason'] = f'Unknown profile: {profile}'
            return validation_result
        
        validation_result['checks']['profile_name'] = True
        
        # Check 2: Model artifacts availability
        model_check = check_model_artifacts_availability(profile, data_bucket)
        validation_result['checks']['model_artifacts'] = model_check
        
        # Check 3: Configuration files availability
        config_check = check_configuration_availability(profile, data_bucket)
        validation_result['checks']['configuration_files'] = config_check
        
        # Check 4: Profile-specific requirements
        profile_specific_check = check_profile_specific_requirements(profile)
        validation_result['checks']['profile_requirements'] = profile_specific_check
        
        # Determine overall validity
        required_checks = ['profile_name', 'model_artifacts']  # Configuration is optional
        all_required_passed = all(validation_result['checks'].get(check, False) for check in required_checks)
        
        if all_required_passed:
            validation_result['is_valid'] = True
            validation_result['reason'] = 'All validation checks passed'
        else:
            failed_checks = [check for check in required_checks if not validation_result['checks'].get(check, False)]
            validation_result['reason'] = f'Failed checks: {", ".join(failed_checks)}'
        
        return validation_result
        
    except Exception as e:
        validation_result['reason'] = f'Validation error: {str(e)}'
        return validation_result

def check_model_artifacts_availability(profile: str, data_bucket: str) -> bool:
    """
    Check if model artifacts are available for the profile
    """
    
    try:
        # Check if there are model artifacts in the model bucket
        # Note: We check data_bucket here as it might contain model references
        model_prefix = f"models/{profile}/"
        
        response = s3_client.list_objects_v2(
            Bucket=data_bucket,
            Prefix=model_prefix,
            MaxKeys=1
        )
        
        # If we find any objects, consider models available
        has_artifacts = 'Contents' in response and len(response['Contents']) > 0
        
        if has_artifacts:
            safe_profile = sanitize_for_logging(profile, 20)
            logger.info(f"Model artifacts found profile={safe_profile}")
        
        return has_artifacts
        
    except Exception as e:
        safe_profile = sanitize_for_logging(profile, 20)
        error_msg = sanitize_for_logging(str(e), 100)
        logger.warning(f"Could not check model artifacts profile={safe_profile} error={error_msg}")
        # Return True to not block validation due to S3 access issues
        return True

def check_configuration_availability(profile: str, data_bucket: str) -> bool:
    """
    Check if configuration files are available for the profile
    """
    
    try:
        # Check for endpoint configurations
        config_prefix = f"endpoint-configurations/{profile}/"
        
        response = s3_client.list_objects_v2(
            Bucket=data_bucket,
            Prefix=config_prefix,
            MaxKeys=1
        )
        
        has_config = 'Contents' in response and len(response['Contents']) > 0
        
        if has_config:
            safe_profile = sanitize_for_logging(profile, 20)
            logger.info(f"Configuration files found profile={safe_profile}")
        
        return has_config
        
    except Exception as e:
        safe_profile = sanitize_for_logging(profile, 20)
        error_msg = sanitize_for_logging(str(e), 100)
        logger.warning(f"Could not check configuration files profile={safe_profile} error={error_msg}")
        # Return True to not block validation due to S3 access issues
        return True

def check_profile_specific_requirements(profile: str) -> bool:
    """
    Check profile-specific requirements
    """
    
    try:
        # Profile-specific validation rules
        profile_requirements = {
            'RNN': {'features': 12, 'special_requirements': []},
            'RN': {'features': 13, 'special_requirements': ['radiation_data']},
            'M': {'features': 12, 'special_requirements': []},
            'S': {'features': 12, 'special_requirements': []},
            'AGR': {'features': 12, 'special_requirements': []},
            'L': {'features': 12, 'special_requirements': []},
            'A6': {'features': 12, 'special_requirements': []}
        }
        
        if profile not in profile_requirements:
            return False
        
        # For now, all profiles pass specific requirements
        # This could be extended to check for specific data requirements
        return True
        
    except Exception as e:
        safe_profile = sanitize_for_logging(profile, 20)
        error_msg = sanitize_for_logging(str(e), 100)
        logger.error(f"Profile requirements check failed profile={safe_profile} error={error_msg}")
        return False

def check_profile_data_availability(profiles: List[str], data_bucket: str, execution_id: str) -> Dict[str, Any]:
    """
    Check data availability for multiple profiles
    """
    
    try:
        logger.info(f"Checking data availability for {len(profiles)} profiles")
        
        data_availability = {}
        
        for profile in profiles:
            try:
                safe_profile = sanitize_for_logging(profile, 20)
                logger.info(f"Checking data availability profile={safe_profile}")
                
                availability_check = {
                    'profile': profile,
                    'model_artifacts_available': check_model_artifacts_availability(profile, data_bucket),
                    'configuration_available': check_configuration_availability(profile, data_bucket),
                    'input_data_available': check_input_data_availability(profile, data_bucket),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Determine overall availability
                availability_check['overall_available'] = (
                    availability_check['model_artifacts_available'] and
                    availability_check['input_data_available']
                )
                
                data_availability[profile] = availability_check
                
            except Exception as e:
                error_msg = sanitize_for_logging(str(e), 100)
                safe_profile = sanitize_for_logging(profile, 20)
                logger.error(f"Error checking data availability profile={safe_profile} error={error_msg}")
                
                data_availability[profile] = {
                    'profile': profile,
                    'error': error_msg,
                    'overall_available': False,
                    'timestamp': datetime.now().isoformat()
                }
        
        # Calculate summary
        available_profiles = [p for p, data in data_availability.items() if data.get('overall_available', False)]
        
        result = {
            'operation': 'check_profile_data_availability',
            'execution_id': execution_id,
            'profiles_checked': profiles,
            'data_availability': data_availability,
            'available_profiles': available_profiles,
            'summary': {
                'total_profiles': len(profiles),
                'available_count': len(available_profiles),
                'availability_rate': round((len(available_profiles) / len(profiles)) * 100, 1) if profiles else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Data availability check completed: {len(available_profiles)}/{len(profiles)} profiles available")
        return result
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        logger.error(f"Data availability check failed: {error_msg}")
        
        return {
            'operation': 'check_profile_data_availability',
            'execution_id': execution_id,
            'error': error_msg,
            'profiles_checked': profiles,
            'timestamp': datetime.now().isoformat()
        }

def check_input_data_availability(profile: str, data_bucket: str) -> bool:
    """
    Check if input data is available for predictions
    """
    
    try:
        # Check for prediction input data
        input_prefix = f"prediction-inputs/{profile}/"
        
        response = s3_client.list_objects_v2(
            Bucket=data_bucket,
            Prefix=input_prefix,
            MaxKeys=1
        )
        
        has_input_data = 'Contents' in response and len(response['Contents']) > 0
        
        if not has_input_data:
            # Also check for general input data that could be used for any profile
            general_input_prefix = "prediction-inputs/general/"
            
            response = s3_client.list_objects_v2(
                Bucket=data_bucket,
                Prefix=general_input_prefix,
                MaxKeys=1
            )
            
            has_input_data = 'Contents' in response and len(response['Contents']) > 0
        
        return has_input_data
        
    except Exception as e:
        safe_profile = sanitize_for_logging(profile, 20)
        error_msg = sanitize_for_logging(str(e), 100)
        logger.warning(f"Could not check input data profile={safe_profile} error={error_msg}")
        # Return True to not block validation due to S3 access issues
        return True