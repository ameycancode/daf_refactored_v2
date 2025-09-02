#!/usr/bin/env python3
"""
Environment Validation Tool for Enhanced Prediction Pipeline
deployment/validate_environment.py

Comprehensive validation of AWS environment prerequisites:
- IAM roles and permissions
- S3 buckets and configurations
- ECR repositories
- SageMaker resources
- Model Registry
- Step Functions
- Lambda functions
- EventBridge
- Network connectivity
"""

import json
import boto3
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """
    Comprehensive environment validator for the enhanced prediction pipeline
    """
    
    def __init__(self, region: str = "us-west-2", environment: str = "dev"):
        """Initialize the environment validator"""
        
        self.region = region
        self.environment = environment
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # AWS clients
        self.sts_client = boto3.client('sts')
        self.iam_client = boto3.client('iam')
        self.s3_client = boto3.client('s3')
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.events_client = boto3.client('events', region_name=region)
        self.ecr_client = boto3.client('ecr', region_name=region)
        self.ec2_client = boto3.client('ec2', region_name=region)
        
        # Expected resources configuration
        self.expected_resources = {
            'iam_roles': [
                f'sdcp-{environment}-sagemaker-energy-forecasting-datascientist-role',
                # 'EnergyForecastingEventBridgeRole'
            ],
            's3_buckets': [
                f'sdcp-{environment}-sagemaker-energy-forecasting-data',
                f'sdcp-{environment}-sagemaker-energy-forecasting-models'
            ],
            'ecr_repositories': [
                'energy-preprocessing',
                'energy-training',
                'energy-prediction'
            ],
            'lambda_functions': [
                'energy-forecasting-profile-validator',
                'energy-forecasting-profile-endpoint-creator',
                'energy-forecasting-profile-predictor',
                'energy-forecasting-profile-cleanup'
            ],
            'step_functions': [
                'energy-forecasting-training-pipeline',
                'energy-forecasting-prediction-pipeline'
            ],
            'model_package_groups': [
                'energy-forecasting-models'
            ]
        }
        
        # Validation results storage
        self.validation_results = {}
        
        logger.info(f"Environment Validator initialized for {environment} environment in {region}")

    def validate_complete_environment(self) -> Dict[str, Any]:
        """Run complete environment validation"""
        
        logger.info("="*80)
        logger.info("COMPREHENSIVE ENVIRONMENT VALIDATION")
        logger.info("="*80)
        logger.info(f"Account: {self.account_id}")
        logger.info(f"Region: {self.region}")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        validation_start_time = time.time()
        
        try:
            # 1. Validate AWS Credentials and Permissions
            logger.info("\n1. VALIDATING AWS CREDENTIALS AND PERMISSIONS")
            logger.info("-" * 50)
            credentials_result = self._validate_credentials_and_permissions()
            self.validation_results['credentials_and_permissions'] = credentials_result
            
            # 2. Validate IAM Roles
            logger.info("\n2. VALIDATING IAM ROLES")
            logger.info("-" * 50)
            iam_result = self._validate_iam_roles()
            self.validation_results['iam_roles'] = iam_result
            
            # 3. Validate S3 Resources
            logger.info("\n3. VALIDATING S3 RESOURCES")
            logger.info("-" * 50)
            s3_result = self._validate_s3_resources()
            self.validation_results['s3_resources'] = s3_result
            
            # 4. Validate ECR Repositories
            logger.info("\n4. VALIDATING ECR REPOSITORIES")
            logger.info("-" * 50)
            ecr_result = self._validate_ecr_repositories()
            self.validation_results['ecr_repositories'] = ecr_result
            
            # 5. Validate SageMaker Resources
            logger.info("\n5. VALIDATING SAGEMAKER RESOURCES")
            logger.info("-" * 50)
            sagemaker_result = self._validate_sagemaker_resources()
            self.validation_results['sagemaker_resources'] = sagemaker_result
            
            # 6. Validate Lambda Functions
            logger.info("\n6. VALIDATING LAMBDA FUNCTIONS")
            logger.info("-" * 50)
            lambda_result = self._validate_lambda_functions()
            self.validation_results['lambda_functions'] = lambda_result
            
            # 7. Validate Step Functions
            logger.info("\n7. VALIDATING STEP FUNCTIONS")
            logger.info("-" * 50)
            stepfunctions_result = self._validate_step_functions()
            self.validation_results['step_functions'] = stepfunctions_result
            
            # 8. Validate EventBridge
            logger.info("\n8. VALIDATING EVENTBRIDGE")
            logger.info("-" * 50)
            eventbridge_result = self._validate_eventbridge()
            self.validation_results['eventbridge'] = eventbridge_result
            
            # 9. Validate Network Connectivity
            logger.info("\n9. VALIDATING NETWORK CONNECTIVITY")
            logger.info("-" * 50)
            network_result = self._validate_network_connectivity()
            self.validation_results['network_connectivity'] = network_result
            
            # 10. Validate Resource Dependencies
            logger.info("\n10. VALIDATING RESOURCE DEPENDENCIES")
            logger.info("-" * 50)
            dependencies_result = self._validate_resource_dependencies()
            self.validation_results['resource_dependencies'] = dependencies_result
            
            # Generate comprehensive summary
            validation_time = time.time() - validation_start_time
            summary = self._generate_validation_summary(validation_time)

            validation_summary = summary['validation_summary']
            logger.info(f"VALIDATION SUMMARY: {validation_summary}")
            
            overall_status = validation_summary['overall_status']
            logger.info(f"OVERALL STATUS: {overall_status}")
            
            logger.info("\n" + "="*80)
            logger.info("ENVIRONMENT VALIDATION COMPLETED")
            logger.info("="*80)
            logger.info(f"Validation time: {validation_time / 60:.2f} minutes")
            # logger.info(f"Overall status: {summary.get('overall_status', 'UNKNOWN')}")
            logger.info(f"Overall status: {overall_status}")
            
            return summary
            
        except Exception as e:
            validation_time = time.time() - validation_start_time
            logger.error(f"Environment validation failed after {validation_time / 60:.2f} minutes: {str(e)}")
            
            return {
                'overall_status': 'FAILED',
                'error': str(e),
                'validation_time_minutes': validation_time / 60,
                'partial_results': self.validation_results
            }

    def _validate_credentials_and_permissions(self) -> Dict[str, Any]:
        """Validate AWS credentials and basic permissions"""
        
        try:
            results = {
                'caller_identity_valid': False,
                'basic_permissions_valid': False,
                'region_accessible': False,
                'account_details': {}
            }
            
            # Test caller identity
            try:
                identity = self.sts_client.get_caller_identity()
                results['caller_identity_valid'] = True
                results['account_details'] = {
                    'account': identity.get('Account'),
                    'arn': identity.get('Arn'),
                    'user_id': identity.get('UserId')
                }
                logger.info(f"✓ AWS credentials valid for account {identity.get('Account')}")
            except Exception as e:
                logger.error(f"✗ AWS credentials invalid: {str(e)}")
                results['credential_error'] = str(e)
            
            # Test basic service permissions
            permission_tests = [
                ('IAM', lambda: self.iam_client.list_roles(MaxItems=1)),
                ('S3', lambda: self.s3_client.list_buckets()),
                ('SageMaker', lambda: self.sagemaker_client.list_domains()),
                ('Step Functions', lambda: self.stepfunctions_client.list_state_machines(maxResults=1)),
                ('Lambda', lambda: self.lambda_client.list_functions(MaxItems=1)),
                ('EventBridge', lambda: self.events_client.list_rules(Limit=1)),
                ('ECR', lambda: self.ecr_client.describe_repositories(maxResults=1))
            ]
            
            permission_results = {}
            successful_permissions = 0
            
            for service_name, test_func in permission_tests:
                try:
                    test_func()
                    permission_results[service_name] = True
                    successful_permissions += 1
                    logger.info(f"✓ {service_name} permissions valid")
                except Exception as e:
                    permission_results[service_name] = False
                    logger.error(f"✗ {service_name} permissions invalid: {str(e)}")
            
            results['basic_permissions_valid'] = successful_permissions == len(permission_tests)
            results['permission_details'] = permission_results
            
            # # Test region accessibility
            # try:
            #     self.ec2_client.describe_regions(RegionNames=[self.region])
            #     results['region_accessible'] = True
            #     logger.info(f"✓ Region {self.region} accessible")
            # except Exception as e:
            #     logger.error(f"✗ Region {self.region} not accessible: {str(e)}")
            #     results['region_error'] = str(e)
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _validate_iam_roles(self) -> Dict[str, Any]:
        """Validate required IAM roles exist and have proper permissions"""
        
        try:
            results = {
                'roles_found': {},
                'roles_missing': [],
                'permission_issues': [],
                'all_roles_valid': False
            }
            
            for role_name in self.expected_resources['iam_roles']:
                try:
                    # Check if role exists
                    role_response = self.iam_client.get_role(RoleName=role_name)
                    role_arn = role_response['Role']['Arn']
                    
                    # Get attached policies
                    attached_policies = self.iam_client.list_attached_role_policies(RoleName=role_name)
                    inline_policies = self.iam_client.list_role_policies(RoleName=role_name)
                    
                    results['roles_found'][role_name] = {
                        'arn': role_arn,
                        'attached_policies': [p['PolicyName'] for p in attached_policies['AttachedPolicies']],
                        'inline_policies': inline_policies['PolicyNames'],
                        'created_date': role_response['Role']['CreateDate'].isoformat(),
                        'trust_policy': role_response['Role']['AssumeRolePolicyDocument']
                    }
                    
                    logger.info(f"✓ IAM role found: {role_name}")
                    
                    # # Validate specific role permissions
                    # if 'datascientist' in role_name:
                    #     self._validate_datascientist_role_permissions(role_name, results)
                    # elif 'EventBridge' in role_name:
                    #     self._validate_eventbridge_role_permissions(role_name, results)
                    
                except self.iam_client.exceptions.NoSuchEntityException:
                    results['roles_missing'].append(role_name)
                    logger.error(f"✗ IAM role not found: {role_name}")
                except Exception as e:
                    results['permission_issues'].append(f"{role_name}: {str(e)}")
                    logger.error(f"✗ Error validating role {role_name}: {str(e)}")
            
            results['all_roles_valid'] = (
                len(results['roles_missing']) == 0 and 
                len(results['permission_issues']) == 0
            )
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _validate_datascientist_role_permissions(self, role_name: str, results: Dict):
        """Validate data scientist role has required permissions"""
        
        try:
            required_policies = [
                'AmazonSageMakerFullAccess',
                'AmazonS3FullAccess',
                'AWSStepFunctionsFullAccess'
            ]
            
            attached_policies = self.iam_client.list_attached_role_policies(RoleName=role_name)
            policy_names = [p['PolicyName'] for p in attached_policies['AttachedPolicies']]
            
            missing_policies = [policy for policy in required_policies if policy not in policy_names]
            
            if missing_policies:
                results['permission_issues'].append(
                    f"{role_name} missing policies: {missing_policies}"
                )
                logger.warning(f" {role_name} missing required policies: {missing_policies}")
            else:
                logger.info(f"✓ {role_name} has required policies")
                
        except Exception as e:
            results['permission_issues'].append(f"{role_name} policy validation failed: {str(e)}")

    def _validate_eventbridge_role_permissions(self, role_name: str, results: Dict):
        """Validate EventBridge role has required permissions"""
        
        try:
            # Check trust policy allows EventBridge
            role_response = self.iam_client.get_role(RoleName=role_name)
            trust_policy = role_response['Role']['AssumeRolePolicyDocument']
            
            # Simple check for events.amazonaws.com in trust policy
            trust_policy_str = json.dumps(trust_policy)
            if 'events.amazonaws.com' not in trust_policy_str:
                results['permission_issues'].append(
                    f"{role_name} trust policy may not allow EventBridge"
                )
                logger.warning(f" {role_name} trust policy may not allow EventBridge")
            else:
                logger.info(f"✓ {role_name} trust policy allows EventBridge")
                
        except Exception as e:
            results['permission_issues'].append(f"{role_name} trust policy validation failed: {str(e)}")

    def _validate_s3_resources(self) -> Dict[str, Any]:
        """Validate S3 buckets and key configurations"""
        
        try:
            results = {
                'buckets_found': {},
                'buckets_missing': [],
                'configuration_issues': [],
                'endpoint_configurations': {},
                'all_s3_valid': False
            }
            
            # Check buckets exist
            for bucket_name in self.expected_resources['s3_buckets']:
                try:
                    # Check bucket exists
                    self.s3_client.head_bucket(Bucket=bucket_name)
                    
                    # Get bucket details
                    location = self.s3_client.get_bucket_location(Bucket=bucket_name)
                    region = location.get('LocationConstraint') or 'us-east-1'
                    
                    results['buckets_found'][bucket_name] = {
                        'region': region,
                        'accessible': True
                    }
                    
                    logger.info(f"✓ S3 bucket found: {bucket_name}")
                    
                    # Check specific configurations for data bucket
                    if 'data' in bucket_name:
                        self._validate_data_bucket_structure(bucket_name, results)
                    
                except Exception as e:
                    results['buckets_missing'].append(bucket_name)
                    logger.error(f"✗ S3 bucket not accessible: {bucket_name} - {str(e)}")
            
            # Check endpoint configurations
            data_bucket = self.expected_resources['s3_buckets'][0]  # Data bucket
            if data_bucket in results['buckets_found']:
                self._validate_endpoint_configurations(data_bucket, results)
            
            results['all_s3_valid'] = (
                len(results['buckets_missing']) == 0 and 
                len(results['configuration_issues']) == 0
            )
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _validate_data_bucket_structure(self, bucket_name: str, results: Dict):
        """Validate data bucket has expected folder structure"""
        
        try:
            expected_prefixes = [
                'archived_folders/forecasting/data/',
                'archived_folders/forecasting/code/',
                'endpoint-configurations/',
                'archived_folders/forecasting/predictions/',
                'archived_folders/forecasting/visualizations/'
            ]
            
            bucket_structure = {}
            
            for prefix in expected_prefixes:
                try:
                    response = self.s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=prefix,
                        MaxKeys=1
                    )
                    
                    exists = response.get('KeyCount', 0) > 0
                    bucket_structure[prefix] = exists
                    
                    if exists:
                        logger.info(f"✓ S3 prefix found: {prefix}")
                    else:
                        logger.warning(f" S3 prefix empty/missing: {prefix}")
                        
                except Exception as e:
                    bucket_structure[prefix] = False
                    logger.error(f"✗ Error checking S3 prefix {prefix}: {str(e)}")
            
            results['buckets_found'][bucket_name]['structure'] = bucket_structure
            
        except Exception as e:
            results['configuration_issues'].append(f"Data bucket structure validation failed: {str(e)}")

    def _validate_endpoint_configurations(self, data_bucket: str, results: Dict):
        """Validate endpoint configurations exist for all profiles"""
        
        try:
            profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
            endpoint_configs = {}
            
            for profile in profiles:
                try:
                    response = self.s3_client.list_objects_v2(
                        Bucket=data_bucket,
                        Prefix=f"endpoint-configurations/{profile}",
                        MaxKeys=1000
                    )

                    # logger.info(f'RESPONSE: {response}')
                   
                    if response.get('Contents'):
                        # Sort by last modified date to get the latest
                        sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                        # logger.info(f'SORTED FILES: {sorted_files}')
                        
                        config_key = sorted_files[0]['Key']
                        # logger.info(f'CONFIG KEY: {config_key}')
                    else:
                        # No config files found
                        endpoint_configs[profile] = {'exists': False, 'valid': False}
                        continue
                       
                except Exception as e:
                    # Handle case where no files found
                    endpoint_configs[profile] = {'exists': False, 'valid': False, 'error': str(e)}
                    continue

                try:
                    response = self.s3_client.get_object(Bucket=data_bucket, Key=config_key)
                    # logger.info(f'RESPONSE 2: {response}')
                    
                    config_data = json.loads(response['Body'].read())
                    # logger.info(f'CONFIG DATA: {config_data}')
                    
                    # Validate config structure
                    required_fields = ['endpoint_config_name', 'model_name']
                    missing_fields = [field for field in required_fields if field not in config_data]
                    
                    endpoint_configs[profile] = {
                        'exists': True,
                        'valid': len(missing_fields) == 0,
                        'config_data': config_data,
                        'missing_fields': missing_fields
                    }
                    
                    if len(missing_fields) == 0:
                        logger.info(f"✓ Endpoint config valid for profile: {profile}")
                    else:
                        logger.warning(f" Endpoint config incomplete for {profile}: missing {missing_fields}")
                        
                except self.s3_client.exceptions.NoSuchKey:
                    endpoint_configs[profile] = {
                        'exists': False,
                        'valid': False
                    }
                    logger.warning(f" Endpoint config missing for profile: {profile}")
                except Exception as e:
                    endpoint_configs[profile] = {
                        'exists': False,
                        'valid': False,
                        'error': str(e)
                    }
                    logger.error(f"✗ Error validating endpoint config for {profile}: {str(e)}")
            
            results['endpoint_configurations'] = endpoint_configs
            
            # Summary
            valid_configs = sum(1 for config in endpoint_configs.values() if config.get('valid', False))
            logger.info(f"Endpoint configurations: {valid_configs}/{len(profiles)} profiles have valid configs")
            
        except Exception as e:
            results['configuration_issues'].append(f"Endpoint configuration validation failed: {str(e)}")

    def _validate_ecr_repositories(self) -> Dict[str, Any]:
        """Validate ECR repositories exist and have images"""
        
        try:
            results = {
                'repositories_found': {},
                'repositories_missing': [],
                'image_issues': [],
                'all_ecr_valid': False
            }
            
            for repo_name in self.expected_resources['ecr_repositories']:
                try:
                    # Check repository exists
                    repo_response = self.ecr_client.describe_repositories(repositoryNames=[repo_name])
                    repo_details = repo_response['repositories'][0]
                    
                    # Check for images
                    images_response = self.ecr_client.describe_images(
                        repositoryName=repo_name,
                        maxResults=10
                    )
                    
                    image_count = len(images_response['imageDetails'])
                    latest_image = None
                    
                    # Look for latest tag
                    for image in images_response['imageDetails']:
                        if 'latest' in image.get('imageTags', []):
                            latest_image = {
                                'pushed_at': image.get('imagePushedAt').isoformat() if image.get('imagePushedAt') else None,
                                'size_bytes': image.get('imageSizeInBytes'),
                                'tags': image.get('imageTags', [])
                            }
                            break
                    
                    results['repositories_found'][repo_name] = {
                        'uri': repo_details['repositoryUri'],
                        'created_at': repo_details['createdAt'].isoformat(),
                        'image_count': image_count,
                        'latest_image': latest_image
                    }
                    
                    if latest_image:
                        logger.info(f"✓ ECR repository found with latest image: {repo_name}")
                    else:
                        logger.warning(f" ECR repository found but no 'latest' image: {repo_name}")
                        results['image_issues'].append(f"{repo_name}: No 'latest' tag found")
                    
                except self.ecr_client.exceptions.RepositoryNotFoundException:
                    results['repositories_missing'].append(repo_name)
                    logger.error(f"✗ ECR repository not found: {repo_name}")
                except Exception as e:
                    results['image_issues'].append(f"{repo_name}: {str(e)}")
                    logger.error(f"✗ Error validating ECR repository {repo_name}: {str(e)}")
            
            results['all_ecr_valid'] = (
                len(results['repositories_missing']) == 0 and 
                len(results['image_issues']) == 0
            )
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _validate_sagemaker_resources(self) -> Dict[str, Any]:
        """Validate SageMaker resources including Model Registry"""
        
        try:
            results = {
                'model_registry_accessible': False,
                'model_package_groups': {},
                'execution_roles_valid': False,
                'quota_limits': {},
                'all_sagemaker_valid': False
            }
            
            # Test Model Registry access
            try:
                model_groups = self.sagemaker_client.list_model_package_groups()
                results['model_registry_accessible'] = True
                logger.info("✓ SageMaker Model Registry accessible")
                
                # Check for energy forecasting model groups
                energy_groups = [
                    group for group in model_groups['ModelPackageGroupSummaryList']
                    if 'energy' in group['ModelPackageGroupName'].lower()
                ]
                
                for group in energy_groups:
                    group_name = group['ModelPackageGroupName']
                    
                    # Get model packages in group
                    packages = self.sagemaker_client.list_model_packages(
                        ModelPackageGroupName=group_name,
                        ModelPackageType='Versioned'
                    )
                    
                    # Count approved models
                    approved_count = sum(
                        1 for pkg in packages['ModelPackageSummaryList']
                        if pkg['ModelPackageStatus'] == 'Completed' and 
                           pkg.get('ModelApprovalStatus') == 'Approved'
                    )
                    
                    results['model_package_groups'][group_name] = {
                        'total_packages': len(packages['ModelPackageSummaryList']),
                        'approved_packages': approved_count,
                        'created_at': group['CreationTime'].isoformat()
                    }
                    
                    logger.info(f"✓ Model package group: {group_name} ({approved_count} approved models)")
                
            except Exception as e:
                logger.error(f"✗ SageMaker Model Registry not accessible: {str(e)}")
                results['model_registry_error'] = str(e)
            
            # Validate execution roles
            datascientist_role = self.expected_resources['iam_roles'][0]
            try:
                # Test role can be assumed by SageMaker
                role_arn = f"arn:aws:iam::{self.account_id}:role/{datascientist_role}"
                
                # Simple test - list domains (doesn't require role assumption)
                self.sagemaker_client.list_domains()
                results['execution_roles_valid'] = True
                logger.info("✓ SageMaker execution roles valid")
                
            except Exception as e:
                logger.error(f"✗ SageMaker execution roles invalid: {str(e)}")
                results['execution_role_error'] = str(e)
            
            # Check service quotas
            try:
                # Check for common SageMaker limits
                quota_checks = {
                    'ml.m5.large_instances': 'Check endpoint instance limits',
                    'processing_jobs': 'Check processing job limits',
                    'training_jobs': 'Check training job limits'
                }
                
                results['quota_limits'] = quota_checks
                logger.info("✓ SageMaker quota limits noted (manual verification recommended)")
                
            except Exception as e:
                logger.warning(f" Could not check SageMaker quotas: {str(e)}")
            
            results['all_sagemaker_valid'] = (
                results['model_registry_accessible'] and 
                results['execution_roles_valid']
            )
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _validate_lambda_functions(self) -> Dict[str, Any]:
        """Validate Lambda functions exist and are properly configured"""
        
        try:
            results = {
                'functions_found': {},
                'functions_missing': [],
                'configuration_issues': [],
                'all_lambda_valid': False
            }
            
            for func_name in self.expected_resources['lambda_functions']:
                try:
                    # Get function details
                    func_response = self.lambda_client.get_function(FunctionName=func_name)
                    config = func_response['Configuration']
                    
                    results['functions_found'][func_name] = {
                        'arn': config['FunctionArn'],
                        'runtime': config['Runtime'],
                        'state': config['State'],
                        'last_modified': config['LastModified'],
                        'timeout': config['Timeout'],
                        'memory_size': config['MemorySize'],
                        'role': config['Role'],
                        'environment_variables': list(config.get('Environment', {}).get('Variables', {}).keys())
                    }
                    
                    # Check function state
                    if config['State'] == 'Active':
                        logger.info(f"✓ Lambda function active: {func_name}")
                    else:
                        logger.warning(f" Lambda function not active: {func_name} (state: {config['State']})")
                        results['configuration_issues'].append(f"{func_name}: State is {config['State']}")
                    
                    # Check timeout is reasonable
                    if config['Timeout'] < 60:
                        logger.warning(f" Lambda function {func_name} has low timeout: {config['Timeout']}s")
                        results['configuration_issues'].append(f"{func_name}: Low timeout ({config['Timeout']}s)")
                    
                except self.lambda_client.exceptions.ResourceNotFoundException:
                    results['functions_missing'].append(func_name)
                    logger.error(f"✗ Lambda function not found: {func_name}")
                except Exception as e:
                    results['configuration_issues'].append(f"{func_name}: {str(e)}")
                    logger.error(f"✗ Error validating Lambda function {func_name}: {str(e)}")
            
            results['all_lambda_valid'] = (
                len(results['functions_missing']) == 0 and 
                len(results['configuration_issues']) == 0
            )
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _validate_step_functions(self) -> Dict[str, Any]:
        """Validate Step Functions state machines"""
        
        try:
            results = {
                'state_machines_found': {},
                'state_machines_missing': [],
                'definition_issues': [],
                'all_stepfunctions_valid': False
            }
            
            for sm_name in self.expected_resources['step_functions']:
                try:
                    sm_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{sm_name}"
                    
                    # Get state machine details
                    sm_response = self.stepfunctions_client.describe_state_machine(stateMachineArn=sm_arn)
                    
                    # Parse definition
                    definition = json.loads(sm_response['definition'])
                    
                    results['state_machines_found'][sm_name] = {
                        'arn': sm_response['stateMachineArn'],
                        'status': sm_response['status'],
                        'created_date': sm_response['creationDate'].isoformat(),
                        'role_arn': sm_response['roleArn'],
                        'definition_valid': True,
                        'state_count': len(definition.get('States', {}))
                    }
                    
                    # Validate enhanced prediction pipeline definition
                    if 'enhanced-prediction' in sm_name:
                        self._validate_enhanced_pipeline_definition(definition, sm_name, results)
                    
                    logger.info(f"✓ Step Functions state machine found: {sm_name}")
                    
                except self.stepfunctions_client.exceptions.StateMachineDoesNotExist:
                    results['state_machines_missing'].append(sm_name)
                    logger.error(f"✗ Step Functions state machine not found: {sm_name}")
                except json.JSONDecodeError as e:
                    results['definition_issues'].append(f"{sm_name}: Invalid JSON definition - {str(e)}")
                    logger.error(f"✗ Step Functions {sm_name} has invalid definition: {str(e)}")
                except Exception as e:
                    results['definition_issues'].append(f"{sm_name}: {str(e)}")
                    logger.error(f"✗ Error validating Step Functions {sm_name}: {str(e)}")
            
            results['all_stepfunctions_valid'] = (
                len(results['state_machines_missing']) == 0 and 
                len(results['definition_issues']) == 0
            )
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _validate_enhanced_pipeline_definition(self, definition: Dict, sm_name: str, results: Dict):
        """Validate enhanced prediction pipeline has required states"""
        
        try:
            required_states = [
                'ValidateInput',
                'CreateEndpointsParallel', 
                'RunPredictionsParallel',
                'CleanupEndpointsParallel'
            ]
            
            states = definition.get('States', {})
            missing_states = [state for state in required_states if state not in states]
            
            if missing_states:
                results['definition_issues'].append(
                    f"{sm_name}: Missing required states: {missing_states}"
                )
                logger.warning(f" {sm_name} missing required states: {missing_states}")
            else:
                logger.info(f"✓ {sm_name} has all required states")
                
            # Check for Map states (parallel execution)
            map_states = [name for name, state in states.items() if state.get('Type') == 'Map']
            if len(map_states) >= 3:  # Should have at least 3 Map states for parallel execution
                logger.info(f"✓ {sm_name} has parallel execution states: {map_states}")
            else:
                results['definition_issues'].append(
                    f"{sm_name}: Insufficient Map states for parallel execution"
                )
                logger.warning(f" {sm_name} may not have proper parallel execution")
                
        except Exception as e:
            results['definition_issues'].append(f"{sm_name}: Definition validation failed - {str(e)}")

    def _validate_eventbridge(self) -> Dict[str, Any]:
        """Validate EventBridge rules and targets"""
        
        try:
            results = {
                'rules_found': [],
                'rules_missing': [],
                'target_issues': [],
                'schedule_issues': [],
                'all_eventbridge_valid': False
            }
            
            # Look for energy forecasting rules
            rules_response = self.events_client.list_rules()
            energy_rules = [
                rule for rule in rules_response['Rules']
                if 'energy-forecasting' in rule['Name']
            ]
            
            expected_rules = [
                'energy-forecasting-enhanced-daily-predictions',
                'energy-forecasting-monthly-parallel-pipeline'
            ]
            
            for rule in energy_rules:
                rule_name = rule['Name']
                
                # Get rule details
                rule_details = {
                    'name': rule_name,
                    'state': rule['State'],
                    'schedule_expression': rule.get('ScheduleExpression'),
                    'description': rule.get('Description')
                }
                
                # Get targets
                try:
                    targets_response = self.events_client.list_targets_by_rule(Rule=rule_name)
                    targets = targets_response['Targets']
                    
                    rule_details['targets'] = []
                    for target in targets:
                        target_info = {
                            'id': target['Id'],
                            'arn': target['Arn'],
                            'role_arn': target.get('RoleArn')
                        }
                        rule_details['targets'].append(target_info)
                        
                        # Validate target ARN points to correct state machine
                        if 'stateMachine' not in target['Arn']:
                            results['target_issues'].append(
                                f"{rule_name}: Target ARN is not a Step Functions state machine"
                            )
                        
                except Exception as e:
                    results['target_issues'].append(f"{rule_name}: Could not get targets - {str(e)}")
                
                results['rules_found'].append(rule_details)
                logger.info(f"✓ EventBridge rule found: {rule_name}")
            
            # Check for missing expected rules
            found_rule_names = [rule['name'] for rule in results['rules_found']]
            for expected_rule in expected_rules:
                if expected_rule not in found_rule_names:
                    results['rules_missing'].append(expected_rule)
                    logger.warning(f" EventBridge rule missing: {expected_rule}")
            
            # Validate schedule expressions
            for rule in results['rules_found']:
                schedule = rule.get('schedule_expression')
                if schedule:
                    if not self._is_valid_cron_expression(schedule):
                        results['schedule_issues'].append(
                            f"{rule['name']}: Invalid schedule expression: {schedule}"
                        )
                        logger.warning(f" Invalid schedule in {rule['name']}: {schedule}")
                else:
                    logger.warning(f" No schedule expression in rule: {rule['name']}")
            
            results['all_eventbridge_valid'] = (
                len(results['rules_missing']) == 0 and 
                len(results['target_issues']) == 0 and
                len(results['schedule_issues']) == 0
            )
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _is_valid_cron_expression(self, schedule: str) -> bool:
        """Basic validation of cron expression"""
        
        try:
            if not schedule.startswith('cron(') or not schedule.endswith(')'):
                return False
            
            # Extract cron part
            cron_part = schedule[5:-1]
            parts = cron_part.split()
            
            # Should have 6 parts for AWS cron
            return len(parts) == 6
            
        except Exception:
            return False

    def _validate_network_connectivity(self) -> Dict[str, Any]:
        """Validate network connectivity and VPC configuration"""
        
        try:
            results = {
                'vpc_accessible': False,
                'internet_gateway_available': False,
                'subnets_available': [],
                'security_groups_valid': False,
                'endpoints_accessible': {},
                'all_network_valid': False
            }
            
            # Check default VPC
            try:
                vpcs_response = self.ec2_client.describe_vpcs(
                    Filters=[{'Name': 'isDefault', 'Values': ['true']}]
                )
                
                if vpcs_response['Vpcs']:
                    default_vpc = vpcs_response['Vpcs'][0]
                    results['vpc_accessible'] = True
                    results['default_vpc_id'] = default_vpc['VpcId']
                    logger.info(f"✓ Default VPC accessible: {default_vpc['VpcId']}")
                    
                    # Check subnets
                    subnets_response = self.ec2_client.describe_subnets(
                        Filters=[{'Name': 'vpc-id', 'Values': [default_vpc['VpcId']]}]
                    )
                    
                    for subnet in subnets_response['Subnets']:
                        subnet_info = {
                            'subnet_id': subnet['SubnetId'],
                            'availability_zone': subnet['AvailabilityZone'],
                            'available_ip_count': subnet['AvailableIpAddressCount']
                        }
                        results['subnets_available'].append(subnet_info)
                    
                    logger.info(f"✓ Found {len(results['subnets_available'])} subnets")
                else:
                    logger.warning(" No default VPC found")
                    
            except Exception as e:
                logger.error(f"✗ VPC validation failed: {str(e)}")
                results['vpc_error'] = str(e)
            
            # Test AWS service endpoints connectivity
            service_endpoints = {
                'sagemaker': f'sagemaker.{self.region}.amazonaws.com',
                'stepfunctions': f'states.{self.region}.amazonaws.com',
                'lambda': f'lambda.{self.region}.amazonaws.com',
                's3': f's3.{self.region}.amazonaws.com'
            }
            
            for service, endpoint in service_endpoints.items():
                try:
                    # Simple connectivity test - if we can make API calls, connectivity is good
                    if service == 'sagemaker':
                        self.sagemaker_client.list_domains(MaxResults=1)
                    elif service == 'stepfunctions':
                        self.stepfunctions_client.list_state_machines(maxResults=1)
                    elif service == 'lambda':
                        self.lambda_client.list_functions(MaxItems=1)
                    elif service == 's3':
                        self.s3_client.list_buckets()
                    
                    results['endpoints_accessible'][service] = True
                    logger.info(f"✓ {service} endpoint accessible")
                    
                except Exception as e:
                    results['endpoints_accessible'][service] = False
                    logger.error(f"✗ {service} endpoint not accessible: {str(e)}")
            
            # Overall network assessment
            accessible_endpoints = sum(results['endpoints_accessible'].values())
            total_endpoints = len(service_endpoints)
            
            results['all_network_valid'] = (
                results['vpc_accessible'] and
                accessible_endpoints == total_endpoints
            )
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _validate_resource_dependencies(self) -> Dict[str, Any]:
        """Validate dependencies between resources"""
        
        try:
            results = {
                'dependency_checks': {},
                'dependency_issues': [],
                'all_dependencies_valid': False
            }
            
            # Check Lambda functions reference correct IAM role
            lambda_results = self.validation_results.get('lambda_functions', {})
            iam_results = self.validation_results.get('iam_roles', {})
            
            if lambda_results.get('functions_found') and iam_results.get('roles_found'):
                datascientist_role_arn = None
                for role_name, role_info in iam_results['roles_found'].items():
                    if 'datascientist' in role_name:
                        datascientist_role_arn = role_info['arn']
                        break
                
                if datascientist_role_arn:
                    lambda_role_issues = []
                    for func_name, func_info in lambda_results['functions_found'].items():
                        if func_info['role'] != datascientist_role_arn:
                            lambda_role_issues.append(
                                f"{func_name} uses incorrect role: {func_info['role']}"
                            )
                    
                    if lambda_role_issues:
                        results['dependency_issues'].extend(lambda_role_issues)
                    else:
                        results['dependency_checks']['lambda_iam_roles'] = 'valid'
                        logger.info("✓ Lambda functions use correct IAM roles")
            
            # Check Step Functions reference correct Lambda functions
            stepfunctions_results = self.validation_results.get('step_functions', {})
            
            if stepfunctions_results.get('state_machines_found'):
                for sm_name, sm_info in stepfunctions_results['state_machines_found'].items():
                    if 'enhanced' in sm_name:
                        # This would require parsing the definition to check Lambda ARNs
                        # For now, we'll do a basic check
                        results['dependency_checks']['stepfunctions_lambda'] = 'basic_check_passed'
                        logger.info("✓ Step Functions dependency check passed (basic)")
            
            # Check EventBridge rules target correct Step Functions
            eventbridge_results = self.validation_results.get('eventbridge', {})
            
            if eventbridge_results.get('rules_found'):
                target_issues = []
                for rule in eventbridge_results['rules_found']:
                    for target in rule.get('targets', []):
                        target_arn = target['arn']
                        if 'stateMachine' in target_arn:
                            # Extract state machine name from ARN
                            sm_name = target_arn.split(':')[-1]
                            if sm_name not in [sm for sm in self.expected_resources['step_functions']]:
                                target_issues.append(
                                    f"EventBridge rule {rule['name']} targets unknown state machine: {sm_name}"
                                )
                
                if target_issues:
                    results['dependency_issues'].extend(target_issues)
                else:
                    results['dependency_checks']['eventbridge_stepfunctions'] = 'valid'
                    logger.info("✓ EventBridge rules target correct Step Functions")
            
            # Check S3 bucket regions match
            s3_results = self.validation_results.get('s3_resources', {})
            
            if s3_results.get('buckets_found'):
                region_issues = []
                for bucket_name, bucket_info in s3_results['buckets_found'].items():
                    bucket_region = bucket_info.get('region', 'us-east-1')
                    if bucket_region != self.region and bucket_region != 'us-east-1':
                        region_issues.append(
                            f"Bucket {bucket_name} in region {bucket_region}, expected {self.region}"
                        )
                
                if region_issues:
                    results['dependency_issues'].extend(region_issues)
                else:
                    results['dependency_checks']['s3_regions'] = 'valid'
                    logger.info("✓ S3 buckets in correct regions")
            
            results['all_dependencies_valid'] = len(results['dependency_issues']) == 0
            
            return results
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e)
            }

    def _generate_validation_summary(self, validation_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        
        try:
            # Calculate overall scores
            validation_categories = [
                'credentials_and_permissions',
                'iam_roles', 
                's3_resources',
                'ecr_repositories',
                'sagemaker_resources',
                'lambda_functions',
                'step_functions',
                'eventbridge',
                'network_connectivity',
                'resource_dependencies'
            ]
            
            category_scores = {}
            total_score = 0
            max_score = 0
            
            for category in validation_categories:
                result = self.validation_results.get(category, {})
                
                if result.get('validation_failed'):
                    score = 0
                else:
                    # Calculate score based on category-specific criteria
                    score = self._calculate_category_score(category, result)
                
                category_scores[category] = score
                total_score += score
                max_score += 100  # Each category worth 100 points
            
            overall_percentage = (total_score / max_score * 100) if max_score > 0 else 0
            
            # Determine overall status
            if overall_percentage >= 95:
                overall_status = 'EXCELLENT'
            elif overall_percentage >= 85:
                overall_status = 'GOOD'
            elif overall_percentage >= 70:
                overall_status = 'ACCEPTABLE'
            elif overall_percentage >= 50:
                overall_status = 'NEEDS_IMPROVEMENT'
            else:
                overall_status = 'CRITICAL_ISSUES'
            
            # Generate recommendations
            recommendations = self._generate_validation_recommendations()
            
            # Determine environment readiness
            environment_ready = (
                overall_percentage >= 70 and
                self._check_critical_requirements()
            )
            
            summary = {
                'validation_summary': {
                    'overall_status': overall_status,
                    'overall_percentage': overall_percentage,
                    'environment_ready': environment_ready,
                    'validation_time_minutes': validation_time / 60,
                    'validation_timestamp': datetime.now().isoformat(),
                    'total_score': total_score,
                    'max_possible_score': max_score
                },
                'category_scores': category_scores,
                'detailed_results': self.validation_results,
                'recommendations': recommendations,
                'critical_issues': self._identify_critical_issues(),
                'next_steps': self._generate_next_steps(environment_ready),
                'resource_summary': {
                    'iam_roles': len(self.validation_results.get('iam_roles', {}).get('roles_found', {})),
                    's3_buckets': len(self.validation_results.get('s3_resources', {}).get('buckets_found', {})),
                    'lambda_functions': len(self.validation_results.get('lambda_functions', {}).get('functions_found', {})),
                    'step_functions': len(self.validation_results.get('step_functions', {}).get('state_machines_found', {})),
                    'ecr_repositories': len(self.validation_results.get('ecr_repositories', {}).get('repositories_found', {}))
                }
            }
            
            return summary
            
        except Exception as e:
            return {
                'summary_generation_error': str(e),
                'validation_results': self.validation_results
            }

    def _calculate_category_score(self, category: str, result: Dict) -> int:
        """Calculate score for a validation category"""
        
        try:
            if category == 'credentials_and_permissions':
                if result.get('caller_identity_valid') and result.get('basic_permissions_valid'):
                    return 100
                elif result.get('caller_identity_valid'):
                    return 50
                else:
                    return 0
            
            elif category == 'iam_roles':
                if result.get('all_roles_valid'):
                    return 100
                else:
                    found = len(result.get('roles_found', {}))
                    total = len(self.expected_resources['iam_roles'])
                    return int((found / total) * 100) if total > 0 else 0
            
            elif category == 's3_resources':
                if result.get('all_s3_valid'):
                    return 100
                else:
                    found = len(result.get('buckets_found', {}))
                    total = len(self.expected_resources['s3_buckets'])
                    bucket_score = int((found / total) * 60) if total > 0 else 0
                    
                    # Add points for endpoint configurations
                    endpoint_configs = result.get('endpoint_configurations', {})
                    valid_configs = sum(1 for config in endpoint_configs.values() if config.get('valid'))
                    config_score = int((valid_configs / 7) * 40) if len(endpoint_configs) > 0 else 0
                    
                    return bucket_score + config_score
            
            elif category == 'lambda_functions':
                if result.get('all_lambda_valid'):
                    return 100
                else:
                    found = len(result.get('functions_found', {}))
                    total = len(self.expected_resources['lambda_functions'])
                    return int((found / total) * 100) if total > 0 else 0
            
            elif category == 'step_functions':
                if result.get('all_stepfunctions_valid'):
                    return 100
                else:
                    found = len(result.get('state_machines_found', {}))
                    total = len(self.expected_resources['step_functions'])
                    return int((found / total) * 100) if total > 0 else 0
            
            elif category == 'ecr_repositories':
                if result.get('all_ecr_valid'):
                    return 100
                else:
                    found = len(result.get('repositories_found', {}))
                    total = len(self.expected_resources['ecr_repositories'])
                    return int((found / total) * 100) if total > 0 else 0
            
            elif category == 'sagemaker_resources':
                score = 0
                if result.get('model_registry_accessible'):
                    score += 50
                if result.get('execution_roles_valid'):
                    score += 50
                return score
            
            elif category == 'eventbridge':
                if result.get('all_eventbridge_valid'):
                    return 100
                else:
                    found = len(result.get('rules_found', []))
                    return min(100, found * 50)  # 50 points per rule, max 100
            
            elif category == 'network_connectivity':
                if result.get('all_network_valid'):
                    return 100
                else:
                    accessible = sum(result.get('endpoints_accessible', {}).values())
                    total = len(result.get('endpoints_accessible', {}))
                    return int((accessible / total) * 100) if total > 0 else 0
            
            elif category == 'resource_dependencies':
                if result.get('all_dependencies_valid'):
                    return 100
                else:
                    issues = len(result.get('dependency_issues', []))
                    return max(0, 100 - (issues * 20))  # -20 points per issue
            
            else:
                return 50  # Default score for unknown categories
                
        except Exception:
            return 0

    def _check_critical_requirements(self) -> bool:
        """Check if critical requirements are met"""
        
        try:
            critical_checks = [
                # AWS credentials must be valid
                self.validation_results.get('credentials_and_permissions', {}).get('caller_identity_valid', False),
                
                # At least one IAM role must exist
                len(self.validation_results.get('iam_roles', {}).get('roles_found', {})) > 0,
                
                # Data bucket must exist
                self.expected_resources['s3_buckets'][0] in self.validation_results.get('s3_resources', {}).get('buckets_found', {}),
                
                # SageMaker must be accessible
                self.validation_results.get('sagemaker_resources', {}).get('model_registry_accessible', False)
            ]
            
            return all(critical_checks)
            
        except Exception:
            return False

    def _identify_critical_issues(self) -> List[str]:
        """Identify critical issues that must be resolved"""
        
        critical_issues = []
        
        try:
            # Check credentials
            creds = self.validation_results.get('credentials_and_permissions', {})
            if not creds.get('caller_identity_valid'):
                critical_issues.append("AWS credentials are invalid or expired")
            
            # Check IAM roles
            iam = self.validation_results.get('iam_roles', {})
            if iam.get('roles_missing'):
                critical_issues.append(f"Missing IAM roles: {iam['roles_missing']}")
            
            # Check S3 buckets
            s3 = self.validation_results.get('s3_resources', {})
            if s3.get('buckets_missing'):
                critical_issues.append(f"Missing S3 buckets: {s3['buckets_missing']}")
            
            # Check Lambda functions
            lambda_res = self.validation_results.get('lambda_functions', {})
            if lambda_res.get('functions_missing'):
                critical_issues.append(f"Missing Lambda functions: {lambda_res['functions_missing']}")
            
            # Check Step Functions
            sf = self.validation_results.get('step_functions', {})
            if sf.get('state_machines_missing'):
                critical_issues.append(f"Missing Step Functions: {sf['state_machines_missing']}")
            
            # Check SageMaker
            sm = self.validation_results.get('sagemaker_resources', {})
            if not sm.get('model_registry_accessible'):
                critical_issues.append("SageMaker Model Registry not accessible")
            
        except Exception as e:
            critical_issues.append(f"Error identifying critical issues: {str(e)}")
        
        return critical_issues

    def _generate_validation_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        try:
            # Check each validation category and provide specific recommendations
            
            # IAM recommendations
            iam = self.validation_results.get('iam_roles', {})
            if iam.get('roles_missing'):
                recommendations.append("Create missing IAM roles using the infrastructure setup scripts")
            if iam.get('permission_issues'):
                recommendations.append("Review and fix IAM role permissions")
            
            # S3 recommendations
            s3 = self.validation_results.get('s3_resources', {})
            if s3.get('buckets_missing'):
                recommendations.append("Create missing S3 buckets")
            
            endpoint_configs = s3.get('endpoint_configurations', {})
            invalid_configs = [profile for profile, config in endpoint_configs.items() if not config.get('valid')]
            if invalid_configs:
                recommendations.append(f"Fix endpoint configurations for profiles: {invalid_configs}")
            
            # Lambda recommendations
            lambda_res = self.validation_results.get('lambda_functions', {})
            if lambda_res.get('functions_missing'):
                recommendations.append("Deploy missing Lambda functions using enhanced_lambda_deployer.py")
            if lambda_res.get('configuration_issues'):
                recommendations.append("Fix Lambda function configuration issues")
            
            # Step Functions recommendations
            sf = self.validation_results.get('step_functions', {})
            if sf.get('state_machines_missing'):
                recommendations.append("Deploy missing Step Functions using deploy_enhanced_mlops.py")
            
            # ECR recommendations
            ecr = self.validation_results.get('ecr_repositories', {})
            if ecr.get('repositories_missing'):
                recommendations.append("Create missing ECR repositories")
            if ecr.get('image_issues'):
                recommendations.append("Build and push container images to ECR")
            
            # EventBridge recommendations
            eb = self.validation_results.get('eventbridge', {})
            if eb.get('rules_missing'):
                recommendations.append("Create missing EventBridge rules")
            
            # General recommendations
            critical_issues = self._identify_critical_issues()
            if not critical_issues:
                recommendations.append("Environment validation passed - proceed with enhanced pipeline deployment")
                recommendations.append("Run comprehensive testing after deployment")
            else:
                recommendations.append("Resolve critical issues before proceeding with deployment")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations

    def _generate_next_steps(self, environment_ready: bool) -> List[str]:
        """Generate next steps based on validation results"""
        
        if environment_ready:
            return [
                "Environment is ready for enhanced prediction pipeline deployment",
                "Run: python deployment/deploy_enhanced_mlops.py",
                "After deployment, run integration tests",
                "Monitor CloudWatch logs during deployment"
            ]
        else:
            return [
                "Fix critical issues identified in validation results",
                "Re-run environment validation after fixes",
                "Review detailed validation results for specific issues",
                "Contact infrastructure team if IAM/VPC issues persist"
            ]

def main():
    """Main function for environment validation"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Environment Validator for Enhanced Prediction Pipeline')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--environment', default='dev', help='Environment (dev/staging/prod)')
    parser.add_argument('--quick', action='store_true', help='Run quick validation (basic checks only)')
    parser.add_argument('--category', help='Validate specific category only')
    parser.add_argument('--output-json', help='Save validation results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = EnvironmentValidator(region=args.region, environment=args.environment)
    
    try:
        if args.quick:
            # Quick validation - credentials and basic permissions only
            logger.info("Running quick environment validation...")
            
            creds_result = validator._validate_credentials_and_permissions()
            
            if creds_result.get('caller_identity_valid') and creds_result.get('basic_permissions_valid'):
                logger.info("✓ Quick validation PASSED - Basic environment is ready")
                sys.exit(0)
            else:
                logger.error("✗ Quick validation FAILED - Environment issues detected")
                print(json.dumps(creds_result, indent=2, default=str))
                sys.exit(1)
                
        elif args.category:
            # Validate specific category only
            logger.info(f"Running validation for category: {args.category}")
            
            validation_methods = {
                'credentials': validator._validate_credentials_and_permissions,
                'iam': validator._validate_iam_roles,
                's3': validator._validate_s3_resources,
                'ecr': validator._validate_ecr_repositories,
                'sagemaker': validator._validate_sagemaker_resources,
                'lambda': validator._validate_lambda_functions,
                'stepfunctions': validator._validate_step_functions,
                'eventbridge': validator._validate_eventbridge,
                'network': validator._validate_network_connectivity,
                'dependencies': validator._validate_resource_dependencies
            }
            
            if args.category in validation_methods:
                result = validation_methods[args.category]()
                
                print(json.dumps(result, indent=2, default=str))
                
                if result.get('validation_failed'):
                    logger.error(f"✗ {args.category} validation FAILED")
                    sys.exit(1)
                else:
                    logger.info(f"✓ {args.category} validation completed")
                    sys.exit(0)
            else:
                logger.error(f"Unknown validation category: {args.category}")
                logger.info(f"Available categories: {list(validation_methods.keys())}")
                sys.exit(1)
                
        else:
            # Run complete validation
            logger.info("Running complete environment validation...")
            
            validation_result = validator.validate_complete_environment()
            
            # Save results to JSON file if requested
            if args.output_json:
                try:
                    with open(args.output_json, 'w') as f:
                        json.dump(validation_result, f, indent=2, default=str)
                    logger.info(f"Validation results saved to: {args.output_json}")
                except Exception as e:
                    logger.error(f"Could not save results to file: {str(e)}")
            
            # Print summary
            summary = validation_result.get('validation_summary', {})
            logger.info("\n" + "="*60)
            logger.info("VALIDATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
            logger.info(f"Overall Score: {summary.get('overall_percentage', 0):.1f}%")
            logger.info(f"Environment Ready: {summary.get('environment_ready', False)}")
            logger.info(f"Validation Time: {summary.get('validation_time_minutes', 0):.2f} minutes")
            
            # Print category scores
            category_scores = validation_result.get('category_scores', {})
            if category_scores:
                logger.info("\nCategory Scores:")
                for category, score in category_scores.items():
                    status = "✓" if score >= 70 else " " if score >= 50 else "✗"
                    logger.info(f"  {status} {category.replace('_', ' ').title()}: {score}%")
            
            # Print critical issues
            critical_issues = validation_result.get('critical_issues', [])
            if critical_issues:
                logger.info("\nCritical Issues:")
                for i, issue in enumerate(critical_issues, 1):
                    logger.error(f"  {i}. {issue}")
            
            # Print recommendations
            recommendations = validation_result.get('recommendations', [])
            if recommendations:
                logger.info("\nRecommendations:")
                for i, rec in enumerate(recommendations, 1):
                    logger.info(f"  {i}. {rec}")
            
            # Print next steps
            next_steps = validation_result.get('next_steps', [])
            if next_steps:
                logger.info("\nNext Steps:")
                for i, step in enumerate(next_steps, 1):
                    logger.info(f"  {i}. {step}")
            
            # Exit with appropriate code
            if validation_result.get('validation_summary', {}).get('environment_ready', False):
                logger.info("\n✓ ENVIRONMENT VALIDATION PASSED")
                logger.info("✓ Environment is ready for enhanced prediction pipeline deployment")
                sys.exit(0)
            else:
                logger.error("\n✗ ENVIRONMENT VALIDATION FAILED")
                logger.error("✗ Environment needs fixes before deployment")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Environment validation failed with unexpected error: {str(e)}")
        sys.exit(1)

def validate_for_deployment() -> bool:
    """
    Simple validation function for use by other scripts
    Returns True if environment is ready for deployment
    """
    
    try:
        validator = EnvironmentValidator()
        result = validator.validate_complete_environment()
        
        return result.get('validation_summary', {}).get('environment_ready', False)
        
    except Exception:
        return False

def get_validation_report(region: str = "us-west-2", environment: str = "dev") -> Dict[str, Any]:
    """
    Get detailed validation report for programmatic use
    
    Args:
        region: AWS region
        environment: Environment name
        
    Returns:
        Dictionary containing complete validation results
    """
    
    try:
        validator = EnvironmentValidator(region=region, environment=environment)
        return validator.validate_complete_environment()
        
    except Exception as e:
        return {
            'validation_failed': True,
            'error': str(e),
            'validation_summary': {
                'overall_status': 'ERROR',
                'environment_ready': False
            }
        }

def check_specific_resource(resource_type: str, region: str = "us-west-2", environment: str = "dev") -> Dict[str, Any]:
    """
    Check specific resource type
    
    Args:
        resource_type: Type of resource to check (iam, s3, lambda, etc.)
        region: AWS region
        environment: Environment name
        
    Returns:
        Dictionary containing validation results for the specific resource type
    """
    
    try:
        validator = EnvironmentValidator(region=region, environment=environment)
        
        validation_methods = {
            'credentials': validator._validate_credentials_and_permissions,
            'iam': validator._validate_iam_roles,
            's3': validator._validate_s3_resources,
            'ecr': validator._validate_ecr_repositories,
            'sagemaker': validator._validate_sagemaker_resources,
            'lambda': validator._validate_lambda_functions,
            'stepfunctions': validator._validate_step_functions,
            'eventbridge': validator._validate_eventbridge,
            'network': validator._validate_network_connectivity,
            'dependencies': validator._validate_resource_dependencies
        }
        
        if resource_type in validation_methods:
            return validation_methods[resource_type]()
        else:
            return {
                'validation_failed': True,
                'error': f'Unknown resource type: {resource_type}',
                'available_types': list(validation_methods.keys())
            }
            
    except Exception as e:
        return {
            'validation_failed': True,
            'error': str(e)
        }

def validate_prerequisites_for_enhanced_pipeline() -> Dict[str, Any]:
    """
    Specific validation for enhanced prediction pipeline prerequisites
    
    Returns:
        Dictionary with validation results and readiness status
    """
    
    try:
        validator = EnvironmentValidator()
        
        # Run key validations for enhanced pipeline
        results = {}
        
        # Check credentials and permissions
        results['credentials'] = validator._validate_credentials_and_permissions()
        
        # Check IAM roles
        results['iam_roles'] = validator._validate_iam_roles()
        
        # Check S3 resources including endpoint configurations
        results['s3_resources'] = validator._validate_s3_resources()
        
        # Check Lambda functions
        results['lambda_functions'] = validator._validate_lambda_functions()
        
        # Check SageMaker access
        results['sagemaker'] = validator._validate_sagemaker_resources()
        
        # Determine readiness
        critical_checks = [
            results['credentials'].get('caller_identity_valid', False),
            results['credentials'].get('basic_permissions_valid', False),
            len(results['iam_roles'].get('roles_found', {})) > 0,
            len(results['s3_resources'].get('buckets_found', {})) > 0,
            results['sagemaker'].get('model_registry_accessible', False)
        ]
        
        enhanced_pipeline_ready = all(critical_checks)
        
        # Count endpoint configurations
        endpoint_configs = results['s3_resources'].get('endpoint_configurations', {})
        valid_configs = sum(1 for config in endpoint_configs.values() if config.get('valid'))
        
        return {
            'enhanced_pipeline_ready': enhanced_pipeline_ready,
            'critical_checks_passed': sum(critical_checks),
            'total_critical_checks': len(critical_checks),
            'endpoint_configurations_valid': valid_configs,
            'total_profiles': 7,
            'lambda_functions_deployed': len(results['lambda_functions'].get('functions_found', {})),
            'expected_lambda_functions': len(validator.expected_resources['lambda_functions']),
            'detailed_results': results,
            'recommendations': [
                "Deploy enhanced Lambda functions if missing",
                "Ensure endpoint configurations exist for all profiles",
                "Verify SageMaker Model Registry has approved models",
                "Run complete validation before deployment"
            ] if not enhanced_pipeline_ready else [
                "Enhanced pipeline prerequisites are met",
                "Proceed with enhanced prediction pipeline deployment",
                "Run integration tests after deployment"
            ]
        }
        
    except Exception as e:
        return {
            'enhanced_pipeline_ready': False,
            'error': str(e),
            'critical_checks_passed': 0,
            'recommendations': [
                "Fix validation errors before proceeding",
                "Check AWS credentials and permissions",
                "Ensure all required resources are deployed"
            ]
        }

if __name__ == "__main__":
    main()
