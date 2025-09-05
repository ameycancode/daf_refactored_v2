#!/usr/bin/env python3
"""
Complete Enhanced MLOps Deployment Script
sdcp_code/deployment/deploy_enhanced_mlops.py

Deploys the complete enhanced prediction pipeline including:
- Enhanced Lambda functions
- Enhanced Step Functions
- Enhanced EventBridge rules
- Container updates
- Environment validation
"""

import json
import subprocess
import boto3
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import time

# Add project paths for sdcp_code structure
current_dir = os.path.dirname(os.path.abspath(__file__))
sdcp_code_dir = os.path.dirname(current_dir)  # sdcp_code directory
project_root = os.path.dirname(sdcp_code_dir)  # repository root
sys.path.append(sdcp_code_dir)
sys.path.append(project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMLOpsDeployer:
    """
    Complete deployment manager for the enhanced MLOps pipeline
    """
    
    def __init__(self, region: str = "us-west-2", environment: str = "dev"):
        """Initialize the enhanced MLOps deployer"""
        
        self.region = region
        self.environment = environment
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # AWS clients
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.events_client = boto3.client('events', region_name=region)
        self.ecr_client = boto3.client('ecr', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.iam_client = boto3.client('iam')
        
        # Environment-aware configuration
        self.config = {
            'region': region,
            'account_id': self.account_id,
            'data_bucket': f'sdcp-{environment}-sagemaker-energy-forecasting-data',
            'model_bucket': f'sdcp-{environment}-sagemaker-energy-forecasting-models',
            'datascientist_role': f'arn:aws:iam::{self.account_id}:role/sdcp-{environment}-sagemaker-energy-forecasting-datascientist-role',
            'eventbridge_role': f'arn:aws:iam::{self.account_id}:role/EnergyForecastingEventBridgeRole',
            'training_state_machine': f'energy-forecasting-{environment}-training-pipeline',
            'enhanced_prediction_state_machine': f'energy-forecasting-{environment}-enhanced-prediction-pipeline',
            'legacy_prediction_state_machine': f'energy-forecasting-{environment}-daily-predictions',
            'containers': [f'energy-preprocessing', f'energy-training']  # environment-agnostic names
        }
        
        # Deployment components
        self.deployment_components = [
            'validate_environment',
            'deploy_enhanced_lambdas',
            'deploy_enhanced_step_functions',
            'deploy_enhanced_containers',
            'setup_enhanced_eventbridge',
            'validate_deployment',
            'run_integration_tests'
        ]
        
        logger.info(f"Enhanced MLOps Deployer initialized for {environment} environment")
        logger.info(f"Using sdcp_code structure from: {sdcp_code_dir}")

    def deploy_complete_enhanced_pipeline(self) -> Dict[str, Any]:
        """Deploy the complete enhanced prediction pipeline"""
        
        logger.info("="*100)
        logger.info("DEPLOYING COMPLETE ENHANCED ENERGY FORECASTING MLOPS PIPELINE")
        logger.info("="*100)
        logger.info(f"Account: {self.account_id}")
        logger.info(f"Region: {self.region}")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        deployment_start_time = time.time()
        deployment_results = {}
        
        try:
            # Step 1: Validate Environment
            logger.info("\n" + "="*60)
            logger.info("STEP 1: VALIDATING ENVIRONMENT")
            logger.info("="*60)
            
            validation_result = self._validate_environment()
            deployment_results['environment_validation'] = validation_result
            
            if not validation_result.get('environment_ready', False):
                raise Exception("Environment validation failed - cannot proceed with deployment")
            
            # Step 2: Deploy Enhanced Lambda Functions
            logger.info("\n" + "="*60)
            logger.info("STEP 2: DEPLOYING ENHANCED LAMBDA FUNCTIONS")
            logger.info("="*60)
            
            lambda_result = self._deploy_enhanced_lambdas()
            deployment_results['enhanced_lambdas'] = lambda_result
            
            # Step 3: Deploy Enhanced Step Functions
            logger.info("\n" + "="*60)
            logger.info("STEP 3: DEPLOYING ENHANCED STEP FUNCTIONS")
            logger.info("="*60)
            
            stepfunctions_result = self._deploy_enhanced_step_functions()
            deployment_results['enhanced_step_functions'] = stepfunctions_result
            
            # Step 4: Build and Push Containers
            logger.info("\n" + "="*60)
            logger.info("STEP 4: BUILDING AND PUSHING CONTAINERS")
            logger.info("="*60)
            
            containers_result = self._build_and_push_containers()
            deployment_results['enhanced_containers'] = containers_result
            
            # Step 5: Setup Enhanced EventBridge
            logger.info("\n" + "="*60)
            logger.info("STEP 5: SETTING UP ENHANCED EVENTBRIDGE")
            logger.info("="*60)
            
            eventbridge_result = self._setup_enhanced_eventbridge()
            deployment_results['enhanced_eventbridge'] = eventbridge_result
            
            # Step 6: Validate Deployment
            logger.info("\n" + "="*60)
            logger.info("STEP 6: VALIDATING COMPLETE DEPLOYMENT")
            logger.info("="*60)
            
            deployment_validation = self._validate_complete_deployment()
            deployment_results['deployment_validation'] = deployment_validation
            
            # Step 7: Run Integration Tests
            logger.info("\n" + "="*60)
            logger.info("STEP 7: RUNNING INTEGRATION TESTS")
            logger.info("="*60)
            
            integration_tests = self._run_integration_tests()
            deployment_results['integration_tests'] = integration_tests
            
            # Generate final summary
            deployment_time = time.time() - deployment_start_time
            summary = self._generate_deployment_summary(deployment_results, deployment_time)

            logger.info(f"DEPLOYMENT RESULTS: {deployment_results}")
            logger.info(f"DEPLOYMENT SUMMARY: {summary['deployment_summary']}")
            
            deployment_summary = summary['deployment_summary']
            overall_success = deployment_summary['overall_success']
            
            logger.info("\n" + "="*100)
            logger.info("ENHANCED MLOPS DEPLOYMENT COMPLETED")
            logger.info("="*100)
            logger.info(f"Total deployment time: {deployment_time / 60:.2f} minutes")
            logger.info(f"Overall success: {overall_success}")
            
            return summary
            
        except Exception as e:
            deployment_time = time.time() - deployment_start_time
            logger.error(f"Enhanced MLOps deployment failed after {deployment_time / 60:.2f} minutes: {str(e)}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'deployment_time_minutes': deployment_time / 60,
                'partial_results': deployment_results,
                'failed_at': self._identify_failure_point(deployment_results)
            }

    def _validate_environment(self) -> Dict[str, Any]:
        """Validate the deployment environment"""
        
        try:
            # Updated import path for sdcp_code structure
            from validate_environment import EnvironmentValidator
            
            validator = EnvironmentValidator(region=self.region, environment=self.environment)
            validation_result = validator.validate_complete_environment()

            logger.info(f"VALIDATION SUMMARY: {validation_result['validation_summary']}")
            validation_summary = validation_result['validation_summary']
            overall_status = validation_summary['overall_status']
            
            logger.info(f"Environment validation: {overall_status}")
            
            return validation_summary
            
        except ImportError:
            logger.warning("Environment validator not available - running basic validation")
            return self._basic_environment_validation()
        except Exception as e:
            logger.error(f"Environment validation failed: {str(e)}")
            return {
                'environment_ready': False,
                'error': str(e)
            }

    def _basic_environment_validation(self) -> Dict[str, Any]:
        """Basic environment validation if validator module not available"""
        
        try:
            validation_results = {
                'iam_roles_exist': False,
                's3_buckets_exist': False,
                'ecr_repositories_exist': False,
                'permissions_valid': False
            }
            
            # Check IAM roles
            try:
                role_name = self.config['datascientist_role'].split('/')[-1]
                self.iam_client.get_role(RoleName=role_name)
                validation_results['iam_roles_exist'] = True
                logger.info("✓ IAM roles validated")
            except Exception as e:
                logger.error(f"✗ IAM role validation failed: {str(e)}")
            
            # Check S3 buckets
            try:
                self.s3_client.head_bucket(Bucket=self.config['data_bucket'])
                self.s3_client.head_bucket(Bucket=self.config['model_bucket'])
                validation_results['s3_buckets_exist'] = True
                logger.info("✓ S3 buckets validated")
            except Exception as e:
                logger.error(f"✗ S3 bucket validation failed: {str(e)}")
            
            # Check ECR repositories
            try:
                repositories = self.ecr_client.describe_repositories()
                energy_repos = [repo for repo in repositories['repositories'] 
                              if 'energy' in repo['repositoryName']]
                validation_results['ecr_repositories_exist'] = len(energy_repos) >= 2
                logger.info(f"✓ ECR repositories validated: {len(energy_repos)} found")
            except Exception as e:
                logger.error(f"✗ ECR repository validation failed: {str(e)}")
            
            # Basic permissions check
            try:
                self.stepfunctions_client.list_state_machines(maxResults=1)
                validation_results['permissions_valid'] = True
                logger.info("✓ Basic permissions validated")
            except Exception as e:
                logger.error(f"✗ Permissions validation failed: {str(e)}")
            
            # Overall assessment
            all_passed = all(validation_results.values())
            
            return {
                'environment_ready': all_passed,
                'validation_results': validation_results,
                'overall_status': 'READY' if all_passed else 'NOT_READY'
            }
            
        except Exception as e:
            return {
                'environment_ready': False,
                'error': str(e)
            }

    def _deploy_enhanced_lambdas(self) -> Dict[str, Any]:
        """Deploy enhanced Lambda functions"""
        
        try:
            # Updated import path for sdcp_code structure
            from lambda_deployer import CompleteLambdaDeployer
            
            deployer = CompleteLambdaDeployer(
                region=self.region, 
                environment=self.environment
            )
            result = deployer.deploy_all_lambda_functions()
            
            return result
            
        except ImportError:
            logger.error("Enhanced Lambda deployer not available")
            return {
                'status': 'failed',
                'error': 'Enhanced Lambda deployer module not found'
            }
        except Exception as e:
            logger.error(f"Enhanced Lambda deployment failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _deploy_enhanced_step_functions(self) -> Dict[str, Any]:
        """Deploy enhanced Step Functions"""
        
        try:
            # Updated import path for sdcp_code structure
            infrastructure_path = os.path.join(sdcp_code_dir, 'infrastructure')
            if infrastructure_path not in sys.path:
                sys.path.append(infrastructure_path)
            
            from step_functions_definitions import get_enhanced_step_functions_with_integration
            
            roles = {"datascientist_role": self.config['datascientist_role']}
            
            result = get_enhanced_step_functions_with_integration(
                roles=roles,
                account_id=self.account_id,
                region=self.region,
                data_bucket=self.config['data_bucket'],
                model_bucket=self.config['model_bucket'],
                # environment=self.environment  # Pass environment for naming
            )
            
            logger.info("✓ Enhanced Step Functions deployed successfully")
            
            return {
                'status': 'success',
                'training_pipeline_arn': result.get('training_pipeline'),
                'enhanced_prediction_pipeline_arn': result.get('enhanced_prediction_pipeline'),
                'deployment_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Enhanced Step Functions deployment failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _build_and_push_containers(self) -> bool:
        """Build and push all container images"""
        
        try:
            # Create ECR repositories if needed
            for repo_name in self.config['containers']:
                try:
                    self.ecr_client.create_repository(repositoryName=repo_name)
                    logger.info(f"  ✓ Created ECR repository: {repo_name}")
                except self.ecr_client.exceptions.RepositoryAlreadyExistsException:
                    logger.info(f"  ✓ ECR repository already exists: {repo_name}")
            
            # Build containers using CodeBuild or local Docker
            try:
                # Try CodeBuild first - Updated path for sdcp_code structure
                scripts_dir = os.path.join(sdcp_code_dir, 'scripts')
                codebuild_script = os.path.join(scripts_dir, 'build_via_codebuild.py')
                
                result = subprocess.run([
                    'python', codebuild_script,
                    '--region', self.region,
                    '--environment', self.environment
                ], capture_output=True, text=True, cwd=project_root)
                
                if result.returncode == 0:
                    logger.info("  ✓ Containers built via CodeBuild")
                    return True
                else:
                    logger.warning("   CodeBuild failed, trying local Docker build")
                    return self._build_containers_locally()
                    
            except Exception as e:
                logger.warning(f"   CodeBuild not available: {str(e)}")
                return self._build_containers_locally()
                
        except Exception as e:
            logger.error(f"Container build failed: {str(e)}")
            return False
    
    def _build_containers_locally(self) -> bool:
        """Build containers locally using Docker"""
        
        try:
            # Get ECR login token
            token_response = self.ecr_client.get_authorization_token()
            token = token_response['authorizationData'][0]['authorizationToken']
            endpoint = token_response['authorizationData'][0]['proxyEndpoint']
            
            # Docker login
            import base64
            username, password = base64.b64decode(token).decode().split(':')
            
            login_result = subprocess.run([
                'docker', 'login', '--username', username, '--password-stdin', endpoint
            ], input=password, text=True, capture_output=True)
            
            if login_result.returncode != 0:
                logger.error("  ✗ Docker ECR login failed")
                return False
            
            # Updated container directories for sdcp_code structure
            container_dirs = {
                'energy-preprocessing': os.path.join(sdcp_code_dir, 'containers', 'preprocessing'),
                'energy-training': os.path.join(sdcp_code_dir, 'containers', 'training'),
            }
            
            for repo_name, container_dir in container_dirs.items():
                if not os.path.exists(container_dir):
                    logger.warning(f"   Container directory not found: {container_dir}")
                    continue
                
                image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repo_name}:latest"
                env_image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repo_name}:{self.environment}-latest"
                
                # Build
                build_result = subprocess.run([
                    'docker', 'build', '-t', image_uri, '-t', env_image_uri, container_dir
                ], capture_output=True, text=True)
                
                if build_result.returncode != 0:
                    logger.error(f"  ✗ Failed to build {repo_name}: {build_result.stderr}")
                    return False
                
                # Push both tags
                for push_uri in [image_uri, env_image_uri]:
                    push_result = subprocess.run([
                        'docker', 'push', push_uri
                    ], capture_output=True, text=True)
                    
                    if push_result.returncode != 0:
                        logger.error(f"  ✗ Failed to push {push_uri}: {push_result.stderr}")
                        return False
                
                logger.info(f"  ✓ Built and pushed: {repo_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Local container build failed: {str(e)}")
            return False
    
    def _setup_enhanced_eventbridge(self) -> Dict[str, Any]:
        """Setup enhanced EventBridge rules for both prediction and training pipelines"""
        
        try:
            # Updated import path for sdcp_code structure
            infrastructure_path = os.path.join(sdcp_code_dir, 'infrastructure')
            if infrastructure_path not in sys.path:
                sys.path.append(infrastructure_path)
            
            from step_functions_definitions import create_enhanced_eventbridge_rules
            
            # Get state machine ARNs
            state_machines = self.stepfunctions_client.list_state_machines()
            
            enhanced_prediction_arn = None
            training_pipeline_arn = None
            
            for sm in state_machines['stateMachines']:
                if sm['name'] == self.config['enhanced_prediction_state_machine']:
                    enhanced_prediction_arn = sm['stateMachineArn']
                elif sm['name'] == self.config['training_state_machine']:
                    training_pipeline_arn = sm['stateMachineArn']
            
            if not enhanced_prediction_arn:
                raise Exception(f"Enhanced prediction state machine not found: {self.config['enhanced_prediction_state_machine']}")
                
            if not training_pipeline_arn:
                raise Exception(f"Training pipeline state machine not found: {self.config['training_state_machine']}")
            
            # Create EventBridge rules for both pipelines
            state_machine_arns = {
                'enhanced_prediction_pipeline': enhanced_prediction_arn,
                'training_pipeline': training_pipeline_arn
            }
            
            rules_result = create_enhanced_eventbridge_rules(
                self.account_id,
                self.region,
                state_machine_arns,
                environment=self.environment  # Pass environment for naming
            )
            
            logger.info("✓ Enhanced EventBridge rules created successfully")
            
            return {
                'status': 'success',
                'rules_created': rules_result,
                'enhanced_prediction_arn': enhanced_prediction_arn,
                'training_pipeline_arn': training_pipeline_arn
            }
            
        except Exception as e:
            logger.error(f"Enhanced EventBridge setup failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _validate_complete_deployment(self) -> Dict[str, Any]:
        """Validate complete deployment including training schedules"""
        
        try:
            validation_results = {}
            
            # Environment-aware Lambda function names
            lambda_functions = [
                f'energy-forecasting-{self.environment}-profile-validator',
                f'energy-forecasting-{self.environment}-profile-endpoint-creator',
                f'energy-forecasting-{self.environment}-profile-predictor',
                f'energy-forecasting-{self.environment}-profile-cleanup'
            ]
            
            lambda_status = {}
            for func_name in lambda_functions:
                try:
                    response = self.lambda_client.get_function(FunctionName=func_name)
                    lambda_status[func_name] = {
                        'exists': True,
                        'state': response['Configuration']['State']
                    }
                except self.lambda_client.exceptions.ResourceNotFoundException:
                    lambda_status[func_name] = {
                        'exists': False,
                        'error': 'Function not found'
                    }
                except Exception as e:
                    lambda_status[func_name] = {
                        'exists': False,
                        'error': str(e)
                    }
            
            validation_results['lambda_functions'] = lambda_status
            
            # Environment-aware Step Function names
            state_machines = [
                self.config['enhanced_prediction_state_machine'],
                self.config['training_state_machine']
            ]
            
            stepfunctions_status = {}
            for sm_name in state_machines:
                try:
                    response = self.stepfunctions_client.describe_state_machine(
                        stateMachineArn=f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{sm_name}"
                    )
                    stepfunctions_status[sm_name] = {
                        'exists': True,
                        'status': response['status']
                    }
                except self.stepfunctions_client.exceptions.StateMachineDoesNotExist:
                    stepfunctions_status[sm_name] = {
                        'exists': False,
                        'error': 'State machine not found'
                    }
                except Exception as e:
                    stepfunctions_status[sm_name] = {
                        'exists': False,
                        'error': str(e)
                    }
            
            validation_results['step_functions'] = stepfunctions_status
            
            # Environment-aware EventBridge rule names
            try:
                rules = self.events_client.list_rules()
                energy_rules = [rule for rule in rules['Rules'] 
                              if f'energy-forecasting-{self.environment}' in rule['Name']]
                
                expected_rules = [
                    f'energy-forecasting-{self.environment}-enhanced-daily-predictions',
                    f'energy-forecasting-{self.environment}-monthly-training-pipeline'
                ]
                
                found_rules = [rule['Name'] for rule in energy_rules]
                
                validation_results['eventbridge_rules'] = {
                    'rules_count': len([rule for rule in found_rules if rule in expected_rules]),
                    'rules_found': [rule for rule in found_rules if rule in expected_rules],
                    'rules_missing': [rule for rule in expected_rules if rule not in found_rules],
                    'all_expected_rules': expected_rules
                }
                
            except Exception as e:
                validation_results['eventbridge_rules'] = {
                    'error': str(e)
                }
            
            # Overall assessment
            lambda_success = sum(1 for status in lambda_status.values() if status.get('exists'))
            stepfunctions_success = sum(1 for status in stepfunctions_status.values() if status.get('exists'))
            eventbridge_success = validation_results.get('eventbridge_rules', {}).get('rules_count', 0)
            
            overall_success = (
                lambda_success == len(lambda_functions) and
                stepfunctions_success == len(state_machines) and
                eventbridge_success >= 2
            )
            
            return {
                'status': 'success' if overall_success else 'partial',
                'overall_success': overall_success,
                'validation_results': validation_results,
                'summary': {
                    'lambda_functions_deployed': f"{lambda_success}/{len(lambda_functions)}",
                    'step_functions_deployed': f"{stepfunctions_success}/{len(state_machines)}",
                    'eventbridge_rules_deployed': f"{eventbridge_success}/2",
                    'deployment_complete': overall_success
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for the enhanced pipeline"""
        
        try:
            # Run basic integration tests
            test_results = {}
            
            # Test 1: Lambda function connectivity
            logger.info("Testing Lambda function connectivity...")
            lambda_test = self._test_lambda_connectivity()
            test_results['lambda_connectivity'] = lambda_test
            
            # Test 2: Step Functions definition validation
            logger.info("Validating Step Functions definitions...")
            stepfunctions_test = self._test_stepfunctions_definitions()
            test_results['stepfunctions_definitions'] = stepfunctions_test
            
            # Test 3: Profile validation test
            logger.info("Testing profile validation...")
            profile_test = self._test_profile_validation()
            test_results['profile_validation'] = profile_test
            
            # Overall assessment
            successful_tests = sum(1 for test in test_results.values() if test.get('status') == 'success')
            total_tests = len(test_results)
            
            return {
                'status': 'success' if successful_tests == total_tests else 'partial',
                'test_results': test_results,
                'summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_lambda_connectivity(self) -> Dict[str, Any]:
        """Test Lambda function connectivity"""
        
        try:
            # Test profile validator with environment-aware naming
            test_event = {
                "operation": "validate_and_filter_profiles",
                "profiles": ["RNN"],
                "data_bucket": self.config['data_bucket']
            }
            
            function_name = f'energy-forecasting-{self.environment}-profile-validator'
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(test_event)
            )
            
            if response['StatusCode'] == 200:
                return {
                    'status': 'success',
                    'message': 'Lambda functions are responsive'
                }
            else:
                return {
                    'status': 'failed',
                    'error': f'Non-200 status code: {response["StatusCode"]}'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_stepfunctions_definitions(self) -> Dict[str, Any]:
        """Test Step Functions definitions"""
        
        try:
            sm_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.config['enhanced_prediction_state_machine']}"
            
            response = self.stepfunctions_client.describe_state_machine(stateMachineArn=sm_arn)
            
            # Basic validation of definition
            definition = json.loads(response['definition'])
            
            required_states = ['ValidateInput', 'CreateEndpointsParallel', 'RunPredictionsParallel', 'CleanupEndpointsParallel']
            missing_states = [state for state in required_states if state not in definition.get('States', {})]
            
            if missing_states:
                return {
                    'status': 'failed',
                    'error': f'Missing required states: {missing_states}'
                }
            else:
                return {
                    'status': 'success',
                    'message': 'Step Functions definition is valid'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_profile_validation(self) -> Dict[str, Any]:
        """Test profile validation functionality"""
        
        try:
            # This would run a quick test of the profile validation
            test_event = {
                "operation": "validate_and_filter_profiles",
                "profiles": ["RNN", "RN"],
                "data_bucket": self.config['data_bucket']
            }
            
            function_name = f'energy-forecasting-{self.environment}-profile-validator'
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(test_event)
            )
            
            if response['StatusCode'] == 200:
                result = json.loads(response['Payload'].read())
                
                if result.get('statusCode') == 200:
                    body = result.get('body', {})
                    valid_count = body.get('valid_profiles_count', 0)
                    
                    return {
                        'status': 'success',
                        'valid_profiles_found': valid_count,
                        'message': f'Profile validation working - found {valid_count} valid profiles'
                    }
                else:
                    return {
                        'status': 'failed',
                        'error': result.get('body', {}).get('error', 'Unknown error')
                    }
            else:
                return {
                    'status': 'failed',
                    'error': f'Lambda invocation failed: {response["StatusCode"]}'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _generate_deployment_summary(self, deployment_results: Dict, deployment_time: float) -> Dict[str, Any]:
        """Generate comprehensive deployment summary"""
       
        try:
            # Calculate success metrics
            successful_components = 0
            total_components = len(deployment_results)
           
            # Define success criteria for each component type
            def is_component_successful(component_name: str, result: Any) -> bool:
                """Determine if a component is successful based on its specific criteria"""
               
                if not isinstance(result, dict):
                    # Handle non-dict results (like True/False for containers)
                    return bool(result)
               
                # Component-specific success criteria
                if component_name == 'environment_validation':
                    return result.get('environment_ready', False) and result.get('overall_status') == 'GOOD'
               
                elif component_name == 'enhanced_lambdas':
                    # Check if all lambda functions were deployed successfully
                    # Assume success if we have function ARNs and no deployment errors
                    if 'error' in result:
                        return False
                    # Check if we have function deployments
                    function_count = len([k for k, v in result.items() if isinstance(v, dict) and 'function_arn' in v])
                    return function_count > 0
               
                elif component_name == 'enhanced_containers':
                    # Containers return True/False
                    return bool(result)
               
                elif component_name == 'enhanced_eventbridge':
                    # Check both prediction and training rules were created successfully
                    if result.get('status') != 'success':
                        return False
                    rules_created = result.get('rules_created', {})
                    # Both rules should be present and not failed
                    prediction_rule_ok = 'enhanced_prediction_rule' in rules_created and not str(rules_created['enhanced_prediction_rule']).startswith('FAILED')
                    training_rule_ok = 'monthly_training_rule' in rules_created and not str(rules_created['monthly_training_rule']).startswith('FAILED')
                    return prediction_rule_ok and training_rule_ok
               
                elif component_name in ['enhanced_step_functions', 'deployment_validation', 'integration_tests']:
                    # These components use 'status' field
                    return result.get('status') == 'success'
               
                else:
                    # Fallback: check for status field or assume success if no error
                    if 'status' in result:
                        return result.get('status') == 'success'
                    elif 'error' in result:
                        return False
                    else:
                        return True
           
            # Count successful components
            for component, result in deployment_results.items():
                is_successful = is_component_successful(component, result)
                if is_successful:
                    successful_components += 1
               
                # Log component status for debugging
                status = 'success' if is_successful else 'failed'
                logger.info(f"==={component}:{status}===")
           
            overall_success = (
                successful_components == total_components and
                deployment_results.get('integration_tests', {}).get('status') == 'success'
            )
           
            # Generate recommendations
            recommendations = self._generate_deployment_recommendations(deployment_results)
           
            summary = {
                'deployment_summary': {
                    'overall_success': overall_success,
                    'deployment_time_minutes': deployment_time / 60,
                    'successful_components': successful_components,
                    'total_components': total_components,
                    'success_rate': (successful_components / total_components * 100) if total_components > 0 else 0,
                    'deployment_timestamp': datetime.now().isoformat()
                },
                'component_results': deployment_results,
                'recommendations': recommendations,
                'next_steps': self._generate_next_steps(deployment_results, overall_success),
                'deployment_artifacts': {
                    'enhanced_lambda_functions': len([k for k, v in deployment_results.get('enhanced_lambdas', {}).items()
                                                    if isinstance(v, dict) and 'function_arn' in v]),
                    'enhanced_step_functions': 1 if deployment_results.get('enhanced_step_functions', {}).get('status') == 'success' else 0,
                    'eventbridge_rules': 1 if deployment_results.get('enhanced_eventbridge', {}).get('status') == 'success' else 0,
                    'integration_tests': deployment_results.get('integration_tests', {}).get('summary', {}).get('total_tests', 0)
                }
            }
           
            return summary
           
        except Exception as e:
            logger.error(f"Error in deployment summary generation: {str(e)}")
            return {
                'summary_generation_error': str(e),
                'partial_deployment_results': deployment_results
            }

    def _generate_deployment_recommendations(self, deployment_results: Dict) -> list:
        """Generate deployment recommendations"""
        
        recommendations = []
        
        # Check Lambda deployment
        lambda_result = deployment_results.get('enhanced_lambdas', {})
        if lambda_result.get('deployment_success_rate', 0) < 100:
            recommendations.append("Fix Lambda function deployment issues before enabling automated predictions")
        
        # Check Step Functions
        stepfunctions_result = deployment_results.get('enhanced_step_functions', {})
        if stepfunctions_result.get('status') != 'success':
            recommendations.append("Resolve Step Functions deployment issues")
        
        # Check integration tests
        integration_result = deployment_results.get('integration_tests', {})
        if integration_result.get('status') != 'success':
            recommendations.append("Address integration test failures before production use")
        
        # Success recommendations
        if not recommendations:
            recommendations.extend([
                "Enhanced prediction pipeline is ready for testing",
                "Run comprehensive tests with test_enhanced_prediction_pipeline.py",
                "Consider enabling daily enhanced predictions",
                "Monitor performance and cost optimization metrics"
            ])
        
        return recommendations

    def _generate_next_steps(self, deployment_results: Dict, overall_success: bool) -> list:
        """Generate next steps based on deployment results"""
        
        if overall_success:
            return [
                "Run comprehensive testing: python scripts/test_enhanced_prediction_pipeline.py",
                "Test with subset of profiles: python scripts/test_enhanced_prediction_pipeline.py --profiles RNN RN",
                "Enable daily predictions when ready",
                "Monitor CloudWatch metrics and costs"
            ]
        else:
            return [
                "Review failed components in deployment results",
                "Fix deployment issues and re-run deployment",
                "Run environment validation: python deployment/validate_environment.py",
                "Check CloudWatch logs for detailed error information"
            ]

    def _identify_failure_point(self, deployment_results: Dict) -> str:
        """Identify where deployment failed"""
        
        for component, result in deployment_results.items():
            if isinstance(result, dict) and result.get('status') == 'failed':
                return component
        
        return 'unknown'

def main():
    """Main function for enhanced MLOps deployment"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced MLOps Pipeline Deployer')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--environment', default='dev', help='Environment (dev/staging/prod)')
    parser.add_argument('--validate-only', action='store_true', help='Only validate environment')
    parser.add_argument('--lambda-only', action='store_true', help='Only deploy Lambda functions')
    parser.add_argument('--stepfunctions-only', action='store_true', help='Only deploy Step Functions')
    parser.add_argument('--test-only', action='store_true', help='Only run integration tests')
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = EnhancedMLOpsDeployer(region=args.region, environment=args.environment)
    
    try:
        if args.validate_only:
            # Only validate environment
            logger.info("Running environment validation only...")
            validation_result = deployer._validate_environment()
            
            if validation_result.get('environment_ready', False):
                logger.info("✓ Environment validation PASSED")
                sys.exit(0)
            else:
                logger.error("✗ Environment validation FAILED")
                logger.error(f"Issues: {validation_result}")
                sys.exit(1)
                
        elif args.lambda_only:
            # Only deploy Lambda functions
            logger.info("Deploying Lambda functions only...")
            lambda_result = deployer._deploy_enhanced_lambdas()
            
            if lambda_result.get('deployment_success_rate', 0) >= 75:
                logger.info("✓ Lambda deployment completed successfully")
                sys.exit(0)
            else:
                logger.error("✗ Lambda deployment failed")
                sys.exit(1)
                
        elif args.stepfunctions_only:
            # Only deploy Step Functions
            logger.info("Deploying Step Functions only...")
            stepfunctions_result = deployer._deploy_enhanced_step_functions()
            
            if stepfunctions_result.get('status') == 'success':
                logger.info("✓ Step Functions deployment completed successfully")
                sys.exit(0)
            else:
                logger.error("✗ Step Functions deployment failed")
                sys.exit(1)
                
        elif args.test_only:
            # Only run integration tests
            logger.info("Running integration tests only...")
            test_result = deployer._run_integration_tests()
            
            if test_result.get('status') == 'success':
                logger.info("✓ Integration tests completed successfully")
                sys.exit(0)
            else:
                logger.error("✗ Integration tests failed")
                sys.exit(1)
                
        else:
            # Deploy complete enhanced pipeline
            logger.info("Deploying complete enhanced MLOps pipeline...")
            
            deployment_result = deployer.deploy_complete_enhanced_pipeline()
            
            if deployment_result.get('deployment_summary', {}).get('overall_success', False):
                logger.info("✓ ENHANCED MLOPS DEPLOYMENT COMPLETED SUCCESSFULLY")
                logger.info("✓ Enhanced prediction pipeline is ready for testing and production use")
                
                # Print next steps
                next_steps = deployment_result.get('next_steps', [])
                if next_steps:
                    logger.info("\nNext Steps:")
                    for i, step in enumerate(next_steps, 1):
                        logger.info(f"{i}. {step}")
                
                sys.exit(0)
            else:
                logger.error("✗ ENHANCED MLOPS DEPLOYMENT FAILED")
                
                # Print failure details
                failed_at = deployment_result.get('failed_at', 'unknown')
                error = deployment_result.get('error', 'Unknown error')
                logger.error(f"Failed at: {failed_at}")
                logger.error(f"Error: {error}")
                
                # Print recommendations
                recommendations = deployment_result.get('recommendations', [])
                if recommendations:
                    logger.info("\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        logger.info(f"{i}. {rec}")
                
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("\nDeployment interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Enhanced MLOps deployment failed with unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
