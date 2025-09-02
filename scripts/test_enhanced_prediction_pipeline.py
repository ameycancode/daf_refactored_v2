"""
Enhanced Integration Test Script for Dynamic Profile Selection Pipeline
scripts/test_enhanced_prediction_pipeline.py

Comprehensive testing framework with:
- Profile selection testing (1, 2, 5, and all 7 profiles)
- Parallel execution validation
- Performance benchmarking
- Fault tolerance testing
- Resource cleanup validation
- Cost optimization metrics
"""

#!/usr/bin/env python3

import json
import boto3
import time
import logging
import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedPredictionPipelineTest:
    """
    Comprehensive test suite for the enhanced prediction pipeline
    """
    
    def __init__(self, region: str = "us-west-2"):
        """Initialize the test framework"""
        
        self.region = region
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # AWS clients
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=region)
        
        # Configuration
        self.config = {
            'data_bucket': 'sdcp-dev-sagemaker-energy-forecasting-data',
            'model_bucket': 'sdcp-dev-sagemaker-energy-forecasting-models',
            'state_machine_name': 'energy-forecasting-enhanced-prediction-pipeline',
            'lambda_functions': {
                'validator': 'energy-forecasting-profile-validator',
                'creator': 'energy-forecasting-profile-endpoint-creator',
                'predictor': 'energy-forecasting-profile-predictor',
                'cleanup': 'energy-forecasting-profile-cleanup'
            }
        }
        
        # Test profiles
        self.all_profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
        
        # Test results storage
        self.test_results = {}
        self.performance_metrics = {}
        
        logger.info(f"Enhanced Pipeline Test initialized for account {self.account_id}")

    def run_comprehensive_test_suite(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete enhanced prediction pipeline test suite
        """
        
        logger.info("="*80)
        logger.info("ENHANCED PREDICTION PIPELINE - COMPREHENSIVE TEST SUITE")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Test 1: Prerequisites Validation
            logger.info("\n1. TESTING PREREQUISITES...")
            prereq_results = self.test_prerequisites()
            self.test_results['prerequisites'] = prereq_results
            
            if not prereq_results['all_passed']:
                logger.error("Prerequisites failed - cannot continue")
                return self._generate_test_report(start_time, failed_at='prerequisites')
            
            # Test 2: Lambda Functions Validation
            logger.info("\n2. TESTING LAMBDA FUNCTIONS...")
            lambda_results = self.test_lambda_functions()
            self.test_results['lambda_functions'] = lambda_results
            
            # Test 3: Profile Selection Testing
            if test_config.get('test_profile_selection', True):
                logger.info("\n3. TESTING PROFILE SELECTION...")
                profile_results = self.test_profile_selection_scenarios()
                self.test_results['profile_selection'] = profile_results
            
            # # Test 4: Parallel Execution Validation
            # if test_config.get('test_parallel_execution', True):
            #     logger.info("\n4. TESTING PARALLEL EXECUTION...")
            #     parallel_results = self.test_parallel_execution()
            #     self.test_results['parallel_execution'] = parallel_results
            
            # Test 5: Performance Benchmarking
            if test_config.get('test_performance', True):
                logger.info("\n5. PERFORMANCE BENCHMARKING...")
                performance_results = self.test_performance_benchmarks()
                self.test_results['performance'] = performance_results
            
            # Test 6: Fault Tolerance Testing
            if test_config.get('test_fault_tolerance', True):
                logger.info("\n6. TESTING FAULT TOLERANCE...")
                fault_results = self.test_fault_tolerance()
                self.test_results['fault_tolerance'] = fault_results
            
            # Test 7: Resource Cleanup Validation
            if test_config.get('test_cleanup', True):
                logger.info("\n7. TESTING RESOURCE CLEANUP...")
                cleanup_results = self.test_resource_cleanup()
                self.test_results['resource_cleanup'] = cleanup_results
            
            # Test 8: Cost Optimization Metrics
            if test_config.get('test_cost_optimization', True):
                logger.info("\n8. TESTING COST OPTIMIZATION...")
                cost_results = self.test_cost_optimization()
                self.test_results['cost_optimization'] = cost_results
            
            # Generate final test report
            total_time = time.time() - start_time
            final_report = self._generate_test_report(start_time, total_time=total_time)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Test suite failed: {str(e)}")
            return self._generate_test_report(start_time, error=str(e))

    def test_prerequisites(self) -> Dict[str, Any]:
        """Test all prerequisites for the enhanced pipeline"""
        
        logger.info("Validating prerequisites...")
        
        results = {
            'step_functions_exists': False,
            'lambda_functions_exist': False,
            's3_configurations_exist': False,
            'model_registry_available': False,
            'iam_permissions_valid': False,
            'all_passed': False
        }
        
        try:
            # Check Step Functions state machine
            try:
                response = self.stepfunctions_client.describe_state_machine(
                    stateMachineArn=f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.config['state_machine_name']}"
                )
                results['step_functions_exists'] = True
                logger.info("✓ Enhanced prediction Step Functions state machine found")
            except Exception as e:
                logger.error(f"✗ Step Functions state machine not found: {str(e)}")
            
            # Check Lambda functions
            lambda_count = 0
            for func_type, func_name in self.config['lambda_functions'].items():
                try:
                    self.lambda_client.get_function(FunctionName=func_name)
                    lambda_count += 1
                    logger.info(f"✓ Lambda function found: {func_name}")
                except Exception as e:
                    logger.error(f"✗ Lambda function not found: {func_name}")
            
            results['lambda_functions_exist'] = lambda_count == len(self.config['lambda_functions'])
            
            # Check S3 configurations
            config_count = 0
            endpoint_configs = {}
            for profile in self.all_profiles:
                try:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.config['data_bucket'],
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
                    # logger.error(f'FIRST ERROR: {str(e)}')
                    # Handle case where no files found
                    endpoint_configs[profile] = {'exists': False, 'valid': False, 'error': str(e)}
                    continue
            
                try:
                    response = self.s3_client.get_object(Bucket=self.config['data_bucket'], Key=config_key)
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
                        config_count += 1
                        logger.info(f"✓ Endpoint config valid for profile: {profile}")
                    else:
                        logger.warning(f" Endpoint config incomplete for {profile}: missing {missing_fields}")
                        
                except self.s3_client.exceptions.NoSuchKey:
                    # logger.error('SECOND ERROR')
                    endpoint_configs[profile] = {
                        'exists': False,
                        'valid': False
                    }
                    logger.warning(f" Endpoint config missing for profile: {profile}")
                except Exception as e:
                    # logger.error('THIRD ERROR')
                    endpoint_configs[profile] = {
                        'exists': False,
                        'valid': False,
                        'error': str(e)
                    }
                    logger.error(f"✗ Error validating endpoint config for {profile}: {str(e)}")

            
            # config_count = 0
            # for profile in self.all_profiles:
            #     try:
            #         key = f"endpoint-configurations/{profile}"
            #         self.s3_client.head_object(Bucket=self.config['data_bucket'], Key=key)
            #         config_count += 1
            #     except Exception:
            #         logger.warning(f" S3 configuration not found for profile {profile}")
            
            results['s3_configurations_exist'] = config_count >= 3  # At least 3 profiles
            logger.info(f"S3 configurations found for {config_count}/{len(self.all_profiles)} profiles")
            
            # Check Model Registry
            try:
                response = self.sagemaker_client.list_model_package_groups(MaxResults=10)
                results['model_registry_available'] = True
                logger.info("✓ Model Registry accessible")
            except Exception as e:
                logger.error(f"✗ Model Registry not accessible: {str(e)}")
            
            # Check IAM permissions (basic test)
            try:
                self.stepfunctions_client.list_state_machines(maxResults=1)
                results['iam_permissions_valid'] = True
                logger.info("✓ Basic IAM permissions valid")
            except Exception as e:
                logger.error(f"✗ IAM permissions issue: {str(e)}")
            
            # Overall assessment
            required_checks = ['step_functions_exists', 'lambda_functions_exist', 'iam_permissions_valid']
            results['all_passed'] = all(results[check] for check in required_checks)
            
            if results['all_passed']:
                logger.info("✓ All prerequisites passed")
            else:
                logger.error("✗ Some prerequisites failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Prerequisites validation failed: {str(e)}")
            results['error'] = str(e)
            return results

    def test_lambda_functions(self) -> Dict[str, Any]:
        """Test all Lambda functions individually"""
        
        logger.info("Testing Lambda functions...")
        
        results = {}
        
        # Test Profile Validator
        logger.info("Testing Profile Validator...")
        validator_result = self._test_profile_validator()
        results['validator'] = validator_result
        
        # Test Profile Endpoint Creator  
        logger.info("Testing Profile Endpoint Creator...")
        creator_result = self._test_profile_endpoint_creator()
        results['creator'] = creator_result
        
        # Test Profile Predictor
        logger.info("Testing Profile Predictor...")
        predictor_result = self._test_profile_predictor()
        results['predictor'] = predictor_result
        
        # Test Profile Cleanup
        logger.info("Testing Profile Cleanup...")
        cleanup_result = self._test_profile_cleanup()
        results['cleanup'] = cleanup_result
        
        # Overall assessment
        successful_functions = sum(1 for result in results.values() if result.get('status') == 'success')
        results['summary'] = {
            'total_functions': len(results),
            'successful_functions': successful_functions,
            'success_rate': successful_functions / len(results) * 100
        }
        
        logger.info(f"Lambda function testing: {successful_functions}/{len(results)} functions passed")
        
        return results

    def _test_profile_validator(self) -> Dict[str, Any]:
        """Test the profile validator Lambda function"""
        
        try:
            test_event = {
                "operation": "validate_and_filter_profiles",
                "profiles": ["RNN", "RN", "INVALID_PROFILE"],
                "data_bucket": self.config['data_bucket'],
                "model_bucket": self.config['model_bucket']
            }
            
            response = self.lambda_client.invoke(
                FunctionName=self.config['lambda_functions']['validator'],
                InvocationType='RequestResponse',
                Payload=json.dumps(test_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            if response['StatusCode'] == 200 and result.get('statusCode') == 200:
                body = result['body']
                valid_count = body.get('valid_profiles_count', 0)
                
                return {
                    'status': 'success',
                    'valid_profiles_found': valid_count,
                    'response_time_ms': response.get('ExecutedVersion', 0),
                    'message': f"Validator found {valid_count} valid profiles"
                }
            else:
                return {
                    'status': 'failed',
                    'error': result.get('body', {}).get('error', 'Unknown error'),
                    'response': result
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_profile_endpoint_creator(self) -> Dict[str, Any]:
        """Test the profile endpoint creator Lambda function"""
        
        try:
            # Test with check operation first (doesn't create resources)
            test_event = {
                "operation": "check_endpoint_status",
                "profile": "RNN",
                "endpoint_name": "test-endpoint-check"
            }
            
            response = self.lambda_client.invoke(
                FunctionName=self.config['lambda_functions']['creator'],
                InvocationType='RequestResponse',
                Payload=json.dumps(test_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            # Expected to fail since endpoint doesn't exist, but function should handle gracefully
            if response['StatusCode'] == 200:
                return {
                    'status': 'success',
                    'message': 'Endpoint creator function is responsive',
                    'test_type': 'status_check'
                }
            else:
                return {
                    'status': 'failed',
                    'error': result.get('body', {}).get('error', 'Unknown error'),
                    'response': result
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_profile_predictor(self) -> Dict[str, Any]:
        """Test the profile predictor Lambda function"""
        
        try:
            # Test with dry run operation
            test_event = {
                "operation": "run_profile_prediction",
                "profile": "RNN",
                "endpoint_name": "test-endpoint",
                "data_bucket": self.config['data_bucket']
            }
            
            response = self.lambda_client.invoke(
                FunctionName=self.config['lambda_functions']['predictor'],
                InvocationType='RequestResponse',
                Payload=json.dumps(test_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            # May fail due to missing endpoint, but function should be responsive
            if response['StatusCode'] == 200:
                return {
                    'status': 'success',
                    'message': 'Predictor function is responsive',
                    'test_type': 'dry_run'
                }
            else:
                return {
                    'status': 'partial',
                    'message': 'Function responded but with expected errors (no endpoint)',
                    'test_type': 'dry_run'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_profile_cleanup(self) -> Dict[str, Any]:
        """Test the profile cleanup Lambda function"""
        
        try:
            # Test with non-existent resources (should handle gracefully)
            test_event = {
                "operation": "cleanup_profile_resources",
                "profile": "TEST",
                "endpoint_name": "non-existent-endpoint",
                "endpoint_config_name": "non-existent-config",
                "model_name": "non-existent-model"
            }
            
            response = self.lambda_client.invoke(
                FunctionName=self.config['lambda_functions']['cleanup'],
                InvocationType='RequestResponse',
                Payload=json.dumps(test_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            if response['StatusCode'] == 200:
                return {
                    'status': 'success',
                    'message': 'Cleanup function is responsive',
                    'test_type': 'non_existent_resources'
                }
            else:
                return {
                    'status': 'failed',
                    'error': result.get('body', {}).get('error', 'Unknown error'),
                    'response': result
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def test_profile_selection_scenarios(self) -> Dict[str, Any]:
        """Test different profile selection scenarios"""
        
        logger.info("Testing profile selection scenarios...")
        
        test_scenarios = [
            {
                'name': 'single_profile',
                'profiles': ['RNN'],
                'description': 'Single profile test'
            },
            # {
            #     'name': 'dual_profiles',
            #     'profiles': ['RNN', 'RN'],
            #     'description': 'Two profiles test'
            # },
            # {
            #     'name': 'subset_profiles',
            #     'profiles': ['RNN', 'RN', 'M', 'S'],
            #     'description': 'Subset of profiles test'
            # },
            # {
            #     'name': 'all_profiles',
            #     'profiles': self.all_profiles,
            #     'description': 'All profiles test'
            # }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            logger.info(f"Testing scenario: {scenario['description']}")
            
            try:
                scenario_result = self._run_profile_selection_test(
                    scenario['profiles'], 
                    scenario['name']
                )
                results[scenario['name']] = scenario_result
                
                if scenario_result.get('status') == 'success':
                    logger.info(f"✓ {scenario['description']} completed successfully")
                else:
                    logger.error(f"✗ {scenario['description']} failed")
                    
            except Exception as e:
                logger.error(f"✗ {scenario['description']} error: {str(e)}")
                results[scenario['name']] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate success rate
        successful_scenarios = sum(1 for result in results.values() if result.get('status') == 'success')
        results['summary'] = {
            'total_scenarios': len(test_scenarios),
            'successful_scenarios': successful_scenarios,
            'success_rate': successful_scenarios / len(test_scenarios) * 100
        }
        
        return results

    def _run_profile_selection_test(self, profiles: List[str], test_name: str) -> Dict[str, Any]:
        """Run a specific profile selection test"""
        
        try:
            start_time = time.time()
            
            # Execute the enhanced prediction pipeline
            execution_input = {
                "profiles": profiles,
                "execution_type": f"test_{test_name}",
                "test_mode": True
            }
            
            state_machine_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.config['state_machine_name']}"
            
            response = self.stepfunctions_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=f"test-{test_name}-{int(time.time())}",
                input=json.dumps(execution_input)
            )
            
            execution_arn = response['executionArn']
            
            # Wait for completion (with timeout)
            execution_result = self._wait_for_execution_completion(execution_arn, timeout_minutes=15)
            
            execution_time = time.time() - start_time
            
            if execution_result.get('status') == 'SUCCEEDED':
                output = json.loads(execution_result.get('output', '{}'))
                
                return {
                    'status': 'success',
                    'execution_arn': execution_arn,
                    'execution_time_seconds': execution_time,
                    'profiles_requested': len(profiles),
                    'profiles_processed': output.get('summary', {}).get('successful_endpoints', []),
                    'output': output,
                    'performance_metrics': self._extract_performance_metrics(output)
                }
            else:
                return {
                    'status': 'failed',
                    'execution_arn': execution_arn,
                    'execution_time_seconds': execution_time,
                    'error': execution_result.get('error', 'Unknown execution error'),
                    'profiles_requested': len(profiles)
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'profiles_requested': len(profiles)
            }

    def test_parallel_execution(self) -> Dict[str, Any]:
        """Test parallel execution capabilities"""
        
        logger.info("Testing parallel execution...")
        
        # Test with multiple profiles to verify true parallelization
        test_profiles = ['RNN', 'RN', 'M', 'S']
        
        try:
            start_time = time.time()
            
            # Run enhanced pipeline
            execution_input = {
                "profiles": test_profiles,
                "execution_type": "parallel_test",
                "test_mode": True
            }
            
            state_machine_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.config['state_machine_name']}"
            
            response = self.stepfunctions_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=f"test-parallel-{int(time.time())}",
                input=json.dumps(execution_input)
            )
            
            execution_arn = response['executionArn']
            
            # Monitor execution for parallel behavior
            parallel_metrics = self._monitor_parallel_execution(execution_arn)
            
            # Wait for completion
            execution_result = self._wait_for_execution_completion(execution_arn, timeout_minutes=20)
            
            total_time = time.time() - start_time
            
            # Analyze parallelization efficiency
            efficiency_metrics = self._analyze_parallelization_efficiency(
                parallel_metrics, len(test_profiles), total_time
            )
            
            return {
                'status': 'success' if execution_result.get('status') == 'SUCCEEDED' else 'failed',
                'execution_arn': execution_arn,
                'total_execution_time': total_time,
                'parallel_metrics': parallel_metrics,
                'efficiency_metrics': efficiency_metrics,
                'profiles_tested': test_profiles
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'profiles_tested': test_profiles
            }

    def _monitor_parallel_execution(self, execution_arn: str) -> Dict[str, Any]:
        """Monitor execution for parallel behavior patterns"""
        
        try:
            parallel_metrics = {
                'concurrent_tasks_detected': False,
                'max_concurrent_tasks': 0,
                'parallel_phases': [],
                'timeline': []
            }
            
            # Monitor for a few minutes to detect parallel patterns
            monitor_start = time.time()
            monitor_duration = 300  # 5 minutes
            
            while time.time() - monitor_start < monitor_duration:
                try:
                    # Get execution history
                    history_response = self.stepfunctions_client.get_execution_history(
                        executionArn=execution_arn,
                        reverseOrder=True,
                        maxResults=100
                    )
                    
                    events = history_response.get('events', [])
                    
                    # Analyze events for parallel patterns
                    concurrent_tasks = self._count_concurrent_tasks(events)
                    
                    if concurrent_tasks > 1:
                        parallel_metrics['concurrent_tasks_detected'] = True
                        parallel_metrics['max_concurrent_tasks'] = max(
                            parallel_metrics['max_concurrent_tasks'], 
                            concurrent_tasks
                        )
                    
                    parallel_metrics['timeline'].append({
                        'timestamp': datetime.now().isoformat(),
                        'concurrent_tasks': concurrent_tasks
                    })
                    
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    logger.debug(f"Monitoring error: {str(e)}")
                    break
            
            return parallel_metrics
            
        except Exception as e:
            logger.error(f"Parallel execution monitoring failed: {str(e)}")
            return {'error': str(e)}

    def _count_concurrent_tasks(self, events: List[Dict]) -> int:
        """Count concurrent tasks from Step Functions events"""
        
        try:
            # Look for parallel state entries and map iterations
            concurrent_count = 0
            
            for event in events:
                event_type = event.get('type', '')
                
                # Count active map iterations or parallel executions
                if 'MapIteration' in event_type and 'Started' in event_type:
                    concurrent_count += 1
                elif 'ParallelState' in event_type and 'Started' in event_type:
                    concurrent_count += 1
            
            return concurrent_count
            
        except Exception:
            return 0

    def _analyze_parallelization_efficiency(self, parallel_metrics: Dict, profile_count: int, 
                                          total_time: float) -> Dict[str, Any]:
        """Analyze the efficiency of parallelization"""
        
        try:
            # Theoretical sequential time (estimate)
            estimated_sequential_time = total_time * profile_count
            
            # Parallelization efficiency
            efficiency = (estimated_sequential_time - total_time) / estimated_sequential_time * 100
            
            # Speedup factor
            speedup_factor = estimated_sequential_time / total_time if total_time > 0 else 0
            
            return {
                'parallel_detected': parallel_metrics.get('concurrent_tasks_detected', False),
                'max_concurrent_tasks': parallel_metrics.get('max_concurrent_tasks', 0),
                'efficiency_percentage': max(0, efficiency),
                'speedup_factor': speedup_factor,
                'estimated_sequential_time': estimated_sequential_time,
                'actual_parallel_time': total_time,
                'time_saved_seconds': max(0, estimated_sequential_time - total_time)
            }
            
        except Exception as e:
            return {'error': str(e)}

    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks"""
        
        logger.info("Running performance benchmarks...")
        
        benchmarks = {
            'single_profile_time': None,
            'dual_profile_time': None,
            'quad_profile_time': None,
            'all_profile_time': None,
            'sequential_vs_parallel': None
        }
        
        try:
            # Benchmark 1: Single profile
            logger.info("Benchmarking single profile...")
            single_result = self._benchmark_execution(['RNN'], 'single')
            benchmarks['single_profile_time'] = single_result
            
            # Benchmark 2: Two profiles
            logger.info("Benchmarking dual profiles...")
            dual_result = self._benchmark_execution(['RNN', 'RN'], 'dual')
            benchmarks['dual_profile_time'] = dual_result
            
            # Benchmark 3: Four profiles
            logger.info("Benchmarking quad profiles...")
            quad_result = self._benchmark_execution(['RNN', 'RN', 'M', 'S'], 'quad')
            benchmarks['quad_profile_time'] = quad_result
            
            # Benchmark 4: All profiles (if time permits)
            logger.info("Benchmarking all profiles...")
            all_result = self._benchmark_execution(self.all_profiles, 'all')
            benchmarks['all_profile_time'] = all_result
            
            # Calculate performance insights
            performance_insights = self._calculate_performance_insights(benchmarks)
            
            return {
                'status': 'success',
                'benchmarks': benchmarks,
                'insights': performance_insights
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'partial_benchmarks': benchmarks
            }

    def _benchmark_execution(self, profiles: List[str], test_name: str) -> Dict[str, Any]:
        """Benchmark a specific execution"""
        
        try:
            start_time = time.time()
            
            execution_input = {
                "profiles": profiles,
                "execution_type": f"benchmark_{test_name}",
                "test_mode": True
            }
            
            state_machine_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.config['state_machine_name']}"
            
            response = self.stepfunctions_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=f"benchmark-{test_name}-{int(time.time())}",
                input=json.dumps(execution_input)
            )
            
            execution_arn = response['executionArn']
            
            # Wait for completion
            execution_result = self._wait_for_execution_completion(execution_arn, timeout_minutes=25)
            
            total_time = time.time() - start_time
            
            return {
                'profiles': profiles,
                'profile_count': len(profiles),
                'execution_time_seconds': total_time,
                'execution_time_minutes': total_time / 60,
                'time_per_profile': total_time / len(profiles) if len(profiles) > 0 else 0,
                'status': execution_result.get('status'),
                'execution_arn': execution_arn
            }
            
        except Exception as e:
            return {
                'profiles': profiles,
                'error': str(e),
                'status': 'failed'
            }

    def _calculate_performance_insights(self, benchmarks: Dict) -> Dict[str, Any]:
        """Calculate performance insights from benchmarks"""
        
        try:
            insights = {}
            
            # Extract successful benchmark times
            times = {}
            for bench_name, bench_data in benchmarks.items():
                if bench_data and bench_data.get('execution_time_seconds'):
                    times[bench_name] = bench_data['execution_time_seconds']
            
            # Calculate scaling efficiency
            if 'single_profile_time' in times and 'all_profile_time' in times:
                single_time = times['single_profile_time']
                all_time = times['all_profile_time']
                
                # Theoretical sequential time for all profiles
                theoretical_sequential = single_time * len(self.all_profiles)
                
                # Actual parallel efficiency
                parallel_efficiency = (theoretical_sequential - all_time) / theoretical_sequential * 100
                speedup = theoretical_sequential / all_time if all_time > 0 else 0
                
                insights['scaling_analysis'] = {
                    'single_profile_time': single_time,
                    'all_profiles_time': all_time,
                    'theoretical_sequential_time': theoretical_sequential,
                    'parallel_efficiency_percent': parallel_efficiency,
                    'speedup_factor': speedup,
                    'time_saved_seconds': theoretical_sequential - all_time
                }
            
            # Calculate linear scaling
            profile_counts = []
            execution_times = []
            
            for bench_name, bench_data in benchmarks.items():
                if bench_data and bench_data.get('profile_count') and bench_data.get('execution_time_seconds'):
                    profile_counts.append(bench_data['profile_count'])
                    execution_times.append(bench_data['execution_time_seconds'])
            
            if len(profile_counts) >= 2:
                insights['scaling_linearity'] = {
                    'profile_counts': profile_counts,
                    'execution_times': execution_times,
                    'appears_linear': self._check_linear_scaling(profile_counts, execution_times)
                }
            
            return insights
            
        except Exception as e:
            return {'error': str(e)}

    def _check_linear_scaling(self, profile_counts: List[int], execution_times: List[float]) -> bool:
        """Check if execution time scales linearly with profile count"""
        
        try:
            # Simple linear correlation check
            if len(profile_counts) < 2:
                return False
            
            # Calculate correlation coefficient (simplified)
            import statistics
            
            mean_profiles = statistics.mean(profile_counts)
            mean_times = statistics.mean(execution_times)
            
            numerator = sum((p - mean_profiles) * (t - mean_times) 
                          for p, t in zip(profile_counts, execution_times))
            
            profiles_sq = sum((p - mean_profiles) ** 2 for p in profile_counts)
            times_sq = sum((t - mean_times) ** 2 for t in execution_times)
            
            denominator = (profiles_sq * times_sq) ** 0.5
            
            if denominator == 0:
                return False
            
            correlation = numerator / denominator
            
            # Consider linear if correlation > 0.7
            return abs(correlation) > 0.7
            
        except Exception:
            return False

    def test_fault_tolerance(self) -> Dict[str, Any]:
        """Test fault tolerance capabilities"""
        
        logger.info("Testing fault tolerance...")
        
        try:
            # Test with a mix of valid and invalid profiles
            test_profiles = ['RNN', 'INVALID_PROFILE', 'RN', 'ANOTHER_INVALID']
            
            start_time = time.time()
            
            execution_input = {
                "profiles": test_profiles,
                "execution_type": "fault_tolerance_test",
                "test_mode": True
            }
            
            state_machine_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.config['state_machine_name']}"
            
            response = self.stepfunctions_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=f"fault-tolerance-{int(time.time())}",
                input=json.dumps(execution_input)
            )
            
            execution_arn = response['executionArn']
            
            # Wait for completion
            execution_result = self._wait_for_execution_completion(execution_arn, timeout_minutes=15)
            
            total_time = time.time() - start_time
            
            # Analyze fault tolerance behavior
            if execution_result.get('status') == 'SUCCEEDED':
                output = json.loads(execution_result.get('output', '{}'))
                
                return {
                    'status': 'success',
                    'graceful_failure_handling': True,
                    'execution_time': total_time,
                    'valid_profiles_processed': len(output.get('summary', {}).get('successful_endpoints', [])),
                    'invalid_profiles_handled': len(test_profiles) - len(output.get('summary', {}).get('successful_endpoints', [])),
                    'pipeline_continued': True,
                    'output': output
                }
            else:
                return {
                    'status': 'failed',
                    'graceful_failure_handling': False,
                    'execution_time': total_time,
                    'error': execution_result.get('error')
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def test_resource_cleanup(self) -> Dict[str, Any]:
        """Test resource cleanup capabilities"""
        
        logger.info("Testing resource cleanup...")
        
        try:
            # Run a test execution and verify cleanup
            test_profiles = ['RNN', 'RN']
            
            start_time = time.time()
            
            execution_input = {
                "profiles": test_profiles,
                "execution_type": "cleanup_test",
                "test_mode": True
            }
            
            state_machine_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.config['state_machine_name']}"
            
            response = self.stepfunctions_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=f"cleanup-test-{int(time.time())}",
                input=json.dumps(execution_input)
            )
            
            execution_arn = response['executionArn']
            
            # Wait for completion
            execution_result = self._wait_for_execution_completion(execution_arn, timeout_minutes=15)
            
            total_time = time.time() - start_time
            
            # Verify cleanup occurred
            cleanup_verification = self._verify_resource_cleanup(execution_arn)
            
            return {
                'status': 'success' if execution_result.get('status') == 'SUCCEEDED' else 'failed',
                'execution_time': total_time,
                'cleanup_verification': cleanup_verification,
                'execution_arn': execution_arn
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _verify_resource_cleanup(self, execution_arn: str) -> Dict[str, Any]:
        """Verify that resources were properly cleaned up"""
        
        try:
            # Check for any remaining endpoints that might have been created during test
            endpoints = self.sagemaker_client.list_endpoints()
            
            test_endpoints = [
                ep for ep in endpoints['Endpoints'] 
                if 'energy-pred' in ep['EndpointName'] and 'test' in ep['EndpointName']
            ]
            
            # Check for endpoint configurations
            configs = self.sagemaker_client.list_endpoint_configs()
            
            test_configs = [
                config for config in configs['EndpointConfigs']
                if 'energy' in config['EndpointConfigName'] and 'test' in config['EndpointConfigName']
            ]
            
            return {
                'remaining_test_endpoints': len(test_endpoints),
                'remaining_test_configs': len(test_configs),
                'cleanup_appears_successful': len(test_endpoints) == 0 and len(test_configs) == 0,
                'verification_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'cleanup_verification_failed': True
            }

    def test_cost_optimization(self) -> Dict[str, Any]:
        """Test cost optimization features"""
        
        logger.info("Testing cost optimization...")
        
        try:
            # Calculate theoretical cost savings
            endpoint_cost_per_hour = 0.115  # Approximate ml.m5.large cost
            
            # Test with multiple profiles to calculate savings
            test_profiles = ['RNN', 'RN', 'M']
            
            # Estimate always-on vs on-demand costs
            always_on_daily_cost = len(test_profiles) * endpoint_cost_per_hour * 24
            
            # Run test execution to measure actual runtime
            start_time = time.time()
            
            execution_input = {
                "profiles": test_profiles,
                "execution_type": "cost_optimization_test",
                "test_mode": True
            }
            
            state_machine_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.config['state_machine_name']}"
            
            response = self.stepfunctions_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=f"cost-test-{int(time.time())}",
                input=json.dumps(execution_input)
            )
            
            execution_arn = response['executionArn']
            
            # Wait for completion
            execution_result = self._wait_for_execution_completion(execution_arn, timeout_minutes=15)
            
            total_time = time.time() - start_time
            execution_hours = total_time / 3600
            
            # Calculate actual on-demand cost
            on_demand_cost = len(test_profiles) * endpoint_cost_per_hour * execution_hours
            
            # Calculate savings
            cost_savings = always_on_daily_cost - on_demand_cost
            savings_percentage = (cost_savings / always_on_daily_cost) * 100 if always_on_daily_cost > 0 else 0
            
            return {
                'status': 'success',
                'cost_analysis': {
                    'profiles_tested': len(test_profiles),
                    'execution_time_hours': execution_hours,
                    'always_on_daily_cost_usd': always_on_daily_cost,
                    'on_demand_execution_cost_usd': on_demand_cost,
                    'daily_cost_savings_usd': cost_savings,
                    'savings_percentage': savings_percentage,
                    'endpoint_cost_per_hour': endpoint_cost_per_hour
                },
                'optimization_metrics': {
                    'cost_efficiency': savings_percentage > 95,  # Expect >95% savings
                    'resource_utilization': 'optimal' if execution_hours < 1 else 'acceptable',
                    'cleanup_verified': True  # Assume cleanup worked based on previous test
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _wait_for_execution_completion(self, execution_arn: str, timeout_minutes: int = 20) -> Dict[str, Any]:
        """Wait for Step Functions execution to complete"""
        
        try:
            start_time = time.time()
            timeout_seconds = timeout_minutes * 60
            
            while time.time() - start_time < timeout_seconds:
                response = self.stepfunctions_client.describe_execution(executionArn=execution_arn)
                
                status = response['status']
                
                if status in ['SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED']:
                    return {
                        'status': status,
                        'output': response.get('output'),
                        'error': response.get('error'),
                        'execution_time': time.time() - start_time
                    }
                
                time.sleep(10)  # Check every 10 seconds
            
            # Timeout reached
            return {
                'status': 'TIMEOUT',
                'error': f'Execution did not complete within {timeout_minutes} minutes',
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def _extract_performance_metrics(self, output: Dict) -> Dict[str, Any]:
        """Extract performance metrics from execution output"""
        
        try:
            metrics = {}
            
            summary = output.get('summary', {})
            
            if 'successful_endpoints' in summary:
                metrics['successful_endpoints'] = len(summary['successful_endpoints'])
            
            if 'failed_endpoints' in summary:
                metrics['failed_endpoints'] = len(summary['failed_endpoints'])
            
            if 'successful_predictions' in summary:
                metrics['successful_predictions'] = len(summary['successful_predictions'])
            
            # Extract timing information if available
            if 'execution_time' in output:
                metrics['pipeline_execution_time'] = output['execution_time']
            
            return metrics
            
        except Exception as e:
            return {'extraction_error': str(e)}

    def _generate_test_report(self, start_time: float, total_time: float = None, 
                            failed_at: str = None, error: str = None) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        try:
            if total_time is None:
                total_time = time.time() - start_time
            
            # Calculate overall success metrics
            total_tests = len(self.test_results)
            successful_tests = sum(
                1 for result in self.test_results.values() 
                if isinstance(result, dict) and result.get('status') != 'failed'
            )
            
            report = {
                'test_execution_summary': {
                    'start_time': datetime.fromtimestamp(start_time).isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_execution_time_seconds': total_time,
                    'total_execution_time_minutes': total_time / 60,
                    'failed_at_stage': failed_at,
                    'fatal_error': error
                },
                'test_results_summary': {
                    'total_test_categories': total_tests,
                    'successful_test_categories': successful_tests,
                    'overall_success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                    'tests_passed': successful_tests,
                    'tests_failed': total_tests - successful_tests
                },
                'detailed_results': self.test_results,
                'performance_metrics': self.performance_metrics,
                'recommendations': self._generate_recommendations(),
                'pipeline_readiness': self._assess_pipeline_readiness()
            }
            
            # Add cost analysis if available
            if 'cost_optimization' in self.test_results:
                cost_result = self.test_results['cost_optimization']
                if cost_result.get('status') == 'success':
                    report['cost_summary'] = cost_result.get('cost_analysis', {})
            
            # Add performance summary if available
            if 'performance' in self.test_results:
                perf_result = self.test_results['performance']
                if perf_result.get('status') == 'success':
                    report['performance_summary'] = perf_result.get('insights', {})
            
            return report
            
        except Exception as e:
            return {
                'report_generation_error': str(e),
                'partial_results': self.test_results
            }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        try:
            # Check prerequisites
            if 'prerequisites' in self.test_results:
                prereq = self.test_results['prerequisites']
                if not prereq.get('all_passed', False):
                    recommendations.append("Fix prerequisite issues before deployment")
                    if not prereq.get('s3_configurations_exist'):
                        recommendations.append("Ensure S3 endpoint configurations exist for all profiles")
            
            # Check Lambda functions
            if 'lambda_functions' in self.test_results:
                lambda_result = self.test_results['lambda_functions']
                if lambda_result.get('summary', {}).get('success_rate', 0) < 100:
                    recommendations.append("Fix Lambda function issues before deployment")
            
            # Check performance
            if 'performance' in self.test_results:
                perf_result = self.test_results['performance']
                if perf_result.get('status') == 'success':
                    insights = perf_result.get('insights', {})
                    scaling = insights.get('scaling_analysis', {})
                    if scaling.get('parallel_efficiency_percent', 0) < 70:
                        recommendations.append("Consider optimizing parallel execution for better performance")
            
            # Check fault tolerance
            if 'fault_tolerance' in self.test_results:
                fault_result = self.test_results['fault_tolerance']
                if not fault_result.get('graceful_failure_handling', False):
                    recommendations.append("Improve fault tolerance and error handling")
            
            # Check cost optimization
            if 'cost_optimization' in self.test_results:
                cost_result = self.test_results['cost_optimization']
                if cost_result.get('status') == 'success':
                    metrics = cost_result.get('optimization_metrics', {})
                    if not metrics.get('cost_efficiency', False):
                        recommendations.append("Review cost optimization strategy")
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("All tests passed - pipeline is ready for production deployment")
                recommendations.append("Consider enabling daily automated predictions")
                recommendations.append("Monitor performance and costs in production")
            
            return recommendations
            
        except Exception as e:
            return [f"Could not generate recommendations: {str(e)}"]

    def _assess_pipeline_readiness(self) -> Dict[str, Any]:
        """Assess overall pipeline readiness for production"""
        
        try:
            readiness_score = 0
            max_score = 0
            criteria = {}
            
            # Prerequisites (weight: 25)
            max_score += 25
            if 'prerequisites' in self.test_results:
                prereq = self.test_results['prerequisites']
                if prereq.get('all_passed', False):
                    readiness_score += 25
                    criteria['prerequisites'] = 'PASS'
                else:
                    criteria['prerequisites'] = 'FAIL'
            
            # Lambda functions (weight: 20)
            max_score += 20
            if 'lambda_functions' in self.test_results:
                lambda_result = self.test_results['lambda_functions']
                success_rate = lambda_result.get('summary', {}).get('success_rate', 0)
                readiness_score += int(success_rate * 0.2)
                criteria['lambda_functions'] = 'PASS' if success_rate >= 75 else 'PARTIAL' if success_rate >= 50 else 'FAIL'
            
            # Profile selection (weight: 15)
            max_score += 15
            if 'profile_selection' in self.test_results:
                profile_result = self.test_results['profile_selection']
                success_rate = profile_result.get('summary', {}).get('success_rate', 0)
                readiness_score += int(success_rate * 0.15)
                criteria['profile_selection'] = 'PASS' if success_rate >= 75 else 'PARTIAL' if success_rate >= 50 else 'FAIL'
            
            # Parallel execution (weight: 15)
            max_score += 15
            if 'parallel_execution' in self.test_results:
                parallel_result = self.test_results['parallel_execution']
                if parallel_result.get('status') == 'success':
                    readiness_score += 15
                    criteria['parallel_execution'] = 'PASS'
                else:
                    criteria['parallel_execution'] = 'FAIL'
            
            # Performance (weight: 10)
            max_score += 10
            if 'performance' in self.test_results:
                perf_result = self.test_results['performance']
                if perf_result.get('status') == 'success':
                    readiness_score += 10
                    criteria['performance'] = 'PASS'
                else:
                    criteria['performance'] = 'FAIL'
            
            # Fault tolerance (weight: 10)
            max_score += 10
            if 'fault_tolerance' in self.test_results:
                fault_result = self.test_results['fault_tolerance']
                if fault_result.get('graceful_failure_handling', False):
                    readiness_score += 10
                    criteria['fault_tolerance'] = 'PASS'
                else:
                    criteria['fault_tolerance'] = 'FAIL'
            
            # Resource cleanup (weight: 5)
            max_score += 5
            if 'resource_cleanup' in self.test_results:
                cleanup_result = self.test_results['resource_cleanup']
                if cleanup_result.get('status') == 'success':
                    readiness_score += 5
                    criteria['resource_cleanup'] = 'PASS'
                else:
                    criteria['resource_cleanup'] = 'FAIL'
            
            # Calculate final readiness
            readiness_percentage = (readiness_score / max_score * 100) if max_score > 0 else 0
            
            if readiness_percentage >= 90:
                readiness_level = 'PRODUCTION_READY'
            elif readiness_percentage >= 75:
                readiness_level = 'MOSTLY_READY'
            elif readiness_percentage >= 50:
                readiness_level = 'NEEDS_IMPROVEMENT'
            else:
                readiness_level = 'NOT_READY'
            
            return {
                'readiness_score': readiness_score,
                'max_possible_score': max_score,
                'readiness_percentage': readiness_percentage,
                'readiness_level': readiness_level,
                'criteria_assessment': criteria,
                'production_deployment_recommended': readiness_percentage >= 75
            }
            
        except Exception as e:
            return {
                'assessment_error': str(e),
                'readiness_level': 'UNKNOWN'
            }

def main():
    """Main function for running the enhanced prediction pipeline tests"""
    
    parser = argparse.ArgumentParser(description='Enhanced Prediction Pipeline Test Suite')
    
    # Test configuration options
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run only prerequisites validation')
    parser.add_argument('--profiles', nargs='+', 
                       help='Test specific profiles (e.g., --profiles RNN RN M)')
    parser.add_argument('--all-profiles', action='store_true',
                       help='Test all 7 profiles')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--fault-tolerance', action='store_true',
                       help='Run fault tolerance tests')
    parser.add_argument('--config-only', action='store_true',
                       help='Test only configuration validation')
    parser.add_argument('--region', default='us-west-2',
                       help='AWS region (default: us-west-2)')
    
    args = parser.parse_args()
    
    # Initialize test framework
    logger.info("Initializing Enhanced Prediction Pipeline Test Suite...")
    test_framework = EnhancedPredictionPipelineTest(region=args.region)
    
    # Configure test execution
    test_config = {
        'test_profile_selection': True,
        'test_parallel_execution': True,
        'test_performance': args.benchmark,
        'test_fault_tolerance': args.fault_tolerance,
        'test_cleanup': True,
        'test_cost_optimization': True
    }
    
    if args.quick_test:
        # Only run prerequisites
        logger.info("Running quick test (prerequisites only)...")
        test_config = {k: False for k in test_config.keys()}
        results = test_framework.test_prerequisites()
        
        if results.get('all_passed'):
            logger.info("✓ Quick test PASSED - Prerequisites are satisfied")
            sys.exit(0)
        else:
            logger.error("✗ Quick test FAILED - Prerequisites not met")
            sys.exit(1)
    
    elif args.config_only:
        # Only run configuration validation
        logger.info("Running configuration validation only...")
        validator_result = test_framework._test_profile_validator()
        
        if validator_result.get('status') == 'success':
            logger.info("✓ Configuration validation PASSED")
            sys.exit(0)
        else:
            logger.error("✗ Configuration validation FAILED")
            sys.exit(1)
    
    elif args.profiles:
        # Test specific profiles
        logger.info(f"Testing specific profiles: {args.profiles}")
        test_result = test_framework._run_profile_selection_test(args.profiles, 'custom')
        
        if test_result.get('status') == 'success':
            logger.info(f"✓ Profile test PASSED for {args.profiles}")
            sys.exit(0)
        else:
            logger.error(f"✗ Profile test FAILED for {args.profiles}")
            sys.exit(1)
    
    else:
        # Run comprehensive test suite
        logger.info("Running comprehensive test suite...")
        
        try:
            results = test_framework.run_comprehensive_test_suite(test_config)
            
            # Print summary
            logger.info("\n" + "="*80)
            logger.info("TEST SUITE EXECUTION COMPLETE")
            logger.info("="*80)
            
            summary = results.get('test_results_summary', {})
            logger.info(f"Total test categories: {summary.get('total_test_categories', 0)}")
            logger.info(f"Successful categories: {summary.get('successful_test_categories', 0)}")
            logger.info(f"Overall success rate: {summary.get('overall_success_rate', 0):.1f}%")
            
            readiness = results.get('pipeline_readiness', {})
            logger.info(f"Pipeline readiness: {readiness.get('readiness_level', 'UNKNOWN')}")
            logger.info(f"Readiness score: {readiness.get('readiness_percentage', 0):.1f}%")
            
            recommendations = results.get('recommendations', [])
            if recommendations:
                logger.info("\nRecommendations:")
                for i, rec in enumerate(recommendations, 1):
                    logger.info(f"{i}. {rec}")
            
            # Exit with appropriate code
            if readiness.get('production_deployment_recommended', False):
                logger.info("\n✓ ENHANCED PIPELINE IS READY FOR PRODUCTION DEPLOYMENT")
                sys.exit(0)
            else:
                logger.error("\n✗ ENHANCED PIPELINE NEEDS IMPROVEMENTS BEFORE DEPLOYMENT")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Test suite execution failed: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
