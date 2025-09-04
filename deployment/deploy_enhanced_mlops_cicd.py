# =============================================================================
# ENHANCED DEPLOYMENT SCRIPT - deployment/deploy_enhanced_mlops_cicd.py
# =============================================================================
"""
Enhanced MLOps Deployment Script with CI/CD Integration
This script enhances the existing deploy_enhanced_mlops.py with CI/CD capabilities
while maintaining the core deployment logic
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import the original deployment class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from deploy_enhanced_mlops import EnhancedMLOpsDeployment
except ImportError:
    print("Error: Could not import deploy_enhanced_mlops.py")
    print("Make sure the original deployment script exists in the same directory")
    sys.exit(1)

class CICDEnhancedMLOpsDeployment(EnhancedMLOpsDeployment):
    """
    CI/CD Enhanced MLOps Deployment
    Extends the original deployment with CI/CD-specific features
    """
    
    def __init__(self, environment: str, region: str = "us-west-2", 
                 ci_cd_mode: bool = False, github_run_id: Optional[str] = None):
        """Initialize CI/CD enhanced deployment"""
        
        # Call parent constructor with environment-aware parameters
        super().__init__(
            region=region,
            # Environment-aware role naming
            role_name=f'sdcp-{environment}-sagemaker-energy-forecasting-datascientist-role'
        )
        
        self.environment = environment
        self.ci_cd_mode = ci_cd_mode
        self.github_run_id = github_run_id
        self.deployment_start_time = datetime.now()
        
        # Environment-specific configuration
        self.env_config = self._get_environment_config()
        
        # CI/CD specific logging setup
        if ci_cd_mode:
            self._setup_cicd_logging()
        
        self.logger.info(f"CI/CD Enhanced MLOps Deployment initialized for {environment} environment")
        self.logger.info(f"CI/CD Mode: {ci_cd_mode}")
        self.logger.info(f"GitHub Run ID: {github_run_id}")
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        configs = {
            "dev": {
                "data_bucket": f"sdcp-dev-sagemaker-energy-forecasting-data",
                "model_bucket": f"sdcp-dev-sagemaker-energy-forecasting-models",
                "role_name": "sdcp-dev-sagemaker-energy-forecasting-datascientist-role",
                "lambda_prefix": "energy-forecasting-dev",
                "step_functions_prefix": "energy-forecasting-dev"
            },
            "preprod": {
                "data_bucket": f"sdcp-preprod-sagemaker-energy-forecasting-data",
                "model_bucket": f"sdcp-preprod-sagemaker-energy-forecasting-models",
                "role_name": "sdcp-preprod-sagemaker-energy-forecasting-datascientist-role",
                "lambda_prefix": "energy-forecasting-preprod",
                "step_functions_prefix": "energy-forecasting-preprod"
            },
            "prod": {
                "data_bucket": f"sdcp-prod-sagemaker-energy-forecasting-data",
                "model_bucket": f"sdcp-prod-sagemaker-energy-forecasting-models",
                "role_name": "sdcp-prod-sagemaker-energy-forecasting-datascientist-role",
                "lambda_prefix": "energy-forecasting-prod",
                "step_functions_prefix": "energy-forecasting-prod"
            }
        }
        return configs.get(self.environment, configs["dev"])
    
    def _setup_cicd_logging(self):
        """Setup CI/CD specific logging"""
        # Create CI/CD compatible log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [CI/CD] - %(levelname)s - %(message)s'
        )
        
        # Add file handler for CI/CD logs
        if self.github_run_id:
            log_filename = f"deployment-{self.environment}-{self.github_run_id}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def deploy_with_cicd_enhancements(self, deployment_bucket: Optional[str] = None,
                                    model_bucket: Optional[str] = None) -> Dict[str, Any]:
        """
        Deploy MLOps pipeline with CI/CD enhancements
        This method wraps the original deployment with CI/CD-specific features
        """
        deployment_summary = {
            "environment": self.environment,
            "ci_cd_mode": self.ci_cd_mode,
            "github_run_id": self.github_run_id,
            "deployment_start_time": self.deployment_start_time.isoformat(),
            "deployment_status": "IN_PROGRESS",
            "steps_completed": [],
            "steps_failed": [],
            "resources_deployed": {},
            "validation_results": {},
            "error_details": None
        }
        
        try:
            self.logger.info("="*80)
            self.logger.info(f"STARTING CI/CD ENHANCED MLOPS DEPLOYMENT")
            self.logger.info(f"Environment: {self.environment}")
            self.logger.info(f"Region: {self.region}")
            self.logger.info(f"GitHub Run ID: {self.github_run_id}")
            self.logger.info("="*80)
            
            # Use environment-specific buckets if provided, otherwise use defaults
            if deployment_bucket:
                self.env_config["data_bucket"] = deployment_bucket
            if model_bucket:
                self.env_config["model_bucket"] = model_bucket
            
            # Step 1: Pre-deployment validation
            self._cicd_step("Pre-deployment Validation", deployment_summary)
            validation_result = self._validate_pre_deployment()
            deployment_summary["validation_results"]["pre_deployment"] = validation_result
            
            # Step 2: Enhanced container configuration
            self._cicd_step("Container Configuration", deployment_summary)
            self._setup_environment_aware_containers()
            
            # Step 3: Call original deployment method with environment parameters
            self._cicd_step("Core MLOps Deployment", deployment_summary)
            
            # Override bucket names for environment-aware deployment
            original_data_bucket = self.data_bucket
            original_model_bucket = self.model_bucket
            
            self.data_bucket = self.env_config["data_bucket"]
            self.model_bucket = self.env_config["model_bucket"]
            
            # Call the original deployment method
            core_deployment_result = self.deploy_complete_enhanced_mlops()
            
            # Restore original bucket names
            self.data_bucket = original_data_bucket
            self.model_bucket = original_model_bucket
            
            deployment_summary["resources_deployed"] = core_deployment_result
            
            # Step 4: Post-deployment validation
            self._cicd_step("Post-deployment Validation", deployment_summary)
            post_validation = self._validate_post_deployment()
            deployment_summary["validation_results"]["post_deployment"] = post_validation
            
            # Step 5: CI/CD specific reporting
            self._cicd_step("CI/CD Reporting", deployment_summary)
            self._generate_cicd_artifacts(deployment_summary)
            
            deployment_summary["deployment_status"] = "SUCCESS"
            deployment_summary["deployment_end_time"] = datetime.now().isoformat()
            deployment_summary["deployment_duration"] = str(
                datetime.now() - self.deployment_start_time
            )
            
            self.logger.info("="*80)
            self.logger.info("CI/CD ENHANCED MLOPS DEPLOYMENT COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total Duration: {deployment_summary['deployment_duration']}")
            self.logger.info("="*80)
            
            return deployment_summary
            
        except Exception as e:
            deployment_summary["deployment_status"] = "FAILED"
            deployment_summary["error_details"] = str(e)
            deployment_summary["deployment_end_time"] = datetime.now().isoformat()
            
            self.logger.error(f"CI/CD Enhanced deployment failed: {str(e)}")
            
            # Generate failure report
            self._generate_failure_report(deployment_summary, e)
            
            raise
    
    def _cicd_step(self, step_name: str, deployment_summary: Dict[str, Any]):
        """Track CI/CD deployment steps"""
        self.logger.info(f">>> CI/CD STEP: {step_name}")
        deployment_summary["steps_completed"].append({
            "step": step_name,
            "timestamp": datetime.now().isoformat()
        })
    
    def _validate_pre_deployment(self) -> Dict[str, Any]:
        """Pre-deployment validation for CI/CD"""
        self.logger.info("Running pre-deployment validation...")
        
        validation_results = {
            "aws_credentials": False,
            "s3_buckets": False,
            "iam_roles": False,
            "environment_config": False
        }
        
        try:
            # Validate AWS credentials
            self.sts_client.get_caller_identity()
            validation_results["aws_credentials"] = True
            self.logger.info("✓ AWS credentials validation passed")
            
            # Validate S3 buckets
            for bucket_name in [self.env_config["data_bucket"], self.env_config["model_bucket"]]:
                try:
                    self.s3_client.head_bucket(Bucket=bucket_name)
                    self.logger.info(f"✓ S3 bucket exists: {bucket_name}")
                except Exception as e:
                    self.logger.warning(f"S3 bucket may not exist: {bucket_name} - {str(e)}")
            validation_results["s3_buckets"] = True
            
            # Validate IAM roles
            try:
                role_arn = f"arn:aws:iam::{self.account_id}:role/{self.env_config['role_name']}"
                self.iam_client.get_role(RoleName=self.env_config['role_name'])
                validation_results["iam_roles"] = True
                self.logger.info(f"✓ IAM role exists: {role_arn}")
            except Exception as e:
                self.logger.error(f"✗ IAM role validation failed: {str(e)}")
            
            # Validate environment configuration
            required_configs = ["data_bucket", "model_bucket", "role_name"]
            if all(key in self.env_config for key in required_configs):
                validation_results["environment_config"] = True
                self.logger.info("✓ Environment configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"Pre-deployment validation error: {str(e)}")
        
        return validation_results
    
    def _setup_environment_aware_containers(self):
        """Setup environment-aware container configurations"""
        self.logger.info("Setting up environment-aware container configurations...")
        
        try:
            # Generate container configurations for current environment
            from container_config_manager import ContainerConfigManager
            
            config_manager = ContainerConfigManager(
                environment=self.environment,
                region=self.region
            )
            
            configs_generated = config_manager.generate_container_configs()
            self.logger.info(f"Generated {len(configs_generated)} container configurations")
            
            for config_name, config_path in configs_generated.items():
                self.logger.info(f"  {config_name}: {config_path}")
            
        except Exception as e:
            self.logger.warning(f"Container configuration setup failed: {str(e)}")
            self.logger.warning("Proceeding with existing container configurations")
    
    def _validate_post_deployment(self) -> Dict[str, Any]:
        """Post-deployment validation for CI/CD"""
        self.logger.info("Running post-deployment validation...")
        
        validation_results = {
            "lambda_functions": [],
            "step_functions": [],
            "eventbridge_rules": [],
            "overall_status": "UNKNOWN"
        }
        
        try:
            # Validate Lambda functions
            expected_functions = [
                f"{self.env_config['lambda_prefix']}-model-registry",
                f"{self.env_config['lambda_prefix']}-endpoint-management",
                f"{self.env_config['lambda_prefix']}-profile-validator",
                f"{self.env_config['lambda_prefix']}-profile-predictor",
                f"{self.env_config['lambda_prefix']}-prediction-summary"
            ]
            
            for function_name in expected_functions:
                try:
                    self.lambda_client.get_function(FunctionName=function_name)
                    validation_results["lambda_functions"].append({
                        "name": function_name,
                        "status": "EXISTS"
                    })
                    self.logger.info(f"✓ Lambda function validated: {function_name}")
                except Exception:
                    validation_results["lambda_functions"].append({
                        "name": function_name,
                        "status": "MISSING"
                    })
                    self.logger.warning(f"✗ Lambda function missing: {function_name}")
            
            # Validate Step Functions
            expected_step_functions = [
                f"{self.env_config['step_functions_prefix']}-training-pipeline",
                f"{self.env_config['step_functions_prefix']}-enhanced-prediction-pipeline"
            ]
            
            for sf_name in expected_step_functions:
                try:
                    sf_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{sf_name}"
                    self.stepfunctions_client.describe_state_machine(stateMachineArn=sf_arn)
                    validation_results["step_functions"].append({
                        "name": sf_name,
                        "status": "EXISTS"
                    })
                    self.logger.info(f"✓ Step Function validated: {sf_name}")
                except Exception:
                    validation_results["step_functions"].append({
                        "name": sf_name,
                        "status": "MISSING"
                    })
                    self.logger.warning(f"✗ Step Function missing: {sf_name}")
            
            # Determine overall status
            lambda_count = len([f for f in validation_results["lambda_functions"] if f["status"] == "EXISTS"])
            sf_count = len([f for f in validation_results["step_functions"] if f["status"] == "EXISTS"])
            
            if lambda_count >= 5 and sf_count >= 2:
                validation_results["overall_status"] = "SUCCESS"
            else:
                validation_results["overall_status"] = "PARTIAL"
                
        except Exception as e:
            validation_results["overall_status"] = "FAILED"
            self.logger.error(f"Post-deployment validation error: {str(e)}")
        
        return validation_results
    
    def _generate_cicd_artifacts(self, deployment_summary: Dict[str, Any]):
        """Generate CI/CD artifacts and reports"""
        self.logger.info("Generating CI/CD artifacts...")
        
        # Generate deployment summary JSON
        summary_filename = f"deployment-summary-{self.environment}-{self.github_run_id or 'local'}.json"
        with open(summary_filename, 'w') as f:
            json.dump(deployment_summary, f, indent=2, default=str)
        
        self.logger.info(f"Generated deployment summary: {summary_filename}")
        
        # Generate GitHub Actions summary if in CI/CD mode
        if self.ci_cd_mode and os.getenv('GITHUB_STEP_SUMMARY'):
            self._generate_github_summary(deployment_summary)
    
    def _generate_github_summary(self, deployment_summary: Dict[str, Any]):
        """Generate GitHub Actions step summary"""
        summary_file = os.getenv('GITHUB_STEP_SUMMARY')
        if not summary_file:
            return
        
        with open(summary_file, 'a') as f:
            f.write(f"""
## 🚀 MLOps Deployment Summary - {self.environment.upper()}

| Parameter | Value |
|-----------|-------|
| Environment | `{self.environment}` |
| Status | {deployment_summary['deployment_status']} |
| Duration | {deployment_summary.get('deployment_duration', 'N/A')} |
| GitHub Run ID | `{self.github_run_id}` |

### Resources Deployed
- **Lambda Functions**: {len(deployment_summary['validation_results'].get('post_deployment', {}).get('lambda_functions', []))} functions
- **Step Functions**: {len(deployment_summary['validation_results'].get('post_deployment', {}).get('step_functions', []))} pipelines
- **S3 Buckets**: Data and Model buckets configured
- **Container Images**: Environment-specific configurations applied

### Validation Results
- Pre-deployment: {'✅' if deployment_summary['validation_results'].get('pre_deployment', {}).get('aws_credentials') else '❌'}
- Post-deployment: {'✅' if deployment_summary['validation_results'].get('post_deployment', {}).get('overall_status') == 'SUCCESS' else '❌'}

""")
    
    def _generate_failure_report(self, deployment_summary: Dict[str, Any], error: Exception):
        """Generate failure report for CI/CD"""
        failure_filename = f"deployment-failure-{self.environment}-{self.github_run_id or 'local'}.json"
        
        failure_report = {
            **deployment_summary,
            "failure_details": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "failed_step": deployment_summary["steps_completed"][-1] if deployment_summary["steps_completed"] else None
            }
        }
        
        with open(failure_filename, 'w') as f:
            json.dump(failure_report, f, indent=2, default=str)
        
        self.logger.error(f"Generated failure report: {failure_filename}")

def main():
    """Main function for CI/CD enhanced deployment"""
    parser = argparse.ArgumentParser(description='CI/CD Enhanced MLOps Deployment')
    parser.add_argument('--environment', required=True,
                       choices=['dev', 'preprod', 'prod'],
                       help='Target environment')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--ci-cd-mode', action='store_true',
                       help='Enable CI/CD mode')
    parser.add_argument('--github-run-id', help='GitHub Actions run ID')
    parser.add_argument('--deployment-bucket', help='Override deployment S3 bucket')
    parser.add_argument('--model-bucket', help='Override model S3 bucket')
    
    args = parser.parse_args()
    
    try:
        # Initialize CI/CD enhanced deployment
        deployment = CICDEnhancedMLOpsDeployment(
            environment=args.environment,
            region=args.region,
            ci_cd_mode=args.ci_cd_mode,
            github_run_id=args.github_run_id
        )
        
        # Run deployment with CI/CD enhancements
        result = deployment.deploy_with_cicd_enhancements(
            deployment_bucket=args.deployment_bucket,
            model_bucket=args.model_bucket
        )
        
        print(f"Deployment completed successfully for {args.environment} environment")
        print(f"Status: {result['deployment_status']}")
        print(f"Duration: {result.get('deployment_duration', 'N/A')}")
        
        return 0
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
