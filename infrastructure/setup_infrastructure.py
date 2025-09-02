#!/usr/bin/env python3
"""
Infrastructure setup for Energy Forecasting System
Uses DataScientist role assumption for all AWS operations
"""

import boto3
import json
import time
from datetime import datetime
import os
import sys

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step_functions_definitions import create_step_functions_with_integration

class EnergyForecastingInfrastructure:
    def __init__(self, region="us-west-2", account_id=None, datascientist_role_name="sdcp-dev-sagemaker-energy-forecasting-datascientist-role"):
        self.region = region
        self.account_id = account_id or boto3.client('sts').get_caller_identity()['Account']
        self.datascientist_role_name = datascientist_role_name
        self.datascientist_role_arn = f"arn:aws:iam::{self.account_id}:role/{datascientist_role_name}"
        
        # Assume DataScientist role and get session
        self.assumed_session = self._assume_datascientist_role()
        
        # Initialize clients with assumed role credentials
        self.ecr_client = self.assumed_session.client('ecr', region_name=region)
        self.s3_client = self.assumed_session.client('s3', region_name=region)
        self.stepfunctions_client = self.assumed_session.client('stepfunctions', region_name=region)
        self.events_client = self.assumed_session.client('events', region_name=region)
        self.iam_client = self.assumed_session.client('iam', region_name=region)
        
        self.setup_config = {
            "project_name": "energy-forecasting",
            "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
            "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
            "repositories": ["energy-preprocessing", "energy-training", "energy-prediction"]
        }
    
    def _assume_datascientist_role(self):
        """Assume DataScientist role and return session with assumed credentials"""
        print(f"Assuming DataScientist role: {self.datascientist_role_arn}")
        
        try:
            # Create STS client with user credentials
            sts_client = boto3.client('sts', region_name=self.region)
            
            # Assume the DataScientist role
            response = sts_client.assume_role(
                RoleArn=self.datascientist_role_arn,
                RoleSessionName=f"EnergyForecasting-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                DurationSeconds=3600  # 1 hour session
            )
            
            # Extract temporary credentials
            credentials = response['Credentials']
            
            # Create session with assumed role credentials
            assumed_session = boto3.Session(
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken'],
                region_name=self.region
            )
            
            print(f"✓ Successfully assumed DataScientist role")
            print(f"  Session expires: {credentials['Expiration']}")
            
            return assumed_session
            
        except Exception as e:
            print(f" Failed to assume DataScientist role: {str(e)}")
            print("Possible issues:")
            print("1. DataScientist role doesn't exist")
            print("2. Your user doesn't have permission to assume the role")
            print("3. Role doesn't have trust relationship with your user")
            raise Exception(f"Role assumption failed: {str(e)}")
    
    def get_existing_roles(self):
        """Get DataScientist role ARN for all AWS services"""
        print("Using DataScientist role for all AWS operations...")
        
        # Verify the DataScientist role exists
        try:
            response = self.iam_client.get_role(RoleName=self.datascientist_role_name)
            role_arn = response['Role']['Arn']
            
            print(f"✓ Found DataScientist role: {self.datascientist_role_name}")
            
            # Use the same role for all services
            roles = {
                'sagemaker_role': role_arn,
                'stepfunctions_role': role_arn,
                'eventbridge_role': role_arn,
                'lambda_role': role_arn,
                'datascientist_role': role_arn
            }
            
            return roles
            
        except self.iam_client.exceptions.NoSuchEntityException:
            raise Exception(
                f"DataScientist role '{self.datascientist_role_name}' not found. "
                f"Please contact admin team to create this role."
            )
    
    def create_ecr_repositories(self):
        """Create ECR repositories for containers"""
        print("Creating ECR repositories...")
        
        repositories = []
        for repo_name in self.setup_config['repositories']:
            try:
                response = self.ecr_client.create_repository(
                    repositoryName=repo_name,
                    imageScanningConfiguration={'scanOnPush': True},
                    imageTagMutability='MUTABLE'
                )
                repositories.append(response['repository']['repositoryUri'])
                print(f"✓ Created repository: {repo_name}")
                
            except self.ecr_client.exceptions.RepositoryAlreadyExistsException:
                # Get existing repository URI
                response = self.ecr_client.describe_repositories(repositoryNames=[repo_name])
                repositories.append(response['repositories'][0]['repositoryUri'])
                print(f"✓ Repository already exists: {repo_name}")
            except Exception as e:
                print(f" Error with repository {repo_name}: {str(e)}")
                # Continue with other repositories
        
        return repositories
    
    def create_step_functions(self, roles):
        """Create Step Function state machines using assumed role"""
        print("Creating Step Functions with assumed role...")
        
        state_machines = create_step_functions_with_integration(
            roles=roles,
            account_id=self.account_id,
            region=self.region,
            data_bucket=self.setup_config['data_bucket'],
            model_bucket=self.setup_config['model_bucket'],
            assumed_session=self.assumed_session  # Pass assumed session
        )
        
        return state_machines
    
    def create_schedules(self, state_machines, roles):
        """Create EventBridge schedules"""
        print("Creating EventBridge schedules...")
        
        # Monthly training schedule (first day of month 2 AM UTC)
        try:
            self.events_client.put_rule(
                Name='energy-forecasting-monthly-training',
                ScheduleExpression='cron(0 2 1 * ? *)',  # Monthly instead of weekly
                State='ENABLED',
                Description='Monthly training pipeline for energy forecasting with model registry'
            )
            
            self.events_client.put_targets(
                Rule='energy-forecasting-monthly-training',
                Targets=[
                    {
                        'Id': '1',
                        'Arn': state_machines['training_pipeline'],
                        'RoleArn': roles['datascientist_role'],  # Use DataScientist role
                        'Input': json.dumps({
                            "PreprocessingJobName": f"energy-preprocessing-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                            "TrainingJobName": f"energy-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                            "PreprocessingImageUri": f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/energy-preprocessing:latest",
                            "TrainingImageUri": f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/energy-training:latest"
                        })
                    }
                ]
            )
            print("✓ Created monthly training schedule")
        except Exception as e:
            print(f"Training schedule issue: {e}")
        
        # Prediction schedule (daily - 1 AM UTC) - FOR FUTURE USE
        try:
            self.events_client.put_rule(
                Name='energy-forecasting-daily-predictions',
                ScheduleExpression='cron(0 1 * * ? *)',
                State='DISABLED',  # Start disabled until prediction container is ready
                Description='Daily predictions for energy forecasting (currently disabled)'
            )
            
            self.events_client.put_targets(
                Rule='energy-forecasting-daily-predictions',
                Targets=[
                    {
                        'Id': '1',
                        'Arn': state_machines['prediction_pipeline'],
                        'RoleArn': roles['datascientist_role'],  # Use DataScientist role
                        'Input': json.dumps({
                            "PredictionJobName": f"energy-prediction-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                            "PredictionImageUri": f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/energy-prediction:latest"
                        })
                    }
                ]
            )
            print("✓ Created daily prediction schedule (disabled)")
        except Exception as e:
            print(f"Prediction schedule issue: {e}")
    
    def verify_permissions(self):
        """Verify required permissions are available through DataScientist role"""
        print("Verifying permissions through DataScientist role...")
        
        try:
            # Test ECR permissions
            self.ecr_client.describe_repositories()
            print("✓ ECR permissions verified")
        except Exception as e:
            print(f" ECR permissions issue: {str(e)}")
            
        try:
            # Test Step Functions permissions
            self.stepfunctions_client.list_state_machines()
            print("✓ Step Functions permissions verified")
        except Exception as e:
            print(f" Step Functions permissions issue: {str(e)}")
            
        try:
            # Test EventBridge permissions
            self.events_client.list_rules()
            print("✓ EventBridge permissions verified")
        except Exception as e:
            print(f" EventBridge permissions issue: {str(e)}")
    
    def setup_infrastructure_without_roles(self):
        """Setup infrastructure using DataScientist role"""
        print(f"Setting up Energy Forecasting infrastructure in {self.region}...")
        print(f"Account ID: {self.account_id}")
        print(f"Using DataScientist role: {self.datascientist_role_name}")
        
        try:
            # Step 1: Verify permissions
            self.verify_permissions()
            
            # Step 2: Get DataScientist role (verify it exists)
            roles = self.get_existing_roles()
            
            # Step 3: Create ECR repositories
            repositories = self.create_ecr_repositories()
            
            # Step 4: Create Step Functions
            state_machines = self.create_step_functions(roles)
            
            # Step 5: Create schedules
            self.create_schedules(state_machines, roles)
            
            print("\n" + "="*60)
            print("INFRASTRUCTURE SETUP COMPLETE!")
            print("="*60)
            
            setup_summary = {
                "timestamp": datetime.now().isoformat(),
                "region": self.region,
                "account_id": self.account_id,
                "datascientist_role": self.datascientist_role_arn,
                "roles_used": roles,
                "repositories": repositories,
                "state_machines": state_machines,
                "schedules": {
                    "training": "Monthly (1st day of month, 2 AM UTC)",
                    "prediction": "Daily (1 AM UTC, currently disabled)"
                },
                "mlops_components": {
                    "step_1": "Containerized Training (containers ready)",
                    "step_2": "Model Registry (Lambda function ready to deploy)",
                    "step_3": "Endpoint Management (Lambda function ready to deploy)",
                    "cost_optimization": "Endpoints deleted after configuration save"
                },
                "next_steps": [
                    "1. Deploy Lambda functions using lambda_deployer.py",
                    "2. Build and push Docker images to ECR repositories", 
                    "3. Test the complete training pipeline",
                    "4. Verify model registry and endpoint management"
                ]
            }
            
            # Save setup summary
            with open('infrastructure_setup_summary.json', 'w') as f:
                json.dump(setup_summary, f, indent=2)
            
            print(f"\nSetup summary saved to: infrastructure_setup_summary.json")
            print(f"\nECR Repositories:")
            for repo in repositories:
                print(f"  - {repo}")
            
            print(f"\nStep Functions:")
            for name, arn in state_machines.items():
                print(f"  - {name}: {arn}")
            
            print(f"\nScheduling:")
            print(f"  - Training: Monthly (1st day of month, 2 AM UTC)")
            print(f"  - Prediction: Daily (1 AM UTC, currently disabled)")
            
            print(f"\nNext steps:")
            print(f"1. Deploy Lambda functions:")
            print(f"   python deployment/lambda_deployer.py")
            print(f"2. Build and push containers:")
            print(f"   bash scripts/deploy_all.sh")
            print(f"3. Test the pipeline:")
            print(f"   python deployment/test_pipeline.py")
            
            return setup_summary
            
        except Exception as e:
            print(f"\n Infrastructure setup failed: {str(e)}")
            
            # Save error details
            error_summary = {
                "timestamp": datetime.now().isoformat(),
                "region": self.region,
                "account_id": self.account_id,
                "datascientist_role": self.datascientist_role_arn,
                "error": str(e),
                "status": "failed",
                "likely_cause": "DataScientist role missing or insufficient permissions",
                "resolution": "Contact admin team to verify DataScientist role permissions"
            }
            
            with open('infrastructure_setup_error.json', 'w') as f:
                json.dump(error_summary, f, indent=2)
            
            print(f"Error details saved to: infrastructure_setup_error.json")
            raise

def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Energy Forecasting Infrastructure')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--role-name', default='sdcp-dev-sagemaker-energy-forecasting-datascientist-role', help='DataScientist role name')
    parser.add_argument('--test-only', action='store_true', help='Only run tests/verification')
    
    args = parser.parse_args()
    
    # Initialize infrastructure
    infrastructure = EnergyForecastingInfrastructure(
        region=args.region,
        datascientist_role_name=args.role_name
    )
    
    if args.test_only:
        print("Running verification tests only...")
        try:
            infrastructure.verify_permissions()
            roles = infrastructure.get_existing_roles()
            print(" All tests passed!")
        except Exception as e:
            print(f" Tests failed: {str(e)}")
            sys.exit(1)
    else:
        # Run full setup
        summary = infrastructure.setup_infrastructure_without_roles()

if __name__ == "__main__":
    main()
