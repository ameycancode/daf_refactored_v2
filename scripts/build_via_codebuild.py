#!/usr/bin/env python3
"""
CodeBuild Container Build Trigger
Triggers AWS CodeBuild to build and push container images
"""

import boto3
import json
import time
import sys
import os
from datetime import datetime

class CodeBuildManager:
    def __init__(self, region="us-west-2"):
        self.region = region
        self.codebuild_client = boto3.client('codebuild', region_name=region)
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        self.role_name = 'sdcp-dev-sagemaker-energy-forecasting-datascientist-role'      
        self.project_name = "energy-forecasting-container-builds"
       
    def create_codebuild_project(self):
        """Create CodeBuild project if it doesn't exist"""
        try:
            # Check if project exists
            response = self.codebuild_client.batch_get_projects(names=[self.project_name])
            print(f"RESPONSE: {response}")
           
            if response['projects']:
                # Project exists
                print(f"✓ CodeBuild project '{self.project_name}' already exists")
                return
            elif self.project_name in response['projectsNotFound']:
                # Project doesn't exist, create it
                print(f"Creating CodeBuild project: {self.project_name}")
               
                project_config = {
                    'name': self.project_name,
                    'description': 'Build and push Energy Forecasting container images',
                    'source': {
                        'type': 'S3',
                        'location': 'sdcp-dev-sagemaker-energy-forecasting-data/codebuild-source/source.zip',
                        'buildspec': 'buildspec.yml'
                    },
                    'artifacts': {
                        'type': 'NO_ARTIFACTS'
                    },
                    'environment': {
                        'type': 'LINUX_CONTAINER',
                        'image': 'aws/codebuild/amazonlinux2-x86_64-standard:3.0',
                        'computeType': 'BUILD_GENERAL1_MEDIUM',
                        'privilegedMode': True,  # Required for Docker builds
                        'environmentVariables': [
                            {
                                'name': 'AWS_DEFAULT_REGION',
                                'value': self.region
                            },
                            {
                                'name': 'AWS_ACCOUNT_ID',
                                'value': self.account_id
                            }
                        ]
                    },
                    'serviceRole': f'arn:aws:iam::{self.account_id}:role/{self.role_name}',
                    'timeoutInMinutes': 30,
                    'tags': [
                        {
                            'key': 'Project',
                            'value': 'EnergyForecasting'
                        },
                        {
                            'key': 'Purpose',
                            'value': 'ContainerBuilds'
                        }
                    ]
                }
               
                response = self.codebuild_client.create_project(**project_config)
                print(f"✓ Created CodeBuild project: {response['project']['arn']}")
            else:
                # Unexpected response
                print("Unexpected response checking project existence")
                return


        except Exception as e:
            print(f"Unknown isue while checking for project existence in CodeBuild: {e}")
   
    def upload_source_to_s3(self):
        """Upload source code to S3 for CodeBuild"""
        import zipfile
        import tempfile
       
        print("Preparing source code for CodeBuild...")
       
        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_zip_path = temp_file.name
       
        # Create zip with necessary files
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add buildspec.yml
            if os.path.exists('buildspec.yml'):
                zipf.write('buildspec.yml')
           
            # Add container directories
            for root, dirs, files in os.walk('containers'):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path)
       
        # Upload to S3
        s3_client = boto3.client('s3', region_name=self.region)
        bucket_name = 'sdcp-dev-sagemaker-energy-forecasting-data'
        s3_key = 'codebuild-source/source.zip'
       
        try:
            # Create bucket if it doesn't exist
            try:
                s3_client.head_bucket(Bucket=bucket_name)
            except:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
                print(f"✓ Created S3 bucket: {bucket_name}")
           
            # Upload source
            s3_client.upload_file(temp_zip_path, bucket_name, s3_key)
            print(f"✓ Source code uploaded to s3://{bucket_name}/{s3_key}")
           
            # Clean up
            os.unlink(temp_zip_path)
           
            return f's3://{bucket_name}/{s3_key}'
           
        except Exception as e:
            print(f"❌ Failed to upload source: {str(e)}")
            os.unlink(temp_zip_path)
            raise
   
    def start_build(self):
        """Start CodeBuild project"""
        print(f"Starting CodeBuild project: {self.project_name}")
       
        try:
            response = self.codebuild_client.start_build(
                projectName=self.project_name,
                environmentVariablesOverride=[
                    {
                        'name': 'BUILD_TIMESTAMP',
                        'value': datetime.now().strftime('%Y%m%d-%H%M%S')
                    }
                ]
            )
           
            build_id = response['build']['id']
            print(f"✓ Build started: {build_id}")
           
            return build_id
           
        except Exception as e:
            print(f"❌ Failed to start build: {str(e)}")
            raise
   
    def wait_for_build(self, build_id, timeout_minutes=30):
        """Wait for build to complete"""
        print(f"Waiting for build to complete: {build_id}")
       
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
       
        while True:
            try:
                response = self.codebuild_client.batch_get_builds(ids=[build_id])
                build = response['builds'][0]
               
                build_status = build['buildStatus']
                current_phase = build.get('currentPhase', 'UNKNOWN')
               
                print(f"Build status: {build_status}, Phase: {current_phase}")
               
                if build_status == 'SUCCEEDED':
                    print("✅ Build completed successfully!")
                   
                    # Print build logs location
                    if 'logs' in build and 'groupName' in build['logs']:
                        log_group = build['logs']['groupName']
                        log_stream = build['logs']['streamName']
                        print(f"Build logs: CloudWatch Logs Group: {log_group}, Stream: {log_stream}")
                   
                    return True
               
                elif build_status in ['FAILED', 'FAULT', 'STOPPED', 'TIMED_OUT']:
                    print(f"❌ Build failed with status: {build_status}")
                   
                    # Print failure reason if available
                    if 'buildStatusDetails' in build:
                        print(f"Failure reason: {build['buildStatusDetails']}")
                   
                    return False
               
                elif build_status == 'IN_PROGRESS':
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        print(f"❌ Build timed out after {timeout_minutes} minutes")
                        return False
                   
                    # Wait before next check
                    time.sleep(30)
               
                else:
                    print(f"Unknown build status: {build_status}")
                    time.sleep(10)
               
            except Exception as e:
                print(f"Error checking build status: {str(e)}")
                time.sleep(10)
   
    def build_containers(self):
        """Complete container build process"""
        try:
            print("🚀 Starting container build process via CodeBuild...")
           
            # Step 1: Create CodeBuild project
            self.create_codebuild_project()
           
            # Step 2: Upload source code
            self.upload_source_to_s3()
           
            # Step 3: Start build
            build_id = self.start_build()
           
            # Step 4: Wait for completion
            success = self.wait_for_build(build_id)
           
            if success:
                print("✅ All container images built and pushed successfully via CodeBuild!")
                return True
            else:
                print("❌ Container build failed")
                return False
               
        except Exception as e:
            print(f"❌ CodeBuild process failed: {str(e)}")
            return False

def main():
    """Main function"""
    import argparse
   
    parser = argparse.ArgumentParser(description='Build containers via CodeBuild')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--create-only', action='store_true', help='Only create CodeBuild project')
   
    args = parser.parse_args()
   
    manager = CodeBuildManager(region=args.region)
   
    if args.create_only:
        print("Creating CodeBuild project only...")
        manager.create_codebuild_project()
    else:
        success = manager.build_containers()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
