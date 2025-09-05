#!/usr/bin/env python3
"""
Verify that DataScientist role exists and has required permissions
Checks single DataScientist role instead of multiple roles
"""

import boto3
import json
from datetime import datetime

def verify_datascientist_role(role_name="sdcp-dev-sagemaker-energy-forecasting-datascientist-role"):
    """Verify DataScientist role exists and is accessible"""
    
    account_id = boto3.client('sts').get_caller_identity()['Account']
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    
    print(" Verifying DataScientist Role Setup...")
    print(f"Role: {role_name}")
    print(f"ARN: {role_arn}")
    
    verification_results = {
        'role_name': role_name,
        'role_arn': role_arn,
        'account_id': account_id,
        'verification_time': datetime.now().isoformat()
    }
    
    # Test 1: Check if role exists using user credentials
    print("\n Test 1: Checking if DataScientist role exists...")
    try:
        iam_client = boto3.client('iam')
        response = iam_client.get_role(RoleName=role_name)
        verification_results['role_exists'] = True
        verification_results['role_created'] = response['Role']['CreateDate'].isoformat()
        print(f" {role_name}: EXISTS")
        
    except iam_client.exceptions.NoSuchEntityException:
        verification_results['role_exists'] = False
        verification_results['error'] = 'Role does not exist'
        print(f" {role_name}: NOT FOUND")
        return verification_results
    except Exception as e:
        verification_results['role_exists'] = False
        verification_results['error'] = str(e)
        print(f" {role_name}: ERROR - {str(e)}")
        return verification_results
    
    # Test 2: Check if we can assume the role
    print("\n Test 2: Testing role assumption...")
    try:
        sts_client = boto3.client('sts')
        response = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=f"VerificationTest-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            DurationSeconds=900  # 15 minutes
        )
        
        verification_results['role_assumable'] = True
        verification_results['session_expiration'] = response['Credentials']['Expiration'].isoformat()
        print(f" Can assume DataScientist role")
        
        # Create assumed session for permission tests
        credentials = response['Credentials']
        assumed_session = boto3.Session(
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )
        
    except Exception as e:
        verification_results['role_assumable'] = False
        verification_results['assumption_error'] = str(e)
        print(f" Cannot assume DataScientist role: {str(e)}")
        print("   This means your user doesn't have permission to assume this role")
        return verification_results
    
    # Test 3: Check required permissions through assumed role
    print("\n Test 3: Testing required permissions through assumed role...")
    
    permission_tests = {
        'ECR': test_ecr_permissions,
        'Step Functions': test_stepfunctions_permissions,
        'EventBridge': test_eventbridge_permissions,
        'Lambda': test_lambda_permissions,
        'SageMaker': test_sagemaker_permissions,
        'S3': test_s3_permissions
    }
    
    verification_results['permissions'] = {}
    
    for service, test_func in permission_tests.items():
        try:
            result = test_func(assumed_session)
            verification_results['permissions'][service] = result
            if result['has_permission']:
                print(f"   {service}: Permissions OK")
            else:
                print(f"   {service}: Missing permissions - {result['error']}")
        except Exception as e:
            verification_results['permissions'][service] = {
                'has_permission': False,
                'error': str(e)
            }
            print(f"   {service}: Test failed - {str(e)}")
    
    # Test 4: Check ECR repositories
    print("\n Test 4: Checking ECR repositories...")
    
    required_repos = ['energy-preprocessing', 'energy-training', 'energy-prediction']
    verification_results['ecr_repositories'] = {}
    
    try:
        ecr_client = assumed_session.client('ecr')
        for repo_name in required_repos:
            try:
                response = ecr_client.describe_repositories(repositoryNames=[repo_name])
                verification_results['ecr_repositories'][repo_name] = {
                    'exists': True,
                    'uri': response['repositories'][0]['repositoryUri']
                }
                print(f"   {repo_name}: EXISTS")
            except ecr_client.exceptions.RepositoryNotFoundException:
                verification_results['ecr_repositories'][repo_name] = {
                    'exists': False,
                    'error': 'Repository does not exist'
                }
                print(f"    {repo_name}: NOT FOUND (will be created during setup)")
            except Exception as e:
                verification_results['ecr_repositories'][repo_name] = {
                    'exists': False,
                    'error': str(e)
                }
                print(f"   {repo_name}: ERROR - {str(e)}")
    except Exception as e:
        print(f"   ECR repository check failed: {str(e)}")
    
    # Save verification results
    with open('datascientist_role_verification.json', 'w') as f:
        json.dump(verification_results, f, indent=2, default=str)
    
    # Summary
    print(f"\n VERIFICATION SUMMARY:")
    print(f"Role Exists: {' ' if verification_results.get('role_exists', False) else ' '}")
    print(f"Role Assumable: {' ' if verification_results.get('role_assumable', False) else ' '}")
    
    # Count permission results
    permissions = verification_results.get('permissions', {})
    permission_count = sum(1 for p in permissions.values() if p.get('has_permission', False))
    total_permissions = len(permissions)
    print(f"Permissions: {permission_count}/{total_permissions} ")
    
    # Check if ready for deployment
    role_ready = (
        verification_results.get('role_exists', False) and 
        verification_results.get('role_assumable', False) and
        permission_count >= 4  # At least ECR, Step Functions, Lambda, SageMaker
    )
    
    if role_ready:
        print("\n READY FOR DEPLOYMENT!")
        print("DataScientist role is properly configured.")
        print("You can proceed with:")
        print("  python infrastructure/setup_infrastructure.py")
        print("  python deployment/lambda_deployer.py")
    else:
        print("\n  NOT READY FOR DEPLOYMENT")
        print("Issues found with DataScientist role configuration.")
        print("Contact admin team to resolve the issues above.")
    
    return role_ready

# Permission test functions
def test_ecr_permissions(session):
    """Test ECR permissions"""
    try:
        ecr_client = session.client('ecr')
        ecr_client.describe_repositories()
        return {'has_permission': True}
    except Exception as e:
        return {'has_permission': False, 'error': str(e)}

def test_stepfunctions_permissions(session):
    """Test Step Functions permissions"""
    try:
        sf_client = session.client('stepfunctions')
        sf_client.list_state_machines()
        return {'has_permission': True}
    except Exception as e:
        return {'has_permission': False, 'error': str(e)}

def test_eventbridge_permissions(session):
    """Test EventBridge permissions"""
    try:
        events_client = session.client('events')
        events_client.list_rules()
        return {'has_permission': True}
    except Exception as e:
        return {'has_permission': False, 'error': str(e)}

def test_lambda_permissions(session):
    """Test Lambda permissions"""
    try:
        lambda_client = session.client('lambda')
        lambda_client.list_functions()
        return {'has_permission': True}
    except Exception as e:
        return {'has_permission': False, 'error': str(e)}

def test_sagemaker_permissions(session):
    """Test SageMaker permissions"""
    try:
        sm_client = session.client('sagemaker')
        sm_client.list_processing_jobs()
        return {'has_permission': True}
    except Exception as e:
        return {'has_permission': False, 'error': str(e)}

def test_s3_permissions(session):
    """Test S3 permissions"""
    try:
        s3_client = session.client('s3')
        s3_client.list_buckets()
        return {'has_permission': True}
    except Exception as e:
        return {'has_permission': False, 'error': str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify DataScientist role setup')
    parser.add_argument('--role-name', default='sdcp-dev-sagemaker-energy-forecasting-datascientist-role', help='DataScientist role name')
    
    args = parser.parse_args()
    
    success = verify_datascientist_role(args.role_name)
    exit(0 if success else 1)
