#!/usr/bin/env python3
"""
Test script for the complete MLOps pipeline
"""

import boto3
import time
import json
from datetime import datetime

def test_training_pipeline():
    """Test the training pipeline"""
   
    stepfunctions = boto3.client('stepfunctions')
   
    # Get account ID
    account_id = boto3.client('sts').get_caller_identity()['Account']
    region = 'us-west-2'
   
    # Start training pipeline execution
    response = stepfunctions.start_execution(
        stateMachineArn=f'arn:aws:states:{region}:{account_id}:stateMachine:energy-forecasting-training-pipeline',
        name=f'test-execution-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        input=json.dumps({
            "PreprocessingJobName": f"test-preprocessing-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "TrainingJobName": f"test-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "TrainingDate": datetime.now().strftime('%Y%m%d'),
            "PreprocessingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-preprocessing:latest",
            "TrainingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-training:latest"
        })
    )
   
    print(f"Started test execution: {response['executionArn']}")
    return response['executionArn']


def monitor_enhanced_execution(execution_arn):
    """Monitor the enhanced pipeline execution including Lambda steps"""
    stepfunctions = boto3.client('stepfunctions')
   
    print(f"Monitoring enhanced execution: {execution_arn}")
   
    while True:
        try:
            response = stepfunctions.describe_execution(executionArn=execution_arn)
            status = response['status']
           
            print(f"Execution status: {status}")
           
            if status == 'SUCCEEDED':
                print(" Enhanced pipeline completed successfully!")
               
                # Parse the output to show Lambda results
                if 'output' in response:
                    output = json.loads(response['output'])
                    if 'model_registry_result' in output:
                        mr_result = output['model_registry_result']['Payload']['body']
                        print(f"Model Registry: {mr_result['successful_count']}/{mr_result['total_models']} models registered")
                   
                    if 'endpoint_result' in output:
                        ep_result = output['endpoint_result']['Payload']['body']
                        print(f"Endpoint Management: {ep_result.get('message', 'Completed')}")
               
                break
               
            elif status == 'FAILED':
                print(" Enhanced pipeline failed!")
                if 'error' in response:
                    print(f"Error: {response['error']}")
                break
               
            elif status in ['RUNNING', 'PENDING']:
                print(" Pipeline running... checking again in 30 seconds")
                time.sleep(30)
               
        except Exception as e:
            print(f"Error monitoring execution: {str(e)}")
            break

if __name__ == "__main__":
    execution_arn = test_training_pipeline()

    # Monitor the enhanced execution
    monitor_enhanced_execution(execution_arn)

    print("Test pipeline started successfully!")
