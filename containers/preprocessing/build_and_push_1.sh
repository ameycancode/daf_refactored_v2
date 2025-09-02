#!/bin/bash
# Build and push script for energy-preprocessing

# Set variables
REGION=us-west-2
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPOSITORY_NAME=energy-preprocessing
IMAGE_TAG=latest

echo "Building and pushing $REPOSITORY_NAME... with acoutn ID $ACCOUNT_ID in region $REGION"

# Get ECR login
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build image
echo "Building Docker image..."
docker build -t $REPOSITORY_NAME .

# Tag image
docker tag $REPOSITORY_NAME:$IMAGE_TAG $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG

# Push image
echo "Pushing image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG

echo "Successfully pushed $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG"
