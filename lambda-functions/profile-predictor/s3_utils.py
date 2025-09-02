"""
S3 Utilities Module
lambda-functions/profile-predictor/s3_utils.py

S3 operations utilities extracted from container logic
Handles all S3 interactions for the Lambda function
"""

import boto3
import logging
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class S3Utils:
    """
    S3 utilities for Lambda function
    Extracted from container S3 operations
    """
   
    def __init__(self):
        """Initialize S3 client"""
        self.s3_client = boto3.client('s3')
        logger.info("S3 utilities initialized")

    def get_object(self, bucket: str, key: str) -> str:
        """
        Get object content from S3
       
        Args:
            bucket: S3 bucket name
            key: S3 object key
           
        Returns:
            Object content as string
        """
       
        try:
            logger.debug(f"Getting object: s3://{bucket}/{key}")
           
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
           
            logger.debug(f"✓ Successfully retrieved object: {len(content)} characters")
           
            return content
           
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}")
            elif error_code == 'NoSuchBucket':
                raise FileNotFoundError(f"S3 bucket not found: {bucket}")
            else:
                raise Exception(f"S3 get_object failed: {str(e)}")
               
        except Exception as e:
            logger.error(f"Failed to get S3 object s3://{bucket}/{key}: {str(e)}")
            raise

    def put_object(self, bucket: str, key: str, body: str, content_type: str = 'text/plain') -> bool:
        """
        Put object to S3
       
        Args:
            bucket: S3 bucket name
            key: S3 object key
            body: Object content
            content_type: MIME type of the content
           
        Returns:
            True if successful
        """
       
        try:
            logger.debug(f"Putting object: s3://{bucket}/{key}")
           
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType=content_type
            )
           
            logger.debug(f"✓ Successfully uploaded object: s3://{bucket}/{key}")
           
            return True
           
        except Exception as e:
            logger.error(f"Failed to put S3 object s3://{bucket}/{key}: {str(e)}")
            raise

    def object_exists(self, bucket: str, key: str) -> bool:
        """
        Check if S3 object exists
       
        Args:
            bucket: S3 bucket name
            key: S3 object key
           
        Returns:
            True if object exists
        """
       
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
           
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['NoSuchKey', '404']:
                return False
            else:
                logger.error(f"Error checking S3 object existence: {str(e)}")
                raise
               
        except Exception as e:
            logger.error(f"Failed to check S3 object existence s3://{bucket}/{key}: {str(e)}")
            raise

    def list_objects(self, bucket: str, prefix: str = '', max_keys: int = 1000) -> list:
        """
        List objects in S3 bucket with prefix
       
        Args:
            bucket: S3 bucket name
            prefix: Object key prefix
            max_keys: Maximum number of keys to return
           
        Returns:
            List of object keys
        """
       
        try:
            logger.debug(f"Listing objects: s3://{bucket}/{prefix}*")
           
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
           
            objects = []
            if 'Contents' in response:
                objects = [obj['Key'] for obj in response['Contents']]
           
            logger.debug(f"✓ Found {len(objects)} objects")
           
            return objects
           
        except Exception as e:
            logger.error(f"Failed to list S3 objects s3://{bucket}/{prefix}: {str(e)}")
            raise

    def get_object_metadata(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Get S3 object metadata
       
        Args:
            bucket: S3 bucket name
            key: S3 object key
           
        Returns:
            Object metadata dictionary
        """
       
        try:
            logger.debug(f"Getting metadata: s3://{bucket}/{key}")
           
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
           
            metadata = {
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType', ''),
                'etag': response.get('ETag', '').strip('"'),
                'metadata': response.get('Metadata', {})
            }
           
            logger.debug(f"✓ Retrieved metadata for s3://{bucket}/{key}")
           
            return metadata
           
        except Exception as e:
            logger.error(f"Failed to get S3 object metadata s3://{bucket}/{key}: {str(e)}")
            raise

    def copy_object(self, source_bucket: str, source_key: str,
                   dest_bucket: str, dest_key: str) -> bool:
        """
        Copy object within S3
       
        Args:
            source_bucket: Source bucket name
            source_key: Source object key
            dest_bucket: Destination bucket name
            dest_key: Destination object key
           
        Returns:
            True if successful
        """
       
        try:
            logger.debug(f"Copying s3://{source_bucket}/{source_key} to s3://{dest_bucket}/{dest_key}")
           
            copy_source = {
                'Bucket': source_bucket,
                'Key': source_key
            }
           
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )
           
            logger.debug(f"✓ Successfully copied object")
           
            return True
           
        except Exception as e:
            logger.error(f"Failed to copy S3 object: {str(e)}")
            raise

    def delete_object(self, bucket: str, key: str) -> bool:
        """
        Delete object from S3
       
        Args:
            bucket: S3 bucket name
            key: S3 object key
           
        Returns:
            True if successful
        """
       
        try:
            logger.debug(f"Deleting object: s3://{bucket}/{key}")
           
            self.s3_client.delete_object(Bucket=bucket, Key=key)
           
            logger.debug(f"✓ Successfully deleted object")
           
            return True
           
        except Exception as e:
            logger.error(f"Failed to delete S3 object s3://{bucket}/{key}: {str(e)}")
            raise

    def generate_presigned_url(self, bucket: str, key: str, expiration: int = 3600) -> str:
        """
        Generate presigned URL for S3 object
       
        Args:
            bucket: S3 bucket name
            key: S3 object key
            expiration: URL expiration time in seconds
           
        Returns:
            Presigned URL string
        """
       
        try:
            logger.debug(f"Generating presigned URL: s3://{bucket}/{key}")
           
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
           
            logger.debug(f"✓ Generated presigned URL (expires in {expiration}s)")
           
            return url
           
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for s3://{bucket}/{key}: {str(e)}")
            raise

    def upload_dataframe(self, df, bucket: str, key: str, file_format: str = 'csv') -> bool:
        """
        Upload pandas DataFrame to S3
       
        Args:
            df: pandas DataFrame
            bucket: S3 bucket name
            key: S3 object key
            file_format: File format ('csv', 'json', 'parquet')
           
        Returns:
            True if successful
        """
       
        try:
            logger.debug(f"Uploading DataFrame to s3://{bucket}/{key} as {file_format}")
           
            from io import StringIO, BytesIO
           
            if file_format.lower() == 'csv':
                buffer = StringIO()
                df.to_csv(buffer, index=False)
                content = buffer.getvalue()
                content_type = 'text/csv'
               
            elif file_format.lower() == 'json':
                buffer = StringIO()
                df.to_json(buffer, orient='records', date_format='iso')
                content = buffer.getvalue()
                content_type = 'application/json'
               
            elif file_format.lower() == 'parquet':
                buffer = BytesIO()
                df.to_parquet(buffer, index=False)
                content = buffer.getvalue()
                content_type = 'application/octet-stream'
               
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
           
            self.put_object(bucket, key, content, content_type)
           
            logger.debug(f"✓ Successfully uploaded DataFrame: {df.shape}")
           
            return True
           
        except Exception as e:
            logger.error(f"Failed to upload DataFrame to s3://{bucket}/{key}: {str(e)}")
            raise

    def download_dataframe(self, bucket: str, key: str, file_format: str = 'csv'):
        """
        Download S3 object as pandas DataFrame
       
        Args:
            bucket: S3 bucket name
            key: S3 object key
            file_format: File format ('csv', 'json', 'parquet')
           
        Returns:
            pandas DataFrame
        """
       
        try:
            logger.debug(f"Downloading DataFrame from s3://{bucket}/{key} as {file_format}")
           
            import pandas as pd
            from io import StringIO, BytesIO
           
            if file_format.lower() == 'csv':
                content = self.get_object(bucket, key)
                df = pd.read_csv(StringIO(content))
               
            elif file_format.lower() == 'json':
                content = self.get_object(bucket, key)
                df = pd.read_json(StringIO(content), orient='records')
               
            elif file_format.lower() == 'parquet':
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read()
                df = pd.read_parquet(BytesIO(content))
               
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
           
            logger.debug(f"✓ Successfully downloaded DataFrame: {df.shape}")
           
            return df
           
        except Exception as e:
            logger.error(f"Failed to download DataFrame from s3://{bucket}/{key}: {str(e)}")
            raise

    def get_bucket_size(self, bucket: str, prefix: str = '') -> Dict[str, Any]:
        """
        Get bucket size and object count
       
        Args:
            bucket: S3 bucket name
            prefix: Object key prefix
           
        Returns:
            Dictionary with size and count information
        """
       
        try:
            logger.debug(f"Calculating bucket size: s3://{bucket}/{prefix}")
           
            total_size = 0
            object_count = 0
           
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
           
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
                        object_count += 1
           
            size_info = {
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_size_gb': round(total_size / (1024 * 1024 * 1024), 4),
                'object_count': object_count,
                'average_size_bytes': round(total_size / object_count, 2) if object_count > 0 else 0
            }
           
            logger.debug(f"✓ Bucket analysis complete: {object_count} objects, {size_info['total_size_mb']} MB")
           
            return size_info
           
        except Exception as e:
            logger.error(f"Failed to analyze bucket size s3://{bucket}/{prefix}: {str(e)}")
            raise
