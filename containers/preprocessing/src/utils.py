"""
Common utilities for Energy Forecasting containers
"""

import os
import boto3
import json
import pandas as pd
from io import StringIO
import logging

logger = logging.getLogger(__name__)

class SageMakerPathManager:
    """Manages file paths for SageMaker containers"""
   
    def __init__(self):
        self.is_sagemaker = self._is_sagemaker_environment()
        self.paths = self._get_paths()
       
    def _is_sagemaker_environment(self):
        """Check if running in SageMaker container"""
        return os.path.exists('/opt/ml')
   
    def _get_paths(self):
        """Get appropriate paths based on environment"""
        if self.is_sagemaker:
            return {
                'input_path': '/opt/ml/processing/input',
                'output_path': '/opt/ml/processing/output',
                'model_path': '/opt/ml/processing/models',
                'code_path': '/opt/ml/processing/code',
                'config_path': '/opt/ml/processing/config'
            }
        else:
            # Local development paths
            return {
                'input_path': './data/input',
                'output_path': './data/output',
                'model_path': './models',
                'code_path': '.',
                'config_path': './config'
            }
   
    @property
    def input_path(self):
        return self.paths['input_path']
   
    @property
    def output_path(self):
        return self.paths['output_path']
   
    @property
    def model_path(self):
        return self.paths['model_path']
   
    @property
    def code_path(self):
        return self.paths['code_path']
   
    @property
    def config_path(self):
        return self.paths['config_path']

class S3Manager:
    """Manages S3 operations for Energy Forecasting"""
   
    def __init__(self, data_bucket, model_bucket=None):
        self.s3_client = boto3.client('s3')
        self.data_bucket = data_bucket
        self.model_bucket = model_bucket or data_bucket
       
    def upload_file(self, local_path, s3_key, bucket=None):
        """Upload file to S3"""
        bucket = bucket or self.data_bucket
        try:
            self.s3_client.upload_file(local_path, bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {str(e)}")
            return False
   
    def download_file(self, s3_key, local_path, bucket=None):
        """Download file from S3"""
        bucket = bucket or self.data_bucket
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download s3://{bucket}/{s3_key}: {str(e)}")
            return False
   
    def upload_dataframe(self, df, s3_key, bucket=None):
        """Upload DataFrame as CSV to S3"""
        bucket = bucket or self.data_bucket
        try:
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            logger.info(f"Uploaded DataFrame to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload DataFrame: {str(e)}")
            return False
   
    def load_dataframe(self, s3_key, bucket=None):
        """Load DataFrame from S3 CSV"""
        bucket = bucket or self.data_bucket
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            df = pd.read_csv(response['Body'])
            logger.info(f"Loaded DataFrame from s3://{bucket}/{s3_key}")
            return df
        except Exception as e:
            logger.error(f"Failed to load DataFrame from s3://{bucket}/{s3_key}: {str(e)}")
            return None
   
    def list_objects(self, prefix, bucket=None):
        """List objects in S3 with given prefix"""
        bucket = bucket or self.data_bucket
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            objects = [obj['Key'] for obj in response.get('Contents', [])]
            return objects
        except Exception as e:
            logger.error(f"Failed to list objects with prefix {prefix}: {str(e)}")
            return []

class ConfigManager:
    """Manages configuration for Energy Forecasting"""
   
    def __init__(self, config_path=None):
        self.path_manager = SageMakerPathManager()
        self.config_path = config_path or self.path_manager.config_path
       
    def load_config(self, config_name):
        """Load configuration file"""
        config_file = os.path.join(self.config_path, f"{config_name}.json")
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config: {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config {config_file}: {str(e)}")
            return {}
   
    def save_config(self, config_name, config_data):
        """Save configuration file"""
        config_file = os.path.join(self.config_path, f"{config_name}.json")
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Saved config: {config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config {config_file}: {str(e)}")
            return False
   
    def get_default_config(self):
        """Get default configuration"""
        return {
            "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
            "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
            "raw_prefix": "archived_folders/forecasting/data/raw/",
            "processed_prefix": "archived_folders/forecasting/data/xgboost/processed/",
            "input_prefix": "archived_folders/forecasting/data/xgboost/input/",
            "output_prefix": "archived_folders/forecasting/data/xgboost/output/",
            "model_prefix": "xgboost/",
            "split_date": "2025-06-24",
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
            "xgboost_params": {
                "n_estimators": [150, 200, 300],
                "learning_rate": [0.03, 0.05, 0.1, 0.2],
                "max_depth": [4, 5, 6, 7]
            }
        }

class ModelConfigGenerator:
    """Generates configuration files for models"""
   
    def __init__(self, path_manager, current_date):
        self.path_manager = path_manager
        self.current_date = current_date
       
    def generate_model_config(self, profiles):
        """Generate model configuration with file paths"""
        config = {
            "test_files": {},
            "model_files": {}
        }
       
        # Generate test file paths
        for profile in profiles:
            suffix = "_r" if profile == "RN" else ""
            test_file = f"df_{profile}_test_{self.current_date}{suffix}.csv"
            config["test_files"][profile] = os.path.join(
                self.path_manager.input_path, test_file
            )
       
        # Generate model file paths
        for profile in profiles:
            model_file = f"{profile}_best_xgboost_{self.current_date}.pkl"
            config["model_files"][profile] = os.path.join(
                self.path_manager.model_path, model_file
            )
       
        return config
   
    def save_model_config(self, config, filename="model_config.json"):
        """Save model configuration"""
        config_file = os.path.join(self.path_manager.output_path, filename)
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
       
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
       
        logger.info(f"Saved model config: {config_file}")
        return config_file

class DataValidator:
    """Validates data integrity for Energy Forecasting"""
   
    def __init__(self):
        self.required_columns = {
            'load_data': ['TradeDate', 'TradeTime', 'LoadProfile', 'RateGroup',
                         'LossAdjustedLoad', 'MeterCount', 'Submission'],
            'temperature_data': ['DATE', 'HourlyDryBulbTemperature'],
            'radiation_data': ['time', 'shortwave_radiation'],
            'processed_data': ['Time', 'Profile', 'Load', 'Count', 'Load_I',
                              'Count_I', 'Year', 'Month', 'Day', 'Hour',
                              'Weekday', 'Season', 'Holiday', 'Workday',
                              'TradeDate', 'Temperature']
        }
   
    def validate_dataframe(self, df, data_type):
        """Validate DataFrame structure and content"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
       
        if data_type not in self.required_columns:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Unknown data type: {data_type}")
            return validation_results
       
        required_cols = self.required_columns[data_type]
       
        # Check required columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing columns: {missing_cols}")
       
        # Check for empty DataFrame
        if df.empty:
            validation_results['valid'] = False
            validation_results['errors'].append("DataFrame is empty")
            return validation_results
       
        # Data type specific validations
        if data_type == 'load_data':
            validation_results = self._validate_load_data(df, validation_results)
        elif data_type == 'temperature_data':
            validation_results = self._validate_temperature_data(df, validation_results)
        elif data_type == 'processed_data':
            validation_results = self._validate_processed_data(df, validation_results)
       
        return validation_results
   
    def _validate_load_data(self, df, validation_results):
        """Validate load data specific requirements"""
        # Check for valid submissions
        valid_submissions = ['Final', 'Initial']
        invalid_submissions = set(df['Submission'].unique()) - set(valid_submissions)
        if invalid_submissions:
            validation_results['warnings'].append(
                f"Invalid submission types found: {invalid_submissions}"
            )
       
        # Check for negative load values
        if 'LossAdjustedLoad' in df.columns:
            negative_loads = df[df['LossAdjustedLoad'] < 0]
            if not negative_loads.empty:
                validation_results['warnings'].append(
                    f"Found {len(negative_loads)} negative load values"
                )
       
        # Check meter count consistency
        if 'MeterCount' in df.columns:
            zero_meters = df[df['MeterCount'] == 0]
            if not zero_meters.empty:
                validation_results['warnings'].append(
                    f"Found {len(zero_meters)} records with zero meter count"
                )
       
        return validation_results
   
    def _validate_temperature_data(self, df, validation_results):
        """Validate temperature data specific requirements"""
        if 'HourlyDryBulbTemperature' in df.columns:
            # Check for reasonable temperature range (Fahrenheit)
            temp_col = df['HourlyDryBulbTemperature']
           
            # Convert to numeric, handling 's' characters
            temp_numeric = pd.to_numeric(
                temp_col.astype(str).str.replace('s', ''),
                errors='coerce'
            )
           
            # Check temperature range (reasonable for San Diego)
            unreasonable_temps = temp_numeric[(temp_numeric < 20) | (temp_numeric > 120)]
            if not unreasonable_temps.empty:
                validation_results['warnings'].append(
                    f"Found {len(unreasonable_temps)} unreasonable temperature values"
                )
           
            # Check for excessive missing values
            missing_temps = temp_numeric.isna().sum()
            missing_pct = (missing_temps / len(df)) * 100
            if missing_pct > 10:
                validation_results['warnings'].append(
                    f"High percentage of missing temperature values: {missing_pct:.1f}%"
                )
       
        return validation_results
   
    def _validate_processed_data(self, df, validation_results):
        """Validate processed data specific requirements"""
        # Check for missing critical values
        critical_columns = ['Load', 'Temperature', 'Count']
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    validation_results['warnings'].append(
                        f"Missing values in {col}: {missing_count}"
                    )
       
        # Check date continuity
        if 'Time' in df.columns:
            df_sorted = df.sort_values('Time')
            time_diffs = df_sorted['Time'].diff()
            expected_diff = pd.Timedelta(hours=1)
           
            gaps = time_diffs[time_diffs > expected_diff]
            if not gaps.empty:
                validation_results['warnings'].append(
                    f"Found {len(gaps)} time gaps in data"
                )
       
        return validation_results

class PerformanceMonitor:
    """Monitor performance metrics for Energy Forecasting"""
   
    def __init__(self):
        self.metrics = {}
   
    def start_timer(self, operation):
        """Start timing an operation"""
        import time
        self.metrics[operation] = {'start_time': time.time()}
   
    def end_timer(self, operation):
        """End timing an operation"""
        import time
        if operation in self.metrics:
            self.metrics[operation]['end_time'] = time.time()
            self.metrics[operation]['duration'] = (
                self.metrics[operation]['end_time'] -
                self.metrics[operation]['start_time']
            )
            logger.info(f"{operation} completed in {self.metrics[operation]['duration']:.2f} seconds")
   
    def get_performance_summary(self):
        """Get performance summary"""
        summary = {}
        total_time = 0
       
        for operation, data in self.metrics.items():
            if 'duration' in data:
                summary[operation] = {
                    'duration_seconds': data['duration'],
                    'duration_minutes': data['duration'] / 60
                }
                total_time += data['duration']
       
        summary['total_time'] = {
            'duration_seconds': total_time,
            'duration_minutes': total_time / 60
        }
       
        return summary
   
    def save_performance_report(self, output_path):
        """Save performance report"""
        summary = self.get_performance_summary()
        report_file = os.path.join(output_path, 'performance_report.json')
       
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
       
        logger.info(f"Performance report saved: {report_file}")
        return report_file
