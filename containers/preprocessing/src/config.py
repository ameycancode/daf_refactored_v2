"""
Simplified Configuration Management for Energy Forecasting System
Container-ready version with Redshift integration
"""

import os
import gc
import psutil
import json
import boto3
import time
import traceback
from datetime import datetime, timedelta
import pytz
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnergyForecastingConfig:
    """Simplified configuration management for containers with Redshift support"""
   
    def __init__(self, config_file=None):
        self.pacific_tz = pytz.timezone("America/Los_Angeles")
        self.current_date = datetime.now(self.pacific_tz).strftime("%Y%m%d")
       
        # Initialize boto3 clients
        try:
            self.s3_client = boto3.client('s3')
            self.redshift_data_client = boto3.client('redshift-data', region_name='us-west-2')
            self.region = boto3.Session().region_name
            self.account_id = boto3.client('sts').get_caller_identity()['Account']
            logger.info(f"AWS connection successful. Region: {self.region}, Account: {self.account_id}")
        except Exception as e:
            logger.warning(f"AWS connection failed: {str(e)}")
            self.s3_client = None
            self.redshift_data_client = None
            self.region = "us-west-2"
            self.account_id = "123456789012"
       
        # Load configuration
        self.config = self._load_configuration(config_file)
       
        logger.info(f"Configuration initialized for date: {self.current_date}")
   
    def _load_configuration(self, config_file):
        """Load configuration from file or use defaults"""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                logger.info(f"Loaded custom configuration from: {config_file}")
               
                # Merge with defaults
                default_config = self._get_default_config()
                self._deep_merge(default_config, custom_config)
                return default_config
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {str(e)}")
       
        # Use default configuration
        logger.info("Using default configuration")
        return self._get_default_config()
   
    def _deep_merge(self, base_dict, update_dict):
        """Deep merge two dictionaries"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
   
    def _get_default_config(self):
        """Get default configuration matching original implementation with Redshift support"""
        return {
            # S3 Configuration (matches your original structure)
            "s3": {
                "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
                "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
                "raw_data_prefix": "archived_folders/forecasting/data/raw/",
                "processed_data_prefix": "archived_folders/forecasting/data/xgboost/processed/",
                "input_data_prefix": "archived_folders/forecasting/data/xgboost/input/",
                "output_data_prefix": "archived_folders/forecasting/data/xgboost/output/",
                "model_prefix": "xgboost/",
                "train_results_prefix": "archived_folders/forecasting/data/xgboost/train_results/"
            },
            
            # Redshift Configuration
            "redshift": {
                "database": "sdcp",
                "cluster_identifier": "sdcp-edp-backend-dev",
                "db_user": "ds_service_user",
                "region": "us-west-2",
                "schema": "edp_cust_dev",
                "table": "caiso_sqmd",
                "query_timeout_seconds": 1800,
                "use_redshift": True,  # Toggle between Redshift and CSV
                "data_reading_period_days": None # All data; 0.3 * 365  # ~109 days, configurable
            },
           
            # Data Processing Configuration (from your original code)
            "data_processing": {
                "split_date": "2025-06-24",
                "profile_start_dates": {
                    "df_RNN": "2022-03-01",
                    "df_RN": "2022-03-01",
                    "df_M": "2021-07-01",
                    "df_S": "2021-07-01",
                    "df_AGR": "2023-05-01",
                    "df_L": "2021-07-01",
                    "df_A6": "2021-07-10"
                },
                "lag_features": {
                    "load_i_lag_days": 14,
                    "load_lag_days": 70
                },
                "profile_mappings": {
                    "RES_Non_NEM": "df_RNN",
                    "RES_NEM": "df_RN",
                    "MEDCI": "df_M",
                    "SMLCOM": "df_S",
                    "AGR": "df_AGR",
                    "LIGHT": "df_L",
                    "A6": "df_A6"
                },
                "holidays": [
                    "2021-01-01", "2021-02-15", "2021-05-31", "2021-07-05", "2021-09-06",
                    "2021-11-11", "2021-11-25", "2021-12-25",
                    "2022-01-01", "2022-02-21", "2022-05-30", "2022-07-04", "2022-09-05",
                    "2022-11-11", "2022-11-24", "2022-12-26",
                    "2023-01-02", "2023-02-20", "2023-05-29", "2023-07-04", "2023-09-04",
                    "2023-11-11", "2023-11-23", "2023-12-25",
                    "2024-01-01", "2024-02-19", "2024-05-27", "2024-07-04", "2024-09-02",
                    "2024-11-11", "2024-11-28", "2024-12-25",
                    "2025-01-01", "2025-02-17", "2025-05-26", "2025-07-04", "2025-09-01",
                    "2025-11-11", "2025-11-27", "2025-12-25"
                ]
            },
           
            # Training Configuration (from your original code)
            "training": {
                "train_cutoff": "2025-05-24",
                "cv_splits": 10,
                "xgboost_params": {
                    "n_estimators": [150, 200, 300],
                    "learning_rate": [0.03, 0.05, 0.1, 0.2],
                    "max_depth": [4, 5, 6, 7]
                },
                "performance_threshold": None,
                "random_state": 42
            },
           
            # API Configuration for Weather/Radiation
            "apis": {
                "weather": {
                    "base_url": "https://api.weather.gov",
                    "station": "KSAN",  # San Diego
                    "location": {
                        "latitude": 32.7157,
                        "longitude": -117.1611
                    }
                },
                "radiation": {
                    "base_url": "https://api.open-meteo.com/v1/forecast",
                    "location": {
                        "latitude": 32.7157,
                        "longitude": -117.1611
                    }
                }
            },
           
            # File naming patterns (from your original code)
            "file_patterns": {
                "raw_files": {
                    "load_data": "SQMD.csv",
                    "temperature_data": "Temperature.csv",
                    "radiation_data": "Radiation.csv"
                },
                "processed_files": {
                    "profile_lagged": "{profile}_lagged_{date}.csv",
                    "profile_train": "{profile}_train_{date}.csv",
                    "profile_test": "{profile}_test_{date}{suffix}.csv"
                },
                "model_files": {
                    "xgboost_model": "{profile}_best_xgboost_{date}.pkl"
                },
                "prediction_files": {
                    "profile_predictions": "{profile}_predictions_{date}.csv",
                    "combined_load": "Combined_Load_{date}.csv",
                    "aggregated_load": "Aggregated_Load_{date}.csv",
                    "weather_forecast": "T_{date}.csv",
                    "radiation_forecast": "shortwave_radiation_{date}.csv"
                }
            },
           
            # Container paths (SageMaker specific)
            "container_paths": {
                "input_path": "/opt/ml/processing/input",
                "output_path": "/opt/ml/processing/output",
                "model_path": "/opt/ml/processing/models",
                "code_path": "/opt/ml/processing/code",
                "config_path": "/opt/ml/processing/config"
            }
        }
   
    # Redshift Configuration getters
    def get_redshift_config(self):
        """Get Redshift configuration"""
        return self.config["redshift"]
    
    def is_redshift_enabled(self):
        """Check if Redshift is enabled"""
        return self.config["redshift"].get("use_redshift", True)
    
    def get_data_reading_period_days(self):
        """Get data reading period in days"""
        return self.config["redshift"].get("data_reading_period_days", None)
   
    # S3 Path Generators (unchanged from original)
    def get_s3_path(self, path_type, **kwargs):
        """Generate S3 paths based on configuration"""
        bucket = self.config["s3"]["data_bucket"]
       
        if path_type == "raw_data":
            return f"s3://{bucket}/{self.config['s3']['raw_data_prefix']}"
        elif path_type == "processed_data":
            return f"s3://{bucket}/{self.config['s3']['processed_data_prefix']}"
        elif path_type == "input_data":
            return f"s3://{bucket}/{self.config['s3']['input_data_prefix']}"
        elif path_type == "output_data":
            return f"s3://{bucket}/{self.config['s3']['output_data_prefix']}"
        elif path_type == "train_results":
            return f"s3://{bucket}/{self.config['s3']['train_results_prefix']}"
        elif path_type == "models":
            bucket = self.config["s3"]["model_bucket"]
            return f"s3://{bucket}/{self.config['s3']['model_prefix']}"
        elif path_type == "temperature_input":
            return f"s3://{bucket}/{self.config['s3']['input_data_prefix']}temperature/"
        elif path_type == "radiation_input":
            return f"s3://{bucket}/{self.config['s3']['input_data_prefix']}radiation/"
        else:
            raise ValueError(f"Unknown path type: {path_type}")
   
    def get_file_path(self, file_type, **kwargs):
        """Generate file paths based on configuration patterns"""
        patterns = self.config["file_patterns"]
       
        # Use current date if not provided
        date = kwargs.get('date', self.current_date)
        profile = kwargs.get('profile', '')
        suffix = kwargs.get('suffix', '')
       
        if file_type in patterns["raw_files"]:
            return patterns["raw_files"][file_type]
        elif file_type in patterns["processed_files"]:
            return patterns["processed_files"][file_type].format(
                profile=profile, date=date, suffix=suffix
            )
        elif file_type in patterns["model_files"]:
            return patterns["model_files"][file_type].format(
                profile=profile, date=date
            )
        elif file_type in patterns["prediction_files"]:
            return patterns["prediction_files"][file_type].format(
                profile=profile, date=date
            )
        else:
            raise ValueError(f"Unknown file type: {file_type}")
   
    def get_full_s3_key(self, path_type, file_type, **kwargs):
        """Get complete S3 key combining path and filename"""
        s3_path = self.get_s3_path(path_type, **kwargs)
        filename = self.get_file_path(file_type, **kwargs)
       
        # Remove s3:// and bucket name to get just the key
        s3_key = s3_path.split('/', 3)[-1] + filename
        return s3_key
   
    # Configuration getters (unchanged from original)
    def get_profiles(self):
        """Get all profile codes"""
        return list(self.config["data_processing"]["profile_mappings"].values())
   
    def get_profile_start_date(self, profile):
        """Get start date for a specific profile"""
        return self.config["data_processing"]["profile_start_dates"].get(profile)
   
    def get_training_config(self):
        """Get training configuration"""
        return self.config["training"]
   
    def get_api_config(self, api_name):
        """Get API configuration"""
        return self.config["apis"].get(api_name, {})
   
    def get_container_paths(self):
        """Get container path configuration"""
        return self.config["container_paths"]
   
    def get_data_processing_config(self):
        """Get data processing configuration"""
        return self.config["data_processing"]
   
    # S3 bucket getters (unchanged from original)
    @property
    def data_bucket(self):
        return self.config["s3"]["data_bucket"]
   
    @property
    def model_bucket(self):
        return self.config["s3"]["model_bucket"]
   
    @property
    def current_date_str(self):
        return self.current_date
   
    # Save configuration
    def save_config(self, filepath):
        """Save current configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {filepath}: {str(e)}")
            return False


class RedshiftDataManager:
    """Redshift data operations using Data API"""
    
    def __init__(self, config: EnergyForecastingConfig):
        self.config = config
        self.redshift_config = config.get_redshift_config()
        self.redshift_client = config.redshift_data_client
        
    def execute_query(self, query, query_limit=None):
        """Execute Redshift query using Data API with pagination support"""
        try:
            if not self.redshift_client:
                raise Exception("Redshift client not available")
                
            if query_limit and query_limit > 0:
                query += f" LIMIT {query_limit}"
                
            logger.info(f"Executing query via Data API on cluster: {self.redshift_config['cluster_identifier']}")
            logger.info(f"Database: {self.redshift_config['database']}, User: {self.redshift_config['db_user']}")
            logger.info(f"Query: {query}")
            
            # Execute the query
            response = self.redshift_client.execute_statement(
                ClusterIdentifier=self.redshift_config['cluster_identifier'],
                Database=self.redshift_config['database'],
                DbUser=self.redshift_config['db_user'],
                Sql=query
            )
            
            query_id = response['Id']
            logger.info(f"Query submitted with ID: {query_id}")
            
            # Wait for completion
            self._wait_for_completion(query_id)
            
            # Get all results with pagination
            df = self._get_paginated_results(query_id)
            
            logger.info(f"Query completed successfully. Retrieved {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error executing query via Data API: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _wait_for_completion(self, query_id):
        """Wait for query completion"""
        max_wait = self.redshift_config.get('query_timeout_seconds', 1800)
        waited = 0
        
        logger.info(f"Waiting for query {query_id} to complete...")
        
        while waited < max_wait:
            try:
                status_response = self.redshift_client.describe_statement(Id=query_id)
                status = status_response['Status']
                
                if status == 'FINISHED':
                    logger.info(f"Query {query_id} completed successfully")
                    return
                elif status == 'FAILED':
                    error_msg = status_response.get('Error', 'Unknown error')
                    logger.error(f"Query {query_id} failed: {error_msg}")
                    raise Exception(f'Query failed: {error_msg}')
                elif status == 'ABORTED':
                    logger.error(f"Query {query_id} was aborted")
                    raise Exception(f'Query was aborted')
                
                # Still running
                if waited % 60 == 0 and waited > 0:  # Log every minute
                    logger.info(f"Query still running... waited {waited}s (status: {status})")
                
                time.sleep(10)
                waited += 10
                
            except Exception as e:
                if 'failed:' in str(e) or 'aborted' in str(e):
                    raise
                else:
                    logger.warning(f"Error checking query status: {str(e)}")
                    time.sleep(10)
                    waited += 10
                    continue
        
        raise Exception(f'Query timed out after {max_wait} seconds')
    
    def _get_paginated_results(self, query_id):
        """Get all results with proper pagination"""
        import pandas as pd
        
        all_records = []
        column_metadata = None
        next_token = None
        page_count = 0
        
        try:
            while True:
                page_count += 1
                logger.info(f"Fetching results page {page_count}...")
                
                # Prepare request parameters
                request_params = {'Id': query_id}
                if next_token:
                    request_params['NextToken'] = next_token
                
                # Get results page
                result_response = self.redshift_client.get_statement_result(**request_params)
                
                # Get column metadata from first page only
                if column_metadata is None:
                    column_metadata = result_response.get('ColumnMetadata', [])
                    logger.info(f"Query has {len(column_metadata)} columns")
                
                # Get records from this page
                page_records = result_response.get('Records', [])
                all_records.extend(page_records)
                
                logger.info(f"Page {page_count}: Retrieved {len(page_records)} records (Total: {len(all_records)})")
                
                # Check if there are more pages
                next_token = result_response.get('NextToken')
                if not next_token:
                    logger.info(f"Pagination complete. Total pages: {page_count}, Total records: {len(all_records)}")
                    break
            
            # Convert to DataFrame
            df = self._convert_to_dataframe(column_metadata, all_records)
            return df
            
        except Exception as e:
            logger.error(f"Error in paginated result retrieval: {str(e)}")
            raise
    
    def _convert_to_dataframe(self, column_metadata, all_records):
        """Convert Redshift Data API results to DataFrame"""
        import pandas as pd
        
        try:
            # Get column names
            column_names = [col['name'] for col in column_metadata]
            
            logger.info(f"Converting {len(all_records)} records with {len(column_names)} columns")
            
            if not all_records:
                return pd.DataFrame(columns=column_names)
            
            # Convert records to list of lists
            data_rows = []
            for record in all_records:
                row = []
                for field in record:
                    # Extract value based on type
                    if 'stringValue' in field:
                        row.append(field['stringValue'])
                    elif 'longValue' in field:
                        row.append(field['longValue'])
                    elif 'doubleValue' in field:
                        row.append(field['doubleValue'])
                    elif 'booleanValue' in field:
                        row.append(field['booleanValue'])
                    elif 'isNull' in field and field['isNull']:
                        row.append(None)
                    else:
                        row.append(str(field))  # Fallback
                data_rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=column_names)
            
            logger.info(f"DataFrame created: {len(df)} rows, {len(column_names)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error converting results to DataFrame: {str(e)}")
            raise
    
    def query_sqmd_data(self, current_date=None, query_limit=None):
        """Query SQMD data from Redshift with time filtering"""
        try:
            if current_date is None:
                current_date = datetime.now()
            
            schema_name = self.redshift_config['schema']
            table_name = self.redshift_config['table']
            
            # Calculate time range based on data reading period
            data_period_days = self.config.get_data_reading_period_days()
            
            if data_period_days:
                start_date = current_date - timedelta(days=data_period_days)
                logger.info(f"Filtering data from {start_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
                
                query = f"""
                SELECT
                    tradedatelocal as tradedate,
                    tradehourstartlocal as tradetime,
                    loadprofile, rategroup, baseload, lossadjustedload, metercount,
                    loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
                    submission, createddate as created
                FROM {schema_name}.{table_name}
                WHERE tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'
                ORDER BY tradedatelocal, tradehourstartlocal
                """
            else:
                logger.info("No time limit set - fetching all available data")
                query = f"""
                SELECT
                    tradedatelocal as tradedate,
                    tradehourstartlocal as tradetime,
                    loadprofile, rategroup, baseload, lossadjustedload, metercount,
                    loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
                    submission, createddate as created
                FROM {schema_name}.{table_name}
                ORDER BY tradedatelocal, tradehourstartlocal
                """
            
            logger.info(f"Executing SQMD data query")
            df = self.execute_query(query, query_limit)
            
            logger.info(f"Retrieved {len(df)} rows of SQMD data from Redshift")
            return df
            
        except Exception as e:
            logger.error(f"Error querying SQMD data: {e}")
            logger.error(traceback.format_exc())
            raise

class MemoryOptimizedRedshiftDataManager(RedshiftDataManager):
    """Memory-optimized version for large datasets"""
    
    def __init__(self, config: EnergyForecastingConfig):
        super().__init__(config)
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 50000))
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
    def query_sqmd_data_chunked(self, current_date=None, chunk_size=None):
        """Query SQMD data in chunks to manage memory"""
        try:
            if chunk_size is None:
                chunk_size = self.chunk_size
                
            logger.info(f"Starting chunked query with chunk size: {chunk_size:,}")
            
            # Get total row count first
            count_query = self._build_count_query(current_date)
            total_rows = self._get_total_row_count(count_query)
            
            if total_rows == 0:
                raise ValueError("No SQMD data available for the specified period")
                
            logger.info(f"Total rows to process: {total_rows:,}")
            logger.info(f"Count Query to process: {total_rows:,}")
            
            # Calculate number of chunks needed
            num_chunks = (total_rows + chunk_size - 1) // chunk_size
            logger.info(f"Will process data in {num_chunks} chunks")
            
            # Process data in chunks
            all_chunks = []
            for chunk_num in range(num_chunks):
                offset = chunk_num * chunk_size
                
                logger.info(f"Processing chunk {chunk_num + 1}/{num_chunks} (rows {offset:,} to {min(offset + chunk_size, total_rows):,})")
                
                # Check memory before processing each chunk
                self._check_memory_usage()
                
                chunk_query = self._build_chunk_query(current_date, chunk_size, offset)
                # if chunk_num < 2:
                #     logger.info(f"Chunk Query to process: {chunk_query}")
                chunk_df = self.execute_query(chunk_query)
                
                if not chunk_df.empty:
                    all_chunks.append(chunk_df)
                    logger.info(f"Chunk {chunk_num + 1} processed: {len(chunk_df):,} rows")
                else:
                    logger.warning(f"Chunk {chunk_num + 1} returned no data")
                
                # Force garbage collection after each chunk
                gc.collect()
            
            if not all_chunks:
                raise ValueError("No data retrieved from any chunks")
            
            # Combine all chunks
            logger.info("Combining all chunks...")
            df_combined = pd.concat(all_chunks, ignore_index=True)

            logger.info(f"Shape of combined dataframe: {df_combined.shape}")
            
            # Clear chunk data from memory
            del all_chunks
            gc.collect()
            
            logger.info(f"Successfully combined all chunks: {len(df_combined):,} total rows")
            return df_combined
            
        except Exception as e:
            logger.error(f"Error in chunked query: {str(e)}")
            # Clean up memory on error
            if 'all_chunks' in locals():
                del all_chunks
            gc.collect()
            raise
    
    def _build_count_query(self, current_date=None):
        """Build query to get total row count"""
        schema_name = self.redshift_config['schema']
        table_name = self.redshift_config['table']
        
        if current_date is None:
            current_date = datetime.now()
        
        data_period_days = self.config.get_data_reading_period_days()
        
        if data_period_days:
            start_date = current_date - timedelta(days=data_period_days)
            where_clause = f"WHERE tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'"
        else:
            where_clause = ""
        
        return f"SELECT COUNT(*) as total_rows FROM {schema_name}.{table_name} {where_clause}"
    
    def _get_total_row_count(self, count_query):
        """Get total row count"""
        try:
            result_df = self.execute_query(count_query)
            total_rows = int(result_df.iloc[0, 0])
            return total_rows
        except Exception as e:
            logger.error(f"Error getting row count: {str(e)}")
            # Fallback: assume large dataset and use chunking anyway
            return 1000000  # Default assumption
    
    def _build_chunk_query(self, current_date=None, limit=50000, offset=0):
        """Build query for a specific chunk"""
        schema_name = self.redshift_config['schema']
        table_name = self.redshift_config['table']
        
        if current_date is None:
            current_date = datetime.now()
        
        data_period_days = self.config.get_data_reading_period_days()
        
        if data_period_days:
            start_date = current_date - timedelta(days=data_period_days)
            where_clause = f"WHERE tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'"
        else:
            where_clause = ""
        
        return f"""
        SELECT
            tradedatelocal as tradedate,
            tradehourstartlocal as tradetime,
            loadprofile, rategroup, baseload, lossadjustedload, metercount,
            loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
            submission, createddate as created
        FROM {schema_name}.{table_name}
        {where_clause}
        ORDER BY tradedatelocal, tradehourstartlocal, loadprofile
        LIMIT {limit} OFFSET {offset}
        """
    
    def _check_memory_usage(self):
        """Check current memory usage and warn if high"""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            available_gb = psutil.virtual_memory().available / (1024**3)
            
            logger.info(f"Memory usage: {memory_percent:.1%}, Available: {available_gb:.1f} GB")
            
            if memory_percent > self.memory_threshold:
                logger.warning(f"High memory usage detected: {memory_percent:.1%}")
                logger.warning("Consider reducing chunk size or upgrading instance type")
                
                # Force aggressive garbage collection
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Could not check memory usage: {str(e)}")


class MemoryOptimizedEnergyForecastingConfig(EnergyForecastingConfig):
    """Memory-optimized configuration with environment variable support"""
    
    def __init__(self, config_file=None):
        super().__init__(config_file)
        
        # Override with environment variables if available
        self._apply_environment_overrides()
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides for container optimization"""
        
        # # Data reading period override
        # env_data_period = os.getenv('DATA_READING_PERIOD_DAYS')
        # if env_data_period:
        #     try:
        #         self.config['redshift']['data_reading_period_days'] = float(env_data_period)
        #         logger.info(f"Override: data_reading_period_days = {env_data_period}")
        #     except ValueError:
        #         logger.warning(f"Invalid DATA_READING_PERIOD_DAYS: {env_data_period}")
        
        # # Query limit override
        # env_query_limit = os.getenv('QUERY_LIMIT')
        # if env_query_limit:
        #     try:
        #         self.config['redshift']['query_limit'] = int(env_query_limit)
        #         logger.info(f"Override: query_limit = {env_query_limit}")
        #     except ValueError:
        #         logger.warning(f"Invalid QUERY_LIMIT: {env_query_limit}")
        
        # Chunk size for memory optimization
        env_chunk_size = os.getenv('CHUNK_SIZE')
        if env_chunk_size:
            try:
                self.config['redshift']['chunk_size'] = int(env_chunk_size)
                logger.info(f"Override: chunk_size = {env_chunk_size}")
            except ValueError:
                logger.warning(f"Invalid CHUNK_SIZE: {env_chunk_size}")
        
        # Memory optimization mode
        if os.getenv('MEMORY_OPTIMIZATION') == '1':
            logger.info("Memory optimization mode enabled")
            self.config['memory_optimization'] = {
                'enabled': True,
                'chunk_processing': True,
                'garbage_collection': True,
                'memory_monitoring': True
            }
    
    def get_query_limit(self):
        """Get query limit for large datasets"""
        return self.config['redshift'].get('query_limit', None)
    
    def get_chunk_size(self):
        """Get chunk size for memory optimization"""
        return self.config['redshift'].get('chunk_size', 50000)
    
    def is_memory_optimization_enabled(self):
        """Check if memory optimization is enabled"""
        return self.config.get('memory_optimization', {}).get('enabled', False)


class S3FileManager:
    """S3 file manager using configuration (unchanged from original)"""
   
    def __init__(self, config: EnergyForecastingConfig):
        self.config = config
        self.s3_client = config.s3_client
   
    def upload_file(self, local_path, s3_key, bucket=None):
        """Upload file to S3"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return False
           
        bucket = bucket or self.config.data_bucket
        try:
            self.s3_client.upload_file(local_path, bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {str(e)}")
            return False
   
    def upload_dataframe(self, df, s3_key, bucket=None):
        """Upload DataFrame as CSV to S3"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return False
           
        import pandas as pd
        from io import StringIO
       
        bucket = bucket or self.config.data_bucket
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
   
    def download_file(self, s3_key, local_path, bucket=None):
        """Download file from S3"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return False
           
        bucket = bucket or self.config.data_bucket
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download s3://{bucket}/{s3_key}: {str(e)}")
            return False
   
    def save_and_upload_dataframe(self, df, local_path, s3_key, bucket=None):
        """Save DataFrame locally and upload to S3"""
        bucket = bucket or self.config.data_bucket
       
        # Save locally first
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            df.to_csv(local_path, index=False)
            logger.info(f"Saved DataFrame locally: {local_path}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame locally: {str(e)}")
            return False
       
        # Upload to S3
        return self.upload_file(local_path, s3_key, bucket)
   
    def save_and_upload_file(self, content, local_path, s3_key, bucket=None):
        """Save content to local file and upload to S3"""
        bucket = bucket or self.config.data_bucket
       
        # Save locally first
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w') as f:
                if isinstance(content, dict):
                    json.dump(content, f, indent=2)
                else:
                    f.write(content)
            logger.info(f"Saved file locally: {local_path}")
        except Exception as e:
            logger.error(f"Failed to save file locally: {str(e)}")
            return False
       
        # Upload to S3
        return self.upload_file(local_path, s3_key, bucket)
