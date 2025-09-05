"""
Energy Forecasting Profile Predictor Lambda Function - Refactored Version
Integrates standalone code logic with Redshift input/output and environment awareness
Maintains single profile processing within Step Functions Map state architecture
"""

import json
import boto3
import logging
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pytz
from io import StringIO

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_runtime_client = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')
redshift_data_client = boto3.client('redshift-data')

class EnvironmentConfig:
    """Environment-aware configuration management"""
    
    def __init__(self, environment: str = None):
        self.environment = environment or os.environ.get('ENVIRONMENT', 'dev')
        self.region = os.environ.get('AWS_REGION', 'us-west-2')
        
    def get_redshift_config(self) -> Dict[str, str]:
        """Get environment-specific Redshift configuration"""
        
        # Environment-specific cluster naming
        if self.environment == 'prod':
            cluster_identifier = "sdcp-edp-backend-prod"
            input_schema = "edp_cust"
            output_schema = "edp_forecasting"
        else:
            cluster_identifier = f"sdcp-edp-backend-{self.environment}"
            input_schema = f"edp_cust_{self.environment}"
            output_schema = f"edp_forecasting_{self.environment}"
        
        return {
            "cluster_identifier": cluster_identifier,
            "database": "sdcp",
            "db_user": "ds_service_user",
            "region": self.region,
            "input_schema": input_schema,
            "input_table": "caiso_sqmd",
            "output_schema": output_schema,
            "output_table": "dayahead_load_forecasts_sdcp"
        }
    
    def get_s3_config(self) -> Dict[str, str]:
        """Get environment-specific S3 configuration"""
        return {
            "data_bucket": f"sdcp-{self.environment}-sagemaker-energy-forecasting-data",
            "model_bucket": f"sdcp-{self.environment}-sagemaker-energy-forecasting-models",
            "input_prefix": "archived_folders/forecasting/data/xgboost/input/",
            "output_prefix": "archived_folders/forecasting/data/xgboost/output/"
        }

class RedshiftDataProcessor:
    """Data processor using standalone logic with Redshift integration"""
    
    def __init__(self, environment_config: EnvironmentConfig):
        self.config = environment_config
        self.redshift_config = environment_config.get_redshift_config()
        self.s3_config = environment_config.get_s3_config()
        
    def query_profile_data(self, profile: str, days_back: int = 100) -> pd.DataFrame:
        """Query data for single profile from Redshift using standalone logic"""
        
        try:
            logger.info(f"Querying Redshift data for profile {profile}")
            
            # Calculate date range (from standalone data_processor.py logic)
            current_date = datetime.now(pytz.timezone('America/Los_Angeles'))
            start_date = current_date - timedelta(days=days_back)
            
            # Build query with profile filtering (adapted from standalone)
            query = f"""
            SELECT
                tradedatelocal as tradedate,
                tradehourstartlocal as tradetime,
                loadprofile, rategroup, baseload, lossadjustedload, metercount,
                loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
                submission, createddate as created
            FROM {self.redshift_config['input_schema']}.{self.redshift_config['input_table']}
            WHERE tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'
            AND loadprofile = '{profile}'
            ORDER BY tradedatelocal, tradehourstartlocal
            """
            
            # Execute query using Data API
            response = redshift_data_client.execute_statement(
                ClusterIdentifier=self.redshift_config['cluster_identifier'],
                Database=self.redshift_config['database'],
                DbUser=self.redshift_config['db_user'],
                Sql=query
            )
            
            query_id = response['Id']
            logger.info(f"Redshift query submitted: {query_id}")
            
            # Wait for completion
            self._wait_for_query_completion(query_id)
            
            # Get paginated results
            df = self._get_all_paginated_results(query_id)
            
            logger.info(f"Retrieved {len(df)} rows for profile {profile}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to query profile data: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame, profile: str, 
                       weather_df: pd.DataFrame, radiation_df: pd.DataFrame = None) -> pd.DataFrame:
        """Complete preprocessing pipeline using standalone logic"""
        
        try:
            logger.info(f"Starting preprocessing for profile {profile}")
            
            if df.empty:
                logger.warning(f"No data for profile {profile}")
                return pd.DataFrame()
            
            # Step 1: Process datetime (from standalone data_processor.py)
            df['TradeDateTime'] = pd.to_datetime(df['tradedate'] + ' ' + df['tradetime'].astype(str) + ':00:00')
            
            # Step 2: Process to hourly data (from standalone logic)
            df_hourly = self._convert_to_hourly_data(df, profile)
            
            # Step 3: Merge with weather data (from standalone logic)
            df_merged = self._merge_with_weather(df_hourly, weather_df)
            
            # Step 4: Add radiation for RN profile (from standalone logic)
            if profile == 'RN' and radiation_df is not None:
                df_merged = self._add_radiation_data(df_merged, radiation_df)
            
            # Step 5: Feature engineering and lag variables (from standalone logic)
            df_featured = self._create_features_and_lags(df_merged, profile)
            
            # Step 6: Count replacement (from standalone logic)
            df_final = self._replace_count_data(df_featured, profile)
            
            logger.info(f"Preprocessing completed for {profile}: {len(df_final)} rows")
            return df_final
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {profile}: {str(e)}")
            raise

    def _wait_for_query_completion(self, query_id: str, max_wait: int = 300):
        """Wait for Redshift query completion"""
        waited = 0
        while waited < max_wait:
            try:
                status_response = redshift_data_client.describe_statement(Id=query_id)
                status = status_response['Status']
                
                if status == 'FINISHED':
                    logger.info(f"Query {query_id} completed successfully")
                    return
                elif status == 'FAILED':
                    error_msg = status_response.get('Error', 'Unknown error')
                    raise Exception(f'Query failed: {error_msg}')
                elif status == 'ABORTED':
                    raise Exception(f'Query was aborted')
                
                time.sleep(10)
                waited += 10
                
            except Exception as e:
                if 'failed:' in str(e) or 'aborted' in str(e):
                    raise
                time.sleep(10)
                waited += 10
                continue
        
        raise Exception(f'Query timed out after {max_wait} seconds')

    def _get_all_paginated_results(self, query_id: str) -> pd.DataFrame:
        """Get all paginated results from Redshift query"""
        all_records = []
        column_metadata = None
        next_token = None
        
        while True:
            try:
                # Get results with pagination
                if next_token:
                    response = redshift_data_client.get_statement_result(
                        Id=query_id, NextToken=next_token
                    )
                else:
                    response = redshift_data_client.get_statement_result(Id=query_id)
                
                # Store column metadata from first page
                if column_metadata is None:
                    column_metadata = response.get('ColumnMetadata', [])
                
                # Add records from this page
                records = response.get('Records', [])
                all_records.extend(records)
                
                # Check for next page
                next_token = response.get('NextToken')
                if not next_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching results: {str(e)}")
                raise
        
        # Convert to DataFrame
        if not all_records or not column_metadata:
            return pd.DataFrame()
        
        # Extract column names
        column_names = [col['name'] for col in column_metadata]
        
        # Extract data rows
        data_rows = []
        for record in all_records:
            row = []
            for field in record:
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
                    row.append(str(field))
            data_rows.append(row)
        
        return pd.DataFrame(data_rows, columns=column_names)

    def _convert_to_hourly_data(self, df: pd.DataFrame, profile: str) -> pd.DataFrame:
        """Convert raw data to hourly format (from standalone logic)"""
        
        # Group by hour and aggregate (from standalone data_processor.py)
        df['Hour'] = df['TradeDateTime'].dt.hour
        df['Date'] = df['TradeDateTime'].dt.date
        df['Year'] = df['TradeDateTime'].dt.year
        df['Month'] = df['TradeDateTime'].dt.month
        df['Day'] = df['TradeDateTime'].dt.day
        df['Weekday'] = df['TradeDateTime'].dt.weekday
        
        # Aggregate by hour (using logic from standalone)
        hourly_df = df.groupby(['Year', 'Month', 'Day', 'Hour']).agg({
            'loadlal': 'sum',  # Loss adjusted load
            'loadmetercount': 'sum',  # Meter count
            'TradeDateTime': 'first'
        }).reset_index()
        
        # Calculate load per meter
        hourly_df['Load_I'] = hourly_df['loadlal'] / hourly_df['loadmetercount']
        hourly_df['Load_I'] = hourly_df['Load_I'].fillna(0)
        
        # Add temporal features (from standalone logic)
        hourly_df['Season'] = hourly_df['Month'].apply(self._get_season)
        hourly_df['Holiday'] = hourly_df['TradeDateTime'].apply(self._is_holiday)
        hourly_df['Workday'] = ((hourly_df['Weekday'] < 5) & (~hourly_df['Holiday'])).astype(int)
        
        return hourly_df

    def _merge_with_weather(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge with weather data (from standalone logic)"""
        
        # Prepare weather data for merging
        weather_df['TradeDateTime'] = pd.to_datetime(weather_df['TradeDateTime'])
        weather_df['Year'] = weather_df['TradeDateTime'].dt.year
        weather_df['Month'] = weather_df['TradeDateTime'].dt.month
        weather_df['Day'] = weather_df['TradeDateTime'].dt.day
        weather_df['Hour'] = weather_df['TradeDateTime'].dt.hour
        
        # Merge on datetime components
        merged_df = df.merge(
            weather_df[['Year', 'Month', 'Day', 'Hour', 'Temperature']],
            on=['Year', 'Month', 'Day', 'Hour'],
            how='left'
        )
        
        # Fill missing temperatures with mean
        merged_df['Temperature'] = merged_df['Temperature'].fillna(merged_df['Temperature'].mean())
        
        return merged_df

    def _add_radiation_data(self, df: pd.DataFrame, radiation_df: pd.DataFrame) -> pd.DataFrame:
        """Add radiation data for RN profile (from standalone logic)"""
        
        # Prepare radiation data
        radiation_df['date'] = pd.to_datetime(radiation_df['date'])
        radiation_df['Year'] = radiation_df['date'].dt.year
        radiation_df['Month'] = radiation_df['date'].dt.month
        radiation_df['Day'] = radiation_df['date'].dt.day
        radiation_df['Hour'] = radiation_df['date'].dt.hour
        
        # Merge radiation data
        merged_df = df.merge(
            radiation_df[['Year', 'Month', 'Day', 'Hour', 'shortwave_radiation']],
            on=['Year', 'Month', 'Day', 'Hour'],
            how='left'
        )
        
        # Fill missing radiation with 0
        merged_df['shortwave_radiation'] = merged_df['shortwave_radiation'].fillna(0)
        
        return merged_df

    def _create_features_and_lags(self, df: pd.DataFrame, profile: str) -> pd.DataFrame:
        """Create lag features (from standalone logic)"""
        
        # Sort by datetime for lag calculation
        df = df.sort_values(['Year', 'Month', 'Day', 'Hour']).reset_index(drop=True)
        
        # Create lag features (from standalone create_lagged_profiles_for_prediction)
        df['Load_I_lag_14_days'] = df['Load_I'].shift(24 * 14)  # 14 days back
        df['Load_lag_70_days'] = df['Load_I'].shift(24 * 70)    # 70 days back
        
        # Fill lag NaNs with median values
        df['Load_I_lag_14_days'] = df['Load_I_lag_14_days'].fillna(df['Load_I'].median())
        df['Load_lag_70_days'] = df['Load_lag_70_days'].fillna(df['Load_I'].median())
        
        return df

    def _replace_count_data(self, df: pd.DataFrame, profile: str) -> pd.DataFrame:
        """Replace count data (from standalone logic)"""
        
        # Simple count replacement strategy (from standalone replace_count_i)
        median_count = df['loadmetercount'].median()
        df['Count'] = df['loadmetercount'].fillna(median_count)
        
        # Ensure positive counts
        df['Count'] = df['Count'].clip(lower=1)
        
        return df

    def _get_season(self, month: int) -> int:
        """Get season from month (from standalone logic)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    def _is_holiday(self, date: pd.Timestamp) -> bool:
        """Simple holiday detection (from standalone logic)"""
        # Simplified holiday detection for common US holidays
        month = date.month
        day = date.day
        
        # Major holidays (simplified)
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas Day
        ]
        
        return (month, day) in holidays

class WeatherDataFetcher:
    """Weather data fetching using enhanced standalone logic"""
    
    def __init__(self, environment_config: EnvironmentConfig):
        self.config = environment_config
        self.s3_config = environment_config.get_s3_config()
        
        # Coordinates for San Diego (from standalone)
        self.latitude = 32.7157
        self.longitude = -117.1611

    def fetch_weather_forecast(self) -> Optional[pd.DataFrame]:
        """Fetch weather forecast (enhanced from standalone weather_forecaster.py)"""
        
        try:
            logger.info("Fetching weather forecast...")
            
            # Try to load existing forecast from S3 first
            existing_forecast = self._load_existing_weather_forecast()
            if existing_forecast is not None:
                logger.info("Using existing weather forecast from S3")
                return existing_forecast
            
            # Fetch new forecast from API (same logic as standalone)
            pacific_tz = pytz.timezone("America/Los_Angeles")
            now = datetime.now(pacific_tz)
            today_str = now.strftime("%Y%m%d")
            tomorrow = now.date() + timedelta(days=1)
            
            # Get forecast grid point
            points_url = f"https://api.weather.gov/points/{self.latitude},{self.longitude}"
            import requests
            points_response = requests.get(points_url, timeout=30)
            points_response.raise_for_status()
            points_data = points_response.json()
            
            # Get hourly forecast
            forecast_hourly_url = points_data['properties']['forecastHourly']
            forecast_response = requests.get(forecast_hourly_url, timeout=30)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            # Parse forecast data
            hourly_forecast = []
            for period in forecast_data['properties']['periods']:
                hourly_forecast.append({
                    'TradeDateTime': pd.to_datetime(period['startTime'], utc=True),
                    'Temperature': period['temperature'],
                })
            
            # Create DataFrame
            weather_df = pd.DataFrame(hourly_forecast)
            weather_df['TradeDateTime'] = weather_df['TradeDateTime'].dt.tz_convert(pacific_tz)
            
            # Filter for tomorrow
            tomorrow_weather = weather_df[
                weather_df['TradeDateTime'].dt.date == tomorrow
            ][['TradeDateTime', 'Temperature']].head(24)
            
            # Save to S3 for caching
            self._save_weather_forecast_to_s3(tomorrow_weather, today_str)
            
            logger.info(f"Weather forecast retrieved: {len(tomorrow_weather)} hours")
            return tomorrow_weather
            
        except Exception as e:
            logger.error(f"Failed to fetch weather forecast: {str(e)}")
            # Return default weather data as fallback
            return self._generate_default_weather()

    def fetch_radiation_forecast(self) -> Optional[pd.DataFrame]:
        """Fetch radiation forecast (enhanced from standalone weather_forecaster.py)"""
        
        try:
            logger.info("Fetching radiation forecast...")
            
            # Try to load existing forecast from S3 first
            existing_forecast = self._load_existing_radiation_forecast()
            if existing_forecast is not None:
                logger.info("Using existing radiation forecast from S3")
                return existing_forecast
            
            # Fetch new forecast from Open-Meteo API
            pacific_tz = pytz.timezone("America/Los_Angeles")
            now = datetime.now(pacific_tz)
            today_str = now.strftime("%Y%m%d")
            tomorrow = now.date() + timedelta(days=1)
            tomorrow_date = tomorrow.strftime('%Y-%m-%d')
            
            # Setup Open-Meteo API request
            import requests
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ["shortwave_radiation"],
                "temperature_unit": "fahrenheit",
                "timezone": "America/Los_Angeles",
                "start_date": tomorrow_date,
                "end_date": tomorrow_date
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse radiation data
            hourly_data = data['hourly']
            times = [datetime.fromisoformat(t) for t in hourly_data['time']]
            radiation_values = hourly_data['shortwave_radiation']
            
            # Create DataFrame
            radiation_df = pd.DataFrame({
                'date': times,
                'shortwave_radiation': radiation_values
            })
            
            # Save to S3 for caching
            self._save_radiation_forecast_to_s3(radiation_df, today_str)
            
            logger.info(f"Radiation forecast retrieved: {len(radiation_df)} hours")
            return radiation_df
            
        except Exception as e:
            logger.error(f"Failed to fetch radiation forecast: {str(e)}")
            # Return default radiation data as fallback
            return self._generate_default_radiation()

    def _load_existing_weather_forecast(self) -> Optional[pd.DataFrame]:
        """Load existing weather forecast from S3"""
        try:
            today_str = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d")
            s3_key = f"{self.s3_config['input_prefix']}temperature/T_{today_str}.csv"
            
            response = s3_client.get_object(Bucket=self.s3_config['data_bucket'], Key=s3_key)
            df = pd.read_csv(response['Body'])
            
            # Check if forecast is recent (within last 6 hours)
            if len(df) >= 24:
                logger.info(f"Loaded existing weather forecast from S3: {s3_key}")
                return df
            
        except Exception:
            pass
        
        return None

    def _load_existing_radiation_forecast(self) -> Optional[pd.DataFrame]:
        """Load existing radiation forecast from S3"""
        try:
            today_str = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d")
            s3_key = f"{self.s3_config['input_prefix']}radiation/shortwave_radiation_{today_str}.csv"
            
            response = s3_client.get_object(Bucket=self.s3_config['data_bucket'], Key=s3_key)
            df = pd.read_csv(response['Body'])
            
            if len(df) >= 24:
                logger.info(f"Loaded existing radiation forecast from S3: {s3_key}")
                return df
            
        except Exception:
            pass
        
        return None

    def _save_weather_forecast_to_s3(self, df: pd.DataFrame, today_str: str):
        """Save weather forecast to S3"""
        try:
            s3_key = f"{self.s3_config['input_prefix']}temperature/T_{today_str}.csv"
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            
            s3_client.put_object(
                Bucket=self.s3_config['data_bucket'],
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            logger.info(f"Weather forecast saved to S3: {s3_key}")
            
        except Exception as e:
            logger.warning(f"Failed to save weather forecast to S3: {str(e)}")

    def _save_radiation_forecast_to_s3(self, df: pd.DataFrame, today_str: str):
        """Save radiation forecast to S3"""
        try:
            s3_key = f"{self.s3_config['input_prefix']}radiation/shortwave_radiation_{today_str}.csv"
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            
            s3_client.put_object(
                Bucket=self.s3_config['data_bucket'],
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            logger.info(f"Radiation forecast saved to S3: {s3_key}")
            
        except Exception as e:
            logger.warning(f"Failed to save radiation forecast to S3: {str(e)}")

    def _generate_default_weather(self) -> pd.DataFrame:
        """Generate default weather data as fallback"""
        pacific_tz = pytz.timezone("America/Los_Angeles")
        tomorrow = datetime.now(pacific_tz) + timedelta(days=1)
        
        # Generate 24 hours of default weather
        times = [tomorrow.replace(hour=h, minute=0, second=0, microsecond=0) for h in range(24)]
        temperatures = [70 + 5 * np.sin(2 * np.pi * h / 24) for h in range(24)]  # Simple sine wave
        
        return pd.DataFrame({
            'TradeDateTime': times,
            'Temperature': temperatures
        })

    def _generate_default_radiation(self) -> pd.DataFrame:
        """Generate default radiation data as fallback"""
        pacific_tz = pytz.timezone("America/Los_Angeles")
        tomorrow = datetime.now(pacific_tz) + timedelta(days=1)
        
        # Generate 24 hours of default radiation
        times = [tomorrow.replace(hour=h, minute=0, second=0, microsecond=0) for h in range(24)]
        # Simple radiation pattern (0 at night, peak at noon)
        radiation = [max(0, 500 * np.sin(np.pi * h / 24)) if 6 <= h <= 18 else 0 for h in range(24)]
        
        return pd.DataFrame({
            'date': times,
            'shortwave_radiation': radiation
        })

class PredictionEngine:
    """Prediction engine using standalone logic with Redshift output"""
    
    def __init__(self, environment_config: EnvironmentConfig):
        self.config = environment_config
        self.redshift_config = environment_config.get_redshift_config()
        self.s3_config = environment_config.get_s3_config()

    def invoke_endpoint_and_save_predictions(self, profile: str, endpoint_name: str, 
                                           test_data: pd.DataFrame) -> Dict[str, Any]:
        """Invoke SageMaker endpoint and save predictions to Redshift"""
        
        try:
            logger.info(f"Invoking endpoint for profile {profile}")
            
            # Prepare data for model inference
            feature_columns = self._get_feature_columns(profile)
            model_input = test_data[feature_columns].fillna(0).values.tolist()
            
            # Invoke SageMaker endpoint
            predictions = self._invoke_sagemaker_endpoint(endpoint_name, model_input)
            
            # Process predictions and save to Redshift
            records_saved = self._save_predictions_to_redshift(profile, test_data, predictions)
            
            result = {
                'status': 'success',
                'profile': profile,
                'endpoint_name': endpoint_name,
                'predictions_count': len(predictions),
                'records_saved': records_saved,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed {profile}: {len(predictions)} predictions, {records_saved} records saved")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process predictions for {profile}: {str(e)}")
            return {
                'status': 'failed',
                'profile': profile,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _get_feature_columns(self, profile: str) -> List[str]:
        """Get feature columns for each profile (from standalone logic)"""
        
        base_features = [
            'Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season',
            'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'
        ]
        
        if profile == 'RN':
            # RN profile includes radiation data
            return base_features + ['shortwave_radiation']
        else:
            return base_features

    def _invoke_sagemaker_endpoint(self, endpoint_name: str, model_input: List) -> List[float]:
        """Invoke SageMaker endpoint"""
        
        try:
            # Prepare payload
            payload = json.dumps(model_input)
            
            # Invoke endpoint
            response = sagemaker_runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            # Extract predictions
            if isinstance(result, dict) and 'predictions' in result:
                predictions = result['predictions']
            elif isinstance(result, list):
                predictions = result
            else:
                predictions = result
            
            # Validate predictions
            if not isinstance(predictions, list):
                raise ValueError(f"Invalid prediction format: expected list, got {type(predictions)}")
            
            return [float(p) for p in predictions]
            
        except Exception as e:
            logger.error(f"Endpoint invocation failed: {str(e)}")
            raise

    def _save_predictions_to_redshift(self, profile: str, test_data: pd.DataFrame, 
                                    predictions: List[float]) -> int:
        """Save predictions to Redshift using standalone logic"""
        
        try:
            logger.info(f"Saving {len(predictions)} predictions for {profile} to Redshift")
            
            # Combine test data with predictions (from standalone prediction_engine.py)
            df = test_data.copy()
            df['Predicted_Load'] = predictions
            
            # Create datetime column for Redshift
            df['tradedatetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
            
            # Calculate load_all (total load = predicted_load * count)
            df['load_all'] = df['Predicted_Load'] * df['Count']
            
            # Map profile and segment (from standalone prediction_engine.py mapping)
            profile_mapping = {
                'RNN': 'RES',
                'RN': 'RES', 
                'M': 'MEDCI',
                'S': 'SMLCOM',
                'AGR': 'AGR',
                'L': 'LIGHT',
                'A6': 'A6'
            }
            
            segment_mapping = {
                'RNN': 'NONSOLAR',
                'RN': 'SOLAR',
                'M': 'ALL',
                'S': 'ALL', 
                'AGR': 'ALL',
                'L': 'ALL',
                'A6': 'ALL'
            }
            
            # Format data for Redshift (from standalone format_combined_data_for_redshift)
            redshift_data = []
            for _, row in df.iterrows():
                redshift_data.append({
                    'tradedatetime': row['tradedatetime'],
                    'predicted_load': float(row['Predicted_Load']),
                    'count': float(row['Count']),
                    'load_all': float(row['load_all']),
                    'profile': profile_mapping.get(profile, profile),
                    'segment': segment_mapping.get(profile, 'ALL')
                })
            
            redshift_df = pd.DataFrame(redshift_data)
            
            # Save to Redshift using Data API
            records_inserted = self._insert_to_redshift(redshift_df)
            
            logger.info(f"Successfully saved {records_inserted} records to Redshift for {profile}")
            return records_inserted
            
        except Exception as e:
            logger.error(f"Failed to save predictions to Redshift: {str(e)}")
            raise

    def _insert_to_redshift(self, df: pd.DataFrame) -> int:
        """Insert data to Redshift using Data API (from standalone logic)"""
        
        try:
            if df.empty:
                return 0
            
            # Build VALUES clause (from standalone build_combined_values_clause)
            values_list = []
            for _, row in df.iterrows():
                tradedatetime = row['tradedatetime'].strftime('%Y-%m-%d %H:%M:%S')
                value_tuple = f"""(
                    '{tradedatetime}', 
                    {row['predicted_load']}, 
                    {row['count']}, 
                    {row['load_all']}, 
                    '{row['profile']}', 
                    '{row['segment']}'
                )"""
                values_list.append(value_tuple.strip())
            
            values_clause = ",\n".join(values_list)
            
            # Build INSERT statement
            schema = self.redshift_config['output_schema']
            table = self.redshift_config['output_table']
            
            insert_sql = f"""
            INSERT INTO {schema}.{table} 
            (tradedatetime, predicted_load, count, load_all, profile, segment)
            VALUES {values_clause}
            """
            
            # Execute INSERT
            response = redshift_data_client.execute_statement(
                ClusterIdentifier=self.redshift_config['cluster_identifier'],
                Database=self.redshift_config['database'],
                DbUser=self.redshift_config['db_user'],
                Sql=insert_sql
            )
            
            query_id = response['Id']
            logger.info(f"Redshift INSERT submitted: {query_id}")
            
            # Wait for completion
            self._wait_for_insert_completion(query_id)
            
            return len(df)
            
        except Exception as e:
            logger.error(f"Redshift insert failed: {str(e)}")
            raise

    def _wait_for_insert_completion(self, query_id: str, max_wait: int = 120):
        """Wait for Redshift INSERT completion"""
        waited = 0
        while waited < max_wait:
            try:
                status_response = redshift_data_client.describe_statement(Id=query_id)
                status = status_response['Status']
                
                if status == 'FINISHED':
                    logger.info(f"Redshift INSERT {query_id} completed successfully")
                    return
                elif status == 'FAILED':
                    error_msg = status_response.get('Error', 'Unknown error')
                    raise Exception(f'INSERT failed: {error_msg}')
                elif status == 'ABORTED':
                    raise Exception(f'INSERT was aborted')
                
                time.sleep(5)
                waited += 5
                
            except Exception as e:
                if 'failed:' in str(e) or 'aborted' in str(e):
                    raise
                time.sleep(5)
                waited += 5
                continue
        
        raise Exception(f'INSERT timed out after {max_wait} seconds')

def sanitize_for_logging(value: str, max_length: int = 50) -> str:
    """Sanitize string values for safe logging"""
    if not isinstance(value, str):
        value = str(value)
    # Remove potentially sensitive characters and limit length
    sanitized = ''.join(c for c in value if c.isalnum() or c in '-_.')
    return sanitized[:max_length]

def lambda_handler(event, context):
    """
    Main Lambda handler - processes single profile from Step Functions Map state
    Integrates standalone code logic with environment awareness
    """
    
    start_time = time.time()
    execution_id = context.aws_request_id
    
    try:
        # Log the event for debugging
        logger.info(f"Lambda execution started: {execution_id}")
        logger.info(f"Event received: {json.dumps(event, default=str)}")
        
        # Extract profile and endpoint from Step Functions Map state
        profile = event.get('profile')
        endpoint_name = event.get('endpoint_name')
        
        if not profile:
            raise ValueError("Profile not specified in event")
        if not endpoint_name:
            raise ValueError("Endpoint name not specified in event")
        
        # Sanitize for logging
        safe_profile = sanitize_for_logging(profile, 10)
        safe_endpoint = sanitize_for_logging(endpoint_name, 100)
        
        logger.info(f"Processing profile={safe_profile} endpoint={safe_endpoint}")
        
        # Initialize environment-aware configuration
        environment = os.environ.get('ENVIRONMENT', 'dev')
        env_config = EnvironmentConfig(environment)
        
        logger.info(f"Environment: {environment}")
        logger.info(f"Redshift cluster: {env_config.get_redshift_config()['cluster_identifier']}")
        logger.info(f"S3 data bucket: {env_config.get_s3_config()['data_bucket']}")
        
        # Initialize components
        data_processor = RedshiftDataProcessor(env_config)
        weather_fetcher = WeatherDataFetcher(env_config)
        prediction_engine = PredictionEngine(env_config)
        
        # Step 1: Fetch weather data (using cached if available)
        logger.info("Step 1: Fetching weather forecast...")
        step_start = time.time()
        weather_df = weather_fetcher.fetch_weather_forecast()
        if weather_df is None or weather_df.empty:
            raise Exception("Failed to fetch weather data")
        logger.info(f"Weather data fetched in {time.time() - step_start:.2f}s")
        
        # Step 2: Fetch radiation data (for RN profile only)
        radiation_df = None
        if profile == 'RN':
            logger.info("Step 2: Fetching radiation forecast for RN profile...")
            step_start = time.time()
            radiation_df = weather_fetcher.fetch_radiation_forecast()
            if radiation_df is None or radiation_df.empty:
                logger.warning("Failed to fetch radiation data for RN profile")
                radiation_df = None
            else:
                logger.info(f"Radiation data fetched in {time.time() - step_start:.2f}s")
        
        # Step 3: Query and preprocess data for the specific profile
        logger.info(f"Step 3: Processing data for profile {safe_profile}...")
        step_start = time.time()
        
        # Query profile data from Redshift
        raw_data = data_processor.query_profile_data(profile)
        if raw_data.empty:
            raise Exception(f"No data found for profile {profile}")
        
        # Preprocess data using standalone logic
        processed_data = data_processor.preprocess_data(raw_data, profile, weather_df, radiation_df)
        if processed_data.empty:
            raise Exception(f"Data preprocessing failed for profile {profile}")
        
        logger.info(f"Data processing completed in {time.time() - step_start:.2f}s")
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        # Step 4: Run prediction and save to Redshift
        logger.info(f"Step 4: Running predictions for profile {safe_profile}...")
        step_start = time.time()
        
        prediction_result = prediction_engine.invoke_endpoint_and_save_predictions(
            profile, endpoint_name, processed_data
        )
        
        logger.info(f"Predictions completed in {time.time() - step_start:.2f}s")
        
        # Prepare response
        total_time = time.time() - start_time
        
        response = {
            'statusCode': 200,
            'body': {
                'status': 'success',
                'profile': profile,
                'endpoint_name': endpoint_name,
                'execution_id': execution_id,
                'execution_time_seconds': round(total_time, 2),
                'data_processed_rows': len(processed_data),
                'prediction_result': prediction_result,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Lambda execution completed successfully in {total_time:.2f}s")
        logger.info(f"Profile {safe_profile}: {prediction_result.get('predictions_count', 0)} predictions, "
                   f"{prediction_result.get('records_saved', 0)} records saved")
        
        return response
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        total_time = time.time() - start_time
        
        logger.error(f"Lambda execution failed after {total_time:.2f}s: {error_msg}")
        
        # Return error response in Step Functions compatible format
        error_response = {
            'statusCode': 500,
            'body': {
                'status': 'failed',
                'error': error_msg,
                'profile': event.get('profile', 'unknown'),
                'endpoint_name': event.get('endpoint_name', 'unknown'),
                'execution_id': execution_id,
                'execution_time_seconds': round(total_time, 2),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return error_response