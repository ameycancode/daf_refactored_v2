"""
Environment-Aware Data Processor for Lambda
Adapted from standalone data_processor.py with Redshift integration
Supports single profile processing within Lambda constraints
"""

import logging
import pandas as pd
import numpy as np
import boto3
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pytz
from io import StringIO

logger = logging.getLogger(__name__)

class LambdaDataProcessor:
    """
    Data processor adapted from standalone code for Lambda environment
    Focuses on single profile processing with Redshift integration
    """
    
    def __init__(self, redshift_config: Dict[str, str], s3_config: Dict[str, str]):
        self.redshift_config = redshift_config
        self.s3_config = s3_config
        self.redshift_client = boto3.client('redshift-data', region_name=redshift_config['region'])
        self.s3_client = boto3.client('s3')
        
    def process_single_profile_data(self, profile: str, days_back: int = 100) -> pd.DataFrame:
        """
        Process data for a single profile using standalone logic
        Optimized for Lambda memory and time constraints
        """
        
        try:
            logger.info(f"Processing data for profile {profile} ({days_back} days back)")
            
            # Step 1: Query raw data from Redshift
            raw_data = self._query_redshift_data(profile, days_back)
            if raw_data.empty:
                logger.warning(f"No raw data retrieved for profile {profile}")
                return pd.DataFrame()
            
            # Step 2: Convert to hourly format (from standalone logic)
            hourly_data = self._convert_to_hourly_data(raw_data)
            logger.info(f"Converted to hourly data: {len(hourly_data)} rows")
            
            # Step 3: Add temporal features (from standalone logic)
            featured_data = self._add_temporal_features(hourly_data)
            logger.info(f"Added temporal features: {len(featured_data)} rows")
            
            # Step 4: Create lag features (from standalone logic)
            lagged_data = self._create_lag_features(featured_data)
            logger.info(f"Created lag features: {len(lagged_data)} rows")
            
            # Step 5: Clean and prepare final dataset
            final_data = self._clean_and_prepare_data(lagged_data, profile)
            logger.info(f"Final processed data: {len(final_data)} rows")
            
            return final_data
            
        except Exception as e:
            logger.error(f"Failed to process data for profile {profile}: {str(e)}")
            raise

    def merge_with_weather_data(self, profile_data: pd.DataFrame, 
                               weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge profile data with weather data using standalone logic
        """
        
        try:
            if profile_data.empty or weather_df.empty:
                logger.warning("Empty profile or weather data for merging")
                return profile_data
            
            logger.info("Merging profile data with weather data")
            
            # Prepare weather data for merging (from standalone logic)
            weather_clean = weather_df.copy()
            weather_clean['TradeDateTime'] = pd.to_datetime(weather_clean['TradeDateTime'])
            weather_clean['Year'] = weather_clean['TradeDateTime'].dt.year
            weather_clean['Month'] = weather_clean['TradeDateTime'].dt.month
            weather_clean['Day'] = weather_clean['TradeDateTime'].dt.day
            weather_clean['Hour'] = weather_clean['TradeDateTime'].dt.hour
            
            # Merge on datetime components (from standalone merge_load_temperature)
            merged_data = profile_data.merge(
                weather_clean[['Year', 'Month', 'Day', 'Hour', 'Temperature']],
                on=['Year', 'Month', 'Day', 'Hour'],
                how='left'
            )
            
            # Fill missing temperatures with mean (from standalone logic)
            mean_temp = merged_data['Temperature'].mean()
            merged_data['Temperature'] = merged_data['Temperature'].fillna(mean_temp)
            
            logger.info(f"Merged weather data: {len(merged_data)} rows, mean temp: {mean_temp:.1f}°F")
            return merged_data
            
        except Exception as e:
            logger.error(f"Failed to merge weather data: {str(e)}")
            raise

    def add_radiation_data_for_rn(self, profile_data: pd.DataFrame, 
                                 radiation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add radiation data for RN profile using standalone logic
        """
        
        try:
            if profile_data.empty or radiation_df is None or radiation_df.empty:
                logger.warning("Cannot add radiation data - missing data")
                return profile_data
            
            logger.info("Adding radiation data for RN profile")
            
            # Prepare radiation data (from standalone add_radiation_to_df_RN)
            radiation_clean = radiation_df.copy()
            radiation_clean['date'] = pd.to_datetime(radiation_clean['date'])
            radiation_clean['Year'] = radiation_clean['date'].dt.year
            radiation_clean['Month'] = radiation_clean['date'].dt.month
            radiation_clean['Day'] = radiation_clean['date'].dt.day
            radiation_clean['Hour'] = radiation_clean['date'].dt.hour
            
            # Merge radiation data
            merged_data = profile_data.merge(
                radiation_clean[['Year', 'Month', 'Day', 'Hour', 'shortwave_radiation']],
                on=['Year', 'Month', 'Day', 'Hour'],
                how='left'
            )
            
            # Fill missing radiation values with 0 (from standalone logic)
            merged_data['shortwave_radiation'] = merged_data['shortwave_radiation'].fillna(0)
            
            logger.info(f"Added radiation data: {len(merged_data)} rows")
            return merged_data
            
        except Exception as e:
            logger.error(f"Failed to add radiation data: {str(e)}")
            raise

    def _query_redshift_data(self, profile: str, days_back: int) -> pd.DataFrame:
        """
        Query Redshift for single profile data using standalone query logic
        """
        
        try:
            # Calculate date range (from standalone query_data)
            pacific_tz = pytz.timezone("America/Los_Angeles")
            current_date = datetime.now(pacific_tz)
            start_date = current_date - timedelta(days=days_back)
            
            # Build query with profile filtering (adapted from standalone)
            schema = self.redshift_config['input_schema']
            table = self.redshift_config['input_table']
            
            query = f"""
            SELECT
                tradedatelocal as tradedate,
                tradehourstartlocal as tradetime,
                loadprofile, rategroup, baseload, lossadjustedload, metercount,
                loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
                submission, createddate as created
            FROM {schema}.{table}
            WHERE tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'
            AND loadprofile = '{profile}'
            AND submission = 'Final'
            ORDER BY tradedatelocal, tradehourstartlocal
            """
            
            logger.info(f"Querying Redshift for profile {profile} from {start_date.strftime('%Y-%m-%d')}")
            
            # Execute query
            response = self.redshift_client.execute_statement(
                ClusterIdentifier=self.redshift_config['cluster_identifier'],
                Database=self.redshift_config['database'],
                DbUser=self.redshift_config['db_user'],
                Sql=query
            )
            
            query_id = response['Id']
            logger.info(f"Redshift query submitted: {query_id}")
            
            # Wait for completion and get results
            self._wait_for_query_completion(query_id)
            result_df = self._get_query_results(query_id)
            
            logger.info(f"Retrieved {len(result_df)} rows for profile {profile}")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to query Redshift data: {str(e)}")
            raise

    def _convert_to_hourly_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw Redshift data to hourly format using standalone logic
        """
        
        try:
            # Parse datetime (from standalone process_load_data_from_redshift)
            raw_data['TradeDateTime'] = pd.to_datetime(
                raw_data['tradedate'] + ' ' + raw_data['tradetime'].astype(str) + ':00:00'
            )
            
            # Extract time components
            raw_data['Year'] = raw_data['TradeDateTime'].dt.year
            raw_data['Month'] = raw_data['TradeDateTime'].dt.month
            raw_data['Day'] = raw_data['TradeDateTime'].dt.day
            raw_data['Hour'] = raw_data['TradeDateTime'].dt.hour
            raw_data['Weekday'] = raw_data['TradeDateTime'].dt.weekday
            
            # Aggregate by hour (from standalone logic)
            hourly_agg = raw_data.groupby(['Year', 'Month', 'Day', 'Hour']).agg({
                'loadlal': 'sum',           # Loss adjusted load
                'loadmetercount': 'sum',    # Meter count
                'TradeDateTime': 'first',   # Keep first datetime
                'Weekday': 'first'          # Keep weekday
            }).reset_index()
            
            # Calculate load per meter (from standalone logic)
            hourly_agg['Load_I'] = hourly_agg['loadlal'] / hourly_agg['loadmetercount']
            hourly_agg['Load_I'] = hourly_agg['Load_I'].fillna(0)
            
            # Rename columns for consistency
            hourly_agg = hourly_agg.rename(columns={
                'loadmetercount': 'Count'
            })
            
            logger.info(f"Converted to hourly data: {len(hourly_agg)} records")
            return hourly_agg
            
        except Exception as e:
            logger.error(f"Failed to convert to hourly data: {str(e)}")
            raise

    def _add_temporal_features(self, hourly_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features using standalone logic
        """
        
        try:
            data = hourly_data.copy()
            
            # Add season (from standalone generate_data_profile)
            data['Season'] = data['Month'].apply(self._get_season)
            
            # Add holiday indicator (simplified from standalone logic)
            data['Holiday'] = data['TradeDateTime'].apply(self._is_holiday).astype(int)
            
            # Add workday indicator (from standalone logic)
            data['Workday'] = ((data['Weekday'] < 5) & (data['Holiday'] == 0)).astype(int)
            
            logger.info(f"Added temporal features to {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to add temporal features: {str(e)}")
            raise

    def _create_lag_features(self, featured_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features using standalone logic
        """
        
        try:
            data = featured_data.copy()
            
            # Sort by datetime for proper lag calculation
            data = data.sort_values(['Year', 'Month', 'Day', 'Hour']).reset_index(drop=True)
            
            # Create lag features (from standalone create_lagged_profiles_for_prediction)
            data['Load_I_lag_14_days'] = data['Load_I'].shift(24 * 14)  # 14 days ago
            data['Load_lag_70_days'] = data['Load_I'].shift(24 * 70)    # 70 days ago
            
            # Fill NaN values with median (from standalone logic)
            median_load = data['Load_I'].median()
            data['Load_I_lag_14_days'] = data['Load_I_lag_14_days'].fillna(median_load)
            data['Load_lag_70_days'] = data['Load_lag_70_days'].fillna(median_load)
            
            logger.info(f"Created lag features for {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to create lag features: {str(e)}")
            raise

    def _clean_and_prepare_data(self, lagged_data: pd.DataFrame, profile: str) -> pd.DataFrame:
        """
        Clean and prepare final dataset using standalone logic
        """
        
        try:
            data = lagged_data.copy()
            
            # Count replacement (from standalone replace_count_i)
            median_count = data['Count'].median()
            data['Count'] = data['Count'].fillna(median_count)
            data['Count'] = data['Count'].clip(lower=1)  # Ensure positive counts
            
            # Remove any remaining NaN values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(0)
            
            # Ensure required columns exist
            required_columns = [
                'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season',
                'Holiday', 'Workday', 'Load_I', 'Load_I_lag_14_days',
                'Load_lag_70_days', 'Count'
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing columns for {profile}: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    data[col] = 0
            
            logger.info(f"Final data preparation completed for {profile}: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to clean and prepare data: {str(e)}")
            raise

    def _get_season(self, month: int) -> int:
        """
        Get season from month (from standalone logic)
        """
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    def _is_holiday(self, date: pd.Timestamp) -> bool:
        """
        Simple holiday detection (from standalone logic)
        """
        month = date.month
        day = date.day
        
        # Major US holidays (simplified)
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas Day
        ]
        
        return (month, day) in holidays

    def _wait_for_query_completion(self, query_id: str, max_wait: int = 300):
        """
        Wait for Redshift query completion
        """
        waited = 0
        while waited < max_wait:
            try:
                status_response = self.redshift_client.describe_statement(Id=query_id)
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

    def _get_query_results(self, query_id: str) -> pd.DataFrame:
        """
        Get all paginated results from Redshift query
        """
        all_records = []
        column_metadata = None
        next_token = None
        
        try:
            while True:
                # Get results with pagination
                if next_token:
                    response = self.redshift_client.get_statement_result(
                        Id=query_id, NextToken=next_token
                    )
                else:
                    response = self.redshift_client.get_statement_result(Id=query_id)
                
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
            
            # Convert to DataFrame
            if not all_records or not column_metadata:
                logger.warning("No data returned from query")
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
            
            df = pd.DataFrame(data_rows, columns=column_names)
            logger.info(f"Query results converted to DataFrame: {len(df)} rows, {len(column_names)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get query results: {str(e)}")
            raise

class PredictionDataBuilder:
    """
    Builds prediction-ready datasets for tomorrow's forecasting
    """
    
    def __init__(self, pacific_tz):
        self.pacific_tz = pacific_tz
        
    def build_prediction_dataset(self, profile: str, historical_data: pd.DataFrame,
                               weather_df: pd.DataFrame, radiation_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build prediction dataset for tomorrow using historical patterns
        """
        
        try:
            logger.info(f"Building prediction dataset for profile {profile}")
            
            # Get tomorrow's date
            tomorrow = datetime.now(self.pacific_tz) + timedelta(days=1)
            
            # Create 24-hour template for tomorrow
            prediction_rows = []
            
            for hour in range(24):
                # Get weather data for this hour
                weather_temp = self._get_weather_for_hour(weather_df, hour)
                
                # Get radiation data for RN profile
                radiation_value = 0
                if profile == 'RN' and radiation_df is not None:
                    radiation_value = self._get_radiation_for_hour(radiation_df, hour)
                
                # Get historical patterns for this hour and day type
                historical_pattern = self._get_historical_pattern(
                    historical_data, tomorrow.weekday(), hour
                )
                
                # Build prediction row
                pred_row = {
                    'Year': tomorrow.year,
                    'Month': tomorrow.month,
                    'Day': tomorrow.day,
                    'Hour': hour,
                    'Weekday': tomorrow.weekday(),
                    'Season': self._get_season(tomorrow.month),
                    'Holiday': int(self._is_holiday(tomorrow)),
                    'Workday': int(tomorrow.weekday() < 5 and not self._is_holiday(tomorrow)),
                    'Temperature': weather_temp,
                    'Load_I_lag_14_days': historical_pattern.get('lag_14', 0.5),
                    'Load_lag_70_days': historical_pattern.get('lag_70', 0.5),
                    'Count': historical_pattern.get('count', 1000)
                }
                
                # Add radiation for RN profile
                if profile == 'RN':
                    pred_row['shortwave_radiation'] = radiation_value
                
                prediction_rows.append(pred_row)
            
            prediction_df = pd.DataFrame(prediction_rows)
            logger.info(f"Built prediction dataset: {len(prediction_df)} hours")
            return prediction_df
            
        except Exception as e:
            logger.error(f"Failed to build prediction dataset: {str(e)}")
            raise

    def _get_weather_for_hour(self, weather_df: pd.DataFrame, hour: int) -> float:
        """Get weather temperature for specific hour"""
        try:
            weather_hour = weather_df[weather_df['TradeDateTime'].dt.hour == hour]
            if not weather_hour.empty:
                return float(weather_hour['Temperature'].iloc[0])
            else:
                # Fallback to default temperature pattern
                return 70 + 5 * np.sin(2 * np.pi * hour / 24)
        except Exception:
            return 70.0  # Default temperature

    def _get_radiation_for_hour(self, radiation_df: pd.DataFrame, hour: int) -> float:
        """Get radiation value for specific hour"""
        try:
            radiation_hour = radiation_df[radiation_df['date'].dt.hour == hour]
            if not radiation_hour.empty:
                return float(radiation_hour['shortwave_radiation'].iloc[0])
            else:
                # Fallback to default radiation pattern
                return max(0, 500 * np.sin(np.pi * hour / 24)) if 6 <= hour <= 18 else 0
        except Exception:
            return 0.0

    def _get_historical_pattern(self, historical_data: pd.DataFrame, 
                              weekday: int, hour: int) -> Dict[str, float]:
        """Get historical patterns for similar day/hour combinations"""
        try:
            # Filter for same weekday and hour
            similar_data = historical_data[
                (historical_data['Weekday'] == weekday) & 
                (historical_data['Hour'] == hour)
            ]
            
            if similar_data.empty:
                # Fallback to same hour regardless of weekday
                similar_data = historical_data[historical_data['Hour'] == hour]
            
            if similar_data.empty:
                # Final fallback to defaults
                return {'lag_14': 0.5, 'lag_70': 0.5, 'count': 1000}
            
            # Get recent patterns (last 30 days)
            recent_data = similar_data.tail(30)
            
            return {
                'lag_14': recent_data['Load_I_lag_14_days'].median(),
                'lag_70': recent_data['Load_lag_70_days'].median(), 
                'count': recent_data['Count'].median()
            }
            
        except Exception as e:
            logger.warning(f"Failed to get historical pattern: {str(e)}")
            return {'lag_14': 0.5, 'lag_70': 0.5, 'count': 1000}

    def _get_season(self, month: int) -> int:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    def _is_holiday(self, date: datetime) -> bool:
        """Simple holiday detection"""
        month = date.month
        day = date.day
        
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas Day
        ]
        
        return (month, day) in holidays

class LambdaDataValidator:
    """
    Data validation and quality checks for Lambda processing
    """
    
    @staticmethod
    def validate_processed_data(df: pd.DataFrame, profile: str) -> Dict[str, Any]:
        """
        Validate processed data quality
        """
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                validation_results['is_valid'] = False
                validation_results['errors'].append("DataFrame is empty")
                return validation_results
            
            # Check required columns
            required_base_columns = [
                'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season',
                'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days',
                'Load_lag_70_days', 'Count'
            ]
            
            if profile == 'RN':
                required_columns = required_base_columns + ['shortwave_radiation']
            else:
                required_columns = required_base_columns
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Missing columns: {missing_columns}")
            
            # Check for excessive NaN values
            nan_threshold = 0.1  # 10% threshold
            for column in df.select_dtypes(include=[np.number]).columns:
                nan_percentage = df[column].isna().sum() / len(df)
                if nan_percentage > nan_threshold:
                    validation_results['warnings'].append(
                        f"Column {column} has {nan_percentage:.1%} NaN values"
                    )
            
            # Check data ranges
            if 'Temperature' in df.columns:
                temp_stats = df['Temperature'].describe()
                if temp_stats['min'] < -50 or temp_stats['max'] > 150:
                    validation_results['warnings'].append(
                        f"Temperature values outside normal range: {temp_stats['min']:.1f} to {temp_stats['max']:.1f}"
                    )
            
            if 'Count' in df.columns:
                if (df['Count'] <= 0).any():
                    validation_results['warnings'].append("Some Count values are zero or negative")
            
            # Add statistics
            validation_results['stats'] = {
                'total_rows': len(df),
                'columns_count': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            logger.info(f"Data validation completed for {profile}: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results

    @staticmethod
    def validate_prediction_input(df: pd.DataFrame, profile: str) -> bool:
        """
        Validate prediction input data
        """
        
        try:
            # Check for 24 hours of data
            if len(df) != 24:
                logger.warning(f"Expected 24 hours of data, got {len(df)}")
                return False
            
            # Check hour sequence
            expected_hours = set(range(24))
            actual_hours = set(df['Hour'].unique())
            if expected_hours != actual_hours:
                logger.warning(f"Missing hours in prediction data: {expected_hours - actual_hours}")
                return False
            
            # Check for required features
            required_features = [
                'Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season',
                'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'
            ]
            
            if profile == 'RN':
                required_features.append('shortwave_radiation')
            
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                logger.error(f"Missing required features for {profile}: {missing_features}")
                return False
            
            # Check for NaN values in required features
            for feature in required_features:
                if df[feature].isna().any():
                    logger.warning(f"NaN values found in feature {feature}")
                    return False
            
            logger.info(f"Prediction input validation passed for {profile}")
            return True
            
        except Exception as e:
            logger.error(f"Prediction input validation failed: {str(e)}")
            return False
