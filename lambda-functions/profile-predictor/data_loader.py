"""
Data Loader Module
lambda-functions/profile-predictor/data_loader.py

Data loading and preparation logic converted from container
Handles profile-specific data loading and feature preparation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pytz
from io import StringIO

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loading and preparation functionality
    Converted from container logic
    """
   
    def __init__(self, data_bucket: str, s3_utils):
        """Initialize data loader"""
        self.data_bucket = data_bucket
        self.s3_utils = s3_utils
       
        logger.info("Data loader initialized")

    def load_profile_test_data(self, profile: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a specific profile
        Converted from container predictions.py logic
        """
       
        try:
            logger.info(f"Loading historical data for profile {profile}")
           
            # Try different possible data locations (same logic as container)
            possible_keys = [
                f"archived_folders/forecasting/data/xgboost/input/{profile}_test.csv",
                f"archived_folders/forecasting/data/xgboost/input/df_{profile}_test.csv",
                f"archived_folders/forecasting/data/processed/{profile}_historical.csv",
                f"archived_folders/forecasting/data/raw/{profile}_test.csv"
            ]
           
            for key in possible_keys:
                try:
                    logger.info(f"Trying to load: s3://{self.data_bucket}/{key}")
                   
                    obj_content = self.s3_utils.get_object(bucket=self.data_bucket, key=key)
                    df = pd.read_csv(StringIO(obj_content))
                   
                    logger.info(f"✓ Successfully loaded profile data: {len(df)} rows, {len(df.columns)} columns")
                    logger.info(f"Columns: {list(df.columns)}")
                   
                    return df
                   
                except Exception as e:
                    logger.debug(f"Could not load from {key}: {str(e)}")
                    continue
           
            # If no data found, create minimal default data (same logic as container)
            logger.warning(f"No historical data found for {profile}, creating default data")
           
            return self._create_default_test_data(profile)
           
        except Exception as e:
            logger.error(f"Failed to load profile data for {profile}: {str(e)}")
            raise Exception(f"Data loading failed for {profile}: {str(e)}")

    def _create_default_test_data(self, profile: str) -> pd.DataFrame:
        """
        Create basic test data structure when no historical data is found
        Same logic as container
        """
       
        try:
            # Create basic test data structure (same logic as container)
            pacific_tz = pytz.timezone("America/Los_Angeles")
            tomorrow = datetime.now(pacific_tz) + timedelta(days=1)
           
            default_data = []
            for hour in range(24):
                default_data.append({
                    'Year': tomorrow.year,
                    'Month': tomorrow.month,
                    'Day': tomorrow.day,
                    'Hour': hour,
                    'Count': 1000,  # Default meter count
                    'Weekday': tomorrow.weekday() + 1,
                    'Season': 1 if tomorrow.month in [6, 7, 8, 9] else 0,
                    'Holiday': 0,
                    'Workday': 1 if tomorrow.weekday() < 5 else 0,
                    'Temperature': 70.0,
                    'Load_I_lag_14_days': 0.5,
                    'Load_lag_70_days': 0.5
                })
           
            df = pd.DataFrame(default_data)
            logger.info(f"Created default data for {profile}: {len(df)} rows")
           
            return df
           
        except Exception as e:
            logger.error(f"Failed to create default data for {profile}: {str(e)}")
            raise

    def prepare_prediction_data(self, profile: str, profile_data: pd.DataFrame,
                              weather_df: Optional[pd.DataFrame],
                              radiation_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare data for prediction based on profile requirements
        Converted from container predictions.py logic
        """
       
        try:
            logger.info(f"Preparing prediction data for profile {profile}")
           
            # Profile-specific feature sets (same as container)
            profile_features = {
                'RNN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                       'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'RN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                      'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days', 'shortwave_radiation'],
                'M': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                     'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'S': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                     'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'AGR': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                       'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'L': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                     'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
                'A6': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                      'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days']
            }
           
            required_features = profile_features.get(profile, profile_features['RNN'])
           
            # Start with profile data (same logic as container)
            test_df = profile_data.copy()
           
            # Merge weather data if available and needed (same logic as container)
            if weather_df is not None and 'Temperature' in required_features:
                test_df = self._merge_weather_data(test_df, weather_df)
           
            # Merge radiation data if available and needed (same logic as container)
            if radiation_df is not None and 'shortwave_radiation' in required_features:
                test_df = self._merge_radiation_data(test_df, radiation_df)
           
            # Ensure all required features are present (same logic as container)
            for feature in required_features:
                if feature not in test_df.columns:
                    default_value = self._get_default_feature_value(feature)
                    test_df[feature] = default_value
                    logger.info(f"Added missing feature {feature} with default value {default_value}")
           
            # Select only required features in the correct order (same logic as container)
            test_df = test_df[required_features]
           
            # Handle missing values (same logic as container)
            test_df = test_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
           
            # Data type conversion (ensure numeric types)
            for feature in required_features:
                if feature in ['Weekday', 'Season', 'Holiday', 'Workday']:
                    test_df[feature] = test_df[feature].astype(int)
                else:
                    test_df[feature] = pd.to_numeric(test_df[feature], errors='coerce').fillna(0)
           
            logger.info(f"✓ Prepared prediction data for {profile}: {test_df.shape}")
            logger.info(f"Features: {list(test_df.columns)}")
           
            # Log data summary
            data_summary = {
                'shape': test_df.shape,
                'features': list(test_df.columns),
                'sample_values': test_df.iloc[0].to_dict() if len(test_df) > 0 else {},
                'null_counts': test_df.isnull().sum().to_dict()
            }
            logger.info(f"Data summary: {data_summary}")
           
            return test_df
           
        except Exception as e:
            logger.error(f"Failed to prepare prediction data for {profile}: {str(e)}")
            raise Exception(f"Data preparation failed for {profile}: {str(e)}")

    def _merge_weather_data(self, test_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weather data with test data
        Same logic as container
        """
       
        try:
            # Match on date/time columns (same logic as container)
            if all(col in weather_df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
                logger.info("Merging weather data on Year/Month/Day/Hour")
               
                test_df = test_df.merge(
                    weather_df[['Year', 'Month', 'Day', 'Hour', 'Temperature']],
                    on=['Year', 'Month', 'Day', 'Hour'],
                    how='left',
                    suffixes=('', '_weather')
                )
                # Use weather temperature if available (same logic as container)
                if 'Temperature_weather' in test_df.columns:
                    test_df['Temperature'] = test_df['Temperature_weather'].fillna(test_df['Temperature'])
                    test_df = test_df.drop('Temperature_weather', axis=1)
                   
                logger.info("✓ Weather data merged successfully")
            else:
                logger.warning("Weather data missing required time columns")
               
            return test_df
           
        except Exception as e:
            logger.error(f"Failed to merge weather data: {str(e)}")
            return test_df

    def _merge_radiation_data(self, test_df: pd.DataFrame, radiation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge radiation data with test data
        Same logic as container
        """
       
        try:
            if all(col in radiation_df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
                logger.info("Merging radiation data on Year/Month/Day/Hour")
               
                test_df = test_df.merge(
                    radiation_df[['Year', 'Month', 'Day', 'Hour', 'shortwave_radiation']],
                    on=['Year', 'Month', 'Day', 'Hour'],
                    how='left'
                )
               
                logger.info("✓ Radiation data merged successfully")
            else:
                logger.warning("Radiation data missing required time columns")
               
            return test_df
           
        except Exception as e:
            logger.error(f"Failed to merge radiation data: {str(e)}")
            return test_df

    def _get_default_feature_value(self, feature: str) -> float:
        """
        Get default values for missing features
        Same logic as container
        """
        defaults = {
            'Count': 1000.0,
            'Year': float(datetime.now().year),
            'Month': float(datetime.now().month),
            'Day': float(datetime.now().day),
            'Hour': 12.0,
            'Weekday': 1.0,
            'Season': 1.0,
            'Holiday': 0.0,
            'Workday': 1.0,
            'Temperature': 70.0,
            'Load_I_lag_14_days': 0.5,
            'Load_lag_70_days': 0.5,
            'shortwave_radiation': 200.0
        }
        return defaults.get(feature, 0.0)

    def validate_data_quality(self, df: pd.DataFrame, profile: str) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics
        """
       
        try:
            quality_metrics = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'null_counts': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'duplicated_rows': df.duplicated().sum()
            }
           
            # Check for critical issues
            critical_issues = []
           
            if len(df) == 0:
                critical_issues.append("Dataset is empty")
           
            if df.isnull().all().any():
                null_columns = df.columns[df.isnull().all()].tolist()
                critical_issues.append(f"Columns with all null values: {null_columns}")
           
            # Check for reasonable value ranges
            if 'Temperature' in df.columns:
                temp_range = (df['Temperature'].min(), df['Temperature'].max())
                if temp_range[0] < -50 or temp_range[1] > 150:
                    critical_issues.append(f"Temperature values out of reasonable range: {temp_range}")
           
            if 'Count' in df.columns:
                if (df['Count'] <= 0).any():
                    critical_issues.append("Found non-positive Count values")
           
            quality_metrics['critical_issues'] = critical_issues
            quality_metrics['data_quality_score'] = self._calculate_quality_score(df, critical_issues)
           
            logger.info(f"Data quality assessment for {profile}: {quality_metrics['data_quality_score']}/100")
           
            return quality_metrics
           
        except Exception as e:
            logger.error(f"Data quality validation failed: {str(e)}")
            return {'error': str(e)}

    def _calculate_quality_score(self, df: pd.DataFrame, critical_issues: list) -> int:
        """Calculate overall data quality score (0-100)"""
       
        score = 100
       
        # Deduct for critical issues
        score -= len(critical_issues) * 20
       
        # Deduct for missing values
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= min(null_percentage * 2, 30)
       
        # Deduct for duplicated rows
        dup_percentage = (df.duplicated().sum() / len(df)) * 100
        score -= min(dup_percentage, 10)
       
        return max(0, int(score))
