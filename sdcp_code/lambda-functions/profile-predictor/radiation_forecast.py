"""
Radiation Forecast Module
lambda-functions/profile-predictor/radiation_forecast.py

Radiation data fetching logic converted from container R_forecast.py
Handles radiation API integration for RN profile
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from io import StringIO
from typing import Optional

logger = logging.getLogger(__name__)

class RadiationForecast:
    """
    Radiation forecast functionality converted from container R_forecast.py
    """
   
    def __init__(self, data_bucket: str, s3_utils):
        """Initialize radiation forecast"""
        self.data_bucket = data_bucket
        self.s3_utils = s3_utils
       
        # Same coordinates as weather (San Diego)
        self.latitude = 32.7157
        self.longitude = -117.1611
       
        logger.info("Radiation forecast initialized for San Diego")

    def fetch_shortwave_radiation(self) -> Optional[pd.DataFrame]:
        """
        Fetch shortwave radiation data for tomorrow
        Converted from container R_forecast.py logic
        """
       
        try:
            logger.info("Fetching shortwave radiation data...")
           
            # Try to load existing forecast from S3 first
            existing_forecast = self._load_existing_radiation_forecast()
            if existing_forecast is not None:
                logger.info("✓ Using existing radiation forecast from S3")
                return existing_forecast
           
            # Fetch new forecast from Open-Meteo API (same logic as container)
            logger.info("Fetching new radiation forecast from Open-Meteo API...")
           
            # Setup for Open-Meteo API (same as container)
            pacific_tz = pytz.timezone("America/Los_Angeles")
            tomorrow = datetime.now(pacific_tz) + timedelta(days=1)
            tomorrow_date = tomorrow.strftime('%Y-%m-%d')
           
            # Define API parameters (same as container)
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
           
            # Fetch radiation data (same as container)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
           
            data = response.json()
           
            # Parse the response (same logic as container)
            hourly_data = data['hourly']
            times = hourly_data['time']
            radiation_values = hourly_data['shortwave_radiation']
           
            # Create DataFrame (same as container)
            radiation_data = []
            for time_str, radiation in zip(times, radiation_values):
                radiation_data.append({
                    'date': pd.to_datetime(time_str),
                    'shortwave_radiation': radiation if radiation is not None else 0.0
                })
           
            df_radiation = pd.DataFrame(radiation_data)
           
            if len(df_radiation) == 0:
                logger.warning("No radiation data received from API")
                return self._get_fallback_radiation_data()
           
            # Add time components for merging (same as container logic)
            df_radiation = self._add_time_components(df_radiation)
           
            # Save to S3 (same as container)
            self._save_radiation_to_s3(df_radiation, datetime.now(pacific_tz))
           
            logger.info(f"✓ Radiation forecast fetched successfully: {len(df_radiation)} data points")
           
            return df_radiation
           
        except requests.exceptions.RequestException as e:
            logger.error(f"Radiation API request failed: {str(e)}")
            return self._get_fallback_radiation_data()
           
        except Exception as e:
            logger.error(f"Radiation forecast fetch failed: {str(e)}")
            return self._get_fallback_radiation_data()

    def _load_existing_radiation_forecast(self) -> Optional[pd.DataFrame]:
        """
        Load existing radiation forecast from S3 if available and recent
        """
       
        try:
            pacific_tz = pytz.timezone("America/Los_Angeles")
            today_str = datetime.now(pacific_tz).strftime("%Y%m%d")
           
            s3_key = f"archived_folders/forecasting/data/radiation/shortwave_radiation_{today_str}.csv"
           
            logger.info(f"Checking for existing radiation forecast: s3://{self.data_bucket}/{s3_key}")
           
            obj_content = self.s3_utils.get_object(bucket=self.data_bucket, key=s3_key)
            df = pd.read_csv(StringIO(obj_content))
           
            # Convert date column and add time components
            df['date'] = pd.to_datetime(df['date'])
            df = self._add_time_components(df)
           
            logger.info(f"✓ Loaded existing radiation forecast: {len(df)} records")
            return df
           
        except Exception as e:
            logger.debug(f"Could not load existing radiation forecast: {str(e)}")
            return None

    def _add_time_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Year, Month, Day, Hour components for data merging
        """
       
        try:
            df = df.copy()
           
            # Ensure date column is properly formatted
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
               
                # Remove timezone info if present
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
               
                # Add time components
                df['Year'] = df['date'].dt.year
                df['Month'] = df['date'].dt.month
                df['Day'] = df['date'].dt.day
                df['Hour'] = df['date'].dt.hour
           
            logger.info("✓ Added time components to radiation data")
           
            return df
           
        except Exception as e:
            logger.error(f"Failed to add time components to radiation data: {str(e)}")
            return df

    def _save_radiation_to_s3(self, radiation_df: pd.DataFrame, current_time: datetime):
        """
        Save radiation forecast to S3 for future use
        Same logic as container
        """
       
        try:
            today_str = current_time.strftime("%Y%m%d")
            s3_key = f"archived_folders/forecasting/data/radiation/shortwave_radiation_{today_str}.csv"
           
            # Prepare data for saving (same format as container)
            save_df = radiation_df.copy()
           
            # Format date for saving (same as container)
            if 'date' in save_df.columns:
                save_df['date'] = save_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
           
            # Convert to CSV and upload
            csv_buffer = StringIO()
            save_df[['date', 'shortwave_radiation']].to_csv(csv_buffer, index=False)
           
            self.s3_utils.put_object(
                bucket=self.data_bucket,
                key=s3_key,
                body=csv_buffer.getvalue(),
                content_type='text/csv'
            )
           
            logger.info(f"✓ Radiation forecast saved to s3://{self.data_bucket}/{s3_key}")
           
        except Exception as e:
            logger.warning(f"Could not save radiation forecast to S3: {str(e)}")

    def _get_fallback_radiation_data(self) -> Optional[pd.DataFrame]:
        """
        Generate fallback radiation data when API is unavailable
        """
       
        try:
            logger.info("Generating fallback radiation data...")
           
            pacific_tz = pytz.timezone("America/Los_Angeles")
            tomorrow = datetime.now(pacific_tz) + timedelta(days=1)
           
            # Generate 24 hours of reasonable radiation data
            fallback_data = []
           
            for hour in range(24):
                # Simple radiation pattern (0 at night, peak around noon)
                if 6 <= hour <= 18:  # Daylight hours
                    # Peak radiation around noon (hour 12)
                    radiation = 800 * max(0, (1 - abs(hour - 12) / 6))
                else:
                    radiation = 0.0  # No radiation at night
               
                fallback_data.append({
                    'date': tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0),
                    'shortwave_radiation': round(radiation, 1),
                    'Year': tomorrow.year,
                    'Month': tomorrow.month,
                    'Day': tomorrow.day,
                    'Hour': hour
                })
           
            df = pd.DataFrame(fallback_data)
           
            logger.info(f"✓ Generated fallback radiation data: {len(df)} records")
           
            return df
           
        except Exception as e:
            logger.error(f"Failed to generate fallback radiation data: {str(e)}")
            return None

    def validate_radiation_data(self, radiation_df: pd.DataFrame) -> bool:
        """
        Validate radiation data quality
        """
       
        try:
            if radiation_df is None or len(radiation_df) == 0:
                logger.error("Radiation data is empty")
                return False
           
            required_columns = ['shortwave_radiation', 'Year', 'Month', 'Day', 'Hour']
            missing_columns = [col for col in required_columns if col not in radiation_df.columns]
           
            if missing_columns:
                logger.error(f"Radiation data missing required columns: {missing_columns}")
                return False
           
            # Check radiation range (should be non-negative)
            radiation_min = radiation_df['shortwave_radiation'].min()
            radiation_max = radiation_df['shortwave_radiation'].max()
           
            if radiation_min < 0:
                logger.warning(f"Radiation data has negative values: min = {radiation_min}")
           
            if radiation_max > 1500:  # Very high but possible
                logger.warning(f"Radiation data has very high values: max = {radiation_max}")
           
            # Check for null values
            null_count = radiation_df['shortwave_radiation'].isnull().sum()
            if null_count > 0:
                logger.warning(f"Radiation data has {null_count} null values")
           
            # Check for reasonable daily pattern (should have some zero values at night)
            night_hours = radiation_df[radiation_df['Hour'].isin([0, 1, 2, 3, 4, 5, 22, 23])]
            if len(night_hours) > 0:
                night_radiation = night_hours['shortwave_radiation'].mean()
                if night_radiation > 50:  # Should be near zero at night
                    logger.warning(f"Unexpected radiation at night: average = {night_radiation}")
           
            logger.info("✓ Radiation data validation passed")
            return True
           
        except Exception as e:
            logger.error(f"Radiation data validation failed: {str(e)}")
            return False

    def get_radiation_statistics(self, radiation_df: pd.DataFrame) -> dict:
        """
        Generate radiation data statistics for monitoring
        """
       
        try:
            if radiation_df is None or len(radiation_df) == 0:
                return {'error': 'No radiation data available'}
           
            stats = {
                'total_records': len(radiation_df),
                'min_radiation': float(radiation_df['shortwave_radiation'].min()),
                'max_radiation': float(radiation_df['shortwave_radiation'].max()),
                'mean_radiation': float(radiation_df['shortwave_radiation'].mean()),
                'total_daily_radiation': float(radiation_df['shortwave_radiation'].sum()),
                'peak_hour': int(radiation_df.loc[radiation_df['shortwave_radiation'].idxmax(), 'Hour']),
                'hours_with_radiation': int((radiation_df['shortwave_radiation'] > 0).sum()),
                'null_values': int(radiation_df['shortwave_radiation'].isnull().sum())
            }
           
            return stats
           
        except Exception as e:
            logger.error(f"Failed to generate radiation statistics: {str(e)}")
            return {'error': str(e)}
