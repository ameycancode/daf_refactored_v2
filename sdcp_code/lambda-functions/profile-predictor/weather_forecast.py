"""
Weather Forecast Module
lambda-functions/profile-predictor/weather_forecast.py

Weather data fetching logic converted from container T_forecast.py
Handles weather API integration and data preprocessing
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from io import StringIO
from typing import Optional

logger = logging.getLogger(__name__)

class WeatherForecast:
    """
    Weather forecast functionality converted from container T_forecast.py
    """
   
    def __init__(self, data_bucket: str, s3_utils):
        """Initialize weather forecast"""
        self.data_bucket = data_bucket
        self.s3_utils = s3_utils
       
        # Hard-coded latitude and longitude for San Diego (same as container)
        self.latitude = 32.7157
        self.longitude = -117.1611
       
        logger.info("Weather forecast initialized for San Diego")

    def fetch_weather_forecast(self) -> Optional[pd.DataFrame]:
        """
        Fetch hourly weather forecast data for tomorrow
        Converted from container T_forecast.py logic
        """
       
        try:
            logger.info("Fetching weather forecast data...")
           
            # Try to load existing forecast from S3 first
            existing_forecast = self._load_existing_forecast()
            if existing_forecast is not None:
                logger.info("✓ Using existing weather forecast from S3")
                return existing_forecast
           
            # Fetch new forecast from API (same logic as container)
            logger.info("Fetching new weather forecast from API...")
           
            # Step 1: Get the forecast grid point for the location (same as container)
            points_url = f"https://api.weather.gov/points/{self.latitude},{self.longitude}"
            points_response = requests.get(points_url, timeout=30)
            points_response.raise_for_status()
            points_data = points_response.json()
           
            # Step 2: Get the hourly forecast URL (same as container)
            forecast_hourly_url = points_data['properties']['forecastHourly']
           
            # Step 3: Fetch the hourly forecast data (same as container)
            forecast_response = requests.get(forecast_hourly_url, timeout=30)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
           
            # Step 4: Parse the hourly forecast data (same as container)
            hourly_forecast = []
            for period in forecast_data['properties']['periods']:
                hourly_forecast.append({
                    'TradeDateTime': pd.to_datetime(period['startTime'], utc=True),
                    'Temperature': period['temperature'],
                    'CloudCover': period.get('shortForecast', '')
                })
           
            # Step 5: Convert to DataFrame (same as container)
            temperature_forecast_df = pd.DataFrame(hourly_forecast)
           
            # Step 6: Convert UTC to Pacific Time (same as container)
            pacific_tz = pytz.timezone("America/Los_Angeles")
            temperature_forecast_df['TradeDateTime'] = temperature_forecast_df['TradeDateTime'].dt.tz_convert(pacific_tz)
           
            # Step 7: Filter data for tomorrow (same as container)
            now_pacific = datetime.now(pacific_tz)
            tomorrow = now_pacific.date() + timedelta(days=1)
           
            tomorrow_forecast_df = temperature_forecast_df[
                temperature_forecast_df['TradeDateTime'].dt.date == tomorrow
            ][['TradeDateTime', 'Temperature']].head(24)
           
            if len(tomorrow_forecast_df) == 0:
                logger.warning("No weather data available for tomorrow, using current day data")
                today = now_pacific.date()
                tomorrow_forecast_df = temperature_forecast_df[
                    temperature_forecast_df['TradeDateTime'].dt.date == today
                ][['TradeDateTime', 'Temperature']].head(24)
           
            # Add time components for merging (same as container logic)
            tomorrow_forecast_df = self._add_time_components(tomorrow_forecast_df)
           
            # Save to S3 (same as container)
            self._save_forecast_to_s3(tomorrow_forecast_df, now_pacific)
           
            logger.info(f"✓ Weather forecast fetched successfully: {len(tomorrow_forecast_df)} data points")
           
            return tomorrow_forecast_df
           
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed: {str(e)}")
            return self._get_fallback_weather_data()
           
        except Exception as e:
            logger.error(f"Weather forecast fetch failed: {str(e)}")
            return self._get_fallback_weather_data()

    def _load_existing_forecast(self) -> Optional[pd.DataFrame]:
        """
        Load existing weather forecast from S3 if available and recent
        """
       
        try:
            pacific_tz = pytz.timezone("America/Los_Angeles")
            today_str = datetime.now(pacific_tz).strftime("%Y%m%d")
           
            s3_key = f"archived_folders/forecasting/data/weather/T_{today_str}.csv"
           
            logger.info(f"Checking for existing forecast: s3://{self.data_bucket}/{s3_key}")
           
            obj_content = self.s3_utils.get_object(bucket=self.data_bucket, key=s3_key)
            df = pd.read_csv(StringIO(obj_content))
           
            # Add time components for consistency
            df['TradeDateTime'] = pd.to_datetime(df['TradeDateTime'])
            df = self._add_time_components(df)
           
            logger.info(f"✓ Loaded existing weather forecast: {len(df)} records")
            return df
           
        except Exception as e:
            logger.debug(f"Could not load existing weather forecast: {str(e)}")
            return None

    def _add_time_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Year, Month, Day, Hour components for data merging
        """
       
        try:
            df = df.copy()
           
            # Ensure TradeDateTime is in Pacific time without timezone info
            if df['TradeDateTime'].dt.tz is not None:
                df['TradeDateTime'] = df['TradeDateTime'].dt.tz_localize(None)
           
            # Add time components
            df['Year'] = df['TradeDateTime'].dt.year
            df['Month'] = df['TradeDateTime'].dt.month
            df['Day'] = df['TradeDateTime'].dt.day
            df['Hour'] = df['TradeDateTime'].dt.hour
           
            logger.info("✓ Added time components to weather data")
           
            return df
           
        except Exception as e:
            logger.error(f"Failed to add time components: {str(e)}")
            return df

    def _save_forecast_to_s3(self, forecast_df: pd.DataFrame, current_time: datetime):
        """
        Save weather forecast to S3 for future use
        Same logic as container
        """
       
        try:
            today_str = current_time.strftime("%Y%m%d")
            s3_key = f"archived_folders/forecasting/data/weather/T_{today_str}.csv"
           
            # Prepare data for saving (same format as container)
            save_df = forecast_df.copy()
           
            # Format TradeDateTime for saving
            if 'TradeDateTime' in save_df.columns:
                save_df['TradeDateTime'] = save_df['TradeDateTime'].dt.strftime('%-m/%-d/%Y %H:%M')
           
            # Convert to CSV and upload
            csv_buffer = StringIO()
            save_df[['TradeDateTime', 'Temperature']].to_csv(csv_buffer, index=False)
           
            self.s3_utils.put_object(
                bucket=self.data_bucket,
                key=s3_key,
                body=csv_buffer.getvalue(),
                content_type='text/csv'
            )
           
            logger.info(f"✓ Weather forecast saved to s3://{self.data_bucket}/{s3_key}")
           
        except Exception as e:
            logger.warning(f"Could not save weather forecast to S3: {str(e)}")

    def _get_fallback_weather_data(self) -> Optional[pd.DataFrame]:
        """
        Generate fallback weather data when API is unavailable
        """
       
        try:
            logger.info("Generating fallback weather data...")
           
            pacific_tz = pytz.timezone("America/Los_Angeles")
            tomorrow = datetime.now(pacific_tz) + timedelta(days=1)
           
            # Generate 24 hours of reasonable temperature data
            fallback_data = []
            base_temp = 70.0  # Reasonable base temperature for San Diego
           
            for hour in range(24):
                # Simple temperature variation (cooler at night, warmer during day)
                temp_variation = 10 * (0.5 + 0.5 * pd.np.sin((hour - 6) * pd.np.pi / 12))
                temperature = base_temp + temp_variation
               
                fallback_data.append({
                    'TradeDateTime': tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0),
                    'Temperature': round(temperature, 1),
                    'Year': tomorrow.year,
                    'Month': tomorrow.month,
                    'Day': tomorrow.day,
                    'Hour': hour
                })
           
            df = pd.DataFrame(fallback_data)
           
            logger.info(f"✓ Generated fallback weather data: {len(df)} records")
           
            return df
           
        except Exception as e:
            logger.error(f"Failed to generate fallback weather data: {str(e)}")
            return None

    def validate_weather_data(self, weather_df: pd.DataFrame) -> bool:
        """
        Validate weather data quality
        """
       
        try:
            if weather_df is None or len(weather_df) == 0:
                logger.error("Weather data is empty")
                return False
           
            required_columns = ['Temperature', 'Year', 'Month', 'Day', 'Hour']
            missing_columns = [col for col in required_columns if col not in weather_df.columns]
           
            if missing_columns:
                logger.error(f"Weather data missing required columns: {missing_columns}")
                return False
           
            # Check temperature range
            temp_min = weather_df['Temperature'].min()
            temp_max = weather_df['Temperature'].max()
           
            if temp_min < -50 or temp_max > 150:
                logger.warning(f"Weather temperatures outside reasonable range: {temp_min}°F to {temp_max}°F")
           
            # Check for null values
            null_count = weather_df['Temperature'].isnull().sum()
            if null_count > 0:
                logger.warning(f"Weather data has {null_count} null temperature values")
           
            logger.info("✓ Weather data validation passed")
            return True
           
        except Exception as e:
            logger.error(f"Weather data validation failed: {str(e)}")
            return False
