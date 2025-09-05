"""
Enhanced Weather Forecaster for Lambda
Combines best practices from standalone and lambda weather modules
Includes caching, error handling, and environment awareness
"""

import logging
import requests
import pandas as pd
import numpy as np
import boto3
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import pytz
from io import StringIO
import time

logger = logging.getLogger(__name__)

class LambdaWeatherForecaster:
    """
    Enhanced weather forecaster combining standalone and lambda best practices
    """
    
    def __init__(self, s3_config: Dict[str, str]):
        self.s3_config = s3_config
        self.s3_client = boto3.client('s3')
        
        # San Diego coordinates (consistent with standalone)
        self.latitude = 32.7157
        self.longitude = -117.1611
        self.pacific_tz = pytz.timezone("America/Los_Angeles")
        
        # API configuration
        self.api_timeout = 30
        self.max_retries = 3
        self.cache_duration_hours = 6

    def fetch_weather_and_radiation(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch both weather and radiation data efficiently
        """
        
        try:
            logger.info("Fetching weather and radiation forecasts")
            
            # Try to use cached data first
            weather_df = self._load_cached_weather_forecast()
            radiation_df = self._load_cached_radiation_forecast()
            
            # Fetch weather if not cached or stale
            if weather_df is None:
                weather_df = self._fetch_fresh_weather_forecast()
                if weather_df is not None:
                    self._cache_weather_forecast(weather_df)
            
            # Fetch radiation if not cached or stale
            if radiation_df is None:
                radiation_df = self._fetch_fresh_radiation_forecast()
                if radiation_df is not None:
                    self._cache_radiation_forecast(radiation_df)
            
            # Validate both datasets
            weather_valid = weather_df is not None and len(weather_df) >= 24
            radiation_valid = radiation_df is not None and len(radiation_df) >= 24
            
            if not weather_valid:
                logger.warning("Using fallback weather data")
                weather_df = self._generate_fallback_weather()
            
            if not radiation_valid:
                logger.warning("Using fallback radiation data")
                radiation_df = self._generate_fallback_radiation()
            
            logger.info(f"Weather forecast: {len(weather_df) if weather_df is not None else 0} hours")
            logger.info(f"Radiation forecast: {len(radiation_df) if radiation_df is not None else 0} hours")
            
            return weather_df, radiation_df
            
        except Exception as e:
            logger.error(f"Failed to fetch weather and radiation: {str(e)}")
            # Return fallback data
            return self._generate_fallback_weather(), self._generate_fallback_radiation()

    def _load_cached_weather_forecast(self) -> Optional[pd.DataFrame]:
        """
        Load cached weather forecast from S3 if recent enough
        """
        
        try:
            today_str = datetime.now(self.pacific_tz).strftime("%Y%m%d")
            s3_key = f"{self.s3_config['input_prefix']}temperature/T_{today_str}.csv"
            
            # Check if file exists and is recent
            try:
                response = self.s3_client.head_object(
                    Bucket=self.s3_config['data_bucket'], 
                    Key=s3_key
                )
                
                # Check if file is recent enough
                last_modified = response['LastModified']
                age_hours = (datetime.now(pytz.UTC) - last_modified).total_seconds() / 3600
                
                if age_hours > self.cache_duration_hours:
                    logger.info(f"Cached weather data is {age_hours:.1f} hours old, fetching fresh data")
                    return None
                
            except Exception:
                # File doesn't exist
                return None
            
            # Load the cached file
            response = self.s3_client.get_object(
                Bucket=self.s3_config['data_bucket'], 
                Key=s3_key
            )
            df = pd.read_csv(response['Body'])
            
            # Validate cached data
            if len(df) >= 24 and 'TradeDateTime' in df.columns and 'Temperature' in df.columns:
                logger.info(f"Using cached weather forecast from S3: {s3_key}")
                df['TradeDateTime'] = pd.to_datetime(df['TradeDateTime'])
                return df
            
        except Exception as e:
            logger.debug(f"Could not load cached weather data: {str(e)}")
        
        return None

    def _load_cached_radiation_forecast(self) -> Optional[pd.DataFrame]:
        """
        Load cached radiation forecast from S3 if recent enough
        """
        
        try:
            today_str = datetime.now(self.pacific_tz).strftime("%Y%m%d")
            s3_key = f"{self.s3_config['input_prefix']}radiation/shortwave_radiation_{today_str}.csv"
            
            # Check if file exists and is recent
            try:
                response = self.s3_client.head_object(
                    Bucket=self.s3_config['data_bucket'], 
                    Key=s3_key
                )
                
                # Check if file is recent enough
                last_modified = response['LastModified']
                age_hours = (datetime.now(pytz.UTC) - last_modified).total_seconds() / 3600
                
                if age_hours > self.cache_duration_hours:
                    logger.info(f"Cached radiation data is {age_hours:.1f} hours old, fetching fresh data")
                    return None
                
            except Exception:
                # File doesn't exist
                return None
            
            # Load the cached file
            response = self.s3_client.get_object(
                Bucket=self.s3_config['data_bucket'], 
                Key=s3_key
            )
            df = pd.read_csv(response['Body'])
            
            # Validate cached data
            if len(df) >= 24 and 'date' in df.columns and 'shortwave_radiation' in df.columns:
                logger.info(f"Using cached radiation forecast from S3: {s3_key}")
                df['date'] = pd.to_datetime(df['date'])
                return df
            
        except Exception as e:
            logger.debug(f"Could not load cached radiation data: {str(e)}")
        
        return None

    def _fetch_fresh_weather_forecast(self) -> Optional[pd.DataFrame]:
        """
        Fetch fresh weather forecast from Weather.gov API with retry logic
        """
        
        try:
            logger.info("Fetching fresh weather forecast from Weather.gov")
            
            # Get forecast grid point with retry
            points_data = self._api_request_with_retry(
                f"https://api.weather.gov/points/{self.latitude},{self.longitude}"
            )
            
            if not points_data:
                raise Exception("Failed to get forecast grid point")
            
            # Get hourly forecast with retry
            forecast_hourly_url = points_data['properties']['forecastHourly']
            forecast_data = self._api_request_with_retry(forecast_hourly_url)
            
            if not forecast_data:
                raise Exception("Failed to get hourly forecast")
            
            # Parse forecast data (enhanced from standalone logic)
            hourly_forecast = []
            for period in forecast_data['properties']['periods']:
                try:
                    hourly_forecast.append({
                        'TradeDateTime': pd.to_datetime(period['startTime'], utc=True),
                        'Temperature': float(period['temperature']),
                        'WindSpeed': self._parse_wind_speed(period.get('windSpeed', '0 mph')),
                        'Humidity': period.get('relativeHumidity', {}).get('value', 50)
                    })
                except Exception as e:
                    logger.debug(f"Skipping malformed forecast period: {str(e)}")
                    continue
            
            if not hourly_forecast:
                raise Exception("No valid forecast periods found")
            
            # Create DataFrame
            weather_df = pd.DataFrame(hourly_forecast)
            weather_df['TradeDateTime'] = weather_df['TradeDateTime'].dt.tz_convert(self.pacific_tz)
            
            # Filter for tomorrow's data
            tomorrow = datetime.now(self.pacific_tz).date() + timedelta(days=1)
            tomorrow_weather = weather_df[
                weather_df['TradeDateTime'].dt.date == tomorrow
            ][['TradeDateTime', 'Temperature']].head(24)
            
            if len(tomorrow_weather) < 24:
                logger.warning(f"Only got {len(tomorrow_weather)} hours of weather data, expected 24")
                # Extend with interpolated data if needed
                tomorrow_weather = self._extend_weather_data(tomorrow_weather, 24)
            
            logger.info(f"Successfully fetched {len(tomorrow_weather)} hours of weather data")
            return tomorrow_weather
            
        except Exception as e:
            logger.error(f"Failed to fetch fresh weather forecast: {str(e)}")
            return None

    def _fetch_fresh_radiation_forecast(self) -> Optional[pd.DataFrame]:
        """
        Fetch fresh radiation forecast from Open-Meteo API with retry logic
        """
        
        try:
            logger.info("Fetching fresh radiation forecast from Open-Meteo")
            
            # Setup request parameters
            tomorrow = datetime.now(self.pacific_tz).date() + timedelta(days=1)
            tomorrow_str = tomorrow.strftime('%Y-%m-%d')
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ["shortwave_radiation"],
                "temperature_unit": "fahrenheit",
                "timezone": "America/Los_Angeles",
                "start_date": tomorrow_str,
                "end_date": tomorrow_str
            }
            
            # Make API request with retry
            response_data = self._api_request_with_retry(url, params=params)
            
            if not response_data:
                raise Exception("Failed to get radiation forecast from Open-Meteo")
            
            # Parse radiation data
            hourly_data = response_data.get('hourly', {})
            times = hourly_data.get('time', [])
            radiation_values = hourly_data.get('shortwave_radiation', [])
            
            if not times or not radiation_values:
                raise Exception("No radiation data in API response")
            
            # Create DataFrame
            radiation_df = pd.DataFrame({
                'date': [datetime.fromisoformat(t) for t in times],
                'shortwave_radiation': radiation_values
            })
            
            # Validate data quality
            if len(radiation_df) < 24:
                logger.warning(f"Only got {len(radiation_df)} hours of radiation data, expected 24")
                radiation_df = self._extend_radiation_data(radiation_df, 24)
            
            # Clean up any negative radiation values
            radiation_df['shortwave_radiation'] = radiation_df['shortwave_radiation'].clip(lower=0)
            
            logger.info(f"Successfully fetched {len(radiation_df)} hours of radiation data")
            return radiation_df
            
        except Exception as e:
            logger.error(f"Failed to fetch fresh radiation forecast: {str(e)}")
            return None

    def _api_request_with_retry(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make API request with retry logic and error handling
        """
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"API request attempt {attempt + 1}/{self.max_retries}: {url}")
                
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=self.api_timeout,
                    headers={'User-Agent': 'Energy Forecasting Lambda/1.0'}
                )
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.Timeout:
                logger.warning(f"API request timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error in API request: {str(e)}")
                break
        
        return None

    def _cache_weather_forecast(self, weather_df: pd.DataFrame) -> None:
        """
        Cache weather forecast to S3
        """
        
        try:
            today_str = datetime.now(self.pacific_tz).strftime("%Y%m%d")
            s3_key = f"{self.s3_config['input_prefix']}temperature/T_{today_str}.csv"
            
            # Format datetime for CSV consistency
            weather_save = weather_df.copy()
            weather_save['TradeDateTime'] = weather_save['TradeDateTime'].dt.strftime('%-m/%-d/%Y %H:%M')
            
            # Save to S3
            csv_buffer = StringIO()
            weather_save.to_csv(csv_buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=self.s3_config['data_bucket'],
                Key=s3_key,
                Body=csv_buffer.getvalue(),
                Metadata={
                    'forecast_type': 'temperature',
                    'generated_timestamp': datetime.now().isoformat(),
                    'data_hours': str(len(weather_df))
                }
            )
            
            logger.info(f"Weather forecast cached to S3: {s3_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache weather forecast: {str(e)}")

    def _cache_radiation_forecast(self, radiation_df: pd.DataFrame) -> None:
        """
        Cache radiation forecast to S3
        """
        
        try:
            today_str = datetime.now(self.pacific_tz).strftime("%Y%m%d")
            s3_key = f"{self.s3_config['input_prefix']}radiation/shortwave_radiation_{today_str}.csv"
            
            # Format datetime for CSV consistency
            radiation_save = radiation_df.copy()
            radiation_save['date'] = radiation_save['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to S3
            csv_buffer = StringIO()
            radiation_save.to_csv(csv_buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=self.s3_config['data_bucket'],
                Key=s3_key,
                Body=csv_buffer.getvalue(),
                Metadata={
                    'forecast_type': 'radiation',
                    'generated_timestamp': datetime.now().isoformat(),
                    'data_hours': str(len(radiation_df))
                }
            )
            
            logger.info(f"Radiation forecast cached to S3: {s3_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache radiation forecast: {str(e)}")

    def _generate_fallback_weather(self) -> pd.DataFrame:
        """
        Generate fallback weather data using historical patterns
        """
        
        try:
            logger.info("Generating fallback weather data")
            
            tomorrow = datetime.now(self.pacific_tz) + timedelta(days=1)
            
            # Generate 24 hours of weather data
            weather_data = []
            
            for hour in range(24):
                # Create datetime for this hour
                dt = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                # Generate temperature using seasonal and daily patterns
                base_temp = self._get_seasonal_base_temperature(tomorrow.month)
                daily_variation = 15 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 2 PM
                temperature = base_temp + daily_variation
                
                weather_data.append({
                    'TradeDateTime': dt,
                    'Temperature': round(temperature, 1)
                })
            
            fallback_df = pd.DataFrame(weather_data)
            logger.info(f"Generated fallback weather data: {len(fallback_df)} hours")
            return fallback_df
            
        except Exception as e:
            logger.error(f"Failed to generate fallback weather: {str(e)}")
            # Return minimal fallback
            tomorrow = datetime.now(self.pacific_tz) + timedelta(days=1)
            times = [tomorrow.replace(hour=h, minute=0, second=0, microsecond=0) for h in range(24)]
            return pd.DataFrame({
                'TradeDateTime': times,
                'Temperature': [70.0] * 24  # Constant 70°F
            })

    def _generate_fallback_radiation(self) -> pd.DataFrame:
        """
        Generate fallback radiation data using solar patterns
        """
        
        try:
            logger.info("Generating fallback radiation data")
            
            tomorrow = datetime.now(self.pacific_tz) + timedelta(days=1)
            
            # Generate 24 hours of radiation data
            radiation_data = []
            
            for hour in range(24):
                # Create datetime for this hour
                dt = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                # Generate radiation using solar angle pattern
                if 6 <= hour <= 18:  # Daylight hours
                    # Peak at solar noon (12:00)
                    solar_angle = np.sin(np.pi * (hour - 6) / 12)
                    base_radiation = 800 * solar_angle  # Peak 800 W/m²
                    
                    # Add seasonal adjustment
                    seasonal_factor = self._get_seasonal_radiation_factor(tomorrow.month)
                    radiation = base_radiation * seasonal_factor
                else:
                    radiation = 0  # No radiation at night
                
                radiation_data.append({
                    'date': dt,
                    'shortwave_radiation': max(0, round(radiation, 1))
                })
            
            fallback_df = pd.DataFrame(radiation_data)
            logger.info(f"Generated fallback radiation data: {len(fallback_df)} hours")
            return fallback_df
            
        except Exception as e:
            logger.error(f"Failed to generate fallback radiation: {str(e)}")
            # Return minimal fallback
            tomorrow = datetime.now(self.pacific_tz) + timedelta(days=1)
            times = [tomorrow.replace(hour=h, minute=0, second=0, microsecond=0) for h in range(24)]
            return pd.DataFrame({
                'date': times,
                'shortwave_radiation': [0.0] * 24  # No radiation
            })
 
    def _get_seasonal_base_temperature(self, month: int) -> float:
        """
        Get seasonal base temperature for San Diego
        """
        # San Diego seasonal temperature patterns
        seasonal_temps = {
            1: 60, 2: 62, 3: 64, 4: 67, 5: 70, 6: 73,
            7: 76, 8: 77, 9: 75, 10: 71, 11: 66, 12: 61
        }
        return seasonal_temps.get(month, 70)
 
    def _get_seasonal_radiation_factor(self, month: int) -> float:
        """
        Get seasonal radiation adjustment factor
        """
        # Higher in summer, lower in winter
        seasonal_factors = {
            1: 0.7, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
            7: 1.2, 8: 1.1, 9: 1.0, 10: 0.9, 11: 0.8, 12: 0.7
        }
        return seasonal_factors.get(month, 1.0)
 
    def _parse_wind_speed(self, wind_str: str) -> float:
        """
        Parse wind speed from string format
        """
        try:
            # Extract numeric value from strings like "5 mph" or "5 to 10 mph"
            import re
            numbers = re.findall(r'\d+', wind_str)
            if numbers:
                return float(numbers[0])
            return 0.0
        except Exception:
            return 0.0
 
    def _extend_weather_data(self, weather_df: pd.DataFrame, target_hours: int) -> pd.DataFrame:
        """
        Extend weather data to target number of hours using interpolation
        """
        
        try:
            if len(weather_df) >= target_hours:
                return weather_df.head(target_hours)
            
            # Get the last available data point
            last_row = weather_df.iloc[-1].copy()
            last_time = last_row['TradeDateTime']
            
            # Generate missing hours
            extended_data = weather_df.copy()
            
            for i in range(len(weather_df), target_hours):
                next_hour = last_time + timedelta(hours=i - len(weather_df) + 1)
                
                # Simple interpolation - could be made more sophisticated
                temp_variation = 2 * np.sin(2 * np.pi * next_hour.hour / 24)
                new_temp = last_row['Temperature'] + temp_variation
                
                new_row = pd.DataFrame({
                    'TradeDateTime': [next_hour],
                    'Temperature': [new_temp]
                })
                
                extended_data = pd.concat([extended_data, new_row], ignore_index=True)
            
            logger.info(f"Extended weather data from {len(weather_df)} to {len(extended_data)} hours")
            return extended_data
            
        except Exception as e:
            logger.error(f"Failed to extend weather data: {str(e)}")
            return weather_df
 
    def _extend_radiation_data(self, radiation_df: pd.DataFrame, target_hours: int) -> pd.DataFrame:
        """
        Extend radiation data to target number of hours using solar patterns
        """
        
        try:
            if len(radiation_df) >= target_hours:
                return radiation_df.head(target_hours)
            
            # Get the base date from existing data
            if not radiation_df.empty:
                base_date = radiation_df.iloc[0]['date'].date()
            else:
                base_date = datetime.now(self.pacific_tz).date() + timedelta(days=1)
            
            # Generate missing hours with solar pattern
            extended_data = radiation_df.copy() if not radiation_df.empty else pd.DataFrame()
            
            for hour in range(len(radiation_df), target_hours):
                dt = datetime.combine(base_date, datetime.min.time().replace(hour=hour % 24))
                dt = self.pacific_tz.localize(dt)
                
                # Generate radiation using solar pattern
                if 6 <= hour % 24 <= 18:
                    solar_angle = np.sin(np.pi * ((hour % 24) - 6) / 12)
                    radiation = 600 * solar_angle  # Simplified pattern
                else:
                    radiation = 0
                
                new_row = pd.DataFrame({
                    'date': [dt],
                    'shortwave_radiation': [max(0, radiation)]
                })
                
                extended_data = pd.concat([extended_data, new_row], ignore_index=True)
            
            logger.info(f"Extended radiation data from {len(radiation_df)} to {len(extended_data)} hours")
            return extended_data
            
        except Exception as e:
            logger.error(f"Failed to extend radiation data: {str(e)}")
            return radiation_df

class WeatherDataValidator:
    """
    Validate weather and radiation data quality
    """
    
    @staticmethod
    def validate_weather_data(weather_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate weather data quality
        """
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Check basic structure
            if weather_df.empty:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Weather data is empty")
                return validation_result
            
            # Check required columns
            required_columns = ['TradeDateTime', 'Temperature']
            missing_columns = [col for col in required_columns if col not in weather_df.columns]
            if missing_columns:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing columns: {missing_columns}")
            
            # Check data completeness
            if len(weather_df) < 24:
                validation_result['warnings'].append(f"Only {len(weather_df)} hours of weather data (expected 24)")
            
            # Check temperature range
            if 'Temperature' in weather_df.columns:
                temp_stats = weather_df['Temperature'].describe()
                if temp_stats['min'] < -20 or temp_stats['max'] > 130:
                    validation_result['warnings'].append(
                        f"Temperature outside reasonable range: {temp_stats['min']:.1f}°F to {temp_stats['max']:.1f}°F"
                    )
                
                validation_result['stats']['temperature'] = {
                    'min': float(temp_stats['min']),
                    'max': float(temp_stats['max']),
                    'mean': float(temp_stats['mean']),
                    'std': float(temp_stats['std'])
                }
            
            # Check for missing values
            null_count = weather_df.isnull().sum().sum()
            if null_count > 0:
                validation_result['warnings'].append(f"Found {null_count} missing values")
            
            validation_result['stats']['total_hours'] = len(weather_df)
            validation_result['stats']['completeness'] = (len(weather_df) - null_count) / len(weather_df) if len(weather_df) > 0 else 0
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
 
    @staticmethod
    def validate_radiation_data(radiation_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate radiation data quality
        """
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Check basic structure
            if radiation_df.empty:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Radiation data is empty")
                return validation_result
            
            # Check required columns
            required_columns = ['date', 'shortwave_radiation']
            missing_columns = [col for col in required_columns if col not in radiation_df.columns]
            if missing_columns:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing columns: {missing_columns}")
            
            # Check data completeness
            if len(radiation_df) < 24:
                validation_result['warnings'].append(f"Only {len(radiation_df)} hours of radiation data (expected 24)")
            
            # Check radiation range
            if 'shortwave_radiation' in radiation_df.columns:
                rad_stats = radiation_df['shortwave_radiation'].describe()
                if rad_stats['min'] < 0:
                    validation_result['warnings'].append("Found negative radiation values")
                
                if rad_stats['max'] > 1500:  # Very high radiation
                    validation_result['warnings'].append(f"Very high radiation values detected: {rad_stats['max']:.1f} W/m²")
                
                validation_result['stats']['radiation'] = {
                    'min': float(rad_stats['min']),
                    'max': float(rad_stats['max']),
                    'mean': float(rad_stats['mean']),
                    'std': float(rad_stats['std'])
                }
            
            # Check for missing values
            null_count = radiation_df.isnull().sum().sum()
            if null_count > 0:
                validation_result['warnings'].append(f"Found {null_count} missing values")
            
            validation_result['stats']['total_hours'] = len(radiation_df)
            validation_result['stats']['completeness'] = (len(radiation_df) - null_count) / len(radiation_df) if len(radiation_df) > 0 else 0
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result