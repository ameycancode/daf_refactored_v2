"""
Environment-Aware Prediction Engine for Lambda
Adapted from standalone prediction_engine.py with Redshift output
Handles single profile prediction and storage within Lambda constraints
"""

import logging
import json
import boto3
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pytz
from io import StringIO

logger = logging.getLogger(__name__)

class LambdaPredictionEngine:
    """
    Prediction engine adapted from standalone code for Lambda environment
    Handles SageMaker endpoint invocation and Redshift storage
    """
    
    def __init__(self, redshift_config: Dict[str, str], s3_config: Dict[str, str]):
        self.redshift_config = redshift_config
        self.s3_config = s3_config
        self.sagemaker_client = boto3.client('sagemaker-runtime')
        self.redshift_client = boto3.client('redshift-data', region_name=redshift_config['region'])
        self.s3_client = boto3.client('s3')
        
        # Profile and segment mappings (from standalone prediction_engine.py)
        self.profile_mapping = {
            'RNN': 'RES',
            'RN': 'RES', 
            'M': 'MEDCI',
            'S': 'SMLCOM',
            'AGR': 'AGR',
            'L': 'LIGHT',
            'A6': 'A6'
        }
        
        self.segment_mapping = {
            'RNN': 'NONSOLAR',
            'RN': 'SOLAR',
            'M': 'ALL',
            'S': 'ALL',
            'AGR': 'ALL',
            'L': 'ALL',
            'A6': 'ALL'
        }

    def run_single_profile_prediction(self, profile: str, endpoint_name: str, 
                                    prediction_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete prediction pipeline for single profile
        """
        
        try:
            logger.info(f"Starting prediction pipeline for profile {profile}")
            
            # Step 1: Validate input data
            if not self._validate_prediction_input(prediction_data, profile):
                raise ValueError(f"Invalid prediction input for profile {profile}")
            
            # Step 2: Prepare features for model
            model_features = self._prepare_model_features(prediction_data, profile)
            
            # Step 3: Invoke SageMaker endpoint
            predictions = self._invoke_sagemaker_endpoint(endpoint_name, model_features)
            
            # Step 4: Process and validate predictions
            processed_predictions = self._process_predictions(predictions, profile)
            
            # Step 5: Save to Redshift
            records_saved = self._save_predictions_to_redshift(
                profile, prediction_data, processed_predictions
            )
            
            # Step 6: Generate summary statistics
            prediction_stats = self._calculate_prediction_statistics(processed_predictions)
            
            result = {
                'status': 'success',
                'profile': profile,
                'endpoint_name': endpoint_name,
                'predictions_count': len(processed_predictions),
                'records_saved': records_saved,
                'statistics': prediction_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction pipeline completed for {profile}: {len(processed_predictions)} predictions")
            return result
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed for {profile}: {str(e)}")
            raise

    def _validate_prediction_input(self, data: pd.DataFrame, profile: str) -> bool:
        """
        Validate prediction input data
        """
        
        try:
            # Check for 24 hours of data
            if len(data) != 24:
                logger.error(f"Expected 24 hours of data for {profile}, got {len(data)}")
                return False
            
            # Check required features
            required_features = self._get_required_features(profile)
            missing_features = [f for f in required_features if f not in data.columns]
            
            if missing_features:
                logger.error(f"Missing features for {profile}: {missing_features}")
                return False
            
            # Check for NaN values
            for feature in required_features:
                if data[feature].isna().any():
                    logger.error(f"NaN values found in feature {feature} for {profile}")
                    return False
            
            # Check hour sequence
            expected_hours = set(range(24))
            actual_hours = set(data['Hour'].unique())
            if expected_hours != actual_hours:
                logger.error(f"Incomplete hour sequence for {profile}: missing {expected_hours - actual_hours}")
                return False
            
            logger.info(f"Prediction input validation passed for {profile}")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for {profile}: {str(e)}")
            return False

    def _get_required_features(self, profile: str) -> List[str]:
        """
        Get required features for each profile (from standalone logic)
        """
        
        base_features = [
            'Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season',
            'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'
        ]
        
        if profile == 'RN':
            # RN profile includes radiation data
            return base_features + ['shortwave_radiation']
        else:
            return base_features

    def _prepare_model_features(self, data: pd.DataFrame, profile: str) -> List[List[float]]:
        """
        Prepare features for model inference
        """
        
        try:
            # Get required features in correct order
            feature_columns = self._get_required_features(profile)
            
            # Extract feature values
            feature_data = data[feature_columns].copy()
            
            # Fill any remaining NaN values
            feature_data = feature_data.fillna(0)
            
            # Convert to list format for SageMaker
            model_input = feature_data.values.tolist()
            
            logger.info(f"Prepared model features for {profile}: {len(model_input)} samples, {len(feature_columns)} features")
            return model_input
            
        except Exception as e:
            logger.error(f"Failed to prepare model features for {profile}: {str(e)}")
            raise

    def _invoke_sagemaker_endpoint(self, endpoint_name: str, model_input: List[List[float]]) -> List[float]:
        """
        Invoke SageMaker endpoint for predictions
        """
        
        try:
            logger.info(f"Invoking SageMaker endpoint: {endpoint_name}")
            
            # Prepare payload
            payload = json.dumps(model_input)
            
            # Invoke endpoint
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            # Extract predictions based on response format
            if isinstance(result, dict) and 'predictions' in result:
                predictions = result['predictions']
            elif isinstance(result, list):
                predictions = result
            else:
                predictions = result
            
            # Validate and convert predictions
            if not isinstance(predictions, list):
                raise ValueError(f"Invalid prediction format: expected list, got {type(predictions)}")
            
            predictions = [float(p) for p in predictions]
            
            logger.info(f"Received {len(predictions)} predictions from endpoint {endpoint_name}")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to invoke SageMaker endpoint {endpoint_name}: {str(e)}")
            raise

    def _process_predictions(self, predictions: List[float], profile: str) -> List[float]:
        """
        Process and validate predictions
        """
        
        try:
            # Convert to numpy array for processing
            pred_array = np.array(predictions)
            
            # Basic validation
            if len(pred_array) != 24:
                raise ValueError(f"Expected 24 predictions for {profile}, got {len(pred_array)}")
            
            # Check for invalid values
            if np.any(np.isnan(pred_array)) or np.any(np.isinf(pred_array)):
                logger.warning(f"Found invalid values in predictions for {profile}")
                # Replace invalid values with median
                valid_mask = np.isfinite(pred_array)
                if np.any(valid_mask):
                    median_val = np.median(pred_array[valid_mask])
                    pred_array[~valid_mask] = median_val
                else:
                    pred_array = np.full_like(pred_array, 0.5)  # Default value
            
            # Ensure non-negative predictions
            pred_array = np.maximum(pred_array, 0)
            
            processed_predictions = pred_array.tolist()
            
            logger.info(f"Processed predictions for {profile}: {len(processed_predictions)} values")
            return processed_predictions
            
        except Exception as e:
            logger.error(f"Failed to process predictions for {profile}: {str(e)}")
            raise

    def _save_predictions_to_redshift(self, profile: str, prediction_data: pd.DataFrame, 
                                    predictions: List[float]) -> int:
        """
        Save predictions to Redshift using standalone logic
        """
        
        try:
            logger.info(f"Saving predictions to Redshift for profile {profile}")
            
            # Combine prediction data with predictions (from standalone logic)
            result_df = prediction_data.copy()
            result_df['Predicted_Load'] = predictions
            
            # Calculate total load (from standalone format_combined_data_for_redshift)
            result_df['Load_All'] = result_df['Predicted_Load'] * result_df['Count']
            
            # Create datetime for Redshift
            result_df['tradedatetime'] = pd.to_datetime(result_df[['Year', 'Month', 'Day', 'Hour']])
            
            # Format data for Redshift insertion
            redshift_records = []
            
            for _, row in result_df.iterrows():
                record = {
                    'tradedatetime': row['tradedatetime'],
                    'predicted_load': round(float(row['Predicted_Load']), 6),
                    'count': int(row['Count']),
                    'load_all': round(float(row['Load_All']), 6),
                    'profile': self.profile_mapping.get(profile, profile),
                    'segment': self.segment_mapping.get(profile, 'ALL')
                }
                redshift_records.append(record)
            
            # Insert to Redshift
            records_inserted = self._execute_redshift_insert(redshift_records)
            
            logger.info(f"Successfully saved {records_inserted} records to Redshift for profile {profile}")
            return records_inserted
            
        except Exception as e:
            logger.error(f"Failed to save predictions to Redshift for {profile}: {str(e)}")
            raise

    def _execute_redshift_insert(self, records: List[Dict[str, Any]]) -> int:
        """
        Execute Redshift INSERT using Data API
        """
        
        try:
            if not records:
                return 0
            
            # Build VALUES clause
            values_list = []
            for record in records:
                tradedatetime = record['tradedatetime'].strftime('%Y-%m-%d %H:%M:%S')
                value_tuple = f"""(
                    '{tradedatetime}', 
                    {record['predicted_load']}, 
                    {record['count']}, 
                    {record['load_all']}, 
                    '{record['profile']}', 
                    '{record['segment']}'
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
            response = self.redshift_client.execute_statement(
                ClusterIdentifier=self.redshift_config['cluster_identifier'],
                Database=self.redshift_config['database'],
                DbUser=self.redshift_config['db_user'],
                Sql=insert_sql
            )
            
            query_id = response['Id']
            logger.info(f"Redshift INSERT submitted: {query_id}")
            
            # Wait for completion
            self._wait_for_insert_completion(query_id)
            
            return len(records)
            
        except Exception as e:
            logger.error(f"Failed to execute Redshift INSERT: {str(e)}")
            raise

    def _wait_for_insert_completion(self, query_id: str, max_wait: int = 120):
        """
        Wait for Redshift INSERT completion
        """
        
        waited = 0
        while waited < max_wait:
            try:
                status_response = self.redshift_client.describe_statement(Id=query_id)
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

    def _calculate_prediction_statistics(self, predictions: List[float]) -> Dict[str, float]:
        """
        Calculate prediction statistics
        """
        
        try:
            if not predictions:
                return {}
            
            pred_array = np.array(predictions)
            
            stats = {
                'min': float(np.min(pred_array)),
                'max': float(np.max(pred_array)),
                'mean': float(np.mean(pred_array)),
                'median': float(np.median(pred_array)),
                'std': float(np.std(pred_array)),
                'total': float(np.sum(pred_array)),
                'count': len(predictions)
            }
            
            # Find peak hour
            peak_hour = int(np.argmax(pred_array))
            stats['peak_hour'] = peak_hour
            stats['peak_load'] = float(pred_array[peak_hour])
            
            # Find minimum hour
            min_hour = int(np.argmin(pred_array))
            stats['min_hour'] = min_hour
            stats['min_load'] = float(pred_array[min_hour])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate prediction statistics: {str(e)}")
            return {}

class PredictionResultProcessor:
    """
    Process and format prediction results for various outputs
    """
    
    def __init__(self, s3_config: Dict[str, str]):
        self.s3_config = s3_config
        self.s3_client = boto3.client('s3')

    def save_detailed_results_to_s3(self, profile: str, prediction_data: pd.DataFrame, 
                                   predictions: List[float], execution_id: str) -> Optional[str]:
        """
        Save detailed prediction results to S3 for analysis
        """
        
        try:
            # Combine data with predictions
            result_df = prediction_data.copy()
            result_df['Predicted_Load'] = predictions
            result_df['Load_All'] = result_df['Predicted_Load'] * result_df['Count']
            result_df['TradeDateTime'] = pd.to_datetime(result_df[['Year', 'Month', 'Day', 'Hour']])
            result_df['execution_id'] = execution_id
            
            # Generate S3 key
            today_str = datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%Y%m%d')
            timestamp = datetime.now().strftime('%H%M%S')
            s3_key = f"{self.s3_config['output_prefix']}{profile}_predictions_{today_str}_{timestamp}.csv"
            
            # Save to S3
            csv_buffer = StringIO()
            result_df.to_csv(csv_buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=self.s3_config['data_bucket'],
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            
            logger.info(f"Detailed results saved to S3: s3://{self.s3_config['data_bucket']}/{s3_key}")
            return f"s3://{self.s3_config['data_bucket']}/{s3_key}"
            
        except Exception as e:
            logger.warning(f"Failed to save detailed results to S3: {str(e)}")
            return None

    def generate_prediction_summary(self, profile: str, predictions: List[float], 
                                  statistics: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive prediction summary
        """
        
        try:
            pacific_tz = pytz.timezone('America/Los_Angeles')
            tomorrow = datetime.now(pacific_tz) + timedelta(days=1)
            
            summary = {
                'profile': profile,
                'forecast_date': tomorrow.strftime('%Y-%m-%d'),
                'total_hours': len(predictions),
                'statistics': statistics,
                'profile_details': {
                    'mapped_profile': self._get_profile_mapping(profile),
                    'mapped_segment': self._get_segment_mapping(profile),
                    'description': self._get_profile_description(profile)
                },
                'hourly_breakdown': {
                    'predictions': predictions,
                    'peak_hour': statistics.get('peak_hour', 0),
                    'min_hour': statistics.get('min_hour', 0)
                },
                'generation_timestamp': datetime.now().isoformat(),
                'forecast_period': {
                    'start_hour': '00:00',
                    'end_hour': '23:00',
                    'timezone': 'America/Los_Angeles'
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate prediction summary: {str(e)}")
            return {}

    def _get_profile_mapping(self, profile: str) -> str:
        """Get mapped profile name"""
        mapping = {
            'RNN': 'RES',
            'RN': 'RES', 
            'M': 'MEDCI',
            'S': 'SMLCOM',
            'AGR': 'AGR',
            'L': 'LIGHT',
            'A6': 'A6'
        }
        return mapping.get(profile, profile)

    def _get_segment_mapping(self, profile: str) -> str:
        """Get mapped segment name"""
        mapping = {
            'RNN': 'NONSOLAR',
            'RN': 'SOLAR',
            'M': 'ALL',
            'S': 'ALL',
            'AGR': 'ALL',
            'L': 'ALL',
            'A6': 'ALL'
        }
        return mapping.get(profile, 'ALL')

    def _get_profile_description(self, profile: str) -> str:
        """Get profile description"""
        descriptions = {
            'RNN': 'Residential Non-NEM customers',
            'RN': 'Residential NEM (solar) customers',
            'M': 'Medium Commercial/Industrial customers',
            'S': 'Small Commercial customers',
            'AGR': 'Agricultural customers',
            'L': 'Lighting customers',
            'A6': 'A6 Rate Group customers'
        }
        return descriptions.get(profile, f'{profile} customers')

class PredictionErrorHandler:
    """
    Handle prediction errors and fallback scenarios
    """
    
    @staticmethod
    def generate_fallback_predictions(profile: str, prediction_data: pd.DataFrame) -> List[float]:
        """
        Generate fallback predictions when model endpoint fails
        """
        
        try:
            logger.warning(f"Generating fallback predictions for {profile}")
            
            # Use simple heuristics based on profile type and time of day
            fallback_predictions = []
            
            for _, row in prediction_data.iterrows():
                hour = row['Hour']
                temperature = row.get('Temperature', 70)
                weekday = row.get('Weekday', 0)
                
                # Base load pattern by profile
                base_patterns = {
                    'RNN': PredictionErrorHandler._residential_pattern(hour, temperature, weekday),
                    'RN': PredictionErrorHandler._residential_solar_pattern(hour, temperature, weekday),
                    'M': PredictionErrorHandler._commercial_pattern(hour, weekday),
                    'S': PredictionErrorHandler._small_commercial_pattern(hour, weekday),
                    'AGR': PredictionErrorHandler._agricultural_pattern(hour, weekday),
                    'L': PredictionErrorHandler._lighting_pattern(hour),
                    'A6': PredictionErrorHandler._a6_pattern(hour, weekday)
                }
                
                prediction = base_patterns.get(profile, 0.5)
                fallback_predictions.append(max(0.01, prediction))  # Ensure positive
            
            logger.info(f"Generated {len(fallback_predictions)} fallback predictions for {profile}")
            return fallback_predictions
            
        except Exception as e:
            logger.error(f"Failed to generate fallback predictions: {str(e)}")
            # Return default flat prediction
            return [0.5] * 24

    @staticmethod
    def _residential_pattern(hour: int, temperature: float, weekday: int) -> float:
        """Residential load pattern"""
        # Higher in morning and evening, influenced by temperature
        base_load = 0.3 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Temperature adjustment
        if temperature > 80:
            base_load += 0.2 * (temperature - 80) / 20  # Cooling load
        elif temperature < 50:
            base_load += 0.15 * (50 - temperature) / 30  # Heating load
        
        # Weekend adjustment
        if weekday >= 5:
            base_load *= 1.1  # Slightly higher on weekends
        
        return max(0.1, base_load)

    @staticmethod
    def _residential_solar_pattern(hour: int, temperature: float, weekday: int) -> float:
        """Residential with solar (NEM) pattern"""
        base_load = PredictionErrorHandler._residential_pattern(hour, temperature, weekday)
        
        # Solar generation reduces net load during daylight hours
        if 8 <= hour <= 17:
            solar_reduction = 0.4 * np.sin(np.pi * (hour - 8) / 9)
            base_load = max(0.05, base_load - solar_reduction)
        
        return base_load

    @staticmethod
    def _commercial_pattern(hour: int, weekday: int) -> float:
        """Medium commercial pattern"""
        if weekday < 5:  # Weekday
            if 7 <= hour <= 18:
                return 0.6 + 0.2 * np.sin(np.pi * (hour - 7) / 11)
            else:
                return 0.2
        else:  # Weekend
            return 0.3

    @staticmethod
    def _small_commercial_pattern(hour: int, weekday: int) -> float:
        """Small commercial pattern"""
        return PredictionErrorHandler._commercial_pattern(hour, weekday) * 0.7

    @staticmethod
    def _agricultural_pattern(hour: int, weekday: int) -> float:
        """Agricultural pattern"""
        # More consistent throughout day, some peak during irrigation hours
        base = 0.4
        if 6 <= hour <= 10 or 16 <= hour <= 20:
            base += 0.2  # Irrigation periods
        return base

    @staticmethod
    def _lighting_pattern(hour: int) -> float:
        """Lighting pattern"""
        # Peak during dark hours
        if 6 <= hour <= 18:
            return 0.1  # Minimal during daylight
        else:
            return 0.8  # High during dark hours

    @staticmethod
    def _a6_pattern(hour: int, weekday: int) -> float:
        """A6 rate pattern"""
        # Similar to commercial but with different timing
        return PredictionErrorHandler._commercial_pattern(hour, weekday) * 0.8

class PredictionMonitor:
    """
    Monitor prediction quality and performance
    """
    
    def __init__(self, s3_config: Dict[str, str]):
        self.s3_config = s3_config
        self.s3_client = boto3.client('s3')

    def log_prediction_metrics(self, profile: str, endpoint_name: str, 
                             prediction_stats: Dict[str, float], 
                             execution_time: float) -> None:
        """
        Log prediction metrics for monitoring
        """
        
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'profile': profile,
                'endpoint_name': endpoint_name,
                'execution_time_seconds': execution_time,
                'prediction_statistics': prediction_stats,
                'quality_indicators': self._calculate_quality_indicators(prediction_stats)
            }
            
            # Log to CloudWatch (would need CloudWatch client)
            logger.info(f"Prediction metrics for {profile}: {json.dumps(metrics, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to log prediction metrics: {str(e)}")

    def _calculate_quality_indicators(self, stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate prediction quality indicators
        """
        
        try:
            quality = {}
            
            # Range check
            load_range = stats.get('max', 0) - stats.get('min', 0)
            quality['load_range'] = load_range
            quality['range_reasonable'] = 0.1 <= load_range <= 10.0
            
            # Peak/minimum ratio
            if stats.get('min', 0) > 0:
                peak_ratio = stats.get('max', 1) / stats.get('min', 1)
                quality['peak_to_min_ratio'] = peak_ratio
                quality['ratio_reasonable'] = 1.5 <= peak_ratio <= 20.0
            
            # Standard deviation check
            std_dev = stats.get('std', 0)
            mean_val = stats.get('mean', 1)
            if mean_val > 0:
                cv = std_dev / mean_val  # Coefficient of variation
                quality['coefficient_of_variation'] = cv
                quality['variability_reasonable'] = 0.1 <= cv <= 1.0
            
            # Overall quality score
            quality_checks = [
                quality.get('range_reasonable', False),
                quality.get('ratio_reasonable', False), 
                quality.get('variability_reasonable', False)
            ]
            quality['overall_score'] = sum(quality_checks) / len(quality_checks)
            quality['quality_grade'] = 'GOOD' if quality['overall_score'] >= 0.7 else 'FAIR' if quality['overall_score'] >= 0.5 else 'POOR'
            
            return quality
            
        except Exception as e:
            logger.error(f"Failed to calculate quality indicators: {str(e)}")
            return {}

    def check_prediction_anomalies(self, profile: str, predictions: List[float]) -> List[str]:
        """
        Check for prediction anomalies
        """
        
        anomalies = []
        
        try:
            pred_array = np.array(predictions)
            
            # Check for flat predictions
            if np.std(pred_array) < 0.01:
                anomalies.append("Predictions are unusually flat")
            
            # Check for extreme values
            mean_val = np.mean(pred_array)
            std_val = np.std(pred_array)
            threshold = 3 * std_val
            
            extreme_indices = np.where(np.abs(pred_array - mean_val) > threshold)[0]
            if len(extreme_indices) > 0:
                anomalies.append(f"Extreme values detected at hours: {extreme_indices.tolist()}")
            
            # Check for unrealistic patterns
            if profile in ['RN'] and not any(6 <= i <= 18 and pred_array[i] < mean_val for i in range(24)):
                anomalies.append("Solar profile missing expected daytime reduction")
            
            if profile in ['L'] and not (np.mean(pred_array[6:18]) < np.mean(pred_array[:6] + pred_array[18:])):
                anomalies.append("Lighting profile missing expected day/night pattern")
            
        except Exception as e:
            logger.error(f"Failed to check prediction anomalies: {str(e)}")
            anomalies.append(f"Anomaly detection failed: {str(e)}")
        
        return anomalies
