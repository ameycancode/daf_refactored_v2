# Model evaluation - Use from previous artifactsfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates the model on the train and test sets and calculates performance metrics.
    
    Parameters:
        model: XGBRegressor
            The trained model.
        X_train: DataFrame
            Training features.
        y_train: Series
            Training target.
        X_test: DataFrame
            Testing features.
        y_test: Series
            Testing target.
    
    Returns:
        metrics: dict
            A dictionary containing evaluation metrics for both train and test sets.
        y_train_pred: ndarray
            Predicted values for the train set.
        y_test_pred: ndarray
            Predicted values for the test set.
    """
    # Predict on train data
    y_train_pred = model.predict(X_train)

    # Predict on test data
    y_test_pred = model.predict(X_test)

    # Compute evaluation metrics for the train set
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred) * 100  # Convert to percentage

    # Compute evaluation metrics for the test set
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred) * 100  # Convert to percentage

    # Store all metrics in a dictionary
    metrics = {
        'RMSE_Train': round(rmse_train, 4),
        'RMSE_Test': round(rmse_test, 4),
        'MAPE_Train': round(mape_train, 4),
        'MAPE_Test': round(mape_test, 4),
        'MSE_Train': round(mse_train, 4),
        'MSE_Test': round(mse_test, 4),
        'MAE_Train': round(mae_train, 4),
        'MAE_Test': round(mae_test, 4),
        'R²_Train': round(r2_train, 4),
        'R²_Test': round(r2_test, 4)
    }

    return metrics, y_train_pred, y_test_pred


def visualize_results(df, y_train, y_train_pred, y_test, y_test_pred, train_cutoff, dataset_name, plot_type):
    """
    Generates interactive plots for actual vs. predicted values and static residual plots.

    Parameters:
        df (DataFrame): The original dataset containing 'Time' and 'Hour' columns.
        y_train (Series or ndarray): Actual training target values.
        y_train_pred (ndarray): Predicted training target values.
        y_test (Series or ndarray): Actual testing target values.
        y_test_pred (ndarray): Predicted testing target values.
        train_cutoff (str or Timestamp): Cutoff date for train-test split.
        dataset_name (str): Title prefix for the plots.
        plot_type (str): Type of plot - 'both', 'actual_vs_predicted', or 'residuals'.

    Returns:
        None (Displays plots)
    """
    
    # Validate plot_type
    valid_plot_types = ['both', 'actual_vs_predicted', 'residuals']
    if plot_type not in valid_plot_types:
        raise ValueError(f"Invalid plot_type: {plot_type}. Use one of {valid_plot_types}")

    # Align data using the original DataFrame's index
    train_mask = df['Time'] < train_cutoff
    test_mask = df['Time'] >= train_cutoff

    train_time = df.loc[train_mask, 'Time'].reset_index(drop=True)
    test_time = df.loc[test_mask, 'Time'].reset_index(drop=True)

    # Plot actual vs predicted
    if plot_type in ['both', 'actual_vs_predicted']:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_time, y=y_train, mode='lines', name='Actual Train Load', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=train_time, y=y_train_pred, mode='lines', name='Predicted Train Load', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=test_time, y=y_test, mode='lines', name='Actual Test Load', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=test_time, y=y_test_pred, mode='lines', name='Predicted Test Load', line=dict(color='red', dash='dash')))

        # Add vertical lines for end of day
        end_of_day_times = df[(df['Time'] >= train_cutoff) & (df['Hour'] == 0)]['Time']
        for time in end_of_day_times:
            fig.add_vline(x=time, line_dash='dash', line_color='gray', opacity=0.5)

        fig.update_layout(
            title=f'{dataset_name} - Actual vs Predicted Load',
            xaxis_title='Time',
            yaxis_title='Load (kWh)',
            template='plotly_white'
        )
        fig.show()
    # Plot residuals
    if plot_type in ['both', 'residuals']:
        #residuals_test = y_test.values.flatten() - y_test_pred
        residuals_test = y_test.values - y_test_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_pred, residuals_test, alpha=0.5, edgecolors='k')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title(f'{dataset_name} - Residuals Plot')
        plt.xlabel('Predicted Load (kWh)')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()



def plot_feature_importance(model, dataset_name):
    """
    Plots the top 10 features based on importance.
    
    Parameters:
        model: XGBRegressor
            The trained model.
        dataset_name: str
            Name of the dataset being processed.
    """
    # Get feature importance scores
    importance = model.get_booster().get_score(importance_type='weight')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)  # Sort by importance

    # Select the top 10 features
    top_10_features = importance[:10]
    top_10_names, top_10_scores = zip(*top_10_features)

    # Plot the top 10 features
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(top_10_names)), top_10_scores, align='center', height=0.5)
    plt.yticks(np.arange(len(top_10_names)), top_10_names, fontsize=10)
    plt.title(f'Top 10 Feature Importance - {dataset_name}', fontsize=14)
    plt.xlabel('F score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
    plt.show(block=False)  
