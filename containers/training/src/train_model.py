import time
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import xgboost as xgb

def train_model(X_train, y_train, cv_splits, xgboost_params = None):
    """
    Trains an XGBoost model using GridSearchCV for hyperparameter tuning and 
    measures the time taken for training.
    
    Parameters:
        X_train: DataFrame
            Training features.
        y_train: Series
            Training target.
        cv_splits: int
            Number of splits for TimeSeriesSplit.
        xgboost_params:Dictionary
            Model parameters. 
    
    Returns:
        best_model: XGBRegressor
            The best estimator found during hyperparameter tuning.
        grid_search: GridSearchCV object
            The GridSearchCV instance containing training details.
        training_time: float
            The total time taken for training (in seconds).
    """
    #Set up parameter grid for XGBoost hyperparameter tuning
    if xgboost_params is None:
        xgboost_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [4, 5, 6, 7, 8, 9, 10],
        #'min_child_weight': [1, 5, 10],
        #'subsample': [0.7, 0.8, 1.0],
        #'colsample_bytree': [0.7, 0.8, 1.0],
        #'colsample_bylevel': [0.7, 0.8, 1.0],
        #'gamma': [0, 0.1, 0.2],
        #'reg_alpha': [0, 0.01, 0.1],
        #'reg_lambda': [1, 1.5, 2],
        #'scale_pos_weight': [1],
        #'max_delta_step': [0, 1, 5],
    }


    # Initialize XGBoost model
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=7)

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Initialize GridSearchCV for model tuning
    grid_search = GridSearchCV(
        estimator=xgboost_model,
        param_grid=xgboost_params,
        #scoring='r2',
        #scoring='neg_mean_absolute_percentage_error',
        scoring= 'neg_mean_absolute_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1  # Utilize all CPU cores
    )

    # Measure the start time
    start_time = time.time()

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Measure the end time
    end_time = time.time()

    # Calculate the total time taken for training
    training_time = end_time - start_time
    training_time_minute = training_time /60
    print(f"Training completed in {training_time_minute:.2f} minute")

    # Get the best model
    best_model = grid_search.best_estimator_

    # Get the best parameters
    best_params = grid_search.best_params_

    # Print the best parameters
    print(f"Best parameters found: {best_params}")

    return best_model, grid_search, best_params, training_time
