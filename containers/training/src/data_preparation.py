import pandas as pd

def encoding_train_test_split(df, train_cutoff):
    """
    Encodes categorical variables (Weekday, Season), splits the data into training and testing sets.

    Args:
        df (DataFrame): Input DataFrame containing 'Weekday', 'Season', 'Load', 'Profile', 'TradeDate', and 'Time'.
        train_cutoff (str): Cutoff date for training data in 'YYYY-MM-DD' format.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, train_cutoff)
    """
    train_cutoff = pd.to_datetime(train_cutoff)

    # Encoding categorical variables safely
    weekday_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                   'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    season_map = {'Summer': 1, 'Winter': 0}
   
    if not df['Weekday'].isin(weekday_map.keys()).all():
        raise ValueError("Unexpected values found in 'Weekday' column.")
    if not df['Season'].isin(season_map.keys()).all():
        raise ValueError("Unexpected values found in 'Season' column.")
   
    df['Weekday'] = df['Weekday'].map(weekday_map)
    df['Season'] = df['Season'].map(season_map)

    # Define features and target
    y = df['Load']
    X = df.drop(columns=['Load', 'Profile', 'TradeDate', 'Load_I'])

    # Split into train and test sets based on cutoff
    X_train = X[X['Time'] < train_cutoff].drop(columns=['Time']).copy()
    y_train = y[X['Time'] < train_cutoff].copy()
    X_test = X[X['Time'] >= train_cutoff].drop(columns=['Time']).copy()
    y_test = y[X['Time'] >= train_cutoff].copy()

    # Reset indices
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test, train_cutoff
