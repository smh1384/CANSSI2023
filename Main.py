import pandas as pd
import numpy as np
import os
import sys
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor

# Configuration
LOG_FILE_PATH = ""
DATA_PATH = 'trots_2013-2022.parquet'
OUTPUT_DIR = ''
Z_SCORE_THRESHOLD = 6
PCA_COMPONENTS = 0.95
RANDOM_STATE = 42
NUMBER_OF_FOLDS_CROSS_VALIDATION = 5
Number_of_Desired_important_Feature = 20
# Set up logging
logging.basicConfig(filename=os.path.join(OUTPUT_DIR, 'algorithm_outputs.txt'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def load_and_preprocess_data(filepath):
    df = pd.read_parquet(filepath, engine='pyarrow')
    # df = df.sample(900, random_state=42)
    df = remove_outliers(df)
    df['RaceStartTime'] = pd.to_datetime(df['RaceStartTime'])
    df = calculate_win_probability(df)
    train_df, val_df = split_data(df)
    return train_df, val_df


def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = stats.zscore(df[numeric_cols])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < Z_SCORE_THRESHOLD).all(axis=1)
    return df[filtered_entries]


def calculate_win_probability(df):
    df['winprobability'] = df.groupby('RaceID')['FinishPosition'].transform(
        lambda x: 1 / x.rank(method='min')
    )
    df['winprobability'] = df.groupby('RaceID')['winprobability'].transform(
        lambda x: x / x.sum()
    )
    assert all(df.groupby('RaceID')['winprobability'].sum().round(10) == 1)
    return df


def split_data(df):
    train_df = df[df['RaceStartTime'] < '2021-11-01']
    val_df = df[df['RaceStartTime'] >= '2021-11-01']
    performance_cols = ['BeatenMargin', 'Disqualified', 'FinishPosition', 'PIRPosition',
                        'Prizemoney', 'RaceOverallTime', 'PriceSP', 'NoFrontCover',
                        'PositionInRunning', 'WideOffRail']
    train_df = train_df.drop(columns=performance_cols)
    val_df = val_df.drop(columns=performance_cols)
    return train_df, val_df


def preprocess_features(train_df, val_df):
    # Separate features and target variable
    X_train = train_df.drop(columns=['winprobability'])
    y_train = train_df['winprobability']
    X_val = val_df.drop(columns=['winprobability'])
    y_val = val_df['winprobability']

    # One-hot encode the categorical columns
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)

    # Ensure both training and validation sets have the same columns
    missing_cols = set(X_train.columns) - set(X_val.columns)
    for col in missing_cols:
        X_val[col] = 0
    X_val = X_val[X_train.columns]

    # Drop columns that are not useful
    cols_to_drop = ['SomeColumnName']  # Replace with actual column names to drop
    X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
    X_val = X_val.drop(columns=cols_to_drop, errors='ignore')

    # Convert Timestamp columns to seconds
    reference_time = pd.Timestamp('2000-01-01')
    for column in X_train.select_dtypes(include=[np.datetime64]).columns:
        X_train[column] = (X_train[column] - reference_time).dt.total_seconds()
        X_val[column] = (X_val[column] - reference_time).dt.total_seconds()

    # Replace infinite values with NaN and fill NaN values
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(X_train.mean(), inplace=True)
    X_val.fillna(X_train.mean(), inplace=True)  # Use means from X_train for X_val to prevent data leakage

    return X_train, y_train, X_val, y_val


def compare_algorithms(X_train, y_train, X_val, y_val):
    # The comparison of different algorithms has been performed in a separate script.
    # The best performing model was identified as Light Gradient Boosting Machine (LGBM).
    # Therefore, only the LGBM model is used in this script.
    # For details on the comparison and selection process, please refer to the relevant Python code in the GitHub repository.

    # Logging the information for record-keeping
    logging.info(
        "The best model has been identified as Light Gradient Boosting Machine (LGBM) from a separate algorithm comparison process.")


def train_and_evaluate_lightgbm(X_train, y_train, X_val, y_val):
    # Train and evaluate the LightGBM model
    best_model = LGBMRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    best_model_name = "Light Gradient Boosting"
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_val)

    # Evaluate the LightGBM model
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    model_performance = {
        'LightGBM': {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
    }
    return best_model, best_model_name, model_performance


def apply_pca_and_cross_validation(X_train, y_train, X_val, y_val, best_model, n_components=PCA_COMPONENTS,
                                   n_folds=NUMBER_OF_FOLDS_CROSS_VALIDATION):
    # Normalize features and apply PCA
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    # Create a pipeline that scales the data and then applies PCA
    pipeline = Pipeline([('scaler', scaler), ('pca', pca)])

    # Transform the training dataset
    X_train_transformed = pipeline.fit_transform(X_train)

    # Check if the best model is a pipeline
    if isinstance(best_model, Pipeline):
        final_estimator = best_model.steps[-1][1]  # Extract the final estimator from the pipeline
    else:
        final_estimator = best_model

    # Train the best model with PCA-transformed data
    final_estimator.fit(X_train_transformed, y_train)

    # Transform the validation dataset
    X_val_transformed = pipeline.transform(X_val)
    y_pred = final_estimator.predict(X_val_transformed)

    # Evaluate the model
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    model_performance = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    # Cross-validation with PCA
    # Reconstruct the pipeline with the best model for cross-validation
    if isinstance(best_model, Pipeline):
        pipeline_with_best_model = Pipeline([
            ('scaler', scaler),
            ('pca', pca),
            (best_model.steps[-1][0], final_estimator)
        ])
    else:
        # Ensure that the string used here is a valid name for the estimator
        pipeline_with_best_model = Pipeline([
            ('scaler', scaler),
            ('pca', pca),
            ('final_estimator', final_estimator)  # Use a generic name or the class name of the estimator
        ])

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline_with_best_model, X_train, y_train, cv=n_folds,
                                scoring='neg_mean_squared_error')
    cv_scores_mean = -np.mean(cv_scores)  # Negate to get positive MSE values
    cv_scores_std = np.std(cv_scores)

    # Store cross-validation results
    cv_results = {
        'Mean MSE': cv_scores_mean,
        'RMSE': np.sqrt(cv_scores_mean),
        'Std': cv_scores_std
    }

    return model_performance, cv_results

    return X_train_selected, X_val_selected, selected_features


def feature_selection(X_train, y_train, X_val, num_features=Number_of_Desired_important_Feature):
    # One-hot encode the categorical columns
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)

    # Ensure both training and validation sets have the same columns
    missing_cols = set(X_train.columns) - set(X_val.columns)
    for col in missing_cols:
        X_val[col] = 0
    X_val = X_val[X_train.columns]

    # Normalize features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert the scaled arrays back into DataFrames to retain column names
    train_columns = X_train.columns.tolist()
    X_train = pd.DataFrame(X_train_scaled, columns=train_columns)
    X_val = pd.DataFrame(X_val_scaled, columns=train_columns)

    # Feature Selection using LightGBM
    lgb_for_selection = LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    lgb_for_selection.fit(X_train, y_train)

    # Get feature importances
    importances = lgb_for_selection.feature_importances_

    # Create the feature_importance_df using the saved column names
    feature_importance_df = pd.DataFrame({'feature': train_columns, 'importance': importances})

    # Select a threshold or number of features
    selected_features = feature_importance_df.nlargest(num_features, 'importance')['feature'].tolist()

    # Apply selection to the train and validation sets
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]

    return X_train_selected, X_val_selected, selected_features


def save_results(model_performance):
    with open(os.path.join(OUTPUT_DIR, "model_performance_results.txt"), "w") as file:
        for model, metrics in model_performance.items():
            file.write(f"{model} - MSE: {metrics[0]}, RMSE: {metrics[1]}, R^2: {metrics[2]}\n")


def forecast(model, X):
    # Apply the trained model to make predictions
    predictions = model.predict(X)
    return predictions


def main():
    # Redirect stdout to capture all algorithm outputs
    original_stdout = sys.stdout
    sys.stdout = open(LOG_FILE_PATH, "w")

    try:
        # Load and preprocess data
        train_df, val_df = load_and_preprocess_data(DATA_PATH)
        X_train, y_train, X_val, y_val = preprocess_features(train_df, val_df)

        # Perform feature selection
        X_train_selected, X_val_selected, selected_features = feature_selection(X_train, y_train, X_val)

        # Compare various algorithms (simulated comparison)
        compare_algorithms(X_train_selected, y_train, X_val_selected, y_val)

        # Train the LightGBM model and evaluate it
        best_model, best_model_name, model_performance = train_and_evaluate_lightgbm(X_train_selected, y_train,
                                                                                     X_val_selected, y_val)

        # Apply PCA and perform cross-validation
        pca_model_performance, cv_results = apply_pca_and_cross_validation(X_train_selected, y_train, X_val_selected,
                                                                           y_val, best_model)

        # Output the performance
        print(f"Forecast Performance on 3 Months Validation Data Set: {pca_model_performance}")

    finally:
        sys.stdout.close()  # Close the logger to save all outputs
        sys.stdout = original_stdout


# Ensure that the main function is called only when the script is executed directly
if __name__ == "__main__":
    main()

