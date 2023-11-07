# CANSSI2023
French Trot Horse Racing: Forecasting Competition
# 1- Brief Introduction to the Machine Learning Workflow 
In this project, we have undertaken a comprehensive machine-learning workflow to predict win probabilities in horse racing. Our approach is methodical, starting from data preprocessing to model selection and evaluation. Below is an overview of the steps we have taken:
## Algorithm Comparison and Selection
We began by comparing a suite of advanced machine learning algorithms to identify the one that performs best with our dataset. The algorithms we considered include:
* **Linear Regression**: A simple yet powerful technique for regression tasks.
* **Lasso Regression**: An extension of linear regression that includes regularization to prevent overfitting.
* **Random Forest Regressor**: An ensemble learning method based on decision tree regressors.
* **Decision Tree Regressor**: A model that predicts the value of a target variable by learning simple decision rules from data features.
* **XGBoost Regressor**: An optimized gradient-boosting machine learning library.
* **Light Gradient Boosting Machine (LightGBM)**: A fast, distributed, high-performance gradient boosting framework.
* **Deep Convolution Neural Network**: techniques for complex pattern recognition tasks, which are particularly useful when dealing with high-dimensional data.
After a thorough comparison based on performance metrics such as Mean Squared Error (MSE) and R-squared (R²), we selected the **Light Gradient Boosting Machine (LightGBM)** for its efficiency and effectiveness for our dataset.

## Feature Selection
To improve our model's performance, we implemented feature selection. This process involves selecting the most important features based on their impact on the prediction variable. By using LightGBM's feature importances, we identified and retained the top features that contribute most significantly to the model's predictive power.

## Data Preprocessing
Our data preprocessing steps were crucial in ensuring the quality of our dataset before feeding it into the machine learning models. These steps included:

* Outlier Removal
We removed outliers from our dataset to prevent them from skewing our model's performance. Outliers were identified using Z-scores, with a threshold set to remove extreme values.
* Handling Missing Values
We addressed missing values in our dataset to avoid inaccuracies in our model's predictions. Missing values were filled with the mean of the respective feature.
* Normalization
We scaled our features to treat all variables equally during the model training process.

## Cross-Validation
To assess the robustness of our LightGBM model, we employed cross-validation. This technique involves partitioning the data into subsets, training the model on some subsets while validating on others. This process helps in evaluating the model's ability to generalize to an independent dataset.

## Conclusion
By meticulously following these steps, we have built a robust predictive model. Our workflow not only ensures that we have a model that performs well on our current dataset but also has the potential to generalize well to new, unseen data. The use of cross-validation and careful feature selection contributes to a model that we can trust for making predictions with a high level of confidence.
