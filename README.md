# Calories Prediction for Workouts

## Overview

This project aims to predict the number of calories burned during a workout based on various features such as age, weight, heart rate, and duration of the exercise. The model is evaluated using the Root Mean Squared Logarithmic Error (RMSLE) metric, which measures the error between predicted and actual values on a logarithmic scale.

## Dataset

The dataset includes the following columns:

* `id`: Unique identifier for each instance.
* `Sex`: Gender of the individual (encoded as 0 for female and 1 for male).
* `Age`: Age of the individual.
* `Height`: Height in centimeters.
* `Weight`: Weight in kilograms.
* `Duration`: Duration of the workout in minutes.
* `Heart_Rate`: Heart rate during the workout.
* `Body_Temp`: Body temperature during the workout.
* `Calories`: Target variable (calories burned).

Additional features are engineered from the raw dataset:

* `BMI`: Body Mass Index calculated as `Weight / (Height / 100)^2`.
* `HR_per_min`: Heart rate per minute, calculated as `Heart_Rate / Duration`.
* `Temp_per_min`: Temperature per minute, calculated as `Body_Temp / Duration`.
* `Effort`: A combined metric of heart rate, body temperature, and duration.
* `Age_Weight`: Product of age and weight.
* `Weight_per_height`: Weight divided by height.
* `log_Duration`: Log-transformed workout duration.
* `log_HR`: Log-transformed heart rate.

## Steps Taken

1. **Preprocessing:**

   * **Categorical Encoding**: The `Sex` column is encoded using `LabelEncoder` to convert it into a numerical format.
   * **Feature Engineering**: New features such as BMI, heart rate per minute, and others are calculated to better capture patterns in the data.
   * **Data Scaling**: Numerical features, except for `Sex`, are standardized using `StandardScaler`.

2. **Modeling:**

   * We use two ensemble models: **CatBoostRegressor** and **XGBRegressor**.
   * Both models are trained using **K-Fold Cross Validation** (5 folds) to reduce overfitting and ensure robustness.
   * A **Stacking** approach is applied where the predictions from both models are combined to make the final prediction.

3. **Training:**

   * The training is performed using `KFold` cross-validation to evaluate the model's performance on different splits of the training data.
   * The models are trained in pipelines where preprocessing steps and model fitting are encapsulated together for each fold.

4. **Prediction:**

   * The final predictions are made by applying weights to the predictions of both models (CatBoost and XGBoost).
   * The final predictions are then transformed back from the logarithmic scale to the original scale using `np.expm1`.

## Evaluation Metric

The performance of the model is evaluated using **Root Mean Squared Logarithmic Error (RMSLE)**:

$$
RMSLE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2}
$$

Where:

* $n$ is the total number of observations in the test set,
* $\hat{y}_i$ is the predicted value for instance $i$,
* $y_i$ is the actual value for instance $i$,
* $\log$ denotes the natural logarithm.

The goal is to minimize this metric, indicating the accuracy of the predictions.

## Submission

The final predictions are saved in a CSV file with the following structure:

```csv
id,Calories
12345,245.56
23456,178.92
...
```

The `Calories` column represents the predicted calories burned.

## Libraries Used

* `pandas`: Data manipulation and analysis.
* `numpy`: Numerical operations.
* `sklearn`: Machine learning preprocessing and modeling.
* `catboost`: CatBoost regression model.
* `xgboost`: XGBoost regression model.

## Conclusion

This project demonstrates the application of machine learning algorithms (CatBoost and XGBoost) for predicting calorie consumption during a workout. By leveraging advanced preprocessing techniques, feature engineering, and model stacking, we have built an effective predictive model.

---
