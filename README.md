🚗 Fuel Efficiency Predictor (MPG Prediction)

This project predicts fuel efficiency (Miles Per Gallon – MPG) of vehicles using Linear Regression and Polynomial Regression based on engine and vehicle attributes.
It demonstrates the complete machine learning pipeline: data preprocessing, feature engineering, model training, evaluation, bias–variance analysis, and cross-validation.

📌 Project Overview

Fuel efficiency is an important metric in automotive design and environmental analysis.
In this project, a regression-based machine learning approach is used to predict MPG using the Auto MPG dataset.

The project compares:

Linear Regression

Polynomial Regression (degree 2 and higher)

and evaluates their performance using multiple metrics.

📊 Dataset

Source: Seaborn built-in mpg dataset

Total records: 398 (after removing missing values)

Target variable: mpg

Features Used

cylinders

displacement

horsepower

weight

acceleration

origin (encoded using one-hot encoding)

🛠️ Technologies & Libraries

Python

Pandas

NumPy

Seaborn

Matplotlib

Scikit-learn

⚙️ Methodology
1. Data Preprocessing

Removed missing values

Selected relevant numerical features

Applied one-hot encoding to categorical feature (origin)

Split data into training and testing sets (80/20)

2. Models Implemented
🔹 Linear Regression

A baseline regression model to understand linear relationships between features and MPG.

🔹 Polynomial Regression

Polynomial features were generated to capture non-linear relationships.

Degree = 2 for detailed analysis

Degrees 1–7 for bias–variance tradeoff study

3. Evaluation Metrics

Each model was evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Adjusted R² Score

4. Residual Analysis

Residual distributions were plotted for both linear and polynomial models

Helps in checking model assumptions and error behavior

5. Feature Importance (Polynomial Model)

Polynomial coefficients were analyzed

Top predictive features were identified based on absolute coefficient magnitude

6. Bias–Variance Tradeoff

Training and testing MSE were calculated for polynomial degrees 1–7

Visualized how model complexity affects overfitting and underfitting

7. Cross-Validation

5-Fold Cross-Validation applied

R² score used to assess generalization performance

📈 Results Summary

Polynomial Regression outperformed Linear Regression in terms of R² and RMSE

Increasing polynomial degree reduced training error but increased test error after a point

🔮 Future Improvements

Add regularization (Ridge, Lasso)

Try tree-based models (Random Forest, Gradient Boosting)

Hyperparameter tuning

Deploy as a web app using Flask or Streamlit

👨‍💻 Author
Abhinav Tiwar

Cross-validation confirmed the robustness of the polynomial model

Vehicle weight, horsepower, and interaction terms were strong predictors of MPG# FUEL-EFFICIENCY-PREDICTOR
