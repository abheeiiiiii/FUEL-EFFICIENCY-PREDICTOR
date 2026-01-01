import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = sns.load_dataset("mpg")
print(data.head())
print(data.info())
data = data.dropna()
data.head()
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'origin']
X = data[features]
y = data['mpg']
X.head()
X_encoded = pd.get_dummies(X, columns=['origin'], drop_first=True)
X_encoded.head()

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1-r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(f"\nLinear Model: MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}, Adjusted R²={adj_r2:.2f}")

plt.figure(figsize=(6,4))
sns.histplot(y_test - y_pred, bins=20, kde=True)
plt.title('Residuals Distribution (Linear Model)')
plt.xlabel('Residual')
plt.show()


poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

lr_poly = LinearRegression()
lr_poly.fit(X_poly_train, y_train)

y_poly_pred = lr_poly.predict(X_poly_test)

mae_poly = mean_absolute_error(y_test, y_poly_pred)
mse_poly = mean_squared_error(y_test, y_poly_pred)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_poly_pred))
r2_poly = r2_score(y_test, y_poly_pred)
adj_r2_poly = 1 - (1-r2_poly) * (len(y_test)-1)/(len(y_test)-X_poly_test.shape[1]-1)
print(f"\nPolynomial Model: MAE={mae_poly:.2f}, MSE={mse_poly:.2f}, RMSE={rmse_poly:.2f}, R²={r2_poly:.2f}, Adjusted R²={adj_r2_poly:.2f}")


plt.figure(figsize=(6,4))
sns.histplot(y_test - y_poly_pred, bins=20, kde=True)
plt.title('Residuals Distribution (Polynomial Model)')
plt.xlabel('Residual')
plt.show()
coef_names = poly.get_feature_names_out(X_encoded.columns)
coefs_df = pd.DataFrame({'Feature': coef_names, 'Coef': lr_poly.coef_})

print("\nTop predictive features (Polynomial Model):")
print(coefs_df.reindex(coefs_df['Coef'].abs().sort_values(ascending=False).index).head(10))

train_errors = []
for d in range(1, 8):
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X_encoded)
    model = LinearRegression().fit(X_poly, y)
    mse = mean_squared_error(y, model.predict(X_poly))
    train_errors.append(mse)

plt.figure(figsize=(7, 5))
plt.plot(range(1, 8), train_errors, marker='o', label='Training MSE')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias–Variance Tradeoff (Training Error Trend)")
plt.grid(True)
plt.legend()
print(plt.show())

test_errors = []
for d in range(1, 8):
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_test_pred = model.predict(X_poly_test)
    mse = mean_squared_error(y_test, y_test_pred)
    test_errors.append(mse)

plt.figure(figsize=(7, 5))
plt.plot(range(1, 8), test_errors, marker='o', label='Test MSE')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Polynomial Degree vs. Test MSE")
plt.grid(True)
plt.legend()
print(plt.show())

plt.figure(figsize=(8, 6))
plt.plot(range(1, 8), train_errors, marker='o', label='Training MSE')
plt.plot(range(1, 8), test_errors, marker='o', label='Test MSE')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias–Variance Tradeoff (Training vs. Test Error)")
plt.xticks(range(1, 8))
plt.grid(True)
plt.legend()
print(plt.show())

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr_poly, poly.fit_transform(X_encoded), y, cv=kf, scoring='r2')

print(f"\nCV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.2f}")