import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from supervised_learning.linear_regression import LinearRegression


def single_variable_regression():
    X, y = make_regression(n_samples=100, n_features=1, noise=11, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Single Variable Linear Regression")
    print("-----------------------------------")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}\n")

    model.summary(X_train, y_train)

    x_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_range_pred = model.predict(x_range)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color="blue", label="Training Data")
    plt.plot(x_range, y_range_pred, color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Single Variable Linear Regression")
    plt.legend()
    plt.show()


def multivariable_regression():
    X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    feature_names = ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"]
    model = LinearRegression(feature_names=feature_names)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Multivariable Linear Regression")
    print("---------------------------------")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}\n")

    model.summary(X_train, y_train)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="green", alpha=0.6)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Multivariable Regression: Actual vs Predicted")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    plt.show()


if __name__ == "__main__":
    single_variable_regression()
    multivariable_regression()
