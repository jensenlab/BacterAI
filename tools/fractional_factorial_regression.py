import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


def perform_regression(features, labels, save_filepath):

    poly_transform = PolynomialFeatures(degree=2, interaction_only=True)
    poly_features = poly_transform.fit_transform(features.values)
    poly_names = poly_transform.get_feature_names(features.columns)

    regressor = LinearRegression()
    regressor.fit(poly_features, labels.values)

    coefficients = pd.DataFrame(regressor.coef_, poly_names, columns=["Coefficient"])
    labels_pred = regressor.predict(poly_features)

    coefficients.to_csv(save_filepath)

    print("Coefficients\n")
    print(coefficients)
    print("MAE:", metrics.mean_absolute_error(labels, labels_pred))
    print("MSE:", metrics.mean_squared_error(labels, labels_pred))
    print(
        "RMSE:", np.sqrt(metrics.mean_squared_error(labels, labels_pred)),
    )


if __name__ == "__main__":

    data = pd.read_csv(
        "fractional_factorial_results_low_high_100000_50split_regularization.csv",
        index_col=0,
    )
    features = data.iloc[:, :-2]
    labels = data.iloc[:, -1]

    perform_regression(
        features, labels, "regression_coefficients_100000_50split_regularization.csv"
    )
