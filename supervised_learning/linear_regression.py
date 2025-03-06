import numpy as np


class LinearRegression:
    """
    A linear regression model implemented with NumPy that supports both simple and multiple regression.
    """

    def __init__(self, fit_intercept=True, feature_names=None):
        if feature_names is not None and not isinstance(
            feature_names, (list, tuple, np.ndarray)
        ):
            raise TypeError("feature_names must be a list, tuple, or numpy array")

        self.fit_intercept = fit_intercept
        self.feature_names = feature_names
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None
        self.n_samples_ = None
        self.n_features_ = None
        self._se = None
        self._is_fitted = False

    def fit(self, X, y):
        """
        Fit the linear model to the training data.
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values")
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("y contains NaN or infinite values")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            raise ValueError(f"X has {X.ndim} dimensions, but only 1 or 2 are allowed")

        if X.shape[0] == 0:
            raise ValueError("X is empty")

        self.n_samples_, self.n_features_ = X.shape

        if len(y) != self.n_samples_:
            raise ValueError(
                f"X has {self.n_samples_} samples, but y has {len(y)} samples"
            )

        if self.feature_names is None:
            self.feature_names_ = [f"X{i+1}" for i in range(self.n_features_)]
        else:
            if len(self.feature_names) != self.n_features_:
                raise ValueError(
                    f"Length of feature_names ({len(self.feature_names)}) does not match number of features ({self.n_features_})"
                )
            self.feature_names_ = self.feature_names

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim > 2:
            raise ValueError(f"y has {y.ndim} dimensions, but only 1 or 2 are allowed")

        if y.shape[1] > 1:
            raise ValueError(
                f"Multiple target variables are not supported. y has {y.shape[1]} columns."
            )

        if self.fit_intercept:
            X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_b = X

        X_rank = np.linalg.matrix_rank(X_b)
        if X_rank < X_b.shape[1]:
            raise ValueError(
                "X is singular or contains collinear features. Cannot compute inverse for least squares solution."
            )

        try:
            X_t_X = X_b.T.dot(X_b)
            X_t_X_inv = np.linalg.inv(X_t_X)
            theta = X_t_X_inv.dot(X_b.T).dot(y)

            y_pred = X_b.dot(theta)
            residuals = y - y_pred

            df = self.n_samples_ - self.n_features_ - (1 if self.fit_intercept else 0)
            if df <= 0:
                raise ValueError(
                    f"Insufficient degrees of freedom: {df}. Need more samples than features."
                )

            mse = np.sum(residuals**2) / df
            var_covar_matrix = mse * X_t_X_inv
            self._se = np.sqrt(np.diag(var_covar_matrix))

            if self.fit_intercept:
                self.intercept_ = float(theta[0, 0])
                self.coef_ = theta[1:].flatten()
            else:
                self.intercept_ = 0.0
                self.coef_ = theta.flatten()

            self._is_fitted = True

            return self

        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"Linear algebra error during fitting: {str(e)}. "
                "This might be due to a singular matrix (features are linearly dependent)"
            )

    def predict(self, X):
        """
        Predict using the linear model.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call 'fit' before making predictions.")

        X = np.array(X, dtype=np.float64)

        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            raise ValueError(f"X has {X.ndim} dimensions, but only 1 or 2 are allowed")

        if X.shape[0] == 0:
            raise ValueError("X is empty")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but LinearRegression is expecting {self.n_features_} features"
            )

        if self.fit_intercept:
            X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_b = X

        if self.fit_intercept:
            theta = np.vstack([[self.intercept_], self.coef_.reshape(-1, 1)])
        else:
            theta = self.coef_.reshape(-1, 1)

        return X_b.dot(theta).flatten()

    def score(self, X, y):
        """
        Calculate the R^2 score of the model.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call 'fit' before calculating score.")

        y = np.array(y, dtype=np.float64)

        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("y contains NaN or infinite values")

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim > 2:
            raise ValueError(f"y has {y.ndim} dimensions, but only 1 or 2 are allowed")

        try:
            y_pred = self.predict(X).reshape(-1, 1)

            if len(y) != len(y_pred):
                raise ValueError(
                    f"X has {len(y_pred)} samples after prediction, but y has {len(y)} samples"
                )

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0

            return 1 - ss_res / ss_tot

        except Exception as e:
            raise ValueError(f"Error calculating score: {str(e)}")

    def summary(self, X, y):
        """
        Print a detailed summary of the fitted model.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call 'fit' before generating summary.")

        try:
            r2 = self.score(X, y)

            df = self.n_samples_ - self.n_features_ - 1
            if df <= 0:
                adjusted_r2 = np.nan
            else:
                adjusted_r2 = 1 - (1 - r2) * (self.n_samples_ - 1) / df

            t_stats = []

            if self.fit_intercept:
                if self._se[0] == 0:
                    t_stat_intercept = np.nan
                else:
                    t_stat_intercept = self.intercept_ / self._se[0]
                t_stats.append(t_stat_intercept)

            for i in range(self.n_features_):
                se_idx = i + 1 if self.fit_intercept else i
                if self._se[se_idx] == 0:
                    t_stat = np.nan
                else:
                    t_stat = self.coef_[i] / self._se[se_idx]
                t_stats.append(t_stat)

            print("\nMultiple Linear Regression Model Summary")
            print("=======================================")
            print(f"Number of samples: {self.n_samples_}")
            print(f"Number of features: {self.n_features_}")
            print(f"R-squared: {r2:.4f}")
            print(f"Adjusted R-squared: {adjusted_r2:.4f}")

            print("\nParameters:")
            print("------------")
            print(
                f"{'Parameter':<15} {'Coefficient':<15} {'Std Error':<15} {'t-value':<15}"
            )

            if self.fit_intercept:
                print(
                    f"{'Intercept':<15} {self.intercept_:<15.4f} {self._se[0]:<15.4f} {t_stats[0]:<15.4f}"
                )
                coef_start_idx = 1
            else:
                coef_start_idx = 0

            for i, name in enumerate(self.feature_names_):
                idx = i + coef_start_idx
                print(
                    f"{name:<15} {self.coef_[i]:<15.4f} {self._se[i+1 if self.fit_intercept else i]:<15.4f} {t_stats[idx]:<15.4f}"
                )

        except Exception as e:
            print(f"Error generating summary: {str(e)}")

    def __repr__(self):
        return f"LinearRegression(fit_intercept={self.fit_intercept}, feature_names={self.feature_names})"

    def __str__(self):
        if not self._is_fitted:
            return "LinearRegression Model (not fitted)"

        model_str = "Linear Regression Model:\n"
        model_str += f"  Intercept: {self.intercept_}\n"

        for i, name in enumerate(self.feature_names_ if self.feature_names_ else []):
            model_str += f"  {name}: {self.coef_[i]}\n"

        return model_str
