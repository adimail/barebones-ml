import numpy as np


def correlation_matrix(y_true, y_pred):
    """
    Compute correlation matrix to evaluate the accuracy of a regression.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns:
    --------
    corr_matrix : ndarray of shape (2, 2)
        Correlation matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true > 0) & (y_pred > 0))
    tn = np.sum((y_true <= 0) & (y_pred <= 0))
    fp = np.sum((y_true <= 0) & (y_pred > 0))
    fn = np.sum((y_true > 0) & (y_pred <= 0))

    corr_matrix = np.array([[tp, fp], [fn, tn]])
    return corr_matrix


def print_correlation_matrix(corr_matrix):
    """
    Print the correlation matrix in a readable format.

    Parameters:
    -----------
    corr_matrix : ndarray of shape (2, 2)
        Correlation matrix.
    """
    print("Correlation Matrix")
    print("==================")
    print("                 Predicted Positive  Predicted Negative")
    print(
        f"Actual Positive      {corr_matrix[0, 0]}                  {corr_matrix[1, 0]}"
    )
    print(
        f"Actual Negative      {corr_matrix[0, 1]}                  {corr_matrix[1, 1]}"
    )
