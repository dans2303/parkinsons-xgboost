import numpy as np
import pandas as pd

from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.combine import SMOTETomek


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        file_path: Full path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def clean_numeric_data(df: pd.DataFrame, target_column: str = "status") -> pd.DataFrame:
    """
    Keep only numeric columns and preserve the target column.

    Args:
        df: Input dataframe.
        target_column: Name of target column.

    Returns:
        Cleaned dataframe with only numeric columns.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_column not in numeric_columns:
        raise ValueError(
            f"Target column '{target_column}' must be numeric for this pipeline."
        )

    cleaned_df = df[numeric_columns].copy()
    cleaned_df = cleaned_df.dropna().reset_index(drop=True)

    return cleaned_df


def split_and_scale(
    df: pd.DataFrame,
    target_column: str = "status",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Split dataframe into train/test sets and scale features.

    Important:
    - split first
    - fit scaler only on training data
    - transform training and test data separately

    Args:
        df: Cleaned dataframe.
        target_column: Name of target column.
        test_size: Fraction of data used for test set.
        random_state: Random seed.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def balance_classes(
    X,
    y=None,
    target_column: str = "status",
    method: str = "tomek",
    random_state: int = 42
):
    """
    Balance classes using either:
    1. New recommended usage:
       balance_classes(X_train, y_train, method="tomek")
    2. Old dataframe-style usage:
       balance_classes(df, target_column="status", method="tomek")

    Supported methods:
    - "tomek": SMOTETomek
    - "oversample": simple random oversampling of minority
    - "undersample": simple random undersampling of majority

    Returns:
    - If X and y are given separately:
        X_resampled, y_resampled
    - If a dataframe is given and y is None:
        resampled dataframe
    """
    if method is None:
        return (X, y) if y is not None else X

    method = method.lower()

    # Case 1: dataframe input
    if y is None and isinstance(X, pd.DataFrame):
        df = X.copy()

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")

        X_df = df.drop(columns=[target_column]).copy()
        y_series = df[target_column].astype(int).copy()

        X_res, y_res = _resample_xy(
            X_df, y_series, method=method, random_state=random_state
        )

        df_res = pd.DataFrame(X_res, columns=X_df.columns)
        df_res[target_column] = y_res
        return df_res

    # Case 2: X and y separately
    if y is None:
        raise ValueError("If X is not a dataframe containing the target column, y must be provided.")

    X_res, y_res = _resample_xy(X, y, method=method, random_state=random_state)

    if isinstance(X, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X.columns)

    if isinstance(y, pd.Series):
        y_res = pd.Series(y_res, name=y.name)

    return X_res, y_res


def _resample_xy(X, y, method: str, random_state: int = 42):
    """
    Internal helper for class balancing on X and y.
    """
    if method == "tomek":
        sampler = SMOTETomek(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled

    y_series = pd.Series(y).reset_index(drop=True)
    X_df = pd.DataFrame(X).reset_index(drop=True)

    class_counts = y_series.value_counts()
    if len(class_counts) != 2:
        raise ValueError("Only binary classification is supported for oversample/undersample.")

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    X_majority = X_df[y_series == majority_class]
    X_minority = X_df[y_series == minority_class]
    y_majority = y_series[y_series == majority_class]
    y_minority = y_series[y_series == minority_class]

    if method == "oversample":
        X_minority_res, y_minority_res = resample(
            X_minority,
            y_minority,
            replace=True,
            n_samples=len(X_majority),
            random_state=random_state
        )

        X_resampled = pd.concat([X_majority, X_minority_res], axis=0)
        y_resampled = pd.concat([y_majority, y_minority_res], axis=0)

    elif method == "undersample":
        X_majority_res, y_majority_res = resample(
            X_majority,
            y_majority,
            replace=False,
            n_samples=len(X_minority),
            random_state=random_state
        )

        X_resampled = pd.concat([X_majority_res, X_minority], axis=0)
        y_resampled = pd.concat([y_majority_res, y_minority], axis=0)

    else:
        raise ValueError(
            f"Unsupported method: {method}. "
            f"Use 'tomek', 'oversample', or 'undersample'."
        )

    return X_resampled, y_resampled


def bootstrap_sample(
    df: pd.DataFrame,
    n_samples: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Bootstrap a dataframe with replacement.

    Args:
        df: Input dataframe.
        n_samples: Number of samples to generate.
        random_state: Random seed.

    Returns:
        Bootstrapped dataframe.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("bootstrap_sample expects a pandas DataFrame.")

    boot_df = df.sample(
        n=n_samples,
        replace=True,
        random_state=random_state
    ).reset_index(drop=True)

    return boot_df