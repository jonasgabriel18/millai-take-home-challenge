import pandas as pd
import numpy as np

def iqr_outlier_detection(column, df: pd.DataFrame):
    """
    Detects outliers in a given column using the IQR method.

    Args:
        column (str): Column name to analyze.
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing only the outlier rows.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def remove_outliers(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """
    Removes outliers from specified numeric_columns using the IQR method.

    Args:
        df (pd.DataFrame): Original DataFrame.
        numeric_columns (list): List of numerical column names to apply outlier removal.

    Returns:
        pd.DataFrame: Cleaned DataFrame without outliers.
    """
    outlier_indices = set()

    for col in numeric_columns:
        outliers = iqr_outlier_detection(col, df)
        outlier_indices.update(outliers.index.tolist())

    df_clean = df.drop(index=outlier_indices)
    return df_clean

def preprocess_data(path_to_df: str, remove_outliers_flag: bool = True) -> pd.DataFrame:
    """
    Preprocesses the data by filling missing values, fixing inconsistencies,
    and optionally removing outliers.

    Args:
        path_to_df (str): Path to the CSV file containing the dataset.
        remove_outliers_flag (bool): Whether to apply outlier removal using the IQR method.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(path_to_df)

    # Fill missing values in categorical fields
    df.loc[df['target_audience_job_titles'].isna(), 'target_audience_job_titles'] = "all"
    df.loc[df['target_audience_interests'].isna(), 'target_audience_interests'] = "all"

    # Fill missing values in numerical fields with 0
    df.fillna({'music_tempo': 0, 'speech_pace': 0, 'watch_time': 0, 'watch_percentage': 0}, inplace=True)

    # Fix logical inconsistency: has a human face but face_count is 0
    df.loc[(df['has_human_face'] == True) & (df['face_count'] == 0), 'face_count'] = 1

    # Remove outliers if flag is set to True
    if remove_outliers_flag:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = remove_outliers(df, numeric_columns)

    return df

if __name__ == "__main__":
    path_to_df = "data/raw/sample.csv"
    df = preprocess_data(path_to_df, remove_outliers_flag=False)

    df.to_csv("data/processed/sample_treated.csv", index=False)