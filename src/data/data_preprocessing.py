import os
import logging
import numpy as np
import pandas as pd
import yaml

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data successfully loaded from %s with shape %s', file_path, df.shape)
        return df
    except FileNotFoundError:
        logger.error('File not found at path: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading %s: %s', file_path, e)
        raise


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant (zero-variance) and identifier columns."""
    try:
        cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
            logger.debug('Dropped zero-variance and ID columns: %s', existing_cols_to_drop)
        return df
    except Exception as e:
        logger.error('Error dropping unnecessary columns: %s', e)
        raise


def encode_target(df: pd.DataFrame, target_col: str = 'Attrition') -> pd.DataFrame:
    """Encode binary target variable (Yes -> 1, No -> 0)."""
    try:
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(
                lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0
            )
            logger.debug('Target column "%s" encoded successfully.', target_col)
        return df
    except Exception as e:
        logger.error('Error encoding target column %s: %s', target_col, e)
        raise


def log_transform_skewed_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, skew_threshold: float = 0.75
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Log-transform numerical features with absolute skewness above threshold."""
    try:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Attrition' in numeric_cols:
            numeric_cols.remove('Attrition')

        skew_vals = train_df[numeric_cols].skew()
        skewed_cols = skew_vals[skew_vals.abs() > skew_threshold].index.tolist()

        logger.debug('Log-transforming skewed features (skew > %.2f): %s', skew_threshold, skewed_cols)

        for col in skewed_cols:
            train_df[col] = np.log1p(train_df[col])
            if col in test_df.columns:
                test_df[col] = np.log1p(test_df[col])

        return train_df, test_df
    except Exception as e:
        logger.error('Error during log transformation: %s', e)
        raise


def encode_categorical_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical features and align columns between train and test sets."""
    try:
        categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'Attrition' in categorical_cols:
            categorical_cols.remove('Attrition')

        logger.debug('Categorical columns to one-hot encode: %s', categorical_cols)

        # One-hot encode train and test
        train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
        test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

        # Align test set columns to match train set exactly
        train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

        # Clean column names (remove spaces)
        train_encoded.columns = train_encoded.columns.str.replace(' ', '')
        test_encoded.columns = test_encoded.columns.str.replace(' ', '')

        logger.debug('Encoded shape - Train: %s, Test: %s', train_encoded.shape, test_encoded.shape)
        return train_encoded, test_encoded
    except Exception as e:
        logger.error('Error during categorical encoding: %s', e)
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = 'data/interim') -> None:
    """Save preprocessed train and test dataframes to output directory."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train_processed.csv')
        test_path = os.path.join(output_dir, 'test_processed.csv')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        

        logger.debug('Preprocessed data successfully saved to %s', output_dir)
    except Exception as e:
        logger.error('Error saving preprocessed data: %s', e)
        raise


def preprocess_data(
    train_path: str = 'data/raw/train.csv',
    test_path: str = 'data/raw/test.csv',
    output_dir: str = 'data/interim',
) -> None:
    """Execute complete data preprocessing pipeline."""
    try:
        logger.info('Starting data preprocessing pipeline...')

        # 1. Load raw train and test data
        train_df = load_data(train_path)
        test_df = load_data(test_path)

        # 2. Drop zero-variance and ID columns
        train_df = drop_unnecessary_columns(train_df)
        test_df = drop_unnecessary_columns(test_df)

        # 3. Encode target column
        train_df = encode_target(train_df, target_col='Attrition')
        test_df = encode_target(test_df, target_col='Attrition')

        # 4. Log-transform skewed numerical features based on train statistics
        train_df, test_df = log_transform_skewed_features(train_df, test_df, skew_threshold=0.75)

        # 5. One-hot encode categorical features and align train/test columns
        train_df, test_df = encode_categorical_features(train_df, test_df)

        # 6. Save preprocessed datasets
        save_data(train_df, test_df, output_dir=output_dir)

        logger.info('Data preprocessing pipeline completed successfully.')
    except Exception as e:
        logger.error('Data preprocessing pipeline failed: %s', e)
        raise


def main():
    preprocess_data(
        train_path='data/raw/train.csv',
        test_path='data/raw/test.csv',
        output_dir='data/interim',
    )


if __name__ == '__main__':
    main()
