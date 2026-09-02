import os
import logging
import pickle
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

# Logging configuration
logger = logging.getLogger('feature_engineering')
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


def load_params(params_path: str = 'params.yaml') -> dict:
    """Load configuration parameters from YAML file."""
    try:
        if os.path.exists(params_path):
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file) or {}
            logger.debug('Parameters loaded from %s', params_path)
            return params
        logger.warning('Params file %s not found. Using empty defaults.', params_path)
        return {}
    except Exception as e:
        logger.error('Error loading %s: %s', params_path, e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Loaded data from %s with shape %s', file_path, df.shape)
        return df
    except FileNotFoundError:
        logger.error('File not found at %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error loading %s: %s', file_path, e)
        raise


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific ratio and interaction features if source columns exist."""
    try:
        df_feat = df.copy()

        # Tenure to Age ratio
        if 'YearsAtCompany' in df_feat.columns and 'Age' in df_feat.columns:
            df_feat['TenurePerAge'] = df_feat['YearsAtCompany'] / (df_feat['Age'] + 1e-5)

        # Years in current role ratio relative to company tenure
        if 'YearsInCurrentRole' in df_feat.columns and 'YearsAtCompany' in df_feat.columns:
            df_feat['YearsInRoleRatio'] = df_feat['YearsInCurrentRole'] / (df_feat['YearsAtCompany'] + 1e-5)

        # Years with current manager ratio relative to company tenure
        if 'YearsWithCurrManager' in df_feat.columns and 'YearsAtCompany' in df_feat.columns:
            df_feat['YearsWithManagerRatio'] = df_feat['YearsWithCurrManager'] / (df_feat['YearsAtCompany'] + 1e-5)

        # Income per working year
        if 'MonthlyIncome' in df_feat.columns and 'TotalWorkingYears' in df_feat.columns:
            df_feat['IncomePerWorkingYear'] = df_feat['MonthlyIncome'] / (df_feat['TotalWorkingYears'] + 1e-5)

        # Promotion to tenure ratio
        if 'YearsSinceLastPromotion' in df_feat.columns and 'YearsAtCompany' in df_feat.columns:
            df_feat['PromotionTenureRatio'] = df_feat['YearsSinceLastPromotion'] / (df_feat['YearsAtCompany'] + 1e-5)

        # Replace any inf or NaN resulting from division
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

        logger.debug('Domain features created. New shape: %s', df_feat.shape)
        return df_feat
    except Exception as e:
        logger.error('Error creating domain features: %s', e)
        raise


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'Attrition',
    scaler_save_path: str = 'models/scaler.pkl',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize features using StandardScaler fitted strictly on training data.
    Saves the fitted scaler artifact using pickle for inference.
    """
    try:
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col != target_col]

        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        # Fit scaler on train only
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_cols,
            index=train_df.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_cols,
            index=test_df.index,
        )

        # Save fitted scaler with pickle
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        with open(scaler_save_path, 'wb') as file:
            pickle.dump(scaler, file)
        logger.debug('Fitted StandardScaler saved to %s', scaler_save_path)

        # Re-attach target column
        if target_col in train_df.columns:
            X_train_scaled[target_col] = train_df[target_col].values
        if target_col in test_df.columns:
            X_test_scaled[target_col] = test_df[target_col].values

        logger.debug('Features scaled. Train shape: %s, Test shape: %s', X_train_scaled.shape, X_test_scaled.shape)
        return X_train_scaled, X_test_scaled
    except Exception as e:
        logger.error('Error during feature scaling: %s', e)
        raise


def save_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    output_dir: str = 'data/processed',
) -> None:
    """Save feature-engineered train and test datasets."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, 'train_features.csv')
        test_path = os.path.join(output_dir, 'test_features.csv')

        train_features.to_csv(train_path, index=False)
        test_features.to_csv(test_path, index=False)

        logger.debug('Features successfully saved to %s and %s', train_path, test_path)
    except Exception as e:
        logger.error('Error saving feature datasets: %s', e)
        raise


def engineer_features(
    train_path: str = 'data/interim/train_processed.csv',
    test_path: str = 'data/interim/test_processed.csv',
    output_dir: str = 'data/processed',
    scaler_save_path: str = 'models/scaler.pkl',
) -> None:
    """Execute complete feature engineering pipeline."""
    try:
        logger.info('Starting feature engineering pipeline...')

        # 1. Load parameters
        params = load_params(params_path='params.yaml')

        # 2. Load preprocessed train and test data
        train_df = load_data(train_path)
        test_df = load_data(test_path)

        # 3. Create domain features
        train_df = create_domain_features(train_df)
        test_df = create_domain_features(test_df)

        # 4. Scale features with StandardScaler and persist scaler artifact
        train_scaled, test_scaled = scale_features(
            train_df=train_df,
            test_df=test_df,
            target_col='Attrition',
            scaler_save_path=scaler_save_path,
        )

        # 5. Save output feature datasets
        save_features(train_scaled, test_scaled, output_dir=output_dir)

        logger.info('Feature engineering pipeline completed successfully.')
    except Exception as e:
        logger.error('Feature engineering pipeline failed: %s', e)
        raise


def main():
    engineer_features(
        train_path='data/interim/train_processed.csv',
        test_path='data/interim/test_processed.csv',
        output_dir='data/processed',
        scaler_save_path='models/scaler.pkl',
    )


if __name__ == '__main__':
    main()
