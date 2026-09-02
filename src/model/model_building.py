import os
import logging
import pickle
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

# Logging configuration
logger = logging.getLogger('model_building')
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
    """Load hyperparameters from YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded successfully from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('Parameters file not found at %s', params_path)
        raise
    except Exception as e:
        logger.error('Error loading parameters: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Loaded training features from %s with shape %s', file_path, df.shape)
        return df
    except FileNotFoundError:
        logger.error('Data file not found at %s', file_path)
        raise
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_params: dict) -> RandomForestClassifier:
    """Instantiate and train the RandomForest model with specified parameters."""
    try:
        n_estimators = model_params.get('n_estimators', 100)
        max_depth = model_params.get('max_depth', 7)
        min_samples_split = model_params.get('min_samples_split', 5)
        min_samples_leaf = model_params.get('min_samples_leaf', 4)
        random_state = model_params.get('random_state', 42)

        logger.info(
            'Training RandomForestClassifier with params: n_estimators=%s, max_depth=%s, '
            'min_samples_split=%s, min_samples_leaf=%s, random_state=%s',
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            random_state,
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

        model.fit(X_train, y_train)
        logger.info('Model training completed successfully.')
        return model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model: RandomForestClassifier, output_path: str = 'models/model.pkl') -> None:
    """Persist the trained model to disk using pickle."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model successfully saved to %s', output_path)
    except Exception as e:
        logger.error('Error saving model to %s: %s', output_path, e)
        raise


def build_model(
    train_path: str = 'data/processed/train_features.csv',
    output_model_path: str = 'models/model.pkl',
    params_path: str = 'params.yaml',
    target_col: str = 'Attrition',
) -> None:
    """Execute the model building and training stage."""
    try:
        logger.info('Starting model building pipeline...')

        # 1. Load parameters
        params = load_params(params_path)
        model_params = params.get('model_building', {})

        # 2. Load feature data
        train_df = load_data(train_path)

        if target_col not in train_df.columns:
            raise KeyError(f'Target column "{target_col}" not found in {train_path}')

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        # 3. Train model
        model = train_model(X_train, y_train, model_params)

        # 4. Save trained model artifact
        save_model(model, output_path=output_model_path)

        logger.info('Model building pipeline completed successfully.')
    except Exception as e:
        logger.error('Model building pipeline failed: %s', e)
        raise


def main():
    build_model(
        train_path='data/processed/train_features.csv',
        output_model_path='models/model.pkl',
        params_path='params.yaml',
        target_col='Attrition',
    )


if __name__ == '__main__':
    main()
