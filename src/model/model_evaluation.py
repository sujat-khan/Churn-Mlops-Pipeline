import os
import json
import logging
import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Logging configuration
logger = logging.getLogger('model_evaluation')
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
    """Load test dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Loaded test features from %s with shape %s', file_path, df.shape)
        return df
    except FileNotFoundError:
        logger.error('Test data file not found at %s', file_path)
        raise
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load trained model artifact from disk using pickle."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model successfully loaded from %s', model_path)
        return model
    except FileNotFoundError:
        logger.error('Model artifact not found at %s', model_path)
        raise
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute classification metrics: Accuracy, Precision, Recall, F1, and ROC-AUC."""
    try:
        y_pred = model.predict(X_test)
        
        # Calculate ROC-AUC if probability estimates are available
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc_auc = None

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        metrics = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4) if roc_auc is not None else None,
        }

        logger.info('Evaluation Metrics:')
        for metric_name, value in metrics.items():
            logger.info('  %s: %s', metric_name, value)

        report = classification_report(y_test, y_pred, zero_division=0)
        logger.debug('Classification Report:\n%s', report)

        return metrics
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, output_path: str = 'reports/metrics.json') -> None:
    """Persist evaluation metrics to a JSON file for DVC/reporting."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved successfully to %s', output_path)
    except Exception as e:
        logger.error('Error saving metrics to %s: %s', output_path, e)
        raise


def evaluate(
    model_path: str = 'models/model.pkl',
    test_path: str = 'data/processed/test_features.csv',
    metrics_path: str = 'reports/metrics.json',
    target_col: str = 'Attrition',
) -> None:
    """Execute complete model evaluation pipeline."""
    try:
        logger.info('Starting model evaluation pipeline...')

        # 1. Load model and test dataset
        model = load_model(model_path)
        test_df = load_data(test_path)

        if target_col not in test_df.columns:
            raise KeyError(f'Target column "{target_col}" not found in {test_path}')

        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # 2. Evaluate model performance
        metrics = evaluate_model(model, X_test, y_test)

        # 3. Save metrics artifact
        save_metrics(metrics, output_path=metrics_path)

        logger.info('Model evaluation pipeline completed successfully.')
    except Exception as e:
        logger.error('Model evaluation pipeline failed: %s', e)
        raise


def main():
    evaluate(
        model_path='models/model.pkl',
        test_path='data/processed/test_features.csv',
        metrics_path='reports/metrics.json',
        target_col='Attrition',
    )


if __name__ == '__main__':
    main()
