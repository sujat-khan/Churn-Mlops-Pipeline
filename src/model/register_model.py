import os
import json
import logging
import mlflow
import dagshub
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = os.getenv("repo_owner", "sujat-khan")
repo_name = os.getenv("repo_name", "Churn-Mlops-Pipeline").strip()

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Logging configuration
logger = logging.getLogger('model_registration')
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


def load_model_info(file_path: str = 'reports/experiment_info.json') -> dict:
    """Load the model run information from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded successfully from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('Model info file not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error loading model info from %s: %s', file_path, e)
        raise


def register_model(model_name: str, model_info: dict, target_stage: str = "Staging") -> None:
    """Register the trained model to the MLflow Model Registry and transition to target stage."""
    try:
        client = mlflow.tracking.MlflowClient()
        run_id = model_info['run_id']
        model_path = model_info.get('model_path', 'model')
        model_uri = f"runs:/{run_id}/{model_path}"

        logger.info('Registering model "%s" from URI: %s', model_name, model_uri)

        version = None
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            version = model_version.version
            logger.info('Model successfully registered as version %s', version)
        except Exception as reg_err:
            logger.warning('mlflow.register_model fallback triggered: %s', reg_err)
            run = client.get_run(run_id)
            artifact_uri = run.info.artifact_uri
            source = f"{artifact_uri}/{model_path}"

            try:
                client.create_registered_model(model_name)
            except Exception:
                pass

            # Check if version for this run already exists
            existing_versions = client.search_model_versions(f"name='{model_name}'")
            for v in existing_versions:
                if v.run_id == run_id:
                    version = v.version
                    break

            if not version:
                created = client.create_model_version(
                    name=model_name,
                    source=source,
                    run_id=run_id
                )
                version = created.version

        # Transition the model version to the target stage and archive existing versions
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=target_stage,
            archive_existing_versions=True
        )

        logger.info('Model "%s" version %s successfully transitioned to stage: %s', model_name, version, target_stage)

    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "RandomForestChurnModel"
        register_model(model_name=model_name, model_info=model_info, target_stage="Staging")
    except Exception as e:
        logger.error('Failed to complete model registration pipeline: %s', e)
        raise


if __name__ == '__main__':
    main()