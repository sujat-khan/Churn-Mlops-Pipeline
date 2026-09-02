import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure Airflow can import from /opt/airflow (project root in container)
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', '/opt/airflow')
if AIRFLOW_HOME not in sys.path:
    sys.path.insert(0, AIRFLOW_HOME)

# Import pipeline stage functions directly from your codebase
from src.data.data_ingestion import main as run_data_ingestion
from src.data.data_preprocessing import main as run_data_preprocessing
from src.features.feature_engineering import main as run_feature_engineering
from src.model.model_building import main as run_model_building
from src.model.model_evaluation import main as run_model_evaluation
from src.model.register_model import main as run_register_model


# Default arguments applied to all tasks
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

# Define the DAG
with DAG(
    dag_id='employee_churn_training_pipeline',
    default_args=default_args,
    description='Automated end-to-end retraining, evaluation, and DagsHub MLflow registration',
    schedule_interval='0 2 * * 1',  # Runs every Monday at 02:00 AM UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['mlops', 'churn-prediction', 'production'],
) as dag:

    # Stage 1: Data Ingestion
    ingest_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=run_data_ingestion,
    )

    # Stage 2: Data Preprocessing
    preprocess_task = PythonOperator(
        task_id='data_preprocessing',
        python_callable=run_data_preprocessing,
    )

    # Stage 3: Feature Engineering & Standardization
    feature_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=run_feature_engineering,
    )

    # Stage 4: Model Training (Random Forest)
    training_task = PythonOperator(
        task_id='model_training',
        python_callable=run_model_building,
    )

    # Stage 5: Evaluation & MLflow Experiment Tracking
    evaluation_task = PythonOperator(
        task_id='model_evaluation',
        python_callable=run_model_evaluation,
    )

    # Stage 6: Automated Model Registration to Staging
    registration_task = PythonOperator(
        task_id='model_registration',
        python_callable=run_register_model,
    )

    # Define linear execution order
    ingest_task >> preprocess_task >> feature_task >> training_task >> evaluation_task >> registration_task
