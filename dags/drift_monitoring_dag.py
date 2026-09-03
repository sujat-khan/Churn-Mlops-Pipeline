from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

from src.monitoring.drift_detection import run_drift_analysis


def check_drift_condition(**context):
    """Returns task ID to follow based on drift check result."""
    no_drift = run_drift_analysis()
    if not no_drift:
        # Drift detected: trigger model retraining DAG
        return "trigger_retraining"
    return "no_drift_detected"


default_args = {
    'owner': 'Sujat',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='model_drift_monitoring_dag',
    default_args=default_args,
    description='Automated scheduled drift check comparing live inferences against baseline',
    schedule_interval='@daily',
    start_date=datetime(2026, 9, 1),
    catchup=False,
    tags=['monitoring', 'drift_detection'],
) as dag:

    audit_drift = BranchPythonOperator(
        task_id='audit_data_drift',
        python_callable=check_drift_condition,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id='trigger_retraining',
        trigger_dag_id='employee_churn_training_pipeline',  # Your existing retraining DAG!
        wait_for_completion=False,
    )

    no_action = EmptyOperator(
        task_id='no_drift_detected',
    )

    audit_drift >> [trigger_retrain, no_action]
