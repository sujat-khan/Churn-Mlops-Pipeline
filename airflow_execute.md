# Apache Airflow MLOps Pipeline — Execution & Architecture Guide

A quick-reference guide explaining the architecture, setup, commands, and key concepts for the Apache Airflow automated training pipeline implemented in this project.

---

## 1. Overview & Objective

Apache Airflow is used as the workflow orchestrator to automate the end-to-end machine learning lifecycle:
1. Data Ingestion (Raw dataset splitting)
2. Data Preprocessing (Cleaning, encoding, skew transforms)
3. Feature Engineering (Domain interaction ratios & standard scaling)
4. Model Training (Random Forest classifier)
5. Model Evaluation (Metrics calculation & MLflow experiment logging)
6. Model Registration (DagsHub MLflow Model Registry promotion to Staging)

---

## 2. Architecture & Components

The cluster runs in Docker using the official CeleryExecutor multi-container architecture defined in `docker-compose-airflow.yml`:

```
+-------------------------------------------------------------+
|                      Airflow Cluster                        |
|                                                             |
|  [Airflow Webserver] <---> [PostgreSQL Database] (Metadata) |
|         ^                          ^                        |
|         |                          |                        |
|  [Airflow Scheduler] <---> [Redis Queue] <---> [Worker]     |
+-------------------------------------------------------------+
                              |
                     [Mounted Volumes]
                 /dags, /src, /data, /models,
                 /reports, params.yaml
```

### Core Services:
* **`postgres`**: PostgreSQL database storing Airflow state, task history, and metadata.
* **`redis`**: Message broker distributing tasks to Celery workers.
* **`airflow-init`**: One-time database migration and admin user creation (`airflow` / `airflow`).
* **`airflow-webserver`**: Web UI dashboard accessible at `http://localhost:8080`.
* **`airflow-scheduler`**: Monitors DAG definitions and triggers scheduled runs.
* **`airflow-worker`**: Executes the Python tasks in isolated Celery workers.

---

## 3. The DAG Pipeline (`dags/churn_training_dag.py`)

The DAG orchestrates 6 sequential tasks using the `PythonOperator`:

```python
# Task Dependency Flow:
ingest_task >> preprocess_task >> feature_task >> training_task >> evaluation_task >> registration_task
```

### Stage Summary:
| Task ID | Function Called | Description |
| :--- | :--- | :--- |
| `data_ingestion` | `src.data.data_ingestion.main` | Splits raw CSV into `train.csv` & `test.csv` based on `params.yaml` |
| `data_preprocessing` | `src.data.data_preprocessing.main` | Drops ID columns, log transforms skewed features, one-hot encodes |
| `feature_engineering` | `src.features.feature_engineering.main` | Computes domain ratios and fits `models/scaler.pkl` |
| `model_training` | `src.model.model_building.main` | Trains `RandomForestClassifier` and saves `models/model.pkl` |
| `model_evaluation` | `src.model.model_evaluation.main` | Computes metrics, logs to DagsHub MLflow, exports `experiment_info.json` |
| `model_registration` | `src.model.register_model.main` | Registers `RandomForestChurnModel` and promotes to `Staging` stage |

---

## 4. Step-by-Step Execution Commands

### 4.1 Start the Airflow Cluster
```powershell
docker compose -f docker-compose-airflow.yml up -d
```

### 4.2 Monitor Startup Logs
```powershell
docker compose -f docker-compose-airflow.yml logs -f airflow-webserver
```
*(Wait until you see `[INFO] Listening at: http://0.0.0.0:8080`)*

### 4.3 Access the Web Dashboard
* **URL:** `http://localhost:8080`
* **Username:** `airflow`
* **Password:** `airflow`

### 4.4 Trigger the DAG
1. Locate **`employee_churn_training_pipeline`** in the DAG list.
2. Toggle the switch to **ON** (Blue).
3. Click the **Play (Trigger DAG)** button on the top right.
4. Click into the DAG to watch tasks turn green in real-time.

### 4.5 Stop the Airflow Cluster
```powershell
docker compose -f docker-compose-airflow.yml down
```

---

## 5. Airflow UI Reference & Concepts

### 5.1 Understanding the Grid View
* **Vertical Columns:** Each column represents **one complete DAG Run**.
* **Small Squares:** Each square represents the execution status of a single task in that run:
  * Solid Green: `success` (Task finished with zero errors)
  * Red: `failed` (Task raised an unhandled exception)
  * Lime Outline: `running` (Task actively running on Celery worker)
  * Orange: `up_for_retry` (Task failed, Airflow is waiting to auto-retry)
  * Grey: `queued` (Waiting for upstream tasks to complete)

### 5.2 Why "Next Run ID" Shows a Past Date
* Airflow schedules runs based on **Data Intervals** (time windows).
* A weekly schedule (`0 2 * * 1` = Every Monday 02:00 UTC) triggers at the **END** of the weekly data window.
* The **Next Run ID** displays the **start timestamp of the interval** it represents, not the physical trigger clock time.

### 5.3 How to Re-run a Single Failed Task
If only one task fails (e.g., network timeout during DagsHub logging):
1. Click on that specific task square in the Grid View.
2. In the right panel, click **Clear**.
3. Airflow will re-run only that task and downstream dependent tasks without re-running earlier completed stages.
