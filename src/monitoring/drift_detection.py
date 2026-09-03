"""
Native Statistical Model & Data Drift Detection Engine
Using pure SciPy, NumPy, and Pandas (Zero external heavy dependencies).

Statistical Methods:
1. Continuous Features: Two-Sample Kolmogorov-Smirnov (KS) Test & Wasserstein Distance
2. Categorical Features: Population Stability Index (PSI) & Distribution Divergence
3. Prediction Drift: Output class distribution comparison vs Baseline Churn Rate
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats


def calculate_numeric_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) for continuous numeric features."""
    if len(actual) < 15:
        return 0.0

    try:
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_pct = (expected_counts + 1e-4) / len(expected)
        actual_pct = (actual_counts + 1e-4) / len(actual)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(max(0.0, psi))
    except Exception:
        return 0.0


def calculate_categorical_psi(ref_series: pd.Series, cur_series: pd.Series) -> float:
    """Calculates Population Stability Index (PSI) for categorical features."""
    if len(cur_series) < 15:
        return 0.0

    try:
        ref_dist = ref_series.value_counts(normalize=True)
        cur_dist = cur_series.value_counts(normalize=True)
        aligned_ref, aligned_cur = ref_dist.align(cur_dist, fill_value=1e-4)

        psi = np.sum((aligned_cur - aligned_ref) * np.log(aligned_cur / aligned_ref))
        return float(max(0.0, psi))
    except Exception:
        return 0.0


def audit_feature_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, feature_cols: list) -> dict:
    """Runs two-sample statistical tests across all feature columns."""
    sample_size = len(cur_df)
    drift_details = {}
    drifted_count = 0

    for col in feature_cols:
        ref_s = ref_df[col].dropna()
        cur_s = cur_df[col].dropna()

        if pd.api.types.is_numeric_dtype(ref_s):
            # 1. Kolmogorov-Smirnov 2-sample test
            ks_stat, p_val = stats.ks_2samp(ref_s, cur_s)
            psi = calculate_numeric_psi(ref_s.values, cur_s.values)
            norm_factor = max(ref_s.std(), 1e-5)
            wasserstein_dist = stats.wasserstein_distance(ref_s, cur_s) / norm_factor

            # Statistical decision rule:
            # For small samples (<15), avoid false alarms
            # For regular samples, drift is flagged if p < 0.05 and observable divergence exists
            if sample_size >= 15:
                is_drifted = bool(p_val < 0.05 and (ks_stat > 0.15 or psi > 0.25))
            else:
                is_drifted = False

            drift_details[col] = {
                "type": "numeric",
                "test": "Kolmogorov-Smirnov (KS)",
                "p_value": round(float(p_val), 4),
                "ks_statistic": round(float(ks_stat), 4),
                "psi": round(float(psi), 4),
                "wasserstein_norm": round(float(wasserstein_dist), 4),
                "drift_detected": is_drifted
            }
        else:
            # 2. Categorical distribution divergence (PSI)
            psi = calculate_categorical_psi(ref_s, cur_s)
            is_drifted = bool(psi > 0.25 and sample_size >= 15)

            drift_details[col] = {
                "type": "categorical",
                "test": "Population Stability Index (PSI)",
                "p_value": "N/A",
                "ks_statistic": "N/A",
                "psi": round(float(psi), 4),
                "wasserstein_norm": "N/A",
                "drift_detected": is_drifted
            }

        if is_drifted:
            drifted_count += 1

    return {
        "sample_size": sample_size,
        "total_features": len(feature_cols),
        "drifted_features": drifted_count,
        "drift_share": round(drifted_count / max(len(feature_cols), 1), 4),
        "all_passed": bool(drifted_count < 4),
        "details": drift_details
    }


def audit_prediction_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> dict:
    """Checks if the model's predicted churn rate has deviated from historical training churn."""
    if "prediction" not in cur_df.columns:
        return {"prediction_drift_detected": False, "details": "No prediction column found"}

    # Baseline churn rate in historical data
    baseline_churn = (ref_df["Attrition"] == "Yes").mean() if "Attrition" in ref_df.columns else 0.16
    current_predicted_churn = (cur_df["prediction"] == 1).mean()

    # Percentage point divergence
    divergence = abs(current_predicted_churn - baseline_churn)
    # Drift alert if production churn rate deviates by more than 15 percentage points
    drift_detected = bool(divergence > 0.15 and len(cur_df) >= 20)

    return {
        "baseline_churn_rate": round(float(baseline_churn), 4),
        "production_churn_rate": round(float(current_predicted_churn), 4),
        "divergence": round(float(divergence), 4),
        "drift_detected": drift_detected
    }


def generate_html_report(results: dict, output_path: str):
    """Generates a standalone, dependency-free interactive HTML monitoring dashboard."""
    features = results["feature_audit"]["details"]
    overall_passed = results["summary"]["all_passed"]
    status_color = "#10b981" if overall_passed else "#ef4444"
    status_text = "PASSED - NO DRIFT" if overall_passed else "ALERT - DRIFT DETECTED"

    rows_html = ""
    for col, d in features.items():
        drift_badge = (
            '<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;">DRIFT</span>'
            if d["drift_detected"]
            else '<span style="background:#dcfce7;color:#15803d;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;">STABLE</span>'
        )
        rows_html += f"""
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px; font-weight: 500;">{col}</td>
            <td style="padding: 12px; text-transform: capitalize; color: #6b7280;">{d['type']}</td>
            <td style="padding: 12px; color: #4b5563;">{d['test']}</td>
            <td style="padding: 12px;">{d.get('p_value', 'N/A')}</td>
            <td style="padding: 12px;">{d.get('ks_statistic', 'N/A')}</td>
            <td style="padding: 12px;">{d.get('psi', 'N/A')}</td>
            <td style="padding: 12px;">{drift_badge}</td>
        </tr>
        """

    pred_data = results["prediction_drift"]
    pred_summary = f"""
    <div class="metric-grid" style="margin-top: 16px;">
        <div class="metric-box">
            <div class="metric-label">Baseline Churn Rate</div>
            <div class="metric-val">{pred_data.get('baseline_churn_rate', 0)*100:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Live Predicted Churn Rate</div>
            <div class="metric-val">{pred_data.get('production_churn_rate', 0)*100:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Prediction Divergence</div>
            <div class="metric-val">{pred_data.get('divergence', 0)*100:.1f}%</div>
        </div>
    </div>
    """ if "baseline_churn_rate" in pred_data else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model & Data Drift Audit Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f9fafb; margin: 0; padding: 24px; color: #111827; }}
        .card {{ background: #ffffff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 28px; margin-bottom: 24px; max-width: 1200px; margin-left: auto; margin-right: auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e5e7eb; padding-bottom: 20px; margin-bottom: 24px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .metric-box {{ background: #f3f4f6; border-radius: 8px; padding: 18px; text-align: center; }}
        .metric-val {{ font-size: 30px; font-weight: 700; color: #1f2937; margin-top: 6px; }}
        .metric-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }}
        th {{ background: #f9fafb; padding: 12px; color: #4b5563; font-weight: 600; border-bottom: 2px solid #e5e7eb; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 26px;">Production Data & Model Drift Report</h1>
                <p style="margin: 4px 0 0 0; color: #6b7280; font-size: 13px;">Engine: <strong>Native SciPy / Scikit-Learn</strong> | Generated: {results['summary']['timestamp']} UTC</p>
            </div>
            <div style="background:{status_color};color:white;padding:10px 22px;border-radius:24px;font-weight:700;font-size:14px;letter-spacing:0.5px;">
                {status_text}
            </div>
        </div>

        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Production Sample Size</div>
                <div class="metric-val">{results['feature_audit']['sample_size']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Features Audited</div>
                <div class="metric-val">{results['feature_audit']['total_features']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Drifted Features</div>
                <div class="metric-val" style="color: {status_color};">{results['feature_audit']['drifted_features']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Drift Share</div>
                <div class="metric-val">{results['feature_audit']['drift_share'] * 100:.1f}%</div>
            </div>
        </div>

        {"<div style='background:#fef3c7;border-left:4px solid #f59e0b;padding:14px;margin-bottom:24px;border-radius:6px;font-size:13px;color:#92400e;'><strong>Sample Size Note:</strong> Audited sample size is small (" + str(results['feature_audit']['sample_size']) + " records). Statistical tests (KS-test and PSI) achieve high statistical confidence when sample size &ge; 30.</div>" if results['feature_audit']['sample_size'] < 30 else ""}

        <h3 style="margin: 24px 0 12px 0;">Prediction Drift (Model Output Distribution)</h3>
        {pred_summary}

        <h3 style="margin: 28px 0 12px 0;">Feature-Level Drift Breakdown</h3>
        <table>
            <thead>
                <tr>
                    <th>Feature Name</th>
                    <th>Data Type</th>
                    <th>Statistical Test</th>
                    <th>p-value</th>
                    <th>KS Statistic</th>
                    <th>PSI</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def run_drift_analysis(
    reference_path: str = "data/raw/train.csv",
    current_path: str = "data/monitoring/production_inferences.csv",
    html_report_path: str = "reports/monitoring/drift_report.html",
    test_results_path: str = "reports/monitoring/drift_summary.json"
) -> bool:
    """Main drift detection entrypoint called by Airflow or CLI."""
    if not os.path.exists(current_path):
        print(f"No inference log found at {current_path}. Run some predictions first.")
        return True

    ref_df = pd.read_csv(reference_path)
    try:
        cur_df = pd.read_csv(current_path, on_bad_lines='skip')
    except Exception:
        cur_df = pd.read_csv(current_path)

    # Exclude metadata columns
    ignore_cols = ["Attrition", "prediction", "probability", "timestamp"]
    feature_cols = [c for c in ref_df.columns if c in cur_df.columns and c not in ignore_cols]

    print(f"Auditing drift across {len(feature_cols)} features on {len(cur_df)} production records...")
    os.makedirs(os.path.dirname(html_report_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_results_path), exist_ok=True)

    # 1. Feature Drift Analysis
    feature_results = audit_feature_drift(ref_df, cur_df, feature_cols)

    # 2. Prediction Output Drift Analysis
    pred_results = audit_prediction_drift(ref_df, cur_df)

    # Overall decision
    overall_passed = bool(feature_results["all_passed"] and not pred_results.get("drift_detected", False))

    results_payload = {
        "summary": {
            "timestamp": datetime.utcnow().isoformat(),
            "all_passed": overall_passed,
            "drift_detected": not overall_passed
        },
        "feature_audit": feature_results,
        "prediction_drift": pred_results
    }

    # Save outputs
    with open(test_results_path, "w") as f:
        json.dump(results_payload, f, indent=4)

    generate_html_report(results_payload, html_report_path)

    print(f"Audit complete: Drift Detected = {not overall_passed} ({feature_results['drifted_features']} drifted features)")
    print(f"  -> JSON Summary: {test_results_path}")
    print(f"  -> HTML Dashboard: {html_report_path}")

    return overall_passed


if __name__ == "__main__":
    run_drift_analysis()
