"""
Pipeline Tests
===============
Unit tests for S3 utilities, MLflow tracking, and PSI drift monitoring.

Run: python -m pytest tests/test_pipeline.py -v
"""

import os
import pytest
import numpy as np


# ── Test: PSI Drift Monitoring ─────────────────────────────────────────
class TestPSIDriftMonitor:

    def test_identical_distributions_zero_psi(self):
        """PSI of identical distributions should be 0."""
        from pipeline.drift_monitor import compute_psi

        np.random.seed(42)
        dist = np.random.normal(3.5, 1.0, 10000)
        psi = compute_psi(dist, dist)

        assert psi == 0.0, f"Expected PSI=0 for identical distributions, got {psi}"

    def test_similar_distributions_low_psi(self):
        """PSI of similar distributions should be < 0.1 (PASS)."""
        from pipeline.drift_monitor import compute_psi

        np.random.seed(42)
        dist_a = np.random.normal(3.5, 1.0, 10000)
        dist_b = np.random.normal(3.5, 1.0, 10000)  # Same params, different samples
        psi = compute_psi(dist_a, dist_b)

        assert psi < 0.1, f"Expected PSI < 0.1 for similar distributions, got {psi}"

    def test_shifted_distributions_high_psi(self):
        """PSI of significantly shifted distributions should be > 0.2 (FAIL)."""
        from pipeline.drift_monitor import compute_psi

        np.random.seed(42)
        dist_a = np.random.normal(3.5, 1.0, 10000)
        dist_b = np.random.normal(5.5, 1.0, 10000)  # Mean shifted by 2.0
        psi = compute_psi(dist_a, dist_b)

        assert psi > 0.2, f"Expected PSI > 0.2 for shifted distributions, got {psi}"

    def test_drift_check_passes_no_shift(self, baseline_distributions):
        """Drift check should PASS when distributions haven't changed."""
        from pipeline.drift_monitor import run_drift_check

        baseline = np.load(baseline_distributions)
        current = {k: baseline[k] for k in baseline.files}

        passed, results = run_drift_check(baseline_distributions, current)

        assert passed, f"Should PASS with no shift, but got: {results}"
        for feature, data in results.items():
            assert data["status"] == "PASS", f"{feature} should be PASS, got {data['status']}"

    def test_drift_check_fails_high_shift(self, baseline_distributions):
        """Drift check should FAIL when distributions shift significantly."""
        from pipeline.drift_monitor import run_drift_check

        baseline = np.load(baseline_distributions)
        np.random.seed(42)

        # Shift all features significantly
        drifted = {}
        for k in baseline.files:
            drifted[k] = baseline[k] + np.random.normal(2.0, 0.5, len(baseline[k]))

        passed, results = run_drift_check(baseline_distributions, drifted)

        assert not passed, "Should FAIL with high shift"

    def test_psi_symmetric(self):
        """PSI should be approximately symmetric (PSI(A,B) ≈ PSI(B,A))."""
        from pipeline.drift_monitor import compute_psi

        np.random.seed(42)
        dist_a = np.random.normal(3.5, 1.0, 10000)
        dist_b = np.random.normal(4.0, 1.0, 10000)

        psi_ab = compute_psi(dist_a, dist_b)
        psi_ba = compute_psi(dist_b, dist_a)

        # PSI isn't perfectly symmetric but should be close
        assert abs(psi_ab - psi_ba) < 0.05, f"PSI not symmetric: {psi_ab:.4f} vs {psi_ba:.4f}"

    def test_format_drift_report(self, baseline_distributions):
        """Drift report should be a formatted string."""
        from pipeline.drift_monitor import run_drift_check, format_drift_report

        baseline = np.load(baseline_distributions)
        current = {k: baseline[k] for k in baseline.files}
        _, results = run_drift_check(baseline_distributions, current)

        report = format_drift_report(results)
        assert isinstance(report, str)
        assert "DRIFT MONITORING REPORT" in report
        assert "PASS" in report


# ── Test: MLflow Promotion Logic ───────────────────────────────────────
class TestMLflowPromotion:

    def test_promote_when_better(self):
        """Model should be promoted when it beats production."""
        from pipeline.mlflow_tracking import promote_model

        promoted, msg = promote_model(
            run_id=None,
            current_metrics={"NDCG@10": 0.45, "HR@10": 0.70},
            production_metrics={"NDCG@10": 0.40, "HR@10": 0.65},
        )

        assert promoted, f"Should promote, but got: {msg}"
        assert "PROMOTED" in msg

    def test_block_when_worse(self):
        """Model should be blocked when it's worse than production."""
        from pipeline.mlflow_tracking import promote_model

        promoted, msg = promote_model(
            run_id=None,
            current_metrics={"NDCG@10": 0.35, "HR@10": 0.55},
            production_metrics={"NDCG@10": 0.40, "HR@10": 0.65},
        )

        assert not promoted, f"Should block, but got: {msg}"
        assert "BLOCKED" in msg

    def test_promote_first_deployment(self):
        """First model should always be promoted (no production baseline)."""
        from pipeline.mlflow_tracking import promote_model

        promoted, msg = promote_model(
            run_id=None,
            current_metrics={"NDCG@10": 0.30},
            production_metrics={},
        )

        assert promoted, f"First deployment should promote, but got: {msg}"

    def test_block_when_equal(self):
        """Model should be blocked when equal to production (must BEAT, not tie)."""
        from pipeline.mlflow_tracking import promote_model

        promoted, msg = promote_model(
            run_id=None,
            current_metrics={"NDCG@10": 0.40},
            production_metrics={"NDCG@10": 0.40},
        )

        assert not promoted, f"Equal metrics should block, but got: {msg}"

    def test_block_missing_metric(self):
        """Should block if promotion metric is missing from results."""
        from pipeline.mlflow_tracking import promote_model

        promoted, msg = promote_model(
            run_id=None,
            current_metrics={"HR@10": 0.70},  # Missing NDCG@10
            production_metrics={"NDCG@10": 0.40},
        )

        assert not promoted, f"Missing metric should block, but got: {msg}"


# ── Test: S3 Utilities ─────────────────────────────────────────────────
class TestS3Utils:

    def test_bucket_name_config(self):
        """S3 bucket name should be configured."""
        from pipeline.s3_utils import S3_BUCKET
        assert S3_BUCKET is not None
        assert len(S3_BUCKET) > 0

    def test_prefix_constants(self):
        """S3 prefix constants should be properly defined."""
        from pipeline.s3_utils import PREFIX_RAW, PREFIX_FEATURES, PREFIX_MODELS, PREFIX_MLFLOW
        assert PREFIX_RAW == "raw/"
        assert PREFIX_FEATURES == "features/"
        assert PREFIX_MODELS == "models/"
        assert PREFIX_MLFLOW == "mlflow-artifacts/"

    def test_verify_setup_without_creds(self):
        """verify_setup should return False without AWS credentials."""
        # Only run if credentials are NOT set
        if os.getenv("AWS_ACCESS_KEY_ID"):
            pytest.skip("AWS credentials are set — skipping no-creds test")

        from pipeline.s3_utils import verify_setup
        result = verify_setup("nonexistent-bucket-xyz")
        assert result is False
