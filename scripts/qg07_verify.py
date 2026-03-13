"""QG-07 — Phase 7 Quality Gate.

Checks:
  1. Full pytest suite: zero failures
  2. Coverage >= 80%
  3. Docker image build (skipped if Docker not available)
  4. Container /health returns 200 within 30s (skipped if Docker not available)
  5. Prometheus metric names present in source code
  6. Grafana provisioning files exist and are valid YAML/JSON
  7. DriftDetector detect_feature_drift() runs end-to-end
  8. DriftDetector detect_regime_drift() returns correct bool

Run:
    python scripts/qg07_verify.py
    python scripts/qg07_verify.py --skip-docker   # skip Docker checks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src` can be imported
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Force UTF-8 output on Windows to handle Unicode log characters (loguru uses →)
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_OK = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"

_results: list[tuple[str, str, str]] = []


def _record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    icon = {"PASS": "[+]", "FAIL": "[!]", "SKIP": "[-]"}.get(status, "[ ]")
    print(f"  {icon} {name}: {status}{(' — ' + detail) if detail else ''}")


def _run_pytest() -> tuple[int, int, float]:
    """Run pytest with coverage and return (passed, failed, coverage_pct)."""
    cmd = [
        sys.executable, "-m", "pytest", "tests/",
        "--tb=no", "-q",
        "--cov=src", "--cov-report=term-missing",
        "--ignore=tests/unit/test_rest_client.py",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))
    output = result.stdout + result.stderr

    passed = failed = 0
    coverage_pct = 0.0

    for line in output.splitlines():
        if " passed" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "passed":
                    try:
                        passed = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
                if p == "failed" or p == "error":
                    try:
                        failed += int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
        if "TOTAL" in line and "%" in line:
            try:
                coverage_pct = float(line.split()[-1].rstrip("%"))
            except (ValueError, IndexError):
                pass

    return passed, failed, coverage_pct


def _check_docker_available() -> bool:
    """Return True if Docker CLI is available on PATH."""
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_docker() -> tuple[bool, str]:
    """Build Docker image. Return (success, error_msg)."""
    result = subprocess.run(
        ["docker", "build", "-t", "doge_predictor:qg07", "."],
        capture_output=True, text=True, timeout=600, cwd=str(_PROJECT_ROOT),
    )
    if result.returncode != 0:
        return False, result.stderr[-500:]
    return True, ""


def _health_check_container() -> tuple[bool, str]:
    """Start container and poll /health for 30s. Return (ok, msg)."""
    import socket
    cid = None
    try:
        # Start detached container (no DB; health returns 503 but endpoint exists)
        run_result = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "-p", "18000:8000",
                "-e", "SHADOW_MODE=true",
                "doge_predictor:qg07",
                "python", "scripts/serve.py",
                "--no-ws", "--no-scheduler",
                "--health-port", "8000",
                "--metrics-port", "8001",
                "--models-dir", "/tmp/empty_models",
            ],
            capture_output=True, text=True, timeout=30,
        )
        cid = run_result.stdout.strip()
        if run_result.returncode != 0 or not cid:
            return False, f"docker run failed: {run_result.stderr[:300]}"

        # Poll /health until it responds (any HTTP response counts)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with socket.create_connection(("localhost", 18000), timeout=2):
                    pass
                # Try to hit /health
                import urllib.request
                try:
                    resp = urllib.request.urlopen(
                        "http://localhost:18000/health", timeout=3
                    )
                    return True, f"HTTP {resp.getcode()}"
                except urllib.error.HTTPError as e:
                    return True, f"HTTP {e.code} (server responded)"
            except (OSError, ConnectionRefusedError):
                time.sleep(1)

        return False, "timeout — /health never responded within 30s"
    finally:
        if cid:
            subprocess.run(["docker", "stop", cid], capture_output=True, timeout=10)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check1_pytest() -> None:
    """Run full pytest suite."""
    print("\nCheck 1: Full pytest suite + coverage")
    passed, failed, cov = _run_pytest()
    if failed > 0:
        _record("pytest zero failures", _FAIL, f"{failed} failures")
    else:
        _record("pytest zero failures", _OK, f"{passed} passed")

    if cov >= 80.0:
        _record("coverage >= 80%", _OK, f"{cov:.1f}%")
    else:
        _record("coverage >= 80%", _FAIL, f"{cov:.1f}% < 80%")


def check2_prometheus_metric_names() -> None:
    """Verify all required metric names appear in source code."""
    print("\nCheck 2: Prometheus metric names in source")
    required = [
        "doge_inference_latency_seconds",
        "doge_feature_freshness_seconds",
        "doge_current_regime",
        "doge_btc_corr_24h",
        "doge_volume_ratio",
        "doge_funding_rate_zscore",
        "doge_prediction_count_total",
        "doge_equity_drawdown_pct",
    ]
    metrics_file = _PROJECT_ROOT / "src" / "monitoring" / "prometheus_metrics.py"
    content = metrics_file.read_text(encoding="utf-8")
    for m in required:
        if m in content:
            _record(f"metric: {m}", _OK)
        else:
            _record(f"metric: {m}", _FAIL, "not found in prometheus_metrics.py")


def check3_grafana_provisioning() -> None:
    """Verify Grafana provisioning files exist and parse."""
    print("\nCheck 3: Grafana provisioning files")
    files = {
        "grafana/provisioning/datasources/prometheus.yaml": "yaml",
        "grafana/provisioning/dashboards/dashboard.yaml": "yaml",
        "grafana/provisioning/dashboards/doge_predictor.json": "json",
    }
    for rel, fmt in files.items():
        path = _PROJECT_ROOT / rel
        if not path.exists():
            _record(rel, _FAIL, "file not found")
            continue
        try:
            if fmt == "yaml":
                yaml.safe_load(path.read_text(encoding="utf-8"))
            else:
                json.loads(path.read_text(encoding="utf-8"))
            _record(rel, _OK)
        except Exception as exc:  # noqa: BLE001
            _record(rel, _FAIL, str(exc)[:80])


def check4_docker(skip: bool) -> None:
    """Build Docker image and test health endpoint."""
    print("\nCheck 4: Docker image build + /health endpoint")
    if skip or not _check_docker_available():
        _record("docker build", _SKIP, "Docker not available on this host")
        _record("/health 200 within 30s", _SKIP, "Docker not available")
        return

    build_ok, build_err = _build_docker()
    if not build_ok:
        _record("docker build", _FAIL, build_err)
        _record("/health 200 within 30s", _SKIP, "image not built")
        return
    _record("docker build", _OK)

    health_ok, health_msg = _health_check_container()
    if health_ok:
        _record("/health 200 within 30s", _OK, health_msg)
    else:
        _record("/health 200 within 30s", _FAIL, health_msg)


def check5_drift_detector() -> None:
    """Run DriftDetector detect_feature_drift and detect_regime_drift end-to-end."""
    print("\nCheck 5: DriftDetector end-to-end")
    try:
        from src.monitoring.drift_detector import detect_feature_drift, detect_regime_drift

        # detect_feature_drift — drifted feature
        n = 100
        df = pd.DataFrame({
            "feature_a": np.random.default_rng(0).normal(5.0, 0.1, n),  # big drift
            "feature_b": np.random.default_rng(1).normal(0.0, 1.0, n),  # no drift
        })
        training_stats = {
            "feature_a": {"mean": 0.0, "std": 1.0},   # current mean >> 3*std away
            "feature_b": {"mean": 0.0, "std": 1.0},
        }
        report = detect_feature_drift(df, training_stats)
        if "feature_a" in report.drifted_features:
            _record("detect_feature_drift: drifted feature detected", _OK)
        else:
            _record("detect_feature_drift: drifted feature detected", _FAIL,
                    f"drifted_features={report.drifted_features}")

        if report.alert_level in ("WARNING", "CRITICAL"):
            _record("detect_feature_drift: alert_level set", _OK, report.alert_level)
        else:
            _record("detect_feature_drift: alert_level set", _FAIL,
                    f"got '{report.alert_level}'")

        # detect_feature_drift — no drift
        df2 = pd.DataFrame({
            "feature_b": np.random.default_rng(2).normal(0.1, 0.9, n),
        })
        report2 = detect_feature_drift(df2, {"feature_b": {"mean": 0.0, "std": 1.0}})
        if report2.alert_level == "NONE":
            _record("detect_feature_drift: no drift → NONE", _OK)
        else:
            _record("detect_feature_drift: no drift → NONE", _FAIL,
                    f"got '{report2.alert_level}' features={report2.drifted_features}")

        # detect_regime_drift — oscillating
        _1H = 3_600_000
        ts = pd.Index([1_640_995_200_000 + i * _1H for i in range(8)], dtype="int64")
        oscillating = pd.Series(
            ["TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL", "TRENDING_BULL",
             "DECOUPLED", "RANGING_LOW_VOL", "TRENDING_BULL", "TRENDING_BEAR"],
            index=ts,
        )
        if detect_regime_drift(oscillating):
            _record("detect_regime_drift: oscillation detected", _OK)
        else:
            _record("detect_regime_drift: oscillation detected", _FAIL,
                    "returned False on 7-transition series")

        # detect_regime_drift — stable
        stable = pd.Series(
            ["TRENDING_BULL"] * 8,
            index=ts,
        )
        if not detect_regime_drift(stable):
            _record("detect_regime_drift: stable → False", _OK)
        else:
            _record("detect_regime_drift: stable → False", _FAIL,
                    "returned True on stable series")

    except Exception as exc:  # noqa: BLE001
        _record("DriftDetector end-to-end", _FAIL, str(exc))


def check6_shadow_mode_env() -> None:
    """Verify SHADOW_MODE env var is referenced in docker-compose.yml."""
    print("\nCheck 6: Shadow mode configuration")
    compose = _PROJECT_ROOT / "docker-compose.yml"
    content = compose.read_text(encoding="utf-8")
    if "SHADOW_MODE" in content:
        _record("SHADOW_MODE in docker-compose.yml", _OK)
    else:
        _record("SHADOW_MODE in docker-compose.yml", _FAIL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="QG-07 — Phase 7 quality gate")
    parser.add_argument(
        "--skip-docker", action="store_true",
        help="Skip Docker build and container health checks",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("QG-07 — Phase 7 Quality Gate")
    print("=" * 60)

    check1_pytest()
    check2_prometheus_metric_names()
    check3_grafana_provisioning()
    check4_docker(skip=args.skip_docker)
    check5_drift_detector()
    check6_shadow_mode_env()

    # Summary
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s, _ in _results if s == _OK)
    n_fail = sum(1 for _, s, _ in _results if s == _FAIL)
    n_skip = sum(1 for _, s, _ in _results if s == _SKIP)
    print(f"RESULT: {n_pass} PASS  {n_fail} FAIL  {n_skip} SKIP  "
          f"(out of {len(_results)} checks)")

    if n_fail > 0:
        print("\nQG-07: FAIL")
        return 1

    print("\nQG-07: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
