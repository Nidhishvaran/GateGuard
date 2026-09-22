import json
import os
import sys
import subprocess
from flask import Flask, render_template, jsonify, request

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_LOGS_FILE = os.path.join(BASE_DIR, "raw_logs.txt")
UPLOADED_FILE = os.path.join(BASE_DIR, "uploaded_raw_logs.txt")
NORMALIZED_FILE = os.path.join(BASE_DIR, "normalized_events.json")
CLEANED_FILE = os.path.join(BASE_DIR, "cleaned_events.json")
VOCAB_FILE = os.path.join(BASE_DIR, "event_vocabulary.json")
SEQUENCE_FILE = os.path.join(BASE_DIR, "logbert_sequences.json")
RESULTS_FILE = os.path.join(BASE_DIR, "lanobert_results.json")


def load_json_safe(filepath, default=None):
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return default


def run_pipeline_for_file(input_log_path):
    """
    Executes the 3 intermediate steps for a given raw log file:
    1. demo.py (Normalization into OCSF)
    2. step2.py (Cleaning, Sequencing, and Vocabulary)
    3. model_changed.py (LogBERT Anomaly Scoring & Pre-Attack Prediction)
    """
    import demo
    import step2

    # Step 1: Normalize
    demo.process_file(input_log_path, NORMALIZED_FILE)

    # Step 2: Sessionize and sequence
    step2.main()

    # Step 3: Run model inference / scoring
    p = subprocess.run([sys.executable, "model_changed.py"], cwd=BASE_DIR, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"model_changed.py error: {p.stderr}")

    return load_json_safe(RESULTS_FILE, {})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze-file", methods=["POST"])
def api_analyze_file():
    target_file = None
    filename = "raw_logs.txt"

    if "file" in request.files and request.files["file"].filename:
        uploaded_file = request.files["file"]
        filename = uploaded_file.filename
        target_file = UPLOADED_FILE
        uploaded_file.save(target_file)
    elif os.path.exists(RAW_LOGS_FILE):
        target_file = RAW_LOGS_FILE
    else:
        return jsonify({"success": False, "error": "No file uploaded and raw_logs.txt not found."}), 400

    try:
        # Read raw lines
        with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        # Run pipeline
        results_data = run_pipeline_for_file(target_file)
        sequences = results_data.get("results", [])
        threshold = results_data.get("threshold", 0.6755)

        # Extract Pre-Attack Precursor details
        pre_attack_info = {
            "detected": False,
            "pattern_events": [],
            "source_ip": "None",
            "anomaly_score": 0.0,
            "threshold": threshold,
            "is_anomalous": False,
            "predicted_attack": None,
            "probed_ports": [],
            "risk_score": 0.0,
            "summary": "No pre-attack pattern detected."
        }

        for seq in sequences:
            if seq.get("sequence_type") == "pre_attack_precursor" or "PRE-ATTACK" in seq.get("classification", ""):
                pred = seq.get("early_prediction", {})
                pre_attack_info = {
                    "detected": True,
                    "pattern_events": seq.get("events", []),
                    "source_ip": seq.get("source", "Unknown"),
                    "anomaly_score": seq.get("anomaly_score", 0.0),
                    "threshold": threshold,
                    "is_anomalous": (seq.get("anomaly_score", 0.0) >= threshold) or ("ANOMALOUS" in seq.get("classification", "")),
                    "classification": seq.get("classification", "ANOMALOUS (PRE-ATTACK RECONNAISSANCE)"),
                    "predicted_attack": pred.get("predicted_attack", "SSH_BRUTE_FORCE (AUTH_SSHD_FAILURE)"),
                    "probed_ports": pred.get("probed_ports", ["445 (SMB)", "3389 (RDP)", "22 (SSH)"]),
                    "risk_score": pred.get("risk_score", 0.86),
                    "summary": pred.get("summary", "Pre-attack reconnaissance pattern detected before attack!")
                }
                break

        normalized_events = load_json_safe(NORMALIZED_FILE, [])

        return jsonify({
            "success": True,
            "filename": filename,
            "raw_lines_count": len(raw_lines),
            "raw_lines": raw_lines,
            "normalized_count": len(normalized_events),
            "normalized_sample": normalized_events[:5],
            "threshold": threshold,
            "total_sequences": len(sequences),
            "sequences": sequences,
            "pre_attack": pre_attack_info
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/current-data")
def api_current_data():
    """Returns currently saved analysis without re-running."""
    results_data = load_json_safe(RESULTS_FILE, {})
    raw_lines = []
    if os.path.exists(RAW_LOGS_FILE):
        with open(RAW_LOGS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

    sequences = results_data.get("results", [])
    threshold = results_data.get("threshold", 0.6755)

    pre_attack_info = {
        "detected": False,
        "pattern_events": [],
        "source_ip": "None",
        "anomaly_score": 0.0,
        "threshold": threshold,
        "is_anomalous": False,
        "predicted_attack": None,
        "probed_ports": [],
        "risk_score": 0.0,
        "summary": "No pre-attack pattern detected."
    }

    for seq in sequences:
        if seq.get("sequence_type") == "pre_attack_precursor" or "PRE-ATTACK" in seq.get("classification", ""):
            pred = seq.get("early_prediction", {})
            pre_attack_info = {
                "detected": True,
                "pattern_events": seq.get("events", []),
                "source_ip": seq.get("source", "Unknown"),
                "anomaly_score": seq.get("anomaly_score", 0.0),
                "threshold": threshold,
                "is_anomalous": (seq.get("anomaly_score", 0.0) >= threshold) or ("ANOMALOUS" in seq.get("classification", "")),
                "classification": seq.get("classification", "ANOMALOUS (PRE-ATTACK RECONNAISSANCE)"),
                "predicted_attack": pred.get("predicted_attack", "SSH_BRUTE_FORCE (AUTH_SSHD_FAILURE)"),
                "probed_ports": pred.get("probed_ports", ["445 (SMB)", "3389 (RDP)", "22 (SSH)"]),
                "risk_score": pred.get("risk_score", 0.86),
                "summary": pred.get("summary", "Pre-attack reconnaissance pattern detected before attack!")
            }
            break

    normalized_events = load_json_safe(NORMALIZED_FILE, [])

    return jsonify({
        "success": True,
        "filename": "raw_logs.txt",
        "raw_lines_count": len(raw_lines),
        "raw_lines": raw_lines,
        "normalized_count": len(normalized_events),
        "normalized_sample": normalized_events[:5],
        "threshold": threshold,
        "total_sequences": len(sequences),
        "sequences": sequences,
        "pre_attack": pre_attack_info
    })


if __name__ == "__main__":
    port = 5000
    print(f"\nSimple Log Security Web UI running at: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
