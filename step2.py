import json
import re
import hashlib
import sys
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "normalized_events.json"

OUTPUT_SEQUENCE_FILE = "logbert_sequences.json"
OUTPUT_VOCAB_FILE = "event_vocabulary.json"
OUTPUT_CLEAN_FILE = "cleaned_events.json"

# Events separated by this many seconds start a new session
SESSION_TIMEOUT = 300

# Maximum sequence length
SEQUENCE_LENGTH = 50


# ============================================================
# 1. LOAD OCSF JSON
# ============================================================

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON file containing a list of events.")

    return data


# ============================================================
# 2. NORMALIZE TEXT
# ============================================================

def normalize_text(value):
    if value is None:
        return "UNKNOWN"

    value = str(value).strip().upper()
    value = re.sub(r"\s+", " ", value)
    return value


# ============================================================
# 3. EXTRACT HELPERS
# ============================================================

def extract_field(message, field):
    if not message:
        return None
    pattern = rf"\b{re.escape(field)}=([^\s]+)"
    match = re.search(pattern, str(message), re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def get_direct_field(event, *names):
    for name in names:
        value = event.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def get_nested_field(event, *paths):
    for path in paths:
        current = event
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is not None and str(current).strip() != "":
            return current
    return None


# ============================================================
# 4. PORT TO SERVICE
# ============================================================

SERVICE_PORTS = {
    20: "FTP_DATA",
    21: "FTP",
    22: "SSH",
    23: "TELNET",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    143: "IMAP",
    443: "HTTPS",
    445: "SMB",
    3389: "RDP",
    3306: "MYSQL",
    5432: "POSTGRESQL",
    6379: "REDIS",
    8080: "HTTP_ALT",
    8443: "HTTPS_ALT",
    9200: "ELASTICSEARCH"
}


def service_from_port(port):
    try:
        port = int(port)
        return SERVICE_PORTS.get(port, "PORT_" + str(port))
    except:
        return "UNKNOWN_SERVICE"


# ============================================================
# 5. TEMPLATE CREATION
# ============================================================

def create_network_template(event):
    message = normalize_text(event.get("message", ""))
    protocol = get_nested_field(event, ("connection_info", "protocol_name"), ("network", "protocol"))
    if not protocol:
        protocol = get_direct_field(event, "protocol", "proto", "protocol_name")
    if not protocol:
        protocol = extract_field(message, "PROTO")
    protocol = normalize_text(protocol or "TCP")

    dst_port = get_nested_field(event, ("dst_endpoint", "port"), ("destination", "port"))
    if not dst_port:
        dst_port = get_direct_field(event, "dst_port", "destination_port", "dport", "DPT")
    if not dst_port:
        dst_port = extract_field(message, "DPT") or extract_field(message, "DST_PORT")

    action = get_direct_field(event, "disposition", "action") or extract_field(message, "ACTION") or "ALLOW"
    action = normalize_text(action)

    service = service_from_port(dst_port)
    return f"NET_{protocol}_{service}_{action}"


def create_auth_template(event):
    message = normalize_text(event.get("message", ""))
    process = get_nested_field(event, ("process", "name"), ("service", "name")) or get_direct_field(event, "service", "process")
    if not process:
        process = extract_field(message, "SERVICE") or "SSHD"
    process = normalize_text(process)

    result = get_direct_field(event, "result", "auth_result", "disposition") or extract_field(message, "RESULT")
    if result:
        result = normalize_text(result)
        if result in ("SUCCESS", "SUCCEEDED", "ACCEPT", "ACCEPTED"):
            return f"AUTH_{process}_SUCCESS"
        if result in ("FAILURE", "FAILED", "FAIL", "DENIED"):
            return f"AUTH_{process}_FAILURE"

    if "ACCEPTED" in message or "SUCCESS" in message:
        return f"AUTH_{process}_SUCCESS"
    if "FAILED" in message or "FAILURE" in message or "INVALID" in message:
        return f"AUTH_{process}_FAILURE"

    return f"AUTH_{process}_OTHER"


def create_event_template(event):
    supplied = get_direct_field(event, "event", "event_name", "event_template")
    if supplied:
        supplied = normalize_text(supplied)
        if supplied.startswith("NET_") or supplied.startswith("AUTH_"):
            return supplied

    class_uid = event.get("class_uid")
    if class_uid == 4001 or get_direct_field(event, "protocol", "proto", "dst_port") is not None:
        return create_network_template(event)
    elif class_uid == 3002 or get_direct_field(event, "service", "result") is not None:
        return create_auth_template(event)

    category = normalize_text(event.get("class_name", "UNKNOWN"))
    activity = normalize_text(event.get("activity_name", "UNKNOWN"))
    return f"{category}_{activity}"


# ============================================================
# 6. SOURCE / DESTINATION RESOLUTION
# ============================================================

def get_source(event):
    src_endpoint = event.get("src_endpoint") or {}
    if isinstance(src_endpoint, dict) and src_endpoint.get("ip"):
        return str(src_endpoint["ip"])

    direct_ip = get_direct_field(event, "src_ip", "source_ip", "source")
    if direct_ip:
        return str(direct_ip)

    message = event.get("message", "")
    return extract_field(message, "SRC") or extract_field(message, "source") or "UNKNOWN_SOURCE"


def get_destination(event):
    dst_endpoint = event.get("dst_endpoint") or {}
    if isinstance(dst_endpoint, dict) and dst_endpoint.get("ip"):
        return str(dst_endpoint["ip"])

    direct_ip = get_direct_field(event, "dst_ip", "destination_ip", "destination", "host")
    if direct_ip:
        return str(direct_ip)

    message = event.get("message", "")
    return extract_field(message, "DST") or extract_field(message, "host") or "UNKNOWN_DESTINATION"


def event_id_from_template(template):
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ============================================================
# 7. CLEAN & CORRELATE EVENTS
# ============================================================

def clean_events(events):
    raw_cleaned = []
    seen_hashes = set()

    for event in events:
        timestamp = event.get("time")
        if timestamp is None:
            continue
        try:
            timestamp = int(timestamp)
        except:
            continue

        raw_hash = event.get("raw_data_hash") or hash(json.dumps(event, sort_keys=True))
        if raw_hash in seen_hashes:
            continue
        seen_hashes.add(raw_hash)

        source = get_source(event)
        destination = get_destination(event)
        template = create_event_template(event)
        eid = event_id_from_template(template)

        cleaned_event = {
            "timestamp": timestamp,
            "source": source,
            "destination": destination,
            "class_uid": event.get("class_uid"),
            "event_template": template,
            "event_id": eid,
            "severity": event.get("severity_id", 1),
            "action": event.get("disposition", "UNKNOWN")
        }
        raw_cleaned.append(cleaned_event)

    raw_cleaned.sort(key=lambda x: x["timestamp"])

    # Correlate server-reported auth logs (where source == destination host)
    # with the recent remote client that connected on port 22/SSH
    recent_ssh_connections = []  # list of (timestamp, client_source, server_destination)
    cleaned = []

    for event in raw_cleaned:
        if "SSH" in event["event_template"] and event["event_template"].startswith("NET_"):
            recent_ssh_connections.append((event["timestamp"], event["source"], event["destination"]))

        if event["event_template"].startswith("AUTH_"):
            if event["source"] == event["destination"]:
                # Find matching remote client that connected to this destination within 5 seconds
                server_host = event["destination"]
                matching_client = None
                for t_conn, c_ip, s_ip in reversed(recent_ssh_connections):
                    if s_ip == server_host and 0 <= (event["timestamp"] - t_conn) <= 5000:
                        matching_client = c_ip
                        break
                if matching_client:
                    event["source"] = matching_client

        cleaned.append(event)

    return cleaned


# ============================================================
# 8. BUILD VOCABULARY & SEQUENCES
# ============================================================

def create_vocabulary(cleaned):
    templates = sorted(set(event["event_template"] for event in cleaned))
    vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    for index, template in enumerate(templates, start=2):
        vocabulary[template] = index
    return vocabulary


def group_by_source(events):
    groups = defaultdict(list)
    for event in events:
        source = event["source"]
        groups[source].append(event)
    return groups


def create_sessions(events):
    sessions = []
    current = []
    previous_time = None

    for event in events:
        current_time = event["timestamp"]
        if previous_time is not None:
            gap = (current_time - previous_time) / 1000.0
            if gap > SESSION_TIMEOUT:
                if current:
                    sessions.append(current)
                current = []
        current.append(event)
        previous_time = current_time

    if current:
        sessions.append(current)
    return sessions


def build_logbert_dataset(cleaned):
    groups = group_by_source(cleaned)
    dataset = []
    sequence_number = 0

    for source, events in groups.items():
        sessions = create_sessions(events)

        for session in sessions:
            if len(session) < 2:
                continue

            full_templates = [e["event_template"] for e in session]
            full_ids = [e["event_id"] for e in session]

            # Check if this session contains an attack progression (e.g. AUTH_SSHD_FAILURE)
            has_attack = any("FAILURE" in t for t in full_templates)
            first_attack_idx = -1
            if has_attack:
                for idx, t in enumerate(full_templates):
                    if "FAILURE" in t:
                        first_attack_idx = idx
                        break

            # If an attack sequence exists and there are pre-attack events,
            # explicitly create a pre-attack precursor sequence as well as the full sequence
            if has_attack and first_attack_idx >= 2:
                precursor_slice = session[:first_attack_idx]
                dataset.append({
                    "sequence_id": sequence_number,
                    "source": source,
                    "sequence_type": "pre_attack_precursor",
                    "start_time": precursor_slice[0]["timestamp"],
                    "end_time": precursor_slice[-1]["timestamp"],
                    "length": len(precursor_slice),
                    "event_ids": [e["event_id"] for e in precursor_slice],
                    "events": [e["event_template"] for e in precursor_slice]
                })
                sequence_number += 1

            # Full session sequence
            dataset.append({
                "sequence_id": sequence_number,
                "source": source,
                "sequence_type": "full_session" if not has_attack else "attack_session",
                "start_time": session[0]["timestamp"],
                "end_time": session[-1]["timestamp"],
                "length": len(session),
                "event_ids": full_ids,
                "events": full_templates
            })
            sequence_number += 1

    return dataset


def convert_to_token_ids(dataset, vocabulary):
    for sequence in dataset:
        sequence["event_ids"] = [
            vocabulary.get(event, vocabulary["<UNK>"])
            for event in sequence["events"]
        ]
    return dataset


def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("OCSF -> LogBERT PREPROCESSING PIPELINE")
    print("=" * 60)

    print("\n[1] Loading OCSF JSON...")
    events = load_json(INPUT_FILE)
    print(f"Loaded {len(events)} events.")

    print("\n[2] Cleaning and normalizing...")
    cleaned = clean_events(events)
    print(f"Clean events: {len(cleaned)}")

    print("\n[3] Creating event vocabulary...")
    vocabulary = create_vocabulary(cleaned)
    print(f"Unique event templates: {len(vocabulary)}")

    print("\n[4] Creating temporal sequences (including pre-attack precursor slices)...")
    dataset = build_logbert_dataset(cleaned)
    print(f"Sequences created: {len(dataset)}")

    print("\n[4A] Event template validation:")
    all_templates = sorted(set(event["event_template"] for event in cleaned))
    for template in all_templates:
        print("  ", template)

    bad_templates = [
        template for template in all_templates
        if "UNKNOWN_SERVICE" in template or template == "UNKNOWN_UNKNOWN"
    ]

    if bad_templates:
        print("\nWARNING: Unknown templates remain:")
        for template in bad_templates:
            print("  ", template)
    else:
        print("\nOK: All templates are valid OCSF canonical security events.")

    print("\n[5] Converting templates to token IDs...")
    dataset = convert_to_token_ids(dataset, vocabulary)

    print("\n[6] Saving output files...")
    save_json(cleaned, OUTPUT_CLEAN_FILE)
    save_json(vocabulary, OUTPUT_VOCAB_FILE)
    save_json(dataset, OUTPUT_SEQUENCE_FILE)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Cleaned events : {OUTPUT_CLEAN_FILE}")
    print(f"Vocabulary     : {OUTPUT_VOCAB_FILE}")
    print(f"Sequences      : {OUTPUT_SEQUENCE_FILE}")

    print("\nGenerated Sequences:")
    for seq in dataset:
        print(f"\n  Sequence ID {seq['sequence_id']} [{seq.get('sequence_type', 'session')}]:")
        print(f"  Source : {seq['source']}")
        print(f"  Events : {' -> '.join(seq['events'])}")


if __name__ == "__main__":
    main()