import re
import json
import hashlib
import sys
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# CONFIGURATION
# ============================================================

SCHEMA_VERSION = "OCSF"
OUTPUT_FILE = "normalized_events.json"


# ============================================================
# TIME
# ============================================================

def current_time_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def parse_time(value):
    if not value:
        return current_time_ms()

    clean_val = str(value).strip()
    if clean_val.endswith("Z"):
        clean_val = clean_val[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(clean_val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%b %d %H:%M:%S",
        "%d/%b/%Y:%H:%M:%S %z"
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(str(value).strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass

    return current_time_ms()


def extract_log_timestamp(log):
    # Match ISO timestamp at the start of log, e.g. 2026-09-22T09:00:01Z
    match = re.match(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)", log)
    if match:
        return parse_time(match.group(1))

    # Match syslog timestamp, e.g. Sep 22 09:00:01
    match_syslog = re.match(r"^([A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2})", log)
    if match_syslog:
        return parse_time(match_syslog.group(1))

    return current_time_ms()


# ============================================================
# GENERAL HELPERS
# ============================================================

def extract(pattern, text, group=1):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(group)
    return None


def to_int(value):
    try:
        return int(value)
    except:
        return None


def hash_raw_log(log):
    return hashlib.sha256(log.encode("utf-8")).hexdigest()


# ============================================================
# BASE OCSF EVENT
# ============================================================

def base_event(
    log,
    category_uid,
    category_name,
    class_uid,
    class_name,
    activity_id,
    activity_name,
    severity_id=1,
    status_id=1,
    timestamp=None
):
    if timestamp is None:
        timestamp = extract_log_timestamp(log)

    event = {
        "category_uid": category_uid,
        "category_name": category_name,
        "class_uid": class_uid,
        "class_name": class_name,
        "activity_id": activity_id,
        "activity_name": activity_name,
        "severity_id": severity_id,
        "status_id": status_id,
        "time": timestamp,
        "message": log,
        "metadata": {
            "version": SCHEMA_VERSION
        },
        "raw_data": log,
        "raw_data_hash": hash_raw_log(log)
    }

    return event


# ============================================================
# NETWORK LOG
# ============================================================

def parse_network_log(log):
    src_ip = extract(r"(?:SRC_IP|SRC|source)[=:]\s*([0-9a-fA-F:.]+)", log)
    dst_ip = extract(r"(?:DST_IP|DST|destination)[=:]\s*([0-9a-fA-F:.]+)", log)
    src_port = extract(r"(?:SPORT|SRC_PORT)[=:]\s*(\d+)", log)
    dst_port = extract(r"(?:DPORT|DST_PORT)[=:]\s*(\d+)", log)
    protocol = extract(r"(?:PROTO|PROTOCOL)[=:]\s*([A-Za-z0-9_-]+)", log)
    action = extract(r"(?:ACTION|ACT)[=:]\s*([A-Za-z0-9_-]+)", log)
    bytes_sent = extract(r"(?:BYTES|SRC_BYTES)[=:]\s*(\d+)", log)
    packets = extract(r"(?:PACKETS|SRC_PACKETS)[=:]\s*(\d+)", log)
    event_name = extract(r"(?:EVENT|EVENT_NAME)[=:]\s*([A-Za-z0-9_-]+)", log)

    event = base_event(
        log,
        category_uid=4,
        category_name="Network Activity",
        class_uid=4001,
        class_name="Network Activity",
        activity_id=1,
        activity_name="Open",
        severity_id=1
    )

    event["src_endpoint"] = {
        "ip": src_ip,
        "port": to_int(src_port)
    }

    event["dst_endpoint"] = {
        "ip": dst_ip,
        "port": to_int(dst_port)
    }

    event["connection_info"] = {
        "protocol_name": protocol,
        "direction": None
    }

    event["traffic"] = {
        "bytes": to_int(bytes_sent),
        "packets": to_int(packets)
    }

    event["disposition"] = action or "ALLOW"
    if event_name:
        event["event"] = event_name
        event["event_template"] = event_name

    return event


# ============================================================
# SSH / AUTHENTICATION LOG
# ============================================================

def parse_auth_log(log):
    username = extract(r"(?:user|for)[=:\s]+([A-Za-z0-9._$@-]+)", log)
    src_ip = extract(r"(?:source|src_ip|from)[=:\s]+([0-9a-fA-F:.]+)", log)
    dst_ip = extract(r"(?:host|dst_ip|destination)[=:\s]+([0-9a-fA-F:.]+)", log)
    src_port = extract(r"(?:port|sport|src_port)[=:\s]+(\d+)", log)
    
    process = extract(r"service[=:\s]+([A-Za-z0-9_.-]+)", log)
    if not process:
        process = extract(r"\b(sshd|sudo|login|sshd-session)\b", log)

    event_name = extract(r"(?:EVENT|EVENT_NAME)[=:]\s*([A-Za-z0-9_-]+)", log)
    result_val = extract(r"(?:result|status|auth_result)[=:]\s*([A-Za-z0-9_-]+)", log)

    failed = bool(
        (result_val and result_val.upper() in ["FAILURE", "FAILED", "FAIL", "DENIED"])
        or (event_name and "FAILURE" in event_name.upper())
        or re.search(r"failed|failure|invalid", log, re.IGNORECASE)
    )

    successful = bool(
        (result_val and result_val.upper() in ["SUCCESS", "SUCCEEDED", "ACCEPT", "ACCEPTED"])
        or (event_name and "SUCCESS" in event_name.upper())
        or re.search(r"accepted|successful|success", log, re.IGNORECASE)
    )

    if failed:
        activity_name = "Logon"
        status_id = 2
        severity_id = 3
        disposition = "FAILURE"
    elif successful:
        activity_name = "Logon"
        status_id = 1
        severity_id = 1
        disposition = "SUCCESS"
    else:
        activity_name = "Logon"
        status_id = 1
        severity_id = 1
        disposition = "UNKNOWN"

    event = base_event(
        log,
        category_uid=3,
        category_name="Identity & Access Management",
        class_uid=3002,
        class_name="Authentication",
        activity_id=1,
        activity_name=activity_name,
        severity_id=severity_id,
        status_id=status_id
    )

    event["actor"] = {
        "user": {
            "name": username
        }
    }

    event["src_endpoint"] = {
        "ip": src_ip,
        "port": to_int(src_port)
    }

    event["dst_endpoint"] = {
        "ip": dst_ip,
        "port": 22 if (process and "sshd" in str(process).lower()) else None
    }

    event["process"] = {
        "name": process or "sshd"
    }

    event["disposition"] = disposition

    if event_name:
        event["event"] = event_name
        event["event_template"] = event_name
    else:
        event_tag = f"AUTH_{str(process or 'SSHD').upper()}_{disposition}"
        event["event"] = event_tag
        event["event_template"] = event_tag

    return event


# ============================================================
# SYSLOG / SYSTEM EVENT
# ============================================================

def parse_syslog(log):
    hostname = extract(r"^[A-Z][a-z]{2}\s+\d+\s+\d+:\d+:\d+\s+(\S+)", log)
    process = extract(r"\s([A-Za-z0-9_.-]+)(?:\[\d+\])?:", log)
    pid = extract(r"\[(\d+)\]", log)
    error = bool(re.search(r"error|failed|failure|critical|denied", log, re.IGNORECASE))

    event = base_event(
        log,
        category_uid=1,
        category_name="System Activity",
        class_uid=1001,
        class_name="System Activity",
        activity_id=1,
        activity_name="Other",
        severity_id=4 if error else 1,
        status_id=2 if error else 1
    )

    event["device"] = {
        "hostname": hostname
    }

    event["process"] = {
        "name": process,
        "pid": to_int(pid)
    }

    return event


# ============================================================
# HTTP LOG
# ============================================================

def parse_http_log(log):
    method = extract(r'"(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)', log)
    uri = extract(r'"(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(\S+)', log)
    status = extract(r'"\s+(\d{3})', log)
    src_ip = extract(r"^(\d+\.\d+\.\d+\.\d+)", log)
    user_agent = extract(r'"([^"]*)"\s*$', log)

    event = base_event(
        log,
        category_uid=4,
        category_name="Network Activity",
        class_uid=4003,
        class_name="HTTP Activity",
        activity_id=1,
        activity_name="Access",
        severity_id=1
    )

    event["src_endpoint"] = {
        "ip": src_ip
    }

    event["http_request"] = {
        "http_method": method,
        "url": {
            "path": uri
        },
        "user_agent": user_agent
    }

    event["http_response"] = {
        "code": to_int(status)
    }

    return event


# ============================================================
# DNS LOG
# ============================================================

def parse_dns_log(log):
    query = extract(r"(?:QUERY|QNAME)[=:]\s*([A-Za-z0-9._-]+)", log)
    query_type = extract(r"(?:TYPE|QTYPE)[=:]\s*([A-Za-z0-9]+)", log)
    response = extract(r"(?:RESPONSE|ANSWER)[=:]\s*([A-Za-z0-9._:-]+)", log)
    src_ip = extract(r"(?:SRC|SRC_IP)[=:]\s*([0-9a-fA-F:.]+)", log)

    event = base_event(
        log,
        category_uid=4,
        category_name="Network Activity",
        class_uid=4002,
        class_name="DNS Activity",
        activity_id=1,
        activity_name="Query",
        severity_id=1
    )

    event["src_endpoint"] = {
        "ip": src_ip
    }

    event["dns_query"] = {
        "hostname": query,
        "type": query_type
    }

    event["dns_response"] = {
        "answer": response
    }

    return event


# ============================================================
# UNKNOWN LOG
# ============================================================

def parse_unknown_log(log):
    event = base_event(
        log,
        category_uid=0,
        category_name="Other",
        class_uid=0,
        class_name="Unknown",
        activity_id=0,
        activity_name="Unknown",
        severity_id=1
    )

    event["unmapped"] = {
        "original_log": log
    }

    return event


# ============================================================
# LOG TYPE DETECTION
# ============================================================

def detect_log_type(log):
    # Authentication
    if (
        re.search(r"\b(?:service=(?:sshd|login|auth)|auth_method=|result=(?:SUCCESS|FAILURE)|event=AUTH_)", log, re.I)
        or re.search(r"failed password|accepted password|authentication failure|invalid user|session opened|session closed", log, re.I)
    ):
        return "auth"

    # Network / firewall
    if (
        (re.search(r"\b(?:SRC|SRC_IP|source)[=:]", log, re.I) and (re.search(r"\b(?:DST|DST_IP|dst_port|DPORT|DPT)[=:]", log, re.I) or re.search(r"\bevent=NET_", log, re.I)))
        or (re.search(r"SRC(?:_IP)?[=:]", log, re.I) and re.search(r"DST(?:_IP)?[=:]", log, re.I))
    ):
        return "network"

    # HTTP
    if re.search(r'"(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s', log, re.I):
        return "http"

    # DNS
    if re.search(r"(QUERY|QNAME|QTYPE|DNS_QUERY)[=:]", log, re.I):
        return "dns"

    # Syslog
    if re.match(r"^[A-Z][a-z]{2}\s+\d+\s+\d+:\d+:\d+", log):
        return "syslog"

    return "unknown"


# ============================================================
# MASTER NORMALIZER
# ============================================================

def normalize_log(log):
    log_type = detect_log_type(log)

    if log_type == "network":
        return parse_network_log(log)
    elif log_type == "auth":
        return parse_auth_log(log)
    elif log_type == "http":
        return parse_http_log(log)
    elif log_type == "dns":
        return parse_dns_log(log)
    elif log_type == "syslog":
        return parse_syslog(log)
    else:
        return parse_unknown_log(log)


# ============================================================
# PROCESS LOG FILE
# ============================================================

def process_file(input_file, output_file):
    events = []

    with open(input_file, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                event = normalize_log(line)
                events.append(event)
            except Exception as e:
                print("Parsing error:", e)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(events, file, indent=2)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_file("raw_logs.txt", OUTPUT_FILE)
    print(f"Normalized logs saved to {OUTPUT_FILE}")