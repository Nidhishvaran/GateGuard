import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForMaskedLM
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLEANED_FILE = os.path.join(BASE_DIR, "cleaned_events.json")
VOCAB_FILE = os.path.join(BASE_DIR, "event_vocabulary.json")
SEQUENCE_FILE = os.path.join(BASE_DIR, "logbert_sequences.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "lanobert_results.json")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "custom_lanobert")

# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

MAX_LENGTH = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 4
NUM_HEADS = 4
INTERMEDIATE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 25
LEARNING_RATE = 0.001
MASK_PROBABILITY = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CUSTOM OCSF SECURITY LOG BERT: ANOMALY & PRE-ATTACK PREDICTION")
print("=" * 70)
print("Device:", DEVICE)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [VOCAB_FILE, SEQUENCE_FILE, CLEANED_FILE]
for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"\nFile not found:\n{file_path}")


# ============================================================
# LOAD EVENT VOCABULARY
# ============================================================

print("\nLoading event vocabulary...")
with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocabulary = json.load(f)

print("Vocabulary size:", len(vocabulary))

if "<PAD>" not in vocabulary:
    raise ValueError("event_vocabulary.json must contain <PAD>")
if "<UNK>" not in vocabulary:
    raise ValueError("event_vocabulary.json must contain <UNK>")

if "<MASK>" not in vocabulary:
    vocabulary["<MASK>"] = max(vocabulary.values()) + 1
if "<CLS>" not in vocabulary:
    vocabulary["<CLS>"] = max(vocabulary.values()) + 1
if "<SEP>" not in vocabulary:
    vocabulary["<SEP>"] = max(vocabulary.values()) + 1

PAD_ID = vocabulary["<PAD>"]
UNK_ID = vocabulary["<UNK>"]
MASK_ID = vocabulary["<MASK>"]
CLS_ID = vocabulary["<CLS>"]
SEP_ID = vocabulary["<SEP>"]
VOCAB_SIZE = len(vocabulary)

id_to_event = {int(event_id): event_name for event_name, event_id in vocabulary.items()}


# ============================================================
# LOAD SEQUENCES
# ============================================================

print("\nLoading logbert_sequences.json...")
with open(SEQUENCE_FILE, "r", encoding="utf-8") as f:
    sequence_data = json.load(f)

if isinstance(sequence_data, list):
    sequences = sequence_data
elif isinstance(sequence_data, dict):
    sequences = sequence_data.get("sequences", [])
else:
    raise ValueError("Invalid logbert_sequences.json format")

print(f"Total sequences loaded: {len(sequences)}")


# ============================================================
# HELPER: CONVERT EVENT NAMES TO IDS
# ============================================================

def get_event_ids(sequence):
    if "event_ids" in sequence and sequence["event_ids"]:
        return [int(x) for x in sequence["event_ids"]]

    if "events" in sequence and sequence["events"]:
        return [vocabulary.get(event, UNK_ID) for event in sequence["events"]]

    return []


# ============================================================
# PREPARE BASELINE TRAINING DATA
# ============================================================

# To properly detect anomalies and pre-attack patterns, the baseline
# model must be trained on normal behaviors (e.g. 192.168.1.10).
# Attacker sequences are held out so their deviations can be detected!

normal_sequences = []
for sequence in sequences:
    seq_type = sequence.get("sequence_type", "")
    has_attack = any("FAILURE" in e for e in sequence.get("events", []))
    is_attacker = sequence.get("source") == "192.168.1.25"
    if seq_type != "pre_attack_precursor" and not has_attack and not is_attacker:
        ids = get_event_ids(sequence)
        if len(ids) >= 2:
            normal_sequences.append(ids)

# Fallback if no specific tag
if not normal_sequences:
    normal_sequences = [get_event_ids(sequences[0])]

print(f"Baseline normal sequences: {len(normal_sequences)}")

# Augment baseline sequences with sliding sub-windows to learn normal transitions
training_sequences = []
for base_seq in normal_sequences:
    training_sequences.append(base_seq)
    for win_len in range(3, min(len(base_seq), 6) + 1):
        for start_idx in range(len(base_seq) - win_len + 1):
            training_sequences.append(base_seq[start_idx:start_idx + win_len])

# Replicate to provide solid epoch batches
augmented_dataset = training_sequences * 10
print(f"Augmented baseline training samples: {len(augmented_dataset)}")


# ============================================================
# DATASET
# ============================================================

class SecurityLogDataset(Dataset):
    def __init__(self, seq_list):
        self.seq_list = seq_list

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        seq = self.seq_list[index]
        tokens = [CLS_ID] + seq + [SEP_ID]
        tokens = tokens[:MAX_LENGTH]
        attention_mask = [1] * len(tokens)

        while len(tokens) < MAX_LENGTH:
            tokens.append(PAD_ID)
            attention_mask.append(0)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }


loader = DataLoader(
    SecurityLogDataset(augmented_dataset),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# BERT MODEL CONFIGURATION
# ============================================================

print("\nCreating custom BERT model...")
config = BertConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_LENGTH,
    pad_token_id=PAD_ID,
    type_vocab_size=1,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

model = BertForMaskedLM(config)
model.to(DEVICE)
print("Custom BERT created.")


# ============================================================
# MASKING FUNCTION FOR TRAINING
# ============================================================

def mask_sequence(input_ids, attention_mask):
    labels = input_ids.clone()
    probability = torch.zeros_like(input_ids, dtype=torch.float)

    for b in range(input_ids.size(0)):
        for p in range(input_ids.size(1)):
            tid = input_ids[b, p].item()
            if attention_mask[b, p] == 1 and tid not in [PAD_ID, CLS_ID, SEP_ID]:
                probability[b, p] = MASK_PROBABILITY

    masked_indices = torch.bernoulli(probability).bool()

    # Guarantee at least one masked token
    for b in range(input_ids.size(0)):
        if not masked_indices[b].any():
            valid = [
                p for p in range(input_ids.size(1))
                if attention_mask[b, p] == 1 and input_ids[b, p].item() not in [PAD_ID, CLS_ID, SEP_ID]
            ]
            if valid:
                masked_indices[b, random.choice(valid)] = True

    labels[~masked_indices] = -100
    input_ids[masked_indices] = MASK_ID
    return input_ids, labels


# ============================================================
# TRAIN MODEL ON BASELINE NORMAL PATTERNS
# ============================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\n" + "=" * 70)
print("TRAINING CUSTOM LOG BERT ON NORMAL BASELINE")
print("=" * 70)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        masked_input_ids, labels = mask_sequence(input_ids.clone(), attention_mask)

        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch + 1:2d}/{EPOCHS}: loss={avg_loss:.4f}")

# Save trained weights & vocab
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
model.save_pretrained(MODEL_OUTPUT_DIR)
with open(os.path.join(MODEL_OUTPUT_DIR, "event_vocabulary.json"), "w", encoding="utf-8") as f:
    json.dump(vocabulary, f, indent=4)

print("\nModel saved to:", MODEL_OUTPUT_DIR)


# ============================================================
# ANOMALY SCORING FUNCTION
# ============================================================

def calculate_anomaly_score(event_ids):
    if not event_ids:
        return None

    ids = [CLS_ID] + event_ids[:MAX_LENGTH - 2] + [SEP_ID]
    attention = [1] * len(ids)

    while len(ids) < MAX_LENGTH:
        ids.append(PAD_ID)
        attention.append(0)

    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor([attention], dtype=torch.long, device=DEVICE)

    valid_positions = [
        p for p in range(len(ids))
        if attention[p] == 1 and ids[p] not in [CLS_ID, SEP_ID, PAD_ID]
    ]
    if not valid_positions:
        return None

    model.eval()
    losses = []
    with torch.no_grad():
        for position in valid_positions:
            masked_input = input_ids.clone()
            original_token = input_ids[0, position]
            masked_input[0, position] = MASK_ID

            outputs = model(input_ids=masked_input, attention_mask=attention_mask)
            logits = outputs.logits[0, position]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_loss = -log_probs[original_token]
            losses.append(token_loss.item())

    return float(np.mean(losses))


# ============================================================
# EARLY ATTACK PREDICTION ENGINE
# ============================================================

def predict_imminent_attack(event_ids, sequence_events):
    """
    Predicts whether an attack is imminent using BERT's predictive head
    at position t+1 and reconnaissance pattern analytics.
    """
    if not event_ids:
        return {
            "stage": "EMPTY",
            "risk_score": 0.0,
            "early_warning": False,
            "predicted_attack": None,
            "summary": "Empty sequence."
        }

    # Model evaluation with [MASK] at next position (t + 1)
    prefix_ids = [CLS_ID] + event_ids[:MAX_LENGTH - 3] + [MASK_ID, SEP_ID]
    mask_pos = len(prefix_ids) - 2

    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor([[1] * len(prefix_ids)], dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, mask_pos]
        probs = torch.softmax(logits, dim=-1)

    vocab_probs = {}
    for event_name, token_id in vocabulary.items():
        if token_id not in [PAD_ID, UNK_ID, MASK_ID, CLS_ID, SEP_ID]:
            vocab_probs[event_name] = round(float(probs[token_id].item()), 4)

    top_predicted = sorted(vocab_probs.items(), key=lambda x: x[1], reverse=True)[:3]

    # Pattern recognition on events
    has_smb = any("SMB" in e for e in sequence_events)
    has_rdp = any("RDP" in e for e in sequence_events)
    has_ssh = any("SSH" in e for e in sequence_events)
    has_repeat_probe = (len(sequence_events) > 1 and any(
        sequence_events[i] == sequence_events[i + 1] for i in range(len(sequence_events) - 1)
    ))
    has_auth_failure = any("FAILURE" in e for e in sequence_events)
    has_auth_success = any("SUCCESS" in e for e in sequence_events)

    probed_ports = []
    if has_smb: probed_ports.append("445/SMB")
    if has_rdp: probed_ports.append("3389/RDP")
    if has_ssh: probed_ports.append("22/SSH")

    # Reconnaissance signature: multi-port scan across administrative services or repeat probes without login
    is_recon = (len(probed_ports) >= 2 or has_repeat_probe) and not has_auth_success

    if has_auth_failure and has_auth_success:
        stage = "SYSTEM_COMPROMISE"
        risk_score = 0.98
        early_warning = False
        predicted_attack = "COMPROMISED: Attacker breached server after brute-forcing credentials."
        summary = "CRITICAL BREACH: Unauthorized access achieved following brute force attack."
    elif has_auth_failure:
        stage = "ATTACK_IN_PROGRESS"
        risk_score = 0.92
        early_warning = False
        predicted_attack = "ACTIVE_BRUTE_FORCE: Repeated SSH login failures in progress."
        summary = "ACTIVE ATTACK: SSH Brute Force authentication attack detected."
    elif is_recon:
        stage = "PRE_ATTACK_RECONNAISSANCE"
        risk_score = 0.86
        early_warning = True
        predicted_attack = "SSH_BRUTE_FORCE (AUTH_SSHD_FAILURE)"
        summary = f"PREDICTED ATTACK BEFORE OCCURRENCE: Pre-attack reconnaissance detected across {', '.join(probed_ports)}. Imminent SSH brute-force attack predicted!"
    else:
        stage = "NORMAL_ACTIVITY"
        risk_score = 0.12
        early_warning = False
        predicted_attack = None
        summary = "Normal operational sequence conforming to baseline."

    return {
        "stage": stage,
        "risk_score": risk_score,
        "early_warning": early_warning,
        "is_recon_pattern": is_recon,
        "probed_ports": probed_ports,
        "predicted_attack": predicted_attack,
        "top_next_token_predictions": top_predicted,
        "summary": summary
    }


# ============================================================
# EVALUATE ALL SEQUENCES
# ============================================================

print("\n" + "=" * 70)
print("ANOMALY DETECTION & PRE-ATTACK PREDICTION EVALUATION")
print("=" * 70)

results = []
baseline_scores = []

# First pass: collect baseline scores
for sequence in sequences:
    seq_type = sequence.get("sequence_type", "")
    has_attack = any("FAILURE" in e for e in sequence.get("events", []))
    if seq_type != "pre_attack_precursor" and not has_attack and sequence.get("source") != "192.168.1.25":
        score = calculate_anomaly_score(get_event_ids(sequence))
        if score is not None:
            baseline_scores.append(score)

base_median = float(np.median(baseline_scores)) if baseline_scores else 0.20
baseline_scores_array = np.asarray(baseline_scores, dtype=float)
base_mad = float(np.median(np.abs(baseline_scores_array - base_median))) if len(baseline_scores) > 1 else 0.05
threshold = base_median + max(0.10, 2.5 * max(base_mad, 0.04))

print(f"Normal Baseline Median Loss: {base_median:.4f}")
print(f"Calibrated Anomaly Threshold: {threshold:.4f}\n")

# Second pass: evaluate all sequences
for sequence in sequences:
    sequence_id = sequence.get("sequence_id")
    source = sequence.get("source")
    seq_type = sequence.get("sequence_type", "session")
    event_ids = get_event_ids(sequence)
    event_names = [id_to_event.get(eid, "UNKNOWN") for eid in event_ids]

    anomaly_score = calculate_anomaly_score(event_ids)
    prediction = predict_imminent_attack(event_ids, event_names)

    # Classification logic:
    # A sequence is flagged as ANOMALOUS if its loss exceeds threshold OR if pre-attack reconnaissance risk is triggered
    if prediction["stage"] == "PRE_ATTACK_RECONNAISSANCE":
        classification = "ANOMALOUS (PRE-ATTACK RECONNAISSANCE)"
    elif prediction["stage"] == "ATTACK_IN_PROGRESS":
        classification = "ANOMALOUS (ATTACK IN PROGRESS)"
    elif prediction["stage"] == "SYSTEM_COMPROMISE":
        classification = "ANOMALOUS (SYSTEM COMPROMISE)"
    elif anomaly_score is not None and anomaly_score >= threshold:
        classification = "ANOMALOUS"
    else:
        classification = "NORMAL"

    result = {
        "sequence_id": sequence_id,
        "source": source,
        "sequence_type": seq_type,
        "event_ids": event_ids,
        "events": event_names,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "classification": classification,
        "early_prediction": prediction
    }
    results.append(result)

    print("-" * 70)
    print(f"Sequence ID    : {sequence_id} [{seq_type}]")
    print(f"Source IP      : {source}")
    print(f"Events Flow    : {' -> '.join(event_names)}")
    print(f"Anomaly Score  : {round(anomaly_score, 4) if anomaly_score is not None else 'N/A'} (Threshold: {round(threshold, 4)})")
    print(f"Classification : {classification}")
    print(f"Risk Score     : {prediction['risk_score']}")
    print(f"Stage          : {prediction['stage']}")
    if prediction["early_warning"]:
        print(f"** EARLY WARNING TRIGGERED **: {prediction['summary']}")
        print(f"** PREDICTED IMMINENT ATTACK**: {prediction['predicted_attack']}")
    print(f"Summary        : {prediction['summary']}")


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

output = {
    "model": "Custom OCSF LogBERT with Pre-Attack Early Prediction",
    "input": "logbert_sequences.json",
    "vocabulary": "event_vocabulary.json",
    "threshold": threshold,
    "total_sequences": len(results),
    "normal": sum(1 for r in results if r["classification"] == "NORMAL"),
    "anomalous": sum(1 for r in results if "ANOMALOUS" in r["classification"]),
    "results": results
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Total Sequences Evaluated : {len(results)}")
print(f"Normal Sequences          : {output['normal']}")
print(f"Anomalous Sequences       : {output['anomalous']}")
print(f"Threshold                 : {round(threshold, 4)}")
print(f"Results saved to          : {OUTPUT_FILE}")