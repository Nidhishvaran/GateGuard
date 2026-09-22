import json
import os
import random

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertConfig,
    BertForMaskedLM
)

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CLEANED_FILE = os.path.join(
    BASE_DIR,
    "cleaned_events.json"
)

VOCAB_FILE = os.path.join(
    BASE_DIR,
    "event_vocabulary.json"
)

SEQUENCE_FILE = os.path.join(
    BASE_DIR,
    "logbert_sequences.json"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "lanobert_results.json"
)

MODEL_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "custom_lanobert"
)


# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

MAX_LENGTH = 32

HIDDEN_SIZE = 128

NUM_LAYERS = 4

NUM_HEADS = 4

INTERMEDIATE_SIZE = 256

BATCH_SIZE = 4

EPOCHS = 30

LEARNING_RATE = 0.001

MASK_PROBABILITY = 0.15

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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

print(
    "CUSTOM OCSF SECURITY LOG BERT"
)

print("=" * 70)

print(
    "Device:",
    DEVICE
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [

    VOCAB_FILE,

    SEQUENCE_FILE,

    CLEANED_FILE

]


for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nFile not found:\n{file_path}"
        )


# ============================================================
# LOAD EVENT VOCABULARY
# ============================================================

print(
    "\nLoading event vocabulary..."
)


with open(
    VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:

    vocabulary = json.load(f)


print(
    "Vocabulary size:",
    len(vocabulary)
)


# ------------------------------------------------------------
# Required special tokens
# ------------------------------------------------------------

if "<PAD>" not in vocabulary:

    raise ValueError(
        "event_vocabulary.json must contain <PAD>"
    )

if "<UNK>" not in vocabulary:

    raise ValueError(
        "event_vocabulary.json must contain <UNK>"
    )


# Add MASK if it doesn't exist
# ------------------------------------------------------------

if "<MASK>" not in vocabulary:

    next_id = max(
        vocabulary.values()
    ) + 1

    vocabulary["<MASK>"] = next_id

    print(
        "Added <MASK> token:",
        next_id
    )


PAD_ID = vocabulary["<PAD>"]

UNK_ID = vocabulary["<UNK>"]

MASK_ID = vocabulary["<MASK>"]

VOCAB_SIZE = len(vocabulary)


# ============================================================
# CREATE REVERSE VOCABULARY
# ============================================================

id_to_event = {

    int(event_id): event_name

    for event_name, event_id
    in vocabulary.items()

}


# ============================================================
# LOAD SEQUENCES
# ============================================================

print(
    "\nLoading logbert_sequences.json..."
)


with open(
    SEQUENCE_FILE,
    "r",
    encoding="utf-8"
) as f:

    sequence_data = json.load(f)


if isinstance(
    sequence_data,
    list
):

    sequences = sequence_data

elif isinstance(
    sequence_data,
    dict
):

    sequences = sequence_data.get(
        "sequences",
        []
    )

else:

    raise ValueError(
        "Invalid logbert_sequences.json format"
    )


print(
    "Sequences:",
    len(sequences)
)


# ============================================================
# DISPLAY INPUT DATA
# ============================================================

print(
    "\nInput sequences:"
)

for sequence in sequences:

    print(
        "\nSequence:",
        sequence.get(
            "sequence_id"
        )
    )

    print(
        "Source:",
        sequence.get(
            "source"
        )
    )

    print(
        "Events:",
        sequence.get(
            "events"
        )
    )

    print(
        "IDs:",
        sequence.get(
            "event_ids"
        )
    )


# ============================================================
# CONVERT EVENT NAMES TO YOUR CUSTOM IDs
# ============================================================

def get_event_ids(sequence):

    # --------------------------------------------------------
    # Prefer your already-created event_ids
    # --------------------------------------------------------

    if "event_ids" in sequence:

        ids = sequence[
            "event_ids"
        ]

        return [
            int(x)
            for x in ids
        ]


    # --------------------------------------------------------
    # Fallback to event names
    # --------------------------------------------------------

    if "events" in sequence:

        ids = []

        for event in sequence["events"]:

            event_id = vocabulary.get(
                event,
                UNK_ID
            )

            ids.append(
                event_id
            )

        return ids


    return []


# ============================================================
# PREPARE TRAINING SEQUENCES
# ============================================================

training_sequences = []


for sequence in sequences:

    ids = get_event_ids(
        sequence
    )

    if len(ids) > 0:

        training_sequences.append(
            ids
        )


print(
    "\nUsable sequences:",
    len(training_sequences)
)


# ============================================================
# IMPORTANT WARNING
# ============================================================

if len(training_sequences) < 20:

    print(
        "\nWARNING:"
    )

    print(
        "You currently have only",
        len(training_sequences),
        "sequences."
    )

    print(
        "This is enough to test the pipeline,"
    )

    print(
        "but NOT enough to train a reliable"
    )

    print(
        "anomaly detector."
    )


# ============================================================
# DATASET
# ============================================================

class SecurityLogDataset(
    Dataset
):

    def __init__(
        self,
        sequences
    ):

        self.sequences = sequences


    def __len__(
        self
    ):

        return len(
            self.sequences
        )


    def __getitem__(
        self,
        index
    ):

        sequence = self.sequences[
            index
        ]


        # ----------------------------------------------------
        # Add CLS
        # ----------------------------------------------------

        tokens = [
            vocabulary["<CLS>"]
        ] + sequence


        # ----------------------------------------------------
        # Add SEP
        # ----------------------------------------------------

        tokens.append(
            vocabulary["<SEP>"]
        )


        # ----------------------------------------------------
        # Truncate
        # ----------------------------------------------------

        tokens = tokens[
            :MAX_LENGTH
        ]


        attention_mask = [
            1
        ] * len(tokens)


        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        while len(tokens) < MAX_LENGTH:

            tokens.append(
                PAD_ID
            )

            attention_mask.append(
                0
            )


        return {

            "input_ids":
                torch.tensor(
                    tokens,
                    dtype=torch.long
                ),

            "attention_mask":
                torch.tensor(
                    attention_mask,
                    dtype=torch.long
                )

        }


# ============================================================
# ADD SPECIAL TOKENS
# ============================================================

if "<CLS>" not in vocabulary:

    vocabulary["<CLS>"] = len(
        vocabulary
    )


if "<SEP>" not in vocabulary:

    vocabulary["<SEP>"] = len(
        vocabulary
    )


CLS_ID = vocabulary["<CLS>"]

SEP_ID = vocabulary["<SEP>"]

VOCAB_SIZE = len(
    vocabulary
)


# ============================================================
# DATASET / DATALOADER
# ============================================================

dataset = SecurityLogDataset(
    training_sequences
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# BERT CONFIGURATION
# ============================================================

print(
    "\nCreating custom BERT model..."
)


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


# ============================================================
# MODEL
# ============================================================

model = BertForMaskedLM(
    config
)


model.to(
    DEVICE
)


print(
    "Custom BERT created."
)


# ============================================================
# MASKING FUNCTION
# ============================================================

def mask_sequence(
    input_ids,
    attention_mask
):

    labels = input_ids.clone()


    # --------------------------------------------------------
    # Start with no masking
    # --------------------------------------------------------

    probability = torch.zeros_like(
        input_ids,
        dtype=torch.float
    )


    # --------------------------------------------------------
    # Only event positions
    # --------------------------------------------------------

    for batch_index in range(
        input_ids.size(0)
    ):

        for position in range(
            input_ids.size(1)
        ):

            token_id = input_ids[
                batch_index,
                position
            ].item()


            if (
                attention_mask[
                    batch_index,
                    position
                ] == 1
                and
                token_id not in [
                    PAD_ID,
                    CLS_ID,
                    SEP_ID
                ]
            ):

                probability[
                    batch_index,
                    position
                ] = MASK_PROBABILITY


    # --------------------------------------------------------
    # Random masking
    # --------------------------------------------------------

    masked_indices = (
        torch.bernoulli(
            probability
        ).bool()
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Guarantee at least one masked event
    # --------------------------------------------------------

    for batch_index in range(
        input_ids.size(0)
    ):

        if not masked_indices[
            batch_index
        ].any():

            valid_positions = []

            for position in range(
                input_ids.size(1)
            ):

                token_id = input_ids[
                    batch_index,
                    position
                ].item()


                if (
                    attention_mask[
                        batch_index,
                        position
                    ] == 1
                    and
                    token_id not in [
                        PAD_ID,
                        CLS_ID,
                        SEP_ID
                    ]
                ):

                    valid_positions.append(
                        position
                    )


            if valid_positions:

                position = random.choice(
                    valid_positions
                )

                masked_indices[
                    batch_index,
                    position
                ] = True


    # --------------------------------------------------------
    # Only calculate loss on masked tokens
    # --------------------------------------------------------

    labels[
        ~masked_indices
    ] = -100


    # --------------------------------------------------------
    # 80% → MASK
    # --------------------------------------------------------

    indices_replaced = (
        torch.bernoulli(
            torch.full_like(
                probability,
                0.8
            )
        ).bool()
        & masked_indices
    )


    input_ids[
        indices_replaced
    ] = MASK_ID


    # --------------------------------------------------------
    # 10% → random token
    # --------------------------------------------------------

    indices_random = (
        torch.bernoulli(
            torch.full_like(
                probability,
                0.5
            )
        ).bool()
        & masked_indices
        & ~indices_replaced
    )


    random_words = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=input_ids.shape,
        device=input_ids.device
    )


    input_ids[
        indices_random
    ] = random_words[
        indices_random
    ]


    return input_ids, labels


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING CUSTOM LOG MODEL"
)

print(
    "=" * 70
)


model.train()


for epoch in range(
    EPOCHS
):

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )


    for batch in progress:

        input_ids = batch[
            "input_ids"
        ].to(
            DEVICE
        )


        attention_mask = batch[
            "attention_mask"
        ].to(
            DEVICE
        )


        # ----------------------------------------------------
        # Mask events
        # ----------------------------------------------------

        masked_input_ids, labels = (
            mask_sequence(
                input_ids.clone(),
                attention_mask
            )
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(

            input_ids=
                masked_input_ids,

            attention_mask=
                attention_mask,

            labels=
                labels

        )


        loss = outputs.loss


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
        )


        progress.set_postfix(
            loss=
                round(
                    loss.item(),
                    4
                )
        )


    average_loss = (
        total_loss
        / max(
            len(loader),
            1
        )
    )


    print(
        f"Epoch {epoch + 1}: "
        f"loss={average_loss:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_OUTPUT_DIR,
    exist_ok=True
)


model.save_pretrained(
    MODEL_OUTPUT_DIR
)


with open(
    os.path.join(
        MODEL_OUTPUT_DIR,
        "event_vocabulary.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        vocabulary,
        f,
        indent=4
    )


print(
    "\nModel saved to:",
    MODEL_OUTPUT_DIR
)


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(
    event_ids
):

    if len(event_ids) == 0:

        return None


    # --------------------------------------------------------
    # Add CLS + SEP
    # --------------------------------------------------------

    ids = [
        CLS_ID
    ] + event_ids + [
        SEP_ID
    ]


    ids = ids[
        :MAX_LENGTH
    ]


    attention = [
        1
    ] * len(ids)


    while len(ids) < MAX_LENGTH:

        ids.append(
            PAD_ID
        )

        attention.append(
            0
        )


    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )


    attention_mask = torch.tensor(
        [attention],
        dtype=torch.long,
        device=DEVICE
    )


    valid_positions = []


    for position in range(
        len(ids)
    ):

        token_id = ids[
            position
        ]


        if (
            attention[position] == 1
            and
            token_id not in [
                CLS_ID,
                SEP_ID,
                PAD_ID
            ]
        ):

            valid_positions.append(
                position
            )


    if not valid_positions:

        return None


    losses = []


    # --------------------------------------------------------
    # Mask each event individually
    # --------------------------------------------------------

    for position in valid_positions:

        masked_input = input_ids.clone()


        original_token = input_ids[
            0,
            position
        ]


        masked_input[
            0,
            position
        ] = MASK_ID


        with torch.no_grad():

            outputs = model(
                input_ids=
                    masked_input,

                attention_mask=
                    attention_mask
            )


        logits = outputs.logits[
            0,
            position
        ]


        log_probabilities = torch.log_softmax(
            logits,
            dim=-1
        )


        loss = -log_probabilities[
            original_token
        ]


        losses.append(
            loss.item()
        )


    return float(
        np.mean(losses)
    )


# ============================================================
# DETECT ALL SEQUENCES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ANOMALY DETECTION"
)

print(
    "=" * 70
)


results = []

scores = []


for sequence in sequences:

    sequence_id = sequence.get(
        "sequence_id"
    )


    event_ids = get_event_ids(
        sequence
    )


    score = calculate_anomaly_score(
        event_ids
    )


    event_names = []

    for event_id in event_ids:

        event_names.append(
            id_to_event.get(
                event_id,
                "<UNKNOWN>"
            )
        )


    result = {

        "sequence_id":
            sequence_id,

        "source":
            sequence.get(
                "source"
            ),

        "event_ids":
            event_ids,

        "events":
            event_names,

        "anomaly_score":
            score

    }


    results.append(
        result
    )


    if score is not None:

        scores.append(
            score
        )


    print(
        "\nSequence:",
        sequence_id
    )

    print(
        "Events:",
        event_names
    )

    print(
        "Score:",
        score
    )


# ============================================================
# THRESHOLD
# ============================================================

if not scores:

    raise RuntimeError(
        "No anomaly scores were generated."
    )


scores_array = np.array(
    scores
)


median = np.median(
    scores_array
)


mad = np.median(
    np.abs(
        scores_array - median
    )
)


if mad > 0:

    threshold = (
        median +
        3 * mad
    )

else:

    threshold = np.percentile(
        scores_array,
        95
    )


# ============================================================
# ANOMALY SCORE DEBUG
# ============================================================

print("\n" + "=" * 70)
print("ANOMALY SCORE DEBUG")
print("=" * 70)

print("All scores:")

for debug_result in results:

    print(
        debug_result["sequence_id"],
        "->",
        round(
            debug_result["anomaly_score"],
            6
        )
    )

print(
    "Median:",
    round(
        median,
        6
    )
)

print(
    "MAD:",
    round(
        mad,
        6
    )
)

print(
    "Threshold:",
    round(
        threshold,
        6
    )
)


# ============================================================
# CLASSIFICATION
# ============================================================

for result in results:

    score = result[
        "anomaly_score"
    ]


    if score is None:

        result[
            "classification"
        ] = "UNKNOWN"


    elif score >= threshold:

        result[
            "classification"
        ] = "ANOMALOUS"


    else:

        result[
            "classification"
        ] = "NORMAL"


    result[
        "threshold"
    ] = threshold


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "model":
        "Custom OCSF LAnoBERT-style BERT",

    "input":
        "logbert_sequences.json",

    "vocabulary":
        "event_vocabulary.json",

    "threshold":
        threshold,

    "total_sequences":
        len(results),

    "normal":
        sum(
            r["classification"]
            == "NORMAL"
            for r in results
        ),

    "anomalous":
        sum(
            r["classification"]
            == "ANOMALOUS"
            for r in results
        ),

    "results":
        results

}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL RESULT"
)

print(
    "=" * 70
)

print(
    "Total:",
    len(results)
)

print(
    "Normal:",
    output["normal"]
)

print(
    "Anomalous:",
    output["anomalous"]
)

print(
    "Threshold:",
    round(
        threshold,
        6
    )
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# SHOW RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SEQUENCE RESULTS"
)

print(
    "=" * 70
)


for result in results:

    print(
        "\nSequence:",
        result["sequence_id"]
    )

    print(
        "Source:",
        result["source"]
    )

    print(
        "Events:",
        " → ".join(
            result["events"]
        )
    )

    print(
        "Anomaly score:",
        round(
            result["anomaly_score"],
            6
        )
        if result["anomaly_score"]
        is not None
        else "UNKNOWN"
    )

    print(
        "Classification:",
        result["classification"]
    )