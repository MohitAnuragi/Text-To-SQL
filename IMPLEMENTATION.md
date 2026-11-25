# Text-to-SQL Query Generation using an Encoder-Decoder Model (T5)

## 1. Problem Statement

Most business data sits in relational databases, and you need SQL to query it. Managers, analysts and other non-technical users often can't write SQL, so they depend on developers for even simple questions like *"How many employees are older than 30?"*.

**Goal:** build an encoder-decoder model that takes a **natural-language question** plus the **database schema** and produces the matching **SQL query**.

```
Input : "How many heads of departments are older than 56?"
        + schema: CREATE TABLE head (age INTEGER)
Output: SELECT COUNT(*) FROM head WHERE age > 56
```

**Real-world uses:** natural-language interfaces to databases, BI dashboards ("ask your data"), chatbots for customer or HR records, and accessibility for non-programmers.

---

## 2. Dataset

**`b-mc2/sql-create-context`** (Hugging Face, CC-BY-4.0)
- About 78,500 examples, built from the **WikiSQL** and **Spider** benchmarks.
- Each row has three fields:
  - `question`: an English question
  - `context`: the table schema as `CREATE TABLE` statements
  - `answer`: the target SQL query
- Why this dataset: it includes the schema with every question, so the model learns to use the table and column names it is given instead of memorising them. This lets it generalise to tables it has never seen. It also loads directly with `load_dataset` (no loading script needed).

**Splits used** (the dataset ships only a `train` split, so we create our own):

| Split | Size | Purpose |
|---|---|---|
| Train | 20,000 (subset, configurable) | Fine-tuning |
| Validation | 2,000 | Loss tracking and best-checkpoint selection |
| Test | 1,000 | Final evaluation (never seen during training) |

---

## 3. Model: T5 (Text-to-Text Transfer Transformer)

T5 is a **Transformer encoder-decoder** model. Every task is framed as "text in → text out", which suits Text→SQL directly.

```
 "translate to SQL: question: ... schema: CREATE TABLE ..."
                         │
                ┌────────▼────────┐
                │  Token Embedding │   SentencePiece tokens
                └────────┬────────┘
                ┌────────▼────────┐
                │     ENCODER      │   6 layers × (Self-Attention + Feed-Forward)
                │ (bidirectional)  │   reads the whole question + schema at once
                └────────┬────────┘
                         │  contextual hidden states (one vector per input token)
                         ▼
                ┌─────────────────┐
  previous  ──► │     DECODER      │   6 layers × (Masked Self-Attention
  SQL tokens    │ (autoregressive) │              + Cross-Attention to encoder
                └────────┬────────┘              + Feed-Forward)
                ┌────────▼────────┐
                │ Linear + Softmax │   probability over vocabulary
                └────────┬────────┘
                         ▼
            "SELECT COUNT(*) FROM head WHERE age > 56"   (one token at a time)
```

| Property | `t5-small` |
|---|---|
| Parameters | ~60 M |
| Encoder / Decoder layers | 6 / 6 |
| Hidden size (d_model) | 512 |
| Attention heads | 8 |
| Feed-forward size | 2048 |

**How the two halves work together**
- **Encoder:** reads the question and schema together. Self-attention links words like *"older than"* to the column `age`, and *"how many"* to `COUNT`.
- **Decoder:** writes the SQL one token at a time. *Masked self-attention* looks at the SQL written so far, and *cross-attention* looks back at the encoder output to decide which column, table or value to copy next.
- **Training:** uses *teacher forcing*. The decoder is fed the correct previous SQL token and trained with cross-entropy loss to predict the next one.
- **Inference:** uses *beam search* (4 beams), which keeps the 4 most probable partial queries at each step.

**Why fine-tune a pretrained model instead of training from scratch:** T5 is already pretrained on a large English corpus (C4), so it understands language. Fine-tuning only has to teach it the English→SQL mapping. This is *transfer learning*, and it gives far better accuracy than a from-scratch LSTM on a few thousand GPU-minutes.

---

## 4. Environment Setup

Use **Google Colab** → `Runtime → Change runtime type → T4 GPU`.

Create a new notebook named `Text_to_SQL_T5.ipynb` and add the cells below in order. Put a short markdown cell above each code cell that explains it (you can copy the explanations from this file).

### Cell 1: Install libraries
```python
!pip install -q transformers datasets accelerate sentencepiece nltk
```

### Cell 2: Imports and configuration
```python
import re, random, sqlite3
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed,
)

MODEL_NAME     = "google-t5/t5-small"
TRAIN_SAMPLES  = 20000      # increase (e.g. 50000) for better accuracy if time allows
VAL_SAMPLES    = 2000
TEST_SAMPLES   = 1000
MAX_INPUT_LEN  = 256        # question + schema tokens
MAX_TARGET_LEN = 128        # SQL tokens
EPOCHS         = 3
BATCH_SIZE     = 16
LR             = 3e-4       # typical learning rate for fine-tuning T5
SEED           = 42

set_seed(SEED)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)")
```

---

## 5. Data Loading and Exploration

### Cell 3: Load and split the dataset
```python
raw = load_dataset("b-mc2/sql-create-context")
print(raw)

split    = raw["train"].train_test_split(test_size=0.1, seed=SEED)
val_test = split["test"].train_test_split(test_size=0.5, seed=SEED)

train_ds = split["train"].shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
val_ds   = val_test["train"].select(range(VAL_SAMPLES))
test_ds  = val_test["test"].select(range(TEST_SAMPLES))

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
```

### Cell 4: Look at some examples
```python
for ex in train_ds.select(range(3)):
    print("QUESTION:", ex["question"])
    print("SCHEMA  :", ex["context"])
    print("SQL     :", ex["answer"])
    print("-" * 80)
```

### Cell 5: Length statistics (used to choose MAX_INPUT_LEN / MAX_TARGET_LEN)
```python
q_len = [len(x.split()) for x in train_ds["question"]]
c_len = [len(x.split()) for x in train_ds["context"]]
a_len = [len(x.split()) for x in train_ds["answer"]]

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
for a, data, title in zip(ax, [q_len, c_len, a_len], ["Question", "Schema", "SQL"]):
    a.hist(data, bins=40); a.set_title(f"{title} length (words)")
plt.tight_layout(); plt.show()
```

---

## 6. Tokenization and Preprocessing

### Cell 6: Tokenizer, plus a fix for the missing `<` symbol
T5's vocabulary has **no `<` character**, so it becomes `<unk>`. That would break queries like `WHERE age < 30`. We detect this and add `<` as a new token. (Section 7 resizes the model's embedding matrix to match.)

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Before fix:", tokenizer.convert_ids_to_tokens(tokenizer("age < 30").input_ids))
if tokenizer.unk_token_id in tokenizer("<").input_ids:
    tokenizer.add_tokens(["<"])
print("After fix :", tokenizer.convert_ids_to_tokens(tokenizer("age < 30").input_ids))
```

### Cell 7: Input format and preprocessing
Question and schema are joined into one input string with a task prefix (the T5 convention):

```
translate to SQL: question: <question> schema: <CREATE TABLE ...>
```

```python
def build_input(question, context):
    return f"translate to SQL: question: {question} schema: {context}"

def preprocess(batch):
    inputs = [build_input(q, c) for q, c in zip(batch["question"], batch["context"])]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LEN, truncation=True)
    labels = tokenizer(text_target=batch["answer"], max_length=MAX_TARGET_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tok_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
tok_val   = val_ds.map(preprocess,   batched=True, remove_columns=val_ds.column_names)
print(tok_train)
```

---

## 7. Model

### Cell 8: Load T5 and resize embeddings
```python
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))   # room for the added '<' token

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params/1e6:.1f} M")
print("Encoder layers:", model.config.num_layers, "| Decoder layers:", model.config.num_decoder_layers)
```

---

## 8. Training

### Cell 9: Training setup
- `DataCollatorForSeq2Seq` pads each batch dynamically and sets padded label positions to `-100` so they are ignored in the loss.
- The best checkpoint (lowest validation loss) is restored at the end.
- `fp16` stays **off** because T5 is known to produce NaN losses in half precision.

```python
collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir="t5-text2sql",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_steps=200,
    logging_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    report_to="none",
    seed=SEED,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    data_collator=collator,
    processing_class=tokenizer,
)
```

### Cell 10: Train
```python
train_result = trainer.train()
print(train_result)
```
On a T4 with the defaults (20k samples, 3 epochs), expect roughly 15–30 minutes.

### Cell 11: Loss curves (for the report)
```python
hist = trainer.state.log_history
tr = [(h["step"], h["loss"])      for h in hist if "loss" in h]
ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]

plt.figure(figsize=(8, 4))
plt.plot(*zip(*tr), label="Train loss")
plt.plot(*zip(*ev), "o-", label="Validation loss")
plt.xlabel("Step"); plt.ylabel("Cross-entropy loss"); plt.title("Training curve")
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

### Cell 12: Save the model
```python
trainer.save_model("t5-text2sql-final")
tokenizer.save_pretrained("t5-text2sql-final")
```

---

## 9. Inference

### Cell 13: SQL generation with beam search
```python
model.eval()

def clean_sql(s):
    """Fix spacing around operators (the added '<' token can lose its spaces when decoded)."""
    s = re.sub(r"\s*(<=|>=|!=|<>|=|<|>)\s*", r" \1 ", s)
    return " ".join(s.split())

def generate_sql(questions, contexts, num_beams=4, batch_size=32):
    preds = []
    for i in range(0, len(questions), batch_size):
        inputs = [build_input(q, c) for q, c in
                  zip(questions[i:i + batch_size], contexts[i:i + batch_size])]
        enc = tokenizer(inputs, max_length=MAX_INPUT_LEN, truncation=True,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_TARGET_LEN, num_beams=num_beams)
        preds.extend(clean_sql(p) for p in tokenizer.batch_decode(out, skip_special_tokens=True))
    return preds
```

### Cell 14: Try it on your own schema
```python
schema = ("CREATE TABLE employees (name VARCHAR, department VARCHAR, "
          "salary INTEGER, age INTEGER, city VARCHAR)")
questions = [
    "How many employees are older than 30?",
    "What is the average salary in the Sales department?",
    "List the names of employees who live in Delhi.",
    "What is the highest salary of employees younger than 25?",
]
for q, sql in zip(questions, generate_sql(questions, [schema] * len(questions))):
    print(f"Q  : {q}\nSQL: {sql}\n")
```

---

## 10. Evaluation

Three metrics, computed on the 1,000 held-out test examples:

| Metric | What it measures |
|---|---|
| **Exact Match (EM)** | The predicted SQL equals the gold SQL after normalisation (case, spacing, trailing `;`). Strict. |
| **BLEU** | n-gram overlap between predicted and gold SQL. Gives partial credit for nearly correct queries. |
| **Valid-SQL rate** | The predicted query actually **runs** in SQLite against the given schema (no syntax or column errors). |

### Cell 15: Metric functions
```python
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def normalize_sql(s):
    s = s.strip().rstrip(";").lower()
    s = re.sub(r"\s*(<=|>=|!=|<>|=|<|>|,|\(|\))\s*", r" \1 ", s)
    return " ".join(s.split())

def is_valid_sql(query, context):
    """Build the schema in an in-memory SQLite DB and check that the query executes."""
    conn = sqlite3.connect(":memory:")
    try:
        for stmt in re.split(r"(?i)(?=CREATE TABLE)", context):
            stmt = stmt.strip().rstrip(",;").strip()
            if stmt:
                conn.execute(stmt)
        conn.execute(query)
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()
```

### Cell 16: Evaluate on the test set
```python
questions, contexts, golds = test_ds["question"], test_ds["context"], test_ds["answer"]
preds = generate_sql(questions, contexts)

em = np.mean([normalize_sql(p) == normalize_sql(g) for p, g in zip(preds, golds)])

bleu = corpus_bleu([[normalize_sql(g).split()] for g in golds],
                   [normalize_sql(p).split() for p in preds],
                   smoothing_function=SmoothingFunction().method1)

# Only count samples whose gold query runs (some dataset schemas are imperfect)
gold_ok  = [is_valid_sql(g, c) for g, c in zip(golds, contexts)]
pred_ok  = [is_valid_sql(p, c) for p, c in zip(preds, contexts)]
valid    = np.mean([p for p, g in zip(pred_ok, gold_ok) if g])

print(f"Exact Match   : {em*100:.2f}%")
print(f"BLEU          : {bleu*100:.2f}")
print(f"Valid-SQL rate: {valid*100:.2f}%  (on {sum(gold_ok)} samples with valid gold SQL)")
```

### Cell 17: Sample predictions and error analysis
```python
print("=== Correct predictions ===")
shown = 0
for q, g, p in zip(questions, golds, preds):
    if normalize_sql(p) == normalize_sql(g):
        print(f"Q   : {q}\nGOLD: {g}\nPRED: {p}\n"); shown += 1
    if shown == 5: break

print("=== Wrong predictions ===")
shown = 0
for q, c, g, p in zip(questions, contexts, golds, preds):
    if normalize_sql(p) != normalize_sql(g):
        print(f"Q     : {q}\nSCHEMA: {c}\nGOLD  : {g}\nPRED  : {p}\n"); shown += 1
    if shown == 5: break
```

When you look at the wrong predictions, you will usually find these kinds of errors (mention them in the report):
- The right structure with a slightly different value string (for example, a capitalisation difference in `'Delhi'`).
- The wrong aggregate (`MAX` vs `COUNT`) when the question is ambiguous.
- A missed `JOIN` condition on multi-table (Spider-style) questions, which are much harder.
- Gold queries that are noisy themselves (the WikiSQL-derived data has some odd values).

---

## 11. Optional: Gradio Web Demo (useful for screenshots)

```python
!pip install -q gradio
import gradio as gr

def demo_fn(question, schema):
    return generate_sql([question], [schema])[0]

gr.Interface(
    fn=demo_fn,
    inputs=[gr.Textbox(label="Question"),
            gr.Textbox(label="Schema (CREATE TABLE ...)", lines=3,
                       value="CREATE TABLE employees (name VARCHAR, department VARCHAR, salary INTEGER, age INTEGER)")],
    outputs=gr.Textbox(label="Generated SQL"),
    title="Text-to-SQL with T5 (Encoder-Decoder)",
).launch(share=True)
```

---

## 12. Results Table (fill in from your run)

| Setting | Value |
|---|---|
| Model | t5-small (60 M params, 6+6 layers) |
| Training samples | 20,000 |
| Epochs | 3 |
| Final train loss | ___ |
| Final validation loss | ___ |
| Test Exact Match | ___ % |
| Test BLEU | ___ |
| Test Valid-SQL rate | ___ % |
| Training time (T4 GPU) | ___ min |

Include in the report: the loss curve plot, 5 correct and 3–5 wrong predictions, the custom-schema demo output, and (optionally) a Gradio screenshot.

---

## 13. Limitations and Future Work
- **Larger models:** `t5-base` (220 M) or `flan-t5-base` usually give clearly better accuracy.
- **Execution accuracy:** run predicted and gold queries on *real* populated databases (the Spider DBs) and compare the results, not just the strings.
- **Constrained decoding** (for example, PICARD) so the model can only produce grammatically valid SQL.
- **Harder queries:** the model is weaker on multi-table joins, nested queries and `GROUP BY ... HAVING`.
- **Safety:** in production, run generated SQL with read-only permissions to avoid destructive queries.

---

## 14. Troubleshooting

| Error | Fix |
|---|---|
| `TypeError: ... unexpected keyword 'processing_class'` | Old transformers version. Replace it with `tokenizer=tokenizer`, or run `!pip install -U transformers` and restart the runtime. |
| `TypeError: ... unexpected keyword 'eval_strategy'` | Old transformers version. Use `evaluation_strategy="epoch"` instead. |
| `CUDA out of memory` | Lower `BATCH_SIZE` to 8 or `MAX_INPUT_LEN` to 192. |
| Loss is `nan` | Make sure `fp16=False`. |
| Training too slow | Lower `TRAIN_SAMPLES` to 10000 or `EPOCHS` to 2. |
| Colab disconnects | Re-run all cells. Checkpoints in `t5-text2sql/` survive only within the same session, so save to Google Drive if needed. |

---

## 15. Submission Checklist
- [ ] `Text_to_SQL_T5.ipynb`, run top to bottom **with outputs visible**
- [ ] Markdown explanation cell above each code cell (problem → data → model → training → evaluation)
- [ ] Loss curve, metrics table and sample predictions included
- [ ] Short report (PDF or in the email body) covering problem, dataset, architecture diagram, results, limitations
- [ ] Reply to the professor's email with the notebook (or a Colab share link set to "Anyone with the link can view") before **25 Nov**

---

## References
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5), JMLR 2020.
- Zhong et al., *Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning* (WikiSQL), 2017.
- Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*, EMNLP 2018.
- Dataset: `b-mc2/sql-create-context`, Hugging Face Hub.
