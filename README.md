# Text-to-SQL with a T5 Encoder-Decoder

Converts English questions into SQL queries by fine-tuning **T5-small** (a Transformer encoder-decoder) on the `b-mc2/sql-create-context` dataset (WikiSQL + Spider).

## Files
| File | What it is |
|---|---|
| `Text_to_SQL_T5.ipynb` | **The submission.** The full notebook, with code and an explanation for every step. |
| `IMPLEMENTATION.md` | Implementation guide and design notes. |

## How to run
1. Go to <https://colab.research.google.com> → **Upload** → choose `Text_to_SQL_T5.ipynb`.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Select **Runtime → Run all**. The whole run takes about 25–40 minutes; training is most of it.
4. Afterwards, every cell has its output (metrics, plots, sample predictions) and you can submit the notebook.

To finish faster, set `TRAIN_SAMPLES = 10000` or `EPOCHS = 2` in the configuration cell.

## What the notebook covers
1. Problem statement and real-world motivation
2. Dataset description and exploration (length histograms, query-type distribution)
3. T5 encoder-decoder architecture, explained with a diagram
4. Tokenization, including a fix for T5's missing `<` token
5. Baseline output *before* fine-tuning
6. Fine-tuning with Hugging Face `Seq2SeqTrainer`, plus the loss curve
7. Inference on a new, unseen table, with the generated SQL executed in SQLite
8. Evaluation: Exact Match, BLEU, and Valid-SQL rate
9. Error analysis: accuracy by query type, correct and wrong examples
10. Cross-attention heatmap showing how the decoder reads the encoder
11. Optional Gradio web demo
12. Conclusion, limitations, future work and references
