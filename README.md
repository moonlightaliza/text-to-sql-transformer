# Text-to-SQL with a Transformer Built from Scratch

An encoder-decoder Transformer ("Attention Is All You Need", Vaswani et al., 2017) implemented from basic PyTorch layers and trained from random initialisation on WikiSQL. The model maps an English question and the column names of one table to a SQL query over that table.

| | |
|---|---|
| Input | English question and the column names of one table |
| Output | `SELECT [agg] col FROM table [WHERE col op value (AND col op value)*]` |
| Dataset | WikiSQL (Zhong, Xiong and Socher, 2017) |
| Parameters | 7,585,600 |
| Framework | PyTorch, SentencePiece |

Constraints followed: no pretrained weights, no `nn.Transformer`, `nn.TransformerEncoder(Layer)`, `nn.TransformerDecoder(Layer)`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, and no Hugging Face `transformers`.

## Repository layout

```
starter/                 given input-side code (data prep, tokenizer, dataset, embeddings)
model/
  attention.py           scaled dot-product attention, multi-head attention
  layers.py              feed-forward, encoder layer, decoder layer
  transformer.py         masks, encoder, decoder, full model
train.py                 training loop, loss, learning-rate schedule
decode.py                greedy decoding, beam search, SQL parser, readable SQL
evaluate_model.py        prediction files, official metrics, component accuracy, attention map, samples
results/                 prediction files, tables, figures, samples.md
text_to_sql_transformer.ipynb   Colab notebook used for the training run
requirements.txt
```

## Dataset

WikiSQL contains 80,654 annotated questions over 24,241 Wikipedia tables. The official splits share no tables, so every evaluation is on unseen tables. Every query has the form `SELECT [agg] col FROM table [WHERE col op value (AND ...)*]` with `agg` in {none, MAX, MIN, COUNT, SUM, AVG} and `op` in {=, >, <}.

### Text-to-text framing

`starter/data_prep.py` converts each example into a source/target string pair. Columns are referred to by special tokens `<c0>`, `<c1>`, ..., so the model points at a column instead of spelling its name. Everything is lowercased.

```
source: what is terrence ross' nationality <sep> <c0> player <c1> no. <c2> nationality <c3> position <c4> years in toronto <c5> school/club team
target: select <c2> where <c0> = terrence ross
```

### Tokenisation

One shared SentencePiece BPE vocabulary of 8,000 pieces is trained on the source and target text of the training split (`starter/tokenizer.py`). `<sep>` and `<c0>`..`<c63>` are user-defined symbols and are never split. Special ids: pad 0, unk 1, bos 2, eos 3. The shared vocabulary allows one embedding matrix for the encoder, the decoder and the output projection.

Source sequences are `src_ids + </s>`. Target sequences are `<s> tgt_ids </s>`. Training drops pairs longer than 160 source or 64 target tokens. Dev and test keep every row in file order.

### Data statistics

| | Train | Dev | Test |
|---|---|---|---|
| Pairs | 56,355 | 8,421 | 15,878 |
| Mean / max source length (tokens) | 42.5 / 222 | 42.5 / 167 | 42.7 / 260 |
| Mean / max target length (tokens) | 14.8 / 65 | 14.8 / 44 | 14.9 / 46 |
| Pairs dropped as too long | 19 | - | - |

Training uses 56,336 pairs (880 batches per epoch at batch size 64).

## Model architecture

Configuration (`model/transformer.py`, `build_model`):

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| Heads h | 4 (d_k = d_v = 64) |
| Encoder / decoder layers | 3 / 3 |
| Feed-forward inner size d_ff | 1024 |
| Dropout | 0.1 |
| Normalisation | post-norm, `LayerNorm(x + Sublayer(x))` |
| Weight sharing | encoder embedding = decoder embedding = output projection |

### Components

- **Input layer (starter).** Token embedding multiplied by sqrt(d_model), plus fixed sinusoidal positional encoding, then dropout. One `TokenEmbedding` instance is shared by the source and target `InputLayer`s.
- **Scaled dot-product attention** (`attention.py`). `softmax(QK^T / sqrt(d_k)) V`. Masked positions are set to `-inf` before the softmax. Returns the output and the attention weights.
- **Multi-head attention** (`attention.py`). Q, K and V are projected to d_model (projections live in the calling layer), split into h heads, attended in parallel, concatenated and projected with `W^O`. The weights of the last call are kept in `.weights` for the attention map.
- **Position-wise feed-forward** (`layers.py`). `W2 (dropout(ReLU(W1 x + b1))) + b2`.
- **Encoder layer.** Self-attention, add and norm, feed-forward, add and norm.
- **Decoder layer.** Masked self-attention, add and norm, cross-attention over the encoder output, add and norm, feed-forward, add and norm.
- **Output layer.** `nn.Linear(d_model, vocab)` whose weight is the shared embedding matrix (the same tensor; `is` check holds). The layer keeps its own bias.

### Masks

- **Padding mask**, shape `(B, 1, 1, L)`: hides `<pad>` keys in encoder self-attention, decoder cross-attention and decoder self-attention.
- **Causal mask**, shape `(T, T)`: lower triangular, combined with the target padding mask for decoder self-attention.

### Parameter count

| Part | Parameters |
|---|---|
| Shared embedding (8,000 x 256) | 2,048,000 |
| Encoder (3 x 789,760) | 2,369,280 |
| Decoder (3 x 1,053,440) | 3,160,320 |
| Output bias | 8,000 |
| **Total** | **7,585,600** |

## Training

Implemented in `train.py`.

| Setting | Value |
|---|---|
| Teacher forcing | decoder input `tgt[:, :-1]`, target `tgt[:, 1:]` |
| Loss | cross-entropy with label smoothing 0.1, `<pad>` ignored |
| Optimiser | Adam, beta1 0.9, beta2 0.98, eps 1e-9 |
| Learning-rate schedule | `d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)`, warmup 4000 steps |
| Batch size / epochs | 64 / 20 (about 17,600 steps) |
| Seed | 0 |
| Checkpoint | lowest dev loss, saved to `checkpoints/best.pt` |
| Hardware | Google Colab GPU |

The dev set is used only to select the checkpoint. The test set is evaluated once, at the end, with the decoding method chosen on dev.

Per-epoch train loss, dev loss and learning rate are in `results/training_results.csv`. The learning rate rises linearly to about 9.4e-4 (epoch 5, step 4000) and then decays as `step^-0.5`. Best dev loss: 2.0844 at epoch 19. Final epoch: train 1.8308, dev 2.0897. The label-smoothed loss cannot reach zero; with epsilon 0.1 and an 8,000-piece vocabulary its floor is roughly 1.2.

## Decoding and SQL conversion

Implemented in `decode.py`.

- **Greedy decoding.** Batched argmax, stops per sequence at `</s>` or after 64 tokens.
- **Beam search.** Beam size 4, one example at a time, stops at `</s>` or after 64 tokens. Finished hypotheses are ranked by log-probability divided by length.
- **Parser** (`parse_sql`). `sp.decode` output is matched against `select [agg] <cK> [where <cI> op value (and <cJ> op value)*]` and converted to WikiSQL's format `{"sel": int, "agg": int, "conds": [[col, op, value], ...]}` using `AGG_OPS` and `COND_OPS` from `data_prep.py`. Unparseable output returns `None`.
- **Prediction files.** One JSON line per example in dev/test file order: `{"query": {...}}`, or `{"error": "parse"}` when parsing fails. No line is skipped.
- **Readable SQL** (`to_readable_sql`, `translate`). Converts a parsed query back to SQL with the real column names, for the front end and for `results/samples.md`.

## Evaluation

Run with `python evaluate_model.py` (dev, greedy and beam) and `python evaluate_model.py --test` (adds the test run).

### Official metrics

Computed with the evaluator shipped with WikiSQL (`WikiSQL/evaluate.py`), not a custom implementation.

- **Logical-form accuracy:** predicted `sel`, `agg` and set of conditions equal the gold query. Value comparison is case-insensitive.
- **Execution accuracy:** the predicted query, run on the real table in `dev.db` / `test.db`, returns the same result as the gold query.
- **Parse-failure rate:** percentage of prediction lines written as `{"error": "parse"}`.

### Component accuracy (dev)

Percentage of dev examples with the correct `sel` column, the correct `agg`, and the correct WHERE clause (same set of conditions, order ignored). Parse failures count as wrong.

### Validation of the evaluation path

Gold dev targets passed through `sp.encode`, `sp.decode` and `parse_sql`, then through the official evaluator, give 99.49% execution and logical-form accuracy with no parse failures. The parser and the evaluator wiring are therefore not a source of error.

## Results

Full outputs are in `results/`. Numbers below are from the checkpoint produced by the training run described above.

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | 9.79 | 16.92 | 0.00 |
| Dev | beam (4) | 10.01 | 17.61 | 0.00 |
| Test | beam (4) | 10.27 | 18.38 | 0.00 |

| Component (dev) | Accuracy (%) |
|---|---|
| `sel` column | 31.74 |
| `agg` | 87.51 |
| WHERE clause | 23.17 |

Beam search gives a small gain over greedy decoding. Aggregation is predicted reliably; column selection and condition construction are the weak components.

### Result files and figures

| File | Content |
|---|---|
| `results/table1.md` | data statistics |
| `results/table3.md` | official metrics |
| `results/table4.md` | component accuracy |
| `results/dev_greedy.jsonl`, `dev_beam.jsonl`, `test_beam.jsonl` | prediction files |
| `results/samples.md` | five correct and five wrong dev examples with failure types |
| `results/training_results.csv` | per-epoch losses and learning rate |
| `results/fig1_positional_encoding.png` | positional-encoding heat-map, first 100 positions x 256 dimensions |
| `results/loss_curve.png` | training and dev loss per epoch |
| `results/fig4_attention.png` | last-layer decoder cross-attention (averaged over heads) for one dev example, generated tokens x source tokens |
| `results/check_starter_output.txt` | output of the starter sanity check |

Failure types assigned in `samples.md` (`find_error`): parse failure, wrong column, wrong aggregation, missing condition, extra condition, wrong value, wrong condition column or operator.

## Reproducing

Run everything from the repository root.

```bash
pip install -r requirements.txt

# data
git clone https://github.com/salesforce/WikiSQL
tar xjf WikiSQL/data.tar.bz2 -C WikiSQL            # creates WikiSQL/data/
python starter/data_prep.py                        # writes {train,dev,test}_pairs.jsonl to the working directory
python -m starter.tokenizer                        # optional; starter/sql_sp.model is committed

# train (writes checkpoints/best.pt)
python train.py

# evaluate (writes results/*)
python evaluate_model.py                           # dev: greedy and beam
python evaluate_model.py --test                    # also runs test once
```

`*_pairs.jsonl` files are generated and not tracked by git. `train.py` and `evaluate_model.py` expect them in the repository root. `evaluate_model.py` calls the official evaluator with `WikiSQL/` as its working directory, so `WikiSQL/data/` must exist. Training on CPU is impractical; a GPU is expected.

## Front end

A web interface that takes a question and a comma-separated list of column names and returns the generated SQL with the real column names, using `decode.translate` and the trained checkpoint. Screenshots and run instructions to be added.

## Limitations and known issues

- **Accuracy is low.** Execution accuracy on dev (16.9% greedy, 17.6% beam) is below the 36% reported for the original LSTM sequence-to-sequence baseline, and below the 20% level that the assignment manual treats as indicating a bug.
- **Probable cause: embedding initialisation.** The model uses `nn.Embedding`'s default N(0,1) initialisation, scaled by sqrt(d_model) = 16 by the starter `TokenEmbedding`, and the same matrix is the output projection. On an untrained model this gives logits with standard deviation about 16 and an initial loss of about 187, against about 9 for a correct start. Epoch-1 train loss of 24.2 reflects this. The scaled embeddings are also roughly 20 times larger than the positional encodings, which weakens position information that column-token binding depends on. The pattern in the results (aggregation 87.5%, column selection 31.7%) is consistent with this. Initialising the shared embedding with standard deviation d_model^-0.5 gives an initial loss near 11 and logit standard deviation near 1. The results in this README predate that change; the effect on accuracy has not yet been measured by retraining.
- **Training budget.** The configuration is fixed by the assignment: about 17,600 optimiser steps with a 4,000-step warmup.
- **Query coverage.** The output format covers single-table `SELECT` queries with one aggregate and `=`, `>`, `<` conditions, as in WikiSQL. Column indices are not validated against the table's real column count in `parse_sql`, so a prediction referencing a nonexistent column parses but fails at `to_readable_sql`.
- **Values are generated, not copied.** Condition values are produced token by token from the BPE vocabulary and are lowercased by the data format. Rare values and punctuation-heavy strings are often corrupted (for example `butler cc (ks)` predicted as `ccksler`).
- **Tables are described only by column names.** Cell contents and column types are not given to the model.
- **Beam search is slow.** It decodes one example per call.
- **Not yet included:** the correctness checks listed in the assignment (causal mask, padding mask, attention rows sum to 1, weight-sharing check, learning-rate plot), the learning-rate schedule figure, and the model/training summary table.

## References

- Vaswani et al., Attention Is All You Need, 2017. https://arxiv.org/abs/1706.03762
- Zhong, Xiong and Socher, Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning, 2017. https://arxiv.org/abs/1709.00103
- WikiSQL data and evaluator. https://github.com/salesforce/WikiSQL