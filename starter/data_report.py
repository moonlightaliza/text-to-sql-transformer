import sys
sys.path.insert(0, ".")
from pathlib import Path
import matplotlib.pyplot as plt
import sentencepiece as spm
from starter.embeddings import PositionalEncoding
from starter.tokenizer import read_pairs

out = Path("../results")
out.mkdir(exist_ok=True)
sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")

cells = []
for split in ["train", "dev", "test"]:
    pairs = read_pairs(f"{split}_pairs.jsonl")
    s = [len(x) + 1 for x in sp.encode([p["src"] for p in pairs])]  # + </s>
    t = [len(x) + 2 for x in sp.encode([p["tgt"] for p in pairs])]  # + <s>, </s>
    dropped = sum(a > 160 or b > 64 for a, b in zip(s, t)) if split == "train" else "–"
    cells.append([f"{len(pairs):,}", f"{sum(s) / len(s):.1f} / {max(s)}", f"{sum(t) / len(t):.1f} / {max(t)}", dropped])

labels = ["Pairs", "Mean / max source length (tokens)", "Mean / max target length (tokens)", "Pairs dropped as too long"]
table = "\n".join(["| | Train | Dev | Test |", "|---|---|---|---|"] + [f"| {l} | " + " | ".join(map(str, row)) + " |" for l, row in zip(labels, zip(*cells))])
(out / "table1.md").write_text(table + "\n")
print(table)

plt.figure(figsize=(10, 4))
plt.imshow(PositionalEncoding(256, dropout=0.0).pe[0, :100].numpy(), aspect="auto", cmap="RdBu")
plt.colorbar()
plt.xlabel("Embedding dimension")
plt.ylabel("Position")
plt.title("Sinusoidal positional encoding (100 x 256)")
plt.savefig(out / "fig1_positional_encoding.png", dpi=150, bbox_inches="tight")