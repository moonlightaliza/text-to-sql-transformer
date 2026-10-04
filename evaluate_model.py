import json
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch

sys.path.insert(0, "starter")
from data_prep import load_split
from dataset import make_loader
from tokenizer import BOS_ID, EOS_ID
from tokenizer import read_pairs
from decode import beam_search, greedy_decode, parse_sql, to_readable_sql, write_predictions
from model.transformer import build_model

def generate_predictions(model, sp, split, decoding):
    device = next(model.parameters()).device
    batch_size = 64 if decoding =='greedy' else 1
    loader = make_loader(f"starter/{split}_pairs.jsonl", sp, train=False, batch_size=batch_size)
    ids = []
    for src, _ in loader:
        src = src.to(device)
        if decoding == 'greedy':
            ids += greedy_decode(model, src)
        else:
            ids.append(beam_search(model, src))
    path = f"results/{split}_{decoding}.jsonl"
    write_predictions([parse_sql(sp.decode(i)) for i in ids], path)
    return path


def run_official_evaluator(split, pred_path):
    out = subprocess.run(
        [sys.executable, "evaluate.py", f"data/{split}.jsonl", f"data/{split}.db", str(Path(pred_path).resolve())],
        cwd='WikiSQL', capture_output=True, text=True, check=True)
    scores = json.loads(out.stdout)
    return 100 * scores["lf_accuracy"], 100 * scores["ex_accuracy"]


def parse_failure_rate(pred_path):
    pred = read_pairs(pred_path)
    return 100 * sum('error' in p for p in preds) / len(preds)

def cond_set(sql):
    return {(c, o, str(v).lower()) for c, o, v in sql['conds']}


def component_accuracy(pred_path, gold_path):
    preds, golds = read_pairs(pred_path), read_pairs(gold_path)
    sel = agg = where = 0
    for pred, gold in zip(preds, golds):
        q, g = pred.get('query'), gold['sql']
        if q is None:
            continue
        sel += q['sel'] == g['sel']
        agg += q['agg'] == g['agg']
        where += cond_set(q) == cond_set(g)

    n = len(golds)
    return 100 * sel / n, 100 * agg / n, 100 * where / n


def find_error(q, gold):
    if q is None:
        return "parse failure"
    if q["sel"] != gold["sel"]:
        return "wrong column"
    if q["agg"] != gold["agg"]:
        return "wrong aggregation"
    pred, true = cond_set(q), cond_set(gold)
    if pred == true:
        return None
    if len(pred) < len(true):
        return "missing condition"
    if len(pred) > len(true):
        return "extra condition"
    if {c[:2] for c in pred} == {c[:2] for c in true}:
        return "wrong value"
    return "wrong condition column or operator"


def plot_attention_map(model, sp, example, out_path):
    device = next(model.parameters()).device
    src = torch.tensor([sp.encode(example['src'])] + [EOS_ID], device=device)
    gen = greedy_decode(model, src)[0]
    ys = torch.tensor([BOS_ID] + gen, device=device)
    with torch.no_grad():
        model(src, ys)
    attn = model.decoder.layers[-1].cross_attention.weights[0].mean(0) # [T+1, S]
    rows = [sp.id_to_piece(i) for i in gen] + ["</s>"]
    cols = [sp.id_to_piece(i) for i in src[0].tolist()]
    plt.figure(figsize=(20, 5))
    plt.imshow(attn.cpu().numpy(), cmap="viridis", aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    plt.yticks(range(len(rows)), rows)
    plt.xlabel("Source tokens")
    plt.ylabel("Generated tokens")
    plt.colorbar()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")




def write_samples(pred_path, out_path):
    examples, tables = load_split("dev")
    right, wrong = [], []
    for ex, pred in zip(examples, read_jsonl(pred_path)):
        q = pred.get("query")
        error = find_error(q, ex["sql"])
        (right if error is None else wrong).append((ex, q, error))
    lines = []
    for ex, q, error in right[:5] + wrong[:5]:
        header = tables[ex["table_id"]]["header"]
        status = "Correct" if error is None else f"Wrong: {error}"
        pred_sql = to_readable_sql(q, header) if q else "(could not parse)"
        lines += [f"### {status}", f"- Question: {ex['question']}",
                  f"- Gold: `{to_readable_sql(ex['sql'], header)}`", f"- Predicted: `{pred_sql}`", ""]
    Path(out_path).write_text("\n".join(lines))


def main():
    Path('results').mkdir(exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file='starter/sql_sp.model')
    model = build_model(sp)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu")["model_state_dict"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    rows, paths, exec_acc = [], {}, {}
    for decoding in ['greedy', 'beam']:
        paths[decoding] = generate_predictions(model, sp, 'dev', decoding)
        lf, ex = run_official_evaluator('dev', paths[decoding])
        rows.append(f"| Dev | {decoding} | {lf:.2f} | {ex:.2f} | {parse_failure_rate(paths[decoding]):.2f} |")
        exec_acc[decoding] = ex
    best = max(exec_acc, key=exec_acc.get)

    if '--test' in sys.argv:
        path = generate_predictions(model, sp, 'test', best)
        lf, ex = run_official_evaluator('test', path)
        rows.append(f"| Test | {best} | {lf:.2f} | {ex:.2f} | {parse_failure_rate(path):.2f} |")

    header = ["| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |", "|---|---|---|---|---|"]
    Path("results/table3.md").write_text("\n".join(header + rows) + "\n")

    sel, agg, where = component_accuracy(paths[best], "WikiSQL/data/dev.jsonl")
    Path("results/table4.md").write_text(
        f"| Component | Accuracy (%) |\n|---|---|\n| sel column | {sel:.2f} |\n| agg | {agg:.2f} |\n| WHERE clause | {where:.2f} |\n")

    plot_attention_map(model, sp, read_jsonl("starter/dev_pairs.jsonl")[0], "results/fig4_attention.png")
    write_samples(paths[best], "results/samples.md")

if __name__ == "__main__":
    main()