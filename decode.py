import json
import re
import sys
import torch
sys.path.insert(0, "starter")
from starter.data_prep import AGG_OPS, COND_OPS, encode_source
from starter.tokenizer import BOS_ID, EOS_ID, PAD_ID
from model.transformer import make_pad_mask, make_tgt_mask

@torch.no_grad()
def greedy_decode(model, src, max_len=64):
    src_mask = make_pad_mask(src, PAD_ID)  # (B, 1, 1, S)
    mem = model.encode(src, src_mask) # (B, S, 256)
    ys = torch.full((src.size(0), 1), BOS_ID, device=src.device) # (B, 1)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device) # (B)
    for _ in range(max_len):
        nxt = model.decode(ys, mem, src_mask, make_tgt_mask(ys, PAD_ID)) # (B, T, V)
        nxt = nxt[:,-1] # (B, V)
        nxt = nxt.argmax(-1) # (B,)
        nxt[done] = PAD_ID
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        done[nxt == EOS_ID] = True
        if done.all():
            break

    return [r[:r.index(EOS_ID)] if EOS_ID in r else r for r in ys[:, 1:].tolist()]


@torch.no_grad()
def beam_search(model, src, beam_size=4, max_len=64):
    src_mask = make_pad_mask(src, PAD_ID)
    mem = model.encode(src, src_mask)
    beams, done = [([BOS_ID], 0.0)], []
    for _ in range(max_len):
        ys = torch.tensor([s for s, _ in beams], device=src.device) # (k, t)
        logits = model.decode(ys, mem.expand(len(beams), -1, -1), src_mask, make_tgt_mask(ys, PAD_ID)) # (k, t, V)
        lp = logits[:, -1].log_softmax(-1) # (k, V)
        cand = []
        for (s, sc), row in zip(beams, lp):
            top = row.topk(beam_size)
            cand += [(s + [i], sc + v) for v, i in zip(top.values.tolist(), top.indices.tolist())]
        beams = []
        for s, sc in sorted(cand, key=lambda c: -c[1])[:beam_size]:
            if s[-1] == EOS_ID:
                done.append((s, sc))
            else:
                beams.append((s, sc))
        if not beams:
            break
    s = max(done + beams, key=lambda b: b[1] / (len(b[0])-1))[0]
    return s[1:-1] if s[-1] == EOS_ID else s[1:]
 




def parse_sql(text):
    m = re.fullmatch(r"select (?:(max|min|count|sum|avg) )?<c(\d+)>(?: where (.+))?", text.strip())
    if not m:
        return None
    agg, sel, conds = m.groups()
    out = []
    for c in re.split(r" and (?=<c\d+> [=<>])", conds) if conds else []:
        cm = re.fullmatch(r"<c(\d+)> ([=<>]) ?(.*)", c)
        if not cm:
            return None
        out.append([int(cm[1]), COND_OPS.index(cm[2]), cm[3]])
    return {'sel': int(sel), 'agg': AGG_OPS.index(agg.upper()) if agg else 0, 'conds': out}



def to_readable_sql(query, header):
    col = header[query['sel']]
    sql = f"SELECT {AGG_OPS[query['agg']]}({col}) FROM table" if query['agg'] else f"SELECT {col} FROM table"
    if query['conds']:
        sql += " WHERE " + " AND ".join(f"{header[c]} {COND_OPS[o]} '{v}'" for c, o, v in query['conds'])
    return sql



def write_predictions(queries, path):
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({'query': q} if q else {'error': 'parse'}) + "\n")


def translate(model, sp, question, cols):
    model.eval()
    src = torch.tensor([sp.encode(encode_source(question, cols)) + [EOS_ID]], device=next(model.parameters()).device)
    q = parse_sql(sp.decode(beam_search(model, src)))
    return to_readable_sql(q, cols) if q else None