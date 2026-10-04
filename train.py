import torch
import os
from starter.dataset import make_loader
from model.transformer import build_model
import sentencepiece as spm
from starter.tokenizer import read_pairs, PAD_ID, BOS_ID, EOS_ID


EPOCHS = 20
BATCH_SIZE = 64
D_MODEL = 256
WARMUP = 4000
LABEL_SMOOTHING = 0.1
SEED = 0
CKPT_DIR = "checkpoints"
USE_STANDIN = False   
MAX_BATCHES = None     
OVERFIT = False  
OVERFIT_STEPS = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Schedule:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1

        lr = self.get_lr(self.step_num)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.optimizer.step()
        

    def get_lr(self, step):
        return (
            self.d_model ** (-0.5)
            * min(
                step ** (-0.5),
                step * self.warmup_steps ** (-1.5)
            )
        )
        


class LabelSmoothedLoss:
    def __init__(self, vocab_size, pad_id, smoothing):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing

    def __call__(self, logits, target):
        # logits: (B, T, V)
        # target: (B, T)

        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(-1, target.unsqueeze(-1), 1.0 - self.smoothing)
            true_dist[:, :, self.pad_id] = 0
            pad_mask = target == self.pad_id
            true_dist[pad_mask] = 0

        loss = -(true_dist * log_probs).sum(dim=-1)
        non_pad = target != self.pad_id

        return loss.masked_select(non_pad).mean()
        


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    # One teacher-forced pass over the training set; returns mean train loss.
    model.train()
    total_loss = 0.0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_input)
        loss = criterion(logits, tgt_output)

        loss.backward()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)
    

def evaluate_loss(model, loader, criterion, device):
    # Mean dev loss without gradient updates.
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            loss = criterion(logits, tgt_output)

            total_loss += loss.item()

    return total_loss / len(loader)
    


def main():
    torch.manual_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
    train_dl = make_loader("starter/train_pairs.jsonl", sp, train=True, batch_size=BATCH_SIZE)
    dev_dl = make_loader("starter/dev_pairs.jsonl", sp, train=False, batch_size=BATCH_SIZE)

    model = build_model(sp).to(DEVICE)

    print(f"Device: {DEVICE}")
    print(f"Parameters: {model.count_parameters():,}")

    criterion = LabelSmoothedLoss(vocab_size=sp.get_piece_size(), pad_id=PAD_ID, smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = Schedule(optimizer, D_MODEL, WARMUP)
    best_dev_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_dl, criterion, optimizer, scheduler, DEVICE)
        dev_loss = evaluate_loss(model, dev_dl, criterion, DEVICE)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"LR: {lr:.6e}"
        )

        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dev_loss": dev_loss,
                },
                f"{CKPT_DIR}/best.pt"
            )

            print("Saved best checkpoint.")
    


if __name__ == "__main__":
    main()