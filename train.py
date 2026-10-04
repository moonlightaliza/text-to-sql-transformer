class Schedule:
    def __init__(self, optimizer, d_model, warmup_steps):
        pass

    def step(self):
        pass

    def get_lr(self, step):
        pass


class LabelSmoothedLoss:
    # Cross-entropy with label smoothing 0.1 that ignores <pad>.
    def __init__(self, vocab_size, pad_id, smoothing):
        pass

    def __call__(self, logits, target):
        pass


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    # One teacher-forced pass over the training set; returns mean train loss.
    pass


def evaluate_loss(model, loader, criterion, device):
    # Mean dev loss without gradient updates.
    pass


def main():
    # Runs 20 epochs, logs train/dev loss and LR per epoch, saves the best-dev-loss checkpoint.
    pass


if __name__ == "__main__":
    main()