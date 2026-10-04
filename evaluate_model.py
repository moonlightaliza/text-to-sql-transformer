def generate_predictions(model, sp, split, decoding):
    # Decodes a whole split and writes results/<split>_<decoding>.jsonl.
    pass


def run_official_evaluator(split, pred_path):
    # Calls WikiSQL's evaluate.py; returns logical-form and execution accuracy.
    pass


def parse_failure_rate(pred_path):
    # Percentage of lines that are {"error": "parse"}.
    pass


def component_accuracy(pred_path, gold_path):
    # Percentage of dev examples with correct sel, agg, and WHERE (condition set, order ignored).
    pass


def plot_attention_map(model, sp, example, out_path):
    # Plots last-layer decoder cross-attention (head-averaged), generated tokens x source tokens.
    pass


def write_samples(pred_path, out_path):
    # Writes results/samples.md: five correct and five wrong dev examples with failure types.
    pass


def main():
    # Runs dev greedy/beam, test once with the chosen decoding, and writes all tables.
    pass


if __name__ == "__main__":
    main()