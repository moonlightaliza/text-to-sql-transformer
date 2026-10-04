def greedy_decode(model, src, max_len=64):
    # Picks the argmax token each step; stops at </s> or max_len.
    pass


def beam_search(model, src, beam_size=4, max_len=64):
    # Keeps the top beam_size hypotheses; stops at </s> or max_len.
    pass


def parse_sql(text):
    # Parses decoded text into {"sel", "agg", "conds"}; returns None if unparseable.
    pass


def to_readable_sql(query, header):
    # Turns a parsed query into SQL with the real column names.
    pass


def write_predictions(queries, path):
    # Writes one JSON line per example in order; {"error": "parse"} for failures.
    pass


def translate(model, sp, question, columns):
    # End-to-end helper for the front end: question + column names -> readable SQL.
    pass