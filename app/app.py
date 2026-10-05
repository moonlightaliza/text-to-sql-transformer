# app/app.py
import streamlit as st
import torch
import sentencepiece as spm
import sys
sys.path.insert(0, "../starter")
sys.path.insert(0, "..")

from model.transformer import Transformer
from starter.tokenizer import PAD_ID
from decode import translate

@st.cache_resource
def load_model():
    sp = spm.SentencePieceProcessor(model_file="../starter/sql_sp.model")
    model = Transformer(
        vocab_size=sp.get_piece_size(),
        pad_id=PAD_ID,
        d_model=256, h=4, n_layers=3, d_ff=1024, dropout=0.1
    )
    checkpoint = torch.load("../checkpoints/best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, sp

model, sp = load_model()

st.title("Text-to-SQL")
st.write("Ask a question about a table, give the column names, get the SQL query.")

question = st.text_input("Question", placeholder="What is Terrence Ross' nationality?")
columns_input = st.text_input("Column names (comma-separated)", placeholder="Player, No., Nationality, Position")

if st.button("Generate SQL"):
    if not question or not columns_input:
        st.warning("Enter both a question and column names.")
    else:
        header = [c.strip() for c in columns_input.split(",")]
        with st.spinner("Generating..."):
            result = translate(model, sp, question, header)

        if result:
            st.code(result, language="sql")
        else:
            st.error("Couldn't parse a valid SQL query from the model's output.")