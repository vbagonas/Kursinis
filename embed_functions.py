import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
import multiprocessing

def init_worker():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Code-400M_R", trust_remote_code=True)
    model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Code-400M_R", trust_remote_code=True)
    model.eval()

def embed_code(code: str):

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    )
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
    return hidden[0, 0].cpu().numpy().tolist()

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    df = pd.read_parquet("subset_100k_part1.parquet")
    codes = df["function_code"].tolist()
    langs = df["language"].tolist()
    total = len(codes)

    vectors = []

    with Pool(processes=8, initializer=init_worker) as pool:
        for vec in tqdm(pool.imap(embed_code, codes, chunksize=10),
                      total=total,
                        desc="Embedding functions",
                        unit="func"):
            vectors.append(vec)
    out = pd.DataFrame({
        "function_code": codes,
        "language":      langs,
        "embedding":     vectors
    })
    out.to_parquet("100k_embedded_part1.parquet", index=False)
    print(f"Wrote {len(out)} rows  ^f^r functions_with_embeddings.parquet")

if __name__ == "__main__":
    main()