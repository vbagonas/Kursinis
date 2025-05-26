import os
from huggingface_hub import login

os.environ["HUGGINGFACE_HUB_TOKEN"] = "Jūsų_HuggingFace_tokenas"  
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool, set_start_method
import gc

MODELS = {
    "codet5p": "Salesforce/codet5p-110m-embedding",
    "sfr": "Salesforce/SFR-Embedding-Code-400M_R", 
    "jina": "jinaai/jina-embeddings-v2-base-code"
}

global_model = None
global_tokenizer = None
global_model_name = None

def init_worker(model_name):
    global global_model, global_tokenizer, global_model_name
    global_model_name = model_name
    
    if model_name == "codet5p":
        global_tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name], trust_remote_code=True)
        global_model = AutoModel.from_pretrained(MODELS[model_name], trust_remote_code=True)
        global_model.eval()
    else:
        global_model = SentenceTransformer(MODELS[model_name], trust_remote_code=True)

def encode_code_worker(code):
    global global_model, global_tokenizer, global_model_name
    
    if global_model_name == "codet5p":
        inputs = global_tokenizer.encode(code, return_tensors="pt")
        with torch.no_grad():
            embedding = global_model(inputs)[0]
            embedding = embedding.numpy().flatten()
    else:
        embedding = global_model.encode([code])[0]
    
    return embedding

def process_data_chunk(data_chunk):
    chunk_indices = data_chunk.index.tolist()
    func1_list = data_chunk['func1'].tolist()
    func2_list = data_chunk['func2'].tolist()
    
    func1_embeddings = [encode_code_worker(code) for code in func1_list]
    func2_embeddings = [encode_code_worker(code) for code in func2_list]
    
    similarities = []
    for i in range(len(func1_embeddings)):
        emb1 = np.array(func1_embeddings[i]).reshape(1, -1)
        emb2 = np.array(func2_embeddings[i]).reshape(1, -1)
        sim = cosine_similarity(emb1, emb2)[0][0]
        similarities.append(sim)
    
    return chunk_indices, func1_embeddings, func2_embeddings, similarities

def process_model_parallel(model_name, df, num_processes=4):
    chunk_size = len(df) // num_processes
    if chunk_size == 0:
        chunk_size = 1
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  
    
    all_indices = []
    all_func1_embeddings = []
    all_func2_embeddings = []
    all_similarities = []
    
    with Pool(processes=num_processes, initializer=init_worker, initargs=(model_name,)) as pool:
        results = list(tqdm(
            pool.imap(process_data_chunk, chunks),
            total=len(chunks),
            desc=f"Processing with {model_name}"
        ))
    
    for chunk_indices, func1_embs, func2_embs, sims in results:
        all_indices.extend(chunk_indices)
        all_func1_embeddings.extend(func1_embs)
        all_func2_embeddings.extend(func2_embs)
        all_similarities.extend(sims)
    
    sorted_indices = sorted(range(len(all_indices)), key=lambda i: all_indices[i])
    all_func1_embeddings = [all_func1_embeddings[i] for i in sorted_indices]
    all_func2_embeddings = [all_func2_embeddings[i] for i in sorted_indices]
    all_similarities = [all_similarities[i] for i in sorted_indices]
    
    embeddings_df = df.copy()
    embeddings_df['code1_embedding'] = [emb.tolist() for emb in all_func1_embeddings]
    embeddings_df['code2_embedding'] = [emb.tolist() for emb in all_func2_embeddings]
    embeddings_df['similarity'] = all_similarities
    
    embeddings_df.to_csv(f"{model_name}_embeddings.csv", index=False)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for threshold in thresholds:
        predictions = [1 if sim >= threshold else 0 for sim in all_similarities]
        true_labels = df['label'].values
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        results.append({
            'model': model_name,
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    results_df = pd.DataFrame(results)
    
    return results_df

def main():
    df = pd.read_csv("clone_test_dataset.csv")
    
    if df['label'].dtype == bool or df['label'].isin(['TRUE', 'FALSE']).all():
        df['label'] = df['label'].map(lambda x: 1 if x == True or x == 'TRUE' else 0)
    
    all_results = []
    num_processes = 1  
    
    for model_name in MODELS.keys():
        results = process_model_parallel(model_name, df, num_processes)
        all_results.append(results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv("all_results.csv", index=False)

if __name__ == "__main__":
    main()
